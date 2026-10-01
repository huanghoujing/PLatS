"""Artifact- and topology-aware P2SD metrics."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable

import numpy as np
from scipy import ndimage

from vesuvius_p2sd.eval.artifact_scores import (
    border_artifact_fraction,
    floating_artifact_component_count,
    floating_artifact_fraction,
    missing_sheet_proxy,
)
from vesuvius_p2sd.eval.connected_components import (
    component_info,
    prompt_connected_mask,
)
from vesuvius_p2sd.eval.distance import distance_transform_edt
from vesuvius_p2sd.eval.topology import betti_numbers
from vesuvius_p2sd.eval.kaggle_surface import (
    KaggleSurfaceMetricConfig,
    compute_kaggle_prompt_component_metrics,
)


@dataclass(frozen=True)
class P2SDMetricConfig:
    threshold: float = 0.5
    tolerance_voxels: tuple[int, ...] = (1, 2)
    erased_border_width: int = 5
    missing_sheet_block_size: int = 8
    max_components: int = 1
    min_largest_component_fraction: float = 0.98
    min_prompt_component_recall: float = 0.98
    max_floating_artifact_fraction: float = 0.01
    floating_artifact_component_min_voxels: int = 100
    max_border_artifact_fraction: float = 0.0
    max_missing_block_fraction: float = 0.25
    max_thickness_excess_fraction: float = 0.5
    distance_backend: str = "cpu_scipy"
    betti_proxy_enabled: bool = True
    kaggle_surface_enabled: bool = False
    kaggle_surface_tolerance_voxels: float = 2.0
    kaggle_surface_connectivity: int = 26
    kaggle_surface_voi_alpha: float = 0.3


def compute_p2sd_metrics(
    pred: np.ndarray,
    gt: np.ndarray,
    *,
    prompt_zyx: tuple[int, int, int] | None = None,
    cfg: P2SDMetricConfig | None = None,
) -> dict[str, float]:
    cfg = cfg or P2SDMetricConfig()
    pred_mask = _as_mask(pred, cfg.threshold)
    gt_mask = _as_mask(gt, 0.5)
    _validate_same_shape(pred_mask, gt_mask)

    metrics: dict[str, float] = {
        "dice": dice(pred_mask, gt_mask),
        "pred_voxels": float(pred_mask.sum()),
        "gt_voxels": float(gt_mask.sum()),
    }

    comp = component_info(pred_mask, prompt_zyx=prompt_zyx)
    for key, value in asdict(comp).items():
        metrics[f"component_{key}"] = float(value)
    # Keep the established component_* fields while exposing the raw count as
    # an explicit fast topology proxy for experiment comparisons.
    metrics["fast_topology_sheet_component_count"] = float(comp.num_components)

    prompt_mask = prompt_connected_mask(pred_mask, prompt_zyx)
    dist_to_gt = (
        distance_transform_edt(~gt_mask, backend=cfg.distance_backend)
        if gt_mask.any()
        else None
    )
    dist_to_pred = (
        distance_transform_edt(~pred_mask, backend=cfg.distance_backend)
        if pred_mask.any()
        else None
    )
    dist_to_prompt = (
        distance_transform_edt(~prompt_mask, backend=cfg.distance_backend)
        if prompt_mask.any()
        else None
    )
    pred_inside = (
        distance_transform_edt(pred_mask, backend=cfg.distance_backend)
        if pred_mask.any()
        else None
    )
    gt_inside = (
        distance_transform_edt(gt_mask, backend=cfg.distance_backend)
        if gt_mask.any()
        else None
    )
    for tau in cfg.tolerance_voxels:
        metrics.update(_tolerant_metrics(
            pred_mask,
            gt_mask,
            tau=tau,
            distance_to_gt=dist_to_gt,
            distance_to_pred=dist_to_pred,
        ))
        metrics[f"floating_artifact_fraction_tau{tau}"] = (
            floating_artifact_fraction(
                pred_mask,
                gt_mask,
                tolerance=tau,
                distance_to_gt=dist_to_gt,
            ))
        metrics[f"floating_artifact_component_count_tau{tau}"] = float(
            floating_artifact_component_count(
                pred_mask,
                gt_mask,
                tolerance=tau,
                distance_to_gt=dist_to_gt,
                min_component_voxels=cfg.floating_artifact_component_min_voxels,
            ))
        metrics[f"prompt_component_gt_recall_tau{tau}"] = tolerant_recall(
            prompt_mask,
            gt_mask,
            tau=tau,
            distance_to_pred=dist_to_prompt,
        )

    metrics["border_artifact_fraction"] = border_artifact_fraction(
        pred_mask, border_width=cfg.erased_border_width)

    miss = missing_sheet_proxy(
        pred_mask,
        gt_mask,
        tolerance=max(cfg.tolerance_voxels),
        block_size=cfg.missing_sheet_block_size,
        distance_to_pred=dist_to_pred,
    )
    metrics.update(miss)
    metrics.update(_hole_proxy(
        pred_mask,
        gt_mask,
        tau=max(cfg.tolerance_voxels),
        distance_to_pred=dist_to_pred,
    ))
    metrics.update(_thickness_stats(pred_mask, prefix="pred", distance_inside=pred_inside))
    metrics.update(_thickness_stats(gt_mask, prefix="gt", distance_inside=gt_inside))
    if cfg.betti_proxy_enabled:
        metrics.update(_betti_metrics(pred_mask, gt_mask))
    metrics.update(composite_subscores(metrics))
    metrics["quality_composite"] = composite_quality(metrics)
    metrics["quality_composite_version"] = COMPOSITE_VERSION
    metrics.update(quality_gates(metrics, cfg))
    metrics.update(failure_severity(metrics))
    if cfg.kaggle_surface_enabled:
        metrics.update(compute_kaggle_prompt_component_metrics(
            pred_mask,
            gt_mask,
            cfg=KaggleSurfaceMetricConfig(
                threshold=0.5,
                surface_tolerance_voxels=cfg.kaggle_surface_tolerance_voxels,
                connectivity=cfg.kaggle_surface_connectivity,
                voi_alpha=cfg.kaggle_surface_voi_alpha,
            ),
        ))
    return metrics


def dice(pred: np.ndarray, gt: np.ndarray) -> float:
    pred = np.asarray(pred, dtype=bool)
    gt = np.asarray(gt, dtype=bool)
    denom = int(pred.sum() + gt.sum())
    if denom == 0:
        return 1.0
    return float(2 * np.logical_and(pred, gt).sum() / denom)


def tolerant_precision(
    pred: np.ndarray,
    gt: np.ndarray,
    *,
    tau: float,
    distance_to_gt: np.ndarray | None = None,
) -> float:
    pred = np.asarray(pred, dtype=bool)
    gt = np.asarray(gt, dtype=bool)
    count = int(pred.sum())
    if count == 0:
        return 1.0 if not gt.any() else 0.0
    if not gt.any():
        return 0.0
    dist_to_gt = (
        np.asarray(distance_to_gt)
        if distance_to_gt is not None
        else ndimage.distance_transform_edt(~gt)
    )
    return float((dist_to_gt[pred] <= tau).mean())


def tolerant_recall(
    pred: np.ndarray,
    gt: np.ndarray,
    *,
    tau: float,
    distance_to_pred: np.ndarray | None = None,
) -> float:
    pred = np.asarray(pred, dtype=bool)
    gt = np.asarray(gt, dtype=bool)
    count = int(gt.sum())
    if count == 0:
        return 1.0
    if not pred.any():
        return 0.0
    dist_to_pred = (
        np.asarray(distance_to_pred)
        if distance_to_pred is not None
        else ndimage.distance_transform_edt(~pred)
    )
    return float((dist_to_pred[gt] <= tau).mean())


COMPOSITE_VERSION = 3.0

# Coverage at or above this earns full substance credit; the surface subscore
# carries the remaining size sensitivity. Below it, the error-absence subscores
# are discounted because the prediction is too small to have been capable of
# committing the errors they measure.
_SUBSTANCE_FULL_CREDIT_RECALL = 0.5


def composite_substance(metrics: dict[str, float]) -> float:
    """How much of the target the prediction actually covers, in [0, 1].

    Version 3 exists because three of the four subscores measure the *absence
    of an error*, and every one of those errors is impossible to commit with no
    prediction. Measured under version 2 on a 48^3 slab:

    ==================  =========
    prediction          composite
    ==================  =========
    perfect              1.000
    tiny (1% of GT)      0.701
    all zeros            0.690
    all ones             0.419
    ==================  =========

    Predicting 1% of a sheet outscored predicting all of it, and the score rose
    monotonically as the prediction shrank -- a smooth incentive to output less.

    Gating on coverage rather than patching each subscore keeps every subscore
    meaning what its name says, and leaves any prediction with >= 50% recall
    numerically identical to version 2. Real runs sit near recall 1.0, so
    historical scores for non-degenerate models are unchanged.
    """

    recall = metrics.get(
        "tolerant_recall_tau2",
        metrics.get("tolerant_recall_tau1", metrics.get("dice", 0.0)),
    )
    return float(np.clip(recall / _SUBSTANCE_FULL_CREDIT_RECALL, 0.0, 1.0))


def composite_quality(metrics: dict[str, float]) -> float:
    """Priority-weighted quality score (version 3).

    Weights derive from the production priority stack, most expensive
    error first:

    - capture (0.35): predicted material far from this GT sheet is very
      likely another sheet merged in — the one error that cannot be amended
      automatically. Steep: 5% floating fraction zeroes the subscore.
    - topology (0.25): single-sheet correctness (one component, no tunnels,
      no cavities). Splits and holes decay the subscore geometrically but,
      per the merge-versus-split asymmetry, one split (~0.10 composite) costs
      well under a saturated capture failure (0.35).
    - surface (0.25): tolerant F1 and prompt-component recall.
    - amendable (0.15): missing blocks, thickness excess, border debris —
      repairable by point re-sampling plus AE reconstruction.

    The weighted sum is then scaled by :func:`composite_substance`, so a
    prediction cannot earn credit for errors it was too small to commit. See
    that function for the version 2 failure this corrects.

    Subscores are exposed as ``score_*`` by :func:`composite_subscores`.
    """
    scores = composite_subscores(metrics)
    weighted = (
        0.35 * scores["score_capture"]
        + 0.25 * scores["score_topology"]
        + 0.25 * scores["score_surface"]
        + 0.15 * scores["score_amendable"]
    )
    return float(np.clip(weighted * scores["score_substance"], 0.0, 1.0))


def composite_subscores(metrics: dict[str, float]) -> dict[str, float]:
    floating = metrics.get(
        "floating_artifact_fraction_tau2",
        metrics.get("floating_artifact_fraction_tau1", 0.0),
    )
    capture = 1.0 - min(floating / 0.05, 1.0)

    if "betti_b0" in metrics:
        components = metrics["betti_b0"]
        splits = max(components - 1.0, 0.0)
        tunnels = metrics.get("betti_b1", 0.0)
        cavities = metrics.get("betti_b2", 0.0)
    else:
        components = metrics.get("component_num_components", 1.0)
        splits = max(components - 1.0, 0.0)
        tunnels = 0.0
        cavities = 0.0
    if components <= 0.0:
        # An empty prediction has no component at all. `max(b0 - 1, 0)` mapped
        # b0=0 onto the same zero-splits value as b0=1, so "predicted nothing"
        # scored identically to "one clean sheet".
        topology = 0.0
    else:
        topology = (
            0.6 ** min(splits, 4.0)
            * 0.8 ** min(tunnels, 5.0)
            * 0.8 ** min(cavities, 5.0)
        )

    f1 = metrics.get("tolerant_f1_tau2", metrics.get("tolerant_f1_tau1", 0.0))
    prompt_recall = metrics.get(
        "prompt_component_gt_recall_tau2",
        metrics.get("prompt_component_gt_recall_tau1", 0.0),
    )
    surface = 0.7 * f1 + 0.3 * prompt_recall

    missing = metrics.get("max_missing_block_fraction", 0.0)
    border = metrics.get("border_artifact_fraction", 0.0)
    thickness_excess = _thickness_excess_fraction(metrics)
    amendable = 1.0 - (
        0.4 * min(missing / 0.5, 1.0)
        + 0.4 * min(thickness_excess / 1.0, 1.0)
        + 0.2 * min(border / 0.05, 1.0)
    )

    return {
        "score_capture": float(np.clip(capture, 0.0, 1.0)),
        "score_topology": float(np.clip(topology, 0.0, 1.0)),
        "score_surface": float(np.clip(surface, 0.0, 1.0)),
        "score_amendable": float(np.clip(amendable, 0.0, 1.0)),
        "score_substance": composite_substance(metrics),
    }


def quality_gates(
    metrics: dict[str, float],
    cfg: P2SDMetricConfig,
) -> dict[str, float]:
    tau = max(cfg.tolerance_voxels)
    prompt_recall = metrics.get(f"prompt_component_gt_recall_tau{tau}", 0.0)
    floating = metrics.get(f"floating_artifact_fraction_tau{tau}", 0.0)
    border = metrics.get("border_artifact_fraction", 0.0)
    largest = metrics.get("component_largest_fraction", 0.0)
    num_components = metrics.get("component_num_components", 0.0)
    missing = metrics.get("max_missing_block_fraction", 0.0)
    thickness_excess = _thickness_excess_fraction(metrics)

    failures = {
        "fail_too_many_components": float(num_components > cfg.max_components),
        "fail_low_largest_component_fraction": float(
            largest < cfg.min_largest_component_fraction),
        "fail_low_prompt_component_recall": float(
            prompt_recall < cfg.min_prompt_component_recall),
        "fail_floating_artifacts": float(
            floating > cfg.max_floating_artifact_fraction),
        "fail_border_artifacts": float(
            border > cfg.max_border_artifact_fraction),
        "fail_missing_sheet": float(
            missing > cfg.max_missing_block_fraction),
        "fail_thickness_excess": float(
            thickness_excess > cfg.max_thickness_excess_fraction),
    }
    if "sheet_topology_defect" in metrics:
        failures["fail_sheet_topology"] = float(metrics["sheet_topology_defect"])
    failures["thickness_excess_fraction"] = float(thickness_excess)
    failures["quality_gate_pass"] = float(
        all(value == 0.0 for key, value in failures.items()
            if key.startswith("fail_")))
    # Severity-aware gate: floating artifacts are the merge/foreign-capture
    # proxy — the one failure that cannot be amended automatically. Everything
    # else is repairable; a sample passing this gate is production-usable
    # after amendment even if strict quality_gate_pass fails.
    failures["hard_gate_pass"] = float(failures["fail_floating_artifacts"] == 0.0)
    return failures


def _betti_metrics(pred_mask: np.ndarray, gt_mask: np.ndarray) -> dict[str, float]:
    """Exact Betti numbers (official V-construction, 6-connected foreground).

    Answers "does this decoding have single-sheet topology": one component,
    no tunnels, no cavities. ``gt_*`` fields flag ground-truth surfaces whose
    generated voxelization itself violates that assumption, so data defects
    are not misread as model failures.
    """
    pred_betti = betti_numbers(pred_mask, construction="V")
    gt_betti = betti_numbers(gt_mask, construction="V")
    return {
        "betti_b0": float(pred_betti[0]),
        "betti_b1": float(pred_betti[1]),
        "betti_b2": float(pred_betti[2]),
        "sheet_topology_defect": float(pred_betti != (1, 0, 0)),
        "gt_betti_b0": float(gt_betti[0]),
        "gt_betti_b1": float(gt_betti[1]),
        "gt_betti_b2": float(gt_betti[2]),
        "gt_topology_defect": float(gt_betti != (1, 0, 0)),
    }


def failure_severity(metrics: dict[str, float]) -> dict[str, float]:
    """Split gate failures by how costly they are to amend downstream.

    Hard failures capture foreign material (likely another sheet merged in);
    those cannot be fixed automatically. Amendable defects are holes, splits,
    missing regions, thickness excess, and border debris, which sparse
    point re-sampling plus AE reconstruction can repair. Does not alter
    ``quality_gate_pass``.
    """
    amendable_keys = (
        "fail_too_many_components",
        "fail_low_largest_component_fraction",
        "fail_missing_sheet",
        "fail_thickness_excess",
        "fail_border_artifacts",
    )
    amendable = any(metrics.get(key, 0.0) for key in amendable_keys)
    amendable = amendable or bool(metrics.get("sheet_topology_defect", 0.0))
    hard = bool(metrics.get("fail_floating_artifacts", 0.0))
    return {
        "hard_failure": float(hard),
        "amendable_defect": float(amendable),
    }


def config_from_mapping(data: dict) -> P2SDMetricConfig:
    tolerances = data.get("tolerance_voxels", (1, 2))
    if isinstance(tolerances, Iterable) and not isinstance(tolerances, (str, bytes)):
        tolerance_tuple = tuple(int(v) for v in tolerances)
    else:
        tolerance_tuple = (int(tolerances),)
    kaggle_surface = data.get("kaggle_surface", {})
    if kaggle_surface is None:
        kaggle_surface = {}
    if not isinstance(kaggle_surface, dict):
        raise ValueError("metrics.kaggle_surface must be a mapping")
    return P2SDMetricConfig(
        threshold=float(data.get("threshold", 0.5)),
        tolerance_voxels=tolerance_tuple,
        erased_border_width=int(data.get("erased_border_width", 5)),
        missing_sheet_block_size=int(data.get("missing_sheet_block_size", 8)),
        max_components=int(data.get("max_components", 1)),
        min_largest_component_fraction=float(
            data.get("min_largest_component_fraction", 0.98)),
        min_prompt_component_recall=float(
            data.get("min_prompt_component_recall", 0.98)),
        max_floating_artifact_fraction=float(
            data.get("max_floating_artifact_fraction", 0.01)),
        floating_artifact_component_min_voxels=int(
            data.get("floating_artifact_component_min_voxels", 100)),
        max_border_artifact_fraction=float(
            data.get("max_border_artifact_fraction", 0.0)),
        max_missing_block_fraction=float(
            data.get("max_missing_block_fraction", 0.25)),
        max_thickness_excess_fraction=float(
            data.get("max_thickness_excess_fraction", 0.5)),
        distance_backend=str(data.get("distance_backend", "cpu_scipy")),
        betti_proxy_enabled=bool(data.get("betti_proxy_enabled", True)),
        kaggle_surface_enabled=bool(kaggle_surface.get("enabled", False)),
        kaggle_surface_tolerance_voxels=float(
            kaggle_surface.get("surface_tolerance_voxels", 2.0)),
        kaggle_surface_connectivity=int(kaggle_surface.get("connectivity", 26)),
            kaggle_surface_voi_alpha=float(kaggle_surface.get("voi_alpha", 0.3)),
    )


def _tolerant_metrics(
    pred: np.ndarray,
    gt: np.ndarray,
    *,
    tau: int,
    distance_to_gt: np.ndarray | None = None,
    distance_to_pred: np.ndarray | None = None,
) -> dict[str, float]:
    precision = tolerant_precision(pred, gt, tau=tau, distance_to_gt=distance_to_gt)
    recall = tolerant_recall(pred, gt, tau=tau, distance_to_pred=distance_to_pred)
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    return {
        f"tolerant_precision_tau{tau}": precision,
        f"tolerant_recall_tau{tau}": recall,
        f"tolerant_f1_tau{tau}": float(f1),
    }


def _hole_proxy(
    pred: np.ndarray,
    gt: np.ndarray,
    *,
    tau: int,
    distance_to_pred: np.ndarray | None = None,
) -> dict[str, float]:
    if not gt.any():
        return {"hole_component_count": 0.0, "max_hole_fraction": 0.0}
    if pred.any():
        dist_to_pred = (
            np.asarray(distance_to_pred)
            if distance_to_pred is not None
            else ndimage.distance_transform_edt(~pred)
        )
        uncovered = gt & (dist_to_pred > tau)
    else:
        uncovered = gt.copy()
    labels, count = ndimage.label(
        uncovered,
        structure=ndimage.generate_binary_structure(3, 3),
    )
    if count == 0:
        return {"hole_component_count": 0.0, "max_hole_fraction": 0.0}
    sizes = np.bincount(labels.ravel(), minlength=count + 1)
    sizes[0] = 0
    return {
        "hole_component_count": float(count),
        "max_hole_fraction": float(sizes.max() / max(int(gt.sum()), 1)),
    }


def _thickness_stats(
    mask: np.ndarray,
    *,
    prefix: str,
    distance_inside: np.ndarray | None = None,
) -> dict[str, float]:
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        return {
            f"{prefix}_mean_radius": 0.0,
            f"{prefix}_max_radius": 0.0,
        }
    dist_inside = (
        np.asarray(distance_inside)
        if distance_inside is not None
        else ndimage.distance_transform_edt(mask)
    )
    vals = dist_inside[mask]
    return {
        f"{prefix}_mean_radius": float(vals.mean()),
        f"{prefix}_max_radius": float(vals.max()),
    }


def _thickness_excess_fraction(metrics: dict[str, float]) -> float:
    pred_radius = metrics.get("pred_mean_radius", 0.0)
    gt_radius = metrics.get("gt_mean_radius", 0.0)
    return max(pred_radius - gt_radius, 0.0) / max(gt_radius, 1e-6)


def _as_mask(x: np.ndarray, threshold: float) -> np.ndarray:
    arr = np.asarray(x)
    if arr.dtype == bool:
        return arr
    return arr > threshold


def _validate_same_shape(a: np.ndarray, b: np.ndarray) -> None:
    if a.shape != b.shape:
        raise ValueError(f"Shape mismatch: {a.shape} vs {b.shape}")
    if a.ndim != 3:
        raise ValueError(f"Expected 3D masks, got shape {a.shape}")
