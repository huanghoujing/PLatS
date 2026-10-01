"""Exact public Kaggle 2025 Vesuvius Surface Detection metric adapter.

Kaggle's Evaluation page links the public metric notebook
``sohier/vesuvius-2025-metric-demo`` and its
``sohier/vesuvius-metric-resources`` dataset. This module delegates directly
to that resource package's ``topometrics.compute_leaderboard_score`` instead
of maintaining a local approximation.

A P2SD query predicts one selected component, not the full multi-sheet volume
required by a Kaggle submission. The resulting composite is consequently
named ``kaggle_prompt_component_score`` rather than a leaderboard score.
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class KaggleSurfaceMetricConfig:
    """Parameters from the public Kaggle metric notebook."""

    threshold: float = 0.5
    surface_tolerance_voxels: float = 2.0
    connectivity: int = 26
    voi_alpha: float = 0.3
    topo_weight: float = 0.3
    surface_dice_weight: float = 0.35
    voi_weight: float = 0.35


class KaggleMetricUnavailableError(RuntimeError):
    """Raised when Kaggle's public metric resource package is not installed."""


REFERENCE_BATCH_BACKEND = "reference_batch"
COUNT_ONLY_BACKEND = "count_only"
BINARY_EXACT_BACKEND = "binary_exact"
COMPACT_EXACT_BACKEND = "compact_exact"
TOPOLOGY_BACKENDS = (
    REFERENCE_BATCH_BACKEND,
    COUNT_ONLY_BACKEND,
    BINARY_EXACT_BACKEND,
    COMPACT_EXACT_BACKEND,
)


def compute_kaggle_prompt_component_metrics(
    pred: np.ndarray,
    gt: np.ndarray,
    *,
    cfg: KaggleSurfaceMetricConfig | None = None,
) -> dict[str, float]:
    """Score one prompted P2SD sheet with the published Kaggle implementation.

    P2SD targets are binary selected-sheet masks. Predictions are thresholded
    before calling the reference package, matching the competition's submitted
    integer TIFF masks and its ``fg_threshold=None`` legacy behavior.
    """

    cfg = cfg or KaggleSurfaceMetricConfig()
    pred_mask = _as_mask(pred, cfg.threshold)
    gt_mask = _as_mask(gt, 0.5)
    _validate_same_shape(pred_mask, gt_mask)
    _validate_public_config(cfg)

    reference = _load_reference_metric()
    report = reference.compute_leaderboard_score(
        predictions=pred_mask.astype(np.uint8),
        labels=gt_mask.astype(np.uint8),
        dims=(0, 1, 2),
        spacing=(1.0, 1.0, 1.0),
        surface_tolerance=cfg.surface_tolerance_voxels,
        voi_connectivity=cfg.connectivity,
        voi_transform="one_over_one_plus",
        voi_alpha=cfg.voi_alpha,
        combine_weights=(cfg.topo_weight, cfg.surface_dice_weight, cfg.voi_weight),
        fg_threshold=None,
        ignore_label=2,
        ignore_mask=None,
    )
    # gt may carry the public ignore label 2, which _as_mask would otherwise
    # count as foreground. Drop those voxels from both masks so the volumetric
    # Dice matches the region the reference scorer actually evaluated.
    ignore = reference._build_ignore_mask(gt, ignore_mask=None, ignore_label=2)
    metrics = _metrics_from_report(
        report,
        score_key="kaggle_prompt_component_score",
        pred_bin=np.where(ignore, False, pred_mask),
        gt_bin=np.where(ignore, False, gt_mask),
    )
    return metrics


def compute_kaggle_case_metrics_batch(
    predictions: Sequence[np.ndarray],
    labels: Sequence[np.ndarray],
    *,
    cfg: KaggleSurfaceMetricConfig | None = None,
    topology_backend: str = REFERENCE_BATCH_BACKEND,
    topology_tile_batch_size: int | None = None,
) -> list[dict[str, float]]:
    """Score a bounded batch of full case volumes with exact public metrics.

    The public scorer evaluates each 3D volume as eight independent topology
    tiles. Its C++ binding also accepts a batch of independent volume pairs.
    This function batches those *unchanged* tiles, then reuses the public
    aggregation, Surface Dice, and VOI operations. It is intended for
    case-level scene evaluation, where process-pickling one PS320 volume per
    prompt variant was otherwise substantial overhead. ``reference_batch``
    is the default public binding path. ``count_only`` returns only the three
    values TopoScore uses per dimension. ``binary_exact`` is a stricter
    uint8/binary-only variant with compact 3D matching records.

    ``labels`` may contain the public ignore label ``2``. This lets a
    thresholded scene contract exclude sheets below its configured size gate
    without relabelling them as background.
    """

    cfg = cfg or KaggleSurfaceMetricConfig()
    _validate_public_config(cfg)
    if topology_backend not in TOPOLOGY_BACKENDS:
        raise ValueError(
            f"topology_backend must be one of {TOPOLOGY_BACKENDS}, got {topology_backend!r}")
    if topology_tile_batch_size is not None and topology_tile_batch_size <= 0:
        raise ValueError("topology_tile_batch_size must be positive or None")
    if len(predictions) != len(labels):
        raise ValueError(
            "Predictions and labels must have the same batch length, got "
            f"{len(predictions)} and {len(labels)}")
    if not predictions:
        return []

    reference = _load_reference_metric()
    prepared = [
        _prepare_public_case_inputs(prediction, label, reference=reference)
        for prediction, label in zip(predictions, labels, strict=True)
    ]
    topo_reports = _compute_batched_topology_reports(
        prepared,
        cfg=cfg,
        reference=reference,
        topology_backend=topology_backend,
        topology_tile_batch_size=topology_tile_batch_size,
    )
    return [
        _case_metrics_from_prepared(item, topo=topo, cfg=cfg, reference=reference)
        for item, topo in zip(prepared, topo_reports, strict=True)
    ]


def _validate_public_config(cfg: KaggleSurfaceMetricConfig) -> None:
    if cfg.connectivity != 26:
        raise ValueError(
            "Kaggle Surface Detection uses 26-connected VOI; "
            f"got connectivity={cfg.connectivity}.")
    if cfg.surface_tolerance_voxels != 2.0:
        raise ValueError(
            "Kaggle's public metric notebook uses surface_tolerance=2.0; "
            f"got {cfg.surface_tolerance_voxels}.")
    if cfg.voi_alpha != 0.3:
        raise ValueError(
            "Kaggle's public metric notebook uses voi_alpha=0.3; "
            f"got {cfg.voi_alpha}.")
    if (cfg.topo_weight, cfg.surface_dice_weight, cfg.voi_weight) != (0.3, 0.35, 0.35):
        raise ValueError(
            "Kaggle's public metric notebook uses weights "
            "(0.3, 0.35, 0.35).")


def _volumetric_dice(pred_bin: np.ndarray, gt_bin: np.ndarray) -> float:
    """Plain volumetric Dice on the scorer's own prepared masks.

    Defined locally rather than imported from ``eval.metrics``, which imports
    this module (importing back would be circular). Callers must pass masks
    that already have the ignore label removed, so this agrees voxel-for-voxel
    with the Surface Dice reported alongside it.
    """

    pred = np.asarray(pred_bin, dtype=bool)
    gt = np.asarray(gt_bin, dtype=bool)
    denom = int(pred.sum() + gt.sum())
    if denom == 0:
        return 1.0
    return float(2 * np.logical_and(pred, gt).sum() / denom)


def _metrics_from_report(
    report,
    *,
    score_key: str,
    pred_bin: np.ndarray,
    gt_bin: np.ndarray,
) -> dict[str, float]:
    """Assemble the reported metric row.

    ``kaggle_volumetric_dice`` is plain overlap Dice. It is NOT part of the
    published composite score; it is reported alongside
    ``kaggle_surface_dice_tau2`` because the two answer different questions —
    surface Dice tolerates a 2-voxel boundary offset, volumetric Dice does not,
    so a large gap between them localizes error to the boundary.
    """

    metrics = {
        score_key: float(report.score),
        "kaggle_toposcore": float(report.topo.toposcore),
        "kaggle_surface_dice_tau2": float(report.surface_dice),
        "kaggle_volumetric_dice": _volumetric_dice(pred_bin, gt_bin),
        "kaggle_voi_total": float(report.voi.voi_total),
        "kaggle_voi_split": float(report.voi.voi_split),
        "kaggle_voi_merge": float(report.voi.voi_merge),
        "kaggle_voi_score": float(report.voi.voi_score),
    }
    # The reference scorer uses NaN for an inactive homology dimension. Omit
    # it from JSON metric rows rather than propagate a non-finite value through
    # the normal validation aggregation.
    for dimension, value in report.topo.topoF1_by_dim.items():
        if np.isfinite(value):
            metrics[f"kaggle_topof1_dim{dimension}"] = float(value)
    return metrics


@dataclass(frozen=True)
class _PreparedPublicCase:
    pred_bin: np.ndarray
    gt_bin: np.ndarray


def _prepare_public_case_inputs(
    prediction: np.ndarray,
    label: np.ndarray,
    *,
    reference,
) -> _PreparedPublicCase:
    """Apply the public scorer's ignore and binarization rules unchanged."""

    pred_raw = np.asarray(prediction)
    gt_raw = np.asarray(label)
    reference._ensure_3d_same_shape(pred_raw, gt_raw)
    ignore = reference._build_ignore_mask(
        gt_raw,
        ignore_mask=None,
        ignore_label=2,
    )
    pred_eval = np.where(ignore, 0, pred_raw)
    gt_eval = np.where(ignore, 0, gt_raw)
    return _PreparedPublicCase(
        pred_bin=reference._nan_safe_binarize(pred_eval, threshold=None),
        gt_bin=reference._nan_safe_binarize(gt_eval, threshold=None),
    )


def _compute_batched_topology_reports(
    prepared: Sequence[_PreparedPublicCase],
    *,
    cfg: KaggleSurfaceMetricConfig,
    reference,
    topology_backend: str,
    topology_tile_batch_size: int | None,
) -> list[object]:
    """Use the reference C++ batch API without changing any topology tile."""

    topology = import_module("topometrics.toposcore")
    betti_matching = import_module("betti_matching")
    flat_predictions: list[np.ndarray] = []
    flat_labels: list[np.ndarray] = []
    case_tile_ranges: list[tuple[int, int]] = []
    for item in prepared:
        topo_prediction = (~item.pred_bin).astype(np.uint8, copy=False)
        topo_label = (~item.gt_bin).astype(np.uint8, copy=False)
        start = len(flat_predictions)
        for tile in reference._octant_slices(topo_prediction.shape, (2, 2, 2)):
            flat_predictions.append(np.ascontiguousarray(topo_prediction[tile]))
            flat_labels.append(np.ascontiguousarray(topo_label[tile]))
        case_tile_ranges.append((start, len(flat_predictions)))

    matcher_batch_size = topology_tile_batch_size or len(flat_predictions)
    tile_reports = []
    for start in range(0, len(flat_predictions), matcher_batch_size):
        stop = min(start + matcher_batch_size, len(flat_predictions))
        if topology_backend == REFERENCE_BATCH_BACKEND:
            tile_reports.extend(
                topology.TopoScore._from_result(result, (0, 1, 2), None)
                for result in betti_matching.compute_matching(
                    flat_predictions[start:stop],
                    flat_labels[start:stop],
                )
            )
        else:
            selected_backend = topology_backend
            # Four-byte cells cover coordinates through 511, including the
            # temporary boundary at shape. Larger public octants retain the
            # established 64-bit exact implementation rather than truncating.
            if topology_backend == COMPACT_EXACT_BACKEND and any(
                max(volume.shape) >= 512
                for volume in flat_predictions[start:stop]
            ):
                selected_backend = BINARY_EXACT_BACKEND
            count_backend = _load_count_backend(selected_backend)
            tile_reports.extend(_topology_reports_from_count_arrays(
                count_backend.compute_matching_counts(
                    flat_predictions[start:stop],
                    flat_labels[start:stop],
                ),
                topology=topology,
            ))
    reports = []
    for start, stop in case_tile_ranges:
        reports.append(topology.TopoScore.aggregate_reports(
            tile_reports[start:stop],
            dims=(0, 1, 2),
            weights=None,
        ))
    return reports


def _topology_reports_from_count_arrays(count_arrays, *, topology) -> list[object]:
    reports = []
    for count_array in count_arrays:
        counts = np.asarray(count_array, dtype=np.int64)
        if counts.shape != (3, 3):
            raise RuntimeError(
                "Count-only Betti backend must return [dimension, "
                "(matched, input1_unmatched, input2_unmatched)], got "
                f"{counts.shape}")
        reports.append(topology.TopoScore.from_counts(
            {
                dimension: (
                    int(counts[dimension, 0]),
                    int(counts[dimension, 0] + counts[dimension, 1]),
                    int(counts[dimension, 0] + counts[dimension, 2]),
                )
                for dimension in range(3)
            },
            dims=(0, 1, 2),
            weights=None,
        ))
    return reports


def _load_count_only_backend():
    try:
        return import_module("betti_matching_count_only")
    except ModuleNotFoundError as exc:
        raise KaggleMetricUnavailableError(
            "The exact count-only Betti backend is not installed. Run "
            "`bash scripts/install_betti_matching_count_only.sh`.") from exc


def _load_binary_exact_backend():
    try:
        return import_module("betti_matching_binary_exact")
    except ModuleNotFoundError as exc:
        raise KaggleMetricUnavailableError(
            "The binary-exact Betti backend is not installed. Run "
            "`bash scripts/install_betti_matching_binary_exact.sh`.") from exc


def _load_count_backend(topology_backend: str):
    if topology_backend == COUNT_ONLY_BACKEND:
        return _load_count_only_backend()
    if topology_backend == BINARY_EXACT_BACKEND:
        return _load_binary_exact_backend()
    if topology_backend == COMPACT_EXACT_BACKEND:
        try:
            return import_module("betti_matching_compact_exact")
        except ModuleNotFoundError as exc:
            raise KaggleMetricUnavailableError(
                "The compact-exact Betti backend is not installed. Run "
                "`bash scripts/install_betti_matching_compact_exact.sh`.") from exc
    raise ValueError(f"No count backend for {topology_backend!r}")


def _case_metrics_from_prepared(
    item: _PreparedPublicCase,
    *,
    topo,
    cfg: KaggleSurfaceMetricConfig,
    reference,
) -> dict[str, float]:
    """Apply the public scorer's Surface Dice, VOI, and composite equations."""

    surface_distance = import_module("surface_distance")
    voi_module = import_module("topometrics.voi")
    spacing = reference._normalize_spacing3d((1.0, 1.0, 1.0))
    if not item.gt_bin.any() and not item.pred_bin.any():
        surface_dice = 1.0
    elif item.gt_bin.any() ^ item.pred_bin.any():
        surface_dice = 0.0
    else:
        distances = surface_distance.compute_surface_distances(
            item.gt_bin.astype(bool),
            item.pred_bin.astype(bool),
            spacing,
        )
        surface_dice = float(surface_distance.compute_surface_dice_at_tolerance(
            distances,
            cfg.surface_tolerance_voxels,
        ))
    voi = voi_module.compute_voi_metrics(
        predictions=item.pred_bin,
        labels=item.gt_bin,
        connectivity=cfg.connectivity,
        use_union_mask=True,
        score_transform="one_over_one_plus",
        alpha=cfg.voi_alpha,
        ignore_label=None,
        ignore_mask=None,
    )
    topo_weight, surface_weight, voi_weight = reference._clamp_and_normalize_weights(
        (cfg.topo_weight, cfg.surface_dice_weight, cfg.voi_weight))
    score = (
        topo_weight * float(topo.toposcore)
        + surface_weight * surface_dice
        + voi_weight * float(voi.voi_score)
    )

    class _Report:
        def __init__(self) -> None:
            self.score = score
            self.topo = topo
            self.surface_dice = surface_dice
            self.voi = voi

    return _metrics_from_report(
        _Report(),
        score_key="kaggle_case_score",
        pred_bin=item.pred_bin,
        gt_bin=item.gt_bin,
    )


def _load_reference_metric():
    try:
        return import_module("topometrics.leaderboard")
    except ModuleNotFoundError as exc:
        raise KaggleMetricUnavailableError(
            "Kaggle's published Vesuvius metric package is required. Run "
            "`bash scripts/install_kaggle_vesuvius_metric.sh`.") from exc


def _as_mask(value: np.ndarray, threshold: float) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != 3:
        raise ValueError(f"Expected a [D,H,W] volume, got shape={array.shape}.")
    return array >= float(threshold)


def _validate_same_shape(left: np.ndarray, right: np.ndarray) -> None:
    if left.shape != right.shape:
        raise ValueError(
            "Prediction and target must share a shape, got "
            f"{left.shape} and {right.shape}.")
