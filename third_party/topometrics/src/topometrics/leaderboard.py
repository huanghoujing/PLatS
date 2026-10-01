# leaderboard.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple, List, Sequence
import numpy as np
import surface_distance as sd

from .toposcore import TopoScore, TopoReport
from .voi import compute_voi_metrics, VOIReport


@dataclass
class LeaderboardReport:
    score: float
    topo: TopoReport
    surface_dice: float
    voi: VOIReport
    params: Dict[str, Any]


# ---------- Internals ----------

def _nan_safe_binarize(x: Any, *, threshold: Optional[float]) -> np.ndarray:
    """
    Nan-safe binarization. If threshold is None, legacy behavior "!= 0" is used,
    but NaNs are treated as background (converted to 0 before comparison).
    If threshold is a float, returns (x > threshold).
    """
    a = np.asarray(x)
    a = np.nan_to_num(a, nan=0.0)
    if threshold is None:
        return a != 0
    return a > float(threshold)


def _ensure_3d_same_shape(pr: np.ndarray, gt: np.ndarray) -> None:
    if pr.shape != gt.shape:
        raise ValueError(f"predictions and labels must have identical shape; got {pr.shape} vs {gt.shape}")
    if pr.ndim != 3:
        raise ValueError(f"Leaderboard expects 3D arrays; got ndim={pr.ndim}")


def _normalize_spacing3d(spacing: Sequence[float] | float) -> Tuple[float, float, float]:
    """
    Ensure spacing is a 3-tuple of floats. Cropping does not change spacing.
    """
    if np.isscalar(spacing):
        v = float(spacing)
        return (v, v, v)
    s = list(spacing)
    if len(s) >= 3:
        return (float(s[0]), float(s[1]), float(s[2]))
    s = s + [1.0] * (3 - len(s))
    return (float(s[0]), float(s[1]), float(s[2]))

def _clamp_and_normalize_weights(w: Sequence[float]) -> Tuple[float, float, float]:
    w = [0.0 if (not np.isfinite(x) or x < 0) else float(x) for x in w]
    s = sum(w)
    if s <= 0:
        return (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0)
    return (w[0] / s, w[1] / s, w[2] / s)


def _build_ignore_mask(
    labels: np.ndarray,
    *,
    ignore_mask: Optional[Any],
    ignore_label: Optional[float | int],
) -> np.ndarray:
    """
    Returns a boolean mask of voxels to ignore (shape == labels.shape).
    Priority: explicit ignore_mask if given; else labels == ignore_label; else no ignore.
    """
    if ignore_mask is not None:
        m = np.asarray(ignore_mask).astype(bool)
        if m.shape != labels.shape:
            raise ValueError(f"ignore_mask shape {m.shape} must match labels shape {labels.shape}")
        return m
    if ignore_label is None:
        return np.zeros(labels.shape, dtype=bool)
    # Numeric comparison; labels are typically integers
    return np.asarray(labels) == ignore_label

def _axis_cuts(n: int, parts: int) -> List[Tuple[int, int]]:
    """
    Split [0, n) into 'parts' contiguous chunks, sizes differ by at most 1.
    Clamps parts to avoid zero-sized slices when n < parts.
    """
    parts = max(1, min(parts, n if n > 0 else 1))
    base = n // parts
    extra = n % parts
    cuts: List[Tuple[int, int]] = []
    start = 0
    for i in range(parts):
        size = base + (1 if i < extra else 0)
        end = start + size
        cuts.append((start, end))
        start = end
    return cuts

def _octant_slices(shape: Tuple[int, int, int], splits: Tuple[int, int, int]) -> List[Tuple[slice, slice, slice]]:
    zcuts = _axis_cuts(shape[0], splits[0])
    ycuts = _axis_cuts(shape[1], splits[1])
    xcuts = _axis_cuts(shape[2], splits[2])
    out: List[Tuple[slice, slice, slice]] = []
    for z0, z1 in zcuts:
        for y0, y1 in ycuts:
            for x0, x1 in xcuts:
                if (z1 - z0) > 0 and (y1 - y0) > 0 and (x1 - x0) > 0:
                    out.append((slice(z0, z1), slice(y0, y1), slice(x0, x1)))
    return out

# ---------- Public API ----------

def compute_leaderboard_score(
    predictions: Any,
    labels: Any,
    *,
    # Topology
    dims: Optional[Iterable[int]] = (0, 1, 2),
    topo_weights: Optional[Iterable[float] | Mapping[int, float]] = None,  # defaults inside TopoScore (0.34,0.33,0.33)

    # Surface Dice
    surface_tolerance: float = 2.0,               # “voxels”, given spacing=(1,1,1)
    spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),

    # VOI
    voi_connectivity: int = 26,
    voi_transform: str = "one_over_one_plus",     # or "exp"
    voi_alpha: float = 1.0,

    # Combination weights (Topo, SurfaceDice, VOI)
    combine_weights: Tuple[float, float, float] = (0.3, 0.35, 0.35),

    # (optional) binarization for safety; None keeps legacy "!=0" semantics
    fg_threshold: Optional[float] = None,

    # NEW: ignore controls
    ignore_label: Optional[float | int] = 2,      # default ignore label
    ignore_mask: Optional[Any] = None,            # optional explicit boolean mask
) -> LeaderboardReport:
    """
    Returns a single leaderboard scalar plus detailed components:
      score = w_topo * TopoScore + w_surf * SurfaceDice + w_voi * VOI_score

    Notes
    -----
    * Inputs are assumed to be 3D single-channel volumes with identical shapes.
    * Voxels marked ignored (by mask or labels == ignore_label) are set to background in both PR and GT.
    * Surface Dice is computed exactly on a union-FG crop (translation-invariant).
    * VOI is computed on binarized masks via 3D connected components (26-conn by default).
    """
    # Arrays + basic checks
    pr_raw = np.asarray(predictions)
    gt_raw = np.asarray(labels)
    _ensure_3d_same_shape(pr_raw, gt_raw)

    # Build ignore mask (priority: explicit mask > label value)
    ign = _build_ignore_mask(gt_raw, ignore_mask=ignore_mask, ignore_label=ignore_label)
    n_ignore = int(ign.sum())

    # Apply ignore: neutralize by setting both PR and GT to background at ignored voxels
    # (Done on float/int volumes before any binarization or topology)
    pr_eval = np.where(ign, 0, pr_raw)
    gt_eval = np.where(ign, 0, gt_raw)

    # 1) Binarize once for all metrics (threshold=None => legacy "!= 0")
    pr_bin = _nan_safe_binarize(pr_eval, threshold=fg_threshold)
    gt_bin = _nan_safe_binarize(gt_eval, threshold=fg_threshold)

    # 1a) Topology (higher better): invert so FG=0, BG=1
    #     This makes H0 == foreground components (via sublevel filtration).
    topo_pr = (~pr_bin).astype(np.uint8)
    topo_gt = (~gt_bin).astype(np.uint8)

    # HARD-CODED: tile TopoScore into 8 subchunks (half along each axis) to reduce RAM.
    # We only aggregate counts and apply weights once, to preserve semantics.
    _SPLITS = (2, 2, 2)
    slices = _octant_slices(topo_pr.shape, _SPLITS)
    tile_reports: List[TopoReport] = []
    for zs, ys, xs in slices:
        tr = TopoScore.compute(
            predictions=topo_pr[zs, ys, xs],
            labels=topo_gt[zs, ys, xs],
            dims=dims,
            weights=None,  # weights applied once after aggregation
        )
        tile_reports.append(tr)
    topo = TopoScore.aggregate_reports(tile_reports, dims=dims, weights=topo_weights)

    # 2) Surface Dice on binary FG masks (higher better)
    sp = _normalize_spacing3d(spacing)

    if not gt_bin.any() and not pr_bin.any():
        surface_dice = 1.0
    elif gt_bin.any() ^ pr_bin.any():
        surface_dice = 0.0
    else:
        sdists = sd.compute_surface_distances(gt_bin.astype(bool), pr_bin.astype(bool), sp)
        surface_dice = float(sd.compute_surface_dice_at_tolerance(sdists, surface_tolerance))

    # 3) VOI (lower better) -> map to [0,1] score (higher better)
    # We pass binarized, already-ignored masks to VOI for consistency.
    voi = compute_voi_metrics(
        predictions=pr_bin,
        labels=gt_bin,
        connectivity=voi_connectivity,
        use_union_mask=True,
        score_transform=voi_transform,
        alpha=voi_alpha,
        # Since we've already applied ignore, turn off in VOI to avoid reprocessing
        ignore_label=None,
        ignore_mask=None,
    )

    # 4) Combine (clamp and renormalize weights)
    w_topo, w_surf, w_voi = _clamp_and_normalize_weights(combine_weights)
    score = w_topo * float(topo.toposcore) + w_surf * surface_dice + w_voi * float(voi.voi_score)

    return LeaderboardReport(
        score=score,
        topo=topo,
        surface_dice=surface_dice,
        voi=voi,
        params=dict(
            dims=list(dims) if dims is not None else None,
            topo_weights=topo_weights,
            fg_threshold=fg_threshold,
            surface_tolerance=surface_tolerance,
            spacing=sp,
            voi_connectivity=voi_connectivity,
            voi_transform=voi_transform,
            voi_alpha=voi_alpha,
            combine_weights=(w_topo, w_surf, w_voi),
            ignore_label=ignore_label,
            ignored_voxels=n_ignore,
        ),
    )
