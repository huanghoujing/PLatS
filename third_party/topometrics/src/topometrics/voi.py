# voi.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Tuple, Optional
import numpy as np

import cc3d
from skimage.metrics import variation_of_information

# Internal heuristic: crop VOI to union-FG bbox if bbox volume <= RATIO * full volume
_VOI_CROP_RATIO = 0.7  # tuneable; exact score preserved, speeds up when FG is localized


@dataclass
class VOIReport:
    voi_total: float         # voi_split + voi_merge (lower is better)
    voi_split: float         # H(GT | PR): over-segmentation term
    voi_merge: float         # H(PR | GT): under-segmentation term
    voi_score: float         # mapped to [0,1], higher is better
    n_foreground: int        # voxels considered (union of FG if use_union_mask=True)
    connectivity: int        # 6/18/26 (3D only)


def _ensure_3d_same_shape(pr: np.ndarray, gt: np.ndarray) -> None:
    if pr.shape != gt.shape:
        raise ValueError(f"predictions and labels must have identical shape; got {pr.shape} vs {gt.shape}")
    if pr.ndim != 3:
        raise ValueError(f"VOI expects 3D arrays; got ndim={pr.ndim}")


def _bbox3d(mask: np.ndarray) -> Optional[Tuple[slice, slice, slice]]:
    assert mask.ndim == 3
    sls = []
    for ax in range(3):
        proj = mask.any(axis=tuple(i for i in range(3) if i != ax))
        idx = np.flatnonzero(proj)
        if idx.size == 0:
            return None
        sls.append(slice(int(idx[0]), int(idx[-1]) + 1))
    return (sls[0], sls[1], sls[2])


def _union_bbox3d(a: np.ndarray, b: np.ndarray) -> Optional[Tuple[slice, slice, slice]]:
    return _bbox3d(a | b)


def _build_ignore_mask(
    labels_arr: np.ndarray,
    *,
    ignore_mask: Optional[Any],
    ignore_label: Optional[float | int],
) -> np.ndarray:
    """
    Returns a boolean mask of voxels to ignore (shape == labels_arr.shape).
    Priority: explicit ignore_mask if given; else labels == ignore_label; else no ignore.
    """
    if ignore_mask is not None:
        m = np.asarray(ignore_mask).astype(bool)
        if m.shape != labels_arr.shape:
            raise ValueError(f"ignore_mask shape {m.shape} must match labels shape {labels_arr.shape}")
        return m
    if ignore_label is None:
        return np.zeros(labels_arr.shape, dtype=bool)
    return labels_arr == ignore_label


def compute_voi_metrics(
    predictions: Any,
    labels: Any,
    *,
    connectivity: int = 26,
    use_union_mask: bool = True,
    score_transform: str = "one_over_one_plus",   # or "exp"
    alpha: float = 1.0,                            # strength for the transform
    # ignore controls (defaults match leaderboard)
    ignore_label: Optional[float | int] = 2,
    ignore_mask: Optional[Any] = None,
) -> VOIReport:
    """
    Compute VOI between *connected-component* labelings derived from binary FG masks.
    Any nonzero voxel is foreground by default (caller may pass boolean masks).

    Ignoring: voxels marked ignored (explicit mask or labels == ignore_label) are set to
    background in BOTH predictions and labels before CC labeling and VOI.

    Direction: skimage.metrics.variation_of_information(a, b) returns (H(a|b), H(b|a)).
    We pass a=GT, b=PR so that:
      - voi_split = H(GT | PR)   (over-segmentation: GT split by PR)
      - voi_merge = H(PR | GT)   (under-segmentation: PR merges GT)

    Returns VOIReport with a score in (0,1], higher is better.
    """
    if alpha < 0:
        raise ValueError(f"alpha must be >= 0; got {alpha}")
    if score_transform not in ("one_over_one_plus", "exp"):
        raise ValueError(f"score_transform must be 'one_over_one_plus' or 'exp'; got {score_transform}")
    if connectivity not in (6, 18, 26):
        raise ValueError(f"3D connectivity must be one of (6, 18, 26); got {connectivity}")

    # Nan-safe arrays
    gt_arr = np.nan_to_num(np.asarray(labels), nan=0.0)
    pr_arr = np.nan_to_num(np.asarray(predictions), nan=0.0)
    _ensure_3d_same_shape(pr_arr, gt_arr)

    # Build and apply ignore mask
    ign = _build_ignore_mask(gt_arr, ignore_mask=ignore_mask, ignore_label=ignore_label)
    if ign.any():
        gt_arr = np.where(ign, 0, gt_arr)
        pr_arr = np.where(ign, 0, pr_arr)

    # Binary foreground (supports bool input as well)
    gt_bin = (gt_arr != 0) if gt_arr.dtype != bool else gt_arr
    pr_bin = (pr_arr != 0) if pr_arr.dtype != bool else pr_arr

    # Early out if nothing remains
    union = gt_bin | pr_bin
    if not union.any():
        return VOIReport(0.0, 0.0, 0.0, 1.0, 0, connectivity)

    # Gate cropping by union-bbox ratio for speed
    slc = _union_bbox3d(gt_bin, pr_bin)
    assert slc is not None
    full_n = int(np.prod(gt_bin.shape, dtype=np.int64))
    bbox_n = int((slc[0].stop - slc[0].start) * (slc[1].stop - slc[1].start) * (slc[2].stop - slc[2].start))
    use_crop = (bbox_n <= int(_VOI_CROP_RATIO * full_n))

    gt_use = gt_bin[slc] if use_crop else gt_bin
    pr_use = pr_bin[slc] if use_crop else pr_bin

    # Connected components (3D only)
    gt_lab = cc3d.connected_components(gt_use.astype(np.uint8), connectivity=connectivity)
    pr_lab = cc3d.connected_components(pr_use.astype(np.uint8), connectivity=connectivity)

    if use_union_mask:
        m = (gt_lab > 0) | (pr_lab > 0)
        a = gt_lab[m]
        b = pr_lab[m]
        n_considered = int(m.sum())
    else:
        a = gt_lab.ravel()
        b = pr_lab.ravel()
        n_considered = a.size

    voi_split, voi_merge = variation_of_information(a, b)  # (H(GT|PR), H(PR|GT))
    voi_total = float(voi_split + voi_merge)

    if score_transform == "exp":
        voi_score = float(np.exp(-alpha * voi_total))
    else:
        voi_score = float(1.0 / (1.0 + alpha * voi_total))

    return VOIReport(
        voi_total=voi_total,
        voi_split=float(voi_split),
        voi_merge=float(voi_merge),
        voi_score=voi_score,
        n_foreground=n_considered,
        connectivity=connectivity,
    )
