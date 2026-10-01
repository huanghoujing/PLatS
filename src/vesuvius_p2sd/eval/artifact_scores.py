"""Artifact-sensitive scores for P2SD predictions."""

from __future__ import annotations

import numpy as np
from scipy import ndimage


def border_mask(shape: tuple[int, int, int], width: int) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    if width <= 0:
        return mask
    for axis in range(3):
        sl = [slice(None)] * 3
        sl[axis] = slice(0, width)
        mask[tuple(sl)] = True
        sl[axis] = slice(-width, None)
        mask[tuple(sl)] = True
    return mask


def floating_artifact_fraction(
    pred: np.ndarray,
    gt: np.ndarray,
    *,
    tolerance: float,
    distance_to_gt: np.ndarray | None = None,
) -> float:
    pred = np.asarray(pred, dtype=bool)
    gt = np.asarray(gt, dtype=bool)
    pred_count = int(pred.sum())
    if pred_count == 0:
        return 0.0
    if not gt.any():
        return 1.0
    dist_to_gt = (
        np.asarray(distance_to_gt)
        if distance_to_gt is not None
        else ndimage.distance_transform_edt(~gt)
    )
    floating = pred & (dist_to_gt > tolerance)
    return float(floating.sum() / pred_count)


def floating_artifact_component_count(
    pred: np.ndarray,
    gt: np.ndarray,
    *,
    tolerance: float,
    distance_to_gt: np.ndarray | None = None,
    min_component_voxels: int = 100,
) -> int:
    """Count predicted components with no voxel within tolerance of the GT.

    This is complementary to floating_artifact_fraction: a distant spur joined
    to the main sheet contributes to the fraction but is not a detached
    component, whereas multiple sufficiently large floating islands are counted
    separately. The default 100-voxel threshold ignores tiny specks.
    """
    pred = np.asarray(pred, dtype=bool)
    gt = np.asarray(gt, dtype=bool)
    labels, count = ndimage.label(pred, structure=ndimage.generate_binary_structure(3, 3))
    if count == 0:
        return 0
    sizes = np.bincount(labels.ravel(), minlength=count + 1)
    label_ids = np.arange(1, count + 1)
    eligible = sizes[label_ids] >= max(0, int(min_component_voxels))
    if not gt.any():
        return int(eligible.sum())
    dist_to_gt = (
        np.asarray(distance_to_gt)
        if distance_to_gt is not None
        else ndimage.distance_transform_edt(~gt)
    )
    touching_labels = np.unique(labels[pred & (dist_to_gt <= tolerance)])
    floating = ~np.isin(label_ids, touching_labels)
    return int((eligible & floating).sum())


def border_artifact_fraction(pred: np.ndarray, *, border_width: int) -> float:
    pred = np.asarray(pred, dtype=bool)
    pred_count = int(pred.sum())
    if pred_count == 0 or border_width <= 0:
        return 0.0
    border = border_mask(pred.shape, border_width)
    return float((pred & border).sum() / pred_count)


def missing_sheet_proxy(
    pred: np.ndarray,
    gt: np.ndarray,
    *,
    tolerance: float,
    block_size: int,
    distance_to_pred: np.ndarray | None = None,
) -> dict[str, float]:
    pred = np.asarray(pred, dtype=bool)
    gt = np.asarray(gt, dtype=bool)
    if not gt.any():
        return {
            "missing_block_count": 0.0,
            "max_missing_block_fraction": 0.0,
            "mean_missing_block_fraction": 0.0,
        }

    if pred.any():
        dist_to_pred = (
            np.asarray(distance_to_pred)
            if distance_to_pred is not None
            else ndimage.distance_transform_edt(~pred)
        )
        covered_gt = gt & (dist_to_pred <= tolerance)
    else:
        covered_gt = np.zeros_like(gt, dtype=bool)

    missing_fractions: list[float] = []
    zdim, ydim, xdim = gt.shape
    for z0 in range(0, zdim, block_size):
        for y0 in range(0, ydim, block_size):
            for x0 in range(0, xdim, block_size):
                sl = (
                    slice(z0, min(z0 + block_size, zdim)),
                    slice(y0, min(y0 + block_size, ydim)),
                    slice(x0, min(x0 + block_size, xdim)),
                )
                gt_block = gt[sl]
                gt_count = int(gt_block.sum())
                if gt_count == 0:
                    continue
                covered = int(covered_gt[sl].sum())
                missing_fractions.append(1.0 - covered / gt_count)

    if not missing_fractions:
        return {
            "missing_block_count": 0.0,
            "max_missing_block_fraction": 0.0,
            "mean_missing_block_fraction": 0.0,
        }
    arr = np.asarray(missing_fractions, dtype=np.float64)
    return {
        "missing_block_count": float((arr > 0.5).sum()),
        "max_missing_block_fraction": float(arr.max()),
        "mean_missing_block_fraction": float(arr.mean()),
    }
