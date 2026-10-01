"""Instance-level matching metrics: matched IoU, AP@tau, merge and split counts.

The scorecard for GT-free instance segmentation. Predicted instance labels are
matched to GT components greedily by IoU (the standard cellpose/stardist
protocol); merges and splits are counted separately because they are the
failure modes the composite hides -- a merge *lowers* the component count and
can pass a count-based check while being exactly the error we care about.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def overlap_matrix(
    pred_labels: np.ndarray,
    gt_labels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Voxel-overlap counts between every (pred, gt) instance pair.

    Returns ``(counts, pred_ids, gt_ids)`` where ``counts[i, j]`` is the voxel
    overlap between ``pred_ids[i]`` and ``gt_ids[j]``. Label 0 is background in
    both volumes and excluded.
    """

    pred = np.asarray(pred_labels).ravel()
    gt = np.asarray(gt_labels).ravel()
    keep = (pred > 0) | (gt > 0)
    pred = pred[keep]
    gt = gt[keep]
    pred_ids = np.unique(pred[pred > 0])
    gt_ids = np.unique(gt[gt > 0])
    if pred_ids.size == 0 or gt_ids.size == 0:
        return np.zeros((pred_ids.size, gt_ids.size), dtype=np.int64), pred_ids, gt_ids
    pred_index = np.zeros(int(pred.max()) + 1, dtype=np.int64)
    pred_index[pred_ids] = np.arange(pred_ids.size)
    gt_index = np.zeros(int(gt.max()) + 1, dtype=np.int64)
    gt_index[gt_ids] = np.arange(gt_ids.size)
    both = (pred > 0) & (gt > 0)
    flat = pred_index[pred[both]] * gt_ids.size + gt_index[gt[both]]
    counts = np.bincount(flat, minlength=pred_ids.size * gt_ids.size)
    return counts.reshape(pred_ids.size, gt_ids.size), pred_ids, gt_ids


def match_instances(
    pred_labels: np.ndarray,
    gt_labels: np.ndarray,
    *,
    iou_threshold: float = 0.5,
    merge_cover_fraction: float = 0.25,
) -> dict[str, Any]:
    """Greedy IoU matching plus merge/split accounting.

    - matched pairs: greedy by IoU descending, one-to-one, kept if IoU >= tau.
    - **merge**: one predicted instance is the majority claimant of >= 2 GT
      sheets. ``merged_gt_pairs`` counts the extra sheets absorbed (a pred
      covering 3 sheets contributes 2).
    - **split**: one GT sheet has >= 2 predicted instances each covering
      >= ``merge_cover_fraction`` of it.
    - AP@tau = TP / (TP + FP + FN), the cellpose/stardist convention.
    """

    counts, pred_ids, gt_ids = overlap_matrix(pred_labels, gt_labels)
    pred_sizes = np.bincount(np.asarray(pred_labels).ravel())[pred_ids] if pred_ids.size else np.array([])
    gt_sizes = np.bincount(np.asarray(gt_labels).ravel())[gt_ids] if gt_ids.size else np.array([])

    iou = np.zeros_like(counts, dtype=np.float64)
    for i in range(pred_ids.size):
        for j in range(gt_ids.size):
            union = pred_sizes[i] + gt_sizes[j] - counts[i, j]
            iou[i, j] = counts[i, j] / union if union else 0.0

    order = np.dstack(np.unravel_index(np.argsort(-iou, axis=None), iou.shape))[0]
    used_pred: set[int] = set()
    used_gt: set[int] = set()
    matches: list[dict[str, float]] = []
    for i, j in order:
        if iou[i, j] < iou_threshold:
            break
        if i in used_pred or j in used_gt:
            continue
        used_pred.add(int(i))
        used_gt.add(int(j))
        matches.append({"pred_id": int(pred_ids[i]), "gt_id": int(gt_ids[j]), "iou": float(iou[i, j])})

    tp = len(matches)
    fp = int(pred_ids.size - tp)
    fn = int(gt_ids.size - tp)

    # Merges: majority claimant per GT sheet (by overlap voxels).
    merges = []
    if pred_ids.size and gt_ids.size:
        majority = counts.argmax(axis=0)          # pred index per gt
        claimed = counts[majority, np.arange(gt_ids.size)]
        by_pred: dict[int, list[int]] = {}
        for j in range(gt_ids.size):
            if claimed[j] > 0:
                by_pred.setdefault(int(majority[j]), []).append(j)
        for i, gts in by_pred.items():
            if len(gts) >= 2:
                merges.append({
                    "pred_id": int(pred_ids[i]),
                    "gt_ids": [int(gt_ids[j]) for j in gts],
                })
    merged_gt_pairs = int(sum(len(m["gt_ids"]) - 1 for m in merges))

    # Splits: GT sheets substantially covered by >= 2 predicted instances.
    splits = []
    for j in range(gt_ids.size):
        if gt_sizes[j] == 0:
            continue
        coverers = np.nonzero(counts[:, j] >= merge_cover_fraction * gt_sizes[j])[0]
        if coverers.size >= 2:
            splits.append({"gt_id": int(gt_ids[j]),
                           "pred_ids": [int(pred_ids[i]) for i in coverers]})

    matched_iou = [m["iou"] for m in matches]
    # AP at a looser threshold from the same greedy IoU ranking: with ~3-voxel
    # sheets and a ~2.8-voxel placement limit, per-sheet IoU tops out near 0.5,
    # so AP@0.5 saturates at model fidelity rather than pipeline quality.
    loose = 0.25
    used_p, used_g, tp_loose = set(), set(), 0
    for i, j in order:
        if iou[i, j] < loose:
            break
        if i in used_p or j in used_g:
            continue
        used_p.add(int(i)); used_g.add(int(j)); tp_loose += 1
    return {
        "ap25": float(tp_loose / max(1, tp_loose + (pred_ids.size - tp_loose) + (gt_ids.size - tp_loose))),
        "true_positives_25": tp_loose,
        "pred_count": int(pred_ids.size),
        "gt_count": int(gt_ids.size),
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "ap": float(tp / max(1, tp + fp + fn)),
        "mean_matched_iou": float(np.mean(matched_iou)) if matched_iou else 0.0,
        "mean_gt_iou": float(sum(matched_iou) / max(1, gt_ids.size)),
        "merge_count": len(merges),
        "merged_gt_pairs": merged_gt_pairs,
        "split_count": len(splits),
        "matches": matches,
        "merges": merges,
        "splits": splits,
    }
