"""Latent regression and sheet-identity losses used by P2SD.

These functions return unweighted terms. The trainer applies the configured
coefficients when assembling the objective; ``weight`` only disables inactive
identity terms. Kept separate so the mathematical method is easy to review.
"""

from __future__ import annotations

import torch
from torch.nn import functional as F


def aux_mse_loss(
    aux_latents,
    z_gt: torch.Tensor,
    weights: list[float],
    *,
    normalizer=None,
    mode: str = "aux_only",
) -> torch.Tensor:
    if not aux_latents or not weights:
        return z_gt.new_zeros(())
    mode = str(mode)
    aux_items = list(aux_latents)
    active_weights = list(weights)
    if mode == "original":
        if len(active_weights) != len(aux_items):
            raise ValueError(
                "p2sd.loss.per_round_mse_weights_mode=original requires one weight "
                f"per refiner round; got {len(active_weights)} weights for {len(aux_items)} rounds"
            )
        aux_items = aux_items[:-1]
        active_weights = active_weights[:-1]
    elif mode == "uniform":
        # The loop modulator emits one aux latent per prompt point and the point
        # count varies per batch under `sampler.n_pos_range`, so a per-round
        # weight list can never line up. One scalar covers every entry. Note the
        # per-round modes would fail QUIETLY here: `zip` truncates to the shorter
        # sequence, silently dropping supervision on the extra rounds.
        active_weights = [float(active_weights[0])] * len(aux_items)
    elif mode != "aux_only":
        raise ValueError(f"Unsupported p2sd.loss.per_round_mse_weights_mode: {mode}")
    normalizer = normalizer or (lambda x: x.float())
    z_target = normalizer(z_gt)
    total = z_gt.new_zeros(())
    for pred, weight in zip(aux_items, active_weights):
        if weight:
            total = total + float(weight) * F.mse_loss(normalizer(pred), z_target)
    return total


def per_round_final_mse_scale(weights: list[float], *, mode: str = "aux_only") -> float:
    if str(mode) == "original" and weights:
        return float(weights[-1])
    if str(mode) in {"aux_only", "uniform"}:
        return 1.0
    if str(mode) == "original":
        return 1.0
    raise ValueError(f"Unsupported p2sd.loss.per_round_mse_weights_mode: {mode}")


def same_sheet_prompt_consistency_loss(
    z_pred: torch.Tensor,
    image_index: torch.Tensor,
    sheet_id: torch.Tensor,
    *,
    weight: float,
) -> tuple[torch.Tensor, int]:
    if weight <= 0 or z_pred.shape[0] < 2:
        return z_pred.new_zeros(()), 0
    stride = int(sheet_id.max().detach().item()) + 1 if sheet_id.numel() else 1
    group_id = image_index.long() * max(1, stride) + sheet_id.long()
    _, inv = torch.unique(group_id, return_inverse=True)
    groups = int(inv.max().detach().item()) + 1 if inv.numel() else 0
    if groups <= 0:
        return z_pred.new_zeros(()), 0
    counts = torch.bincount(inv, minlength=groups)
    multi = counts > 1
    if not multi.any():
        return z_pred.new_zeros(()), 0
    z = z_pred.float()
    group_sum = z.new_zeros(groups, *z.shape[1:])
    group_sum.index_add_(0, inv, z)
    group_mean = group_sum / counts.view(groups, *([1] * (z.dim() - 1))).clamp_min(1).float()
    member_mask = multi[inv]
    loss = (z[member_mask] - group_mean[inv[member_mask]]).pow(2).mean()
    return loss, int(multi.sum().detach().item())


def different_sheet_prompt_contrast_loss(
    z_pred: torch.Tensor,
    image_index: torch.Tensor,
    sheet_id: torch.Tensor,
    *,
    weight: float,
    margin: float,
) -> tuple[torch.Tensor, int, torch.Tensor]:
    """Push latents of DIFFERENT sheets in the same volume at least ``margin`` apart.

    The complement of ``same_sheet_prompt_consistency_loss``: that one pulls
    same-(image, sheet) predictions together, this one hinges on the mean
    squared distance between same-image different-sheet predictions --
    ``relu(margin - d)`` averaged over such pairs, so pairs already farther than
    the margin contribute nothing. Calibration on `0017` (normalized latents):
    same-sheet d ~ 0.0003, different-sheet mean 1.77 / p10 0.72 / min 0.31, so
    the default margin 1.0 targets the closest ~20% of sheet pairs -- the
    confusable ones behind wrong-sheet pickup and touching-sheet cases.

    Returns ``(loss, pair_count, mean_distance)``; zeros when no such pair is in
    the batch (requires ``pair_sampling.max_components_per_sample >= 2``, see
    ``selection_mode: balanced_sheets``).
    """

    if weight <= 0 or z_pred.shape[0] < 2:
        return z_pred.new_zeros(()), 0, z_pred.new_zeros(())
    same_image = image_index.view(-1, 1) == image_index.view(1, -1)
    different_sheet = sheet_id.view(-1, 1) != sheet_id.view(1, -1)
    pair_mask = torch.triu(same_image & different_sheet, diagonal=1)
    left, right = pair_mask.nonzero(as_tuple=True)
    if left.numel() == 0:
        return z_pred.new_zeros(()), 0, z_pred.new_zeros(())
    z = z_pred.float().flatten(1)
    distances = (z[left] - z[right]).pow(2).mean(dim=1)
    loss = F.relu(float(margin) - distances).mean()
    return loss, int(left.numel()), distances.detach().mean()
