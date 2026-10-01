"""One training step of the image-conditioned sheet decoder (0065): tiled fine stages, GT targets."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def _per_sheet_dice(prob: torch.Tensor, target: torch.Tensor, valid: torch.Tensor, sheet: torch.Tensor,
                    n_sheets: int) -> torch.Tensor:
    """Soft dice per sheet over its tile centres (prob/target/valid: (N,1,c,c,c); sheet: (N,) index)."""
    p = (prob * valid).flatten(1).sum(1)
    t = (target * valid).flatten(1).sum(1)
    i = (prob * target * valid).flatten(1).sum(1)
    inter = torch.zeros(n_sheets, device=prob.device, dtype=prob.dtype).index_add_(0, sheet, i)
    denom = torch.zeros(n_sheets, device=prob.device, dtype=prob.dtype).index_add_(0, sheet, p + t)
    mass = torch.zeros(n_sheets, device=prob.device, dtype=prob.dtype).index_add_(0, sheet, t)
    dice = 1.0 - (2 * inter + 1e-6) / (denom + 1e-6)
    keep = mass > 0
    return dice[keep].mean() if bool(keep.any()) else dice.sum() * 0.0


def sheet_decoder_step(sdec, target_ae, *, image, image_taps, z_pred_raw, z_gt, mask, batch, image_index, device,
                       max_sheets, gt_mix, pos_weight, bce_weight, dice_weight, aux_weight, aux_pos_weight,
                       gate_logit, max_extra_tiles):
    n_pairs = int(z_pred_raw.shape[0])
    k = min(max_sheets, n_pairs)
    idx = torch.randperm(n_pairs, device=device)[:k] if k < n_pairs else torch.arange(n_pairs, device=device)
    img_idx = image_index.to(device).long()[idx]
    use_gt = torch.rand(k, device=device) < gt_mix
    z_in = torch.where(use_gt.view(-1, 1, 1, 1, 1), z_gt[idx].detach(), z_pred_raw[idx].detach())
    target = mask[idx].float()
    if "ignore" in batch:
        ign = batch["ignore"]
        if ign.ndim == 5:
            ign = ign[:, 0]
        valid = (ign.to(device) == 0).unsqueeze(1).float()[img_idx]
    else:
        valid = torch.ones_like(target)
    tile, hq = target_ae.SPARSE_TILE, target_ae.SPARSE_HALO4
    c = slice(hq * 4, hq * 4 + tile)

    skips = {f: image_taps[f][img_idx] for f in sdec.COARSE_FACTORS}
    h4, aux4 = sdec.coarse(z_in, skips)
    # 1/4 occupancy target for the gate (any GT voxel in the 4^3 cell); cell valid if any voxel valid
    occ4 = F.max_pool3d(target, kernel_size=4, stride=4)
    valid4 = F.max_pool3d(valid, kernel_size=4, stride=4)
    w4 = valid4 * torch.where(occ4 > 0.5, torch.full_like(occ4, aux_pos_weight), torch.ones_like(occ4))
    aux_bce = (F.binary_cross_entropy_with_logits(aux4.float(), occ4, reduction="none") * w4).sum() / valid4.sum().clamp_min(1.0)

    with torch.no_grad():
        gate = aux4.detach() > gate_logit
        gt_tiles = target_ae.active_tiles(occ4 > 0.5, tile4=tile // 4, dilate=1)
        gate_tiles = target_ae.active_tiles(gate, tile4=tile // 4, dilate=1)
        extra = gate_tiles & ~gt_tiles
        # cap the non-GT tiles per step (untrained gate = everything active -> OOM)
        n_extra = int(extra.sum())
        if n_extra > max_extra_tiles * k:
            flat = extra.flatten()
            pos = flat.nonzero()[:, 0]
            keep = pos[torch.randperm(pos.numel(), device=device)[: max_extra_tiles * k]]
            flat = torch.zeros_like(flat)
            flat[keep] = True
            extra = flat.view_as(extra)
        tiles = gt_tiles | extra
        gate_recall = float(((gate_tiles & gt_tiles).sum() / gt_tiles.sum().clamp_min(1)).item())
    stem2 = sdec.image_stem(image)[img_idx]
    full = {4: h4, 2: stem2, 1: torch.cat([image[img_idx].to(h4.dtype), target.to(h4.dtype), valid.to(h4.dtype)], dim=1)}
    t, valids, tidx = target_ae.gather_tile_set(tiles, full)
    logits_t = sdec.fine(t[4], t[2], t[1][:, :1], valid=valids)
    logits_c = logits_t[..., c, c, c].float()
    target_c = t[1][:, 1:2][..., c, c, c].float()
    valid_c = t[1][:, 2:3][..., c, c, c].float()
    w = valid_c * torch.where(target_c > 0.5, torch.full_like(target_c, pos_weight), torch.ones_like(target_c))
    bce = (F.binary_cross_entropy_with_logits(logits_c, target_c, reduction="none") * w).sum() / valid_c.sum().clamp_min(1.0)
    prob = torch.sigmoid(logits_c)
    dice = _per_sheet_dice(prob, target_c, valid_c, tidx[:, 0], k)
    loss = bce_weight * bce + dice_weight * dice + aux_weight * aux_bce
    with torch.no_grad():
        volume_ratio = float((((logits_c > 0).float() * valid_c).sum() / (target_c * valid_c).sum().clamp_min(1.0)).item())
    stats = {"bce": float(bce.detach()), "dice": float(dice.detach()), "aux_bce": float(aux_bce.detach()),
             "tiles": float(tidx.shape[0]), "gate_recall": gate_recall, "volume_ratio": volume_ratio,
             "gt_fraction": float(use_gt.float().mean())}
    return loss, stats


def union_decoder_step(udec, target_ae, *, image, image_taps, context, batch, device, pos_weight, bce_weight,
                       dice_weight, aux_weight, aux_pos_weight, gate_logit, max_tiles_per_image):
    """0066: union-mask decoder step. z = the P2SD image context (per image); target = union of all GT
    sheets, ignore-masked; tiles = (aux gate | GT) capped at max_tiles_per_image (random subset)."""
    comp = batch["component_label"]
    if comp.ndim == 5:
        comp = comp[:, 0]
    target = (comp.to(device) > 0).float().unsqueeze(1)
    if "ignore" in batch:
        ign = batch["ignore"]
        if ign.ndim == 5:
            ign = ign[:, 0]
        valid = (ign.to(device) == 0).float().unsqueeze(1)
    else:
        valid = torch.ones_like(target)
    b = int(target.shape[0])
    tile, hq = target_ae.SPARSE_TILE, target_ae.SPARSE_HALO4
    c = slice(hq * 4, hq * 4 + tile)
    h4, aux4 = udec.coarse(context, {f: image_taps[f] for f in udec.COARSE_FACTORS})
    occ4 = F.max_pool3d(target, kernel_size=4, stride=4)
    valid4 = F.max_pool3d(valid, kernel_size=4, stride=4)
    w4 = valid4 * torch.where(occ4 > 0.5, torch.full_like(occ4, aux_pos_weight), torch.ones_like(occ4))
    aux_bce = (F.binary_cross_entropy_with_logits(aux4.float(), occ4, reduction="none") * w4).sum() / valid4.sum().clamp_min(1.0)
    with torch.no_grad():
        gt_tiles = target_ae.active_tiles(occ4 > 0.5, tile4=tile // 4, dilate=1)
        gate_tiles = target_ae.active_tiles(aux4.detach() > gate_logit, tile4=tile // 4, dilate=1)
        tiles = gt_tiles | gate_tiles
        gate_recall = float(((gate_tiles & gt_tiles).sum() / gt_tiles.sum().clamp_min(1)).item())
        for bi in range(b):
            flat = tiles[bi].flatten()
            pos = flat.nonzero()[:, 0]
            if pos.numel() > max_tiles_per_image:
                keep = pos[torch.randperm(pos.numel(), device=device)[:max_tiles_per_image]]
                flat = torch.zeros_like(flat)
                flat[keep] = True
                tiles[bi] = flat.view_as(tiles[bi])
    stem2 = udec.image_stem(image)
    full = {4: h4, 2: stem2, 1: torch.cat([image.to(h4.dtype), target.to(h4.dtype), valid.to(h4.dtype)], dim=1)}
    t, valids, tidx = target_ae.gather_tile_set(tiles, full)
    logits_t = udec.fine(t[4], t[2], t[1][:, :1], valid=valids)
    logits_c = logits_t[..., c, c, c].float()
    target_c = t[1][:, 1:2][..., c, c, c].float()
    valid_c = t[1][:, 2:3][..., c, c, c].float()
    w = valid_c * torch.where(target_c > 0.5, torch.full_like(target_c, pos_weight), torch.ones_like(target_c))
    bce = (F.binary_cross_entropy_with_logits(logits_c, target_c, reduction="none") * w).sum() / valid_c.sum().clamp_min(1.0)
    dice = _per_sheet_dice(torch.sigmoid(logits_c), target_c, valid_c, tidx[:, 0], b)
    loss = bce_weight * bce + dice_weight * dice + aux_weight * aux_bce
    with torch.no_grad():
        volume_ratio = float((((logits_c > 0).float() * valid_c).sum() / (target_c * valid_c).sum().clamp_min(1.0)).item())
    stats = {"bce": float(bce.detach()), "dice": float(dice.detach()), "aux_bce": float(aux_bce.detach()),
             "tiles": float(tidx.shape[0]), "gate_recall": gate_recall, "volume_ratio": volume_ratio}
    return loss, stats
