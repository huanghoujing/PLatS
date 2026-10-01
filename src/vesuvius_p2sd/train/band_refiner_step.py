"""0069: training step of the sheet band refiner (models/band_refiner.py) inside train_p2sd.py.

Per step, for up to `max_sheets` prompted sheets of the batch: decode the sheet's latent (P2SD
prediction with prob 1-gt_mix, AE-encoded GT latent otherwise) through the FROZEN 0065a decoder
exactly as production does (gated tiles) -> logits + h4; band = dilate(logits > 0, radius); one
random `patch`^3 crop centred on a band voxel; every band voxel of the crop is a token (capped at
`max_train_tokens` by uniform subsampling); the refiner predicts a delta per token; loss = BCE +
soft dice of (logit + delta) against the thin GT sheet over the crop's band tokens (ignore masked).
Stats include the same losses for the un-refined logits, so the gain is visible per step.

0069d (rank_weight > 0): a pairwise hinge over (GT-sheet token, off-sheet token) pairs — the score
of a point on the GT must exceed the score of a point off it by `rank_margin` logits. BCE trains
each token against its own fg/bg calibration and never orders two tokens against each other;
under the band's ~7:1 negative prior that calibration cuts faint sheet interiors at 0.5 (the 0069c
holes). The hinge is prior-free and is exactly the between-token order the decode threshold needs.
"""
from __future__ import annotations

import torch
from torch.nn import functional as F

from vesuvius_p2sd.models.band_refiner import band_from_mask, decode_one


def _bce_dice(logits: torch.Tensor, target: torch.Tensor, valid: torch.Tensor, pos_weight: float):
    w = valid * torch.where(target > 0.5, torch.full_like(target, pos_weight), torch.ones_like(target))
    bce = (F.binary_cross_entropy_with_logits(logits, target, reduction="none") * w).sum() / w.sum().clamp_min(1.0)
    p = torch.sigmoid(logits) * valid
    t = target * valid
    dice = 1.0 - (2 * (p * t).sum() + 1e-6) / (p.sum() + t.sum() + 1e-6)
    return bce, dice


_LOCAL_OVERSAMPLE = 16


def _rank_pairs(coords_local, p, tgt, val, dist_gt, *, shell, max_pos, pairs_per_pos, off_min, off_max, local_frac):
    """0069d: (positive, negative) token index pairs for the pairwise hinge. Positives = GT-sheet tokens;
    eligible negatives = tokens more than `shell` Chebyshev voxels from the GT sheet (the ±shell ring is
    the same physical surface under thin, ±1-2 voxel labels, so it is never a ranking partner). Local
    pairs put the negative at a random Chebyshev offset in [off_min, off_max] of the positive — the
    across-normal neighbours that decide whether a faint region is a sheet or a hole; global pairs draw
    the negative uniformly from the crop, which is what makes one decode threshold valid everywhere."""
    device = tgt.device
    pos = torch.nonzero((tgt > 0.5) & (val > 0.5)).squeeze(1)
    neg_ok = (tgt < 0.5) & (val > 0.5) & (dist_gt > shell)
    neg = torch.nonzero(neg_ok).squeeze(1)
    if pos.numel() == 0 or neg.numel() == 0:
        return None
    if pos.numel() > max_pos:
        pos = pos[torch.randperm(pos.numel(), device=device)[:max_pos]]
    n_total = pos.numel() * pairs_per_pos
    n_local = int(round(n_total * local_frac))
    i_g = pos[torch.randint(pos.numel(), (n_total - n_local,), device=device)]
    j_g = neg[torch.randint(neg.numel(), (n_total - n_local,), device=device)]
    if n_local > 0:
        # In-plane offsets land on the GT/shell and off-plane ones often miss the band token
        # sample, so only ~5-15 % of candidates survive: oversample, then truncate to n_local.
        n_try = n_local * _LOCAL_OVERSAMPLE
        idx_vol = torch.full((p, p, p), -1, dtype=torch.long, device=device)
        idx_vol[coords_local[:, 0], coords_local[:, 1], coords_local[:, 2]] = torch.arange(coords_local.shape[0], device=device)
        i_l = pos[torch.randint(pos.numel(), (n_try,), device=device)]
        off = torch.randint(-off_max, off_max + 1, (n_try, 3), device=device)
        tgt_vox = coords_local[i_l] + off
        ok = (off.abs().amax(dim=1) >= off_min) & (tgt_vox >= 0).all(dim=1) & (tgt_vox < p).all(dim=1)
        i_l, tgt_vox = i_l[ok], tgt_vox[ok]
        j_l = idx_vol[tgt_vox[:, 0], tgt_vox[:, 1], tgt_vox[:, 2]]
        ok = j_l >= 0
        i_l, j_l = i_l[ok], j_l[ok]
        ok = neg_ok[j_l]
        i_l, j_l = i_l[ok][:n_local], j_l[ok][:n_local]
    else:
        i_l = j_l = i_g[:0]
    return torch.cat([i_l, i_g]), torch.cat([j_l, j_g]), int(i_l.numel())


def band_refiner_step(brf, dec, target_ae, *, image, image_taps, z_pred_raw, z_gt, mask, batch, image_index, device,
                      max_sheets, gt_mix, pos_weight, bce_weight, dice_weight, patch, max_train_tokens, radius,
                      gate_logit, gate_dilate, binseg_model=None, rank_weight=0.0, rank_margin=2.0, rank_shell=2,
                      rank_offset_min=3, rank_offset_max=6, rank_pairs_per_pos=4, rank_max_pos=16384,
                      rank_local_frac=0.5):
    # 0069c: frozen binseg forward on the (augmented) raw-intensity image, exactly the tensor the
    # proposer sees in auto_instance_seg; its sigmoid prob is a per-token feature.
    binseg_prob = None
    if int(getattr(brf.cfg, "binseg_channels", 0)) > 0:
        if binseg_model is None:
            raise RuntimeError("band_refiner.binseg_channels > 0 requires binseg_model")
        with torch.no_grad():
            binseg_prob = torch.sigmoid(binseg_model(image).float())
    n_pairs = int(z_pred_raw.shape[0])
    k = min(max_sheets, n_pairs)
    idx = torch.randperm(n_pairs, device=device)[:k] if k < n_pairs else torch.arange(n_pairs, device=device)
    img_idx = image_index.to(device).long()[idx]
    use_gt = torch.rand(k, device=device) < gt_mix
    z_in = torch.where(use_gt.view(-1, 1, 1, 1, 1), z_gt[idx].detach(), z_pred_raw[idx].detach())
    target_all = mask[idx].float()
    if "ignore" in batch:
        ign = batch["ignore"]
        if ign.ndim == 5:
            ign = ign[:, 0]
        valid_all = (ign.to(device) == 0).unsqueeze(1).float()[img_idx]
    else:
        valid_all = torch.ones_like(target_all)
    with torch.no_grad():
        stem2_all = dec.image_stem(image)
    taps_keys = sorted(brf.cfg.tap_channels)
    losses, stats = [], {"bce": 0.0, "dice": 0.0, "base_bce": 0.0, "base_dice": 0.0, "tokens": 0.0,
                         "recall": 0.0, "precision": 0.0, "base_recall": 0.0, "base_precision": 0.0, "sheets": 0.0,
                         "rank": 0.0, "base_rank": 0.0, "rank_acc": 0.0, "base_rank_acc": 0.0, "pairs": 0.0,
                         "local_pairs": 0.0, "rank_local": 0.0, "base_rank_local": 0.0, "rank_acc_local": 0.0,
                         "base_rank_acc_local": 0.0}
    for i in range(k):
        b = int(img_idx[i])
        skips = {f: image_taps[f][b:b + 1] for f in dec.COARSE_FACTORS}
        with torch.no_grad():
            logits, h4 = decode_one(dec, target_ae, z_in[i:i + 1], skips, stem2_all[b:b + 1], image[b:b + 1],
                                    gate_logit=gate_logit, gate_dilate=gate_dilate)
            m = logits > 0
            if not bool(m.any()):
                continue
            band, dist = band_from_mask(m, radius)
            d, h, w = logits.shape[-3:]
            nz = torch.nonzero(band[0, 0])
            centre = nz[torch.randint(nz.shape[0], (1,), device=device)][0]
            p = min(patch, d)
            origin = torch.stack([torch.clamp(centre[j] - p // 2, 0, int(logits.shape[-3 + j]) - p) for j in range(3)])
            sub = band[0, 0, origin[0]:origin[0] + p, origin[1]:origin[1] + p, origin[2]:origin[2] + p]
            coords = torch.nonzero(sub) + origin.view(1, 3)
            if coords.shape[0] > max_train_tokens:
                coords = coords[torch.randperm(coords.shape[0], device=device)[:max_train_tokens]]
            taps = {f: image_taps[f][b:b + 1] for f in taps_keys}
            feat = brf.features(coords, origin, image=image[b:b + 1], taps=taps, h4=h4, stem2=stem2_all[b:b + 1],
                                logits=logits, mask=m, dist=dist,
                                binseg=None if binseg_prob is None else binseg_prob[b:b + 1])
            base = brf._gather(logits, coords).float().clamp(-20.0, 20.0)
            tgt = brf._gather(target_all[i:i + 1], coords).float()
            val = brf._gather(valid_all[i:i + 1], coords).float()
            pairs = None
            if rank_weight > 0:
                _, dist_gt = band_from_mask(target_all[i:i + 1] > 0.5, rank_shell)
                pairs = _rank_pairs(coords - origin.view(1, 3), p, tgt, val, brf._gather(dist_gt, coords).long(),
                                    shell=rank_shell, max_pos=rank_max_pos, pairs_per_pos=rank_pairs_per_pos,
                                    off_min=rank_offset_min, off_max=rank_offset_max, local_frac=rank_local_frac)
        delta = brf.forward_tokens(feat).float()
        refined = base + delta if brf.cfg.residual else delta
        bce, dice = _bce_dice(refined, tgt, val, pos_weight)
        loss_i = bce_weight * bce + dice_weight * dice
        if pairs is not None:
            pi, pj, n_local = pairs
            hinge = F.relu(rank_margin - (refined[pi] - refined[pj]))
            rank = hinge.mean()
            loss_i = loss_i + rank_weight * rank
            with torch.no_grad():
                base_hinge = F.relu(rank_margin - (base[pi] - base[pj]))
                acc, base_acc = (refined[pi] > refined[pj]).float(), (base[pi] > base[pj]).float()
                stats["rank"] += rank.item(); stats["base_rank"] += base_hinge.mean().item()
                stats["rank_acc"] += acc.mean().item(); stats["base_rank_acc"] += base_acc.mean().item()
                stats["pairs"] += float(pi.numel()); stats["local_pairs"] += float(n_local)
                if n_local > 0:  # local pairs come first in the concatenation
                    stats["rank_local"] += hinge[:n_local].mean().item()
                    stats["base_rank_local"] += base_hinge[:n_local].mean().item()
                    stats["rank_acc_local"] += acc[:n_local].mean().item()
                    stats["base_rank_acc_local"] += base_acc[:n_local].mean().item()
        losses.append(loss_i)
        with torch.no_grad():
            bb, bd = _bce_dice(base, tgt, val, pos_weight)
            pos = (tgt > 0.5) & (val > 0.5)
            pr = (refined > 0) & (val > 0.5)
            pb = (base > 0) & (val > 0.5)
            stats["bce"] += bce.item(); stats["dice"] += dice.item()
            stats["base_bce"] += bb.item(); stats["base_dice"] += bd.item()
            stats["tokens"] += float(coords.shape[0])
            stats["recall"] += ((pr & pos).sum() / pos.sum().clamp_min(1)).item()
            stats["precision"] += ((pr & pos).sum() / pr.sum().clamp_min(1)).item()
            stats["base_recall"] += ((pb & pos).sum() / pos.sum().clamp_min(1)).item()
            stats["base_precision"] += ((pb & pos).sum() / pb.sum().clamp_min(1)).item()
            stats["sheets"] += 1.0
    if not losses:
        return torch.zeros((), device=device, requires_grad=True), stats
    n = max(stats["sheets"], 1.0)
    for key in list(stats):
        if key != "sheets":
            stats[key] /= n
    return torch.stack(losses).mean(), stats
