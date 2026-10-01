"""Train a binary sheet-segmentation decoder on a frozen P2SD trunk.

Prompt-free: the target is the union of all labeled sheets, and the loss is
masked to the labeled region -- the source labels leave ~57% of a typical
volume unlabeled (see ``data/build_ignore_masks.py``), so ignore-voxels carry
zero loss weight. BCE and Dice are combined but always logged separately.

Only the fresh decoder trains; the trunk (image encoder + image-context
attention, warm-started from a P2SD checkpoint) is frozen.
"""

from __future__ import annotations

import argparse
import math
import os
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from vesuvius_p2sd.data.dataset import build_dataset
from vesuvius_p2sd.models.binary_seg import build_binary_seg_model
from vesuvius_p2sd.research.run_status import update_heartbeat, write_json
from vesuvius_p2sd.train.common import (
    MetricWindow,
    amp_dtype,
    append_jsonl,
    autocast_context,
    get_device,
    load_ae_from_config,
    prepare_run,
    save_checkpoint,
    seed_everything,
    should_stop,
)
from vesuvius_p2sd.train.loss_plots import write_loss_plots


def masked_bce_and_dice(
    logits: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor,
    *,
    pos_weight: float,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    """BCE and soft Dice restricted to labeled (valid) voxels.

    ``pos_weight`` is a fixed scalar, deliberately small (<= ~2): an
    auto-balanced weight at the true foreground rate would be ~25x and
    systematically thicken predictions. Dice is computed on valid voxels only by
    zeroing both probability and target on ignore -- ignored voxels then
    contribute to neither intersection nor denominator.
    """

    logits = logits.float()
    target = target.float()
    valid = valid.float()
    weight = valid * torch.where(target > 0.5, torch.full_like(target, float(pos_weight)), torch.ones_like(target))
    bce_map = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    valid_total = valid.sum().clamp_min(1.0)
    bce = (bce_map * weight).sum() / valid_total
    prob = torch.sigmoid(logits) * valid
    masked_target = target * valid
    dims = tuple(range(1, prob.ndim))
    inter = (prob * masked_target).sum(dim=dims)
    target_mass = masked_target.sum(dim=dims)
    denom = prob.sum(dim=dims) + target_mass
    per_sample_dice = 1.0 - ((2 * inter + 1e-6) / (denom + 1e-6))
    # Soft dice is degenerate on an empty masked target (any residual
    # probability drives it to ~1 regardless of quality). Crop training makes
    # empty-foreground samples routine, so exclude them from the dice mean --
    # BCE still supervises those crops, which is exactly the false-positive
    # suppression an empty region should teach.
    has_target = target_mass > 0
    if bool(has_target.any()):
        dice = per_sample_dice[has_target].mean()
    else:
        dice = logits.new_zeros(())
    stats = {
        "valid_fraction": float((valid.sum() / valid.numel()).detach().cpu()),
        "foreground_fraction": float((masked_target.sum() / valid_total).detach().cpu()),
        "dice_samples": float(has_target.sum().detach().cpu()),
    }
    return bce, dice, stats


def border_mask(shape: tuple[int, ...], width: int, device) -> torch.Tensor:
    """True on the outer ``width``-voxel shell of a (z, y, x) volume."""

    mask = torch.zeros(shape, dtype=torch.bool, device=device)
    if width <= 0:
        return mask
    w = int(width)
    mask[:w], mask[-w:] = True, True
    mask[:, :w], mask[:, -w:] = True, True
    mask[..., :w], mask[..., -w:] = True, True
    return mask


def _sample_mask_coords(
    mask: torch.Tensor, count: int, generator: torch.Generator | None
) -> torch.Tensor | None:
    coords = mask.nonzero(as_tuple=False)
    if coords.shape[0] == 0:
        return None
    idx = torch.randint(coords.shape[0], (min(count, coords.shape[0]),),
                        generator=generator, device=coords.device)
    return coords.index_select(0, idx)


def _embed_at_coords(
    embedding: torch.Tensor, item: int, coords: torch.Tensor, full_shape
) -> torch.Tensor:
    """Sample the embedding at fine-grid voxel coordinates.

    Voxel-centre convention for cross-resolution sampling: fine voxel c maps
    to fraction (c + 0.5) / N with align_corners=False, so a stride-s coarse
    cell is sampled at the centre of the fine block it covers — the same
    convention as ``sheet_ae.sample_latent_at_points``. The previous
    ``c / (N-1)`` + align_corners=True form misaligned a stride-32 tap by up
    to half a coarse cell (~16 voxels) near volume edges (fixed 2026-08-13).
    """
    device = embedding.device
    size = torch.tensor(full_shape, device=device, dtype=torch.float32).clamp_min(1)
    grid = ((coords.to(device).float() + 0.5) / size) * 2 - 1
    grid_xyz = grid[:, [2, 1, 0]].view(1, -1, 1, 1, 3)
    sampled = F.grid_sample(embedding[item:item + 1].float(), grid_xyz,
                            mode="bilinear", align_corners=False)
    return F.normalize(sampled[0, :, :, 0, 0].transpose(0, 1), dim=1)


ANCHOR_PROJECTION_SEED = 20260813
ANCHOR_GRID = (10, 10, 10)
_anchor_projection_cache: dict = {}


def anchor_projection_matrix(flat_dim: int, out_dim: int, device) -> torch.Tensor:
    """Fixed random Johnson-Lindenstrauss projection, seeded and cached.

    Sheet identity lives in the SPATIAL pattern of the AE latent: measured on
    0032, same-volume different-sheet pair cosine is 0.099 for FLATTENED
    latents but 0.894 after whole-grid mean pooling and 0.978 after
    sheet-masked pooling — any spatial pooling collapses identity (this also
    explains the neutral P1 latent-prompt token). A fixed random projection
    of the flattened latent approximately preserves pairwise angles, giving
    compact 64-d anchors that stay separated; the same seeded matrix can map
    pipeline fingerprints into the identical space at inference.
    """
    key = (int(flat_dim), int(out_dim), str(device))
    if key not in _anchor_projection_cache:
        generator = torch.Generator(device="cpu").manual_seed(ANCHOR_PROJECTION_SEED)
        matrix = torch.randn(int(out_dim), int(flat_dim), generator=generator)
        matrix = matrix / float(flat_dim) ** 0.5
        _anchor_projection_cache[key] = matrix.to(device)
    return _anchor_projection_cache[key]


@torch.no_grad()
def sheet_latent_anchors(
    target_ae,
    component_label: torch.Tensor,
    *,
    min_sheet_voxels: int,
    anchor_dim: int,
    encode_batch: int = 4,
) -> dict[tuple[int, int], torch.Tensor]:
    """Normalized random-projected FLATTENED AE latents per (item, sheet).

    The latent grid is adaptively pooled to a fixed ANCHOR_GRID first so the
    flattened dimension (and therefore the projection matrix) is identical
    across 320^3 and 384^3 cases, then projected to ``anchor_dim`` with the
    fixed JL matrix (see anchor_projection_matrix for why pooling alone is
    wrong).
    """
    jobs: list[tuple[int, int]] = []
    for item in range(component_label.shape[0]):
        labels_item = component_label[item]
        for sheet in torch.unique(labels_item[labels_item > 0]):
            sheet = int(sheet)
            if int((labels_item == sheet).sum()) >= min_sheet_voxels:
                jobs.append((item, sheet))
    anchors: dict[tuple[int, int], torch.Tensor] = {}
    for start in range(0, len(jobs), encode_batch):
        chunk = jobs[start:start + encode_batch]
        masks = torch.stack([
            (component_label[item] == sheet).float()
            for item, sheet in chunk
        ]).unsqueeze(1)
        z = target_ae.encode(masks).float()
        z = F.adaptive_avg_pool3d(z, ANCHOR_GRID).flatten(1)
        projection = anchor_projection_matrix(z.shape[1], anchor_dim, z.device)
        projected = F.normalize(z @ projection.t(), dim=1)
        for (item, sheet), vector in zip(chunk, projected):
            anchors[(item, sheet)] = vector
    return anchors


def voxel_latent_distill_loss(
    embedding: torch.Tensor,
    component_label: torch.Tensor,
    anchors: dict[tuple[int, int], torch.Tensor],
    *,
    pos_samples: int = 1024,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Point each sheet voxel's embedding at its sheet's pooled AE latent.

    The successor of the prototype-contrast loss (0024/0025, NEGATIVE): the
    anchor is no longer an in-batch estimate but a FIXED external target from
    the repulsion-trained AE latent space. Cosine regression only -- distinct
    anchors are already near-orthogonal, so cross-sheet discrimination comes
    from the target geometry rather than sampled negatives. Cross-sheet
    cosine is logged for monitoring.
    """

    losses = []
    stats = {"distill_sheets": 0.0, "distill_cos": 0.0, "distill_cross_cos": 0.0}
    full_shape = component_label.shape[1:]
    cross_terms = 0.0
    cross_count = 0
    for (item, sheet), anchor in anchors.items():
        coords = _sample_mask_coords(component_label[item] == sheet, pos_samples, generator)
        if coords is None:
            continue
        sampled = _embed_at_coords(embedding, item, coords, full_shape)
        cos = sampled @ anchor.to(sampled.dtype)
        losses.append((1.0 - cos).mean())
        stats["distill_sheets"] += 1.0
        stats["distill_cos"] += float(cos.mean().detach().cpu())
        for (other_item, other_sheet), other_anchor in anchors.items():
            if other_item == item and other_sheet != sheet:
                cross_terms += float((sampled.detach() @ other_anchor.to(sampled.dtype)).mean().cpu())
                cross_count += 1
    if not losses:
        return embedding.new_zeros(()), stats
    n = max(1.0, stats["distill_sheets"])
    stats["distill_cos"] /= n
    stats["distill_cross_cos"] = cross_terms / max(1, cross_count)
    return torch.stack(losses).mean(), stats


def voxel_prototype_contrast_loss(
    embedding: torch.Tensor,
    component_label: torch.Tensor,
    ignore: torch.Tensor,
    *,
    temperature: float = 0.1,
    pos_samples: int = 1024,
    neg_samples: int = 4096,
    center_samples: int = 2048,
    min_sheet_voxels: int = 500,
    border_width: int = 5,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Per-sheet prototype classifiers over sampled voxel embeddings.

    For each labeled sheet: its center (normalized mean embedding over
    ``center_samples`` of its own voxels) is applied as a cosine classifier --
    own voxels must score high (within-sheet consistency), everything else low
    (cross-sheet discrimination). Negatives are other sheets' voxels, labeled
    background, AND the ignore region minus the outer ``border_width`` shell:
    the source labels are sheet-complete, so a non-border ignore voxel is
    guaranteed to belong to no labeled sheet -- unlabeled NEIGHBOR sheets
    become free hard negatives. Only the border is excluded (erased
    continuations of labeled sheets live there).

    ``embedding`` may be at a coarser resolution than ``component_label``
    (the tap stage); sampled voxel coordinates are read via trilinear
    grid_sample, and normalization happens after sampling.
    """

    b = embedding.shape[0]
    losses = []
    stats = {"contrast_sheets": 0.0, "contrast_pos_cos": 0.0, "contrast_neg_cos": 0.0}
    full_shape = component_label.shape[1:]
    border = border_mask(full_shape, border_width, component_label.device)

    def sample_coords(mask: torch.Tensor, count: int) -> torch.Tensor | None:
        return _sample_mask_coords(mask, count, generator)

    def embed_at(item: int, coords: torch.Tensor) -> torch.Tensor:
        return _embed_at_coords(embedding, item, coords, full_shape)

    for i in range(b):
        labels_i = component_label[i]
        ignore_i = ignore[i].bool()
        sheet_ids = [int(s) for s in torch.unique(labels_i[labels_i > 0])
                     if int((labels_i == s).sum()) >= min_sheet_voxels]
        if not sheet_ids:
            continue
        negatives_common = sample_coords(
            ((labels_i == 0) & ~ignore_i) | (ignore_i & ~border), neg_samples)
        for sheet in sheet_ids:
            own = labels_i == sheet
            center_coords = sample_coords(own, center_samples)
            pos_coords = sample_coords(own, pos_samples)
            other = sample_coords((labels_i > 0) & ~own, neg_samples // 2)
            if center_coords is None or pos_coords is None or negatives_common is None:
                continue
            center = F.normalize(embed_at(i, center_coords).mean(dim=0), dim=0)
            pos = embed_at(i, pos_coords) @ center
            neg_coords = negatives_common if other is None else torch.cat([negatives_common, other])
            neg = embed_at(i, neg_coords) @ center
            logits = torch.cat([pos, neg]) / float(temperature)
            target = torch.cat([torch.ones_like(pos), torch.zeros_like(neg)])
            losses.append(F.binary_cross_entropy_with_logits(logits, target))
            stats["contrast_sheets"] += 1.0
            stats["contrast_pos_cos"] += float(pos.mean().detach().cpu())
            stats["contrast_neg_cos"] += float(neg.mean().detach().cpu())
    if not losses:
        return embedding.new_zeros(()), stats
    n = max(1.0, stats["contrast_sheets"])
    stats["contrast_pos_cos"] /= n
    stats["contrast_neg_cos"] /= n
    return torch.stack(losses).mean(), stats


def sample_context_crop(
    context: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor,
    *,
    crop_grid: int,
    generator: torch.Generator | None = None,
    attempts: int = 4,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Random grid-ALIGNED crop of the context and the matching output region.

    One context voxel maps to ``factor^3`` output voxels (factor = 32 at
    PS320), so offsets are drawn in grid units and scaled -- a misaligned crop
    would train the decoder against shifted targets. Draws up to ``attempts``
    offsets and keeps the first whose crop contains any labeled foreground;
    otherwise the last draw stands (an all-background crop still teaches
    false-positive suppression through BCE).
    """

    grid = int(context.shape[-1])
    crop_grid = int(crop_grid)
    if crop_grid <= 0 or crop_grid >= grid:
        return context, target, valid
    factor = int(target.shape[-1]) // grid
    span = grid - crop_grid + 1
    chosen = None
    for _ in range(max(1, attempts)):
        offsets = torch.randint(0, span, (3,), generator=generator).tolist()
        gz, gy, gx = (int(v) for v in offsets)
        vz, vy, vx = gz * factor, gy * factor, gx * factor
        target_crop = target[..., vz:vz + crop_grid * factor,
                             vy:vy + crop_grid * factor, vx:vx + crop_grid * factor]
        valid_crop = valid[..., vz:vz + crop_grid * factor,
                           vy:vy + crop_grid * factor, vx:vx + crop_grid * factor]
        chosen = ((gz, gy, gx), target_crop, valid_crop)
        if bool((target_crop.bool() & valid_crop.bool()).any()):
            break
    (gz, gy, gx), target_crop, valid_crop = chosen
    context_crop = context[..., gz:gz + crop_grid, gy:gy + crop_grid, gx:gx + crop_grid]
    return context_crop, target_crop, valid_crop


def load_trunk_weights(model, checkpoint_path: str) -> None:
    """Load a P2SD checkpoint into the frozen trunk; the decoder stays fresh.

    Strict over the trunk's own parameter set, so a typo'd path or a
    wrong-architecture checkpoint cannot half-load silently.
    """

    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = state.get("model", state)
    model.trunk.load_state_dict(state, strict=True)


def main(argv: list[str] | None = None) -> int:
    # cuDNN off: a cuDNN kernel in this trainer's step sporadically reads an
    # unmapped page on Blackwell (Xid 31 MMU fault; cuDNN 9 / CUDA 13.2 /
    # torch 2.12). 5 launches with cuDNN crashed within 1.6k steps (0038,
    # 0039 r0-r3; CUDA_LAUNCH_BLOCKING placed one fault in a cuDNN backward
    # kernel; SDPA-only disable did NOT help), while a full-disable replay ran
    # clean past every prior crash step at no measurable step-time cost.
    # VESUVIUS_ENABLE_CUDNN=1 re-enables for re-testing after stack upgrades.
    if not os.environ.get("VESUVIUS_ENABLE_CUDNN"):
        torch.backends.cudnn.enabled = False
        torch.backends.cuda.enable_cudnn_sdp(False)
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", required=True)
    parser.add_argument("--run_id", default=None)
    args, overrides = parser.parse_known_args(argv)
    cfg, run_dir = prepare_run(
        config_path=args.config_path,
        overrides=overrides,
        run_id=args.run_id,
        entrypoint="train_binary_seg",
    )
    seed_everything(int(cfg.get("seed", cfg.get("training", {}).get("seed", 42))))

    device = get_device(cfg)
    dtype = amp_dtype(cfg.get("training", {}).get("amp_dtype"))
    training_cfg = cfg.get("training", {})
    seg_cfg = cfg.get("binary_seg", {})
    loss_cfg = seg_cfg.get("loss", {})
    bce_weight = float(loss_cfg.get("bce_weight", 1.0))
    dice_weight = float(loss_cfg.get("dice_weight", 1.0))
    pos_weight = float(loss_cfg.get("pos_weight", 2.0))
    decode_crop_grid = int(seg_cfg.get("decode_crop_grid_voxels", 0))
    contrast_cfg = seg_cfg.get("contrast", {})
    contrast_enabled = bool(contrast_cfg.get("enabled", False))
    contrast_mode = str(contrast_cfg.get("mode", "prototype"))
    if contrast_mode not in {"prototype", "latent_distill"}:
        raise ValueError(f"binary_seg.contrast.mode must be prototype|latent_distill, got {contrast_mode!r}")
    contrast_weight = float(contrast_cfg.get("weight", 0.2))
    contrast_warmup_epochs = float(contrast_cfg.get("warmup_epochs", 10))
    if contrast_enabled and decode_crop_grid:
        raise ValueError("binary_seg.contrast requires full-volume decode (decode_crop_grid_voxels: 0)")

    data_cfg = cfg.get("data", {})
    if not bool(data_cfg.get("load_ignore", False)):
        raise ValueError(
            "binary segmentation requires data.load_ignore: true -- without the ignore "
            "mask the loss treats ~57% unlabeled voxels as negative")

    model = build_binary_seg_model(cfg).to(device)
    distill_target_ae = None
    if contrast_enabled and contrast_mode == "latent_distill":
        target_cfg = cfg.get("target_ae", {})
        if not target_cfg.get("checkpoint_path"):
            raise ValueError(
                "contrast.mode=latent_distill requires target_ae.config_path and "
                "target_ae.checkpoint_path (the frozen anchor AE)")
        distill_target_ae, _ = load_ae_from_config(
            target_cfg.get("config_path"), target_cfg.get("checkpoint_path"))
        distill_target_ae = distill_target_ae.to(device).eval()
        for parameter in distill_target_ae.parameters():
            parameter.requires_grad_(False)
    resume_path = training_cfg.get("resume_checkpoint_path")
    resume_state = None
    if resume_path:
        # True resume: model + optimizer + step, continuing the step-based
        # cosine schedule in the same run dir (metrics/checkpoints append).
        resume_state = torch.load(resume_path, map_location="cpu", weights_only=False)
        model.load_state_dict(resume_state["model"], strict=True)
        print(f"[resume] loaded {resume_path} at step {int(resume_state.get('step', 0))}", flush=True)
    warm_start = training_cfg.get("warm_start_path")
    if resume_path:
        pass
    elif warm_start:
        # Full-model warm start (trunk AND decoder) from a previous binary-seg
        # run. Strict over everything the checkpoint should cover: the only
        # tolerated gaps are heads the checkpoint predates (context_refiner
        # keeps its identity init, contrast_head its fresh init) -- any other
        # missing or unexpected key is still a hard error, so a
        # wrong-architecture checkpoint cannot half-load.
        fresh_ok = ("context_refiner.", "contrast_head.")
        state = torch.load(warm_start, map_location="cpu", weights_only=False)
        state = state.get("model", state)
        swap_trunk = training_cfg.get("trunk_checkpoint_path")
        if swap_trunk:
            # Trunk SWAP under a warm-started decoder/refiner: binseg-side
            # weights come from warm_start_path, the frozen trunk from this
            # P2SD checkpoint (e.g. a context-objective ft). Drop the warm
            # checkpoint's trunk and tolerate trunk gaps here;
            # load_trunk_weights below stays strict over the swapped trunk,
            # so nothing half-loads.
            state = {k: v for k, v in state.items() if not k.startswith("trunk.")}
            fresh_ok = fresh_ok + ("trunk.",)
        incompatible = model.load_state_dict(state, strict=False)
        bad_missing = [k for k in incompatible.missing_keys if not k.startswith(fresh_ok)]
        if bad_missing or incompatible.unexpected_keys:
            raise ValueError(
                f"warm start {warm_start} is not architecture-compatible: "
                f"missing={bad_missing} unexpected={incompatible.unexpected_keys}")
        fresh = [k for k in incompatible.missing_keys
                 if k.startswith(fresh_ok) and not k.startswith("trunk.")]
        pending_trunk = [k for k in incompatible.missing_keys if k.startswith("trunk.")]
        print(f"[warm_start] full model from {warm_start}"
              + (f"; {len(fresh)} refiner/contrast tensors fresh" if fresh else "")
              + (f"; {len(pending_trunk)} trunk tensors pending swap" if pending_trunk else ""),
              flush=True)
        if swap_trunk:
            load_trunk_weights(model, swap_trunk)
            print(f"[warm_start] trunk swapped in from {swap_trunk}", flush=True)
    else:
        trunk_ckpt = training_cfg.get("trunk_checkpoint_path")
        if not trunk_ckpt:
            raise ValueError(
                "training.trunk_checkpoint_path is required (the trunk must be warm), "
                "or pass training.warm_start_path for a full-model warm start")
        load_trunk_weights(model, trunk_ckpt)
    trainable = model.trainable_parameters()
    total = sum(p.numel() for p in model.parameters())
    trainable_count = sum(p.numel() for p in trainable)
    print(
        f"[model] parameters total={total:,} trainable={trainable_count:,} "
        f"frozen={total - trainable_count:,}",
        flush=True,
    )
    if not warm_start and not resume_path:
        print(f"[trunk] loaded {trunk_ckpt}; encoder+image-context frozen, decoder fresh", flush=True)

    dataset = build_dataset(data_cfg, split=str(data_cfg.get("train_split", "train")))
    num_workers = int(training_cfg.get("num_workers", 8))
    loader = DataLoader(
        dataset,
        batch_size=int(training_cfg.get("batch_size", 1)),
        shuffle=True,
        num_workers=num_workers,
        prefetch_factor=int(training_cfg.get("prefetch_factor", 4)) if num_workers > 0 else None,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        drop_last=True,
    )
    steps_per_epoch = len(loader)
    epochs = int(training_cfg.get("epochs", 500))
    max_steps = epochs * steps_per_epoch
    # Same contract as the AE/P2SD trainers: an explicit positive
    # training.max_steps bounds the run (preflights). It was silently ignored
    # here until 2026-08-12, which turned a 3-step preflight into a full run.
    raw_max_steps = int(training_cfg.get("max_steps", 0) or 0)
    if raw_max_steps > 0:
        max_steps = min(max_steps, raw_max_steps)
    opt_cfg = cfg.get("optimizer", {})
    groups = opt_cfg.get("groups") or [{}]
    base_lr = float(groups[0].get("lr", opt_cfg.get("lr", 1e-4)))
    min_lr_scale = float(opt_cfg.get("schedule", {}).get("min_lr_scale", 0.0))
    optimizer = torch.optim.AdamW(
        trainable, lr=base_lr, weight_decay=float(opt_cfg.get("weight_decay", 0.02)))
    start_step = 0
    if resume_state is not None and "optimizer" in resume_state:
        optimizer.load_state_dict(resume_state["optimizer"])
        start_step = int(resume_state.get("step", 0))
        print(f"[resume] optimizer restored; resuming cosine at step {start_step}/{max_steps}", flush=True)
    grad_clip = float(training_cfg.get("grad_clip", 1.0))
    print(f"[lr_schedule] cosine lr={base_lr:.3e} -> {base_lr * min_lr_scale:.3e} "
          f"over {max_steps} steps", flush=True)
    write_json(run_dir / "state.json", {
        "status": "running", "epochs": epochs,
        "steps_per_epoch": steps_per_epoch, "max_steps": max_steps,
    })

    step = start_step
    window = MetricWindow()
    start = time.monotonic()
    start_epoch = start_step // max(1, steps_per_epoch)
    for epoch in range(start_epoch + 1, epochs + 1):
        # Same stop-file contract as the AE/P2SD trainers (scripts/
        # request_stop.sh); this trainer ignored it until 2026-08-13.
        if should_stop(run_dir):
            save_checkpoint(run_dir / "last.pt", model=model, optimizer=optimizer,
                            step=step, cfg=cfg, best=None)
            write_json(run_dir / "state.json", {"status": "stop_requested", "step": step})
            update_heartbeat(run_dir, status="stop_requested", step=step)
            return 0
        epoch_start = time.monotonic()
        if step >= max_steps:
            break
        for batch in loader:
            if step >= max_steps:
                break
            cosine = 0.5 * (1.0 + math.cos(math.pi * step / max_steps))
            lr = base_lr * (min_lr_scale + (1.0 - min_lr_scale) * cosine)
            for group in optimizer.param_groups:
                group["lr"] = lr
            image = batch["image"].to(device, non_blocking=True).float()
            component_label = batch["component_label"].to(device, non_blocking=True)
            ignore_label = batch["ignore"].to(device, non_blocking=True)
            target = (component_label > 0).unsqueeze(1)
            valid = (ignore_label == 0).unsqueeze(1)
            contrast = None
            with autocast_context(device, dtype):
                context = model.encode_context(image)
                # Global attention (if configured) refines the FULL grid before
                # any crop -- RoPE coords and cross-sheet structure both need
                # the whole volume. The trunk always sees the full volume
                # (frozen, cheap); only the decode may run on an aligned crop --
                # ~(grid/crop)^3 less compute per step (see decode_context).
                context = model.refine_context(context)
                context, target, valid = sample_context_crop(
                    context, target, valid, crop_grid=decode_crop_grid)
                if contrast_enabled:
                    distill_only = (
                        contrast_mode == "latent_distill"
                        and bce_weight == 0.0 and dice_weight == 0.0
                        and str(contrast_cfg.get("tap", "decoder_stage")) == "context"
                    )
                    if distill_only:
                        # Frozen decoder + zero seg weights: skip the
                        # full-volume decode, train refiner+head on distill.
                        logits, embedding = None, model.embed_context(context)
                    else:
                        logits, embedding = model.decode_context_with_embedding(context)
                    if contrast_mode == "latent_distill":
                        anchors = sheet_latent_anchors(
                            distill_target_ae, component_label,
                            min_sheet_voxels=int(contrast_cfg.get("min_sheet_voxels", 500)),
                            anchor_dim=int(contrast_cfg.get("dim", 64)))
                        contrast, contrast_stats = voxel_latent_distill_loss(
                            embedding, component_label, anchors,
                            pos_samples=int(contrast_cfg.get("pos_samples", 1024)))
                    else:
                        contrast, contrast_stats = voxel_prototype_contrast_loss(
                            embedding, component_label, ignore_label,
                            temperature=float(contrast_cfg.get("temperature", 0.1)),
                            pos_samples=int(contrast_cfg.get("pos_samples", 1024)),
                            neg_samples=int(contrast_cfg.get("neg_samples", 4096)),
                            center_samples=int(contrast_cfg.get("center_samples", 2048)),
                            min_sheet_voxels=int(contrast_cfg.get("min_sheet_voxels", 500)),
                            border_width=int(contrast_cfg.get("border_width", 5)))
                else:
                    logits = model.decode_context(context)
            if logits is not None:
                bce, dice, stats = masked_bce_and_dice(logits, target, valid, pos_weight=pos_weight)
            else:
                bce = dice = context.new_zeros(())
                stats = {"valid_fraction": 0.0, "foreground_fraction": 0.0, "dice_samples": 0.0}
            loss = bce_weight * bce + dice_weight * dice
            if contrast is not None:
                # Ramp keeps the converged warm-started decoder from being
                # yanked by the new objective at step 0.
                ramp = min(1.0, epoch / max(1e-6, contrast_warmup_epochs))
                loss = loss + contrast_weight * ramp * contrast
            if not bool(torch.isfinite(loss.detach()).all().item()):
                row = {"status": "failed", "reason": "nonfinite_loss", "step": step,
                       "bce": float(bce.detach().cpu()), "dice": float(dice.detach().cpu())}
                append_jsonl(run_dir / "metrics.jsonl", row)
                write_json(run_dir / "state.json", row)
                update_heartbeat(run_dir, status="failed", step=step, loss=row["bce"])
                raise RuntimeError(f"binary-seg loss non-finite at step {step}: {row}")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(trainable, grad_clip)
            optimizer.step()
            step += 1
            window.track("grad_norm", grad_norm)
            row_metrics = {
                "loss": loss,
                "bce": bce,
                "dice": dice,
                "grad_norm": grad_norm,
                "valid_fraction": loss.new_tensor(stats["valid_fraction"]),
                "foreground_fraction": loss.new_tensor(stats["foreground_fraction"]),
                "dice_samples": loss.new_tensor(stats["dice_samples"]),
            }
            if contrast is not None:
                row_metrics["contrast"] = contrast
                for key, value in contrast_stats.items():
                    row_metrics[key] = loss.new_tensor(float(value))
            window.add(row_metrics)

        epoch_time = time.monotonic() - epoch_start
        row = {
            "step": step,
            "epoch": float(epoch),
            "lr": float(optimizer.param_groups[0]["lr"]),
            "elapsed_s": float(time.monotonic() - start),
            "epoch_time_s": float(epoch_time),
            "estimated_epoch_time_s": float(epoch_time),
            "seconds_per_step": float(epoch_time / max(1, steps_per_epoch)),
            "pos_weight": pos_weight,
            "bce_weight": bce_weight,
            "dice_weight": dice_weight,
            **window.means(),
        }
        window.reset()
        append_jsonl(run_dir / "metrics.jsonl", row)
        update_heartbeat(run_dir, status="running", step=step, loss=row.get("loss"), epoch_time_s=epoch_time)
        save_checkpoint(run_dir / "last.pt", model=model, optimizer=optimizer, step=step, cfg=cfg, best=None)
        if epoch % int(training_cfg.get("plot_interval_epochs", 25)) == 0:
            write_loss_plots(run_dir, loss_keys=("loss", "bce", "dice"))

    write_loss_plots(run_dir, loss_keys=("loss", "bce", "dice"))
    write_json(run_dir / "state.json", {
        "status": "complete", "epochs": epochs,
        "steps_per_epoch": steps_per_epoch, "max_steps": max_steps,
    })
    update_heartbeat(run_dir, status="complete", step=step)
    print(f"[done] {epochs} epochs, {step} steps", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
