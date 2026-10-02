"""Estimate target AE latent channel statistics for P2SD normalization."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from vesuvius_p2sd.train.common import (
    amp_dtype,
    autocast_context,
    get_device,
    load_ae_from_config,
    make_loader,
    maybe_channels_last_3d,
    move_batch,
    seed_everything,
)
from vesuvius_p2sd.train.train_p2sd import build_p2sd_pair_batch
from vesuvius_p2sd.utils.config import load_config


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--max_batches", type=int, default=64)
    parser.add_argument("--use_pair_sampling", action=argparse.BooleanOptionalAction, default=True)
    args, overrides = parser.parse_known_args(argv)

    cfg = load_config(args.config_path, overrides)
    stats = estimate_latent_stats(
        cfg,
        split=args.split,
        max_batches=int(args.max_batches),
        use_pair_sampling=bool(args.use_pair_sampling),
    )
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(stats, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(stats, sort_keys=True), flush=True)
    return 0


@torch.no_grad()
def estimate_latent_stats(
    cfg: dict[str, Any],
    *,
    split: str,
    max_batches: int,
    use_pair_sampling: bool,
) -> dict[str, Any]:
    seed_everything(int(cfg.get("seed", 42)))
    device = get_device(cfg)
    target_cfg = cfg.get("target_ae", {})
    target_ae, _ = load_ae_from_config(
        target_cfg.get("config_path", "configs/ae/single_sheet_sparse_5down_dense_input.yaml"),
        target_cfg.get("checkpoint_path"),
    )
    target_ae = target_ae.to(device).eval()
    channels_last = bool(cfg.get("training", {}).get("channels_last_3d", False))
    if channels_last:
        target_ae = target_ae.to(memory_format=torch.channels_last_3d)
    loader = make_loader(cfg, split, shuffle=False)
    dtype = amp_dtype(cfg.get("training", {}).get("amp_dtype"))

    sum_c = None
    sumsq_c = None
    count = 0
    batches = 0
    samples = 0
    for batch_index, batch in enumerate(loader):
        if batch_index >= max_batches:
            break
        batch = move_batch(batch, device)
        if use_pair_sampling:
            pairs = build_p2sd_pair_batch(batch, cfg, device)
            mask = pairs["target_mask"].float()
        else:
            mask = batch["mask"].float()
        mask = maybe_channels_last_3d(mask, channels_last)
        with autocast_context(device, dtype):
            z = target_ae.encode(mask)
        z = z.float()
        reduce_dims = (0, 2, 3, 4)
        batch_sum = z.sum(dim=reduce_dims)
        batch_sumsq = z.pow(2).sum(dim=reduce_dims)
        if sum_c is None:
            sum_c = torch.zeros_like(batch_sum)
            sumsq_c = torch.zeros_like(batch_sumsq)
        sum_c += batch_sum
        sumsq_c += batch_sumsq
        count += int(z.shape[0] * z.shape[2] * z.shape[3] * z.shape[4])
        samples += int(z.shape[0])
        batches += 1

    if count <= 0 or sum_c is None or sumsq_c is None:
        raise RuntimeError("No latent samples were produced for statistics")

    mean = sum_c / float(count)
    var = (sumsq_c / float(count) - mean.pow(2)).clamp_min(0.0)
    std = var.sqrt().clamp_min(1e-6)
    return {
        "format": "vesuvius_p2sd_latent_channel_stats_v1",
        "split": split,
        "batches": batches,
        "samples": samples,
        "count_per_channel": count,
        "latent_channels": int(mean.numel()),
        "use_pair_sampling": bool(use_pair_sampling),
        "mean": [float(v) for v in mean.cpu()],
        "std": [float(v) for v in std.cpu()],
        "rms_mean": float((sumsq_c / float(count)).sqrt().mean().cpu()),
        "std_mean": float(std.mean().cpu()),
        "std_min": float(std.min().cpu()),
        "std_max": float(std.max().cpu()),
        "mean_abs": float(mean.abs().mean().cpu()),
    }


if __name__ == "__main__":
    raise SystemExit(main())
