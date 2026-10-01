"""Training-time input corruption for the denoising sheet AE.

The AE historically autoencoded the clean sheet mask, so at deployment it can
only smooth what it is given. Corrupting the *input* while supervising against
the *clean* mask turns it into a denoiser/inpainter: spherical holes teach it
to close erosion damage far larger than morphological closing can (radius up
to ~10 voxels vs closing's 1-2), and random voxel dropout teaches it to
re-densify thin, ragged partial sheets. The corrupted tensor is only ever the
model input; every loss target (occupancy, distance field, query distances)
stays anchored to the clean mask, which the dataset built the targets from.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class AEInputCorruptionConfig:
    apply_prob: float = 0.7
    hole_count_range: tuple[int, int] = (1, 6)
    hole_radius_range: tuple[float, float] = (2.0, 10.0)
    dropout_prob_range: tuple[float, float] = (0.0, 0.3)
    # Hole SHAPE family (user-directed 2026-08-14): rotated anisotropic
    # ellipsoids; per-axis scale = radius * uniform(axis_ratio_range) * 1.6,
    # with probability slit_prob one axis collapses to uniform(slit_thickness_
    # range) voxels (a crack), and the boundary is noise-jittered.
    axis_ratio_range: tuple[float, float] = (0.35, 1.0)
    slit_prob: float = 0.25
    slit_thickness_range: tuple[float, float] = (1.0, 2.5)
    boundary_jitter: float = 0.35
    # Guard for the empty-input BF16 hazard (RMSNorm backward blowup on empty
    # regions) and for degenerate "hallucinate the sheet from nothing" targets:
    # if corruption would leave less than this fraction of the sheet, the
    # sample falls back to its clean input.
    min_survive_fraction: float = 0.25


def resolve_ae_input_corruption(data_cfg: dict[str, Any]) -> AEInputCorruptionConfig | None:
    cfg = data_cfg.get("input_corruption", {}) or {}
    if not bool(cfg.get("enabled", False)):
        return None
    holes = cfg.get("holes", {}) or {}
    dropout = cfg.get("dropout", {}) or {}
    count_range = tuple(int(v) for v in holes.get("count_range", (1, 6)))
    radius_range = tuple(float(v) for v in holes.get("radius_range", (2.0, 10.0)))
    dropout_range = tuple(float(v) for v in dropout.get("prob_range", (0.0, 0.3)))
    if len(count_range) != 2 or count_range[0] > count_range[1] or count_range[0] < 0:
        raise ValueError(f"input_corruption.holes.count_range must be [lo, hi] with 0 <= lo <= hi, got {count_range}")
    if len(radius_range) != 2 or radius_range[0] > radius_range[1] or radius_range[0] <= 0:
        raise ValueError(f"input_corruption.holes.radius_range must be [lo, hi] with 0 < lo <= hi, got {radius_range}")
    if len(dropout_range) != 2 or not 0.0 <= dropout_range[0] <= dropout_range[1] <= 1.0:
        raise ValueError(f"input_corruption.dropout.prob_range must be [lo, hi] within [0, 1], got {dropout_range}")
    axis_ratio_range = tuple(float(v) for v in holes.get("axis_ratio_range", (0.35, 1.0)))
    slit_thickness_range = tuple(float(v) for v in holes.get("slit_thickness_range", (1.0, 2.5)))
    return AEInputCorruptionConfig(
        apply_prob=float(cfg.get("apply_prob", 0.7)),
        hole_count_range=count_range,
        hole_radius_range=radius_range,
        dropout_prob_range=dropout_range,
        min_survive_fraction=float(cfg.get("min_survive_fraction", 0.25)),
        axis_ratio_range=axis_ratio_range,
        slit_prob=float(holes.get("slit_prob", 0.25)),
        slit_thickness_range=slit_thickness_range,
        boundary_jitter=float(holes.get("boundary_jitter", 0.35)),
    )


def corrupt_sheet_mask(mask: torch.Tensor, cfg: AEInputCorruptionConfig) -> torch.Tensor:
    """Return a corrupted copy of a [B, 1, D, H, W] binary sheet mask.

    Per sample (with probability ``apply_prob``): drop foreground voxels with a
    rate drawn from ``dropout_prob_range`` (partial-erosion look), then erase
    ``hole_count_range`` spheres of radius ``hole_radius_range`` centered on
    foreground voxels (holes in the sheet). Uses the global torch RNG, so runs
    stay reproducible under the trainer's RNG capture as long as corruption
    happens at a fixed point in the step.
    """
    device = mask.device
    corrupted = mask.clone()
    for sample_index in range(mask.shape[0]):
        if float(torch.rand((), device=device)) >= cfg.apply_prob:
            continue
        volume = corrupted[sample_index, 0]
        foreground = volume.nonzero()
        clean_count = int(foreground.shape[0])
        if clean_count == 0:
            continue
        drop_low, drop_high = cfg.dropout_prob_range
        drop_prob = float(torch.empty((), device=device).uniform_(drop_low, drop_high))
        if drop_prob > 0:
            dropped = torch.rand(clean_count, device=device) < drop_prob
            drop_indices = foreground[dropped]
            volume[drop_indices[:, 0], drop_indices[:, 1], drop_indices[:, 2]] = 0.0
        count_low, count_high = cfg.hole_count_range
        hole_count = int(torch.randint(count_low, count_high + 1, (), device=device))
        for _ in range(hole_count):
            center = foreground[int(torch.randint(clean_count, (), device=device))]
            radius = float(torch.empty((), device=device).uniform_(*cfg.hole_radius_range))
            axes = torch.empty(3, device=device).uniform_(*cfg.axis_ratio_range) * radius * 1.6
            if float(torch.rand((), device=device)) < cfg.slit_prob:
                axes[int(torch.randint(3, (), device=device))] = float(
                    torch.empty((), device=device).uniform_(*cfg.slit_thickness_range))
            _erase_ellipsoid(volume, center, axes, jitter=cfg.boundary_jitter)
        survive_fraction = float(volume.sum()) / float(clean_count)
        if survive_fraction < cfg.min_survive_fraction:
            corrupted[sample_index, 0] = mask[sample_index, 0]
    return corrupted


def _erase_sphere(volume: torch.Tensor, center_zyx: torch.Tensor, radius: float) -> None:
    reach = int(math.ceil(radius))
    lows = [max(int(center_zyx[axis]) - reach, 0) for axis in range(3)]
    highs = [min(int(center_zyx[axis]) + reach + 1, int(volume.shape[axis])) for axis in range(3)]
    if any(low >= high for low, high in zip(lows, highs)):
        return
    axes = [
        torch.arange(low, high, device=volume.device, dtype=torch.float32) - float(center_zyx[axis])
        for axis, (low, high) in enumerate(zip(lows, highs))
    ]
    grid_z, grid_y, grid_x = torch.meshgrid(axes[0], axes[1], axes[2], indexing="ij")
    sphere = grid_z.pow(2) + grid_y.pow(2) + grid_x.pow(2) <= radius * radius
    region = volume[lows[0]:highs[0], lows[1]:highs[1], lows[2]:highs[2]]
    region[sphere] = 0.0


def _erase_ellipsoid(
    volume: torch.Tensor,
    center_zyx: torch.Tensor,
    axes: torch.Tensor,
    *,
    jitter: float = 0.0,
) -> None:
    """Erase a randomly ROTATED anisotropic ellipsoid with a ragged boundary.

    Rotation via a random orthonormal basis (QR of a Gaussian matrix, global
    torch RNG); boundary raggedness via per-voxel noise added to the
    normalized ellipsoid distance (``jitter`` = noise amplitude), so edges are
    irregular rather than clean quadric surfaces.
    """
    device = volume.device
    reach = int(math.ceil(float(axes.max()) * (1.0 + jitter)))
    lows = [max(int(center_zyx[axis]) - reach, 0) for axis in range(3)]
    highs = [min(int(center_zyx[axis]) + reach + 1, int(volume.shape[axis])) for axis in range(3)]
    if any(low >= high for low, high in zip(lows, highs)):
        return
    grids = torch.meshgrid(
        *[torch.arange(low, high, device=device, dtype=torch.float32)
          for low, high in zip(lows, highs)],
        indexing="ij",
    )
    offsets = torch.stack([g - float(center_zyx[i]) for i, g in enumerate(grids)], dim=-1)
    basis, _ = torch.linalg.qr(torch.randn(3, 3, device=device))
    local = offsets @ basis
    distance = (local / axes.clamp_min(0.75)).pow(2).sum(dim=-1)
    if jitter > 0:
        distance = distance + jitter * torch.randn_like(distance)
    region = tuple(slice(low, high) for low, high in zip(lows, highs))
    volume[region] = torch.where(distance <= 1.0, torch.zeros_like(volume[region]), volume[region])
