"""GPU construction of configurable AE signed/unsigned distance targets."""

from __future__ import annotations

from typing import Any

import torch


CPU_SCIPY_BACKEND = "cpu_scipy"
GPU_CUCIM_BACKEND = "gpu_cucim"
DISTANCE_TARGET_BACKENDS = {CPU_SCIPY_BACKEND, GPU_CUCIM_BACKEND}


def distance_target_backend(data_cfg: dict[str, Any]) -> str:
    backend = str(data_cfg.get("distance_target_backend", CPU_SCIPY_BACKEND))
    if backend not in DISTANCE_TARGET_BACKENDS:
        choices = "|".join(sorted(DISTANCE_TARGET_BACKENDS))
        raise ValueError(f"data.distance_target_backend must be {choices}, got {backend!r}")
    return backend


def prepare_ae_distance_targets(
    batch: dict[str, Any],
    data_cfg: dict[str, Any],
) -> dict[str, Any]:
    """Build configured cuCIM distance targets after a batch reaches CUDA."""
    if distance_target_backend(data_cfg) != GPU_CUCIM_BACKEND:
        return batch

    distance_cfg = data_cfg.get("distance_field", {})
    query_cfg = data_cfg.get("query_distance", {})
    distance_enabled = bool(distance_cfg.get("enabled", False))
    query_enabled = bool(query_cfg.get("enabled", False))
    if not distance_enabled and not query_enabled:
        return batch

    mask = batch.get("mask")
    if not torch.is_tensor(mask):
        raise RuntimeError("GPU distance targets require batch['mask']")
    if mask.device.type != "cuda":
        raise RuntimeError("data.distance_target_backend=gpu_cucim requires a CUDA device")
    if mask.ndim != 5 or mask.shape[1] != 1:
        raise ValueError(f"AE masks must have shape [B, 1, D, H, W], got {tuple(mask.shape)}")

    binary_mask = mask[:, 0] > 0.5
    signed_voxels = signed_distance_edt_cucim(binary_mask)
    clip_voxels = max(
        float(distance_cfg.get("clip_voxels", query_cfg.get("clip_voxels", 16.0))),
        1.0,
    )
    target_mode = str(distance_cfg.get("target_mode", "signed"))
    normalized = normalize_distance_target(
        signed_voxels,
        clip_voxels=clip_voxels,
        target_mode=target_mode,
    )

    if distance_enabled:
        batch["distance_field"] = normalized.unsqueeze(1)
    if query_enabled:
        points, distances = sample_query_distance_points_gpu(
            signed_voxels=signed_voxels,
            normalized_distance=normalized,
            count=int(query_cfg.get("num_points", 0)),
            band_voxels=float(query_cfg.get("band_voxels", 8.0)),
            sampling=str(query_cfg.get("sampling", "near_band_uniform")),
            inside_fraction=float(query_cfg.get("inside_fraction", 1.0 / 3.0)),
            near_outside_fraction=float(query_cfg.get("near_outside_fraction", 1.0 / 3.0)),
        )
        batch["query_points"] = points
        batch["query_distances"] = distances
        if bool(query_cfg.get("subvoxel_jitter", False)):
            points, signed_normalized = jitter_query_points_subvoxel(
                points,
                signed_voxels=signed_voxels,
                clip_voxels=clip_voxels,
            )
            batch["query_points"] = points
            batch["query_signed_distances"] = signed_normalized
            batch["query_distances"] = signed_normalized.abs()
            batch["query_occupancy"] = (signed_normalized < 0).float()
    return batch


def jitter_query_points_subvoxel(
    points_zyx: torch.Tensor,
    *,
    signed_voxels: torch.Tensor,
    clip_voxels: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Jitter voxel-center queries by U(-0.5, 0.5) with interpolated signed targets.

    Lattice-only supervision leaves the decision boundary between voxel
    centers unconstrained (~0.5 voxel slack per side). Interpolating the
    *signed* EDT at continuous positions constrains the zero crossing
    sub-voxel: the signed field is linear through the boundary, where the
    unsigned field has a kink and would interpolate wrongly. Occupancy labels
    become the sign of the interpolated field.
    """
    if points_zyx.numel() == 0:
        return points_zyx, points_zyx.new_zeros(points_zyx.shape[:-1])
    shape = signed_voxels.shape[-3:]
    jittered = points_zyx + (torch.rand_like(points_zyx) - 0.5)
    for axis in range(3):
        jittered[..., axis].clamp_(0.0, float(shape[axis] - 1))
    field = (signed_voxels / float(clip_voxels)).clamp(-1.0, 1.0).unsqueeze(1)
    unit = (jittered + 0.5) / jittered.new_tensor([float(s) for s in shape])
    grid = torch.stack(
        [unit[..., 2], unit[..., 1], unit[..., 0]], dim=-1).mul(2.0).sub(1.0)
    sampled = torch.nn.functional.grid_sample(
        field.float(),
        grid.view(points_zyx.shape[0], -1, 1, 1, 3).float(),
        mode="bilinear",
        padding_mode="border",
        align_corners=False,
    )
    return jittered, sampled[:, 0, :, 0, 0]


def signed_distance_edt_cucim(binary_mask: torch.Tensor) -> torch.Tensor:
    """Return exact outside-minus-inside EDT for CUDA masks shaped [B, D, H, W]."""
    if binary_mask.device.type != "cuda":
        raise RuntimeError("cuCIM EDT requires a CUDA tensor")
    if binary_mask.ndim != 4:
        raise ValueError(f"binary_mask must have shape [B, D, H, W], got {tuple(binary_mask.shape)}")
    try:
        import cupy as cp
        from cucim.core.operations.morphology import distance_transform_edt
    except ImportError as exc:  # pragma: no cover - depends on optional environment
        raise ImportError(
            "gpu_cucim distance targets require the gpu-edt extra: "
            "pip install -e '.[gpu-edt]'"
        ) from exc

    signed = torch.empty(binary_mask.shape, device=binary_mask.device, dtype=torch.float32)
    for batch_index in range(binary_mask.shape[0]):
        cupy_mask = cp.from_dlpack(binary_mask[batch_index].contiguous())
        inside = distance_transform_edt(cupy_mask, float64_distances=False)
        signed[batch_index].copy_(-torch.from_dlpack(inside))
        outside = distance_transform_edt(~cupy_mask, float64_distances=False)
        signed[batch_index].add_(torch.from_dlpack(outside))
    return signed


def normalize_distance_target(
    signed_voxels: torch.Tensor,
    *,
    clip_voxels: float,
    target_mode: str,
) -> torch.Tensor:
    """Normalize signed distance or unsigned distance-to-boundary targets."""
    if target_mode == "signed":
        return signed_voxels.clamp(-clip_voxels, clip_voxels).div(clip_voxels)
    if target_mode == "unsigned":
        return signed_voxels.abs().clamp(0.0, clip_voxels).div(clip_voxels)
    raise ValueError(f"distance target_mode must be signed|unsigned, got {target_mode!r}")


def sample_query_distance_points_gpu(
    *,
    signed_voxels: torch.Tensor,
    normalized_distance: torch.Tensor,
    count: int,
    band_voxels: float,
    sampling: str = "near_band_uniform",
    inside_fraction: float = 1.0 / 3.0,
    near_outside_fraction: float = 1.0 / 3.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample legacy or explicitly sign-balanced point-query supervision."""
    count = int(count)
    batch_size, depth, height, width = signed_voxels.shape
    if count <= 0:
        points = signed_voxels.new_zeros((batch_size, 0, 3))
        distances = signed_voxels.new_zeros((batch_size, 0))
        return points, distances

    if sampling == "balanced_regions":
        return _sample_balanced_query_distance_points_gpu(
            signed_voxels=signed_voxels,
            normalized_distance=normalized_distance,
            count=count,
            band_voxels=band_voxels,
            inside_fraction=inside_fraction,
            near_outside_fraction=near_outside_fraction,
        )
    if sampling != "near_band_uniform":
        raise ValueError(f"query sampling must be near_band_uniform|balanced_regions, got {sampling!r}")
    band_count = count // 2
    random_count = count - band_count
    total_voxels = int(depth * height * width)
    batch_points = []
    batch_distances = []
    for batch_index in range(batch_size):
        sample_points = []
        sample_random_count = random_count
        if band_count > 0:
            band = torch.nonzero(
                signed_voxels[batch_index].abs() <= float(band_voxels),
                as_tuple=False,
            )
            if band.shape[0] > 0:
                indices = torch.randint(band.shape[0], (band_count,), device=band.device)
                sample_points.append(band[indices])
            else:
                sample_random_count += band_count
        if sample_random_count > 0:
            flat = torch.randint(total_voxels, (sample_random_count,), device=signed_voxels.device)
            z = torch.div(flat, height * width, rounding_mode="floor")
            remainder = flat.remainder(height * width)
            y = torch.div(remainder, width, rounding_mode="floor")
            x = remainder.remainder(width)
            sample_points.append(torch.stack((z, y, x), dim=1))
        points = torch.cat(sample_points, dim=0)
        points = points[torch.randperm(points.shape[0], device=points.device)]
        zyx = points.unbind(dim=1)
        distances = normalized_distance[batch_index][zyx]
        batch_points.append(points.float())
        batch_distances.append(distances)
    return torch.stack(batch_points), torch.stack(batch_distances)


def _sample_balanced_query_distance_points_gpu(
    *,
    signed_voxels: torch.Tensor,
    normalized_distance: torch.Tensor,
    count: int,
    band_voxels: float,
    inside_fraction: float,
    near_outside_fraction: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Equalize gradients from the thin interior and both exterior regions."""
    if not 0.0 <= inside_fraction <= 1.0 or not 0.0 <= near_outside_fraction <= 1.0:
        raise ValueError("query region fractions must lie in [0, 1]")
    if inside_fraction + near_outside_fraction > 1.0:
        raise ValueError("inside_fraction + near_outside_fraction must not exceed 1")
    counts = (int(round(count * inside_fraction)), int(round(count * near_outside_fraction)))
    counts = (*counts, count - sum(counts))
    batch_points, batch_distances = [], []
    for batch_index in range(signed_voxels.shape[0]):
        signed = signed_voxels[batch_index]
        regions = (
            signed < 0,
            (signed > 0) & (signed <= float(band_voxels)),
            signed > float(band_voxels),
        )
        parts = []
        for region, region_count in zip(regions, counts, strict=True):
            if region_count <= 0:
                continue
            candidates = torch.nonzero(region, as_tuple=False)
            if candidates.shape[0] == 0:
                candidates = torch.nonzero(signed > 0, as_tuple=False)
            indices = torch.randint(candidates.shape[0], (region_count,), device=signed.device)
            parts.append(candidates[indices])
        points = torch.cat(parts, dim=0)
        points = points[torch.randperm(points.shape[0], device=points.device)]
        zyx = points.unbind(dim=1)
        batch_points.append(points.float())
        batch_distances.append(normalized_distance[batch_index][zyx])
    return torch.stack(batch_points), torch.stack(batch_distances)
