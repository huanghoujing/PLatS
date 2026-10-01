"""Repo-local Vesuvius sheet datasets."""

from __future__ import annotations

import json
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy import ndimage
from torch.utils.data import Dataset

from vesuvius_p2sd.data.distance_targets import CPU_SCIPY_BACKEND, distance_target_backend


@dataclass(frozen=True)
class PatchAugmentationConfig:
    """Training-only image and geometry augmentations for one patch."""

    flip_p: float = 0.0
    xy_rotate_p: float = 0.0
    xy_rand_rotate_p: float = 0.0
    translation_max_pad_voxels: int = 0
    # Probability of applying pad-then-crop translation on a sample (when
    # translation_max_pad_voxels > 0). Default 1.0 keeps the historical
    # always-on behavior; lower values leave some samples in canonical position.
    translation_p: float = 1.0
    photometric_device: str = "cpu"
    gaussian_noise_p: float = 0.0
    gaussian_noise_std: tuple[float, float] = (0.01, 0.1)
    intensity_p: float = 0.0
    intensity_scale: tuple[float, float] = (0.9, 1.1)
    intensity_shift: tuple[float, float] = (-0.1, 0.1)
    gamma_p: float = 0.0
    gamma: tuple[float, float] = (0.8, 1.2)
    # Zoom-in ("enlarging spacing", user directive 2026-09-05): with probability zoom_p a sub-crop of side
    # patch / s (s ~ U[zoom_range]) containing the anchor is resampled to the patch size (image trilinear, labels
    # nearest), so sheets appear s x thicker and s x further apart -- the model learns to decode upsampled
    # (compressed) regions instead of shattering on them.
    zoom_p: float = 0.0
    zoom_range: tuple[float, float] = (1.0, 2.0)
    # After zooming, thin every sheet back to its medial ~zoom_thin_voxels voxels (user 2026-09-05: "thin the sheet to
    # be ~3 voxel if too thick") so the label convention (a thin surface band) and the AE's thin-sheet prior hold at
    # every scale; 0 = keep the s x thicker resampled labels.
    zoom_thin_voxels: int = 0

    @property
    def has_geometric_transform(self) -> bool:
        return (
            self.flip_p > 0.0
            or self.xy_rotate_p > 0.0
            or self.xy_rand_rotate_p > 0.0
            or self.zoom_p > 0.0
            or (self.translation_max_pad_voxels > 0 and self.translation_p > 0.0)
        )


@dataclass(frozen=True)
class PatchSampleConfig:
    patch_size: tuple[int, int, int] = (64, 64, 64)
    min_component_voxels: int = 1000
    component_select_mode: str = "random"
    num_positive_points: int = 1
    num_negative_points: int = 0
    distance_field_enabled: bool = False
    distance_clip_voxels: float = 16.0
    distance_target_mode: str = "signed"
    query_distance_enabled: bool = False
    query_distance_points: int = 0
    query_distance_band_voxels: float = 8.0
    query_distance_sampling: str = "near_band_uniform"
    query_distance_inside_fraction: float = 1.0 / 3.0
    query_distance_near_outside_fraction: float = 1.0 / 3.0
    randomize_each_access: bool = False
    # Load per-case ignore.npy (unlabeled region + erased border; see
    # data/build_ignore_masks.py) and return it, transformed alongside the
    # labels, as sample["ignore"]. Requires the *_ignore manifests.
    load_ignore: bool = False
    load_vertices: bool = False
    max_vertices: int = 4096
    node_weight_radius: int = 0
    node_weight_mid: float = 1.0
    augmentation: PatchAugmentationConfig = field(default_factory=PatchAugmentationConfig)
    # P2SD pair sampling rebuilds targets from component_label, so its train
    # loader can omit this otherwise large float32 tensor.
    include_mask: bool = True
    max_retries: int = 24
    samples_per_epoch: int | None = None
    # Per-worker LRU: each case otherwise retains two mmap file descriptors
    # and an optional packed ignore array for the lifetime of a persistent worker.
    case_cache_size: int = 16
    seed: int = 42


@dataclass(frozen=True)
class ComponentStat:
    component_id: int
    count: int
    bbox: tuple[tuple[int, int], ...]


class VesuviusSheetPatchDataset(Dataset):
    """Sample image crops, component labels, and optional sheet masks from a manifest."""

    def __init__(
        self,
        *,
        dataset_root: str | Path,
        split: str,
        cfg: PatchSampleConfig,
    ) -> None:
        self.dataset_root = Path(dataset_root)
        self.split = split
        self.cfg = cfg
        manifest_name = f"manifest_{split}_ignore.jsonl" if cfg.load_ignore else f"manifest_{split}.jsonl"
        manifest = self.dataset_root / manifest_name
        if not manifest.exists():
            raise FileNotFoundError(manifest)
        self.rows = load_jsonl(manifest)
        if not self.rows:
            raise ValueError(f"No rows in {manifest}")
        if cfg.case_cache_size < 0:
            raise ValueError("case_cache_size must be nonnegative")
        self._case_cache: OrderedDict[int, tuple[np.ndarray, np.ndarray, np.ndarray | None]] = OrderedDict()
        self._component_stats_cache: dict[int, list[ComponentStat]] = {}
        self._component_stats_preload = load_component_stats_cache(self.dataset_root, split)
        self._access_count = 0

    def __len__(self) -> int:
        if self.cfg.samples_per_epoch is not None:
            return int(self.cfg.samples_per_epoch)
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row_index = index % len(self.rows)
        access = self._next_access_count() if self.cfg.randomize_each_access else 0
        rng = np.random.default_rng(self.cfg.seed + index * 1009 + access * 9176)
        image, comps, packed_ignore = self._load_case(row_index)
        for _ in range(self.cfg.max_retries):
            sample, target_voxels = self._sample_once(image, comps, packed_ignore, row_index, rng)
            if target_voxels >= self.cfg.min_component_voxels:
                return sample
        sample, _ = self._sample_once(image, comps, packed_ignore, row_index, rng)
        return sample

    def _next_access_count(self) -> int:
        value = self._access_count
        self._access_count += 1
        return value

    def _load_case(self, row_index: int) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
        cached = self._case_cache.get(row_index)
        if cached is not None:
            self._case_cache.move_to_end(row_index)
            return cached
        row = self.rows[row_index]
        image = np.load(row["image_path"], mmap_mode="r")
        comps = np.load(row["components_path"], mmap_mode="r")
        packed_ignore = None
        if self.cfg.load_ignore:
            if "ignore_path" not in row:
                raise KeyError(f"load_ignore=True but row {row.get('case_id')} has no ignore_path")
            # Cache the bit-packed form (~4 MB/case); unpack per access.
            packed_ignore = np.load(row["ignore_path"])
        if self.cfg.case_cache_size > 0:
            self._case_cache[row_index] = (image, comps, packed_ignore)
            while len(self._case_cache) > self.cfg.case_cache_size:
                # Drop ownership rather than forcibly closing an mmap that a
                # caller may still reference. Sample crops are owned copies.
                self._case_cache.popitem(last=False)
        return image, comps, packed_ignore

    def _load_vertices(self, row_index: int):
        """vertices.npz next to components.npy -> (xyz_zyx float32 [N,3], normal_zyx float32 [N,3], component int64 [N]) or None."""
        if not self.cfg.load_vertices:
            return None
        if not hasattr(self, "_vertex_cache"):
            self._vertex_cache = OrderedDict()
        if row_index in self._vertex_cache:
            self._vertex_cache.move_to_end(row_index)
            return self._vertex_cache[row_index]
        row = self.rows[row_index]
        path = Path(row["components_path"]).parent / "vertices.npz"
        out = None
        if path.exists():
            with np.load(path) as v:
                if len(v["xyz"]):
                    out = (
                        np.ascontiguousarray(v["xyz"][:, ::-1].astype(np.float32)),
                        np.ascontiguousarray(v["normal"][:, ::-1].astype(np.float32)),
                        v["component"].astype(np.int64),
                    )
        if self.cfg.case_cache_size > 0:
            self._vertex_cache[row_index] = out
            while len(self._vertex_cache) > self.cfg.case_cache_size:
                self._vertex_cache.popitem(last=False)
        return out


    def _sample_once(
        self,
        image: np.ndarray,
        comps: np.ndarray,
        packed_ignore: np.ndarray | None,
        row_index: int,
        rng: np.random.Generator,
    ) -> tuple[dict[str, Any], int]:
        row = self.rows[row_index]
        stats = self._component_stats(row_index, comps)
        comp = choose_component_stat(
            stats,
            min_voxels=self.cfg.min_component_voxels,
            mode=self.cfg.component_select_mode,
            rng=rng,
        )
        comp_id = comp.component_id if comp is not None else 0
        if comp is None:
            center = np.array(image.shape) // 2
        else:
            center = sample_component_center(comps, comp, rng)
        start = crop_start_around(center, image.shape, self.cfg.patch_size, rng)
        output_offset = sample_crop_output_offset(
            image.shape,
            start,
            self.cfg.patch_size,
            rng,
        )
        # Keep the stored image dtype through the CPU loader. The trainers cast
        # after the non-blocking device transfer, avoiding an unnecessary PS320
        # float32 host copy and halving the transfer payload for float16 data.
        image_crop = np.array(
            crop_array(image, start, self.cfg.patch_size, output_offset=output_offset),
            copy=True,
        )
        comp_crop = np.array(
            crop_array(comps, start, self.cfg.patch_size, output_offset=output_offset),
            copy=True,
        )
        ignore_crop = None
        if packed_ignore is not None:
            full_ignore = np.unpackbits(packed_ignore, count=int(np.prod(image.shape)))
            full_ignore = full_ignore.reshape(image.shape).astype(bool)
            ignore_crop = np.array(
                crop_array(full_ignore, start, self.cfg.patch_size, output_offset=output_offset),
                copy=True,
            )
        anchor_point = center - start + output_offset if comp is not None else None
        verts = self._load_vertices(row_index)
        v_pts = v_nrm = v_comp = None
        if verts is not None:
            vx, vn, vc = verts
            rel = vx - np.asarray(start, np.float32) + np.asarray(output_offset, np.float32)
            keep = (rel >= -0.5).all(1) & (rel < np.asarray(self.cfg.patch_size, np.float32) - 0.5).all(1)
            v_pts, v_nrm, v_comp = rel[keep], vn[keep], vc[keep]
        augmented = apply_patch_augmentations(
            image_crop,
            comp_crop,
            self.cfg.augmentation,
            rng,
            anchor_point=anchor_point,
            extra_label=ignore_crop,
            points=v_pts,
            normals=v_nrm,
        )
        if v_pts is not None:
            augmented, v_pts, v_nrm = augmented[:-2], augmented[-2], augmented[-1]
        if ignore_crop is not None:
            image_crop, comp_crop, ignore_crop = augmented
        else:
            image_crop, comp_crop = augmented
        mask = (comp_crop == comp_id).astype(np.float32)
        # Spatial transforms happen before prompt sampling. The original crop
        # center is no longer meaningful after a flip or rotation, so sample
        # all positive prompts from the transformed component instead.
        preferred_pos = (
            [center - start + output_offset]
            if comp is not None and not self.cfg.augmentation.has_geometric_transform
            else None
        )
        prompt_pos = sample_points(mask > 0, self.cfg.num_positive_points, rng, preferred_points=preferred_pos)
        prompt_neg = sample_negative_points(comp_crop, mask > 0, self.cfg.num_negative_points, rng)
        prompt_points, prompt_labels = combine_prompt_points(prompt_pos, prompt_neg)
        sample = {
            "image": torch.from_numpy(image_crop[None]),
            # Keep the on-disk dtype (uint8 for this data): the int32 cast used
            # here previously quadrupled the loader's memory traffic for this
            # tensor (~131 MB -> 33 MB per PS320 sample). Consumers that need
            # wider integers call .long() after the device transfer.
            "component_label": torch.from_numpy(np.ascontiguousarray(comp_crop)),
            "prompt_points": torch.from_numpy(prompt_points.astype(np.float32, copy=False)),
            "prompt_labels": torch.from_numpy(prompt_labels.astype(np.int64, copy=False)),
            "prompt_zyx": tuple(int(v) for v in prompt_pos[0]) if len(prompt_pos) else None,
            "case_id": row["case_id"],
            "component_id": int(comp_id),
            "crop_start": tuple(int(v) for v in start),
        }
        if ignore_crop is not None:
            sample["ignore"] = torch.from_numpy(ignore_crop.astype(np.uint8, copy=False))
        if self.cfg.include_mask:
            sample["mask"] = torch.from_numpy(mask[None])
        if self.cfg.load_vertices:
            M = int(self.cfg.max_vertices)
            pts_pad = np.zeros((M, 3), np.float32); nrm_pad = np.zeros((M, 3), np.float32); comp_pad = np.zeros((M,), np.int64); valid_pad = np.zeros((M,), bool)
            if v_pts is not None and len(v_pts):
                # the component id under the (transformed) node voxel, from the transformed labels
                q = np.clip(np.round(v_pts).astype(np.int64), 0, np.asarray(comp_crop.shape) - 1)
                v_comp = comp_crop[q[:, 0], q[:, 1], q[:, 2]].astype(np.int64)
                if len(v_pts) > M:
                    sel = rng.choice(len(v_pts), M, replace=False); v_pts, v_nrm, v_comp = v_pts[sel], v_nrm[sel], v_comp[sel]
                k = len(v_pts); pts_pad[:k] = v_pts; nrm_pad[:k] = v_nrm; comp_pad[:k] = v_comp; valid_pad[:k] = True
            sample["vertices_zyx"] = torch.from_numpy(pts_pad); sample["vertex_normals_zyx"] = torch.from_numpy(nrm_pad)
            sample["vertex_component"] = torch.from_numpy(comp_pad); sample["vertex_valid"] = torch.from_numpy(valid_pad)
            if self.cfg.node_weight_radius > 0:
                weight = np.full(comp_crop.shape, 255, np.uint8)
                band = comp_crop > 0
                if band.any():
                    near = np.zeros(comp_crop.shape, bool)
                    if v_pts is not None and len(v_pts):
                        q = np.clip(np.round(v_pts).astype(np.int64), 0, np.asarray(comp_crop.shape) - 1)
                        near[q[:, 0], q[:, 1], q[:, 2]] = True
                        near = ndimage.binary_dilation(near, iterations=int(self.cfg.node_weight_radius))
                    weight[band & ~near] = np.uint8(round(255 * float(self.cfg.node_weight_mid)))
                sample["dense_weight"] = torch.from_numpy(weight)
        add_optional_distance_targets(sample, mask > 0, self.cfg, rng)
        return sample, int(mask.sum())

    def _component_stats(self, row_index: int, comps: np.ndarray) -> list[ComponentStat]:
        cached = self._component_stats_cache.get(row_index)
        if cached is not None:
            return cached
        row = self.rows[row_index]
        preloaded = self._component_stats_preload.get(str(row.get("case_id", "")))
        if preloaded is not None:
            cached_components_path, stats = preloaded
            if not cached_components_path or str(cached_components_path) == str(row.get("components_path", "")):
                self._component_stats_cache[row_index] = stats
                return stats
        stats = build_component_stats(comps)
        self._component_stats_cache[row_index] = stats
        return stats


class SyntheticSheetDataset(Dataset):
    """Tiny deterministic dataset for shape tests and CPU/GPU smoke runs."""

    def __init__(
        self,
        *,
        length: int = 8,
        patch_size: tuple[int, int, int] = (64, 64, 64),
        seed: int = 0,
        num_positive_points: int = 1,
        num_negative_points: int = 0,
        distance_field_enabled: bool = False,
        distance_clip_voxels: float = 16.0,
        distance_target_mode: str = "signed",
        query_distance_enabled: bool = False,
        query_distance_points: int = 0,
        query_distance_band_voxels: float = 8.0,
        query_distance_sampling: str = "near_band_uniform",
        query_distance_inside_fraction: float = 1.0 / 3.0,
        query_distance_near_outside_fraction: float = 1.0 / 3.0,
        include_mask: bool = True,
    ) -> None:
        self.length = int(length)
        self.patch_size = patch_size
        self.seed = int(seed)
        self.num_positive_points = int(num_positive_points)
        self.num_negative_points = int(num_negative_points)
        self.distance_field_enabled = bool(distance_field_enabled)
        self.distance_clip_voxels = float(distance_clip_voxels)
        self.distance_target_mode = str(distance_target_mode)
        self.query_distance_enabled = bool(query_distance_enabled)
        self.query_distance_points = int(query_distance_points)
        self.query_distance_band_voxels = float(query_distance_band_voxels)
        self.query_distance_sampling = str(query_distance_sampling)
        self.query_distance_inside_fraction = float(query_distance_inside_fraction)
        self.query_distance_near_outside_fraction = float(query_distance_near_outside_fraction)
        self.include_mask = bool(include_mask)

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> dict[str, Any]:
        rng = np.random.default_rng(self.seed + index)
        shape = self.patch_size
        zdim, ydim, xdim = shape
        image = rng.normal(0.0, 0.03, size=shape).astype(np.float32)
        zz = zdim // 2 + int(rng.integers(-2, 3))
        y_grid, x_grid = np.meshgrid(np.arange(ydim), np.arange(xdim), indexing="ij")
        curve = ydim // 2 + 5 * np.sin((x_grid - xdim / 2) / 9.0)
        sheet2d = np.abs(y_grid - curve) <= max(2, ydim // 12)
        mask = np.zeros(shape, dtype=bool)
        for dz in (-1, 0, 1):
            z = np.clip(zz + dz, 0, zdim - 1)
            mask[z] = sheet2d
        mask = ndimage.binary_closing(mask, iterations=1)
        image[mask] += 1.0
        comp = mask.astype(np.int32)
        prompt_pos = sample_points(mask, self.num_positive_points, rng)
        prompt_neg = sample_negative_points(comp, mask, self.num_negative_points, rng)
        prompt_points, prompt_labels = combine_prompt_points(prompt_pos, prompt_neg)
        sample = {
            "image": torch.from_numpy(image[None]),
            "component_label": torch.from_numpy(comp),
            "prompt_points": torch.from_numpy(prompt_points.astype(np.float32, copy=False)),
            "prompt_labels": torch.from_numpy(prompt_labels.astype(np.int64, copy=False)),
            "prompt_zyx": tuple(int(v) for v in prompt_pos[0]),
            "case_id": f"synthetic_{index:04d}",
            "component_id": 1,
            "crop_start": (0, 0, 0),
        }
        if self.include_mask:
            sample["mask"] = torch.from_numpy(mask.astype(np.float32)[None])
        cfg = PatchSampleConfig(
            patch_size=self.patch_size,
            distance_field_enabled=self.distance_field_enabled,
            distance_clip_voxels=self.distance_clip_voxels,
            distance_target_mode=self.distance_target_mode,
            query_distance_enabled=self.query_distance_enabled,
            query_distance_points=self.query_distance_points,
            query_distance_band_voxels=self.query_distance_band_voxels,
            query_distance_sampling=self.query_distance_sampling,
            query_distance_inside_fraction=self.query_distance_inside_fraction,
            query_distance_near_outside_fraction=self.query_distance_near_outside_fraction,
            include_mask=self.include_mask,
        )
        add_optional_distance_targets(sample, mask, cfg, rng)
        return sample


def build_dataset(data_cfg: dict[str, Any], split: str) -> Dataset:
    patch_size = tuple(int(v) for v in data_cfg.get("patch_size", data_cfg.get("crop_size", [64, 64, 64])))
    prompt = data_cfg.get("prompt", {})
    distance_cfg = data_cfg.get("distance_field", {})
    query_cfg = data_cfg.get("query_distance", {})
    distance_enabled = bool(distance_cfg.get("enabled", False))
    query_enabled = bool(query_cfg.get("enabled", False))
    distance_target_mode = str(distance_cfg.get("target_mode", "signed"))
    target_backend = distance_target_backend(data_cfg)
    build_targets_in_worker = target_backend == CPU_SCIPY_BACKEND
    include_mask = bool(data_cfg.get(f"{split}_include_mask", data_cfg.get("include_mask", True)))
    distance_clip = float(distance_cfg.get("clip_voxels", query_cfg.get("clip_voxels", 16.0)))
    augmentation = build_patch_augmentation_config(
        data_cfg.get("augmentation", {}),
        split=split,
        patch_size=patch_size,
    )
    if bool(data_cfg.get("synthetic", False)):
        return SyntheticSheetDataset(
            length=int(data_cfg.get("synthetic_length", 8)),
            patch_size=patch_size,
            seed=int(data_cfg.get("seed", 42)) + (0 if split == "train" else 10000),
            num_positive_points=int(prompt.get("num_positive_points", 1)),
            num_negative_points=int(prompt.get("num_negative_points", 0)),
            distance_field_enabled=distance_enabled and build_targets_in_worker,
            distance_clip_voxels=distance_clip,
            distance_target_mode=distance_target_mode,
            query_distance_enabled=query_enabled and build_targets_in_worker,
            query_distance_points=int(query_cfg.get("num_points", 0)),
            query_distance_band_voxels=float(query_cfg.get("band_voxels", 8.0)),
            query_distance_sampling=str(query_cfg.get("sampling", "near_band_uniform")),
            query_distance_inside_fraction=float(query_cfg.get("inside_fraction", 1.0 / 3.0)),
            query_distance_near_outside_fraction=float(query_cfg.get("near_outside_fraction", 1.0 / 3.0)),
            include_mask=include_mask,
        )
    cfg = PatchSampleConfig(
        patch_size=patch_size,
        min_component_voxels=int(data_cfg.get("min_component_voxels", 1000)),
        component_select_mode=str(data_cfg.get("component_select_mode", "random")),
        num_positive_points=int(prompt.get("num_positive_points", 1)),
        num_negative_points=int(prompt.get("num_negative_points", 0)),
        distance_field_enabled=distance_enabled and build_targets_in_worker,
        distance_clip_voxels=distance_clip,
        distance_target_mode=distance_target_mode,
        query_distance_enabled=query_enabled and build_targets_in_worker,
        query_distance_points=int(query_cfg.get("num_points", 0)),
        query_distance_band_voxels=float(query_cfg.get("band_voxels", 8.0)),
        query_distance_sampling=str(query_cfg.get("sampling", "near_band_uniform")),
        query_distance_inside_fraction=float(query_cfg.get("inside_fraction", 1.0 / 3.0)),
        query_distance_near_outside_fraction=float(query_cfg.get("near_outside_fraction", 1.0 / 3.0)),
        randomize_each_access=bool(data_cfg.get("randomize_each_access", False)),
        load_ignore=bool(data_cfg.get("load_ignore", False)),
        load_vertices=bool(data_cfg.get("load_vertices", False)),
        max_vertices=int(data_cfg.get("max_vertices", 4096)),
        node_weight_radius=int(data_cfg.get("node_weight_radius", 0)),
        node_weight_mid=float(data_cfg.get("node_weight_mid", 1.0)),
        augmentation=augmentation,
        include_mask=include_mask,
        max_retries=int(data_cfg.get("patch_max_retries", 24)),
        samples_per_epoch=data_cfg.get(f"{split}_samples_per_epoch"),
        case_cache_size=int(data_cfg.get("case_cache_size", 16)),
        seed=int(data_cfg.get("seed", 42)) + (0 if split == "train" else 10000),
    )
    return VesuviusSheetPatchDataset(
        dataset_root=data_cfg["dataset_root"],
        split=split,
        cfg=cfg,
    )


def build_patch_augmentation_config(
    augmentation_cfg: Any,
    *,
    split: str,
    patch_size: tuple[int, int, int],
) -> PatchAugmentationConfig:
    """Parse one YAML augmentation block and disable it outside training."""

    if split != "train":
        return PatchAugmentationConfig()
    if augmentation_cfg is None:
        augmentation_cfg = {}
    if not isinstance(augmentation_cfg, dict):
        raise ValueError("data.augmentation must be a mapping")

    photometric = augmentation_cfg.get("photometric", {})
    if photometric is None:
        photometric = {}
    if not isinstance(photometric, dict):
        raise ValueError("data.augmentation.photometric must be a mapping")

    cfg = PatchAugmentationConfig(
        flip_p=_probability(augmentation_cfg, "flip_p", default=0.0),
        xy_rotate_p=_probability(augmentation_cfg, "xy_rotate_p", default=0.0),
        xy_rand_rotate_p=_probability(augmentation_cfg, "xy_rand_rotate_p", default=0.0),
        translation_max_pad_voxels=_nonnegative_integer(
            augmentation_cfg,
            "translation_max_pad_voxels",
            default=0,
        ),
        translation_p=_probability(augmentation_cfg, "translation_p", default=1.0),
        photometric_device=_photometric_device(photometric),
        gaussian_noise_p=_probability(photometric, "gaussian_noise_p", default=0.0),
        gaussian_noise_std=_float_range(photometric, "gaussian_noise_std", default=(0.01, 0.1)),
        intensity_p=_probability(photometric, "intensity_p", default=0.0),
        intensity_scale=_float_range(photometric, "intensity_scale", default=(0.9, 1.1)),
        intensity_shift=_float_range(photometric, "intensity_shift", default=(-0.1, 0.1)),
        gamma_p=_probability(photometric, "gamma_p", default=0.0),
        gamma=_float_range(photometric, "gamma", default=(0.8, 1.2)),
        zoom_p=_probability(augmentation_cfg, "zoom_p", default=0.0),
        zoom_range=_float_range(augmentation_cfg, "zoom_range", default=(1.0, 2.0)),
        zoom_thin_voxels=int(augmentation_cfg.get("zoom_thin_voxels", 0)),
    )
    if cfg.xy_rotate_p > 0.0 and patch_size[1] != patch_size[2]:
        raise ValueError(
            "data.augmentation.xy_rotate_p requires equal H/W patch dimensions, "
            f"got patch_size={patch_size}"
        )
    return cfg


def zoom_in_patch(
    image: np.ndarray,
    component_label: np.ndarray,
    scale: float,
    rng: np.random.Generator,
    *,
    anchor_point: np.ndarray | None = None,
    extra_label: np.ndarray | None = None,
    thin_voxels: int = 0,
):
    """Resample a sub-crop of side round(size / scale) (containing the anchor when given) back to the full
    patch size: image trilinear (float32 math, cast back), labels nearest via index maps. Returns the zoomed
    (image, component_label, extra_label, anchor_point) in the new frame."""
    import torch
    import torch.nn.functional as F

    shape = np.asarray(image.shape)
    sub = np.maximum(np.round(shape / scale).astype(int), 8)
    start = np.zeros(3, dtype=int)
    for axis in range(3):
        lo_max = int(shape[axis] - sub[axis])
        if anchor_point is not None:
            a = int(anchor_point[axis])
            lo = max(0, a - sub[axis] + 1); hi = min(lo_max, a)
            start[axis] = int(rng.integers(lo, hi + 1)) if hi >= lo else int(np.clip(a - sub[axis] // 2, 0, lo_max))
        else:
            start[axis] = int(rng.integers(0, lo_max + 1))
    sl = tuple(slice(int(st), int(st + sz)) for st, sz in zip(start, sub))
    img = torch.from_numpy(np.ascontiguousarray(image[sl]).astype(np.float32))[None, None]
    img = F.interpolate(img, size=tuple(int(v) for v in shape), mode="trilinear", align_corners=False)[0, 0].numpy()
    if np.issubdtype(image.dtype, np.integer):
        info = np.iinfo(image.dtype)
        img = np.clip(np.rint(img), info.min, info.max)
    image_out = img.astype(image.dtype)
    idx = [np.clip(np.floor((np.arange(int(shape[a])) + 0.5) * sub[a] / shape[a]).astype(int) + start[a], 0, int(shape[a]) - 1)
           for a in range(3)]
    grid = np.ix_(idx[0], idx[1], idx[2])
    comp_out = np.ascontiguousarray(component_label[grid])
    extra_out = np.ascontiguousarray(extra_label[grid]) if extra_label is not None else None
    if thin_voxels > 0:
        comp_out = thin_sheets(comp_out, thin_voxels, scale)
    anchor_out = None
    if anchor_point is not None:
        anchor_out = np.clip(np.floor((np.asarray(anchor_point) - start + 0.5) * shape / sub).astype(int), 0, shape - 1)
    return image_out, comp_out, extra_out, anchor_out


def thin_sheets(component_label: np.ndarray, target_voxels: int, scale: float) -> np.ndarray:
    """Keep the medial ~target_voxels of every sheet thicker than that: a foreground voxel survives when its distance
    to the background is within target_voxels / 2 of the local maximum (the sheet's mid-plane), found with a max
    filter whose window covers the thickest zoomed sheet (~3 * scale voxels). Sheets already thinner are untouched;
    component ids are preserved."""
    from scipy import ndimage as ndi

    fg = component_label > 0
    if not fg.any():
        return component_label
    edt = ndi.distance_transform_edt(fg)
    win = int(2 * np.ceil(1.5 * scale) + 1)
    local_max = ndi.maximum_filter(edt, size=win)
    keep = fg & (edt >= local_max - 0.5 * target_voxels)
    return np.where(keep, component_label, 0).astype(component_label.dtype)


def apply_patch_augmentations(
    image: np.ndarray,
    component_label: np.ndarray,
    cfg: PatchAugmentationConfig,
    rng: np.random.Generator,
    *,
    anchor_point: np.ndarray | None = None,
    extra_label: np.ndarray | None = None,
    points: np.ndarray | None = None,
    normals: np.ndarray | None = None,
) -> tuple[np.ndarray, ...]:
    """Apply paired spatial transforms and image-only photometric transforms.

    ``points`` ([N,3] float, zyx voxel coordinates in the crop; index i = the centre of voxel i) and ``normals``
    ([N,3] unit vectors, zyx) are transformed with the same flips / rot90 / translation as the labels and returned
    appended to the tuple (0075 vertex emphasis). Zoom and free-angle rotation are not implemented for points.

    Pad-then-crop translation shifts the owned crop in place, which is exactly
    equivalent to virtual zero padding followed by a same-size crop while
    avoiding a larger PS320 temporary. A supplied anchor remains in view.
    """

    if image.ndim != 3 or component_label.ndim != 3:
        raise ValueError(
            "Patch augmentation expects image and component_label as [D,H,W], "
            f"got {image.shape} and {component_label.shape}"
        )
    if image.shape != component_label.shape:
        raise ValueError(
            "Patch augmentation requires matching image/component-label shapes, "
            f"got {image.shape} and {component_label.shape}"
        )

    image_dtype = image.dtype
    label_dtype = component_label.dtype
    anchor = _validate_anchor_point(anchor_point, image.shape)
    if cfg.translation_max_pad_voxels > 0 and rng.random() < cfg.translation_p:
        shifts = sample_pad_then_crop_shifts(
            image.shape,
            max_pad=cfg.translation_max_pad_voxels,
            rng=rng,
            anchor_point=anchor,
        )
        image = translate_zero_filled(image, shifts)
        component_label = translate_zero_filled(component_label, shifts)
        if extra_label is not None:
            extra_label = translate_zero_filled(extra_label, shifts)
        if points is not None:
            points = points + np.asarray(shifts, np.float32)
    if cfg.zoom_p > 0.0 and rng.random() < cfg.zoom_p:
        s = float(rng.uniform(cfg.zoom_range[0], cfg.zoom_range[1]))
        if s > 1.0 + 1e-6 and points is not None:
            raise NotImplementedError("zoom augmentation is not implemented for vertex points (set zoom_p 0 with load_vertices)")
        if s > 1.0 + 1e-6:
            image, component_label, extra_label, anchor = zoom_in_patch(
                image, component_label, s, rng, anchor_point=anchor, extra_label=extra_label,
                thin_voxels=cfg.zoom_thin_voxels)
    if cfg.flip_p > 0.0:
        for axis in range(3):
            if rng.random() < cfg.flip_p:
                image = np.flip(image, axis=axis)
                component_label = np.flip(component_label, axis=axis)
                if extra_label is not None:
                    extra_label = np.flip(extra_label, axis=axis)
                if points is not None:
                    points = points.copy(); points[:, axis] = (image.shape[axis] - 1) - points[:, axis]
                    if normals is not None:
                        normals = normals.copy(); normals[:, axis] = -normals[:, axis]
    if cfg.xy_rotate_p > 0.0 and rng.random() < cfg.xy_rotate_p:
        k = int(rng.integers(1, 4))
        shape_before = image.shape
        image = np.rot90(image, k=k, axes=(1, 2))
        component_label = np.rot90(component_label, k=k, axes=(1, 2))
        if extra_label is not None:
            extra_label = np.rot90(extra_label, k=k, axes=(1, 2))
        if points is not None:
            points = rot90_points_zyx(points, k, shape_before)
            if normals is not None:
                normals = rot90_points_zyx(normals, k, shape_before, vector=True)
    if cfg.xy_rand_rotate_p > 0.0 and rng.random() < cfg.xy_rand_rotate_p:
        if points is not None:
            raise NotImplementedError("free-angle rotation is not implemented for vertex points (set xy_rand_rotate_p 0 with load_vertices)")
        angle = float(rng.uniform(-180.0, 180.0))
        # scipy.ndimage does not implement geometric interpolation for float16.
        # Keep its FP32 working array local to this optional CPU augmentation
        # and cast back before the DataLoader returns the image.
        image = ndimage.rotate(
            image.astype(np.float32, copy=False),
            angle,
            axes=(1, 2),
            reshape=False,
            order=1,
            mode="constant",
            cval=0.0,
        )
        component_label = ndimage.rotate(
            component_label,
            angle,
            axes=(1, 2),
            reshape=False,
            order=0,
            mode="constant",
            cval=0,
        )
        if extra_label is not None:
            extra_label = ndimage.rotate(
                extra_label.astype(np.uint8, copy=False),
                angle,
                axes=(1, 2),
                reshape=False,
                order=0,
                mode="constant",
                cval=0,
            )

    if cfg.photometric_device == "cpu" and (
        cfg.gaussian_noise_p > 0.0
        or cfg.intensity_p > 0.0
        or cfg.gamma_p > 0.0
    ):
        image_float = image.astype(np.float32, copy=False)
        if cfg.gaussian_noise_p > 0.0 and rng.random() < cfg.gaussian_noise_p:
            std = float(rng.uniform(*cfg.gaussian_noise_std))
            image_float = image_float + rng.normal(0.0, std, size=image.shape).astype(np.float32)
        if cfg.intensity_p > 0.0 and rng.random() < cfg.intensity_p:
            scale = float(rng.uniform(*cfg.intensity_scale))
            shift = float(rng.uniform(*cfg.intensity_shift))
            image_float = image_float * scale + shift
        if cfg.gamma_p > 0.0 and rng.random() < cfg.gamma_p:
            gamma = float(rng.uniform(*cfg.gamma))
            image_min = float(image_float.min())
            image_range = float(image_float.max() - image_min) + 1.0e-8
            image_float = ((image_float - image_min) / image_range) ** gamma * image_range + image_min
        image = image_float.astype(image_dtype, copy=False)

    out: tuple[np.ndarray, ...]
    if extra_label is not None:
        out = (
            np.ascontiguousarray(image, dtype=image_dtype),
            np.ascontiguousarray(component_label, dtype=label_dtype),
            np.ascontiguousarray(extra_label).astype(bool, copy=False),
        )
    else:
        out = (
            np.ascontiguousarray(image, dtype=image_dtype),
            np.ascontiguousarray(component_label, dtype=label_dtype),
        )
    if points is not None:
        out = out + (np.ascontiguousarray(points, dtype=np.float32), None if normals is None else np.ascontiguousarray(normals, dtype=np.float32))
    return out


def rot90_points_zyx(points: np.ndarray, k: int, shape: tuple[int, ...], *, vector: bool = False) -> np.ndarray:
    """Map zyx coordinates (or direction vectors when vector=True) through np.rot90(volume, k, axes=(1, 2)).

    np.rot90 with axes=(1, 2) is a rotation from axis 1 (y) towards axis 2 (x): for k=1 the output voxel
    [z, y', x'] holds input [z, x', (W-1) - y'] ... verified against volumes in runs/meshlabel_20260910/test_vertex_aug.py.
    """
    k = int(k) % 4
    p = np.array(points, dtype=np.float32, copy=True)
    h, w = int(shape[1]), int(shape[2])
    for _ in range(k):
        y = p[:, 1].copy(); x = p[:, 2].copy()
        # one np.rot90(axes=(1,2)) step: out[z, y', x'] = in[z, x', W-1-y']  -> the point (y, x) moves to (y', x') = (W-1-x, y)
        if vector:
            p[:, 1] = -x; p[:, 2] = y
        else:
            p[:, 1] = (w - 1) - x; p[:, 2] = y
        h, w = w, h
    return p


def apply_gpu_photometric_augmentations(
    image: torch.Tensor,
    cfg: PatchAugmentationConfig,
    *,
    generator: torch.Generator,
) -> torch.Tensor:
    """Apply the configured image-only photometrics after CUDA transfer.

    The input is the float32 P2SD image tensor. Geometry, labels, masks, and
    prompts have already been prepared on the CPU; this function changes only
    image intensities. A dedicated generator keeps these random draws isolated
    from model RNG state.
    """

    if cfg.photometric_device != "cuda":
        return image
    if image.device.type != "cuda":
        raise ValueError("GPU photometric augmentation requires a CUDA image tensor")
    if image.ndim != 5:
        raise ValueError(f"GPU photometric augmentation expects [B,C,D,H,W], got {image.shape}")
    if not image.dtype.is_floating_point:
        raise ValueError(f"GPU photometric augmentation requires floating image dtype, got {image.dtype}")

    for sample_index in range(image.shape[0]):
        sample = image[sample_index]
        if cfg.gaussian_noise_p > 0.0 and _torch_random_bool(
            cfg.gaussian_noise_p,
            device=image.device,
            generator=generator,
        ):
            std = _torch_uniform(
                cfg.gaussian_noise_std,
                device=image.device,
                generator=generator,
            )
            noise = torch.randn(
                sample.shape,
                device=image.device,
                dtype=sample.dtype,
                generator=generator,
            )
            sample.add_(noise.mul_(std))
        if cfg.intensity_p > 0.0 and _torch_random_bool(
            cfg.intensity_p,
            device=image.device,
            generator=generator,
        ):
            scale = _torch_uniform(
                cfg.intensity_scale,
                device=image.device,
                generator=generator,
            )
            shift = _torch_uniform(
                cfg.intensity_shift,
                device=image.device,
                generator=generator,
            )
            sample.mul_(scale).add_(shift)
        if cfg.gamma_p > 0.0 and _torch_random_bool(
            cfg.gamma_p,
            device=image.device,
            generator=generator,
        ):
            gamma = _torch_uniform(cfg.gamma, device=image.device, generator=generator)
            image_min = sample.amin()
            image_range = (sample.amax() - image_min).add_(1.0e-8)
            sample.sub_(image_min).div_(image_range).pow_(gamma).mul_(image_range).add_(image_min)
    return image


def _torch_random_bool(
    probability: float,
    *,
    device: torch.device,
    generator: torch.Generator,
) -> bool:
    return bool(torch.rand((), device=device, generator=generator) < probability)


def _torch_uniform(
    bounds: tuple[float, float],
    *,
    device: torch.device,
    generator: torch.Generator,
) -> torch.Tensor:
    lower, upper = bounds
    return torch.empty((), device=device).uniform_(lower, upper, generator=generator)


def _probability(mapping: dict[str, Any], key: str, *, default: float) -> float:
    value = float(mapping.get(key, default))
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"data.augmentation.{key} must be in [0, 1], got {value}")
    return value


def _nonnegative_integer(mapping: dict[str, Any], key: str, *, default: int) -> int:
    value = mapping.get(key, default)
    if isinstance(value, bool):
        raise ValueError(f"data.augmentation.{key} must be a non-negative integer, got {value}")
    try:
        integer = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"data.augmentation.{key} must be a non-negative integer, got {value}") from exc
    if integer != value or integer < 0:
        raise ValueError(f"data.augmentation.{key} must be a non-negative integer, got {value}")
    return integer


def _photometric_device(photometric: dict[str, Any]) -> str:
    value = str(photometric.get("device", "cpu")).lower()
    if value not in {"cpu", "cuda"}:
        raise ValueError(
            "data.augmentation.photometric.device must be 'cpu' or 'cuda', "
            f"got {value!r}")
    return value


def _validate_anchor_point(
    anchor_point: np.ndarray | None,
    shape: tuple[int, int, int],
) -> np.ndarray | None:
    if anchor_point is None:
        return None
    anchor = np.asarray(anchor_point, dtype=np.int64)
    if anchor.shape != (3,) or np.any(anchor < 0) or np.any(anchor >= np.asarray(shape)):
        raise ValueError(
            f"anchor_point must be an in-bounds [D,H,W] coordinate for shape {shape}, got {anchor_point}")
    return anchor


def sample_pad_then_crop_shifts(
    shape: tuple[int, int, int],
    *,
    max_pad: int,
    rng: np.random.Generator,
    anchor_point: np.ndarray | None,
) -> tuple[int, int, int]:
    """Sample shifts induced by independently padded sides and a same-size crop."""

    if max_pad < 0:
        raise ValueError(f"max_pad must be non-negative, got {max_pad}")
    if max_pad == 0:
        return (0, 0, 0)

    shifts = []
    for axis, size in enumerate(shape):
        pad_before = int(rng.integers(0, max_pad + 1))
        pad_after = int(rng.integers(0, max_pad + 1))
        crop_low = 0
        crop_high = pad_before + pad_after
        if anchor_point is not None:
            anchor = int(anchor_point[axis])
            crop_low = max(crop_low, pad_before + anchor - size + 1)
            crop_high = min(crop_high, pad_before + anchor)
        if crop_low > crop_high:
            raise RuntimeError(
                "No virtual crop can retain the requested anchor; "
                f"axis={axis} size={size} pads=({pad_before}, {pad_after})")
        crop_start = int(rng.integers(crop_low, crop_high + 1))
        shifts.append(pad_before - crop_start)
    return tuple(shifts)


def translate_zero_filled(array: np.ndarray, shifts: tuple[int, int, int]) -> np.ndarray:
    """Shift a [D,H,W] crop with zero fill using one combined slice copy."""

    if len(shifts) != array.ndim:
        raise ValueError(f"Expected one shift per array axis, got {shifts} for shape {array.shape}")
    if not any(shifts):
        return array

    source = []
    destination = []
    for size, shift in zip(array.shape, shifts):
        shift = int(shift)
        if abs(shift) >= size:
            return np.zeros_like(array)
        source.append(slice(max(0, -shift), min(size, size - shift)))
        destination.append(slice(max(0, shift), min(size, size + shift)))

    out = np.zeros_like(array)
    out[tuple(destination)] = array[tuple(source)]
    return out


def _float_range(
    mapping: dict[str, Any],
    key: str,
    *,
    default: tuple[float, float],
) -> tuple[float, float]:
    value = mapping.get(key, default)
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"data.augmentation.{key} must contain exactly two numbers")
    lower, upper = (float(value[0]), float(value[1]))
    if lower > upper:
        raise ValueError(f"data.augmentation.{key} must be ordered, got {value}")
    return lower, upper


def collate_sheet_batch(samples: list[dict[str, Any]]) -> dict[str, Any]:
    tensor_keys = [
        "image",
        "mask",
        "component_label",
        "prompt_points",
        "prompt_labels",
        "distance_field",
        "query_points",
        "query_distances",
        # "ignore" is deliberately NOT collated for P2SD (user 2026-09-11: the Kaggle ignore region lies outside the
        # sheets and is intended to be background); the loader still builds sample["ignore"] for the binseg trainer.
        # 0075 vertex emphasis (padded per sample, so they stack): mesh nodes and the node-weighted dense weight
        "vertices_zyx",
        "vertex_normals_zyx",
        "vertex_component",
        "vertex_valid",
        "dense_weight",
    ]
    batch: dict[str, Any] = {}
    for key in tensor_keys:
        if all(key in sample for sample in samples):
            batch[key] = torch.stack([sample[key] for sample in samples], dim=0)
    batch["case_id"] = [sample["case_id"] for sample in samples]
    batch["component_id"] = torch.tensor([sample["component_id"] for sample in samples], dtype=torch.long)
    batch["crop_start"] = [sample["crop_start"] for sample in samples]
    batch["prompt_zyx"] = [sample["prompt_zyx"] for sample in samples]
    for key in ("probe_id", "prompt_set_id"):
        if all(key in sample for sample in samples):
            batch[key] = [str(sample[key]) for sample in samples]
    return batch


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_component_stats_cache(
    dataset_root: str | Path,
    split: str,
) -> dict[str, tuple[str | None, list[ComponentStat]]]:
    """Load optional per-case component stats for a split.

    The cache avoids repeated full-volume scans in DataLoader workers. Missing
    or invalid cache files are ignored so existing datasets keep working.
    """

    path = Path(dataset_root) / f"component_stats_{split}.jsonl"
    if not path.exists():
        return {}
    out: dict[str, tuple[str | None, list[ComponentStat]]] = {}
    for row in load_jsonl(path):
        case_id = str(row.get("case_id", ""))
        if not case_id:
            continue
        stats = [
            ComponentStat(
                component_id=int(item["component_id"]),
                count=int(item["count"]),
                bbox=tuple(
                    (int(pair[0]), int(pair[1]))
                    for pair in item.get("bbox", [])
                ),
            )
            for item in row.get("components", [])
        ]
        out[case_id] = (row.get("components_path"), stats)
    return out


def component_stats_cache_row(row: dict[str, Any], stats: list[ComponentStat]) -> dict[str, Any]:
    return {
        "case_id": row["case_id"],
        "components_path": row.get("components_path"),
        "shape": row.get("shape"),
        "components": [
            {
                "component_id": int(item.component_id),
                "count": int(item.count),
                "bbox": [[int(lo), int(hi)] for lo, hi in item.bbox],
            }
            for item in stats
        ],
    }


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")


def choose_component_id(
    comps: np.ndarray,
    *,
    min_voxels: int,
    mode: str,
    rng: np.random.Generator,
) -> int:
    ids, counts = np.unique(comps, return_counts=True)
    valid = [(int(i), int(c)) for i, c in zip(ids, counts) if int(i) > 0 and int(c) >= min_voxels]
    if not valid:
        valid = [(int(i), int(c)) for i, c in zip(ids, counts) if int(i) > 0]
    if not valid:
        return 0
    valid.sort(key=lambda x: x[1], reverse=True)
    if mode == "largest":
        return valid[0][0]
    if mode == "random":
        return valid[int(rng.integers(0, len(valid)))][0]
    raise ValueError(f"Unsupported component_select_mode: {mode}")


def build_component_stats(comps: np.ndarray) -> list[ComponentStat]:
    max_label = int(np.max(comps))
    if max_label <= 0:
        return []
    counts = np.bincount(np.asarray(comps).reshape(-1), minlength=max_label + 1)
    objects = ndimage.find_objects(comps)
    stats = []
    for component_id in range(1, max_label + 1):
        count = int(counts[component_id]) if component_id < len(counts) else 0
        if count <= 0:
            continue
        obj = objects[component_id - 1] if component_id - 1 < len(objects) else None
        if obj is None:
            continue
        bbox = tuple((int(sl.start), int(sl.stop)) for sl in obj)
        stats.append(ComponentStat(component_id=component_id, count=count, bbox=bbox))
    stats.sort(key=lambda item: item.count, reverse=True)
    return stats


def choose_component_stat(
    stats: list[ComponentStat],
    *,
    min_voxels: int,
    mode: str,
    rng: np.random.Generator,
) -> ComponentStat | None:
    valid = [item for item in stats if item.count >= min_voxels]
    if not valid:
        valid = list(stats)
    if not valid:
        return None
    if mode == "largest":
        return valid[0]
    if mode == "random":
        return valid[int(rng.integers(0, len(valid)))]
    raise ValueError(f"Unsupported component_select_mode: {mode}")


def sample_component_center(
    comps: np.ndarray,
    stat: ComponentStat,
    rng: np.random.Generator,
    *,
    max_rejection_tries: int = 1024,
) -> np.ndarray:
    lows = np.asarray([lo for lo, _ in stat.bbox], dtype=np.int64)
    highs = np.asarray([hi for _, hi in stat.bbox], dtype=np.int64)
    for _ in range(max_rejection_tries):
        coord = np.asarray(
            [int(rng.integers(int(lo), int(hi))) for lo, hi in zip(lows, highs)],
            dtype=np.int64,
        )
        if int(comps[tuple(coord)]) == stat.component_id:
            return coord

    slices = tuple(slice(int(lo), int(hi)) for lo, hi in stat.bbox)
    coords = np.argwhere(np.asarray(comps[slices]) == stat.component_id)
    if len(coords) == 0:
        return (lows + highs) // 2
    return lows + coords[int(rng.integers(0, len(coords)))]


def crop_start_around(
    center: np.ndarray,
    shape: tuple[int, int, int],
    patch_size: tuple[int, int, int],
    rng: np.random.Generator,
) -> np.ndarray:
    start = []
    for c, size, patch in zip(center, shape, patch_size):
        if size <= patch:
            start.append(0)
            continue
        low = max(0, int(c) - patch + 1)
        high = min(int(c), size - patch)
        if high < low:
            value = min(max(0, int(c) - patch // 2), size - patch)
        else:
            value = int(rng.integers(low, high + 1))
        start.append(value)
    return np.asarray(start, dtype=np.int64)


def crop_array(
    arr: np.ndarray,
    start: np.ndarray,
    patch_size: tuple[int, int, int],
    *,
    output_offset: np.ndarray | None = None,
) -> np.ndarray:
    slices = tuple(slice(int(s), int(s) + int(p)) for s, p in zip(start, patch_size))
    crop = np.asarray(arr[slices])
    if crop.shape == patch_size:
        return crop
    out = np.zeros(patch_size, dtype=arr.dtype)
    if output_offset is None:
        output_offset = np.zeros(3, dtype=np.int64)
    offset = np.asarray(output_offset, dtype=np.int64)
    if offset.shape != (3,) or np.any(offset < 0):
        raise ValueError(f"output_offset must be non-negative [D,H,W], got {output_offset}")
    if np.any(offset + np.asarray(crop.shape) > np.asarray(patch_size)):
        raise ValueError(
            "output_offset places the crop outside the patch: "
            f"offset={tuple(int(value) for value in offset)} "
            f"crop_shape={crop.shape} patch_size={patch_size}"
        )
    dst = tuple(slice(int(offset[axis]), int(offset[axis]) + size) for axis, size in enumerate(crop.shape))
    out[dst] = crop
    return out


def sample_crop_output_offset(
    shape: tuple[int, int, int],
    start: np.ndarray,
    patch_size: tuple[int, int, int],
    rng: np.random.Generator,
) -> np.ndarray:
    """Randomly place an undersized source crop inside the fixed patch."""

    offset = []
    for size, source_start, patch in zip(shape, start, patch_size):
        crop_size = min(max(0, int(size) - int(source_start)), int(patch))
        max_offset = int(patch) - crop_size
        offset.append(int(rng.integers(0, max_offset + 1)) if max_offset > 0 else 0)
    return np.asarray(offset, dtype=np.int64)


def sample_points(
    mask: np.ndarray,
    count: int,
    rng: np.random.Generator,
    preferred_points: list[np.ndarray] | None = None,
) -> np.ndarray:
    count = int(count)
    if count <= 0:
        return np.zeros((0, 3), dtype=np.float32)
    points = []
    for point in preferred_points or []:
        coord = np.asarray(point, dtype=np.int64)
        if coord.shape != (3,):
            continue
        if np.any(coord < 0) or np.any(coord >= np.asarray(mask.shape)):
            continue
        if mask[tuple(coord)]:
            points.append(coord.astype(np.float32))
            if len(points) == count:
                return np.stack(points, axis=0).astype(np.float32, copy=False)
    if not bool(np.any(mask)):
        return np.zeros((count, 3), dtype=np.float32)
    remaining = count - len(points)
    sampled = sample_random_foreground_points(mask, remaining, rng)
    if not points:
        return sampled
    return np.concatenate([np.stack(points, axis=0), sampled], axis=0).astype(np.float32, copy=False)


def sample_random_foreground_points(
    mask: np.ndarray,
    count: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample positive voxels without materializing a full `(N, 3)` coordinate array."""
    if count <= 0:
        return np.zeros((0, 3), dtype=np.float32)
    flat_mask = np.asarray(mask, dtype=bool).reshape(-1)
    if not bool(flat_mask.any()):
        return np.zeros((count, 3), dtype=np.float32)

    # Sheet crops are normally dense enough that rejection sampling avoids the
    # costly `np.argwhere(mask)` allocation over a PS320 volume. Accepted flat
    # indices are unique, matching the no-replacement path used previously.
    accepted: list[int] = []
    accepted_set: set[int] = set()
    max_attempts = max(4096, 2048 * count)
    for _ in range(max_attempts):
        if len(accepted) == count:
            break
        flat_index = int(rng.integers(flat_mask.size))
        if flat_index in accepted_set or not flat_mask[flat_index]:
            continue
        accepted.append(flat_index)
        accepted_set.add(flat_index)

    if len(accepted) < count:
        # Very sparse masks make rejection inefficient. `flatnonzero` has a
        # smaller temporary representation than an `(N, 3)` coordinate array,
        # and preserves the old replacement behavior when needed.
        foreground = np.flatnonzero(flat_mask)
        needed = count - len(accepted)
        if len(foreground) >= count:
            remaining_foreground = foreground[~np.isin(foreground, accepted)]
            choices = rng.choice(len(remaining_foreground), size=needed, replace=False)
            accepted.extend(int(remaining_foreground[index]) for index in np.asarray(choices).reshape(-1))
        else:
            choices = rng.choice(len(foreground), size=needed, replace=True)
            accepted.extend(int(foreground[index]) for index in np.asarray(choices).reshape(-1))

    coords = np.column_stack(np.unravel_index(np.asarray(accepted, dtype=np.int64), mask.shape))
    return coords.astype(np.float32, copy=False)


def sample_negative_points(
    components: np.ndarray,
    positive_mask: np.ndarray,
    count: int,
    rng: np.random.Generator,
) -> np.ndarray:
    count = int(count)
    if count <= 0:
        return np.zeros((0, 3), dtype=np.float32)
    neg_mask = ~positive_mask
    coords = np.argwhere(neg_mask)
    if len(coords) == 0:
        coords = np.argwhere(np.ones_like(components, dtype=bool))
    indices = rng.choice(len(coords), size=count, replace=len(coords) < count)
    return coords[indices].astype(np.float32)


def combine_prompt_points(
    pos: np.ndarray,
    neg: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    points = np.concatenate([pos, neg], axis=0).astype(np.float32, copy=False)
    labels = np.concatenate([
        np.ones((len(pos),), dtype=np.int64),
        np.zeros((len(neg),), dtype=np.int64),
    ])
    return points, labels


def add_optional_distance_targets(
    sample: dict[str, Any],
    mask: np.ndarray,
    cfg: PatchSampleConfig,
    rng: np.random.Generator,
) -> None:
    if not cfg.distance_field_enabled and not cfg.query_distance_enabled:
        return
    signed_voxels, normalized = normalized_signed_distance(
        mask,
        clip_voxels=cfg.distance_clip_voxels,
        target_mode=cfg.distance_target_mode,
    )
    if cfg.distance_field_enabled:
        sample["distance_field"] = torch.from_numpy(normalized[None])
    if cfg.query_distance_enabled and cfg.query_distance_points > 0:
        points, distances = sample_query_distance_points(
            signed_voxels=signed_voxels,
            normalized_distance=normalized,
            count=cfg.query_distance_points,
            band_voxels=cfg.query_distance_band_voxels,
            sampling=cfg.query_distance_sampling,
            inside_fraction=cfg.query_distance_inside_fraction,
            near_outside_fraction=cfg.query_distance_near_outside_fraction,
            rng=rng,
        )
        sample["query_points"] = torch.from_numpy(points)
        sample["query_distances"] = torch.from_numpy(distances)


def normalized_signed_distance(
    mask: np.ndarray,
    *,
    clip_voxels: float,
    target_mode: str = "signed",
) -> tuple[np.ndarray, np.ndarray]:
    mask_bool = mask.astype(bool, copy=False)
    clip = max(float(clip_voxels), 1.0)
    inside = ndimage.distance_transform_edt(mask_bool)
    outside = ndimage.distance_transform_edt(~mask_bool)
    signed_voxels = outside - inside
    if target_mode == "signed":
        normalized = np.clip(signed_voxels, -clip, clip) / clip
    elif target_mode == "unsigned":
        normalized = np.clip(np.abs(signed_voxels), 0.0, clip) / clip
    else:
        raise ValueError(f"distance target_mode must be signed|unsigned, got {target_mode!r}")
    return signed_voxels.astype(np.float32, copy=False), normalized.astype(np.float32, copy=False)


def sample_query_distance_points(
    *,
    signed_voxels: np.ndarray,
    normalized_distance: np.ndarray,
    count: int,
    band_voxels: float,
    sampling: str = "near_band_uniform",
    inside_fraction: float = 1.0 / 3.0,
    near_outside_fraction: float = 1.0 / 3.0,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    count = int(count)
    if count <= 0:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0,), dtype=np.float32)
    if sampling == "balanced_regions":
        return _sample_balanced_query_distance_points_cpu(
            signed_voxels=signed_voxels,
            normalized_distance=normalized_distance,
            count=count,
            band_voxels=band_voxels,
            inside_fraction=inside_fraction,
            near_outside_fraction=near_outside_fraction,
            rng=rng,
        )
    if sampling != "near_band_uniform":
        raise ValueError(f"query sampling must be near_band_uniform|balanced_regions, got {sampling!r}")
    band_count = count // 2
    random_count = count - band_count
    coords = []
    if band_count > 0:
        band = np.argwhere(np.abs(signed_voxels) <= float(band_voxels))
        if len(band) > 0:
            indices = rng.choice(len(band), size=band_count, replace=len(band) < band_count)
            coords.append(band[indices])
        else:
            random_count += band_count
    if random_count > 0:
        flat = rng.integers(0, int(np.prod(signed_voxels.shape)), size=random_count)
        coords.append(np.column_stack(np.unravel_index(flat, signed_voxels.shape)))
    points = np.concatenate(coords, axis=0).astype(np.float32, copy=False)
    rng.shuffle(points, axis=0)
    zyx = tuple(points.astype(np.int64).T)
    distances = normalized_distance[zyx].astype(np.float32, copy=False)
    return points, distances


def _sample_balanced_query_distance_points_cpu(
    *,
    signed_voxels: np.ndarray,
    normalized_distance: np.ndarray,
    count: int,
    band_voxels: float,
    inside_fraction: float,
    near_outside_fraction: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Balance thin-sheet interior, near exterior, and far exterior queries."""
    if not 0.0 <= inside_fraction <= 1.0 or not 0.0 <= near_outside_fraction <= 1.0:
        raise ValueError("query region fractions must lie in [0, 1]")
    if inside_fraction + near_outside_fraction > 1.0:
        raise ValueError("inside_fraction + near_outside_fraction must not exceed 1")
    counts = (
        int(round(count * inside_fraction)),
        int(round(count * near_outside_fraction)),
    )
    counts = (*counts, count - sum(counts))
    regions = (
        signed_voxels < 0,
        (signed_voxels > 0) & (signed_voxels <= float(band_voxels)),
        signed_voxels > float(band_voxels),
    )
    points = []
    for region, region_count in zip(regions, counts, strict=True):
        candidates = np.argwhere(region)
        if region_count <= 0:
            continue
        if len(candidates) == 0:
            candidates = np.argwhere(signed_voxels > 0)
        indices = rng.choice(len(candidates), size=region_count, replace=len(candidates) < region_count)
        points.append(candidates[indices])
    result = np.concatenate(points, axis=0).astype(np.float32, copy=False)
    rng.shuffle(result)
    zyx = tuple(result.astype(np.int64).T)
    return result, normalized_distance[zyx].astype(np.float32, copy=False)
