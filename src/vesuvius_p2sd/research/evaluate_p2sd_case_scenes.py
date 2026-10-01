"""Evaluate all fixed prompted sheets as four full-scene case predictions.

For each case and prompt-set id, this evaluator unions the predicted masks for
every threshold-eligible sheet.  It shares one P2SD image-encoder/context
pass across all prompted sheets in that case, then sends the four scene unions
through the exact public Kaggle metric implementation in bounded batches.  The
``topology_proxies`` mode replays the same predictions but skips the expensive
public metric calculation to measure raw connected-component proxies only.

The ``eligible_only`` target contract marks sheets below the manifest's size
threshold with public ignore label ``2``.  It is intentionally reported as an
eligible-case score, not as a literal full-submission leaderboard result.
"""

from __future__ import annotations

import argparse
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np
import torch

from vesuvius_p2sd.data.dataset import load_jsonl
from vesuvius_p2sd.data.scene_probes import SceneCaseProbe, SceneProbeManifest, load_scene_probe_manifest
from vesuvius_p2sd.eval.connected_components import component_count
from vesuvius_p2sd.eval.kaggle_surface import compute_kaggle_case_metrics_batch
from vesuvius_p2sd.eval.probes import pairwise_prediction_dice_stats
from vesuvius_p2sd.eval.topology import GpuSheetContactTracker, SheetContactTracker
from vesuvius_p2sd.models.p2sd import build_p2sd
from vesuvius_p2sd.research.run_status import write_json
from vesuvius_p2sd.train.common import (
    amp_dtype,
    autocast_context,
    get_device,
    load_ae_from_config,
    load_model_state,
    maybe_channels_last_3d,
)
from vesuvius_p2sd.train.train_p2sd import build_static_latent_codec, decoded_foreground_probability
from vesuvius_p2sd.utils.config import load_config


ELIGIBLE_ONLY_CONTRACT = "eligible_only"
ALL_SHEETS_CONTRACT = "all_sheets"
TARGET_CONTRACTS = (ELIGIBLE_ONLY_CONTRACT, ALL_SHEETS_CONTRACT)
FULL_EVALUATION_MODE = "full"
TOPOLOGY_PROXIES_EVALUATION_MODE = "topology_proxies"

# Raw Variation of Information is a distance: 0 is perfect, larger is worse. The
# `kaggle_*` summary loop reports a "worst" case per metric, and taking min() for
# these would report the BEST case under a name that reads as the worst.
# `kaggle_voi_score` is NOT in this set -- it is the one_over_one_plus transform
# of VOI, so higher is better like every other reported score.
LOWER_IS_BETTER_KAGGLE_METRICS = frozenset({
    "kaggle_voi_total",
    "kaggle_voi_split",
    "kaggle_voi_merge",
})
EVALUATION_MODES = (FULL_EVALUATION_MODE, TOPOLOGY_PROXIES_EVALUATION_MODE)

# Per (case, prompt-set): how many pairs of predicted sheets touch or overlap.
UNION_PAIR_COUNT_KEY = "sheet_contact_pair_count"


def evaluate_case_scenes(
    *,
    checkpoint_path: str | Path,
    source_run_dir: str | Path,
    scene_manifest_path: str | Path,
    output_dir: str | Path,
    target_contract: str,
    min_component_voxels: int = 0,
    topology_batch_size: int = 2,
    topology_tile_batch_size: int = 4,
    prompt_batch_size: int = 8,
    cpu_workers: int = 4,
    topology_backend: str = "reference_batch",
    evaluation_mode: str = FULL_EVALUATION_MODE,
    device_name: str | None = None,
    shard_index: int = 0,
    num_shards: int = 1,
    refine_head_path: str | Path | None = None,
) -> dict[str, Any]:
    """Evaluate one checkpoint with deterministic all-sheet case scenes."""

    if target_contract not in TARGET_CONTRACTS:
        raise ValueError(f"target_contract must be one of {TARGET_CONTRACTS}, got {target_contract!r}")
    if evaluation_mode not in EVALUATION_MODES:
        raise ValueError(f"evaluation_mode must be one of {EVALUATION_MODES}, got {evaluation_mode!r}")
    if topology_batch_size <= 0 or topology_tile_batch_size <= 0 or prompt_batch_size <= 0:
        raise ValueError(
            "topology_batch_size, topology_tile_batch_size, and prompt_batch_size must be positive")
    if cpu_workers <= 0:
        raise ValueError("cpu_workers must be positive")
    if num_shards <= 0 or not (0 <= shard_index < num_shards):
        raise ValueError(f"invalid shard {shard_index}/{num_shards}")

    checkpoint_path = Path(checkpoint_path)
    source_run_dir = Path(source_run_dir)
    scene_manifest_path = Path(scene_manifest_path)
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Evaluation output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    source_cfg = load_config(source_run_dir / "resolved_config.yaml")
    if device_name is not None:
        source_cfg["device"] = device_name
    patch_size = _patch_size_from_config(source_cfg)
    manifest = load_scene_probe_manifest(scene_manifest_path, patch_size=patch_size)
    source_rows = _load_source_rows(source_cfg, manifest)
    device = get_device(source_cfg)
    dtype = amp_dtype(source_cfg.get("training", {}).get("amp_dtype"))
    channels_last = bool(source_cfg.get("training", {}).get("channels_last_3d", False))

    target_cfg = source_cfg.get("target_ae", {})
    target_ae, _ = load_ae_from_config(
        target_cfg.get("config_path"),
        target_cfg.get("checkpoint_path"),
    )
    target_ae = target_ae.to(device).eval()
    model = build_p2sd(source_cfg, latent_channels=target_ae.latent_channels).to(device)
    if channels_last:
        target_ae = target_ae.to(memory_format=torch.channels_last_3d)
        model = model.to(memory_format=torch.channels_last_3d)
    checkpoint = load_model_state(model, checkpoint_path)
    model.eval()
    latent_codec = build_static_latent_codec(
        source_cfg.get("p2sd", {}).get("loss", {}).get("latent_normalization", {}),
        latent_channels=target_ae.latent_channels,
        device=device,
    )
    if refine_head_path:
        # 0061: decode through the image-conditioned refinement head.
        from vesuvius_p2sd.models.refine_head import load_refined_decoder

        target_ae = load_refined_decoder(target_ae, model.image_encoder, refine_head_path, device)
        print(f"[refine] decoding through {refine_head_path}", flush=True)
    threshold = float(source_cfg.get("metrics", {}).get("threshold", 0.5))

    # Round-robin sharding balances large and small cases across workers.
    shard_cases = list(manifest.cases[shard_index::num_shards])
    case_rows: list[dict[str, Any]] = []
    sheet_rows: list[dict[str, Any]] = []
    start_time = time.monotonic()
    for case_index, case in enumerate(shard_cases, start=1):
        source_row = source_rows[case.case_id]
        result = _evaluate_one_case(
            model=model,
            target_ae=target_ae,
            latent_codec=latent_codec,
            source_row=source_row,
            case=case,
            patch_size=patch_size,
            target_contract=target_contract,
            threshold=threshold,
            min_component_voxels=min_component_voxels,
            prompt_batch_size=prompt_batch_size,
            cpu_workers=cpu_workers,
            topology_batch_size=topology_batch_size,
            topology_tile_batch_size=topology_tile_batch_size,
            topology_backend=topology_backend,
            evaluation_mode=evaluation_mode,
            device=device,
            dtype=dtype,
            channels_last=channels_last,
        )
        case_rows.extend(result["case_rows"])
        sheet_rows.extend(result["sheet_rows"])
        elapsed = time.monotonic() - start_time
        progress = {
            "case_index": case_index,
            "case_count": len(shard_cases),
            "shard": f"{shard_index}/{num_shards}",
            "case_id": case.case_id,
            "elapsed_s": elapsed,
            "mean_s_per_case": elapsed / case_index,
            "estimated_remaining_s": (elapsed / case_index) * (len(shard_cases) - case_index),
        }
        write_json(output_dir / "progress.json", progress)
        print(progress, flush=True)

    summary = _summarize_rows(
        case_rows=case_rows,
        sheet_rows=sheet_rows,
        manifest=manifest,
        checkpoint_path=checkpoint_path,
        checkpoint=checkpoint,
        source_run_dir=source_run_dir,
        scene_manifest_path=scene_manifest_path,
        target_contract=target_contract,
        topology_batch_size=topology_batch_size,
        topology_tile_batch_size=topology_tile_batch_size,
        prompt_batch_size=prompt_batch_size,
        min_component_voxels=min_component_voxels,
        topology_backend=topology_backend,
        evaluation_mode=evaluation_mode,
        elapsed_s=time.monotonic() - start_time,
    )
    _write_jsonl(output_dir / "case_metrics.jsonl", case_rows)
    _write_jsonl(output_dir / "sheet_prompt_consistency.jsonl", sheet_rows)
    write_json(output_dir / "summary.json", summary)
    return summary


@torch.no_grad()
def _evaluate_one_case(
    *,
    model,
    target_ae,
    latent_codec,
    source_row: dict[str, Any],
    case: SceneCaseProbe,
    patch_size: tuple[int, int, int],
    target_contract: str,
    threshold: float,
    min_component_voxels: int,
    prompt_batch_size: int,
    cpu_workers: int,
    topology_batch_size: int,
    topology_tile_batch_size: int,
    topology_backend: str,
    evaluation_mode: str,
    device: torch.device,
    dtype: torch.dtype | None,
    channels_last: bool,
) -> dict[str, list[dict[str, Any]]]:
    image = np.load(source_row["image_path"], mmap_mode="r")
    if tuple(image.shape) != case.source_shape:
        raise ValueError(
            f"Case {case.case_id} image shape does not match manifest: "
            f"image={image.shape}, manifest={case.source_shape}")
    components = None
    if evaluation_mode == FULL_EVALUATION_MODE:
        components = np.load(source_row["components_path"], mmap_mode="r")
        _validate_case_source(case, image, components)
    image_tensor = _canvas_image_tensor(
        image,
        source_shape=case.source_shape,
        canvas_offset=case.canvas_offset,
        patch_size=patch_size,
        device=device,
        channels_last=channels_last,
    )
    prompt_ids = case_prompt_ids(case)
    variant_count = len(prompt_ids)
    source_slices = _source_canvas_slices(case.canvas_offset, case.source_shape)
    use_gpu_topology = device.type == "cuda"
    contact_trackers = [
        GpuSheetContactTracker(case.source_shape, device=device)
        if use_gpu_topology
        else SheetContactTracker(case.source_shape)
        for _ in prompt_ids
    ]
    union_gpu = [
        torch.zeros(case.source_shape, dtype=torch.bool, device=device)
        for _ in prompt_ids
    ]
    sheet_rows: list[dict[str, Any]] = []

    with autocast_context(device, dtype):
        _, image_tokens, image_coords, image_context = model.encode_image_context_from_image(image_tensor)
        if hasattr(target_ae, "set_image"):
            target_ae.set_image(image_tensor)

    # Decode all (component, prompt-variant) pairs in flat GPU batches so the
    # batch size is no longer capped at one component's variant count, and
    # overlap the remaining CPU metrics (connected components, pairwise dice)
    # with decoding via a small thread pool.
    for component in case.components:
        if len(component.prompt_sets) != variant_count:
            raise RuntimeError(
                f"Component {component.component_id} has {len(component.prompt_sets)} prompt sets; "
                f"expected {variant_count}")
    flat_prompts = [
        (component_index, variant_index, prompt_set)
        for component_index, component in enumerate(case.components)
        for variant_index, prompt_set in enumerate(component.prompt_sets)
    ]
    component_masks: list[list[np.ndarray | None]] = [
        [None] * variant_count for _ in case.components
    ]
    count_futures: dict[tuple[int, int], Any] = {}
    with ThreadPoolExecutor(max_workers=cpu_workers) as pool:
        for chunk in _chunks(flat_prompts, prompt_batch_size):
            prompt_points = torch.tensor(
                [prompt.points_zyx for _, _, prompt in chunk],
                device=device,
                dtype=torch.float32,
            )
            prompt_labels = torch.tensor(
                [prompt.labels for _, _, prompt in chunk],
                device=device,
                dtype=torch.long,
            )
            image_index = torch.zeros(len(chunk), device=device, dtype=torch.long)
            with autocast_context(device, dtype):
                output = model.forward_from_image_context(
                    image_tokens,
                    image_coords,
                    image_context,
                    prompt_points,
                    prompt_labels,
                    image_shape=patch_size,
                    image_index=image_index,
                )
                decoded_logits = target_ae.decode(latent_codec.raw_prediction(output["latent"]))
            probabilities = decoded_foreground_probability(decoded_logits).float()[:, 0]
            for (component_index, variant_index, _), volume in zip(chunk, probabilities):
                mask_device = (volume >= threshold)[source_slices]
                contact_trackers[variant_index].add(mask_device)
                union_gpu[variant_index] |= mask_device
                mask = np.ascontiguousarray(mask_device.cpu().numpy(), dtype=bool)
                if min_component_voxels > 0:
                    mask = filter_small_components(mask, min_component_voxels)
                component_masks[component_index][variant_index] = mask
                count_futures[(component_index, variant_index)] = pool.submit(component_count, mask)
        union_masks = [
            np.ascontiguousarray(union.cpu().numpy(), dtype=bool) for union in union_gpu
        ]
        if min_component_voxels > 0:
            union_masks = [filter_small_components(u, min_component_voxels) for u in union_masks]
        union_count_futures = [pool.submit(component_count, union) for union in union_masks]
        dice_futures = [
            pool.submit(pairwise_prediction_dice_stats, masks) for masks in component_masks
        ]
        union_dice_future = pool.submit(pairwise_prediction_dice_stats, union_masks)
        for component_index, component in enumerate(case.components):
            counts = [
                float(count_futures[(component_index, variant_index)].result())
                for variant_index in range(variant_count)
            ]
            gt_dice_summary: dict[str, float] = {}
            if components is not None:
                # Per-sheet GT Dice (user directive 2026-08-26: intermediate evals
                # must report accuracy, not only prompt consistency): each prompt
                # variant's decoded sheet against its own GT component.
                gt_mask = np.asarray(components) == component.component_id
                gt_voxels = int(np.count_nonzero(gt_mask))
                gt_dice = []
                for variant_index in range(variant_count):
                    variant_mask = component_masks[component_index][variant_index]
                    if variant_mask is None:
                        continue
                    inter = int(np.count_nonzero(variant_mask & gt_mask))
                    gt_dice.append(2.0 * inter / max(int(np.count_nonzero(variant_mask)) + gt_voxels, 1))
                if gt_dice:
                    gt_dice_summary = {
                        "prompt_prediction_gt_dice_mean": float(np.mean(gt_dice)),
                        "prompt_prediction_gt_dice_min": float(np.min(gt_dice)),
                        "prompt_prediction_gt_dice_max": float(np.max(gt_dice)),
                    }
            sheet_rows.append({
                "case_id": case.case_id,
                "component_id": component.component_id,
                "target_voxels": component.target_voxels,
                **_count_summary(counts, prefix="fast_topology_sheet_component_count"),
                **dice_futures[component_index].result(),
                **gt_dice_summary,
            })
        union_component_counts = [float(f.result()) for f in union_count_futures]
        union_consistency = union_dice_future.result()

    if evaluation_mode == FULL_EVALUATION_MODE:
        if components is None:
            raise RuntimeError("Full evaluation requires source components")
        label = build_scene_target_label(
            components,
            eligible_component_ids=[component.component_id for component in case.components],
            target_contract=target_contract,
        )
        if min_component_voxels > 0:
            # Symmetric filter: drop sub-threshold GT foreground components too, so
            # the size filter is applied identically to prediction and target. A
            # no-op on the current data (GT foreground has no components < ~16k
            # voxels; sub-threshold sheets are already ignore=2), but it keeps the
            # comparison airtight if a future GT ever has small foreground pieces.
            foreground = label == 1
            kept = filter_small_components(foreground, min_component_voxels)
            removed = foreground & ~kept
            if removed.any():
                label = np.where(removed, np.uint8(0), label)
        case_metrics = []
        for start in range(0, len(union_masks), topology_batch_size):
            stop = min(start + topology_batch_size, len(union_masks))
            case_metrics.extend(compute_kaggle_case_metrics_batch(
                union_masks[start:stop],
                [label] * (stop - start),
                topology_backend=topology_backend,
                topology_tile_batch_size=topology_tile_batch_size,
            ))
    else:
        case_metrics = [{} for _ in union_masks]
    case_rows = [
        {
            "case_id": case.case_id,
            "prompt_set_id": prompt_id,
            "target_contract": target_contract,
            "eligible_component_count": len(case.components),
            "eligible_target_voxels": int(sum(component.target_voxels for component in case.components)),
            "scene_prediction_voxels": int(prediction.sum()),
            "fast_topology_volume_union_component_count": union_count,
            **tracker.summary(),
            **union_consistency,
            **metrics,
        }
        for prompt_id, prediction, union_count, tracker, metrics in zip(
            prompt_ids, union_masks, union_component_counts, contact_trackers, case_metrics,
            strict=True)
    ]
    return {"case_rows": case_rows, "sheet_rows": sheet_rows}


def filter_small_components(mask: np.ndarray, min_voxels: int, *, connectivity: int = 3) -> np.ndarray:
    """Drop connected components smaller than ``min_voxels`` from a binary mask.

    Post-processing lever for topology (component-count) error: the union/sheet
    component counts are raw (every 1-voxel speck counts), so a size threshold
    removes spurious fragments before counting and scoring. connectivity=3 is
    26-connectivity, matching ``connected_components.component_count``'s default.
    ``min_voxels <= 0`` is a no-op, preserving the unfiltered behaviour.
    """

    if min_voxels <= 0 or not mask.any():
        return mask
    try:
        import cc3d

        # dust() removes components with fewer than `threshold` voxels in one
        # optimized pass (~2x faster than scipy label+bincount on 320^3). 26-conn
        # matches connectivity=3; verified voxel-identical to the scipy path.
        conn = {1: 6, 2: 18, 3: 26}[int(connectivity)]
        return cc3d.dust(
            np.ascontiguousarray(mask), threshold=int(min_voxels),
            connectivity=conn, in_place=False,
        ).astype(bool)
    except Exception:
        from scipy import ndimage

        structure = ndimage.generate_binary_structure(3, connectivity)
        labels, n = ndimage.label(mask, structure=structure)
        if n == 0:
            return mask
        sizes = np.bincount(labels.reshape(-1))
        keep = sizes >= int(min_voxels)
        keep[0] = False  # background
        return keep[labels]


def _count_summary(values: Sequence[float], prefix: str) -> dict[str, float]:
    """Summarize raw 26-connected component counts across prompt variants."""
    array = np.asarray(list(values), dtype=np.float64)
    if array.size == 0:
        return {f"{prefix}_mean": 0.0, f"{prefix}_max": 0.0}
    return {
        f"{prefix}_mean": float(array.mean()),
        f"{prefix}_max": float(array.max()),
    }


def _component_count_summary(
    masks: Sequence[np.ndarray],
    *,
    prefix: str,
) -> dict[str, float]:
    return _count_summary([component_count(mask) for mask in masks], prefix)


def _canvas_image_tensor(
    image: np.ndarray,
    *,
    source_shape: tuple[int, int, int],
    canvas_offset: tuple[int, int, int],
    patch_size: tuple[int, int, int],
    device: torch.device,
    channels_last: bool,
) -> torch.Tensor:
    canvas = np.zeros(patch_size, dtype=np.asarray(image).dtype)
    canvas[_source_canvas_slices(canvas_offset, source_shape)] = image
    tensor = torch.from_numpy(canvas[None, None]).to(device, non_blocking=True).float()
    return maybe_channels_last_3d(tensor, channels_last)


def _crop_native_canvas(
    canvas: np.ndarray,
    canvas_offset: tuple[int, int, int],
    source_shape: tuple[int, int, int],
) -> np.ndarray:
    return np.ascontiguousarray(canvas[_source_canvas_slices(canvas_offset, source_shape)], dtype=bool)


def _source_canvas_slices(
    canvas_offset: tuple[int, int, int],
    source_shape: tuple[int, int, int],
) -> tuple[slice, slice, slice]:
    return tuple(slice(offset, offset + size) for offset, size in zip(canvas_offset, source_shape))


def build_scene_target_label(
    components: np.ndarray,
    *,
    eligible_component_ids: Sequence[int],
    target_contract: str,
) -> np.ndarray:
    """Build an all-sheet or threshold-aligned target using public ignore label 2."""

    component_array = np.asarray(components)
    if component_array.ndim != 3:
        raise ValueError(f"components must be [D,H,W], got {component_array.shape}")
    if target_contract == ALL_SHEETS_CONTRACT:
        return np.ascontiguousarray(component_array > 0, dtype=np.uint8)
    if target_contract != ELIGIBLE_ONLY_CONTRACT:
        raise ValueError(f"Unsupported target contract: {target_contract}")
    if not eligible_component_ids:
        raise ValueError("eligible_only target requires at least one eligible component")
    maximum_id = int(component_array.max(initial=0))
    lookup = np.zeros(maximum_id + 1, dtype=np.uint8)
    lookup[np.asarray(eligible_component_ids, dtype=np.int64)] = 1
    label = np.where(component_array > 0, 2, 0).astype(np.uint8, copy=False)
    eligible = lookup[component_array]
    label[eligible > 0] = 1
    return np.ascontiguousarray(label)


def case_prompt_ids(case: SceneCaseProbe) -> tuple[str, ...]:
    if not case.components:
        raise ValueError(f"Scene case {case.case_id} has no components")
    return tuple(prompt.prompt_set_id for prompt in case.components[0].prompt_sets)


def _load_source_rows(cfg: dict[str, Any], manifest: SceneProbeManifest) -> dict[str, dict[str, Any]]:
    dataset_root = Path(cfg.get("data", {}).get("dataset_root", ""))
    if not dataset_root:
        raise ValueError("source run config needs data.dataset_root")
    rows = load_jsonl(dataset_root / f"manifest_{manifest.split}.jsonl")
    by_case_id = {str(row["case_id"]): row for row in rows}
    missing = [case.case_id for case in manifest.cases if case.case_id not in by_case_id]
    if missing:
        raise ValueError(f"Scene manifest cases are absent from {manifest.split} data: {missing[:5]}")
    return by_case_id


def _patch_size_from_config(cfg: dict[str, Any]) -> tuple[int, int, int]:
    values = cfg.get("data", {}).get("patch_size", cfg.get("data", {}).get("crop_size", []))
    patch_size = tuple(int(value) for value in values)
    if len(patch_size) != 3:
        raise ValueError(f"source run config needs a three-dimensional data.patch_size, got {values}")
    return patch_size


def _validate_case_source(case: SceneCaseProbe, image: np.ndarray, components: np.ndarray) -> None:
    if tuple(image.shape) != case.source_shape or tuple(components.shape) != case.source_shape:
        raise ValueError(
            f"Case {case.case_id} source shapes do not match manifest: image={image.shape}, "
            f"components={components.shape}, manifest={case.source_shape}")


def _chunks(items: Sequence[Any], size: int) -> Iterator[Sequence[Any]]:
    for start in range(0, len(items), size):
        yield items[start:start + size]


def _summarize_rows(
    *,
    case_rows: Sequence[dict[str, Any]],
    sheet_rows: Sequence[dict[str, Any]],
    manifest: SceneProbeManifest,
    checkpoint_path: Path,
    checkpoint: dict[str, Any],
    source_run_dir: Path,
    scene_manifest_path: Path,
    target_contract: str,
    topology_batch_size: int,
    topology_tile_batch_size: int,
    prompt_batch_size: int,
    min_component_voxels: int,
    topology_backend: str,
    evaluation_mode: str,
    elapsed_s: float,
) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_step": int(checkpoint.get("step", 0)),
        "source_run_dir": str(source_run_dir),
        # Recorded so a finished run can be replayed (see
        # research.visualize_worst_scene_cases) without re-deriving the settings
        # that produced its numbers.
        "scene_manifest_path": str(scene_manifest_path),
        "min_component_voxels": int(min_component_voxels),
        "scene_manifest_dataset_root": manifest.dataset_root,
        "scene_manifest_split": manifest.split,
        "scene_manifest_min_target_voxels": manifest.min_target_voxels,
        "target_contract": target_contract,
        "evaluation_mode": evaluation_mode,
        "target_contract_description": (
            "Sheets below min_target_voxels are ignore label 2; this is not a literal all-sheet leaderboard score."
            if target_contract == ELIGIBLE_ONLY_CONTRACT
            else "All annotated sheets are foreground, including sheets without a prompt."
        ),
        "case_count": len(manifest.cases),
        "eligible_component_count": int(sum(len(case.components) for case in manifest.cases)),
        "eligible_target_voxels": int(sum(
            component.target_voxels for case in manifest.cases for component in case.components
        )),
        "prompt_set_ids": list(manifest.prompt_set_ids),
        "topology_batch_size": topology_batch_size,
        "topology_tile_batch_size": topology_tile_batch_size,
        "topology_backend": topology_backend,
        "prompt_batch_size": prompt_batch_size,
        "elapsed_s": elapsed_s,
    }
    metric_names = sorted({
        key for row in case_rows for key, value in row.items()
        if key.startswith("kaggle_") and isinstance(value, (int, float))
    })
    for metric in metric_names:
        values = np.asarray([float(row[metric]) for row in case_rows if metric in row], dtype=np.float64)
        if values.size:
            summary[f"case_variant_macro_{metric}"] = float(values.mean())
            worst = values.max() if metric in LOWER_IS_BETTER_KAGGLE_METRICS else values.min()
            summary[f"case_variant_worst_{metric}"] = float(worst)
        for prompt_id in manifest.prompt_set_ids:
            prompt_values = np.asarray([
                float(row[metric]) for row in case_rows
                if row.get("prompt_set_id") == prompt_id and metric in row
            ], dtype=np.float64)
            if prompt_values.size:
                summary[f"{prompt_id}_macro_{metric}"] = float(prompt_values.mean())
    _append_summary_means(summary, case_rows, prefix="case_union_prompt_", keys=(
        "prompt_prediction_pairwise_dice_mean",
        "prompt_prediction_pairwise_dice_min",
    ))
    _append_component_count_summary(
        summary,
        case_rows,
        prefix="case_union_",
        key="fast_topology_volume_union_component_count",
    )
    _append_summary_means(summary, case_rows, prefix="case_scene_", keys=(
        "sheet_contact_pair_count",
        "sheet_contact_component_count",
        "sheet_contact_overlap_voxels",
        "sheet_contact_free",
    ))
    _append_contact_case_summary(summary, case_rows)
    _append_summary_means(summary, sheet_rows, prefix="sheet_prompt_", keys=(
        "prompt_prediction_pairwise_dice_mean",
        "prompt_prediction_pairwise_dice_min",
    ))
    _append_summary_means(summary, sheet_rows, prefix="sheet_prompt_", keys=(
        "prompt_prediction_gt_dice_mean",
        "prompt_prediction_gt_dice_min",
    ))
    _append_component_count_summary(
        summary,
        sheet_rows,
        prefix="sheet_prompt_",
        key="fast_topology_sheet_component_count_mean",
    )
    _append_component_count_summary(
        summary,
        sheet_rows,
        prefix="sheet_prompt_",
        key="fast_topology_sheet_component_count_max",
    )
    return summary


def _append_contact_case_summary(
    summary: dict[str, Any],
    rows: Sequence[dict[str, Any]],
) -> None:
    """Count the distinct cases where predicted sheets touch under any prompt.

    ``case_scene_macro_sheet_contact_free`` averages over (case, prompt-set)
    rows, so a scene that only collides under one of its four prompts is diluted
    to 0.75 and reads as mostly fine. Sheet contact is really a per-case defect
    -- the scene either has sheets bleeding into each other or it does not -- and
    the case count is the number worth quoting.
    """

    pairs_by_case: dict[str, float] = {}
    for row in rows:
        if UNION_PAIR_COUNT_KEY not in row:
            continue
        case_id = str(row.get("case_id", ""))
        pairs_by_case[case_id] = pairs_by_case.get(case_id, 0.0) + float(row[UNION_PAIR_COUNT_KEY])
    if not pairs_by_case:
        return
    contact_cases = sum(1 for pairs in pairs_by_case.values() if pairs > 0.0)
    summary["case_scene_sheet_contact_case_count"] = int(contact_cases)
    summary["case_scene_sheet_contact_evaluated_case_count"] = int(len(pairs_by_case))
    summary["case_scene_sheet_contact_case_fraction"] = float(contact_cases / len(pairs_by_case))


def _append_summary_means(
    summary: dict[str, Any],
    rows: Sequence[dict[str, Any]],
    *,
    prefix: str,
    keys: Sequence[str],
) -> None:
    for key in keys:
        values = np.asarray([float(row[key]) for row in rows if key in row], dtype=np.float64)
        if values.size:
            summary[f"{prefix}macro_{key}"] = float(values.mean())


def _append_component_count_summary(
    summary: dict[str, Any],
    rows: Sequence[dict[str, Any]],
    *,
    prefix: str,
    key: str,
) -> None:
    values = np.asarray([float(row[key]) for row in rows if key in row], dtype=np.float64)
    if values.size:
        summary[f"{prefix}macro_{key}"] = float(values.mean())
        summary[f"{prefix}worst_{key}"] = float(values.max())


def _write_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    import json

    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, sort_keys=True) + "\n")


def _merge_shard_rows(
    shard_dirs: Sequence[Path],
    file_name: str,
    case_order: dict[str, int],
) -> list[dict[str, Any]]:
    """Concatenate shard row files and restore manifest case order."""
    rows: list[dict[str, Any]] = []
    for shard_dir in shard_dirs:
        rows.extend(load_jsonl(shard_dir / file_name))
    missing = {str(row["case_id"]) for row in rows} - set(case_order)
    if missing:
        raise RuntimeError(f"Shard rows reference cases missing from manifest: {sorted(missing)}")
    rows.sort(key=lambda row: case_order[str(row["case_id"])])
    return rows


def per_shard_prompt_batch(prompt_batch_size: int, parallel: int, num_devices: int = 1) -> int:
    """Decode batch per shard: the aggregate is a PER-DEVICE memory budget.

    With one device, `parallel` shards share it. With N devices the shards are
    spread round-robin, so only ceil(parallel / N) shards coexist on any one
    device -- each can take a correspondingly larger slice. Keeping this exact
    also keeps multi-GPU runs batch-identical to the single-GPU runs they are
    compared against (2 GPUs x 6 shards -> batch 3, same as 1 GPU x 3 shards).
    """
    import math

    shards_per_device = math.ceil(parallel / max(1, num_devices))
    return max(2, math.ceil(prompt_batch_size / shards_per_device))


def run_parallel_scene_evaluation(
    *,
    checkpoint_path: str | Path,
    source_run_dir: str | Path,
    scene_manifest_path: str | Path,
    output_dir: str | Path,
    target_contract: str,
    parallel: int,
    prompt_batch_size: int = 8,
    cpu_workers: int = 4,
    topology_batch_size: int = 2,
    topology_tile_batch_size: int = 4,
    topology_backend: str = "reference_batch",
    evaluation_mode: str = FULL_EVALUATION_MODE,
    min_component_voxels: int = 0,
    device_name: str | None = None,
    cuda_devices: Sequence[str] | None = None,
    refine_head_path: str | Path | None = None,
) -> dict[str, Any]:
    """Shard cases across subprocesses and merge the results.

    Used both by the CLI (--parallel) and by the P2SD training loop during its
    eval window, where training is paused and the evaluator owns the device.
    `cuda_devices` spreads the shards round-robin over several GPUs by setting
    each shard's CUDA_VISIBLE_DEVICES (e.g. ("0", "1") with --parallel 6 puts
    three shards on each GPU and roughly halves the wall time).
    """
    import os
    import subprocess
    import sys

    if parallel <= 0:
        raise ValueError("parallel must be positive")
    cuda_devices = tuple(str(device) for device in cuda_devices) if cuda_devices else ()
    if cuda_devices and device_name is not None:
        raise ValueError("cuda_devices and device_name are mutually exclusive")
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Evaluation output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    per_shard_batch = per_shard_prompt_batch(
        prompt_batch_size, parallel, num_devices=max(1, len(cuda_devices)))
    shard_env = {**os.environ, "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}
    print(
        f"[parallel] {parallel} shards, prompt_batch_size {per_shard_batch} each "
        f"(aggregate {prompt_batch_size} per device"
        + (f", devices {','.join(cuda_devices)})" if cuda_devices else ")"),
        flush=True,
    )
    start_time = time.monotonic()
    processes = []
    shard_dirs = []
    for shard_index in range(parallel):
        shard_dir = output_dir / f"shard_{shard_index:02d}"
        shard_dirs.append(shard_dir)
        command = [
            sys.executable, "-m", "vesuvius_p2sd.research.evaluate_p2sd_case_scenes",
            "--checkpoint_path", str(checkpoint_path),
            "--source_run_dir", str(source_run_dir),
            "--scene_manifest_path", str(scene_manifest_path),
            "--output_dir", str(shard_dir),
            "--target_contract", target_contract,
            "--topology_batch_size", str(topology_batch_size),
            "--topology_tile_batch_size", str(topology_tile_batch_size),
            "--prompt_batch_size", str(per_shard_batch),
            "--cpu_workers", str(cpu_workers),
            "--topology_backend", topology_backend,
            "--evaluation_mode", evaluation_mode,
            "--min_component_voxels", str(min_component_voxels),
            "--shard_index", str(shard_index),
            "--num_shards", str(parallel),
        ]
        if device_name is not None:
            command.extend(["--device", device_name])
        if refine_head_path:
            command.extend(["--refine_head_path", str(refine_head_path)])
        env = shard_env
        if cuda_devices:
            env = {**shard_env, "CUDA_VISIBLE_DEVICES": cuda_devices[shard_index % len(cuda_devices)]}
        log_path = output_dir / f"shard_{shard_index:02d}.log"
        processes.append((shard_index, subprocess.Popen(
            command, stdout=log_path.open("w"), stderr=subprocess.STDOUT, env=env)))
    failures = [index for index, process in processes if process.wait() != 0]
    if failures:
        raise RuntimeError(
            f"Shards {failures} failed; see {output_dir}/shard_XX.log")

    source_cfg = load_config(Path(source_run_dir) / "resolved_config.yaml")
    manifest = load_scene_probe_manifest(
        scene_manifest_path, patch_size=_patch_size_from_config(source_cfg))
    case_order = {case.case_id: index for index, case in enumerate(manifest.cases)}
    case_rows = _merge_shard_rows(shard_dirs, "case_metrics.jsonl", case_order)
    sheet_rows = _merge_shard_rows(shard_dirs, "sheet_prompt_consistency.jsonl", case_order)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    summary = _summarize_rows(
        case_rows=case_rows,
        sheet_rows=sheet_rows,
        manifest=manifest,
        checkpoint_path=Path(checkpoint_path),
        checkpoint=checkpoint,
        source_run_dir=Path(source_run_dir),
        scene_manifest_path=Path(scene_manifest_path),
        target_contract=target_contract,
        topology_batch_size=topology_batch_size,
        topology_tile_batch_size=topology_tile_batch_size,
        prompt_batch_size=prompt_batch_size,
        min_component_voxels=min_component_voxels,
        topology_backend=topology_backend,
        evaluation_mode=evaluation_mode,
        elapsed_s=time.monotonic() - start_time,
    )
    summary["parallel_shards"] = parallel
    # The exact per-shard decode batch, so replays (worst-case viz) can reproduce
    # voxel counts without re-deriving the split -- which depends on the device
    # count for multi-GPU runs.
    summary["per_shard_prompt_batch_size"] = per_shard_batch
    if cuda_devices:
        summary["parallel_cuda_devices"] = list(cuda_devices)
    _write_jsonl(output_dir / "case_metrics.jsonl", case_rows)
    _write_jsonl(output_dir / "sheet_prompt_consistency.jsonl", sheet_rows)
    write_json(output_dir / "summary.json", summary)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint_path", required=True)
    parser.add_argument("--source_run_dir", required=True)
    parser.add_argument("--scene_manifest_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--target_contract", choices=TARGET_CONTRACTS, required=True)
    parser.add_argument("--topology_batch_size", type=int, default=2)
    parser.add_argument("--topology_tile_batch_size", type=int, default=4)
    parser.add_argument("--prompt_batch_size", type=int, default=8)
    parser.add_argument("--cpu_workers", type=int, default=4)
    parser.add_argument("--shard_index", type=int, default=0)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument(
        "--parallel",
        type=int,
        default=1,
        help="Spawn this many shard subprocesses and merge results.",
    )
    parser.add_argument(
        "--cuda_devices",
        default=None,
        help="Comma-separated GPU ids to spread --parallel shards over round-robin "
             "(e.g. '0,1' with --parallel 6 puts 3 shards on each GPU, ~halving wall "
             "time). Default: all shards on the current device. The per-shard batch "
             "is derived per device, keeping results batch-identical to a single-GPU "
             "run with the same shards-per-device.",
    )
    parser.add_argument(
        "--topology_backend",
        choices=("reference_batch", "count_only", "binary_exact", "compact_exact"),
        default="reference_batch",
    )
    parser.add_argument(
        "--evaluation_mode",
        choices=EVALUATION_MODES,
        default=FULL_EVALUATION_MODE,
        help="Use topology_proxies to skip the public Kaggle metric calculation.",
    )
    parser.add_argument("--min_component_voxels", type=int, default=500,
        help="Drop connected components smaller than this (predicted + GT) before counting/scoring. 0=off. Default 500 (adopted 2026-07-24: removes false-positive specks, near-zeroes component-count error, dice unchanged).")
    parser.add_argument("--visualize_worst_n", type=int, default=0,
        help="After scoring, write NIfTI volumes for the N worst-fragmented sheets and the N scenes "
             "with the most sheet-to-sheet contact into <output_dir>/worst_case_viz. 0 disables.")
    parser.add_argument("--device", default=None)
    parser.add_argument("--refine_head_path", default=None,
                        help="SheetRefineHead checkpoint (refine_last.pt): decode through the "
                             "image-conditioned refinement (0061).")
    args = parser.parse_args(argv)
    cuda_devices = [d.strip() for d in args.cuda_devices.split(",") if d.strip()] if args.cuda_devices else None
    if cuda_devices and args.parallel <= 1:
        raise SystemExit("--cuda_devices requires --parallel > 1")
    if args.parallel > 1:
        if args.num_shards != 1 or args.shard_index != 0:
            raise SystemExit("--parallel cannot be combined with explicit shard arguments")
        summary = run_parallel_scene_evaluation(
            checkpoint_path=args.checkpoint_path,
            source_run_dir=args.source_run_dir,
            scene_manifest_path=args.scene_manifest_path,
            output_dir=args.output_dir,
            target_contract=args.target_contract,
            parallel=args.parallel,
            prompt_batch_size=args.prompt_batch_size,
            cpu_workers=args.cpu_workers,
            topology_batch_size=args.topology_batch_size,
            topology_tile_batch_size=args.topology_tile_batch_size,
            topology_backend=args.topology_backend,
            evaluation_mode=args.evaluation_mode,
            min_component_voxels=args.min_component_voxels,
            refine_head_path=args.refine_head_path,
            device_name=args.device,
            cuda_devices=cuda_devices,
        )
        print(summary, flush=True)
        _maybe_visualize_worst_cases(args)
        return 0
    summary = evaluate_case_scenes(
        checkpoint_path=args.checkpoint_path,
        source_run_dir=args.source_run_dir,
        scene_manifest_path=args.scene_manifest_path,
        output_dir=args.output_dir,
        target_contract=args.target_contract,
        topology_batch_size=args.topology_batch_size,
        topology_tile_batch_size=args.topology_tile_batch_size,
        prompt_batch_size=args.prompt_batch_size,
        cpu_workers=args.cpu_workers,
        topology_backend=args.topology_backend,
        evaluation_mode=args.evaluation_mode,
        min_component_voxels=args.min_component_voxels,
        device_name=args.device,
        shard_index=args.shard_index,
        num_shards=args.num_shards,
        refine_head_path=args.refine_head_path,
    )
    print(summary, flush=True)
    if args.num_shards == 1:
        _maybe_visualize_worst_cases(args)
    return 0


def _maybe_visualize_worst_cases(args: argparse.Namespace) -> None:
    """Dump NIfTI volumes for the worst cases once the scores are on disk.

    Deliberately post-hoc and non-fatal: the evaluation is the expensive part and
    a viewer artifact is not worth losing it over. Never runs inside a shard --
    it needs the merged rows.
    """

    if args.visualize_worst_n <= 0:
        return
    from vesuvius_p2sd.research.visualize_worst_scene_cases import visualize_worst_cases

    try:
        index = visualize_worst_cases(
            eval_dir=args.output_dir,
            scene_manifest_path=args.scene_manifest_path,
            top_n=args.visualize_worst_n,
            min_component_voxels=args.min_component_voxels,
            device_name=args.device,
        )
    except Exception as error:  # noqa: BLE001 - telemetry must not fail the run
        print(f"[worst_viz] skipped: {type(error).__name__}: {error}", flush=True)
        return
    print(
        f"[worst_viz] wrote {len(index['worst_sheet_cc'])} sheet and "
        f"{len(index['worst_union_contact'])} union case(s) to {args.output_dir}/worst_case_viz",
        flush=True,
    )


if __name__ == "__main__":
    raise SystemExit(main())
