"""Dump NIfTI visualizations for the worst cases of a finished case-scene eval.

``evaluate_p2sd_case_scenes`` reports aggregate topology numbers but throws the
volumes away, so a bad component count is a number with no picture attached.
This module reads a finished evaluation directory, ranks the failures, and
re-decodes *only* those cases to write inspectable NIfTI volumes.

Two rankings, matching the two component-count metrics the evaluator reports:

``worst_sheet_cc``
    Per prompted sheet, ranked by ``fast_topology_sheet_component_count_max``:
    the sheets whose prediction fragments into more than one blob (or vanishes
    entirely, which ranks worst of all). Writes the prediction with each
    connected component given its own label id, so the spurious fragment is
    immediately visible next to the sheet it broke off from.

``worst_union_contact``
    Per (case, prompt-set) scene union, ranked by ``sheet_contact_overlap_voxels``:
    the scenes where predicted sheets bleed into each other. Writes the union,
    a per-sheet id volume, and the contact/overlap volume that produced the
    metric.

Every output directory is named ``rank<NN>_<case_id>...`` and also records the
case id inside ``metrics.json``, alongside the originating evaluator row.

Masks are reproduced with the evaluator's exact conventions: the size filter is
applied per sheet and to the union *after* accumulation, while sheet-contact
tracking uses unfiltered masks (see ``evaluate_p2sd_case_scenes``). Pass the same
``--min_component_voxels`` the evaluation used or the pictures will not match the
numbers.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from vesuvius_p2sd.data.dataset import load_jsonl
from vesuvius_p2sd.data.scene_probes import SceneCaseProbe, load_scene_probe_manifest
from vesuvius_p2sd.eval.connected_components import label_components
from vesuvius_p2sd.models.p2sd import build_p2sd
from vesuvius_p2sd.research.evaluate_p2sd_case_scenes import (
    ELIGIBLE_ONLY_CONTRACT,
    TARGET_CONTRACTS,
    UNION_PAIR_COUNT_KEY as UNION_PAIR_KEY,
    _canvas_image_tensor,
    _chunks,
    _load_source_rows,
    _patch_size_from_config,
    _source_canvas_slices,
    build_scene_target_label,
    case_prompt_ids,
    filter_small_components,
)
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


SHEET_COUNT_MAX_KEY = "fast_topology_sheet_component_count_max"
SHEET_COUNT_MEAN_KEY = "fast_topology_sheet_component_count_mean"
UNION_OVERLAP_KEY = "sheet_contact_overlap_voxels"
UNION_COUNT_KEY = "fast_topology_volume_union_component_count"

IMAGE_DTYPES = ("uint8", "float32", "none")

NIFTI_AXIS_ORDER = "z,y,x (same as the source .npy arrays)"


def sheet_defect_score(row: dict[str, Any]) -> float:
    """How badly a prompted sheet fragments. ``0.0`` means one blob per prompt.

    An all-empty prediction (``count_max == 0``) scores ``inf``: predicting
    nothing is a worse failure than any amount of fragmentation, and it would
    otherwise sort as "better than perfect" under a raw count comparison.
    """

    count_max = float(row.get(SHEET_COUNT_MAX_KEY, 0.0))
    count_mean = float(row.get(SHEET_COUNT_MEAN_KEY, 0.0))
    if count_max <= 0.0:
        return math.inf
    return (count_max - 1.0) + 0.01 * max(count_mean - 1.0, 0.0)


def select_worst_sheets(
    sheet_rows: Sequence[dict[str, Any]],
    *,
    top_n: int,
) -> list[dict[str, Any]]:
    """Rank prompted sheets worst-first by predicted component count."""

    scored = [(sheet_defect_score(row), row) for row in sheet_rows]
    scored = [(score, row) for score, row in scored if score > 0.0]
    scored.sort(key=lambda item: (
        -item[0],
        float(item[1].get("prompt_prediction_pairwise_dice_min", 1.0)),
        str(item[1].get("case_id", "")),
        int(item[1].get("component_id", 0)),
    ))
    selections = []
    for rank, (score, row) in enumerate(scored[:max(int(top_n), 0)]):
        selections.append({
            "rank": rank,
            "case_id": str(row["case_id"]),
            "component_id": int(row["component_id"]),
            "defect_score": None if math.isinf(score) else float(score),
            "empty_prediction": math.isinf(score),
            "eval_row": dict(row),
        })
    return selections


def select_worst_union_contacts(
    case_rows: Sequence[dict[str, Any]],
    *,
    top_n: int,
    dedupe_cases: bool = True,
) -> list[dict[str, Any]]:
    """Rank scene unions worst-first by how much predicted sheets touch/overlap.

    With ``dedupe_cases`` (the default) each case contributes only its worst
    prompt variant, so ``top_n`` distinct cases come back instead of the same
    bad case four times.
    """

    rows = [row for row in case_rows if float(row.get(UNION_PAIR_KEY, 0.0)) > 0.0]
    rows.sort(key=lambda row: (
        -float(row.get(UNION_OVERLAP_KEY, 0.0)),
        -float(row.get(UNION_PAIR_KEY, 0.0)),
        str(row.get("case_id", "")),
        str(row.get("prompt_set_id", "")),
    ))
    if dedupe_cases:
        seen: set[str] = set()
        deduped = []
        for row in rows:
            case_id = str(row.get("case_id", ""))
            if case_id in seen:
                continue
            seen.add(case_id)
            deduped.append(row)
        rows = deduped
    selections = []
    for rank, row in enumerate(rows[:max(int(top_n), 0)]):
        selections.append({
            "rank": rank,
            "case_id": str(row["case_id"]),
            "prompt_set_id": str(row["prompt_set_id"]),
            "contact_overlap_voxels": float(row.get(UNION_OVERLAP_KEY, 0.0)),
            "contact_pair_count": float(row.get(UNION_PAIR_KEY, 0.0)),
            "eval_row": dict(row),
        })
    return selections


def eval_prompt_batch_size(summary: dict[str, Any]) -> int:
    """The decode batch size a shard of that evaluation actually used.

    ``summary["prompt_batch_size"]`` is the *aggregate* the CLI was given;
    ``run_parallel_scene_evaluation`` divides it across shards. Reusing the
    aggregate would repack the batches and stop the volumes from reproducing the
    evaluator's voxel counts. Evaluations since the multi-GPU flag record the
    exact per-shard value (the division depends on the device count); older
    single-device summaries fall back to recomputing it the way the runner did.
    """

    recorded = summary.get("per_shard_prompt_batch_size")
    if recorded is not None:
        return int(recorded)
    aggregate = int(summary.get("prompt_batch_size", 8))
    shards = int(summary.get("parallel_shards", 1) or 1)
    return max(2, math.ceil(aggregate / shards)) if shards > 1 else aggregate


def _safe_token(value: Any) -> str:
    text = str(value)
    return "".join(char if (char.isalnum() or char in "-_.") else "_" for char in text) or "unnamed"


def _sphere_offsets(radius: int) -> list[tuple[int, int, int]]:
    radius = max(int(radius), 0)
    limit = radius * radius
    return [
        (dz, dy, dx)
        for dz in range(-radius, radius + 1)
        for dy in range(-radius, radius + 1)
        for dx in range(-radius, radius + 1)
        if dz * dz + dy * dy + dx * dx <= limit
    ]


def stamp_prompt_points(
    shape: tuple[int, int, int],
    points_zyx: Sequence[Sequence[int]],
    values: Sequence[int],
    *,
    radius: int = 2,
) -> np.ndarray:
    """Render prompt points as small filled spheres so they are visible in a viewer."""

    volume = np.zeros(shape, dtype=np.uint16)
    offsets = _sphere_offsets(radius)
    for point, value in zip(points_zyx, values, strict=True):
        z, y, x = (int(coordinate) for coordinate in point)
        for dz, dy, dx in offsets:
            zz, yy, xx = z + dz, y + dy, x + dx
            if 0 <= zz < shape[0] and 0 <= yy < shape[1] and 0 <= xx < shape[2]:
                volume[zz, yy, xx] = np.uint16(value)
    return volume


def label_by_size(mask: np.ndarray, *, connectivity: int = 3) -> tuple[np.ndarray, list[int]]:
    """Label connected components with id 1 = largest, so extras read as 2, 3, ...

    Uses the same 26-connectivity the evaluator counts with, so the number of
    distinct labels equals the reported component count.
    """

    labels, count = label_components(mask, connectivity=connectivity)
    if count == 0:
        return np.zeros(mask.shape, dtype=np.uint16), []
    sizes = np.bincount(labels.reshape(-1), minlength=count + 1)
    sizes[0] = 0
    order = np.argsort(-sizes[1:], kind="stable") + 1
    # uint16 keeps the volumes small; an unfiltered mask can exceed 65535 specks.
    dtype = np.uint16 if count < np.iinfo(np.uint16).max else np.int32
    remap = np.zeros(count + 1, dtype=dtype)
    remap[order] = np.arange(1, count + 1, dtype=dtype)
    return remap[labels], [int(sizes[old]) for old in order]


def overlap_volume(gt_mask: np.ndarray, pred_mask: np.ndarray) -> np.ndarray:
    """1 = GT only, 2 = prediction only, 3 = both. Matches the training viz coding."""

    volume = np.zeros(gt_mask.shape, dtype=np.uint8)
    volume[gt_mask & ~pred_mask] = 1
    volume[pred_mask & ~gt_mask] = 2
    volume[gt_mask & pred_mask] = 3
    return volume


def _dice(a: np.ndarray, b: np.ndarray) -> float:
    total = int(a.sum()) + int(b.sum())
    if total == 0:
        return 1.0
    return float(2.0 * int(np.logical_and(a, b).sum()) / total)


def _save_nifti(path: Path, array: np.ndarray) -> str:
    import nibabel as nib

    path.parent.mkdir(parents=True, exist_ok=True)
    affine = np.eye(4, dtype=np.float32)
    nib.save(nib.Nifti1Image(np.ascontiguousarray(array), affine), str(path))
    return path.name


def _image_for_nifti(image: np.ndarray, image_dtype: str) -> tuple[np.ndarray | None, dict[str, Any]]:
    if image_dtype == "none":
        return None, {}
    array = np.asarray(image)
    info = {
        "source_dtype": str(array.dtype),
        "source_min": float(array.min()),
        "source_max": float(array.max()),
    }
    if image_dtype == "uint8":
        # This dataset stores 0..255 in float16, so uint8 is lossless here and
        # ~8x smaller. source_min/source_max above make any clipping detectable.
        return np.clip(array.astype(np.float32), 0.0, 255.0).astype(np.uint8), info
    return array.astype(np.float32), info


@torch.no_grad()
def _render_case(
    *,
    model,
    target_ae,
    latent_codec,
    source_row: dict[str, Any],
    case: SceneCaseProbe,
    patch_size: tuple[int, int, int],
    threshold: float,
    min_component_voxels: int,
    prompt_batch_size: int,
    sheet_component_ids: set[int],
    union_variants: set[int],
    device: torch.device,
    dtype: torch.dtype | None,
    channels_last: bool,
    decode_all_prompts: bool = True,
) -> dict[str, Any]:
    """Re-decode one case, keeping only the sheets/variants that were selected.

    Mirrors ``evaluate_p2sd_case_scenes._evaluate_one_case``: unfiltered masks
    feed the union and the sheet-contact bookkeeping, the size filter is applied
    per sheet and once more to the accumulated union.

    ``decode_all_prompts`` submits every (component, prompt) pair in the
    evaluator's order even though most results are discarded. That looks
    wasteful, and it is, but under autocast the summed reductions depend on how
    a batch is packed: decoding only the selected prompts repacks the batches
    and shifts borderline voxels, which moves voxel-count metrics by a few
    percent. Skipping the unused decodes is faster and still reproduces the
    integer component counts exactly, but not ``sheet_contact_overlap_voxels``.
    """

    image = np.load(source_row["image_path"], mmap_mode="r")
    image_tensor = _canvas_image_tensor(
        image,
        source_shape=case.source_shape,
        canvas_offset=case.canvas_offset,
        patch_size=patch_size,
        device=device,
        channels_last=channels_last,
    )
    source_slices = _source_canvas_slices(case.canvas_offset, case.source_shape)
    prompt_ids = case_prompt_ids(case)
    wanted_components = {
        index for index, component in enumerate(case.components)
        if component.component_id in sheet_component_ids
    }
    flat_prompts = [
        (component_index, variant_index, prompt_set)
        for component_index, component in enumerate(case.components)
        for variant_index, prompt_set in enumerate(component.prompt_sets)
        if decode_all_prompts
        or component_index in wanted_components
        or variant_index in union_variants
    ]

    # Union bookkeeping lives on the device and mirrors GpuSheetContactTracker:
    # `sheet_ids` is its label volume (last writer wins) and drives contact pairs.
    union_gpu = {
        variant: torch.zeros(case.source_shape, dtype=torch.bool, device=device)
        for variant in union_variants
    }
    cover_gpu = {
        variant: torch.zeros(case.source_shape, dtype=torch.uint8, device=device)
        for variant in union_variants
    }
    sheet_ids_gpu = {
        variant: torch.zeros(case.source_shape, dtype=torch.int16, device=device)
        for variant in union_variants
    }
    contacts: dict[int, set[tuple[int, int]]] = {variant: set() for variant in union_variants}
    overlap_voxels: dict[int, int] = {variant: 0 for variant in union_variants}
    sheet_masks: dict[tuple[int, int], np.ndarray] = {}

    with autocast_context(device, dtype):
        _, image_tokens, image_coords, image_context = model.encode_image_context_from_image(image_tensor)

    for chunk in _chunks(flat_prompts, prompt_batch_size):
        prompt_points = torch.tensor(
            [prompt.points_zyx for _, _, prompt in chunk], device=device, dtype=torch.float32)
        prompt_labels = torch.tensor(
            [prompt.labels for _, _, prompt in chunk], device=device, dtype=torch.long)
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
            if variant_index in union_gpu:
                labels = sheet_ids_gpu[variant_index]
                if bool(mask_device.any()):
                    overlap_voxels[variant_index] += int((labels[mask_device] != 0).sum())
                    reach = torch.nn.functional.max_pool3d(
                        mask_device[None, None].float(), kernel_size=3, stride=1, padding=1,
                    )[0, 0] > 0
                    for value in torch.unique(labels[reach]).tolist():
                        if value > 0:
                            contacts[variant_index].add((int(value) - 1, component_index))
                    labels[mask_device] = component_index + 1
                union_gpu[variant_index] |= mask_device
                cover_gpu[variant_index] += mask_device.to(torch.uint8)
            if component_index in wanted_components:
                mask = np.ascontiguousarray(mask_device.cpu().numpy(), dtype=bool)
                if min_component_voxels > 0:
                    mask = filter_small_components(mask, min_component_voxels)
                sheet_masks[(component_index, variant_index)] = mask

    unions: dict[int, np.ndarray] = {}
    for variant, union in union_gpu.items():
        array = np.ascontiguousarray(union.cpu().numpy(), dtype=bool)
        if min_component_voxels > 0:
            array = filter_small_components(array, min_component_voxels)
        unions[variant] = array
    covers = {
        variant: np.ascontiguousarray(cover.cpu().numpy()) for variant, cover in cover_gpu.items()
    }
    sheet_ids = {
        variant: np.ascontiguousarray(labels.cpu().numpy())
        for variant, labels in sheet_ids_gpu.items()
    }
    return {
        "image": np.asarray(image),
        "prompt_ids": prompt_ids,
        "sheet_masks": sheet_masks,
        "unions": unions,
        "covers": covers,
        "sheet_ids": sheet_ids,
        "contacts": {variant: sorted(pairs) for variant, pairs in contacts.items()},
        "contact_overlap_voxels": overlap_voxels,
    }


def _write_sheet_selection(
    *,
    selection: dict[str, Any],
    case: SceneCaseProbe,
    rendered: dict[str, Any],
    components: np.ndarray,
    output_dir: Path,
    image_payload: np.ndarray | None,
    image_info: dict[str, Any],
    prompt_radius: int,
    min_component_voxels: int,
) -> dict[str, Any]:
    component_id = selection["component_id"]
    component_index = next(
        index for index, component in enumerate(case.components)
        if component.component_id == component_id
    )
    component = case.components[component_index]
    directory = output_dir / f"rank{selection['rank']:02d}_{_safe_token(case.case_id)}_c{component_id}"
    directory.mkdir(parents=True, exist_ok=True)

    gt_sheet = np.ascontiguousarray(np.asarray(components) == component_id)
    files: dict[str, str] = {"gt_sheet": _save_nifti(directory / "gt_sheet.nii.gz", gt_sheet.astype(np.uint8))}
    if image_payload is not None:
        files["image"] = _save_nifti(directory / "image.nii.gz", image_payload)

    offset = np.asarray(case.canvas_offset, dtype=np.int64)
    prompt_details: dict[str, Any] = {}
    for variant_index, prompt_id in enumerate(rendered["prompt_ids"]):
        mask = rendered["sheet_masks"].get((component_index, variant_index))
        if mask is None:
            continue
        labels, sizes = label_by_size(mask)
        token = _safe_token(prompt_id)
        files[f"pred_components_{prompt_id}"] = _save_nifti(
            directory / f"pred_components_{token}.nii.gz", labels)
        files[f"gt_pred_overlap_{prompt_id}"] = _save_nifti(
            directory / f"gt_pred_overlap_{token}.nii.gz", overlap_volume(gt_sheet, mask))
        prompt_set = component.prompt_sets[variant_index]
        source_points = (np.asarray(prompt_set.points_zyx, dtype=np.int64) - offset[None]).tolist()
        # Positive prompts get label 1, negative prompts label 2.
        values = [1 if int(label) > 0 else 2 for label in prompt_set.labels]
        files[f"prompts_{prompt_id}"] = _save_nifti(
            directory / f"prompts_{token}.nii.gz",
            stamp_prompt_points(case.source_shape, source_points, values, radius=prompt_radius))
        prompt_details[prompt_id] = {
            "predicted_component_count": len(sizes),
            "predicted_component_voxels": sizes,
            "predicted_voxels": int(mask.sum()),
            "dice_vs_gt_sheet": _dice(gt_sheet, mask),
            "prompt_points_zyx_source": source_points,
            "prompt_points_zyx_canvas": [list(map(int, point)) for point in prompt_set.points_zyx],
            "prompt_labels": [int(label) for label in prompt_set.labels],
        }

    metrics = {
        "kind": "worst_sheet_component_count",
        "rank": selection["rank"],
        "case_id": case.case_id,
        "component_id": component_id,
        "target_voxels": int(component.target_voxels),
        "gt_sheet_voxels": int(gt_sheet.sum()),
        "defect_score": selection["defect_score"],
        "empty_prediction": selection["empty_prediction"],
        "source_shape": list(case.source_shape),
        "canvas_offset": list(case.canvas_offset),
        "min_component_voxels": min_component_voxels,
        "eval_row": selection["eval_row"],
        "prompt_sets": prompt_details,
        "files": files,
        "image": image_info,
        "nifti_axis_order": NIFTI_AXIS_ORDER,
        "nifti_label_maps": {
            "gt_sheet": {"0": "background", "1": "gt_sheet"},
            "pred_components": {"0": "background", "1": "largest predicted component",
                                "2+": "extra components, descending size"},
            "gt_pred_overlap": {"0": "background", "1": "gt_only", "2": "pred_only", "3": "overlap"},
            "prompts": {"0": "background", "1": "positive prompt point", "2": "negative prompt point"},
        },
    }
    write_json(directory / "metrics.json", metrics)
    return {"directory": directory.name, **{k: metrics[k] for k in ("case_id", "component_id", "rank")}}


def _write_union_selection(
    *,
    selection: dict[str, Any],
    case: SceneCaseProbe,
    rendered: dict[str, Any],
    components: np.ndarray,
    target_contract: str,
    output_dir: Path,
    image_payload: np.ndarray | None,
    image_info: dict[str, Any],
    prompt_radius: int,
    min_component_voxels: int,
) -> dict[str, Any]:
    prompt_id = selection["prompt_set_id"]
    variant_index = list(rendered["prompt_ids"]).index(prompt_id)
    token = _safe_token(prompt_id)
    directory = output_dir / f"rank{selection['rank']:02d}_{_safe_token(case.case_id)}_{token}"
    directory.mkdir(parents=True, exist_ok=True)

    label = build_scene_target_label(
        components,
        eligible_component_ids=[component.component_id for component in case.components],
        target_contract=target_contract,
    )
    if min_component_voxels > 0:
        foreground = label == 1
        kept = filter_small_components(foreground, min_component_voxels)
        removed = foreground & ~kept
        if removed.any():
            label = np.where(removed, np.uint8(0), label)

    union = rendered["unions"][variant_index]
    cover = rendered["covers"][variant_index]
    sheet_ids = rendered["sheet_ids"][variant_index]
    union_labels, union_sizes = label_by_size(union)
    contact = np.zeros(case.source_shape, dtype=np.uint8)
    contact[union] = 1
    contact[cover >= 2] = 2

    # The public scorer drops ignore voxels from both masks before scoring, so a
    # prediction landing on a sub-threshold sheet is neither credited nor
    # penalized. Code it as 4 rather than hiding it inside "pred_only".
    ignore = label == 2
    eligible = label == 1
    scene_overlap = overlap_volume(eligible, union)
    scene_overlap[ignore & union] = 4
    scene_overlap[ignore & ~union] = 0

    files = {
        "gt_label": _save_nifti(directory / "gt_label.nii.gz", label),
        f"union_components_{prompt_id}": _save_nifti(
            directory / f"union_components_{token}.nii.gz", union_labels),
        f"gt_pred_overlap_{prompt_id}": _save_nifti(
            directory / f"gt_pred_overlap_{token}.nii.gz", scene_overlap),
        f"sheet_ids_{prompt_id}": _save_nifti(directory / f"sheet_ids_{token}.nii.gz", sheet_ids),
        f"sheet_contact_{prompt_id}": _save_nifti(directory / f"sheet_contact_{token}.nii.gz", contact),
    }
    if image_payload is not None:
        files["image"] = _save_nifti(directory / "image.nii.gz", image_payload)

    offset = np.asarray(case.canvas_offset, dtype=np.int64)
    points: list[list[int]] = []
    values: list[int] = []
    sheet_index_map: dict[str, dict[str, Any]] = {}
    for component_index, component in enumerate(case.components):
        prompt_set = component.prompt_sets[variant_index]
        source_points = (np.asarray(prompt_set.points_zyx, dtype=np.int64) - offset[None]).tolist()
        points.extend(source_points)
        values.extend([component_index + 1] * len(source_points))
        sheet_index_map[str(component_index + 1)] = {
            "component_id": int(component.component_id),
            "target_voxels": int(component.target_voxels),
            "prompt_points_zyx_source": source_points,
        }
    files[f"prompts_{prompt_id}"] = _save_nifti(
        directory / f"prompts_{token}.nii.gz",
        stamp_prompt_points(case.source_shape, points, values, radius=prompt_radius))

    contact_pairs = [
        {
            "sheet_ids": [first + 1, second + 1],
            "component_ids": [
                int(case.components[first].component_id),
                int(case.components[second].component_id),
            ],
        }
        for first, second in rendered["contacts"][variant_index]
    ]
    metrics = {
        "kind": "worst_union_sheet_contact",
        "rank": selection["rank"],
        "case_id": case.case_id,
        "prompt_set_id": prompt_id,
        "target_contract": target_contract,
        "sheet_count": len(case.components),
        "contact_pair_count": len(contact_pairs),
        "contact_pairs": contact_pairs,
        "contact_overlap_voxels": int(rendered["contact_overlap_voxels"][variant_index]),
        "union_component_count": len(union_sizes),
        "union_component_voxels": union_sizes,
        "union_voxels": int(union.sum()),
        "gt_eligible_voxels": int(eligible.sum()),
        # Ignore voxels are removed from both masks first, so this reproduces
        # kaggle_volumetric_dice rather than a stricter variant of it.
        "dice_vs_gt_eligible": _dice(eligible, union & ~ignore),
        "predicted_voxels_on_ignored_sheets": int((union & ignore).sum()),
        "source_shape": list(case.source_shape),
        "canvas_offset": list(case.canvas_offset),
        "min_component_voxels": min_component_voxels,
        "eval_row": selection["eval_row"],
        "sheet_index_map": sheet_index_map,
        "files": files,
        "image": image_info,
        "nifti_axis_order": NIFTI_AXIS_ORDER,
        "nifti_label_maps": {
            "gt_label": {"0": "background", "1": "eligible sheet (scored foreground)",
                         "2": "ignore (sub-threshold sheet)"},
            "union_components": {"0": "background", "1": "largest predicted component",
                                 "2+": "extra components, descending size"},
            "gt_pred_overlap": {"0": "background", "1": "gt_only", "2": "pred_only", "3": "overlap",
                                "4": "prediction on an ignored sub-threshold sheet (unscored)"},
            "sheet_ids": {"0": "background", "n": "sheet n, see sheet_index_map (last writer wins "
                                                  "where sheets overlap, matching the contact tracker)"},
            "sheet_contact": {"0": "background", "1": "union foreground",
                              "2": "two or more predicted sheets cover this voxel"},
            "prompts": {"0": "background", "n": "prompt point of sheet n, see sheet_index_map"},
        },
    }
    write_json(directory / "metrics.json", metrics)
    return {"directory": directory.name, **{k: metrics[k] for k in ("case_id", "prompt_set_id", "rank")}}


def visualize_worst_cases(
    *,
    eval_dir: str | Path,
    scene_manifest_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    checkpoint_path: str | Path | None = None,
    source_run_dir: str | Path | None = None,
    target_contract: str | None = None,
    top_n: int = 8,
    min_component_voxels: int | None = None,
    prompt_batch_size: int | None = None,
    prompt_radius: int = 2,
    image_dtype: str = "uint8",
    dedupe_union_cases: bool = True,
    decode_all_prompts: bool = True,
    device_name: str | None = None,
) -> dict[str, Any]:
    """Write NIfTI volumes for the worst sheets and worst touching-sheet unions."""

    if image_dtype not in IMAGE_DTYPES:
        raise ValueError(f"image_dtype must be one of {IMAGE_DTYPES}, got {image_dtype!r}")
    eval_dir = Path(eval_dir)
    summary_path = eval_dir / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Not a finished evaluation directory (no summary.json): {eval_dir}")
    import json as _json

    summary = _json.loads(summary_path.read_text(encoding="utf-8"))
    checkpoint_path = Path(checkpoint_path or summary["checkpoint_path"])
    source_run_dir = Path(source_run_dir or summary["source_run_dir"])
    target_contract = target_contract or summary.get("target_contract", ELIGIBLE_ONLY_CONTRACT)
    if target_contract not in TARGET_CONTRACTS:
        raise ValueError(f"target_contract must be one of {TARGET_CONTRACTS}, got {target_contract!r}")
    if prompt_batch_size is None:
        prompt_batch_size = eval_prompt_batch_size(summary)
    # Evaluations recorded before these keys existed must be told explicitly.
    if scene_manifest_path is None:
        scene_manifest_path = summary.get("scene_manifest_path")
        if not scene_manifest_path:
            raise ValueError(
                f"{summary_path} predates scene_manifest_path being recorded; pass it explicitly")
    if min_component_voxels is None:
        min_component_voxels = summary.get("min_component_voxels")
        if min_component_voxels is None:
            raise ValueError(
                f"{summary_path} predates min_component_voxels being recorded; pass it explicitly "
                "(the volumes will not reproduce the reported counts under a different filter)")
    min_component_voxels = int(min_component_voxels)
    output_dir = Path(output_dir) if output_dir is not None else eval_dir / "worst_case_viz"
    output_dir.mkdir(parents=True, exist_ok=True)

    sheet_rows = load_jsonl(eval_dir / "sheet_prompt_consistency.jsonl")
    case_rows = load_jsonl(eval_dir / "case_metrics.jsonl")
    sheet_selections = select_worst_sheets(sheet_rows, top_n=top_n)
    union_selections = select_worst_union_contacts(
        case_rows, top_n=top_n, dedupe_cases=dedupe_union_cases)
    if not sheet_selections and not union_selections:
        index = {"eval_dir": str(eval_dir), "worst_sheet_cc": [], "worst_union_contact": [],
                 "note": "no defective sheets and no touching sheets in this evaluation"}
        write_json(output_dir / "index.json", index)
        return index

    source_cfg = load_config(source_run_dir / "resolved_config.yaml")
    if device_name is not None:
        source_cfg["device"] = device_name
    patch_size = _patch_size_from_config(source_cfg)
    manifest = load_scene_probe_manifest(scene_manifest_path, patch_size=patch_size)
    source_rows = _load_source_rows(source_cfg, manifest)
    cases_by_id = {case.case_id: case for case in manifest.cases}
    device = get_device(source_cfg)
    dtype = amp_dtype(source_cfg.get("training", {}).get("amp_dtype"))
    channels_last = bool(source_cfg.get("training", {}).get("channels_last_3d", False))

    target_cfg = source_cfg.get("target_ae", {})
    target_ae, _ = load_ae_from_config(target_cfg.get("config_path"), target_cfg.get("checkpoint_path"))
    target_ae = target_ae.to(device).eval()
    model = build_p2sd(source_cfg, latent_channels=target_ae.latent_channels).to(device)
    if channels_last:
        target_ae = target_ae.to(memory_format=torch.channels_last_3d)
        model = model.to(memory_format=torch.channels_last_3d)
    load_model_state(model, checkpoint_path)
    model.eval()
    latent_codec = build_static_latent_codec(
        source_cfg.get("p2sd", {}).get("loss", {}).get("latent_normalization", {}),
        latent_channels=target_ae.latent_channels,
        device=device,
    )
    threshold = float(source_cfg.get("metrics", {}).get("threshold", 0.5))

    # One decode pass per case covers every selection that touches it.
    per_case_sheets: dict[str, list[dict[str, Any]]] = {}
    per_case_unions: dict[str, list[dict[str, Any]]] = {}
    for selection in sheet_selections:
        per_case_sheets.setdefault(selection["case_id"], []).append(selection)
    for selection in union_selections:
        per_case_unions.setdefault(selection["case_id"], []).append(selection)

    sheet_dir = output_dir / "worst_sheet_cc"
    union_dir = output_dir / "worst_union_contact"
    sheet_index: list[dict[str, Any]] = []
    union_index: list[dict[str, Any]] = []
    for case_id in sorted(set(per_case_sheets) | set(per_case_unions)):
        case = cases_by_id.get(case_id)
        if case is None:
            raise RuntimeError(f"Evaluation rows reference case {case_id!r} absent from the manifest")
        prompt_ids = case_prompt_ids(case)
        sheet_component_ids = {item["component_id"] for item in per_case_sheets.get(case_id, [])}
        union_variants = {
            list(prompt_ids).index(item["prompt_set_id"]) for item in per_case_unions.get(case_id, [])
        }
        rendered = _render_case(
            model=model,
            target_ae=target_ae,
            latent_codec=latent_codec,
            source_row=source_rows[case_id],
            case=case,
            patch_size=patch_size,
            threshold=threshold,
            min_component_voxels=min_component_voxels,
            prompt_batch_size=prompt_batch_size,
            sheet_component_ids=sheet_component_ids,
            union_variants=union_variants,
            device=device,
            dtype=dtype,
            channels_last=channels_last,
            decode_all_prompts=decode_all_prompts,
        )
        components = np.load(source_rows[case_id]["components_path"])
        image_payload, image_info = _image_for_nifti(rendered["image"], image_dtype)
        for selection in per_case_sheets.get(case_id, []):
            sheet_index.append(_write_sheet_selection(
                selection=selection, case=case, rendered=rendered, components=components,
                output_dir=sheet_dir, image_payload=image_payload, image_info=image_info,
                prompt_radius=prompt_radius, min_component_voxels=min_component_voxels))
        for selection in per_case_unions.get(case_id, []):
            union_index.append(_write_union_selection(
                selection=selection, case=case, rendered=rendered, components=components,
                target_contract=target_contract, output_dir=union_dir, image_payload=image_payload,
                image_info=image_info, prompt_radius=prompt_radius,
                min_component_voxels=min_component_voxels))
        print(f"[worst_viz] {case_id}: {len(sheet_component_ids)} sheet(s), "
              f"{len(union_variants)} union(s)", flush=True)

    index = {
        "eval_dir": str(eval_dir),
        "checkpoint_path": str(checkpoint_path),
        "source_run_dir": str(source_run_dir),
        "scene_manifest_path": str(scene_manifest_path),
        "target_contract": target_contract,
        "min_component_voxels": min_component_voxels,
        "prompt_batch_size": int(prompt_batch_size),
        "decode_all_prompts": bool(decode_all_prompts),
        "top_n": int(top_n),
        "image_dtype": image_dtype,
        "prompt_radius": int(prompt_radius),
        "nifti_axis_order": NIFTI_AXIS_ORDER,
        "worst_sheet_cc": sorted(sheet_index, key=lambda item: item["rank"]),
        "worst_union_contact": sorted(union_index, key=lambda item: item["rank"]),
        "worst_sheet_cc_ranking_key": SHEET_COUNT_MAX_KEY,
        "worst_union_contact_ranking_key": UNION_OVERLAP_KEY,
    }
    write_json(output_dir / "index.json", index)
    return index


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval_dir", required=True,
                        help="A finished evaluate_p2sd_case_scenes output directory.")
    parser.add_argument("--scene_manifest_path", default=None,
                        help="Defaults to the manifest the evaluation recorded.")
    parser.add_argument("--output_dir", default=None,
                        help="Defaults to <eval_dir>/worst_case_viz.")
    parser.add_argument("--checkpoint_path", default=None, help="Defaults to the evaluated checkpoint.")
    parser.add_argument("--source_run_dir", default=None, help="Defaults to the evaluated run dir.")
    parser.add_argument("--target_contract", choices=TARGET_CONTRACTS, default=None)
    parser.add_argument("--top_n", type=int, default=8)
    parser.add_argument("--min_component_voxels", type=int, default=None,
                        help="Defaults to the value the evaluation recorded. Overriding it stops "
                             "the volumes reproducing the reported counts.")
    parser.add_argument("--prompt_batch_size", type=int, default=None,
                        help="Defaults to the per-shard batch size the evaluation used. Changing it "
                             "repacks the decode batches and perturbs voxel-level numbers.")
    parser.add_argument("--skip_unused_prompts", action="store_true",
                        help="Decode only the selected prompts. Faster, but only the integer "
                             "component counts still reproduce the evaluation exactly.")
    parser.add_argument("--prompt_radius", type=int, default=2,
                        help="Prompt points are stamped as spheres of this radius so they are visible.")
    parser.add_argument("--image_dtype", choices=IMAGE_DTYPES, default="uint8",
                        help="uint8 is lossless for this 0..255 dataset and ~8x smaller than float32; "
                             "none skips the image volume.")
    parser.add_argument("--keep_all_union_prompts", action="store_true",
                        help="Rank union rows per (case, prompt) instead of one row per case.")
    parser.add_argument("--device", default=None)
    args = parser.parse_args(argv)
    index = visualize_worst_cases(
        eval_dir=args.eval_dir,
        scene_manifest_path=args.scene_manifest_path,
        output_dir=args.output_dir,
        checkpoint_path=args.checkpoint_path,
        source_run_dir=args.source_run_dir,
        target_contract=args.target_contract,
        top_n=args.top_n,
        min_component_voxels=args.min_component_voxels,
        prompt_batch_size=args.prompt_batch_size,
        prompt_radius=args.prompt_radius,
        image_dtype=args.image_dtype,
        dedupe_union_cases=not args.keep_all_union_prompts,
        decode_all_prompts=not args.skip_unused_prompts,
        device_name=args.device,
    )
    print(index, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
