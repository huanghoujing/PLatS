"""Deterministic all-sheet scene probes for case-level P2SD evaluation."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from vesuvius_p2sd.data.dataset import ComponentStat, build_component_stats, load_jsonl
from vesuvius_p2sd.data.probes import _spread_prompt_points


@dataclass(frozen=True)
class ScenePromptSet:
    prompt_set_id: str
    points_zyx: tuple[tuple[int, int, int], ...]
    labels: tuple[int, ...]


@dataclass(frozen=True)
class SceneComponentProbe:
    component_id: int
    target_voxels: int
    prompt_sets: tuple[ScenePromptSet, ...]


@dataclass(frozen=True)
class SceneCaseProbe:
    case_id: str
    source_shape: tuple[int, int, int]
    canvas_offset: tuple[int, int, int]
    components: tuple[SceneComponentProbe, ...]


@dataclass(frozen=True)
class SceneProbeManifest:
    dataset_root: str
    split: str
    patch_size: tuple[int, int, int]
    min_target_voxels: int
    positive_points_per_prompt: int
    prompt_set_ids: tuple[str, ...]
    cases: tuple[SceneCaseProbe, ...]


def build_scene_probe_manifest(
    *,
    dataset_root: str | Path,
    split: str,
    patch_size: tuple[int, int, int],
    min_target_voxels: int,
    prompts_per_sheet: int,
    positive_points_per_prompt: int,
    seed: int,
) -> dict[str, Any]:
    """Create one fixed prompt suite for every eligible component in each case."""

    dataset_root = Path(dataset_root)
    patch_size = tuple(int(value) for value in patch_size)
    min_target_voxels = int(min_target_voxels)
    prompts_per_sheet = int(prompts_per_sheet)
    positive_points_per_prompt = int(positive_points_per_prompt)
    if len(patch_size) != 3 or any(value <= 0 for value in patch_size):
        raise ValueError(f"patch_size must be three positive dimensions, got {patch_size}")
    if min_target_voxels <= 0:
        raise ValueError(f"min_target_voxels must be positive, got {min_target_voxels}")
    if prompts_per_sheet <= 0 or positive_points_per_prompt <= 0:
        raise ValueError("prompts_per_sheet and positive_points_per_prompt must be positive")

    rows = load_jsonl(dataset_root / f"manifest_{split}.jsonl")
    cases = []
    for case_index, row in enumerate(rows):
        case_id = str(row["case_id"])
        components = np.load(row["components_path"], mmap_mode="r")
        shape = tuple(int(value) for value in components.shape)
        if len(shape) != 3 or any(size > patch for size, patch in zip(shape, patch_size)):
            raise ValueError(
                f"Scene probe {case_id} has shape {shape}; only source volumes fitting "
                f"inside the evaluation canvas {patch_size} are supported")
        offset = tuple((patch - size) // 2 for size, patch in zip(shape, patch_size))
        probes = []
        for stat in build_component_stats(components):
            if stat.count < min_target_voxels:
                continue
            probes.append(_build_component_probe(
                components=components,
                stat=stat,
                case_index=case_index,
                canvas_offset=offset,
                prompts_per_sheet=prompts_per_sheet,
                positive_points_per_prompt=positive_points_per_prompt,
                seed=seed,
            ))
        if not probes:
            raise ValueError(
                f"Scene probe case {case_id} has no components meeting "
                f"min_target_voxels={min_target_voxels}")
        cases.append({
            "case_id": case_id,
            "source_shape": list(shape),
            "canvas_offset": list(offset),
            "components": probes,
        })

    return {
        "version": 1,
        "kind": "p2sd_case_scene",
        "dataset_root": str(dataset_root),
        "split": str(split),
        "patch_size": list(patch_size),
        "min_target_voxels": min_target_voxels,
        "prompts_per_sheet": prompts_per_sheet,
        "positive_points_per_prompt": positive_points_per_prompt,
        "seed": int(seed),
        "cases": cases,
    }


def _build_component_probe(
    *,
    components: np.ndarray,
    stat: ComponentStat,
    case_index: int,
    canvas_offset: tuple[int, int, int],
    prompts_per_sheet: int,
    positive_points_per_prompt: int,
    seed: int,
) -> dict[str, Any]:
    target = np.asarray(components == stat.component_id)
    prompt_rng = np.random.default_rng(np.random.SeedSequence([
        int(seed),
        int(case_index),
        int(stat.component_id),
    ]))
    total_points = prompts_per_sheet * positive_points_per_prompt
    native_points = _spread_prompt_points(target, count=total_points, rng=prompt_rng)
    if native_points.shape[0] != total_points:
        raise RuntimeError(
            f"Component {stat.component_id} has only {native_points.shape[0]} prompt points; "
            f"requested {total_points}")
    offset = np.asarray(canvas_offset, dtype=np.int64)
    canvas_points = native_points + offset[None]
    prompt_sets = []
    for prompt_index in range(prompts_per_sheet):
        start = prompt_index * positive_points_per_prompt
        stop = start + positive_points_per_prompt
        prompt_sets.append({
            "id": f"p{prompt_index:02d}",
            "points_zyx": canvas_points[start:stop].astype(int).tolist(),
            "labels": [1] * positive_points_per_prompt,
        })
    return {
        "component_id": int(stat.component_id),
        "target_voxels": int(stat.count),
        "prompt_sets": prompt_sets,
    }


def load_scene_probe_manifest(
    path: str | Path,
    *,
    patch_size: tuple[int, int, int] | None = None,
) -> SceneProbeManifest:
    path = Path(path)
    with path.open("r", encoding="utf-8") as file:
        raw = json.load(file)
    if not isinstance(raw, dict) or raw.get("version") != 1 or raw.get("kind") != "p2sd_case_scene":
        raise ValueError(f"Unsupported scene probe manifest: {path}")
    manifest_patch = tuple(int(value) for value in raw.get("patch_size", []))
    if len(manifest_patch) != 3:
        raise ValueError(f"Scene manifest patch_size must be [D,H,W]: {path}")
    if patch_size is not None and tuple(int(value) for value in patch_size) != manifest_patch:
        raise ValueError(
            f"Scene manifest patch size {manifest_patch} does not match requested {patch_size}")
    raw_cases = raw.get("cases", [])
    if not isinstance(raw_cases, list) or not raw_cases:
        raise ValueError(f"Scene manifest has no cases: {path}")
    cases = tuple(_parse_scene_case(item, index=index, patch_size=manifest_patch)
                  for index, item in enumerate(raw_cases))
    case_ids = [case.case_id for case in cases]
    if len(case_ids) != len(set(case_ids)):
        raise ValueError(f"Scene manifest has duplicate case ids: {path}")
    prompt_ids = tuple(prompt.prompt_set_id for prompt in cases[0].components[0].prompt_sets)
    if not prompt_ids:
        raise ValueError(f"Scene manifest has no prompt ids: {path}")
    for case in cases:
        for component in case.components:
            if tuple(prompt.prompt_set_id for prompt in component.prompt_sets) != prompt_ids:
                raise ValueError(
                    "Every scene component must use the same ordered prompt-set ids; "
                    f"case={case.case_id} component={component.component_id}")
    return SceneProbeManifest(
        dataset_root=str(raw.get("dataset_root", "")),
        split=str(raw.get("split", "val")),
        patch_size=manifest_patch,
        min_target_voxels=int(raw.get("min_target_voxels", 0)),
        positive_points_per_prompt=int(raw.get("positive_points_per_prompt", 0)),
        prompt_set_ids=prompt_ids,
        cases=cases,
    )


def _parse_scene_case(item: Any, *, index: int, patch_size: tuple[int, int, int]) -> SceneCaseProbe:
    if not isinstance(item, dict):
        raise ValueError(f"Scene case {index} must be an object")
    case_id = str(item.get("case_id", ""))
    shape = tuple(int(value) for value in item.get("source_shape", []))
    offset = tuple(int(value) for value in item.get("canvas_offset", []))
    if not case_id or len(shape) != 3 or len(offset) != 3:
        raise ValueError(f"Scene case {index} needs case_id, source_shape, and canvas_offset")
    if any(size <= 0 or size > patch for size, patch in zip(shape, patch_size)):
        raise ValueError(f"Scene case {case_id} has invalid source shape {shape}")
    if any(value < 0 for value in offset) or any(
        value + size > patch for value, size, patch in zip(offset, shape, patch_size)
    ):
        raise ValueError(f"Scene case {case_id} has invalid canvas offset {offset}")
    raw_components = item.get("components", [])
    if not isinstance(raw_components, list) or not raw_components:
        raise ValueError(f"Scene case {case_id} has no component probes")
    components = tuple(
        _parse_scene_component(component, case_id=case_id, index=component_index, patch_size=patch_size)
        for component_index, component in enumerate(raw_components)
    )
    ids = [component.component_id for component in components]
    if len(ids) != len(set(ids)):
        raise ValueError(f"Scene case {case_id} has duplicate component ids")
    return SceneCaseProbe(
        case_id=case_id,
        source_shape=shape,
        canvas_offset=offset,
        components=components,
    )


def _parse_scene_component(
    item: Any,
    *,
    case_id: str,
    index: int,
    patch_size: tuple[int, int, int],
) -> SceneComponentProbe:
    if not isinstance(item, dict):
        raise ValueError(f"Scene component {case_id}/{index} must be an object")
    component_id = int(item.get("component_id", 0))
    target_voxels = int(item.get("target_voxels", 0))
    if component_id <= 0 or target_voxels <= 0:
        raise ValueError(f"Scene component {case_id}/{index} has invalid id or target size")
    raw_prompts = item.get("prompt_sets", [])
    if not isinstance(raw_prompts, list) or not raw_prompts:
        raise ValueError(f"Scene component {case_id}/{component_id} has no prompt sets")
    prompts = tuple(
        _parse_scene_prompt(prompt, case_id=case_id, component_id=component_id, index=prompt_index, patch_size=patch_size)
        for prompt_index, prompt in enumerate(raw_prompts)
    )
    prompt_ids = [prompt.prompt_set_id for prompt in prompts]
    if len(prompt_ids) != len(set(prompt_ids)):
        raise ValueError(f"Scene component {case_id}/{component_id} has duplicate prompt ids")
    return SceneComponentProbe(
        component_id=component_id,
        target_voxels=target_voxels,
        prompt_sets=prompts,
    )


def _parse_scene_prompt(
    item: Any,
    *,
    case_id: str,
    component_id: int,
    index: int,
    patch_size: tuple[int, int, int],
) -> ScenePromptSet:
    if not isinstance(item, dict):
        raise ValueError(f"Scene prompt {case_id}/{component_id}/{index} must be an object")
    prompt_set_id = str(item.get("id", f"p{index:02d}"))
    points = tuple(tuple(int(value) for value in point) for point in item.get("points_zyx", []))
    labels = tuple(int(value) for value in item.get("labels", []))
    if not prompt_set_id or not points or len(points) != len(labels):
        raise ValueError(f"Scene prompt {case_id}/{component_id}/{index} is invalid")
    for point in points:
        if len(point) != 3 or any(value < 0 or value >= size for value, size in zip(point, patch_size)):
            raise ValueError(f"Scene prompt {case_id}/{component_id}/{prompt_set_id} is outside the canvas")
    if not any(label > 0 for label in labels):
        raise ValueError(f"Scene prompt {case_id}/{component_id}/{prompt_set_id} has no positive point")
    return ScenePromptSet(prompt_set_id=prompt_set_id, points_zyx=points, labels=labels)


def write_scene_probe_manifest(path: str | Path, payload: dict[str, Any]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, sort_keys=True)
        file.write("\n")
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset_root", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--patch_size", nargs=3, type=int, required=True)
    parser.add_argument("--min_target_voxels", type=int, required=True)
    parser.add_argument("--prompts_per_sheet", type=int, default=4)
    parser.add_argument("--positive_points_per_prompt", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    payload = build_scene_probe_manifest(
        dataset_root=args.dataset_root,
        split=args.split,
        patch_size=tuple(args.patch_size),
        min_target_voxels=args.min_target_voxels,
        prompts_per_sheet=args.prompts_per_sheet,
        positive_points_per_prompt=args.positive_points_per_prompt,
        seed=args.seed,
    )
    output = write_scene_probe_manifest(args.output_path, payload)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
