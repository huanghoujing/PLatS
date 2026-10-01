"""Fixed evaluation probes for reproducible sheet and prompt diagnostics."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from vesuvius_p2sd.data.dataset import (
    ComponentStat,
    build_component_stats,
    crop_array,
    crop_start_around,
    load_jsonl,
    sample_component_center,
)


@dataclass(frozen=True)
class ProbePromptSet:
    prompt_set_id: str
    points_zyx: tuple[tuple[int, int, int], ...]
    labels: tuple[int, ...]


@dataclass(frozen=True)
class FixedProbeCase:
    probe_id: str
    case_id: str
    component_id: int
    crop_start: tuple[int, int, int]
    target_voxels: int
    prompt_sets: tuple[ProbePromptSet, ...]


class FixedProbeDataset(Dataset):
    """Load explicit crop/component/prompt probes without stochastic sampling."""

    def __init__(
        self,
        *,
        dataset_root: str | Path,
        manifest_path: str | Path,
        patch_size: tuple[int, int, int],
        split: str = "val",
    ) -> None:
        self.dataset_root = Path(dataset_root)
        self.manifest_path = Path(manifest_path)
        self.patch_size = tuple(int(value) for value in patch_size)
        self.split = str(split)
        self.cases = load_fixed_probe_manifest(self.manifest_path, patch_size=self.patch_size)
        source_rows = load_jsonl(self.dataset_root / f"manifest_{self.split}.jsonl")
        self.rows_by_case_id = {str(row["case_id"]): row for row in source_rows}
        missing = sorted({case.case_id for case in self.cases} - set(self.rows_by_case_id))
        if missing:
            raise ValueError(
                f"Probe manifest references {len(missing)} missing {self.split} cases: {missing[:3]}")
        self.entries = [
            (case, prompt_set)
            for case in self.cases
            for prompt_set in case.prompt_sets
        ]
        if not self.entries:
            raise ValueError(f"Probe manifest contains no prompt sets: {self.manifest_path}")
        self._case_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, index: int) -> dict[str, Any]:
        probe_case, prompt_set = self.entries[int(index)]
        image, components = self._load_case(probe_case.case_id)
        start = np.asarray(probe_case.crop_start, dtype=np.int64)
        image_crop = np.array(
            crop_array(image, start, self.patch_size),
            dtype=np.float32,
            copy=True,
        )
        component_crop = np.array(
            crop_array(components, start, self.patch_size),
            dtype=np.int32,
            copy=True,
        )
        mask = component_crop == int(probe_case.component_id)
        target_voxels = int(mask.sum())
        if target_voxels != int(probe_case.target_voxels):
            raise ValueError(
                f"Probe target changed for {probe_case.probe_id}: "
                f"manifest={probe_case.target_voxels}, current={target_voxels}")
        points = np.asarray(prompt_set.points_zyx, dtype=np.float32)
        labels = np.asarray(prompt_set.labels, dtype=np.int64)
        _validate_prompt_set(mask, points, labels, probe_case.probe_id, prompt_set.prompt_set_id)
        positive = np.flatnonzero(labels > 0)
        prompt_zyx = None if positive.size == 0 else tuple(
            int(round(float(value))) for value in points[int(positive[0])])
        return {
            "image": torch.from_numpy(image_crop[None]),
            "mask": torch.from_numpy(mask.astype(np.float32, copy=False)[None]),
            "component_label": torch.from_numpy(component_crop),
            "prompt_points": torch.from_numpy(points),
            "prompt_labels": torch.from_numpy(labels),
            "prompt_zyx": prompt_zyx,
            "case_id": probe_case.case_id,
            "component_id": int(probe_case.component_id),
            "crop_start": probe_case.crop_start,
            "probe_id": probe_case.probe_id,
            "prompt_set_id": prompt_set.prompt_set_id,
            "target_voxels": target_voxels,
        }

    def _load_case(self, case_id: str) -> tuple[np.ndarray, np.ndarray]:
        cached = self._case_cache.get(case_id)
        if cached is not None:
            return cached
        row = self.rows_by_case_id[case_id]
        image = np.load(row["image_path"], mmap_mode="r")
        components = np.load(row["components_path"], mmap_mode="r")
        self._case_cache[case_id] = (image, components)
        return image, components


def load_fixed_probe_manifest(
    path: str | Path,
    *,
    patch_size: tuple[int, int, int] | None = None,
) -> tuple[FixedProbeCase, ...]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"Probe manifest must be a JSON object: {path}")
    version = int(payload.get("version", 0))
    if version != 1:
        raise ValueError(f"Unsupported probe manifest version {version}: {path}")
    manifest_patch = payload.get("patch_size")
    if patch_size is not None and manifest_patch is not None:
        expected = tuple(int(value) for value in patch_size)
        observed = tuple(int(value) for value in manifest_patch)
        if observed != expected:
            raise ValueError(
                f"Probe manifest patch size {observed} does not match configured patch size {expected}")
    raw_cases = payload.get("cases", [])
    if not isinstance(raw_cases, list):
        raise ValueError(f"Probe manifest cases must be a list: {path}")
    cases = tuple(_parse_probe_case(item, index=index) for index, item in enumerate(raw_cases))
    ids = [case.probe_id for case in cases]
    if len(ids) != len(set(ids)):
        raise ValueError(f"Probe ids must be unique: {path}")
    return cases


def build_probe_manifest(
    *,
    dataset_root: str | Path,
    split: str,
    patch_size: tuple[int, int, int],
    min_target_voxels: int,
    num_sheets: int,
    prompts_per_sheet: int,
    seed: int,
    positive_points_per_prompt: int = 1,
) -> dict[str, Any]:
    """Build a deterministic, editable fixed-probe manifest from real data."""

    dataset_root = Path(dataset_root)
    rows = load_jsonl(dataset_root / f"manifest_{split}.jsonl")
    if not rows:
        raise ValueError(f"No rows in {dataset_root / f'manifest_{split}.jsonl'}")
    patch_size = tuple(int(value) for value in patch_size)
    num_sheets = int(num_sheets)
    prompts_per_sheet = int(prompts_per_sheet)
    positive_points_per_prompt = int(positive_points_per_prompt)
    if num_sheets <= 0:
        raise ValueError(f"num_sheets must be positive, got {num_sheets}")
    if prompts_per_sheet <= 0:
        raise ValueError(f"prompts_per_sheet must be positive, got {prompts_per_sheet}")
    if positive_points_per_prompt <= 0:
        raise ValueError(
            "positive_points_per_prompt must be positive, "
            f"got {positive_points_per_prompt}")

    rng = np.random.default_rng(int(seed))
    selected: list[dict[str, Any]] = []
    selected_keys: set[tuple[str, int, tuple[int, int, int]]] = set()
    stats_by_case_id: dict[str, list[ComponentStat]] = {}

    # One pass gives each source case at most one probe. Later passes create
    # distinct component/crop targets when the requested suite is larger than
    # the validation-case count.
    required_passes = (num_sheets + len(rows) - 1) // len(rows)
    max_passes = max(8, 8 * required_passes)
    for _ in range(max_passes):
        if len(selected) >= num_sheets:
            break
        selected_before_pass = len(selected)
        for row_index in rng.permutation(len(rows)):
            if len(selected) >= num_sheets:
                break
            row = rows[int(row_index)]
            case_id = str(row["case_id"])
            components = np.load(row["components_path"], mmap_mode="r")
            stats = stats_by_case_id.get(case_id)
            if stats is None:
                stats = [
                    item for item in build_component_stats(components)
                    if item.count >= int(min_target_voxels)
                ]
                stats_by_case_id[case_id] = stats
            chosen = _sample_unique_probe_crop(
                case_id=case_id,
                components=components,
                stats=stats,
                patch_size=patch_size,
                min_target_voxels=int(min_target_voxels),
                selected_keys=selected_keys,
                rng=rng,
            )
            if chosen is None:
                continue
            component_id, start, target = chosen
            total_prompt_points = prompts_per_sheet * positive_points_per_prompt
            prompts = _spread_prompt_points(target, count=total_prompt_points, rng=rng)
            if prompts.shape[0] < total_prompt_points:
                continue
            prompt_sets = []
            for prompt_index in range(prompts_per_sheet):
                start_index = prompt_index * positive_points_per_prompt
                end_index = start_index + positive_points_per_prompt
                prompt_sets.append({
                    "id": f"p{prompt_index:02d}",
                    "points_zyx": [
                        point.astype(int).tolist()
                        for point in prompts[start_index:end_index]
                    ],
                    "labels": [1] * positive_points_per_prompt,
                })
            start_tuple = tuple(int(value) for value in start)
            selected_keys.add((case_id, int(component_id), start_tuple))
            selected.append({
                "id": f"{case_id}_c{int(component_id)}_{len(selected):03d}",
                "case_id": case_id,
                "component_id": int(component_id),
                "crop_start": list(start_tuple),
                "target_voxels": int(target.sum()),
                "prompt_sets": prompt_sets,
            })
        if len(selected) == selected_before_pass:
            break
    if len(selected) != num_sheets:
        raise RuntimeError(
            f"Could only create {len(selected)} of {num_sheets} probes with "
            f"min_target_voxels={min_target_voxels}")
    return {
        "version": 1,
        "dataset_root": str(dataset_root),
        "split": str(split),
        "patch_size": list(patch_size),
        "min_target_voxels": int(min_target_voxels),
        "prompts_per_sheet": prompts_per_sheet,
        "positive_points_per_prompt": positive_points_per_prompt,
        "selected_source_case_count": len({str(case["case_id"]) for case in selected}),
        "seed": int(seed),
        "cases": selected,
    }


def _sample_unique_probe_crop(
    *,
    case_id: str,
    components: np.ndarray,
    stats: list[ComponentStat],
    patch_size: tuple[int, int, int],
    min_target_voxels: int,
    selected_keys: set[tuple[str, int, tuple[int, int, int]]],
    rng: np.random.Generator,
) -> tuple[int, np.ndarray, np.ndarray] | None:
    """Sample one eligible crop that is not already in the manifest."""

    if not stats:
        return None
    for component_index in rng.permutation(len(stats)):
        stat = stats[int(component_index)]
        for _ in range(64):
            center = sample_component_center(components, stat, rng)
            start = crop_start_around(center, components.shape, patch_size, rng)
            start_tuple = tuple(int(value) for value in start)
            key = (case_id, int(stat.component_id), start_tuple)
            if key in selected_keys:
                continue
            crop = crop_array(components, start, patch_size)
            target = crop == stat.component_id
            if int(target.sum()) >= min_target_voxels:
                return int(stat.component_id), start, target
    return None


def write_probe_manifest(path: str | Path, payload: dict[str, Any]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")
    return path


def _parse_probe_case(item: Any, *, index: int) -> FixedProbeCase:
    if not isinstance(item, dict):
        raise ValueError(f"Probe case {index} must be an object")
    probe_id = str(item.get("id", ""))
    case_id = str(item.get("case_id", ""))
    if not probe_id or not case_id:
        raise ValueError(f"Probe case {index} needs non-empty id and case_id")
    start = tuple(int(value) for value in item.get("crop_start", []))
    if len(start) != 3:
        raise ValueError(f"Probe {probe_id} crop_start must have three coordinates")
    target_voxels = int(item.get("target_voxels", -1))
    if target_voxels < 0:
        raise ValueError(f"Probe {probe_id} needs non-negative target_voxels")
    raw_prompt_sets = item.get("prompt_sets", [])
    if not isinstance(raw_prompt_sets, list) or not raw_prompt_sets:
        raise ValueError(f"Probe {probe_id} needs at least one prompt_set")
    prompt_sets = tuple(_parse_prompt_set(raw, probe_id=probe_id, index=i)
                        for i, raw in enumerate(raw_prompt_sets))
    ids = [prompt.prompt_set_id for prompt in prompt_sets]
    if len(ids) != len(set(ids)):
        raise ValueError(f"Prompt-set ids must be unique within probe {probe_id}")
    return FixedProbeCase(
        probe_id=probe_id,
        case_id=case_id,
        component_id=int(item["component_id"]),
        crop_start=start,
        target_voxels=target_voxels,
        prompt_sets=prompt_sets,
    )


def _parse_prompt_set(item: Any, *, probe_id: str, index: int) -> ProbePromptSet:
    if not isinstance(item, dict):
        raise ValueError(f"Prompt set {index} in {probe_id} must be an object")
    prompt_set_id = str(item.get("id", f"p{index:02d}"))
    raw_points = item.get("points_zyx", [])
    points = tuple(tuple(int(value) for value in point) for point in raw_points)
    labels = tuple(int(value) for value in item.get("labels", []))
    if not points or any(len(point) != 3 for point in points):
        raise ValueError(f"Prompt set {prompt_set_id} in {probe_id} needs [N, 3] points")
    if len(points) != len(labels):
        raise ValueError(f"Prompt set {prompt_set_id} in {probe_id} has mismatched labels")
    return ProbePromptSet(prompt_set_id=prompt_set_id, points_zyx=points, labels=labels)


def _validate_prompt_set(
    target: np.ndarray,
    points: np.ndarray,
    labels: np.ndarray,
    probe_id: str,
    prompt_set_id: str,
) -> None:
    if points.ndim != 2 or points.shape[1] != 3 or labels.shape != (points.shape[0],):
        raise ValueError(f"Invalid prompt tensors for {probe_id}/{prompt_set_id}")
    if not np.any(labels > 0):
        raise ValueError(f"Probe {probe_id}/{prompt_set_id} has no positive prompt")
    shape = np.asarray(target.shape)
    for point, label in zip(points, labels):
        coord = np.asarray(np.rint(point), dtype=np.int64)
        if np.any(coord < 0) or np.any(coord >= shape):
            raise ValueError(f"Prompt outside crop for {probe_id}/{prompt_set_id}: {coord.tolist()}")
        inside = bool(target[tuple(coord)])
        if int(label) > 0 and not inside:
            raise ValueError(f"Positive prompt outside target for {probe_id}/{prompt_set_id}")
        if int(label) <= 0 and inside:
            raise ValueError(f"Negative prompt inside target for {probe_id}/{prompt_set_id}")


def _spread_prompt_points(
    target: np.ndarray,
    *,
    count: int,
    rng: np.random.Generator,
) -> np.ndarray:
    coords = np.argwhere(target)
    if coords.shape[0] == 0 or count <= 0:
        return np.zeros((0, 3), dtype=np.int64)
    if coords.shape[0] > 8192:
        keep = rng.choice(coords.shape[0], size=8192, replace=False)
        coords = coords[keep]
    center = coords.mean(axis=0, keepdims=True)
    first = int(np.square(coords - center).sum(axis=1).argmin())
    selected = [first]
    min_distance = np.square(coords - coords[first]).sum(axis=1)
    while len(selected) < min(int(count), coords.shape[0]):
        next_index = int(min_distance.argmax())
        selected.append(next_index)
        min_distance = np.minimum(min_distance, np.square(coords - coords[next_index]).sum(axis=1))
    return coords[np.asarray(selected, dtype=np.int64)].astype(np.int64, copy=False)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Create a fixed P2SD probe manifest.")
    parser.add_argument("--dataset_root", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--patch_size", nargs=3, type=int, required=True)
    parser.add_argument("--min_target_voxels", type=int, required=True)
    parser.add_argument("--num_sheets", type=int, default=8)
    parser.add_argument("--prompts_per_sheet", type=int, default=4)
    parser.add_argument(
        "--positive_points_per_prompt",
        type=int,
        default=1,
        help="Positive points in each prompt set; prompt sets remain distinct variants.",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    payload = build_probe_manifest(
        dataset_root=args.dataset_root,
        split=args.split,
        patch_size=tuple(args.patch_size),
        min_target_voxels=args.min_target_voxels,
        num_sheets=args.num_sheets,
        prompts_per_sheet=args.prompts_per_sheet,
        positive_points_per_prompt=args.positive_points_per_prompt,
        seed=args.seed,
    )
    output = write_probe_manifest(args.output_path, payload)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
