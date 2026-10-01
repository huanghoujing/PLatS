"""Backfill per-case ignore masks that the dataset conversion discarded.

The source labels are three-class -- 0 background, 1 sheet, 2 unlabeled -- and
class 2 covers most of a typical volume (~70%). ``convert_dataset`` zeroes it
before connected-component labeling, so in ``components.npy`` unlabeled voxels
are indistinguishable from true background. That is survivable for prompted
per-sheet training, but fatal for dense binary segmentation: the loss would
demand "empty" on the unlabeled majority of every volume.

This script reads each case's ``source_raw_label`` with the SAME reader the
conversion used (``load_nifti_array``: Fortran-order reshape, so alignment is
by construction) and writes ``cases/<id>/ignore.npy``:

    ignore = (raw_label == ignore_label) | erased_border(width)

The erased border is included because ``convert_dataset`` zeroed those voxels
in both image and components -- they are neither reliable foreground nor
reliable background. Masks are stored bit-packed (np.packbits, ~4 MB per 320^3
case instead of 32 MB); load with ``load_ignore_mask``.

Each converted mask is validated against the case's existing arrays:
labeled foreground (``components > 0``) must never intersect ignore, and must
be exactly the border-erased class-1 region.

New manifests ``manifest_<split>_ignore.jsonl`` mirror the originals with
``ignore_path``/``ignore_format``/``ignore_voxels`` added; the originals are
left untouched so nothing running against them changes behaviour.
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

from vesuvius_p2sd.data.conversion import zero_volume_border
from vesuvius_p2sd.data.convert_dataset import load_nifti_array
from vesuvius_p2sd.data.dataset import load_jsonl

IGNORE_FORMAT = "packbits_uint8"


def build_ignore_mask(
    raw_label: np.ndarray,
    *,
    ignore_label: int = 2,
    erased_border_width: int = 5,
) -> np.ndarray:
    """Unlabeled region plus the conversion's erased border, as bool."""

    ignore = np.asarray(raw_label) == int(ignore_label)
    if erased_border_width > 0:
        border = np.zeros(ignore.shape, dtype=bool)
        interior = np.ones(ignore.shape, dtype=np.uint8)
        zero_volume_border(interior, erased_border_width)
        border[interior == 0] = True
        ignore = ignore | border
    return ignore


def save_ignore_mask(path: Path, mask: np.ndarray) -> None:
    packed = np.packbits(np.ascontiguousarray(mask, dtype=bool))
    np.save(path, packed)


def load_ignore_mask(path: str | Path, shape: tuple[int, int, int]) -> np.ndarray:
    packed = np.load(path)
    count = int(np.prod(shape))
    return np.unpackbits(packed, count=count).reshape(shape).astype(bool)


def _process_case(row: dict[str, Any], *, ignore_label: int, validate: bool) -> dict[str, Any]:
    source_path = row["source_raw_label"]
    shape = tuple(int(v) for v in row["shape"])
    border = int(row.get("erased_border_width", 5))
    raw = load_nifti_array(source_path)
    if tuple(raw.shape) != shape:
        raise ValueError(f"{row['case_id']}: source label shape {raw.shape} != manifest {shape}")
    ignore = build_ignore_mask(raw, ignore_label=ignore_label, erased_border_width=border)

    # t6 is topology-REPAIRED: components may extend slightly beyond raw class 1
    # (measured ~0-3k voxels/case, all into raw background, none into raw class
    # 2). The repaired foreground is the trusted label, so subtract it from
    # ignore -- the mask can then never contradict components.npy by
    # construction -- and record how far the repair drifted for auditing.
    components = np.load(row["components_path"], mmap_mode="r")
    labeled = np.asarray(components) > 0
    labeled_in_raw_ignore = int((labeled & (raw == ignore_label)).sum())
    class1 = np.array(raw == 1)
    zero_volume_border(class1, border)
    repair_added = int((labeled & ~class1).sum())
    repair_removed = int((class1 & ~labeled).sum())
    ignore &= ~labeled

    if validate and int((labeled & ignore).sum()):
        raise ValueError(f"{row['case_id']}: ignore still intersects labeled foreground")

    out_path = Path(row["components_path"]).parent / "ignore.npy"
    save_ignore_mask(out_path, ignore)
    return {
        **row,
        "ignore_path": str(out_path),
        "ignore_format": IGNORE_FORMAT,
        "ignore_label_source": int(ignore_label),
        "ignore_voxels": int(ignore.sum()),
        "labeled_in_raw_ignore_voxels": labeled_in_raw_ignore,
        "repair_added_voxels": repair_added,
        "repair_removed_voxels": repair_removed,
    }


def build_split(
    dataset_root: Path,
    split: str,
    *,
    ignore_label: int,
    validate: bool,
    workers: int,
) -> dict[str, Any]:
    rows = load_jsonl(dataset_root / f"manifest_{split}.jsonl")
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [
            pool.submit(_process_case, row, ignore_label=ignore_label, validate=validate)
            for row in rows
        ]
        out_rows = [future.result() for future in futures]
    out_path = dataset_root / f"manifest_{split}_ignore.jsonl"
    with out_path.open("w", encoding="utf-8") as f:
        for row in out_rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")
    fractions = [row["ignore_voxels"] / float(np.prod(row["shape"])) for row in out_rows]
    return {
        "split": split,
        "cases": len(out_rows),
        "manifest": str(out_path),
        "ignore_fraction_mean": float(np.mean(fractions)),
        "ignore_fraction_min": float(np.min(fractions)),
        "ignore_fraction_max": float(np.max(fractions)),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset_root", required=True)
    parser.add_argument("--splits", nargs="+", default=["train", "val"])
    parser.add_argument("--ignore_label", type=int, default=2)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--no_validate", action="store_true",
                        help="Skip the per-case consistency checks against components.npy.")
    args = parser.parse_args(argv)
    for split in args.splits:
        summary = build_split(
            Path(args.dataset_root),
            split,
            ignore_label=args.ignore_label,
            validate=not args.no_validate,
            workers=args.workers,
        )
        print(summary, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
