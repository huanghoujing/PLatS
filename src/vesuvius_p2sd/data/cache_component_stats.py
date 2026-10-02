"""Precompute component stats used by patch sampling."""

from __future__ import annotations

import argparse
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from vesuvius_p2sd.data.dataset import (
    build_component_stats,
    component_stats_cache_row,
    load_jsonl,
    write_jsonl,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", required=True)
    parser.add_argument("--split", default="all", choices=["all", "train", "val"])
    parser.add_argument("--workers", type=int, default=max(1, min(8, os.cpu_count() or 1)))
    args = parser.parse_args(argv)

    dataset_root = Path(args.dataset_root)
    splits = ["train", "val"] if args.split == "all" else [args.split]
    for split in splits:
        build_split_cache(dataset_root, split=split, workers=int(args.workers))
    return 0


def build_split_cache(dataset_root: Path, *, split: str, workers: int) -> Path:
    manifest = dataset_root / f"manifest_{split}.jsonl"
    rows = load_jsonl(manifest)
    if not rows:
        raise ValueError(f"No rows in {manifest}")
    workers = max(1, int(workers))
    out_path = dataset_root / f"component_stats_{split}.jsonl"
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")

    if workers == 1:
        out_rows = [_build_one(row) for row in rows]
    else:
        out_rows_by_index: dict[int, dict] = {}
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(_build_one, row): index
                for index, row in enumerate(rows)
            }
            for future in as_completed(futures):
                out_rows_by_index[futures[future]] = future.result()
        out_rows = [out_rows_by_index[index] for index in range(len(rows))]

    write_jsonl(tmp_path, out_rows)
    tmp_path.replace(out_path)
    return out_path


def _build_one(row: dict) -> dict:
    comps = np.load(row["components_path"], mmap_mode="r")
    stats = build_component_stats(comps)
    return component_stats_cache_row(row, stats)


if __name__ == "__main__":
    raise SystemExit(main())
