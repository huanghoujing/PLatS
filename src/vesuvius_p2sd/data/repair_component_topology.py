"""Repair GT sheet topology to official 6-connected watertightness.

The Kaggle TopoScore sees foreground under the V-construction (6-connected),
so a B-spline surface rasterized with diagonal-only contacts is full of
pinhole tunnels even when it looks watertight under 26-connectivity. This
tool audits every component of a converted dataset and repairs defective
sheets with guarded escalating morphological closing:

- per-sheet, never on the label union;
- a fusion guard forbids adding any voxel that overlaps or 26-touches a
  different component (repairing single-sheet topology must not create
  inter-sheet contacts, the one unamendable error class);
- escalating structuring elements (3^3, then 5^3) until Betti == (1, 0, 0);
- sheets that cannot reach (1, 0, 0) under the guard keep their best guarded
  repair and are flagged, never silently dropped.

Output is a new versioned dataset root: unchanged arrays are hardlinked,
repaired ``components.npy`` rewritten, manifests and per-case meta updated
with provenance, and a full per-sheet audit written to
``topology_repair_report.json``.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
from scipy import ndimage

from vesuvius_p2sd.eval.topology import betti_numbers

SE_SIZES = (3, 5)
GUARD_STRUCTURE = np.ones((3, 3, 3), dtype=bool)


def repair_sheet_mask(
    mask: np.ndarray,
    forbidden: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Guarded escalating closing of one sheet mask.

    ``forbidden`` marks voxels that must never be claimed (other components
    and their 26-neighborhoods). Returns the repaired mask and an audit row.
    """
    before = betti_numbers(mask, construction="V")
    info: dict[str, Any] = {
        "betti_before": list(before),
        "betti_after": list(before),
        "method": "none",
        "added_voxels": 0,
        "repaired": before == (1, 0, 0),
    }
    if before == (1, 0, 0):
        return mask, info
    best_mask = mask
    best_betti = before
    for se_size in SE_SIZES:
        structure = np.ones((se_size,) * 3, dtype=bool)
        closed = ndimage.binary_closing(best_mask, structure=structure)
        candidate = best_mask | (closed & ~best_mask & ~forbidden)
        betti = betti_numbers(candidate, construction="V")
        if sum(betti) < sum(best_betti) or (betti == (1, 0, 0)):
            best_mask = candidate
            best_betti = betti
            info["method"] = f"guarded_close{se_size}"
        if best_betti == (1, 0, 0):
            break
    info["betti_after"] = list(best_betti)
    info["added_voxels"] = int((best_mask & ~mask).sum())
    info["repaired"] = best_betti == (1, 0, 0)
    return best_mask, info


def repair_case_components(components: np.ndarray) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Audit and repair every component of one case, sequentially guarded.

    Later components see voxels added for earlier ones as forbidden, so
    repairs can never introduce inter-sheet contact among themselves.
    """
    components = components.copy()
    reports: list[dict[str, Any]] = []
    component_ids = [int(v) for v in np.unique(components) if v > 0]
    pad = max(SE_SIZES) // 2 + 1
    for component_id in component_ids:
        mask_full = components == component_id
        bounds = ndimage.find_objects(mask_full.astype(np.int8), max_label=1)[0]
        window = tuple(
            slice(max(s.start - pad, 0), min(s.stop + pad, dim))
            for s, dim in zip(bounds, components.shape)
        )
        local = components[window]
        mask = local == component_id
        others = (local > 0) & ~mask
        forbidden = ndimage.binary_dilation(others, structure=GUARD_STRUCTURE)
        repaired, info = repair_sheet_mask(mask, forbidden)
        info["component_id"] = component_id
        added = repaired & ~mask
        if added.any():
            local[added] = component_id
        reports.append(info)
    return components, reports


def process_case(
    row: dict[str, Any],
    *,
    source_root: str,
    output_root: str,
    audit_only: bool,
) -> dict[str, Any]:
    source_root_path = Path(source_root)
    output_root_path = Path(output_root)
    case_id = str(row["case_id"])
    components = np.load(source_root_path / "cases" / case_id / "components.npy")
    repaired, sheet_reports = repair_case_components(components)
    changed = any(report["added_voxels"] for report in sheet_reports)
    case_report = {
        "case_id": case_id,
        "changed": bool(changed),
        "components": sheet_reports,
    }
    if audit_only:
        return case_report

    case_dir = output_root_path / "cases" / case_id
    case_dir.mkdir(parents=True, exist_ok=True)
    source_case_dir = source_root_path / "cases" / case_id
    _hardlink(source_case_dir / "image.npy", case_dir / "image.npy")
    if changed:
        np.save(case_dir / "components.npy", repaired.astype(components.dtype))
    else:
        _hardlink(source_case_dir / "components.npy", case_dir / "components.npy")

    counts = np.bincount(repaired.reshape(-1))
    foreground_voxels = int(counts[1:].sum())
    max_component_voxels = int(counts[1:].max()) if counts.size > 1 else 0
    meta = json.loads((source_case_dir / "meta.json").read_text(encoding="utf-8"))
    meta.update({
        "components_path": str(case_dir / "components.npy"),
        "image_path": str(case_dir / "image.npy"),
        "foreground_voxels": foreground_voxels,
        "max_component_voxels": max_component_voxels,
        "topology_repair": {
            "source_components_path": str(source_case_dir / "components.npy"),
            "changed": bool(changed),
            "defective_before": sum(
                1 for r in sheet_reports if tuple(r["betti_before"]) != (1, 0, 0)),
            "unrepaired": sum(1 for r in sheet_reports if not r["repaired"]),
        },
    })
    (case_dir / "meta.json").write_text(
        json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    case_report["manifest_row"] = {
        **row,
        "components_path": meta["components_path"],
        "image_path": meta["image_path"],
        "foreground_voxels": foreground_voxels,
        "max_component_voxels": max_component_voxels,
    }
    return case_report


def _hardlink(source: Path, target: Path) -> None:
    if target.exists():
        target.unlink()
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def _worker(args: tuple) -> dict[str, Any]:
    row, source_root, output_root, audit_only = args
    return process_case(
        row,
        source_root=source_root,
        output_root=output_root,
        audit_only=audit_only,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source_root", required=True)
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--audit_only", action="store_true")
    args = parser.parse_args(argv)

    source_root = Path(args.source_root)
    output_root = Path(args.output_root)
    if not args.audit_only and output_root.exists() and any(output_root.iterdir()):
        raise SystemExit(f"Refusing to write into non-empty {output_root}")

    rows = [
        json.loads(line)
        for line in (source_root / "manifest_all.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    tasks = [(row, str(source_root), str(output_root), args.audit_only) for row in rows]
    reports: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for index, report in enumerate(pool.map(_worker, tasks, chunksize=2)):
            reports.append(report)
            if (index + 1) % 50 == 0 or index + 1 == len(tasks):
                print(f"[{index + 1}/{len(tasks)}] cases audited", flush=True)

    sheets = [r for case in reports for r in case["components"]]
    defective = [r for r in sheets if tuple(r["betti_before"]) != (1, 0, 0)]
    unrepaired = [r for r in sheets if not r["repaired"]]
    summary = {
        "source_root": str(source_root),
        "output_root": None if args.audit_only else str(output_root),
        "cases": len(reports),
        "sheets": len(sheets),
        "defective_sheets": len(defective),
        "repaired_sheets": len(defective) - len(unrepaired),
        "unrepaired_sheets": len(unrepaired),
        "added_voxels_total": int(sum(r["added_voxels"] for r in sheets)),
    }
    print(json.dumps(summary, indent=2))

    report_root = output_root if not args.audit_only else source_root
    report_root.mkdir(parents=True, exist_ok=True)
    (report_root / "topology_repair_report.json").write_text(
        json.dumps({"summary": summary, "cases": reports}, indent=2) + "\n",
        encoding="utf-8",
    )
    if args.audit_only:
        return 0

    manifest_rows = {case["case_id"]: case["manifest_row"] for case in reports}
    for split in ("all", "train", "val"):
        source_manifest = source_root / f"manifest_{split}.jsonl"
        if not source_manifest.exists():
            continue
        with (output_root / f"manifest_{split}.jsonl").open("w", encoding="utf-8") as handle:
            for line in source_manifest.read_text(encoding="utf-8").splitlines():
                case_id = str(json.loads(line)["case_id"])
                handle.write(json.dumps(manifest_rows[case_id]) + "\n")
    dataset_summary = json.loads(
        (source_root / "dataset_summary.json").read_text(encoding="utf-8"))
    dataset_summary["topology_repair"] = summary
    (output_root / "dataset_summary.json").write_text(
        json.dumps(dataset_summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if (source_root / "previews").exists():
        shutil.copytree(source_root / "previews", output_root / "previews", dirs_exist_ok=True)
    print(
        "Component stats must be regenerated: "
        f".venv/bin/python -m vesuvius_p2sd.data.cache_component_stats --dataset_root {output_root}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
