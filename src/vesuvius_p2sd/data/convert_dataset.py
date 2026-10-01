"""Convert Vesuvius sheet data into the clean repo-local contract."""

from __future__ import annotations

import argparse
import gzip
import json
import shutil
import struct
from pathlib import Path
from typing import Any

import numpy as np
from scipy import ndimage

from vesuvius_p2sd.data.conversion import zero_volume_border
from vesuvius_p2sd.train.visualization import save_p2sd_overview_png


def convert_dataset(
    *,
    source_raw_dir: str | Path | None = None,
    source_preprocessed_dir: str | Path | None = None,
    output_root: str | Path,
    component_dir: str | Path | None = None,
    component_policy: str = "shape_match",
    erased_border_width: int = 5,
    ignore_label: int | None = 2,
    val_fraction: float = 0.1,
    seed: int = 42,
    limit_cases: int | None = None,
    overwrite: bool = False,
    image_dtype: str = "float16",
    max_preview_cases: int = 4,
) -> Path:
    if source_raw_dir is None and source_preprocessed_dir is None:
        raise ValueError("Provide source_raw_dir or source_preprocessed_dir")
    if source_raw_dir is not None and source_preprocessed_dir is not None:
        raise ValueError("Use only one of source_raw_dir or source_preprocessed_dir")
    source_dir = Path(source_raw_dir or source_preprocessed_dir)
    source_mode = "raw" if source_raw_dir is not None else "preprocessed"
    output = Path(output_root)
    comp_dir = Path(component_dir) if component_dir else None
    if not source_dir.exists():
        raise FileNotFoundError(source_dir)
    if output.exists() and overwrite:
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)
    cases_dir = output / "cases"
    cases_dir.mkdir(exist_ok=True)

    source_files = list_source_cases(source_dir, source_mode)
    if limit_cases is not None:
        source_files = source_files[: int(limit_cases)]
    if not source_files:
        raise ValueError(f"No source cases found in {source_dir}")

    rng = np.random.default_rng(seed)
    case_rows: list[dict[str, Any]] = []
    stats = {
        "source_mode": source_mode,
        "source_raw_dir": str(source_dir) if source_mode == "raw" else None,
        "source_preprocessed_dir": str(source_dir) if source_mode == "preprocessed" else None,
        "component_dir": str(comp_dir) if comp_dir else None,
        "component_policy": component_policy,
        "output_root": str(output),
        "erased_border_width": int(erased_border_width),
        "ignore_label": ignore_label,
        "num_cases": 0,
        "total_foreground_voxels": 0,
        "component_source_counts": {},
    }

    for index, source_path in enumerate(source_files):
        if source_mode == "raw":
            row = convert_raw_case(
                image_path=source_path,
                raw_dir=source_dir,
                cases_dir=cases_dir,
                component_dir=comp_dir,
                component_policy=component_policy,
                erased_border_width=erased_border_width,
                ignore_label=ignore_label,
                image_dtype=image_dtype,
                overwrite=overwrite,
            )
        else:
            row = convert_preprocessed_case(
                npz_path=source_path,
                cases_dir=cases_dir,
                component_dir=comp_dir,
                erased_border_width=erased_border_width,
                ignore_label=ignore_label,
                image_dtype=image_dtype,
                overwrite=overwrite,
            )
        case_rows.append(row)
        stats["num_cases"] += 1
        stats["total_foreground_voxels"] += int(row["foreground_voxels"])
        source_key = row["component_source"]
        stats["component_source_counts"][source_key] = (
            int(stats["component_source_counts"].get(source_key, 0)) + 1
        )
        if index < max_preview_cases:
            save_preview(output / "previews" / f"{row['case_id']}.png", row)

    order = rng.permutation(len(case_rows))
    val_count = max(1, int(round(len(case_rows) * val_fraction))) if len(case_rows) > 1 else 0
    val_indices = set(int(i) for i in order[:val_count])
    train_rows = [r for i, r in enumerate(case_rows) if i not in val_indices]
    val_rows = [r for i, r in enumerate(case_rows) if i in val_indices]
    if not train_rows and val_rows:
        train_rows.append(val_rows.pop())

    write_jsonl(output / "manifest_all.jsonl", case_rows)
    write_jsonl(output / "manifest_train.jsonl", train_rows)
    write_jsonl(output / "manifest_val.jsonl", val_rows)
    stats["num_train"] = len(train_rows)
    stats["num_val"] = len(val_rows)
    stats["mean_foreground_voxels"] = (
        float(stats["total_foreground_voxels"]) / max(1, stats["num_cases"]))
    (output / "dataset_summary.json").write_text(
        json.dumps(stats, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return output


def list_source_cases(source_dir: Path, source_mode: str) -> list[Path]:
    if source_mode == "raw":
        return sorted((source_dir / "imagesTr").glob("*_0000.nii.gz"))
    return sorted(source_dir.glob("*.npz"))


def convert_raw_case(
    *,
    image_path: Path,
    raw_dir: Path,
    cases_dir: Path,
    component_dir: Path | None,
    component_policy: str,
    erased_border_width: int,
    ignore_label: int | None,
    image_dtype: str,
    overwrite: bool,
) -> dict[str, Any]:
    case_id = image_path.name.removesuffix("_0000.nii.gz")
    label_path = raw_dir / "labelsTr" / f"{case_id}.nii.gz"
    if not label_path.exists():
        raise FileNotFoundError(label_path)
    case_dir = cases_dir / case_id
    image_out = case_dir / "image.npy"
    comp_out = case_dir / "components.npy"
    meta_path = case_dir / "meta.json"
    if image_out.exists() and comp_out.exists() and meta_path.exists() and not overwrite:
        return json.loads(meta_path.read_text(encoding="utf-8"))

    case_dir.mkdir(parents=True, exist_ok=True)
    image = load_nifti_array(image_path).astype(np.float32, copy=False)
    raw_label = load_nifti_array(label_path)
    if image.shape != raw_label.shape:
        raise ValueError(f"Raw image shape {image.shape} != raw label shape {raw_label.shape} for {case_id}")
    image = np.array(image, copy=True)
    zero_volume_border(image, erased_border_width)
    components, component_source = load_raw_components(
        case_id=case_id,
        raw_label=raw_label,
        component_dir=component_dir,
        component_policy=component_policy,
        erased_border_width=erased_border_width,
        ignore_label=ignore_label,
    )
    components = relabel_components(components)
    components = smallest_component_dtype(components)
    image = image.astype(np.dtype(image_dtype), copy=False)

    np.save(image_out, image)
    np.save(comp_out, components)
    meta = build_case_meta(
        case_id=case_id,
        image_path=image_out,
        comp_path=comp_out,
        image=image,
        components=components,
        component_source=component_source,
        source_key="source_raw_image",
        source_path=image_path,
        erased_border_width=erased_border_width,
    )
    meta["source_raw_label"] = str(label_path)
    meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return meta


def convert_preprocessed_case(
    *,
    npz_path: Path,
    cases_dir: Path,
    component_dir: Path | None,
    erased_border_width: int,
    ignore_label: int | None,
    image_dtype: str,
    overwrite: bool,
) -> dict[str, Any]:
    case_id = npz_path.stem
    case_dir = cases_dir / case_id
    image_path = case_dir / "image.npy"
    comp_path = case_dir / "components.npy"
    meta_path = case_dir / "meta.json"
    if image_path.exists() and comp_path.exists() and meta_path.exists() and not overwrite:
        return json.loads(meta_path.read_text(encoding="utf-8"))

    case_dir.mkdir(parents=True, exist_ok=True)
    with np.load(npz_path) as data:
        image = np.asarray(data["data"][0], dtype=np.float32)
        seg = np.asarray(data["seg"][0])

    image = np.array(image, copy=True)
    zero_volume_border(image, erased_border_width)

    components, component_source = load_preprocessed_components(
        case_id=case_id,
        seg=seg,
        component_dir=component_dir,
        erased_border_width=erased_border_width,
        ignore_label=ignore_label,
    )
    components = relabel_components(components)
    components = smallest_component_dtype(components)
    image = image.astype(np.dtype(image_dtype), copy=False)

    np.save(image_path, image)
    np.save(comp_path, components)
    meta = build_case_meta(
        case_id=case_id,
        image_path=image_path,
        comp_path=comp_path,
        image=image,
        components=components,
        component_source=component_source,
        source_key="source_npz",
        source_path=npz_path,
        erased_border_width=erased_border_width,
    )
    meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return meta


def build_case_meta(
    *,
    case_id: str,
    image_path: Path,
    comp_path: Path,
    image: np.ndarray,
    components: np.ndarray,
    component_source: str,
    source_key: str,
    source_path: Path,
    erased_border_width: int,
) -> dict[str, Any]:
    comp_ids, counts = np.unique(components, return_counts=True)
    fg_counts = counts[comp_ids > 0]
    meta = {
        "case_id": case_id,
        "image_path": str(image_path),
        "components_path": str(comp_path),
        "shape": list(image.shape),
        "image_dtype": str(image.dtype),
        "components_dtype": str(components.dtype),
        "num_components": int((comp_ids > 0).sum()),
        "foreground_voxels": int(fg_counts.sum()) if len(fg_counts) else 0,
        "max_component_voxels": int(fg_counts.max()) if len(fg_counts) else 0,
        "component_source": component_source,
        source_key: str(source_path),
        "erased_border_width": int(erased_border_width),
    }
    return meta


def load_raw_components(
    *,
    case_id: str,
    raw_label: np.ndarray,
    component_dir: Path | None,
    component_policy: str,
    erased_border_width: int,
    ignore_label: int | None,
) -> tuple[np.ndarray, str]:
    if component_policy not in {"shape_match", "never", "require_match"}:
        raise ValueError(f"Unsupported component_policy: {component_policy}")
    if component_policy != "never" and component_dir is not None:
        cc_path = component_dir / f"{case_id}_cc_t5.npy"
        if cc_path.exists():
            external = np.asarray(np.load(cc_path), dtype=np.int32)
            if external.shape == raw_label.shape:
                external = np.array(external, copy=True)
                zero_volume_border(external, erased_border_width)
                return external, "cc_t5_raw_shape_match"
            if component_policy == "require_match":
                raise ValueError(
                    f"{cc_path} shape {external.shape} does not match raw label shape {raw_label.shape}"
                )
            return connected_components_from_label(
                raw_label,
                erased_border_width=erased_border_width,
                ignore_label=ignore_label,
            ), "raw_label_connected_components_component_shape_mismatch"
    return connected_components_from_label(
        raw_label,
        erased_border_width=erased_border_width,
        ignore_label=ignore_label,
    ), "raw_label_connected_components"


def connected_components_from_label(
    labels: np.ndarray,
    *,
    erased_border_width: int,
    ignore_label: int | None,
) -> np.ndarray:
    cleaned = np.array(labels, copy=True)
    cleaned[cleaned < 0] = 0
    if ignore_label is not None:
        cleaned[cleaned == ignore_label] = 0
    zero_volume_border(cleaned, erased_border_width)
    cc, _ = ndimage.label(cleaned > 0)
    return cc.astype(np.int32, copy=False)


def load_nifti_array(path: str | Path) -> np.ndarray:
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rb") as f:
        header = f.read(348)
        if len(header) != 348:
            raise ValueError(f"Invalid NIfTI header in {path}")
        sizeof_hdr_le = struct.unpack("<i", header[:4])[0]
        endian = "<" if sizeof_hdr_le == 348 else ">"
        if struct.unpack(endian + "i", header[:4])[0] != 348:
            raise ValueError(f"Invalid NIfTI sizeof_hdr in {path}")
        dim = struct.unpack(endian + "8h", header[40:56])
        ndim = int(dim[0])
        shape = tuple(int(v) for v in dim[1:1 + ndim])
        datatype = int(struct.unpack(endian + "h", header[70:72])[0])
        vox_offset = int(float(struct.unpack(endian + "f", header[108:112])[0]))
        slope = float(struct.unpack(endian + "f", header[112:116])[0])
        intercept = float(struct.unpack(endian + "f", header[116:120])[0])
        dtype = nifti_dtype(datatype, endian)
        if vox_offset > 348:
            f.read(vox_offset - 348)
        count = int(np.prod(shape))
        data = f.read(count * dtype.itemsize)
    arr = np.frombuffer(data, dtype=dtype, count=count).reshape(shape, order="F")
    if slope not in {0.0, 1.0} or intercept != 0.0:
        slope = 1.0 if slope == 0.0 else slope
        arr = arr.astype(np.float32) * slope + intercept
    return np.array(arr, copy=True)


def nifti_dtype(datatype: int, endian: str) -> np.dtype:
    mapping = {
        2: "u1",
        4: "i2",
        8: "i4",
        16: "f4",
        64: "f8",
        256: "i1",
        512: "u2",
        768: "u4",
    }
    if datatype not in mapping:
        raise ValueError(f"Unsupported NIfTI datatype code: {datatype}")
    code = mapping[datatype]
    if code.endswith("1"):
        return np.dtype(code)
    return np.dtype(endian + code)


def convert_case(
    *,
    npz_path: Path,
    cases_dir: Path,
    component_dir: Path | None,
    erased_border_width: int,
    ignore_label: int | None,
    image_dtype: str,
    overwrite: bool,
) -> dict[str, Any]:
    return convert_preprocessed_case(
        npz_path=npz_path,
        cases_dir=cases_dir,
        component_dir=component_dir,
        erased_border_width=erased_border_width,
        ignore_label=ignore_label,
        image_dtype=image_dtype,
        overwrite=overwrite,
    )


def load_preprocessed_components(
    *,
    case_id: str,
    seg: np.ndarray,
    component_dir: Path | None,
    erased_border_width: int,
    ignore_label: int | None,
) -> tuple[np.ndarray, str]:
    if component_dir is not None:
        cc_path = component_dir / f"{case_id}_cc_t5.npy"
        if cc_path.exists():
            inner = np.asarray(np.load(cc_path), dtype=np.int32)
            full = np.zeros(seg.shape, dtype=np.int32)
            if erased_border_width > 0:
                dst = tuple(slice(erased_border_width, -erased_border_width) for _ in range(3))
                if full[dst].shape != inner.shape:
                    raise ValueError(
                        f"{cc_path} shape {inner.shape} does not match trimmed "
                        f"seg shape {full[dst].shape}")
                full[dst] = inner
            else:
                if inner.shape != full.shape:
                    raise ValueError(f"{cc_path} shape {inner.shape} != seg shape {full.shape}")
                full = inner
            zero_volume_border(full, erased_border_width)
            return full, "cc_t5"

    labels = np.array(seg, copy=True)
    labels[labels < 0] = 0
    if ignore_label is not None:
        labels[labels == ignore_label] = 0
    zero_volume_border(labels, erased_border_width)
    cc, _ = ndimage.label(labels > 0)
    return cc.astype(np.int32, copy=False), "seg_connected_components"


def relabel_components(labels: np.ndarray) -> np.ndarray:
    labels = np.asarray(labels)
    out = np.zeros(labels.shape, dtype=np.int32)
    ids = [int(v) for v in np.unique(labels) if int(v) > 0]
    for new_id, old_id in enumerate(ids, start=1):
        out[labels == old_id] = new_id
    return out


def smallest_component_dtype(labels: np.ndarray) -> np.ndarray:
    max_id = int(labels.max()) if labels.size else 0
    if max_id <= np.iinfo(np.uint8).max:
        return labels.astype(np.uint8, copy=False)
    if max_id <= np.iinfo(np.uint16).max:
        return labels.astype(np.uint16, copy=False)
    return labels.astype(np.int32, copy=False)


def save_preview(path: Path, row: dict[str, Any]) -> None:
    image = np.load(row["image_path"], mmap_mode="r")
    comps = np.load(row["components_path"], mmap_mode="r")
    mask = comps > 0
    prompt = choose_prompt(mask)
    save_p2sd_overview_png(
        path,
        image=np.asarray(image),
        gt=np.asarray(mask),
        pred=np.asarray(mask),
        prompt_zyx=prompt,
        metrics={"num_components": float(row["num_components"])},
        meta={"case_id": row["case_id"], "preview": True},
    )


def choose_prompt(mask: np.ndarray) -> tuple[int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    return tuple(int(v) for v in coords[len(coords) // 2])


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_raw_dir", default=None)
    parser.add_argument("--source_preprocessed_dir", default=None)
    parser.add_argument("--component_dir", default=None)
    parser.add_argument(
        "--component_policy",
        default="shape_match",
        choices=["shape_match", "never", "require_match"],
    )
    parser.add_argument("--output_root", default="datasets/vesuvius554_raw_clean_t5")
    parser.add_argument("--erased_border_width", type=int, default=5)
    parser.add_argument("--ignore_label", type=int, default=2)
    parser.add_argument("--val_fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit_cases", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--image_dtype", default="float16")
    parser.add_argument("--max_preview_cases", type=int, default=4)
    args = parser.parse_args(argv)
    path = convert_dataset(**vars(args))
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
