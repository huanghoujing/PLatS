"""Run the published 9um ink model on a bounded public surface-volume crop.

Run from the project root. Outputs are in flattened surface coordinates, NOT
native scroll coordinates. No labels or prompt points are fabricated when absent.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import itertools
import json
from pathlib import Path
import subprocess
import sys
import time
import urllib.request

import nibabel as nib
import numpy as np


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def fetch(url: str, cache: Path) -> bytes:
    """Cache successful responses atomically; never turn failed reads into air."""
    path = cache / hashlib.sha256(url.encode()).hexdigest()
    if path.exists():
        return path.read_bytes()
    cache.mkdir(parents=True, exist_ok=True)
    for attempt in range(3):
        try:
            with urllib.request.urlopen(url, timeout=30) as response:
                data = response.read()
            temporary = path.with_suffix(".partial")
            temporary.write_bytes(data)
            temporary.replace(path)
            return data
        except OSError:
            if attempt == 2:
                raise
            time.sleep(attempt + 1)
    raise AssertionError("unreachable")


def read_crop(config: dict) -> tuple[np.ndarray, dict]:
    """Read level 0 of a native 9um, uncompressed, uint8 Zarr v2 render.

    Deliberately reject other encodings/resolutions instead of guessing axes,
    silently filling missing chunks, or loading a whole scroll into memory.
    """
    url = config["source_url"].rstrip("/")
    cache = Path(config["cache_dir"])
    meta = json.loads(fetch(url + "/0/.zarray", cache))
    attrs = json.loads(fetch(url + "/.zattrs", cache))
    if (meta.get("zarr_format") != 2 or meta.get("compressor") is not None
            or meta.get("filters") or meta.get("dtype") != "|u1"
            or meta.get("order") != "C" or len(meta["shape"]) != 3):
        raise ValueError("Expected an uncompressed C-order uint8 Zarr v2 ZYX render")
    multiscale = attrs["multiscales"][0]
    if [(a["name"], a["unit"]) for a in multiscale["axes"]] != [
        (a, "micrometer") for a in ("z", "y", "x")
    ]:
        raise ValueError("Expected ZYX axes in micrometers")
    level = next(d for d in multiscale["datasets"] if d["path"] == "0")
    transforms = level["coordinateTransformations"]
    spacing = next(t["scale"] for t in transforms if t["type"] == "scale")
    if not np.allclose(spacing, config["spacing_zyx_um"]):
        raise ValueError("Configured spacing disagrees with source metadata")
    if any(t["type"] not in ("scale", "translation") or
           (t["type"] == "translation" and any(t["translation"])) for t in transforms):
        raise ValueError("This baseline supports only scale and zero translation")
    if not all(8 <= s <= 11 for s in spacing):
        raise ValueError("This checkpoint requires an approximately isotropic 9um render")
    origin = np.array([0, *config["origin_yx"]], dtype=int)
    shape = np.array([meta["shape"][0], *config["shape_yx"]], dtype=int)
    chunks = np.array(meta["chunks"], dtype=int)
    if np.any(origin < 0) or np.any(shape <= 0) or np.any(origin + shape > meta["shape"]):
        raise ValueError("Crop must fit inside the source volume")
    if int(np.prod(shape)) > 128 * 1024**2:
        raise ValueError("Crop exceeds the 128 MiB baseline limit; use smaller regions")
    coords = list(itertools.product(*(range(int(lo), int(hi) + 1) for lo, hi in
                  zip(origin // chunks, (origin + shape - 1) // chunks))))
    separator = meta.get("dimension_separator", ".")
    if separator not in ("/", "."):
        raise ValueError("Unsupported chunk separator")
    result = np.empty(tuple(shape), dtype=np.uint8)

    def read_one(coord):
        chunk_url = url + "/0/" + separator.join(map(str, coord))
        data = fetch(chunk_url, cache)
        if len(data) != int(np.prod(chunks)):
            raise ValueError(f"Unexpected chunk byte count: {chunk_url}")
        return coord, np.frombuffer(data, dtype=np.uint8).reshape(tuple(chunks))

    with ThreadPoolExecutor(max_workers=8) as pool:
        for coord, chunk in pool.map(read_one, coords):
            chunk_origin = np.array(coord) * chunks
            start = np.maximum(origin, chunk_origin)
            stop = np.minimum(origin + shape, chunk_origin + chunks)
            dst = tuple(slice(int(a), int(b)) for a, b in zip(start-origin, stop-origin))
            src = tuple(slice(int(a), int(b)) for a, b in zip(start-chunk_origin, stop-chunk_origin))
            result[dst] = chunk[src]
    return result, {"zarray": meta, "zattrs": attrs, "chunks_read": len(coords)}


def positions(length: int, patch: int, stride: int) -> list[int]:
    if not 1 <= stride <= patch or length < patch:
        raise ValueError("Require length >= patch and 1 <= stride <= patch")
    return sorted(set([*range(0, length - patch + 1, stride), length - patch]))


def tiled_predict(volume, predict, *, patch: int, stride: int, batch_size: int):
    """Blend raw probabilities with nonzero Hann weights, including edge pixels."""
    if volume.ndim != 3 or batch_size < 1:
        raise ValueError("Expected a ZYX volume and positive batch size")
    height, width = volume.shape[1:]
    tiles = list(itertools.product(positions(height, patch, stride), positions(width, patch, stride)))
    window = np.maximum(np.hanning(patch), 1e-3).astype(np.float32)
    weight = np.outer(window, window)
    total = np.zeros((height, width), np.float32)
    count = np.zeros_like(total)
    for start in range(0, len(tiles), batch_size):
        batch_tiles = tiles[start:start + batch_size]
        batch = np.stack([volume[:, y:y+patch, x:x+patch] for y, x in batch_tiles])
        predictions = np.asarray(predict(batch), dtype=np.float32)
        if (predictions.shape != (len(batch_tiles), patch, patch)
                or not np.isfinite(predictions).all()
                or predictions.min() < 0 or predictions.max() > 1):
            raise ValueError("Predictor must return finite BYX probabilities in [0, 1]")
        for (y, x), prediction in zip(batch_tiles, predictions):
            total[y:y+patch, x:x+patch] += prediction * weight
            count[y:y+patch, x:x+patch] += weight
    return total / count


def save_nifti(path: Path, volume: np.ndarray, config: dict) -> None:
    """ZYX array to XYZ NIfTI in the source's flat (u,v,depth) mm coordinates."""
    affine = np.diag([*(np.array(config["spacing_zyx_um"])[::-1] / 1000), 1.0])
    affine[:3, 3] = affine[:3, :3] @ np.array([*config["origin_yx"][::-1], 0])
    nii = nib.Nifti1Image(np.ascontiguousarray(volume.transpose(2, 1, 0)), affine)
    nii.header.set_xyzt_units("mm")
    nii.set_sform(affine, code=2)  # aligned local coordinates, not native scan RAS
    nii.set_qform(affine, code=0)
    nib.save(nii, path)


def run(config: dict, out: Path) -> None:
    import torch

    out.mkdir(parents=True, exist_ok=False)
    for name in ("inputs", "predictions", "viz", "logs"):
        (out / name).mkdir()
    write_json(out / "config.json", config)
    started = time.monotonic()

    def event(stage, **values):
        row = {"stage": stage, "elapsed_seconds": time.monotonic()-started, **values}
        with (out / "logs/events.jsonl").open("a") as stream:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
        print(json.dumps(row), flush=True)

    try:
        villa = Path(config["villa_root"]).resolve()
        revision = subprocess.check_output(["git", "-C", str(villa), "rev-parse", "HEAD"], text=True).strip()
        if revision != config["villa_revision"]:
            raise ValueError("Official model checkout does not match the pinned revision")
        if subprocess.check_output(["git", "-C", str(villa), "status", "--porcelain"], text=True).strip():
            raise ValueError("Official model checkout has local changes")
        sys.path.insert(0, str(villa / "vesuvius/src"))
        from vesuvius.ink_detection.config import InkConfig
        from vesuvius.ink_detection.models.model import make_model
        from vesuvius.ink_detection.data.normalization import normalize_image
        from vesuvius.ink_detection.models.checkpoint import select_inference_weights

        checkpoint_path = Path(config["checkpoint"])
        checkpoint_hash = sha256(checkpoint_path)
        if checkpoint_hash != config["checkpoint_sha256"]:
            raise ValueError("Checkpoint bytes do not match the pinned SHA256")
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        model_config = InkConfig.from_mapping(payload["config"])
        if model_config.data.mode != "flat":
            raise ValueError("Expected a flat surface-volume model")
        depth, patch, patch_x = model_config.model.crop_size
        if patch != patch_x:
            raise ValueError("Expected square model patches")
        device = torch.device(config["device"])
        model = make_model(model_config)
        weights_name, weights = select_inference_weights(payload)
        model.load_state_dict(weights, strict=True)
        model.to(device).eval()
        event("model_loaded", device=str(device), weights=weights_name)
        write_json(out / "inputs/checkpoint_config.json", payload["config"])
        volume, source = read_crop(config)
        np.save(out / "inputs/image.npy", volume)
        write_json(out / "inputs/source.json", source)
        save_nifti(out / "viz/image.nii.gz", volume, config)
        event("input_ready", shape=list(volume.shape))

        @torch.inference_mode()
        def predict(batch):
            normalized = np.stack([normalize_image(p.copy(), model_config.data.normalization) for p in batch])
            image = torch.from_numpy(normalized[:, None]).to(device)
            return model(image)["ink"][:, 0].float().sigmoid().cpu().numpy()

        results = []
        for direction in config["directions"]:
            if direction not in ("forward", "reverse"):
                raise ValueError("Direction must be forward or reverse")
            for start in config["layer_starts"]:
                if start < 0 or start + depth > volume.shape[0]:
                    raise ValueError("Depth window falls outside the surface volume")
                window = volume[start:start + depth]
                if direction == "reverse":
                    window = window[::-1].copy()
                prediction = tiled_predict(window, predict, patch=patch, stride=config["stride"],
                                           batch_size=config["batch_size"])
                name = f"{direction}_z{start:02d}-{start+depth:02d}"
                np.save(out / f"predictions/{name}.npy", prediction)
                # Repeat the 2D map only for overlay inspection across raw slices.
                # This does NOT estimate the depth occupied by ink.
                overlay = np.broadcast_to(prediction, volume.shape)
                save_nifti(out / f"viz/pred_{name}.nii.gz", overlay, config)
                results.append({"name": name, "mean": float(prediction.mean()),
                                "std": float(prediction.std()),
                                "min": float(prediction.min()), "max": float(prediction.max())})
                event("prediction_saved", **results[-1])
        manifest = {
            "status": "complete", "purpose": config["purpose"],
            "coordinate_system": "flattened surface u/v/depth; not native scroll XYZ",
            "prediction_semantics": "raw 2D probabilities; NIfTI repeats each map through depth for viewing only",
            "gt": None, "points": None, "held_out_metrics": None,
            "checkpoint_sha256": checkpoint_hash, "villa_revision": revision,
            "torch_version": torch.__version__,
            "device": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
            "input_sha256": sha256(out / "inputs/image.npy"),
            "area_cm2": float(np.prod(np.array(config["shape_yx"]) * np.array(config["spacing_zyx_um"])[1:] / 10000)),
            "predictions": results, "elapsed_seconds": time.monotonic()-started,
        }
        write_json(out / "manifest.json", manifest)
        event("complete")
    except Exception as exc:
        write_json(out / "manifest.json", {"status": "failed", "error": str(exc)})
        event("failed", error=str(exc))
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/ink/first_letters_baseline.json"))
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    out = args.out or Path(config["run_root"]) / f"{stamp}_{config['scroll']}_ink9um_baseline"
    run(config, out)


if __name__ == "__main__":
    main()
