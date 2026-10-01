"""Deterministic PNG visualization for P2SD predictions."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

try:
    import nibabel as nib
except ImportError:  # NIFTI export is reported clearly when the optional dependency is absent.
    nib = None

from vesuvius_p2sd.eval.connected_components import prompt_connected_mask


@dataclass(frozen=True)
class P2SDVisualizationState:
    image_3d: np.ndarray
    gt_mask: np.ndarray
    pred_prob: np.ndarray
    pred_mask: np.ndarray
    ae_prob: np.ndarray | None
    ae_mask: np.ndarray | None
    points: np.ndarray
    labels: np.ndarray
    primary_prompt: tuple[int, int, int] | None
    errors: dict[str, np.ndarray]
    prompt_component: np.ndarray
    detached_component: np.ndarray
    base_volume: np.ndarray


def prune_visualization_groups(
    directory: str | Path,
    *,
    keep_latest: int | None,
    stem_prefixes: tuple[str, ...] = (),
) -> None:
    """Keep only the newest visualization groups in a directory."""

    if keep_latest is None or int(keep_latest) <= 0:
        return
    directory = Path(directory)
    if not directory.exists():
        return
    groups: dict[str, list[Path]] = {}
    for path in directory.iterdir():
        if not path.is_file():
            continue
        key = _visualization_group_key(path)
        if key is None:
            continue
        if stem_prefixes and not key.startswith(stem_prefixes):
            continue
        groups.setdefault(key, []).append(path)
    if len(groups) <= int(keep_latest):
        return
    newest_first = sorted(
        groups.items(),
        key=lambda item: max(path.stat().st_mtime for path in item[1]),
        reverse=True,
    )
    for _, paths in newest_first[int(keep_latest):]:
        for path in paths:
            path.unlink(missing_ok=True)


def prune_visualization_step_groups(
    directory: str | Path,
    *,
    keep_latest: int | None,
    step_prefix: str,
) -> None:
    """Prune complete evaluation steps while retaining all prompt variants."""

    if keep_latest is None or int(keep_latest) <= 0:
        return
    directory = Path(directory)
    if not directory.exists():
        return
    groups: dict[str, list[Path]] = {}
    for path in directory.iterdir():
        if not path.is_file():
            continue
        key = _visualization_group_key(path)
        if key is None or not key.startswith(step_prefix):
            continue
        step_text = key[len(step_prefix):].split("_", 1)[0]
        if not step_text.isdigit():
            continue
        groups.setdefault(step_prefix + step_text, []).append(path)
    if len(groups) <= int(keep_latest):
        return
    newest_first = sorted(
        groups.items(),
        key=lambda item: max(path.stat().st_mtime for path in item[1]),
        reverse=True,
    )
    for _, paths in newest_first[int(keep_latest):]:
        for path in paths:
            path.unlink(missing_ok=True)


def prune_visualization_epoch_directories(
    directory: str | Path,
    *,
    keep_latest: int | None,
    preserve_epochs: set[int] | None = None,
) -> None:
    """Keep recent epoch galleries while retaining configured milestones."""

    if keep_latest is None or int(keep_latest) <= 0:
        return
    directory = Path(directory)
    if not directory.exists():
        return
    preserve_epochs = preserve_epochs or set()
    epoch_dirs: list[tuple[int, Path]] = []
    for path in directory.iterdir():
        if not path.is_dir() or not path.name.startswith("epoch_"):
            continue
        epoch_text = path.name.removeprefix("epoch_")
        if epoch_text.isdigit():
            epoch_dirs.append((int(epoch_text), path))
    removable = [(epoch, path) for epoch, path in epoch_dirs if epoch not in preserve_epochs]
    if len(removable) <= int(keep_latest):
        return
    newest_first = sorted(removable, key=lambda item: item[0], reverse=True)
    for _, path in newest_first[int(keep_latest):]:
        for child in sorted(path.rglob("*"), reverse=True):
            if child.is_file() or child.is_symlink():
                child.unlink(missing_ok=True)
            elif child.is_dir():
                child.rmdir()
        path.rmdir()


def resolve_visualization_milestone_epoch(
    visualization_cfg: dict[str, Any],
    *,
    step: int,
    steps_per_epoch: int,
) -> int | None:
    """Return the configured milestone epoch reached by this evaluation step."""

    epoch = float(step) / max(1, int(steps_per_epoch))
    candidates = [float(value) for value in visualization_cfg.get("milestone_epochs", [])]
    interval = visualization_cfg.get("milestone_interval_epochs")
    if interval is not None and float(interval) > 0:
        interval_value = float(interval)
        nearest = round(epoch / interval_value) * interval_value
        if nearest > 0:
            candidates.append(nearest)
    for candidate in candidates:
        if abs(epoch - candidate) <= 1e-6:
            return int(round(candidate))
    return None


def _visualization_group_key(path: Path) -> str | None:
    if path.name.endswith(".nii.gz"):
        stem = path.name[:-7]
    elif path.suffix in {".png", ".json", ".npz", ".ply"}:
        stem = path.stem
    else:
        return None
    for suffix in ("_volumes", "_error_points", "_gt_points", "_pred_points", "_projection", "_gt", "_pred", "_gt_pred_overlap"):
        if stem.endswith(suffix):
            return stem[: -len(suffix)]
    return stem


def save_p2sd_overview_png(
    path: str | Path,
    *,
    image: np.ndarray,
    gt: np.ndarray,
    pred: np.ndarray,
    prompt_zyx: tuple[int, int, int] | None = None,
    metrics: dict[str, float] | None = None,
    meta: dict[str, Any] | None = None,
) -> Path:
    """Save a stable 2D overview panel from 3D arrays."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    image_3d = _as_3d(image)
    gt_mask = _as_3d(gt) > 0.5
    pred_mask = _as_3d(pred) > 0.5
    z = _choose_slice(gt_mask, pred_mask, prompt_zyx)

    base = _normalize_to_uint8(image_3d[z])
    gt_panel = _mask_panel(base, gt_mask[z], color=(0, 200, 0))
    pred_panel = _mask_panel(base, pred_mask[z], color=(255, 80, 0))
    err_panel = _error_panel(base, gt_mask[z], pred_mask[z])
    img_panel = Image.fromarray(base, mode="L").convert("RGB")
    if prompt_zyx is not None and int(prompt_zyx[0]) == z:
        for panel in (img_panel, gt_panel, pred_panel, err_panel):
            _draw_prompt(panel, (int(prompt_zyx[2]), int(prompt_zyx[1])))

    title_h = 18
    panels = [img_panel, gt_panel, pred_panel, err_panel]
    labels = ["image", "gt", "pred", "fp/fn"]
    w, h = panels[0].size
    canvas = Image.new("RGB", (w * len(panels), h + title_h), "white")
    draw = ImageDraw.Draw(canvas)
    for i, (panel, label) in enumerate(zip(panels, labels)):
        canvas.paste(panel, (i * w, title_h))
        draw.text((i * w + 4, 2), label, fill=(0, 0, 0))
    _save_png(canvas, path)

    sidecar = {
        "slice_z": z,
        "prompt_zyx": prompt_zyx,
        "metrics": metrics or {},
        "meta": meta or {},
    }
    with path.with_suffix(".json").open("w", encoding="utf-8") as f:
        json.dump(sidecar, f, indent=2, sort_keys=True)
    return path


def prepare_p2sd_visualization_state(
    *,
    image: np.ndarray,
    gt: np.ndarray,
    pred: np.ndarray,
    ae_recon: np.ndarray | None = None,
    prompt_points_zyx: np.ndarray | None = None,
    prompt_labels: np.ndarray | None = None,
    prompt_zyx: tuple[int, int, int] | None = None,
    threshold: float = 0.5,
    tolerance_voxels: float = 2.0,
) -> P2SDVisualizationState:
    """Prepare geometry once for paired diagnostic and projection PNGs."""

    image_3d = _as_3d(image)
    gt_mask = _as_3d(gt) > 0.5
    pred_prob = _as_3d(pred).astype(np.float32, copy=False)
    pred_mask = pred_prob > float(threshold)
    ae_prob = None if ae_recon is None else _as_3d(ae_recon).astype(np.float32, copy=False)
    ae_mask = None if ae_prob is None else ae_prob > float(threshold)
    points, labels = _normalize_prompt_points(prompt_points_zyx, prompt_labels, prompt_zyx)
    primary_prompt = _primary_positive_prompt(points, labels)
    errors = _tolerant_error_masks(gt_mask, pred_mask, tolerance_voxels=tolerance_voxels)
    prompt_component = prompt_connected_mask(pred_mask, primary_prompt)
    return P2SDVisualizationState(
        image_3d=image_3d,
        gt_mask=gt_mask,
        pred_prob=pred_prob,
        pred_mask=pred_mask,
        ae_prob=ae_prob,
        ae_mask=ae_mask,
        points=points,
        labels=labels,
        primary_prompt=primary_prompt,
        errors=errors,
        prompt_component=prompt_component,
        detached_component=pred_mask & ~prompt_component,
        base_volume=_normalize_to_uint8(image_3d),
    )


def save_p2sd_diagnostic_png(
    path: str | Path,
    *,
    image: np.ndarray,
    gt: np.ndarray,
    pred: np.ndarray,
    ae_recon: np.ndarray | None = None,
    prompt_points_zyx: np.ndarray | None = None,
    prompt_labels: np.ndarray | None = None,
    prompt_zyx: tuple[int, int, int] | None = None,
    metrics: dict[str, float] | None = None,
    meta: dict[str, Any] | None = None,
    threshold: float = 0.5,
    tolerance_voxels: float = 2.0,
    prediction_label: str = "p2sd",
    visual_state: P2SDVisualizationState | None = None,
) -> Path:
    """Save prompt- and error-centered 2D diagnostic slices from 3D arrays."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = visual_state or prepare_p2sd_visualization_state(
        image=image,
        gt=gt,
        pred=pred,
        ae_recon=ae_recon,
        prompt_points_zyx=prompt_points_zyx,
        prompt_labels=prompt_labels,
        prompt_zyx=prompt_zyx,
        threshold=threshold,
        tolerance_voxels=tolerance_voxels,
    )
    gt_mask = state.gt_mask
    pred_mask = state.pred_mask
    ae_mask = state.ae_mask
    points = state.points
    labels = state.labels
    primary_prompt = state.primary_prompt
    errors = state.errors
    prompt_component = state.prompt_component
    detached_component = state.detached_component

    slice_indices = [_choose_slice(gt_mask, pred_mask, primary_prompt)]
    error_slice = _choose_error_slice(errors)
    if error_slice not in slice_indices:
        slice_indices.append(error_slice)

    base_volume = state.base_volume
    rows: list[tuple[str, list[Image.Image]]] = []
    for z in slice_indices:
        base = base_volume[z]
        panels = [
            Image.fromarray(base, mode="L").convert("RGB"),
            _mask_panel(base, gt_mask[z], color=(0, 200, 0)),
        ]
        labels_for_panels = ["image", "gt"]
        if ae_mask is not None:
            panels.append(_mask_panel(base, ae_mask[z], color=(170, 80, 255)))
            labels_for_panels.append("ae recon")
        panels.extend([
            _mask_panel(base, pred_mask[z], color=(255, 80, 0)),
            _tolerant_error_panel(base, errors, z),
            _topology_panel(base, prompt_component[z], detached_component[z]),
        ])
        labels_for_panels.extend([str(prediction_label), "tolerant fp/fn", "topology"])
        for panel in panels:
            _draw_prompt_points(panel, points, labels, slice_z=z)
        rows.append((f"z={z}", _label_panels(panels, labels_for_panels)))

    canvas = _stack_labeled_rows(rows)
    _save_png(canvas, path)
    sidecar = {
        "slice_z": int(slice_indices[0]),
        "slice_zs": [int(z) for z in slice_indices],
        "prompt_zyx": primary_prompt,
        "prompt_points_zyx": points.tolist(),
        "prompt_labels": labels.tolist(),
        "threshold": float(threshold),
        "tolerance_voxels": float(tolerance_voxels),
        "metrics": metrics or {},
        "meta": meta or {},
        "colors": _diagnostic_color_legend(),
    }
    _write_sidecar(path.with_suffix(".json"), sidecar)
    return path


def save_p2sd_projection_png(
    path: str | Path,
    *,
    image: np.ndarray,
    gt: np.ndarray,
    pred: np.ndarray,
    ae_recon: np.ndarray | None = None,
    prompt_points_zyx: np.ndarray | None = None,
    prompt_labels: np.ndarray | None = None,
    prompt_zyx: tuple[int, int, int] | None = None,
    metrics: dict[str, float] | None = None,
    meta: dict[str, Any] | None = None,
    threshold: float = 0.5,
    tolerance_voxels: float = 2.0,
    prediction_label: str = "p2sd",
    visual_state: P2SDVisualizationState | None = None,
) -> Path:
    """Save a maximum-area oblique projection for geometry and artifact review."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = visual_state or prepare_p2sd_visualization_state(
        image=image,
        gt=gt,
        pred=pred,
        ae_recon=ae_recon,
        prompt_points_zyx=prompt_points_zyx,
        prompt_labels=prompt_labels,
        prompt_zyx=prompt_zyx,
        threshold=threshold,
        tolerance_voxels=tolerance_voxels,
    )
    image_3d = state.image_3d
    gt_mask = state.gt_mask
    pred_mask = state.pred_mask
    ae_mask = state.ae_mask
    points = state.points
    labels = state.labels
    primary_prompt = state.primary_prompt
    projection = _make_oblique_projection(
        image_3d,
        gt_mask | pred_mask,
        gt_mask,
        pred_mask,
        state.errors,
        state.prompt_component,
        state.detached_component,
    )
    base = projection["base"]
    label = "oblique max-area projection"
    row_specs: list[tuple[str, np.ndarray | None, tuple[int, int, int] | None]] = [
        ("image", projection["base"], None),
        ("gt", projection["gt"], (0, 200, 0)),
    ]
    if ae_mask is not None:
        ae_projection = _project_volume_oblique(ae_mask, projection["basis"], projection["canvas"])
        row_specs.append(("ae recon", ae_projection, (170, 80, 255)))
    row_specs.append((str(prediction_label), projection["pred"], (255, 80, 0)))

    rows: list[tuple[str, list[Image.Image]]] = []
    for row_label, volume, color in row_specs:
        panels = []
        if color is None:
            panel = Image.fromarray(base, mode="L").convert("RGB")
        else:
            panel = _mask_panel(base, np.asarray(volume, dtype=bool), color=color)
        _draw_oblique_prompt_points(panel, points, labels, projection)
        panels.append(panel)
        rows.append((row_label, panels))

    overlap_panel = _gt_pred_overlap_panel(projection["gt"], projection["pred"])
    _draw_oblique_prompt_points(
        overlap_panel,
        points,
        labels,
        projection,
        positive_color=(0, 255, 255),
        negative_color=(0, 100, 255),
    )
    rows.append(("gt-pred overlap", [overlap_panel]))

    canvas = _stack_labeled_rows(rows)
    canvas = _append_color_legend(canvas, {
        "prediction": (255, 255, 255),
        "GT over prediction": (255, 220, 0),
        "positive prompt": (0, 255, 255),
        "negative prompt": (0, 100, 255),
    })
    _save_png(canvas, path)
    sidecar = {
        "projection_method": "pca_max_area_oblique",
        "projection_normal_zyx": projection["normal"].tolist(),
        "projection_basis_zyx": projection["basis"].tolist(),
        "projection_label": label,
        "prompt_zyx": primary_prompt,
        "prompt_points_zyx": points.tolist(),
        "prompt_labels": labels.tolist(),
        "threshold": float(threshold),
        "tolerance_voxels": float(tolerance_voxels),
        "metrics": metrics or {},
        "meta": meta or {},
        "colors": {
            "prediction": [255, 255, 255],
            "gt_over_prediction": [255, 220, 0],
            "positive_prompt": [0, 255, 255],
            "negative_prompt": [0, 100, 255],
        },
    }
    _write_sidecar(path.with_suffix(".json"), sidecar)
    return path


def _make_oblique_projection(
    image: np.ndarray,
    support: np.ndarray,
    gt: np.ndarray,
    pred: np.ndarray,
    errors: dict[str, np.ndarray],
    prompt_component: np.ndarray,
    detached_component: np.ndarray,
) -> dict[str, Any]:
    """Project volumes onto the plane normal to their thinnest PCA direction."""
    coords = np.argwhere(support)
    if coords.shape[0] < 3:
        normal = np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
    else:
        sample = coords[::max(1, coords.shape[0] // 8192)]
        _, vectors = np.linalg.eigh(np.cov(sample.astype(np.float32), rowvar=False))
        normal = vectors[:, 0].astype(np.float32)
        normal /= max(float(np.linalg.norm(normal)), 1e-8)
    reference = np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
    if abs(float(np.dot(reference, normal))) > 0.9:
        reference = np.asarray([0.0, 1.0, 0.0], dtype=np.float32)
    basis_u = np.cross(normal, reference)
    basis_u /= max(float(np.linalg.norm(basis_u)), 1e-8)
    basis_v = np.cross(normal, basis_u)
    basis = np.stack([basis_u, basis_v], axis=0)
    all_coords = np.indices(image.shape, dtype=np.float32).reshape(3, -1).T
    uv = all_coords @ basis.T
    lo = uv.min(axis=0)
    hi = uv.max(axis=0)
    size = int(max(image.shape))
    scale = (size - 1) / max(float(np.max(hi - lo)), 1e-6)

    def project(volume: np.ndarray, *, values: np.ndarray | None = None) -> np.ndarray:
        mask = np.asarray(volume, dtype=bool).reshape(-1)
        pixels = np.floor((uv[mask] - lo) * scale).astype(np.int64)
        out = np.zeros((size, size), dtype=np.float32 if values is not None else bool)
        if values is None:
            out[pixels[:, 1], pixels[:, 0]] = True
        else:
            np.maximum.at(out, (pixels[:, 1], pixels[:, 0]), np.asarray(values).reshape(-1)[mask])
        return out

    # Image projection uses all voxels; max pooling keeps bright sheet structure visible.
    image_projection = project(np.ones_like(image, dtype=bool), values=image)
    result = {
        "base": _normalize_to_uint8(image_projection),
        "gt": project(gt),
        "pred": project(pred),
        "errors": {key: project(value) for key, value in errors.items()},
        "prompt_component": project(prompt_component),
        "detached_component": project(detached_component),
        "normal": normal,
        "basis": basis,
        "canvas": (lo, scale, size),
    }
    return result


def _project_volume_oblique(volume: np.ndarray, basis: np.ndarray, canvas: tuple[np.ndarray, float, int]) -> np.ndarray:
    lo, scale, size = canvas
    coords = np.argwhere(np.asarray(volume, dtype=bool))
    out = np.zeros((size, size), dtype=bool)
    if coords.size:
        uv = coords.astype(np.float32) @ basis.T
        pixels = np.floor((uv - lo) * scale).astype(np.int64)
        valid = (pixels[:, 0] >= 0) & (pixels[:, 0] < size) & (pixels[:, 1] >= 0) & (pixels[:, 1] < size)
        out[pixels[valid, 1], pixels[valid, 0]] = True
    return out


def _projected_error_panel(base: np.ndarray, errors: dict[str, np.ndarray]) -> Image.Image:
    return _overlay_masks(base, [
        (errors["tp"], (0, 200, 0)),
        (errors["near_fp"], (255, 210, 0)),
        (errors["near_fn"], (0, 220, 220)),
        (errors["far_fp"], (255, 45, 45)),
        (errors["far_fn"], (0, 120, 255)),
    ])


def _gt_pred_overlap_panel(gt: np.ndarray, pred: np.ndarray) -> Image.Image:
    """Render prediction in white with GT painted over it in yellow."""
    gt = np.asarray(gt, dtype=bool)
    pred = np.asarray(pred, dtype=bool)
    rgb = np.zeros((*gt.shape, 3), dtype=np.uint8)
    rgb[pred] = (255, 255, 255)
    rgb[gt] = (255, 220, 0)
    return Image.fromarray(rgb, mode="RGB")


def _draw_oblique_prompt_points(
    panel: Image.Image,
    points: np.ndarray,
    labels: np.ndarray,
    projection: dict[str, Any],
    *,
    positive_color: tuple[int, int, int] = (255, 255, 0),
    negative_color: tuple[int, int, int] = (0, 220, 255),
) -> None:
    if points.ndim != 2 or points.shape[1] != 3:
        return
    lo, scale, size = projection["canvas"]
    uv = points.astype(np.float32) @ projection["basis"].T
    pixels = np.floor((uv - lo) * scale).astype(np.int64)
    draw = ImageDraw.Draw(panel)
    for (px, py), label in zip(pixels, labels):
        if 0 <= px < size and 0 <= py < size:
            radius = 5 if int(label) > 0 else 4
            color = positive_color if int(label) > 0 else negative_color
            draw.ellipse((px - radius, py - radius, px + radius, py + radius), fill=color, outline=(0, 0, 0), width=1)


def save_p2sd_3d_artifacts(
    path_prefix: str | Path,
    *,
    image: np.ndarray,
    gt: np.ndarray,
    pred: np.ndarray,
    ae_recon: np.ndarray | None = None,
    prompt_zyx: tuple[int, int, int] | None = None,
    metrics: dict[str, float] | None = None,
    meta: dict[str, Any] | None = None,
    threshold: float = 0.5,
    max_points: int = 200_000,
    save_npz: bool = False,
    save_ply: bool = False,
) -> dict[str, Path]:
    """Save volume arrays and a colored point-cloud error view for 3D inspection."""

    prefix = Path(path_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    image_3d = _as_3d(image).astype(np.float16, copy=False)
    gt_mask = (_as_3d(gt) > 0.5)
    pred_prob = _as_3d(pred).astype(np.float16, copy=False)
    pred_mask = pred_prob > float(threshold)
    if nib is None:
        raise RuntimeError("NIFTI visualization requires nibabel; install the project dependencies")
    payload = {
        "image": image_3d,
        "gt_mask": gt_mask.astype(np.uint8),
        "pred_prob": pred_prob,
        "pred_mask": pred_mask.astype(np.uint8),
    }
    if ae_recon is not None:
        ae_prob = _as_3d(ae_recon).astype(np.float16, copy=False)
        payload["ae_recon_prob"] = ae_prob
        payload["ae_recon_mask"] = (ae_prob > float(threshold)).astype(np.uint8)
    npz_path = prefix.with_name(prefix.name + "_volumes.npz")
    if save_npz:
        np.savez_compressed(npz_path, **payload)

    # Keep separate GT/pred volumes and a compact integer-coded comparison volume.
    # The arrays intentionally retain the repository's z,y,x voxel order.
    overlap = np.zeros(gt_mask.shape, dtype=np.uint8)
    overlap[gt_mask & ~pred_mask] = 1
    overlap[pred_mask & ~gt_mask] = 2
    overlap[gt_mask & pred_mask] = 3
    affine = np.eye(4, dtype=np.float32)
    nifti_paths = {
        "gt": prefix.with_name(prefix.name + "_gt.nii.gz"),
        "pred": prefix.with_name(prefix.name + "_pred.nii.gz"),
        "gt_pred_overlap": prefix.with_name(prefix.name + "_gt_pred_overlap.nii.gz"),
    }
    nib.save(nib.Nifti1Image(gt_mask.astype(np.uint8), affine), str(nifti_paths["gt"]))
    nib.save(nib.Nifti1Image(pred_mask.astype(np.uint8), affine), str(nifti_paths["pred"]))
    nib.save(nib.Nifti1Image(overlap, affine), str(nifti_paths["gt_pred_overlap"]))

    ply_path = prefix.with_name(prefix.name + "_error_points.ply")
    gt_ply_path = prefix.with_name(prefix.name + "_gt_points.ply")
    pred_ply_path = prefix.with_name(prefix.name + "_pred_points.ply")
    if save_ply:
        _save_error_point_cloud_ply(ply_path, gt_mask=gt_mask, pred_mask=pred_mask,
                                    prompt_zyx=prompt_zyx, max_points=max_points)
        _save_mask_point_cloud_ply(gt_ply_path, mask=gt_mask, color=(0, 200, 0),
                                   prompt_zyx=prompt_zyx, max_points=max_points)
        _save_mask_point_cloud_ply(pred_ply_path, mask=pred_mask, color=(255, 80, 0),
                                   prompt_zyx=prompt_zyx, max_points=max_points)

    sidecar_path = prefix.with_suffix(".json")
    with sidecar_path.open("w", encoding="utf-8") as f:
        json.dump({
            "threshold": float(threshold),
            "max_points": int(max_points),
            "prompt_zyx": prompt_zyx,
            "metrics": metrics or {},
            "meta": meta or {},
            "files": {
                **({"volumes": npz_path.name} if save_npz else {}),
                **({"error_points": ply_path.name, "gt_points": gt_ply_path.name,
                    "pred_points": pred_ply_path.name} if save_ply else {}),
                "gt": nifti_paths["gt"].name,
                "pred": nifti_paths["pred"].name,
                "gt_pred_overlap": nifti_paths["gt_pred_overlap"].name,
            },
            "nifti_label_maps": {
                "gt": {"0": "background", "1": "ground_truth"},
                "pred": {"0": "background", "1": "prediction"},
                "gt_pred_overlap": {
                    "0": "background", "1": "gt_only", "2": "pred_only", "3": "overlap"
                },
            },
            "nifti_axis_order": "z,y,x (same as source arrays)",
            "colors": {
                "gt_points": [0, 200, 0],
                "pred_points": [255, 80, 0],
                "true_positive": [0, 200, 0],
                "false_positive": [255, 80, 0],
                "false_negative": [0, 120, 255],
                "prompt": [255, 255, 0],
            },
        }, f, indent=2, sort_keys=True)
    return {
        **({"volumes": npz_path} if save_npz else {}),
        **({"error_points": ply_path, "gt_points": gt_ply_path,
            "pred_points": pred_ply_path} if save_ply else {}),
        **nifti_paths,
        "sidecar": sidecar_path,
    }


def save_p2sd_train_batch_png(
    path: str | Path,
    *,
    image: np.ndarray,
    target_mask: np.ndarray,
    prompt_points_zyx: np.ndarray,
    prompt_labels: np.ndarray,
    component_label: np.ndarray | None = None,
    meta: dict[str, Any] | None = None,
) -> Path:
    """Save a train-pair debug view for checking sampled data and prompts."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    image_3d = _as_3d(image)
    target = _as_3d(target_mask) > 0.5
    components = None if component_label is None else _as_3d(component_label).astype(np.int64, copy=False)
    points = np.asarray(prompt_points_zyx, dtype=np.float32)
    labels = np.asarray(prompt_labels, dtype=np.int64)
    z = _choose_train_slice(target, points, labels)

    base = _normalize_to_uint8(image_3d[z])
    image_panel = Image.fromarray(base, mode="L").convert("RGB")
    component_panel = (
        Image.fromarray(base, mode="L").convert("RGB")
        if components is None
        else _component_panel(base, components[z])
    )
    target_panel = _mask_panel(base, target[z], color=(0, 200, 0))
    prompt_panel = Image.fromarray(base, mode="L").convert("RGB")
    for panel in (image_panel, component_panel, target_panel, prompt_panel):
        _draw_prompt_points(panel, points, labels, slice_z=z)

    title_h = 18
    panels = [image_panel, component_panel, target_panel, prompt_panel]
    panel_labels = ["image", "components", "target", "prompts"]
    w, h = panels[0].size
    canvas = Image.new("RGB", (w * len(panels), h + title_h), "white")
    draw = ImageDraw.Draw(canvas)
    for i, (panel, label) in enumerate(zip(panels, panel_labels)):
        canvas.paste(panel, (i * w, title_h))
        draw.text((i * w + 4, 2), label, fill=(0, 0, 0))
    _save_png(canvas, path)

    sidecar = {
        "slice_z": int(z),
        "target_voxels": int(target.sum()),
        "prompt_points_zyx": points.tolist(),
        "prompt_labels": labels.tolist(),
        "meta": meta or {},
        "colors": {
            "target": [0, 200, 0],
            "positive_prompt": [255, 255, 0],
            "negative_prompt": [0, 220, 255],
        },
    }
    if components is not None:
        selected = np.unique(components[target])
        sidecar["target_component_labels"] = [int(v) for v in selected if int(v) > 0]
        sidecar["component_labels_on_slice"] = [
            int(v) for v in np.unique(components[z]) if int(v) > 0
        ]
    with path.with_suffix(".json").open("w", encoding="utf-8") as f:
        json.dump(sidecar, f, indent=2, sort_keys=True)
    return path


def _normalize_prompt_points(
    prompt_points_zyx: np.ndarray | None,
    prompt_labels: np.ndarray | None,
    prompt_zyx: tuple[int, int, int] | None,
) -> tuple[np.ndarray, np.ndarray]:
    if prompt_points_zyx is None:
        if prompt_zyx is None:
            return np.zeros((0, 3), dtype=np.float32), np.zeros((0,), dtype=np.int64)
        return np.asarray([prompt_zyx], dtype=np.float32), np.ones((1,), dtype=np.int64)
    points = np.asarray(prompt_points_zyx, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"prompt_points_zyx must have shape [N, 3], got {points.shape}")
    if prompt_labels is None:
        labels = np.ones((points.shape[0],), dtype=np.int64)
    else:
        labels = np.asarray(prompt_labels, dtype=np.int64).reshape(-1)
    if labels.shape[0] != points.shape[0]:
        raise ValueError(
            "prompt_labels length must match prompt_points_zyx: "
            f"{labels.shape[0]} != {points.shape[0]}")
    return points, labels


def _primary_positive_prompt(
    points: np.ndarray,
    labels: np.ndarray,
) -> tuple[int, int, int] | None:
    if points.shape[0] == 0:
        return None
    positive = np.flatnonzero(labels > 0)
    if positive.size == 0:
        return None
    point = points[int(positive[0])]
    return tuple(int(round(float(value))) for value in point)


def _tolerant_error_masks(
    gt: np.ndarray,
    pred: np.ndarray,
    *,
    tolerance_voxels: float,
) -> dict[str, np.ndarray]:
    gt = np.asarray(gt, dtype=bool)
    pred = np.asarray(pred, dtype=bool)
    if gt.shape != pred.shape:
        raise ValueError(f"Expected matching GT/pred shapes, got {gt.shape} and {pred.shape}")
    fp = pred & ~gt
    fn = gt & ~pred
    if gt.any():
        dist_to_gt = ndimage.distance_transform_edt(~gt)
    else:
        dist_to_gt = np.full(gt.shape, np.inf, dtype=np.float32)
    if pred.any():
        dist_to_pred = ndimage.distance_transform_edt(~pred)
    else:
        dist_to_pred = np.full(gt.shape, np.inf, dtype=np.float32)
    tolerance = float(tolerance_voxels)
    return {
        "tp": gt & pred,
        "near_fp": fp & (dist_to_gt <= tolerance),
        "far_fp": fp & (dist_to_gt > tolerance),
        "near_fn": fn & (dist_to_pred <= tolerance),
        "far_fn": fn & (dist_to_pred > tolerance),
    }


def _choose_error_slice(errors: dict[str, np.ndarray]) -> int:
    far = errors["far_fp"] | errors["far_fn"]
    support = far if far.any() else (
        errors["near_fp"] | errors["near_fn"] | errors["tp"])
    if support.any():
        counts = support.reshape(support.shape[0], -1).sum(axis=1)
        return int(counts.argmax())
    return int(support.shape[0] // 2)


def _tolerant_error_panel(
    base: np.ndarray,
    errors: dict[str, np.ndarray],
    z: int,
) -> Image.Image:
    return _overlay_masks(
        base,
        [
            (errors["tp"][z], (0, 200, 0)),
            (errors["near_fp"][z], (255, 210, 0)),
            (errors["near_fn"][z], (0, 220, 220)),
            (errors["far_fp"][z], (255, 45, 45)),
            (errors["far_fn"][z], (0, 120, 255)),
        ],
    )


def _topology_panel(
    base: np.ndarray,
    prompt_component: np.ndarray,
    detached_component: np.ndarray,
) -> Image.Image:
    return _overlay_masks(
        base,
        [
            (prompt_component, (0, 200, 180)),
            (detached_component, (220, 0, 220)),
        ],
    )


def _projected_tolerant_error_panel(
    base: np.ndarray,
    errors: dict[str, np.ndarray],
    *,
    axis: int,
) -> Image.Image:
    return _overlay_masks(
        base,
        [
            (np.any(errors["tp"], axis=axis), (0, 200, 0)),
            (np.any(errors["near_fp"], axis=axis), (255, 210, 0)),
            (np.any(errors["near_fn"], axis=axis), (0, 220, 220)),
            (np.any(errors["far_fp"], axis=axis), (255, 45, 45)),
            (np.any(errors["far_fn"], axis=axis), (0, 120, 255)),
        ],
    )


def _projected_topology_panel(
    base: np.ndarray,
    prompt_component: np.ndarray,
    detached_component: np.ndarray,
    *,
    axis: int,
) -> Image.Image:
    return _topology_panel(
        base,
        np.any(prompt_component, axis=axis),
        np.any(detached_component, axis=axis),
    )


def _overlay_masks(
    base: np.ndarray,
    masks_and_colors: list[tuple[np.ndarray, tuple[int, int, int]]],
) -> Image.Image:
    rgb = Image.fromarray(np.asarray(base, dtype=np.uint8), mode="L").convert("RGB")
    arr = np.asarray(rgb).copy()
    for mask, color in masks_and_colors:
        mask = np.asarray(mask, dtype=bool)
        if mask.any():
            arr[mask] = (
                0.45 * arr[mask] + 0.55 * np.asarray(color, dtype=np.uint8)
            ).astype(np.uint8)
    return Image.fromarray(arr, mode="RGB")


def _draw_projected_prompt_points(
    panel: Image.Image,
    points_zyx: np.ndarray,
    labels: np.ndarray,
    *,
    axis: int,
) -> None:
    if points_zyx.ndim != 2 or points_zyx.shape[1] != 3:
        return
    draw = ImageDraw.Draw(panel)
    width, height = panel.size
    for point, label in zip(points_zyx, labels):
        z, y, x = (int(round(float(value))) for value in point)
        if axis == 0:
            px, py = x, y
        elif axis == 1:
            px, py = x, z
        elif axis == 2:
            px, py = y, z
        else:
            raise ValueError(f"Unsupported projection axis: {axis}")
        if not (0 <= px < width and 0 <= py < height):
            continue
        color = (255, 255, 0) if int(label) > 0 else (0, 220, 255)
        radius = 5 if int(label) > 0 else 4
        draw.ellipse((px - radius, py - radius, px + radius, py + radius), fill=color, outline=(0, 0, 0), width=1)


def _label_panels(panels: list[Image.Image], labels: list[str]) -> list[Image.Image]:
    if len(panels) != len(labels):
        raise ValueError("Each panel needs exactly one label")
    title_h = 18
    labeled = []
    for panel, label in zip(panels, labels):
        canvas = Image.new("RGB", (panel.width, panel.height + title_h), "white")
        canvas.paste(panel, (0, title_h))
        ImageDraw.Draw(canvas).text((4, 2), label, fill=(0, 0, 0))
        labeled.append(canvas)
    return labeled


def _stack_labeled_rows(rows: list[tuple[str, list[Image.Image]]]) -> Image.Image:
    if not rows or not rows[0][1]:
        raise ValueError("At least one labeled panel row is required")
    row_label_w = 90
    row_widths = [sum(panel.width for panel in panels) for _, panels in rows]
    width = row_label_w + max(row_widths)
    height = sum(max(panel.height for panel in panels) for _, panels in rows)
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    y = 0
    for row_label, panels in rows:
        row_h = max(panel.height for panel in panels)
        draw.text((4, y + 2), row_label, fill=(0, 0, 0))
        x = row_label_w
        for panel in panels:
            canvas.paste(panel, (x, y))
            x += panel.width
        y += row_h
    return canvas


def _diagnostic_color_legend() -> dict[str, list[int]]:
    return {
        "gt_or_true_positive": [0, 200, 0],
        "ae_reconstruction": [170, 80, 255],
        "prediction": [255, 80, 0],
        "near_false_positive": [255, 210, 0],
        "far_false_positive": [255, 45, 45],
        "near_false_negative": [0, 220, 220],
        "far_false_negative": [0, 120, 255],
        "prompt_connected_component": [0, 200, 180],
        "detached_prediction_component": [220, 0, 220],
        "positive_prompt": [255, 255, 0],
        "negative_prompt": [0, 220, 255],
    }


def _append_color_legend(
    canvas: Image.Image,
    colors: dict[str, tuple[int, int, int] | list[int]],
) -> Image.Image:
    """Append a compact legend so a PNG remains interpretable without its JSON."""
    entries = list(colors.items())
    row_height = 22
    columns = 2
    rows = (len(entries) + columns - 1) // columns
    legend_width = max(canvas.width, 360)
    legend = Image.new("RGB", (legend_width, rows * row_height + 6), "white")
    draw = ImageDraw.Draw(legend)
    for index, (label, color) in enumerate(entries):
        row, column = divmod(index, columns)
        x = column * (legend_width // columns) + 6
        y = row * row_height + 4
        rgb = tuple(int(value) for value in color)
        draw.rectangle((x, y, x + 12, y + 12), fill=rgb, outline=(0, 0, 0))
        draw.text((x + 17, y - 1), label.replace("_", " "), fill=(0, 0, 0))
    result = Image.new("RGB", (max(canvas.width, legend.width), canvas.height + legend.height), "white")
    result.paste(canvas, (0, 0))
    result.paste(legend, (0, canvas.height))
    return result


def _save_png(canvas: Image.Image, path: Path, scale: int = 2) -> None:
    """Save readable inspection PNGs at a stable 2x raster resolution."""
    if scale > 1:
        canvas = canvas.resize((canvas.width * scale, canvas.height * scale), Image.Resampling.LANCZOS)
    canvas.save(path)


def _write_sidecar(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def _as_3d(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x)
    while arr.ndim > 3:
        arr = arr[0]
    if arr.ndim != 3:
        raise ValueError(f"Expected 3D volume or leading singleton dims, got {x.shape}")
    return arr


def _choose_slice(
    gt: np.ndarray,
    pred: np.ndarray,
    prompt_zyx: tuple[int, int, int] | None,
) -> int:
    if prompt_zyx is not None and 0 <= int(prompt_zyx[0]) < gt.shape[0]:
        return int(prompt_zyx[0])
    support = gt | pred
    if support.any():
        counts = support.reshape(support.shape[0], -1).sum(axis=1)
        return int(counts.argmax())
    return gt.shape[0] // 2


def _choose_train_slice(
    target: np.ndarray,
    points_zyx: np.ndarray,
    labels: np.ndarray,
) -> int:
    if points_zyx.ndim == 2 and points_zyx.shape[1] == 3 and points_zyx.shape[0] > 0:
        positive = np.flatnonzero(labels > 0)
        idx = int(positive[0]) if positive.size else 0
        z = int(round(float(points_zyx[idx, 0])))
        if 0 <= z < target.shape[0]:
            return z
    if target.any():
        counts = target.reshape(target.shape[0], -1).sum(axis=1)
        return int(counts.argmax())
    return target.shape[0] // 2


def _normalize_to_uint8(img: np.ndarray) -> np.ndarray:
    arr = np.asarray(img, dtype=np.float32)
    lo, hi = np.percentile(arr, [1, 99])
    if hi <= lo:
        hi = float(arr.max())
        lo = float(arr.min())
    if hi <= lo:
        return np.zeros(arr.shape, dtype=np.uint8)
    arr = np.clip((arr - lo) / (hi - lo), 0, 1)
    return (arr * 255).astype(np.uint8)


def _mask_panel(base: np.ndarray, mask: np.ndarray, *, color: tuple[int, int, int]) -> Image.Image:
    rgb = Image.fromarray(base, mode="L").convert("RGB")
    arr = np.asarray(rgb).copy()
    mask = np.asarray(mask, dtype=bool)
    overlay = np.asarray(color, dtype=np.uint8)
    arr[mask] = (0.45 * arr[mask] + 0.55 * overlay).astype(np.uint8)
    return Image.fromarray(arr, mode="RGB")


def _error_panel(base: np.ndarray, gt: np.ndarray, pred: np.ndarray) -> Image.Image:
    rgb = Image.fromarray(base, mode="L").convert("RGB")
    arr = np.asarray(rgb).copy()
    gt = np.asarray(gt, dtype=bool)
    pred = np.asarray(pred, dtype=bool)
    fp = pred & ~gt
    fn = gt & ~pred
    tp = gt & pred
    arr[tp] = (0.45 * arr[tp] + np.array([0, 200, 0]) * 0.55).astype(np.uint8)
    arr[fp] = (0.45 * arr[fp] + np.array([255, 80, 0]) * 0.55).astype(np.uint8)
    arr[fn] = (0.45 * arr[fn] + np.array([0, 120, 255]) * 0.55).astype(np.uint8)
    return Image.fromarray(arr, mode="RGB")


def _component_panel(base: np.ndarray, labels: np.ndarray) -> Image.Image:
    rgb = Image.fromarray(base, mode="L").convert("RGB")
    arr = np.asarray(rgb).copy()
    labels = np.asarray(labels)
    palette = np.asarray([
        [0, 200, 0],
        [255, 80, 0],
        [0, 120, 255],
        [220, 0, 220],
        [255, 220, 0],
        [0, 220, 220],
        [255, 140, 0],
        [120, 80, 255],
    ], dtype=np.uint8)
    mask = labels > 0
    if mask.any():
        colors = palette[(labels[mask].astype(np.int64) - 1) % len(palette)]
        arr[mask] = (0.35 * arr[mask] + 0.65 * colors).astype(np.uint8)
    return Image.fromarray(arr, mode="RGB")


def _draw_prompt(panel: Image.Image, xy: tuple[int, int]) -> None:
    draw = ImageDraw.Draw(panel)
    x, y = xy
    r = 4
    draw.ellipse((x - r, y - r, x + r, y + r), outline=(255, 255, 0), width=2)


def _draw_prompt_points(
    panel: Image.Image,
    points_zyx: np.ndarray,
    labels: np.ndarray,
    *,
    slice_z: int,
) -> None:
    if points_zyx.ndim != 2 or points_zyx.shape[1] != 3:
        return
    draw = ImageDraw.Draw(panel)
    for point, label in zip(points_zyx, labels):
        z, y, x = (int(round(float(v))) for v in point)
        if z != int(slice_z):
            continue
        color = (255, 255, 0) if int(label) > 0 else (0, 220, 255)
        r = 5 if int(label) > 0 else 4
        draw.ellipse((x - r, y - r, x + r, y + r), fill=color, outline=(0, 0, 0), width=1)


def _save_error_point_cloud_ply(
    path: Path,
    *,
    gt_mask: np.ndarray,
    pred_mask: np.ndarray,
    prompt_zyx: tuple[int, int, int] | None,
    max_points: int,
) -> None:
    tp = gt_mask & pred_mask
    fp = pred_mask & ~gt_mask
    fn = gt_mask & ~pred_mask
    coords_rgb = [
        (_mask_coords_rgb(tp, (0, 200, 0)), "tp"),
        (_mask_coords_rgb(fp, (255, 80, 0)), "fp"),
        (_mask_coords_rgb(fn, (0, 120, 255)), "fn"),
    ]
    points = [coords for coords, _ in coords_rgb if coords.size]
    cloud = np.concatenate(points, axis=0) if points else np.empty((0, 6), dtype=np.float32)
    if prompt_zyx is not None:
        z, y, x = (int(v) for v in prompt_zyx)
        if 0 <= z < gt_mask.shape[0] and 0 <= y < gt_mask.shape[1] and 0 <= x < gt_mask.shape[2]:
            prompt = np.asarray([[x, y, z, 255, 255, 0]], dtype=np.float32)
            cloud = np.concatenate([prompt, cloud], axis=0)
    cloud = _limit_points_deterministic(cloud, max_points=max_points)
    _write_ascii_ply(path, cloud)


def _save_mask_point_cloud_ply(
    path: Path,
    *,
    mask: np.ndarray,
    color: tuple[int, int, int],
    prompt_zyx: tuple[int, int, int] | None,
    max_points: int,
) -> None:
    cloud = _mask_coords_rgb(mask, color)
    if prompt_zyx is not None:
        z, y, x = (int(v) for v in prompt_zyx)
        if 0 <= z < mask.shape[0] and 0 <= y < mask.shape[1] and 0 <= x < mask.shape[2]:
            prompt = np.asarray([[x, y, z, 255, 255, 0]], dtype=np.float32)
            cloud = np.concatenate([prompt, cloud], axis=0)
    cloud = _limit_points_deterministic(cloud, max_points=max_points)
    _write_ascii_ply(path, cloud)


def _mask_coords_rgb(mask: np.ndarray, rgb: tuple[int, int, int]) -> np.ndarray:
    coords_zyx = np.argwhere(mask)
    if coords_zyx.size == 0:
        return np.empty((0, 6), dtype=np.float32)
    coords_xyz = coords_zyx[:, [2, 1, 0]].astype(np.float32, copy=False)
    colors = np.broadcast_to(np.asarray(rgb, dtype=np.float32), (coords_xyz.shape[0], 3))
    return np.concatenate([coords_xyz, colors], axis=1)


def _limit_points_deterministic(cloud: np.ndarray, *, max_points: int) -> np.ndarray:
    max_points = int(max_points)
    if max_points <= 0 or cloud.shape[0] <= max_points:
        return cloud
    idx = np.linspace(0, cloud.shape[0] - 1, num=max_points, dtype=np.int64)
    return cloud[idx]


def _write_ascii_ply(path: Path, cloud: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {cloud.shape[0]}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")
        for x, y, z, r, g, b in cloud:
            f.write(f"{x:.3f} {y:.3f} {z:.3f} {int(r)} {int(g)} {int(b)}\n")
