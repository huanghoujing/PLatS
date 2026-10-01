"""Consistent per-sheet/union scoring and inspectable NIFTI tuples."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import nibabel as nib
import numpy as np

from vesuvius_p2sd.eval.connected_components import component_count
from vesuvius_p2sd.eval.kaggle_surface import compute_kaggle_case_metrics_batch


def border_only_ignore(source_ignore, border_width):
    """Ignore the rectangular conversion shell, never inward unlabeled holes.

    Geometry, not connectivity to an edge, defines the shell: an unlabeled
    region connected to the edge but extending inward is still scored inside.
    Callers must treat the original inward ignore as GT background while
    leaving predictions there intact.
    """
    source_ignore = np.asarray(source_ignore, bool)
    width = int(border_width)
    if source_ignore.ndim != 3 or width < 0 or 2 * width >= min(source_ignore.shape):
        raise ValueError("Border width must leave a nonempty 3D interior")
    shell = np.zeros(source_ignore.shape, bool)
    if width:
        shell[:] = True
        shell[width:-width, width:-width, width:-width] = False
    return shell


def annotated_box_bounds(source_ignore):
    """Half-open ZYX bounds of originally non-ignored voxels, including BG."""
    valid = ~np.asarray(source_ignore, bool)
    if valid.ndim != 3 or not valid.any():
        raise ValueError("Source ignore must leave a nonempty 3D scoring region")
    bounds = []
    for axis in range(3):
        occupied = np.flatnonzero(np.any(valid, axis=tuple(i for i in range(3) if i != axis)))
        bounds.append((int(occupied[0]), int(occupied[-1] + 1)))
    return tuple(bounds)


def rectangular_border_ignore(source_ignore):
    """Retain ignored outer slabs; score every voxel inside the valid-data box."""
    bounds = annotated_box_bounds(source_ignore)
    shell = np.ones(np.shape(source_ignore), bool)
    shell[tuple(slice(lo, hi) for lo, hi in bounds)] = False
    return shell


def dice(pred, gt, valid=None):
    pred, gt = np.asarray(pred, bool), np.asarray(gt, bool)
    if valid is not None:
        pred, gt = pred & valid, gt & valid
    denominator = int(pred.sum()) + int(gt.sum())
    return float(2 * np.count_nonzero(pred & gt) / denominator) if denominator else 1.0


def score_masks(pred, gt, ignore, *, topology_backend="compact_exact", topology_tile_batch_size=1):
    """Public formula on identical raw masks/ignore, with no hidden cleanup.

    Used for both a prompted sheet and the full patch union. A per-sheet use of
    the public formula is a diagnostic, not an official instance leaderboard.
    """
    pred, gt, ignore = np.asarray(pred, bool), np.asarray(gt, bool), np.asarray(ignore, bool)
    if not (pred.shape == gt.shape == ignore.shape) or pred.ndim != 3:
        raise ValueError("prediction, GT and ignore must have matching 3D shapes")
    label = gt.astype(np.uint8)
    label[ignore] = 2
    result = compute_kaggle_case_metrics_batch([pred], [label],
        topology_backend=topology_backend, topology_tile_batch_size=topology_tile_batch_size)[0]
    return {
        "dice": result["kaggle_volumetric_dice"],
        "surface_dice_tau2": result["kaggle_surface_dice_tau2"],
        "leaderboard_formula_score": result["kaggle_case_score"],
        "toposcore": result["kaggle_toposcore"],
        "voi_score": result["kaggle_voi_score"],
        "voi_split": result["kaggle_voi_split"],
        "voi_merge": result["kaggle_voi_merge"],
        "pred_components_raw": int(component_count(pred)),
        "pred_components_scored": int(component_count(pred & ~ignore)),
        "gt_components_scored": int(component_count(gt & ~ignore)),
        "pred_voxels_scored": int(np.count_nonzero(pred & ~ignore)),
        "gt_voxels_scored": int(np.count_nonzero(gt & ~ignore)),
    }


def update_instances(instance_ids, best_probability, probability, mask, instance_id):
    """Assign overlaps using model confidence; equal confidence uses lower ID."""
    claim = mask & ((probability > best_probability) |
                    ((probability == best_probability) &
                     ((instance_ids == 0) | (instance_id < instance_ids))))
    instance_ids[claim] = instance_id
    best_probability[claim] = probability[claim]


def save_nifti(path, array):
    """Repository convention: array indices are ZYX; affine is voxel identity."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name("." + path.name + ".tmp.nii.gz")
    nib.save(nib.Nifti1Image(np.asarray(array), np.eye(4)), temporary)
    temporary.replace(path)


def shared_assets(root, case_id, image, ignore, source_paths, *, policy="source", source_ignore=None):
    signatures = []
    for value in source_paths:
        path = Path(value).resolve()
        stat = path.stat()
        signatures.append([str(path), stat.st_size, stat.st_mtime_ns])
    key = hashlib.sha256(json.dumps([signatures, policy]).encode()).hexdigest()[:16]
    target = Path(root) / case_id / key
    target.mkdir(parents=True, exist_ok=True)
    # Reuse the original CT cache across scoring policies as well as steps.
    image_key = hashlib.sha256(json.dumps(signatures).encode()).hexdigest()[:16]
    image_path = Path(root) / case_id / image_key / "image.nii.gz"
    if not image_path.exists():
        save_nifti(image_path, image)
    if not (target / "image.nii.gz").exists():
        (target / "image.nii.gz").symlink_to(os.path.relpath(image_path, target))
    if not (target / "ignore.nii.gz").exists():
        save_nifti(target / "ignore.nii.gz", np.asarray(ignore, np.uint8))
    if source_ignore is not None and not (target / "source_ignore.nii.gz").exists():
        save_nifti(target / "source_ignore.nii.gz", np.asarray(source_ignore, np.uint8))
    (target / "source.json").write_text(json.dumps({
        "source_signatures": signatures, "array_axes": "ZYX", "affine": "voxel identity",
        "shape": list(image.shape), "ignore_policy": policy}, indent=2) + "\n")
    return target


def link_assets(folder, assets):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    for name in ["image.nii.gz", "ignore.nii.gz"]:
        target = Path(assets) / name
        link = folder / name
        if not link.is_symlink():
            link.symlink_to(os.path.relpath(target, folder))
    if (Path(assets) / "source_ignore.nii.gz").exists():
        link = folder / "source_ignore.nii.gz"
        if not link.is_symlink():
            link.symlink_to(os.path.relpath(Path(assets) / link.name, folder))


def write_sheet_tuple(folder, *, assets, gt, prediction, points_zyx, labels, metadata):
    """One complete viewing tuple per sheet and prompt variant, at source shape."""
    folder = Path(folder)
    link_assets(folder, assets)
    points = np.zeros(gt.shape, np.uint8)
    coordinates = np.rint(points_zyx).astype(np.int64)
    if coordinates.shape != (len(labels), 3):
        raise ValueError("Prompt coordinates/labels differ")
    for coordinate, label in zip(coordinates, labels, strict=True):
        if label < 0:
            continue
        if np.any(coordinate < 0) or np.any(coordinate >= np.asarray(gt.shape)):
            raise ValueError("Prompt outside source volume after removing canvas offset")
        value = 1 if label > 0 else 2
        index = tuple(coordinate)
        if points[index] not in (0, value):
            raise ValueError("Conflicting prompt labels at one voxel")
        points[index] = value
    save_nifti(folder / "prompt_points.nii.gz", points)
    save_nifti(folder / "gt_sheet.nii.gz", np.asarray(gt, np.uint8))
    save_nifti(folder / "p2sd_decoded_sheet.nii.gz", np.asarray(prediction, np.uint8))
    (folder / "tuple.json").write_text(json.dumps({**metadata,
        "points_zyx": np.asarray(points_zyx).tolist(), "point_labels": list(labels),
        "prompt_values": {"positive": 1, "negative": 2}, "array_axes": "ZYX",
        "prediction": "raw decoded mask; ignore applied only when scoring",
        "files": ["image.nii.gz", "prompt_points.nii.gz", "gt_sheet.nii.gz",
                  "p2sd_decoded_sheet.nii.gz", "ignore.nii.gz"]}, indent=2) + "\n")
