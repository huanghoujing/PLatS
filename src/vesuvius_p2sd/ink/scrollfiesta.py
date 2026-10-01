"""P2SD probability NIfTI -> ScrollFiesta CLI grid, preserving voxel coordinates.

One grid is one prompted sheet. Do not combine independently predicted instances
into a binary union. CLI OBJ columns are ZYX; exported interoperability OBJ/TIFXYZ
coordinates are XYZ in the source scan's level-0 voxel frame.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import time

import nibabel as nib
import numpy as np
import tifffile
from scipy import ndimage, sparse
from scipy.sparse.csgraph import connected_components

from .baseline import sha256, save_nifti
from .geometry import render, geometry_stats


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def load_zyx(path):
    nii = nib.load(path)
    return np.asarray(nii.dataobj).transpose(2, 1, 0), nii


def save_like(path, array, template):
    out = nib.Nifti1Image(np.ascontiguousarray(array.transpose(2, 1, 0)), template.affine)
    out.header.set_xyzt_units("mm")
    out.set_sform(template.affine, code=2)
    out.set_qform(template.affine, code=0)
    nib.save(out, path)


def write_grid(raw, probability, out, threshold=0.5, cube_size=128):
    """Write exact little-endian multipage ZYX TIFFs; no implicit padding."""
    out = Path(out)
    if raw.ndim != 3 or raw.shape != probability.shape:
        raise ValueError("Raw and probability volumes must have identical 3D shapes")
    if raw.dtype != np.uint8 or any(n % cube_size for n in raw.shape):
        raise ValueError("Raw must be uint8 and dimensions divisible by cube size")
    if not 0 < threshold < 1 or not np.isfinite(probability).all():
        raise ValueError("Invalid probability or threshold")
    if probability.min() < 0 or probability.max() > 1:
        raise ValueError("Expected probabilities in [0, 1]")
    for kind in ("RAW", "PRED"):
        (out / f"cubes_{kind}").mkdir(parents=True, exist_ok=True)
    ids = []
    for z in range(0, raw.shape[0], cube_size):
        for y in range(0, raw.shape[1], cube_size):
            for x in range(0, raw.shape[2], cube_size):
                region = np.s_[z:z+cube_size, y:y+cube_size, x:x+cube_size]
                cube_id = f"z{z:05d}_y{y:05d}_x{x:05d}"
                pred = (probability[region] >= threshold).astype(np.uint8) * 255
                for kind, data in (("RAW", raw[region]), ("PRED", pred)):
                    if kind == "PRED" and not data.any():
                        continue
                    tifffile.imwrite(out / f"cubes_{kind}" / f"{cube_id}.tif",
                                     np.ascontiguousarray(data), byteorder="<",
                                     photometric="minisblack", compression=None,
                                     rowsperstrip=cube_size)
                if pred.any():
                    ids.append(cube_id)
    write_json(out / "cubes_PRED/present.json", ids)
    return ids


def prepare(source, out, points=1, crop_size=256, threshold=0.5):
    source, out = Path(source), Path(out)
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite {out}")
    raw, nii = load_zyx(source / "native/image.nii.gz")
    prob_path = source / f"native/pred_p2sd_{points}points.nii.gz"
    prob, prob_nii = load_zyx(prob_path)
    observed, obs_nii = load_zyx(source / "native/observed.nii.gz")
    if raw.shape != prob.shape or observed.shape != raw.shape:
        raise ValueError("Input shape mismatch")
    for other in (prob_nii, obs_nii):
        if not np.allclose(nii.affine, other.affine, atol=1e-6):
            raise ValueError("Input affine mismatch")
    start = (np.array(raw.shape) - crop_size) // 2
    if (start < 0).any() or crop_size % 128:
        raise ValueError("Crop must fit input and be divisible by 128")
    region = tuple(slice(int(a), int(a+crop_size)) for a in start)
    if not observed[region].all():
        raise ValueError("This adapter requires completely observed CT context")
    if raw.dtype != np.uint8:
        raise ValueError("Expected the original uint8 CT, without rescaling")
    metadata = json.loads((source / "prepared.json").read_text())
    prompts = np.array(json.loads((source / "native/points.json").read_text())["points_zyx"][:points])
    if len(prompts) != points or not (((prompts-start) >= 0) & ((prompts-start) < crop_size)).all():
        raise ValueError("Requested prompts must exist and lie inside the crop")
    out.mkdir(parents=True)
    ids = write_grid(raw[region], prob[region], out / "grid", threshold)
    affine = nii.affine.copy()
    affine[:3, 3] = nib.affines.apply_affine(nii.affine, start[::-1])
    template = nib.Nifti1Image(np.empty((1, 1, 1)), affine)
    native = out / "native"
    native.mkdir()
    for name, array in (("image", raw[region]), ("pred_probability", prob[region]),
                        ("pred_binary", (prob[region] >= threshold).astype(np.uint8)),
                        ("observed", observed[region].astype(np.uint8))):
        save_like(native / f"{name}.nii.gz", array, template)
    point_volume = np.zeros((crop_size,)*3, np.uint8)
    for i, p in enumerate(prompts-start):
        point_volume[tuple(np.rint(p).astype(int))] = i+1
    save_like(native / "points.nii.gz", point_volume, template)
    gt_path = source / "native/gt_sheet_reference.nii.gz"
    if gt_path.exists():
        gt, gt_nii = load_zyx(gt_path)
        if gt.shape != raw.shape or not np.allclose(gt_nii.affine, nii.affine):
            raise ValueError("Reference overlay grid mismatch")
        save_like(native / "gt_partial_reference.nii.gz", gt[region].astype(np.uint8), template)
    manifest = {
        "status": "prepared", "source_patch": str(source.resolve()), "prompt_count": points,
        "threshold": threshold, "crop_start_zyx": start.tolist(), "crop_size": crop_size,
        "grid_origin_source_level2_zyx": (np.array(metadata["origin_zyx"])+start).tolist(),
        "source_level0_from_level2": "(xyz_level2 + 0.375) * 4",
        "grid_xyz_to_physical_mm": affine.tolist(), "spacing_um": metadata["spacing_um"],
        "prompt_grid_zyx": (prompts-start).tolist(), "cube_size": 128,
        "prediction_cubes": ids, "input_sha256": {
            str(p.relative_to(source)): sha256(p) for p in
            [source / "native/image.nii.gz", prob_path, source / "native/points.json"]},
        "scope": "Development-only single-sheet integration. Reference geometry is an overlay only; no reference UV or ink label enters meshing or flattening.",
        "context": "Centered 256-cube from cached 320-cube; raw and prediction outside this crop are not supplied to mesher. Rendering rejects samples outside the crop.",
    }
    write_json(out / "manifest.json", manifest)
    write_json(out / "grid/manifest.json", {"chunk_size": 128, "shape_zyx": [crop_size]*3,
                                            "coordinate_frame": "local crop voxels"})
    return manifest


def run_mesh(out, build, upstream):
    out, build, upstream = Path(out), Path(build).resolve(), Path(upstream).resolve()
    manifest = json.loads((out / "manifest.json").read_text())
    revision = subprocess.check_output(["git", "-C", str(upstream), "rev-parse", "HEAD"], text=True).strip()
    if revision != "5d957e9c21529580cc43eda8bf980fef02486b36":
        raise ValueError("Unexpected ScrollFiesta revision")
    if subprocess.check_output(["git", "-C", str(upstream), "status", "--porcelain"], text=True).strip():
        raise ValueError("ScrollFiesta checkout has modifications")
    if (out / "meshing").exists():
        raise FileExistsError("Meshing output exists; use a fresh run")
    command = [str(build / "grid_pipeline"), str((out / "grid").resolve()),
               str((out / "meshing").resolve()), "--halo", "13",
               "--threads-per-cube", "4", "--max-concurrent", "2",
               "--exe", str(build / "cube_mesh"), "--weld", str(build / "grid_weld")]
    manifest.update(status="meshing", upstream_revision=revision, command=command,
                    executable_sha256={name: sha256(build/name) for name in
                                       ("cube_mesh", "grid_pipeline", "grid_weld")})
    write_json(out / "manifest.json", manifest)
    started = time.monotonic()
    with (out / "meshing.log").open("w") as log:
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=3600)
    manifest.update(status="meshed" if result.returncode == 0 else "mesh_failed",
                    mesh_exit_code=result.returncode, mesh_seconds=time.monotonic()-started)
    write_json(out / "manifest.json", manifest)
    if result.returncode:
        raise RuntimeError(f"ScrollFiesta failed; see {out / 'meshing.log'}")


def read_cli_obj(path):
    """Convert CLI ZYX to XYZ and reverse winding under the odd permutation."""
    vertices, faces = [], []
    for line in Path(path).read_text().splitlines():
        parts = line.split()
        if parts and parts[0] == "v":
            vertices.append([float(v) for v in parts[1:4]][::-1])
        elif parts and parts[0] == "f":
            if len(parts) != 4:
                raise ValueError("Expected triangular OBJ")
            faces.append([int(parts[i].split('/')[0])-1 for i in (1, 3, 2)])
    vertices, faces = np.array(vertices), np.array(faces, dtype=np.int32)
    if not len(vertices) or not len(faces) or faces.min() < 0 or faces.max() >= len(vertices):
        raise ValueError("Empty or invalid mesh")
    return vertices, faces


def mesh_audit(vertices, faces):
    edges = np.sort(np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]), axis=1)
    edges, counts = np.unique(edges, axis=0, return_counts=True)
    graph = sparse.coo_matrix((np.ones(len(edges)), (edges[:, 0], edges[:, 1])),
                              shape=(len(vertices),)*2).tocsr()
    _, labels = connected_components(graph, directed=False)
    used = np.unique(faces)
    return {"vertices": len(vertices), "faces": len(faces),
            "components": len(np.unique(labels[used])),
            "boundary_edges": int((counts == 1).sum()),
            "nonmanifold_edges": int((counts > 2).sum()),
            "euler_characteristic": int(len(used)-len(edges)+len(faces))}


def local_chart(vertices, faces):
    """A local PCA chart for the integration smoke test, not whole-scroll unroll.

    Rasterize actual triangles (never a Delaunay fill across holes). Exclude UV
    collisions when two covering faces disagree by more than two native voxels.
    """
    center = vertices.mean(0)
    _, _, axes = np.linalg.svd(vertices-center, full_matrices=False)
    for axis in axes:
        if axis[np.abs(axis).argmax()] < 0:
            axis *= -1
    normal = np.cross(axes[0], axes[1])
    uv = (vertices-center) @ axes[:2].T
    uv -= uv.min(0)
    width, height = np.ceil(uv.max(0)).astype(int)+1
    if max(width, height) > 2048:
        raise ValueError("Local chart exceeds bounded pilot size")
    xyz = np.zeros((height, width, 3), np.float32)
    normals = np.zeros_like(xyz)
    hits = np.zeros((height, width), bool)
    ambiguous = np.zeros_like(hits)
    degenerate = 0
    for face in faces:
        p = uv[face]
        a, b = p[1]-p[0], p[2]-p[0]
        determinant = a[0]*b[1]-a[1]*b[0]
        if abs(determinant) < 1e-8:
            degenerate += 1
            continue
        lo = np.maximum(np.floor(p.min(0)).astype(int), 0)
        hi = np.minimum(np.ceil(p.max(0)).astype(int), [width-1, height-1])
        yy, xx = np.mgrid[lo[1]:hi[1]+1, lo[0]:hi[0]+1]
        qx, qy = xx-p[0, 0], yy-p[0, 1]
        s = (qx*b[1]-qy*b[0])/determinant
        t = (a[0]*qy-a[1]*qx)/determinant
        inside = (s >= -1e-7) & (t >= -1e-7) & (s+t <= 1+1e-7)
        y, x = yy[inside], xx[inside]
        values = ((1-s[inside]-t[inside])[:, None]*vertices[face[0]]
                  + s[inside, None]*vertices[face[1]] + t[inside, None]*vertices[face[2]])
        ambiguous[y, x] |= hits[y, x] & (np.linalg.norm(xyz[y, x]-values, axis=-1) > 2)
        n = np.cross(vertices[face[1]]-vertices[face[0]], vertices[face[2]]-vertices[face[0]])
        n /= max(np.linalg.norm(n), 1e-12)
        if n @ normal < 0:
            n *= -1
        xyz[y, x], normals[y, x], hits[y, x] = values, n, True
    valid = hits & ~ambiguous
    return uv, xyz, normals, valid, {
        "method": "local PCA projection of ScrollFiesta triangles; not native whole-scroll registration",
        "shape_yx": [int(height), int(width)], "ambiguous_pixels": int(ambiguous.sum()),
        "covered_pixels": int(hits.sum()), "degenerate_projected_faces": degenerate,
        "basis_xyz": axes[:2].tolist(), "normal_xyz": normal.tolist(),
        **geometry_stats(xyz, ndimage.binary_erosion(valid, iterations=2)),
    }


def finish(out, ink_settings=None):
    out = Path(out)
    manifest = json.loads((out / "manifest.json").read_text())
    if manifest["status"] not in ("meshed", "local_integration_complete"):
        raise ValueError("A successful mesh run is required")
    vertices, faces = read_cli_obj(out / "meshing/welded.obj")
    audit = mesh_audit(vertices, faces)
    upstream_audit = json.loads((out / "meshing/welded.obj.weld_report.json").read_text())
    # Do not certify nonmanifold output merely because grid_pipeline returned 0.
    if audit["nonmanifold_edges"]:
        write_json(out / "mesh_audit.json", audit)
        raise ValueError("Nonmanifold mesh; retained for diagnosis, not promoted")
    if not upstream_audit["embedded_geometry"]["certificate"]:
        raise ValueError("Upstream embedded-geometry certificate failed")
    uv, xyz, normals, valid, chart = local_chart(vertices, faces)
    if not valid.any():
        raise ValueError("No unambiguous chart pixels")
    raw, nii = load_zyx(out / "native/image.nii.gz")
    flat = out / "flat"
    segment = out / "tifxyz"
    flat.mkdir(exist_ok=True)
    segment.mkdir(exist_ok=True)
    origin_xyz = np.array(manifest["grid_origin_source_level2_zyx"])[::-1]
    level0_xyz = (xyz+origin_xyz+0.375)*4
    for i, axis in enumerate("xyz"):
        tifffile.imwrite(segment / f"{axis}.tif", np.where(valid, level0_xyz[..., i], -1).astype(np.float32))
    tifffile.imwrite(segment / "mask.tif", valid.astype(np.uint8)*255)
    write_json(segment / "meta.json", {"type": "seg", "format": "tifxyz", "uuid": out.name,
               "name": f"p2sd_scrollfiesta_{out.name}", "width": valid.shape[1],
               "height": valid.shape[0], "scale": [0.25, 0.25],
               "bbox": [level0_xyz[valid].min(0).tolist(), level0_xyz[valid].max(0).tolist()],
               "coordinate_frame": "source scan level-0 XYZ voxels",
               "sampling_step_source_voxels": 4,
               "scale_convention": "stored/full-resolution UV dimension, as read by villa tifxyz_label_transfer",
               "local_chart": True})
    with (out / "mesh_source_level0_xyz.obj").open("w") as stream:
        for v in (vertices+origin_xyz+0.375)*4:
            stream.write("v " + " ".join(f"{c:.6f}" for c in v) + "\n")
        for p in uv*4:
            stream.write(f"vt {p[0]:.6f} {p[1]:.6f}\n")
        for face in faces+1:
            stream.write("f " + " ".join(f"{v}/{v}" for v in face) + "\n")
    # Dense triangle samples give a native overlay; no filled volume is claimed.
    surface = np.zeros(raw.shape, np.uint8)
    for face in faces:
        tri = vertices[face]
        steps = max(1, int(np.ceil(max(np.linalg.norm(tri[i]-tri[j]) for i,j in [(0,1),(1,2),(2,0)]))*2))
        a, b = np.meshgrid(np.arange(steps+1)/steps, np.arange(steps+1)/steps)
        keep = a+b <= 1
        points = tri[0]+a[keep,None]*(tri[1]-tri[0])+b[keep,None]*(tri[2]-tri[0])
        indices = np.rint(points[:, ::-1]).astype(int)
        inside = ((indices >= 0) & (indices < np.array(raw.shape))).all(1)
        surface[tuple(indices[inside].T)] = 1
    save_like(out / "native/pred_scrollfiesta_surface.nii.gz", surface, nii)
    # TIFXYZ stores grid vertices; a native-resolution rendered pixel is at
    # (y+0.5, x+0.5), matching villa's public surface renderer convention.
    xyz, normals, valid = sample_chart_centers(xyz, normals, valid)
    np.savez_compressed(out / "render_geometry.npz", xyz_grid=xyz, normals=normals, valid=valid)
    offsets = np.arange(21)-9.875
    viz = {"spacing_zyx_um": [manifest["spacing_um"]]*3, "origin_yx": [0, 0]}
    from scipy.spatial import cKDTree
    yy, xx = np.where(valid)
    distances, nearest = cKDTree(xyz[valid]).query(np.array(manifest["prompt_grid_zyx"])[:, ::-1])
    point_map = np.zeros(valid.shape, np.uint8)
    for i, (distance, index) in enumerate(zip(distances, nearest)):
        if distance <= 4:
            point_map[yy[index], xx[index]] = i+1
    save_nifti(flat / "points.nii.gz", np.repeat(point_map[None], 21, axis=0), viz)
    settings = json.loads(Path(ink_settings).read_text()) if ink_settings else None
    model = None
    if settings:
        from .model import PublishedInk
        model = PublishedInk(settings)
        write_json(out / "ink_settings.json", settings)
    coverage = {}
    # Recto side is not identifiable from an unsigned sheet probability. Export
    # both fixed orientations, without selecting the stronger ink prediction.
    for name, sign in (("normal_plus", 1), ("normal_minus", -1)):
        stack, supported = render(raw, xyz, sign*normals, [0, 0, 0], offsets)
        support = valid & supported
        stack[:, ~support] = 0
        stack = np.clip(np.rint(stack), 0, 255).astype(np.uint8)
        np.save(flat / f"image_{name}.npy", stack)
        save_nifti(flat / f"image_{name}.nii.gz", stack, viz)
        save_nifti(flat / f"valid_{name}.nii.gz", np.repeat(support[None], 21, axis=0).astype(np.uint8), viz)
        coverage[name] = {"fully_observed_depth_pixels": int(support.sum())}
        if model:
            pred = model.predict(stack, start=4)
            pred[~support] = 0
            np.save(flat / f"pred_ink_{name}.npy", pred)
            save_nifti(flat / f"pred_ink_{name}.nii.gz", np.repeat(pred[None], 21, axis=0), viz)
    metrics = {"mesh": audit, "upstream_audit": upstream_audit, "chart": chart, "render": coverage,
               "prompt_to_raster_distance_voxels": distances.tolist(),
               "prompts_within_4_voxels": int((distances <= 4).sum()),
               "ink_accuracy": None, "letter_readability": None,
               "limitations": ["Local PCA chart; whole-scroll registration not run without a calibrated axis.",
                               "Independent UV has no transferred ink GT. Predictions are exploratory.",
                               "Both normal directions retained; recto orientation unresolved.",
                               "Ink model context can contain masked UV holes and crop borders; no accuracy is claimed.",
                               "Topology counts do not certify absence of self-intersections."]}
    write_json(out / "metrics.json", metrics)
    manifest["status"] = "local_integration_complete"
    manifest["chart"] = chart["method"]
    write_json(out / "manifest.json", manifest)
    return metrics


def sample_chart_centers(xyz, normals, valid):
    yy, xx = np.mgrid[:valid.shape[0], :valid.shape[1]].astype(float)+0.5
    def interpolate(array):
        return ndimage.map_coordinates(array, [yy, xx], order=1, mode="constant", cval=0)
    sampled = np.stack([interpolate(xyz[..., i]) for i in range(3)], axis=-1)
    normal = np.stack([interpolate(normals[..., i]) for i in range(3)], axis=-1)
    length = np.linalg.norm(normal, axis=-1)
    supported = (interpolate(valid.astype(float)) > 0.99999) & (length > 1e-8)
    normal /= np.maximum(length[..., None], 1e-8)
    return sampled, normal, supported


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--points", type=int, choices=[1, 8], default=1)
    parser.add_argument("--stage", choices=["prepare", "mesh", "finish"], default="prepare")
    parser.add_argument("--ink-settings", type=Path)
    parser.add_argument("--build", type=Path, default=Path(".external/scrollfiesta-build"))
    parser.add_argument("--upstream", type=Path, default=Path(".external/scrollfiesta"))
    args = parser.parse_args()
    if args.stage == "prepare":
        if not args.source:
            parser.error("--source is required for prepare")
        prepare(args.source, args.out, args.points)
    elif args.stage == "mesh":
        run_mesh(args.out, args.build, args.upstream)
    else:
        finish(args.out, args.ink_settings)


if __name__ == "__main__":
    main()
