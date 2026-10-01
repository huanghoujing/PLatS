"""Independent mesh -> SLIM UV -> CT layers -> frozen ink, with no reference UV.

Each connected component is flattened separately. No holes are filled to create
ink support. Both surface-normal directions are exported without label selection.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import igl
import numpy as np
import tifffile
from scipy import sparse
from scipy.sparse.linalg import spsolve

from .baseline import save_nifti
from .geometry import render
from .scrollfiesta import load_zyx, save_like, read_cli_obj, mesh_audit, write_json


def distortion(v, f, uv):
    tri = v[f]
    a, b = tri[:, 1]-tri[:, 0], tri[:, 2]-tri[:, 0]
    length = np.linalg.norm(a, axis=1)
    twice_area = np.linalg.norm(np.cross(a, b), axis=1)
    source = np.zeros((len(f), 2, 2))
    source[:, 0, 0] = length
    source[:, 0, 1] = np.sum(a*b, axis=1)/length
    source[:, 1, 1] = twice_area/length
    target = np.stack((uv[f[:, 1]]-uv[f[:, 0]], uv[f[:, 2]]-uv[f[:, 0]]), axis=-1)
    jacobian = target @ np.linalg.inv(source)
    singular = np.linalg.svd(jacobian, compute_uv=False)
    energy = np.sum(singular**2 + 1/np.maximum(singular, 1e-20)**2, axis=1)
    return {"area_voxels2": float(twice_area.sum()/2),
            "flipped_or_degenerate_faces": int((np.linalg.det(jacobian) <= 1e-10).sum()),
            "symmetric_dirichlet_energy": float(np.average(energy, weights=twice_area)),
            "stretch_p05_p50_p95": np.percentile(singular, [5, 50, 95]).tolist(),
            "anisotropy_p95": float(np.percentile(singular[:, 0]/singular[:, 1], 95))}


def simplify_dense(v, f, max_faces=20000):
    """Bound SLIM cost while checking topology and sampled geometric retention."""
    before = mesh_audit(v, f)
    if len(f) <= max_faces:
        return v, f, {"applied": False}
    candidate, faces, _, _ = igl.decimate(np.asfortranarray(v),
        np.asfortranarray(f, dtype=np.int32), max_faces, block_intersections=True)
    after = mesh_audit(candidate, faces)
    if (after["nonmanifold_edges"] or after["components"] != before["components"]
        or after["euler_characteristic"] != before["euler_characteristic"]
        or len(igl.boundary_loop_all(faces)) != len(igl.boundary_loop_all(f))):
        raise ValueError("Dense-mesh simplification changed topology")
    # Check all vertices plus face centroids in both directions. This is a
    # sampled distance bound, not a certificate over every surface point.
    source_samples = np.concatenate((v, v[f].mean(1)))
    target_samples = np.concatenate((candidate, candidate[faces].mean(1)))
    distance = np.sqrt(np.concatenate((igl.point_mesh_squared_distance(source_samples, candidate, faces)[0],
                                      igl.point_mesh_squared_distance(target_samples, v, f)[0])))
    audit = {"applied": True, "before": before, "after": after,
        "sampled_distance_p95_max_voxels": [float(np.percentile(distance, 95)), float(distance.max())],
        "intersection_blocking": True, "target_faces": max_faces,
        "error_limits_voxels": {"p95": .25, "maximum": 1.0}}
    if np.percentile(distance, 95) > .25 or distance.max() > 1:
        raise ValueError(f"Dense-mesh simplification exceeded geometric tolerance: {audit}")
    if len(faces) > max_faces*2:
        raise ValueError("Dense-mesh simplification could not reach bounded chart size")
    return np.asarray(candidate), np.asarray(faces, np.int64), audit


def flatten(v, f, iterations=100):
    v, f = np.asarray(v, np.float64), np.asarray(f, np.int64)
    before = mesh_audit(v, f)
    if before["nonmanifold_edges"] or before["components"] != 1:
        raise ValueError("Flatten one manifold component at a time")
    loops = igl.boundary_loop_all(f)
    if not loops:
        raise ValueError("Closed component has no sheet boundary")
    original = np.arange(len(v))
    if len(loops) != 1 or before["euler_characteristic"] != 1:
        cuts = {tuple(sorted((a, b))) for path in igl.cut_to_disk(f) for a, b in zip(path, path[1:])}
        flags = np.array([[tuple(sorted((face[k], face[(k+1)%3]))) in cuts for k in range(3)] for face in f])
        v, f, original = igl.cut_mesh(v, f, flags)
    loops = igl.boundary_loop_all(f)
    if len(loops) != 1 or mesh_audit(v, f)["euler_characteristic"] != 1:
        raise ValueError("Disk cut did not produce one disk")
    boundary = np.array(loops[0], dtype=np.int32)
    area = np.linalg.norm(np.cross(v[f[:, 1]]-v[f[:, 0]], v[f[:, 2]]-v[f[:, 0]]), axis=1).sum()/2
    circle = igl.map_vertices_to_circle(v, boundary)*np.sqrt(area/np.pi)
    # Positive uniform Tutte weights give an injective convex-boundary start,
    # including obtuse triangles where cotangent harmonic weights can flip.
    edges = np.concatenate((f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]))
    edges = np.unique(np.sort(edges, axis=1), axis=0)
    adjacency = sparse.coo_matrix((np.ones(2*len(edges)),
        (np.r_[edges[:, 0], edges[:, 1]], np.r_[edges[:, 1], edges[:, 0]])), shape=(len(v), len(v))).tocsr()
    laplacian = sparse.diags(np.asarray(adjacency.sum(1)).ravel())-adjacency
    interior = np.setdiff1d(np.arange(len(v)), boundary)
    uv = np.zeros((len(v), 2)); uv[boundary] = circle
    if len(interior):
        uv[interior] = spsolve(laplacian[interior][:, interior].tocsc(), -laplacian[interior][:, boundary]@circle)
    # Orient the convex initialization consistently with the face winding.
    p = uv[f]
    a, b = p[:, 1]-p[:, 0], p[:, 2]-p[:, 0]
    signed = a[:, 0]*b[:, 1]-a[:, 1]*b[:, 0]
    if np.median(signed) < 0:
        uv[:, 0] *= -1
    initial = distortion(v, f, uv)
    if initial["flipped_or_degenerate_faces"]:
        raise ValueError("Initialization contains flipped or degenerate faces")
    data = igl.slim_precompute(np.asfortranarray(v), np.asfortranarray(f, dtype=np.int32),
        np.asfortranarray(uv), igl.MappingEnergyType.SYMMETRIC_DIRICHLET,
        np.empty(0, np.int32), np.empty((0, 2), order="F"), 0.0)
    uv = igl.slim_solve(data, iterations)
    # Fix arbitrary in-plane rotation using scan Z, independent of labels.
    direction = np.linalg.lstsq(uv-uv.mean(0), v[:, 2]-v[:, 2].mean(), rcond=None)[0]
    if np.linalg.norm(direction) > 1e-8:
        direction /= np.linalg.norm(direction)
        uv = uv @ np.stack(([direction[1], -direction[0]], direction), axis=1)
    uv -= uv.min(0)
    final = distortion(v, f, uv)
    if not np.isfinite(uv).all() or final["flipped_or_degenerate_faces"]:
        raise ValueError("SLIM produced an invalid chart")
    return v, f, uv, {"method": "libigl 2.6.3 symmetric Dirichlet SLIM; uniform Tutte initialization",
        "iterations": iterations, "mesh_before_cut": before, "mesh_after_cut": mesh_audit(v, f),
        "duplicated_seam_vertices": len(v)-before["vertices"], "initial": initial, "final": final}


def rasterize(v, f, uv, max_dimension=4096):
    width, height = np.ceil(uv.max(0)).astype(int)+1
    if max(width, height) > max_dimension:
        raise ValueError(f"Chart exceeds raster limit {max_dimension}")
    xyz = np.zeros((height, width, 3), np.float32)
    normals = np.zeros_like(xyz)
    hits = np.zeros((height, width), bool); ambiguous = hits.copy()
    vertex_normals = igl.per_vertex_normals(v, f.astype(np.int64))
    for face in f:
        p = uv[face]; a, b = p[1]-p[0], p[2]-p[0]
        det = a[0]*b[1]-a[1]*b[0]
        if abs(det) < 1e-10:
            continue
        lo = np.maximum(np.floor(p.min(0)).astype(int), 0)
        hi = np.minimum(np.ceil(p.max(0)).astype(int), [width-1, height-1])
        yy, xx = np.mgrid[lo[1]:hi[1]+1, lo[0]:hi[0]+1]
        qx, qy = xx+.5-p[0, 0], yy+.5-p[0, 1]
        s, t = (qx*b[1]-qy*b[0])/det, (a[0]*qy-a[1]*qx)/det
        inside = (s >= -1e-7) & (t >= -1e-7) & (s+t <= 1+1e-7)
        y, x = yy[inside], xx[inside]
        weights = np.stack((1-s[inside]-t[inside], s[inside], t[inside]), axis=1)
        values = weights@v[face]
        ambiguous[y, x] |= hits[y, x] & (np.linalg.norm(xyz[y, x]-values, axis=-1) > 2)
        xyz[y, x], normals[y, x], hits[y, x] = values, weights@vertex_normals[face], True
    norm = np.linalg.norm(normals, axis=-1)
    valid = hits & ~ambiguous & (norm > .5)
    normals /= np.maximum(norm[..., None], 1e-12)
    return xyz, normals, valid, {"shape_yx": [int(height), int(width)], "covered_pixels": int(hits.sum()),
        "ambiguous_pixels": int(ambiguous.sum()), "valid_pixels": int(valid.sum()), "pixel_centers_uv": "(x+0.5,y+0.5)"}


def process(directory, ink, simplify=True):
    d = Path(directory); settings = json.loads((d/"input.json").read_text())
    candidates = list((d/"mesh").glob("*all.obj"))
    if len(candidates) != 1:
        raise ValueError(f"Expected one combined mesh, found {candidates}")
    v, f = read_cli_obj(candidates[0]); n, labels = igl.facet_components(f.astype(np.int64))
    raw, template = load_zyx(d/"native/image.nii.gz")
    seed_xyz = np.array(settings["seed_crop_zyx"])[::-1]
    _, seed_face, _ = igl.point_mesh_squared_distance(seed_xyz[None], v, f.astype(np.int64))
    seed_component = int(labels[int(seed_face[0])])
    origin_xyz = np.array(settings["crop_origin_level2_zyx"])[::-1]
    offsets = np.arange(21)-9.875
    nifti_config = {"spacing_zyx_um": settings["spacing_zyx_um"], "origin_yx": [0, 0]}
    summary = {"id": settings["id"], "method": settings["method"], "mesh": mesh_audit(v, f),
               "reference_geometry_used": False, "dense_mesh_policy": "try bounded simplification then retain original" if simplify else "retain original", "components": []}
    for ci in range(n):
        selected = f[labels == ci]; used, inverse = np.unique(selected, return_inverse=True)
        cv, cf = v[used], inverse.reshape(-1, 3)
        out = d/f"component_{ci:02d}"; out.mkdir(exist_ok=True)
        entry = {"component": ci, "is_seed_component": ci == seed_component,
                 "seed_vertex_distance_voxels": float(np.linalg.norm(cv-seed_xyz, axis=1).min())}
        try:
            print(json.dumps({"stage": "flatten_start", "directory": str(d), "component": ci,
                              "faces": len(cf), "time": time.time()}), flush=True)
            attempts = []
            for target_faces in ((20000, 40000, 80000, 160000) if simplify else ()):
                try:
                    sv, sf, simplification = simplify_dense(cv, cf, max_faces=target_faces)
                    simplification["earlier_attempts"] = attempts
                    cv, cf = sv, sf
                    break
                except ValueError as error:
                    attempts.append(str(error))
            else:
                simplification = {"applied": False, "earlier_attempts": attempts,
                    "reason": "Retaining original geometry; optional simplification cannot reject a valid source chart"}
            cv, cf, uv, audit = flatten(cv, cf)
            xyz, normals, valid, raster = rasterize(cv, cf, uv)
            entry.update(simplification=simplification, flatten=audit, raster=raster)
            np.savez_compressed(out/"surface.npz", vertices_xyz=cv, faces=cf, uv=uv,
                xyz_local=xyz, normals=normals, valid=valid)
            global_xyz = (xyz+origin_xyz+.375)*4
            tifdir = out/"tifxyz"; tifdir.mkdir(exist_ok=True)
            for k, axis in enumerate("xyz"):
                tifffile.imwrite(tifdir/f"{axis}.tif", np.where(valid, global_xyz[..., k], -1).astype(np.float32))
            tifffile.imwrite(tifdir/"mask.tif", valid.astype(np.uint8)*255)
            write_json(tifdir/"meta.json", {"type": "seg", "format": "tifxyz", "uuid": f"{settings['id']}_{settings['method']}_{ci}",
                "width": valid.shape[1], "height": valid.shape[0], "scale": [.25, .25],
                "coordinate_frame": "source level-0 XYZ voxels", "local_chart": True})
            with (out/"mesh_source_level0_xyz.obj").open("w") as stream:
                for p in (cv+origin_xyz+.375)*4: stream.write("v " + " ".join(f"{a:.6f}" for a in p)+"\n")
                for p in uv*4: stream.write(f"vt {p[0]:.6f} {p[1]:.6f}\n")
                for face in cf+1: stream.write("f " + " ".join(f"{a}/{a}" for a in face)+"\n")
            point = np.zeros(valid.shape, np.uint8)
            distances = np.where(valid, np.linalg.norm(xyz-seed_xyz, axis=-1), np.inf)
            nearest = np.unravel_index(distances.argmin(), valid.shape)
            if valid.any() and ci == seed_component:
                point[nearest] = 1
                entry["point_projection_distance_voxels"] = float(distances[nearest])
            surface = np.zeros(raw.shape, np.uint8)
            pos = np.rint(xyz[valid][:, ::-1]).astype(int)
            pos = pos[((pos >= 0) & (pos < raw.shape)).all(1)]
            surface[tuple(pos.T)] = 1
            save_like(out/"surface_native.nii.gz", surface, template)
            for name, sign in (("normal_plus", 1), ("normal_minus", -1)):
                flat = out/name; flat.mkdir(exist_ok=True)
                stack, inside = render(raw, xyz, normals*sign, [0, 0, 0], offsets)
                support = valid & inside
                stack[:, ~support] = 0
                h, w = support.shape
                padded = np.pad(stack, ((0, 0), (0, max(0, ink.patch-h)),
                                        (0, max(0, ink.patch-w))))
                pred = ink.predict(padded, start=4)[:h, :w].astype(np.float32)
                pred[~support] = 0
                np.save(flat/"ink.npy", pred)
                for filename, array in (("image", stack.astype(np.float32)),
                    ("pred", np.broadcast_to(pred, stack.shape)),
                    ("point", np.broadcast_to(point, stack.shape)),
                    ("support", np.broadcast_to(support.astype(np.uint8), stack.shape))):
                    save_nifti(flat/f"{filename}.nii.gz", array, nifti_config)
                entry[name] = {"supported_pixels": int(support.sum()), "ink_threshold": .4,
                    "zero_padding_yx_for_minimum_model_patch": [max(0, ink.patch-h), max(0, ink.patch-w)],
                    "ink_pixels_above_threshold": int(((pred >= .4) & support).sum())}
            entry["status"] = "ink_complete"
        except ValueError as error:
            entry.update(status="rejected", reason=str(error))
        write_json(out/"result.json", entry)
        print(json.dumps({"stage": "component_complete", "directory": str(d), "component": ci,
                          "status": entry["status"], "reason": entry.get("reason"), "time": time.time()}), flush=True)
        summary["components"].append(entry)
    write_json(d/"reading.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("directories", nargs="+")
    parser.add_argument("--no-simplify", action="store_true")
    args = parser.parse_args()
    from .model import PublishedInk
    settings = json.loads(Path("configs/ink/first_letters_baseline.json").read_text())
    settings["device"] = "cuda:0"
    ink = PublishedInk(settings)
    for directory in args.directories:
        started = time.monotonic()
        result = process(directory, ink, simplify=not args.no_simplify)
        print(json.dumps({"directory": directory, "seconds": time.monotonic()-started,
            "components": [{"component": c["component"], "status": c["status"],
                            "reason": c.get("reason")} for c in result["components"]]}), flush=True)


if __name__ == "__main__":
    main()
