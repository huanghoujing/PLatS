"""Test-time augmentation over the EXACT symmetries of the training augmentation (per-axis flips, xy rotations by
90/180/270 degrees; `data/dataset.py`): the 16-element group D4(y,x) x flip(z). Averaging is done on full-resolution
outputs only (user directive 2026-09-05: the AE latent is not flip-then-flip-back equivalent, the decoded volume is).

A transform is (rot_k, flip_z, flip_x): rot90 by rot_k on the (H, W) = (y, x) axes, then optional flips of z and x
(flip y = rot180 + flip x, so this parametrisation spans the whole group). Volumes are torch tensors (..., D, H, W);
prompt points are integer (z, y, x) voxel coordinates. The point transform is DERIVED at import by probing torch's
rot90/flip on a tiny volume, so it cannot drift from the volume transform (self-test: `python -m ...tta`).
"""
from __future__ import annotations

import numpy as np
import torch

GROUP16 = [(k, fz, fx) for k in range(4) for fz in (0, 1) for fx in (0, 1)]
FLIPS8 = [(k, fz, fx) for k in (0, 2) for fz in (0, 1) for fx in (0, 1)]   # rot180 + flip x = flip y


def transforms(n: int) -> list[tuple[int, int, int]]:
    """n = 1 identity, 8 the axis flips, 16 the full group."""
    if n <= 1:
        return [(0, 0, 0)]
    if n == 8:
        return list(FLIPS8)
    if n == 16:
        return list(GROUP16)
    raise ValueError(f"tta count must be 1, 8 or 16, got {n}")


def apply(vol: torch.Tensor, t: tuple[int, int, int]) -> torch.Tensor:
    """Forward transform of a (..., D, H, W) tensor."""
    k, fz, fx = t
    out = torch.rot90(vol, k, dims=(-2, -1)) if k else vol
    dims = [d for d, f in ((-3, fz), (-1, fx)) if f]
    return torch.flip(out, dims) if dims else out


def invert(vol: torch.Tensor, t: tuple[int, int, int]) -> torch.Tensor:
    """Inverse transform (undo flips, then the rotation)."""
    k, fz, fx = t
    dims = [d for d, f in ((-3, fz), (-1, fx)) if f]
    out = torch.flip(vol, dims) if dims else vol
    return torch.rot90(out, -k, dims=(-2, -1)) if k else out


def _probe(t: tuple[int, int, int], shape=(3, 5, 7)):
    """Axis map of a transform: for target axis j, (source axis, reversed) — found by transforming coordinate ramps."""
    D, H, W = shape
    ramps = [torch.arange(D).view(D, 1, 1).expand(D, H, W), torch.arange(H).view(1, H, 1).expand(D, H, W),
             torch.arange(W).view(1, 1, W).expand(D, H, W)]
    moved = [apply(r.clone(), t) for r in ramps]
    axis_map = []
    for j in range(3):   # which source ramp varies along target axis j, and in which direction
        for s in range(3):
            m = moved[s]
            idx = [slice(0, 1)] * 3; idx[j] = slice(None)
            line = m[tuple(idx)].flatten()
            if line.numel() > 1 and (line[1:] != line[:-1]).all():
                axis_map.append((s, bool(line[0] > line[1]))); break
        else:
            raise RuntimeError(f"probe failed for {t}")
    return axis_map


_AXIS_MAPS = {t: _probe(t) for t in GROUP16}


def apply_points(points_zyx, t: tuple[int, int, int], shape) -> np.ndarray:
    """Forward transform of integer (z, y, x) points into the transformed volume's frame; shape = SOURCE (D, H, W)."""
    pts = np.asarray(points_zyx, dtype=np.int64).reshape(-1, 3)
    out = np.empty_like(pts)
    tshape = out_shape(shape, t)
    for j, (s, rev) in enumerate(_AXIS_MAPS[t]):
        out[:, j] = (tshape[j] - 1 - pts[:, s]) if rev else pts[:, s]
    return out


def out_shape(shape, t: tuple[int, int, int]):
    D, H, W = shape
    return (D, W, H) if t[0] % 2 else (D, H, W)


def _selftest():
    rng = np.random.default_rng(0)
    for shape in ((4, 6, 9), (320, 320, 320)):
        D, H, W = shape
        pts = np.unique(np.stack([rng.integers(0, D, 50), rng.integers(0, H, 50), rng.integers(0, W, 50)], 1), axis=0)
        vol = torch.zeros((D, H, W), dtype=torch.int64)
        for i, p in enumerate(pts, 1):
            vol[tuple(p)] = i
        for t in GROUP16:
            moved = apply(vol, t)
            q = apply_points(pts, t, shape)
            assert tuple(moved.shape) == out_shape(shape, t), (t, moved.shape)
            got = moved[torch.from_numpy(q[:, 0]), torch.from_numpy(q[:, 1]), torch.from_numpy(q[:, 2])].numpy()
            assert (got == np.arange(1, len(pts) + 1)).all(), f"point transform wrong for {t}"
            assert torch.equal(invert(moved, t), vol), f"inverse wrong for {t}"
        assert len({tuple(apply(vol, t).flatten().tolist()) for t in GROUP16}) == 16, "transforms not distinct"
    print("tta self-test OK: 16 distinct transforms, points and inverses consistent")


if __name__ == "__main__":
    _selftest()
