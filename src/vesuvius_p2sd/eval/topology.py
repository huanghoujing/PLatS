"""Fast Betti-number proxy for binary 3D masks.

Computes (b0, b1, b2) = (components, tunnels, cavities) from connected
components plus the cubical Euler characteristic, instead of persistent
homology. For a binary volume this is exact for the chosen cubical
construction and runs in well under a second at PS320, which makes
per-sample topology affordable at every evaluation instead of only in the
octant-tiled official matcher.

Constructions:

- ``"T"``: voxels are closed unit cubes. Foreground uses 26-connectivity,
  enclosed cavities use 6-connectivity. This matches the repo's existing
  component metrics (``connectivity=3``).
- ``"V"``: voxels are vertices. Foreground uses 6-connectivity, cavities use
  26-connectivity. This is the dual convention used by sublevel-set cubical
  filtrations on inverted binary masks (the official matcher input path).

The two constructions differ only on sub-voxel contacts (diagonal touches);
``tests/test_topology_proxy.py`` pins both against analytic shapes and the
exact count-only matcher.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage

from vesuvius_p2sd.eval.connected_components import label_components


def _or_reduce_windows(volume: np.ndarray, axes: tuple[int, ...]) -> np.ndarray:
    """OR of every 2-window along each axis in ``axes`` (shrinks by 1 per axis)."""
    out = volume
    for axis in axes:
        lead = tuple(slice(None) for _ in range(axis))
        out = out[(*lead, slice(None, -1))] | out[(*lead, slice(1, None))]
    return out


def _and_reduce_windows(volume: np.ndarray, axes: tuple[int, ...]) -> np.ndarray:
    """AND of every 2-window along each axis in ``axes`` (shrinks by 1 per axis)."""
    out = volume
    for axis in axes:
        lead = tuple(slice(None) for _ in range(axis))
        out = out[(*lead, slice(None, -1))] & out[(*lead, slice(1, None))]
    return out


def euler_characteristic(mask: np.ndarray, *, construction: str = "T") -> int:
    """Euler characteristic of the cubical complex built from a binary mask."""
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 3:
        raise ValueError(f"Expected a 3D mask, got shape {mask.shape}")
    if construction == "V":
        # Voxels are vertices; cells are spanned by fully-foreground windows.
        vertices = int(mask.sum())
        edges = sum(int(_and_reduce_windows(mask, (axis,)).sum()) for axis in (0, 1, 2))
        faces = sum(
            int(_and_reduce_windows(mask, axes).sum())
            for axes in ((0, 1), (0, 2), (1, 2))
        )
        cubes = int(_and_reduce_windows(mask, (0, 1, 2)).sum())
        return vertices - edges + faces - cubes
    if construction == "T":
        # Voxels are closed unit cubes; a grid cell belongs to the closure if
        # any incident voxel is foreground, so pad and OR over windows.
        padded = np.pad(mask, 1)
        cubes = int(mask.sum())
        # A face orthogonal to `axis` is incident to the 2-window along it;
        # an edge parallel to an axis is incident to the 2x2 window over the
        # other two axes; a vertex to the full 2x2x2 window.
        faces = sum(
            int(_trim_except(_or_reduce_windows(padded, (axis,)), axis).sum())
            for axis in (0, 1, 2)
        )
        edges = sum(
            int(_trim_except(_or_reduce_windows(padded, axes), *axes).sum())
            for axes in ((0, 1), (0, 2), (1, 2))
        )
        vertices = int(_or_reduce_windows(padded, (0, 1, 2)).sum())
        return vertices - edges + faces - cubes
    raise ValueError(f"construction must be 'T' or 'V', got {construction!r}")


def _trim_except(volume: np.ndarray, *window_axes: int) -> np.ndarray:
    """Drop the padding layer on every axis that was not shrunk by a window."""
    slices = [
        slice(None) if axis in window_axes else slice(1, -1)
        for axis in range(volume.ndim)
    ]
    return volume[tuple(slices)]


def _cavity_count(mask: np.ndarray, *, connectivity: int) -> int:
    """Count background components fully enclosed by the foreground."""
    background = np.pad(~mask, 1, constant_values=True)
    structure = ndimage.generate_binary_structure(3, connectivity)
    labels, count = ndimage.label(background, structure=structure)
    # The padded border belongs to the single outside component.
    outside = labels[0, 0, 0]
    return int(count - (1 if outside > 0 else 0))


class SheetContactTracker:
    """Track touching/overlap between decoded sheet masks in one volume.

    Sheets are added one at a time; contacts are recorded when a new sheet's
    mask overlaps or is 26-adjacent to any previously added sheet. Memory is
    one int16 label canvas regardless of sheet count, and dilation runs only
    inside each new mask's padded bounding box.

    Per the production priorities, any contact between different decoded
    sheets is a hard failure: merged sheets cannot be amended automatically.
    """

    def __init__(self, shape: tuple[int, int, int], *, connectivity: int = 3) -> None:
        self._labels = np.zeros(shape, dtype=np.int16)
        self._structure = ndimage.generate_binary_structure(3, connectivity)
        self._contacts: set[tuple[int, int]] = set()
        self._overlap_voxels = 0
        self._count = 0

    def add(self, mask: np.ndarray) -> int:
        """Add one sheet mask; returns its index within this tracker."""
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != self._labels.shape:
            raise ValueError(
                f"mask shape {mask.shape} does not match tracker shape {self._labels.shape}"
            )
        index = self._count
        self._count += 1
        if mask.any():
            bounds = ndimage.find_objects(mask.astype(np.int8), max_label=1)[0]
            padded = tuple(
                slice(max(s.start - 1, 0), min(s.stop + 1, dim))
                for s, dim in zip(bounds, mask.shape)
            )
            local_mask = mask[padded]
            local_labels = self._labels[padded]
            self._overlap_voxels += int((local_labels[local_mask] != 0).sum())
            reach = ndimage.binary_dilation(local_mask, structure=self._structure)
            for touched in np.unique(local_labels[reach]):
                if touched > 0:
                    self._contacts.add((int(touched) - 1, index))
            local_labels[local_mask] = index + 1
        return index

    def summary(self) -> dict[str, float]:
        involved = {sheet for pair in self._contacts for sheet in pair}
        return {
            "sheet_count": float(self._count),
            "sheet_contact_pair_count": float(len(self._contacts)),
            "sheet_contact_component_count": float(len(involved)),
            "sheet_contact_overlap_voxels": float(self._overlap_voxels),
            "sheet_contact_free": float(not self._contacts),
        }


class GpuSheetContactTracker:
    """`SheetContactTracker` on a CUDA device via max-pool dilation.

    Accepts boolean torch tensors that may still live on the GPU (e.g. fresh
    model decodings), so contact tracking costs milliseconds instead of the
    CPU dilation seconds and avoids a device-to-host copy. Produces the same
    summary as the CPU tracker (parity-tested).
    """

    def __init__(self, shape: tuple[int, int, int], *, device) -> None:
        import torch

        self._torch = torch
        self._labels = torch.zeros(shape, dtype=torch.int16, device=device)
        self._contacts: set[tuple[int, int]] = set()
        self._overlap_voxels = 0
        self._count = 0

    def add(self, mask) -> int:
        torch = self._torch
        mask = mask.to(self._labels.device, torch.bool)
        if mask.shape != self._labels.shape:
            raise ValueError(
                f"mask shape {tuple(mask.shape)} does not match tracker shape "
                f"{tuple(self._labels.shape)}"
            )
        index = self._count
        self._count += 1
        if bool(mask.any()):
            self._overlap_voxels += int((self._labels[mask] != 0).sum())
            reach = torch.nn.functional.max_pool3d(
                mask[None, None].float(), kernel_size=3, stride=1, padding=1,
            )[0, 0] > 0
            touched = torch.unique(self._labels[reach])
            for value in touched.tolist():
                if value > 0:
                    self._contacts.add((int(value) - 1, index))
            self._labels[mask] = index + 1
        return index

    def summary(self) -> dict[str, float]:
        involved = {sheet for pair in self._contacts for sheet in pair}
        return {
            "sheet_count": float(self._count),
            "sheet_contact_pair_count": float(len(self._contacts)),
            "sheet_contact_component_count": float(len(involved)),
            "sheet_contact_overlap_voxels": float(self._overlap_voxels),
            "sheet_contact_free": float(not self._contacts),
        }


def betti_numbers(mask: np.ndarray, *, construction: str = "T") -> tuple[int, int, int]:
    """Exact (b0, b1, b2) of a binary 3D mask under the chosen construction."""
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 3:
        raise ValueError(f"Expected a 3D mask, got shape {mask.shape}")
    if not mask.any():
        return (0, 0, 0)
    if construction == "T":
        fg_connectivity, bg_connectivity = 3, 1
    elif construction == "V":
        fg_connectivity, bg_connectivity = 1, 3
    else:
        raise ValueError(f"construction must be 'T' or 'V', got {construction!r}")
    _, b0 = label_components(mask, connectivity=fg_connectivity)
    b2 = _cavity_count(mask, connectivity=bg_connectivity)
    chi = euler_characteristic(mask, construction=construction)
    b1 = b0 + b2 - chi
    if b1 < 0:
        raise RuntimeError(
            f"Inconsistent topology: b0={b0}, b2={b2}, chi={chi} give negative b1"
        )
    return (int(b0), int(b1), int(b2))
