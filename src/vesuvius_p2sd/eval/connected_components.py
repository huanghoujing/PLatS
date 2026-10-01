"""Connected-component utilities for 3D masks."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage

try:
    import cc3d
except ImportError:  # Keep analysis-only utilities usable in minimal CPU environments.
    cc3d = None


@dataclass(frozen=True)
class ComponentInfo:
    num_components: int
    largest_size: int
    largest_fraction: float
    small_component_count: int
    prompt_component_size: int
    prompt_component_fraction: float


def label_components(mask: np.ndarray, *, connectivity: int = 3) -> tuple[np.ndarray, int]:
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 3:
        raise ValueError(f"Expected a 3D mask, got shape {mask.shape}")
    structure = ndimage.generate_binary_structure(3, connectivity)
    labels, count = ndimage.label(mask, structure=structure)
    return labels.astype(np.int32, copy=False), int(count)


def component_count(mask: np.ndarray, *, connectivity: int = 3) -> int:
    """Return the raw connected-component count without GT-dependent filtering.

    ``cc3d`` avoids allocating a labeled volume when callers need only the
    count.  SciPy remains the compatibility fallback and the labeled backend
    for metrics that need component sizes or masks.
    """
    mask = np.ascontiguousarray(mask, dtype=bool)
    if mask.ndim != 3:
        raise ValueError(f"Expected a 3D mask, got shape {mask.shape}")
    if cc3d is not None:
        cc3d_connectivity = {1: 6, 2: 18, 3: 26}.get(connectivity)
        if cc3d_connectivity is None:
            raise ValueError(f"Expected connectivity 1, 2, or 3, got {connectivity}")
        _, count = cc3d.connected_components(
            mask,
            connectivity=cc3d_connectivity,
            return_N=True,
        )
        return int(count)
    _, count = label_components(mask, connectivity=connectivity)
    return count


def component_info(
    mask: np.ndarray,
    *,
    prompt_zyx: tuple[int, int, int] | None = None,
    small_component_max_voxels: int = 32,
) -> ComponentInfo:
    labels, count = label_components(mask)
    total = int(np.asarray(mask, dtype=bool).sum())
    if count == 0 or total == 0:
        return ComponentInfo(0, 0, 0.0, 0, 0, 0.0)

    sizes = np.bincount(labels.ravel(), minlength=count + 1)
    sizes[0] = 0
    largest_size = int(sizes.max())
    small_count = int(((sizes > 0) & (sizes <= small_component_max_voxels)).sum())
    prompt_size = _prompt_component_size(labels, sizes, prompt_zyx)
    return ComponentInfo(
        num_components=count,
        largest_size=largest_size,
        largest_fraction=largest_size / max(total, 1),
        small_component_count=small_count,
        prompt_component_size=prompt_size,
        prompt_component_fraction=prompt_size / max(total, 1),
    )


def prompt_connected_mask(
    mask: np.ndarray,
    prompt_zyx: tuple[int, int, int] | None,
) -> np.ndarray:
    labels, count = label_components(mask)
    if count == 0:
        return np.zeros_like(mask, dtype=bool)
    sizes = np.bincount(labels.ravel(), minlength=count + 1)
    sizes[0] = 0
    label_id = _prompt_label(labels, sizes, prompt_zyx)
    if label_id <= 0:
        label_id = int(sizes.argmax())
    return labels == label_id


def _prompt_component_size(
    labels: np.ndarray,
    sizes: np.ndarray,
    prompt_zyx: tuple[int, int, int] | None,
) -> int:
    label_id = _prompt_label(labels, sizes, prompt_zyx)
    return int(sizes[label_id]) if label_id > 0 else 0


def _prompt_label(
    labels: np.ndarray,
    sizes: np.ndarray,
    prompt_zyx: tuple[int, int, int] | None,
) -> int:
    if prompt_zyx is None:
        return 0
    z, y, x = (int(v) for v in prompt_zyx)
    if not (0 <= z < labels.shape[0]
            and 0 <= y < labels.shape[1]
            and 0 <= x < labels.shape[2]):
        return 0
    label_id = int(labels[z, y, x])
    return label_id if label_id < len(sizes) else 0
