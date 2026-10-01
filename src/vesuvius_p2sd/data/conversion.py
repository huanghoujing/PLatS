"""Dataset conversion helpers."""

from __future__ import annotations

import numpy as np


def erase_ignore_and_border(
    image: np.ndarray,
    labels: np.ndarray,
    *,
    ignore_label: int | None,
    erase_border_width: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    image = np.array(image, copy=True)
    labels = np.array(labels, copy=True)
    if ignore_label is not None:
        labels[labels == ignore_label] = 0
    if erase_border_width > 0:
        zero_volume_border(image, erase_border_width)
        zero_volume_border(labels, erase_border_width)
    return image, labels


def zero_volume_border(volume: np.ndarray, width: int) -> None:
    if width <= 0:
        return
    if volume.ndim < 3:
        raise ValueError(f"Expected at least 3 spatial dims, got shape {volume.shape}")
    spatial_axes = range(volume.ndim - 3, volume.ndim)
    for axis in spatial_axes:
        sl = [slice(None)] * volume.ndim
        sl[axis] = slice(0, width)
        volume[tuple(sl)] = 0
        sl[axis] = slice(-width, None)
        volume[tuple(sl)] = 0
