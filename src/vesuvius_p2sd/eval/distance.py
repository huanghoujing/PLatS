"""Exact Euclidean distance transforms for evaluation metrics."""

from __future__ import annotations

import numpy as np
from scipy import ndimage


def distance_transform_edt(mask: np.ndarray, *, backend: str) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool)
    if backend == "cpu_scipy":
        return ndimage.distance_transform_edt(mask)
    if backend != "gpu_cucim":
        raise ValueError(
            "metrics.distance_backend must be cpu_scipy|gpu_cucim, "
            f"got {backend!r}"
        )
    try:
        import cupy as cp
        from cucim.core.operations.morphology import distance_transform_edt as cucim_edt
    except ImportError as exc:  # pragma: no cover - depends on optional environment
        raise ImportError(
            "gpu_cucim evaluation metrics require the gpu-edt extra: "
            "pip install -e '.[gpu-edt]'"
        ) from exc
    return cp.asnumpy(cucim_edt(cp.asarray(mask), float64_distances=False))
