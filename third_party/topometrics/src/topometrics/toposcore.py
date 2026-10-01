# toposcore.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple, Mapping
import numpy as np

from ._bm_loader import load_betti_matching
bm = load_betti_matching()  # exposes compute_matching(...)

# default weights for dims (0,1,2)
_DEFAULT_WEIGHTS = (0.34, 0.33, 0.33)


@dataclass
class TopoReport:
    toposcore: float
    topoF1_by_dim: Dict[int, float]
    counts_by_dim: Dict[int, Tuple[int, int, int]]  # (m_k, p_k, g_k)
    dims: List[int]


class TopoScore:
    """
    Compute Topological Score (weighted Topological-F1) directly from predictions/labels.

    - By default uses per-dimension weights w0=0.34, w1=0.33, w2=0.33.
    - Only *active* dimensions (those with p_k + g_k > 0) contribute.
      Weights are automatically renormalized over the active dims.
    - Convention: input1 == predictions, input2 == labels (GT)
    """

    @classmethod
    def compute(
        cls,
        predictions: Any,
        labels: Any,
        dims: Optional[Iterable[int]] = None,
        weights: Optional[Iterable[float] | Mapping[int, float]] = None,
    ) -> TopoReport:
        """
        Parameters
        ----------
        predictions, labels : Any
            Passed through to the Betti matcher.
        dims : Iterable[int] or None
            Which homology dimensions to evaluate (e.g., (0,1,2)). If None, inferred.
        weights : iterable or mapping or None
            If iterable, interpreted index-wise as w[k] for k in dims (e.g., length>=3 for dims 0..2).
            If mapping, keys are dimension ints -> weight.
            If None, uses defaults (0.34, 0.33, 0.33).
        """
        result = bm.compute_matching(predictions, labels)
        return cls._from_result(result, dims, weights)

    # ---------- Internals ----------

    @classmethod
    def _from_result(
        cls,
        result: Any,
        dims: Optional[Iterable[int]],
        weights_in: Optional[Iterable[float] | Mapping[int, float]],
    ) -> TopoReport:
        dims_list = cls._infer_dims_if_needed(result, dims)

        topoF1_by_dim: Dict[int, float] = {}
        counts_by_dim: Dict[int, Tuple[int, int, int]] = {}
        active_dims: List[int] = []

        for k in dims_list:
            m_k, p_k, g_k = cls._counts_for_dim(result, k)
            counts_by_dim[k] = (m_k, p_k, g_k)
            denom = p_k + g_k
            if denom > 0:
                if g_k != 0:
                    topoF1_k = (2.0 * m_k) / float(denom)
                else:
                    topoF1_k = 0.5 / (float(denom) + 0.5) # adding pseudocount to avoid 0 independently of p_k
                topoF1_by_dim[k] = float(topoF1_k)
                active_dims.append(k)  
            else:
                topoF1_by_dim[k] = float("nan")  # keep NaN for inactive (preserves drop-in behavior)

        if not active_dims:
            score = 1.0  # convention
        else:
            w = cls._weights_for_dims(dims_list, weights_in)
            w_active = np.array([w.get(k, 0.0) for k in active_dims], dtype=float)
            if not np.isfinite(w_active).all() or w_active.sum() <= 0:
                w_active = np.ones(len(active_dims), dtype=float)
            w_active = w_active / w_active.sum()
            vals = np.array([topoF1_by_dim[k] for k in active_dims], dtype=float)
            score = float((w_active * vals).sum())

        return TopoReport(
            toposcore=score,
            topoF1_by_dim=topoF1_by_dim,
            counts_by_dim=counts_by_dim,
            dims=dims_list,
        )

    @staticmethod
    def _weights_for_dims(
        dims_list: List[int],
        weights_in: Optional[Iterable[float] | Mapping[int, float]],
    ) -> Dict[int, float]:
        """
        Build a {dim -> weight} dict. Supports:
          - None: defaults (0.34, 0.33, 0.33) for dims 0..2, zeros otherwise
          - Mapping[int,float]: direct, merged over defaults (can pass a partial mapping)
          - Iterable[float]: index-aligned to dims_list positions if lengths match,
                             else treated as global sequence with index == dim
        """
        default_map = {0: _DEFAULT_WEIGHTS[0], 1: _DEFAULT_WEIGHTS[1], 2: _DEFAULT_WEIGHTS[2]}
        if weights_in is None:
            return {k: default_map.get(k, 0.0) for k in dims_list}

        if isinstance(weights_in, Mapping):
            out = {k: default_map.get(k, 0.0) for k in dims_list}
            out.update({int(k): float(v) for k, v in weights_in.items()})
            return out

        seq = list(weights_in)
        if len(seq) >= (max(dims_list) + 1 if dims_list else 0):
            return {k: float(seq[k]) for k in dims_list}
        out = {}
        for i, k in enumerate(dims_list):
            out[k] = float(seq[i]) if i < len(seq) else 0.0
        return out

    @staticmethod
    def _infer_dims_if_needed(result: Any, dims: Optional[Iterable[int]]) -> List[int]:
        if dims is not None:
            return list(dims)
        lengths = [
            len(getattr(result, "input1_matched_birth_coordinates", [])),
            len(getattr(result, "input1_unmatched_birth_coordinates", [])),
            len(getattr(result, "input2_unmatched_birth_coordinates", [])),
            len(getattr(result, "input2_matched_birth_coordinates", [])),
        ]
        max_len = max(lengths) if lengths else 0
        return list(range(max_len))

    @staticmethod
    def _count_points(arr: Any) -> int:
        """
        Count how many 'points' (features) are represented by a container.
        Handles:
          - None -> 0
          - list/tuple of points/arrays -> len(...)
          - numpy object arrays (1-D) -> len(...)
          - 2-D numeric arrays -> rows as points (fallback to max axis if ambiguous)
          - 1-D numeric arrays -> 1 point (treat as a single coordinate)
          - everything else -> 0
        """
        if arr is None:
            return 0

        if isinstance(arr, (list, tuple)):
            return len(arr)

        try:
            a = np.asarray(arr)
        except Exception:
            return 0

        if a.size == 0:
            return 0

        if a.dtype == object:
            return int(a.shape[0])

        if a.ndim == 1:
            return 1

        if a.ndim == 2:
            if a.shape[1] in (1, 2, 3):
                return int(a.shape[0])
            if a.shape[0] in (1, 2, 3):
                return int(a.shape[1])
            return int(a.shape[0])

        if a.ndim >= 3:
            return int(a.shape[0])

        return 0

    @classmethod
    def _counts_for_dim(cls, result: Any, k: int) -> Tuple[int, int, int]:
        m_k = cls._count_points(cls._safe_index(getattr(result, "input1_matched_birth_coordinates", []), k))
        p_un_k = cls._count_points(cls._safe_index(getattr(result, "input1_unmatched_birth_coordinates", []), k))
        g_un_k = cls._count_points(cls._safe_index(getattr(result, "input2_unmatched_birth_coordinates", []), k))
        p_k = m_k + p_un_k
        g_k = m_k + g_un_k
        return m_k, p_k, g_k

    @staticmethod
    def _safe_index(seq: Any, k: int) -> Any:
        try:
            return seq[k]
        except Exception:
            return None

    @classmethod
    def from_counts(
        cls,
        counts_by_dim: Mapping[int, Tuple[int, int, int]],
        dims: Optional[Iterable[int]] = None,
        weights: Optional[Iterable[float] | Mapping[int, float]] = None,
    ) -> TopoReport:
        """
        Build a TopoReport directly from aggregated (m_k, p_k, g_k) counts.
        Mirrors _from_result logic to keep semantics identical.
        """
        dims_list = sorted(counts_by_dim.keys()) if dims is None else list(dims)

        topoF1_by_dim: Dict[int, float] = {}
        active_dims: List[int] = []

        for k in dims_list:
            m_k, p_k, g_k = counts_by_dim.get(k, (0, 0, 0))
            denom = p_k + g_k
            if denom > 0:
                if g_k != 0:
                    topoF1_k = (2.0 * m_k) / float(denom)
                else:
                    topoF1_k = 0.5 / (float(denom) + 0.5)
                topoF1_by_dim[k] = float(topoF1_k)
                active_dims.append(k)
            else:
                topoF1_by_dim[k] = float("nan")

        if not active_dims:
            score = 1.0
        else:
            w = cls._weights_for_dims(dims_list, weights)
            w_active = np.array([w.get(k, 0.0) for k in active_dims], dtype=float)
            if not np.isfinite(w_active).all() or w_active.sum() <= 0:
                w_active = np.ones(len(active_dims), dtype=float)
            w_active = w_active / w_active.sum()
            vals = np.array([topoF1_by_dim[k] for k in active_dims], dtype=float)
            score = float((w_active * vals).sum())

        return TopoReport(
            toposcore=score,
            topoF1_by_dim=topoF1_by_dim,
            counts_by_dim={k: counts_by_dim.get(k, (0, 0, 0)) for k in dims_list},
            dims=dims_list,
        )

    @classmethod
    def aggregate_reports(
        cls,
        reports: Iterable[TopoReport],
        dims: Optional[Iterable[int]] = None,
        weights: Optional[Iterable[float] | Mapping[int, float]] = None,
    ) -> TopoReport:
        """
        Sum (m_k, p_k, g_k) across reports and rebuild a single TopoReport.
        """
        combined: Dict[int, Tuple[int, int, int]] = {}
        for r in reports:
            for k, (m, p, g) in r.counts_by_dim.items():
                M, P, G = combined.get(k, (0, 0, 0))
                combined[k] = (M + m, P + p, G + g)
        return cls.from_counts(combined, dims=dims, weights=weights)
