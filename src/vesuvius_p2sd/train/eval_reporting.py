"""Per-sample evaluation reporting: tail statistics and a worst-K gallery.

Aggregate means stop carrying information once a model is good on average;
production readiness is decided by the tail. This module persists one row per
validation sample to ``eval_samples.jsonl``, appends tail statistics to the
aggregate metrics row, and renders the K worst samples of each evaluation as
projection PNGs under ``viz/worst/step_XXXXXX/``.
"""

from __future__ import annotations

import heapq
from pathlib import Path
from typing import Any

import numpy as np

from vesuvius_p2sd.train.common import append_jsonl
from vesuvius_p2sd.train.visualization import (
    prune_visualization_step_groups,
    save_p2sd_projection_png,
)

TAIL_KEYS = ("quality_composite", "dice", "tolerant_f1_tau2")


def severity_score(metrics: dict[str, float]) -> float:
    """Rank samples worst-first: hard failures, then amendable, then quality."""
    return (
        2.0 * float(metrics.get("hard_failure", 0.0))
        + 1.0 * float(metrics.get("amendable_defect", 0.0))
        + (1.0 - float(metrics.get("quality_composite", 0.0)))
    )


class EvalSampleReporter:
    """Collect per-sample metrics during one evaluation pass."""

    def __init__(
        self,
        *,
        run_dir: Path,
        step: int,
        task: str,
        metrics_cfg: dict[str, Any] | None = None,
    ) -> None:
        metrics_cfg = metrics_cfg or {}
        self.run_dir = Path(run_dir)
        self.step = int(step)
        self.task = str(task)
        self.save_samples = bool(metrics_cfg.get("save_eval_samples", True))
        self.worst_k = int(metrics_cfg.get("worst_gallery_k", 4))
        self.keep_latest = int(metrics_cfg.get("worst_gallery_keep_latest", 4))
        self.rows: list[dict[str, Any]] = []
        self._heap: list[tuple[float, int, dict[str, Any]]] = []
        self._counter = 0

    @property
    def gallery_enabled(self) -> bool:
        return self.worst_k > 0

    def compact_volumes(
        self,
        *,
        pred: np.ndarray,
        gt: np.ndarray,
        image: np.ndarray | None = None,
    ) -> dict[str, np.ndarray]:
        """Downcast volumes so at most worst_k + in-flight copies stay resident."""
        volumes = {
            "pred": np.asarray(pred, dtype=np.float16),
            "gt": np.asarray(gt, dtype=np.uint8),
        }
        if image is not None:
            volumes["image"] = np.asarray(image, dtype=np.float16)
        return volumes

    def add(
        self,
        *,
        identity: dict[str, Any],
        metrics: dict[str, float],
        volumes: dict[str, np.ndarray] | None = None,
    ) -> None:
        row = {
            "step": self.step,
            "task": self.task,
            **identity,
            **{key: value for key, value in metrics.items() if isinstance(value, (int, float))},
        }
        self.rows.append(row)
        if volumes is None or not self.gallery_enabled:
            return
        entry = (severity_score(metrics), self._counter, {
            "identity": identity,
            "metrics": metrics,
            "volumes": volumes,
        })
        self._counter += 1
        if len(self._heap) < self.worst_k:
            heapq.heappush(self._heap, entry)
        else:
            # Min-heap on severity keeps the K most severe samples.
            heapq.heappushpop(self._heap, entry)

    def tail_stats(self) -> dict[str, float]:
        stats: dict[str, float] = {}
        for key in TAIL_KEYS:
            values = np.asarray(
                [float(row[key]) for row in self.rows if key in row], dtype=np.float64)
            if values.size:
                stats[f"{key}_p10"] = float(np.percentile(values, 10))
                stats[f"{key}_worst"] = float(values.min())
        return stats

    def finalize(self) -> dict[str, float]:
        """Persist per-sample rows, render the gallery, return tail statistics."""
        if self.save_samples and self.rows:
            for row in self.rows:
                append_jsonl(self.run_dir / "eval_samples.jsonl", row)
        if self._heap:
            gallery_dir = self.run_dir / "viz" / "worst"
            worst_first = sorted(self._heap, key=lambda item: -item[0])
            for rank, (_, _, sample) in enumerate(worst_first):
                identity = sample["identity"]
                volumes = sample["volumes"]
                name = (
                    f"step_{self.step:06d}_rank{rank:02d}"
                    f"_case_{identity.get('case_id', 'unknown')}"
                    f"_component_{identity.get('component_id', 'unknown')}.png"
                )
                save_p2sd_projection_png(
                    gallery_dir / name,
                    image=np.asarray(
                        volumes.get("image", volumes["gt"]), dtype=np.float32),
                    gt=np.asarray(volumes["gt"], dtype=np.float32),
                    pred=np.asarray(volumes["pred"], dtype=np.float32),
                    metrics=sample["metrics"],
                    meta={"step": self.step, "task": self.task, **identity},
                )
            prune_visualization_step_groups(
                gallery_dir,
                keep_latest=self.keep_latest,
                step_prefix="step_",
            )
        return self.tail_stats()
