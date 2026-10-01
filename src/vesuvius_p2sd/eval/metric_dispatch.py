"""Bounded ordered dispatch for expensive per-sample evaluation metrics."""

from __future__ import annotations

from collections import deque
from concurrent.futures import Future, ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass
from multiprocessing import get_context
from typing import Any, Callable, Generic, TypeVar


PayloadT = TypeVar("PayloadT")
ResultT = TypeVar("ResultT")


@dataclass(frozen=True)
class MetricDispatchSettings:
    workers: int = 0
    max_in_flight: int = 1
    backend: str = "thread"

    @classmethod
    def from_metrics_config(cls, metrics_cfg: dict[str, Any]) -> "MetricDispatchSettings":
        workers = max(0, int(metrics_cfg.get("parallel_workers", 0)))
        backend = str(metrics_cfg.get("parallel_backend", "thread"))
        if backend not in {"thread", "process"}:
            raise ValueError(
                "metrics.parallel_backend must be thread|process, "
                f"got {backend!r}")
        default_in_flight = max(1, workers * 2)
        max_in_flight = int(
            metrics_cfg.get("parallel_max_in_flight", default_in_flight))
        if max_in_flight < 1:
            raise ValueError("metrics.parallel_max_in_flight must be at least 1")
        if workers > 0 and max_in_flight < workers:
            raise ValueError(
                "metrics.parallel_max_in_flight must be at least metrics.parallel_workers")

        kaggle_surface = metrics_cfg.get("kaggle_surface", {})
        if isinstance(kaggle_surface, dict) and bool(kaggle_surface.get("enabled", False)):
            # Betti matching holds multiple full-volume topology structures.
            # Start checkpoint scoring conservatively; a measured profile can
            # explicitly raise this nested cap later.
            topology_workers = max(0, int(kaggle_surface.get("parallel_workers", 1)))
            workers = min(workers, topology_workers)
            max_in_flight = 1 if workers == 0 else min(max_in_flight, workers)
        return cls(workers=workers, max_in_flight=max_in_flight, backend=backend)


class OrderedMetricDispatcher(Generic[PayloadT, ResultT]):
    """Run metric work concurrently while returning results in submit order."""

    def __init__(self, settings: MetricDispatchSettings):
        self.settings = settings
        if settings.workers <= 0:
            self._executor = None
        elif settings.backend == "process":
            # ``spawn`` never inherits the parent's CUDA context. This keeps
            # CPU-only full-volume metrics safe to run beside GPU inference.
            self._executor = ProcessPoolExecutor(
                max_workers=settings.workers,
                mp_context=get_context("spawn"),
            )
        else:
            self._executor = ThreadPoolExecutor(
                max_workers=settings.workers,
                thread_name_prefix="p2sd_metric",
            )
        self._pending: deque[tuple[PayloadT, Future[ResultT]]] = deque()

    def submit(
        self,
        payload: PayloadT,
        fn: Callable[..., ResultT],
        *args: Any,
        **kwargs: Any,
    ) -> list[tuple[PayloadT, ResultT]]:
        if self._executor is None:
            return [(payload, fn(*args, **kwargs))]
        self._pending.append((payload, self._executor.submit(fn, *args, **kwargs)))
        if len(self._pending) <= self.settings.max_in_flight:
            return []
        return [self._pop_oldest()]

    def drain(self) -> list[tuple[PayloadT, ResultT]]:
        completed = []
        while self._pending:
            completed.append(self._pop_oldest())
        return completed

    def _pop_oldest(self) -> tuple[PayloadT, ResultT]:
        payload, future = self._pending.popleft()
        return payload, future.result()

    def close(self) -> None:
        if self._executor is not None:
            self._executor.shutdown(wait=True)
            self._executor = None

    def __enter__(self) -> "OrderedMetricDispatcher[PayloadT, ResultT]":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()
