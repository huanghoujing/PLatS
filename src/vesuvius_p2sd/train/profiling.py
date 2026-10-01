"""Opt-in CUDA profiling for bounded P2SD training runs."""

from __future__ import annotations

import statistics
import time
from collections import defaultdict
from contextlib import AbstractContextManager, contextmanager
from pathlib import Path
from typing import Any

import torch
from torch import nn

from vesuvius_p2sd.research.run_status import write_json


class P2SDStepProfiler:
    """Capture annotated training-stage timing without changing normal runs."""

    def __init__(
        self,
        training_cfg: dict[str, Any],
        run_dir: Path,
        model: nn.Module,
        device: torch.device,
    ) -> None:
        profile_cfg = training_cfg.get("profile", {})
        self.enabled = isinstance(profile_cfg, dict) and bool(profile_cfg.get("enabled", False))
        self.run_dir = run_dir
        self.device = device
        self.step_index = 0
        self.warmup_steps = int(profile_cfg.get("warmup_steps", 4)) if self.enabled else 0
        self.active_steps = int(profile_cfg.get("active_steps", 8)) if self.enabled else 0
        self.capture_trace = bool(profile_cfg.get("capture_trace", True)) if self.enabled else False
        if self.enabled and (self.warmup_steps < 0 or self.active_steps <= 0):
            raise ValueError("training.profile requires warmup_steps >= 0 and active_steps > 0")
        self.trace_path = run_dir / str(profile_cfg.get("trace_name", "profile_trace.json"))
        self.summary_path = run_dir / str(profile_cfg.get("summary_name", "profile_summary.json"))
        self._step_start_s: float | None = None
        self._active_step_seconds: list[float] = []
        self._active_pair_counts: list[int] = []
        self._active_source_crop_counts: list[int] = []
        self._active_stage_events: list[tuple[str, torch.cuda.Event, torch.cuda.Event]] = []
        self._stage_cuda_milliseconds: dict[str, list[float]] = defaultdict(list)
        self._hook_handles: list[Any] = []
        self._module_contexts: dict[int, list[AbstractContextManager]] = {}
        self._restore_methods: list[tuple[Any, str, Any]] = []
        self._profiler = None

        if not self.enabled:
            return
        if device.type != "cuda":
            raise ValueError("training.profile currently requires a CUDA device")
        if self.capture_trace:
            activities = [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
            schedule = torch.profiler.schedule(
                wait=0,
                warmup=self.warmup_steps,
                active=self.active_steps,
                repeat=1,
            )
            self._profiler = torch.profiler.profile(
                activities=activities,
                schedule=schedule,
                record_shapes=False,
                profile_memory=True,
                with_stack=False,
            )
            self._install_module_hooks(model)
        torch.cuda.reset_peak_memory_stats(device)

    def _install_module_hooks(self, model: nn.Module) -> None:
        self._wrap_image_encoder_encode(model.image_encoder)
        stage_modules: list[tuple[str, nn.Module]] = [
            ("p2sd/grid_projection", model.grid_proj),
            ("p2sd/prompt_mlp", model.prompt_mlp),
            ("p2sd/prompt_modulation", model.cond_proj),
        ]
        stage_modules.extend(("p2sd/modulator_attention", block) for block in model.modulator_blocks)
        stage_modules.extend(("p2sd/refiner_attention", block) for block in model.blocks)
        for name, module in stage_modules:
            self._hook_handles.append(module.register_forward_pre_hook(self._make_pre_hook(name)))
            self._hook_handles.append(module.register_forward_hook(self._make_post_hook()))

    def _wrap_image_encoder_encode(self, image_encoder: nn.Module) -> None:
        original_encode = image_encoder.encode

        def profiled_encode(x: torch.Tensor) -> torch.Tensor:
            with self.range("p2sd/image_encoder"):
                return original_encode(x)

        self._restore_methods.append((image_encoder, "encode", original_encode))
        image_encoder.encode = profiled_encode

    def _make_pre_hook(self, name: str):
        def hook(module: nn.Module, _inputs: tuple[Any, ...]) -> None:
            context = self.range(name)
            context.__enter__()
            self._module_contexts.setdefault(id(module), []).append(context)

        return hook

    def _make_post_hook(self):
        def hook(module: nn.Module, _inputs: tuple[Any, ...], _output: Any) -> None:
            contexts = self._module_contexts.get(id(module), [])
            if contexts:
                contexts.pop().__exit__(None, None, None)

        return hook

    def start(self) -> None:
        if self._profiler is not None:
            self._profiler.start()

    @contextmanager
    def range(self, name: str):
        if not self.enabled:
            yield
            return
        record_context = None
        if self.capture_trace:
            record_context = torch.profiler.record_function(name)
            record_context.__enter__()
        start = None
        if self._active_step():
            start = torch.cuda.Event(enable_timing=True)
            start.record()
        try:
            yield
        finally:
            if start is not None:
                end = torch.cuda.Event(enable_timing=True)
                end.record()
                self._active_stage_events.append((name, start, end))
            if record_context is not None:
                record_context.__exit__(None, None, None)

    def _active_step(self) -> bool:
        next_step = self.step_index + 1
        return self.warmup_steps < next_step <= self.warmup_steps + self.active_steps

    def begin_step(self) -> None:
        if not self.enabled:
            return
        torch.cuda.synchronize(self.device)
        self._step_start_s = time.perf_counter()

    def end_step(self, pair_count: int, source_crop_count: int) -> None:
        if not self.enabled:
            return
        torch.cuda.synchronize(self.device)
        if self._step_start_s is None:
            raise RuntimeError("P2SD profiler ended a step that did not start")
        self.step_index += 1
        if self.warmup_steps < self.step_index <= self.warmup_steps + self.active_steps:
            self._active_step_seconds.append(time.perf_counter() - self._step_start_s)
            self._active_pair_counts.append(int(pair_count))
            self._active_source_crop_counts.append(int(source_crop_count))
            for name, start, end in self._active_stage_events:
                self._stage_cuda_milliseconds[name].append(float(start.elapsed_time(end)))
        self._active_stage_events.clear()
        self._step_start_s = None
        if self._profiler is not None:
            self._profiler.step()

    def close(self) -> None:
        if not self.enabled:
            return
        if self._profiler is not None:
            self._profiler.stop()
            self._profiler.export_chrome_trace(str(self.trace_path))
        for handle in self._hook_handles:
            handle.remove()
        self._hook_handles.clear()
        for owner, attribute, original in self._restore_methods:
            setattr(owner, attribute, original)
        self._restore_methods.clear()
        peak_allocated = torch.cuda.max_memory_allocated(self.device)
        peak_reserved = torch.cuda.max_memory_reserved(self.device)
        event_rows = (
            aggregate_event_rows(self._profiler.key_averages())
            if self._profiler is not None
            else []
        )
        stage_rows = [row for row in event_rows if row["key"].startswith(("target_ae/", "p2sd/"))]
        sdpa_rows = [
            row
            for row in event_rows
            if "scaled_dot_product" in row["key"] or "flash_attention" in row["key"]
        ]
        active_total_s = sum(self._active_step_seconds)
        active_pairs = sum(self._active_pair_counts)
        active_source_crops = sum(self._active_source_crop_counts)
        summary = {
            "status": "complete",
            "warmup_steps": self.warmup_steps,
            "active_steps": self.active_steps,
            "timed_steps": len(self._active_step_seconds),
            "median_step_seconds": median_or_none(self._active_step_seconds),
            "mean_step_seconds": mean_or_none(self._active_step_seconds),
            "source_crops_per_second": (
                active_source_crops / active_total_s if active_total_s > 0 else None
            ),
            "source_crops_per_step": mean_or_none(
                [float(value) for value in self._active_source_crop_counts]
            ),
            "prompt_pairs_per_second": active_pairs / active_total_s if active_total_s > 0 else None,
            "prompt_pairs_per_step": mean_or_none([float(value) for value in self._active_pair_counts]),
            "peak_memory_allocated_bytes": int(peak_allocated),
            "peak_memory_reserved_bytes": int(peak_reserved),
            "chrome_trace": str(self.trace_path) if self.capture_trace else None,
            "capture_trace": self.capture_trace,
            "cuda_stage_timings_ms": {
                name: {
                    "calls": len(values),
                    "mean_ms": mean_or_none(values),
                    "median_ms": median_or_none(values),
                    "total_ms": float(sum(values)),
                }
                for name, values in sorted(self._stage_cuda_milliseconds.items())
            },
            "stage_events": stage_rows,
            "sdpa_events": sdpa_rows,
            "uses_flash_sdpa": any("flash" in row["key"].lower() for row in sdpa_rows),
        }
        write_json(self.summary_path, summary)

    @staticmethod
    def _event_row(event: Any) -> dict[str, Any]:
        return {
            "key": str(event.key),
            "calls": int(event.count),
            "cpu_total_us": float(getattr(event, "cpu_time_total", 0.0)),
            "device_total_us": float(
                getattr(event, "device_time_total", getattr(event, "cuda_time_total", 0.0))
            ),
            "self_device_total_us": float(
                getattr(event, "self_device_time_total", getattr(event, "self_cuda_time_total", 0.0))
            ),
        }


def mean_or_none(values: list[float]) -> float | None:
    return float(statistics.fmean(values)) if values else None


def median_or_none(values: list[float]) -> float | None:
    return float(statistics.median(values)) if values else None


def aggregate_event_rows(events: Any) -> list[dict[str, Any]]:
    rows_by_key: dict[str, dict[str, Any]] = {}
    for event in events:
        row = P2SDStepProfiler._event_row(event)
        existing = rows_by_key.get(row["key"])
        if existing is None:
            rows_by_key[row["key"]] = row
            continue
        existing["calls"] += row["calls"]
        existing["cpu_total_us"] += row["cpu_total_us"]
        existing["device_total_us"] += row["device_total_us"]
        existing["self_device_total_us"] += row["self_device_total_us"]
    return sorted(rows_by_key.values(), key=lambda row: row["key"])
