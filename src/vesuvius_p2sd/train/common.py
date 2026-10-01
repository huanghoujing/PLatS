"""Shared training helpers."""

from __future__ import annotations

import json
import math
import os
import random
import shutil
import subprocess
import sys
import importlib
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Sampler

from vesuvius_p2sd.data.dataset import build_dataset, collate_sheet_batch
from vesuvius_p2sd.data.probes import FixedProbeDataset
from vesuvius_p2sd.models.ae import build_sheet_ae
from vesuvius_p2sd.research.run_status import init_run_dir, update_heartbeat, write_json
from vesuvius_p2sd.utils.config import get_run_dir, load_config
from vesuvius_p2sd.utils.param_groups import build_param_groups


def prepare_run(
    *,
    config_path: str,
    overrides: list[str],
    run_id: str | None,
    entrypoint: str,
) -> tuple[dict[str, Any], Path]:
    cfg = load_config(config_path, overrides)
    if run_id is not None:
        cfg.setdefault("run", {})["run_id"] = run_id
    run_dir = get_run_dir(cfg)
    init_run_dir(run_dir, command=sys.argv)
    (run_dir / "pid.txt").write_text(f"{os.getpid()}\n", encoding="utf-8")
    write_run_metadata(run_dir, cfg)
    update_heartbeat(run_dir, status="starting", entrypoint=entrypoint)
    return cfg, run_dir


def write_run_metadata(run_dir: Path, cfg: dict[str, Any]) -> None:
    with (run_dir / "resolved_config.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    def git_output(*args: str) -> str:
        try:
            return subprocess.check_output(
                ["git", *args],
                text=True,
                stderr=subprocess.STDOUT,
            ).strip()
        except Exception as exc:  # pragma: no cover - environment dependent
            return f"git {' '.join(args)} unavailable: {exc}"

    git_commit = git_output("rev-parse", "HEAD")
    git_status = git_output("status", "--short")
    write_json(run_dir / "env.json", {
        "cwd": os.getcwd(),
        "python": sys.executable,
        "argv": sys.argv,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "git_commit": git_commit,
        "git_dirty": bool(git_status) and not git_status.startswith("git "),
    })
    (run_dir / "git_status.txt").write_text(
        f"commit {git_commit}\n{git_status}\n", encoding="utf-8"
    )
    write_json(run_dir / "lineage.json", build_lineage_record(cfg))
    write_json(run_dir / "best.json", {})


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class SameCasePairBatchSampler(Sampler[list[int]]):
    """Batches whose first two indices resolve to the same manifest row.

    ``VesuviusSheetPatchDataset`` maps ``index -> row`` via ``index % num_rows``
    and seeds its per-access rng with the raw index, so ``r`` and
    ``r + num_rows`` load the same case twice with independent component draws.
    That gives every batch one same-volume candidate pair for the AE latent
    repulsion loss (the loss guards against the occasional draw of the same
    component twice). The remaining slots come from other rows, walked through
    shuffled permutations that persist across epochs so long runs still cover
    every case evenly. Batch count per epoch matches the plain shuffled loader
    (``ceil(num_samples / batch_size)``), so epoch/step accounting is unchanged.
    """

    def __init__(self, *, num_rows: int, num_samples: int, batch_size: int, seed: int) -> None:
        if batch_size < 2:
            raise ValueError(f"same_case_pair_batches needs batch_size >= 2, got {batch_size}")
        if num_rows < 2:
            raise ValueError(f"same_case_pair_batches needs at least 2 manifest rows, got {num_rows}")
        self.num_rows = int(num_rows)
        self.num_samples = int(num_samples)
        self.batch_size = int(batch_size)
        self._rng = np.random.default_rng(int(seed) + 617)
        self._pool: list[int] = []

    def __len__(self) -> int:
        return max(1, math.ceil(self.num_samples / self.batch_size))

    def _draw_rows(self, count: int) -> list[int]:
        rows = []
        while len(rows) < count:
            if not self._pool:
                self._pool = [int(v) for v in self._rng.permutation(self.num_rows)]
            rows.append(self._pool.pop())
        return rows

    def __iter__(self):
        for _ in range(len(self)):
            rows = self._draw_rows(self.batch_size - 1)
            yield [rows[0], rows[0] + self.num_rows, *rows[1:]]


def make_loader(cfg: dict[str, Any], split: str, *, shuffle: bool) -> DataLoader:
    data_cfg = cfg.get("data", {})
    ds = build_dataset(data_cfg, split)
    training = cfg.get("training", {})
    batch_size = int(training.get("batch_size", 1))
    workers = int(training.get("num_workers", 0))
    loader_kwargs = {
        "num_workers": workers,
        "pin_memory": torch.cuda.is_available(),
        "collate_fn": collate_sheet_batch,
    }
    if split == "train" and bool(data_cfg.get("same_case_pair_batches", False)):
        rows = getattr(ds, "rows", None)
        if not rows:
            raise ValueError("data.same_case_pair_batches requires a manifest-backed dataset")
        loader_kwargs["batch_sampler"] = SameCasePairBatchSampler(
            num_rows=len(rows),
            num_samples=len(ds),
            batch_size=batch_size,
            seed=int(data_cfg.get("seed", 42)),
        )
    else:
        loader_kwargs["batch_size"] = batch_size
        loader_kwargs["shuffle"] = shuffle
    if workers > 0:
        loader_kwargs["persistent_workers"] = bool(training.get("persistent_workers", False))
        loader_kwargs["timeout"] = float(training.get("data_timeout_seconds", 0))
        if training.get("prefetch_factor") is not None:
            loader_kwargs["prefetch_factor"] = int(training.get("prefetch_factor"))
    return DataLoader(ds, **loader_kwargs)


def make_fixed_probe_loader(cfg: dict[str, Any]) -> DataLoader | None:
    """Build the small deterministic probe loader configured for diagnostics."""

    viz_cfg = cfg.get("visualization", {})
    probe_cfg = viz_cfg.get("probes", {})
    if not bool(probe_cfg.get("enabled", False)):
        return None
    manifest_path = probe_cfg.get("manifest_path")
    if not manifest_path:
        raise ValueError("visualization.probes.enabled requires manifest_path")
    data_cfg = cfg.get("data", {})
    if bool(data_cfg.get("synthetic", False)):
        raise ValueError("Fixed probe manifests require a real dataset, not data.synthetic=true")
    patch_size = tuple(int(value) for value in data_cfg.get("patch_size", data_cfg.get("crop_size", [64, 64, 64])))
    ds = FixedProbeDataset(
        dataset_root=data_cfg["dataset_root"],
        manifest_path=manifest_path,
        patch_size=patch_size,
        split=str(probe_cfg.get("split", "val")),
    )
    training = cfg.get("training", {})
    batch_size = int(probe_cfg.get("batch_size", training.get("batch_size", 1)))
    workers = int(probe_cfg.get("num_workers", 0))
    loader_kwargs = {
        "batch_size": batch_size,
        "shuffle": False,
        "num_workers": workers,
        "pin_memory": torch.cuda.is_available(),
        "collate_fn": collate_sheet_batch,
    }
    if workers > 0:
        loader_kwargs["persistent_workers"] = bool(probe_cfg.get("persistent_workers", False))
        if probe_cfg.get("prefetch_factor") is not None:
            loader_kwargs["prefetch_factor"] = int(probe_cfg["prefetch_factor"])
    return DataLoader(ds, **loader_kwargs)


def cycle(loader: DataLoader):
    while True:
        for batch in loader:
            yield batch


def get_device(cfg: dict[str, Any]) -> torch.device:
    requested = str(cfg.get("device", cfg.get("training", {}).get("device", "cuda")))
    if requested.startswith("cuda") and not torch.cuda.is_available():
        requested = "cpu"
    return torch.device(requested)


def amp_dtype(name: str | None) -> torch.dtype | None:
    if name is None or str(name).lower() in {"none", "float32", "fp32"}:
        return None
    name = str(name).lower()
    if name in {"bfloat16", "bf16"}:
        return torch.bfloat16
    if name in {"float16", "fp16"}:
        return torch.float16
    raise ValueError(f"Unsupported amp dtype: {name}")


def autocast_context(device: torch.device, dtype: torch.dtype | None):
    enabled = dtype is not None and device.type == "cuda"
    return torch.autocast(device_type=device.type, dtype=dtype or torch.float32, enabled=enabled)


def move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    out = dict(batch)
    for key in [
        "image",
        "mask",
        "component_label",
        "prompt_points",
        "prompt_labels",
        "component_id",
        "distance_field",
        "query_points",
        "query_distances",
    ]:
        if key in out and torch.is_tensor(out[key]):
            out[key] = out[key].to(device, non_blocking=True)
    return out


def maybe_channels_last_3d(tensor: torch.Tensor, enabled: bool) -> torch.Tensor:
    if enabled and tensor.ndim == 5:
        return tensor.contiguous(memory_format=torch.channels_last_3d)
    return tensor


def dice_loss_from_logits(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    # Keep scalar reconstruction math out of the BF16 autocast region. The
    # Conv3d path remains BF16; this only promotes sigmoid and reductions.
    prob = torch.sigmoid(logits.float())
    target = target.float()
    dims = tuple(range(1, prob.ndim))
    inter = (prob * target).sum(dim=dims)
    denom = prob.sum(dim=dims) + target.sum(dim=dims)
    dice = (2 * inter + eps) / (denom + eps)
    return 1.0 - dice.mean()


def weighted_bce_with_logits(
    logits: torch.Tensor,
    target: torch.Tensor,
    *,
    pos_weight: float | str | None = None,
    max_auto_pos_weight: float = 32.0,
) -> torch.Tensor:
    logits = logits.float()
    target = target.float()
    if pos_weight is None or pos_weight == 1 or pos_weight == "none":
        return torch.nn.functional.binary_cross_entropy_with_logits(logits, target)
    if isinstance(pos_weight, str) and pos_weight == "auto":
        pos = target.sum().clamp_min(1.0)
        neg = target.numel() - pos
        value = (neg / pos).clamp(min=1.0, max=float(max_auto_pos_weight))
    else:
        value = torch.as_tensor(float(pos_weight), device=logits.device, dtype=logits.dtype)
    return torch.nn.functional.binary_cross_entropy_with_logits(
        logits,
        target,
        pos_weight=value.to(device=logits.device, dtype=logits.dtype),
    )


def outside_dilation_loss_from_logits(
    logits: torch.Tensor,
    target: torch.Tensor,
    *,
    tolerance: int = 2,
) -> torch.Tensor:
    prob = torch.sigmoid(logits.float())
    target = target.float()
    if tolerance <= 0:
        allowed = target > 0.5
    else:
        kernel = 2 * int(tolerance) + 1
        allowed = torch.nn.functional.max_pool3d(
            target.float(),
            kernel_size=kernel,
            stride=1,
            padding=int(tolerance),
        ) > 0.5
    outside = (~allowed).to(prob.dtype)
    denom = target.sum().clamp_min(1.0)
    return (prob * outside).sum() / denom


def border_probability_loss_from_logits(
    logits: torch.Tensor,
    *,
    border_width: int,
    target: torch.Tensor | None = None,
) -> torch.Tensor:
    if border_width <= 0:
        return logits.new_zeros(())
    prob = torch.sigmoid(logits.float())
    mask = torch.zeros_like(prob, dtype=torch.bool)
    w = int(border_width)
    mask[..., :w, :, :] = True
    mask[..., -w:, :, :] = True
    mask[..., :, :w, :] = True
    mask[..., :, -w:, :] = True
    mask[..., :, :, :w] = True
    mask[..., :, :, -w:] = True
    denom = target.float().sum().clamp_min(1.0) if target is not None else mask.sum().clamp_min(1)
    return (prob * mask.to(prob.dtype)).sum() / denom


def collect_nonfinite_gradient_details(model: torch.nn.Module) -> dict[str, Any]:
    """Return original backward failures before clipping can spread them."""

    parameters = []
    for name, parameter in model.named_parameters():
        grad = parameter.grad
        if grad is None:
            continue
        grad_float = grad.detach().float()
        finite = torch.isfinite(grad_float)
        if bool(finite.all().item()):
            continue
        finite_values = grad_float[finite]
        parameters.append({
            "name": name,
            "shape": list(grad.shape),
            "nan_count": int(torch.isnan(grad_float).sum().item()),
            "inf_count": int(torch.isinf(grad_float).sum().item()),
            "numel": int(grad.numel()),
            "finite_abs_max": (
                float(finite_values.abs().max().cpu()) if finite_values.numel() else None
            ),
        })
    return {
        "parameter_count": len(parameters),
        "parameters": parameters,
    }


def capture_rng_state() -> dict[str, Any]:
    """Capture RNG state immediately before a model forward for failure replay."""

    state: dict[str, Any] = {
        "python": random.getstate(),
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = [item.cpu() for item in torch.cuda.get_rng_state_all()]
    return state


def restore_rng_state(state: dict[str, Any]) -> None:
    """Restore a state produced by :func:`capture_rng_state`."""

    random.setstate(state["python"])
    torch.set_rng_state(state["torch_cpu"])
    if "torch_cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def copy_to_cpu(value: Any) -> Any:
    """Recursively detach tensor payloads for a portable failure reproducer."""

    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: copy_to_cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [copy_to_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(copy_to_cpu(item) for item in value)
    return value


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, sort_keys=True) + "\n")


def make_optimizer(model: torch.nn.Module, cfg: dict[str, Any]) -> torch.optim.Optimizer:
    opt_cfg = cfg.get("optimizer", {})
    groups = opt_cfg.get("groups")
    if groups:
        param_groups, summaries = build_param_groups(
            model.named_parameters(),
            groups,
            allow_unmatched=bool(opt_cfg.get("allow_unmatched", True)),
        )
        if not param_groups:
            raise ValueError("All parameters are frozen; no optimizer groups to train")
        for summary in summaries:
            print(
                f"[param_group] {summary.name} selector={summary.selector} "
                f"lr={summary.lr} matched={summary.count} "
                f"requires_grad={summary.requires_grad_count} "
                f"excluded_by_zero_lr={summary.excluded_by_zero_lr}",
                flush=True,
            )
    else:
        param_groups = [{
            "params": [p for p in model.parameters() if p.requires_grad],
            "lr": float(opt_cfg.get("lr", cfg.get("training", {}).get("lr", 1e-4))),
        }]
    name = str(opt_cfg.get("name", "adamw")).lower()
    weight_decay = float(opt_cfg.get("weight_decay", cfg.get("training", {}).get("weight_decay", 0.0)))
    if name == "adamw":
        return torch.optim.AdamW(param_groups, weight_decay=weight_decay)
    if name == "adam":
        return torch.optim.Adam(param_groups, weight_decay=weight_decay)
    raise ValueError(f"Unsupported optimizer: {name}")


def configured_lrs(optimizer: torch.optim.Optimizer) -> list[float]:
    """Snapshot the per-group learning rates the config asked for.

    Call this immediately after ``make_optimizer`` and before any
    ``load_state_dict``, then pass the result to ``restore_configured_lrs``.
    """

    return [float(group["lr"]) for group in optimizer.param_groups]


def restore_configured_lrs(
    optimizer: torch.optim.Optimizer,
    base_lrs: list[float],
) -> None:
    """Re-assert the config's learning rates after loading optimizer state.

    ``Optimizer.load_state_dict`` keeps each *saved* param group wholesale and
    copies only ``params`` across, so a checkpoint's ``lr`` silently replaces
    the one the config asked for. Two consequences, both of which this undoes:

    * Editing ``optimizer.lr`` and resuming was a no-op -- the run continued at
      the checkpoint's rate, and the log reported the config value.
    * With an LR schedule, ``LRSchedule`` captures its base LRs from the
      optimizer, so a resume would re-anchor the schedule on an already-decayed
      rate and decay a second time from there.

    The config is the authority for learning rate; the checkpoint is the
    authority for optimizer moments.
    """

    for group, base_lr in zip(optimizer.param_groups, base_lrs, strict=True):
        group["lr"] = base_lr


class LRSchedule:
    """Step-indexed learning-rate multiplier applied to every optimizer group.

    The multiplier is a pure function of the global step, so resuming needs no
    scheduler state: a run restarted at step N recomputes the same LR that an
    uninterrupted run would have had. Per-group base LRs are captured at
    construction, so discriminative-LR configs keep their ratios and ``lr: 0``
    groups stay at zero.

    ``name: constant`` (the default) reproduces the historical behaviour
    exactly, so existing configs are unaffected.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        total_steps: int,
        name: str = "constant",
        warmup_steps: int = 0,
        min_lr_scale: float = 0.0,
    ) -> None:
        self.name = str(name).lower()
        if self.name not in {"constant", "cosine"}:
            raise ValueError(f"Unsupported optimizer.schedule.name: {name!r} (use constant|cosine)")
        self.base_lrs = [float(g["lr"]) for g in optimizer.param_groups]
        self.total_steps = max(1, int(total_steps))
        self.warmup_steps = max(0, int(warmup_steps))
        self.min_lr_scale = float(min_lr_scale)
        if not 0.0 <= self.min_lr_scale <= 1.0:
            raise ValueError(f"optimizer.schedule.min_lr_scale must be in [0, 1], got {self.min_lr_scale}")
        if self.warmup_steps >= self.total_steps:
            raise ValueError(
                f"optimizer.schedule.warmup_steps ({self.warmup_steps}) must be < total steps ({self.total_steps})"
            )

    @property
    def is_constant(self) -> bool:
        return self.name == "constant" and self.warmup_steps == 0

    def scale_at(self, step: int) -> float:
        if self.warmup_steps > 0 and step <= self.warmup_steps:
            return float(step) / float(self.warmup_steps)
        if self.name == "constant":
            return 1.0
        denom = max(1, self.total_steps - self.warmup_steps)
        progress = min(1.0, max(0.0, (step - self.warmup_steps) / denom))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.min_lr_scale + (1.0 - self.min_lr_scale) * cosine

    def apply(self, optimizer: torch.optim.Optimizer, step: int) -> float:
        scale = self.scale_at(step)
        for group, base_lr in zip(optimizer.param_groups, self.base_lrs):
            group["lr"] = base_lr * scale
        return float(optimizer.param_groups[0]["lr"])

    def describe(self) -> str:
        if self.is_constant:
            return f"constant lr={self.base_lrs[0]:.3e}"
        first = self.base_lrs[0]
        return (
            f"{self.name} lr={first:.3e} -> {first * self.min_lr_scale:.3e} "
            f"over {self.total_steps} steps (warmup {self.warmup_steps})"
        )


def make_lr_schedule(
    optimizer: torch.optim.Optimizer,
    cfg: dict[str, Any],
    total_steps: int,
) -> LRSchedule:
    sched_cfg = (cfg.get("optimizer", {}) or {}).get("schedule") or {}
    return LRSchedule(
        optimizer,
        total_steps=total_steps,
        name=sched_cfg.get("name", "constant"),
        warmup_steps=int(sched_cfg.get("warmup_steps", 0) or 0),
        min_lr_scale=float(sched_cfg.get("min_lr_scale", 0.0) or 0.0),
    )


def count_parameters(model: torch.nn.Module) -> dict[str, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    tensors = sum(1 for _ in model.parameters())
    trainable_tensors = sum(1 for p in model.parameters() if p.requires_grad)
    return {
        "total": int(total),
        "trainable": int(trainable),
        "frozen": int(total - trainable),
        "tensors": int(tensors),
        "trainable_tensors": int(trainable_tensors),
    }


def format_model_report(name: str, model: torch.nn.Module) -> str:
    counts = count_parameters(model)
    return "\n".join([
        f"[model] {name}",
        (
            "[model] "
            f"parameters total={counts['total']:,} "
            f"trainable={counts['trainable']:,} "
            f"frozen={counts['frozen']:,} "
            f"tensors={counts['tensors']} "
            f"trainable_tensors={counts['trainable_tensors']}"
        ),
        str(model),
    ])


def print_model_report(
    name: str,
    model: torch.nn.Module,
    run_dir: Path,
    *,
    print_to_stdout: bool = True,
) -> None:
    report = format_model_report(name, model)
    safe_name = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in name)
    (run_dir / f"model_{safe_name}.txt").write_text(report + "\n", encoding="utf-8")
    if print_to_stdout:
        print(report, flush=True)


def save_checkpoint(
    path: Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    cfg: dict[str, Any],
    best: dict[str, Any] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.tmp")
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": int(step),
        "config": cfg,
        "best": best or {},
    }, temporary_path)
    # Keep the previous checkpoint intact until the new payload is complete.
    os.replace(temporary_path, path)


def snapshot_checkpoint_files(
    run_dir: Path, *, step: int, filenames: list[str], keep_last: int,
    keep_every_steps: int = 0,
) -> Path:
    """Publish a complete, immutable checkpoint set, then prune unpinned history.

    Call synchronously after saving every active head at the same training step.
    Copies also protect snapshots of heads whose writers overwrite in place.
    A PIN file exempts a snapshot from automatic retention.
    """
    if keep_last < 1:
        raise ValueError("checkpoint snapshots require keep_last >= 1")
    root = run_dir / "checkpoints"
    root.mkdir(parents=True, exist_ok=True)
    destination = root / f"step_{step:06d}"
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite checkpoint snapshot: {destination}")
    temporary = Path(tempfile.mkdtemp(prefix=f".step_{step:06d}_", dir=root))
    try:
        for filename in filenames:
            shutil.copy2(run_dir / filename, temporary / filename)
        write_json(temporary / "checkpoint.json", {"step": int(step), "files": filenames})
        os.replace(temporary, destination)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    snapshots = sorted(p for p in root.glob("step_*") if (p / "checkpoint.json").is_file())
    for old in snapshots[:-keep_last]:
        old_step = int(json.loads((old / "checkpoint.json").read_text())["step"])
        if (old / "PIN").exists() or (keep_every_steps > 0 and old_step % keep_every_steps == 0):
            continue
        shutil.rmtree(old)
    return destination


def load_model_state(model: torch.nn.Module, checkpoint_path: str | Path, *, strict: bool = True) -> dict[str, Any]:
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    state = ckpt.get("model", ckpt.get("model_state_dict", ckpt))
    model.load_state_dict(state, strict=strict)
    return ckpt if isinstance(ckpt, dict) else {}


class BrainSAMAETargetAdapter(torch.nn.Module):
    """Adapter for Brain-SAM AE factories with encode/decode-only usage."""

    def __init__(
        self,
        model: torch.nn.Module,
        *,
        latent_channels: int | None = None,
        use_isolated_mask_encode: bool = True,
    ) -> None:
        super().__init__()
        self.model = model
        self._latent_channels = int(
            latent_channels
            if latent_channels is not None
            else getattr(model, "embedding_dim", 0)
        )
        if self._latent_channels <= 0:
            raise ValueError("Brain-SAM AE adapter requires latent_channels or model.embedding_dim")
        self.use_isolated_mask_encode = bool(use_isolated_mask_encode)

    @property
    def latent_channels(self) -> int:
        return self._latent_channels

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_isolated_mask_encode:
            fast_encode = getattr(self.model, "encode_isolated_masks", None)
            if fast_encode is not None:
                return fast_encode(x.float())
        return self.model.encode(x.float())

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.model.decode(z)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        z = self.encode(x)
        return {"latent": z, "logits": self.decode(z)}


def load_ae_from_config(config_path: str | Path, checkpoint_path: str | Path | None = None):
    cfg = load_config(config_path, [])
    if str(cfg.get("model", {}).get("type", "")).lower() in {"brain_sam_ae", "brain_sam_factory"}:
        return load_brain_sam_ae_from_config(cfg, checkpoint_path), cfg
    model = build_sheet_ae(cfg)
    if checkpoint_path:
        load_model_state(model, checkpoint_path, strict=True)
    return model, cfg


def load_brain_sam_ae_from_config(
    cfg: dict[str, Any],
    checkpoint_path: str | Path | None,
) -> BrainSAMAETargetAdapter:
    model_cfg = cfg.get("model", {})
    repo_root = Path(model_cfg.get("repo_root", "/home/x/Project/Brain-SAM")).expanduser().resolve()
    code_root = repo_root / "Brain-SAM"
    if not code_root.exists():
        code_root = repo_root
    code_root_str = str(code_root)
    if code_root_str not in sys.path:
        sys.path.insert(0, code_root_str)

    module_name = str(model_cfg.get("module", "modeling.noise_ae_aniso_unet"))
    factory_name = str(model_cfg["factory_name"])
    factory_kwargs = dict(model_cfg.get("factory_kwargs", {}))
    if checkpoint_path:
        factory_kwargs["load_state_dict"] = False
    module = importlib.import_module(module_name)
    factory = getattr(module, factory_name)
    model = factory(**factory_kwargs)
    if checkpoint_path:
        load_model_state(
            model,
            checkpoint_path,
            strict=bool(model_cfg.get("strict", True)),
        )
    return BrainSAMAETargetAdapter(
        model,
        latent_channels=model_cfg.get("latent_channels"),
        use_isolated_mask_encode=bool(model_cfg.get("use_isolated_mask_encode", True)),
    )


def should_stop(run_dir: Path) -> bool:
    return (run_dir / "stop_requested").exists()


class MetricWindow:
    """Accumulate per-step scalar tensors on-device; emit window means.

    Training rows are logged once per epoch by default; a single-step point
    sample of the loss would be too noisy at that cadence, so the row carries
    the mean of every step since the previous row. Sums stay on the GPU
    (`detach().float()`, no `.item()`) so accumulation adds no per-step sync.
    """

    def __init__(self) -> None:
        self._sums: dict[str, torch.Tensor] = {}
        self._tracked: dict[str, list[torch.Tensor]] = {}
        self._count = 0

    def add(self, values: dict[str, torch.Tensor | None]) -> None:
        self._count += 1
        for key, value in values.items():
            if value is None:
                continue
            value = value.detach().float()
            existing = self._sums.get(key)
            self._sums[key] = value if existing is None else existing + value

    def track(self, key: str, value: torch.Tensor | None) -> None:
        """Retain the full per-step history for a metric, not just its sum.

        A mean cannot show a spike, a saturating threshold, or the moment a
        distribution shifts. `grad_norm` is the motivating case: at a window
        mean of 8.9 against `grad_clip: 1.0` you cannot tell from the row
        whether clipping fired on every step or none. Costs one small tensor
        per step and a single sync per window.
        """
        if value is None:
            return
        self._tracked.setdefault(key, []).append(value.detach().float().reshape(()))

    def distribution_stats(self, *, clip_thresholds: dict[str, float] | None = None) -> dict[str, float]:
        out: dict[str, float] = {}
        for key, values in self._tracked.items():
            if not values:
                continue
            stacked = torch.stack(values)
            out[f"{key}_max"] = float(stacked.max().item())
            out[f"{key}_p99"] = float(stacked.quantile(0.99).item())
            out[f"{key}_p50"] = float(stacked.median().item())
            threshold = (clip_thresholds or {}).get(key)
            if threshold is not None:
                out[f"{key}_clip_hit_rate"] = float((stacked > threshold).float().mean().item())
        return out

    def means(self) -> dict[str, float]:
        if not self._count:
            return {}
        return {key: float((value / self._count).item()) for key, value in self._sums.items()}

    @property
    def window_steps(self) -> int:
        """Steps covered by the current window; a row's means are over this."""
        return self._count

    def reset(self) -> None:
        self._sums = {}
        self._tracked = {}
        self._count = 0


def resolve_log_interval_steps(training_cfg: dict, steps_per_epoch: int) -> int:
    """Training-row cadence: explicit config wins, else one row per epoch."""
    if "log_interval_epochs" in training_cfg:
        return max(1, int(round(float(training_cfg["log_interval_epochs"]) * max(1, steps_per_epoch))))
    if "log_interval_steps" in training_cfg:
        return max(1, int(training_cfg["log_interval_steps"]))
    return max(1, int(steps_per_epoch))


def build_lineage_record(cfg: dict[str, Any]) -> dict[str, Any]:
    """Machine-readable run ancestry, so history lives in data, not run names.

    Parent = the run whose checkpoint this run resumes; the target AE is
    recorded separately for P2SD runs. `run.lineage_reason` is the one-line
    human why (set it in the config or as an override on every resume).
    """
    import datetime

    def run_of(path_value: Any) -> dict[str, Any] | None:
        if not path_value:
            return None
        path = Path(str(path_value))
        parts = path.parts
        run_name = parts[parts.index("runs") + 1] if "runs" in parts else None
        return {"run": run_name, "checkpoint": str(path)}

    training_cfg = cfg.get("training", {})
    return {
        "created_at": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        "reason": str(cfg.get("run", {}).get("lineage_reason", "")) or None,
        "parent": run_of(
            training_cfg.get("resume_checkpoint_path") or training_cfg.get("resume_from")),
        "target_ae": run_of(cfg.get("target_ae", {}).get("checkpoint_path")),
    }
