"""Load a jointly trained P2SD model and its own foreground decoder."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from vesuvius_p2sd.models.binary_seg import BinarySegFromP2SD
from vesuvius_p2sd.models.p2sd import build_p2sd
from vesuvius_p2sd.train.common import load_ae_from_config
from vesuvius_p2sd.train.train_p2sd import build_static_latent_codec
from vesuvius_p2sd.utils.config import load_config


def load_joint_run(
    run_dir: str | Path,
    *,
    repository_root: str | Path,
    device: torch.device,
    checkpoint_dir: str | Path | None = None,
    load_foreground: bool = True,
) -> tuple[Any, Any, Any, Any, dict]:
    """Load an AE, P2SD and optional co-trained binary head for inference.

    The P2SD checkpoint stores the training config. Relative paths in that
    config are relative to the repository root, as in the training commands.
    A snapshot directory can select an intermediate matching checkpoint pair.
    Head and P2SD steps must agree: independently read live ``last`` files may
    otherwise come from different steps while training is saving checkpoints.
    """
    root = Path(repository_root).resolve()
    run = Path(run_dir).resolve()
    checkpoints = Path(checkpoint_dir).resolve() if checkpoint_dir else run

    def resolve(value: str | Path) -> Path:
        path = Path(value)
        return path if path.is_absolute() else root / path

    p2sd_path = checkpoints / "last.pt"
    saved = torch.load(p2sd_path, map_location="cpu", weights_only=False)
    config = saved.get("config") or load_config(run / "resolved_config.yaml")
    target = config["target_ae"]
    ae_config = resolve(target["config_path"])
    ae_path = resolve(target["checkpoint_path"])
    ae, _ = load_ae_from_config(str(ae_config), str(ae_path))
    ae = ae.to(device).eval()
    model = build_p2sd(config, latent_channels=ae.latent_channels).to(device).eval()
    model.load_state_dict(saved["model"], strict=True)

    normalization = dict(config["p2sd"]["loss"].get("latent_normalization", {}))
    files = {"p2sd": str(p2sd_path), "ae": str(ae_path), "ae_config": str(ae_config)}
    for key in ("stats_path", "channel_stats_path"):
        if normalization.get(key):
            normalization[key] = str(resolve(normalization[key]))
            files["latent_statistics"] = normalization[key]
    codec = build_static_latent_codec(
        normalization, latent_channels=ae.latent_channels, device=device,
    )

    foreground = None
    if load_foreground:
        dense_path = checkpoints / "dense_last.pt"
        dense_saved = torch.load(dense_path, map_location="cpu", weights_only=False)
        if saved.get("step") is None or dense_saved.get("step") != saved["step"]:
            raise ValueError(
                "P2SD and binary checkpoint steps must match; use a completed "
                "checkpoints/step_XXXXXX snapshot. "
                f"Got {saved.get('step')} and {dense_saved.get('step')}."
            )
        dense_config = dense_saved.get("dense_aux") or config["p2sd"]["dense_aux"]
        refiner = dense_config.get("refiner", {})
        foreground = BinarySegFromP2SD(
            model,
            decoder_channels=dense_config.get("decoder_channels", (512, 256, 128, 64, 32, 16)),
            num_groups=int(dense_config.get("num_groups", 8)),
            dropout=float(dense_config.get("dropout", 0.0)),
            refiner_depth=int(refiner.get("depth", 0)),
            refiner_heads=int(refiner.get("heads", 8)),
            refiner_mlp_ratio=float(refiner.get("mlp_ratio", 4.0)),
            refiner_rope_mode=str(refiner.get("rope_mode", "per_axis")),
        ).to(device).eval()
        # Check each non-trunk module strictly, without copying a second trunk.
        head_state = dense_saved["model"]
        modules = {"context_refiner": foreground.context_refiner,
                   "decoder": foreground.decoder, "head": foreground.head}
        expected = {f"{name}.{key}" for name, module in modules.items()
                    for key in module.state_dict()}
        if set(head_state) != expected:
            raise ValueError(
                f"Binary checkpoint keys differ: missing={sorted(expected - set(head_state))}, "
                f"unexpected={sorted(set(head_state) - expected)}"
            )
        for name, module in modules.items():
            module.load_state_dict(
                {key[len(name) + 1:]: value for key, value in head_state.items()
                 if key.startswith(name + ".")}, strict=True,
            )
        files["binary_head"] = str(dense_path)

    for module in (model, ae):
        module.requires_grad_(False)
    if foreground is not None:
        foreground.requires_grad_(False)
    provenance = {"mode": "joint_training", "run_dir": str(run),
                  "checkpoint_dir": str(checkpoints), "step": saved.get("step"),
                  "files": files}
    return model, ae, codec, foreground, provenance
