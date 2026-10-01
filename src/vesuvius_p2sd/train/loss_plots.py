"""Best-effort loss-vs-epoch plots written to a run dir during training.

Kept deliberately dependency-tolerant: matplotlib is imported lazily inside the
function so importing this module is cheap (the scene-eval shards import the
training package), and a missing matplotlib or malformed metrics never raises
into the training loop -- plotting is telemetry, not training.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Sequence

DEFAULT_LOSS_KEYS = ("latent_mse", "latent_mse_loss", "geo_loss", "coordinate_query_loss", "aux_mse", "loss")


def write_loss_plots(
    run_dir: str | Path,
    *,
    loss_keys: Sequence[str] = DEFAULT_LOSS_KEYS,
    out_subdir: str = "plots",
) -> list[str]:
    """Write one ``loss_<key>.png`` per loss (loss vs epoch) into run_dir/plots.

    Returns the list of written paths (empty on any failure). Safe to call every
    N epochs from the training loop.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return []

    run_dir = Path(run_dir)
    metrics_path = run_dir / "metrics.jsonl"
    if not metrics_path.exists():
        return []

    by_epoch: dict[str, dict[float, float]] = {k: {} for k in loss_keys}
    try:
        with metrics_path.open() as f:
            for line in f:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if row.get("split", "train") != "train" or "epoch" not in row:
                    continue
                # Keep every logged training window: rounding to integer epochs
                # discarded most points and validation could overwrite them.
                epoch = float(row["epoch"])
                for key in loss_keys:
                    value = row.get(key)
                    if isinstance(value, (int, float)) and math.isfinite(value):
                        by_epoch[key][epoch] = float(value)
    except OSError:
        return []

    out_dir = run_dir / out_subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[str] = []
    for key in loss_keys:
        series = by_epoch[key]
        if not series:
            continue
        epochs = sorted(series)
        values = [series[e] for e in epochs]
        try:
            fig = plt.figure(figsize=(8, 5))
            plt.plot(epochs, values, lw=1.3, color="#1f77b4")
            if all(v > 0 for v in values):
                plt.yscale("log")
            plt.xlabel("epoch")
            label = {"latent_mse": "Latent MSE (raw)",
                     "latent_mse_loss": "Latent MSE (channel normalized)",
                     "geo_loss": "Geometric consistency loss"}.get(key, key)
            plt.ylabel(label)
            plt.title(f"{label} vs epoch — {run_dir.name}")
            plt.grid(True, which="both", alpha=0.25)
            plt.tight_layout()
            path = out_dir / f"loss_{key}.png"
            fig.savefig(path, dpi=120)
            plt.close(fig)
            written.append(str(path))
        except Exception:
            try:
                plt.close("all")
            except Exception:
                pass
    return written
