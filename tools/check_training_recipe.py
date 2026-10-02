#!/usr/bin/env python3
"""Exercise the scratch-training chain on small synthetic volumes, without CUDA."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import yaml

ROOT = Path(__file__).resolve().parents[1]


def run(module: str, *arguments: str) -> None:
    environment = dict(os.environ)
    environment.update(PYTHONPATH=str(ROOT / "src"), OMP_NUM_THREADS="2", MKL_NUM_THREADS="2")
    subprocess.run(
        [sys.executable, "-m", module, *arguments],
        cwd=ROOT, env=environment, check=True,
    )


def small_config(name: str, output: Path) -> dict:
    config = yaml.safe_load((ROOT / "configs/training" / name).read_text())
    config["device"] = "cpu"
    config["run"]["output_root"] = str(output)
    config["data"].update(
        synthetic=True, synthetic_length=4, patch_size=[64, 64, 64],
        same_case_pair_batches=False, load_ignore=False,
    )
    config["data"]["augmentation"] = {}
    config["training"].update(
        batch_size=2, num_workers=0, max_steps=2, amp_dtype="float32",
        channels_last_3d=False, eval_on_last_step=False,
        save_interval_steps=1, log_interval_steps=1,
    )
    for key in ("save_interval_epochs", "log_interval_epochs"):
        config["training"].pop(key, None)
    config["optimizer"]["schedule"] = {"name": "constant"}
    config["visualization"] = {"save_png": False, "save_3d": False}
    return config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = (args.output or Path(tempfile.mkdtemp(prefix="plats-train-check-"))).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        parser.error("--output must be empty; checkpoints are not overwritten")

    ae = small_config("ae_scratch.yaml", output)
    ae_path = output / "ae.yaml"
    ae_path.write_text(yaml.safe_dump(ae, sort_keys=False))
    run("vesuvius_p2sd.train.train_ae", "--config_path", str(ae_path))

    p2sd = small_config("p2sd_scratch.yaml", output)
    p2sd["target_ae"].update(
        config_path=str(ae_path), checkpoint_path=str(output / "ae_scratch/last.pt"),
    )
    stats_path = output / "ae_scratch/latent_stats.json"
    p2sd["p2sd"]["loss"]["latent_normalization"]["stats_path"] = str(stats_path)
    p2sd["p2sd"]["dense_aux"]["crop_grid"] = 1
    p2sd_path = output / "p2sd.yaml"
    p2sd_path.write_text(yaml.safe_dump(p2sd, sort_keys=False))
    run(
        "vesuvius_p2sd.research.estimate_ae_latent_stats",
        "--config_path", str(p2sd_path), "--output_path", str(stats_path),
        "--split", "train", "--max_batches", "2",
    )
    run("vesuvius_p2sd.train.train_p2sd", "--config_path", str(p2sd_path))

    import torch

    for path in (output / "ae_scratch/last.pt", output / "p2sd_scratch/last.pt",
                 output / "p2sd_scratch/dense_last.pt"):
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        assert checkpoint["step"] == 2, (path, checkpoint.get("step"))
        assert all(torch.isfinite(value).all() for value in checkpoint["model"].values())
    result = {"status": "passed", "steps_per_stage": 2, "crop": [64] * 3,
              "device": "cpu", "output": str(output),
              "scope": "synthetic scratch optimization; not convergence or GPU performance"}
    (output / "validation.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
