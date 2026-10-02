#!/usr/bin/env python3
"""Check joint checkpoint loading and shared-context inference on a 64³ volume."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vesuvius_p2sd.joint_inference import load_joint_run
from vesuvius_p2sd.research.auto_instance_seg import run_case_cluster


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_dir", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(2)
    device = torch.device("cpu")
    model, ae, codec, foreground, source = load_joint_run(
        args.run_dir, repository_root=ROOT, device=device,
    )
    assert foreground.trunk is model
    image = np.full((64, 64, 64), 100, dtype=np.uint8)
    tensor = torch.from_numpy(image[None, None].astype(np.float32))
    with torch.inference_mode():
        cache = model.encode_image_context_from_image(tensor)
        cached_logits = foreground.decode_context(foreground.refine_context(cache[3]))
        direct_logits = foreground(tensor)
        torch.testing.assert_close(cached_logits, direct_logits, rtol=0, atol=0)

        # A zero threshold guarantees seed candidates for this plumbing check;
        # it is not a segmentation-quality evaluation or production threshold.
        mask = cached_logits.sigmoid()[0, 0].numpy() >= 0.0
        settings = dict(
            binseg_model=None, p2sd_model=model, target_ae=ae, latent_codec=codec,
            image=image, valid=np.ones(image.shape, bool), device=device,
            dtype=torch.float32, binary_threshold=0.0, sheet_threshold=0.5,
            min_component_voxels=1, cluster_points=8, cluster_min_points=1,
            cluster_mse_threshold=1e6, prompt_points_per_sheet=1,
            external_mask=mask, seed=13,
        )
        with patch.object(model, "encode_image_context_from_image",
                          wraps=model.encode_image_context_from_image) as encode:
            cached = run_case_cluster(**settings, image_context_cache=cache)
            assert encode.call_count == 0, "Cached inference encoded CT again"
            uncached = run_case_cluster(**settings)
            assert encode.call_count == 1
        np.testing.assert_array_equal(cached["instance_ids"], uncached["instance_ids"])
        assert cached["points"] == uncached["points"]

    with tempfile.TemporaryDirectory(prefix="plats-mismatched-head-") as directory:
        snapshot = Path(directory)
        (snapshot / "last.pt").symlink_to((args.run_dir / "last.pt").resolve())
        torch.save({"step": int(source["step"]) + 1, "model": {}}, snapshot / "dense_last.pt")
        try:
            load_joint_run(args.run_dir, repository_root=ROOT, device=device,
                           checkpoint_dir=snapshot)
        except ValueError as error:
            assert "steps must match" in str(error), error
        else:
            raise AssertionError("Mismatched P2SD/binary steps were accepted")

    result = {"status": "passed", "step": source["step"], "crop": [64] * 3,
              "same_trunk_object": True, "binary_logits_bitwise_equal": True,
              "cached_and_uncached_instances_equal": True,
              "cached_clustering_extra_image_encodes": 0,
              "mismatched_checkpoint_steps_rejected": True}
    if args.output:
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
