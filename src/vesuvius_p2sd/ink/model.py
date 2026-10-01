"""Strict adapter for the pinned published ink model; no training side effects."""
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch

from .baseline import sha256, tiled_predict


class PublishedInk:
    def __init__(self, settings):
        self.settings = settings
        root = Path(settings["villa_root"]).resolve()
        revision = subprocess.check_output(["git","-C",str(root),"rev-parse","HEAD"],text=True).strip()
        if revision != settings["villa_revision"]:
            raise ValueError("Unexpected upstream model revision")
        if subprocess.check_output(["git","-C",str(root),"status","--porcelain"],text=True).strip():
            raise ValueError("Upstream source has modifications")
        self.checkpoint_hash = sha256(Path(settings["checkpoint"]))
        if self.checkpoint_hash != settings["checkpoint_sha256"]:
            raise ValueError("Unexpected checkpoint hash")
        sys.path.insert(0,str(root/"vesuvius/src"))
        from vesuvius.ink_detection.config import InkConfig
        from vesuvius.ink_detection.models.model import make_model
        from vesuvius.ink_detection.models.checkpoint import select_inference_weights
        from vesuvius.ink_detection.data.normalization import normalize_image
        payload = torch.load(settings["checkpoint"],map_location="cpu",weights_only=True)
        self.training_config = payload["config"]
        self.config = InkConfig.from_mapping(self.training_config)
        self.depth,self.patch,px = self.config.model.crop_size
        if self.config.data.mode != "flat" or self.patch != px:
            raise ValueError("Expected square flat-model patches")
        self.device = torch.device(settings["device"])
        self.model = make_model(self.config)
        self.weights_name, weights = select_inference_weights(payload)
        self.model.load_state_dict(weights,strict=True)
        self.model.to(self.device).eval()
        self.normalize = normalize_image

    @torch.inference_mode()
    def batch(self, batch):
        normalized = np.stack([self.normalize(p.copy(),self.config.data.normalization) for p in batch])
        image = torch.from_numpy(normalized[:,None]).to(self.device)
        return self.model(image)["ink"][:,0].float().sigmoid().cpu().numpy()

    def predict(self, image, start, reverse=False):
        if start < 0 or start+self.depth > image.shape[0]:
            raise ValueError("Invalid depth window")
        window = image[start:start+self.depth]
        if reverse:
            window = window[::-1].copy()
        return tiled_predict(window,self.batch,patch=self.patch,stride=self.settings["stride"],
                             batch_size=self.settings["batch_size"])
