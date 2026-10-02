"""Inference and native-coordinate data handling, independent of the web UI."""
from __future__ import annotations

import hashlib
import json
import struct
import time
from pathlib import Path

import numpy as np

from .inputs import load_array, load_crop
from .exports import export_session


def pad_crop(image: np.ndarray):
    offset = (320 - np.array(image.shape)) // 2
    bounds = tuple(slice(int(o), int(o + n)) for o, n in zip(offset, image.shape))
    canvas = np.zeros((320, 320, 320), dtype=np.uint8)
    canvas[bounds] = image
    return canvas, bounds, offset


def validate_points(points, shape) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3 or not 1 <= len(points) <= 8:
        raise ValueError('Use 1–8 positive points on the same sheet.')
    if not np.isfinite(points).all() or not ((points >= 0) & (points < shape)).all():
        raise ValueError('A prompt lies outside the native crop.')
    return points


def packed_mask(mask: np.ndarray) -> bytes:
    return np.packbits(mask.reshape(-1), bitorder='little').tobytes()


def surface_bytes(mask: np.ndarray) -> bytes:
    """Exact voxel-mask isosurface, closed at crop edges, in array coordinates.

    Little-endian wire format: uint32 vertex/triangle counts, float32 XYZ
    vertices, uint32 triangle indices. No decimation of thin sheets is applied.
    """
    from skimage.measure import marching_cubes
    where = np.nonzero(mask)
    if not len(where[0]):
        return struct.pack('<II', 0, 0)
    lower = np.array([a.min() for a in where])
    upper = np.array([a.max() + 1 for a in where])
    bounds = tuple(slice(int(a), int(b)) for a, b in zip(lower, upper))
    block = np.pad(mask[bounds], 1).astype(np.float32)
    vertices, faces, _, _ = marching_cubes(block, level=0.5, allow_degenerate=False)
    vertices += lower - 1
    return (struct.pack('<II', len(vertices), len(faces))
            + vertices.astype('<f4').tobytes() + faces.astype('<u4').tobytes())


class PromptEngine:
    """One model stack and one cached CT context; called by a single worker."""
    def __init__(self, repository_root, bundle_root, device='cuda:0',
                 run_dir=None, checkpoint_dir=None, threads=4):
        self.root = Path(repository_root)
        self.bundle = Path(bundle_root)
        self.device_name = device
        self.run_dir, self.checkpoint_dir = run_dir, checkpoint_dir
        self.threads = threads
        self.model = None
        self.context = None
        self.image_key = None
        self.encode_count = 0
        self.source = {}
        self.foreground = None

    def _load(self):
        import torch
        from vesuvius_p2sd.models.ae import build_sheet_ae
        from vesuvius_p2sd.models.p2sd import build_p2sd
        from vesuvius_p2sd.train.common import load_model_state
        from vesuvius_p2sd.train.train_p2sd import build_static_latent_codec
        from vesuvius_p2sd.utils.config import load_config
        self.device = torch.device(self.device_name)
        torch.set_num_threads(self.threads)
        if self.device.type == 'cuda' and not torch.cuda.is_available():
            raise ValueError('CUDA is unavailable; choose an available GPU or --device cpu.')
        if self.device.type == 'cuda' and not torch.cuda.is_bf16_supported():
            raise ValueError('The reference inference requires a BF16-capable GPU.')
        self.dtype = torch.bfloat16 if self.device.type == 'cuda' else torch.float32
        if self.run_dir:
            from vesuvius_p2sd.joint_inference import load_joint_run
            self.model, self.ae, self.codec, _, self.source = load_joint_run(
                self.run_dir, repository_root=self.root, device=self.device,
                checkpoint_dir=self.checkpoint_dir, load_foreground=False)
        else:
            config = load_config(self.bundle / 'configs/p2sd.yaml')
            ae_path, model_path = self.bundle / 'weights/ae.pt', self.bundle / 'weights/p2sd.pt'
            if not ae_path.is_file() or not model_path.is_file():
                raise ValueError('Missing released weights. Extract the reproduction archive or set --bundle-root to it.')
            self.ae = build_sheet_ae(load_config(self.bundle / 'configs/ae.yaml')).to(self.device).eval()
            load_model_state(self.ae, ae_path)
            self.model = build_p2sd(config, latent_channels=self.ae.latent_channels).to(self.device).eval()
            load_model_state(self.model, model_path)
            normalization = dict(config['p2sd']['loss']['latent_normalization'])
            normalization['stats_path'] = str(self.bundle / 'configs/latent_stats.json')
            self.codec = build_static_latent_codec(normalization,
                latent_channels=self.ae.latent_channels, device=self.device)
            self.source = dict(mode='released_0058', files=dict(ae=str(ae_path), p2sd=str(model_path),
                ae_config=str(self.bundle / 'configs/ae.yaml'),
                p2sd_config=str(self.bundle / 'configs/p2sd.yaml'),
                latent_statistics=normalization['stats_path']))
        self.model.requires_grad_(False)
        self.ae.requires_grad_(False)
        self.source['sha256'] = {
            name: hashlib.sha256(Path(path).read_bytes()).hexdigest()
            for name, path in self.source['files'].items()}

    def prepare(self, image, image_key):
        import torch
        started = time.perf_counter()
        if self.model is None:
            self._load()
        if self.image_key != image_key:
            self.context = None  # release the old crop before encoding the new one
            canvas, self.bounds, self.offset = pad_crop(image)
            tensor = torch.from_numpy(canvas[None, None].astype(np.float32)).to(self.device)
            with torch.inference_mode(), torch.autocast(self.device.type, dtype=self.dtype,
                                                        enabled=self.device.type == 'cuda'):
                _, tokens, coords, context = self.model.encode_image_context_from_image(tensor)
            self.context = (tokens, coords, context)
            self.image_key = image_key
            self.encode_count += 1
        return time.perf_counter() - started

    def predict(self, image, image_key, points):
        import torch
        from vesuvius_p2sd.research.auto_instance_seg import _decode_prompt_sets
        points = validate_points(points, np.array(image.shape))
        preparation_seconds = self.prepare(image, image_key)
        started = time.perf_counter()
        with torch.inference_mode(), torch.autocast(self.device.type, dtype=self.dtype,
                                                    enabled=self.device.type == 'cuda'):
            probability = _decode_prompt_sets(self.model, self.ae, self.codec,
                *self.context, [points + self.offset], (320, 320, 320), self.device, self.dtype)[0]
        probability = np.ascontiguousarray(probability[self.bounds])
        return probability, dict(prepare_seconds=preparation_seconds,
            decode_seconds=time.perf_counter() - started, image_encodes=self.encode_count)


    def automatic(self, image, image_key, progress=lambda message: None):
        """The same automatic recipe as plats.py, reusing the CT context."""
        import torch
        from vesuvius_p2sd.research import auto_instance_seg as auto
        from vesuvius_p2sd.utils.config import load_config
        from vesuvius_p2sd.train.common import load_model_state
        started = time.perf_counter()
        self.prepare(image, image_key)
        settings_path = self.bundle / 'configs/automatic.json'
        if not settings_path.is_file():
            settings_path = self.root / 'configs/automatic.json'
        settings = json.loads(settings_path.read_text())
        progress('Automatic: loading foreground decoder')
        if self.foreground is None:
            if self.run_dir:
                from vesuvius_p2sd.joint_inference import load_joint_run
                trunk, ae, codec, head, provenance = load_joint_run(
                    self.run_dir, repository_root=self.root, device=self.device,
                    checkpoint_dir=self.checkpoint_dir, load_foreground=True)
                current_hash = hashlib.sha256(Path(provenance['files']['p2sd']).read_bytes()).hexdigest()
                if current_hash != self.source['sha256']['p2sd']:
                    raise ValueError('The checkpoint changed after model loading. Restart with a fixed snapshot.')
                head.trunk = self.model
                self.foreground = head
                self.source['files']['binary_head'] = provenance['files']['binary_head']
                del trunk, ae, codec
            else:
                from vesuvius_p2sd.models.binary_seg import build_binary_seg_model
                head = build_binary_seg_model(load_config(self.bundle / 'configs/foreground.yaml')).to(self.device).eval()
                path = self.bundle / 'weights/foreground.pt'
                load_model_state(head, path)
                head.requires_grad_(False)
                self.foreground = head
                self.source['files']['foreground'] = str(path)
            self.source['sha256'].update({name: hashlib.sha256(Path(path).read_bytes()).hexdigest()
                for name, path in self.source['files'].items() if name not in self.source['sha256']})
        canvas, bounds, offset = pad_crop(image)
        valid = np.zeros(canvas.shape, bool)
        valid[bounds] = True
        tensor = torch.from_numpy(canvas[None, None].astype(np.float32)).to(self.device)
        progress('Automatic: predicting foreground')
        with torch.inference_mode(), torch.autocast(self.device.type, dtype=self.dtype,
                                                   enabled=self.device.type == 'cuda'):
            if self.run_dir:
                refined = self.foreground.refine_context(self.context[2])
                logits = self.foreground.decode_context(refined)
                del refined
            else:
                logits = self.foreground(tensor)
            foreground = (logits.float().sigmoid()[0, 0] >= settings['binary_threshold']).cpu().numpy()
            del logits, tensor
            progress('Automatic: clustering 512 seeds and decoding sheets')
            # Same batch size as the released CLI; avoid accumulating full 320³
            # decoder activations for a whole cluster's prompt candidates.
            previous_batch = auto._DECODE_BATCH
            auto._DECODE_BATCH = 1
            try:
                result = auto.run_case_cluster(binseg_model=None, p2sd_model=self.model,
                    target_ae=self.ae, latent_codec=self.codec, image=canvas, valid=valid,
                    device=self.device, dtype=self.dtype, external_mask=foreground,
                    image_context_cache=(None, *self.context), **settings)
            finally:
                auto._DECODE_BATCH = previous_batch
        points = result['points']
        points['clicks'] = (np.asarray(points['clicks']).reshape(-1, 3) - offset).tolist()
        points['sheets'] = {str(k): (np.asarray(v).reshape(-1, 3) - offset).tolist()
                            for k, v in points['sheets'].items()}
        return dict(labels=result['instance_ids'][bounds].astype(np.uint16),
            foreground=foreground[bounds].copy(), points=points, clustering=result['seed_log'],
            settings=settings, timings=dict(total_seconds=time.perf_counter()-started,
                                             image_encodes=self.encode_count))
