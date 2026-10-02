"""Inference and native-coordinate data handling, independent of the web UI."""
from __future__ import annotations

import hashlib
import json
import struct
import time
from pathlib import Path

import numpy as np


def load_array(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Preserve file array axes; never silently reorient CT or normalize it."""
    path = Path(path).expanduser().resolve(strict=True)
    affine = np.eye(4)
    if path.suffix == '.npy':
        array = np.load(path, allow_pickle=False)
    elif path.name.endswith(('.nii', '.nii.gz')):
        import nibabel as nib
        image = nib.load(path)
        array, affine = np.asarray(image.dataobj), image.affine
    elif path.suffix.lower() in ('.tif', '.tiff'):
        import tifffile
        array = tifffile.imread(path)
    else:
        raise ValueError('Use a .npy, .nii, .nii.gz, .tif or .tiff volume.')
    if array.ndim != 3 or any(n < 32 or n > 320 for n in array.shape):
        raise ValueError('Select a 3D crop with each side between 32 and 320 voxels.')
    return np.ascontiguousarray(array), affine


def load_crop(path: str, gt_path: str = '') -> dict:
    image, affine = load_array(path)
    if image.dtype != np.uint8:
        raise ValueError('CT must be preprocessed uint8, as used during training. No automatic rescaling is applied.')
    gt = None
    if gt_path:
        reference, gt_affine = load_array(gt_path)
        if reference.shape != image.shape or not np.allclose(affine, gt_affine):
            raise ValueError('Reference shape and voxel affine must match the CT.')
        gt = np.asarray(reference > 0, dtype=np.uint8)
    return dict(image=image, affine=affine, gt=gt,
                path=str(Path(path).expanduser().resolve()),
                gt_path=str(Path(gt_path).expanduser().resolve()) if gt_path else '')


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


def export_session(folder: Path, crop: dict, sheets: dict, source: dict):
    """Save each sheet separately; an overlap count avoids silently merging IDs."""
    import nibabel as nib
    folder.mkdir(parents=True, exist_ok=False)
    def nifti(path, values):
        nib.save(nib.Nifti1Image(values, crop['affine']), str(path))
    nifti(folder / 'image.nii.gz', crop['image'])
    if crop['gt'] is not None:
        nifti(folder / 'gt_union.nii.gz', crop['gt'])
    overlaps = np.zeros(crop['image'].shape, np.uint16)
    records = []
    for sheet_id, sheet in sorted(sheets.items()):
        target = folder / f'sheet_{sheet_id:02d}'
        target.mkdir()
        mask = sheet['probability'] >= sheet['threshold']
        overlaps += mask
        nifti(target / 'prediction.nii.gz', mask.astype(np.uint8))
        nifti(target / 'probability.nii.gz', sheet['probability'].astype(np.float32))
        points = np.zeros(mask.shape, np.uint8)
        indices = np.minimum(np.rint(sheet['points']).astype(int), np.array(mask.shape) - 1)
        for point in indices:
            points[tuple(point)] = 1
        nifti(target / 'prompt_points.nii.gz', points)
        # Each folder is independently convenient to open in a NIFTI viewer.
        (target / 'image.nii.gz').symlink_to('../image.nii.gz')
        if crop['gt'] is not None:
            (target / 'gt_union.nii.gz').symlink_to('../gt_union.nii.gz')
        record = dict(id=sheet_id, points=sheet['points'], threshold=sheet['threshold'],
                      timings=sheet['timings'], coordinate_order='native input array axes [0,1,2]')
        (target / 'points.json').write_text(json.dumps(record, indent=2) + '\n')
        records.append(record)
    nifti(folder / 'union.nii.gz', (overlaps > 0).astype(np.uint8))
    nifti(folder / 'overlap_count.nii.gz', overlaps)
    record = dict(image=crop['path'], gt=crop['gt_path'], shape=list(crop['image'].shape),
        affine=crop['affine'].tolist(), image_array_sha256=hashlib.sha256(crop['image'].tobytes()).hexdigest(),
        sheets=records, model=source, postprocessing='none; raw probability threshold',
        geometry='3D display is in native voxel coordinates; NIFTI export preserves the input affine.')
    (folder / 'session.json').write_text(json.dumps(record, indent=2) + '\n')
    return str(folder.resolve())
