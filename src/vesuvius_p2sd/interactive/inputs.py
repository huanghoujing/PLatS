"""Explicit challenge TIFF / OME-Zarr conversions to model XYZ coordinates."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from urllib.parse import urlsplit

import numpy as np


def is_tiff(path):
    return Path(path).suffix.lower() in ('.tif', '.tiff')


def load_array(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Read file array axes verbatim; load_crop applies TIFF ZYX -> XYZ."""
    path = Path(path).expanduser().resolve(strict=True)
    affine = np.eye(4)
    if path.suffix == '.npy':
        array = np.load(path, allow_pickle=False)
    elif path.name.endswith(('.nii', '.nii.gz')):
        import nibabel as nib
        image = nib.load(path)
        array, affine = np.asarray(image.dataobj), image.affine
    elif is_tiff(path):
        import tifffile
        array = tifffile.imread(path)
    else:
        raise ValueError('Use a .npy, .nii, .nii.gz, .tif or .tiff volume.')
    if array.ndim != 3 or any(n < 32 or n > 320 for n in array.shape):
        raise ValueError('Select a 3D crop with each side between 32 and 320 voxels.')
    return np.ascontiguousarray(array), affine


def reference_labels(values, kind='instances'):
    """Keep sheet IDs, or identify six-connected components of a binary GT."""
    if kind == 'instances':
        if not np.issubdtype(values.dtype, np.integer) or values.min() < 0:
            raise ValueError('Reference instance labels must be nonnegative integers.')
        if values.max() > np.iinfo(np.uint32).max:
            raise ValueError('Reference IDs exceed uint32.')
        return np.asarray(values, dtype=np.uint32)
    if kind not in ('binary', 'challenge'):
        raise ValueError('Reference type must be instances, binary or challenge.')
    import cc3d
    if kind == 'challenge':
        if not np.isin(values, [0, 1, 2]).all():
            raise ValueError('Challenge labels must use 0=background, 1=sheet, 2=ignore.')
        foreground = values == 1
    else:
        foreground = values > 0
    return cc3d.connected_components(np.ascontiguousarray(foreground), connectivity=6).astype(np.uint32)


def load_crop(path: str, gt_path: str = '', reference_kind='instances') -> dict:
    image, affine = load_array(path)
    if is_tiff(path):
        image = np.ascontiguousarray(image.transpose(2, 1, 0))
    if image.dtype != np.uint8:
        raise ValueError('CT must be preprocessed uint8, as used during training. No automatic rescaling is applied.')
    gt, labels = None, None
    if gt_path:
        reference, gt_affine = load_array(gt_path)
        if is_tiff(gt_path):
            reference = reference.transpose(2, 1, 0)
        both_nifti = all(str(p).endswith(('.nii', '.nii.gz')) for p in (path, gt_path))
        if reference.shape != image.shape or (both_nifti and not np.allclose(affine, gt_affine)):
            raise ValueError('Reference shape and voxel affine must match the CT.')
        labels = reference_labels(reference, reference_kind)
        gt = np.asarray(labels > 0, dtype=np.uint8)
    return dict(image=image, affine=affine, gt=gt, reference=labels,
                reference_kind=reference_kind, path=str(Path(path).expanduser().resolve()),
                gt_path=str(Path(gt_path).expanduser().resolve()) if gt_path else '',
                source=dict(kind='tiff' if is_tiff(path) else 'local',
                    file_axes='ZYX' if is_tiff(path) else 'XYZ', model_axes='XYZ',
                    origin_xyz=[0, 0, 0]))


def _http_url(url):
    url = str(url).strip().rstrip('/')
    parts = urlsplit(url)
    if parts.scheme == 's3':
        url = f'https://{parts.netloc}.s3.amazonaws.com{parts.path}'
        parts = urlsplit(url)
    if parts.scheme not in ('http', 'https') or not parts.netloc:
        raise ValueError('Provide an HTTP(S) OME-Zarr URL, or an s3:// bucket URL.')
    if parts.query or parts.fragment:
        raise ValueError('Use the public Zarr root or array URL without query parameters.')
    return url


def load_zarr_crop(url, start_xyz, *, level=0, cache_dir=None, patch_size=320):
    """Read only intersecting chunks; starts are XYZ indices in the chosen level.

    OME axes, scale, and translation are preserved in provenance. If a plain
    3D array has no axes metadata, the explicit official-CT convention is ZYX.
    The viewer uses XYZ voxels; TIFF exports reverse that permutation exactly.
    """
    import aiohttp
    import zarr
    from zarr.storage import FsspecStore
    url = _http_url(url)
    origin = np.asarray(start_xyz, dtype=np.float64)
    if origin.shape != (3,) or not np.isfinite(origin).all() or (origin < 0).any() or not (origin == np.floor(origin)).all():
        raise ValueError('Starting X, Y and Z must be nonnegative integer voxel coordinates.')
    origin = origin.astype(np.int64)
    level = int(level)
    if level < 0:
        raise ValueError('Resolution level must be nonnegative.')
    options = {'client_kwargs': {'timeout': aiohttp.ClientTimeout(total=60)}}
    def open_url(address):
        return zarr.open(FsspecStore.from_url(address, storage_options=options, read_only=True), mode='r')
    node = open_url(url)
    attrs, dataset_path = dict(node.attrs), str(level)
    if isinstance(node, zarr.Group):
        array = node[dataset_path]
        if not isinstance(array, zarr.Array):
            raise ValueError('The selected resolution is not an array.')
        array_url = f'{url}/{dataset_path}'
    else:
        array, array_url = node, url
        # Direct /0 URLs should still use the parent OME axes/transforms.
        parent = url.rsplit('/', 1)[0]
        if url.rsplit('/', 1)[-1].isdigit():
            dataset_path = url.rsplit('/', 1)[-1]
            attrs = dict(open_url(parent).attrs)
    if array.ndim != 3 or array.dtype != np.uint8:
        raise ValueError('Select an official 3D uint8 CT array, not a surface-volume/channel array.')
    multiscales = attrs.get('multiscales') or attrs.get('ome', {}).get('multiscales') or []
    metadata = multiscales[0] if multiscales else {}
    axes = [a['name'] if isinstance(a, dict) else a for a in metadata.get('axes', [])]
    if not axes:
        axes = list(getattr(array.metadata, 'dimension_names', None) or ['z', 'y', 'x'])
    axes = [a.lower() for a in axes]
    if sorted(axes) != ['x', 'y', 'z']:
        raise ValueError(f'Expected spatial X/Y/Z axes; got {axes}.')
    source_origin = np.array([origin['xyz'.index(a)] for a in axes])
    if (source_origin + patch_size > array.shape).any():
        raise ValueError(f'The {patch_size}³ patch exceeds array shape {array.shape} in {axes} order.')
    selected = next((d for d in metadata.get('datasets', []) if d['path'] == dataset_path), {})
    transforms = selected.get('coordinateTransformations', []) + metadata.get('coordinateTransformations', [])
    scale, translation = np.ones(3), np.zeros(3)
    for transform in transforms:
        if transform['type'] == 'scale':
            factor = np.asarray(transform['scale'])
            scale *= factor
            translation *= factor
        elif transform['type'] == 'translation':
            translation += transform['translation']
        else:
            raise ValueError('Only OME scale and translation transforms are supported.')
    perm = [axes.index(a) for a in 'xyz']
    affine = np.eye(4)
    affine[:3, :3] = np.diag(scale[perm])
    affine[:3, 3] = origin * scale[perm] + translation[perm]
    source = dict(kind='zarr', url=url, array_url=array_url, level=dataset_path,
        origin_xyz=origin.tolist(), source_axes=axes, model_axes='XYZ',
        source_shape=list(array.shape), patch_shape_xyz=[patch_size]*3,
        coordinate_transformations=transforms, ome_axes=metadata.get('axes', []),
        axes_fallback=not bool(metadata.get('axes') or getattr(array.metadata, 'dimension_names', None)))
    cache = None
    if cache_dir:
        digest = hashlib.sha256(json.dumps(source, sort_keys=True).encode()).hexdigest()
        cache = Path(cache_dir) / f'{digest}.npz'
    if cache and cache.is_file():
        with np.load(cache) as saved:
            image = saved['image']
    else:
        slices = tuple(slice(int(o), int(o + patch_size)) for o in source_origin)
        with zarr.config.set({'async.concurrency': 8}):
            raw = np.asarray(array[slices])
        image = np.ascontiguousarray(raw.transpose(perm))
        if cache:
            cache.parent.mkdir(parents=True, exist_ok=True)
            temporary = cache.with_suffix('.tmp')
            with temporary.open('wb') as stream:
                np.savez_compressed(stream, image=image)
            temporary.replace(cache)
    source['image_sha256'] = hashlib.sha256(image.tobytes()).hexdigest()
    return dict(image=image, affine=affine, gt=None, reference=None,
                reference_kind='instances', path=array_url, gt_path='', source=source)
