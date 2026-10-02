"""Lossless TIFF and NIFTI export of the exact displayed inference results."""
import hashlib
import json
from pathlib import Path

import numpy as np


def write_tiff(path, values_xyz):
    import tifffile
    # TIFF pages follow the challenge's [Z,Y,X] order. Deflate is lossless for
    # integer IDs as well as probabilities; never convert IDs to display RGB.
    tifffile.imwrite(path, np.ascontiguousarray(values_xyz.transpose(2, 1, 0)),
                     photometric='minisblack', compression='deflate',
                     compressionargs={'level': 6}, metadata={'axes': 'ZYX'})


def export_session(folder: Path, crop: dict, sheets: dict, source: dict,
                   *, formats=('nifti',), automatic=None):
    import nibabel as nib
    if not formats or any(f not in ('nifti', 'tiff') for f in formats):
        raise ValueError('Choose NIFTI, TIFF, or both export formats.')
    folder.mkdir(parents=True, exist_ok=False)

    def save(base, values):
        if 'nifti' in formats:
            nib.save(nib.Nifti1Image(values, crop['affine']), str(base) + '.nii.gz')
        if 'tiff' in formats:
            write_tiff(str(base) + '.tif', values)

    save(folder / 'image', crop['image'])
    if crop['gt'] is not None:
        save(folder / 'gt_union', crop['gt'])
        if crop.get('reference') is not None:
            save(folder / 'gt_instances', crop['reference'])
    overlaps = np.zeros(crop['image'].shape, np.uint16)
    records = []
    for sheet_id, sheet in sorted(sheets.items()):
        target = folder / f'sheet_{sheet_id:02d}'
        target.mkdir()
        mask = sheet['probability'] >= sheet['threshold']
        overlaps += mask
        save(target / 'prediction', mask.astype(np.uint8))
        save(target / 'probability', sheet['probability'].astype(np.float32))
        points = np.zeros(mask.shape, np.uint8)
        indices = np.minimum(np.rint(sheet['points']).astype(int), np.array(mask.shape) - 1)
        for point in indices:
            points[tuple(point)] = 1
        save(target / 'prompt_points', points)
        # Each sheet folder remains independently convenient to inspect.
        for extension in (['nii.gz'] if 'nifti' in formats else []) + (['tif'] if 'tiff' in formats else []):
            for name in ['image'] + (['gt_union'] if crop['gt'] is not None else []):
                (target / f'{name}.{extension}').symlink_to(f'../{name}.{extension}')
        record = dict(id=sheet_id, points=sheet['points'], threshold=sheet['threshold'],
                      timings=sheet['timings'], coordinate_order='model XYZ; TIFF indices are ZYX')
        (target / 'points.json').write_text(json.dumps(record, indent=2) + '\n')
        records.append(record)
    if sheets:
        save(folder / 'union', (overlaps > 0).astype(np.uint8))
        save(folder / 'overlap_count', overlaps)
    if automatic is not None:
        target = folder / 'automatic'
        target.mkdir()
        save(target / 'instances', automatic['labels'])
        save(target / 'prediction', (automatic['labels'] > 0).astype(np.uint8))
        save(target / 'foreground', automatic['foreground'].astype(np.uint8))
        (target / 'clustering.json').write_text(json.dumps(automatic['clustering'], indent=2) + '\n')
        (target / 'points.json').write_text(json.dumps(automatic['points'], indent=2) + '\n')
    record = dict(image=crop['path'], gt=crop['gt_path'], shape=list(crop['image'].shape),
        affine=crop['affine'].tolist(), image_array_sha256=hashlib.sha256(crop['image'].tobytes()).hexdigest(),
        input_source=crop.get('source'), sheets=records, model=source, formats=list(formats),
        tiff=dict(axes='ZYX', compression='lossless Deflate', instance_dtype='uint16'),
        automatic=None if automatic is None else {k: automatic[k] for k in ('revision', 'settings', 'timings')},
        postprocessing='Prompted: raw threshold. Automatic: released clustering/cleanup recipe.',
        geometry='Viewer/model XYZ; TIFF ZYX. Remote origin is in input_source.origin_xyz at the selected resolution.')
    (folder / 'session.json').write_text(json.dumps(record, indent=2) + '\n')
    return str(folder.resolve())
