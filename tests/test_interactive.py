"""Native coordinates, NIFTI provenance, and stale asynchronous result safety."""
import json
import struct
import threading
import time
from pathlib import Path
import sys

import nibabel as nib
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from vesuvius_p2sd.interactive.core import (
    export_session, load_crop, packed_mask, pad_crop, surface_bytes, validate_points,
)
from vesuvius_p2sd.interactive.server import App


def test_native_padding_and_input_contract(tmp_path):
    image = np.arange(32*35*38, dtype=np.uint8).reshape(32,35,38)
    np.save(tmp_path/'image.npy', image)
    crop = load_crop(str(tmp_path/'image.npy'))
    canvas, bounds, offset = pad_crop(crop['image'])
    assert np.array_equal(canvas[bounds], image)
    assert list(offset) == [144,142,141]
    for point in ([0,0,0], [31,34,37], [12,17,25]):
        assert canvas[tuple(np.array(point)+offset)] == image[tuple(point)]
    with pytest.raises(ValueError):
        validate_points([[32,0,0]], image.shape)
    with pytest.raises(ValueError):
        validate_points([[np.nan,0,0]], image.shape)
    with pytest.raises(ValueError):
        validate_points([[1,1,1]]*9, image.shape)
    np.save(tmp_path/'float.npy', image.astype(np.float32))
    with pytest.raises(ValueError, match='uint8'):
        load_crop(str(tmp_path/'float.npy'))


def test_surface_and_mask_native_coordinates():
    mask = np.zeros((32,35,38), bool)
    mask[7:11,13:17,3:9] = True
    raw = surface_bytes(mask)
    nv, nf = struct.unpack('<II', raw[:8])
    vertices = np.frombuffer(raw, '<f4', offset=8, count=nv*3).reshape(-1,3)
    assert nf > 0
    np.testing.assert_array_equal(vertices.min(0), [6.5,12.5,2.5])
    np.testing.assert_array_equal(vertices.max(0), [10.5,16.5,8.5])
    restored = np.unpackbits(np.frombuffer(packed_mask(mask), np.uint8),bitorder='little')[:mask.size]
    np.testing.assert_array_equal(restored.reshape(mask.shape),mask)
    assert surface_bytes(np.zeros_like(mask)) == struct.pack('<II',0,0)


def test_nifti_export_preserves_affine_points_and_sheet_overlaps(tmp_path):
    image = np.zeros((32,35,38), np.uint8)
    affine = np.diag([.5,.7,1.2,1.]);affine[:3,3]=[13,-4,8]
    nib.save(nib.Nifti1Image(image,affine),tmp_path/'image.nii.gz')
    crop=load_crop(str(tmp_path/'image.nii.gz'))
    probability=np.zeros(image.shape,np.float32);probability[8:12,13:17,20:24]=.7
    sheet=dict(probability=probability,threshold=.5,points=[[9,14,21]],timings={})
    folder=tmp_path/'export'
    export_session(folder,crop,{1:sheet,2:sheet},{'mode':'test'})
    saved=nib.load(folder/'sheet_01/prediction.nii.gz')
    np.testing.assert_allclose(saved.affine,affine)
    np.testing.assert_array_equal(np.asarray(saved.dataobj),probability>=.5)
    points=np.asarray(nib.load(folder/'sheet_01/prompt_points.nii.gz').dataobj)
    assert points.sum()==1 and points[9,14,21]==1
    overlaps=np.asarray(nib.load(folder/'overlap_count.nii.gz').dataobj)
    assert overlaps.max()==2 and overlaps.sum()==128
    assert (folder/'sheet_01/image.nii.gz').resolve()==folder/'image.nii.gz'
    assert json.loads((folder/'session.json').read_text())['shape']==[32,35,38]


class FakeEngine:
    device_name='test'
    source={'mode':'test'}
    encode_count=0
    def __init__(self):
        self.started=threading.Event();self.release=threading.Event();self.calls=0
    def prepare(self,image,key):
        self.encode_count+=1
        return 0.
    def predict(self,image,key,points):
        self.calls+=1;self.started.set()
        assert self.release.wait(5)
        probability=np.zeros(image.shape,np.float32);probability[4:8,5:9,6:10]=.8
        return probability,dict(prepare_seconds=0,decode_seconds=0,image_encodes=1)


def wait_job(app,job):
    deadline=time.monotonic()+10
    while time.monotonic()<deadline:
        with app.lock:
            result=dict(app.jobs[job['job']])
        if result['status'] in ('done','error'):
            assert result['status']=='done',result
            return result['result']
        time.sleep(.01)
    raise AssertionError('Job timed out')


def make_app(tmp_path):
    engine=FakeEngine()
    crop=dict(image=np.zeros((32,32,32),np.uint8),gt=None,affine=np.eye(4),path='test',gt_path='')
    return App(engine,crop,tmp_path),engine


def test_clear_during_decode_never_resurrects_sheet(tmp_path):
    app,engine=make_app(tmp_path)
    try:
        job=app.predict(dict(generation=1,sheet=1,points=[[5,6,7]]))
        assert engine.started.wait(5)
        app.mutate('/api/delete',dict(generation=1,sheet=1))
        engine.release.set()
        assert wait_job(app,job)['discarded']
        assert not app.sheets
    finally:
        engine.release.set();app.worker.shutdown()


def test_threshold_reuses_probability_and_export_rejects_stale_revision(tmp_path):
    app,engine=make_app(tmp_path);engine.release.set()
    try:
        first=wait_job(app,app.predict(dict(generation=1,sheet=1,points=[[5,6,7]])))
        second=wait_job(app,app.predict(dict(generation=1,sheet=1,points=[[5,6,7]],threshold=.7)))
        assert engine.calls==1 and second['timings']['reused_probability']
        with pytest.raises(ValueError,match='changed'):
            app.mutate('/api/export',dict(generation=1,revisions={'1':first['revision']}))
        with pytest.raises(ValueError,match='crop changed'):
            app.predict(dict(generation=0,sheet=1,points=[[5,6,7]]))
    finally:
        app.worker.shutdown()
