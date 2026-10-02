"""Challenge axes, reference semantics, lossless label export and remote ROI reads."""
import functools
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import threading
from pathlib import Path
import sys

import numpy as np
import pytest
import tifffile
import zarr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from vesuvius_p2sd.interactive.inputs import load_crop, load_zarr_crop, reference_labels
from vesuvius_p2sd.interactive.exports import export_session


def test_asymmetric_challenge_tiff_roundtrip_and_reference_ids(tmp_path):
    image = np.arange(32*35*38,dtype=np.uint8).reshape(32,35,38)
    labels = np.zeros(image.shape,np.uint16)
    labels[3:8,9:14,20:25] = 513
    tifffile.imwrite(tmp_path/'image.tif',image,photometric='minisblack')
    tifffile.imwrite(tmp_path/'labels.tif',labels,photometric='minisblack')
    crop=load_crop(str(tmp_path/'image.tif'),str(tmp_path/'labels.tif'))
    np.testing.assert_array_equal(crop['image'],image.transpose(2,1,0))
    np.testing.assert_array_equal(crop['reference'],labels.transpose(2,1,0))
    automatic=dict(labels=crop['reference'].astype(np.uint16),foreground=crop['gt'],
                   revision=1,clustering=[],points={},settings={},timings={})
    folder=tmp_path/'export'
    export_session(folder,crop,{}, {},formats=['tiff'],automatic=automatic)
    np.testing.assert_array_equal(tifffile.imread(folder/'image.tif'),image)
    np.testing.assert_array_equal(tifffile.imread(folder/'automatic/instances.tif'),labels)
    with tifffile.TiffFile(folder/'automatic/instances.tif') as handle:
        assert handle.series[0].axes=='ZYX'
        assert all(page.compression.name in ('DEFLATE','ADOBE_DEFLATE') for page in handle.pages)
        assert handle.series[0].dtype==np.uint16


def test_challenge_ignore_is_not_a_reference_sheet():
    labels=np.zeros((32,32,32),np.uint8)
    labels[5:8,5:8,5:8]=1
    labels[20:23,20:23,20:23]=1
    labels[:3]=2
    result=reference_labels(labels,'challenge')
    assert result.max()==2 and np.count_nonzero(result)==54
    assert not result[:3].any()
    assert reference_labels(labels,'instances')[:3].max()==2


def test_remote_nonzero_xyz_reads_exact_roi_and_rejects_bounds(tmp_path):
    root=zarr.open_group(str(tmp_path/'ct.zarr'),mode='w',zarr_format=2)
    raw=np.arange(75*80*90,dtype=np.uint8).reshape(75,80,90)
    root.create_array('0',data=raw,chunks=(16,16,16))
    root.attrs['multiscales']=[{'axes':[{'name':a} for a in 'zyx'],
        'datasets':[{'path':'0','coordinateTransformations':[{'type':'scale','scale':[2,3,4]}]}]}]
    class Quiet(SimpleHTTPRequestHandler):
        requests=[]
        def log_message(self,*args):pass
        def do_GET(self):
            self.requests.append(self.path)
            super().do_GET()
    server=ThreadingHTTPServer(('127.0.0.1',0),functools.partial(Quiet,directory=str(tmp_path)))
    thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    url=f'http://127.0.0.1:{server.server_port}/ct.zarr'
    try:
        crop=load_zarr_crop(url,[21,17,9],patch_size=32,cache_dir=tmp_path/'cache')
        np.testing.assert_array_equal(crop['image'],raw[9:41,17:49,21:53].transpose(2,1,0))
        np.testing.assert_array_equal(np.diag(crop['affine'])[:3],[4,3,2])
        np.testing.assert_array_equal(crop['affine'][:3,3],[84,51,18])
        chunk_requests=[p for p in Quiet.requests if p.rsplit('/',1)[-1].replace('.','').isdigit()]
        assert 0<len(chunk_requests)<=27,chunk_requests
        Quiet.requests.clear()
        cached=load_zarr_crop(url+'/0',[21,17,9],patch_size=32,cache_dir=tmp_path/'cache')
        np.testing.assert_array_equal(cached['image'],crop['image'])
        with pytest.raises(ValueError,match='exceeds'):
            load_zarr_crop(url,[70,17,9],patch_size=32)
        with pytest.raises(ValueError,match='integer'):
            load_zarr_crop(url,[.5,0,0],patch_size=32)
    finally:
        server.shutdown();server.server_close();thread.join()


def test_automatic_keeps_prompted_sheets_and_rejects_stale_export(tmp_path):
    from test_interactive import make_app, wait_job
    app, engine=make_app(tmp_path)
    engine.release.set()
    labels=np.zeros((32,32,32),np.uint16);labels[3:9,5:8,7:10]=1
    engine.automatic=lambda image,key,progress: dict(labels=labels,foreground=labels>0,
        settings={},timings={},points={},clustering=[])
    try:
        wait_job(app,app.predict(dict(generation=1,sheet=1,points=[[5,6,7]])))
        before=app.sheets[1]
        result=wait_job(app,app.automatic(dict(generation=1)))
        assert len(result['automatic']['instances'])==1
        assert app.sheets[1] is before
        with pytest.raises(ValueError,match='Automatic predictions changed'):
            app.mutate('/api/export',dict(generation=1,revisions={},automatic_revision=0,formats=['tiff']))
        exported=wait_job(app,app.mutate('/api/export',dict(generation=1,revisions={},automatic_revision=1,formats=['tiff'])))
        np.testing.assert_array_equal(tifffile.imread(Path(exported['path'])/'automatic/instances.tif'),labels.transpose(2,1,0))
    finally:
        app.worker.shutdown()
