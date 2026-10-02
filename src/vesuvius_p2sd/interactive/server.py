"""Loopback-only HTTP app; one background worker owns the GPU model/cache."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import gzip
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import mimetypes
from pathlib import Path
import secrets
import threading
import time
import traceback
from urllib.parse import urlsplit

import numpy as np

from .core import PromptEngine, export_session, load_crop, packed_mask, surface_bytes, validate_points
from .inputs import load_zarr_crop

STATIC = Path(__file__).with_name('static')


class App:
    def __init__(self, engine, crop, output):
        self.engine, self.crop, self.output = engine, crop, Path(output)
        self.token = secrets.token_urlsafe(24)
        self.generation = 1
        self.sheets, self.revisions, self.jobs = {}, {}, {}
        self.automatic_result = None
        self.automatic_revision = 0
        self.surface_cache = {}
        self.lock = threading.RLock()
        self.worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix='plats-inference')
        self.ready = 'Preparing model and CT features'
        self.submit('warmup', self.warmup)

    def warmup(self):
        try:
            seconds = self.engine.prepare(self.crop['image'], self.generation)
            self.ready = f'CT features ready ({seconds:.1f}s); click a sheet'
        except Exception as error:
            self.ready = f'Model preparation failed: {error}'
            raise
        return dict(seconds=seconds)

    def submit(self, kind, function):
        with self.lock:
            # Keep a bounded history; results contain metadata, never volumes.
            for key in list(self.jobs):
                if len(self.jobs) < 100:
                    break
                if self.jobs[key]['status'] in ('done', 'error'):
                    del self.jobs[key]
            if sum(j['status'] in ('queued', 'running') for j in self.jobs.values()) >= 16:
                raise ValueError('Too many queued jobs; wait for inference to finish.')
            job_id = secrets.token_hex(8)
            self.jobs[job_id] = dict(kind=kind, status='queued')
        def run():
            with self.lock:
                self.jobs[job_id]['status'] = 'running'
            try:
                result = function()
                with self.lock:
                    self.jobs[job_id].update(status='done', result=result)
            except Exception as error:
                traceback.print_exc()
                with self.lock:
                    self.jobs[job_id].update(status='error', error=str(error))
        self.worker.submit(run)
        return dict(job=job_id)

    def metadata(self):
        with self.lock:
            reference = self.crop.get('reference')
            ids, counts = (np.unique(reference, return_counts=True) if reference is not None else ([], []))
            reference_ids = [dict(id=int(i), voxels=int(c)) for i, c in zip(ids, counts) if i > 0]
            automatic = self.automatic_result
            automatic_info = None
            if automatic is not None:
                auto_ids, auto_counts = np.unique(automatic['labels'], return_counts=True)
                automatic_info = dict(revision=automatic['revision'], timings=automatic['timings'],
                    instances=[dict(id=int(i), voxels=int(c)) for i, c in zip(auto_ids, auto_counts) if i > 0])
            return dict(shape=list(self.crop['image'].shape), path=self.crop['path'],
                gt_path=self.crop['gt_path'], has_gt=self.crop['gt'] is not None,
                generation=self.generation, ready=self.ready, device=self.engine.device_name,
                coordinate_order='XYZ model/viewer; TIFF files use ZYX',
                source=self.crop.get('source', {}), reference_kind=self.crop.get('reference_kind', 'instances'),
                reference_ids=reference_ids, automatic=automatic_info,
                model=self.engine.source.get('mode', 'loading'),
                image_encodes=self.engine.encode_count,
                sheets=[dict(id=k, points=s['points'], threshold=s['threshold'],
                             revision=s['revision'], timings=s['timings'])
                        for k, s in sorted(self.sheets.items())])

    def predict(self, payload):
        with self.lock:
            if payload['generation'] != self.generation:
                raise ValueError('The crop changed; reload the viewer.')
            sheet_id = int(payload['sheet'])
            if not 1 <= sheet_id <= 16:
                raise ValueError('Use sheet IDs 1–16.')
            crop, generation = self.crop, self.generation
            points = validate_points(payload['points'], np.array(crop['image'].shape)).tolist()
            threshold = float(payload.get('threshold', 0.5))
            if not 0.05 <= threshold <= 0.95:
                raise ValueError('Threshold must be between 0.05 and 0.95.')
            revision = self.revisions.get(sheet_id, 0) + 1
            self.revisions[sheet_id] = revision
        def run():
            with self.lock:
                if generation != self.generation or self.revisions.get(sheet_id) != revision:
                    return dict(discarded=True)
                previous = self.sheets.get(sheet_id)
            # Threshold edits reuse full-precision probability, without a GPU pass.
            started = time.perf_counter()
            if previous is not None and previous['points'] == points:
                probability = previous['probability']
                timings = dict(prepare_seconds=0, decode_seconds=0, image_encodes=self.engine.encode_count,
                               reused_probability=True)
            else:
                probability, timings = self.engine.predict(crop['image'], generation, points)
            mask = probability >= threshold
            mesh = surface_bytes(mask)
            sheet = dict(points=points, probability=probability, threshold=threshold,
                         mask=packed_mask(mask), mesh=mesh, timings=timings, revision=revision)
            timings['total_seconds'] = time.perf_counter() - started
            with self.lock:
                if generation != self.generation or self.revisions.get(sheet_id) != revision:
                    return dict(discarded=True)
                self.sheets[sheet_id] = sheet
            return dict(sheet=sheet_id, revision=revision, generation=generation,
                        voxels=int(mask.sum()), triangles=int.from_bytes(mesh[4:8], 'little'), timings=timings)
        return self.submit('predict', run)

    def automatic(self, payload):
        with self.lock:
            if payload['generation'] != self.generation:
                raise ValueError('The crop changed; reload the viewer.')
            generation, crop = self.generation, self.crop
            self.automatic_revision += 1
            revision = self.automatic_revision
        def run():
            def progress(message):
                self.ready = message
            result = self.engine.automatic(crop['image'], generation, progress)
            result['revision'] = revision
            with self.lock:
                if generation != self.generation or revision != self.automatic_revision:
                    return dict(discarded=True)
                self.automatic_result = result
                self.surface_cache = {k: v for k, v in self.surface_cache.items() if k[0] != 'automatic'}
                self.ready = 'Automatic instances ready; CT context retained for prompting'
            return self.metadata()
        return self.submit('automatic', run)

    def surface(self, payload):
        kind, identity = payload['kind'], int(payload['id'])
        if kind not in ('reference', 'automatic'):
            raise ValueError('Choose a reference or automatic surface.')
        with self.lock:
            if payload['generation'] != self.generation:
                raise ValueError('The crop changed; reload the viewer.')
            if kind == 'reference':
                labels = self.crop.get('reference')
                revision = self.generation
            else:
                labels = self.automatic_result['labels'] if self.automatic_result is not None else None
                revision = self.automatic_revision
            if labels is None or identity <= 0 or not np.any(labels == identity):
                raise ValueError('No voxels for the selected reference/instance ID.')
            generation = self.generation
            key = (kind, generation, revision, identity)
        def run():
            if key not in self.surface_cache:
                mesh = surface_bytes(labels == identity)
                with self.lock:
                    if generation != self.generation:
                        return dict(discarded=True)
                    # Meshes are requested lazily, so a dense automatic volume
                    # does not create every full-resolution surface at once.
                    if len(self.surface_cache) >= 16:
                        self.surface_cache.pop(next(iter(self.surface_cache)))
                    self.surface_cache[key] = mesh
            return dict(key='/'.join(map(str, key)), triangles=int.from_bytes(self.surface_cache[key][4:8], 'little'))
        return self.submit('surface', run)

    def mutate(self, path, payload):
        if path == '/api/automatic':
            return self.automatic(payload)
        if path == '/api/surface':
            return self.surface(payload)
        if path == '/api/predict':
            return self.predict(payload)
        if path == '/api/delete':
            with self.lock:
                if payload['generation'] != self.generation:
                    raise ValueError('The crop changed; reload the viewer.')
                sheet_id = int(payload['sheet'])
                self.revisions[sheet_id] = self.revisions.get(sheet_id, 0) + 1
                self.sheets.pop(sheet_id, None)
            return dict(deleted=sheet_id)
        if path == '/api/load':
            def load():
                if payload.get('source') == 'zarr':
                    crop = load_zarr_crop(payload['url'], payload['start_xyz'],
                        level=payload.get('level', 0), cache_dir=self.output.parent / 'zarr_cache')
                else:
                    crop = load_crop(str(payload['image']), str(payload.get('gt', '')),
                                     payload.get('reference_kind', 'instances'))
                with self.lock:
                    self.crop = crop
                    self.generation += 1
                    self.automatic_result = None
                    self.automatic_revision += 1
                    self.surface_cache.clear()
                    self.sheets.clear()
                    self.revisions.clear()
                    self.ready = 'Encoding the new CT crop'
                self.warmup()
                return self.metadata()
            return self.submit('load', load)
        if path == '/api/export':
            # Export only exact revisions displayed in the browser. Edited but
            # unsubmitted prompts must never be paired with an older mask.
            with self.lock:
                expected = {int(k): int(v) for k, v in payload['revisions'].items()}
                auto_revision = payload.get('automatic_revision')
                automatic = self.automatic_result if auto_revision is not None else None
                if payload['generation'] != self.generation or (not expected and automatic is None):
                    raise ValueError('Decode a sheet or run automatic segmentation before exporting.')
                if auto_revision is not None and (automatic is None or automatic['revision'] != auto_revision):
                    raise ValueError('Automatic predictions changed; refresh before exporting.')
                if any(k not in self.sheets or self.sheets[k]['revision'] != v for k, v in expected.items()):
                    raise ValueError('Predictions changed; wait for decoding before exporting.')
                crop = self.crop
                sheets = {k: self.sheets[k] for k in expected}
            folder = self.output / (datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_') + secrets.token_hex(3))
            formats = payload.get('formats', ['nifti'])
            if not formats or any(f not in ('nifti', 'tiff') for f in formats):
                raise ValueError('Choose NIFTI, TIFF, or both export formats.')
            return self.submit('export', lambda: dict(path=export_session(folder, crop, sheets,
                self.engine.source, formats=formats, automatic=automatic)))
        raise KeyError(path)


class Handler(BaseHTTPRequestHandler):
    server_version = 'PLatSViewer/2'

    def log_message(self, fmt, *args):
        # Polling/slice access is frequent; keep the terminal readable.
        if args and str(args[1] if len(args) > 1 else '').startswith(('4', '5')):
            super().log_message(fmt, *args)

    def send(self, data, mime='application/json', status=200):
        if isinstance(data, dict):
            data = json.dumps(data, allow_nan=False).encode()
        compressed = len(data) > 4096 and 'gzip' in self.headers.get('Accept-Encoding', '')
        if compressed:
            data = gzip.compress(data, compresslevel=1)
        self.send_response(status)
        self.send_header('Content-Type', mime)
        self.send_header('Content-Length', str(len(data)))
        self.send_header('Cache-Control', 'no-store' if self.path.startswith('/api/') else 'no-cache')
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.send_header('Referrer-Policy', 'no-referrer')
        if compressed:
            self.send_header('Content-Encoding', 'gzip')
        self.end_headers()
        self.wfile.write(data)

    def authorized(self):
        return secrets.compare_digest(self.headers.get('X-PLatS-Token', ''), self.server.app.token)

    def do_GET(self):
        try:
            path = urlsplit(self.path).path
            app = self.server.app
            if path.startswith('/api/'):
                if not self.authorized():
                    return self.send(dict(error='Open the complete viewer URL printed by the server (including #token).'), status=403)
                with app.lock:
                    if path == '/api/meta':
                        return self.send(app.metadata())
                    if path in ('/api/image', '/api/gt'):
                        array = app.crop['image' if path.endswith('image') else 'gt']
                        if array is None:
                            raise ValueError('No reference volume loaded.')
                        return self.send(array.tobytes(), 'application/octet-stream')
                    if path == '/api/reference':
                        labels = app.crop.get('reference')
                        if labels is None:
                            raise ValueError('No reference labels loaded.')
                        return self.send(labels.astype('<u4').tobytes(), 'application/octet-stream')
                    if path == '/api/automatic-labels':
                        if app.automatic_result is None:
                            raise ValueError('Run automatic segmentation first.')
                        return self.send(app.automatic_result['labels'].astype('<u2').tobytes(), 'application/octet-stream')
                    if path.startswith('/api/surface/'):
                        _, _, _, kind, generation, revision, identity = path.split('/')
                        key = (kind, int(generation), int(revision), int(identity))
                        return self.send(app.surface_cache[key], 'application/octet-stream')
                    if path.startswith('/api/jobs/'):
                        job = dict(app.jobs[path.rsplit('/', 1)[1]])
                        if job['kind'] == 'automatic':
                            job['message'] = app.ready
                        return self.send(job)
                    if path.startswith(('/api/mask/', '/api/mesh/')):
                        _, _, kind, key = path.split('/')
                        return self.send(app.sheets[int(key)][kind], 'application/octet-stream')
                return self.send(dict(error='Unknown API route'), status=404)
            relative = 'index.html' if path == '/' else path.lstrip('/')
            if path == '/favicon.ico':
                return self.send(b'', 'image/x-icon', status=204)
            file = (STATIC / relative).resolve()
            if not file.is_relative_to(STATIC.resolve()) or not file.is_file():
                return self.send(dict(error='Not found'), status=404)
            self.send(file.read_bytes(), mimetypes.guess_type(str(file))[0] or 'application/octet-stream')
        except (KeyError, ValueError) as error:
            self.send(dict(error=str(error)), status=400)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def do_POST(self):
        try:
            if not self.authorized():
                return self.send(dict(error='Invalid viewer token'), status=403)
            length = int(self.headers.get('Content-Length', 0))
            if length <= 0 or length > 65536 or self.headers.get('Content-Type') != 'application/json':
                return self.send(dict(error='Expected a small JSON request'), status=400)
            payload = json.loads(self.rfile.read(length))
            self.send(self.server.app.mutate(urlsplit(self.path).path, payload))
        except (ValueError, KeyError, TypeError) as error:
            self.send(dict(error=str(error)), status=400)
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as error:
            traceback.print_exc()
            self.send(dict(error=str(error)), status=500)


def main(repository_root):
    parser = argparse.ArgumentParser(description='PLatS interactive browser viewer (SSH port forwarding supported).')
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument('--image', help='Preprocessed uint8 crop; challenge TIFF is ZYX, NPY/NIFTI is XYZ')
    inputs.add_argument('--zarr-url', help='Official CT OME-Zarr root or array URL')
    parser.add_argument('--start-xyz', nargs=3, type=int, metavar=('X', 'Y', 'Z'), default=[0, 0, 0])
    parser.add_argument('--level', type=int, default=0, help='Zarr resolution level (default 0)')
    parser.add_argument('--reference-kind', choices=['instances', 'binary', 'challenge'], default='instances')
    parser.add_argument('--gt', default='', help='Optional aligned GT foreground/instance volume')
    parser.add_argument('--bundle-root', type=Path, default=repository_root, help='Extracted release with configs/ and weights/')
    parser.add_argument('--run-dir', type=Path, help='Optional jointly trained P2SD run instead of release weights')
    parser.add_argument('--checkpoint-dir', type=Path, help='Optional snapshot within the training run')
    parser.add_argument('--training-root', type=Path, default=repository_root, help='Resolve training config paths here')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--port', type=int, default=8787)
    parser.add_argument('--token-file', type=Path, help='Private token file to retain the URL across restarts')
    parser.add_argument('--output', type=Path, default=Path(repository_root) / 'outputs/interactive')
    args = parser.parse_args()
    if args.checkpoint_dir and not args.run_dir:
        parser.error('--checkpoint-dir requires --run-dir')
    if not 1 <= args.threads <= 64:
        parser.error('--threads must be 1–64')
    if args.zarr_url:
        if args.gt:
            parser.error('--gt is currently for local crops; load aligned local CT/reference to compare a sheet')
        crop = load_zarr_crop(args.zarr_url, args.start_xyz, level=args.level,
                              cache_dir=args.output.parent / 'zarr_cache')
    else:
        crop = load_crop(args.image, args.gt, args.reference_kind)
    engine = PromptEngine(args.training_root, args.bundle_root, args.device,
                          args.run_dir, args.checkpoint_dir, args.threads)
    app = App(engine, crop, args.output)
    if args.token_file:
        if args.token_file.exists():
            token = args.token_file.read_text().strip()
            if len(token) < 32:
                parser.error('--token-file must contain a token of at least 32 characters')
            app.token = token
        else:
            import os
            args.token_file.parent.mkdir(parents=True, exist_ok=True)
            descriptor = os.open(args.token_file, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(descriptor, 'w') as stream:
                stream.write(app.token)
    server = ThreadingHTTPServer(('127.0.0.1', args.port), Handler)
    server.daemon_threads = True
    server.app = app
    print(f'PLatS viewer: http://localhost:{server.server_port}/#{app.token}', flush=True)
    print(f'SSH tunnel: ssh -N -L {server.server_port}:127.0.0.1:{server.server_port} USER@SERVER', flush=True)
    print(f'Exports: {args.output.resolve()}\nModel preparation runs in the background.', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        app.worker.shutdown(wait=True, cancel_futures=True)


if __name__ == '__main__':
    main(Path(__file__).resolve().parents[3])
