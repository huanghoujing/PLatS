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

STATIC = Path(__file__).with_name('static')


class App:
    def __init__(self, engine, crop, output):
        self.engine, self.crop, self.output = engine, crop, Path(output)
        self.token = secrets.token_urlsafe(24)
        self.generation = 1
        self.sheets, self.revisions, self.jobs = {}, {}, {}
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
            return dict(shape=list(self.crop['image'].shape), path=self.crop['path'],
                gt_path=self.crop['gt_path'], has_gt=self.crop['gt'] is not None,
                generation=self.generation, ready=self.ready, device=self.engine.device_name,
                coordinate_order='Native array axes 0 / 1 / 2 (no reorientation)',
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

    def mutate(self, path, payload):
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
                crop = load_crop(str(payload['image']), str(payload.get('gt', '')))
                with self.lock:
                    self.crop = crop
                    self.generation += 1
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
                if payload['generation'] != self.generation or not expected:
                    raise ValueError('Decode at least one sheet before exporting.')
                if any(k not in self.sheets or self.sheets[k]['revision'] != v for k, v in expected.items()):
                    raise ValueError('Predictions changed; wait for decoding before exporting.')
                crop = self.crop
                sheets = {k: self.sheets[k] for k in expected}
            folder = self.output / (datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_') + secrets.token_hex(3))
            return self.submit('export', lambda: dict(path=export_session(folder, crop, sheets, self.engine.source)))
        raise KeyError(path)


class Handler(BaseHTTPRequestHandler):
    server_version = 'PLatSViewer/1'

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
                    if path.startswith('/api/jobs/'):
                        return self.send(app.jobs[path.rsplit('/', 1)[1]])
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
    parser.add_argument('--image', required=True, help='Preprocessed uint8 crop, each side 32–320; .npy/.nii[.gz]/.tif')
    parser.add_argument('--gt', default='', help='Optional aligned GT foreground/instance volume')
    parser.add_argument('--bundle-root', type=Path, default=repository_root, help='Extracted release with configs/ and weights/')
    parser.add_argument('--run-dir', type=Path, help='Optional jointly trained P2SD run instead of release weights')
    parser.add_argument('--checkpoint-dir', type=Path, help='Optional snapshot within the training run')
    parser.add_argument('--training-root', type=Path, default=repository_root, help='Resolve training config paths here')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--port', type=int, default=8787)
    parser.add_argument('--output', type=Path, default=Path(repository_root) / 'outputs/interactive')
    args = parser.parse_args()
    if args.checkpoint_dir and not args.run_dir:
        parser.error('--checkpoint-dir requires --run-dir')
    if not 1 <= args.threads <= 64:
        parser.error('--threads must be 1–64')
    crop = load_crop(args.image, args.gt)
    engine = PromptEngine(args.training_root, args.bundle_root, args.device,
                          args.run_dir, args.checkpoint_dir, args.threads)
    app = App(engine, crop, args.output)
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
