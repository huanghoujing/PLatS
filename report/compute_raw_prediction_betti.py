#!/usr/bin/env python3
"""Count whole-volume Betti numbers on frozen prediction unions, without GT."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time

for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[name] = '1'

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from vesuvius_p2sd.eval.topology import betti_numbers


def sha(path: Path) -> str:
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def count_one(task: dict) -> dict:
    start = time.perf_counter()
    path = Path(task['path'])
    digest = sha(path)
    if digest != task['sha256']:
        raise ValueError(f'Prediction changed: {path}')
    with np.load(path, allow_pickle=False) as archive:
        mask = np.ascontiguousarray(archive[task['array_key']] > 0)
    if list(mask.shape) != task['shape']:
        raise ValueError(f'Native shape mismatch: {path}: {mask.shape}')
    # Do not load or apply ignore/GT masks, crop the volume, or filter IDs.
    counts = list(betti_numbers(mask, construction='V'))
    expected = task.get('previous_betti')
    if expected is not None and counts != expected:
        raise ValueError(f'Counts disagree with previous audit: {path}: {counts} vs {expected}')
    return {'case_id': task['case_id'], 'method': task['method'],
            'shape': list(mask.shape), 'foreground_voxels': int(mask.sum()),
            'betti': counts, 'prediction_sha256': digest,
            'source': task['relative_path'], 'array_key': task['array_key'],
            'matches_previous_audit': counts == expected if expected is not None else None,
            'seconds': time.perf_counter() - start}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--prediction_root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=16)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    tasks = []
    for case in manifest['cases']:
        for method, source in case['predictions'].items():
            tasks.append(dict(source, case_id=case['case_id'], method=method,
                              shape=case['shape'], relative_path=source['path'],
                              path=str(args.prediction_root / source['path'])))
    keys = [(t['case_id'], t['method']) for t in tasks]
    if len(keys) != len(set(keys)):
        raise ValueError('Duplicate case/method in manifest')
    start = time.perf_counter()
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(count_one, task) for task in tasks]
        for future in as_completed(futures):
            rows.append(future.result())
            if len(rows) % 20 == 0 or len(rows) == len(tasks):
                print(f'{len(rows)}/{len(tasks)} volumes: {time.perf_counter() - start:.1f}s', flush=True)
    rows.sort(key=lambda row: (row['method'], row['case_id']))
    summary = {}
    for method in sorted({row['method'] for row in rows}):
        group = [row for row in rows if row['method'] == method]
        counts = np.asarray([row['betti'] for row in group], dtype=np.int64)
        summary[method] = {'cases': len(group), 'mean_betti': counts.mean(0).tolist(),
                           'median_betti': np.median(counts, axis=0).tolist(),
                           'sum_betti': counts.sum(0).tolist(),
                           'nonzero_cases': (counts > 0).sum(0).tolist(),
                           'p90_betti': np.quantile(counts, 0.9, axis=0).tolist()}
    result = {'status': 'complete', 'created_utc': datetime.now(timezone.utc).isoformat(),
              'case_count': len(manifest['cases']), 'workers': args.workers,
              'wall_seconds': time.perf_counter() - start,
              'protocol': {'construction': 'V', 'foreground_connectivity': 6,
                           'background_connectivity': 26, 'volume': 'native prediction union',
                           'ignore_processing': False, 'gt_loaded': False,
                           'additional_filtering': False,
                           'inference_postprocessing': 'retained as in Tables 1 and 2 inputs',
                           'aggregation': 'equal weight per case'},
              'source_sha256': {'manifest': sha(args.manifest), 'script': sha(Path(__file__)),
                                'topology': sha(ROOT / 'src/vesuvius_p2sd/eval/topology.py'),
                                'connected_components': sha(ROOT / 'src/vesuvius_p2sd/eval/connected_components.py')},
              'summary': summary, 'rows': rows}
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'raw_prediction_betti.json').write_text(json.dumps(result, indent=2) + '\n')
    with (args.output / 'raw_prediction_betti.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['method', 'case_id', 'b0', 'b1', 'b2', 'foreground_voxels', 'prediction_sha256'])
        for row in rows:
            writer.writerow([row['method'], row['case_id'], *row['betti'],
                             row['foreground_voxels'], row['prediction_sha256']])
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
