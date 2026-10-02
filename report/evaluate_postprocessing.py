"""Frozen 106-case AE/closing/dusting ablation for report Tables 1--3.

Run from the research workspace (or set PLATS_RESEARCH_ROOT). Predictions and
intermediates are separate from all historical results. Inference never loads
GT or ignore masks. Scoring starts after all four new arms are frozen.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import sys
import time

for variable in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[variable] = '1'

import cc3d
import numpy as np
from scipy import ndimage

ROOT = Path(os.environ.get('PLATS_RESEARCH_ROOT', Path.cwd())).resolve()
OUT = ROOT / 'runs_from_260914/evaluation/16_postprocessing_hidden106'
sys.path.insert(0, str(ROOT / 'src'))
from vesuvius_p2sd.eval.topology import betti_numbers

SOURCE = ROOT / 'runs_from_260914/evaluation/09_paper_raw_prediction_betti/input_manifest.json'
OLD = ROOT / 'runs_from_260914/evaluation/05_paper_winner_ignore106_gpu0_gpu2/results.json'
AE = ROOT / 'runs/0032_ae_t6_denoise_repel_e150_r1'
ARMS = ('plats_ae', 'plats_ae_close', 'plats_ae_close_dust', 'winner_close_dust')
FIELDS = ('dice', 'surface_dice_tau2', 'toposcore', 'voi_score', 'leaderboard_formula_score')


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def sha(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def load_instances(case, method):
    source = case['predictions'][method]
    path = ROOT / source['path']
    assert sha(path) == source['sha256']
    with np.load(path) as archive:
        values = archive[source['array_key']]
    assert list(values.shape) == case['shape']
    if method == 'winner':
        values = cc3d.connected_components(np.ascontiguousarray(values > 0), connectivity=6)
    return values.astype(np.int32)


def prepare():
    cases = read(SOURCE)['cases']
    assert len(cases) == len({c['case_id'] for c in cases}) == 106
    plan = dict(case_count=106, cases=cases, source_manifest_sha256=sha(SOURCE),
                ae_checkpoint=str(AE / 'last.pt'), ae_checkpoint_sha256=sha(AE / 'last.pt'),
                ae_config_sha256=sha(AE / 'resolved_config.yaml'),
                recipe=dict(ae='0032 epoch150; encode and decode each existing 0058 instance',
                            threshold=0.5, threshold_comparison='strictly greater', precision='bfloat16',
                            native_padding='Center-pad 256 cubes to 320, then crop back; 320 cubes unchanged',
                            overlaps='Highest AE probability, ties lower original ID',
                            closing='Per original ID, additive guarded 3^3 then 5^3; accept reduced sum of V Betti counts; skip (1,0,0)',
                            closing_guard='Forbid new voxels overlapping or 26-touching another ID; sequential ascending IDs',
                            dust_max_voxels=5, dust_connectivity=6,
                            restoration_full_instance_min_voxels=5000),
                arms=list(ARMS), inference_uses_gt_or_ignore=False,
                scoring='Same native cases and public formula as original tables; standard erasure and full restoration diagnostic',
                limitations=['Recipe selected after inspecting three released-test outliers; not a fresh blind evaluation.',
                             'Winner identities are pre-cleanup six-connected components, not physical sheet IDs.',
                             'Dust is per-ID connected pieces <=5; restoration filter is whole IDs with full size >=5000.'],
                created_unix=time.time(), script_sha256=sha(__file__))
    if (OUT / 'plan.json').exists():
        existing = read(OUT / 'plan.json')
        assert existing['recipe'] == plan['recipe'] and existing['cases'] == cases
    else:
        write(OUT / 'plan.json', plan)


def guarded_close(instances):
    """Existing repair_component_topology algorithm, copied for reproduction.

    Preserve every original voxel. A candidate is accepted only if its Betti
    sum decreases; this heuristic is not a topology or geometry guarantee.
    """
    instances = instances.copy()
    reports = []
    for identity in np.unique(instances):
        if identity == 0:
            continue
        full = instances == identity
        bounds = ndimage.find_objects(full.astype(np.uint8), max_label=1)[0]
        window = tuple(slice(max(s.start - 3, 0), min(s.stop + 3, dim))
                       for s, dim in zip(bounds, instances.shape))
        local = instances[window]
        mask = local == identity
        before = tuple(betti_numbers(mask, construction='V'))
        best, counts = mask, before
        sizes_used = []
        if before != (1, 0, 0):
            forbidden = ndimage.binary_dilation((local > 0) & ~mask, structure=np.ones((3,) * 3, bool))
            for size in (3, 5):
                closed = ndimage.binary_closing(best, structure=np.ones((size,) * 3, bool))
                candidate = best | (closed & ~best & ~forbidden)
                candidate_counts = tuple(betti_numbers(candidate, construction='V'))
                if sum(candidate_counts) < sum(counts) or candidate_counts == (1, 0, 0):
                    best, counts = candidate, candidate_counts
                    sizes_used.append(size)
                if counts == (1, 0, 0):
                    break
        added = best & ~mask
        local[added] = identity
        reports.append(dict(id=int(identity), before=list(before), after=list(counts),
                            kernels_accepted=sizes_used, added_voxels=int(added.sum())))
    return instances, reports


def dust(instances):
    instances = instances.copy()
    reports = []
    for identity, bounds in enumerate(ndimage.find_objects(instances), 1):
        if bounds is None:
            continue
        local = instances[bounds]
        labels = cc3d.connected_components(np.ascontiguousarray(local == identity), connectivity=6)
        sizes = np.bincount(labels.ravel())
        remove = (sizes <= 5) & (np.arange(len(sizes)) > 0)
        removed_sizes = sizes[remove].tolist()
        local[remove[labels]] = 0
        reports.append(dict(id=identity, removed_component_sizes=removed_sizes,
                            removed_voxels=sum(removed_sizes)))
    return instances, reports


def save_stage(case, arm, instances, audit):
    import nibabel as nib
    dest = OUT / 'predictions' / arm / case['case_id']
    dest.mkdir(parents=True, exist_ok=True)
    values = instances.astype(np.int32)
    np.savez_compressed(dest / 'instances.npz', inst=values)
    # Match the existing native NIFTI convention; image/GT/points are linked.
    source = ROOT / 'runs_from_260914/evaluation/04_paper_automatic_hidden106_gpu0_gpu2/0058_kaggle/cases' / case['case_id']
    affine = nib.load(source / 'pred_instances.nii.gz').affine
    nib.save(nib.Nifti1Image(values, affine), dest / 'pred_instances.nii.gz')
    for filename in (('image.nii.gz', 'sampled_points.nii.gz') if arm.startswith('plats') else ('image.nii.gz',)):
        link = dest / filename
        if not link.exists():
            link.symlink_to(os.path.relpath(source / filename, dest))
    record = dict(case_id=case['case_id'], arm=arm, native_shape=list(values.shape),
                  instances_sha256=sha(dest / 'instances.npz'),
                  raw_betti=list(betti_numbers(values > 0, construction='V')),
                  foreground_voxels=int(np.count_nonzero(values)),
                  prediction_ids=[int(v) for v in np.unique(values) if v],
                  labels_or_ignore_loaded=False, audit=audit)
    write(dest / 'inference.json', record)
    return record


def postprocess(case, winner=False):
    final_arm = 'winner_close_dust' if winner else 'plats_ae_close_dust'
    if (OUT / 'predictions' / final_arm / case['case_id'] / 'inference.json').exists():
        return case['case_id']
    if winner:
        instances = load_instances(case, 'winner')
    else:
        with np.load(OUT / 'predictions/plats_ae' / case['case_id'] / 'instances.npz') as archive:
            instances = archive['inst']
    closed, closing_audit = guarded_close(instances)
    assert np.all(closed[instances > 0] == instances[instances > 0])
    if not winner:
        save_stage(case, 'plats_ae_close', closed, dict(closing=closing_audit))
    cleaned, dust_audit = dust(closed)
    assert np.all((cleaned == 0) | (cleaned == closed))
    save_stage(case, final_arm, cleaned, dict(closing=closing_audit, dust=dust_audit))
    return case['case_id']


def infer(gpu, shard, shards):
    import torch
    from vesuvius_p2sd.train.common import load_ae_from_config
    plan = read(OUT / 'plan.json')
    assert sha(AE / 'last.pt') == plan['ae_checkpoint_sha256']
    torch.set_num_threads(2)
    device = torch.device(f'cuda:{gpu}')
    model, _ = load_ae_from_config(AE / 'resolved_config.yaml', AE / 'last.pt')
    model = model.to(device).eval()
    selected = plan['cases'][shard::shards]
    with ProcessPoolExecutor(max_workers=6, mp_context=multiprocessing.get_context('spawn')) as pool:
        pending = []
        for index, case in enumerate(selected):
            dest = OUT / 'predictions/plats_ae' / case['case_id']
            if not (dest / 'inference.json').exists():
                start = time.monotonic()
                inst = load_instances(case, 'plats_auto')
                shape = inst.shape
                offset = tuple((320 - s) // 2 for s in shape)
                bounds = tuple(slice(o, o + s) for o, s in zip(offset, shape))
                best_prob = torch.zeros(shape, device=device, dtype=torch.float32)
                best_id = torch.zeros(shape, device=device, dtype=torch.int32)
                for identity in np.unique(inst):
                    if identity == 0:
                        continue
                    padded = np.zeros((320,) * 3, np.float32)
                    padded[bounds] = inst == identity
                    x = torch.from_numpy(padded)[None, None].to(device)
                    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
                        logits = model.decode(model.encode(x))[0, 0][bounds].float()
                    assert torch.isfinite(logits).all()
                    prob = logits.sigmoid()
                    claim = (logits > 0) & (prob > best_prob)
                    best_id[claim] = int(identity)
                    best_prob[claim] = prob[claim]
                    del x, logits, prob, claim
                result = best_id.cpu().numpy()
                save_stage(case, 'plats_ae', result, dict(seconds=time.monotonic() - start,
                    checkpoint_sha256=plan['ae_checkpoint_sha256'], canvas_offset=list(offset)))
                del best_prob, best_id
            pending.append(pool.submit(postprocess, case))
            if len(pending) >= 12:
                pending.pop(0).result()
            write(OUT / f'gpu{gpu}_status.json', dict(decoded=index + 1, total=len(selected), latest=case['case_id']))
            print(f'GPU{gpu}: {index + 1}/{len(selected)} {case["case_id"]}', flush=True)
        for future in pending:
            future.result()
    write(OUT / f'gpu{gpu}_status.json', dict(status='complete', total=len(selected)))


def winner(workers):
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(postprocess, case, True) for case in read(OUT / 'plan.json')['cases']]
        for index, future in enumerate(as_completed(futures), 1):
            print('winner cleanup', index, future.result(), flush=True)


def freeze():
    records = []
    for arm in ARMS:
        for case in read(OUT / 'plan.json')['cases']:
            dest = OUT / 'predictions' / arm / case['case_id']
            record = read(dest / 'inference.json')
            assert sha(dest / 'instances.npz') == record['instances_sha256']
            assert record['native_shape'] == case['shape'] and not record['labels_or_ignore_loaded']
            records.append(record)
    write(OUT / 'prediction_freeze.json', dict(status='complete', prediction_count=len(records),
          plan_sha256=sha(OUT / 'plan.json'), records=records, labels_or_ignore_loaded=False))


def prepare_case_scores(case):
    from vesuvius_p2sd.data.build_ignore_masks import load_ignore_mask
    tasks, references = {}, []
    name = case['case_id']
    meta = read(ROOT / 'datasets/hf_kaggle_202607/cases' / name / 'meta.json')
    gt = np.load(meta['components_path']) > 0
    ignore = load_ignore_mask(meta['ignore_path'], gt.shape)
    gt &= ~ignore
    gt_path = OUT / 'scoring_inputs' / name / 'gt.npz'
    gt_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(gt_path, mask=gt)
    for arm in ARMS:
        dest = OUT / 'predictions' / arm / name
        inference = read(dest / 'inference.json')
        assert sha(dest / 'instances.npz') == inference['instances_sha256']
        with np.load(dest / 'instances.npz') as archive:
            instances = archive['inst']
        counts = np.bincount(instances.ravel())
        visible = np.bincount(instances[~ignore], minlength=len(counts))
        keep = (counts >= 5000) & (visible > 0)
        keep[0] = False
        masks = dict(standard=(instances > 0) & ~ignore, restored_filtered=keep[instances])
        for variant, mask in masks.items():
            digest = hashlib.sha256(np.packbits(mask).tobytes()).hexdigest()
            key = name + '_' + digest
            path = OUT / 'scoring_inputs' / name / (digest + '.npz')
            if key not in tasks:
                np.savez_compressed(path, mask=mask)
                tasks[key] = dict(key=key, case_id=name, prediction_path=str(path), gt_path=str(gt_path),
                                  prediction_sha256=sha(path), gt_sha256=sha(gt_path))
            references.append(dict(arm=arm, case_id=name, variant=variant, key=key))
        # Link viewing assets only now, after inference is frozen.
        source = ROOT / 'runs_from_260914/evaluation/04_paper_automatic_hidden106_gpu0_gpu2/0058_kaggle/cases' / name
        for filename in ('gt_components.nii.gz', 'source_ignore.nii.gz'):
            link = dest / filename
            if not link.exists():
                link.symlink_to(os.path.relpath(source / filename, dest))
    return list(tasks.values()), references


def prepare_scores(workers=16):
    """Deduplicate identical masks, preparing independent native cases in parallel."""
    frozen = read(OUT / 'prediction_freeze.json')
    assert frozen['prediction_count'] == 424
    tasks, references = [], []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for case_tasks, case_refs in pool.map(prepare_case_scores, read(OUT / 'plan.json')['cases']):
            tasks.extend(case_tasks)
            references.extend(case_refs)
    write(OUT / 'score_tasks.json', dict(tasks=tasks, references=references,
           logical_scores=len(references), unique_scores=len(tasks), prediction_freeze_sha256=sha(OUT / 'prediction_freeze.json')))
    print('Scoring:', len(references), 'logical pairs;', len(tasks), 'distinct binary pairs', flush=True)


def score_one(task):
    import resource
    from vesuvius_p2sd.eval.sheet_report import score_masks
    path = OUT / 'scores' / (task['key'] + '.json')
    if path.exists():
        return read(path)
    start = time.monotonic()
    assert sha(task['prediction_path']) == task['prediction_sha256']
    assert sha(task['gt_path']) == task['gt_sha256']
    with np.load(task['prediction_path']) as archive:
        pred = archive['mask']
    with np.load(task['gt_path']) as archive:
        gt = archive['mask']
    metrics = score_masks(pred, gt, np.zeros(gt.shape, bool), topology_backend='compact_exact', topology_tile_batch_size=1)
    result = dict(key=task['key'], case_id=task['case_id'], metrics=metrics,
                  seconds=time.monotonic() - start, peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                  prediction_sha256=task['prediction_sha256'], gt_sha256=task['gt_sha256'])
    write(path, result)
    return result


def score(workers):
    tasks = read(OUT / 'score_tasks.json')['tasks']
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(score_one, task) for task in tasks]
        for index, future in enumerate(as_completed(futures), 1):
            result = future.result()
            if index % 10 == 0 or index == len(tasks):
                print('scored', index, '/', len(tasks), 'last_seconds', round(result['seconds'], 1), flush=True)
                write(OUT / 'score_status.json', dict(done=index, total=len(tasks)))


def summarize():
    job = read(OUT / 'score_tasks.json')
    frozen = read(OUT / 'prediction_freeze.json')
    lookup = {(r['arm'], r['case_id']): r for r in frozen['records']}
    rows = []
    for arm in ARMS:
        for case in read(OUT / 'plan.json')['cases']:
            name = case['case_id']
            metrics = {}
            for ref in job['references']:
                if ref['arm'] == arm and ref['case_id'] == name:
                    metrics[ref['variant']] = read(OUT / 'scores' / (ref['key'] + '.json'))['metrics']
            assert len(metrics) == 2
            rows.append(dict(arm=arm, case_id=name, metrics=metrics, raw_betti=lookup[arm, name]['raw_betti']))
    previous = read(OLD)
    for original, name in [('winner', 'winner'), ('plats_auto', 'plats_base')]:
        for row in previous['rows']:
            if row['arm'] == original:
                rows.append(dict(arm=name, case_id=row['case_id'], raw_betti=row['raw_betti'],
                    metrics={v:row['scores'][v]['metrics'] for v in ('standard', 'restored_filtered')}))
    means = {}
    for arm in ('winner', 'winner_close_dust', 'plats_base', 'plats_ae', 'plats_ae_close', 'plats_ae_close_dust'):
        group = [r for r in rows if r['arm'] == arm]
        assert len(group) == len({r['case_id'] for r in group}) == 106
        betti = np.array([r['raw_betti'] for r in group])
        entry = {v:{k:float(np.mean([r['metrics'][v][k] for r in group])) for k in FIELDS}
                 for v in ('standard', 'restored_filtered')}
        entry['raw_betti'] = dict(mean=betti.mean(0).tolist(), median=np.median(betti, 0).tolist(),
                                 total=betti.sum(0).tolist(), nonzero_cases=(betti > 0).sum(0).tolist())
        for variant in ('standard', 'restored_filtered'):
            m = entry[variant]
            assert abs(m['leaderboard_formula_score'] - (.35*m['surface_dice_tau2']+.30*m['toposcore']+.35*m['voi_score'])) < 1e-10
        means[arm] = entry
    write(OUT / 'results.json', dict(status='complete', cases=106, means=means, rows=rows,
           plan=read(OUT / 'plan.json'), prediction_freeze_sha256=sha(OUT / 'prediction_freeze.json')))
    print(json.dumps(means, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare', 'infer', 'winner', 'freeze', 'prepare_scores', 'score', 'summarize'])
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--shard', type=int, default=0)
    parser.add_argument('--shards', type=int, default=2)
    parser.add_argument('--workers', type=int, default=48)
    args = parser.parse_args()
    if args.action == 'infer':
        infer(args.gpu, args.shard, args.shards)
    elif args.action in ('winner', 'score', 'prepare_scores'):
        globals()[args.action](args.workers)
    else:
        globals()[args.action]()
