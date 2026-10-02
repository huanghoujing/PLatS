#!/usr/bin/env python3
"""Run the unchanged automatic recipe with seed support restricted to one CT plane."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time
from unittest.mock import patch

import numpy as np
import torch
import vesuvius_p2sd.research.auto_instance_seg as automatic
from vesuvius_p2sd.eval.sheet_report import save_nifti


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--research_root', type=Path, required=True)
    parser.add_argument('--public_root', type=Path, required=True)
    args = parser.parse_args()
    root = args.research_root.resolve()
    os.chdir(root)
    out = root/'runs_from_260914/evaluation/12_paper_slice_seed_example'
    out.mkdir(parents=True, exist_ok=True)
    case, z = 'sample_00860', 160
    base = root/'runs_from_260914/evaluation/04_paper_automatic_hidden106_gpu0_gpu2'
    plan = json.loads((base/'plan.json').read_text())
    ct_record = next(r for r in json.loads((base/'inference_manifest.json').read_text())['cases'] if r['case_id']==case)
    image = np.load(ct_record['image_path'])
    assert image.shape == (320,320,320)
    def sha(path):
        with Path(path).open('rb') as f:
            return hashlib.file_digest(f,'sha256').hexdigest()
    assert sha(ct_record['image_path']) == ct_record['image_sha256']
    foreground_record = json.loads((base/'foreground'/case/'complete.json').read_text())
    assert sha(base/'foreground'/case/'mask.npz') == foreground_record['mask_sha256']
    with np.load(base/'foreground'/case/'mask.npz') as f:
        original_fg = f['mask']
    seed_support = np.zeros_like(original_fg)
    seed_support[:,:,z] = original_fg[:,:,z]
    settings = json.loads((args.public_root/'configs/automatic.json').read_text())
    assert settings == plan['settings']
    torch.set_num_threads(2)
    automatic._DECODE_BATCH = 2
    device = torch.device('cuda:0')
    model_run = root/'runs/0058_p2sd_joint_dense_ps160_warm0033_e150'
    assert sha(model_run/'last.pt') == plan['models']['0058_kaggle']['checkpoint_sha256']
    model, ae, codec, _ = automatic.load_p2sd_stack(model_run, device)
    captured = {}
    original_cluster = automatic.cluster_latents

    def capture(latents, **kwargs):
        groups = original_cluster(latents, **kwargs)
        assert not captured, 'Expected exactly one initial clustering pass.'
        captured['latents'] = latents.detach().float().cpu().flatten(1)
        captured['groups'] = groups
        return groups

    started = time.monotonic()
    with torch.inference_mode(), patch.object(automatic, 'cluster_latents', capture):
        result = automatic.run_case_cluster(binseg_model=None,p2sd_model=model,
            target_ae=ae,latent_codec=codec,image=image,valid=np.ones_like(image,dtype=bool),
            device=device,dtype=torch.bfloat16,external_mask=seed_support,**settings)
    points = np.asarray(result['points']['clicks'], dtype=np.int16)
    assert len(points)==512 and (points[:,2]==z).all()
    assert seed_support[tuple(points.T)].all()
    assert len(np.unique(points,axis=0))==len(points)
    # The seed support is planar; the validity mask and decoded sheets remain 3D.
    instances = result['instance_ids']
    assert np.count_nonzero(instances)>np.count_nonzero(instances[:,:,z])
    groups = captured['groups']
    kept = [g for g in groups if len(g)>settings['cluster_min_points']]
    log = result['seed_log'][0]
    assert [len(g) for g in kept]==log['cluster_sizes']
    assert len(kept)==len(log['assigned'])
    identity=np.zeros(len(points),dtype=np.int16)
    cluster_id=np.zeros(len(points),dtype=np.int16)
    for i,g in enumerate(groups,1):cluster_id[g]=i
    for g,assigned in zip(kept,log['assigned']):identity[g]=max(0,assigned)
    with torch.inference_mode():
        latent=captured['latents'].to(device,dtype=torch.float64)
        distance=(torch.cdist(latent,latent).square()/latent.shape[1]).clamp_min(0)
        distance.fill_diagonal_(0)
        distance=distance.float().cpu().numpy()
    for a,b in [(0,1),(23,58),(137,390),(290,511)]:
        expected=(captured['latents'][a].double()-captured['latents'][b].double()).square().mean().item()
        np.testing.assert_allclose(distance[a,b],expected,rtol=2e-6,atol=1e-8)
    np.savez_compressed(out/'figure_data.npz',image=image[:,:,z].T,points=points,
        identity=identity,cluster_id=cluster_id,distance=distance,instances=instances[:,:,z].T)
    np.savez_compressed(out/'instances.npz',inst=instances)
    np.savez_compressed(out/'latents.npz',latent=captured['latents'].numpy().astype(np.float16))
    for name,value in [('points.json',result['points']),('seeds.json',result['seed_log'])]:
        (out/name).write_text(json.dumps(value,indent=2)+'\n')
    save_nifti(out/'pred_instances.nii.gz',instances.astype(np.int16))
    pointmap=np.zeros(image.shape,dtype=np.uint8);pointmap[tuple(points.T)]=1
    save_nifti(out/'sampled_points.nii.gz',pointmap)
    link=out/'image.nii.gz'
    if not link.exists():link.symlink_to(base/'foreground'/case/'image.nii.gz')
    record=dict(case=case,native_z=z,seed=0,seed_count=len(points),all_seeds_on_displayed_plane=True,
        sampling='Uniform foreground sampling without replacement, restricted to native z=160.',
        foreground_source='Same cached 0022 proposer as the original volume-seed example.',
        cluster_count=len(groups),retained_cluster_count=len(kept),
        retained_seed_count=int((identity>0).sum()),discarded_seed_count=int((identity==0).sum()),
        final_instance_count=int(len(np.unique(instances[instances>0]))),
        inference_seconds=time.monotonic()-started,settings=settings,
        labels_or_ignore_loaded=False,three_dimensional_CT_and_decoding=True,
        selection='Same existing case and slice; first fixed-seed run, without result-based selection.',
        benchmark_tables_updated=False,
        source_checkpoint_sha256=plan['models']['0058_kaggle']['checkpoint_sha256'])
    (out/'capture.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record,indent=2),flush=True)

if __name__=='__main__':main()
