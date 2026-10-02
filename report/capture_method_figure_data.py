"""Capture genuine examples for the three method figures, without retraining."""
import argparse
import os
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import distance_transform_edt
from vesuvius_p2sd.research.auto_instance_seg import load_p2sd_stack, cluster_latents
from vesuvius_p2sd.train.ae_corruption import resolve_ae_input_corruption, corrupt_sheet_mask
from vesuvius_p2sd.utils.config import load_config

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--research_root', type=Path, required=True)
    parser.add_argument('--public_root', type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    ROOT = args.research_root.resolve()
    public_root = args.public_root.resolve()
    os.chdir(ROOT)  # Original run configs contain research-relative paths.
    OUT = ROOT / 'runs_from_260914/evaluation/11_paper_three_method_figures'
    OUT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.manual_seed(42)
    device = torch.device('cuda:0')
    source = ROOT/'runs_from_260914/evaluation/08_paper_method_visuals'
    with np.load(source/'intermediates.npz') as cache:
        codes = torch.from_numpy(cache['latent'].astype(np.float32))
        points, identity = cache['points_xyz'], cache['identity']
    clusters = cluster_latents(codes, mse_threshold=.02)
    kept = [c for c in clusters if len(c) > 8]
    auto = ROOT/'runs_from_260914/evaluation/04_paper_automatic_hidden106_gpu0_gpu2/0058_kaggle/cases/sample_00860'
    log = json.loads((auto/'seeds.json').read_text())[0]
    assert [len(c) for c in kept] == log['cluster_sizes']
    replayed = np.zeros(len(points), np.int16)
    cluster_id = np.zeros(len(points), np.int16)
    for i, members in enumerate(clusters, 1):
        cluster_id[members] = i
    for members, final_id in zip(kept, log['assigned']):
        replayed[members] = final_id
    np.testing.assert_array_equal(replayed, identity)
    with torch.inference_mode():
        x = codes.to(device, dtype=torch.float64)
        distance = (torch.cdist(x, x).square() / x.shape[1]).clamp_min(0)
        distance.fill_diagonal_(0)
        distance = distance.float().cpu().numpy()
    del x
    print('Verified saved clustering and computed full-code distance matrix.', flush=True)

    case = ROOT/'datasets/hf_kaggle_202607/cases/sample_00860'
    meta = json.loads((case/'meta.json').read_text())
    image = np.load(meta['image_path'])
    instances = np.load(meta['components_path'])
    gt = instances == 9
    other = instances == 8
    zslice = 160
    model, ae, codec, cfg = load_p2sd_stack(ROOT/'runs/0058_p2sd_joint_dense_ps160_warm0033_e150', device)
    ae_cfg = load_config(ROOT/'runs/0032_ae_t6_denoise_repel_e150_r1/resolved_config.yaml')
    clean = torch.from_numpy(gt.astype(np.float32))[None, None].to(device)
    second = torch.from_numpy(other.astype(np.float32))[None, None].to(device)
    corrupt_cfg = resolve_ae_input_corruption(ae_cfg['data'])
    corrupted = corrupt_sheet_mask(clean, corrupt_cfg)
    assert (corrupted != clean).any(), 'The fixed corruption seed must actually corrupt the input.'
    prompts = np.array(json.loads((public_root/'examples/sample_00860/points.json').read_text())['points'])
    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
        output = ae(corrupted)
        teacher_raw = ae.encode(clean)
        other_raw = ae.encode(second)
        teacher = codec.model_target(teacher_raw)
        cosine = F.cosine_similarity(teacher_raw.float().flatten(1), other_raw.float().flatten(1)).item()
        _, tokens, coords, context = model.encode_image_context_from_image(torch.from_numpy(image.astype(np.float32))[None, None].to(device))
        pred = model.forward_from_image_context(tokens, coords, context,
            torch.tensor(prompts[None], device=device, dtype=torch.float32),
            torch.ones((1,len(prompts)), device=device, dtype=torch.long),
            image_shape=image.shape, image_index=torch.zeros(1, device=device, dtype=torch.long))['latent']
        pred_mask = ae.decode(codec.raw_prediction(pred)).sigmoid()
        code_mse = (pred.float() - teacher.float()).square().mean().item()
    def plane(t):
        return t[0,0,:,:,zslice].detach().float().cpu().numpy().T
    target_distance = distance_transform_edt(~gt)
    np.savez_compressed(OUT/'figure_data.npz',
        image=image[:,:,zslice].T, clean=gt[:,:,zslice].T, second=other[:,:,zslice].T,
        corrupted=plane(corrupted), reconstruction=plane(output['logits'].sigmoid()),
        dense_distance=plane(output['distance']), target_distance=(target_distance[:,:,zslice].T/16).clip(0,1),
        predicted_sheet=plane(pred_mask), gt_union=(instances[:,:,zslice].T>0),
        teacher=teacher.float().cpu().numpy(), predicted=pred.float().cpu().numpy(),
        ae_code=output['latent'].float().cpu().numpy(), other_code=other_raw.float().cpu().numpy(),
        prompts=prompts, points=points, identity=identity, cluster_id=cluster_id, distance=distance)
    record = dict(case='sample_00860', native_z=zslice, ae_sheet_ids=[9,8], corruption_seed=42,
        corruption_config=corrupt_cfg.__dict__, cosine_raw_ae_codes=cosine, prompt_code_mse=code_mse,
        clustering_replay_identity_exact=True, seed_count=len(points), cluster_count=len(clusters),
        retained_cluster_count=len(kept), retained_seeds=int((identity>0).sum()), discarded_seeds=int((identity==0).sum()),
        display='Native XYZ data; matched z=160 planes transposed for XY display. Corruption uses the original AE training recipe.',
        selection='Same previously published example; GT band 9 and adjacent band 8, without outcome-based selection.',
        purpose='Illustration of existing model behavior; not a new benchmark or ablation.')
    (OUT/'capture.json').write_text(json.dumps(record, indent=2)+'\n')
    print(json.dumps(record, indent=2), flush=True)

if __name__ == '__main__':
    main()
