#!/usr/bin/env python3
"""Portable PLatS inference and exact scoring entry point (copied into the bundle)."""
import argparse, json, os, sys, time, hashlib
from pathlib import Path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'third_party/topometrics/src'))
sys.path.insert(0, str(ROOT / 'third_party/betti_compact/build'))

def read(p):
    return json.loads(Path(p).read_text())

def write(p, d):
    Path(p).write_text(json.dumps(d, indent=2) + '\n')

def sha(p):
    with Path(p).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()

def load_cfg(name):
    import yaml
    c = yaml.safe_load((ROOT / 'configs' / f'{name}.yaml').read_text())
    if name == 'p2sd':
        c['target_ae']['config_path'] = str(ROOT / 'configs/ae.yaml')
        c['target_ae']['checkpoint_path'] = str(ROOT / 'weights/ae.pt')
        c['p2sd']['loss']['latent_normalization']['stats_path'] = str(ROOT / 'configs/latent_stats.json')
    return c

def stack(device):
    from vesuvius_p2sd.models.ae import build_sheet_ae
    from vesuvius_p2sd.models.p2sd import build_p2sd
    from vesuvius_p2sd.train.common import load_model_state
    from vesuvius_p2sd.train.train_p2sd import build_static_latent_codec
    c = load_cfg('p2sd')
    ae = build_sheet_ae(load_cfg('ae')).to(device).eval()
    load_model_state(ae, ROOT / 'weights/ae.pt')
    model = build_p2sd(c, latent_channels=ae.latent_channels).to(device).eval()
    load_model_state(model, ROOT / 'weights/p2sd.pt')
    codec = build_static_latent_codec(c['p2sd']['loss']['latent_normalization'],
        latent_channels=ae.latent_channels,
        device=device)
    return (model, ae, codec)

def volume(path):
    import numpy as np
    a = np.load(path, allow_pickle=False)
    if a.ndim != 3 or any((n > 320 or n < 32 for n in a.shape)):
        raise ValueError('Expected a 3D crop, each side 32..320 voxels')
    if a.dtype != np.uint8:
        raise ValueError('Expected uint8 CT as in the training/evaluation preprocessing')
    offset = np.array([(320 - n) // 2 for n in a.shape])
    bounds = tuple((slice(int(o), int(o + n)) for o, n in zip(offset, a.shape)))
    canvas = np.zeros((320,) * 3, np.uint8)
    canvas[bounds] = a
    valid = np.zeros(canvas.shape, bool)
    valid[bounds] = True
    return (a, canvas, valid, bounds, offset)

def predict(args):
    import numpy as np, torch
    from vesuvius_p2sd.research import auto_instance_seg as auto
    from vesuvius_p2sd.eval.sheet_report import save_nifti
    start = time.time()
    torch.set_num_threads(args.threads)
    auto._DECODE_BATCH = 1
    device = torch.device(args.device)
    dtype = torch.bfloat16 if device.type == 'cuda' else torch.float32
    if device.type == 'cuda' and (not torch.cuda.is_bf16_supported()):
        raise ValueError('The released comparison uses BF16; select a BF16-capable GPU or CPU')
    raw, canvas, valid, bounds, offset = volume(args.image)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    model, ae, codec = stack(device)
    tensor = torch.from_numpy(canvas[None, None].astype(np.float32)).to(device)
    with torch.inference_mode():
        if args.command == 'prompt':
            points = np.asarray(read(args.points)['points'], np.float32)[:args.count]
            if points.ndim != 2 or points.shape[1] != 3 or len(points) == 0 or (not np.isfinite(points).all()):
                raise ValueError('points must be nonempty finite N x 3 coordinates')
            if not ((points >= 0) & (points < np.array(raw.shape))).all():
                raise ValueError('Prompt outside native crop')
            with torch.autocast(device.type, dtype=dtype, enabled=device.type == 'cuda'):
                _, tokens, coords, context = model.encode_image_context_from_image(tensor)
                probability = auto._decode_prompt_sets(
                    model,
                    ae,
                    codec,
                    tokens,
                    coords,
                    context,
                    [points + offset],
                    canvas.shape,
                    device,
                    dtype,
                )[0][bounds]
            prediction = (probability >= 0.5).astype(np.uint8)
            np.save(out / 'probability.npy', probability)
            prompt_map = np.zeros(raw.shape, np.uint8)
            for p in np.rint(points).astype(int):
                prompt_map[tuple(np.minimum(p, np.array(raw.shape) - 1))] = 1
            save_nifti(out / 'prompt_points.nii.gz', prompt_map)
            write(out / 'points.json', dict(points=points.tolist(), coordinate_order='input array axes'))
        else:
            if args.foreground:
                fg = np.load(args.foreground) > 0
                if fg.shape != raw.shape:
                    raise ValueError('Foreground shape must match CT')
                mask = np.zeros(canvas.shape, bool)
                mask[bounds] = fg
            else:
                from vesuvius_p2sd.models.binary_seg import build_binary_seg_model
                from vesuvius_p2sd.train.common import load_model_state
                proposer = build_binary_seg_model(load_cfg('foreground')).to(device).eval()
                load_model_state(proposer, ROOT / 'weights/foreground.pt')
                with torch.autocast(device.type, dtype=dtype, enabled=device.type == 'cuda'):
                    mask = (proposer(tensor).float().sigmoid()[0, 0] >= 0.6).cpu().numpy()
                del proposer
            settings = read(ROOT / 'configs/automatic.json')
            result = auto.run_case_cluster(
                binseg_model=None,
                p2sd_model=model,
                target_ae=ae,
                latent_codec=codec,
                image=canvas,
                valid=valid,
                device=device,
                dtype=dtype,
                external_mask=mask,
                **settings,
            )
            prediction = result['instance_ids'][bounds].astype(np.int16)
            np.save(out / 'foreground.npy', mask[bounds])
            save_nifti(out / 'foreground.nii.gz', mask[bounds].astype(np.uint8))
            write(out / 'clustering.json', result['seed_log'])
            points = result['points']
            points['clicks'] = (np.array(points['clicks']).reshape(-1, 3) - offset).tolist()
            points['sheets'] = {
                k: (np.array(v).reshape(-1, 3) - offset).tolist()
                for k, v in points['sheets'].items()
            }
            write(out / 'points.json', points)
            prompt_map = np.zeros(raw.shape, np.uint8)
            for p in points['clicks']:
                prompt_map[tuple(p)] = 1
            save_nifti(out / 'prompt_points.nii.gz', prompt_map)
    np.save(out / 'prediction.npy', prediction)
    save_nifti(out / 'prediction.nii.gz', prediction)
    save_nifti(out / 'image.nii.gz', raw)
    write(
        out / 'run.json',
        dict(
            command=args.command,
            seconds=time.time() - start,
            device=str(device),
            dtype=str(dtype),
            input_sha256=sha(args.image),
            prediction_sha256=sha(out / 'prediction.npy'),
            input_shape=list(raw.shape),
            canvas_offset=offset.tolist(),
            source_module=auto.__file__,
            code_root=str(ROOT),
            weights_sha256={
                p.name: sha(p) for p in sorted((ROOT / 'weights').glob('*.pt'))
            },
        ),
    )
    print(out.resolve())

def score(args):
    import numpy as np
    from vesuvius_p2sd.eval.sheet_report import score_masks, rectangular_border_ignore, save_nifti
    pred = np.load(args.prediction) > 0
    gt = np.load(args.gt) > 0
    ignore = np.load(args.ignore).astype(bool) if args.ignore else np.zeros(gt.shape, bool)
    if args.policy == 'annotated-box':
        ignore = rectangular_border_ignore(ignore)
    result = score_masks(pred, gt, ignore, topology_backend='compact_exact', topology_tile_batch_size=1)
    result['ignore_policy'] = args.policy
    write(args.output, result)
    save_nifti(Path(args.output).parent / 'gt_sheet.nii.gz', gt.astype(np.uint8))
    print(json.dumps(result, indent=2))

def verify(args):
    manifest = read(ROOT / 'manifest.json')
    errors = []
    for name, digest in manifest['files_sha256'].items():
        p = ROOT / name
        if not p.is_file() or sha(p) != digest:
            errors.append(name)
    if errors:
        raise RuntimeError('Missing or changed files: ' + str(errors))
    print(f"Verified {len(manifest['files_sha256'])} files")

def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    for name in ['prompt', 'automatic']:
        q = sub.add_parser(name)
        q.add_argument('--image', required=True)
        q.add_argument('--output', required=True)
        q.add_argument('--device', default='cuda:0')
        q.add_argument('--threads', type=int, default=4)
        if name == 'prompt':
            q.add_argument('--points', required=True)
            q.add_argument('--count', type=int, choices=[1, 2, 4, 8], default=8)
        else:
            q.add_argument('--foreground',
                help='Optional precomputed binary .npy; otherwise run the included proposer')
    q = sub.add_parser('score')
    q.add_argument('--prediction', required=True)
    q.add_argument('--gt', required=True)
    q.add_argument('--ignore')
    q.add_argument('--policy', choices=['source', 'annotated-box'], default='source')
    q.add_argument('--output', required=True)
    sub.add_parser('verify')
    args = p.parse_args()
    {'prompt': predict, 'automatic': predict, 'score': score, 'verify': verify}[args.command](args)
if __name__ == '__main__':
    main()
