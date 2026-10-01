#!/usr/bin/env python3
"""Build a portable, local PLatS reproduction bundle without publishing it."""
import ast,hashlib,importlib.metadata,json,shutil,subprocess,sys
from pathlib import Path
import numpy as np
import torch,yaml
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'submission/PLatS'
SOURCE=ROOT/'src'
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def write(p,d):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(d,indent=2)+'\n')
def copy(source,target):target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
def module_file(name):
 p=SOURCE.joinpath(*name.split('.'))
 return p.with_suffix('.py') if p.with_suffix('.py').exists() else p/'__init__.py'
def source_closure():
 todo=['vesuvius_p2sd.research.auto_instance_seg','vesuvius_p2sd.train.train_ae',
       'vesuvius_p2sd.train.train_p2sd','vesuvius_p2sd.train.train_binary_seg',
       'vesuvius_p2sd.eval.sheet_report','vesuvius_p2sd.data.prepare_t6',
       'vesuvius_p2sd.ink.reading','vesuvius_p2sd.ink.model']
 seen=set()
 while todo:
  name=todo.pop();p=module_file(name)
  if name in seen or not p.is_file():continue
  seen.add(name);copy(p,OUT/'src'/p.relative_to(SOURCE))
  parts=name.split('.')
  for n in range(1,len(parts)):todo.append('.'.join(parts[:n]))
  for node in ast.walk(ast.parse(p.read_text())):
   if isinstance(node,ast.Import):todo.extend(x.name for x in node.names if x.name.startswith('vesuvius_p2sd'))
   elif isinstance(node,ast.ImportFrom):
    base=node.module or ''
    if node.level:
     parent=parts if p.name=='__init__.py' else parts[:-1]
     base='.'.join(parent[:len(parent)-node.level+1]+([base] if base else []))
    if base.startswith('vesuvius_p2sd'):
     todo.append(base);todo.extend(base+'.'+x.name for x in node.names)
 return sorted(seen)
def main():
 OUT.mkdir(parents=True,exist_ok=True)
 modules=source_closure();print('Source modules',len(modules),flush=True)
 copy(ROOT/'scripts/submission_cli.py',OUT/'plats.py')
 runs={'ae':'0032_ae_t6_denoise_repel_e150_r1','p2sd':'0058_p2sd_joint_dense_ps160_warm0033_e150',
       'foreground':'0022_binseg_ps320_ctxattn4_rope_frozen0021_ft200'}
 weights={}
 for role,name in runs.items():
  source=ROOT/'runs'/name;original=yaml.safe_load((source/'resolved_config.yaml').read_text())
  copy(source/'resolved_config.yaml',OUT/'provenance'/f'{role}_training_original.yaml')
  cfg={k:v for k,v in original.items() if k in {'model','p2sd','binary_seg','target_ae'}}
  if role=='p2sd':
   cfg['target_ae'].update(config_path='configs/ae.yaml',checkpoint_path='weights/ae.pt')
   stats=Path(original['p2sd']['loss']['latent_normalization']['stats_path'])
   copy(ROOT/stats,OUT/'configs/latent_stats.json')
   cfg['p2sd']['loss']['latent_normalization']['stats_path']='configs/latent_stats.json'
  # Warm-start paths belong only to the archived training configurations.
  def clean(d):
   if isinstance(d,dict):
    for k in list(d):
     if k in ('init_checkpoint','resume_checkpoint_path'):d.pop(k)
     else:clean(d[k])
   elif isinstance(d,list):
    for v in d:clean(v)
  clean(cfg)
  target=OUT/'configs'/f'{role}.yaml';target.parent.mkdir(exist_ok=True);target.write_text(yaml.safe_dump(cfg,sort_keys=False))
  payload=torch.load(source/'last.pt',map_location='cpu',weights_only=False)
  state={k:v.detach().cpu().contiguous() for k,v in payload['model'].items()}
  target=OUT/'weights'/f'{role}.pt';target.parent.mkdir(exist_ok=True)
  torch.save({'model':state},target)
  saved=torch.load(target,map_location='cpu',weights_only=True)['model']
  assert saved.keys()==state.keys() and all(torch.equal(saved[k],v) for k,v in state.items())
  weights[role]=dict(run=name,original_sha256=sha(source/'last.pt'),inference_sha256=sha(target),
    tensors=len(state),step=payload.get('step'),tensor_equality_verified=True)
  print(role,target.stat().st_size,flush=True)
 auto=ROOT/'runs_from_260914/evaluation/04_paper_automatic_hidden106_gpu0_gpu2'
 sys.path.insert(0,str(ROOT/'scripts'))
 from paper_automatic_hidden106 import SETTINGS
 write(OUT/'configs/automatic.json',SETTINGS)
 case='sample_00860';demo=OUT/'examples'/case;demo.mkdir(parents=True,exist_ok=True)
 meta=json.loads((ROOT/'datasets/hf_kaggle_202607/cases'/case/'meta.json').read_text())
 copy(Path(meta['image_path']),demo/'image.npy')
 components=np.load(meta['components_path']);ignore=np.unpackbits(np.load(meta['ignore_path']),count=components.size).reshape(components.shape).astype(bool)
 np.save(demo/'ignore.npy',ignore);np.save(demo/'gt_instances.npy',components)
 evalroot=ROOT/'runs_from_260914/evaluation/03_paper_hidden106_three_checkpoints_gpu0_gpu2'
 manifest=json.loads((evalroot/'manifests/points_08.json').read_text())
 case_plan=next(c for c in manifest['cases'] if c['case_id']==case);sheet=case_plan['components'][0]
 identity=sheet['component_id'];np.save(demo/'gt_sheet.npy',(components==identity).astype(np.uint8))
 write(demo/'points.json',dict(points=sheet['prompt_sets'][0]['points_zyx'],sheet_id=identity,
    coordinate_order='native .npy axis order (X,Y,Z for this example)',source='Frozen 8-point evaluation manifest; nested prefixes 1/2/4/8'))
 with np.load(auto/'foreground'/case/'mask.npz') as a:np.save(demo/'foreground_expected.npy',a['mask'])
 with np.load(auto/'0058_kaggle/cases'/case/'instances.npz') as a:np.save(demo/'instances_expected.npy',a['inst'])
 prompted=evalroot/'0058_kaggle/points_08/cases'/case/'sheets'/f'sheet_{identity:04d}'/'p00'
 import nibabel as nib
 np.save(demo/'prompt_expected.npy',np.asanyarray(nib.load(prompted/'p2sd_decoded_sheet.nii.gz').dataobj).astype(np.uint8))
 copy(prompted/'metrics.json',demo/'prompt_expected_metrics.json')
 write(demo/'provenance.json',dict(case=case,selection='Same crop as the report method/failure figures; first eligible sheet in frozen evaluation order',
    source=meta['source'],original_metadata=meta,image_sha256=sha(demo/'image.npy'),
    preprocessing='Original TIFF [Z,Y,X] transposed to [X,Y,Z]; uint8 CT; five-voxel image border zeroed. GT borders retained.',
    reference_use='GT supplies prompt coordinates and scoring targets; automatic inference does not access GT or ignore'))
 for role,source in [('betti_compact',ROOT/'.external/Betti-Matching-3D-compact-exact'),
                     ('topometrics/external/Betti-Matching-3D',ROOT/'.external/Betti-Matching-3D')]:
  dest=OUT/'third_party'/role
  shutil.copytree(source/'src',dest/'src',dirs_exist_ok=True)
  for f in ['CMakeLists.txt','LICENSE']:copy(source/f,dest/f)
 source=ROOT/'.external/kaggle_vesuvius_metric_resources/topological-metrics-kaggle';dest=OUT/'third_party/topometrics'
 shutil.copytree(source/'src',dest/'src',dirs_exist_ok=True,ignore=shutil.ignore_patterns('__pycache__','*.egg-info'))
 for f in ['pyproject.toml','LICENSE','README.md']:copy(source/f,dest/f)
 for f in ['betti_matching_compact_exact.patch','betti_matching_binary_exact.patch']:
  if (ROOT/'research/reference_patches'/f).exists():copy(ROOT/'research/reference_patches'/f,OUT/'provenance'/f)
 packages=['numpy','scipy','connected-components-3d','nibabel','PyYAML','matplotlib','Pillow','tifffile','zarr','surface-distance','pybind11','cmake','scikit-image']
 (OUT/'requirements.txt').write_text('\n'.join(n+'=='+importlib.metadata.version(n) for n in packages)+'\n')
 write(OUT/'provenance/environment.json',dict(python=sys.version,torch=torch.__version__,cuda=torch.version.cuda,
    source_revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()))
 write(OUT/'provenance/weights.json',weights)
 write(OUT/'provenance/source_modules.json',dict(modules=modules,files_sha256={str(p.relative_to(OUT)):sha(p) for p in (OUT/'src').rglob('*.py')}))
 copy(__file__,OUT/'provenance/build_progress_prize_bundle.py')
 print(OUT,flush=True)
if __name__=='__main__':main()
