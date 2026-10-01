#!/usr/bin/env python3
"""Publishable prediction-only exports of all completed released-test evaluations."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import hashlib,json,os,shutil,subprocess,time
import nibabel as nib
ROOT=Path(__file__).resolve().parents[1]
P=ROOT/'runs_from_260914/evaluation/03_paper_hidden106_three_checkpoints_gpu0_gpu2'
A=ROOT/'runs_from_260914/evaluation/04_paper_automatic_hidden106_gpu0_gpu2'
OUT=ROOT/'runs_from_260914/releases/01_hidden106_predictions'
MODELS={'0058_kaggle':'0058','original_0076':'0076','0076_plus_100k':'0076_plus_100k'}
BASE='https://github.com/huanghoujing/PLatS/releases/download/v0.1.0-progress-prize/'
def read(p):return json.loads(Path(p).read_text())
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def write(p,d):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(d,indent=2)+'\n')
def link(p,q):
 q.parent.mkdir(parents=True,exist_ok=True)
 if not q.exists():os.link(p,q)
 assert sha(p)==sha(q)
def archive(task):
 kind,model,count=task;label=MODELS[model]
 name=f'PLatS_hidden106_{label}_'+(f'prompted_{count:02d}point' if kind=='prompted' else 'automatic')
 stage=OUT/'staging'/name;stage.mkdir(parents=True,exist_ok=True)
 source=P/model/f'points_{count:02d}' if kind=='prompted' else A/model
 manifest=read(P/'manifests/points_08.json');cases=manifest['cases'];assert len(cases)==106
 original={};sheets=0;volumes=0
 def include(p,rel):
  original[str(rel)]=dict(path=str(p.relative_to(ROOT)),sha256=sha(p))
  link(p,stage/rel)
 def volume(p,rel,shape):
  nonlocal volumes
  assert not p.is_symlink() and p.is_file(),p
  assert list(nib.load(p).shape)==shape,(str(p),shape)
  include(p,rel);volumes+=1
 if kind=='prompted':
  summary=read(source/'summary.json');assert summary['status']=='complete' and summary['sheet_prompt_count']==764
  for filename in ['summary.json','protocol.json','case_metrics.jsonl','sheet_metrics.jsonl']:
   include(source/filename,Path(filename))
  include(P/'manifests'/f'points_{count:02d}.json',Path('prompt_manifest.source.json'))
  checkpoint=read(P/'plan.json')['models'][model]
 else:
  include(A/'results.json',Path('all_checkpoint_metrics.json'))
  include(A/'plan.json',Path('automatic_plan.source.json'))
  checkpoint=read(A/'plan.json')['models'][model]
 for case in cases:
  cid=case['case_id'];shape=case['source_shape'];case_dir=Path('cases')/cid
  meta=read(ROOT/'datasets/hf_kaggle_202607/cases'/cid/'meta.json')
  assert meta['orientation']=='tif.transpose(2,1,0) = project-internal (t6) order'
  write(stage/case_dir/'coordinates.json',dict(case_id=cid,array_axes='XYZ',shape=shape,
   source_tiff_conversion='Original TIFF [Z,Y,X] transposed with (2,1,0). NIFTIs use identity voxel affine.',
   legacy_names='The source manifests use *_zyx names for the same numerical [X,Y,Z] indices; do not transpose these points.',
   data_source=meta['source'],image_border_zero=meta['image_border_zero']))
  d=source/'cases'/cid
  if kind=='prompted':
   assert (d/'complete.json').is_file()
   for component in case['components']:
    sid=component['component_id'];rel=case_dir/'sheets'/f'sheet_{sid:04d}'/'p00';t=d/'sheets'/f'sheet_{sid:04d}'/'p00'
    for filename in ['p2sd_decoded_sheet.nii.gz','prompt_points.nii.gz']:volume(t/filename,rel/filename,shape)
    include(t/'metrics.json',rel/'metrics.json');include(t/'tuple.json',rel/'tuple.source.json')
    info=read(t/'tuple.json');assert len(info['points_zyx'])==count
    assert all(0<=v<shape[axis] for pt in info['points_zyx'] for axis,v in enumerate(pt))
    info['array_axes']='XYZ';info['points_xyz']=info['points_zyx']
    info['coordinate_note']='points_xyz and legacy points_zyx contain identical input-array coordinates. Original ZYX label was incorrect.'
    info['files']=['prompt_points.nii.gz','p2sd_decoded_sheet.nii.gz','metrics.json']
    info['not_included']=['image.nii.gz','gt_sheet.nii.gz','ignore.nii.gz']
    write(stage/rel/'tuple.json',info);sheets+=1
   rel=case_dir/'unions/p00';t=d/'unions/p00'
   for filename in ['p2sd_union.nii.gz','p2sd_instances.nii.gz']:volume(t/filename,rel/filename,shape)
   include(t/'metrics.json',rel/'metrics.json')
  else:
   info=read(d/'inference.json')
   assert sha(d/'instances.npz')==info['instances_sha256']
   assert sha(d/'pred_instances.nii.gz')==info['prediction_sha256']
   for filename in ['pred_instances.nii.gz','sampled_points.nii.gz']:volume(d/filename,case_dir/filename,shape)
   for filename in ['instances.npz','points.json','seeds.json','inference.json','metrics.json']:include(d/filename,case_dir/filename)
 metadata=dict(kind=kind,checkpoint=label,prompt_count=count,cases=106,prompted_sheets=sheets,
    nifti_volumes=volumes,array_axes='XYZ',checkpoint_provenance=checkpoint,
    source_predictions_unchanged=True,source_files=original)
 if kind=='prompted':assert sheets==764
 (stage/'README.md').write_text(f'''# PLatS full released-test predictions: {label} / {kind}

All 106 released Kaggle test cubes. These are previously inspected, now-public
cases, not a fresh blind challenge submission. NIFTIs are original saved masks;
no new inference, smoothing or topology cleanup is applied during export.

This archive contains {sheets} prompted sheet predictions and {volumes} NIFTIs.
Prompt count: {count if count else 'automatic (512 foreground seeds per case)'}.
Checkpoint provenance and per-file SHA256 hashes are in manifest.json.

Array coordinates are [X,Y,Z], matching TIFF.transpose(2,1,0), not the historical
ZYX variable names. Native dimensions are 320³ (102 cases) or 256³ (4 cases).
Tuple/automatic points are native-array coordinates; NIFTIs have an identity voxel affine.
In prompt_manifest.source.json, points are in the padded 320-cube canvas; subtract
the case canvas_offset for native coordinates (32 on each axis for 256-cubes).
Original metadata is retained as *.source.json. Prompted tuple.json corrects
its axis label and adds points_xyz without changing numerical coordinates.

Prompted evaluation ignores only the exterior annotated-region rectangle;
inward source-ignore voxels count as background. Automatic evaluation uses
original source-ignore erasure. The saved masks themselves have not had ignore
erased. Prompted union and ID volumes are included; overlapping sheet masks
are resolved by the original confidence ownership rule. Automatic IDs include
the frozen clustering/cleanup pipeline, rather than raw independent decodes.

CT, GT and ignore volumes are not duplicated here. Their original data sources
and case IDs are in cases/*/coordinates.json. Obtain those released annotations
for rescoring; the accompanying metrics and fixed points allow inspection of
all reported results. Each archive is independent and extracts into its own folder.
''')
 metadata['files_sha256']={str(f.relative_to(stage)):sha(f) for f in sorted(stage.rglob('*')) if f.is_file() and f.name!='manifest.json'}
 write(stage/'manifest.json',metadata)
 target=OUT/(name+'.tar.gz')
 subprocess.run(['tar','-I','pigz -p 4 -1','-cf',str(target),'-C',str(stage.parent),name],check=True)
 assert target.stat().st_size<1_900_000_000,'Split archive before upload'
 result=dict(name=target.name,bytes=target.stat().st_size,sha256=sha(target),url=BASE+target.name,
             kind=kind,checkpoint=label,prompt_count=count,cases=106,prompted_sheets=sheets,
             files=len(metadata['files_sha256'])+1,source_files_verified=len(original))
 write(OUT/'completed'/(name+'.json'),result)
 print(json.dumps(result),flush=True)
 return result

def main():
 OUT.mkdir(parents=True,exist_ok=True)
 tasks=[('prompted',m,n) for m in MODELS for n in [1,2,4,8]]+[('automatic',m,None) for m in MODELS]
 with ThreadPoolExecutor(max_workers=4) as pool:rows=list(pool.map(archive,tasks))
 assert sum(r['prompted_sheets'] for r in rows)==9168
 index=dict(status='complete',release='https://github.com/huanghoujing/PLatS/releases/tag/v0.1.0-progress-prize',
  cases=106,checkpoints=list(MODELS.values()),prompt_counts=[1,2,4,8],prompted_sheets=9168,
  prompted_case_unions=1272,automatic_cases=318,arrays='XYZ',assets=rows)
 write(OUT/'PLatS_hidden106_index.json',index)
 (OUT/'PLatS_hidden106_SHA256SUMS.txt').write_text(''.join(r['sha256']+'  '+r['name']+'\n' for r in rows))
 lines=['# Full released Kaggle test predictions','',
  'All **106 cases**, **9,168 prompted sheet predictions**, **1,272 prompted unions**, and **318 automatic instance predictions**.',
  'The primary report model is **0058**; **0076** and **0076+100k** are supplementary.','',
  'The cases are the released former hidden test set, with public annotations and prior inspection. This is not a new blind submission.',
  'Archives contain NIFTI predictions, prompt coordinates/maps, per-sheet and per-case scores, checkpoint identities, and file hashes. CT and reference labels are not duplicated.',
  'Coordinates are XYZ (native TIFF transpose 2,1,0). The old ZYX tuple label is corrected in exported tuple.json; masks and point values are unchanged.','',
  '| Checkpoint | Evaluation | Download | Size MiB |','|---|---|---|---:|']
 for r in rows:
  desc=f"{r['prompt_count']} point(s), 764 sheets + 106 unions" if r['kind']=='prompted' else 'Automatic instances, 106 cases'
  lines.append(f"| {r['checkpoint']} | {desc} | [{r['name']}]({r['url']}) | {r['bytes']/2**20:.1f} |")
 lines+=['',f'[Checksums]({BASE}PLatS_hidden106_SHA256SUMS.txt) · [Machine-readable index]({BASE}PLatS_hidden106_index.json)','',
  'Verify downloaded archives with `sha256sum --ignore-missing -c PLatS_hidden106_SHA256SUMS.txt`, then extract with `tar -xzf ARCHIVE.tar.gz`. Each archive has its own folder and internal manifest.']
 (OUT/'PLatS_hidden106_README.md').write_text('\n'.join(lines)+'\n')
 print('COMPLETE',sum(r['bytes'] for r in rows),flush=True)
if __name__=='__main__':main()
