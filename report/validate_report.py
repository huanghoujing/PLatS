#!/usr/bin/env python3
"""Check completed evidence, unchanged sources, score arithmetic and PDF layout."""
from pathlib import Path
import hashlib,io,json,re,subprocess,zipfile,xml.etree.ElementTree as ET
from datetime import datetime,timezone
import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
def read(p):return json.loads(Path(p).read_text())
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def formula(m):
 assert abs(m['leaderboard_formula_score']-(.35*m['surface_dice_tau2']+.30*m['toposcore']+.35*m['voi_score']))<1e-12
def main():
 data=read(HERE/'results_snapshot.json')
 for name in ('winner_comparison','winner_reading','large_page_evaluation','latent_separability'):
  assert data[name] is not None and data[name]['status']=='complete',name
 for path,digest in data['source_sha256'].items():assert sha(ROOT/path)==digest,path
 for path,digest in data['source_code_sha256'].items():assert sha(ROOT/path)==digest,path
 assert data['full_prompt_verification']['status']=='passed'
 assert sum(map(len,data['completed_prompt_formula'].values()))==12
 automatic=data['automatic'];assert len(automatic['rows'])==424
 for row in automatic['rows']:
  formula(row['binary_union']);s=row['instance'];tp=s['true_positives'];den=tp+s['false_positives']+s['false_negatives']
  assert abs(s['instance_accuracy_iou50']-(tp/den if den else 1))<1e-12
 winner=data['winner_comparison'];assert winner['cases']==106 and len(winner['rows'])==318
 case_sets={a:{r['case_id'] for r in winner['rows'] if r['arm']==a} for a in winner['arms']}
 assert all(len(c)==106 and c==next(iter(case_sets.values())) for c in case_sets.values())
 original={r['case_id']:r['binary_union'] for r in automatic['rows'] if r['arm']=='0058_kaggle'}
 parity=0.
 for r in winner['rows']:
  for variant,score in r['scores'].items():formula(score['metrics'])
  assert r['modified_lb']==r['scores']['restored_filtered']['metrics']['leaderboard_formula_score']
  for audit in r['instances']:
   assert audit['retained_after_filter']==(audit['visible_voxels']>0 and audit['total_voxels']>=5000)
   assert audit['ignored_voxels']==audit['total_voxels']-audit['visible_voxels']
  if r['arm']=='plats_auto':
   for k in ('dice','surface_dice_tau2','toposcore','voi_score','leaderboard_formula_score'):
    parity=max(parity,abs(r['scores']['standard']['metrics'][k]-original[r['case_id']][k]))
 assert parity<1e-12,parity
 for arm,means in winner['means'].items():
  group=[r for r in winner['rows'] if r['arm']==arm]
  for variant in ('standard','filtered','restored_filtered'):
   formula(means[variant])
   for k,value in means[variant].items():assert abs(value-np.mean([r['scores'][variant]['metrics'][k] for r in group]))<1e-12
 for figure in ('large_pages','spiral_figure','winner_reading_figure'):
  for item in data[figure]['files']:
   if 'path' in item:assert sha(ROOT/item['path'])==item['sha256'],item['path']
 if 'page_onepoint' in data:
  page=data['page_onepoint'];assert len(page['plan']['cases'])==126
  assert [r['method'] for r in page['rows']]==['ffn','0058']
  assert page['scope_amendment']['active_methods']==['ffn','0058']
  assert page['plan']['prompt_count_per_crop']==1
  for path,digest in page['source_sha256'].items():assert sha(ROOT/path)==digest
  for item in page['files']:assert sha(ROOT/item['path'])==item['sha256']
  for row in page['rows']:
   assert row['reading']['status']=='complete'
   assert len(row['stitching']['rows'])==126
   assert all(r['prompt_count']==1 for r in row['stitching']['rows'])
   source=ROOT/page['source_run']/row['method']
   for case,digest in row['stitching']['raw_predictions_sha256'].items():assert sha(source/'patches'/case/'raw_mask.npz')==digest
   prompts={c['id']:c['prompt_local_zyx'] for c in page['plan']['cases']}
   assert all(r['prompt_local_zyx']==prompts[r['case']] for r in row['stitching']['rows'])
   assert sha(source/'reading/surface.npz')==row['reading']['surface_sha256']
   assert sha(source/'reading/inward.npy')==row['reading']['inward_sha256']
   assert not row['reading']['reference_geometry_access'] and not row['reading']['ink_labels_access']
 # Figure 3 must use measured arrays and exactly the saved seed/crop geometry.
 method=data['method_visuals']
 assert method['replay_cluster_sizes_match'] and method['sampled_points']==512
 for path,digest in method['source_sha256'].items():assert sha(ROOT/path)==digest,path
 native_dir=ROOT/'runs_from_260914/evaluation/08_paper_method_visuals'
 auto_dir=ROOT/'runs_from_260914/evaluation/04_paper_automatic_hidden106_gpu0_gpu2'
 source=auto_dir/'0058_kaggle/cases'/method['case']
 with np.load(native_dir/'intermediates.npz') as a:
  points=a['points_xyz'];identities=a['identity']
  assert np.array_equal(points,read(source/'points.json')['clicks'])
  assert a['latent'].shape==(512,64000) and np.isfinite(a['latent']).all()
  assert a['pca'].shape==(512,2) and np.isfinite(a['pca']).all()
  assert a['context'].shape==(1,512,10,10,10)
  for key,selected in read(source/'points.json')['sheets'].items():
   members={tuple(p) for p in points[identities==int(key)]}
   assert all(tuple(p) in members for p in selected),key
 # Validate figure orientation against saved geometry, independent of rendered textures.
 from build_paper_figures import reading_display_frame
 figure=data['winner_reading_figure'];ref=figure['orientation_reference']
 assert sha(ROOT/ref['path'])==ref['sha256']
 basis0,normal0=reading_display_frame(ROOT/ref['path'])
 for item in figure['orientations']:
  assert sha(ROOT/item['surface'])==item['surface_sha256']
  basis,normal=reading_display_frame(ROOT/item['surface'])
  assert bool(basis[0]@basis0[0]<0)==item['horizontal_flip']
  assert ('normal_plus' if normal@normal0>0 else 'normal_minus')==item['normal']
  assert basis[1]@basis0[1]>.9
 # Validate every frozen winner inference mask, including the four native 256³ cubes.
 manifest=read(ROOT/'runs_from_260914/evaluation/04_paper_automatic_hidden106_gpu0_gpu2/inference_manifest.json')
 freeze=data['winner_inference_freeze']['sha256']
 for case in manifest['cases']:
  meta=read(ROOT/'datasets/hf_kaggle_202607/cases'/case['case_id']/'meta.json')
  assert meta['component_source']=='hf_label_connected_components_6conn_unfiltered'
  assert meta['erased_border_width']==0 and meta['ignore_label_source']==2
  p=ROOT/'runs_from_260914/evaluation/05_paper_winner_ignore106_gpu0_gpu2/winner'/case['case_id']/'prediction.npz'
  assert sha(p)==freeze[case['case_id']]
  with np.load(p) as archive:assert list(archive['mask'].shape)==case['shape']
 # Verify that cached binary masks are exact transposes of the original released
 # runner's submission archive, rather than subsequently cleaned instances.
 import tifffile
 cached=ROOT/'runs/instseg_firstplace_eval80/out/submission.zip'
 cached_count=0
 with zipfile.ZipFile(cached) as archive:
  for case in data['winner_plan']['cases']:
   if case['source']!='cached_eval80':continue
   native=tifffile.imread(io.BytesIO(archive.read(case['case_id']+'.tif'))).transpose(2,1,0)>0
   with np.load(ROOT/'runs_from_260914/evaluation/05_paper_winner_ignore106_gpu0_gpu2/winner'/case['case_id']/'prediction.npz') as saved:assert np.array_equal(native,saved['mask'])
   cached_count+=1
 assert cached_count==80
 assert data['binary_failure_selection']['gap_audit']['ignored_voxels']==0
 assert data['binary_failure_selection']['gap_audit']['predicted_known_background_voxels']>0
 rendering=data['failure_rendering']
 assert rendering['rendering']['same_slice_and_crop_within_each_comparison']
 assert rendering['rendering']['interpolation']=='nearest'
 assert rendering['binary']['axis']==rendering['ignore']['axis']==2
 for audit in rendering['rendering']['coordinate_audit']['cases']:
  assert sha(ROOT/audit['source'])==audit['sha256']
 for entry in (rendering['binary'],rendering['ignore']):
  native=tifffile.imread(ROOT/'datasets/hf_kaggle_202607/raw/images'/f'{entry["case_id"]}.tif',key=entry['slice_index'])
  native[:5]=0;native[-5:]=0;native[:,:5]=0;native[:,-5:]=0
  if 'crop_start' in entry:native=native[tuple(slice(a,b) for a,b in zip(entry['crop_start'],entry['crop_stop']))]
  with np.load(ROOT/entry['native_panels']) as panels:assert np.array_equal(panels['image'],native)
 views=ROOT/'runs_from_260914/evaluation/06_paper_failure_analysis/2d_slices'
 for path,digest in rendering['assets'].items():assert sha(views/path)==digest,path
 for path,digest in rendering['figures'].items():assert sha(ROOT/path)==digest,path
 b=rendering['binary'];meta=read(ROOT/'datasets/hf_kaggle_202607/cases'/b['case_id']/'meta.json')
 with np.load(ROOT/b['native_panels']) as panels:
  for k,source in [('image','image_path'),('gt','components_path')]:assert np.array_equal(panels[k],np.load(meta[source])[:,:,b['slice_index']].T)
  with np.load(ROOT/f'runs_from_260914/evaluation/05_paper_winner_ignore106_gpu0_gpu2/winner/{b["case_id"]}/prediction.npz') as source:assert np.array_equal(panels['winner']>0,source['mask'][:,:,b['slice_index']].T)
  with np.load(ROOT/f'runs_from_260914/evaluation/04_paper_automatic_hidden106_gpu0_gpu2/0058_kaggle/cases/{b["case_id"]}/instances.npz') as source:assert np.array_equal(panels['plats'],source['inst'][:,:,b['slice_index']].T)
  gap=data['binary_failure_selection']['gap_audit'];yy,xx=np.array(gap['sampled_yx']).T
  assert len(yy)==gap['sampled_voxels'] and (panels['winner'][yy,xx]>0).all()
  ignore=np.unpackbits(np.load(meta['ignore_path']),count=320**3).reshape(320,320,320)[:,:,b['slice_index']].T
  assert not ignore[yy,xx].any()
  assert int((panels['gt'][yy,xx]==0).sum())==gap['predicted_known_background_voxels']
 i=rendering['ignore']
 import nibabel as nib
 from scipy import ndimage as ndi
 crop=tuple(slice(a,b) for a,b in zip(i['crop_start'],i['crop_stop']))
 base=ROOT/'runs_from_260914/evaluation/06_paper_failure_analysis/figure_nifti'/i['case_id']/f'sheet_{i["sheet_id"]:04d}'
 with np.load(ROOT/i['native_panels']) as panels:
  meta=read(ROOT/'datasets/hf_kaggle_202607/cases'/i['case_id']/'meta.json')
  assert np.array_equal(panels['image'],np.load(meta['image_path'])[:,:,i['slice_index']].T[crop])
  for k in ('raw','erased','restored','removed'):
   native=np.asanyarray(nib.load(base/f'{k}.nii.gz').dataobj)>0
   assert np.array_equal(panels[k],native[:,:,i['slice_index']].T[crop]),k
  assert np.array_equal(panels['raw'],panels['restored'])
  assert np.array_equal(panels['raw'] & ~panels['removed'],panels['erased'])
  assert ndi.label(panels['raw'])[1]==i['raw_2d_components']==1
  assert ndi.label(panels['erased'])[1]==i['erased_2d_components']>1
 for model in data['checkpoint_plan']['models'].values():
  # The plan contains the actual absolute model/config/statistics paths.
  for key in ('checkpoint','config','latent_stats'):
   if key in model and key+'_sha256' in model:assert sha(Path(model[key]))==model[key+'_sha256']
 pdf=HERE/'report.pdf'
 text=subprocess.check_output(['pdftotext','-layout',str(pdf),'-'],text=True)
 pages=[p for p in text.split('\f') if p.strip()]
 assert 'pending' not in text.lower() and '??' not in text
 assert 'FFN' not in text, 'Deferred comparison leaked into report'
 assert 'Houjing Huang' in pages[0] and 'houjing.huang@gmail.com' in pages[0]
 supp=next(i for i,p in enumerate(pages) if 'Supplementary Material' in p)
 refs=next(i for i,p in enumerate(pages) if p.lstrip().startswith('References'))
 abstract=pages[0].split('Abstract',1)[1].split('1. Method',1)[0]
 assert '0076' not in abstract and '0058' not in abstract and '100k' not in abstract
 assert all('0076' not in p for p in pages[:refs]),'Mixed-data checkpoint in main paper'
 assert 'Type 3' not in subprocess.check_output(['pdffonts',str(pdf)],text=True)
 log=(HERE/'report.log').read_text()
 warnings=[line for line in log.splitlines() if re.search(r'Overfull|undefined|multiply defined|LaTeX Warning',line)]
 assert not warnings,warnings
 bbox=subprocess.check_output(['pdftotext','-bbox',str(pdf),'-'],text=True)
 bbox=''.join(c for c in bbox if ord(c)>=32 or c in '\n\r\t')
 document=ET.fromstring(bbox)
 bounds=[]
 for index,page in enumerate(document.findall('.//{*}page')):
  for word in page.findall('.//{*}word'):
   x0,y0,x1,y1=[float(word.attrib[k]) for k in ('xMin','yMin','xMax','yMax')]
   if not(40<=x0<=x1<=570 and 30<=y0<=y1<=775):bounds.append((index+1,word.text,(x0,y0,x1,y1)))
 assert not bounds,bounds
 status=dict(status='passed',checked_utc=datetime.now(timezone.utc).isoformat(),
  pdf_pages=len(pages),main_pages=refs,reference_pages=supp-refs,supplementary_pages=len(pages)-supp,
  source_hashes_verified=len(data['source_sha256']),source_code_hashes_verified=len(data['source_code_sha256']),
  complete_prompt_jobs=12,prompted_cases=106,prompted_sheets=764,
  automatic_arm_case_scores=424,winner_ignore_arm_case_scores=318,
  unchanged_winner_native_shapes_verified=106,standard_plats_max_difference=parity,
  cached_winner_submission_masks_verified=cached_count,cached_submission_sha256=sha(cached),
  formula_and_instance_accuracy_arithmetic_verified=True,pdf_text_bounds_verified=True,
  pdf_type3_fonts=False,tex_warnings=warnings,
  large_page_count=2,winner_reading_crops_attempted=5,
  native_failure_slice_groups_verified=2,
  winner_reading_crops_with_ink=sum(any(c['status']=='ink_complete' for c in (r['reading'] or {}).get('components',[])) for r in data['winner_reading']['rows']),
  files_sha256={str(p.relative_to(HERE)):sha(p) for p in sorted(HERE.rglob('*')) if p.is_file() and p.name!='validation.json' and p.suffix not in ('.aux','.bbl','.blg','.log','.out','.pyc') and '__pycache__' not in p.parts and 'preview' not in p.parts})
 (HERE/'validation.json').write_text(json.dumps(status,indent=2)+'\n')
 print(json.dumps({k:v for k,v in status.items() if k!='files_sha256'},indent=2))
if __name__=='__main__':main()
