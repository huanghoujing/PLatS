#!/usr/bin/env python3
"""Export the validated report and native page PNGs to a clean new folder."""
from pathlib import Path
import hashlib,json,re,shutil
from datetime import datetime,timezone
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
OUT=ROOT/'runs_from_260914/reports/11_plats_submission'
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def active_tex(text):
 """Follow only the existing-file branch when collecting TeX dependencies."""
 token=r'\IfFileExists'
 while token in text:
  start=text.index(token);position=start+len(token);groups=[]
  for _ in range(3):
   while text[position].isspace():position+=1
   assert text[position]=='{'
   begin=position+1;depth=1;position+=1
   while depth:
    if text[position]=='{' and text[position-1]!='\\':depth+=1
    if text[position]=='}' and text[position-1]!='\\':depth-=1
    position+=1
   groups.append(text[begin:position-1])
  path,yes,no=groups
  text=text[:start]+(yes if (HERE/path).exists() else no)+text[position:]
 return text
def main():
 validation=json.loads((HERE/'validation.json').read_text())
 assert validation['status']=='passed'
 assert sha(HERE/'report.pdf')==validation['files_sha256']['report.pdf']
 files={Path(p) for p in ('report.tex','report.pdf','references.bib','cvpr.sty',
  'ieeenat_fullname.bst','results_snapshot.json','validation.json',
  'typesetting_provenance.json','README.md','build_assets.py',
  'build_paper_figures.py','build_page_comparison.py','audit_failure_coordinates.py','build_failure_figures.py','validate_report.py','export_report.py')}
 todo=[Path('report.tex')]
 while todo:
  p=todo.pop();text=active_tex((HERE/p).read_text())
  for match in re.findall(r'\\input\{([^}]+)\}',text):
   child=Path(match).with_suffix('.tex')
   if child not in files:files.add(child);todo.append(child)
  for match in re.findall(r'\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}',text):files.add(Path(match))
 for relative in list(files):
  if relative.parts[0]=='figures' and relative.suffix=='.pdf' and (HERE/relative.with_suffix('.png')).exists():files.add(relative.with_suffix('.png'))
 # Optional generated status text is included by \IfFileExists + \input.
 OUT.mkdir(parents=True,exist_ok=True)
 records=[]
 for relative in sorted(files):
  source=HERE/relative;assert source.exists(),source
  if str(relative) in validation['files_sha256']:
   assert sha(source)==validation['files_sha256'][str(relative)],relative
  target=OUT/relative;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
  records.append(dict(path=str(relative),sha256=sha(target),source=str(source.relative_to(ROOT))))
 native=ROOT/'runs_from_260914/first_letters/23_paper_large0058_gpu0_gpu2'
 images=[]
 for region in ('page_a','page_b'):
  for name in ('ct','inward','outward'):
   source=native/region/'p2sd/reading'/f'{name}.png'
   target=OUT/'native_pngs'/region/f'{name}.png';target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
   record=dict(path=str(target.relative_to(OUT)),sha256=sha(target),source=str(source.relative_to(ROOT)))
   records.append(record);images.append(record)
 body=['<!doctype html><meta charset="utf-8"><title>PLatS technical report</title>',
  '<style>body{font:17px sans-serif;margin:28px;max-width:1500px}table{width:100%}td{width:50%;vertical-align:top}img{width:100%}p{line-height:1.5}</style>',
  '<h1>PLatS: Papyrus Surface Segmentation in Latent Space</h1>',
  f'<p><a href="report.pdf">Technical report PDF</a>: {validation["main_pages"]} main pages, {validation["reference_pages"]} reference page, {validation["supplementary_pages"]} Supplementary pages. <a href="validation.json">Validation</a> · <a href="results_snapshot.json">Measurements and provenance</a></p>',
  '<p>Native large-region outputs from Kaggle-only 0058. Dark ink pixels indicate higher probability; gray denotes missing support. Known ink-training scroll. These images are not verified transcriptions.</p>']
 for region in ('page_a','page_b'):
  body.append(f'<h2>{region}</h2><table><tr><th>CT texture</th><th>Inward ink probability</th></tr><tr><td><a href="native_pngs/{region}/ct.png"><img src="native_pngs/{region}/ct.png"></a></td><td><a href="native_pngs/{region}/inward.png"><img src="native_pngs/{region}/inward.png"></a></td></tr></table><p><a href="native_pngs/{region}/outward.png">Outward ink PNG</a></p>')
 body.append('<p><a href="../../first_letters/23_paper_large0058_gpu0_gpu2/index.html">Large-page surfaces and native ink tiles</a></p>')
 (OUT/'index.html').write_text('\n'.join(body)+'\n')
 records.append(dict(path='index.html',sha256=sha(OUT/'index.html'),source='generated export index'))
 views=ROOT/'runs_from_260914/evaluation/06_paper_failure_analysis/2d_slices'
 provenance=json.loads((views/'provenance.json').read_text())
 for name,digest in provenance['assets'].items():
  source=views/name;assert sha(source)==digest,name
  target=OUT/'slice_panels'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
  records.append(dict(path=str(target.relative_to(OUT)),sha256=sha(target),source=str(source.relative_to(ROOT))))
 shutil.copy2(views/'provenance.json',OUT/'slice_panels/provenance.json')
 records.append(dict(path='slice_panels/provenance.json',sha256=sha(OUT/'slice_panels/provenance.json'),source=str((views/'provenance.json').relative_to(ROOT))))
 index=OUT/'index.html'
 index.write_text(index.read_text()+'<p><a href="figures/binary_failure_main.png">Figure 4 binary slice</a> · <a href="figures/ignore_failure.png">Figure 4 ignore cuts</a> · <a href="figures/binary_failure_zoom.png">Figure 6 enlarged slice</a> · <a href="slice_panels/provenance.json">Native slice coordinates and provenance</a></p>\n')
 next(r for r in records if r['path']=='index.html')['sha256']=sha(index)
 (OUT/'manifest.json').write_text(json.dumps(dict(created_utc=datetime.now(timezone.utc).isoformat(),
  validation=validation,files=records,native_pngs_unmodified=True),indent=2)+'\n')
 print(OUT)
if __name__=='__main__':main()
