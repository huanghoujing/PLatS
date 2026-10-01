"""Figure 7: completed matched page-scale reconstructions, with fixed native scale."""
from pathlib import Path
import json,hashlib
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
RUN=ROOT/'runs_from_260914/first_letters/25_paper_page_onepoint_gpu0_gpu2'
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def read(p):return json.loads(Path(p).read_text())
def build(data):
 if not (RUN/'complete.json').exists():return
 assert read(RUN/'complete.json')['status']=='complete'
 plan=read(RUN/'plan.json');amendment=read(RUN/'scope_amendment.json')
 assert amendment['active_methods']==['ffn','0058']
 assert read(RUN/'complete.json')['methods']==['ffn','0058']
 records=[];arrays=[];rows=[]
 for path in [Path(__file__),ROOT/'scripts/paper_page_onepoint.py',ROOT/'scripts/paper_page_winner_adapter.py',ROOT/'scripts/finalize_page_onepoint_report.py',RUN/'controller_at_launch.py']:
  data['source_code_sha256'][str(path.relative_to(ROOT))]=sha(path)
 assert sha(RUN/'controller_at_launch.py')==plan['controller_sha256']
 for method,title in [('ffn','FFN'),('0058','PLatS 0058')]:
  d=RUN/method/'reading';complete=read(d/'complete.json');geometry=read(d/'geometry.json');stitch=read(RUN/method/'stitching.json')
  assert complete['status']=='complete'
  assert len(stitch['rows'])==len(plan['cases'])==126
  im=[]
  for name in ('ct','inward'):
   path=d/f'{name}.png';im.append(np.asarray(Image.open(path)));records.append(dict(path=str(path.relative_to(ROOT)),sha256=sha(path)))
  assert im[0].shape==im[1].shape
  arrays.append(im);rows.append(dict(method=method,title=title,area_mm2=geometry['area_mm2'],reading=complete,geometry=geometry,stitching=stitch))
 # Common canvas dimensions retain native voxel scale, without image registration.
 shape=np.max([im[0].shape for im in arrays],axis=0)
 fig,axes=plt.subplots(2,2,figsize=(11,11))
 for col,(ims,row) in enumerate(zip(arrays,rows)):
  for r,(im,name) in enumerate(zip(ims,('CT','Inward ink'))):
   canvas=np.full(shape,180,np.uint8);offset=(shape-np.array(im.shape))//2
   canvas[offset[0]:offset[0]+im.shape[0],offset[1]:offset[1]+im.shape[1]]=im
   axes[r,col].imshow(canvas,cmap='gray',vmin=0,vmax=255,interpolation='nearest');axes[r,col].axis('off')
   axes[r,col].set_title(f"{row['title']}: {name}\n{row['area_mm2']/100:.2f} cm² reconstructed",fontsize=10)
 fig.subplots_adjust(left=.005,right=.995,bottom=.005,top=.95,wspace=.03,hspace=.12)
 fig.savefig(HERE/'figures/page_onepoint.png',dpi=260);plt.close(fig)
 data['page_onepoint']=dict(plan=plan,scope_amendment=amendment,rows=rows,files=records,
  display='Same native pixel scale; centered gray padding only; independent SLIM layouts; inward normals; scan Z vertical. No evaluation geometry or ink labels.',
  source_run=str(RUN.relative_to(ROOT)),source_sha256={str(p.relative_to(ROOT)):sha(p) for p in [RUN/'plan.json',RUN/'scope_amendment.json',RUN/'complete.json',*[RUN/m/k for m in ('ffn','0058') for k in ('stitching.json','reading/complete.json','reading/geometry.json')]]})
 caption=(r'\newcommand{\PageOnepointCaption}{Page-scale FFN and Kaggle-only PLatS comparison on PHerc0139 page A, in the region of Figure 8. '
  r'Both receive the same 126 overlapping $320^3$ crops and one shared physical prompt per crop (224-voxel step). '
  r'Prompts are sparsely sampled from a frozen earlier 0076 prediction; this can favor the PLatS family and inherit prior drift. '
  r'Unique-ray extraction and overlap rejection are shared; each primary connected chart is independently flattened. '
  r'Rows show CT and inward ink at the same native scale with gray padding for missing support; layouts are not pixel-registered. '
  r'No evaluation surface, reference UV or ink labels guide reconstruction. This tests repeated sparse prompting and stitching, not autonomous single-seed growth.}'+'\n')
 (HERE/'tables/page_onepoint_caption.tex').write_text(caption)
 (HERE/'tables/page_onepoint_results.tex').write_text(
  '\\paragraph{Page-scale repeated prompting.} Figure~\\ref{fig:winnerreading} replaces the single-cube illustration with the matched page experiment. '
  +' '.join(f"{r['title']} delivers {r['area_mm2']/100:.2f}~cm$^2$ from {r['stitching']['nonempty_crops']}/126 nonempty local reconstructions." for r in rows)
  +' The gallery preserves all raw masks, overlap conflicts, disconnected support and both normal directions. Primary-chart area alone does not establish sheet identity or readable text.\n')
