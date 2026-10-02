"""Editable vector diagrams and figures derived from frozen experiment outputs."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch,Rectangle
plt.rcParams.update({'pdf.fonttype':42,'ps.fonttype':42,'font.size':8.5})
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
def save(fig,name):
 fig.savefig(HERE/'figures'/f'{name}.pdf',bbox_inches='tight',pad_inches=.03)
 fig.savefig(HERE/'figures'/f'{name}.png',dpi=240,bbox_inches='tight',pad_inches=.03)
 plt.close(fig)
def canvas(height):
 fig,ax=plt.subplots(figsize=(6.9,height));ax.set(xlim=(0,6.9),ylim=(0,height));ax.axis('off')
 fig.subplots_adjust(left=0,right=1,bottom=0,top=1);return fig,ax
def box(ax,x,y,w,h,text,color='#e9f1f8',fontsize=8.5):
 ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.035',facecolor=color,edgecolor='#415b73',lw=.8))
 ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=fontsize)
def arrow(ax,a,b,label=None,dashed=False,color='#415b73'):
 ax.annotate('',xy=b,xytext=a,arrowprops=dict(arrowstyle='->',lw=1,color=color,linestyle='--' if dashed else '-'))
 if label:ax.text((a[0]+b[0])/2,(a[1]+b[1])/2+.09,label,ha='center',fontsize=7.5,color=color)
def architecture(*, include_inference=True):
 # Two compact diagrams: one central flow, with objectives on separate rows.
 blue='#e3eff9'; gray='#edf0f2'; gold='#fff1d9'
 fig,ax=canvas(1.48)
 nodes=[(.06,.89,1.02,'Corrupted\nsheet mask',gray),
        (1.42,.89,.96,'Encoder E',blue),(2.72,.89,.83,'Code z',blue),
        (3.89,.89,1.16,'Decoder D',blue),(5.4,.89,1.38,'Reconstructed\nsheet Q',gray)]
 for x,y,w,t,c in nodes:box(ax,x,y,w,.43,t,c,9)
 for left,right in zip(nodes,nodes[1:]):arrow(ax,(left[0]+left[2]+.04,1.105),(right[0]-.04,1.105))
 box(ax,.06,.11,2.06,.45,'Occupancy: BCE + Dice',gold,8.6)
 box(ax,2.4,.11,2.06,.45,'Dense / query: Smooth L1',gold,8.6)
 box(ax,4.74,.11,2.04,.45,'Code / spatial regularizers',gold,8.6)
 ax.text(3.45,.70,'All heads depend on the code  ·  No skip connections',ha='center',fontsize=8,color='#435666')
 save(fig,'ae')
 fig,ax=canvas(2.13)
 # Supervision row: a frozen teacher supplies the latent target.
 box(ax,.07,1.64,.96,.38,'GT sheet',gray,9)
 box(ax,1.42,1.64,1.02,.38,'Frozen AE E',gray,9)
 box(ax,3.35,1.64,2.23,.38,'Latent MSE + identity losses',gold,8.8)
 arrow(ax,(1.07,1.83),(1.38,1.83))
 arrow(ax,(2.48,1.83),(3.31,1.83),dashed=True)
 # Prediction row.
 for x,w,t,c in [(.07,.96,'CT crop X',gray),(1.42,1.02,'CT encoder +\ncontext C',blue),
                 (2.83,1.42,'Point transformer T',blue),(4.64,.93,'Code z',blue),
                 (5.95,.84,'Frozen D',gray)]:box(ax,x,.83,w,.42,t,c,8.8)
 for a,b in [((1.07,1.04),(1.38,1.04)),((2.48,1.04),(2.79,1.04)),
             ((4.29,1.04),(4.60,1.04)),((5.61,1.04),(5.91,1.04))]:arrow(ax,a,b)
 arrow(ax,(5.10,1.29),(5.10,1.60),dashed=True)
 ax.text(6.37,.65,'Sheet mask',ha='center',fontsize=8.5,color='#435666')
 # Positive prompts enter the transformer; dense binary supervision branches left.
 box(ax,.07,.06,1.05,.39,'Co-trained B',blue,8.7)
 box(ax,1.48,.06,1.10,.39,'Union BCE + Dice',gold,8)
 box(ax,2.94,.06,1.2,.39,'Positive points P',gray,8.8)
 arrow(ax,(3.54,.49),(3.54,.79))
 ax.annotate('',xy=(.60,.49),xytext=(1.72,.79),arrowprops=dict(arrowstyle='->',lw=1,color='#415b73',connectionstyle='arc3,rad=.1'))
 arrow(ax,(1.16,.255),(1.44,.255),dashed=True)
 ax.text(4.63,.20,'Blue: trainable   Gray: frozen / data',fontsize=7.9,color='#435666')
 save(fig,'training')
 if include_inference:
  actual_inference()

def actual_inference():
 from matplotlib.colors import ListedColormap
 base=ROOT/'runs_from_260914/evaluation/08_paper_method_visuals'
 record=read(base/'provenance.json');case=record['case'];z=record['native_z']
 auto=ROOT/'runs_from_260914/evaluation/04_paper_automatic_hidden106_gpu0_gpu2'
 meta=read(ROOT/'datasets/hf_kaggle_202607/cases'/case/'meta.json')
 ct=np.load(meta['image_path'])[:,:,z].T
 with np.load(auto/'0058_kaggle/cases'/case/'instances.npz') as a:inst=a['inst'][:,:,z].T
 with np.load(base/'intermediates.npz') as a:
  points=a['points_xyz'];identity=a['identity'];xy=a['pca']
  features=np.sqrt(np.mean(a['context'][0].astype(np.float64)**2,axis=0))[:,:,z//32].T
  variance=float(a['pca_variance_fraction'])
 # One palette links codes to saved final instance IDs; discarded seeds are gray.
 colors=np.vstack(([.70,.70,.70,1],plt.get_cmap('tab20')(np.arange(20))))
 cmap=ListedColormap(colors);norm=matplotlib.colors.BoundaryNorm(np.arange(-.5,21.5),21)
 fig,axs=plt.subplots(1,4,figsize=(7.05,2.30))
 titles=['1  Foreground seeds','2  CT features','3  Latent clustering','4  Decoded sheets']
 notes=['CT + seeds near this slice','Channel RMS, native grid',f'Actual codes · PCA ({variance:.0%})','Same CT plane · final IDs']
 for ax,title in zip(axs,titles):
  ax.set_title(title,fontsize=9,fontweight='bold',loc='left',pad=8)
  ax.set_xticks([]);ax.set_yticks([])
  for sp in ax.spines.values():sp.set_visible(False)
 axs[0].imshow(ct,cmap='gray',vmin=0,vmax=255,interpolation='nearest')
 near=abs(points[:,2]-z)<=12
 axs[0].scatter(points[near,0],points[near,1],s=11,c='#ffcf46',edgecolors='#1e252a',linewidths=.25)
 axs[1].imshow(features,cmap='magma',interpolation='nearest')
 dropped=identity==0
 axs[2].scatter(*xy[dropped].T,s=5,color='#c7c7c7',linewidths=0,rasterized=True)
 axs[2].scatter(*xy[~dropped].T,s=8,c=identity[~dropped],cmap=cmap,norm=norm,linewidths=.15,edgecolors='white',rasterized=True)
 axs[2].set_box_aspect(1)
 axs[2].margins(.08)
 axs[3].imshow(ct,cmap='gray',vmin=0,vmax=255,interpolation='nearest')
 axs[3].imshow(np.ma.masked_where(inst==0,inst),cmap=cmap,norm=norm,alpha=.94,interpolation='nearest')
 for ax,note in zip(axs,notes):ax.text(.5,-.10,note,transform=ax.transAxes,ha='center',va='top',fontsize=7.6)
 fig.subplots_adjust(left=.005,right=.995,bottom=.20,top=.84,wspace=.10)
 save(fig,'inference')
def read(p):return json.loads(Path(p).read_text())
def large_pages(data):
 from PIL import Image
 from build_assets import sha
 base=ROOT/'runs_from_260914/first_letters/23_paper_large0058_gpu0_gpu2'
 if not (base/'complete.json').exists():return
 fig,axs=plt.subplots(1,4,figsize=(11,5.0));records=[]
 for j,region in enumerate(('page_a','page_b')):
  d=base/region/'p2sd/reading'
  for k,name in enumerate(('ct','inward')):
   p=d/f'{name}.png';axs[j*2+k].imshow(np.asarray(Image.open(p)),cmap='gray',vmin=0,vmax=255);axs[j*2+k].axis('off')
   axs[j*2+k].set_title(f'{region.replace("_"," ")}: '+('CT texture' if k==0 else 'Ink probability'),fontsize=11)
   records.append(dict(path=str(p.relative_to(ROOT)),sha256=sha(p)))
 fig.subplots_adjust(left=.005,right=.995,bottom=.005,top=.94,wspace=.06)
 fig.savefig(HERE/'figures/large_ink.png',dpi=280);plt.close(fig)
 data['large_pages']=dict(model='0058',complete=read(base/'complete.json'),files=records,
    selection='Both preregistered page regions; independent layouts; inward orientation fixed geometrically',
    regions={r:dict(growth=read(base/r/'p2sd/growth.json'),reading=read(base/r/'p2sd/reading/complete.json')) for r in ('page_a','page_b')})

def reading_display_frame(path):
 # Use only reconstructed geometry, never CT appearance, reference labels or ink.
 with np.load(path) as a:
  valid=a['valid']; xy=np.argwhere(valid)[:,::-1];xyz=a['xyz_local'][valid]
  basis=np.linalg.lstsq(np.c_[xy,np.ones(len(xy))],xyz,rcond=None)[0][:2]
  normal=a['normals'][valid].mean(0)
 return basis/np.linalg.norm(basis,axis=1,keepdims=True),normal/np.linalg.norm(normal)

def downstream_figures(data):
 from PIL import Image
 from build_assets import sha
 records=[]
 base=ROOT/'runs_from_260914/first_letters/21_p2sd_spiral_inputs_gpu0_gpu2'
 methods=[('p2sd','Direct PLatS'),('spiral','Published tracks'),('p2sd_only','PLatS only fit'),('m7_plus_p2sd','Mixed-input fit')]
 fig,axs=plt.subplots(1,4,figsize=(11,4.5))
 for j,(arm,title) in enumerate(methods):
  d=base/'page_a'/arm/'reading'
  p=d/'inward.png';axs[j].imshow(np.asarray(Image.open(p)),cmap='gray',vmin=0,vmax=255);axs[j].axis('off')
  axs[j].set_title(title,fontsize=11)
  records.append(dict(path=str(p.relative_to(ROOT)),sha256=sha(p)))
 fig.subplots_adjust(left=.01,right=.99,bottom=.005,top=.94,wspace=.06,hspace=.03)
 fig.savefig(HERE/'figures/spiral_integration.png',dpi=240);plt.close(fig)
 data['spiral_figure']=dict(model='original_0076',region='page_a',files=records,layout='Independent layouts, full pages, common grayscale; inward ink only')
 if data['winner_reading'] is None:return
 winner=ROOT/'runs_from_260914/first_letters/24_paper_winner_onepoint_gpu0_gpu2'
 plats=ROOT/'runs_from_260914/first_letters/22_paper_ffn_three_checkpoints_onepoint_gpu0_gpu2'
 rows=data['winner_reading']['rows']
 eligible=[r for r in rows if not r['case'].startswith('pherc0814') and any(c['is_seed_component'] and c['status']=='ink_complete' for c in (r['reading'] or {}).get('components',[]))]
 case=eligible[0]['case'] if eligible else 'pherc0139-w016_patch02'
 selection='First evaluation crop in predefined order with an accepted winner seed chart; no selection by CT/ink appearance'
 records=[];fig,axs=plt.subplots(2,3,figsize=(11,5.5))
 reference=plats/case/'ffn/component_00/surface.npz'
 reference_basis,reference_normal=reading_display_frame(reference)
 orientations=[]
 for i,(base,method,title) in enumerate([(plats,'ffn','FFN'),(winner,'winner','Winner'),(plats,'0058_kaggle','PLatS 0058')]):
  for j,case in enumerate((case,)):
   d=base/case/method
   components=[r for r in read(d/'reading.json')['components'] if r['is_seed_component']] if (d/'reading.json').exists() else []
   assert len(components)<=1
   accepted=len(components)==1 and components[0]['status']=='ink_complete'
   if accepted:
    comp=components[0]['component'];surface=d/f'component_{comp:02d}/surface.npz'
    basis,normal=reading_display_frame(surface)
    horizontal_cosine=float(basis[0]@reference_basis[0]);vertical_cosine=float(basis[1]@reference_basis[1])
    normal_cosine=float(normal@reference_normal)
    assert abs(horizontal_cosine)>.9 and vertical_cosine>.9 and abs(normal_cosine)>.9
    flip_horizontal=horizontal_cosine<0
    side='normal_plus' if normal_cosine>0 else 'normal_minus'
    orientations.append(dict(method=method,surface=str(surface.relative_to(ROOT)),surface_sha256=sha(surface),
      horizontal_cosine=horizontal_cosine,vertical_cosine=vertical_cosine,normal_cosine=normal_cosine,
      horizontal_flip=flip_horizontal,normal=side))
   for k,name in enumerate(('ct','ink')):
    ax=axs[k,i];ax.axis('off');ax.set_title(f'{title}: '+name,fontsize=11)
    if accepted:
     p=d/f'component_{comp:02d}/{side}/{name}.png'
     im=np.asarray(Image.open(p))
     if flip_horizontal:im=im[:,::-1]
     ax.imshow(im,cmap='gray',vmin=0,vmax=255)
     records.append(dict(path=str(p.relative_to(ROOT)),sha256=sha(p),selection='Seed-nearest component; sampling side and horizontal display direction aligned by reconstructed geometry',horizontal_flip=flip_horizontal,normal=side))
    else:
     reason=components[0].get('reason','rejected') if components else 'Mesh/flatten attempt unsuccessful'
     ax.text(.5,.55,'No completed '+('CT texture' if name=='ct' else 'ink output'),ha='center',va='center',transform=ax.transAxes,fontsize=11)
     ax.text(.5,.38,reason[:150],ha='center',va='center',wrap=True,transform=ax.transAxes,fontsize=9)
     records.append(dict(method=method,case=case,status='no_accepted_seed_chart',reason=reason))
 fig.subplots_adjust(left=.005,right=.995,bottom=.005,top=.94,wspace=.06,hspace=.10)
 fig.savefig(HERE/'figures/winner_reading.png',dpi=240);plt.close(fig)
 data['winner_reading_figure']=dict(case=case,selection=selection,files=records,layout='Independent charts; horizontal direction and sampling side aligned to FFN using reconstructed geometry only; no pixel registration',
  orientation_reference=dict(path=str(reference.relative_to(ROOT)),sha256=sha(reference)),orientations=orientations)
 (HERE/'tables/winner_reading_figure_caption.tex').write_text(
  r'\newcommand{\WinnerReadingCaption}{'+'FFN, winner and Kaggle-only PLatS downstream control on '+case.replace('_',r'\_')+'. '
  'Each uses the same one physical seed, independent meshing/flattening and the same ink model. '
  'Rows show CT texture and ink; columns are independent layouts, not registered pixels. '
  'The seed-nearest component is shown. Horizontal direction and sampling side are aligned to FFN using reconstructed geometry only; both PLatS rows are flipped horizontally and use its saved normal-minus output. '
  'The crop is the first in the predefined evaluation order with an accepted winner seed chart, without selection by ink appearance. '
  'All crops, both normals, all components and failed attempts are retained in the gallery.}\n')
 count=sum(any(c['status']=='ink_complete' for c in (r['reading'] or {}).get('components',[])) for r in rows)
 seed_count=sum(any(c['is_seed_component'] and c['status']=='ink_complete' for c in (r['reading'] or {}).get('components',[])) for r in rows if not r['case'].startswith('pherc0814'))
 charts=sum(c['status']=='ink_complete' for r in rows for c in (r['reading'] or {}).get('components',[]))
 (HERE/'tables/winner_reading_status.tex').write_text(
  f'Of five attempted crops (four evaluation, one development), {count} deliver at least one independently flattened ink chart, '
  f'with {charts} accepted charts in total; {seed_count} of four evaluation crops deliver the seed-nearest chart. '
  'These counts describe pipeline execution, not readable letters. Rejected charts and bounded runtime failures are retained. '
  'Mesh-stage/crop limits are 1,800/4,200 seconds, with three parallel workers.\n')
if __name__=='__main__':architecture()
