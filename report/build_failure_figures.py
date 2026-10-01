"""Native 2D slices of saved binary-merger and ignore-erasure examples."""
import json
from pathlib import Path
import cc3d
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import nibabel as nib
import numpy as np
from scipy import ndimage as ndi
from build_assets import sha

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
OUT=ROOT/'runs_from_260914/evaluation/06_paper_failure_analysis'
SLICES=OUT/'2d_slices'
plt.rcParams.update({'font.size':11,'pdf.fonttype':42})

def read(p):return json.loads(Path(p).read_text())
def volume(p):return np.asanyarray(nib.load(p).dataobj)>0
def plane(a,s,axis):
    assert axis == 2, "TIFF z is internal axis 2"
    return a[:,:,s].T
def rgb(labels):
    palette=np.asarray(matplotlib.colormaps['tab20'].colors)
    result=np.ones((*labels.shape,3));fg=labels>0
    result[fg]=palette[(labels[fg].astype(int)-1)%len(palette)]
    return result
def mask_rgb(mask,removed=None):
    result=np.ones((*mask.shape,3));result[mask]=[.20,.49,.64]
    if removed is not None:result[removed]=[.94,.43,.12]
    return result
def row(images,titles,target,roi=None,gap=None):
    aspect=images[0].shape[0]/images[0].shape[1]
    fig,axs=plt.subplots(1,len(images),figsize=(11,max(2,2.65*aspect+.4)))
    for i,(ax,im,title) in enumerate(zip(axs,images,titles)):
        ax.imshow(im,cmap='gray',vmin=0,vmax=255,interpolation='nearest')
        ax.set_title(title,fontsize=11,pad=7);ax.axis('off')
        if roi is not None:
            y,x,side=roi;ax.add_patch(Rectangle((x-.5,y-.5),side,side,
                fill=False,linewidth=1.3,edgecolor='#e9701a'))
        if gap is not None and i in (1,2):
            a,b=gap
            ax.plot([a[1],b[1]],[a[0],b[0]],color='#c52626',lw=1.2,ls='--')
            ax.scatter([a[1],b[1]],[a[0],b[0]],s=18,facecolor='white',edgecolor='#c52626',linewidth=.9)
    fig.subplots_adjust(left=.005,right=.995,bottom=.01,top=.89,wspace=.08)
    fig.savefig(target,dpi=260);plt.close(fig)

def binary_slices(selected):
    case=selected['case_id'];meta=read(ROOT/'datasets/hf_kaggle_202607/cases'/case/'meta.json')
    image=np.load(meta['image_path']);gt=np.load(meta['components_path'])
    with np.load(ROOT/f'runs_from_260914/evaluation/05_paper_winner_ignore106_gpu0_gpu2/winner/{case}/prediction.npz') as a:
        winner=cc3d.connected_components(np.ascontiguousarray(a['mask']),connectivity=6)
    with np.load(ROOT/f'runs_from_260914/evaluation/04_paper_automatic_hidden106_gpu0_gpu2/0058_kaggle/cases/{case}/instances.npz') as a:pred=a['inst']
    z=selected['slice_index'];y,x,side=selected['zoom_yx']
    arrays=dict(image=plane(image,z,2),gt=plane(gt,z,2),winner=plane(winner,z,2),plats=plane(pred,z,2))
    native=SLICES/'binary_slice.npz';np.savez_compressed(native,**arrays)
    images=[arrays['image'],rgb(arrays['gt']),rgb(arrays['winner']),rgb(arrays['plats'])]
    titles=['CT: z='+str(z),'GT annotated bands','Winner: binary CC IDs','PLatS: automatic IDs']
    row(images,titles,HERE/'figures/binary_failure_main.png',roi=[y,x,side])
    gap=[np.asarray(selected['gap_audit'][k])-np.array([y,x]) for k in ('start_yx','end_yx')]
    row([im[y:y+side,x:x+side] for im in images],titles,
        HERE/'figures/binary_failure_zoom.png',gap=gap)
    return dict(case_id=case,axis=2,slice_index=z,zoom_yx=[y,x,side],
                native_panels=str(native.relative_to(ROOT)),source_hashes=selected['sources'])

def ignore_slices(selected):
    base=OUT/'figure_nifti'/selected['case_id']/f'sheet_{selected["sheet_id"]:04d}'
    raw=volume(base/'raw.nii.gz');erased=volume(base/'erased.nii.gz')
    restored=volume(base/'restored.nii.gz');removed=volume(base/'removed.nii.gz')
    assert np.array_equal(raw,restored) and np.array_equal(raw & ~removed,erased)
    # Choose a continuous native z section, matching the binary comparison axis.
    # This display choice changes neither the case/sheet nor any score input.
    candidates=[]
    for axis in (2,):
        for s in range(raw.shape[axis]):
            r=plane(raw,s,axis);cut=plane(erased,s,axis)
            count=int(r.sum());lost=int((r & ~cut).sum())
            if count<1000 or lost<20:continue
            _,before=ndi.label(r);_,after=ndi.label(cut)
            if before==1 and after>1:candidates.append((after,lost,count,axis,s))
    assert candidates,'No eligible native continuous slice with ignore-induced cuts'
    after,lost,count,axis,s=max(candidates)
    r=plane(raw,s,axis);points=np.argwhere(r)
    lo=np.maximum(points.min(0)-8,0);hi=np.minimum(points.max(0)+9,r.shape)
    crop=tuple(slice(int(a),int(b)) for a,b in zip(lo,hi))
    meta=read(ROOT/'datasets/hf_kaggle_202607/cases'/selected['case_id']/'meta.json')
    image=np.load(meta['image_path']);assert image.shape==raw.shape
    arrays=dict(image=plane(image,s,axis)[crop],raw=r[crop],
        erased=plane(erased,s,axis)[crop],restored=plane(restored,s,axis)[crop],
        removed=plane(removed,s,axis)[crop])
    native=SLICES/'ignore_slice.npz';np.savez_compressed(native,**arrays)
    titles=[f'CT: z={s}','Raw sheet','After ignore erasure','Restored sheet']
    row([arrays['image'],mask_rgb(arrays['raw'],arrays['removed']),
         mask_rgb(arrays['erased']),mask_rgb(arrays['restored'])],titles,HERE/'figures/ignore_failure.png')
    return dict(case_id=selected['case_id'],sheet_id=selected['sheet_id'],axis=axis,slice_index=s,
        crop_start=lo.tolist(),crop_stop=hi.tolist(),raw_2d_components=1,erased_2d_components=after,
        raw_slice_voxels=count,removed_slice_voxels=lost,native_panels=str(native.relative_to(ROOT)),
        selection='Maximum visible 2D component count after erasure among raw continuous native z slices with >=1000 foreground and >=20 removed voxels',
        source_hashes={str(p.relative_to(ROOT)):sha(p) for p in [base/f'{n}.nii.gz' for n in ('raw','erased','restored','removed')]+[Path(meta['image_path'])]})

def main():
    SLICES.mkdir(parents=True,exist_ok=True)
    binary=read(OUT/'binary_figure_selection.json');ignore=read(OUT/'ignore_figure_selection.json')
    for p,d in binary['sources'].items():assert sha(ROOT/p)==d,p
    assert sha(ROOT/ignore['path'])==ignore['prediction_sha256']
    b=binary_slices(binary);i=ignore_slices(ignore)
    rendering=dict(method='Native 2D array slices; nearest-neighbor display; no projection or resampling',
        same_slice_and_crop_within_each_comparison=True,interpolation='nearest',
        directory=str(SLICES.relative_to(ROOT)),
        coordinate_convention='TIFF stack [z,y,x]; internal [x,y,z]; z plane = internal[:,:,z].T',
        coordinate_audit=read(OUT/'coordinate_audit.json'))
    binary['rendering']=dict(**rendering,**b);ignore['rendering']=dict(**rendering,**i)
    ignore['topology']='Whole 3D exact V-construction voxel counts; the slice shows selected cuts, not all tunnels'
    (OUT/'binary_figure_selection.json').write_text(json.dumps(binary,indent=2)+'\n')
    (OUT/'ignore_figure_selection.json').write_text(json.dumps(ignore,indent=2)+'\n')
    provenance=dict(rendering=rendering,binary=b,ignore=i,
        figures={str(p.relative_to(ROOT)):sha(p) for p in [HERE/'figures'/n for n in
            ('binary_failure_main.png','binary_failure_zoom.png','ignore_failure.png')]},
        assets={p.name:sha(p) for p in SLICES.iterdir() if p.is_file() and p.name!='provenance.json'})
    (SLICES/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    print(json.dumps(dict(binary=b,ignore=i),indent=2))
if __name__=='__main__':main()
