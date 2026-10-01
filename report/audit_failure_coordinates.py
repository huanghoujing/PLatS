"""Audit TIFF/internal coordinates and select a readable native z-section example."""
from itertools import combinations
from pathlib import Path
import json
import numpy as np
import tifffile
import cc3d
from scipy import ndimage as ndi
from build_failure_figures import ROOT, OUT, read, sha

def main():
    audits=[]
    for case in ('sample_00865','sample_00860','sample_00829'):
        base=ROOT/'datasets/hf_kaggle_202607'
        path=base/'raw/images'/f'{case}.tif'
        source=tifffile.imread(path);internal=np.load(base/'cases'/case/'image.npy')
        expected=source.transpose(2,1,0).copy()
        for axis in range(3):
            sl=[slice(None)]*3;sl[axis]=slice(0,5);expected[tuple(sl)]=0
            sl[axis]=slice(-5,None);expected[tuple(sl)]=0
        assert np.array_equal(internal,expected),case
        audits.append(dict(case_id=case,source=str(path.relative_to(ROOT)),sha256=sha(path),
            internal_equals_tiff_transpose_210_with_5_voxel_zero_border=True))
    case='sample_00860';z=160
    meta=read(ROOT/'datasets/hf_kaggle_202607/cases'/case/'meta.json')
    paths=[ROOT/f'runs_from_260914/evaluation/05_paper_winner_ignore106_gpu0_gpu2/winner/{case}/prediction.npz',
        ROOT/f'runs_from_260914/evaluation/04_paper_automatic_hidden106_gpu0_gpu2/0058_kaggle/cases/{case}/instances.npz']
    paths += [Path(meta[k]) for k in ('components_path','image_path','ignore_path')]
    g=np.load(meta['components_path'])[:,:,z].T
    ignore=np.unpackbits(np.load(meta['ignore_path']),count=320**3).reshape(320,320,320)[:,:,z].T.astype(bool)
    w=cc3d.connected_components(np.ascontiguousarray(np.load(paths[0])['mask']),connectivity=6)[:,:,z].T
    cached=read(OUT/'winner'/f'{case}.json')
    merged=max(cached['merged_components'],key=lambda r:r['covered_bands'])
    labels,_=ndi.label(w==merged['id']);options=[]
    for component in np.unique(labels)[1:]:
        ids=[int(i) for i in np.unique(g[labels==component]) if i]
        for a,b in combinations(ids,2):
            aa=np.argwhere((g==a)&(labels==component));bb=(g==b)&(labels==component)
            dist,idx=ndi.distance_transform_edt(~bb,return_indices=True)
            for start in aa:
                end=idx[:,start[0],start[1]];d=dist[tuple(start)]
                if not 5<=d<=30:continue
                line=np.unique(np.rint(np.linspace(start,end,int(np.ceil(d))*3+1)).astype(int),axis=0)
                yy,xx=line.T
                if ignore[yy,xx].any() or not (labels[yy,xx]==component).all():continue
                bg=int((g[yy,xx]==0).sum())
                if bg<4:continue
                options.append((bg,-float(d),int(component),a,b,start.tolist(),end.tolist(),line.tolist()))
    assert options
    bg,neg_d,component,a,b,start,end,line=max(options)
    center=(np.array(start)+end)//2;lo=np.clip(center-45,0,230)
    selected=dict(case_id=case,**merged,slice_axis=2,slice_index=z,zoom_yx=[*lo.tolist(),90],
        within_slice_connected_component=component,bridge_pair_gt_ids=[a,b],closest_gt_distance_vox=-neg_d,
        gap_audit=dict(start_yx=start,end_yx=end,sampled_yx=line,sampled_voxels=len(line),ignored_voxels=0,
            predicted_known_background_voxels=bg,interpretation='Unique rasterized straight gap; every voxel is winner foreground and none is ignored'),
        winner_raw_betti=cached['raw_betti'],sources={str(p.relative_to(ROOT)):sha(p) for p in paths},
        selection='Visually clear lamellar CT case among the top 20 cached merger candidates; central TIFF z=160; longest eligible known-background gap between nearest GT/winner intersections (5 to 30 voxels) within one 2D component')
    (OUT/'binary_figure_selection.json').write_text(json.dumps(selected,indent=2)+'\n')
    audit=dict(convention='TIFF stack [z,y,x]; project internal [x,y,z]. Display z by internal[:,:,z].T; no interpolation.',
        previous_error='Figure code mislabeled internal axis 0 (TIFF x) as z. CT and masks were mutually aligned.',cases=audits)
    (OUT/'coordinate_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    print(json.dumps(selected,indent=2))
if __name__=='__main__':main()
