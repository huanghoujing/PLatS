#!/usr/bin/env python3
"""Render three method figures from measured, locally captured model outputs.

The NPZ is produced by capture_method_figure_data.py. It contains actual
matched CT/mask planes and full-code distances, never synthetic predictions.
"""
from pathlib import Path
import argparse
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
from matplotlib.colors import ListedColormap, BoundaryNorm, LogNorm

plt.rcParams.update({'font.size': 8, 'pdf.fonttype': 42, 'svg.fonttype': 'none'})
COLORS = dict(train='#dcecf9', frozen='#eee6f4', data='white', code='#d9efe5',
              loss='#ffedc7', edge='#354b5d', blue='#297aaf', orange='#dc792e')

def canvas(height):
    fig, ax = plt.subplots(figsize=(7.2, 7.2 * height / 100))
    ax.set(xlim=(0, 100), ylim=(0, height))
    ax.axis('off')
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    return fig, ax

def label(ax, x, y, text, size=8, align='center', **kw):
    ax.text(x, y, text, ha=align, va='center', fontsize=size, **kw)

def box(ax, x, y, w, h, text, kind='train', size=8):
    ax.add_patch(FancyBboxPatch((x,y), w,h, boxstyle='round,pad=.18,rounding_size=.55',
        facecolor=COLORS[kind], edgecolor='#876a9b' if kind=='frozen' else COLORS['edge'],
        linewidth=.8, linestyle='--' if kind=='frozen' else '-'))
    label(ax, x+w/2, y+h/2, text, size)

def arrow(ax, points, dashed=False, color=None):
    color = color or COLORS['edge']
    if len(points)>2:
        ax.plot(*np.array(points[:-1]).T, color=color, lw=.85, ls='--' if dashed else '-')
    ax.annotate('', xy=points[-1], xytext=points[-2], arrowprops={
        'arrowstyle':'->', 'lw':.85, 'color':color, 'linestyle':'--' if dashed else '-'})

def code(ax, x, y, w, h, symbol):
    # A schematic tensor glyph identifies a code object, not a projection or
    # feature-intensity measurement. No invented latent values are displayed.
    for offset in (.8, .4, 0):
        ax.add_patch(Rectangle((x+offset,y+offset),w,h,fc=COLORS['code'],
                               ec=COLORS['edge'],lw=.7))
    label(ax,x+w/2,y+h/2,symbol,12)

def image_panel(fig, rect, values, title='', cmap='gray_r', vmin=0, vmax=1, color=None):
    # rect is expressed in the same physical coordinate system as the canvas.
    height = fig.get_figheight()/fig.get_figwidth()*100
    x,y,w,h = rect
    iax = fig.add_axes([x/100,y/height,w/100,h/height])
    if color:
        cmap = ListedColormap(['white',color])
    iax.imshow(values,cmap=cmap,vmin=vmin,vmax=vmax,interpolation='nearest')
    iax.set_xticks([]);iax.set_yticks([])
    for spine in iax.spines.values():
        spine.set_color('#a7b1b9');spine.set_linewidth(.5)
    if title: iax.set_title(title,fontsize=7.5,pad=3)
    return iax

def save(fig, out, name):
    out.mkdir(parents=True,exist_ok=True)
    for ext in ('pdf','png','svg'):
        path=out/f'{name}.{ext}'
        fig.savefig(path,dpi=260,bbox_inches='tight',pad_inches=.04,
                    metadata={'Creator':'PLatS measured method figures'})
        if ext=='svg': path.write_text('\n'.join(s.rstrip() for s in path.read_text().splitlines())+'\n')
    plt.close(fig)

def ae_figure(data, record, out):
    fig, ax=canvas(43)
    # Same central XY crop at native z=160 in every image panel. No mask
    # thickening, hand correction or independently selected slices.
    roi=np.s_[40:200,80:240]
    view=lambda key: data[key][roi]
    image_panel(fig,(.5,28,12,12),view('corrupted'),r'Corrupted $\widetilde S$')
    box(ax,16,31,9,6,'Encoder\n$E$')
    code(ax,29,30,8,8,'$z$')
    box(ax,41,31,9,6,'Decoder\nfeatures')
    box(ax,53,32,9,5,'Occupancy\nheads',size=7.2)
    image_panel(fig,(65,28,11,11),view('reconstruction')>=.5,r'Reconstruction $Q$')
    box(ax,78,31,8,6,'BCE\n+ Dice','loss',7.2)
    image_panel(fig,(89,28,11,11),view('clean'),r'Clean target $S$')
    for start,end in [(12.5,15.7),(25.3,28.7),(38,40.7),(50.3,52.7),(62.3,64.7),(76.3,77.7)]:
        arrow(ax,[(start,34),(end,34)])
    arrow(ax,[(88.7,34),(86.3,34)])
    label(ax,19,26,'No encoder–decoder skips',7.6)
    box(ax,76.5,23.1,12,4,'Outside / border\npenalty','loss',6.7)
    arrow(ax,[(70.5,28),(70.5,25.1),(76.2,25.1)])
    arrow(ax,[(94.5,28),(94.5,25.1),(88.8,25.1)])

    box(ax,53,16.3,9,5,'Dense\ndistance',size=7.2)
    image_panel(fig,(65,10,11,11),view('dense_distance'),'$U$',cmap='magma',vmin=0,vmax=1)
    box(ax,78,14,8,5,'Smooth\nL1','loss',7.2)
    image_panel(fig,(89,10,11,11),view('target_distance'),'GT distance $d_S$',cmap='magma',vmin=0,vmax=1)
    arrow(ax,[(45.5,30.7),(45.5,18.8),(52.7,18.8)])
    arrow(ax,[(62.3,18.8),(64.7,16.5)])
    arrow(ax,[(76.3,16.5),(77.7,16.5)])
    arrow(ax,[(88.7,16.5),(86.3,16.5)])

    box(ax,51,1.7,13,5,'Query head\n$h(z,q)$',size=7.5)
    box(ax,77,1.7,10,5,'Smooth L1','loss',7.1)
    label(ax,95,4.2,'$d_S(q)$',8.5)
    arrow(ax,[(33,29.7),(33,24),(48,24),(48,4.2),(50.7,4.2)])
    arrow(ax,[(64.3,4.2),(76.7,4.2)])
    label(ax,70.5,6.2,'distance at $q$',6.9)
    arrow(ax,[(91,4.2),(87.3,4.2)])
    arrow(ax,[(94.5,9.7),(94.5,6.3)])

    # A measured two-sheet example makes the repulsion objective concrete.
    label(ax,.5,21.4,'Different sheets → distinguishable codes',8,align='left',weight='bold')
    image_panel(fig,(1,11,8,8),view('clean'),color=COLORS['blue'])
    image_panel(fig,(1,1,8,8),view('second'),color=COLORS['orange'])
    box(ax,12,7.5,7,5,'$E$')
    code(ax,22,12.5,6,5,'$z_i$');code(ax,22,2.5,6,5,'$z_j$')
    arrow(ax,[(9.3,15),(10.5,15),(11.7,11)])
    arrow(ax,[(9.3,5),(10.5,5),(11.7,9)])
    arrow(ax,[(19.3,11),(21.7,15)])
    arrow(ax,[(19.3,9),(21.7,5)])
    box(ax,32,8,12,5,'Cosine-margin\nrepulsion','loss',7.2)
    arrow(ax,[(28.3,15),(30,15),(31.7,12)])
    arrow(ax,[(28.3,5),(30,5),(31.7,9)])
    label(ax,38,4.3,f"Measured cos\n{record['cosine_raw_ae_codes']:.3f}",7)
    save(fig,out,'sheet_representation')

def point_figure(data, out):
    fig,ax=canvas(41)
    image_panel(fig,(.5,27,11,11),data['clean'],'GT sheet $S$')
    box(ax,18,29,15,6,'Frozen AE\nencoder $E$','frozen',7.8)
    code(ax,59,27,10,9,r'$\bar z$')
    label(ax,64,39,'Target code',8,weight='bold')
    arrow(ax,[(11.8,32),(17.7,32)])
    arrow(ax,[(33.3,32),(58.7,32)])
    label(ax,45.5,34.5,r'fixed normalization $\mathcal{N}$',7.1)
    box(ax,60,21,8,3.5,'MSE','loss',8)
    arrow(ax,[(64,26.7),(64,24.8)])
    arrow(ax,[(64,19.8),(64,20.7)])

    iax=image_panel(fig,(.5,8.5,12,12),data['image'],'CT + prompts $P$',cmap='gray',vmax=255)
    points=data['prompts'];near=abs(points[:,2]-160)<=12
    iax.scatter(points[near,0],points[near,1],s=32,facecolors='none',edgecolors='#fff16c',linewidths=1.2)
    box(ax,18,11,12,6,'CT encoder\n+ context $C$',size=7.4)
    box(ax,34,11,5,6,'$F$','code',10)
    box(ax,43,11,11,6,'Point\ntransformer $T$',size=7.4)
    code(ax,59,10,10,9,r'$\widehat{\bar z}$')
    label(ax,64,8.2,'Predicted code',7.6)
    for start,end in [(12.8,17.7),(30.3,33.7),(39.3,42.7),(54.3,58.7)]:
        arrow(ax,[(start,14),(end,14)])
    arrow(ax,[(12.8,19),(14,19),(14,23),(48.5,23),(48.5,17.3)])
    label(ax,27.5,24.5,'Prompt coordinates',7.4)
    label(ax,6.5,6.4,r'$z\pm12$ slab shown',6.8)

    box(ax,42,1,12,5.8,'Binary decoder $B$\nresidual blocks',size=6.9)
    arrow(ax,[(36.5,10.7),(36.5,3.9),(41.7,3.9)])
    label(ax,57.5,3.9,'$V$',8)
    arrow(ax,[(54.3,3.9),(55.8,3.9)])
    box(ax,61,1,10,5.8,'BCE\n+ Dice','loss',7.2)
    arrow(ax,[(58.6,3.9),(60.7,3.9)])
    image_panel(fig,(75,.9,6,6),data['gt_union'])
    label(ax,89,3.8,'GT union $Y$',7.6)
    arrow(ax,[(74.7,3.9),(71.3,3.9)])

    # Inference is explicitly outside all training-loss paths.
    ax.add_patch(FancyBboxPatch((73,8),26.5,15,boxstyle='round,pad=.2',
        fill=False,edgecolor='#8d75a1',linestyle='--',linewidth=.9))
    label(ax,86.5,24.6,'INFERENCE ONLY',8,weight='bold',color='#765a8a')
    box(ax,76,12,9,6,'Frozen\n'+r'$D\circ\mathcal{N}^{-1}$','frozen',7.3)
    image_panel(fig,(89,9,10,10),data['predicted_sheet']>=.5)
    label(ax,94,21,r'Sheet $\widehat S$',7.8)
    arrow(ax,[(70.3,14),(75.7,14)])
    arrow(ax,[(85.3,14),(88.7,14)])
    for x,y,kind,text in [(76,36,'data','Data'),(88,36,'train','Trainable'),(76,31,'frozen','Frozen'),(88,31,'loss','Loss')]:
        box(ax,x,y,2.4,2.4,'',kind)
        label(ax,x+3.3,y+1.2,text,6.9,align='left')
    save(fig,out,'point_to_latent')

def _discovery_row(axes, data, distance_norm, *, plane_only):
    points, identity, groups = data['points'], data['identity'], data['cluster_id']
    palette = np.vstack(([.65,.65,.65,1], plt.get_cmap('tab20')(np.arange(20))))
    assert identity.max() < len(palette)
    cmap = ListedColormap(palette)
    norm = BoundaryNorm(np.arange(-.5,21.5), len(palette))
    titles = ['Foreground seeds', 'Full-code MSE', 'Clustered seeds', 'Decoded sheets']
    for ax, title in zip(axes, titles):
        ax.set_title(title, loc='left', fontsize=8, weight='bold', pad=8)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_box_aspect(1)
        for spine in ax.spines.values(): spine.set_visible(False)
    near = np.abs(points[:,2]-160) <= 12
    exact = points[:,2] == 160
    if plane_only:
        assert exact.all(), 'Every displayed seed must lie on the exact plane.'
    for col in (0,2):
        ax = axes[col]
        ax.imshow(data['image'], cmap='gray', vmin=0, vmax=255)
        for on_plane in (False,True):
            selected = near & (exact if on_plane else ~exact)
            colors = palette[identity[selected]] if col==2 else np.tile([1,.97,.65,1], (selected.sum(),1))
            ax.scatter(points[selected,0], points[selected,1], s=3.5 if plane_only else 12,
                facecolors=colors if on_plane else 'none', edgecolors=colors,
                linewidths=.2 if plane_only else .7)
        ax.set(xlim=(-.5,319.5), ylim=(319.5,-.5))
    # Ordering is for display only. Actual leader clustering uses full-code
    # distance to running centroids, with the same unchanged recipe in both rows.
    order = np.lexsort((np.arange(len(points)), groups, np.where(identity>0,identity,1000)))
    sorted_ids, sorted_groups = identity[order], groups[order]
    matrix = data['distance'][np.ix_(order,order)]
    im = axes[1].imshow(np.maximum(matrix,1e-5), cmap='magma_r', norm=distance_norm, interpolation='nearest')
    boundaries = np.r_[0,np.flatnonzero(np.diff(sorted_groups))+1,len(points)]
    for lo,hi in zip(boundaries[:-1],boundaries[1:]):
        if sorted_ids[lo]>0:
            axes[1].add_patch(Rectangle((lo-.5,lo-.5),hi-lo,hi-lo,fill=False,ec='white',lw=.35))
    cutoff = int((identity>0).sum())
    axes[1].axvline(cutoff-.5,color='#41b9c6',ls='--',lw=.65)
    axes[1].axhline(cutoff-.5,color='#41b9c6',ls='--',lw=.65)
    stripe = palette[sorted_ids][None,:,:]
    top = axes[1].inset_axes([0,1.005,1,.035]); top.imshow(stripe,aspect='auto'); top.axis('off')
    left = axes[1].inset_axes([-.04,0,.035,1]); left.imshow(stripe.transpose(1,0,2),aspect='auto'); left.axis('off')
    axes[3].imshow(data['image'],cmap='gray',vmin=0,vmax=255)
    axes[3].imshow(np.ma.masked_where(data['instances']==0,data['instances']),
                   cmap=cmap,norm=norm,alpha=.94,interpolation='nearest')
    return im


def discovery_figure(data, record, research, out):
    source = research/'runs_from_260914/evaluation/04_paper_automatic_hidden106_gpu0_gpu2/0058_kaggle/cases/sample_00860/instances.npz'
    with np.load(source) as f:
        volume = {**data, 'instances':f['inst'][:,:,160].T}
    folder = research/'runs_from_260914/evaluation/12_paper_slice_seed_example'
    with np.load(folder/'figure_data.npz') as f:
        planar = {key:f[key] for key in f.files}
    np.testing.assert_array_equal(volume['image'],planar['image'])
    distance_norm = LogNorm(vmin=1e-5,vmax=max(volume['distance'].max(),planar['distance'].max()))
    fig, axes = plt.subplots(2,4,figsize=(7.2,5.2),gridspec_kw={'width_ratios':[1,1.18,1,1]})
    fig.subplots_adjust(left=.005,right=.995,top=.89,bottom=.20,hspace=.70,wspace=.20)
    _discovery_row(axes[0],volume,distance_norm,plane_only=False)
    im = _discovery_row(axes[1],planar,distance_norm,plane_only=True)
    fig.canvas.draw()
    for row, row_data, title in [(0,volume,'A · Seeds sampled throughout the 3D crop'),
                                 (1,planar,'B · All seeds sampled on the displayed z=160 plane')]:
        top = max(ax.get_position().y1 for ax in axes[row])
        kept = int((row_data['identity']>0).sum())
        fig.text(.005,top+.075,title,fontsize=8.3,weight='bold',va='bottom')
        bottom = min(ax.get_position().y0 for ax in axes[row])
        x = (axes[row,1].get_position().x0+axes[row,1].get_position().x1)/2
        fig.text(x,bottom-.023,f'{kept} kept / {512-kept} discarded',ha='center',va='top',fontsize=7)
    upper_top = max(ax.get_position().y1 for ax in axes[0])
    fig.text(.005,upper_top+.043,'Filled markers: z=160. Hollow markers: z=148–172 slab.',fontsize=6.9,va='bottom')
    centers = [(ax.get_position().x0+ax.get_position().x1)/2 for ax in axes[1]]
    fig.text(centers[0],.105,'Encode CT once\nOne code per seed',ha='center',va='top',fontsize=7.1)
    cax = fig.add_axes([axes[1,1].get_position().x0,.104,axes[1,1].get_position().width,.012])
    cb = fig.colorbar(im,cax=cax,orientation='horizontal',ticks=[1e-5,1e-2,1])
    cb.ax.tick_params(labelsize=6.4,length=2,pad=1); cb.outline.set_linewidth(.4)
    fig.text(centers[1],.058,'MSE · shared log scale',ha='center',va='top',fontsize=6.7)
    fig.text(centers[2],.105,'Groups → multi-point\nprompts → code prediction',ha='center',va='top',fontsize=7.1)
    fig.text(centers[3],.105,'Frozen sheet decoder\n+ instance cleanup',ha='center',va='top',fontsize=7.1)
    for a,b in zip(centers,centers[1:]):
        fig.text((a+b)/2,.13,'→',ha='center',fontsize=11,color=COLORS['edge'])
    fig.text(.005,.008,'Same CT plane in both rows; 512 seeds per run. Instance colors are local to each row. Gray: discarded.',fontsize=7)
    save(fig,out,'automatic_discovery')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--research_root',type=Path,required=True)
    p.add_argument('--data_dir',type=Path)
    p.add_argument('--output_dir',type=Path,default=Path(__file__).resolve().parent/'figures')
    args=p.parse_args()
    folder=args.data_dir or args.research_root/'runs_from_260914/evaluation/11_paper_three_method_figures'
    record=json.loads((folder/'capture.json').read_text())
    with np.load(folder/'figure_data.npz') as f:data={k:f[k] for k in f.files}
    ae_figure(data,record,args.output_dir)
    point_figure(data,args.output_dir)
    discovery_figure(data,record,args.research_root,args.output_dir)
    print(args.output_dir)

if __name__=='__main__':main()
