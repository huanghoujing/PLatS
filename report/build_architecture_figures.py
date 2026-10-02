#!/usr/bin/env python3
"""Draw code-audited model diagrams; no experiment data or GPU is required."""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['svg.fonttype'] = 'none'  # Keep labels editable as text.
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

ROOT = Path(__file__).resolve().parent
COLORS = {'module': '#e5eff8', 'data': '#f5f6f7', 'frozen': '#e7e7e7',
          'loss': '#fff0cf', 'latent': '#dceee6', 'text': '#243447'}


def canvas(height=7.0):
    fig, ax = plt.subplots(figsize=(14, height))
    ax.set(xlim=(0, 14), ylim=(0, height))
    ax.axis('off')
    fig.subplots_adjust(left=.01, right=.99, top=.98, bottom=.02)
    return fig, ax


def box(ax, x, y, w, h, title, kind='module', size=13):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.035',
                              facecolor=COLORS[kind], edgecolor='#60758a', lw=1.2))
    ax.text(x+w/2, y+h/2, title, ha='center', va='center', fontsize=size,
            color=COLORS['text'], linespacing=1.25)


def arrow(ax, start, end, *, dashed=False, via=None):
    if via:
        points=[start, *via, end]
        ax.plot([p[0] for p in points[:-1]], [p[1] for p in points[:-1]],
                color='#60758a', lw=1.3, ls='--' if dashed else '-')
        start=points[-2]
    ax.annotate('', xy=end, xytext=start,
                arrowprops=dict(arrowstyle='->', lw=1.3, color='#60758a',
                                linestyle='--' if dashed else '-'))


def text(ax, x, y, value, size=13, **kw):
    ax.text(x, y, value, fontsize=size, color=COLORS['text'], **kw)


def save(fig, name):
    for extension in ('pdf', 'png', 'svg'):
        path = ROOT / 'figures' / f'{name}.{extension}'
        fig.savefig(path, dpi=220, facecolor='white')
        if extension == 'svg':
            path.write_text('\n'.join(line.rstrip() for line in path.read_text().splitlines()) + '\n')
    plt.close(fig)


def ae_detail():
    fig, ax = canvas(7.1)
    text(ax, .15, 6.78, 'AE: learn a single-sheet code and freeze it for P2SD', 18, weight='bold')
    row=[(.15,1.35,'Clean sheet\nS','data'),(1.9,1.5,'Corruption\nholes / dropout','data'),
         (3.8,2.15,'Sheet encoder E\ndense (reference)','module'),
         (6.35,1.4,'Code z\n64 × 10³','latent'),
         (8.15,1.7,'Latent noise\nz ⊙ (1 + ξ)','data'),
         (10.25,3.5,'Shared dense decoder\n5 × residual + upsample','module')]
    for x,w,title,kind in row: box(ax,x,5.5,w,.9,title,kind,12.5)
    for left,right in zip(row,row[1:]): arrow(ax,(left[0]+left[1],5.95),(right[0],5.95))
    text(ax,4.86,5.15,'Sparse occupied-voxel encoder retained as an alternative',11.5,ha='center')
    heads=[(.2,3.0,'Intermediate occupancy\n20³ / 40³ / 80³ / 160³'),
           (3.75,3.0,'Final occupancy\n320³ logits'),
           (7.3,3.0,'Dense unsigned distance\n320³ values'),
           (10.85,2.95,'Query distance MLP\nsampled code + query q')]
    for x,w,title in heads: box(ax,x,3.35,w,.8,title,size=12.5)
    # Decoder heads share features; intermediate heads attach at their stages.
    arrow(ax,(11.5,5.5),(1.7,4.15),via=[(11.5,4.83),(1.7,4.83)])
    arrow(ax,(12.1,5.5),(5.25,4.15),via=[(12.1,4.67),(5.25,4.67)])
    arrow(ax,(12.7,5.5),(8.8,4.15),via=[(12.7,4.5),(8.8,4.5)])
    # Query head reads the noisy code, not dense decoder features.
    arrow(ax,(9,5.5),(12.325,4.15),via=[(9,5.0),(13.9,5.0),(13.9,4.3),(12.325,4.3)])
    losses=['BCE + Dice\nweights .125 / .25 / .5 / .5',
            'BCE + Dice\n+ outside / border penalties',
            'Smooth L1\nweight .15', 'Smooth L1\nweight .15']
    targets=['Resized clean sheet','Clean sheet S','Clipped distance to S','Distance to S at q']
    for (x,w,_),loss,target in zip(heads,losses,targets):
        box(ax,x,2.05,w,.85,loss,'loss',12.5)
        arrow(ax,(x+w/2,3.35),(x+w/2,2.9))
        box(ax,x,1.02,w,.58,target,'data',12.2)
        arrow(ax,(x+w/2,1.6),(x+w/2,2.05),dashed=True)
    text(ax,.2,.52,'Code regularization: distinct sheets in one crop → cosine repulsion on z (weight .1, margin .5).',12.5)
    text(ax,.2,.13,'All reconstruction heads depend on the bottleneck. No encoder–decoder skips; no explicit topology loss.',12.5)
    save(fig,'ae_modules')


def p2sd_detail():
    fig, ax = canvas(9)
    text(ax,.15,8.65,'P2SD: shared image computation, then compact per-prompt prediction',18,weight='bold')
    box(ax,.2,7.25,1.35,.8,'CT X\n320³','data')
    box(ax,1.95,7.25,2.1,.8,'Image encoder\nfactor 32')
    box(ax,4.45,7.25,2.45,.8,'Image context\nprojection + 4 blocks')
    box(ax,7.35,7.25,2.1,.8,'Cached context F\n1,000 × 512','latent')
    for a,b in [((1.55,7.65),(1.95,7.65)),((4.05,7.65),(4.45,7.65)),((6.9,7.65),(7.35,7.65))]: arrow(ax,a,b)
    text(ax,10,7.75,'Computed once per CT crop',13,weight='bold')
    text(ax,10,7.3,'Reused across seed / prompt groups',12)
    box(ax,.2,5.35,1.35,.85,'K points P\npositive labels','data',12)
    box(ax,1.95,5.35,3.05,.85,'Prompt composition\nFourier coords + F(p) + label\nMLP → K × 512',size=12)
    box(ax,5.45,5.35,2.4,.85,'Prompt modulator\n1 prefix-attention block',size=12)
    box(ax,8.3,5.35,2.2,.85,'Latent refiner\n4 attention blocks',size=12)
    box(ax,10.95,5.35,2.85,.85,'Final prediction head\nLayerNorm + linear\nẑ normalized: 64 × 10³','latent',12)
    for a,b in [((1.55,5.77),(1.95,5.77)),((5,5.77),(5.45,5.77)),((7.85,5.77),(8.3,5.77)),((10.5,5.77),(10.95,5.77))]: arrow(ax,a,b)
    arrow(ax,(7.9,7.25),(3.47,6.2),via=[(7.9,6.8),(3.47,6.8)])
    arrow(ax,(8.55,7.25),(6.65,6.2),via=[(8.55,6.56),(6.65,6.56)])
    text(ax,5.52,5.05,'Keep grid outputs; discard prompt prefix',10.5)
    box(ax,8.3,3.98,2.2,.65,'Auxiliary code heads\nafter refiner blocks',size=11.8)
    arrow(ax,(9.4,5.35),(9.4,4.63))
    box(ax,10.95,3.98,2.85,.65,'Final + auxiliary MSE\nand sheet identity losses','loss',11.8)
    arrow(ax,(12.375,5.35),(12.375,4.63))
    arrow(ax,(10.5,4.305),(10.95,4.305))
    box(ax,.2,3.2,2,.7,'Clean GT sheet S','data',12)
    box(ax,2.65,3.2,2.35,.7,'Frozen AE encoder E','frozen',12)
    box(ax,5.45,3.2,2.4,.7,'Normalize code\nfixed channel μ / σ','frozen',12)
    for a,b in [((2.2,3.55),(2.65,3.55)),((5,3.55),(5.45,3.55))]:arrow(ax,a,b)
    arrow(ax,(7.85,3.55),(12.375,3.98),dashed=True,via=[(12.375,3.55)])
    # Binary auxiliary path branches from unprompted image context.
    box(ax,.2,1.55,2.4,.8,'Binary context refiner\n4 blocks on full F',size=11.8)
    arrow(ax,(5.0,7.25),(1.4,2.35),via=[(5.0,6.98),(.06,6.98),(.06,2.6),(1.4,2.6)])
    box(ax,3.05,1.55,3,.8,'Binary decoder + head\n5³ context crop → 160³\n5 convolutional upsamples',size=11.8)
    box(ax,6.5,1.55,2.6,.8,'Union BCE + Dice\nignore-masked GT union','loss',11.8)
    arrow(ax,(2.6,1.95),(3.05,1.95));arrow(ax,(6.05,1.95),(6.5,1.95))
    box(ax,10.2,1.3,3.6,1.12,'Inference / validation only\nDenormalize ẑ → frozen AE D\n→ full-resolution sheet','frozen',12)
    arrow(ax,(13.8,5.78),(13.8,2.42),dashed=True,via=[(13.94,5.78),(13.94,2.62),(13.8,2.62)])
    text(ax,.2,.76,'Blue: trained modules    Green: codes / context    Gray: data or frozen (explicitly labeled)    Gold: losses',11.8)
    text(ax,.2,.32,'No full-resolution sheet decode for the latent training loss. The binary branch still decodes one crop per CT batch.',12)
    save(fig,'p2sd_modules')


if __name__ == '__main__':
    ae_detail()
    p2sd_detail()
