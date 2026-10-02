#!/usr/bin/env python3
"""Draw the two-stage overview and measured inference panels for the paper."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import FancyBboxPatch
import numpy as np
from scipy.stats import rankdata

HERE = Path(__file__).resolve().parent
plt.rcParams.update({'pdf.fonttype': 42, 'svg.fonttype': 'none', 'font.size': 8.5})
COLORS = {'train': '#e3eff9', 'frozen': '#e5e7eb', 'data': '#f5f6f7',
          'code': '#dceee6', 'loss': '#fff1d9', 'line': '#415b73'}


def box(ax, x, y, width, height, label, kind='train', fontsize=8.5):
    ax.add_patch(FancyBboxPatch((x, y), width, height,
                 boxstyle='round,pad=.025', facecolor=COLORS[kind],
                 edgecolor=COLORS['line'], linewidth=.8))
    ax.text(x + width / 2, y + height / 2, label,
            ha='center', va='center', fontsize=fontsize)


def arrow(ax, start, end, dashed=False):
    ax.annotate('', xy=end, xytext=start, arrowprops={
        'arrowstyle': '->', 'lw': .9, 'color': COLORS['line'],
        'linestyle': '--' if dashed else '-'})


def save(fig, output, name):
    output.mkdir(parents=True, exist_ok=True)
    for extension in ('pdf', 'png', 'svg'):
        path = output / f'{name}.{extension}'
        fig.savefig(path, dpi=240, bbox_inches='tight', pad_inches=.04,
                    metadata={'Creator': 'PLatS measured method figures'})
        if extension == 'svg':
            path.write_text('\n'.join(line.rstrip() for line in path.read_text().splitlines()) + '\n')
    plt.close(fig)


def training_overview(output):
    fig, ax = plt.subplots(figsize=(7.1, 3.75))
    ax.set(xlim=(0, 7.1), ylim=(0, 3.75))
    ax.axis('off')
    fig.subplots_adjust(left=.005, right=.995, top=.995, bottom=.005)

    ax.text(.06, 3.53, '(a) Stage 1: learn the sheet AE', weight='bold', fontsize=10)
    nodes = [(.06, 1.02, 'Corrupted\nsheet mask', 'data'),
             (1.39, .98, 'Encoder E', 'train'),
             (2.68, .72, 'Code z', 'code'),
             (3.72, 1.18, 'Decoder D\n+ voxel heads', 'train'),
             (5.21, 1.78, 'Sheet Q + dense\ndistance U', 'data')]
    for x, width, label, kind in nodes:
        box(ax, x, 2.86, width, .45, label, kind)
    for left, right in zip(nodes, nodes[1:]):
        arrow(ax, (left[0] + left[1] + .025, 3.085), (right[0] - .025, 3.085))
    box(ax, .06, 2.18, 2.3, .39, 'Code repulsion + spatial penalties', 'loss', 8)
    box(ax, 2.68, 2.18, 1.30, .39, 'Query head h(z, q)', fontsize=8)
    box(ax, 4.32, 2.18, 2.67, .39, 'BCE + Dice / Smooth L1\nclean sheet + distance targets', 'loss', 8)
    arrow(ax, (3.04, 2.835), (3.33, 2.60))
    arrow(ax, (4.01, 2.375), (4.29, 2.375))
    arrow(ax, (6.1, 2.835), (6.1, 2.60))
    ax.text(.08, 2.70, 'No encoder–decoder skips', fontsize=7.8, color=COLORS['line'])
    ax.plot([0, 7.1], [2.01, 2.01], color='#b9c2cc', lw=.6)

    ax.text(.06, 1.81, '(b) Stage 2: freeze the AE; learn point-to-latent prediction',
            weight='bold', fontsize=10)
    box(ax, .06, 1.26, .99, .35, 'Clean sheet S', 'data', 8.2)
    box(ax, 1.39, 1.26, 1.18, .35, 'Frozen N ∘ E', 'frozen', 8.3)
    box(ax, 3.62, 1.26, 2.14, .35, 'Code MSE + identity losses', 'loss', 8.2)
    arrow(ax, (1.08, 1.435), (1.36, 1.435))
    arrow(ax, (2.60, 1.435), (3.59, 1.435), dashed=True)
    ax.text(3.10, 1.48, r'$\bar z$', ha='center', fontsize=9)

    nodes = [(.06, .99, 'CT crop X', 'data'),
             (1.39, 1.18, 'CT encoder +\ncontext C', 'train'),
             (2.91, 1.40, 'Point transformer T\n+ code heads', 'train'),
             (4.64, .89, r'Code $\widehat{\bar z}$', 'code'),
             (5.86, 1.13, 'Frozen N⁻¹, D\nsheet at inference', 'frozen')]
    for x, width, label, kind in nodes:
        box(ax, x, .57, width, .43, label, kind, 8)
    for left, right in zip(nodes, nodes[1:]):
        arrow(ax, (left[0] + left[1] + .025, .785), (right[0] - .025, .785))
    arrow(ax, (5.08, 1.025), (5.08, 1.23), dashed=True)
    box(ax, .06, .04, 1.10, .33, 'Binary head B', fontsize=8.2)
    box(ax, 1.46, .04, 1.25, .33, 'Union BCE + Dice', 'loss', 7.7)
    box(ax, 2.98, .04, 1.32, .33, 'Positive points P', 'data', 8.2)
    arrow(ax, (1.75, .54), (.61, .40))
    arrow(ax, (1.19, .205), (1.43, .205))
    arrow(ax, (3.64, .40), (3.64, .54))
    ax.text(4.64, .14, 'Blue: trainable   Gray: frozen', fontsize=7.8, color=COLORS['line'])
    save(fig, output, 'training_overview')


def measured_distances(research_root, expected):
    base = research_root / 'runs_from_260914/evaluation/07_paper0058_latent_separability_gpu2/cases'
    same, different = [], []
    paths = sorted(base.glob('*/distances.npz'))
    for path in paths:
        with np.load(path, allow_pickle=False) as archive:
            distance, labels = archive['distance'], archive['labels']
        left, right = np.triu_indices(len(labels), 1)
        values = distance[left, right]
        equal = labels[left] == labels[right]
        same.append(values[equal])
        different.append(values[~equal])
    same, different = np.concatenate(same), np.concatenate(different)
    assert len(paths) == expected['cases'] == 106
    assert len(same) == expected['intra_pairs'] and len(different) == expected['inter_pairs']
    np.testing.assert_allclose(np.percentile(same, [50, 90, 99]), expected['intra_p50_p90_p99'])
    np.testing.assert_allclose(np.percentile(different, [1, 50]), expected['inter_p01_p50'])
    n, m = len(same), len(different)
    auc = (rankdata(-np.r_[same, different])[:n].sum() - n * (n + 1) / 2) / (n * m)
    np.testing.assert_allclose(auc, expected['pair_auroc'], rtol=0, atol=1e-8)
    return same, different


def inference_figure(research_root, output):
    snapshot = json.loads((HERE / 'results_snapshot.json').read_text())
    base = research_root / 'runs_from_260914/evaluation/08_paper_method_visuals'
    record = json.loads((base / 'provenance.json').read_text())
    case, z = record['case'], record['native_z']
    meta = json.loads((research_root / 'datasets/hf_kaggle_202607/cases' / case / 'meta.json').read_text())
    image = np.load(meta['image_path'], mmap_mode='r')[:, :, z].T
    source = research_root / 'runs_from_260914/evaluation/04_paper_automatic_hidden106_gpu0_gpu2/0058_kaggle/cases' / case
    with np.load(source / 'instances.npz', allow_pickle=False) as archive:
        instances = archive['inst'][:, :, z].T
    with np.load(base / 'intermediates.npz', allow_pickle=False) as archive:
        points, identity = archive['points_xyz'], archive['identity']
        projection = archive['pca']
        variance = float(archive['pca_variance_fraction'])
    assert len(points) == record['sampled_points'] == 512
    assert int((identity > 0).sum()) == record['retained_seed_points']
    same, different = measured_distances(research_root, snapshot['latent_separability'])

    # Predicted final IDs, never GT identities, supply the shared palette for
    # seeds, PCA clusters and decoded masks. Gray marks discarded hypotheses.
    colors = np.vstack(([.70, .70, .70, 1.], plt.get_cmap('tab20')(np.arange(20))))
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(np.arange(-.5, 21.5), cmap.N)
    fig, axes = plt.subplots(1, 4, figsize=(7.5, 2.60), gridspec_kw={'width_ratios': [1, 1.12, 1, 1]})
    titles = ['(a) Foreground seeds', '(b) Code distances', '(c) Predicted clusters', '(d) Decoded sheets']
    notes = ['Final ID colors', 'GT prompts · 106 crops', f'PCA display ({variance:.0%})', 'Same slice / ID colors']
    for ax, title in zip(axes, titles):
        ax.set_title(title, fontsize=8.4, weight='bold', loc='left', pad=8)
        ax.set_box_aspect(1)
    for index in (0, 2, 3):
        axes[index].set_xticks([])
        axes[index].set_yticks([])
        for spine in axes[index].spines.values():
            spine.set_visible(False)

    axes[0].imshow(image, cmap='gray', vmin=0, vmax=255, interpolation='nearest')
    near = np.abs(points[:, 2] - z) <= 12
    axes[0].scatter(points[near, 0], points[near, 1], s=16, c=identity[near], cmap=cmap,
                    norm=norm, edgecolors='#1e252a', linewidths=.3)

    ax = axes[1]
    # Plot both complete empirical CDFs over common log-spaced coordinates.
    # Separate normalization prevents the more numerous different-sheet pairs
    # from visually overwhelming same-sheet pairs. No test-set threshold fit.
    minimum = min(float(v[v > 0].min()) for v in (same, different))
    maximum = max(float(same.max()), float(different.max()))
    grid = np.geomspace(minimum / 1.2, maximum * 1.2, 350)
    for values, label, color, style in [(same, 'Same sheet', '#1f2937', '-'),
                                        (different, 'Different sheet', '#969696', '--')]:
        fraction = np.searchsorted(np.sort(values), grid, side='right') / len(values)
        ax.plot(grid, fraction, label=label, color=color, linestyle=style, linewidth=1.4)
    ax.set_xscale('log')
    ax.set(xlim=(grid[0], grid[-1]), ylim=(0, 1.02))
    ax.set_yticks([0, .5, 1])
    ax.set_xlabel('Full-code MSE', fontsize=7.8, labelpad=1.5)
    ax.set_ylabel('Pair fraction', fontsize=7.8, labelpad=1.5)
    ax.tick_params(labelsize=7, length=2)
    ax.legend(loc='upper left', fontsize=6.7, frameon=False, handlelength=1.3, borderaxespad=.1)
    ax.spines[['top', 'right']].set_visible(False)

    dropped = identity == 0
    axes[2].scatter(*projection[dropped].T, s=6, c=[colors[0]], linewidths=0, rasterized=True)
    axes[2].scatter(*projection[~dropped].T, s=10, c=identity[~dropped], cmap=cmap,
                    norm=norm, edgecolors='white', linewidths=.15, rasterized=True)
    axes[2].margins(.08)
    axes[3].imshow(image, cmap='gray', vmin=0, vmax=255, interpolation='nearest')
    axes[3].imshow(np.ma.masked_where(instances == 0, instances), cmap=cmap, norm=norm,
                    alpha=.94, interpolation='nearest')
    fig.subplots_adjust(left=.005, right=.995, bottom=.25, top=.85, wspace=.30)
    fig.canvas.draw()
    for ax, note in zip(axes, notes):
        position = ax.get_position()
        fig.text((position.x0 + position.x1) / 2, .05, note,
                 ha='center', va='bottom', fontsize=7, color=COLORS['line'])
    save(fig, output, 'inference_relationships')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--research_root', type=Path)
    parser.add_argument('--output_dir', type=Path, default=HERE / 'figures')
    parser.add_argument('--training_only', action='store_true')
    args = parser.parse_args()
    if not args.training_only and args.research_root is None:
        parser.error('--research_root is required for measured inference panels')
    training_overview(args.output_dir)
    if not args.training_only:
        inference_figure(args.research_root, args.output_dir)
    print(args.output_dir.resolve())


if __name__ == '__main__':
    main()
