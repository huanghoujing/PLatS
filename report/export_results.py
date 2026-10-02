"""Export aggregate-only paper tables and a readable local evaluation report."""
import argparse
import csv
import hashlib
import html
import json
from pathlib import Path

LABELS = {'winner': 'Winner', 'winner_close_dust': 'Winner + close + dust',
          'plats_base': 'PLatS', 'plats_ae': 'PLatS + AE',
          'plats_ae_close': 'PLatS + AE + close',
          'plats_ae_close_dust': 'PLatS + repair'}
FIELDS = ('dice', 'surface_dice_tau2', 'toposcore', 'voi_score', 'leaderboard_formula_score')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def leaderboard_table(means, variant):
    rows = []
    for arm in ('winner', 'plats_base', 'plats_ae_close_dust'):
        metrics = means[arm][variant]
        rows.append(LABELS[arm] + ' & ' + ' & '.join(f'{metrics[k]:.4f}' for k in FIELDS) + r' \\')
    if variant == 'standard':
        caption = r'Original source-ignore erasure on all 106 released test cubes. Equal-case means; automatic inference, no GT prompts. Repair adds AE reconstruction, guarded closing and dusting before erasure. LB is reproduced locally.'
        label = 'tab:lb'
    else:
        caption = r'Restoration diagnostic on the same 106 cases, using the full-size 5,000-voxel instance gate. All terms use restored predictions and erased GT. Repair precedes restoration. This is not official LB.'
        label = 'tab:modified'
    return '\n'.join([r'\begin{table}[t]', r'\centering', r'\small',
        r'\setlength{\tabcolsep}{3pt}', r'\begin{tabular}{lrrrrr}', r'\toprule',
        r'Method & Dice & SD$_2$ & Topo & VOI & LB \\', r'\midrule', *rows,
        r'\bottomrule', r'\end{tabular}', r'\caption{' + caption + '}',
        r'\label{' + label + '}', r'\end{table}', ''])


def betti_table(means):
    rows = []
    for arm in ('winner', 'plats_base', 'plats_ae_close_dust'):
        b = means[arm]['raw_betti']
        rows.append(f"{LABELS[arm]} & {b['mean'][1]:.3f} & {b['median'][1]:.1f} & {b['mean'][2]:.3f} & {b['median'][2]:.1f}" + r' \\')
    return '\n'.join([r'\begin{table}[t]', r'\centering', r'\small',
        r'\setlength{\tabcolsep}{4pt}', r'\begin{tabular}{lrrrr}', r'\toprule',
        r' & \multicolumn{2}{c}{$b_1$: tunnels} & \multicolumn{2}{c}{$b_2$: cavities} \\',
        r'Method & Mean & Median & Mean & Median \\', r'\midrule', *rows,
        r'\bottomrule', r'\end{tabular}',
        r'\caption{Full prediction unions before ignore processing on the same 106 native crops. V-construction Betti counts use six-connected foreground and equal case weights. Repair is part of inference; no erasure, restoration or 5,000-voxel evaluation filter is applied here.}',
        r'\label{tab:rawpredbetti}', r'\end{table}', ''])


def ablation_table(means):
    rows = []
    for arm in LABELS:
        entry = means[arm]
        values = [entry[v][k] for v in ('standard', 'restored_filtered')
                  for k in ('surface_dice_tau2', 'toposcore', 'voi_score', 'leaderboard_formula_score')]
        rows.append(LABELS[arm] + ' & ' + ' & '.join(f'{v:.4f}' for v in values)
                    + f" & {entry['raw_betti']['mean'][1]:.3f} & {entry['raw_betti']['mean'][2]:.3f}" + r' \\')
    return '\n'.join([r'\begin{table*}[t]', r'\centering', r'\scriptsize',
        r'\setlength{\tabcolsep}{3pt}', r'\begin{tabular}{lrrrrrrrrrr}', r'\toprule',
        r' & \multicolumn{4}{c}{Original ignore erasure} & \multicolumn{4}{c}{Restoration diagnostic} & \multicolumn{2}{c}{Before ignore} \\',
        r'Method & SD$_2$ & Topo & VOI & LB & SD$_2$ & Topo & VOI & Formula & Mean $b_1$ & Mean $b_2$ \\',
        r'\midrule', *rows, r'\bottomrule', r'\end{tabular}',
        r'\caption{Fixed postprocessing ablation on all 106 cases. AE is the existing 0032 model; close denotes guarded per-ID closing; dust removes six-connected pieces of at most five voxels. The final PLatS repair row includes all three steps. Winner cleanup uses its pre-cleanup six-connected components as ID proxies. The 5,000-voxel whole-ID gate applies only in the restoration diagnostic.}',
        r'\label{tab:postprocessing}', r'\end{table*}', ''])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--paper', type=Path)
    args = parser.parse_args()
    result = json.loads((args.run / 'results.json').read_text())
    means = result['means']
    aggregate = dict(status='complete', cases=106, means=means,
                     recipe=result['plan']['recipe'], limitations=result['plan']['limitations'],
                     source_results_sha256=sha(args.run / 'results.json'),
                     plan_sha256=sha(args.run / 'plan.json'),
                     prediction_freeze_sha256=result['prediction_freeze_sha256'],
                     ae_checkpoint_sha256=result['plan']['ae_checkpoint_sha256'])
    (args.run / 'aggregate_results.json').write_text(json.dumps(aggregate, indent=2) + '\n')
    with (args.run / 'scores.csv').open('w') as handle:
        writer = csv.writer(handle)
        writer.writerow(['arm', 'case_id', 'protocol', *FIELDS, 'raw_b0', 'raw_b1', 'raw_b2'])
        for row in result['rows']:
            for protocol, metrics in row['metrics'].items():
                writer.writerow([row['arm'], row['case_id'], protocol, *[metrics[k] for k in FIELDS], *row['raw_betti']])
    page = '''<!doctype html><meta charset="utf-8"><title>106-case postprocessing evaluation</title>
<style>body{font:16px system-ui;max-width:1200px;margin:40px auto;padding:0 20px;line-height:1.5}table{border-collapse:collapse}th,td{border:1px solid #ccc;padding:8px;text-align:right}th:first-child,td:first-child{text-align:left}</style>
<h1>PLatS repair: all 106 released test crops</h1>
<p>Fixed 0032 AE reconstruction, probability ownership on overlaps, additive guarded per-ID closing (3³ then 5³), and per-ID removal of six-connected pieces ≤5 voxels. No GT or ignore masks enter inference. The recipe was selected after inspecting three released-test outliers; this is not a fresh blind evaluation. Original outputs are retained. Standard LB erases source ignore; restoration instead retains visibly supported IDs with full size ≥5,000 and compares restored masks against erased GT.</p>'''
    for protocol, title in [('standard', 'Original source-ignore scoring'), ('restored_filtered', 'Restoration diagnostic; not official LB')]:
        page += '<h2>' + title + '</h2><table><tr><th>Method</th><th>Dice</th><th>Surface Dice</th><th>Topo</th><th>VOI</th><th>Formula</th></tr>'
        for arm in LABELS:
            page += '<tr><td>' + html.escape(LABELS[arm]) + '</td>' + ''.join(f'<td>{means[arm][protocol][k]:.6f}</td>' for k in FIELDS) + '</tr>'
        page += '</table>'
    page += '<h2>Whole-union topology before ignore processing</h2><table><tr><th>Method</th><th>Mean b0</th><th>Mean b1</th><th>Median b1</th><th>Mean b2</th><th>Median b2</th></tr>'
    for arm in LABELS:
        b = means[arm]['raw_betti']
        vals = [b['mean'][0], b['mean'][1], b['median'][1], b['mean'][2], b['median'][2]]
        page += '<tr><td>' + html.escape(LABELS[arm]) + '</td>' + ''.join(f'<td>{v:.6f}</td>' for v in vals) + '</tr>'
    page += '</table><p><a href="results.json">Full local results</a> · <a href="scores.csv">Per-case CSV</a> · <a href="plan.json">Frozen recipe</a> · <a href="prediction_freeze.json">Output hashes</a></p><p>Each predictions/arm/case folder contains native instance masks and NIFTIs with linked image, annotation, and source-ignore viewing assets. PLatS folders also link the sampled seeds.</p>'
    (args.run / 'report.html').write_text(page)
    if args.paper:
        (args.paper / 'tables/standard_lb.tex').write_text(leaderboard_table(means, 'standard'))
        (args.paper / 'tables/modified_lb.tex').write_text(leaderboard_table(means, 'restored_filtered'))
        (args.paper / 'tables/raw_prediction_betti.tex').write_text(betti_table(means))
        (args.paper / 'tables/postprocessing_ablation.tex').write_text(ablation_table(means))
        (args.paper / 'postprocessing_results.json').write_text(json.dumps(aggregate, indent=2) + '\n')


if __name__ == '__main__':
    main()
