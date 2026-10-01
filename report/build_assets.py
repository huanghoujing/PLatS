#!/usr/bin/env python3
"""Snapshot completed measurements and build the report's tables and figures.

Run with the project virtualenv. This never starts inference or changes a run.
Incomplete new comparisons remain visibly pending until all cases finish.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
from datetime import datetime, timezone

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
EVAL = ROOT/'runs_from_260914/evaluation/03_paper_hidden106_three_checkpoints_gpu0_gpu2'
AUTO = ROOT/'runs_from_260914/evaluation/04_paper_automatic_hidden106_gpu0_gpu2'
FFN = ROOT/'runs_from_260914/first_letters/22_paper_ffn_three_checkpoints_onepoint_gpu0_gpu2'
LABELS = {'0058_kaggle':'0058', 'original_0076':'0076', '0076_plus_100k':'0076+100k'}
FIELDS = ('dice', 'surface_dice_tau2', 'toposcore', 'voi_score', 'leaderboard_formula_score')


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def table(name, environment, caption, label, spec, header, rows):
    text = [f'\\begin{{{environment}}}[t]', '\\centering', '\\small',
        '\\setlength{\\tabcolsep}{4pt}', f'\\begin{{tabular}}{{{spec}}}', '\\toprule',
        header, '\\midrule', *[r+r' \\' for r in rows], '\\bottomrule',
        '\\end{tabular}', '\\caption{'+caption+'}', '\\label{'+label+'}',
        f'\\end{{{environment}}}']
    (HERE/'tables'/f'{name}.tex').write_text('\n'.join(text)+'\n')


def snapshot():
    sources = {}
    def capture(path):
        payload = Path(path).read_bytes()
        value = json.loads(payload)
        sources[str(path.relative_to(ROOT))] = hashlib.sha256(payload).hexdigest()
        return value
    summaries = {}
    points = {}
    for label in LABELS:
        summaries[label] = capture(EVAL/label/'points_08/summary.json')
        assert summaries[label]['status']=='complete'
        assert summaries[label]['case_variant_count']==106 and summaries[label]['sheet_prompt_count']==764
        points[label] = {}
        for n in (1, 2, 4, 8):
            path = EVAL/label/f'points_{n:02d}/summary.json'
            if path.exists() and read(path).get('status')=='complete':
                current = capture(path)
                assert current['case_variant_count']==106 and current['sheet_prompt_count']==764
                points[label][str(n)] = current['sheet']['leaderboard_formula_score']['mean']
    automatic = None
    if (AUTO/'results.json').exists() and read(AUTO/'results.json').get('status')=='complete':
        automatic = capture(AUTO/'results.json')
        assert automatic['cases']==106 and automatic['checkpoint_count']==3
        assert len(automatic['rows'])==424
    data = dict(created_utc=datetime.now(timezone.utc).isoformat(),
        code_revision=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        code_revision_description='Base revision before this report revision; source SHA256 hashes identify the current controller and figure code.',
        asset_builder_sha256=sha(Path(__file__)),
        prompted_eight=summaries, completed_prompt_formula=points,
        raw_topology=capture(EVAL/'paper_analysis/betti_census.json')['summaries'],
        ffn=capture(FFN/'results.json'), ffn_raw_topology=capture(FFN/'raw_betti.json'),
        automatic=automatic, automatic_status=capture(AUTO/'status.json'),
        ink_model_settings=capture(ROOT/'configs/ink/first_letters_baseline.json'),
        eight_point_verification=capture(EVAL/'eight_point_verification.json'),
        checkpoint_plan=capture(EVAL/'plan.json'), ffn_plan=capture(FFN/'plan.json'),
        source_sha256=sources)
    data['full_prompt_verification']=capture(EVAL/'verification.json')
    data['paired_case_bootstrap']=capture(EVAL/'paper_analysis/paired_case_bootstrap.json')
    failure=ROOT/'runs_from_260914/evaluation/06_paper_failure_analysis'
    data['ignore_cut_census']=capture(failure/'results.json')
    data['binary_failure_selection']=capture(failure/'binary_figure_selection.json')
    data['ignore_failure_selection']=capture(failure/'ignore_figure_selection.json')
    data['failure_rendering']=capture(failure/'2d_slices/provenance.json')
    winner=ROOT/'runs_from_260914/evaluation/05_paper_winner_ignore106_gpu0_gpu2'
    data['winner_plan']=capture(winner/'plan.json')
    data['winner_inference_freeze']=capture(winner/'inference_complete.json')
    data['winner_comparison']=capture(winner/'results.json') if (winner/'results.json').exists() else None
    reading=ROOT/'runs_from_260914/first_letters/24_paper_winner_onepoint_gpu0_gpu2'
    data['winner_reading_plan']=capture(reading/'plan.json')
    data['winner_reading']=capture(reading/'results.json') if (reading/'results.json').exists() else None
    for name in ('prediction_freeze','gallery_provenance'):
        if (reading/f'{name}.json').exists():data['winner_reading_'+name]=capture(reading/f'{name}.json')
    large=ROOT/'runs_from_260914/first_letters/23_paper_large0058_gpu0_gpu2'
    data['large_page_plan']=capture(large/'plan.json')
    data['large_page_evaluation']=capture(large/'results.json') if (large/'results.json').exists() else None
    data['spiral_integration']=capture(ROOT/'runs_from_260914/first_letters/21_p2sd_spiral_inputs_gpu0_gpu2/results.json')
    data['method_visuals']=capture(ROOT/'runs_from_260914/evaluation/08_paper_method_visuals/provenance.json')
    data['latent_separability']=capture(ROOT/'runs_from_260914/evaluation/07_paper0058_latent_separability_gpu2/results.json')
    data['source_code_sha256']={str(p.relative_to(ROOT)):sha(p) for p in [
        HERE/'build_paper_figures.py', HERE/'build_failure_figures.py',
        ROOT/'scripts/paper_inference_figure_capture.py',
        *[ROOT/'scripts'/f'paper_{name}.py' for name in (
            'winner_ignore_benchmark','winner_gpu_adapter','failure_analysis',
            'latent_separability','large_0058','large0058_eval','winner_reading')]]}
    return data


def main_and_supplementary_tables(data):
    """Keep main model comparisons concise; full checkpoint matrix at the end."""
    comparison=data['winner_comparison']
    for name,variant,caption,label in [
        ('standard_lb','standard','Original source-ignore erasure on all 106 released test cubes. Equal-case means; automatic inference, no GT prompts. LB is the public formula reproduced locally.','tab:lb'),
        ('modified_lb','restored_filtered','Restoration diagnostic on the same 106 cases, using the full-size 5,000-voxel gate. All three terms use restored predictions and erased GT. This is not official LB.','tab:modified')]:
        rows=[]
        for arm,title in [('winner','Winner'),('plats_auto','PLatS')]:
            vals=None if comparison is None else comparison['means'][arm][variant]
            rows.append(title+' & '+' & '.join('--' if vals is None else f'{vals[k]:.4f}' for k in FIELDS))
        table(name,'table',caption,label,'lrrrrr',r'Method & Dice & SD$_2$ & Topo & VOI & LB \\',rows)
    if comparison is None:
        (HERE/'tables/restoration_interpretation.tex').write_text('')
        (HERE/'tables/restoration_controls.tex').write_text('')
    else:
        m=comparison['means'];change={a:m[a]['restored_filtered']['leaderboard_formula_score']-m[a]['standard']['leaderboard_formula_score'] for a in ('winner','plats_auto')}
        (HERE/'tables/restoration_interpretation.tex').write_text(
          'The winner retains the higher combined score under both protocols. '
          f'Restored TopoScore is {m["plats_auto"]["restored_filtered"]["toposcore"]:.4f} for PLatS versus {m["winner"]["restored_filtered"]["toposcore"]:.4f} for the winner. '
          f'The restored-formula change is {change["winner"]:+.4f} for the winner and {change["plats_auto"]:+.4f} for automatic PLatS. '
          'Changes combine erasure sensitivity, filtering and penalties for unknown foreground; this is not a new challenge ranking.\n')
        rows=[]
        for a,title in [('winner','Winner'),('plats_auto','PLatS auto'),('plats_gt8','PLatS GT8 union')]:
            v=m[a];g=[r for r in comparison['rows'] if r['arm']==a]
            isolated=float(np.mean([r['topology_isolation_score'] for r in g]))
            vals=[v['standard']['leaderboard_formula_score'],v['filtered']['leaderboard_formula_score'],v['restored_filtered']['leaderboard_formula_score'],isolated]
            rows.append(title+' & '+' & '.join(f'{x:.4f}' for x in vals))
        table('restoration_controls','table*','Ignore-restoration controls. GT8 union uses deterministic sheet-ID ownership on overlaps and retains its original union exactly. It uses GT prompts and is not an automatic discovery result. The topology-only hybrid replaces only TopoScore.','tab:restorecontrols','lrrrr',r'Method & Standard LB & Filter only & Restore + filter & Topology-only diagnostic \\',rows)
    rows=[]
    for a,title in [('ffn','FFN'),('0058_kaggle','PLatS')]:
        group=[r for r in data['ffn']['rows'] if r['method']==a and not r['case'].startswith('pherc0814')]
        vals=np.mean([r['geometry']['reference_to_pred_p50_p90'] for r in group],axis=0)
        rows.append(title+' & '+' & '.join(f'{x:.3f}' for x in vals))
    table('ffn_main','table','Same one seed per crop. Reference-to-prediction distances in voxels, averaged over four evaluation crops.','tab:ffnmain','lrr',r'Method & Median & p90 \\',rows)
    s=data['latent_separability']
    (HERE/'tables/latent_separability.tex').write_text(
       f'On {s["cases"]} cases and {s["points"]:,} fixed GT points, the median same-band MSE is {s["intra_p50_p90_p99"][0]:.4f}, '
       f'versus {s["inter_p01_p50"][1]:.4f} between bands. Pair AUROC is {s["pair_auroc"]:.4f}; '
       f'mean per-case nearest-band accuracy is {100*s["nearest_band_accuracy_case_mean"]:.2f}\\%. '
       f'At the inherited 0.02 threshold, {100*s["intra_over_002"]:.2f}\\% of same-band pairs exceed the threshold and '
       f'{100*s["inter_under_002"]:.2f}\\% of different-band pairs fall below it.\n')
    for kind,name,label in [('sheet','prompt_matrix','tab:promptmatrix'),('union','union_matrix','tab:unionmatrix')]:
        rows=[]
        for a,title in LABELS.items():
            for n in (1,2,4,8):
                s=read(EVAL/a/f'points_{n:02d}/summary.json')
                assert s['status']=='complete'
                rows.append(title+f' & {n} & '+' & '.join(f'{s[kind][k]["mean"]:.4f}' for k in FIELDS))
        table(name,'table*',f'Complete {kind} results: all three checkpoints, all four nested prompt counts. Annotated-box ignore; no prediction cleanup. Formula scores are diagnostics under this policy.',''+label,'lrrrrrr',r'Checkpoint & Points & Dice & SD$_2$ & Topo & VOI & Formula \\',rows)
    rows=[]
    if data['large_page_evaluation'] is not None:
        for region,r in data['large_page_evaluation']['regions'].items():
            g=r['geometry']['growth'];dist=r['geometry']['target_distance_median_p90_voxels']
            rows.append(region.replace('_',r'\_')+f' & {r["complete"]["area_mm2"]/100:.2f} & {g["accepted"]}/{g["attempted"]} & {g["typical_accepted_join_median_voxels"]:.3f} & {dist[0]:.3f} & {dist[1]:.3f}')
    table('large_pages','table*','0058 large-page outputs. Areas are reconstructed mesh areas; joins include accepted windows only. Target distances exclude reference-boundary samples and are measured after all ink outputs are frozen.','tab:largepages','lrrrrr',r'Page & Area cm$^2$ & Accepted/attempted & Join median (vox) & Target median & Target p90 \\',rows)
    rows=[]
    integration=data['spiral_integration']['regions']['page_a']
    for a,title in [('p2sd','Direct PLatS'),('spiral','Published tracks'),('p2sd_only','PLatS patches only'),('m7_plus_p2sd','Tracks + PLatS')]:
        r=integration[a];ink=integration['ink'][a];dist=r['geometry']['target_distance_median_p90_voxels']
        vals=[r['complete']['area_mm2']/100,*dist,ink['common']['ink_AP'],ink['common']['ink_Dice'],ink['labeled_coverage'],ink['delivered']['ink_Dice']]
        rows.append(title+' & '+' & '.join(f'{x:.3f}' if i<3 else f'{x:.4f}' for i,x in enumerate(vals)))
    table('spiral_integration','table*','Original 0076 integration pilot on page A. Common ink uses identical pixels across all four arms; coverage and delivered Dice expose missing output. Published tracks are not verified as the Kaggle winner.','tab:spiralintegration','lrrrrrrr',r'Input/method & Area cm$^2$ & Median vox & p90 vox & Ink AP & Ink Dice & Coverage & Delivered Dice \\',rows)
    rows=[]
    if data['winner_reading'] is not None:
        group=[r for r in data['winner_reading']['rows'] if not r['case'].startswith('pherc0814')]
        vals=np.concatenate([np.mean([r['geometry'][k] for r in group],axis=0) for k in ('reference_to_pred_p50_p90','pred_to_partial_reference_p50_p90')])
        rows.append('Winner component'+' & '+' & '.join(f'{x:.3f}' for x in vals))
        for a,title in [('0058_kaggle','PLatS 0058')]:
            group=[r for r in data['ffn']['rows'] if r['method']==a and not r['case'].startswith('pherc0814')]
            vals=np.concatenate([np.mean([r['geometry'][k] for r in group],axis=0) for k in ('reference_to_pred_p50_p90','pred_to_partial_reference_p50_p90')])
            rows.append(title+' & '+' & '.join(f'{x:.3f}' for x in vals))
    table('winner_reading','table','Winner seed-selected binary component versus one-point PLatS on the same four evaluation crops. Native voxel distances; incomplete reference extents affect reverse tails.','tab:winnerreading','lrrrr',r'& \multicolumn{2}{c}{R$\to$P} & \multicolumn{2}{c}{P$\to$R} \\'+'\n'+r'Method & Median & p90 & Median & p90 \\',rows)


def make_tables(data):
    rows = []
    for label, title in LABELS.items():
        g = data['raw_topology'][label]
        values = [*g['sheet']['raw']['mean_betti'], 100*g['sheet']['raw']['fraction_100'], *g['union']['raw']['mean_betti']]
        rows.append(title+' & '+' & '.join(f'{v:.3f}' if i!=3 else f'{v:.2f}\\%' for i,v in enumerate(values)))
    g = data['raw_topology']['0058_kaggle']
    vals = [*g['sheet']['raw']['gt_mean_betti'], 100*g['sheet']['raw']['gt_fraction_100'], *g['union']['raw']['gt_mean_betti']]
    rows.append('GT & '+' & '.join(f'{v:.3f}' if i!=3 else f'{v:.2f}\\%' for i,v in enumerate(vals)))
    table('raw_topology', 'table*',
        'Whole-mask raw Betti counts before scoring erasure or component cleanup. Sheet statistics cover 764 eligible bands; union statistics cover 106 cases. $P_{100}$ denotes the fraction of sheets with $(b_0,b_1,b_2)=(1,0,0)$. These counts do not replace spatial feature matching.',
        'tab:topology', 'lrrrrrrr',
        r'& \multicolumn{3}{c}{Per-sheet means} & & \multicolumn{3}{c}{Union means} \\'+'\n'+
        r'Checkpoint & $b_0$ & $b_1$ & $b_2$ & $P_{100}$ & $b_0$ & $b_1$ & $b_2$ \\', rows)
    rows, topology = [], []
    for label, title in [('ffn','FFN'), *LABELS.items()]:
        group = [r for r in data['ffn']['rows'] if r['method']==label and not r['case'].startswith('pherc0814')]
        assert len(group)==4
        values = np.concatenate([np.mean([r['geometry'][key] for r in group], axis=0)
                  for key in ('reference_to_pred_p50_p90', 'pred_to_partial_reference_p50_p90')])
        rows.append(title+' & '+' & '.join(f'{v:.3f}' for v in values))
        raw = [r['raw_betti'] for r in data['ffn_raw_topology']['rows'] if r['method']==label and not r['case'].startswith('pherc0814')]
        topology.append(title+' & '+' & '.join(f'{v:.2f}' for v in np.mean(raw, axis=0)))
    table('ffn_distances', 'table',
        'Matched one-point FFN pilot: means of per-crop median/p90 distances in voxels over four evaluation crops. R denotes partial reference samples, P predicted voxel centers. Unannotated extents can inflate P$\\to$R distance.',
        'tab:ffn', 'lrrrr',
        r'& \multicolumn{2}{c}{R$\to$P} & \multicolumn{2}{c}{P$\to$R} \\'+'\n'+
        r'Method & Median & p90 & Median & p90 \\', rows)
    table('ffn_topology', 'table',
        'Mean raw voxel topology over the same four FFN evaluation crops. Threshold 0.5; V construction; no meshing or cleanup. Counts are not matched to a fully annotated reference sheet.',
        'tab:ffntopo', 'lrrr', r'Method & $b_0$ & $b_1$ & $b_2$ \\', topology)
    if data['automatic'] is None:
        (HERE/'tables/automatic.tex').write_text(
            '\\paragraph{Evaluation status.} Automatic inference has finished for all three checkpoints, and exact post-hoc scoring is still running in this snapshot. No automatic accuracy or formula result is reported until all 106 cases and all four arms are complete.\n')
    else:
        rows, instance = [], []
        for label, title in [('foreground_cc6','Foreground CC6'), *LABELS.items()]:
            r = data['automatic']['means'][label]
            rows.append(title+' & '+' & '.join(f'{r["binary_union"][key]:.4f}' for key in FIELDS))
            keys = ('instance_accuracy_iou25', 'instance_accuracy_iou50', 'mean_gt_iou', 'pred_count', 'gt_count', 'merge_count', 'split_count')
            instance.append(title+' & '+' & '.join(f'{r["instance"][key]:.3f}' for key in keys))
        table('automatic_union', 'table*',
              'Completed automatic segmentation on 106 cubes under original source ignore. One common foreground and the same 512 sampled points are used for every P2SD checkpoint. The CC6 baseline labels the foreground directly.',
              'tab:auto', 'lrrrrr', r'Arm & Dice & Surface Dice & TopoScore & VOI & Formula \\', rows)
        table('automatic_instances', 'table*',
              'Per-case means of annotated-band instance diagnostics. Acc$_\\tau$ is TP/(TP+FP+FN) at IoU $\\tau$, not integrated AP. GT IoU sums IoUs of matches at threshold 0.50 and divides by the GT count, assigning zero to unmatched bands. Pred/GT are counts; merge and split definitions are those of the inherited matching implementation.',
              'tab:autoinst', 'lrrrrrrr',
              r'Arm & Acc$_{.25}$ & Acc$_{.50}$ & GT IoU$_{.50}$ & Pred & GT & Merges & Splits \\', instance)



def main():
    for directory in ('figures','tables'):(HERE/directory).mkdir(exist_ok=True)
    from build_paper_figures import architecture,large_pages,downstream_figures
    data=snapshot();make_tables(data);main_and_supplementary_tables(data)
    architecture();large_pages(data);downstream_figures(data)
    (HERE/'results_snapshot.json').write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')
    print('Snapshot:',data['created_utc'],'automatic complete:',data['automatic'] is not None)
    print('Completed prompt jobs:',sum(len(v) for v in data['completed_prompt_formula'].values()),'/12')


if __name__=='__main__':main()
