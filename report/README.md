# PLatS technical report

**PLatS: Papyrus Surface Segmentation in Latent Space**

Houjing Huang · houjing.huang@gmail.com

Open [report.pdf](report.pdf). The main paper develops one idea: a compact
sheet code for both reconstruction and instance discovery. It includes related
work, automatic segmentation, experiments and remaining evidence gaps.
Exact objectives, module connections and secondary experiments follow in the
same PDF as Supplementary Material. FFN comparisons are excluded from this draft.

## Edit and build

Edit `sections/abstract.tex`, `sections/introduction.tex`, `sections/method.tex`,
`sections/experiments.tex` and `sections/discussion.tex` for the main paper. Supplementary source is `sections/supplementary.tex`, with
`sections/model_details.tex` for objectives and `sections/architecture.tex` for
module connections and efficiency. `sections/query_extensions.tex` describes
retained query supervision and the unevaluated AE-consistency proposal.
Main experiments use the Kaggle-only model;
extra-data checkpoints remain supplementary.

The detailed supplementary module figures need no datasets, weights or GPU:

```bash
python report/build_architecture_figures.py
cd report
latexmk -pdf report.tex
```

A Tectonic installation can alternatively compile `report.tex`. The bundled
`cvpr.sty` and bibliography style use the revision recorded in
`typesetting_provenance.json`. No template download is required.

The supplementary figures are also available as standalone
[AE PNG](figures/ae_modules.png), [P2SD PNG](figures/p2sd_modules.png), PDF and SVG.
They show the frozen teacher, normalized target, prompt composition, modulator,
refiner, prediction heads, binary branch and attached losses. The binary branch
has **no encoder–decoder skip connections**; this corrects the earlier prose.
The final prediction head is distinct from the auxiliary projections.
The [new training recipe diagram](figures/p2sd_training_recipe.png) shows
late binary cropping and is included in the PDF. The historical reference
diagram remains available separately; benchmark settings are explicitly distinguished.

The efficiency discussion distinguishes tensor sizes from runtime: 1,000 latent
sites, 64,000 code scalars, shared CT context, and full decoding only when needed.
It includes retained-sheet decoding and the historical separate foreground
proposer. New-run inference reuses CT context for its co-trained binary head.
No wall-clock speedup is claimed without a controlled benchmark.

## Frozen experimental evidence

`results_snapshot.json` identifies the original experiment inputs and hashes.
The original `validation.json` records the submission's experimental audit; it
is historical, not a certification of every later prose/layout edit. Branch
validation for the initial cleanup is in
`../provenance/make_idea_clear_validation.json`; the later recipe/inference
checks are in `../provenance/late_crop_joint_inference_validation.json`.
The initial CVPR writing revision is audited in `revision_validation.json`.
It changes presentation and corrects the 0076 GT-query description.
The raw-prediction Betti table was checked in
`raw_prediction_betti_validation.json`. The previous method revision is checked in `method_clarity_validation.json`.
The initial three-figure redesign is checked in `three_figures_validation.json`.
The added exact-plane seed example is checked in `slice_seed_figure_validation.json`. Matched ablations, runtime
measurements and spatially disjoint generalization remain open experiments.

Tables cover all 106 released, previously inspected test cubes. Training IDs
were excluded; this alone does not certify spatial independence. Ignore-mask
restoration scores are diagnostic and separate from the official erasure score.
The large-page PNGs show reconstructed CT/ink surfaces, including gaps and drift;
they do not establish new text recovery.

The legacy `build_assets.py`, failure analysis, page comparison and validation
scripts record how the original research workspace assembled those experiments.
They require external experiment outputs identified in the snapshot. Do not run
them as a standalone paper-build command or to regenerate the new architecture
figures. Normal typesetting uses the supplied tables and images directly.

The full released-test prediction archives are linked in
[the download index](../predictions/README.md). Training and source-reading
instructions are in [TRAINING.md](../docs/TRAINING.md) and
[CODE_REVIEW.md](../docs/CODE_REVIEW.md).

## Original prediction topology census (Table 3 baselines)

The table contains freshly recomputed Betti counts for the same 106 winner/automatic PLatS
unions as Tables 1 and 2. No GT or ignore masks are read. The full native volumes
are used, with no additional erasure, restoration or size filtering. Existing
inference postprocessing remains in the predictions. Means and medians describe
these masks, not errors against a reference topology.

The per-case CSV, JSON and input manifest remain in the local experiment folder
`runs_from_260914/evaluation/09_paper_raw_prediction_betti/` in the research
workspace. Recompute from that manifest and its frozen prediction files:

```bash
python report/compute_raw_prediction_betti.py \
  --manifest /path/to/09_paper_raw_prediction_betti/input_manifest.json \
  --prediction_root /path/to/research/workspace \
  --output /tmp/plats-raw-betti --workers 16
```

The script verifies input hashes and native shapes, and checks agreement with
the previous raw audit. It needs NumPy/SciPy and the repository source; no GPU,
model inference or exact feature matching is needed for these whole-mask counts.

## Three main method figures

The main method still follows AE training, point-to-latent training and automatic
clustering/decoding. The three figures now explain these stages separately:

- [Sheet representation](figures/sheet_representation.png): matched corrupted,
  reconstructed and clean mask slices, explicit supervised heads, and a measured
  two-sheet code-repulsion example.
- [Point-to-latent prediction](figures/point_to_latent.png): aligned target and
  predicted codes with their MSE, a shared binary branch, and an inference-only
  frozen sheet decoder. The binary branch has residual blocks, not encoder-to-decoder skips.
- [Automatic discovery](figures/automatic_discovery.png): all 512 full-code
  distances, cluster boundaries and excluded seeds, colored seed groups on CT,
  and decoded sheets. Row A retains the volume-seed example; row B samples
  all 512 seeds from the exact displayed plane. Off-plane markers in A are
  explicitly distinguished. Colors are local to each row.

The main figures use actual outputs from the existing research runs. Rebuild
the local capture on an available GPU, then render without further inference:

```bash
CUDA_VISIBLE_DEVICES=0 python report/capture_method_figure_data.py --research_root /path/to/research/workspace
CUDA_VISIBLE_DEVICES=0 python report/capture_slice_seed_figure.py --research_root /path/to/research/workspace --public_root "$PWD"
python report/build_explanatory_method_figures.py --research_root /path/to/research/workspace
```

The original research checkpoints and arrays must be present. The capture is
saved under `runs_from_260914/evaluation/11_paper_three_method_figures/` in that
workspace. The exact-plane run is in `runs_from_260914/evaluation/12_paper_slice_seed_example/`.
Source arrays stay local; rendered PDF/PNG/SVG assets and their
builders accompany the paper. The figure revision itself does not change benchmark results.
Earlier figure assets and audits remain historical records.

## Fixed repair evaluation (Tables 1--3)

The tables retain their original rows and add PLatS with fixed AE repair,
guarded per-sheet closing, and dusting. The supplementary ablation includes
each stage and the winner with the same morphological cleanup. Aggregate
measurements are in [postprocessing_results.json](postprocessing_results.json);
the current revision audit is [postprocessing_validation.json](postprocessing_validation.json).
This is an exploratory extension after inspecting three released-test outliers.

The local research record is
`runs_from_260914/evaluation/16_postprocessing_hidden106/`: frozen recipe,
native instance NIFTIs, images/annotation links, per-case scores, output hashes,
and an HTML comparison. Tiny-piece dusting uses <=5 voxels before scoring;
the restoration diagnostic separately retains full instances >=5,000 voxels
with visible support. Table 3 includes inference cleanup but no ignore processing.

Reproduce using the original research workspace and its frozen input manifests:

```bash
export PLATS_RESEARCH_ROOT=/path/to/research/workspace
python report/evaluate_postprocessing.py prepare
# Run these two inference commands concurrently on available GPUs:
python report/evaluate_postprocessing.py infer --gpu 0 --shard 0
python report/evaluate_postprocessing.py infer --gpu 2 --shard 1
python report/evaluate_postprocessing.py winner --workers 16
python report/evaluate_postprocessing.py freeze
python report/evaluate_postprocessing.py prepare_scores --workers 16
python report/evaluate_postprocessing.py score --workers 96
python report/evaluate_postprocessing.py summarize
python report/export_results.py --run "$PLATS_RESEARCH_ROOT/runs_from_260914/evaluation/16_postprocessing_hidden106" --paper report
```

Choose a scoring worker count suitable for available CPU/RAM. The evaluation
uses the existing exact compact topology backend. Identical masks share one
score computation; completed atomic score files are reusable. The scoring
adapter was checked against an unchanged published case with zero difference.
The original `results_snapshot.json`, checkpoint weights and inference recipe
are unchanged; the new postprocessing is explicitly a separate inference arm.
