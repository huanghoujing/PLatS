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

The new module figures are reproducible without datasets, model weights or a GPU:

```bash
python report/build_architecture_figures.py
python -c "from report.build_paper_figures import architecture; architecture(include_inference=False)"
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
The current PDF, including the raw-prediction Betti table, is checked in
`raw_prediction_betti_validation.json`. Matched ablations, runtime
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

## Raw prediction topology (Table 3)

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
