# PLatS technical report

**PLatS: Papyrus Surface Segmentation in Latent Space**

Houjing Huang · houjing.huang@gmail.com

Open [report.pdf](report.pdf). The main paper presents the sheet representation,
point-conditioned latent prediction, automatic segmentation and experiments.
Exact objectives, module connections and secondary experiments follow in the
same PDF as Supplementary Material. FFN comparisons are excluded from this draft.

## Edit and build

Edit `sections/method.tex`, `sections/experiments.tex` and `sections/abstract.tex`
for the main paper. Supplementary source is `sections/supplementary.tex`, with
`sections/model_details.tex` for objectives and `sections/architecture.tex` for
module connections and efficiency. Main experiments use the Kaggle-only model;
extra-data checkpoints remain supplementary.

The new module figures are reproducible without datasets, model weights or a GPU:

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
late binary cropping; the paper figure labels the historical reference recipe.

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
