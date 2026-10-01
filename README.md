# PLatS: Papyrus Surface Segmentation in Latent Space

Houjing Huang · houjing.huang@gmail.com

A point-conditioned transformer predicts a single-sheet autoencoder code. The
frozen decoder reconstructs that sheet; latent-code clustering supports automatic
instance discovery from a binary foreground. This standalone reproduction bundle
contains the Kaggle-only 0058 model, its 0032 AE, and the 0022 foreground proposer.
It includes actual inference weights, not links into the original workspace.
Use the [complete reproduction archive](https://github.com/huanghoujing/PLatS/releases/download/v0.1.0-progress-prize/PLatS_progress_prize.tar.gz): it includes the weights and example arrays omitted from Git.
The [release page](https://github.com/huanghoujing/PLatS/releases/tag/v0.1.0-progress-prize) also provides its SHA256 checksum.

```bash
curl -fL -O https://github.com/huanghoujing/PLatS/releases/download/v0.1.0-progress-prize/PLatS_progress_prize.tar.gz
curl -fL -O https://github.com/huanghoujing/PLatS/releases/download/v0.1.0-progress-prize/PLatS_progress_prize.tar.gz.sha256
sha256sum -c PLatS_progress_prize.tar.gz.sha256
tar -xzf PLatS_progress_prize.tar.gz
cd PLatS
```

## Full released-test predictions

[Download index](predictions/README.md): all 106 released Kaggle test cases,
including **9,168 prompted sheet masks** (1/2/4/8 points, three checkpoints),
**1,272 prompted unions**, and **318 automatic instance predictions**.
The primary checkpoint is 0058; 0076 and 0076+100k are supplementary.
NIFTIs, point coordinates, scores and checksums are supplied as separate release
assets. These are the released former hidden-test cases, not a fresh blind test.

## Start here

Use Python 3.12 and an NVIDIA GPU supporting BF16. Tested software versions are
pinned below; compilation of the exact scorer also needs a C++17 compiler.

```bash
python3.12 -m venv .venv
. .venv/bin/activate
python -m pip install torch==2.12.0 --index-url https://download.pytorch.org/whl/cu132
python -m pip install -r requirements.txt
python -m pip install --no-deps -e .
python plats.py verify
```

The tested PyTorch build is `2.12.0+cu132`. Install the matching PyTorch/CUDA build
for the machine; a different build may introduce small numerical differences.
CPU inference is available with `--device cpu` but is much slower.

## Reconstruct one prompted sheet

```bash
python plats.py prompt --image examples/sample_00860/image.npy \
  --points examples/sample_00860/points.json --count 8 --output outputs/prompt
```

Use `--count 1`, `2`, `4` or `8` for the nested fixed prompts. The command saves
`image.nii.gz`, `prompt_points.nii.gz`, `prediction.nii.gz`, `prediction.npy`,
probabilities, and runtime/weight provenance. The mask is raw: no component cleanup.
The example is the first eligible sheet from the frozen case-00860 evaluation,
not a best-performing sheet. Saved expected outputs expose its reconstruction errors.

## Discover instances automatically

```bash
python plats.py automatic --image examples/sample_00860/image.npy \
  --output outputs/automatic
```

This runs the included foreground proposer, samples 512 seeds, clusters their
codes, tests seven eight-point subsets per retained cluster, decodes and resolves
overlaps using the frozen benchmark settings. Small and unstable clusters are
excluded. It does not read GT or ignore masks. `--foreground path.npy` optionally
reuses a binary mask. Final IDs, foreground, seeds and clustering records are saved.
Set `CUDA_VISIBLE_DEVICES` to select a physical GPU; `--device cuda:0` then uses
that visible GPU. Process separate crops on separate GPUs for parallel inference.

## Exact score and reproduction check

```bash
python tools/build_metrics.py
python plats.py score --prediction outputs/prompt/prediction.npy \
  --gt examples/sample_00860/gt_sheet.npy --ignore examples/sample_00860/ignore.npy \
  --policy annotated-box --output outputs/prompt/metrics.json
python tools/check_example.py --prompt outputs/prompt --automatic outputs/automatic
python tools/check_import_paths.py
```

Scoring reports Dice, Surface Dice, exact matched TopoScore, VOI and the public
weighted formula. It also writes `gt_sheet.nii.gz` beside the prediction tuple.
`annotated-box` ignores only the exterior annotated-region box, treating inward
ignore regions as background; this is the prompted-sheet diagnostic. `source`
uses original ignore erasure for automatic binary-union evaluation. The latter can
be run against `gt_instances.npy`: all positive IDs become binary foreground.
A per-sheet formula score is not an official instance leaderboard score.

## Input contract

Inputs are preprocessed uint8 CT `.npy` arrays, with at most 320 voxels per axis.
Points index those array axes in exactly the same order. The bundled sample uses
`[X,Y,Z]`, obtained from a TIFF `[Z,Y,X]` stack by `transpose(2,1,0)`; its five-voxel
image border is already zeroed. No intensity normalization, border erasure or axis
permutation is silently applied by the CLI. Smaller crops are centered in a 320³
canvas and predictions are cropped back. NIFTIs use an identity voxel affine;
physical scanner coordinates require the original volume metadata.

## Contents and scope

- `src/`: the model, training, data and evaluation modules needed by the entry points.
- `weights/`: inference-only state dictionaries; every tensor equals its source checkpoint.
- `configs/`: portable architecture, latent statistics and frozen automatic settings.
- `examples/`: one released test CT, reference masks, prompts and expected predictions.
- `report/`: technical report and source; the deferred seeded-growing comparison is excluded.
- `evidence/`: full-test result summaries, fixed prompt manifests and large-page CT/ink PNGs.
- `provenance/`: checkpoint hashes, original training configs, environment and source hashes.
- `third_party/`: exact scorer sources with their upstream licenses.

The small example reproduces the central inference and scoring path. Full 106-case
results require the remaining released data; the example does not reproduce their
aggregate by itself. Evidence manifests identify those cases and prompt sets.
The original training configurations record historical warm starts and dataset
paths. They are archival records, not portable one-command training recipes;
retraining requires the repaired training dataset and initialization lineage.
Training entry points are `vesuvius_p2sd.train.train_ae`, `train_p2sd`, and
`train_binary_seg`, each taking `--config_path`. The supplied inference configs
are deliberately separate from these historical training configs.

The report's large-page PNGs are frozen evidence. Core mesh/ink modules are included,
but reproducing page growth additionally requires the scroll volume, ink checkpoint
and the documented region/controller artifacts; the small demo makes no page-growth
or ink-training reproduction claim. Training data and source licensing are recorded
in `NOTICE.md`. No submission form or external upload is performed by these tools.

## Completed portability check

The entry points were executed from `/tmp`, with included weights and configs.
Automatic foreground and final instance IDs match the frozen benchmark exactly.
Single-sheet inference has 0.99704 Dice agreement with the previously batched
prediction; its public-formula difference is -0.0000915 (BF16 execution is not
bitwise invariant to batch layout). All five score differences are below 0.00035.
The bundled exact C++ extensions were compiled from source and exercised.
`tools/check_import_paths.py` verifies every loaded PLatS and scoring module is
inside this folder. See `provenance/` for measurements. Dependencies were tested
with the existing recorded environment; a fresh dependency download was not tested.
