# PLatS: Papyrus Surface Segmentation in Latent Space

Houjing Huang · houjing.huang@gmail.com

## Videos

Click a preview to play. Both segmentation demos use the Kaggle-only **0058**
PLatS checkpoint. Videos generated using **GPT-6-Astra xhigh**.

### Oral presentation · 3:55

The method and its development: learning compact sheet representations, predicting a sheet from point prompts, and discovering instances automatically.

https://github.com/user-attachments/assets/96abfccb-0c03-45be-9b7a-ed5e281f6a05

[Download MP4](https://github.com/huanghoujing/PLatS/releases/download/v0.1.0-progress-prize/PLatS_oral_CC.mp4)

### Official Zarr result demo · 2:04

Seven scroll crops with automatic instances, one-point decoding, Kaggle 1st-place comparisons, rotating 3D surfaces, and CT textures.

https://github.com/user-attachments/assets/564c6e5b-78dc-4a4b-a367-6ce35ea863cc

[Download MP4](https://github.com/huanghoujing/PLatS/releases/download/v0.1.0-progress-prize/remote_zarr.mp4)

### Labeled Kaggle result demo · 2:44

Neighboring prompted sheets and automatic instances compared with GT and the Kaggle winner, including slice/3D overlays and segmentation metrics.

https://github.com/user-attachments/assets/8d05cfe8-d7e1-4242-a91b-bb8febac0a63

[Download MP4](https://github.com/huanghoujing/PLatS/releases/download/v0.1.0-progress-prize/kaggle_gt_neighbors.mp4)

## Overview

A point-conditioned transformer predicts a single-sheet autoencoder code. The
frozen decoder reconstructs that sheet; latent-code clustering supports automatic
instance discovery from a binary foreground. This standalone reproduction bundle
contains the Kaggle-only 0058 model, its 0032 AE, and the 0022 foreground proposer.
It includes actual inference weights, not links into the original workspace.
Use the [complete reproduction archive](https://github.com/huanghoujing/PLatS/releases/download/v0.1.0-progress-prize/PLatS_progress_prize.tar.gz): it includes the weights and example arrays omitted from Git.
The [release page](https://github.com/huanghoujing/PLatS/releases/tag/v0.1.0-progress-prize) also provides its SHA256 checksum.

On `make_idea_clear`, start with the [architecture and efficiency guide](docs/ARCHITECTURE.md),
[training-from-scratch instructions](docs/TRAINING.md), and
[code review map and proposed removals](docs/CODE_REVIEW.md).
The release archive is the frozen submission version; it does not contain these
branch updates. No released weights are needed for the new scratch-training path.

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

## Interactive inference over SSH

[Launch the browser viewer](docs/INTERACTIVE_INFERENCE.md) for three linked
orthogonal CT slices, click-to-prompt sheet decoding, and rotatable 3D surfaces.
The GPU server caches CT features across clicks. Separate colored sheets,
reference-sheet 3D comparison, enlarged single-slice layouts, and automatic
instance segmentation are included. Load challenge TIFFs or a 320³ region from
an official CT Zarr URL using starting XYZ coordinates. Export NIFTI or losslessly
compressed TIFF labels in challenge ZYX order.

```bash
python -m pip install -r requirements-viewer.txt
python viewer.py --image examples/sample_00860/image.npy \
  --gt examples/sample_00860/gt_instances.npy --device cuda:0 --port 8787
# On your local computer:
ssh -N -L 8787:127.0.0.1:8787 USER@SERVER
```

Open the complete localhost URL printed by the server, including its `#token`.
Use `--bundle-root /path/to/extracted/PLatS` when the released weights are outside
this checkout. Inputs are preprocessed uint8 crops of at most 320³ voxels.

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
For a newly trained model, add `--run_dir runs_from_260914/training/p2sd_scratch`.
This uses the **co-trained binary head** and shares CT context with the point
model. The new training recipe crops only before the last binary upsampling
stage; the released reference settings and scores remain historical.
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
For a new model, follow [training from scratch](docs/TRAINING.md): prepare and
repair training labels, train the AE, compute its latent statistics, then train
P2SD with its joint binary branch. Portable recipes are in `configs/training/`.
The sparse AE encoder is retained with an optional training config. This new
recipe has passed a short synthetic optimization check; full training accuracy
has not been measured. The historical configs in `provenance/` record the
warm starts that produced the released checkpoints.

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
