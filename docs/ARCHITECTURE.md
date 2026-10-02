# Architecture and efficiency

Dimensions below describe the Kaggle-only reference at a 320×320×320 crop.
Spatial axes follow the input array; legacy variable names ending in `_zyx`
do not permute the native array axes.

## Single-sheet AE

`NoSkipSheetAE` in `models/ae/sheet_ae.py` encodes a corrupted binary sheet into
`z: [B,64,10,10,10]`. The released dense encoder uses a factor-four patch stem,
stage-matched residual convolutions, RMS normalization and factor-32 total
reduction. There are no encoder–decoder skips. The sparse alternative in
`models/ae/sparse_encoder.py` computes on occupied voxels and densifies its
bottleneck for the same decoder. It is retained for efficiency experiments.

The training decoder receives `z * (1 + uniform(-.02,.02))`. Five residual /
transposed-convolution stages increase spatial size 10→20→40→80→160→320, with
channels 64→48→32→24→16→8. Final 1×1 convolutions predict occupancy logits and
unsigned distance. Four intermediate 1×1 occupancy heads supervise sizes
20, 40, 80 and 160. A separate query MLP samples the noisy latent trilinearly,
concatenates normalized query coordinates, and predicts one unsigned distance.

| Output / regularizer | Supervision |
|---|---|
| Final occupancy | Positive-weighted BCE + soft Dice |
| Intermediate occupancy | BCE + Dice, weights .125/.25/.5/.5 |
| Dense unsigned distance | Smooth L1, weight .15 |
| Query unsigned distance | Smooth L1, weight .15 |
| Distinct sheet codes in one crop | Cosine repulsion, margin .5, weight .1 |
| Occupancy outside a 2-voxel target tolerance | Foreground-normalized penalty, weight .05 |
| Occupancy at the 5-voxel border | Foreground-normalized penalty, weight .05 |

Distances are clipped at 16 voxels and normalized by 16. No explicit topology
loss is used. See `train_ae.py::compute_ae_loss_terms` for exact arithmetic.

## P2SD modules and their connections

| Module | Input → output | Implementation |
|---|---|---|
| Image encoder | CT → 512×10³ features | `P2SDImageEncoder`, early convolutional factor-four stem, total factor 32 |
| Image context | Grid projection + four attention blocks → 1,000 tokens of width 512 | `grid_proj`, `image_context_blocks`; 3D per-axis RoPE |
| Prompt composition | Fourier point coordinates + sampled contextual features + label → K tokens of width 512 | `compose_prompts`, `prompt_mlp`; eight Fourier bands; retain all positive tokens |
| Prompt modulator | K prompt tokens prefixed to 1,000 context tokens → conditioned grid | One prefix-attention block; align coordinate frames, discard prefix outputs |
| Latent refiner | Conditioned grid → four successive 1,000-token grids | `blocks`, four self-attention blocks, eight heads, SwiGLU MLPs |
| Prediction heads | Grid → 64×10³ sheet code | Intermediate `aux_out`; final `LayerNorm` + `out` |
| Frozen AE decoder | Denormalized final code → 320³ sheet logits | Used for inference/validation; no active decoded sheet loss in 0058 |
| Binary branch | Unprompted image context → union logits | Four additional context-attention blocks, five convolutional upsamples, 1×1 occupancy head |

The binary branch shares the image encoder **and image context** with P2SD.
Its gradient updates both during joint training. It has no encoder–decoder
skip connections: residual connections inside blocks are a different operation.
The joint trainer restores shared-trunk trainability after constructing
`BinarySegFromP2SD`, whose standalone default freezes its trunk.

The frozen AE encodes clean GT sheets. Per-channel training-split mean and
standard deviation normalize those codes; MSE supervises predicted normalized
codes. In the reference `original` weighting mode, auxiliary outputs after
refiner blocks 1–3 receive weights 0/.5/1, and the separate final normalized
head receives weight 1. The fourth auxiliary head is stored in the checkpoint
but is not used by that objective. Same-sheet prompt consistency (weight .5)
and different-sheet MSE-margin hinge (weight .25, margin 1) act on final codes.

The historical 0058 recipe cropped a 5³ context subgrid before binary decoding.
The **new training recipe** uses `crop_stage: second_last`: refine all 10³ context
sites, decode the full 160³ intermediate feature volume, crop 80³ features, then
run the last ×2 upsampling/residual/head to supervise 160³ output voxels.
The earlier decoder stages keep full-volume context. `crop_grid: 5` still denotes
5 × 32 = 160 output voxels per axis. The loss is ignore-masked BCE (positive
weight 2) + Dice, once per CT batch rather than once per prompt group.

P2SD's optional `latent_distance_head` learns query distances directly from GT
EDT, independently of the frozen AE query head. Its loss and vertex/query
utilities remain available. The older `coordinate_query` objective instead
matches frozen-AE query predictions. The current P2SD-owned head is distance-only;
query occupancy/GT target utilities also remain in the implicit AE path.

## Where efficiency comes from

| Representation | Spatial sites | Scalars | BF16 tensor storage |
|---|---:|---:|---:|
| Image context, 512×10³ | 1,000 | 512,000 | 0.977 MiB |
| Sheet code, 64×10³ | 1,000 | 64,000 | 0.122 MiB |
| One full-resolution output, 1×320³ | 32,768,000 | 32,768,000 | 62.5 MiB |

The latent grid has 32,768 times fewer spatial sites. Accounting for its 64
channels, a sheet code contains **512 times fewer scalars** than a single
full-resolution mask. These are tensor-size ratios, **not runtime speedups**.
Training memory additionally includes activations, gradients and optimizer state;
attention computation still depends quadratically on token count.

Image encoding/context is shared across prompt groups. During seed clustering,
we predict compact codes for N seeds, compare codes, and decode only retained
sheet hypotheses. With K decoded hypotheses the approximate work is

`shared CT + N × point-to-code + clustering + K × AE decode`.

A direct full-resolution per-seed predictor would instead pay its upsampling
cost for every seed. Our implementation also tests multiple multi-point subsets
per retained cluster, adding code-prediction work before its final decode.
Geometric cleanup and full-resolution output storage remain necessary.
New-model inference (`plats.py automatic --run_dir ...`) loads the co-trained
binary head and shares the cached image context with P2SD. It adds one full-volume
binary decode, not a second CT encoding. The historical released-model path
(without `--run_dir`) still uses a separately trained proposer and its additional
encoding; published benchmark results refer to that historical configuration.

During P2SD training the teacher encoder remains necessary, but the frozen AE
decoder does not execute for the active latent objective. The binary auxiliary
branch still decodes its 160³ crop, and validation decodes full sheets. Thus the
efficiency claim concerns repeated prompt processing and supervision, not an
absence of all upsampling. Moving binary cropping to the final stage increases
its training activation memory relative to bottleneck cropping. A wall-clock
speedup versus a matched direct decoder
requires a controlled benchmark and is not yet claimed.
