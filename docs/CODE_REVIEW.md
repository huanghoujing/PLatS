# Code review and cleanup scope

This branch improves the public PLatS repository. The larger research repository,
experiment outputs, published archives and checkpoint tensors remain historical
records. The reference implementation is the Kaggle-only 0058 architecture; 0076
and its continuation are relevant supplementary experiments.

## Proposed removals — listed before deletion

The following are **candidates, not deleted modules**. None is active in the
released 0058 model or the 0076 latent-prediction path. Removing them also requires
removing their optional trainer/evaluator dispatch, not just deleting files.

| Candidate modules | Associated objectives to remove with them | Reason |
|---|---|---|
| `models/refine_head.py` and `p2sd.refine` trainer branch | Post-AE residual-refinement BCE/Dice, volume and surface-band penalties | Separate image-conditioned decoder experiments; not the latent reconstruction used in the report |
| `models/sheet_decoder.py`, `train/sheet_decoder_step.py`, `p2sd.sheet_decoder` and `p2sd.union_decoder` branches | Alternative sheet/union occupancy and auxiliary decoder objectives | Alternative replacement decoder, not the frozen AE decoder |
| `models/band_refiner.py`, `train/band_refiner_step.py`, `p2sd.band_refiner` branch | Band-refinement BCE/Dice and foreground/off-sheet ranking hinge | Depends on the replacement decoder above |
| `models/point_decoder.py` (`SparsePointUnionDecoder`) and its checkpoint dispatch | No standalone trainer in this public bundle; remove its unused inference dispatch | Separate sparse union decoder; **not the sparse AE encoder** |

Do not remove these candidates until the deletion scope has been reviewed. They
are coupled to optional configurations, so their removal is a separate functional
change from the initial readability pass.

## Explicitly retained

- Dense **and sparse** AE encoders, their shared dense decoder and stage-matching
  utilities. `models/ae/sparse_encoder.py` remains available for efficiency work.
- AE occupancy BCE/Dice, intermediate occupancy heads, dense/query distance
  Smooth L1, input corruption, latent noise, same-volume code repulsion, outside
  tolerance and border penalties. These are active in the reference AE.
- Image encoder, image context, prompt composition, prefix modulator, latent
  refiner, final/auxiliary code heads, joint binary context refiner and decoder.
- Final/auxiliary latent MSE, same-sheet consistency, different-sheet hinge and
  binary BCE/Dice. Their implementations are exposed in a small loss module.
- Flip/rotation consistency and vertex/query supervision needed by the
  extra-data supplementary models.
- Optional AE point decoder, prompt modulator variants and decoded/query losses
  useful for focused ablations. Disabled does not by itself mean irrelevant.
- Existing useful comments, checkpoint key names and public import paths.

## Read the active method in this order

1. `configs/training/ae_scratch.yaml` and `p2sd_scratch.yaml`: runnable recipe.
2. `models/ae/sheet_ae.py`: `NoSkipSheetAE`, then its encoder and decoder heads.
3. `models/p2sd/full_attn.py`: `encode_image_context_from_image`,
   `compose_prompts`, `modulate_grid_tokens`, `forward_from_image_context`.
4. `models/binary_seg.py`: the shared-context auxiliary branch.
5. `train/p2sd_losses.py`: latent regression and identity objectives.
6. `train/train_ae.py::compute_ae_loss_terms` and `train/train_p2sd.py::main`:
   objective assembly and optimization.
7. `plats.py`, then `research/auto_instance_seg.py`: inference orchestration.

The public entry-point script is formatted for human review. Loss helpers are
moved without changing their arithmetic; the trainer re-exports their old names.
No trained module, state-dict key or loss is deleted in this first pass.

## Corrections found by tracing code

The binary decoder has residual blocks, but **no encoder–decoder skip
connections**. It receives the image-context grid, applies its own attention
refiner, and upsamples through five convolutional stages. Earlier report prose
confused residual connections with encoder–decoder skips; the revised figure and
description follow `BinarySegFromP2SD`.

In the reference `original` auxiliary-weight mode, weights `[0, .5, 1, 1]`
supervise auxiliary heads after blocks 1–3 with `[0, .5, 1]` and the **separate
final LayerNorm + linear head** with weight 1. The fourth auxiliary projection is
instantiated for checkpoint compatibility but its output is not supervised by
this setting. The supplementary equations now distinguish these heads.
