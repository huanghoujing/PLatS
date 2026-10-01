# PLatS technical report

**PLatS: Papyrus Surface Segmentation in Latent Space**

PLatS means Papyrus Latent Space and names the overall framework. P2SD names
the point-to-sheet decoder within it.

The short main paper contains Abstract, Method and Experiments. References
and Supplementary Material follow in the same PDF. The main model is 0058
(Kaggle-only training); external-data 0076 and its 100k-step continuation appear
only in Supplementary. It uses the official CVPR 2026 author-kit style with
page numbers, without a review/submission banner.
The template is from [cvpr-org/author-kit](https://github.com/cvpr-org/author-kit),
at the pinned revision in `typesetting_provenance.json`.

Open [report.pdf](report.pdf). Edit [report.tex](report.tex) and the four files
under `sections/`. Measurements, their source hashes, checkpoint identities
and figure-selection provenance are in [results_snapshot.json](results_snapshot.json).
The revised report is exported under
`runs_from_260914/reports/11_plats_submission/`; earlier drafts are
preserved in `01_latent_sheet_cvpr/`, `02_plats_concise_supplementary/` and
`03_plats_3d_figures/`, `04_plats_readable_3d_views/` and `05_plats_native_slices/` and `06_plats_title_z_slices/`. Tables and figures are generated from
the snapshot's completed inputs. [validation.json](validation.json) records
case coverage, unchanged source hashes, score arithmetic and PDF checks.

Main evidence includes the released first-place ensemble versus automatic
0058 on all 106 released test cubes, before and after the explicit ignore-cut
restoration diagnostic. Actual binary-merger and ignore-cut figures use saved
native masks. Three separate diagrams define AE training, joint point-to-sheet
training and automatic inference. Whole-page CT/ink images replace the old
single-patch illustration. Figures 4 and 6 now show matched native 2D slices
and an enlarged contact crop. Both rows of Figure 4 use TIFF-stack z (internal axis 2, transposed for display).
The earlier figures mislabeled internal axis 0 as z. An exact source-TIFF
comparison now guards the coordinate mapping; the clearer case 00860 replaces
00865 in the top row. Run `audit_failure_coordinates.py` before rebuilding
the failure figures to reproduce the audit and selection. The raw-sheet panel marks ignored prediction
voxels in orange; the erased/restored panels show the corresponding cuts.
Native slice arrays and coordinates are saved under `slice_panels/` in the
export. The earlier 3D viewers remain in the previous exports. No score inputs
are changed by this presentation revision.

Supplementary retains all 1/2/4/8-point results for three checkpoints (9,168
sheet tuples), automatic instance diagnostics, the latent identity analysis,
the matched one-point FFN/winner controls, and spiral integration controls.
The published m7 tracks in the spiral control are not verified as first-place
outputs. Rejected or timed-out charts are unsuccessful downstream attempts,
even when a controller finishes normally. No SOTA or new readable-text claim
is made from these released, previously inspected data.

Experiment outputs and native galleries, relative to the project root:

| Run | Evidence |
|---|---|
| `evaluation/03_paper_hidden106_three_checkpoints_gpu0_gpu2` | Full prompt matrix, raw topology, NIFTI tuples and verification |
| `evaluation/04_paper_automatic_hidden106_gpu0_gpu2` | Automatic segmentation and annotated-band instance diagnostics |
| `evaluation/05_paper_winner_ignore106_gpu0_gpu2` | Winner/0058 standard and restoration scores; filtering/hybrid controls |
| `evaluation/06_paper_failure_analysis` | Native failure masks, selection rules and hashes |
| `evaluation/07_paper0058_latent_separability_gpu2` | 6,112 fixed single-point codes; same/different-band distances |
| `first_letters/22_paper_ffn_three_checkpoints_onepoint_gpu0_gpu2` | Matched-seed FFN pilot and all component CT/ink PNGs |
| `first_letters/23_paper_large0058_gpu0_gpu2/index.html` | Two large 0058 pages; native PNGs, ink tiles and surfaces |
| `first_letters/24_paper_winner_onepoint_gpu0_gpu2/index.html` | Winner and 0058 side by side, both normals and explicit failures |

All paths in this table start with `runs_from_260914/`. Controllers are
`scripts/paper_winner_ignore_benchmark.py`, `paper_failure_analysis.py`,
`paper_latent_separability.py`, `paper_large_0058.py`,
`paper_large0058_eval.py`, and `paper_winner_reading.py`.

From the project root, refresh measurements and figures with:

```bash
MPLCONFIGDIR=/tmp/p2sd_cvpr_mpl_cache .venv/bin/python research/papers/latent_sheet_report/audit_failure_coordinates.py
MPLCONFIGDIR=/tmp/p2sd_cvpr_mpl_cache .venv/bin/python research/papers/latent_sheet_report/build_failure_figures.py
MPLCONFIGDIR=/tmp/p2sd_cvpr_mpl_cache .venv/bin/python research/papers/latent_sheet_report/build_assets.py
```

Compile with a normal LaTeX/BibTeX toolchain, or the locally cached Tectonic:

```bash
cd research/papers/latent_sheet_report
XDG_CACHE_HOME=/tmp/p2sd_cvpr_tex_cache ../../../.external/paper_typesetting/tectonic \
  --only-cached --untrusted --keep-logs report.tex
```

After compilation, run `.venv/bin/python research/papers/latent_sheet_report/validate_report.py`
from the project root. Validation requires all comparisons to finish and rejects
pending PDF text. Refreshing assets does not rerun inference. Review the text
when adding results. The source template
revision and local typesetting engine download hash are recorded in
[typesetting_provenance.json](typesetting_provenance.json).

This is a technical-report draft. It is not an uploaded conference submission;
author names and publication metadata can be supplied when preparing one.

Figure 7 uses reconstructed 3D geometry to align the horizontal display direction
and sampling normal to the FFN chart. PLatS uses a horizontal flip in both rows
and its already saved normal-minus CT/ink output; FFN and winner use normal-plus.
The charts remain independently flattened, without pixel registration. All
original predictions and quantitative scores are unchanged. The orientation
cosines and source surface hashes are recorded in `results_snapshot.json`.

The Figure 7 page replacement is controlled by
`scripts/paper_page_onepoint.py` and `scripts/finalize_page_onepoint_report.py`.
It appears only after FFN and PLatS finish the frozen 126-crop plan in
`runs_from_260914/first_letters/25_paper_page_onepoint_gpu0_gpu2/`.
The earlier single-crop figure remains available in export 08.

The user removed the unprompted winner from the page-scale comparison.
Figure 7 now compares only FFN and PLatS; the original frozen prompt plan
is retained, with the scope change recorded in `scope_amendment.json`.
Completed PLatS outputs and FFN crops are reused. Three FFN shards migrate
to freed GPU0 after saving their current crop; the others continue on GPU2.

## Simplified report revision

The main text presents the sheet prior, point-conditioned prediction and automatic
inference, followed by the standard LB, ignore-restoration and downstream results.
Full objectives, heads, coefficients and audit details are in Supplementary, including
`sections/model_details.tex`. Figures 1 and 2 use a single main flow with separate
supervision rows. Figure 3 uses measured data from case 00860: CT/seeds, cached
CT-feature RMS, PCA of all 512 point codes, and the saved decoded instances.
Gray codes are discarded seeds. PCA is for display, never used for clustering.
The forward replay matches frozen retained-cluster sizes. Exact intermediate
arrays and source hashes are in `evaluation/08_paper_method_visuals/`; regenerate
them only with `scripts/paper_inference_figure_capture.py` on GPU0. Ordinary
asset builds reuse these arrays and do not run inference.

## Submission scope

Author: Houjing Huang (houjing.huang@gmail.com). FFN and the combined prompted
reading comparison are deferred and excluded from the current PDF and export.
Historical experiment files remain archived. The primary evidence is the
Kaggle winner comparison, prompted topology, automatic instances, and PLatS
large-page outputs. The clean reproduction package is `submission/PLatS/`.
