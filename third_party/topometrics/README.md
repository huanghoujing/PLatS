# Topological Metrics for 3D Instance Segmentation of Surfaces

A small toolkit that combines **topology (Betti matching)**, **Surface Dice**, and **VOI** into one leaderboard score for 3D surface segmentations.

## Quickstart

```bash
# create env
conda env create -f environment.yml
conda activate topo-metrics-3d

# build Betti-Matching-3D (submodule + C++/pybind)
make build-betti

# install this package (editable)
make dev
```

## Leaderboard score

```
Leaderboard = 0.3 * TopoScore + 0.35 * SurfaceDice@τ + 0.35 * VOI_score
```

* **TopoScore**: weighted Topological‑F1 over homology dims $k \in \{0,1,2\}$  
  ($k{=}0$ components, $k{=}1$ tunnels, $k{=}2$ cavities) with default weights **(0.34, 0.33, 0.33)**, renormalized over **active** dims.
* **SurfaceDice@$\,\tau$**: from `surface-distance` (default $\tau = 1.0$ voxel; `spacing=(1,1,1)`).
* **VOI_score**: $1 / (1 + \mathrm{VOI}_{\text{total}})$ (higher is better).

> Inactive dims (no features in pred+GT) are reported as **NaN** and **excluded** from TopoScore.

## Demos

Synthetic warped-surface cases with parallel evaluation + napari visualization are under `examples/` (128³ recommended).

```bash
python examples/demo_surfaces.py
```

## Build notes

* The compiled module lands at `external/Betti-Matching-3D/build/betti_matching*.so` and is auto-loaded.
* If CMake finds system Python, ensure you’re in the conda env and re-run `make build-betti`.

## Quick start (CLI)

Evaluate paired `.tif/.tiff` volumes from two folders and write a CSV + plots:

```bash
python metric_compute.py \
  --gt-dir /path/to/gt \
  --pred-dir /path/to/pred \
  --outdir ./metrics_out \
  --spacing 1 1 1 \
  --surface-tolerance 2.0 \
  --ignore-label 2 \
  --workers 4
```

With per‑case ignore masks (boolean .tif sharing the same stem as the case):

```bash
python metric_compute.py \
  --gt-dir /gt \
  --pred-dir /pred \
  --ignore-mask-dir /ignore_masks \
  --outdir ./metrics_out
```

This writes:

* `metrics_out/metrics_per_case.csv`
* `metrics_out/metrics_summary.json`
* Histograms: `hist_*.png`

---

## Quick start (Python API)

```python
from topometrics import compute_leaderboard_score
import tifffile

# pr, gt are 3D arrays with identical shape (Z, Y, X)
pr = tifffile.imread("pred_case01.tif")
gt = tifffile.imread("gt_case01.tif")

rep = compute_leaderboard_score(
    predictions=pr,
    labels=gt,
    dims=(0,1,2),
    spacing=(1.0, 1.0, 1.0),          # (z, y, x)
    surface_tolerance=2.0,            # in spacing units
    voi_connectivity=26,
    voi_transform="one_over_one_plus",
    voi_alpha=0.3,
    combine_weights=(0.3, 0.35, 0.35),  # (Topo, SurfaceDice, VOI)
    fg_threshold=None,                # None => legacy "!= 0"; else uses "x > threshold"
    ignore_label=2,                   # voxels with this GT label are ignored
    ignore_mask=None,                 # or pass an explicit boolean mask
)

print("Leaderboard score:", rep.score)                # scalar in [0,1]
print("Topo score:", rep.topo.toposcore)              # [0,1]
print("Surface Dice:", rep.surface_dice)              # [0,1]
print("VOI score:", rep.voi.voi_score)                # (0,1]
print("VOI split/merge:", rep.voi.voi_split, rep.voi.voi_merge)
print("Params used:", rep.params)
```

---

## Mathematical definition

We take as reference the findings in [this paper](https://arxiv.org/pdf/2412.14619v1), in which it is stated that the most highly correlated metrics with what a human being deems to be "good" for topology-aware segmentation are the [Betti-matching error](https://arxiv.org/abs/2407.04683) and the Variation of Information (VOI), which quantify the degree of over- and under- segmentation.

### Topological score

Taking inspiration from the [Betti-matching error](https://arxiv.org/abs/2407.04683), we define a topological score in [0,1].

For each homology dimension $k \in \mathcal{K}$ (default $\{0,1,2\}$), the matcher returns:

* $m_k$: matched features,  
* $p_k$: predicted total,  
* $g_k$: GT total.

Per‑dimension Topological‑F1:

```math
\mathrm{TopoF1}_k =
\begin{cases}
\dfrac{2 m_k}{p_k + g_k}, & \text{if } p_k + g_k > 0 \land g_k \neq 0,\\
\dfrac{0.5}{p_k + 0.5}, & \text{if } p_k > 0 \land g_k = 0,\\
\text{NaN}, & \text{otherwise.}
\end{cases}
```

Active dimensions:

```math
\mathcal{A} = \{\, k \mid p_k + g_k > 0 \,\}.
```

Let default weights be $w_0{=}0.34,\; w_1{=}0.33,\; w_2{=}0.33$ (configurable). Renormalize over active dims:

```math
\tilde w_k = \frac{w_k}{\sum_{j \in \mathcal{A}} w_j}, \quad k \in \mathcal{A}.
```

Topological score:

```math
S_{\text{topo}} =
\begin{cases}
1, & \text{if } \mathcal{A}=\varnothing,\\
\sum_{k \in \mathcal{A}} \tilde w_k \,\mathrm{TopoF1}_k, & \text{otherwise.}
\end{cases}
```

Hence $S_{\text{topo}} \in [0,1]$.

---

### Surface Dice at tolerance

The reference paper is [this](https://arxiv.org/pdf/1809.04430) from the Deepmind team. It's a modified Dice score for binary segmentation of surfaces that takes into account that there could be some inaccuracies in the labels, and so we define a spatial tolerance in the predictions to not overpenalize results that still capture the geometry, but are not exactly voxel accurate.

With spacing $(s_z,s_y,s_x)$ and tolerance $\tau$ (in those units), define voxelized surfaces $(\partial P, \partial G)$ and the $\tau$‑neighborhood $\mathcal{N}_\tau(\cdot)$ from `surface_distance`. Then

```math
\mathrm{SD}_\tau =
\frac{
\lvert \partial P \cap \mathcal{N}_\tau(\partial G)\rvert
+
\lvert \partial G \cap \mathcal{N}_\tau(\partial P)\rvert
}{
\lvert \partial P \rvert + \lvert \partial G \rvert
}.
```

Edge cases: both empty $\Rightarrow \mathrm{SD}_\tau{=}1$; exactly one empty $\Rightarrow 0$.

---

### VOI score

Compute 3D connected components of $P$ and $G$ to get labelings $L_P, L_G$ (connectivity $c \in \{6,18,26\}$, default $26$). Evaluate on the union‑FG mask.

```math
\mathrm{VOI}(G,P) = H(G \mid P) + H(P \mid G).
```

Map “lower is better” VOI to a “higher is better” score in $(0,1]$:

```math
S_{\text{VOI}} =
\begin{cases}
\dfrac{1}{1 + \alpha\,\mathrm{VOI}}, & \text{(default transform)},\\
e^{-\alpha\,\mathrm{VOI}}, & \text{(alternative)}
\end{cases}
\qquad \alpha \ge 0.
```

If both volumes are empty, $\mathrm{VOI}{=}0$ and $S_{\text{VOI}}{=}1$.

---

### Final leaderboard score

Let combination weights $(w_T,w_S,w_V)$ correspond to (Topo, Surface, VOI). Clamp negatives to 0 and renormalize to sum 1. The leaderboard scalar is

```math
\boxed{
\mathrm{LB} = \hat w_T S_{\text{topo}} + \hat w_S \mathrm{SD}_\tau + \hat w_V S_{\text{VOI}}
}
\in [0,1].
```

---

### Ignore handling, binarization, connectivity, and spacing

* **Ignore**: Voxels marked ignored are set to **background in both** $P$ and $G$ **before** all computations (topology, Surface Dice, VOI).
  * Ignore can be specified by a **GT label value** (default `ignore_label=2`) or an **explicit boolean mask**.
* **Binarization**:
  * If `fg_threshold is None` (default): foreground is `x != 0` (NaNs are treated as 0).
  * Else: foreground is `x > fg_threshold`.
* **Connectivity** (VOI only): `6`, `18`, or `26` (3D); default `26`.
* **Spacing**: a 3‑tuple $(s_z, s_y, s_x)$. Tolerance ($\tau$) is in the **same physical units**. Cropping does not alter spacing.

---

## Inputs/Outputs

* **Inputs**: 3D arrays (Z, Y, X) with identical shapes for prediction and GT. The CLI loads `.tif/.tiff` with optional `--allow-2d` (adds a singleton Z).
* **Outputs (CLI)**:

  * `metrics_per_case.csv` with columns:

    ```
    case, leaderboard, toposcore, topoF1_0, topoF1_1, topoF1_2,
    surface_dice, voi_score, voi_total, voi_split, voi_merge
    ```
  * `metrics_summary.json` with mean/median/std/IQR and lower/upper VaR/CVaR (for VOI total).
  * Histograms: `hist_leaderboard.png`, `hist_toposcore.png`, `hist_surface_dice.png`,
    `hist_voi_score.png`, `hist_voi_total.png`.

## License

MIT
