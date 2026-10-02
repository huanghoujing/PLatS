# Interactive sheet inference

The browser app shows three linked orthogonal CT slices and a rotatable 3D
surface view. Click a sheet to predict it with the released 0058 PLatS model,
then add up to eight positive points to refine the same sheet. CT features are
computed once per loaded crop and reused across points and sheets. Inference
runs on the server; the browser renders slices and surfaces locally.

## Start on the GPU server

Use the same Python environment as `plats.py`; there are no additional Python
runtime dependencies. Browser rendering assets are bundled, so the viewer does
not contact a CDN. In the repository with the release weights and example arrays:

```bash
python viewer.py \
  --image examples/sample_00860/image.npy \
  --gt examples/sample_00860/gt_instances.npy \
  --device cuda:0 --port 8787 \
  --output runs_from_260914/interactive/exports
```

The Git repository excludes weights and example arrays. If the reproduction
archive was extracted elsewhere, pass its root with `--bundle-root` and use
absolute image/reference paths. Example for the existing research workspace:

```bash
~/Project/vesuvius_p2sd/.venv/bin/python ~/Project/PLatS/viewer.py \
  --bundle-root ~/Project/vesuvius_p2sd/submission/PLatS \
  --image ~/Project/vesuvius_p2sd/submission/PLatS/examples/sample_00860/image.npy \
  --gt ~/Project/vesuvius_p2sd/submission/PLatS/examples/sample_00860/gt_instances.npy \
  --device cuda:0 --port 8787 \
  --output ~/Project/vesuvius_p2sd/runs_from_260914/interactive/exports
```

Run this in a terminal or `tmux` session. The app prints a localhost URL with a
random `#token`. It listens on `127.0.0.1` only. On your local computer:

```bash
ssh -N -L 8787:127.0.0.1:8787 USER@SERVER
```

Open the **complete URL printed by the server**, including its token, in a
browser with WebGL2. The token is retained in that tab's session storage and
removed from the address bar. Keep the SSH tunnel and server running. This is
a single-user research viewer; multiple tabs share the server's loaded crop.

## Controls

- **Click a slice:** add a positive point to the active sheet. All other slices
  move to that point. Points on other slice planes are not drawn on the current
  plane; every point is shown in 3D.
- **Shift-click:** move the crosshair without adding a prompt.
- **Mouse wheel / sliders:** move each slice independently.
- **Decode after each click:** enabled initially; disable it to place several
  points before clicking **Decode sheet** or pressing **Enter**.
- **Undo point / Backspace:** remove the last point. The × beside a coordinate
  removes that point. **Clear sheet** clears the active group.
- **+ New / N:** start another independently colored sheet (up to 16 per session).
- **Threshold:** change the active sheet's mask. Its cached probability is reused;
  this does not rerun the model. Display window and overlay opacity also do not
  affect model input or inference.
- **3D:** drag to rotate, right-drag to pan, wheel to zoom. CT planes and predicted
  surfaces can be toggled separately. Small axis labels identify array axes.
- **Open another crop:** enter paths on the server. This replaces the current
  session and recomputes CT features. Export results first if needed.

Prompt edits invalidate the displayed old prediction immediately. Late results
from older prompts are discarded. GPU work is serialized in one background
worker so the interface remains usable while decoding. There are no negative
clicks: the released model was trained with positive sheet prompts.

## Inputs, coordinates, and outputs

CT inputs must be preprocessed `uint8` 3D `.npy`, `.nii`, `.nii.gz`, `.tif`, or
`.tiff` arrays, with each dimension between 32 and 320. Use a crop for larger
volumes. No normalization, axis permutation, border erasure, or NIFTI
canonicalization is applied. The bundled array uses `[X,Y,Z]`; a TIFF loaded
in `[Z,Y,X]` stays in that order. Slice labels deliberately use **axis 0/1/2**
instead of assuming anatomical orientation. Smaller crops are centered in the
model's 320³ canvas and decoded masks are cropped back.

The optional reference must have the same shape and voxel affine as the image.
Positive reference values form its displayed union; supply a clean foreground
or instance label map, not a mask whose positive values encode ignore regions.
The reference is only displayed and exported; it never supplies prompts or model
inputs. Slice and 3D views use native voxel coordinates. NIFTI exports preserve
the CT affine; `.npy` and TIFF inputs use an identity affine.

**Export NIFTIs** creates a new timestamped folder on the server containing:

- `image.nii.gz`, optional `gt_union.nii.gz`, and `session.json` with exact points,
  thresholds, timings, input checksum, and model/config checksums.
- One `sheet_XX/` folder per decoded sheet: `prompt_points.nii.gz`,
  `prediction.nii.gz`, `probability.nii.gz`, `points.json`, and links to CT/GT.
- `union.nii.gz` and `overlap_count.nii.gz`. Sheets remain separate; overlapping
  prompted masks are not silently converted into exclusive instance IDs.

Predictions are raw thresholded outputs, without AE repair, closing, dusting,
ignore-mask erasure, or leaderboard scoring. The 3D surface uses marching cubes
on the same mask shown in the slices, without downsampling thin sheets. Full
probabilities remain on the server; compact bit-packed masks and meshes are
sent to the browser.

## Other checkpoints

For a jointly trained checkpoint, replace `--bundle-root` with:

```bash
--run-dir /path/to/training/run \
--checkpoint-dir /path/to/training/run/checkpoints/step_100000 \
--training-root /path/to/workspace
```

`--checkpoint-dir` is optional; `--training-root` resolves paths stored in the
training configuration. The foreground decoder is not needed for this prompted
viewer. `--device cpu` is supported but has not been performance-tested.

## Validation

`python -m pytest -q tests/test_interactive.py` checks asymmetric crop padding,
point bounds, mask/mesh axis agreement, NIFTI affine and overlap preservation,
clearing a sheet during a running decode, cached threshold changes, and stale
export rejection. These checks use small synthetic volumes and no GPU.

The released 320³ example was also exercised in Chromium with real GPU0
inference: clicks on all three axes, crosshair navigation, one-point decoding,
reference overlays, threshold changes, separate sheets, and NIFTI export.
The exported probability volume exactly matched the released CLI one-point result
(maximum absolute difference 0). On GPU0, a warm decode took about 0.30 seconds,
or 0.90 seconds including exact surface extraction; transfer and browser drawing
add latency. Browser refresh restores already decoded sheets from the server.

The optional browser check uses Playwright with Chromium and a fresh server
loaded with `sample_00860`:

```bash
python tools/check_interactive_browser.py --url 'http://localhost:8787/#TOKEN' \
  --output /tmp/plats-browser-check
```

A local run record and screenshots are under
`runs_from_260914/interactive/01_browser_inference/` in the research workspace.
