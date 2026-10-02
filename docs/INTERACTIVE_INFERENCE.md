# Interactive sheet inference

The browser viewer supports prompted sheets and automatic instance discovery.
It shows three linked slices, an enlarged single-slice layout, and rotatable 3D
prediction/reference surfaces. CT features are cached on the GPU server and
reused across clicks and automatic clustering.

## Launch and connect over SSH

Use the same environment as `plats.py`, with the optional viewer transports and
TIFF codecs installed:

```bash
python -m pip install -r requirements-viewer.txt
python viewer.py \
  --image examples/sample_00860/image.npy \
  --gt examples/sample_00860/gt_instances.npy \
  --device cuda:0 --port 8787 \
  --output runs_from_260914/interactive/exports
```

The Git repository excludes model weights and example arrays. Extract the
reproduction archive into this checkout, or pass
`--bundle-root /path/to/extracted/PLatS` and absolute image/reference paths.
For the existing research workspace:

```bash
~/Project/vesuvius_p2sd/.venv/bin/python ~/Project/PLatS/viewer.py \
  --bundle-root ~/Project/vesuvius_p2sd/submission/PLatS \
  --image ~/Project/vesuvius_p2sd/submission/PLatS/examples/sample_00860/image.npy \
  --gt ~/Project/vesuvius_p2sd/submission/PLatS/examples/sample_00860/gt_instances.npy \
  --device cuda:0 --port 8787 \
  --output ~/Project/vesuvius_p2sd/runs_from_260914/interactive/exports
```

Keep the server in a terminal or `tmux`. On your local computer:

```bash
ssh -N -L 8787:127.0.0.1:8787 USER@SERVER
```

Open the **complete localhost URL printed by the server, including `#token`**.
It is retained in that browser tab's session storage and removed from the address
bar. WebGL2 is needed for 3D. The server listens only on `127.0.0.1`; all browser
assets are bundled. This is a single-user research app: tabs share the current
crop. `--token-file /private/path/viewer.token` optionally retains the URL across
restarts; protect this file like the printed token.

## Prompting and viewing

- **Click:** add a positive point to the active sheet; other slices move to it.
  Use 1–8 points on the same sheet. Only points on the exact displayed plane
  appear in a slice; all appear in 3D.
- **Shift-click:** move the crosshair without adding a point.
- **Wheel / sliders:** scroll individual slices.
- **Decode after each click:** enabled initially. Disable it to place points
  before pressing **Decode sheet / Enter**.
- **Undo point / Backspace**, × beside a point, or **Clear sheet**: edit prompts.
  Old predictions disappear immediately; stale background results are discarded.
- **+ New / N:** create another independently colored prompted sheet (up to 16).
- **Layout:** choose three slices or one enlarged X/Y/Z slice. Uncheck
  **Show 3D panel** to devote the workspace to the slice. This enlarges native
  voxels; it does not resample the CT into higher physical resolution.
- **CT window / opacity:** change display only. **Threshold** changes the active
  prompted mask using its cached probability, without another model pass.
- **3D:** drag to rotate, right-drag to pan, wheel to zoom. The renderer redraws
  on changes rather than continuously rendering an idle view.

## Reference-sheet comparison

Open an aligned reference together with the local CT. Select its interpretation:

- **Sheet instance IDs:** preserve positive IDs exactly.
- **Challenge labels:** `0=background, 1=sheet, 2=ignore`; ignore is excluded.
- **Binary:** positive voxels are foreground.

Binary/challenge foreground is split into six-connected components for selection;
these components are not guaranteed to represent physical sheet identities.
Use an instance-labeled or single-sheet reference when identity matters.

Choose **Reference sheet**, then enable **Selected reference in 3D**. Its green
translucent surface is shown alongside the prompted/automatic prediction.
**Reference contours** shows the selected ID on slices. References are used only
for display/export; they never enter inference or automatic seed selection.

## Challenge TIFF files

**Open another crop → Local file / challenge TIFF** accepts a multipage CT TIFF
and optional label TIFF, or use:

```bash
python viewer.py --image /data/test_images/CASE.tif \
  --gt /data/labels/CASE.tif --reference-kind challenge --device cuda:0
```

TIFF pages are read as **Z/Y/X** and transposed to the model/viewer's **X/Y/Z**.
NumPy and NIFTI array axes are retained and interpreted as X/Y/Z. Each dimension
must be 32–320; smaller crops are center-padded to 320³ and cropped back after
inference. CT must already be uint8. No intensity normalization or border erasure
is applied. NIFTIs are not canonicalized. If both CT and reference are NIFTIs,
their affines must match; every reference must match the CT shape after conversion.

## Official remote CT Zarr

Choose **Open another crop → Official CT Zarr URL**, paste an OME-Zarr root or
array URL, and enter the starting **X, Y, Z** coordinates of a 320³ patch.
Coordinates are voxel indices at the selected resolution level; they are not
micrometers or a patch-center coordinate. Root URLs use the selected level
(default 0); an array URL such as `...zarr/0` fixes the level.

For example:

```bash
python viewer.py \
  --zarr-url https://vesuvius-challenge-open-data.s3.amazonaws.com/PHerc0139/volumes/20250728140407-9.362um-1.2m-113keV-masked.zarr \
  --start-xyz 3612 3942 4572 --level 0 --device cuda:0
```

Only intersecting chunks are read, with eight concurrent requests. The extracted
patch is cached in `zarr_cache/` beside the export directory. OME axis metadata
controls the permutation to XYZ; a plain 3D array without axis metadata explicitly
falls back to the official CT ZYX convention. Inputs must be 3D uint8. Bounds are
checked; a patch extending past the source volume is rejected rather than clipped.
Public HTTP(S) and `s3://` URLs are supported. Source axes, transforms, level,
origin, and checksum are saved in the export. NIFTI affines preserve OME scale
and translation; their coordinate units follow the source metadata.

## Automatic instance segmentation

Click **Segment all sheets**. The released 0058 stack uses the released 0022
foreground proposer and the same frozen 512-seed clustering/decoding recipe as
`plats.py automatic`. A training-run checkpoint uses its own co-trained binary
head. Neither path uses GT or manual prompts. Automatic results are separate from
prompted sheets, so running it does not discard manual work.

All automatic IDs can be overlaid on the slices. **Inspect instance** selects one
for 3D; surfaces are extracted lazily to avoid loading every large mesh at once.
The automatic ID volume remains complete. Export records clustering, seeds,
settings, and timings. Automatic masks include the released recipe's cleanup;
prompted masks remain raw probability thresholds. Neither path applies the
paper's optional AE repair/closing/dusting or leaderboard ignore processing.

For a jointly trained model, use `--run-dir /path/to/run`, optional
`--checkpoint-dir /path/to/run/checkpoints/step_100000`, and
`--training-root /path/to/workspace` to resolve saved config paths. Automatic
inference requires the matching `dense_last.pt`; prompting alone does not.
`--device cpu` is available but has not been performance-tested.

## Lossless export

Choose **NIFTI**, **TIFF**, or **NIFTI + TIFF**, then **Export results**. A fresh
timestamped folder on the server contains:

- CT, optional reference union/IDs, and `session.json` with source coordinates,
  checkpoints/config checksums, prompts, settings, and timings.
- Each `sheet_XX/`: mask, float32 probability, prompt-point volume, coordinates,
  and links to CT/reference. Prompted masks may overlap; `union` and
  `overlap_count` preserve that information without inventing exclusive IDs.
- `automatic/instances`: the complete **uint16 instance labels**;
  `automatic/prediction`: the **uint8 binary union** in challenge format;
  foreground proposer mask, seed coordinates, and clustering records.

Every TIFF is a multipage **ZYX** stack with **lossless Deflate compression**.
Instance IDs are never replaced by display colors. NIFTIs use model XYZ arrays
with the input affine. Remote coordinates stay in `session.json`; TIFF pixel
indices are local to the exported patch. Probability values remain float32.

## Checks

```bash
python -m pytest -q tests/test_interactive.py tests/test_viewer_formats.py
```

Checks cover asymmetric TIFF/XYZ round-trips, IDs above 255, reference ignore
semantics, Deflate tags, remote chunk selection and bounds, prompt coordinates,
NIFTI affines, and stale asynchronous results. Chromium testing also exercised
reference/prediction surfaces, focus layouts, TIFF loading/export, the automatic
button, and actual PHerc0139 remote loading and prompting.

On the released example, automatic labels matched the frozen prediction exactly:
14 instances in 14.1 seconds on GPU0. A warm prompted decode took about 0.30 s
(0.90 s including mesh extraction), with bitwise CLI probability agreement. The
real remote 320³ read took 5.8 s on this server. Network/rendering latency varies.
Local records are in `runs_from_260914/interactive/01_browser_inference/` and
`02_viewer_extensions/` in the research workspace.
