import { SurfaceView } from "./scene.js";
const $ = (id) => document.getElementById(id);
const mappings = [
  { fixed: 0, h: 2, v: 1 },
  { fixed: 1, h: 2, v: 0 },
  { fixed: 2, h: 0, v: 1 },
];
const colors = [
  "#ff9982",
  "#77c9ff",
  "#c6a0ff",
  "#f0d16d",
  "#71ddd0",
  "#f99cd6",
  "#a8d56e",
  "#eeb981",
  "#ed647b",
  "#589cef",
  "#916eef",
  "#b6bd5c",
  "#40a88e",
  "#d877bb",
  "#b0ceab",
  "#c6985d",
];
const state = {
  meta: null,
  image: null,
  gt: null,
  slices: [0, 0, 0],
  sheets: new Map(),
  active: 1,
  busy: false,
  loading: false,
  pending: new Set(),
  boxes: [],
  timer: null,
};
const token = location.hash.slice(1) || sessionStorage.getItem("plats-token");
if (token) {
  sessionStorage.setItem("plats-token", token);
  history.replaceState(null, "", location.pathname);
}
let scene;
try {
  scene = new SurfaceView($("scene"));
} catch (error) {
  $("scene").textContent =
    `3D requires WebGL2: ${error.message}. Slice prompting remains available.`;
}
const planes = mappings.map(() => document.createElement("canvas"));

async function api(path, body, binary = false) {
  const response = await fetch(path, {
    headers: {
      "X-PLatS-Token": token || "",
      ...(body ? { "Content-Type": "application/json" } : {}),
    },
    ...(body ? { method: "POST", body: JSON.stringify(body) } : {}),
  });
  if (!response.ok) {
    let message;
    try {
      message = (await response.json()).error;
    } catch {
      message = `HTTP ${response.status}`;
    }
    throw new Error(message);
  }
  return binary ? response.arrayBuffer() : response.json();
}
function status(message, busy = false, error = false) {
  $("status").textContent = message;
  $("status").className = error ? "error" : busy ? "busy" : "";
}
function notice(message, error = false) {
  $("notice").textContent = message;
  $("notice").hidden = false;
  $("notice").className = error ? "error" : "";
}
$("notice").onclick = () => {
  $("notice").hidden = true;
};
function active() {
  return state.sheets.get(state.active);
}
function newSheet() {
  const id = Array.from({ length: 16 }, (_, i) => i + 1).find((id) => !state.sheets.has(id));
  if (!id) {
    notice(
      "Up to 16 sheets can be kept in this session. Export and open the crop again to start another session.",
    );
    return;
  }
  state.sheets.set(id, {
    id,
    color: colors[id - 1],
    points: [],
    threshold: 0.5,
    edit: 0,
    prediction: null,
    dirty: false,
  });
  state.active = id;
  renderControls();
  draw();
}
function renderControls() {
  const sheet = active();
  if (!sheet) return;
  $("sheets").replaceChildren();
  for (const s of state.sheets.values()) {
    const b = document.createElement("button");
    b.className = "sheet" + (s.id === state.active ? " active" : "");
    const dot = document.createElement("i");
    dot.style.background = s.color;
    const text = document.createElement("span");
    text.textContent = `Sheet ${s.id}`;
    const small = document.createElement("small");
    small.textContent = s.dirty
      ? "changed"
      : s.prediction
        ? "decoded"
        : `${s.points.length} points`;
    b.append(dot, text, small);
    b.onclick = () => {
      state.active = s.id;
      renderControls();
      draw();
    };
    $("sheets").append(b);
  }
  $("points").replaceChildren();
  sheet.points.forEach((p, i) => {
    const li = document.createElement("li");
    li.textContent = p.join(", ");
    const remove = document.createElement("button");
    remove.textContent = "×";
    remove.title = `Remove point ${i + 1}`;
    remove.onclick = () => {
      sheet.points.splice(i, 1);
      edited(sheet);
    };
    li.append(remove);
    $("points").append(li);
  });
  $("point-count").textContent = `${sheet.points.length} / 8 positive points · axes 0, 1, 2`;
  $("threshold").value = sheet.threshold;
  $("threshold-value").textContent = sheet.threshold.toFixed(2);
  $("decode").disabled = !sheet.points.length || state.loading;
  $("undo").disabled = !sheet.points.length || state.loading;
  $("export").disabled =
    state.loading ||
    state.busy ||
    [...state.sheets.values()].some((s) => s.dirty) ||
    ![...state.sheets.values()].some((s) => s.prediction);
  $("new-sheet").disabled = state.loading;
}
function edited(sheet) {
  sheet.edit++;
  sheet.dirty = sheet.points.length > 0;
  sheet.prediction = null;
  scene?.removeSheet(sheet.id);
  if (!sheet.points.length) {
    state.pending.delete(sheet.id);
    api("/api/delete", { sheet: sheet.id, generation: state.meta.generation }).catch((e) =>
      notice(e.message, true),
    );
  }
  renderControls();
  draw();
  scene?.updatePoints(state.sheets);
  clearTimeout(state.timer);
  if ($("auto").checked && sheet.points.length)
    state.timer = setTimeout(() => requestDecode(sheet.id), 200);
}
function indexAt(point) {
  const [, ny, nz] = state.meta.shape;
  return (point[0] * ny + point[1]) * nz + point[2];
}
function draw() {
  if (!state.image) return;
  const shape = state.meta.shape,
    alpha = +$("opacity").value,
    showGT = $("show-gt").checked && state.gt;
  const lo = +$("low").value,
    windowScale = 255 / (Math.max(lo + 1, +$("high").value) - lo);
  const strides = [shape[1] * shape[2], shape[2], 1];
  const ready = [...state.sheets.values()]
    .filter((s) => s.prediction)
    .map((s) => ({ ...s, rgb: s.color.match(/\w\w/g).map((h) => parseInt(h, 16)) }));
  for (let axis = 0; axis < 3; axis++) {
    const { fixed, h, v } = mappings[axis],
      w = shape[h],
      height = shape[v],
      off = planes[axis];
    if (off.width !== w || off.height !== height) {
      off.width = w;
      off.height = height;
    }
    const pixels = new ImageData(w, height),
      p = [0, 0, 0];
    p[fixed] = state.slices[fixed];
    for (let y = 0; y < height; y++)
      for (let x = 0; x < w; x++) {
        p[h] = x;
        p[v] = y;
        const k = indexAt(p),
          j = (y * w + x) * 4;
        let grey = Math.min(255, Math.max(0, Math.round((state.image[k] - lo) * windowScale)));
        let rgb = [grey, grey, grey];
        for (const sheet of ready)
          if (sheet.prediction.mask[k])
            rgb = rgb.map((value, c) => value * (1 - alpha) + sheet.rgb[c] * alpha);
        if (showGT && state.gt[k]) {
          const edge =
            x === 0 ||
            y === 0 ||
            x === w - 1 ||
            y === height - 1 ||
            !state.gt[k - strides[h]] ||
            !state.gt[k + strides[h]] ||
            !state.gt[k - strides[v]] ||
            !state.gt[k + strides[v]];
          if (edge) rgb = [116, 237, 135];
        }
        pixels.data[j] = rgb[0];
        pixels.data[j + 1] = rgb[1];
        pixels.data[j + 2] = rgb[2];
        pixels.data[j + 3] = 255;
      }
    off.getContext("2d").putImageData(pixels, 0, 0);
    const canvas = $(`slice-${axis}`),
      rect = canvas.getBoundingClientRect(),
      dpr = Math.min(devicePixelRatio, 2);
    canvas.width = Math.round(rect.width * dpr);
    canvas.height = Math.round(rect.height * dpr);
    const c = canvas.getContext("2d");
    c.setTransform(dpr, 0, 0, dpr, 0, 0);
    c.imageSmoothingEnabled = false;
    const scale = Math.min(rect.width / w, rect.height / height),
      bw = w * scale,
      bh = height * scale,
      bx = (rect.width - bw) / 2,
      by = (rect.height - bh) / 2;
    state.boxes[axis] = { x: bx, y: by, w: bw, h: bh };
    c.drawImage(off, bx, by, bw, bh);
    const cx = bx + (state.slices[h] + 0.5) * scale,
      cy = by + (state.slices[v] + 0.5) * scale;
    c.strokeStyle = "#e6f3ff88";
    c.lineWidth = 1;
    c.setLineDash([4, 5]);
    c.beginPath();
    c.moveTo(cx, by);
    c.lineTo(cx, by + bh);
    c.moveTo(bx, cy);
    c.lineTo(bx + bw, cy);
    c.stroke();
    c.setLineDash([]);
    for (const sheet of state.sheets.values())
      sheet.points.forEach((point, i) => {
        if (point[fixed] !== state.slices[fixed]) return;
        const x = bx + (point[h] + 0.5) * scale,
          y = by + (point[v] + 0.5) * scale;
        c.beginPath();
        c.arc(x, y, sheet.id === state.active ? 5 : 4, 0, Math.PI * 2);
        c.fillStyle = sheet.color;
        c.fill();
        c.strokeStyle = "#08111d";
        c.lineWidth = 1.6;
        c.stroke();
        c.font = "bold 11px system-ui";
        c.lineWidth = 3;
        c.strokeStyle = "#08111d";
        c.strokeText(String(i + 1), x + 7, y - 5);
        c.fillStyle = sheet.color;
        c.fillText(String(i + 1), x + 7, y - 5);
      });
    $(`slider-${axis}`).value = state.slices[axis];
    $(`slice-value-${axis}`).textContent = `${state.slices[axis]} / ${shape[axis] - 1}`;
  }
  scene?.updatePlanes(state.slices, $("planes").checked);
  scene?.showSurfaces($("surfaces").checked);
  $("window-value").textContent = `${$("low").value}–${$("high").value}`;
  $("opacity-value").textContent = `${Math.round(alpha * 100)}%`;
}
async function waitJob(job) {
  while (true) {
    const result = await api(`/api/jobs/${job}`);
    if (result.status === "error") throw new Error(result.error);
    if (result.status === "done") return result.result;
    await new Promise((resolve) => setTimeout(resolve, 250));
  }
}
async function fetchPrediction(sheet, result) {
  const [packed, mesh] = await Promise.all([
    api(`/api/mask/${sheet.id}`, null, true),
    api(`/api/mesh/${sheet.id}`, null, true),
  ]);
  const bytes = new Uint8Array(packed),
    mask = new Uint8Array(state.image.length);
  for (let i = 0; i < mask.length; i++) mask[i] = (bytes[i >> 3] >> (i & 7)) & 1;
  return { ...result, mask, mesh };
}
async function requestDecode(id = state.active) {
  const sheet = state.sheets.get(id);
  if (!sheet?.points.length || state.loading) return;
  state.pending.add(id);
  if (state.busy) return;
  state.busy = true;
  renderControls();
  try {
    while (state.pending.size) {
      const key = state.pending.values().next().value;
      state.pending.delete(key);
      const s = state.sheets.get(key);
      if (!s?.points.length) continue;
      const edit = s.edit,
        generation = state.meta.generation;
      status(
        `Decoding sheet ${key} from ${s.points.length} point${s.points.length === 1 ? "" : "s"}…`,
        true,
      );
      const job = await api("/api/predict", {
        sheet: key,
        points: s.points,
        threshold: s.threshold,
        generation,
      });
      const result = await waitJob(job.job);
      if (result.discarded || s.edit !== edit || state.meta.generation !== generation) continue;
      const prediction = await fetchPrediction(s, result);
      if (s.edit !== edit || state.meta.generation !== generation) continue;
      s.prediction = prediction;
      s.dirty = false;
      scene?.setSheet(key, prediction.mesh, s.color);
      const t = result.timings;
      $("timing").textContent =
        `Last decode: ${t.decode_seconds.toFixed(2)}s · with surface: ${t.total_seconds.toFixed(2)}s. CT encodes: ${t.image_encodes}.`;
      $("mesh-info").textContent =
        `Sheet ${key}: ${result.voxels.toLocaleString()} voxels · ${result.triangles.toLocaleString()} triangles`;
      status(`Sheet ${key} ready · shared CT features reused`);
      renderControls();
      draw();
    }
  } catch (error) {
    notice(error.message, true);
    status("Inference failed — see message", false, true);
    state.pending.clear();
  } finally {
    state.busy = false;
    renderControls();
  }
}
async function loadData() {
  state.meta = await api("/api/meta");
  const [image, gt] = await Promise.all([
    api("/api/image", null, true),
    state.meta.has_gt ? api("/api/gt", null, true) : Promise.resolve(null),
  ]);
  state.image = new Uint8Array(image);
  state.gt = gt ? new Uint8Array(gt) : null;
  state.slices = state.meta.shape.map((n) => Math.floor(n / 2));
  state.sheets.clear();
  state.pending.clear();
  $("crop-name").textContent = state.meta.path.split("/").pop();
  $("crop-name").title = state.meta.path;
  $("crop-info").textContent = `${state.meta.shape.join(" × ")} · uint8 · ${state.meta.device}`;
  $("image-path").value = state.meta.path;
  $("gt-path").value = state.meta.gt_path;
  $("show-gt").disabled = !state.meta.has_gt;
  mappings.forEach((_, i) => {
    $(`slider-${i}`).max = state.meta.shape[i] - 1;
    planes[i].width = state.meta.shape[mappings[i].h];
    planes[i].height = state.meta.shape[mappings[i].v];
  });
  scene?.reset(state.meta.shape, mappings, planes);
  for (const saved of state.meta.sheets || []) {
    const sheet = {
      ...saved,
      color: colors[saved.id - 1],
      edit: 0,
      prediction: null,
      dirty: false,
    };
    sheet.prediction = await fetchPrediction(sheet, saved);
    state.sheets.set(sheet.id, sheet);
    scene?.setSheet(sheet.id, sheet.prediction.mesh, sheet.color);
  }
  if (state.sheets.size) {
    state.active = state.sheets.keys().next().value;
    renderControls();
    draw();
  } else newSheet();
  scene?.updatePoints(state.sheets);
  status(state.meta.ready);
  if (state.meta.ready.startsWith("Preparing")) pollPreparation(state.meta.generation);
}
async function pollPreparation(generation) {
  await new Promise((resolve) => setTimeout(resolve, 1000));
  if (state.meta.generation !== generation) return;
  try {
    const meta = await api("/api/meta");
    if (!state.busy) status(meta.ready, false, meta.ready.includes("failed"));
    if (meta.ready.startsWith("Preparing")) pollPreparation(generation);
  } catch (error) {
    status(error.message, false, true);
  }
}
for (let axis = 0; axis < 3; axis++) {
  const canvas = $(`slice-${axis}`);
  // Pointer events preserve subpixel coordinates; MouseEvent.click can round
  // them to integer CSS pixels and shift a thin-sheet prompt by one voxel.
  canvas.addEventListener("pointerup", (event) => {
    if (event.button !== 0 || !state.image || state.loading) return;
    const rect = canvas.getBoundingClientRect(),
      box = state.boxes[axis],
      x = event.clientX - rect.left,
      y = event.clientY - rect.top;
    if (x < box.x || y < box.y || x >= box.x + box.w || y >= box.y + box.h) return;
    const { h, v } = mappings[axis],
      point = [...state.slices];
    point[h] = Math.floor(((x - box.x) / box.w) * state.meta.shape[h]);
    point[v] = Math.floor(((y - box.y) / box.h) * state.meta.shape[v]);
    state.slices = point;
    if (!event.shiftKey) {
      const sheet = active();
      if (sheet.points.length >= 8) {
        notice(
          "This model supports up to eight positive points. Remove one before adding another.",
        );
        draw();
        return;
      }
      if (!sheet.points.some((p) => p.every((value, i) => value === point[i]))) {
        sheet.points.push([...point]);
        edited(sheet);
      }
    }
    draw();
  });
  canvas.addEventListener(
    "wheel",
    (event) => {
      event.preventDefault();
      if (!state.image) return;
      state.slices[axis] = Math.max(
        0,
        Math.min(state.meta.shape[axis] - 1, state.slices[axis] + Math.sign(event.deltaY)),
      );
      draw();
    },
    { passive: false },
  );
  $(`slider-${axis}`).oninput = (event) => {
    state.slices[axis] = +event.target.value;
    draw();
  };
  new ResizeObserver(() => draw()).observe(canvas);
}
$("new-sheet").onclick = newSheet;
$("undo").onclick = () => {
  const sheet = active();
  if (sheet?.points.length) {
    sheet.points.pop();
    edited(sheet);
  }
};
$("clear").onclick = () => {
  const sheet = active();
  sheet.points = [];
  edited(sheet);
};
$("decode").onclick = () => requestDecode();
$("threshold").oninput = () => {
  active().threshold = +$("threshold").value;
  $("threshold-value").textContent = active().threshold.toFixed(2);
};
$("threshold").onchange = () => {
  if (active().points.length) edited(active());
};
$("low").oninput = () => {
  if (+$("low").value >= +$("high").value) $("high").value = +$("low").value + 1;
  draw();
};
$("high").oninput = () => {
  if (+$("high").value <= +$("low").value) $("low").value = +$("high").value - 1;
  draw();
};
for (const id of ["opacity", "show-gt", "planes", "surfaces"]) $(id).oninput = draw;
$("reset-camera").onclick = () => scene?.resetCamera();
$("auto").onchange = () => {
  if ($("auto").checked && active()?.dirty) requestDecode();
};
$("load").onclick = async () => {
  if (state.busy) {
    notice("Wait for the current decode before opening another crop.");
    return;
  }
  state.loading = true;
  clearTimeout(state.timer);
  renderControls();
  status("Opening crop and caching CT features…", true);
  try {
    const result = await api("/api/load", { image: $("image-path").value, gt: $("gt-path").value });
    await waitJob(result.job);
    await loadData();
  } catch (error) {
    notice(error.message, true);
    status("Could not open crop", false, true);
  } finally {
    state.loading = false;
    renderControls();
  }
};
$("export").onclick = async () => {
  const revisions = Object.fromEntries(
    [...state.sheets.values()]
      .filter((s) => s.prediction)
      .map((s) => [s.id, s.prediction.revision]),
  );
  state.loading = true;
  renderControls();
  status("Writing NIFTIs on the server…", true);
  try {
    const job = await api("/api/export", { generation: state.meta.generation, revisions });
    const result = await waitJob(job.job);
    notice(`Saved CT, prompts, probabilities, masks, and session provenance:\n${result.path}`);
    status("NIFTI export complete");
  } catch (error) {
    notice(error.message, true);
    status("Export failed", false, true);
  } finally {
    state.loading = false;
    renderControls();
  }
};
document.addEventListener("keydown", (event) => {
  if (["INPUT", "TEXTAREA", "SELECT"].includes(document.activeElement?.tagName) || state.loading)
    return;
  if (event.key === "Enter") {
    event.preventDefault();
    requestDecode();
  }
  if (event.key === "Backspace") {
    event.preventDefault();
    $("undo").click();
  }
  if (event.key.toLowerCase() === "n") {
    event.preventDefault();
    newSheet();
  }
});
loadData().catch((error) => {
  notice(error.message, true);
  status("Connection failed", false, true);
});
// Expose read-only diagnostics for coordinate and browser integration checks.
window.platsDiagnostics = () => ({
  shape: state.meta?.shape,
  slices: [...state.slices],
  busy: state.busy,
  sheets: [...state.sheets.values()].map((s) => ({
    id: s.id,
    points: s.points,
    dirty: s.dirty,
    revision: s.prediction?.revision,
  })),
  boxes: state.boxes,
  webgl: !!scene,
});
