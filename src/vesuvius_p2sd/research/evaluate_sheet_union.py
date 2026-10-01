"""Fixed-prompt sheet/union evaluation with raw NIFTI tuples and public scores."""
from __future__ import annotations

import argparse
from collections import deque
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch

from vesuvius_p2sd.data.build_ignore_masks import load_ignore_mask
from vesuvius_p2sd.data.dataset import load_jsonl
from vesuvius_p2sd.data.scene_probes import load_scene_probe_manifest
from vesuvius_p2sd.eval.sheet_report import (
    annotated_box_bounds, rectangular_border_ignore, border_only_ignore, dice, link_assets, save_nifti, score_masks, shared_assets, update_instances, write_sheet_tuple,
)
from vesuvius_p2sd.models.p2sd import build_p2sd
from vesuvius_p2sd.research.evaluate_p2sd_case_scenes import _canvas_image_tensor, _source_canvas_slices
from vesuvius_p2sd.research.run_status import write_json
from vesuvius_p2sd.train.common import amp_dtype, autocast_context, load_ae_from_config, load_model_state
from vesuvius_p2sd.train.train_p2sd import build_static_latent_codec, decoded_foreground_probability
from vesuvius_p2sd.utils.config import load_config


class PendingCaseScores:
    """Bound GPU-produced scoring work while committing cases in manifest order."""

    def __init__(self, limit, finalize):
        if limit < 1:
            raise ValueError('max_in_flight_cases must be positive')
        self.limit, self.finalize, self.jobs = limit, finalize, deque()

    def add(self, job):
        self.jobs.append(job)
        while self.jobs and all(f.done() for f in self.jobs[0]['futures']):
            self.finalize(self.jobs.popleft())
        if len(self.jobs) >= self.limit:
            self.finalize(self.jobs.popleft())

    def finish(self):
        while self.jobs:
            self.finalize(self.jobs.popleft())


def summarize(sheet_rows, case_rows):
    def statistics(rows, keys):
        result = {}
        for key in keys:
            values = np.asarray([r[key] for r in rows if key in r], float)
            if values.size:
                result[key] = {"mean": float(values.mean()), "median": float(np.median(values)),
                               "p10": float(np.quantile(values, .1)), "min": float(values.min())}
        return result
    score_keys = ["dice", "surface_dice_tau2", "leaderboard_formula_score", "toposcore", "voi_score"]
    sheet = statistics(sheet_rows, score_keys + ["latent_raw_mse", "latent_normalized_mse",
        "latent_relative_mse", "ae_reconstruction_dice", "pred_components_raw", "pred_components_scored"])
    union = statistics(case_rows, score_keys + ["prompted_union_dice"])
    return {"sheet_prompt_count": len(sheet_rows), "case_variant_count": len(case_rows),
            "sheet": sheet, "union": union}


def write_reports(output, sheet_rows, case_rows, protocol):
    for name, rows in [("sheet_metrics.jsonl", sheet_rows), ("case_metrics.jsonl", case_rows)]:
        (output / name).write_text("".join(json.dumps(row) + "\n" for row in rows))
    summary = {**protocol, **summarize(sheet_rows, case_rows)}
    write_json(output / "summary.json", summary)
    lines = ["# Sheet and patch-union evaluation", "", "| Scope | Dice | Surface Dice @2 vox | Public-formula score | TopoScore | VOI score |",
             "|---|---:|---:|---:|---:|---:|"]
    for name in ["sheet", "union"]:
        s = summary[name]
        lines.append("| " + name + " | " + " | ".join(f"{s[k]['mean']:.5f}" for k in
            ["dice", "surface_dice_tau2", "leaderboard_formula_score", "toposcore", "voi_score"]) + " |")
    lines += ["", "| Latent / AE diagnostic | Sheet mean |", "|---|---:|"]
    for key in ["latent_raw_mse", "latent_normalized_mse", "latent_relative_mse", "ae_reconstruction_dice"]:
        if key in summary["sheet"]:
            lines.append(f"| {key} | {summary['sheet'][key]['mean']:.6f} |")
    lines += ["", "Normalized MSE uses the training AE's per-channel statistics. Relative MSE divides each sheet's normalized MSE by its mean-latent predictor MSE.",
        "AE reconstruction Dice measures GT → frozen AE → mask; it is a reconstruction reference, not a mathematical upper bound."]
    lines += ["", "Ignore policy: " + protocol["ignore_policy"] + ". No closing, component filtering, or ignore restoration.",
        "Union scores include all annotated GT sheets, including those without prompts. Prompted-union Dice is also recorded.",
        "Sheet scores apply the public binary formula to each fixed prompted sheet; they are not an official instance leaderboard.",
        "These are GT-prompted reconstruction scores, not automatic instance-discovery results.",
        "NIFTIs are raw masks in source ZYX array coordinates with voxel-identity affine; scoring erases ignore from both sides.",
        "", "Each `cases/<case>/sheets/<id>/<prompt>/` contains image (symlink), prompt points, GT sheet, decoded sheet, ignore (symlink), and metrics."]
    (output / "REPORT.md").write_text("\n".join(lines) + "\n")
    return summary


@torch.inference_mode()
def evaluate(*, checkpoint_path, source_run_dir, scene_manifest_path, output_dir,
             prompt_ids=("p00",), prompt_batch_size=4, cpu_workers=4,
             topology_backend="compact_exact", max_cases=None, resume=False, ignore_policy="border_box",
             max_in_flight_cases=3, topology_tile_batch_size=1):
    run, output = Path(source_run_dir), Path(output_dir)
    if output.exists() and any(output.iterdir()) and not resume:
        raise FileExistsError(output)
    output.mkdir(parents=True, exist_ok=True)
    cfg = load_config(run / "resolved_config.yaml")
    patch_size = tuple(cfg["data"]["patch_size"])
    manifest = load_scene_probe_manifest(scene_manifest_path, patch_size=patch_size)
    source_manifest = Path(cfg["data"]["dataset_root"]) / f"manifest_{manifest.split}_ignore.jsonl"
    if not source_manifest.exists():
        raise FileNotFoundError(f"Real ignore-mask manifest required: {source_manifest}")
    rows = {r["case_id"]: r for r in load_jsonl(source_manifest)}
    device = torch.device(cfg.get("device", "cuda"))
    dtype = amp_dtype(cfg["training"].get("amp_dtype"))
    channels_last = bool(cfg["training"].get("channels_last_3d", False))
    ae, _ = load_ae_from_config(cfg["target_ae"]["config_path"], cfg["target_ae"]["checkpoint_path"])
    ae = ae.to(device).eval()
    model = build_p2sd(cfg, latent_channels=ae.latent_channels).to(device).eval()
    if channels_last:
        model = model.to(memory_format=torch.channels_last_3d)
        ae = ae.to(memory_format=torch.channels_last_3d)
    checkpoint = load_model_state(model, checkpoint_path)
    step = int(checkpoint["step"])
    del checkpoint
    codec = build_static_latent_codec(cfg["p2sd"]["loss"]["latent_normalization"], latent_channels=ae.latent_channels, device=device)
    threshold = float(cfg.get("metrics", {}).get("threshold", .5))
    cases = list(manifest.cases)[:max_cases]
    if ignore_policy not in {"border_box", "border_only", "source"}:
        raise ValueError(ignore_policy)
    protocol = {"protocol": {"border_box": "sheet_union_v3_box", "border_only": "sheet_union_v2_border", "source": "sheet_union_v1"}[ignore_policy], "checkpoint_path": str(Path(checkpoint_path).resolve()),
        "checkpoint_step": step, "source_run_dir": str(run), "scene_manifest_path": str(scene_manifest_path),
        "source_manifest": str(source_manifest), "case_count": len(cases), "prompt_ids": list(prompt_ids),
        "prompt_batch_size": prompt_batch_size, "threshold": threshold, "topology_backend": topology_backend,
        "ignore_policy": ({"border_box": "outside source-valid bounding box ignored; inward source ignore is GT background; predictions retained",
            "border_only": "fixed conversion border only; inward source ignore is GT background; predictions retained",
            "source": "source ignore only; erase both sides for scoring"}[ignore_policy]),
        "border_widths": {c.case_id: int(rows[c.case_id]["erased_border_width"]) for c in cases},
        "component_filter": 0,
        "target_contract": "all_sheets", "sheet_score_contract": "fixed GT-prompted binary sheet pairs",
        "instance_overlap_policy": "highest probability; ties choose lower prompt-associated ID",
        "surface_tolerance_voxels": 2, "leaderboard_weights": [.3, .35, .35]}
    if resume and (output / "protocol.json").exists():
        if json.loads((output / "protocol.json").read_text()) != protocol:
            raise ValueError("Cannot resume evaluation with a different protocol/checkpoint")
    write_json(output / "protocol.json", protocol)
    with (output / "execution.jsonl").open("a") as stream:
        stream.write(json.dumps({"unix_time": time.time(), "cpu_workers": cpu_workers,
            "topology_tile_batch_size": topology_tile_batch_size, "resume": resume, "max_in_flight_cases": max_in_flight_cases}) + "\n")
    all_sheets, all_cases = [], []
    started = time.monotonic()
    completed_cases = 0

    def finalize(job):
        nonlocal completed_cases
        folder = job['folder']
        if 'saved' in job:
            saved = job['saved']
        else:
            sheet_results, case_results = [], []
            for jobs, results in [(job['sheet_jobs'], sheet_results), (job['union_jobs'], case_results)]:
                for tuple_dir, metrics, future in jobs:
                    metrics.update(future.result())
                    write_json(tuple_dir / 'metrics.json', metrics)
                    results.append(metrics)
            saved = {'sheet_rows': sheet_results, 'case_rows': case_results}
            write_json(folder / 'complete.json', saved)
        all_sheets.extend(saved['sheet_rows']); all_cases.extend(saved['case_rows'])
        completed_cases += 1
        progress = {'cases_completed': completed_cases, 'cases_total': len(cases), 'case_id': folder.name,
                    'elapsed_s': time.monotonic() - started}
        write_json(output / 'progress.json', progress)
        print(json.dumps(progress), flush=True)
        write_reports(output, all_sheets, all_cases, protocol)

    pending_cases = PendingCaseScores(max_in_flight_cases, finalize)
    # Isolate native topology scoring from the CUDA producer and its Python
    # interpreter; spawn also avoids forking an initialized CUDA context.
    with ProcessPoolExecutor(max_workers=cpu_workers,
            mp_context=multiprocessing.get_context("spawn")) as pool:
        for case in cases:
            folder = output / "cases" / case.case_id
            if resume and (folder / "complete.json").exists():
                saved = json.loads((folder / "complete.json").read_text())
                pending_cases.add(dict(folder=folder, saved=saved, futures=[]))
                continue
            row = rows[case.case_id]
            image = np.load(row["image_path"], mmap_mode="r")
            gt_ids = np.asarray(np.load(row["components_path"], mmap_mode="r"))
            source_ignore = load_ignore_mask(row["ignore_path"], image.shape)
            width = int(row["erased_border_width"])
            ignore = (rectangular_border_ignore(source_ignore) if ignore_policy == "border_box" else
                      border_only_ignore(source_ignore, width) if ignore_policy == "border_only" else source_ignore)
            gt_ids = np.where(source_ignore, 0, gt_ids)
            if tuple(image.shape) != case.source_shape or gt_ids.shape != image.shape:
                raise ValueError(f"Source/manifest shape mismatch: {case.case_id}")
            assets = shared_assets(run / "eval_assets", case.case_id, image, ignore,
                                   [row[k] for k in ["image_path", "components_path", "ignore_path"]],
                                   policy=f"{ignore_policy}:border_width={width}", source_ignore=source_ignore)
            source_slices = _source_canvas_slices(case.canvas_offset, case.source_shape)
            tensor = _canvas_image_tensor(image, source_shape=case.source_shape,
                canvas_offset=case.canvas_offset, patch_size=patch_size, device=device, channels_last=channels_last)
            with autocast_context(device, dtype):
                _, tokens, coords, context = model.encode_image_context_from_image(tensor)
            sheet_jobs, union_jobs = [], []
            selected_gt = np.isin(gt_ids, [c.component_id for c in case.components])
            for prompt_id in prompt_ids:
                ids = np.zeros(image.shape, np.uint16)
                best = np.full(image.shape, -np.inf, np.float32)
                hits = np.zeros(image.shape, np.uint16)
                prompts = [(c, next(p for p in c.prompt_sets if p.prompt_set_id == prompt_id)) for c in case.components]
                for start in range(0, len(prompts), prompt_batch_size):
                    chunk = prompts[start:start + prompt_batch_size]
                    points = torch.tensor([p.points_zyx for _, p in chunk], device=device, dtype=torch.float32)
                    labels = torch.tensor([p.labels for _, p in chunk], device=device, dtype=torch.long)
                    gt_canvas = torch.zeros((len(chunk), 1, *patch_size), device=device)
                    for i, (component, prompt) in enumerate(chunk):
                        native_points = np.asarray(prompt.points_zyx) - np.asarray(case.canvas_offset)
                        if any(label > 0 and gt_ids[tuple(point)] != component.component_id
                               for point, label in zip(native_points, prompt.labels, strict=True)):
                            raise ValueError(f"Stale positive prompt: {case.case_id}/{component.component_id}")
                        gt_canvas[(i, 0, *source_slices)] = torch.as_tensor(gt_ids == component.component_id, device=device)
                    with autocast_context(device, dtype):
                        predicted = model.forward_from_image_context(tokens, coords, context, points, labels,
                            image_shape=patch_size, image_index=torch.zeros(len(chunk), dtype=torch.long, device=device))["latent"]
                        raw = codec.raw_prediction(predicted)
                        probability = decoded_foreground_probability(ae.decode(raw)).float()[:, 0]
                        target_latent = ae.encode(gt_canvas)
                        ae_masks = decoded_foreground_probability(ae.decode(target_latent))[:, 0] >= threshold
                    target_model = codec.normalize(target_latent).float()
                    pred_model = codec.normalize(raw).float()
                    raw_mse = (raw.float() - target_latent.float()).square().flatten(1).mean(1).cpu().numpy()
                    normalized_mse = (pred_model - target_model).square().flatten(1).mean(1).cpu().numpy()
                    baseline_mse = target_model.square().flatten(1).mean(1).cpu().numpy()
                    for i, (component, prompt) in enumerate(chunk):
                        prob = np.asarray(probability[i][source_slices].cpu(), dtype=np.float32)
                        pred, gt = prob >= threshold, gt_ids == component.component_id
                        hits += pred
                        update_instances(ids, best, prob, pred, component.component_id)
                        tuple_dir = folder / "sheets" / f"sheet_{component.component_id:04d}" / prompt_id
                        meta = {"case_id": case.case_id, "sheet_id": component.component_id, "prompt_id": prompt_id,
                            "checkpoint_step": step, "threshold": threshold, "canvas_offset_zyx": list(case.canvas_offset),
                            "ignore_policy": ignore_policy, "conversion_border_width": width,
                            "scoring_box_zyx": annotated_box_bounds(ignore) if ignore_policy != "source" else None}
                        write_sheet_tuple(tuple_dir, assets=assets, gt=gt, prediction=pred,
                            points_zyx=np.asarray(prompt.points_zyx)-np.asarray(case.canvas_offset), labels=prompt.labels, metadata=meta)
                        metrics = {**meta, "tuple_dir": str(tuple_dir.relative_to(output)),
                            "latent_raw_mse": float(raw_mse[i]), "latent_normalized_mse": float(normalized_mse[i]),
                            "latent_relative_mse": float(normalized_mse[i] / max(float(baseline_mse[i]), 1e-12)),
                            "ae_reconstruction_dice": dice(np.asarray(ae_masks[i][source_slices].cpu()), gt, ~ignore)}
                        sheet_jobs.append((tuple_dir, metrics, pool.submit(score_masks, pred, gt, ignore, topology_backend=topology_backend,
                            topology_tile_batch_size=topology_tile_batch_size)))
                union = hits > 0
                union_folder = folder / "unions" / prompt_id
                link_assets(union_folder, assets)
                for name, volume in [("gt_union", gt_ids > 0), ("gt_prompted_union", selected_gt),
                        ("p2sd_union", union), ("gt_instances", gt_ids), ("p2sd_instances", ids)]:
                    save_nifti(union_folder / (name + ".nii.gz"), np.asarray(volume, np.uint16 if "instances" in name else np.uint8))
                union_future = pool.submit(score_masks, union, gt_ids > 0, ignore, topology_backend=topology_backend,
                    topology_tile_batch_size=topology_tile_batch_size)
                case_row = {"case_id": case.case_id, "prompt_id": prompt_id,
                    "ignore_policy": ignore_policy, "conversion_border_width": width,
                    "scoring_box_zyx": annotated_box_bounds(ignore) if ignore_policy != "source" else None,
                    "prompted_union_dice": dice(union, selected_gt, ~ignore),
                    "prompted_sheet_count": len(prompts), "all_gt_sheet_count": int(len(np.unique(gt_ids[gt_ids > 0])))}
                union_jobs.append((union_folder, case_row, union_future))
            pending_cases.add(dict(folder=folder, sheet_jobs=sheet_jobs, union_jobs=union_jobs,
                                   futures=[f for _, _, f in sheet_jobs + union_jobs]))
        pending_cases.finish()
    summary = write_reports(output, all_sheets, all_cases, {**protocol, "status": "complete", "elapsed_s": time.monotonic()-started})
    write_json(run / "latest_sheet_evaluation.json", {"checkpoint_step": step,
        "output_dir": str(output.resolve()), "sheet_prompt_count": len(all_sheets),
        "case_variant_count": len(all_cases)})
    return summary


def run_report_subprocess(*, checkpoint_path, source_run_dir, scene_manifest_path, output_dir, **settings):
    output = Path(output_dir)
    output.parent.mkdir(parents=True, exist_ok=True)
    log = output.parent / (output.name + ".log")
    command = [sys.executable, "-m", __name__, "--checkpoint_path", str(checkpoint_path),
        "--source_run_dir", str(source_run_dir), "--scene_manifest_path", str(scene_manifest_path),
        "--output_dir", str(output)]
    for key, value in settings.items():
        command.extend(["--" + key, str(value)])
    with log.open("w") as stream:
        subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=True, env=os.environ.copy())
    return json.loads((output / "summary.json").read_text())


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ["checkpoint_path", "source_run_dir", "scene_manifest_path", "output_dir"]:
        ap.add_argument("--" + name, required=True)
    ap.add_argument("--prompt_ids", default="p00")
    ap.add_argument("--prompt_batch_size", type=int, default=4)
    ap.add_argument("--cpu_workers", type=int, default=4)
    ap.add_argument("--max_in_flight_cases", type=int, default=3)
    ap.add_argument("--topology_tile_batch_size", type=int, default=1)
    ap.add_argument("--topology_backend", default="compact_exact")
    ap.add_argument("--max_cases", type=int)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--ignore_policy", choices=["border_box", "border_only", "source"], default="border_box")
    args = vars(ap.parse_args()); args["prompt_ids"] = tuple(args["prompt_ids"].split(","))
    evaluate(**args)


if __name__ == "__main__":
    main()
