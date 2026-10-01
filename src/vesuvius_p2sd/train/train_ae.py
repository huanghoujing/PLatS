"""Train the sheet autoencoder."""

from __future__ import annotations

import argparse
import json
import math
import time
import traceback
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from vesuvius_p2sd.data.distance_targets import prepare_ae_distance_targets
from vesuvius_p2sd.eval.metric_dispatch import MetricDispatchSettings, OrderedMetricDispatcher
from vesuvius_p2sd.eval.metrics import compute_p2sd_metrics, config_from_mapping
from vesuvius_p2sd.models.ae import build_sheet_ae
from vesuvius_p2sd.research.run_status import update_heartbeat, write_json, write_resume_context
from vesuvius_p2sd.train.ae_corruption import corrupt_sheet_mask, resolve_ae_input_corruption
from vesuvius_p2sd.train.common import (
    amp_dtype,
    append_jsonl,
    autocast_context,
    cycle,
    configured_lrs,
    restore_configured_lrs,
    dice_loss_from_logits,
    get_device,
    make_loader,
    make_lr_schedule,
    make_optimizer,
    maybe_channels_last_3d,
    move_batch,
    prepare_run,
    print_model_report,
    save_checkpoint,
    seed_everything,
    should_stop,
    border_probability_loss_from_logits,
    MetricWindow,
    capture_rng_state,
    collect_nonfinite_gradient_details,
    copy_to_cpu,
    outside_dilation_loss_from_logits,
    resolve_log_interval_steps,
    restore_rng_state,
    weighted_bce_with_logits,
)
from vesuvius_p2sd.train.eval_reporting import EvalSampleReporter
from vesuvius_p2sd.train.visualization import (
    prune_visualization_groups,
    prepare_p2sd_visualization_state,
    resolve_visualization_milestone_epoch,
    save_p2sd_3d_artifacts,
    save_p2sd_diagnostic_png,
    save_p2sd_projection_png,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", required=True)
    parser.add_argument("--run_id", default=None)
    parser.add_argument("--dry_run", action="store_true")
    args, overrides = parser.parse_known_args(argv)

    cfg, run_dir = prepare_run(
        config_path=args.config_path,
        overrides=overrides,
        run_id=args.run_id,
        entrypoint="train_ae",
    )
    raw_max_steps = cfg.get("training", {}).get("max_steps")
    if args.dry_run or raw_max_steps == 0 or raw_max_steps == "0":
        write_json(run_dir / "state.json", {
            "status": "dry_run_complete",
            "message": "Config validated; AE training skipped.",
        })
        update_heartbeat(run_dir, status="dry_run_complete")
        return 0

    seed_everything(int(cfg.get("seed", cfg.get("training", {}).get("seed", 42))))
    configure_training_backend(cfg.get("training", {}))
    device = get_device(cfg)
    model = build_sheet_ae(cfg).to(device)
    if bool(cfg.get("training", {}).get("channels_last_3d", False)):
        model = model.to(memory_format=torch.channels_last_3d)
    print_model_report(
        "ae",
        model,
        run_dir,
        print_to_stdout=bool(cfg.get("logging", {}).get("print_model", True)),
    )
    optimizer = make_optimizer(model, cfg)
    start_step, resumed_best = maybe_resume_ae(model, optimizer, cfg)
    train_loader = make_loader(cfg, "train", shuffle=True)
    val_loader = make_loader(cfg, "val", shuffle=False)
    train_iter = cycle(train_loader)
    dtype = amp_dtype(cfg.get("training", {}).get("amp_dtype"))
    steps_per_epoch = max(1, len(train_loader))
    max_steps, target_epochs = resolve_max_steps(cfg.get("training", {}), steps_per_epoch)
    lr_schedule = make_lr_schedule(optimizer, cfg, max_steps)
    print(f"[lr_schedule] {lr_schedule.describe()}", flush=True)
    log_interval = resolve_log_interval_steps(cfg.get("training", {}), steps_per_epoch)
    eval_interval = resolve_interval_steps(cfg.get("training", {}), "eval", steps_per_epoch, max_steps)
    save_interval = resolve_interval_steps(cfg.get("training", {}), "save", steps_per_epoch, max_steps)
    recon = cfg.get("loss", {})
    bce_weight = float(recon.get("bce_weight", 1.0))
    dice_weight = float(recon.get("dice_weight", 1.0))
    pos_weight = recon.get("pos_weight", "auto")
    max_auto_pos_weight = float(recon.get("max_auto_pos_weight", 32.0))
    outside_weight = float(recon.get("outside_dilation_weight", 0.0))
    outside_tolerance = int(recon.get("outside_dilation_tolerance", 2))
    border_weight = float(recon.get("border_weight", 0.0))
    distance_weight = float(recon.get("distance_weight", 0.0))
    query_distance_weight = float(recon.get("query_distance_weight", 0.0))
    distance_beta = float(recon.get("distance_smooth_l1_beta", 0.1))
    deep_supervision_weights = resolve_deep_supervision_weights(recon)
    repulsion_cfg = recon.get("same_volume_latent_repulsion", {}) or {}
    repulsion_weight = float(repulsion_cfg.get("weight", 0.0))
    repulsion_margin = float(repulsion_cfg.get("margin", 0.5))
    # P4 (task #10): up-weight repulsion on pairs in physical contact
    # (table from research/build_contact_table.py).
    repulsion_contact_weight = float(repulsion_cfg.get("contact_weight", 1.0))
    repulsion_contact_table = None
    if repulsion_cfg.get("contact_table"):
        repulsion_contact_table = json.load(open(str(repulsion_cfg["contact_table"])))
        print(f"[repulsion] contact table: {sum(len(v) for v in repulsion_contact_table.values())} "
              f"pairs across {sum(1 for v in repulsion_contact_table.values() if v)} cases, "
              f"contact_weight={repulsion_contact_weight}", flush=True)
    input_corruption = resolve_ae_input_corruption(cfg.get("data", {}))
    border_width = int(cfg.get("data", {}).get("erased_border_width", cfg.get("metrics", {}).get("erased_border_width", 0)))
    nonfinite_gradient_policy = str(cfg.get("training", {}).get("nonfinite_gradient_policy", "fail"))
    if nonfinite_gradient_policy not in {"fail", "skip"}:
        raise ValueError("training.nonfinite_gradient_policy must be fail|skip")
    nonfinite_gradient_max_skips = int(cfg.get("training", {}).get("nonfinite_gradient_max_skips", 0))
    if nonfinite_gradient_max_skips < 0:
        raise ValueError("training.nonfinite_gradient_max_skips must be non-negative")
    replay_nonfinite_gradient_skip = bool(
        cfg.get("training", {}).get("replay_nonfinite_gradient_skip", False)
    )
    nonfinite_gradient_skips = 0
    best = resumed_best or {"score": -math.inf, "step": 0}
    if resumed_best is not None:
        write_json(run_dir / "best.json", best)
    last_log_time = time.perf_counter()
    last_log_step = start_step
    epoch_start_time = time.perf_counter()
    epoch_start_step = start_step
    stopped_early = False
    last_completed_step = start_step
    log_window = MetricWindow()
    write_json(run_dir / "state.json", {
        "status": "running",
        "max_steps": max_steps,
        "epochs": target_epochs,
        "steps_per_epoch": steps_per_epoch,
        "start_step": start_step,
    })

    for step in range(start_step + 1, max_steps + 1):
        if should_stop(run_dir):
            stopped_early = True
            write_json(run_dir / "state.json", {
                "status": "stop_requested",
                "step": last_completed_step,
            })
            update_heartbeat(run_dir, status="stop_requested", step=last_completed_step)
            break
        model.train()
        batch = move_batch(next(train_iter), device)
        batch = prepare_ae_distance_targets(batch, cfg.get("data", {}))
        mask = maybe_channels_last_3d(batch["mask"].float(), bool(cfg.get("training", {}).get("channels_last_3d", False)))
        # Corruption happens BEFORE the RNG capture so a replay that restores
        # the captured state and reruns the forward on the saved corrupted
        # tensor consumes the same in-forward RNG draws (latent noise).
        model_input = corrupt_sheet_mask(mask, input_corruption) if input_corruption is not None else mask
        if step == start_step + 1:
            save_batch_nifti(run_dir, f"train_step{step:07d}",
                             {"input_degraded": model_input, "target_clean": mask}, batch)
        optimizer.zero_grad(set_to_none=True)
        forward_rng_state = capture_rng_state()
        with autocast_context(device, dtype):
            out = model(model_input, query_points=batch.get("query_points"))
        loss_terms = compute_ae_loss_terms(
            out,
            mask,
            batch,
            bce_weight=bce_weight,
            dice_weight=dice_weight,
            pos_weight=pos_weight,
            max_auto_pos_weight=max_auto_pos_weight,
            outside_weight=outside_weight,
            outside_tolerance=outside_tolerance,
            border_weight=border_weight,
            border_width=border_width,
            distance_weight=distance_weight,
            query_distance_weight=query_distance_weight,
            distance_beta=distance_beta,
            deep_supervision_weights=deep_supervision_weights,
            channels_last_3d=bool(cfg.get("training", {}).get("channels_last_3d", False)),
            point_interior_weight=float(recon.get("point_interior_weight", 1.0)),
            point_interior_band=float(recon.get("point_interior_band", 0.0)),
            same_volume_repulsion_weight=repulsion_weight,
            same_volume_repulsion_margin=repulsion_margin,
            same_volume_repulsion_contact_table=repulsion_contact_table,
            same_volume_repulsion_contact_weight=repulsion_contact_weight,
        )
        logits = out.get("logits")
        loss = loss_terms["loss"]
        bce = loss_terms["bce"]
        dloss = loss_terms["dice_loss"]
        outside_loss = loss_terms["outside_loss"]
        border_loss = loss_terms["border_loss"]
        deep_supervision_loss = loss_terms["deep_supervision_loss"]
        distance_loss = loss_terms["distance_loss"]
        query_distance_loss = loss_terms["query_distance_loss"]
        latent_repulsion_loss = loss_terms["latent_repulsion_loss"]
        if not bool(torch.isfinite(loss.detach()).all().item()):
            row = {
                "status": "failed",
                "reason": "nonfinite_loss",
                "step": step,
                "loss": float(loss.detach().float().cpu()),
                "bce": float(bce.detach().float().cpu()),
                "dice_loss": float(dloss.detach().float().cpu()),
                "outside_loss": float(outside_loss.detach().float().cpu()),
                "border_loss": float(border_loss.detach().float().cpu()),
                "deep_supervision_loss": float(deep_supervision_loss.detach().float().cpu()),
                "distance_loss": float(distance_loss.detach().float().cpu()),
                "query_distance_loss": float(query_distance_loss.detach().float().cpu()),
                "latent_repulsion_loss": float(latent_repulsion_loss.detach().float().cpu()),
            }
            append_jsonl(run_dir / "metrics.jsonl", row)
            write_json(run_dir / "state.json", row)
            save_ae_nonfinite_reproducer(
                run_dir=run_dir,
                model=model,
                optimizer=optimizer,
                step=step,
                cfg=cfg,
                batch=batch,
                model_mask=model_input,
                forward_rng_state=forward_rng_state,
                reason=row["reason"],
                loss_terms=loss_terms,
                gradient_details=None,
                target_mask=mask if input_corruption is not None else None,
            )
            update_heartbeat(run_dir, status="failed", step=step, loss=row["loss"])
            raise RuntimeError(f"AE loss became non-finite at step {step}: {row}")
        loss.backward()
        # Inspect before clip_grad_norm_: clipping a single NaN gradient would
        # multiply every otherwise-finite gradient by NaN and erase its source.
        nonfinite_gradients = collect_nonfinite_gradient_details(model)
        if nonfinite_gradients["parameter_count"]:
            can_skip_nonfinite_gradients = (
                nonfinite_gradient_policy == "skip"
                and nonfinite_gradient_skips < nonfinite_gradient_max_skips
            )
            row = {
                "status": "skipped_nonfinite_gradients" if can_skip_nonfinite_gradients else "failed",
                "reason": "nonfinite_gradients",
                "step": step,
                "loss": float(loss.detach().float().cpu()),
                "bce": float(bce.detach().float().cpu()),
                "dice_loss": float(dloss.detach().float().cpu()),
                "outside_loss": float(outside_loss.detach().float().cpu()),
                "border_loss": float(border_loss.detach().float().cpu()),
                "deep_supervision_loss": float(deep_supervision_loss.detach().float().cpu()),
                "distance_loss": float(distance_loss.detach().float().cpu()),
                "query_distance_loss": float(query_distance_loss.detach().float().cpu()),
                "latent_repulsion_loss": float(latent_repulsion_loss.detach().float().cpu()),
                "nonfinite_gradient_parameter_count": nonfinite_gradients["parameter_count"],
                "nonfinite_gradient_skips": nonfinite_gradient_skips + int(can_skip_nonfinite_gradients),
                "nonfinite_gradient_max_skips": nonfinite_gradient_max_skips,
            }
            append_jsonl(run_dir / "metrics.jsonl", row)
            write_json(run_dir / "nonfinite_gradients.json", {
                "step": step,
                "status": row["status"],
                "skips_used": row["nonfinite_gradient_skips"],
                "max_skips": nonfinite_gradient_max_skips,
                "loss_terms": {
                    "loss": row["loss"],
                    "bce": row["bce"],
                    "dice_loss": row["dice_loss"],
                    "outside_loss": row["outside_loss"],
                    "border_loss": row["border_loss"],
                    "deep_supervision_loss": row["deep_supervision_loss"],
                    "distance_loss": row["distance_loss"],
                    "query_distance_loss": row["query_distance_loss"],
                    "latent_repulsion_loss": row["latent_repulsion_loss"],
                },
                "gradients": nonfinite_gradients,
            })
            save_ae_nonfinite_reproducer(
                run_dir=run_dir,
                model=model,
                optimizer=optimizer,
                step=step,
                cfg=cfg,
                batch=batch,
                model_mask=model_input,
                forward_rng_state=forward_rng_state,
                reason=row["reason"],
                loss_terms=loss_terms,
                gradient_details=nonfinite_gradients,
                target_mask=mask if input_corruption is not None else None,
            )
            if not can_skip_nonfinite_gradients or replay_nonfinite_gradient_skip:
                replay_ae_nonfinite_backward(
                    model=model,
                    optimizer=optimizer,
                    mask=mask,
                    model_input=model_input,
                    batch=batch,
                    forward_rng_state=forward_rng_state,
                    dtype=dtype,
                    loss_kwargs={
                        "bce_weight": bce_weight,
                        "dice_weight": dice_weight,
                        "pos_weight": pos_weight,
                        "max_auto_pos_weight": max_auto_pos_weight,
                        "outside_weight": outside_weight,
                        "outside_tolerance": outside_tolerance,
                        "border_weight": border_weight,
                        "border_width": border_width,
                        "distance_weight": distance_weight,
                        "query_distance_weight": query_distance_weight,
                        "distance_beta": distance_beta,
                        "deep_supervision_weights": deep_supervision_weights,
                        "channels_last_3d": bool(cfg.get("training", {}).get("channels_last_3d", False)),
                        "point_interior_weight": float(recon.get("point_interior_weight", 1.0)),
                        "point_interior_band": float(recon.get("point_interior_band", 0.0)),
                        "same_volume_repulsion_weight": repulsion_weight,
                        "same_volume_repulsion_margin": repulsion_margin,
                    },
                    run_dir=run_dir,
                )
            if can_skip_nonfinite_gradients:
                nonfinite_gradient_skips += 1
                optimizer.zero_grad(set_to_none=True)
                last_completed_step = step
                update_heartbeat(
                    run_dir,
                    status="running",
                    step=step,
                    loss=row["loss"],
                    nonfinite_gradient_skips=nonfinite_gradient_skips,
                )
                if step % eval_interval == 0 or step == max_steps:
                    score = evaluate_ae(
                        model,
                        val_loader,
                        cfg,
                        run_dir,
                        step,
                        steps_per_epoch,
                        device,
                        dtype,
                    )
                    if score > best["score"]:
                        best = {"score": float(score), "step": step}
                        write_json(run_dir / "best.json", best)
                        save_checkpoint(
                            run_dir / "best.pt",
                            model=model,
                            optimizer=optimizer,
                            step=step,
                            cfg=cfg,
                            best=best,
                        )
                if step % save_interval == 0 or step == max_steps:
                    save_checkpoint(run_dir / "last.pt", model=model, optimizer=optimizer, step=step, cfg=cfg, best=best)
                    if (
                        bool(cfg.get("training", {}).get("save_epoch_checkpoints", False))
                        and step % steps_per_epoch == 0
                    ):
                        epoch_index = int(step // steps_per_epoch)
                        save_checkpoint(
                            run_dir / f"epoch_{epoch_index:04d}.pt",
                            model=model,
                            optimizer=optimizer,
                            step=step,
                            cfg=cfg,
                            best=best,
                        )
                continue
            write_json(run_dir / "state.json", row)
            update_heartbeat(run_dir, status="failed", step=step, loss=row["loss"])
            raise RuntimeError(
                f"AE gradients became non-finite at step {step}: {row}; "
                f"details={run_dir / 'nonfinite_gradients.json'}")
        grad_clip = cfg.get("training", {}).get("grad_clip")
        max_grad_norm = float(grad_clip) if grad_clip is not None else float("inf")
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        if not bool(torch.isfinite(grad_norm.detach()).all().item()):
            raise RuntimeError(
                "clip_grad_norm_ produced a non-finite norm after finite per-parameter gradients"
            )
        lr_schedule.apply(optimizer, step)
        optimizer.step()
        last_completed_step = step
        log_window.add({
            "loss": loss,
            "bce": bce,
            "dice_loss": dloss,
            "outside_loss": outside_loss,
            "border_loss": border_loss,
            "deep_supervision_loss": deep_supervision_loss,
            "distance_loss": distance_loss,
            "query_distance_loss": query_distance_loss,
            "latent_repulsion_loss": latent_repulsion_loss,
            "latent_repulsion_pairs": loss_terms["latent_repulsion_pairs"],
            "latent_repulsion_cosine": loss_terms["latent_repulsion_cosine"],
            "grad_norm": grad_norm,
        })

        if step % log_interval == 0:
            now = time.perf_counter()
            elapsed = max(now - last_log_time, 1e-9)
            step_delta = max(step - last_log_step, 1)
            step_rate = float(step_delta / elapsed)
            batch_size = int(mask.shape[0])
            # Rows are window MEANS over the steps since the previous row
            # (one row per epoch by default), not single-step point samples.
            window_means = log_window.means()
            log_window.reset()
            mean_terms = {
                key: torch.tensor(window_means.get(key, 0.0))
                for key in ("loss", "bce", "dice_loss", "outside_loss", "border_loss",
                            "deep_supervision_loss", "distance_loss", "query_distance_loss",
                            "latent_repulsion_loss")
            }
            row = {
                "step": step,
                "epoch": float(step / steps_per_epoch),
                "seconds_per_step": float(elapsed / step_delta),
                "steps_per_second": step_rate,
                "samples_per_second": float(step_rate * batch_size),
                "batch_size": batch_size,
                **window_means,
                "deep_supervision_levels": float(min(len(out.get("aux_logits", [])), len(deep_supervision_weights))),
                **loss_contribution_scalars(
                    mean_terms,
                    bce_weight=bce_weight,
                    dice_weight=dice_weight,
                    outside_weight=outside_weight,
                    border_weight=border_weight,
                    distance_weight=distance_weight,
                    query_distance_weight=query_distance_weight,
                    same_volume_repulsion_weight=repulsion_weight,
                ),
            }
            append_jsonl(run_dir / "metrics.jsonl", row)
            update_heartbeat(run_dir, status="running", step=step, loss=row["loss"])
            last_log_time = now
            last_log_step = step

        if step % steps_per_epoch == 0:
            now = time.perf_counter()
            epoch_steps = max(1, step - epoch_start_step)
            epoch_time_s = now - epoch_start_time
            append_jsonl(run_dir / "metrics.jsonl", {
                "split": "train_epoch",
                "step": step,
                "epoch": float(step / steps_per_epoch),
                "epoch_index": int(step // steps_per_epoch),
                "epoch_time_s": float(epoch_time_s),
                "seconds_per_step": float(epoch_time_s / epoch_steps),
                "samples_per_second": float(mask.shape[0] * epoch_steps / epoch_time_s),
            })
            update_heartbeat(
                run_dir,
                status="running",
                step=step,
                loss=float(loss.detach().cpu()),
                epoch_time_s=float(epoch_time_s),
            )
            epoch_start_time = now
            epoch_start_step = step

        if logits is not None and should_save_ae_train_3d(
                cfg.get("visualization", {}), step, steps_per_epoch):
            save_ae_train_3d(
                batch=batch,
                mask=mask,
                model_input=model_input,
                logits=logits,
                loss_values={
                    "loss": float(loss.detach().cpu()),
                    "bce": float(bce.detach().cpu()),
                    "dice_loss": float(dloss.detach().cpu()),
                },
                cfg=cfg,
                run_dir=run_dir,
                step=step,
            )

        if step % eval_interval == 0 or step == max_steps:
            score = evaluate_ae(
                model,
                val_loader,
                cfg,
                run_dir,
                step,
                steps_per_epoch,
                device,
                dtype,
            )
            if score > best["score"]:
                best = {"score": float(score), "step": step}
                write_json(run_dir / "best.json", best)
                save_checkpoint(run_dir / "best.pt", model=model, optimizer=optimizer, step=step, cfg=cfg, best=best)

        if step % save_interval == 0 or step == max_steps:
            save_checkpoint(run_dir / "last.pt", model=model, optimizer=optimizer, step=step, cfg=cfg, best=best)
            if (
                bool(cfg.get("training", {}).get("save_epoch_checkpoints", False))
                and step % steps_per_epoch == 0
            ):
                epoch_index = int(step // steps_per_epoch)
                save_checkpoint(
                    run_dir / f"epoch_{epoch_index:04d}.pt",
                    model=model,
                    optimizer=optimizer,
                    step=step,
                    cfg=cfg,
                    best=best,
                )

    final_status = "stop_requested" if stopped_early else "complete"
    write_json(run_dir / "state.json", {
        "status": final_status,
        "step": last_completed_step,
        "max_steps": max_steps,
        "epochs": target_epochs,
        "steps_per_epoch": steps_per_epoch,
        "best": best,
    })
    update_heartbeat(run_dir, status=final_status, step=last_completed_step)
    write_resume_context(run_dir)
    return 0


def configure_training_backend(training_cfg: dict) -> None:
    torch.backends.cudnn.enabled = bool(training_cfg.get("cudnn_enabled", True))
    cudnn_deterministic = bool(training_cfg.get("cudnn_deterministic", False))
    torch.backends.cudnn.deterministic = cudnn_deterministic
    torch.backends.cudnn.benchmark = bool(training_cfg.get("cudnn_benchmark", True)) and not cudnn_deterministic
    matmul_precision = training_cfg.get("matmul_precision", "high")
    if matmul_precision:
        torch.set_float32_matmul_precision(str(matmul_precision))


def maybe_resume_ae(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    cfg: dict,
) -> tuple[int, dict | None]:
    training_cfg = cfg.get("training", {})
    resume_path = training_cfg.get("resume_checkpoint_path") or training_cfg.get("resume_from")
    if not resume_path:
        return 0, None
    checkpoint = torch.load(resume_path, map_location="cpu")
    state = checkpoint.get("model", checkpoint.get("model_state_dict"))
    if state is None:
        raise ValueError(f"Checkpoint has no model state: {resume_path}")
    model.load_state_dict(state, strict=True)
    if bool(training_cfg.get("resume_optimizer", True)) and "optimizer" in checkpoint:
        base_lrs = configured_lrs(optimizer)
        optimizer.load_state_dict(checkpoint["optimizer"])
        checkpoint_lrs = configured_lrs(optimizer)
        restore_configured_lrs(optimizer, base_lrs)
        if checkpoint_lrs != base_lrs:
            print(
                f"[resume] optimizer state restored; learning rate taken from config "
                f"{base_lrs} (checkpoint held {checkpoint_lrs})",
                flush=True,
            )
    start_step = int(checkpoint.get("step", 0))
    best = checkpoint.get("best")
    if best is None:
        best = {"score": -math.inf, "step": 0}
    if bool(training_cfg.get("reset_best_on_resume", False)):
        best = {"score": -math.inf, "step": 0, "source_resume_step": start_step}
    if bool(training_cfg.get("reset_step_on_resume", False)):
        start_step = 0
    print(f"[resume] loaded {resume_path} at step {checkpoint.get('step', 0)}; starting at step {start_step}", flush=True)
    return start_step, best


def resolve_max_steps(training_cfg: dict, steps_per_epoch: int) -> tuple[int, float]:
    raw = training_cfg.get("max_steps")
    if raw is not None and int(raw) > 0:
        max_steps = int(raw)
        return max_steps, float(max_steps / max(1, steps_per_epoch))
    epochs = float(training_cfg.get("epochs", 1))
    max_steps = max(1, int(math.ceil(epochs * max(1, steps_per_epoch))))
    return max_steps, epochs


def resolve_interval_steps(
    training_cfg: dict,
    name: str,
    steps_per_epoch: int,
    max_steps: int,
) -> int:
    epoch_key = f"{name}_interval_epochs"
    step_key = f"{name}_interval_steps"
    if epoch_key in training_cfg:
        return max(1, int(round(float(training_cfg[epoch_key]) * max(1, steps_per_epoch))))
    return int(training_cfg.get(step_key, max(1, max_steps)))


def resolve_deep_supervision_weights(loss_cfg: dict) -> list[float]:
    if "deep_supervision_weights" in loss_cfg:
        return [float(v) for v in loss_cfg.get("deep_supervision_weights", [])]
    deep_cfg = loss_cfg.get("deep_supervision", {})
    return [float(v) for v in deep_cfg.get("weights", [])]


def should_save_ae_train_3d(
    visualization_cfg: dict,
    step: int,
    steps_per_epoch: int,
) -> bool:
    if not bool(visualization_cfg.get("save_train_3d", False)):
        return False
    interval_epochs = visualization_cfg.get("train_3d_interval_epochs")
    if interval_epochs is not None:
        interval = max(1, int(round(float(interval_epochs) * max(1, steps_per_epoch))))
        return step % interval == 0
    interval = int(visualization_cfg.get("train_3d_interval_steps", 0) or 0)
    return interval > 0 and step % interval == 0


@torch.no_grad()
def save_ae_train_3d(
    *,
    batch: dict,
    mask: torch.Tensor,
    logits: torch.Tensor,
    loss_values: dict[str, float],
    cfg: dict,
    run_dir: Path,
    step: int,
    model_input: torch.Tensor | None = None,
) -> None:
    """Save bounded NIFTI train artifacts from the actual optimization batch."""

    viz_cfg = cfg.get("visualization", {})
    metrics_cfg = cfg.get("metrics", {})
    max_items = min(int(mask.shape[0]), max(1, int(viz_cfg.get("train_3d_max_items", 1))))
    # "image" is what the model actually saw: under input corruption that is
    # the holed/eroded mask, so the artifact shows the denoising behaviour.
    input_volume = mask if model_input is None else model_input
    for sample_index in range(max_items):
        component_id = int(batch["component_id"][sample_index].detach().item())
        crop_start = tuple(int(value) for value in batch["crop_start"][sample_index])
        payload = {
            "image": input_volume[sample_index, 0].detach().float().cpu().numpy(),
            "gt": mask[sample_index, 0].detach().float().cpu().numpy(),
            "pred": torch.sigmoid(logits[sample_index, 0]).detach().float().cpu().numpy(),
            "metrics": loss_values,
        }
        _save_ae_3d_visualization(
            run_dir / "viz3d" / "train" / f"ae_step_{step:06d}_item_{sample_index:02d}",
            payload=payload,
            metrics_cfg=metrics_cfg,
            meta={
                "step": step,
                "task": "ae_train",
                "conditioned": False,
                "case_id": str(batch["case_id"][sample_index]),
                "component_id": component_id,
                "crop_start": crop_start,
                "target_voxels": int(mask[sample_index, 0].sum().detach().item()),
            },
            viz_cfg=viz_cfg,
        )
    prune_visualization_groups(
        run_dir / "viz3d" / "train",
        keep_latest=resolve_visualization_keep_latest(viz_cfg, "keep_latest_train_3d"),
        stem_prefixes=("ae_step_",),
    )


def deep_supervision_loss_from_logits(
    aux_logits: list[torch.Tensor] | tuple[torch.Tensor, ...],
    target: torch.Tensor,
    *,
    weights: list[float],
    bce_weight: float,
    dice_weight: float,
    pos_weight: float | str | None,
    max_auto_pos_weight: float,
) -> torch.Tensor:
    if not aux_logits or not weights:
        return target.new_zeros(())
    if len(weights) > len(aux_logits):
        raise ValueError(
            "loss.deep_supervision weights has more entries than model aux logits: "
            f"{len(weights)} weights for {len(aux_logits)} aux outputs"
        )
    total = target.new_zeros(())
    for logits, weight in zip(aux_logits, weights):
        if not weight:
            continue
        target_i = match_target_to_logits(target, logits)
        bce = weighted_bce_with_logits(
            logits,
            target_i,
            pos_weight=pos_weight,
            max_auto_pos_weight=max_auto_pos_weight,
        )
        dloss = dice_loss_from_logits(logits, target_i)
        total = total + float(weight) * (float(bce_weight) * bce + float(dice_weight) * dloss)
    return total


def save_batch_nifti(
    run_dir: Path,
    tag: str,
    tensors: dict[str, torch.Tensor],
    batch: dict,
    *,
    max_samples: int = 4,
) -> None:
    """Dump batch volumes as NIFTI under <run_dir>/batch_nifti/<tag>/.

    Must-have batch visibility (user-directed 2026-08-14): the AE trains on
    CORRUPTED inputs (denoising), and the only way to sanity-check what the
    degradation actually looks like is to open the real batch tensors in a
    viewer. Identity affine, (z, y, x) axis order -- the same convention as
    instances_to_case_nifti. Filenames carry case/component ids.
    """
    import nibabel as nib

    out = run_dir / "batch_nifti" / tag
    out.mkdir(parents=True, exist_ok=True)
    count = min(max_samples, next(iter(tensors.values())).shape[0])
    case_ids = batch.get("case_id", ["?"] * count)
    component_ids = batch.get("component_id")
    for i in range(count):
        case = str(case_ids[i])
        comp = int(component_ids[i]) if component_ids is not None else -1
        for name, tensor in tensors.items():
            arr = tensor[i, 0].detach().float().cpu().numpy().astype(np.float32)
            nib.save(nib.Nifti1Image(arr, np.eye(4, dtype=np.float32)),
                     str(out / f"s{i}_{case}_c{comp}_{name}.nii.gz"))
    print(f"[batch_nifti] {tag}: {count} samples x {len(tensors)} volumes -> {out}", flush=True)


def same_volume_latent_repulsion_loss(
    latent: torch.Tensor,
    case_ids: list,
    component_ids: torch.Tensor,
    *,
    margin: float,
    contact_table: dict | None = None,
    contact_weight: float = 1.0,
) -> tuple[torch.Tensor, int, torch.Tensor]:
    """Push latents of DIFFERENT sheets from the SAME volume apart.

    The AE counterpart of the P2SD ``different_sheet_prompt_contrast_loss``:
    downstream clustering separates click candidates by latent fingerprint, so
    latents of confusable neighbouring wraps must not collapse together.
    Hinged cosine similarity ``relu(cos - margin)`` over same-case
    different-component pairs: cosine is scale-free, so unlike a squared
    distance hinge it cannot be satisfied by inflating latent norms (the
    latent is the P2SD regression target and its scale is normalized away by
    latent stats anyway). Pairs enter batches via
    ``data.same_case_pair_batches``; a batch without pairs (validation, or a
    duplicate component draw) contributes exactly zero.

    Returns ``(loss, pair_count, mean_pair_cosine)``.
    """
    if latent.shape[0] < 2:
        return latent.new_zeros(()), 0, latent.new_zeros(())
    case_index = {}
    case_codes = torch.tensor(
        [case_index.setdefault(str(case_id), len(case_index)) for case_id in case_ids],
        device=latent.device,
    )
    same_case = case_codes.view(-1, 1) == case_codes.view(1, -1)
    different_component = component_ids.view(-1, 1) != component_ids.view(1, -1)
    pair_mask = torch.triu(same_case & different_component, diagonal=1)
    left, right = pair_mask.nonzero(as_tuple=True)
    if left.numel() == 0:
        return latent.new_zeros(()), 0, latent.new_zeros(())
    z = F.normalize(latent.float().flatten(1), dim=1)
    cosine = (z[left] * z[right]).sum(dim=1)
    hinge = F.relu(cosine - float(margin))
    if contact_table is not None and contact_weight != 1.0:
        # P4 (task #10): up-weight pairs in physical contact -- the exact
        # pairs whose latent collapse causes decode/cluster merges. Weighted
        # MEAN keeps the loss scale comparable to the unweighted champion.
        weights = []
        for l_index, r_index in zip(left.tolist(), right.tolist()):
            pair = sorted((int(component_ids[l_index]), int(component_ids[r_index])))
            in_contact = pair in contact_table.get(str(case_ids[l_index]), [])
            weights.append(float(contact_weight) if in_contact else 1.0)
        weight_tensor = torch.tensor(weights, device=hinge.device, dtype=hinge.dtype)
        loss = (hinge * weight_tensor).sum() / weight_tensor.sum()
    else:
        loss = hinge.mean()
    return loss, int(left.numel()), cosine.detach().mean()


def compute_ae_loss_terms(
    out: dict[str, torch.Tensor],
    mask: torch.Tensor,
    batch: dict,
    *,
    bce_weight: float,
    dice_weight: float,
    pos_weight: float | str | None,
    max_auto_pos_weight: float,
    outside_weight: float,
    outside_tolerance: int,
    border_weight: float,
    border_width: int,
    distance_weight: float,
    query_distance_weight: float,
    distance_beta: float,
    deep_supervision_weights: list[float],
    channels_last_3d: bool,
    point_interior_weight: float = 1.0,
    point_interior_band: float = 0.0,
    same_volume_repulsion_weight: float = 0.0,
    same_volume_repulsion_margin: float = 0.5,
    same_volume_repulsion_contact_table: dict | None = None,
    same_volume_repulsion_contact_weight: float = 1.0,
) -> dict[str, torch.Tensor]:
    """Compute AE supervision outside the BF16 model-autocast region."""

    if "logits" not in out:
        return compute_point_ae_loss_terms(
            out,
            mask,
            batch,
            bce_weight=bce_weight,
            dice_weight=dice_weight,
            pos_weight=pos_weight,
            max_auto_pos_weight=max_auto_pos_weight,
            distance_weight=distance_weight,
            query_distance_weight=query_distance_weight,
            distance_beta=distance_beta,
            point_interior_weight=point_interior_weight,
            point_interior_band=point_interior_band,
        )
    logits = out["logits"]
    bce = weighted_bce_with_logits(
        logits,
        mask,
        pos_weight=pos_weight,
        max_auto_pos_weight=max_auto_pos_weight,
    )
    dloss = dice_loss_from_logits(logits, mask)
    outside_loss = outside_dilation_loss_from_logits(
        logits,
        mask,
        tolerance=outside_tolerance,
    )
    border_loss = border_probability_loss_from_logits(
        logits,
        border_width=border_width,
        target=mask,
    )
    deep_supervision_loss = deep_supervision_loss_from_logits(
        out.get("aux_logits", []),
        mask,
        weights=deep_supervision_weights,
        bce_weight=bce_weight,
        dice_weight=dice_weight,
        pos_weight=pos_weight,
        max_auto_pos_weight=max_auto_pos_weight,
    )
    distance_loss = mask.new_zeros(())
    if distance_weight > 0:
        if "distance" not in out:
            raise RuntimeError("loss.distance_weight > 0 but AE distance head is disabled")
        if "distance_field" not in batch:
            raise RuntimeError("loss.distance_weight > 0 but data.distance_field.enabled is false")
        distance_target = maybe_channels_last_3d(
            batch["distance_field"].float(),
            channels_last_3d,
        )
        distance_loss = F.smooth_l1_loss(
            out["distance"].float(),
            distance_target.float(),
            beta=distance_beta,
        )
    query_distance_loss = mask.new_zeros(())
    if query_distance_weight > 0:
        if "query_distance" not in out:
            raise RuntimeError(
                "loss.query_distance_weight > 0 but AE coordinate query head or data query points are disabled"
            )
        if "query_distances" not in batch:
            raise RuntimeError("loss.query_distance_weight > 0 but data.query_distance.enabled is false")
        query_distance_loss = F.smooth_l1_loss(
            out["query_distance"].float(),
            batch["query_distances"].float(),
            beta=distance_beta,
        )
    latent_repulsion_loss = mask.new_zeros(())
    latent_repulsion_pairs = mask.new_zeros(())
    latent_repulsion_cosine = mask.new_zeros(())
    if same_volume_repulsion_weight > 0:
        if "latent" not in out:
            raise RuntimeError("loss.same_volume_latent_repulsion.weight > 0 but model output has no latent")
        latent_repulsion_loss, pair_count, latent_repulsion_cosine = same_volume_latent_repulsion_loss(
            out["latent"],
            batch["case_id"],
            batch["component_id"],
            margin=same_volume_repulsion_margin,
            contact_table=same_volume_repulsion_contact_table,
            contact_weight=same_volume_repulsion_contact_weight,
        )
        latent_repulsion_pairs = mask.new_tensor(float(pair_count))
    loss = (
        bce_weight * bce
        + dice_weight * dloss
        + outside_weight * outside_loss
        + border_weight * border_loss
        + distance_weight * distance_loss
        + query_distance_weight * query_distance_loss
        + deep_supervision_loss
        + same_volume_repulsion_weight * latent_repulsion_loss
    )
    return {
        "loss": loss,
        "bce": bce,
        "dice_loss": dloss,
        "outside_loss": outside_loss,
        "border_loss": border_loss,
        "deep_supervision_loss": deep_supervision_loss,
        "distance_loss": distance_loss,
        "query_distance_loss": query_distance_loss,
        "latent_repulsion_loss": latent_repulsion_loss,
        "latent_repulsion_pairs": latent_repulsion_pairs,
        "latent_repulsion_cosine": latent_repulsion_cosine,
    }


def point_occupancy_targets(mask: torch.Tensor, points_zyx: torch.Tensor) -> torch.Tensor:
    """Gather binary occupancy labels at (possibly fractional) voxel points."""
    volume = mask[:, 0]
    indices = points_zyx.round().long()
    for axis in range(3):
        indices[..., axis].clamp_(0, volume.shape[axis + 1] - 1)
    batch_index = torch.arange(volume.shape[0], device=mask.device)[:, None]
    return volume[batch_index, indices[..., 0], indices[..., 1], indices[..., 2]]


def compute_point_ae_loss_terms(
    out: dict[str, torch.Tensor],
    mask: torch.Tensor,
    batch: dict,
    *,
    bce_weight: float,
    dice_weight: float,
    pos_weight: float | str | None,
    max_auto_pos_weight: float,
    distance_weight: float,
    query_distance_weight: float,
    distance_beta: float,
    point_interior_weight: float = 1.0,
    point_interior_band: float = 0.0,
) -> dict[str, torch.Tensor]:
    """Point-decoder supervision: occupancy BCE/dice and distance at samples.

    Dense-map terms (outside dilation, border, dense distance field, deep
    supervision) have no dense prediction to act on and stay zero; geometry
    is carried by the query-distance term at the same sampled points.
    """
    if "point_logits" not in out:
        raise RuntimeError(
            "point-decoder AE requires data.query_distance.enabled sampled points")
    if distance_weight > 0:
        raise RuntimeError(
            "loss.distance_weight must be 0 for decoder_type=point; "
            "use loss.query_distance_weight for geometry supervision")
    point_logits = out["point_logits"]
    if "query_occupancy" in batch:
        # Sub-voxel jittered queries: labels are the sign of the interpolated
        # signed EDT, consistent with the continuous positions.
        targets = batch["query_occupancy"].float()
    else:
        targets = point_occupancy_targets(mask, batch["query_points"])
    bce = weighted_bce_with_logits(
        point_logits,
        targets,
        pos_weight=pos_weight,
        max_auto_pos_weight=max_auto_pos_weight,
    )
    dloss = dice_loss_from_logits(point_logits, targets)
    zero = mask.new_zeros(())
    query_distance_loss = zero
    if query_distance_weight > 0:
        if "point_signed_distance" in out:
            # Signed-distance head: sign comes from occupancy at the same
            # points, so boundary sharpness is carried by the geometry channel.
            if "query_signed_distances" in batch:
                signed_target = batch["query_signed_distances"].float()
            else:
                signed_target = batch["query_distances"].float() * (1.0 - 2.0 * targets)
            if point_interior_weight > 1.0 and point_interior_band > 0.0:
                # Pinholes come from sign flips where |signed target| is small
                # (thin-sheet interiors and the immediate outside): up-weight
                # that band so regression precision concentrates at the
                # zero crossing instead of the easy far field.
                per_point = F.smooth_l1_loss(
                    out["point_signed_distance"].float(),
                    signed_target,
                    beta=distance_beta,
                    reduction="none",
                )
                weights = 1.0 + (float(point_interior_weight) - 1.0) * (
                    signed_target.abs() < float(point_interior_band)).float()
                query_distance_loss = (per_point * weights).sum() / weights.sum()
            else:
                query_distance_loss = F.smooth_l1_loss(
                    out["point_signed_distance"].float(),
                    signed_target,
                    beta=distance_beta,
                )
        else:
            query_distance_loss = F.smooth_l1_loss(
                out["query_distance"].float(),
                batch["query_distances"].float(),
                beta=distance_beta,
            )
    loss = (
        bce_weight * bce
        + dice_weight * dloss
        + query_distance_weight * query_distance_loss
    )
    return {
        "loss": loss,
        "bce": bce,
        "dice_loss": dloss,
        "outside_loss": zero,
        "border_loss": zero,
        "deep_supervision_loss": zero,
        "distance_loss": zero,
        "query_distance_loss": query_distance_loss,
        "latent_repulsion_loss": zero,
        "latent_repulsion_pairs": zero,
        "latent_repulsion_cosine": zero,
    }


def loss_contribution_scalars(
    loss_terms: dict[str, torch.Tensor],
    *,
    bce_weight: float,
    dice_weight: float,
    outside_weight: float,
    border_weight: float,
    distance_weight: float,
    query_distance_weight: float,
    same_volume_repulsion_weight: float = 0.0,
) -> dict[str, float]:
    """Expose raw task losses and their exact contribution to total loss."""
    def scalar(name: str, weight: float) -> float:
        return float((float(weight) * loss_terms[name]).detach().float().cpu())

    occupancy = scalar("bce", bce_weight) + scalar("dice_loss", dice_weight)
    scalars = {
        "loss_occupancy_weighted": occupancy,
        "loss_bce_weighted": scalar("bce", bce_weight),
        "loss_dice_weighted": scalar("dice_loss", dice_weight),
        "loss_outside_weighted": scalar("outside_loss", outside_weight),
        "loss_border_weighted": scalar("border_loss", border_weight),
        "loss_deep_supervision_weighted": float(loss_terms["deep_supervision_loss"].detach().float().cpu()),
        "loss_distance_weighted": scalar("distance_loss", distance_weight),
        "loss_query_distance_weighted": scalar("query_distance_loss", query_distance_weight),
    }
    if "latent_repulsion_loss" in loss_terms:
        scalars["loss_latent_repulsion_weighted"] = scalar(
            "latent_repulsion_loss", same_volume_repulsion_weight
        )
    return scalars


def save_ae_nonfinite_reproducer(
    *,
    run_dir: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    cfg: dict,
    batch: dict,
    model_mask: torch.Tensor,
    forward_rng_state: dict,
    reason: str,
    loss_terms: dict[str, torch.Tensor],
    gradient_details: dict | None,
    target_mask: torch.Tensor | None = None,
) -> None:
    """Persist the precise pre-update state needed to replay one AE failure."""

    required_batch_keys = (
        "distance_field",
        "query_points",
        "query_distances",
        "case_id",
        "component_id",
        "crop_start",
        "prompt_zyx",
    )
    try:
        payload = {
            "format_version": 1,
            "reason": reason,
            "step": int(step),
            "config": cfg,
            "backend": {
                "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
                "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
            },
            "rng_state_before_forward": copy_to_cpu(forward_rng_state),
            "model": copy_to_cpu(model.state_dict()),
            "optimizer": copy_to_cpu(optimizer.state_dict()),
            # "mask" is the tensor the model actually consumed (the corrupted
            # input when input corruption is active); "target_mask" is the
            # clean supervision target when the two differ.
            "batch": copy_to_cpu({
                "mask": model_mask,
                **({"target_mask": target_mask} if target_mask is not None else {}),
                **{key: batch[key] for key in required_batch_keys if key in batch},
            }),
            "loss_terms": {
                key: float(value.detach().float().cpu())
                for key, value in loss_terms.items()
            },
            "gradient_details": gradient_details,
        }
        torch.save(payload, run_dir / "nonfinite_reproducer.pt")
    except Exception:
        (run_dir / "nonfinite_reproducer_error.txt").write_text(
            traceback.format_exc(),
            encoding="utf-8",
        )


def replay_ae_nonfinite_backward(
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    mask: torch.Tensor,
    batch: dict,
    forward_rng_state: dict,
    dtype: torch.dtype | None,
    loss_kwargs: dict,
    run_dir: Path,
    model_input: torch.Tensor | None = None,
) -> None:
    """Replay the failed batch with anomaly detection after gradients are saved.

    ``model_input`` is the tensor the failing forward consumed (the corrupted
    mask under input corruption); ``mask`` stays the clean loss target.
    """

    result: dict[str, object]
    forward_input = mask if model_input is None else model_input
    optimizer.zero_grad(set_to_none=True)
    restore_rng_state(forward_rng_state)
    try:
        with torch.autograd.detect_anomaly(check_nan=True):
            with autocast_context(mask.device, dtype):
                out = model(forward_input, query_points=batch.get("query_points"))
            replay_loss_terms = compute_ae_loss_terms(out, mask, batch, **loss_kwargs)
            replay_loss_terms["loss"].backward()
        replay_gradients = collect_nonfinite_gradient_details(model)
        result = {
            "status": (
                "nonfinite_gradients_reproduced"
                if replay_gradients["parameter_count"]
                else "clean_replay"
            ),
            "gradient_details": replay_gradients,
        }
    except Exception:
        result = {
            "status": "anomaly_exception",
            "traceback": traceback.format_exc(),
        }
    finally:
        optimizer.zero_grad(set_to_none=True)
    write_json(run_dir / "nonfinite_anomaly_replay.json", result)


def match_target_to_logits(target: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
    target_shape = tuple(int(v) for v in target.shape[-3:])
    logit_shape = tuple(int(v) for v in logits.shape[-3:])
    if target_shape == logit_shape:
        return target
    if all(t >= l for t, l in zip(target_shape, logit_shape)):
        if all(t % l == 0 for t, l in zip(target_shape, logit_shape)):
            kernel = tuple(max(1, t // l) for t, l in zip(target_shape, logit_shape))
            return F.max_pool3d(target.float(), kernel_size=kernel, stride=kernel)
        return F.adaptive_max_pool3d(target.float(), output_size=logit_shape)
    return F.interpolate(target.float(), size=logit_shape, mode="nearest")


@torch.no_grad()
def evaluate_ae(
    model,
    val_loader,
    cfg,
    run_dir: Path,
    step: int,
    steps_per_epoch: int,
    device,
    dtype,
) -> float:
    model.eval()
    metrics_cfg = cfg.get("metrics", {})
    metric_cfg = config_from_mapping(metrics_cfg)
    eval_batches = int(metrics_cfg.get("eval_batches", 1))
    val_min_target_voxels = int(metrics_cfg.get("val_min_target_voxels", 0) or 0)
    loss_cfg = cfg.get("loss", {})
    loss_kwargs = {
        "bce_weight": float(loss_cfg.get("bce_weight", 1.0)),
        "dice_weight": float(loss_cfg.get("dice_weight", 1.0)),
        "pos_weight": loss_cfg.get("pos_weight", "auto"),
        "max_auto_pos_weight": float(loss_cfg.get("max_auto_pos_weight", 32.0)),
        "outside_weight": float(loss_cfg.get("outside_dilation_weight", 0.0)),
        "outside_tolerance": int(loss_cfg.get("outside_dilation_tolerance", 2)),
        "border_weight": float(loss_cfg.get("border_weight", 0.0)),
        "border_width": int(cfg.get("data", {}).get("erased_border_width", 0)),
        "distance_weight": float(loss_cfg.get("distance_weight", 0.0)),
        "query_distance_weight": float(loss_cfg.get("query_distance_weight", 0.0)),
        "distance_beta": float(loss_cfg.get("distance_smooth_l1_beta", 0.1)),
        "deep_supervision_weights": resolve_deep_supervision_weights(loss_cfg),
        "channels_last_3d": bool(cfg.get("training", {}).get("channels_last_3d", False)),
        "point_interior_weight": float(loss_cfg.get("point_interior_weight", 1.0)),
        "point_interior_band": float(loss_cfg.get("point_interior_band", 0.0)),
        "same_volume_repulsion_weight": float(
            (loss_cfg.get("same_volume_latent_repulsion", {}) or {}).get("weight", 0.0)
        ),
        "same_volume_repulsion_margin": float(
            (loss_cfg.get("same_volume_latent_repulsion", {}) or {}).get("margin", 0.5)
        ),
    }
    metric_rows = []
    task_loss_rows = []
    first_payload = None
    first_payload_submitted = False
    eval_seen_samples = 0
    eval_excluded_small_target_samples = 0
    dispatch_settings = MetricDispatchSettings.from_metrics_config(metrics_cfg)
    reporter = EvalSampleReporter(
        run_dir=run_dir,
        step=step,
        task="ae",
        metrics_cfg=metrics_cfg,
    )

    def consume_metrics(payload: dict, metrics: dict[str, float]) -> None:
        nonlocal first_payload
        metric_rows.append(metrics)
        reporter.add(
            identity=payload["identity"],
            metrics=metrics,
            volumes=payload.get("volumes"),
        )
        if payload["first_payload"] is not None:
            first_payload = {**payload["first_payload"], "metrics": metrics}

    with OrderedMetricDispatcher[dict, dict[str, float]](dispatch_settings) as metric_dispatcher:
        for batch_index, batch in enumerate(val_loader):
            if batch_index >= eval_batches:
                break
            if should_stop(run_dir):
                update_heartbeat(run_dir, status="stop_requested", step=step)
                return -math.inf
            batch = move_batch(batch, device)
            batch = prepare_ae_distance_targets(batch, cfg.get("data", {}))
            mask = maybe_channels_last_3d(
                batch["mask"].float(),
                bool(cfg.get("training", {}).get("channels_last_3d", False)),
            )
            if batch_index == 0 and not (run_dir / "batch_nifti" / "val").exists():
                save_batch_nifti(run_dir, "val", {"input_clean": mask}, batch)
            with autocast_context(device, dtype):
                out = model(mask, query_points=batch.get("query_points"))
                if "logits" not in out:
                    # Point-decoder AE: materialize dense logits for metrics
                    # and visualization via chunked queries, gradient-free.
                    with torch.no_grad():
                        out["logits"] = model.decode(out["latent"])
            loss_terms = compute_ae_loss_terms(out, mask, batch, **loss_kwargs)
            # Field-decode Dice (protocol addition 2026-08-14, user-directed):
            # how much SHAPE the auxiliary distance heads carry, decoded as
            # distance < 2 voxels (0032 reference: dense 0.445 / query 0.430
            # vs occupancy 0.916; see research/evaluate_ae_field_decode.py).
            # Heads regress the clip-normalized target, so tau = 2/clip.
            field_decode_rows = {}
            clip_voxels = float(cfg.get("data", {}).get("distance_field", {}).get("clip_voxels", 16.0))
            if "distance" in out:
                decoded = out["distance"].float() < (2.0 / clip_voxels)
                target_bool = mask.bool()
                inter = float((decoded & target_bool).sum())
                field_decode_rows["val_distance_decode_dice_t2"] = (
                    2.0 * inter / max(float(decoded.sum()) + float(target_bool.sum()), 1.0))
            if "query_distance" in out and "query_distances" in batch:
                pred_in = out["query_distance"].float() < (2.0 / clip_voxels)
                gt_in = batch["query_distances"].float() < (2.0 / clip_voxels)
                inter = float((pred_in & gt_in).sum())
                field_decode_rows["val_query_decode_dice_t2"] = (
                    2.0 * inter / max(float(pred_in.sum()) + float(gt_in.sum()), 1.0))
            task_loss_rows.append({
                **field_decode_rows,
                "val_loss": float(loss_terms["loss"].detach().float().cpu()),
                "val_bce": float(loss_terms["bce"].detach().float().cpu()),
                "val_dice_loss": float(loss_terms["dice_loss"].detach().float().cpu()),
                "val_distance_loss": float(loss_terms["distance_loss"].detach().float().cpu()),
                "val_query_distance_loss": float(loss_terms["query_distance_loss"].detach().float().cpu()),
                **{
                    f"val_{key}": value
                    for key, value in loss_contribution_scalars(
                        loss_terms,
                        bce_weight=loss_kwargs["bce_weight"],
                        dice_weight=loss_kwargs["dice_weight"],
                        outside_weight=loss_kwargs["outside_weight"],
                        border_weight=loss_kwargs["border_weight"],
                        distance_weight=loss_kwargs["distance_weight"],
                        query_distance_weight=loss_kwargs["query_distance_weight"],
                        same_volume_repulsion_weight=loss_kwargs["same_volume_repulsion_weight"],
                    ).items()
                },
            })
            prob = torch.sigmoid(out["logits"]).float().cpu().numpy()
            gt = mask.float().cpu().numpy()
            for sample_index in range(prob.shape[0]):
                eval_seen_samples += 1
                target_voxels = int(mask[sample_index, 0].sum().detach().item())
                if target_voxels < val_min_target_voxels:
                    eval_excluded_small_target_samples += 1
                    continue
                payload = {
                    "first_payload": None,
                    "identity": {
                        "case_id": str(batch["case_id"][sample_index]),
                        "component_id": int(batch["component_id"][sample_index].detach().item()),
                        "crop_start": tuple(int(value) for value in batch["crop_start"][sample_index]),
                        "target_voxels": target_voxels,
                    },
                }
                if reporter.gallery_enabled:
                    payload["volumes"] = reporter.compact_volumes(
                        pred=prob[sample_index, 0],
                        gt=gt[sample_index, 0],
                    )
                if not first_payload_submitted:
                    payload["first_payload"] = {
                        "image": mask[sample_index, 0].float().cpu().numpy(),
                        "gt": gt[sample_index, 0],
                        "pred": prob[sample_index, 0],
                        "case_id": batch["case_id"][sample_index],
                        "component_id": int(batch["component_id"][sample_index].detach().item()),
                        "crop_start": tuple(int(value) for value in batch["crop_start"][sample_index]),
                        "target_voxels": target_voxels,
                    }
                    first_payload_submitted = True
                completed = metric_dispatcher.submit(
                    payload,
                    compute_p2sd_metrics,
                    prob[sample_index, 0],
                    gt[sample_index, 0],
                    prompt_zyx=batch["prompt_zyx"][sample_index],
                    cfg=metric_cfg,
                )
                for completed_payload, metrics in completed:
                    consume_metrics(completed_payload, metrics)
        for completed_payload, metrics in metric_dispatcher.drain():
            consume_metrics(completed_payload, metrics)
    if not metric_rows:
        raise RuntimeError(
            "No validation samples met metrics.val_min_target_voxels="
            f"{val_min_target_voxels} across {eval_seen_samples} sampled cases")
    metrics = average_metric_rows(metric_rows)
    tail_stats = reporter.finalize()
    task_losses = {
        key: float(np.mean([item[key] for item in task_loss_rows]))
        for key in task_loss_rows[0]
    }
    row = {
        "step": step,
        "epoch": float(step / max(1, steps_per_epoch)),
        "split": "val",
        "eval_samples": len(metric_rows),
        "eval_seen_samples": eval_seen_samples,
        "eval_excluded_small_target_samples": eval_excluded_small_target_samples,
        "val_min_target_voxels": val_min_target_voxels,
        **task_losses,
        **metrics,
        **tail_stats,
    }
    append_jsonl(run_dir / "metrics.jsonl", row)
    viz_cfg = cfg.get("visualization", {})
    milestone_epoch = resolve_visualization_milestone_epoch(
        viz_cfg,
        step=step,
        steps_per_epoch=steps_per_epoch,
    )
    if bool(viz_cfg.get("save_png", True)) and first_payload is not None:
        _save_ae_visual_pngs(
            run_dir / "viz" / f"ae_step_{step:06d}.png",
            payload=first_payload,
            metrics_cfg=metrics_cfg,
            meta={
                "step": step,
                "task": "ae",
                "conditioned": False,
                "case_id": first_payload["case_id"],
                "component_id": first_payload["component_id"],
                "crop_start": first_payload["crop_start"],
                "target_voxels": first_payload["target_voxels"],
            },
        )
        prune_visualization_groups(
            run_dir / "viz",
            keep_latest=resolve_visualization_keep_latest(
                viz_cfg,
                "keep_latest_val_png",
                "val_png_keep_latest",
            ),
            stem_prefixes=("ae_step_",),
        )
        if milestone_epoch is not None:
            _save_ae_visual_pngs(
                run_dir / "viz" / "milestones" / f"ae_epoch_{milestone_epoch:06d}.png",
                payload=first_payload,
                metrics_cfg=metrics_cfg,
                meta={
                    "step": step,
                    "epoch": milestone_epoch,
                    "task": "ae_milestone",
                    "conditioned": False,
                    "case_id": first_payload["case_id"],
                    "component_id": first_payload["component_id"],
                    "crop_start": first_payload["crop_start"],
                    "target_voxels": first_payload["target_voxels"],
                },
            )
    save_rolling_3d = bool(viz_cfg.get("save_3d", False)) and not bool(
        viz_cfg.get("save_3d_on_milestones_only", False))
    save_milestone_3d = milestone_epoch is not None and bool(
        viz_cfg.get("milestone_save_3d", False))
    if (save_rolling_3d or save_milestone_3d) and first_payload is not None:
        if save_rolling_3d:
            _save_ae_3d_visualization(
                run_dir / "viz3d" / f"ae_step_{step:06d}",
                payload=first_payload,
                metrics_cfg=metrics_cfg,
                meta={"step": step, "task": "ae", "conditioned": False},
                viz_cfg=viz_cfg,
            )
        if save_milestone_3d:
            _save_ae_3d_visualization(
                run_dir / "viz3d" / "milestones" / f"ae_epoch_{milestone_epoch:06d}",
                payload=first_payload,
                metrics_cfg=metrics_cfg,
                meta={
                    "step": step,
                    "epoch": milestone_epoch,
                    "task": "ae_milestone",
                    "conditioned": False,
                },
                viz_cfg=viz_cfg,
            )
    if save_rolling_3d:
        prune_visualization_groups(
            run_dir / "viz3d",
            keep_latest=resolve_visualization_keep_latest(
                viz_cfg,
                "keep_latest_val_3d",
                "val_3d_keep_latest",
            ),
            stem_prefixes=("ae_step_",),
        )
    return float(metrics.get("quality_composite", metrics.get("dice", 0.0)))


def _save_ae_visual_pngs(
    path: Path,
    *,
    payload: dict,
    metrics_cfg: dict,
    meta: dict,
) -> None:
    threshold = float(metrics_cfg.get("threshold", 0.5))
    tolerance = _max_metric_tolerance(metrics_cfg)
    visual_state = prepare_p2sd_visualization_state(
        image=payload["image"],
        gt=payload["gt"],
        pred=payload["pred"],
        threshold=threshold,
        tolerance_voxels=tolerance,
    )
    save_p2sd_diagnostic_png(
        path,
        image=payload["image"],
        gt=payload["gt"],
        pred=payload["pred"],
        prompt_zyx=None,
        metrics=payload["metrics"],
        meta=meta,
        threshold=threshold,
        tolerance_voxels=tolerance,
        prediction_label="ae",
        visual_state=visual_state,
    )
    save_p2sd_projection_png(
        path.with_name(path.stem + "_projection.png"),
        image=payload["image"],
        gt=payload["gt"],
        pred=payload["pred"],
        prompt_zyx=None,
        metrics=payload["metrics"],
        meta=meta,
        threshold=threshold,
        tolerance_voxels=tolerance,
        prediction_label="ae",
        visual_state=visual_state,
    )


def _save_ae_3d_visualization(
    path_prefix: Path,
    *,
    payload: dict,
    metrics_cfg: dict,
    meta: dict,
    viz_cfg: dict,
) -> None:
    save_p2sd_3d_artifacts(
        path_prefix,
        image=payload["image"],
        gt=payload["gt"],
        pred=payload["pred"],
        prompt_zyx=None,
        metrics=payload["metrics"],
        meta=meta,
        threshold=float(metrics_cfg.get("threshold", 0.5)),
        max_points=int(viz_cfg.get("save_3d_max_points", 200_000)),
    )


def _max_metric_tolerance(metrics_cfg: dict) -> int:
    tolerances = metrics_cfg.get("tolerance_voxels", [2])
    if isinstance(tolerances, (list, tuple)):
        return max(int(value) for value in tolerances) if tolerances else 0
    return int(tolerances)


def resolve_visualization_keep_latest(viz_cfg: dict, *keys: str) -> int | None:
    for key in keys:
        if key in viz_cfg:
            return int(viz_cfg[key])
    if "keep_latest" in viz_cfg:
        return int(viz_cfg["keep_latest"])
    return None


def average_metric_rows(rows: list[dict[str, float]]) -> dict[str, float]:
    keys = sorted({key for row in rows for key in row if isinstance(row.get(key), (int, float))})
    out = {}
    for key in keys:
        values = [float(row[key]) for row in rows if isinstance(row.get(key), (int, float))]
        if values:
            out[key] = float(sum(values) / len(values))
    return out


if __name__ == "__main__":
    raise SystemExit(main())
