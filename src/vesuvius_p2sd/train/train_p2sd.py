"""Train full-attention point-to-sheet decoding."""

from __future__ import annotations

import argparse
import json
import shutil
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F

from vesuvius_p2sd.data.dataset import (
    apply_gpu_photometric_augmentations,
    build_patch_augmentation_config,
)
from vesuvius_p2sd.data.distance_targets import (
    prepare_ae_distance_targets,
    sample_query_distance_points_gpu,
    signed_distance_edt_cucim,
)
from vesuvius_p2sd.eval.metric_dispatch import (
    MetricDispatchSettings,
    OrderedMetricDispatcher,
)
from vesuvius_p2sd.eval.metrics import compute_p2sd_metrics, config_from_mapping
from vesuvius_p2sd.eval.probes import (
    aggregate_fixed_probe_rows,
    pairwise_prediction_dice_stats,
)
from vesuvius_p2sd.models.p2sd import build_p2sd
from vesuvius_p2sd.research.run_status import update_heartbeat, write_json, write_resume_context
from vesuvius_p2sd.train.train_binary_seg import (
    sheet_latent_anchors,
    voxel_latent_distill_loss,
    voxel_prototype_contrast_loss,
)
from vesuvius_p2sd.train.common import (
    MetricWindow,
    resolve_log_interval_steps,
    amp_dtype,
    append_jsonl,
    autocast_context,
    collect_nonfinite_gradient_details,
    cycle,
    configured_lrs,
    restore_configured_lrs,
    get_device,
    load_ae_from_config,
    make_fixed_probe_loader,
    make_loader,
    make_lr_schedule,
    make_optimizer,
    maybe_channels_last_3d,
    move_batch,
    prepare_run,
    print_model_report,
    save_checkpoint,
    snapshot_checkpoint_files,
    seed_everything,
    should_stop,
    weighted_bce_with_logits,
)
from vesuvius_p2sd.train.profiling import P2SDStepProfiler
# Re-export these helpers to preserve existing evaluator imports.
from vesuvius_p2sd.train.p2sd_losses import (
    aux_mse_loss,
    per_round_final_mse_scale,
    same_sheet_prompt_consistency_loss,
    different_sheet_prompt_contrast_loss,
)
from vesuvius_p2sd.train.eval_reporting import EvalSampleReporter
from vesuvius_p2sd.train.loss_plots import write_loss_plots
from vesuvius_p2sd.train.visualization import (
    prune_visualization_epoch_directories,
    prune_visualization_groups,
    prepare_p2sd_visualization_state,
    save_p2sd_3d_artifacts,
    save_p2sd_diagnostic_png,
    save_p2sd_projection_png,
    save_p2sd_train_batch_png,
    resolve_visualization_milestone_epoch,
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
        entrypoint="train_p2sd",
    )
    raw_max_steps = cfg.get("training", {}).get("max_steps")
    if args.dry_run or raw_max_steps == 0 or raw_max_steps == "0":
        write_json(run_dir / "state.json", {
            "status": "dry_run_complete",
            "message": "Config validated; P2SD training skipped.",
        })
        update_heartbeat(run_dir, status="dry_run_complete")
        return 0

    seed_everything(int(cfg.get("seed", cfg.get("training", {}).get("seed", 42))))
    configure_torch_runtime(cfg.get("training", {}))
    device = get_device(cfg)
    target_cfg = cfg.get("target_ae", {})
    target_trainable = bool(target_cfg.get("trainable", False))
    deduplicate_frozen_targets = bool(target_cfg.get("deduplicate_frozen_pair_targets", True))
    target_ae, _ = load_ae_from_config(
        target_cfg.get("config_path", "configs/ae/single_sheet_sparse_5down_dense_input.yaml"),
        target_cfg.get("checkpoint_path"),
    )
    target_ae = target_ae.to(device).eval()
    for p in target_ae.parameters():
        p.requires_grad_(target_trainable)

    model = build_p2sd(cfg, latent_channels=target_ae.latent_channels).to(device)
    if bool(cfg.get("training", {}).get("channels_last_3d", False)):
        model = model.to(memory_format=torch.channels_last_3d)
        target_ae = target_ae.to(memory_format=torch.channels_last_3d)
    logging_cfg = cfg.get("logging", {})
    print_model_report(
        "p2sd",
        model,
        run_dir,
        print_to_stdout=bool(logging_cfg.get("print_model", True)),
    )
    print_model_report(
        "target_ae",
        target_ae,
        run_dir,
        print_to_stdout=bool(logging_cfg.get("print_target_ae_model", False)),
    )
    # Joint dense objective (user directive 2026-08-26): 0022's context-refiner +
    # conv decoder + head attached to THIS model's (trainable) image encoder,
    # trained on grid-aligned context crops (crop_grid 5 = 160^3 voxels at
    # PS320) with the binseg trainer's ignore-masked BCE+Dice, alongside the
    # P2SD latent losses. last.pt stays P2SD-format; the head is saved
    # separately as dense_last.pt.
    dense_cfg = cfg.get("p2sd", {}).get("dense_aux", {}) or {}
    dense_weight = float(dense_cfg.get("weight", 0.0))
    dense = None
    opt_model = model
    if dense_weight > 0:
        from vesuvius_p2sd.models.binary_seg import BinarySegFromP2SD

        trainable_flags = {id(p): bool(p.requires_grad) for p in model.parameters()}
        refiner_cfg = dense_cfg.get("refiner", {}) or {}
        dense = BinarySegFromP2SD(
            model,
            decoder_channels=dense_cfg.get("decoder_channels", (512, 256, 128, 64, 32, 16)),
            num_groups=int(dense_cfg.get("num_groups", 8)),
            dropout=float(dense_cfg.get("dropout", 0.0)),
            refiner_depth=int(refiner_cfg.get("depth", 0)),
            refiner_heads=int(refiner_cfg.get("heads", 8)),
            refiner_mlp_ratio=float(refiner_cfg.get("mlp_ratio", 4.0)),
            refiner_rope_mode=str(refiner_cfg.get("rope_mode", "per_axis")),
        )
        # BinarySegFromP2SD freezes its trunk in __init__; restore the P2SD
        # freeze policy (the whole point here is a trainable encoder).
        for p in model.parameters():
            p.requires_grad_(trainable_flags[id(p)])
        init_ckpt = dense_cfg.get("init_checkpoint")
        if init_ckpt:
            ck = torch.load(init_ckpt, map_location="cpu", weights_only=False)
            head_state = {k: v for k, v in ck.get("model", ck).items() if not k.startswith("trunk.")}
            missing, unexpected = dense.load_state_dict(head_state, strict=False)
            missing = [k for k in missing if not k.startswith("trunk.")]
            if missing or unexpected:
                raise ValueError(f"dense_aux init mismatch: missing={missing[:5]} unexpected={unexpected[:5]}")
            print(f"[dense_aux] loaded {len(head_state)} head tensors from {init_ckpt}", flush=True)
        dense = dense.to(device)
        if bool(cfg.get("training", {}).get("channels_last_3d", False)):
            dense = dense.to(memory_format=torch.channels_last_3d)
        opt_model = torch.nn.ModuleDict({
            "p2sd": model,
            "dense_refiner": dense.context_refiner,
            "dense_decoder": dense.decoder,
            "dense_head": dense.head,
        })
        print(f"[dense_aux] weight {dense_weight} crop_grid {int(dense_cfg.get('crop_grid', 5))} "
              f"pos_weight {float(dense_cfg.get('pos_weight', 2.0))} bce {float(dense_cfg.get('bce_weight', 1.0))} "
              f"dice {float(dense_cfg.get('dice_weight', 1.0))}; encoder trainable params "
              f"{sum(p.numel() for n, p in model.named_parameters() if n.startswith('image_encoder') and p.requires_grad)}", flush=True)
    dense_crop_grid = int(dense_cfg.get("crop_grid", 5))
    # Where the training crop of the bin-seg decoder is taken (user 2026-09-11): "context" = a crop_grid^3 block of the
    # 10^3 token grid decoded alone (every conv sees zero padding on the crop faces, unlike full-grid inference);
    # "second_last" = the full grid is decoded to the second-to-last stage (half resolution) and the crop is taken there,
    # so only the last upsample + logits see crop padding. The output crop is crop_grid * 32 voxels either way.
    dense_crop_stage = str(dense_cfg.get("crop_stage", "context"))
    if dense_crop_stage not in ("context", "second_last"):
        raise ValueError(f"dense_aux.crop_stage must be context|second_last, got {dense_crop_stage!r}")
    dense_pos_weight = float(dense_cfg.get("pos_weight", 2.0))
    dense_bce_w = float(dense_cfg.get("bce_weight", 1.0))
    dense_dice_w = float(dense_cfg.get("dice_weight", 1.0))

    def save_dense_state(run_dir_: Path) -> None:
        if dense is None:
            return
        path = run_dir_ / "dense_last.pt"
        temporary = path.with_name(".dense_last.pt.tmp")
        torch.save({"model": {k: v for k, v in dense.state_dict().items() if not k.startswith("trunk.")},
                    "dense_aux": dense_cfg, "step": last_completed_step}, temporary)
        temporary.replace(path)

    # 0061 (2026-08-27): image-conditioned refinement head on the AE decode.
    # Everything else frozen (P2SD + AE); the head fuses the AE decoder taps
    # (1/4, 1/2, full) with the frozen encoder's 1/4 feature + raw image and
    # predicts a residual on the AE logits. Trained on FULL-volume decodes of
    # P2SD's own predictions (the exact inference path), ignore-masked
    # BCE+Dice vs the thin GT sheet. Saved separately as refine_last.pt.
    refine_cfg = cfg.get("p2sd", {}).get("refine", {}) or {}
    refine_weight = float(refine_cfg.get("weight", 0.0))
    refine = None
    if refine_weight > 0:
        from vesuvius_p2sd.models.refine_head import build_refine_head

        if dense is not None:
            raise ValueError("p2sd.refine and p2sd.dense_aux are mutually exclusive")
        refine = build_refine_head(refine_cfg).to(device)
        refine.checkpoint_level1 = bool(refine_cfg.get("checkpoint_level1", False))
        if bool(refine_cfg.get("freeze_p2sd", True)):
            for p in model.parameters():
                p.requires_grad_(False)
        init_ckpt = refine_cfg.get("init_checkpoint")
        if init_ckpt:
            ck = torch.load(init_ckpt, map_location="cpu", weights_only=False)
            refine.load_state_dict(ck.get("model", ck), strict=True)
            print(f"[refine] loaded head from {init_ckpt}", flush=True)
        opt_model = torch.nn.ModuleDict({"p2sd": model, "refine_head": refine})
        print(f"[refine] weight {refine_weight} max_sheets {int(refine_cfg.get('max_sheets', 2))} "
              f"pos_weight {float(refine_cfg.get('pos_weight', 2.0))} ae_factors {refine.ae_factors} "
              f"image_factors {refine.image_factors} widths {refine.widths}; head params "
              f"{sum(p.numel() for p in refine.parameters())}; p2sd trainable params "
              f"{sum(p.numel() for p in model.parameters() if p.requires_grad)}", flush=True)
    refine_max_sheets = int(refine_cfg.get("max_sheets", 2))
    refine_pos_weight = float(refine_cfg.get("pos_weight", 2.0))
    refine_bce_w = float(refine_cfg.get("bce_weight", 1.0))
    refine_dice_w = float(refine_cfg.get("dice_weight", 1.0))
    refine_target_mode = str(refine_cfg.get("target_mode", "gt"))   # gt | ae_gt (0062)
    if refine_target_mode not in {"gt", "ae_gt", "ae_gt_hard"}:
        raise ValueError(f"p2sd.refine.target_mode must be gt|ae_gt|ae_gt_hard, got {refine_target_mode!r}")
    refine_volume_weight = float(refine_cfg.get("volume_weight", 0.0))
    refine_band_weight = float(refine_cfg.get("band_weight", 0.0))      # 0063: extra loss weight on the surface band
    refine_band_inner = int(refine_cfg.get("band_inner", 2))
    refine_band_outer = int(refine_cfg.get("band_outer", 6))
    refine_sparse = bool(refine_cfg.get("sparse", False))                # 0064: block-sparse tiles (arch v2)
    refine_tile_threshold = float(refine_cfg.get("tile_threshold", -4.0))

    def save_refine_state(run_dir_: Path) -> None:
        if refine is None:
            return
        torch.save({"model": refine.state_dict(), "refine": refine_cfg}, run_dir_ / "refine_last.pt")

    # 0065 (2026-08-28): image-conditioned sheet decoder trained from scratch on
    # GT masks (models/sheet_decoder.py); P2SD + encoder frozen. Replaces the AE
    # decoder at inference (decoder_last.pt, loaded via --refine_head_path).
    sdec_cfg = cfg.get("p2sd", {}).get("sheet_decoder", {}) or {}
    sdec_weight = float(sdec_cfg.get("weight", 0.0))
    sdec = None
    if sdec_weight > 0:
        from vesuvius_p2sd.models.sheet_decoder import build_sheet_decoder

        if dense is not None or refine is not None:
            raise ValueError("p2sd.sheet_decoder is exclusive with dense_aux / refine")
        sdec = build_sheet_decoder(sdec_cfg).to(device)
        sdec.checkpoint_level1 = bool(sdec_cfg.get("checkpoint_level1", True))
        if bool(sdec_cfg.get("freeze_p2sd", True)):
            for p in model.parameters():
                p.requires_grad_(False)
        if sdec_cfg.get("init_checkpoint"):
            ck = torch.load(sdec_cfg["init_checkpoint"], map_location="cpu", weights_only=False)
            sdec.load_state_dict(ck["model"], strict=True)
            print(f"[sheet_decoder] loaded {sdec_cfg['init_checkpoint']}", flush=True)
        opt_model = torch.nn.ModuleDict({"p2sd": model, "sheet_decoder": sdec})
        print(f"[sheet_decoder] weight {sdec_weight} widths {sdec.widths} stem {sdec.stem_channels} params "
              f"{sum(p.numel() for p in sdec.parameters())}; max_sheets {int(sdec_cfg.get('max_sheets', 4))} "
              f"latent_gt_mix {float(sdec_cfg.get('latent_gt_mix', 0.25))} pos_weight {float(sdec_cfg.get('pos_weight', 1.0))} "
              f"aux_weight {float(sdec_cfg.get('aux_weight', 0.3))} gate_logit {float(sdec_cfg.get('gate_logit', -3.0))} "
              f"max_extra_tiles {int(sdec_cfg.get('max_extra_tiles', 150))}", flush=True)
    # 0066: union-mask decoder (multi-task with the sheet decoder; the seed model at inference)
    udec_cfg = cfg.get("p2sd", {}).get("union_decoder", {}) or {}
    udec_weight = float(udec_cfg.get("weight", 0.0))
    udec = None
    if udec_weight > 0:
        from vesuvius_p2sd.models.sheet_decoder import build_sheet_decoder as _build_udec

        if sdec is None:
            raise ValueError("p2sd.union_decoder requires p2sd.sheet_decoder")
        udec_cfg = {**udec_cfg, "latent_channels": int(udec_cfg.get("latent_channels", 512))}
        udec = _build_udec(udec_cfg).to(device)
        udec.checkpoint_level1 = bool(udec_cfg.get("checkpoint_level1", True))
        opt_model = torch.nn.ModuleDict({"p2sd": model, "sheet_decoder": sdec, "union_decoder": udec})
        print(f"[union_decoder] weight {udec_weight} widths {udec.widths} params {sum(p.numel() for p in udec.parameters())}; "
              f"max_tiles_per_image {int(udec_cfg.get('max_tiles_per_image', 200))} pos_weight {float(udec_cfg.get('pos_weight', 1.0))}", flush=True)
    # 0069 (2026-08-29): sheet band refiner (models/band_refiner.py) on the FROZEN 0065a decoder —
    # global self-attention over all band voxels of a 160^3 patch around each decoded sheet.
    brf_cfg = cfg.get("p2sd", {}).get("band_refiner", {}) or {}
    brf_weight = float(brf_cfg.get("weight", 0.0))
    brf = brf_dec = brf_binseg = None
    brf_dec_cfg: dict = {}
    if brf_weight > 0:
        from vesuvius_p2sd.models.band_refiner import build_band_refiner
        from vesuvius_p2sd.models.sheet_decoder import load_sheet_decoder

        if sdec is not None or refine is not None or dense is not None:
            raise ValueError("p2sd.band_refiner is exclusive with sheet_decoder / refine / dense_aux")
        for p in model.parameters():
            p.requires_grad_(False)
        brf_dec, brf_dec_cfg = load_sheet_decoder(brf_cfg["decoder_checkpoint"], device)
        for p in brf_dec.parameters():
            p.requires_grad_(False)
        brf_dec.checkpoint_level1 = False
        brf = build_band_refiner(brf_cfg).to(device)
        # last.pt stores only the p2sd model, so a mid-run resume must reload the head from
        # band_last.pt or it silently restarts from random init (found on the 0069c ep34 resume).
        if brf_cfg.get("resume_band_path"):
            band_state = torch.load(brf_cfg["resume_band_path"], map_location="cpu", weights_only=False)
            brf.load_state_dict(band_state["model"], strict=True)
            print(f"[band_refiner] resumed head weights from {brf_cfg['resume_band_path']}", flush=True)
        opt_model = torch.nn.ModuleDict({"p2sd": model, "band_refiner": brf})
        if brf.cfg.binseg_channels:
            from vesuvius_p2sd.models.band_refiner import load_frozen_binseg

            brf_binseg = load_frozen_binseg(brf_cfg["binseg_run_dir"], device)
            print(f"[band_refiner] binseg feature from {brf_cfg['binseg_run_dir']} "
                  f"({brf.cfg.binseg_channels} ch)", flush=True)
        print(f"[band_refiner] weight {brf_weight} params {sum(p.numel() for p in brf.parameters())} cfg {brf.cfg}; "
              f"frozen decoder {brf_cfg['decoder_checkpoint']} gate {float(brf_dec_cfg.get('gate_logit', -3.0))}", flush=True)
    brf_gate_logit = float(brf_cfg.get("gate_logit", brf_dec_cfg.get("gate_logit", -3.0)))
    brf_gate_dilate = int(brf_cfg.get("gate_dilate", brf_dec_cfg.get("gate_dilate", 1)))
    brf_max_sheets = int(brf_cfg.get("max_sheets", 4))
    brf_gt_mix = float(brf_cfg.get("gt_mix", 0.25))
    brf_pos_weight = float(brf_cfg.get("pos_weight", 1.0))
    brf_bce_w = float(brf_cfg.get("bce_weight", 1.0))
    brf_dice_w = float(brf_cfg.get("dice_weight", 1.0))
    brf_patch = int(brf_cfg.get("patch", 160))
    brf_max_tokens = int(brf_cfg.get("max_train_tokens", 150000))
    brf_radius = int(brf_cfg.get("radius", 5))
    # 0069d: pairwise hinge (fg token must out-score off-sheet tokens by a margin), see band_refiner_step.
    brf_rank_kw = {"rank_weight": float(brf_cfg.get("rank_weight", 0.0)),
                   "rank_margin": float(brf_cfg.get("rank_margin", 2.0)),
                   "rank_shell": int(brf_cfg.get("rank_shell", 2)),
                   "rank_offset_min": int(brf_cfg.get("rank_offset_min", 3)),
                   "rank_offset_max": int(brf_cfg.get("rank_offset_max", 6)),
                   "rank_pairs_per_pos": int(brf_cfg.get("rank_pairs_per_pos", 4)),
                   "rank_max_pos": int(brf_cfg.get("rank_max_pos", 16384)),
                   "rank_local_frac": float(brf_cfg.get("rank_local_frac", 0.5))}
    if brf is not None and brf_rank_kw["rank_weight"] > 0:
        print(f"[band_refiner] pairwise hinge {brf_rank_kw} bce_weight {brf_bce_w} dice_weight {brf_dice_w}", flush=True)

    def save_brf_state(run_dir_: Path) -> None:
        if brf is None:
            return
        torch.save({"model": brf.state_dict(), "band_refiner": brf_cfg,
                    "decoder_checkpoint": brf_cfg["decoder_checkpoint"],
                    "binseg_run_dir": brf_cfg.get("binseg_run_dir")}, run_dir_ / "band_last.pt")

    udec_pos_weight = float(udec_cfg.get("pos_weight", 1.0))
    udec_bce_w = float(udec_cfg.get("bce_weight", 1.0))
    udec_dice_w = float(udec_cfg.get("dice_weight", 1.0))
    udec_aux_w = float(udec_cfg.get("aux_weight", 0.3))
    udec_aux_pos_weight = float(udec_cfg.get("aux_pos_weight", 2.0))
    udec_gate_logit = float(udec_cfg.get("gate_logit", -3.0))
    udec_max_tiles = int(udec_cfg.get("max_tiles_per_image", 200))
    sdec_detach_latent = bool(sdec_cfg.get("detach_latent", True))
    sdec_max_sheets = int(sdec_cfg.get("max_sheets", 4))
    sdec_gt_mix = float(sdec_cfg.get("latent_gt_mix", 0.25))
    sdec_pos_weight = float(sdec_cfg.get("pos_weight", 1.0))
    sdec_bce_w = float(sdec_cfg.get("bce_weight", 1.0))
    sdec_dice_w = float(sdec_cfg.get("dice_weight", 1.0))
    sdec_aux_w = float(sdec_cfg.get("aux_weight", 0.3))
    sdec_aux_pos_weight = float(sdec_cfg.get("aux_pos_weight", 2.0))
    sdec_gate_logit = float(sdec_cfg.get("gate_logit", -3.0))
    sdec_max_extra = int(sdec_cfg.get("max_extra_tiles", 150))

    def save_sdec_state(run_dir_: Path) -> None:
        if sdec is None:
            return
        torch.save({"model": sdec.state_dict(), "sheet_decoder": sdec_cfg}, run_dir_ / "decoder_last.pt")
        if udec is not None:
            torch.save({"model": udec.state_dict(), "union_decoder": udec_cfg}, run_dir_ / "union_last.pt")

    optimizer = make_optimizer(opt_model, cfg)
    start_step, best = maybe_resume_p2sd(model, optimizer, cfg)
    raw_model = model
    model = maybe_compile_model(model, cfg.get("training", {}))
    target_encode = maybe_compile_target_encode(target_ae, cfg.get("training", {}))
    step_profiler = P2SDStepProfiler(
        cfg.get("training", {}),
        run_dir,
        raw_model,
        device,
    )
    train_loader = make_loader(cfg, "train", shuffle=True)
    val_loader = make_loader(cfg, "val", shuffle=False)
    fixed_probe_loader = make_fixed_probe_loader(cfg)
    train_iter = cycle(train_loader)
    dtype = amp_dtype(cfg.get("training", {}).get("amp_dtype"))
    patch_size = tuple(int(value) for value in cfg.get("data", {}).get("patch_size", [64, 64, 64]))
    train_augmentation = build_patch_augmentation_config(
        cfg.get("data", {}).get("augmentation", {}),
        split="train",
        patch_size=patch_size,
    )
    gpu_photometric_generator = None
    if train_augmentation.photometric_device == "cuda":
        if device.type != "cuda":
            raise ValueError("data.augmentation.photometric.device=cuda requires device=cuda")
        gpu_photometric_generator = torch.Generator(device=device)
        gpu_photometric_generator.manual_seed(int(cfg.get("seed", 42)) + 917)
    steps_per_epoch = max(1, len(train_loader))
    max_steps, target_epochs = resolve_max_steps(cfg.get("training", {}), steps_per_epoch)
    lr_schedule = make_lr_schedule(optimizer, cfg, max_steps)
    print(f"[lr_schedule] {lr_schedule.describe()}", flush=True)
    log_interval = resolve_log_interval_steps(cfg.get("training", {}), steps_per_epoch)
    log_window = MetricWindow()
    eval_interval = resolve_interval_steps(cfg.get("training", {}), "eval", steps_per_epoch, max_steps)
    save_interval = resolve_interval_steps(cfg.get("training", {}), "save", steps_per_epoch, max_steps)
    scene_eval_settings = resolve_scene_eval_settings(cfg, steps_per_epoch, max_steps)
    if scene_eval_settings is not None and refine is not None:
        # Milestones must decode through the head: save it with last.pt and
        # hand the shards its path.
        scene_eval_settings["refine_head"] = True
        scene_eval_settings["refine_head_file"] = "refine_last.pt"
        scene_eval_settings["save_extra_state"] = save_refine_state
    if scene_eval_settings is not None and sdec is not None:
        scene_eval_settings["refine_head"] = True
        scene_eval_settings["refine_head_file"] = "decoder_last.pt"
        scene_eval_settings["save_extra_state"] = save_sdec_state
    if scene_eval_settings is not None and brf is not None:
        scene_eval_settings["refine_head"] = True
        scene_eval_settings["refine_head_file"] = "band_last.pt"
        scene_eval_settings["save_extra_state"] = save_brf_state
    eval_on_last_step = bool(cfg.get("training", {}).get("eval_on_last_step", True))
    save_on_last_step = bool(cfg.get("training", {}).get("save_on_last_step", True))
    loss_cfg = cfg.get("p2sd", {}).get("loss", {})
    aux_weights = [float(v) for v in loss_cfg.get("per_round_mse_weights", [])]
    aux_weights_mode = str(loss_cfg.get("per_round_mse_weights_mode", "aux_only"))
    decoded_weight = float(loss_cfg.get("decoded_bce_weight", 0.0))
    decoded_interval = max(1, int(loss_cfg.get("decoded_bce_interval_steps", 1)))
    decoded_warmup = int(loss_cfg.get("decoded_bce_warmup_steps", 0))
    decoded_pos_weight = loss_cfg.get("decoded_pos_weight", "auto")
    decoded_max_auto_pos_weight = float(loss_cfg.get("decoded_max_auto_pos_weight", 32.0))
    # Coarse-stage decoded loss (2026-08-25): supervise the AE decoder's
    # deep-supervision aux logits at the listed stages instead of the full
    # 320^3 decode -- the same BCE+Dice/target-downsampling the AE itself
    # trains with -- so the decoded term can run every step.
    decoded_stages = [int(v) for v in (loss_cfg.get("decoded_stages") or [])]
    decoded_stage_weights = [float(v) for v in (loss_cfg.get("decoded_stage_weights") or [])]
    decoded_bce_term_weight = float(loss_cfg.get("decoded_bce_term_weight", 1.0))
    decoded_dice_weight = float(loss_cfg.get("decoded_dice_weight", 0.0))
    # decoded_target_mode (2026-08-26): "gt" = round-1 BCE+Dice against
    # max-pooled GT occupancy (rejected: teaches bloat); "ae_soft" = distill
    # the AE's coarse aux logits of z_pred toward those of z_gt (the target
    # latent) with soft-target BCE -- AE-consistent, no pooling artifact.
    decoded_target_mode = str(loss_cfg.get("decoded_target_mode", "gt"))
    if decoded_target_mode not in {"gt", "ae_soft", "crop_gt"}:
        raise ValueError(f"p2sd.loss.decoded_target_mode must be gt|ae_soft|crop_gt, got {decoded_target_mode!r}")
    # crop_gt (2026-08-26): FULL-resolution per-sheet decoded loss on a
    # grid-aligned crop of the predicted latent (decoded_crop_grid^3 latent
    # cells -> 32x that many voxels), BCE+Dice against the actual thin GT sheet
    # crop, ignore-masked. No pooled targets (0055) and no distillation (0057);
    # ~1/8 of a full decode at crop 5.
    decoded_crop_grid = int(loss_cfg.get("decoded_crop_grid", 5))
    if decoded_target_mode == "crop_gt" and decoded_weight > 0:
        print(f"[decoded_loss] crop_gt: full-res per-sheet decode on {decoded_crop_grid}^3 latent cells "
              f"x{decoded_weight} every {decoded_interval} step(s); bce {decoded_bce_term_weight} dice {decoded_dice_weight} "
              f"pos_weight {decoded_pos_weight}", flush=True)
    if decoded_stages and decoded_weight > 0:
        if len(decoded_stage_weights) != len(decoded_stages):
            raise ValueError("p2sd.loss.decoded_stage_weights must match decoded_stages")
        have = set(getattr(target_ae, "deep_supervision_indices", {}).values())
        if not set(decoded_stages) <= have:
            raise ValueError(f"target AE lacks deep-supervision heads for stages {decoded_stages} (has {sorted(have)})")
        print(f"[decoded_loss] coarse aux stages {decoded_stages} weights {decoded_stage_weights} "
              f"x{decoded_weight} every {decoded_interval} step(s); bce {decoded_bce_term_weight} dice {decoded_dice_weight}; "
              f"target {decoded_target_mode}", flush=True)
    consistency_weight = float(loss_cfg.get("same_sheet_prompt_consistency_weight", 0.0))
    contrast_weight = float(loss_cfg.get("different_sheet_prompt_contrast_weight", 0.0))
    # Geometric consistency at the OUTPUT resolution (user directive 2026-09-05): the same prompts on a flipped /
    # xy-rotated image must decode (through the frozen AE, full-res crop) to the transformed-back same sheet. The
    # latent is not equivariant (measured: flipped decodes agree at dice 0.6-0.9 only), so the term acts on decoded
    # probabilities of a decoded_crop_grid^3-cell crop and its image under the transform. grad "both" back-propagates
    # through both views, "one" detaches the transformed view (cheaper, mean-teacher-like).
    geo_cfg = loss_cfg.get("geo_consistency", {}) or {}
    geo_weight = float(geo_cfg.get("weight", 0.0))
    geo_interval = max(1, int(geo_cfg.get("interval_steps", 1)))
    geo_transforms = int(geo_cfg.get("transforms", 16))
    geo_grad = str(geo_cfg.get("grad", "both"))
    geo_crop_grid = int(geo_cfg.get("crop_grid", decoded_crop_grid))
    if geo_weight > 0:
        print(f"[geo_consistency] weight {geo_weight} every {geo_interval} step(s), {geo_transforms} transforms, "
              f"grad {geo_grad}, crop {geo_crop_grid}^3 latent cells", flush=True)
    contrast_margin = float(loss_cfg.get("different_sheet_prompt_contrast_margin", 1.0))
    latent_prompt_cfg = cfg.get("p2sd", {}).get("latent_prompt", {}) or {}
    latent_prompt_mode = str(latent_prompt_cfg.get("mode", "none"))
    if latent_prompt_mode not in {"none", "teacher_centroid"}:
        raise ValueError(f"p2sd.latent_prompt.mode must be none|teacher_centroid, got {latent_prompt_mode!r}")
    latent_prompt_present_prob = float(latent_prompt_cfg.get("present_prob", 0.8))
    latent_prompt_teacher_range = [int(v) for v in latent_prompt_cfg.get("teacher_clicks_range", [1, 4])]
    coordinate_query_cfg = loss_cfg.get("coordinate_query", {})
    coordinate_query_enabled = bool(coordinate_query_cfg.get("enabled", False))
    coordinate_query_weight = float(coordinate_query_cfg.get("weight", 0.0))
    coordinate_query_num_points = int(coordinate_query_cfg.get("num_points", 0))
    coordinate_query_beta = float(coordinate_query_cfg.get("smooth_l1_beta", 0.1))
    coordinate_query_sampling = str(coordinate_query_cfg.get("sampling", "uniform"))
    coordinate_query_band_voxels = float(coordinate_query_cfg.get("band_voxels", 8.0))
    coordinate_query_inside_fraction = float(coordinate_query_cfg.get("inside_fraction", 1.0 / 3.0))
    coordinate_query_near_outside_fraction = float(
        coordinate_query_cfg.get("near_outside_fraction", 1.0 / 3.0)
    )
    if coordinate_query_enabled and coordinate_query_num_points <= 0:
        raise ValueError("p2sd.loss.coordinate_query.num_points must be positive when enabled")
    # Task-12: JL-anchor distill on image_context (jointly with the P2SD
    # objective). Requires p2sd.context_distill.dim > 0 on the model side.
    context_distill_cfg = loss_cfg.get("context_distill", {}) or {}
    context_distill_weight = float(context_distill_cfg.get("weight", 0.0))
    context_distill_pos_samples = int(context_distill_cfg.get("pos_samples", 1024))
    context_distill_min_sheet_voxels = int(context_distill_cfg.get("min_sheet_voxels", 500))
    # GT-EDT latent distance head (user-directed 2026-08-14): supervise a
    # FRESH head on the predicted latent with true EDT at sampled points --
    # not distilled through the AE's weak query head. Reuses the AE's target
    # machinery verbatim (same normalization/clip/sampling conventions).
    # Contrast twin of the context-distill objective (user challenge
    # 2026-08-14: 0040's prototype contrast had the best sheet-MEAN
    # orthogonality ever measured (0.73/0.05), and the modulator READS context
    # by pooling many tokens -- the averaging regime where that geometry wins
    # and per-point noise washes out). Same 64-d head; only the objective
    # differs from context_distill. No ignore mask in P2SD batches, so
    # negatives are other labeled sheets + background only.
    context_contrast_cfg = loss_cfg.get("context_contrast", {}) or {}
    context_contrast_weight = float(context_contrast_cfg.get("weight", 0.0))
    latent_distance_cfg = loss_cfg.get("latent_distance", {}) or {}
    latent_distance_weight = float(latent_distance_cfg.get("weight", 0.0))
    latent_distance_target_cfg = {
        "distance_target_backend": "gpu_cucim",
        "query_distance": {
            "enabled": True,
            "num_points": int(latent_distance_cfg.get("num_points", 4096)),
            "band_voxels": float(latent_distance_cfg.get("band_voxels", 8.0)),
            "sampling": str(latent_distance_cfg.get("sampling", "balanced_regions")),
            "clip_voxels": float(latent_distance_cfg.get("clip_voxels", 16.0)),
            "subvoxel_jitter": bool(latent_distance_cfg.get("subvoxel_jitter", True)),
        },
    }
    latent_distance_beta = float(latent_distance_cfg.get("smooth_l1_beta", 0.1))
    # 0075 (user 2026-09-11): vertex point supervision through the same GT-distance head -- at every mesh grid node of
    # the prompted sheet the signed distance to the band is -band_half (the node lies on the surface, band = EDT <= 1.5),
    # and at node + t * normal it is |t| - band_half; targets normalised like the EDT targets (/clip, clamped).
    vertex_query_cfg = latent_distance_cfg.get("vertex_queries", {}) or {}
    vertex_query_enabled = bool(vertex_query_cfg.get("enabled", False))
    vertex_query_weight = float(vertex_query_cfg.get("weight", 1.0))
    vertex_query_offsets = [float(v) for v in vertex_query_cfg.get("offsets", [0.0, 1.0, 2.0, 4.0, 8.0])]
    vertex_query_max_nodes = int(vertex_query_cfg.get("max_nodes", 1024))
    vertex_query_band_half = float(vertex_query_cfg.get("band_half", 1.5))
    vertex_query_clip = float(latent_distance_cfg.get("clip_voxels", 16.0))
    latent_norm_cfg = loss_cfg.get("latent_normalization", {})
    static_latent_codec = build_static_latent_codec(
        latent_norm_cfg,
        latent_channels=target_ae.latent_channels,
        device=device,
    )
    if best is None:
        best = {"score": -math.inf, "step": 0}
    report_best_path = run_dir / "sheet_union_best.json"
    if (cfg.get("scene_eval", {}).get("report_profile") in {"sheet_union_v1", "sheet_union_v2_border", "sheet_union_v3_box"}
            and not cfg.get("training", {}).get("reset_best_on_resume", False)
            and report_best_path.exists()):
        # Evaluation follows snapshot publication, so a snapshot can contain
        # an older best record. Preserve a completed report across restarts.
        report_best = json.loads(report_best_path.read_text())
        if (report_best.get("report_protocol") == sheet_report_protocol_key(cfg)
                and report_best["step"] <= start_step
                and (run_dir / report_best["checkpoint_path"]).is_file()
                and (best.get("metric") != report_best["metric"]
                     or best.get("report_protocol") != report_best["report_protocol"]
                     or report_best["score"] > best["score"])):
            best = report_best
    write_json(run_dir / "best.json", best)
    write_json(run_dir / "state.json", {
        "status": "running",
        "max_steps": max_steps,
        "epochs": target_epochs,
        "steps_per_epoch": steps_per_epoch,
        "start_step": start_step,
    })
    train_start_time = time.perf_counter()
    epoch_start_time = train_start_time
    epoch_start_step = start_step
    epoch_pair_count = 0
    preceding_checkpoint_time_s = 0.0
    preceding_post_epoch_overhead_s = 0.0
    live_timing_enabled = (
        device.type == "cuda"
        and bool(cfg.get("training", {}).get("live_timing", {}).get("enabled", False))
    )
    live_timing_records: list[
        tuple[
            float,
            float,
            torch.cuda.Event,
            torch.cuda.Event,
            torch.cuda.Event | None,
            torch.cuda.Event | None,
        ]
    ] = []
    last_completed_step = start_step
    stopped_early = False

    def save_current_training_state(step_: int) -> None:
        save_checkpoint(run_dir / "last.pt", model=raw_model, optimizer=optimizer,
                        step=step_, cfg=cfg, best=best)
        filenames = ["last.pt"]
        for active, filename, saver in (
            (dense, "dense_last.pt", save_dense_state),
            (refine, "refine_last.pt", save_refine_state),
            (sdec, "decoder_last.pt", save_sdec_state),
            (brf, "band_last.pt", save_brf_state),
        ):
            if active is not None:
                saver(run_dir)
                filenames.append(filename)
        snapshots = cfg.get("training", {}).get("checkpoint_snapshots", {}) or {}
        if bool(snapshots.get("enabled", False)):
            destination = run_dir / "checkpoints" / f"step_{step_:06d}"
            # A stop immediately after a periodic save already has this step.
            if not destination.exists():
                snapshot_checkpoint_files(
                    run_dir, step=step_, filenames=filenames,
                    keep_last=int(snapshots.get("keep_last", 3)),
                    keep_every_steps=int(snapshots.get("keep_every_steps", 0)),
                )
    log_param_dynamics_enabled = bool(cfg.get("training", {}).get("log_param_dynamics", False))
    loss_plot_interval = int(cfg.get("training", {}).get("loss_plot_interval_epochs", 25))
    param_dynamics_snapshot: dict[str, torch.Tensor] = {}
    if log_param_dynamics_enabled:
        # Snapshot the initial weights so the first epoch's drift is measured
        # from initialization.
        param_dynamics_snapshot = {
            name: p.detach().float().to("cpu").clone()
            for name, p in raw_model.named_parameters()
        }
    step_profiler.start()

    for step in range(start_step + 1, max_steps + 1):
        if should_stop(run_dir):
            stopped_early = True
            save_current_training_state(last_completed_step)
            write_json(run_dir / "state.json", {
                "status": "stop_requested",
                "step": last_completed_step,
            })
            update_heartbeat(run_dir, status="stop_requested", step=last_completed_step)
            break
        model.train()
        batch_fetch_start_time = time.perf_counter()
        batch = next(train_iter)
        batch_fetch_time_s = time.perf_counter() - batch_fetch_start_time
        batch_prepare_start_time = time.perf_counter()
        batch = move_batch(batch, device)
        image = maybe_channels_last_3d(batch["image"].float(), bool(cfg.get("training", {}).get("channels_last_3d", False)))
        gpu_photo_start = None
        gpu_photo_end = None
        if gpu_photometric_generator is not None:
            if live_timing_enabled:
                gpu_photo_start = torch.cuda.Event(enable_timing=True)
                gpu_photo_start.record()
            image = apply_gpu_photometric_augmentations(
                image,
                train_augmentation,
                generator=gpu_photometric_generator,
            )
            if live_timing_enabled:
                gpu_photo_end = torch.cuda.Event(enable_timing=True)
                gpu_photo_end.record()
        pairs = build_p2sd_pair_batch(batch, cfg, device)
        mask = maybe_channels_last_3d(pairs["target_mask"].float(), bool(cfg.get("training", {}).get("channels_last_3d", False)))
        batch_prepare_time_s = time.perf_counter() - batch_prepare_start_time
        pair_count = int(mask.shape[0])
        target_encode_count = pair_count
        target_reuse_factor = 1.0
        if should_save_train_batch_viz(cfg, step, start_step):
            save_train_batch_viz(batch, pairs, image, cfg, run_dir, step)
        optimizer.zero_grad(set_to_none=True)
        step_profiler.begin_step()
        gpu_start = None
        if live_timing_enabled:
            gpu_start = torch.cuda.Event(enable_timing=True)
            gpu_start.record()
        with autocast_context(device, dtype):
            if target_trainable:
                z_gt = target_encode(mask)
            else:
                target_mask = mask
                target_inverse = None
                if deduplicate_frozen_targets:
                    target_mask, target_inverse = unique_p2sd_target_masks(
                        mask,
                        pairs["image_index"],
                        pairs["sheet_id"],
                    )
                    target_encode_count = int(target_mask.shape[0])
                    target_reuse_factor = float(pair_count / max(target_encode_count, 1))
                with torch.inference_mode(), step_profiler.range("target_ae/encode"):
                    z_gt = target_encode(target_mask)
                    if target_inverse is not None:
                        z_gt = z_gt.index_select(0, target_inverse)
                z_gt = z_gt.detach().clone()
            with step_profiler.range("p2sd/forward"):
                # Latent-conditioned training (self-distillation): with prob
                # present_prob the pair also receives the average of the LIVE
                # model's own no-grad single-click latents -- exactly the
                # noisy cluster-centroid distribution the pipeline feeds at
                # inference. Presence is per-step so tensors stay dense; the
                # absent branch keeps the model usable without the token.
                use_latent_prompt = (
                    latent_prompt_mode == "teacher_centroid"
                    and float(torch.rand(())) < latent_prompt_present_prob
                )
                if use_latent_prompt:
                    feat, image_tokens, image_coords, image_context = \
                        model.encode_image_context_from_image(image)
                    teacher_clicks = sample_mask_clicks(
                        mask,
                        int(torch.randint(latent_prompt_teacher_range[0],
                                          latent_prompt_teacher_range[1] + 1, ())),
                    )
                    latent_prompt_tensor = teacher_centroid_latent(
                        model,
                        image_tokens, image_coords, image_context, feat,
                        teacher_clicks,
                        image_index=pairs["image_index"],
                        image_shape=tuple(int(v) for v in image.shape[-3:]),
                    )
                    out = model.forward_from_image_context(
                        image_tokens, image_coords, image_context,
                        pairs["prompt_points"].float(),
                        pairs["prompt_labels"],
                        image_shape=tuple(int(v) for v in image.shape[-3:]),
                        image_index=pairs["image_index"],
                        image_latent=feat,
                        latent_prompt=latent_prompt_tensor,
                    )
                elif refine is not None or sdec is not None or brf is not None:
                    # One encoder pass gives the 1/32 feature for the regressor AND
                    # the pyramid taps for the head/decoder; no_grad only when frozen.
                    if model.freeze_image_encoder:
                        model.image_encoder.eval()
                    with torch.set_grad_enabled(not model.freeze_image_encoder):
                        feat, image_taps = model.image_encoder.encode_with_pyramid(image)
                    image_tokens, image_coords, image_context = model.encode_image_context(feat, image.dtype)
                    out = model.forward_from_image_context(
                        image_tokens, image_coords, image_context,
                        pairs["prompt_points"].float(),
                        pairs["prompt_labels"],
                        image_shape=tuple(int(v) for v in image.shape[-3:]),
                        image_index=pairs["image_index"],
                        image_latent=feat,
                    )
                elif dense is not None:
                    feat, image_tokens, image_coords, image_context = \
                        model.encode_image_context_from_image(image)
                    out = model.forward_from_image_context(
                        image_tokens, image_coords, image_context,
                        pairs["prompt_points"].float(),
                        pairs["prompt_labels"],
                        image_shape=tuple(int(v) for v in image.shape[-3:]),
                        image_index=pairs["image_index"],
                        image_latent=feat,
                    )
                else:
                    out = model(
                        image,
                        pairs["prompt_points"].float(),
                        pairs["prompt_labels"],
                        image_index=pairs["image_index"],
                    )
            dense_loss = torch.zeros((), device=device)
            dense_bce = torch.zeros((), device=device)
            dense_dice = torch.zeros((), device=device)
            dense_stats: dict[str, float] = {}
            if dense is not None:
                from vesuvius_p2sd.train.train_binary_seg import masked_bce_and_dice, sample_context_crop

                with step_profiler.range("dense/forward"):
                    comp = batch["component_label"]
                    if comp.ndim == 5:
                        comp = comp[:, 0]
                    dense_fg = (comp.to(device) > 0).float().unsqueeze(1)
                    if "ignore" in batch:
                        ign = batch["ignore"]
                        if ign.ndim == 5:
                            ign = ign[:, 0]
                        dense_valid = (ign.to(device) == 0).float().unsqueeze(1)
                    else:
                        dense_valid = torch.ones_like(dense_fg)
                    if "dense_weight" in batch:   # 0075 node-weighted band: weight in [0,1] per voxel, 255 = 1
                        dw = batch["dense_weight"]
                        if dw.ndim == 5:
                            dw = dw[:, 0]
                        dense_valid = dense_valid * (dw.to(device).float().unsqueeze(1) / 255.0)
                    ctx = dense.refine_context(image_context)
                    if dense_crop_stage == "second_last":
                        feat = dense.decoder[:-2](ctx)                       # full grid up to the second-to-last stage
                        token_factor = int(dense_fg.shape[-1]) // int(ctx.shape[-1])
                        stage_factor = int(dense_fg.shape[-1]) // int(feat.shape[-1])
                        crop_feat = max(1, dense_crop_grid * token_factor // stage_factor)
                        feat_crop, fg_crop, valid_crop = sample_context_crop(
                            feat, dense_fg, dense_valid, crop_grid=crop_feat)
                        dense_logits = dense.head(dense.decoder[-2:](feat_crop))
                    else:
                        ctx_crop, fg_crop, valid_crop = sample_context_crop(
                            ctx, dense_fg, dense_valid, crop_grid=dense_crop_grid)
                        dense_logits = dense.decode_context(ctx_crop)
                    dense_bce, dense_dice, dense_stats = masked_bce_and_dice(
                        dense_logits, fg_crop, valid_crop, pos_weight=dense_pos_weight)
                    dense_loss = dense_bce_w * dense_bce + dense_dice_w * dense_dice
            z_pred = out["latent"]
            if z_pred.shape != z_gt.shape:
                raise RuntimeError(
                    "P2SD latent shape mismatch before loss: "
                    f"z_pred={tuple(z_pred.shape)} z_gt={tuple(z_gt.shape)}"
                )
            latent_codec, latent_norm_log = build_latent_loss_codec(
                z_gt,
                latent_norm_cfg,
                static_codec=static_latent_codec,
            )
            z_pred_raw = latent_codec.raw_prediction(z_pred)
            z_gt_model = latent_codec.model_target(z_gt)
            raw_latent_mse = F.mse_loss(z_pred_raw.float(), z_gt.float())
            if latent_codec.predict_normalized:
                latent_mse_loss = F.mse_loss(z_pred.float(), z_gt_model.float())
                aux_loss = aux_mse_loss(
                    out["aux_latents"],
                    z_gt_model,
                    aux_weights,
                    mode=aux_weights_mode,
                )
            else:
                latent_mse_loss = F.mse_loss(latent_codec.normalize(z_pred), latent_codec.normalize(z_gt))
                aux_loss = aux_mse_loss(
                    out["aux_latents"],
                    z_gt,
                    aux_weights,
                    normalizer=latent_codec.normalize,
                    mode=aux_weights_mode,
                )
            final_mse_scale = per_round_final_mse_scale(
                aux_weights,
                mode=aux_weights_mode,
            )
            consistency_loss, consistency_groups = same_sheet_prompt_consistency_loss(
                z_pred,
                pairs["image_index"],
                pairs["sheet_id"],
                weight=consistency_weight,
            )
            contrast_loss, contrast_pairs, contrast_distance = different_sheet_prompt_contrast_loss(
                z_pred,
                pairs["image_index"],
                pairs["sheet_id"],
                weight=contrast_weight,
                margin=contrast_margin,
            )
            coordinate_query_loss = z_pred.new_zeros(())
            coordinate_query_teacher = None
            coordinate_query_prediction = None
            if coordinate_query_enabled:
                (
                    coordinate_query_loss,
                    coordinate_query_teacher,
                    coordinate_query_prediction,
                ) = coordinate_query_distillation_loss(
                    target_ae,
                    z_pred_raw,
                    z_gt,
                    image_shape=tuple(int(value) for value in mask.shape[-3:]),
                    num_points=coordinate_query_num_points,
                    smooth_l1_beta=coordinate_query_beta,
                    target_mask=mask,
                    image_index=pairs["image_index"],
                    sheet_id=pairs["sheet_id"],
                    sampling=coordinate_query_sampling,
                    band_voxels=coordinate_query_band_voxels,
                    inside_fraction=coordinate_query_inside_fraction,
                    near_outside_fraction=coordinate_query_near_outside_fraction,
                )
            context_distill_loss = z_pred.new_zeros(())
            context_distill_stats: dict[str, float] = {}
            if context_distill_weight > 0 or context_contrast_weight > 0:
                if getattr(model, "context_distill_head", None) is None:
                    raise ValueError(
                        "context distill/contrast weight > 0 needs p2sd.context_distill.dim > 0")
                component_grid = batch["component_label"].to(device, non_blocking=True).long()
                if component_grid.ndim == 5 and component_grid.shape[1] == 1:
                    component_grid = component_grid[:, 0]
                ctx_embedding = model.context_distill_head(out["image_context"])
            if context_distill_weight > 0:
                anchors = sheet_latent_anchors(
                    target_ae, component_grid,
                    min_sheet_voxels=context_distill_min_sheet_voxels,
                    anchor_dim=int(model.context_distill_head.out_channels))
                context_distill_loss, context_distill_stats = voxel_latent_distill_loss(
                    ctx_embedding, component_grid, anchors,
                    pos_samples=context_distill_pos_samples)
            context_contrast_loss = z_pred.new_zeros(())
            context_contrast_stats: dict[str, float] = {}
            if context_contrast_weight > 0:
                context_contrast_loss, context_contrast_stats = voxel_prototype_contrast_loss(
                    ctx_embedding, component_grid,
                    torch.zeros_like(component_grid),
                    temperature=float(context_contrast_cfg.get("temperature", 0.1)),
                    pos_samples=int(context_contrast_cfg.get("pos_samples", 1024)),
                    neg_samples=int(context_contrast_cfg.get("neg_samples", 4096)),
                    center_samples=int(context_contrast_cfg.get("center_samples", 2048)),
                    min_sheet_voxels=int(context_contrast_cfg.get("min_sheet_voxels", 500)),
                    border_width=int(context_contrast_cfg.get("border_width", 5)))
            latent_distance_loss = z_pred.new_zeros(())
            if latent_distance_weight > 0:
                if getattr(model, "latent_distance_head", None) is None:
                    raise ValueError(
                        "p2sd.loss.latent_distance.weight > 0 needs p2sd.latent_distance.hidden_dim > 0")
                gt_targets = prepare_ae_distance_targets(
                    {"mask": mask}, latent_distance_target_cfg)
                predicted = model.query_latent_distance(
                    z_pred_raw, gt_targets["query_points"],
                    tuple(int(v) for v in mask.shape[-3:]))
                latent_distance_loss = F.smooth_l1_loss(
                    predicted.float(), gt_targets["query_distances"].float(),
                    beta=latent_distance_beta)
            vertex_query_loss = z_pred.new_zeros(()); vertex_query_count = 0
            if latent_distance_weight > 0 and vertex_query_enabled and "vertices_zyx" in batch:
                vq_points, vq_targets, vq_valid = build_vertex_queries(
                    batch, pairs["image_index"], pairs["sheet_id"], device,
                    offsets=vertex_query_offsets, max_nodes=vertex_query_max_nodes,
                    band_half=vertex_query_band_half, clip_voxels=vertex_query_clip)
                vertex_query_count = int(vq_valid.sum())
                if vertex_query_count > 0:
                    vq_pred = model.query_latent_distance(
                        z_pred_raw, vq_points, tuple(int(v) for v in mask.shape[-3:]))
                    vertex_query_loss = F.smooth_l1_loss(
                        vq_pred.float()[vq_valid], vq_targets.float()[vq_valid], beta=latent_distance_beta)
                    latent_distance_loss = latent_distance_loss + vertex_query_weight * vertex_query_loss
            loss = (
                final_mse_scale * latent_mse_loss
                + aux_loss
                + consistency_weight * consistency_loss
                + contrast_weight * contrast_loss
                + coordinate_query_weight * coordinate_query_loss
                + context_distill_weight * context_distill_loss
                + context_contrast_weight * context_contrast_loss
                + latent_distance_weight * latent_distance_loss
                + dense_weight * dense_loss
            )
            decoded_bce = torch.zeros((), device=device)
            apply_decoded_bce = should_apply_decoded_bce(
                step,
                weight=decoded_weight,
                interval=decoded_interval,
                warmup_steps=decoded_warmup,
            )
            if apply_decoded_bce and decoded_stages:
                from vesuvius_p2sd.train.train_ae import deep_supervision_loss_from_logits

                aux_logits = target_ae.decode_aux_until(z_pred_raw, decoded_stages)
                if decoded_target_mode == "ae_soft":
                    with torch.no_grad():
                        aux_teacher = target_ae.decode_aux_until(z_gt, decoded_stages)
                    decoded_bce = torch.zeros((), device=device)
                    for w_k, a_k, t_k in zip(decoded_stage_weights, aux_logits, aux_teacher, strict=True):
                        t_prob = torch.sigmoid(t_k.float())
                        # soft-target BCE minus the teacher's own entropy -> 0 at a perfect match
                        term = F.binary_cross_entropy_with_logits(a_k.float(), t_prob) \
                            - F.binary_cross_entropy_with_logits(t_k.float(), t_prob)
                        decoded_bce = decoded_bce + float(w_k) * term
                else:
                    decoded_bce = deep_supervision_loss_from_logits(
                        [a.float() for a in aux_logits],
                        mask.float(),
                        weights=decoded_stage_weights,
                        bce_weight=decoded_bce_term_weight,
                        dice_weight=decoded_dice_weight,
                        pos_weight=decoded_pos_weight,
                        max_auto_pos_weight=decoded_max_auto_pos_weight,
                    )
            elif apply_decoded_bce and decoded_target_mode == "crop_gt":
                from vesuvius_p2sd.train.train_binary_seg import masked_bce_and_dice

                grid = int(z_pred_raw.shape[-1])
                factor = int(mask.shape[-1]) // grid
                g = min(decoded_crop_grid, grid)
                fg_any = (mask.detach().sum(dim=0, keepdim=True) > 0)
                if "ignore" in batch:
                    ign_full = batch["ignore"]
                    if ign_full.ndim == 5:
                        ign_full = ign_full[:, 0]
                    valid_full = (ign_full.to(device) == 0).unsqueeze(1)
                else:
                    valid_full = torch.ones_like(fg_any)
                # Same selection rule as sample_context_crop: up to 4 draws,
                # keep the first crop containing labeled sheet voxels.
                span = grid - g + 1
                offsets = None
                for _ in range(4):
                    cand = [int(v) for v in torch.randint(0, span, (3,)).tolist()]
                    sl = tuple(slice(o * factor, (o + g) * factor) for o in cand)
                    offsets = cand
                    if bool((fg_any[(..., *sl)] & valid_full[(..., *sl)]).any()):
                        break
                gz, gy, gx = offsets
                sl = (slice(gz * factor, (gz + g) * factor), slice(gy * factor, (gy + g) * factor),
                      slice(gx * factor, (gx + g) * factor))
                z_crop = z_pred_raw[..., gz:gz + g, gy:gy + g, gx:gx + g]
                crop_logits = target_ae.decode(z_crop)
                mask_crop = mask[(..., *sl)].float()
                valid_crop = valid_full[(..., *sl)].float().expand_as(mask_crop)
                crop_pos_weight = float(decoded_pos_weight) if not isinstance(decoded_pos_weight, str) else 2.0
                crop_bce, crop_dice, _ = masked_bce_and_dice(
                    crop_logits.float(), mask_crop, valid_crop, pos_weight=crop_pos_weight)
                decoded_bce = decoded_bce_term_weight * crop_bce + decoded_dice_weight * crop_dice
            elif apply_decoded_bce:
                decoded_logits = target_ae.decode(z_pred_raw)
                decoded_bce = decoded_mask_loss(
                    decoded_logits.float(),
                    mask.float(),
                    pos_weight=decoded_pos_weight,
                    max_auto_pos_weight=decoded_max_auto_pos_weight,
                )
            loss = loss + decoded_weight * decoded_bce
            geo_loss = torch.zeros((), device=device)
            geo_dice = torch.zeros((), device=device)
            if geo_weight > 0 and step % geo_interval == 0 and dense is not None:
                from vesuvius_p2sd.research import tta as _tta

                with step_profiler.range("geo_consistency"):
                    t = _tta.transforms(geo_transforms)[int(torch.randint(1, geo_transforms, ()))]   # never the identity
                    full_shape = tuple(int(v) for v in image.shape[-3:])
                    image_t = _tta.apply(image, t)
                    pts = pairs["prompt_points"].detach().cpu().numpy()
                    pts_t = np.stack([_tta.apply_points(np.rint(p).astype(np.int64), t, full_shape) for p in pts])
                    pts_t = torch.from_numpy(pts_t.astype(np.float32)).to(device)
                    with torch.set_grad_enabled(geo_grad == "both"):
                        feat_t, tokens_t, coords_t, context_t = model.encode_image_context_from_image(image_t)
                        out_t = model.forward_from_image_context(
                            tokens_t, coords_t, context_t, pts_t, pairs["prompt_labels"],
                            image_shape=_tta.out_shape(full_shape, t), image_index=pairs["image_index"], image_latent=feat_t)
                        z_t_raw = latent_codec.raw_prediction(out_t["latent"])
                    grid = int(z_pred_raw.shape[-1]); factor = int(mask.shape[-1]) // grid; g = min(geo_crop_grid, grid)
                    fg_any = (mask.detach().sum(dim=0, keepdim=True) > 0)
                    span = grid - g + 1; offsets = None
                    for _ in range(4):
                        cand = [int(v) for v in torch.randint(0, span, (3,)).tolist()]
                        sl = tuple(slice(o * factor, (o + g) * factor) for o in cand); offsets = cand
                        if bool(fg_any[(..., *sl)].any()):
                            break
                    gz, gy, gx = offsets
                    corners = np.array([[gz, gy, gx], [gz + g - 1, gy + g - 1, gx + g - 1]])
                    tc = _tta.apply_points(corners, t, (grid, grid, grid)).min(axis=0)
                    z_crop = z_pred_raw[..., gz:gz + g, gy:gy + g, gx:gx + g]
                    z_t_crop = z_t_raw[..., tc[0]:tc[0] + g, tc[1]:tc[1] + g, tc[2]:tc[2] + g]
                    logits_a = target_ae.decode(z_crop).float()
                    with torch.set_grad_enabled(geo_grad == "both"):
                        logits_b = _tta.invert(target_ae.decode(z_t_crop), t).float()
                    if geo_grad != "both":
                        logits_b = logits_b.detach()
                    pa = torch.sigmoid(logits_a); pb = torch.sigmoid(logits_b)
                    inter = (pa * pb).sum(dim=(1, 2, 3, 4)); denom = (pa * pa).sum(dim=(1, 2, 3, 4)) + (pb * pb).sum(dim=(1, 2, 3, 4))
                    geo_loss = (1.0 - (2.0 * inter + 1.0) / (denom + 1.0)).mean()
                    with torch.no_grad():
                        ma = pa > 0.5; mb = pb > 0.5
                        geo_dice = (2.0 * (ma & mb).sum(dim=(1, 2, 3, 4)).float() / ((ma.sum(dim=(1, 2, 3, 4)) + mb.sum(dim=(1, 2, 3, 4))).float() + 1.0)).mean()
                    loss = loss + geo_weight * geo_loss
            refine_loss = torch.zeros((), device=device)
            refine_bce = torch.zeros((), device=device)
            refine_dice = torch.zeros((), device=device)
            refine_base_bce = torch.zeros((), device=device)
            refine_base_dice = torch.zeros((), device=device)
            refine_volume_ratio = torch.ones((), device=device)
            refine_tiles_n = 0.0
            if refine is not None:
                from vesuvius_p2sd.train.train_binary_seg import masked_bce_and_dice

                with step_profiler.range("refine/forward"):
                    n_pairs = int(z_pred_raw.shape[0])
                    k = min(refine_max_sheets, n_pairs)
                    idx = (torch.randperm(n_pairs, device=z_pred_raw.device)[:k] if k < n_pairs
                           else torch.arange(n_pairs, device=z_pred_raw.device))
                    img_idx = pairs["image_index"].to(z_pred_raw.device).long()[idx]
                    need_teacher = refine_target_mode in ("ae_gt", "ae_gt_hard")
                    if "ignore" in batch:
                        ign = batch["ignore"]
                        if ign.ndim == 5:
                            ign = ign[:, 0]
                        valid_vol = (ign.to(device) == 0).unsqueeze(1).float()[img_idx]
                    else:
                        valid_vol = torch.ones((int(idx.shape[0]), 1) + tuple(mask.shape[-3:]), device=device)
                    if refine_sparse:
                        # 0064: dense AE decodes (exact; the AE's GroupNorm forbids tiling it), then
                        # the head runs only on the tiles where the base decode, the teacher or the
                        # target is active (SheetAE tile helpers); the loss is taken on the tiles'
                        # exact 32^3 centres.
                        tile, hq = target_ae.SPARSE_TILE, target_ae.SPARSE_HALO4
                        tile_c = slice(hq * 4, hq * 4 + tile)
                        with torch.no_grad():
                            base_full, taps_full = target_ae.decode_with_taps(z_pred_raw[idx].detach(), (4, 2, 1))
                            active = base_full > refine_tile_threshold
                            teacher_full = None
                            if need_teacher:
                                teacher_full = target_ae.decode(z_gt[idx].detach())
                                active = active | (teacher_full > refine_tile_threshold)
                            active = active | (mask[idx] > 0)
                            active4 = F.max_pool3d(active.float(), kernel_size=4, stride=4) > 0
                            tiles = target_ae.active_tiles(active4, tile4=tile // 4, dilate=1)
                            c1 = int(taps_full[1].shape[1])
                            stack1 = [taps_full[1], base_full.to(taps_full[1].dtype), image[img_idx].to(taps_full[1].dtype),
                                      mask[idx].to(taps_full[1].dtype), valid_vol.to(taps_full[1].dtype)]
                            if need_teacher:
                                stack1.append(teacher_full.to(taps_full[1].dtype))
                            full = {4: taps_full[4], 2: taps_full[2], 1: torch.cat(stack1, dim=1)}
                            t, valids_t, tidx = target_ae.gather_tile_set(tiles, full)
                            img4_t, _ = target_ae.gather_tiles(image_taps[4][img_idx], tiles, tile // 4, hq)
                        ae_taps_t = {4: t[4], 2: t[2], 1: t[1][:, :c1]}
                        base_t = t[1][:, c1:c1 + 1]
                        image_t = t[1][:, c1 + 1:c1 + 2]
                        refined_full = refine(base_t, ae_taps_t, {4: img4_t * valids_t[4]}, image_t, valid=valids_t)
                        refined_logits = refined_full[..., tile_c, tile_c, tile_c].float()
                        base_logits = base_t[..., tile_c, tile_c, tile_c].float()
                        mask_sel = t[1][:, c1 + 2:c1 + 3][..., tile_c, tile_c, tile_c].float()
                        refine_valid = t[1][:, c1 + 3:c1 + 4][..., tile_c, tile_c, tile_c].float()
                        teacher_logits = t[1][:, c1 + 4:c1 + 5][..., tile_c, tile_c, tile_c].float() if need_teacher else None
                        refine_tiles_n = float(tidx.shape[0])
                    else:
                        with torch.no_grad():
                            base_logits, ae_taps = target_ae.decode_with_taps(
                                z_pred_raw[idx].detach(), refine.ae_factors)
                            teacher_logits = target_ae.decode(z_gt[idx].detach()) if need_teacher else None
                        img_taps_sel = {f: image_taps[f][img_idx] for f in refine.image_factors}
                        refined_logits = refine(base_logits, ae_taps, img_taps_sel, image[img_idx])
                        mask_sel = mask[idx].float()
                        refine_valid = valid_vol
                        refine_tiles_n = 0.0
                    if need_teacher:
                        # 0062: distill toward the AE's own decode of the GT latent (thin, b1~0, same
                        # thickness convention as the base decode) — the head can only gain by
                        # DISPLACING the surface. ae_gt_hard (0063): binary teacher (no hedge halo).
                        refine_target = ((teacher_logits.float() > 0).float() if refine_target_mode == "ae_gt_hard"
                                         else torch.sigmoid(teacher_logits.float()))
                    else:
                        refine_target = mask_sel
                    refine_valid = refine_valid.expand_as(refine_target)
                    refine_weight_map = refine_valid
                    if refine_band_weight > 0:
                        # 0063: extra weight on the 2-6 voxel band around the
                        # teacher surface AND around the base surface — the
                        # tau=2 deficit lives there (surface anatomy 08-27),
                        # while the per-voxel loss mass sits in the +-1 band.
                        with torch.no_grad():
                            def _dil(m, r):
                                return F.max_pool3d(m, kernel_size=2 * r + 1, stride=1, padding=r)
                            t_hard = (refine_target > 0.5).float()
                            b_hard = (base_logits > 0).float()
                            band = ((_dil(t_hard, refine_band_outer) - _dil(t_hard, refine_band_inner))
                                    + (_dil(b_hard, refine_band_outer) - _dil(b_hard, refine_band_inner))).clamp_(0.0, 1.0)
                            refine_weight_map = refine_valid * (1.0 + refine_band_weight * band)
                    refine_bce, refine_dice, _ = masked_bce_and_dice(
                        refined_logits.float(), refine_target, refine_weight_map, pos_weight=refine_pos_weight)
                    with torch.no_grad():
                        refine_base_bce, refine_base_dice, _ = masked_bce_and_dice(
                            base_logits.float(), refine_target, refine_weight_map, pos_weight=refine_pos_weight)
                        base_vol = ((base_logits > 0).float() * refine_valid).sum().clamp_min(1.0)
                        refine_volume_ratio = ((refined_logits.detach() > 0).float() * refine_valid).sum() / base_vol
                    refine_loss = refine_bce_w * refine_bce + refine_dice_w * refine_dice
                    if refine_volume_weight > 0:
                        # soft volume-preservation: relative excess/deficit of probability mass vs the base decode
                        base_mass = (torch.sigmoid(base_logits.float()) * refine_valid).sum().clamp_min(1.0)
                        ref_mass = (torch.sigmoid(refined_logits.float()) * refine_valid).sum()
                        refine_loss = refine_loss + refine_volume_weight * (ref_mass / base_mass - 1.0).abs()
                loss = loss + refine_weight * refine_loss
            sdec_loss = torch.zeros((), device=device)
            sdec_stats: dict[str, float] = {}
            if sdec is not None:
                from vesuvius_p2sd.train.sheet_decoder_step import sheet_decoder_step

                with step_profiler.range("sheet_decoder/forward"):
                    sdec_loss, sdec_stats = sheet_decoder_step(
                        sdec, target_ae, image=image, image_taps=image_taps,
                        z_pred_raw=(z_pred_raw.detach() if sdec_detach_latent else z_pred_raw),
                        z_gt=z_gt, mask=mask, batch=batch, image_index=pairs["image_index"], device=device,
                        max_sheets=sdec_max_sheets, gt_mix=sdec_gt_mix, pos_weight=sdec_pos_weight,
                        bce_weight=sdec_bce_w, dice_weight=sdec_dice_w, aux_weight=sdec_aux_w,
                        aux_pos_weight=sdec_aux_pos_weight, gate_logit=sdec_gate_logit, max_extra_tiles=sdec_max_extra)
                loss = loss + sdec_weight * sdec_loss
            udec_loss = torch.zeros((), device=device)
            udec_stats: dict[str, float] = {}
            if udec is not None:
                from vesuvius_p2sd.train.sheet_decoder_step import union_decoder_step

                with step_profiler.range("union_decoder/forward"):
                    udec_loss, udec_stats = union_decoder_step(
                        udec, target_ae, image=image, image_taps=image_taps, context=image_context, batch=batch,
                        device=device, pos_weight=udec_pos_weight, bce_weight=udec_bce_w, dice_weight=udec_dice_w,
                        aux_weight=udec_aux_w, aux_pos_weight=udec_aux_pos_weight, gate_logit=udec_gate_logit,
                        max_tiles_per_image=udec_max_tiles)
                loss = loss + udec_weight * udec_loss
            brf_loss = torch.zeros((), device=device)
            brf_stats: dict[str, float] = {}
            if brf is not None:
                from vesuvius_p2sd.train.band_refiner_step import band_refiner_step

                with step_profiler.range("band_refiner/forward"):
                    brf_loss, brf_stats = band_refiner_step(
                        brf, brf_dec, target_ae, image=image, image_taps=image_taps, z_pred_raw=z_pred_raw, z_gt=z_gt,
                        mask=mask, batch=batch, image_index=pairs["image_index"], device=device,
                        max_sheets=brf_max_sheets, gt_mix=brf_gt_mix, pos_weight=brf_pos_weight, bce_weight=brf_bce_w,
                        dice_weight=brf_dice_w, patch=brf_patch, max_train_tokens=brf_max_tokens, radius=brf_radius,
                        gate_logit=brf_gate_logit, gate_dilate=brf_gate_dilate, binseg_model=brf_binseg, **brf_rank_kw)
                loss = loss + brf_weight * brf_loss
        if not bool(torch.isfinite(loss.detach()).all().item()):
            row = {
                "status": "failed",
                "reason": "nonfinite_loss",
                "step": step,
                "loss": float(loss.detach().float().cpu()),
                "latent_mse": float(raw_latent_mse.detach().float().cpu()),
                "latent_mse_loss": float(latent_mse_loss.detach().float().cpu()),
                "aux_mse": float(aux_loss.detach().float().cpu()),
                "same_sheet_prompt_consistency": float(consistency_loss.detach().float().cpu()),
                "different_sheet_prompt_contrast": float(contrast_loss.detach().float().cpu()),
                "coordinate_query_loss": float(coordinate_query_loss.detach().float().cpu()),
                "decoded_bce": float(decoded_bce.detach().float().cpu()),
                "geo_loss": float(geo_loss.detach().float().cpu()),
                "geo_dice": float(geo_dice.detach().float().cpu()),
                "dense_loss": float(dense_loss.detach().float().cpu()),
                "dense_bce": float(dense_bce.detach().float().cpu()),
                "dense_dice": float(dense_dice.detach().float().cpu()),
                "dense_fg_fraction": float(dense_stats.get("foreground_fraction", 0.0)),
                "refine_loss": float(refine_loss.detach().float().cpu()),
                "refine_bce": float(refine_bce.detach().float().cpu()),
                "refine_dice": float(refine_dice.detach().float().cpu()),
                "refine_base_bce": float(refine_base_bce.detach().float().cpu()),
                "refine_base_dice": float(refine_base_dice.detach().float().cpu()),
                "refine_volume_ratio": float(refine_volume_ratio.detach().float().cpu()),
                "refine_tiles": float(refine_tiles_n),
                "sdec_loss": float(sdec_loss.detach().float().cpu()),
                "sdec_bce": float(sdec_stats.get("bce", 0.0)),
                "sdec_dice": float(sdec_stats.get("dice", 0.0)),
                "sdec_aux_bce": float(sdec_stats.get("aux_bce", 0.0)),
                "sdec_tiles": float(sdec_stats.get("tiles", 0.0)),
                "sdec_gate_recall": float(sdec_stats.get("gate_recall", 0.0)),
                "sdec_volume_ratio": float(sdec_stats.get("volume_ratio", 0.0)),
                "udec_loss": float(udec_loss.detach().float().cpu()),
                "udec_bce": float(udec_stats.get("bce", 0.0)),
                "udec_dice": float(udec_stats.get("dice", 0.0)),
                "udec_aux_bce": float(udec_stats.get("aux_bce", 0.0)),
                "udec_tiles": float(udec_stats.get("tiles", 0.0)),
                "udec_gate_recall": float(udec_stats.get("gate_recall", 0.0)),
                "udec_volume_ratio": float(udec_stats.get("volume_ratio", 0.0)),
                "brf_loss": float(brf_loss.detach().float().cpu()),
                "brf_bce": float(brf_stats.get("bce", 0.0)),
                "brf_dice": float(brf_stats.get("dice", 0.0)),
                "brf_base_bce": float(brf_stats.get("base_bce", 0.0)),
                "brf_base_dice": float(brf_stats.get("base_dice", 0.0)),
                "brf_tokens": float(brf_stats.get("tokens", 0.0)),
                "brf_recall": float(brf_stats.get("recall", 0.0)),
                "brf_precision": float(brf_stats.get("precision", 0.0)),
                "brf_base_recall": float(brf_stats.get("base_recall", 0.0)),
                "brf_base_precision": float(brf_stats.get("base_precision", 0.0)),
                "brf_rank": float(brf_stats.get("rank", 0.0)),
                "brf_base_rank": float(brf_stats.get("base_rank", 0.0)),
                "brf_rank_acc": float(brf_stats.get("rank_acc", 0.0)),
                "brf_base_rank_acc": float(brf_stats.get("base_rank_acc", 0.0)),
                "brf_pairs": float(brf_stats.get("pairs", 0.0)),
                "brf_local_pairs": float(brf_stats.get("local_pairs", 0.0)),
                "brf_rank_local": float(brf_stats.get("rank_local", 0.0)),
                "brf_base_rank_local": float(brf_stats.get("base_rank_local", 0.0)),
                "brf_rank_acc_local": float(brf_stats.get("rank_acc_local", 0.0)),
                "brf_base_rank_acc_local": float(brf_stats.get("base_rank_acc_local", 0.0)),
            }
            append_jsonl(run_dir / "metrics.jsonl", row)
            write_json(run_dir / "state.json", row)
            update_heartbeat(run_dir, status="failed", step=step, loss=row["loss"])
            raise RuntimeError(f"P2SD loss became non-finite at step {step}: {row}")
        with step_profiler.range("p2sd/backward"):
            loss.backward()
        # Inspect before clipping: a single NaN gradient would otherwise cause
        # clip_grad_norm_ to overwrite every finite gradient with NaN.
        nonfinite_gradients = collect_nonfinite_gradient_details(raw_model)
        if nonfinite_gradients["parameter_count"]:
            row = {
                "status": "failed",
                "reason": "nonfinite_gradients",
                "step": step,
                "loss": float(loss.detach().float().cpu()),
                "latent_mse": float(raw_latent_mse.detach().float().cpu()),
                "latent_mse_loss": float(latent_mse_loss.detach().float().cpu()),
                "aux_mse": float(aux_loss.detach().float().cpu()),
                "same_sheet_prompt_consistency": float(consistency_loss.detach().float().cpu()),
                "different_sheet_prompt_contrast": float(contrast_loss.detach().float().cpu()),
                "coordinate_query_loss": float(coordinate_query_loss.detach().float().cpu()),
                "decoded_bce": float(decoded_bce.detach().float().cpu()),
                "geo_loss": float(geo_loss.detach().float().cpu()),
                "geo_dice": float(geo_dice.detach().float().cpu()),
                "dense_loss": float(dense_loss.detach().float().cpu()),
                "dense_bce": float(dense_bce.detach().float().cpu()),
                "dense_dice": float(dense_dice.detach().float().cpu()),
                "dense_fg_fraction": float(dense_stats.get("foreground_fraction", 0.0)),
                "refine_loss": float(refine_loss.detach().float().cpu()),
                "refine_bce": float(refine_bce.detach().float().cpu()),
                "refine_dice": float(refine_dice.detach().float().cpu()),
                "refine_base_bce": float(refine_base_bce.detach().float().cpu()),
                "refine_base_dice": float(refine_base_dice.detach().float().cpu()),
                "refine_volume_ratio": float(refine_volume_ratio.detach().float().cpu()),
                "refine_tiles": float(refine_tiles_n),
                "sdec_loss": float(sdec_loss.detach().float().cpu()),
                "sdec_bce": float(sdec_stats.get("bce", 0.0)),
                "sdec_dice": float(sdec_stats.get("dice", 0.0)),
                "sdec_aux_bce": float(sdec_stats.get("aux_bce", 0.0)),
                "sdec_tiles": float(sdec_stats.get("tiles", 0.0)),
                "sdec_gate_recall": float(sdec_stats.get("gate_recall", 0.0)),
                "sdec_volume_ratio": float(sdec_stats.get("volume_ratio", 0.0)),
                "udec_loss": float(udec_loss.detach().float().cpu()),
                "udec_bce": float(udec_stats.get("bce", 0.0)),
                "udec_dice": float(udec_stats.get("dice", 0.0)),
                "udec_aux_bce": float(udec_stats.get("aux_bce", 0.0)),
                "udec_tiles": float(udec_stats.get("tiles", 0.0)),
                "udec_gate_recall": float(udec_stats.get("gate_recall", 0.0)),
                "udec_volume_ratio": float(udec_stats.get("volume_ratio", 0.0)),
                "brf_loss": float(brf_loss.detach().float().cpu()),
                "brf_bce": float(brf_stats.get("bce", 0.0)),
                "brf_dice": float(brf_stats.get("dice", 0.0)),
                "brf_base_bce": float(brf_stats.get("base_bce", 0.0)),
                "brf_base_dice": float(brf_stats.get("base_dice", 0.0)),
                "brf_tokens": float(brf_stats.get("tokens", 0.0)),
                "brf_recall": float(brf_stats.get("recall", 0.0)),
                "brf_precision": float(brf_stats.get("precision", 0.0)),
                "brf_base_recall": float(brf_stats.get("base_recall", 0.0)),
                "brf_base_precision": float(brf_stats.get("base_precision", 0.0)),
                "brf_rank": float(brf_stats.get("rank", 0.0)),
                "brf_base_rank": float(brf_stats.get("base_rank", 0.0)),
                "brf_rank_acc": float(brf_stats.get("rank_acc", 0.0)),
                "brf_base_rank_acc": float(brf_stats.get("base_rank_acc", 0.0)),
                "brf_pairs": float(brf_stats.get("pairs", 0.0)),
                "brf_local_pairs": float(brf_stats.get("local_pairs", 0.0)),
                "brf_rank_local": float(brf_stats.get("rank_local", 0.0)),
                "brf_base_rank_local": float(brf_stats.get("base_rank_local", 0.0)),
                "brf_rank_acc_local": float(brf_stats.get("rank_acc_local", 0.0)),
                "brf_base_rank_acc_local": float(brf_stats.get("base_rank_acc_local", 0.0)),
                "nonfinite_gradient_parameter_count": nonfinite_gradients["parameter_count"],
            }
            append_jsonl(run_dir / "metrics.jsonl", row)
            write_json(run_dir / "state.json", row)
            write_json(run_dir / "nonfinite_gradients.json", {
                "step": step,
                "loss_terms": {
                    "loss": row["loss"],
                    "latent_mse": row["latent_mse"],
                    "latent_mse_loss": row["latent_mse_loss"],
                    "aux_mse": row["aux_mse"],
                    "same_sheet_prompt_consistency": row["same_sheet_prompt_consistency"],
                    "decoded_bce": row["decoded_bce"],
                    "geo_loss": row.get("geo_loss", 0.0),
                    "geo_dice": row.get("geo_dice", 0.0),
                },
                "gradients": nonfinite_gradients,
            })
            save_checkpoint(
                run_dir / "last_finite_before_failure.pt",
                model=raw_model,
                optimizer=optimizer,
                step=last_completed_step,
                cfg=cfg,
                best=best,
            )
            update_heartbeat(run_dir, status="failed", step=step, loss=row["loss"])
            raise RuntimeError(
                f"P2SD gradients became non-finite at step {step}: {row}; "
                f"details={run_dir / 'nonfinite_gradients.json'}"
            )
        grad_clip = cfg.get("training", {}).get("grad_clip")
        grad_norm = None
        if grad_clip is not None:
            grad_norm = torch.nn.utils.clip_grad_norm_(raw_model.parameters(), float(grad_clip))
            if not bool(torch.isfinite(grad_norm.detach()).all().item()):
                raise RuntimeError(
                    "clip_grad_norm_ produced a non-finite norm after finite per-parameter gradients"
                )
        current_lr = lr_schedule.apply(optimizer, step)
        with step_profiler.range("p2sd/optimizer"):
            optimizer.step()
        if gpu_start is not None:
            gpu_end = torch.cuda.Event(enable_timing=True)
            gpu_end.record()
            live_timing_records.append(
                (batch_fetch_time_s, batch_prepare_time_s, gpu_start, gpu_end, gpu_photo_start, gpu_photo_end)
            )
        last_completed_step = step
        epoch_pair_count += pair_count
        step_profiler.end_step(pair_count, int(image.shape[0]))
        log_window.track("grad_norm", grad_norm)
        log_window.add({
            "loss": loss,
            "latent_mse": raw_latent_mse,
            "latent_mse_loss": latent_mse_loss,
            "aux_mse": aux_loss,
            "same_sheet_prompt_consistency": consistency_loss,
            "different_sheet_prompt_contrast": contrast_loss,
            "different_sheet_prompt_contrast_distance": contrast_distance,
            # MetricWindow accepts tensors only (it calls .detach()).
            "different_sheet_prompt_contrast_pairs": contrast_loss.new_tensor(float(contrast_pairs)),
            "coordinate_query_loss": coordinate_query_loss,
            "decoded_bce": decoded_bce,
            "geo_loss": geo_loss,
            "geo_dice": geo_dice,
            "dense_loss": dense_loss,
            "dense_bce": dense_bce,
            "dense_dice": dense_dice,
            "dense_fg_fraction": dense_loss.new_tensor(float(dense_stats.get("foreground_fraction", 0.0))),
            "refine_loss": refine_loss,
            "refine_bce": refine_bce,
            "refine_dice": refine_dice,
            "refine_base_bce": refine_base_bce,
            "refine_base_dice": refine_base_dice,
            "refine_volume_ratio": refine_volume_ratio,
            "refine_tiles": refine_volume_ratio.new_tensor(float(refine_tiles_n)),
            "sdec_loss": sdec_loss,
            "sdec_bce": sdec_loss.new_tensor(float(sdec_stats.get("bce", 0.0))),
            "sdec_dice": sdec_loss.new_tensor(float(sdec_stats.get("dice", 0.0))),
            "sdec_aux_bce": sdec_loss.new_tensor(float(sdec_stats.get("aux_bce", 0.0))),
            "sdec_tiles": sdec_loss.new_tensor(float(sdec_stats.get("tiles", 0.0))),
            "sdec_gate_recall": sdec_loss.new_tensor(float(sdec_stats.get("gate_recall", 0.0))),
            "sdec_volume_ratio": sdec_loss.new_tensor(float(sdec_stats.get("volume_ratio", 0.0))),
            "udec_loss": udec_loss,
            "brf_loss": brf_loss,
            "brf_bce": brf_loss.new_tensor(float(brf_stats.get("bce", 0.0))),
            "brf_dice": brf_loss.new_tensor(float(brf_stats.get("dice", 0.0))),
            "brf_base_bce": brf_loss.new_tensor(float(brf_stats.get("base_bce", 0.0))),
            "brf_base_dice": brf_loss.new_tensor(float(brf_stats.get("base_dice", 0.0))),
            "brf_tokens": brf_loss.new_tensor(float(brf_stats.get("tokens", 0.0))),
            "brf_recall": brf_loss.new_tensor(float(brf_stats.get("recall", 0.0))),
            "brf_precision": brf_loss.new_tensor(float(brf_stats.get("precision", 0.0))),
            "brf_base_recall": brf_loss.new_tensor(float(brf_stats.get("base_recall", 0.0))),
            "brf_base_precision": brf_loss.new_tensor(float(brf_stats.get("base_precision", 0.0))),
            "brf_rank": brf_loss.new_tensor(float(brf_stats.get("rank", 0.0))),
            "brf_base_rank": brf_loss.new_tensor(float(brf_stats.get("base_rank", 0.0))),
            "brf_rank_acc": brf_loss.new_tensor(float(brf_stats.get("rank_acc", 0.0))),
            "brf_base_rank_acc": brf_loss.new_tensor(float(brf_stats.get("base_rank_acc", 0.0))),
            "brf_pairs": brf_loss.new_tensor(float(brf_stats.get("pairs", 0.0))),
            "brf_local_pairs": brf_loss.new_tensor(float(brf_stats.get("local_pairs", 0.0))),
            "brf_rank_local": brf_loss.new_tensor(float(brf_stats.get("rank_local", 0.0))),
            "brf_base_rank_local": brf_loss.new_tensor(float(brf_stats.get("base_rank_local", 0.0))),
            "brf_rank_acc_local": brf_loss.new_tensor(float(brf_stats.get("rank_acc_local", 0.0))),
            "brf_base_rank_acc_local": brf_loss.new_tensor(float(brf_stats.get("base_rank_acc_local", 0.0))),
            "udec_bce": udec_loss.new_tensor(float(udec_stats.get("bce", 0.0))),
            "udec_dice": udec_loss.new_tensor(float(udec_stats.get("dice", 0.0))),
            "udec_aux_bce": udec_loss.new_tensor(float(udec_stats.get("aux_bce", 0.0))),
            "udec_tiles": udec_loss.new_tensor(float(udec_stats.get("tiles", 0.0))),
            "udec_gate_recall": udec_loss.new_tensor(float(udec_stats.get("gate_recall", 0.0))),
            "udec_volume_ratio": udec_loss.new_tensor(float(udec_stats.get("volume_ratio", 0.0))),
            "context_distill_loss": context_distill_loss,
            "context_contrast_loss": context_contrast_loss,
            "context_contrast_pos_cos": context_contrast_loss.new_tensor(
                float(context_contrast_stats.get("contrast_pos_cos", 0.0))),
            "context_contrast_neg_cos": context_contrast_loss.new_tensor(
                float(context_contrast_stats.get("contrast_neg_cos", 0.0))),
            "latent_distance_loss": latent_distance_loss,
            "vertex_query_loss": vertex_query_loss,
            "vertex_query_count": latent_distance_loss.new_tensor(float(vertex_query_count)),
            "context_distill_cos": context_distill_loss.new_tensor(
                float(context_distill_stats.get("distill_cos", 0.0))),
            "context_distill_cross_cos": context_distill_loss.new_tensor(
                float(context_distill_stats.get("distill_cross_cos", 0.0))),
            "grad_norm": grad_norm,
        })

        if step % log_interval == 0:
            timing_log = {}
            if live_timing_records:
                torch.cuda.synchronize(device)
                timing_log = {
                    "data_fetch_wall_s_mean": float(np.mean([item[0] for item in live_timing_records])),
                    "batch_prepare_wall_s_mean": float(np.mean([item[1] for item in live_timing_records])),
                    "gpu_train_step_s_mean": float(
                        np.mean([item[2].elapsed_time(item[3]) / 1000.0 for item in live_timing_records])
                    ),
                    "live_timing_steps": float(len(live_timing_records)),
                }
                gpu_photo_timings = [
                    item[4].elapsed_time(item[5]) / 1000.0
                    for item in live_timing_records
                    if item[4] is not None and item[5] is not None
                ]
                if gpu_photo_timings:
                    timing_log["gpu_photometric_s_mean"] = float(np.mean(gpu_photo_timings))
                live_timing_records.clear()
            target_voxels = pairs.get("target_voxels")
            if target_voxels is None:
                target_voxels = pairs["target_mask"].flatten(1).sum(1)
            target_voxels = target_voxels.float()
            latent_diag_log = build_latent_diagnostic_log(
                z_pred_raw.detach(),
                z_gt.detach(),
                target_ae,
                cfg,
                step=step,
                device=device,
                dtype=dtype,
            )
            elapsed_s = time.perf_counter() - train_start_time
            steps_done = max(1, step - start_step)
            seconds_per_step = elapsed_s / steps_done
            # Loss channels below are window MEANS over the steps since the
            # previous row (one row per epoch by default).
            window_means = log_window.means()
            window_steps = log_window.window_steps
            window_stats = log_window.distribution_stats(
                clip_thresholds=({"grad_norm": float(grad_clip)} if grad_clip is not None else None)
            )
            log_window.reset()
            row = {
                "step": step,
                "epoch": float(step / steps_per_epoch),
                "elapsed_s": float(elapsed_s),
                "seconds_per_step": float(seconds_per_step),
                "estimated_epoch_time_s": float(seconds_per_step * steps_per_epoch),
                "lr": float(optimizer.param_groups[0]["lr"]),
                "grad_clip": float(grad_clip) if grad_clip is not None else None,
                "window_steps": int(window_steps),
                **window_stats,
                **window_means,
                "latent_mse_loss_scale": float(final_mse_scale),
                "latent_prediction_normalized": float(latent_codec.predict_normalized),
                "same_sheet_prompt_consistency_groups": float(consistency_groups),
                "coordinate_query_weight": float(coordinate_query_weight),
                "coordinate_query_num_points": float(coordinate_query_num_points),
                "decoded_bce_applied": float(apply_decoded_bce),
                "p2sd_num_pairs": float(pair_count),
                "target_ae_encode_masks": float(target_encode_count),
                "target_ae_encode_reuse_factor": float(target_reuse_factor),
                "p2sd_target_voxels_mean": float(target_voxels.mean().detach().cpu()),
                "p2sd_target_voxels_min": float(target_voxels.min().detach().cpu()),
                "p2sd_target_voxels_max": float(target_voxels.max().detach().cpu()),
            }
            row.update(timing_log)
            if coordinate_query_teacher is not None and coordinate_query_prediction is not None:
                row.update(tensor_stats_for_log("coordinate_query_teacher", coordinate_query_teacher.detach()))
                row.update(tensor_stats_for_log("coordinate_query_prediction", coordinate_query_prediction.detach()))
            if latent_codec.predict_normalized:
                row.update(tensor_stats_for_log("latent_pred_model", z_pred.detach().float()))
                row.update(tensor_stats_for_log("latent_gt_model", z_gt_model.detach().float()))
            row.update(latent_norm_log)
            row.update(latent_diag_log)
            # Do not fill current run logs with zero-valued diagnostics from
            # disabled experimental heads (including legacy recall/accuracy).
            inactive_prefixes = tuple(prefix for prefix, head in [
                ("dense_", dense), ("refine_", refine), ("sdec_", sdec),
                ("udec_", udec), ("brf_", brf),
            ] if head is None)
            row = {key: value for key, value in row.items()
                   if not key.startswith(inactive_prefixes)}
            append_jsonl(run_dir / "metrics.jsonl", row)
            update_heartbeat(run_dir, status="running", step=step, loss=row["loss"])

        if step % steps_per_epoch == 0:
            now = time.perf_counter()
            epoch_steps = max(1, step - epoch_start_step)
            epoch_time_s = now - epoch_start_time
            train_epoch_time_s = max(0.0, epoch_time_s - preceding_post_epoch_overhead_s)
            append_jsonl(run_dir / "metrics.jsonl", {
                "split": "train_epoch",
                "step": step,
                "epoch": float(step / steps_per_epoch),
                "epoch_index": int(step // steps_per_epoch),
                "epoch_time_s": float(epoch_time_s),
                "seconds_per_step": float(epoch_time_s / epoch_steps),
                # `epoch_time_s` has historically included post-epoch work
                # from the preceding boundary. Keep it for continuity while
                # exposing the actual train-loop time separately.
                "train_epoch_time_s": float(train_epoch_time_s),
                "train_seconds_per_step": float(train_epoch_time_s / epoch_steps),
                "preceding_checkpoint_time_s": float(preceding_checkpoint_time_s),
                "preceding_post_epoch_overhead_s": float(preceding_post_epoch_overhead_s),
                "p2sd_pairs_in_epoch": float(epoch_pair_count),
            })
            update_heartbeat(
                run_dir,
                status="running",
                step=step,
                loss=float(loss.detach().cpu()),
                epoch_time_s=float(epoch_time_s),
            )
            if log_param_dynamics_enabled:
                # Grads from this epoch's final step are still live here (the
                # next iteration's zero_grad has not run yet).
                param_dynamics_snapshot = log_param_dynamics(
                    run_dir,
                    raw_model,
                    param_dynamics_snapshot,
                    float(optimizer.param_groups[0]["lr"]),
                    step,
                    steps_per_epoch,
                )
            if loss_plot_interval > 0 and (step // steps_per_epoch) % loss_plot_interval == 0:
                # Best-effort loss-vs-epoch PNGs into run_dir/plots. Never fatal.
                try:
                    write_loss_plots(run_dir)
                except Exception as exc:  # pragma: no cover - telemetry only
                    print(f"[loss_plots] skipped: {exc}", flush=True)
            epoch_start_time = now
            epoch_start_step = step
            epoch_pair_count = 0

        is_last_step = step == max_steps
        # Persist the completed training state before validation or an expensive
        # probe event. A SIGKILL/OOM during evaluation must not discard an
        # otherwise recoverable epoch.
        should_save = step % save_interval == 0 or (is_last_step and save_on_last_step)
        checkpoint_time_s = 0.0
        post_epoch_start_time = time.perf_counter() if step % steps_per_epoch == 0 else None
        if should_save:
            checkpoint_start_time = time.perf_counter()
            save_current_training_state(step)
            checkpoint_time_s = time.perf_counter() - checkpoint_start_time
            append_jsonl(run_dir / "metrics.jsonl", {
                "split": "checkpoint",
                "step": step,
                "checkpoint_time_s": float(checkpoint_time_s),
                "checkpoint_path": "last.pt",
            })

        evaluation_time_s = 0.0
        if bool(cfg.get("training", {}).get("standard_validation", True)) and (
            step % eval_interval == 0 or (is_last_step and eval_on_last_step)
        ):
            evaluation_start_time = time.perf_counter()
            score = evaluate_p2sd(
                model,
                target_ae,
                val_loader,
                cfg,
                run_dir,
                step,
                steps_per_epoch,
                device,
                dtype,
                fixed_probe_loader=fixed_probe_loader,
            )
            if score > best["score"]:
                best = {"score": float(score), "step": step}
                write_json(run_dir / "best.json", best)
                save_checkpoint(run_dir / "best.pt", model=raw_model, optimizer=optimizer, step=step, cfg=cfg, best=best)
            evaluation_time_s = time.perf_counter() - evaluation_start_time

        if scene_eval_settings is not None and (
            step % scene_eval_settings["interval_steps"] == 0
            or (is_last_step and scene_eval_settings["on_last_step"])
        ):
            run_scene_evaluation_from_training(
                settings=scene_eval_settings,
                run_dir=run_dir,
                step=step,
                steps_per_epoch=steps_per_epoch,
                raw_model=raw_model,
                optimizer=optimizer,
                cfg=cfg,
                best=best,
            )

        if post_epoch_start_time is not None:
            preceding_checkpoint_time_s = checkpoint_time_s
            preceding_post_epoch_overhead_s = time.perf_counter() - post_epoch_start_time
            if evaluation_time_s > 0.0:
                append_jsonl(run_dir / "metrics.jsonl", {
                    "split": "post_epoch_overhead",
                    "step": step,
                    "checkpoint_time_s": float(checkpoint_time_s),
                    "evaluation_time_s": float(evaluation_time_s),
                    "post_epoch_overhead_s": float(preceding_post_epoch_overhead_s),
                })

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
    step_profiler.close()
    write_resume_context(run_dir)
    return 0


def configure_torch_runtime(training_cfg: dict) -> None:
    cudnn_deterministic = bool(training_cfg.get("cudnn_deterministic", False))
    torch.backends.cudnn.deterministic = cudnn_deterministic
    torch.backends.cudnn.benchmark = bool(training_cfg.get("cudnn_benchmark", True)) and not cudnn_deterministic
    matmul_precision = training_cfg.get("matmul_precision", "high")
    if matmul_precision:
        torch.set_float32_matmul_precision(str(matmul_precision))


def maybe_compile_model(model: torch.nn.Module, training_cfg: dict) -> torch.nn.Module:
    compile_cfg = training_cfg.get("compile", {})
    enabled = bool(compile_cfg.get("model", False)) if isinstance(compile_cfg, dict) else bool(compile_cfg)
    if not enabled:
        return model
    mode = str(compile_cfg.get("mode", "reduce-overhead")) if isinstance(compile_cfg, dict) else "reduce-overhead"
    fullgraph = bool(compile_cfg.get("fullgraph", False)) if isinstance(compile_cfg, dict) else False
    print(f"[compile] model mode={mode} fullgraph={fullgraph}", flush=True)
    return torch.compile(model, mode=mode, fullgraph=fullgraph)


def maybe_compile_target_encode(target_ae: torch.nn.Module, training_cfg: dict):
    compile_cfg = training_cfg.get("compile", {})
    enabled = bool(compile_cfg.get("target_ae_encode", False)) if isinstance(compile_cfg, dict) else False
    if not enabled:
        return target_ae.encode
    mode = str(compile_cfg.get("mode", "reduce-overhead")) if isinstance(compile_cfg, dict) else "reduce-overhead"
    fullgraph = bool(compile_cfg.get("fullgraph", False)) if isinstance(compile_cfg, dict) else False
    print(f"[compile] target_ae.encode mode={mode} fullgraph={fullgraph}", flush=True)
    return torch.compile(target_ae.encode, mode=mode, fullgraph=fullgraph)


def _param_dynamics_layer_key(name: str) -> str:
    """Bucket a parameter name into a layer/stage for dynamics logging."""
    parts = name.split(".")
    if name.startswith("image_encoder.stem"):
        return "image_encoder.stem"
    if name.startswith("image_encoder.encoder.") and len(parts) > 2 and parts[2].isdigit():
        return f"image_encoder.encoder.{parts[2]}"
    for prefix in ("blocks", "image_context_blocks"):
        if len(parts) > 1 and parts[0] == prefix and parts[1].isdigit():
            return f"{prefix}.{parts[1]}"
    return parts[0]


def log_param_dynamics(
    run_dir: Path,
    model: torch.nn.Module,
    prev_snapshot: dict[str, torch.Tensor],
    lr: float,
    step: int,
    steps_per_epoch: int,
) -> dict[str, torch.Tensor]:
    """Append one per-layer weight/grad dynamics row to param_dynamics.jsonl.

    For each layer we record grad norm, weight norm, the relative weight movement
    since the previous snapshot (drift = ||W_t - W_prev|| / ||W_prev||), and the
    effective update ratio lr*||g||/||W||. A layer whose drift and update ratio
    decay toward zero has stopped learning and is a freeze candidate. Called once
    per epoch, so the extra host syncs are negligible. Returns the new snapshot.
    """
    from collections import defaultdict

    agg: dict[str, dict[str, float]] = defaultdict(
        lambda: {"g2": 0.0, "w2": 0.0, "d2": 0.0, "wprev2": 0.0, "n": 0}
    )
    new_snapshot: dict[str, torch.Tensor] = {}
    for name, p in model.named_parameters():
        w = p.detach().float()
        new_snapshot[name] = w.to("cpu").clone()
        a = agg[_param_dynamics_layer_key(name)]
        a["w2"] += float(w.pow(2).sum())
        if p.grad is not None:
            a["g2"] += float(p.grad.detach().float().pow(2).sum())
        prev = prev_snapshot.get(name)
        if prev is not None:
            prev = prev.to(w.device)
            a["d2"] += float((w - prev).pow(2).sum())
            a["wprev2"] += float(prev.pow(2).sum())
        a["n"] += p.numel()

    layers: dict[str, dict[str, float]] = {}
    for key, a in agg.items():
        w_norm = math.sqrt(a["w2"])
        g_norm = math.sqrt(a["g2"])
        w_prev_norm = math.sqrt(a["wprev2"])
        layers[key] = {
            "grad_norm": g_norm,
            "weight_norm": w_norm,
            "drift": math.sqrt(a["d2"]) / (w_prev_norm + 1e-12) if w_prev_norm > 0 else 0.0,
            "update_ratio": lr * g_norm / (w_norm + 1e-12) if w_norm > 0 else 0.0,
            "num_params": int(a["n"]),
        }
    append_jsonl(run_dir / "param_dynamics.jsonl", {
        "split": "param_dynamics",
        "step": int(step),
        "epoch": float(step / steps_per_epoch),
        "lr": float(lr),
        "layers": layers,
    })
    return new_snapshot


def load_p2sd_state_dict(
    model: torch.nn.Module,
    state: dict,
    *,
    strict: bool = True,
    allowed_prefixes: tuple[str, ...] = ("modulator_blocks",),
) -> dict:
    """Load a P2SD checkpoint into ``model``.

    ``strict=True`` is a plain strict load (unchanged behaviour). ``strict=False``
    supports warm-starting when a submodule was swapped for a differently-shaped
    one -- e.g. replacing the prefix-attn prompt modulator with the SAM-style
    two-way modulator. In that mode, checkpoint keys that collide in name but
    mismatch in shape are dropped so the new submodule starts fresh, and the
    resulting missing/unexpected keys are tolerated ONLY when every one of them
    lives under ``allowed_prefixes``. Any mismatch outside that allowlist is
    still fatal, so a genuinely wrong checkpoint can never load silently.

    Returns a summary ``{"fresh", "dropped", "reshaped"}`` of tolerated keys.
    """
    if strict:
        model.load_state_dict(state, strict=True)
        return {"fresh": 0, "dropped": 0, "reshaped": 0}
    model_shapes = {k: tuple(v.shape) for k, v in model.state_dict().items()}
    filtered: dict = {}
    reshaped: list[str] = []
    for k, v in state.items():
        want = model_shapes.get(k)
        if want is not None and want != tuple(v.shape):
            if k.startswith(allowed_prefixes):
                reshaped.append(k)
                continue
            raise RuntimeError(
                f"resume: shape mismatch for '{k}' outside allowed prefixes "
                f"{allowed_prefixes}: checkpoint {tuple(v.shape)} vs model {want}"
            )
        filtered[k] = v
    incompat = model.load_state_dict(filtered, strict=False)
    stray = [
        k
        for k in list(incompat.missing_keys) + list(incompat.unexpected_keys)
        if not k.startswith(allowed_prefixes)
    ]
    if stray:
        shown = ", ".join(stray[:20])
        more = f" (+{len(stray) - 20} more)" if len(stray) > 20 else ""
        raise RuntimeError(
            "resume: non-strict load left mismatched keys outside allowed prefixes "
            f"{allowed_prefixes}: {shown}{more}"
        )
    return {
        "fresh": len(incompat.missing_keys),
        "dropped": len(incompat.unexpected_keys),
        "reshaped": len(reshaped),
    }


def maybe_resume_p2sd(
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
    resume_strict = bool(training_cfg.get("resume_strict", True))
    allowed_prefixes = tuple(training_cfg.get("resume_partial_prefixes", ("modulator_blocks",)))
    summary = load_p2sd_state_dict(
        model, state, strict=resume_strict, allowed_prefixes=allowed_prefixes
    )
    if not resume_strict:
        print(
            f"[resume] partial load (strict=false): {summary['fresh']} fresh + "
            f"{summary['dropped']} dropped + {summary['reshaped']} reshaped keys, all within "
            f"{allowed_prefixes}; everything else loaded from checkpoint",
            flush=True,
        )
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
    source_step = start_step
    if bool(training_cfg.get("reset_best_on_resume", False)):
        best = {"score": -math.inf, "step": 0, "source_resume_step": source_step}
    if bool(training_cfg.get("reset_step_on_resume", False)):
        start_step = 0
    print(f"[resume] loaded {resume_path} at step {source_step}; starting at step {start_step}", flush=True)
    return start_step, best


def should_apply_decoded_bce(
    step: int,
    *,
    weight: float,
    interval: int,
    warmup_steps: int,
) -> bool:
    if weight <= 0:
        return False
    if warmup_steps > 0 and step <= warmup_steps:
        return True
    return step % max(1, interval) == 0


def coordinate_query_distillation_loss(
    target_ae: torch.nn.Module,
    z_pred_raw: torch.Tensor,
    z_target: torch.Tensor,
    *,
    image_shape: tuple[int, int, int],
    num_points: int,
    smooth_l1_beta: float,
    target_mask: torch.Tensor | None = None,
    image_index: torch.Tensor | None = None,
    sheet_id: torch.Tensor | None = None,
    sampling: str = "uniform",
    band_voxels: float = 8.0,
    inside_fraction: float = 1.0 / 3.0,
    near_outside_fraction: float = 1.0 / 3.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Distill the frozen AE coordinate-query task into P2SD's latent output."""
    if z_pred_raw.shape != z_target.shape:
        raise ValueError(
            "coordinate query distillation requires matching latent shapes, got "
            f"{tuple(z_pred_raw.shape)} and {tuple(z_target.shape)}"
        )
    query_distance = getattr(target_ae, "query_distance", None)
    if not callable(query_distance):
        raise RuntimeError(
            "p2sd.loss.coordinate_query.enabled requires a target AE with an enabled coordinate query head"
        )
    if num_points <= 0:
        raise ValueError("coordinate query distillation requires num_points > 0")

    points_zyx = coordinate_query_points(
        target_mask=target_mask,
        image_index=image_index,
        sheet_id=sheet_id,
        batch_size=z_pred_raw.shape[0],
        image_shape=image_shape,
        num_points=num_points,
        sampling=sampling,
        band_voxels=band_voxels,
        inside_fraction=inside_fraction,
        near_outside_fraction=near_outside_fraction,
        device=z_pred_raw.device,
    )
    # Smooth L1 saves its target for the student backward, so this must create
    # a normal no-grad tensor rather than an inference tensor.
    with torch.no_grad():
        teacher = query_distance(z_target, points_zyx, image_shape)
    prediction = query_distance(z_pred_raw, points_zyx, image_shape)
    loss = F.smooth_l1_loss(
        prediction.float(),
        teacher.float(),
        beta=float(smooth_l1_beta),
    )
    return loss, teacher, prediction


def coordinate_query_points(
    *,
    target_mask: torch.Tensor | None,
    image_index: torch.Tensor | None,
    sheet_id: torch.Tensor | None,
    batch_size: int,
    image_shape: tuple[int, int, int],
    num_points: int,
    sampling: str,
    band_voxels: float,
    inside_fraction: float,
    near_outside_fraction: float,
    device: torch.device,
) -> torch.Tensor:
    """Sample P2SD coordinate queries, balancing thin-sheet regions when requested."""
    if sampling == "uniform":
        point_axes = [
            torch.randint(int(size), (batch_size, num_points), device=device)
            for size in image_shape
        ]
        return torch.stack(point_axes, dim=-1).float()
    if sampling != "balanced_regions":
        raise ValueError(
            "coordinate query sampling must be uniform|balanced_regions, got "
            f"{sampling!r}"
        )
    if target_mask is None or image_index is None or sheet_id is None:
        raise ValueError("balanced coordinate queries require target_mask, image_index, and sheet_id")
    if target_mask.shape[0] != batch_size:
        raise ValueError("coordinate query target_mask batch must match predicted latent batch")
    if target_mask.ndim != 5 or target_mask.shape[1] != 1:
        raise ValueError(f"coordinate query masks must be [B, 1, D, H, W], got {tuple(target_mask.shape)}")
    if tuple(int(value) for value in target_mask.shape[-3:]) != image_shape:
        raise ValueError("coordinate query mask shape must match image_shape")

    # Repeated prompt variants share one selected sheet. Compute its expensive
    # EDT and query coordinates once, then restore the prompt-pair order.
    unique_masks, inverse = unique_p2sd_target_masks(target_mask, image_index, sheet_id)
    signed_voxels = signed_distance_edt_cucim(unique_masks[:, 0] > 0.5)
    unique_points, _ = sample_query_distance_points_gpu(
        signed_voxels=signed_voxels,
        normalized_distance=signed_voxels,
        count=num_points,
        band_voxels=band_voxels,
        sampling="balanced_regions",
        inside_fraction=inside_fraction,
        near_outside_fraction=near_outside_fraction,
    )
    return unique_points.index_select(0, inverse)


def resolve_max_steps(training_cfg: dict, steps_per_epoch: int) -> tuple[int, float]:
    raw = training_cfg.get("max_steps")
    if raw is not None and int(raw) > 0:
        max_steps = int(raw)
        return max_steps, float(max_steps / max(1, steps_per_epoch))
    epochs = float(training_cfg.get("epochs", training_cfg.get("max_epochs", 1)))
    max_steps = max(1, int(math.ceil(epochs * max(1, steps_per_epoch))))
    return max_steps, epochs


def sheet_report_protocol_key(cfg: dict) -> dict:
    scene = cfg.get("scene_eval", {})
    return {"profile": scene.get("report_profile", "sheet_union_v1"), "manifest_path": str(Path(scene["manifest_path"]).resolve()),
            "prompt_ids": str(scene.get("prompt_ids", "p00")),
            "threshold": float(cfg.get("metrics", {}).get("threshold", .5))}


def resolve_scene_eval_settings(
    cfg: dict,
    steps_per_epoch: int,
    max_steps: int,
) -> dict | None:
    """Parse the optional in-training scene-evaluation block.

    Scene evaluation decodes every prompted sheet of every manifest case and
    reports whole-volume contact/topology metrics. It runs in shard
    subprocesses during the eval window, when training is paused and the
    device is otherwise idle.
    """
    scene_cfg = cfg.get("scene_eval", {}) or {}
    if not bool(scene_cfg.get("enabled", False)):
        return None
    manifest_path = scene_cfg.get("manifest_path")
    if not manifest_path:
        raise ValueError("scene_eval.enabled requires scene_eval.manifest_path")
    if "interval_epochs" in scene_cfg:
        interval = max(1, int(round(float(scene_cfg["interval_epochs"]) * max(1, steps_per_epoch))))
    else:
        interval = int(scene_cfg.get("interval_steps", max(1, max_steps)))
    return {
        "interval_steps": max(1, interval),
        "report_profile": str(scene_cfg.get("report_profile", "legacy")),
        "prompt_ids": str(scene_cfg.get("prompt_ids", "p00")),
        "topology_backend": str(scene_cfg.get("topology_backend", "binary_exact")),
        "manifest_path": str(manifest_path),
        "target_contract": str(scene_cfg.get("target_contract", "eligible_only")),
        "evaluation_mode": str(scene_cfg.get("evaluation_mode", "topology_proxies")),
        "min_component_voxels": int(scene_cfg.get("min_component_voxels", 500)),
        "parallel": int(scene_cfg.get("parallel", 3)),
        "prompt_batch_size": int(scene_cfg.get("prompt_batch_size", 8)),
        "cpu_workers": int(scene_cfg.get("cpu_workers", 4)),
        "on_last_step": bool(scene_cfg.get("on_last_step", True)),
    }


def run_scene_evaluation_from_training(
    *,
    settings: dict,
    run_dir: Path,
    step: int,
    steps_per_epoch: int,
    raw_model,
    optimizer,
    cfg: dict,
    best: dict,
) -> None:
    """Run the sharded scene evaluation against the just-saved trainer state.

    Failures are logged, never raised: scene evaluation is telemetry and must
    not kill a multi-week training run.
    """
    start_time = time.perf_counter()
    row: dict[str, Any] = {
        "split": "scene_val",
        "step": step,
        "epoch": float(step / max(1, steps_per_epoch)),
    }
    try:
        # Everything below is inside the try to honour this function's contract.
        # The lazy import in particular used to sit outside it: a source edit
        # landing while training was live left a partially-updated tree, and the
        # ImportError raised here killed run 0008 at ep50 -- no scene_eval dir,
        # no scene_val row, just a dead run. The evaluator imports helpers back
        # from this module, so the import must stay lazy; it must not be fatal.
        from vesuvius_p2sd.research.evaluate_p2sd_case_scenes import run_parallel_scene_evaluation

        save_checkpoint(run_dir / "last.pt", model=raw_model, optimizer=optimizer, step=step, cfg=cfg, best=best)
        if settings.get("save_extra_state") is not None:
            settings["save_extra_state"](run_dir)
        output_dir = run_dir / "scene_eval" / f"step_{step:06d}"
        suffix = 0
        while output_dir.exists() and any(output_dir.iterdir()):
            suffix += 1
            output_dir = run_dir / "scene_eval" / f"step_{step:06d}_rerun{suffix}"
        torch.cuda.empty_cache()
        if settings.get("report_profile") in {"sheet_union_v1", "sheet_union_v2_border", "sheet_union_v3_box"}:
            from vesuvius_p2sd.research.evaluate_sheet_union import run_report_subprocess

            evaluated_checkpoint = run_dir / "checkpoints" / f"step_{step:06d}" / "last.pt"
            if not evaluated_checkpoint.exists():
                # Preserve the exact evaluated weights even when periodic
                # snapshots are disabled in another configuration.
                evaluated_checkpoint = run_dir / "eval_checkpoints" / output_dir.name / "last.pt"
                evaluated_checkpoint.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(run_dir / "last.pt", evaluated_checkpoint)
            summary = run_report_subprocess(
                checkpoint_path=evaluated_checkpoint, source_run_dir=run_dir,
                scene_manifest_path=settings["manifest_path"], output_dir=output_dir,
                prompt_ids=settings["prompt_ids"], prompt_batch_size=settings["prompt_batch_size"],
                cpu_workers=settings["cpu_workers"], topology_backend=settings["topology_backend"],
                ignore_policy={"sheet_union_v1": "source", "sheet_union_v2_border": "border_only",
                               "sheet_union_v3_box": "border_box"}[settings["report_profile"]],
            )
            for scope in ("sheet", "union"):
                for key, values in summary[scope].items():
                    row[f"{scope}_{key}"] = values["mean"]
            row.update(status="complete", report_profile=settings["report_profile"],
                       scene_eval_dir=str(output_dir.relative_to(run_dir)))
            metric = "union_leaderboard_formula_score"
            score = row[metric]
            protocol_key = sheet_report_protocol_key({**cfg, "scene_eval": settings})
            persisted_path = run_dir / "sheet_union_best.json"
            if persisted_path.exists():
                persisted = json.loads(persisted_path.read_text())
                if (persisted.get("report_protocol") == protocol_key
                        and (run_dir / persisted["checkpoint_path"]).is_file()
                        and (best.get("report_protocol") != protocol_key or persisted["score"] > best["score"])):
                    # A CPU-only backfill may finish while training is live.
                    best.update(persisted)
            if (best.get("metric") != metric or best.get("report_protocol") != protocol_key
                    or score > best["score"]):
                best.update(score=score, metric=metric, step=step,
                    checkpoint_path=str(evaluated_checkpoint.relative_to(run_dir)), report_protocol=protocol_key)
                (evaluated_checkpoint.parent / "PIN").touch()
                write_json(run_dir / "best.json", best)
                write_json(run_dir / "sheet_union_best.json", best)
            row["scene_eval_elapsed_s"] = time.perf_counter() - start_time
            append_jsonl(run_dir / "metrics.jsonl", row)
            return
        summary = run_parallel_scene_evaluation(
            checkpoint_path=run_dir / "last.pt",
            source_run_dir=run_dir,
            scene_manifest_path=settings["manifest_path"],
            output_dir=output_dir,
            target_contract=settings["target_contract"],
            parallel=settings["parallel"],
            prompt_batch_size=settings["prompt_batch_size"],
            cpu_workers=settings["cpu_workers"],
            evaluation_mode=settings["evaluation_mode"],
            min_component_voxels=settings["min_component_voxels"],
            refine_head_path=(run_dir / settings.get("refine_head_file", "refine_last.pt")) if settings.get("refine_head") else None,
        )
        row.update({
            "status": "complete",
            "scene_contact_free": summary.get("case_scene_macro_sheet_contact_free"),
            "scene_contact_pair_count": summary.get("case_scene_macro_sheet_contact_pair_count"),
            "scene_contact_overlap_voxels": summary.get("case_scene_macro_sheet_contact_overlap_voxels"),
            "scene_union_component_count": summary.get(
                "case_union_macro_fast_topology_volume_union_component_count"),
            "scene_union_component_count_worst": summary.get(
                "case_union_worst_fast_topology_volume_union_component_count"),
            "scene_sheet_component_count": summary.get(
                "sheet_prompt_macro_fast_topology_sheet_component_count_mean"),
            "scene_eval_dir": str(output_dir.relative_to(run_dir)),
        })
    except Exception as exc:
        row.update({"status": "failed", "error": str(exc)[:500]})
        print(f"[scene_eval] failed at step {step}: {exc}", flush=True)
    row["scene_eval_elapsed_s"] = float(time.perf_counter() - start_time)
    append_jsonl(run_dir / "metrics.jsonl", row)


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


def should_save_train_batch_viz(cfg: dict, step: int, start_step: int) -> bool:
    viz_cfg = cfg.get("visualization", {})
    if not bool(viz_cfg.get("save_train_batch", False)):
        return False
    if step == start_step + 1:
        return True
    interval = int(viz_cfg.get("train_batch_interval_steps", 0) or 0)
    return interval > 0 and step % interval == 0


@torch.no_grad()
def save_train_batch_viz(
    batch: dict,
    pairs: dict[str, torch.Tensor],
    image: torch.Tensor,
    cfg: dict,
    run_dir: Path,
    step: int,
) -> None:
    viz_cfg = cfg.get("visualization", {})
    max_items = max(1, int(viz_cfg.get("train_batch_max_items", 4)))
    pair_count = int(pairs["target_mask"].shape[0])
    component_label = batch.get("component_label")
    if component_label is not None:
        component_label = component_label.long()
        if component_label.ndim == 5 and component_label.shape[1] == 1:
            component_label = component_label[:, 0]
    for pair_idx in range(min(max_items, pair_count)):
        image_idx = int(pairs["image_index"][pair_idx].detach().cpu().item())
        sheet_id = int(pairs["sheet_id"][pair_idx].detach().cpu().item())
        comp = None
        if component_label is not None:
            comp = component_label[image_idx].detach().cpu().numpy()
        save_p2sd_train_batch_png(
            run_dir / "train_viz" / f"step_{step:06d}_pair_{pair_idx:02d}.png",
            image=image[image_idx, 0].detach().float().cpu().numpy(),
            target_mask=pairs["target_mask"][pair_idx, 0].detach().float().cpu().numpy(),
            component_label=comp,
            prompt_points_zyx=pairs["prompt_points"][pair_idx].detach().float().cpu().numpy(),
            prompt_labels=pairs["prompt_labels"][pair_idx].detach().long().cpu().numpy(),
            meta={
                "step": int(step),
                "pair_index": int(pair_idx),
                "image_index": int(image_idx),
                "sheet_id": int(sheet_id),
                "task": "p2sd_train_batch",
            },
        )
    prune_visualization_groups(
        run_dir / "train_viz",
        keep_latest=resolve_visualization_keep_latest(
            viz_cfg,
            "keep_latest_train_batch",
            "train_batch_keep_latest",
        ),
        stem_prefixes=("step_",),
    )


def resolve_visualization_keep_latest(viz_cfg: dict, *keys: str) -> int | None:
    for key in keys:
        if key in viz_cfg:
            return int(viz_cfg[key])
    if "keep_latest" in viz_cfg:
        return int(viz_cfg["keep_latest"])
    return None


def should_evaluate_fixed_probes(
    visualization_cfg: dict[str, Any],
    *,
    step: int,
    steps_per_epoch: int,
    milestone_epoch: int | None,
) -> bool:
    """Keep a large deterministic probe suite off the ordinary eval cadence."""

    probe_cfg = visualization_cfg.get("probes", {})
    if milestone_epoch is not None and bool(probe_cfg.get("evaluate_at_milestones", True)):
        return True
    interval = probe_cfg.get("eval_interval_epochs")
    if interval is None:
        # Existing manifests evaluated on every validation call before this
        # optional cadence was introduced.
        return True
    interval = float(interval)
    if interval <= 0:
        return False
    epoch = float(step) / max(1, int(steps_per_epoch))
    nearest_multiple = round(epoch / interval)
    return nearest_multiple > 0 and abs(epoch - nearest_multiple * interval) <= 1e-6


class LatentCodec:
    def __init__(
        self,
        *,
        mode: str,
        mean: torch.Tensor | None = None,
        std: torch.Tensor | None = None,
        predict_normalized: bool = False,
    ) -> None:
        self.mode = mode
        self.mean = mean
        self.std = std
        self.predict_normalized = bool(predict_normalized)

    @property
    def enabled(self) -> bool:
        return self.mean is not None and self.std is not None

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        if not self.enabled:
            return x.float()
        return (x.float() - self.mean.to(device=x.device, dtype=torch.float32)) / self.std.to(
            device=x.device,
            dtype=torch.float32,
        )

    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        if not self.enabled:
            return x.float()
        return x.float() * self.std.to(device=x.device, dtype=torch.float32) + self.mean.to(
            device=x.device,
            dtype=torch.float32,
        )

    def model_target(self, z_gt_raw: torch.Tensor) -> torch.Tensor:
        if self.predict_normalized:
            return self.normalize(z_gt_raw)
        return z_gt_raw.float()

    def raw_prediction(self, z_pred_model: torch.Tensor) -> torch.Tensor:
        if self.predict_normalized:
            return self.denormalize(z_pred_model)
        return z_pred_model.float()


def build_static_latent_codec(
    norm_cfg: dict,
    *,
    latent_channels: int,
    device: torch.device,
) -> LatentCodec:
    if not norm_cfg or not bool(norm_cfg.get("enabled", False)):
        return LatentCodec(mode="none")
    mode = str(norm_cfg.get("mode", "batch_channel")).lower()
    predict_normalized = bool(norm_cfg.get("predict_normalized", False))
    if mode in {"batch_channel", "per_batch_channel"}:
        if predict_normalized:
            raise ValueError(
                "p2sd.loss.latent_normalization.predict_normalized requires static "
                "decode stats; use mode=global_channel with mean/std or stats_path"
            )
        return LatentCodec(mode=mode)
    if mode not in {"global_channel", "channel"}:
        raise ValueError(f"Unsupported p2sd.loss.latent_normalization.mode: {mode}")
    eps = float(norm_cfg.get("eps", 1e-6))
    mean, std = load_latent_channel_stats(norm_cfg, latent_channels, device=device, eps=eps)
    return LatentCodec(
        mode="global_channel",
        mean=mean,
        std=std,
        predict_normalized=predict_normalized,
    )


def build_latent_loss_codec(
    z_gt: torch.Tensor,
    norm_cfg: dict,
    *,
    static_codec: LatentCodec | None = None,
) -> tuple[LatentCodec, dict[str, float | str]]:
    if not norm_cfg or not bool(norm_cfg.get("enabled", False)):
        return LatentCodec(mode="none"), {
            "latent_loss_normalization": "none",
            "latent_loss_normalized": 0.0,
        }
    if static_codec is not None and static_codec.enabled:
        log = build_latent_codec_log(static_codec)
        return static_codec, log
    mode = str(norm_cfg.get("mode", "batch_channel")).lower()
    eps = float(norm_cfg.get("eps", 1e-6))
    if mode in {"global_channel", "channel"}:
        codec = build_static_latent_codec(
            norm_cfg,
            latent_channels=int(z_gt.shape[1]),
            device=z_gt.device,
        )
        return codec, build_latent_codec_log(codec)
    if mode not in {"batch_channel", "per_batch_channel"}:
        raise ValueError(f"Unsupported p2sd.loss.latent_normalization.mode: {mode}")
    reduce_dims = (0, 2, 3, 4)
    z_ref = z_gt.detach().float()
    mean = z_ref.mean(dim=reduce_dims, keepdim=True)
    std = z_ref.std(dim=reduce_dims, keepdim=True, unbiased=False).clamp_min(eps)

    codec = LatentCodec(mode="batch_channel", mean=mean, std=std, predict_normalized=False)
    log = {
        "latent_loss_normalization": "batch_channel",
        "latent_loss_normalized": 1.0,
        "latent_prediction_space": "raw",
        "latent_norm_mean_abs": float(mean.abs().mean().detach().cpu()),
        "latent_norm_std_mean": float(std.mean().detach().cpu()),
        "latent_norm_std_min": float(std.min().detach().cpu()),
        "latent_norm_std_max": float(std.max().detach().cpu()),
    }
    return codec, log


def build_latent_loss_normalizer(z_gt: torch.Tensor, norm_cfg: dict):
    codec, log = build_latent_loss_codec(z_gt, norm_cfg)
    return codec.normalize, log


def build_latent_codec_log(codec: LatentCodec) -> dict[str, float | str]:
    log: dict[str, float | str] = {
        "latent_loss_normalization": codec.mode,
        "latent_loss_normalized": 1.0,
        "latent_prediction_space": "normalized" if codec.predict_normalized else "raw",
    }
    if codec.enabled:
        mean = codec.mean.detach().float()
        std = codec.std.detach().float()
        log.update({
            "latent_norm_mean_abs": float(mean.abs().mean().cpu()),
            "latent_norm_std_mean": float(std.mean().cpu()),
            "latent_norm_std_min": float(std.min().cpu()),
            "latent_norm_std_max": float(std.max().cpu()),
        })
    return log


def load_latent_channel_stats(
    norm_cfg: dict,
    latent_channels: int,
    *,
    device: torch.device,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    data: dict[str, Any] = {}
    stats_path = norm_cfg.get("stats_path") or norm_cfg.get("channel_stats_path")
    if stats_path:
        data.update(read_latent_stats_file(Path(stats_path)))
    data.update({key: value for key, value in norm_cfg.items() if key in {"mean", "std", "var"}})
    mean_value = first_present(data, "mean", "channel_mean", "latent_mean")
    std_value = first_present(data, "std", "channel_std", "latent_std")
    if std_value is None:
        var_value = first_present(data, "var", "channel_var", "latent_var")
        if var_value is not None:
            std_value = torch.as_tensor(var_value, dtype=torch.float32).sqrt().tolist()
    if mean_value is None or std_value is None:
        raise ValueError(
            "global_channel latent normalization requires mean/std values or a stats_path "
            "containing mean and std"
        )
    mean = latent_channel_tensor(mean_value, "mean", latent_channels, device=device)
    std = latent_channel_tensor(std_value, "std", latent_channels, device=device).clamp_min(float(eps))
    return mean, std


def read_latent_stats_file(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix.lower() in {".pt", ".pth"}:
        data = torch.load(path, map_location="cpu")
    else:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Latent stats file must contain a mapping: {path}")
    return data


def first_present(data: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in data:
            return data[key]
    return None


def latent_channel_tensor(
    value: Any,
    name: str,
    latent_channels: int,
    *,
    device: torch.device,
) -> torch.Tensor:
    tensor = torch.as_tensor(value, dtype=torch.float32, device=device).flatten()
    if tensor.numel() == 1:
        tensor = tensor.expand(int(latent_channels))
    if tensor.numel() != int(latent_channels):
        raise ValueError(
            f"latent stats {name} must be scalar or length {latent_channels}, "
            f"got {tensor.numel()}"
        )
    return tensor.view(1, int(latent_channels), 1, 1, 1)


@torch.no_grad()
def build_latent_diagnostic_log(
    z_pred: torch.Tensor,
    z_gt: torch.Tensor,
    target_ae,
    cfg: dict,
    *,
    step: int,
    device: torch.device,
    dtype: torch.dtype | None,
) -> dict[str, float]:
    diag_cfg = cfg.get("p2sd", {}).get("diagnostics", {})
    if not bool(diag_cfg.get("enabled", False)):
        return {}
    z_pred_f = z_pred.float()
    z_gt_f = z_gt.float()
    zero = torch.zeros_like(z_gt_f)
    log = {
        "latent_zero_mse": float(F.mse_loss(zero, z_gt_f).detach().cpu()),
        "latent_pred_zero_mse": float(F.mse_loss(z_pred_f, zero).detach().cpu()),
    }
    log.update(tensor_stats_for_log("latent_gt", z_gt_f))
    log.update(tensor_stats_for_log("latent_pred", z_pred_f))
    decode_interval = int(diag_cfg.get("decode_interval_steps", 0) or 0)
    if decode_interval > 0 and step % decode_interval == 0:
        target_ae.eval()
        with autocast_context(device, dtype):
            pred_logits = target_ae.decode(z_pred.to(device=device).detach())
            pred_prob = decoded_foreground_probability(pred_logits)
            log.update(probability_stats_for_log("decoded_pred", pred_prob))
            if bool(diag_cfg.get("decode_gt", False)):
                gt_logits = target_ae.decode(z_gt.to(device=device).detach())
                gt_prob = decoded_foreground_probability(gt_logits)
                log.update(probability_stats_for_log("decoded_gt", gt_prob))
    return log


def tensor_stats_for_log(prefix: str, x: torch.Tensor) -> dict[str, float]:
    return {
        f"{prefix}_mean": float(x.mean().detach().cpu()),
        f"{prefix}_std": float(x.std().detach().cpu()),
        f"{prefix}_min": float(x.min().detach().cpu()),
        f"{prefix}_max": float(x.max().detach().cpu()),
        f"{prefix}_absmean": float(x.abs().mean().detach().cpu()),
    }


def probability_stats_for_log(prefix: str, prob: torch.Tensor) -> dict[str, float]:
    voxels = (prob > 0.5).flatten(1).sum(1).float()
    return {
        f"{prefix}_prob_mean": float(prob.mean().detach().cpu()),
        f"{prefix}_prob_max": float(prob.max().detach().cpu()),
        f"{prefix}_voxels_gt_0p5_mean": float(voxels.mean().detach().cpu()),
        f"{prefix}_voxels_gt_0p5_min": float(voxels.min().detach().cpu()),
        f"{prefix}_voxels_gt_0p5_max": float(voxels.max().detach().cpu()),
    }


def decoded_foreground_probability(logits: torch.Tensor) -> torch.Tensor:
    logits = logits.float()
    if logits.ndim != 5:
        raise ValueError(f"decoded logits must be [B,C,D,H,W], got {tuple(logits.shape)}")
    channels = int(logits.shape[1])
    if channels == 1:
        return torch.sigmoid(logits)
    if channels == 2:
        return torch.softmax(logits, dim=1)[:, 1:2]
    raise ValueError(f"Unsupported decoded logit channel count: {channels}")


def decoded_mask_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    *,
    pos_weight: float | str | None,
    max_auto_pos_weight: float,
) -> torch.Tensor:
    if logits.shape[1] == 1:
        return weighted_bce_with_logits(
            logits,
            target,
            pos_weight=pos_weight,
            max_auto_pos_weight=max_auto_pos_weight,
        )
    if logits.shape[1] == 2:
        labels = target[:, 0].long()
        weight = None
        if isinstance(pos_weight, str) and pos_weight == "auto":
            pos = target.sum().clamp_min(1.0)
            neg = target.numel() - pos
            fg_weight = (neg / pos).clamp(min=1.0, max=float(max_auto_pos_weight))
            weight = torch.stack([fg_weight.new_ones(()), fg_weight])
        elif pos_weight not in {None, 1, "none"}:
            fg_weight = torch.as_tensor(float(pos_weight), device=logits.device, dtype=logits.dtype)
            weight = torch.stack([fg_weight.new_ones(()), fg_weight])
        return F.cross_entropy(logits.float(), labels, weight=weight)
    raise ValueError(f"Unsupported decoded logit channel count: {logits.shape[1]}")


def build_vertex_queries(
    batch: dict,
    image_index: torch.Tensor,
    sheet_id: torch.Tensor,
    device: torch.device,
    *,
    offsets: list[float],
    max_nodes: int,
    band_half: float,
    clip_voxels: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per pair: the mesh nodes of the prompted sheet (vertex_component == sheet id) and their offsets along the node
    normal; targets = the normalised signed distance to the 3-voxel band ((|t| - band_half) / clip, clamped to [-1, 1]).
    Returns points [P, Q, 3] (zyx), targets [P, Q], valid [P, Q] (padded pairs / nodes are invalid)."""
    pts_all = batch["vertices_zyx"].to(device).float()          # [B, M, 3]
    nrm_all = batch["vertex_normals_zyx"].to(device).float()    # [B, M, 3]
    comp_all = batch["vertex_component"].to(device).long()      # [B, M]
    valid_all = batch["vertex_valid"].to(device).bool()         # [B, M]
    n_off = len(offsets); Q = max_nodes * n_off; P = int(image_index.numel())
    points = torch.zeros((P, Q, 3), device=device); targets = torch.zeros((P, Q), device=device); valid = torch.zeros((P, Q), dtype=torch.bool, device=device)
    off = torch.tensor(offsets, device=device).float()
    tgt_per_off = ((off.abs() - band_half) / clip_voxels).clamp(-1.0, 1.0)
    for p in range(P):
        b = int(image_index[p]); sid = int(sheet_id[p])
        sel = valid_all[b] & (comp_all[b] == sid) & (nrm_all[b].norm(dim=-1) > 0.5)
        idx = sel.nonzero(as_tuple=False).squeeze(-1)
        if idx.numel() == 0:
            continue
        if idx.numel() > max_nodes:
            idx = idx[torch.randperm(idx.numel(), device=device)[:max_nodes]]
        n = idx.numel(); base = pts_all[b, idx]; nrm = nrm_all[b, idx]
        q = (base[:, None, :] + off[None, :, None] * nrm[:, None, :]).reshape(n * n_off, 3)
        t = tgt_per_off[None, :].expand(n, n_off).reshape(n * n_off)
        points[p, :n * n_off] = q; targets[p, :n * n_off] = t; valid[p, :n * n_off] = True
    # queries outside the volume are dropped (the head clamps coordinates; keep the targets honest)
    shape = torch.tensor(batch["component_label"].shape[-3:], device=device).float()
    inside = (points >= -0.5).all(-1) & (points <= shape[None, None, :] - 0.5).all(-1)
    valid = valid & inside
    return points, targets, valid


def build_p2sd_pair_batch(batch: dict, cfg: dict, device: torch.device) -> dict[str, torch.Tensor]:
    p2sd_cfg = cfg.get("p2sd", {})
    pair_cfg = p2sd_cfg.get("pair_sampling", {})
    prompt_comp = p2sd_cfg.get("prompt_composition", {})
    sampler_cfg = prompt_comp.get("sampler", p2sd_cfg.get("prompt", {}))
    requested_pairs = int(pair_cfg.get("pairs_per_step", 0))
    if requested_pairs <= 0 or "component_label" not in batch:
        return default_p2sd_pairs(batch, device)

    component_label = batch["component_label"].long()
    if component_label.ndim == 5 and component_label.shape[1] == 1:
        component_label = component_label[:, 0]
    if component_label.ndim != 4:
        raise ValueError(f"component_label must be [B,D,H,W], got {tuple(component_label.shape)}")
    min_voxels = int(pair_cfg.get("min_voxels_per_sheet", cfg.get("data", {}).get("min_component_voxels", 1)))
    max_components = int(pair_cfg.get("max_components_per_sample", pair_cfg.get("max_components", 0)) or 0)
    component_select_mode = str(pair_cfg.get("component_select_mode", "largest"))
    eligible_b, eligible_sheet, eligible_voxels = collect_eligible_sheets(
        component_label,
        min_voxels,
        max_components_per_sample=max_components,
        component_select_mode=component_select_mode,
    )
    if eligible_b.numel() == 0:
        return default_p2sd_pairs(batch, device)

    replace = bool(pair_cfg.get("sample_with_replacement", True))
    selection_mode = str(pair_cfg.get("selection_mode", "uniform"))
    use_replacement = replace or requested_pairs > eligible_b.numel()
    if selection_mode == "uniform":
        if use_replacement:
            sel = torch.randint(eligible_b.numel(), (requested_pairs,), device=device)
        else:
            sel = torch.randperm(eligible_b.numel(), device=device)[:requested_pairs]
    elif selection_mode == "size_weighted":
        weights = eligible_voxels.float().clamp_min(1)
        sel = torch.multinomial(weights, requested_pairs, replacement=use_replacement)
    elif selection_mode == "balanced_sheets":
        # Round-robin over the (shuffled) eligible (image, sheet) entries. With
        # max_components_per_sample >= 2 and pairs_per_step >= 2 * entries this
        # guarantees every eligible sheet appears at least twice, so each step
        # carries BOTH same-sheet duplicates (consistency loss) and same-image
        # different-sheet pairs (contrast loss). Uniform sampling with
        # replacement makes either group a coin flip per step instead.
        order = torch.randperm(eligible_b.numel(), device=device)
        repeats = (requested_pairs + eligible_b.numel() - 1) // eligible_b.numel()
        sel = order.repeat(repeats)[:requested_pairs]
    else:
        raise ValueError(f"Unsupported p2sd.pair_sampling.selection_mode: {selection_mode}")
    image_index = eligible_b[sel]
    sheet_id = eligible_sheet[sel]
    target_voxels = eligible_voxels[sel]
    selected_labels = component_label.index_select(0, image_index)
    target = selected_labels == sheet_id.view(-1, 1, 1, 1)
    target_mask = target.unsqueeze(1).float()
    prompt_points, prompt_labels = sample_pair_prompts(
        selected_labels,
        target,
        sheet_id,
        n_pos=resolve_positive_point_count(sampler_cfg),
        n_neg=resolve_negative_point_count(sampler_cfg),
        neg_source=str(sampler_cfg.get("neg_source", "background_or_other_sheet")),
        pos_jitter_voxels=int(sampler_cfg.get("pos_jitter_voxels", 0)),
    )
    return {
        "image_index": image_index.long(),
        "sheet_id": sheet_id.long(),
        "target_mask": target_mask,
        "target_voxels": target_voxels.float(),
        "prompt_points": prompt_points,
        "prompt_labels": prompt_labels,
    }


def unique_p2sd_target_masks(
    target_mask: torch.Tensor,
    image_index: torch.Tensor,
    sheet_id: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select one target mask per repeated `(image_index, sheet_id)` key.

    Frozen target AEs run in evaluation/inference mode, so every occurrence of
    the same sheet mask has the same latent. The returned inverse index restores
    original prompt-pair order after a single encode of each unique target.
    """
    pair_count = int(target_mask.shape[0])
    if image_index.ndim != 1 or sheet_id.ndim != 1:
        raise ValueError("image_index and sheet_id must be one-dimensional")
    if image_index.shape[0] != pair_count or sheet_id.shape[0] != pair_count:
        raise ValueError("target masks and target keys must have the same batch length")
    if pair_count == 0:
        return target_mask, torch.empty(0, device=target_mask.device, dtype=torch.long)

    keys = torch.stack((image_index.long(), sheet_id.long()), dim=1)
    unique_keys, inverse = torch.unique(keys, dim=0, sorted=True, return_inverse=True)
    representative = torch.full(
        (unique_keys.shape[0],),
        pair_count,
        device=target_mask.device,
        dtype=torch.long,
    )
    pair_indices = torch.arange(pair_count, device=target_mask.device, dtype=torch.long)
    representative.scatter_reduce_(0, inverse, pair_indices, reduce="amin", include_self=True)
    unique_masks = target_mask.index_select(0, representative)
    if target_mask.is_contiguous(memory_format=torch.channels_last_3d):
        unique_masks = unique_masks.contiguous(memory_format=torch.channels_last_3d)
    return unique_masks, inverse


def default_p2sd_pairs(batch: dict, device: torch.device) -> dict[str, torch.Tensor]:
    image = batch["image"]
    b = image.shape[0]
    sheet_id = batch.get("component_id")
    if sheet_id is None:
        sheet_id = torch.arange(1, b + 1, device=device)
    sheet_id = sheet_id.to(device=device, dtype=torch.long)
    target_mask = batch.get("mask")
    if target_mask is not None:
        target_mask = target_mask.float()
    else:
        component_label = batch.get("component_label")
        if component_label is None:
            raise KeyError("P2SD fallback requires mask or component_label in the batch")
        component_label = component_label.to(device=device)
        if component_label.ndim == 5 and component_label.shape[1] == 1:
            component_label = component_label[:, 0]
        if component_label.ndim != 4:
            raise ValueError(f"component_label must be [B,D,H,W], got {tuple(component_label.shape)}")
        target_mask = (component_label == sheet_id.view(-1, 1, 1, 1)).unsqueeze(1).float()
    return {
        "image_index": torch.arange(b, device=device, dtype=torch.long),
        "sheet_id": sheet_id,
        "target_mask": target_mask,
        "target_voxels": target_mask.flatten(1).sum(1),
        "prompt_points": batch["prompt_points"].float(),
        "prompt_labels": batch["prompt_labels"].long(),
    }


def collect_eligible_sheets(
    component_label: torch.Tensor,
    min_voxels: int,
    *,
    max_components_per_sample: int = 0,
    component_select_mode: str = "largest",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    pair_b = []
    pair_sheet = []
    pair_voxels = []
    for b in range(component_label.shape[0]):
        labels, counts = torch.unique(component_label[b], return_counts=True)
        valid = (labels > 0) & (counts >= int(min_voxels))
        if valid.any():
            sheets = labels[valid]
            sheet_counts = counts[valid]
        else:
            nonzero = labels > 0
            sheets = labels[nonzero]
            sheet_counts = counts[nonzero]
        if sheets.numel() == 0:
            continue
        sheets, sheet_counts = cap_eligible_sheets(
            sheets,
            sheet_counts,
            max_components_per_sample=max_components_per_sample,
            component_select_mode=component_select_mode,
        )
        pair_b.append(torch.full((sheets.numel(),), b, device=component_label.device, dtype=torch.long))
        pair_sheet.append(sheets.long())
        pair_voxels.append(sheet_counts.float())
    if not pair_b:
        empty = torch.empty(0, device=component_label.device, dtype=torch.long)
        empty_float = torch.empty(0, device=component_label.device, dtype=torch.float32)
        return empty, empty, empty_float
    return torch.cat(pair_b, dim=0), torch.cat(pair_sheet, dim=0), torch.cat(pair_voxels, dim=0)


def cap_eligible_sheets(
    sheets: torch.Tensor,
    counts: torch.Tensor,
    *,
    max_components_per_sample: int,
    component_select_mode: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    max_components = int(max_components_per_sample)
    if max_components <= 0 or sheets.numel() <= max_components:
        return sheets, counts
    order = torch.argsort(counts, descending=True)
    sorted_sheets = sheets[order]
    sorted_counts = counts[order]
    if component_select_mode == "largest":
        keep = torch.arange(max_components, device=sheets.device)
    elif component_select_mode == "random":
        keep = torch.randperm(sorted_sheets.numel(), device=sheets.device)[:max_components]
        keep = keep.sort().values
    else:
        raise ValueError(
            "Unsupported p2sd.pair_sampling.component_select_mode: "
            f"{component_select_mode}"
        )
    return sorted_sheets[keep], sorted_counts[keep]


def resolve_positive_point_count(sampler_cfg: dict[str, Any]) -> int:
    """Positive prompt points for this batch, optionally drawn from a range.

    `sampler.n_pos_range: [lo, hi]` makes the model click-count agnostic -- it
    sees 1-point and 8-point prompts in the same run instead of exactly one
    count, which is what lets a single checkpoint be scored at any K.

    Drawn per BATCH, not per pair: `prompt_points` is a dense `[M, P, 3]` tensor
    with no padding mask, so a per-pair count would need one plumbed through the
    prompt MLP and every modulator.
    """

    default = int(sampler_cfg.get("n_pos", sampler_cfg.get("num_positive_points", 1)))
    span = sampler_cfg.get("n_pos_range")
    if not span:
        return default
    values = [int(value) for value in span]
    if len(values) != 2 or values[0] < 1 or values[1] < values[0]:
        raise ValueError(
            "p2sd.prompt_composition.sampler.n_pos_range must be [lo, hi] with "
            f"1 <= lo <= hi, got {span}"
        )
    low, high = values
    weights = sampler_cfg.get("n_pos_weights")
    if weights:
        # Weighted draw over [lo..hi] -- e.g. a fingerprint fine-tune trains
        # K=1-heavy ([0.5, 0.2, 0.2, 0.1] over [1,4]) to match how the
        # instance pipeline actually calls the model (single clicks).
        values = [float(w) for w in weights]
        if len(values) != high - low + 1 or any(w < 0 for w in values) or sum(values) <= 0:
            raise ValueError(
                f"n_pos_weights must be {high - low + 1} non-negative weights "
                f"for n_pos_range {span}, got {weights}")
        draw = torch.multinomial(torch.tensor(values), 1).item()
        return low + int(draw)
    return int(torch.randint(low, high + 1, (1,)).item())


def resolve_negative_point_count(sampler_cfg: dict[str, Any]) -> int:
    """Negative prompt points per batch; ``n_neg_range: [lo, hi]`` draws
    uniformly (lo may be 0 so the model keeps its no-negatives behaviour on a
    fraction of batches). Same per-BATCH constraint as the positive count."""

    default = int(sampler_cfg.get("n_neg", sampler_cfg.get("num_negative_points", 0)))
    span = sampler_cfg.get("n_neg_range")
    if not span:
        return default
    values = [int(value) for value in span]
    if len(values) != 2 or values[0] < 0 or values[1] < values[0]:
        raise ValueError(
            f"n_neg_range must be [lo, hi] with 0 <= lo <= hi, got {span}")
    return int(torch.randint(values[0], values[1] + 1, (1,)).item())


def sample_pair_prompts(
    selected_labels: torch.Tensor,
    target: torch.Tensor,
    sheet_id: torch.Tensor,
    *,
    n_pos: int,
    n_neg: int,
    neg_source: str,
    pos_jitter_voxels: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    points = []
    labels = []
    for i in range(target.shape[0]):
        pos = sample_points_from_mask(target[i], n_pos)
        if pos_jitter_voxels > 0 and pos.numel():
            # Click-noise augmentation for the fingerprint role: deployment
            # clicks come from the FAT binseg foreground, so many land 1-3
            # voxels OFF the sheet (gap or neighbor territory). Jittered
            # training points teach "a near-miss click still belongs to the
            # nearest sheet's fingerprint" -- the ambiguity that chain-merges
            # clusters. Deliberately allowed to leave the sheet.
            jitter = torch.randint(
                -int(pos_jitter_voxels), int(pos_jitter_voxels) + 1,
                pos.shape, device=pos.device, dtype=torch.long)
            bound = torch.tensor(target.shape[1:], device=pos.device, dtype=torch.float32) - 1
            pos = (pos + jitter.float()).clamp(min=torch.zeros(3, device=pos.device), max=bound)
        neg_mask = negative_prompt_mask(selected_labels[i], target[i], sheet_id[i], neg_source)
        neg = sample_points_from_mask(neg_mask, n_neg)
        points.append(torch.cat([pos, neg], dim=0))
        labels.append(torch.cat([
            torch.ones(n_pos, device=target.device, dtype=torch.long),
            torch.zeros(n_neg, device=target.device, dtype=torch.long),
        ], dim=0))
    if not points:
        return (
            torch.empty(0, n_pos + n_neg, 3, device=target.device),
            torch.empty(0, n_pos + n_neg, device=target.device, dtype=torch.long),
        )
    return torch.stack(points, dim=0).float(), torch.stack(labels, dim=0)


def sample_points_from_mask(mask: torch.Tensor, count: int) -> torch.Tensor:
    count = int(count)
    if count <= 0:
        return torch.empty(0, 3, device=mask.device, dtype=torch.float32)
    coords = mask.nonzero(as_tuple=False)
    if coords.numel() == 0:
        shape = torch.tensor(mask.shape, device=mask.device, dtype=torch.float32)
        return (shape.view(1, 3) - 1).clamp_min(0).expand(count, 3) * 0.5
    idx = torch.randint(coords.shape[0], (count,), device=mask.device)
    return coords.index_select(0, idx).float()


def negative_prompt_mask(
    labels: torch.Tensor,
    target: torch.Tensor,
    sheet_id: torch.Tensor,
    neg_source: str,
) -> torch.Tensor:
    if neg_source == "background":
        mask = labels == 0
    elif neg_source == "other_sheet":
        mask = (labels > 0) & (labels != sheet_id)
    elif neg_source == "background_or_other_sheet":
        mask = labels != sheet_id
    else:
        raise ValueError(f"Unsupported prompt_composition.sampler.neg_source: {neg_source}")
    if not mask.any():
        mask = ~target
    if not mask.any():
        mask = torch.ones_like(target, dtype=torch.bool)
    return mask


def sample_mask_clicks(mask: torch.Tensor, count: int) -> torch.Tensor:
    """[M, count, 3] random foreground voxels per pair mask (repeats allowed).

    Feeds the latent-prompt teacher: single random on-sheet clicks, matching
    the pipeline's ``sample_foreground_points`` distribution.
    """
    clicks = []
    for pair_index in range(mask.shape[0]):
        foreground = mask[pair_index, 0].nonzero()
        if foreground.shape[0] == 0:
            clicks.append(mask.new_zeros(count, 3))
            continue
        chosen = torch.randint(foreground.shape[0], (count,), device=foreground.device)
        clicks.append(foreground[chosen].to(mask.dtype))
    return torch.stack(clicks)


def teacher_centroid_latent(
    model,
    image_tokens: torch.Tensor,
    image_coords: torch.Tensor,
    image_context: torch.Tensor,
    image_latent: torch.Tensor,
    clicks: torch.Tensor,
    *,
    image_index: torch.Tensor,
    image_shape: tuple[int, int, int],
) -> torch.Tensor:
    """Average of the model's own no-grad single-click latent predictions.

    One modulator+refiner pass per click column (the expensive image encode is
    shared), averaged in latent space -- the same construction as a pipeline
    fingerprint-cluster centroid, produced by the LIVE model so the
    conditioning distribution tracks the student as it trains.
    """
    with torch.no_grad():
        positive = torch.ones(clicks.shape[0], 1, dtype=torch.long, device=clicks.device)
        latents = []
        for click_column in range(clicks.shape[1]):
            out = model.forward_from_image_context(
                image_tokens, image_coords, image_context,
                clicks[:, click_column:click_column + 1, :].float(),
                positive,
                image_shape=image_shape,
                image_index=image_index,
                image_latent=image_latent,
            )
            latents.append(out["latent"].float())
        return torch.stack(latents).mean(dim=0).detach()


def _compute_metrics_with_optional_ae_recon(
    prediction: np.ndarray,
    target: np.ndarray,
    prompt_zyx,
    metric_cfg,
    ae_recon_prediction: np.ndarray | None,
) -> tuple[dict[str, float], dict[str, float] | None]:
    metrics = compute_p2sd_metrics(
        prediction,
        target,
        prompt_zyx=prompt_zyx,
        cfg=metric_cfg,
    )
    ae_recon_metrics = (
        compute_p2sd_metrics(
            ae_recon_prediction,
            target,
            prompt_zyx=prompt_zyx,
            cfg=metric_cfg,
        )
        if ae_recon_prediction is not None
        else None
    )
    return metrics, ae_recon_metrics


@torch.no_grad()
def evaluate_p2sd(
    model,
    target_ae,
    val_loader,
    cfg,
    run_dir: Path,
    step: int,
    steps_per_epoch: int,
    device,
    dtype,
    fixed_probe_loader=None,
    force_fixed_probes: bool = False,
) -> float:
    model.eval()
    target_ae.eval()
    metrics_cfg = cfg.get("metrics", {})
    metric_cfg = config_from_mapping(metrics_cfg)
    eval_batches = int(metrics_cfg.get("eval_batches", 1))
    val_min_target_voxels = int(metrics_cfg.get("val_min_target_voxels", 0) or 0)
    log_ae_recon = bool(metrics_cfg.get("log_ae_recon", True))
    latent_norm_cfg = cfg.get("p2sd", {}).get("loss", {}).get("latent_normalization", {})
    latent_codec = build_static_latent_codec(
        latent_norm_cfg,
        latent_channels=target_ae.latent_channels,
        device=device,
    )
    metric_rows = []
    ae_recon_metric_rows = []
    first_payload = None
    first_payload_submitted = False
    eval_seen_samples = 0
    eval_excluded_small_target_samples = 0
    dispatch_settings = MetricDispatchSettings.from_metrics_config(metrics_cfg)
    reporter = EvalSampleReporter(
        run_dir=run_dir,
        step=step,
        task="p2sd",
        metrics_cfg=metrics_cfg,
    )

    def consume_metrics(
        payload: dict[str, Any],
        result: tuple[dict[str, float], dict[str, float] | None],
    ) -> None:
        nonlocal first_payload
        metrics, ae_recon_metrics = result
        metric_rows.append(metrics)
        reporter.add(
            identity=payload["identity"],
            metrics=metrics,
            volumes=payload.get("volumes"),
        )
        if ae_recon_metrics is not None:
            ae_recon_metric_rows.append(ae_recon_metrics)
        if payload["first_payload"] is not None:
            first_payload = {
                **payload["first_payload"],
                "metrics": metrics,
                "ae_recon_metrics": ae_recon_metrics,
            }

    with OrderedMetricDispatcher[dict[str, Any], tuple[dict[str, float], dict[str, float] | None]](
        dispatch_settings,
    ) as metric_dispatcher:
        for batch_index, batch in enumerate(val_loader):
            if batch_index >= eval_batches:
                break
            batch = move_batch(batch, device)
            image = batch["image"].float()
            mask = batch["mask"].float()
            with autocast_context(device, dtype):
                out = model(image, batch["prompt_points"].float(), batch["prompt_labels"])
                decoded_logits = target_ae.decode(latent_codec.raw_prediction(out["latent"]))
                ae_recon_logits = None
                if log_ae_recon:
                    ae_recon_logits = target_ae.decode(target_ae.encode(mask))
            prob = decoded_foreground_probability(decoded_logits).cpu().numpy()
            ae_recon_prob = (
                decoded_foreground_probability(ae_recon_logits).cpu().numpy()
                if ae_recon_logits is not None else None
            )
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
                        image=image[sample_index, 0].float().cpu().numpy(),
                    )
                if first_payload is None and not first_payload_submitted:
                    first_payload_submitted = True
                    payload["first_payload"] = {
                        "image": image[sample_index, 0].float().cpu().numpy(),
                        "gt": gt[sample_index, 0],
                        "pred": prob[sample_index, 0],
                        "ae_recon": (
                            None if ae_recon_prob is None
                            else ae_recon_prob[sample_index, 0]
                        ),
                        "prompt_zyx": batch["prompt_zyx"][sample_index],
                        "prompt_points_zyx": batch["prompt_points"][sample_index].detach().float().cpu().numpy(),
                        "prompt_labels": batch["prompt_labels"][sample_index].detach().long().cpu().numpy(),
                        "case_id": batch["case_id"][sample_index],
                        "component_id": int(batch["component_id"][sample_index].detach().item()),
                        "crop_start": tuple(int(value) for value in batch["crop_start"][sample_index]),
                        "target_voxels": target_voxels,
                    }
                for completed_payload, result in metric_dispatcher.submit(
                    payload,
                    _compute_metrics_with_optional_ae_recon,
                    prob[sample_index, 0],
                    gt[sample_index, 0],
                    batch["prompt_zyx"][sample_index],
                    metric_cfg,
                    None if ae_recon_prob is None else ae_recon_prob[sample_index, 0],
                ):
                    consume_metrics(completed_payload, result)
        for completed_payload, result in metric_dispatcher.drain():
            consume_metrics(completed_payload, result)
    if not metric_rows:
        raise RuntimeError(
            "No validation samples met metrics.val_min_target_voxels="
            f"{val_min_target_voxels} across {eval_seen_samples} sampled cases")
    metrics = average_metric_rows(metric_rows)
    tail_stats = reporter.finalize()
    ae_recon_metrics = average_metric_rows(ae_recon_metric_rows) if ae_recon_metric_rows else {}
    row = {
        "step": step,
        "epoch": float(step / max(1, steps_per_epoch)),
        "split": "val",
        "eval_samples": len(metric_rows),
        "eval_seen_samples": eval_seen_samples,
        "eval_excluded_small_target_samples": eval_excluded_small_target_samples,
        "val_min_target_voxels": val_min_target_voxels,
        **metrics,
        **tail_stats,
        **{f"ae_recon_{key}": value for key, value in ae_recon_metrics.items()},
    }
    append_jsonl(run_dir / "metrics.jsonl", row)
    viz_cfg = cfg.get("visualization", {})
    milestone_epoch = resolve_visualization_milestone_epoch(
        viz_cfg,
        step=step,
        steps_per_epoch=steps_per_epoch,
    )
    if bool(viz_cfg.get("save_png", True)) and first_payload is not None:
        _save_p2sd_visual_pngs(
            run_dir / "viz" / f"p2sd_step_{step:06d}.png",
            payload=first_payload,
            metrics_cfg=metrics_cfg,
            meta={
                "step": step,
                "task": "p2sd",
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
            stem_prefixes=("p2sd_step_",),
        )
        if milestone_epoch is not None:
            _save_p2sd_visual_pngs(
                run_dir / "viz" / "milestones" / f"p2sd_epoch_{milestone_epoch:06d}.png",
                payload=first_payload,
                metrics_cfg=metrics_cfg,
                meta={
                    "step": step,
                    "epoch": milestone_epoch,
                    "task": "p2sd_milestone",
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
            _save_p2sd_3d_visualization(
                run_dir / "viz3d" / f"p2sd_step_{step:06d}",
                payload=first_payload,
                metrics_cfg=metrics_cfg,
                meta={"step": step, "task": "p2sd"},
                viz_cfg=viz_cfg,
            )
        if save_milestone_3d:
            _save_p2sd_3d_visualization(
                run_dir / "viz3d" / "milestones" / f"p2sd_epoch_{milestone_epoch:06d}",
                payload=first_payload,
                metrics_cfg=metrics_cfg,
                meta={"step": step, "epoch": milestone_epoch, "task": "p2sd_milestone"},
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
            stem_prefixes=("p2sd_step_",),
        )
    should_run_fixed_probes = force_fixed_probes or should_evaluate_fixed_probes(
        viz_cfg,
        step=step,
        steps_per_epoch=steps_per_epoch,
        milestone_epoch=milestone_epoch,
    )
    if fixed_probe_loader is not None and should_run_fixed_probes:
        _evaluate_fixed_p2sd_probes(
            model,
            target_ae,
            fixed_probe_loader,
            cfg=cfg,
            run_dir=run_dir,
            step=step,
            steps_per_epoch=steps_per_epoch,
            device=device,
            dtype=dtype,
            metric_cfg=metric_cfg,
            latent_codec=latent_codec,
            milestone_epoch=milestone_epoch,
        )
    return float(metrics.get("quality_composite", metrics.get("dice", 0.0)))


def _save_p2sd_visual_pngs(
    path: Path,
    *,
    payload: dict[str, Any],
    metrics_cfg: dict[str, Any],
    meta: dict[str, Any],
) -> None:
    threshold = float(metrics_cfg.get("threshold", 0.5))
    tolerance = _max_metric_tolerance(metrics_cfg)
    metrics = _payload_visual_metrics(payload)
    visual_state = prepare_p2sd_visualization_state(
        image=payload["image"],
        gt=payload["gt"],
        pred=payload["pred"],
        ae_recon=payload.get("ae_recon"),
        prompt_points_zyx=payload.get("prompt_points_zyx"),
        prompt_labels=payload.get("prompt_labels"),
        prompt_zyx=payload.get("prompt_zyx"),
        threshold=threshold,
        tolerance_voxels=tolerance,
    )
    save_p2sd_diagnostic_png(
        path,
        image=payload["image"],
        gt=payload["gt"],
        pred=payload["pred"],
        ae_recon=payload.get("ae_recon"),
        prompt_points_zyx=payload.get("prompt_points_zyx"),
        prompt_labels=payload.get("prompt_labels"),
        prompt_zyx=payload.get("prompt_zyx"),
        metrics=metrics,
        meta=meta,
        threshold=threshold,
        tolerance_voxels=tolerance,
        visual_state=visual_state,
    )
    save_p2sd_projection_png(
        path.with_name(path.stem + "_projection.png"),
        image=payload["image"],
        gt=payload["gt"],
        pred=payload["pred"],
        ae_recon=payload.get("ae_recon"),
        prompt_points_zyx=payload.get("prompt_points_zyx"),
        prompt_labels=payload.get("prompt_labels"),
        prompt_zyx=payload.get("prompt_zyx"),
        metrics=metrics,
        meta=meta,
        threshold=threshold,
        tolerance_voxels=tolerance,
        visual_state=visual_state,
    )


def _save_p2sd_3d_visualization(
    path_prefix: Path,
    *,
    payload: dict[str, Any],
    metrics_cfg: dict[str, Any],
    meta: dict[str, Any],
    viz_cfg: dict[str, Any],
) -> None:
    meta = {
        **meta,
        "prompt_points_zyx": _to_jsonable_points(payload.get("prompt_points_zyx")),
        "prompt_labels": _to_jsonable_labels(payload.get("prompt_labels")),
    }
    save_p2sd_3d_artifacts(
        path_prefix,
        image=payload["image"],
        gt=payload["gt"],
        pred=payload["pred"],
        ae_recon=payload.get("ae_recon"),
        prompt_zyx=payload.get("prompt_zyx"),
        metrics=_payload_visual_metrics(payload),
        meta=meta,
        threshold=float(metrics_cfg.get("threshold", 0.5)),
        max_points=int(viz_cfg.get("save_3d_max_points", 200_000)),
        save_npz=bool(viz_cfg.get("save_3d_npz", False)),
        save_ply=bool(viz_cfg.get("save_3d_ply", False)),
    )


@torch.no_grad()
def _evaluate_fixed_p2sd_probes(
    model,
    target_ae,
    probe_loader,
    *,
    cfg: dict[str, Any],
    run_dir: Path,
    step: int,
    steps_per_epoch: int,
    device,
    dtype,
    metric_cfg,
    latent_codec,
    milestone_epoch: int | None,
) -> None:
    """Evaluate fixed sheets across prompt variants without changing checkpoint score."""

    model.eval()
    target_ae.eval()
    metrics_cfg = cfg.get("metrics", {})
    viz_cfg = cfg.get("visualization", {})
    probe_cfg = viz_cfg.get("probes", {})
    log_ae_recon = bool(metrics_cfg.get("log_ae_recon", True))
    min_target_voxels = int(metrics_cfg.get("val_min_target_voxels", 0) or 0)
    prompt_rows: list[dict[str, Any]] = []
    expected_prompt_counts = _fixed_probe_prompt_counts(probe_loader)
    pending_prediction_masks: dict[str, list[np.ndarray]] = {}
    pairwise_prediction_dice: dict[str, dict[str, float]] = {}
    excluded_small_target = 0
    dispatch_settings = MetricDispatchSettings.from_metrics_config(metrics_cfg)

    def consume_metrics(payload: dict[str, Any], sample_metrics: dict[str, float]) -> None:
        probe_id = payload["probe_id"]
        row = {
            **payload["row"],
            **sample_metrics,
        }
        prompt_rows.append(row)
        masks_for_probe = pending_prediction_masks.setdefault(probe_id, [])
        masks_for_probe.append(payload["prediction"] > float(metrics_cfg.get("threshold", 0.5)))
        expected_count = expected_prompt_counts.get(probe_id)
        if expected_count is not None and len(masks_for_probe) == expected_count:
            pairwise_prediction_dice[probe_id] = pairwise_prediction_dice_stats(masks_for_probe)
            del pending_prediction_masks[probe_id]

    with OrderedMetricDispatcher[dict[str, Any], dict[str, float]](dispatch_settings) as metric_dispatcher:
        for batch in probe_loader:
            batch = move_batch(batch, device)
            image = batch["image"].float()
            mask = batch["mask"].float()
            with autocast_context(device, dtype):
                out = model(image, batch["prompt_points"].float(), batch["prompt_labels"])
                decoded_logits = target_ae.decode(latent_codec.raw_prediction(out["latent"]))
            prob = decoded_foreground_probability(decoded_logits).cpu().numpy()
            gt = mask.float().cpu().numpy()
            for sample_index in range(prob.shape[0]):
                target_voxels = int(mask[sample_index, 0].sum().detach().item())
                if target_voxels < min_target_voxels:
                    excluded_small_target += 1
                    continue
                probe_id = str(batch["probe_id"][sample_index])
                prompt_labels = batch["prompt_labels"][sample_index]
                payload = {
                    "probe_id": probe_id,
                    "prediction": prob[sample_index, 0],
                    "row": {
                        "probe_id": probe_id,
                        "prompt_set_id": str(batch["prompt_set_id"][sample_index]),
                        "case_id": str(batch["case_id"][sample_index]),
                        "component_id": int(batch["component_id"][sample_index].detach().item()),
                        "crop_start": [int(value) for value in batch["crop_start"][sample_index]],
                        "target_voxels": target_voxels,
                        "prompt_point_count": int(prompt_labels.shape[0]),
                        "positive_prompt_count": int((prompt_labels > 0).sum().detach().item()),
                    },
                }
                for completed_payload, sample_metrics in metric_dispatcher.submit(
                    payload,
                    compute_p2sd_metrics,
                    prob[sample_index, 0],
                    gt[sample_index, 0],
                    prompt_zyx=batch["prompt_zyx"][sample_index],
                    cfg=metric_cfg,
                ):
                    consume_metrics(completed_payload, sample_metrics)
        for completed_payload, sample_metrics in metric_dispatcher.drain():
            consume_metrics(completed_payload, sample_metrics)

    if not prompt_rows:
        raise RuntimeError(
            "No fixed probes met metrics.val_min_target_voxels="
            f"{min_target_voxels}; excluded={excluded_small_target}")
    tolerance = _max_metric_tolerance(metrics_cfg)
    per_sheet, summary = aggregate_fixed_probe_rows(
        prompt_rows,
        prediction_masks=pending_prediction_masks,
        pairwise_prediction_dice=pairwise_prediction_dice,
        tolerance_voxels=tolerance,
    )
    epoch = float(step / max(1, steps_per_epoch))
    append_jsonl(run_dir / "probe_metrics.jsonl", {
        "step": step,
        "epoch": epoch,
        "split": "fixed_probe",
        "probe_prompt_rows": len(prompt_rows),
        "probe_excluded_small_target_samples": excluded_small_target,
        "val_min_target_voxels": min_target_voxels,
        **summary,
    })
    detail_path = run_dir / "probe_metrics" / f"p2sd_step_{step:06d}.json"
    write_json(detail_path, {
        "step": step,
        "epoch": epoch,
        "prompt_rows": prompt_rows,
        "per_sheet": per_sheet,
        "summary": summary,
    })
    prune_visualization_groups(
        detail_path.parent,
        keep_latest=int(probe_cfg.get("keep_latest_metrics", 8)),
        stem_prefixes=("p2sd_step_",),
    )
    if milestone_epoch is not None:
        write_json(detail_path.parent / "milestones" / f"p2sd_epoch_{milestone_epoch:06d}.json", {
            "step": step,
            "epoch": epoch,
            "prompt_rows": prompt_rows,
            "per_sheet": per_sheet,
            "summary": summary,
        })

    save_png = bool(probe_cfg.get("save_png", viz_cfg.get("save_png", True)))
    normal_count = int(probe_cfg.get("render_prompt_sets_per_sheet", 2))
    milestone_count = int(probe_cfg.get("milestone_render_prompt_sets_per_sheet", normal_count))
    normal_selections = _select_probe_visual_rows(
        prompt_rows,
        max_per_probe=normal_count,
        max_probes=_probe_visual_max_sheets(probe_cfg, "max_rendered_sheets"),
    )
    milestone_selections = _select_probe_visual_rows(
        prompt_rows,
        max_per_probe=milestone_count,
        max_probes=_probe_visual_max_sheets(probe_cfg, "milestone_max_rendered_sheets"),
    )
    render_selections = milestone_selections if milestone_epoch is not None else normal_selections
    event_epoch = int(round(epoch))
    png_epoch_dir = run_dir / "viz" / "probes" / f"epoch_{event_epoch:06d}"
    nifti_epoch_dir = run_dir / "viz3d" / "probes" / f"epoch_{event_epoch:06d}"
    if save_png and render_selections:
        _render_fixed_probe_visuals(
            model,
            target_ae,
            probe_loader,
            selections=render_selections,
            per_sheet_by_id={str(item["probe_id"]): item for item in per_sheet},
            prompt_rows_by_key={
                (str(row["probe_id"]), str(row["prompt_set_id"])): row
                for row in prompt_rows
            },
            cfg=cfg,
            run_dir=run_dir,
            step=step,
            epoch=epoch,
            device=device,
            dtype=dtype,
            metric_cfg=metric_cfg,
            latent_codec=latent_codec,
            log_ae_recon=log_ae_recon,
            milestone_epoch=milestone_epoch,
            png_epoch_dir=png_epoch_dir,
            nifti_epoch_dir=nifti_epoch_dir,
        )
    milestone_epochs = {int(round(float(value))) for value in viz_cfg.get("milestone_epochs", [])}
    if milestone_epoch is not None:
        milestone_epochs.add(int(milestone_epoch))
    for gallery_root in (png_epoch_dir.parent, nifti_epoch_dir.parent):
        prune_visualization_epoch_directories(
            gallery_root,
            keep_latest=int(probe_cfg.get("keep_latest_png", 4)),
            preserve_epochs=milestone_epochs,
        )


def _fixed_probe_prompt_counts(probe_loader) -> dict[str, int]:
    """Return the known prompt-variant count for each fixed sheet."""

    counts: dict[str, int] = {}
    for probe_case, _ in getattr(probe_loader.dataset, "entries", ()):
        probe_id = str(probe_case.probe_id)
        counts[probe_id] = counts.get(probe_id, 0) + 1
    return counts


def _render_fixed_probe_visuals(
    model,
    target_ae,
    probe_loader,
    *,
    selections: dict[tuple[str, str], str],
    per_sheet_by_id: dict[str, dict[str, Any]],
    prompt_rows_by_key: dict[tuple[str, str], dict[str, Any]],
    cfg: dict[str, Any],
    run_dir: Path,
    step: int,
    epoch: float,
    device,
    dtype,
    metric_cfg,
    latent_codec,
    log_ae_recon: bool,
    milestone_epoch: int | None,
    png_epoch_dir: Path,
    nifti_epoch_dir: Path,
) -> None:
    """Re-evaluate only selected probe variants and render each one immediately."""

    metrics_cfg = cfg.get("metrics", {})
    probe_cfg = cfg.get("visualization", {}).get("probes", {})
    task = "p2sd_fixed_probe_milestone" if milestone_epoch is not None else "p2sd_fixed_probe"
    for batch in probe_loader:
        selected_indices = [
            index
            for index, (probe_id, prompt_set_id) in enumerate(zip(batch["probe_id"], batch["prompt_set_id"]))
            if (str(probe_id), str(prompt_set_id)) in selections
        ]
        if not selected_indices:
            continue
        batch = move_batch(_select_batch_items(batch, selected_indices), device)
        image = batch["image"].float()
        mask = batch["mask"].float()
        with autocast_context(device, dtype):
            out = model(image, batch["prompt_points"].float(), batch["prompt_labels"])
            decoded_logits = target_ae.decode(latent_codec.raw_prediction(out["latent"]))
            ae_recon_logits = target_ae.decode(target_ae.encode(mask)) if log_ae_recon else None
        prob = decoded_foreground_probability(decoded_logits).cpu().numpy()
        ae_recon_prob = (
            decoded_foreground_probability(ae_recon_logits).cpu().numpy()
            if ae_recon_logits is not None else None
        )
        gt = mask.float().cpu().numpy()
        for sample_index in range(prob.shape[0]):
            probe_id = str(batch["probe_id"][sample_index])
            prompt_set_id = str(batch["prompt_set_id"][sample_index])
            key = (probe_id, prompt_set_id)
            row = prompt_rows_by_key[key]
            prompt_zyx = batch["prompt_zyx"][sample_index]
            ae_recon_metrics = None
            if ae_recon_prob is not None:
                ae_recon_metrics = compute_p2sd_metrics(
                    ae_recon_prob[sample_index, 0],
                    gt[sample_index, 0],
                    prompt_zyx=prompt_zyx,
                    cfg=metric_cfg,
                )
            payload = {
                "image": image[sample_index, 0].detach().float().cpu().numpy(),
                "gt": gt[sample_index, 0],
                "pred": prob[sample_index, 0],
                "ae_recon": None if ae_recon_prob is None else ae_recon_prob[sample_index, 0],
                "prompt_zyx": prompt_zyx,
                "prompt_points_zyx": batch["prompt_points"][sample_index].detach().float().cpu().numpy(),
                "prompt_labels": batch["prompt_labels"][sample_index].detach().long().cpu().numpy(),
                "case_id": row["case_id"],
                "component_id": row["component_id"],
                "crop_start": tuple(row["crop_start"]),
                "target_voxels": row["target_voxels"],
                "probe_id": probe_id,
                "prompt_set_id": prompt_set_id,
                "metrics": {key: value for key, value in row.items() if isinstance(value, (int, float))},
                "ae_recon_metrics": ae_recon_metrics,
            }
            role = selections[key]
            probe_dir = png_epoch_dir / role / _safe_path_token(probe_id)
            _save_p2sd_visual_pngs(
                probe_dir / f"{_safe_path_token(prompt_set_id)}.png",
                payload=payload,
                metrics_cfg=metrics_cfg,
                meta={
                    "step": step,
                    "epoch": epoch,
                    "task": task,
                    "selection_role": role,
                    "case_id": payload["case_id"],
                    "component_id": payload["component_id"],
                    "crop_start": payload["crop_start"],
                    "target_voxels": payload["target_voxels"],
                    "probe_id": probe_id,
                    "prompt_set_id": prompt_set_id,
                    "per_sheet": per_sheet_by_id[probe_id],
                },
            )
            if bool(probe_cfg.get("save_3d", True)):
                _save_p2sd_3d_visualization(
                    nifti_epoch_dir / role / _safe_path_token(probe_id) / _safe_path_token(prompt_set_id),
                    payload=payload,
                    metrics_cfg=metrics_cfg,
                    meta={
                        "step": step,
                        "epoch": epoch,
                        "task": task,
                        "selection_role": role,
                        "probe_id": probe_id,
                        "prompt_set_id": prompt_set_id,
                    },
                    viz_cfg=cfg.get("visualization", {}),
                )


def _select_batch_items(batch: dict[str, Any], indices: list[int]) -> dict[str, Any]:
    """Keep only selected fixed-probe samples before moving them to the GPU."""

    return {
        key: value[indices] if isinstance(value, torch.Tensor) else [value[index] for index in indices]
        for key, value in batch.items()
    }


def _select_probe_visual_rows(
    rows: list[dict[str, Any]],
    *,
    max_per_probe: int,
    max_probes: int | None = None,
) -> dict[tuple[str, str], str]:
    """Select compact probe identities; full volume arrays are never retained."""

    if max_per_probe <= 0:
        return {}
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["probe_id"]), []).append(row)
    probe_ids = _select_probe_ids_for_visualization(grouped, max_probes=max_probes)
    anchor_count = min(len(probe_ids), max(1, max_probes // 2)) if max_probes else len(probe_ids)
    anchor_ids = set(probe_ids[:anchor_count])
    selected: dict[tuple[str, str], str] = {}
    for probe_id in probe_ids:
        group = grouped[probe_id]
        candidates = [group[0]]
        candidates.extend(sorted(
            group[1:],
            key=lambda row: (
                _probe_visual_metric(row, "quality_composite"),
                -_probe_visual_metric(row, "floating_artifact_fraction_tau2"),
            ),
        ))
        for row in candidates:
            key = (probe_id, str(row["prompt_set_id"]))
            selected[key] = "anchor" if probe_id in anchor_ids else "worst"
            if sum(1 for selected_probe_id, _ in selected if selected_probe_id == probe_id) >= max_per_probe:
                break
    return selected


def _select_probe_visual_payloads(
    payloads: list[dict[str, Any]],
    *,
    max_per_probe: int,
    max_probes: int | None = None,
) -> list[dict[str, Any]]:
    if max_per_probe <= 0:
        return []
    grouped: dict[str, list[dict[str, Any]]] = {}
    for payload in payloads:
        grouped.setdefault(str(payload["probe_id"]), []).append(payload)
    probe_ids = _select_probe_ids_for_visualization(grouped, max_probes=max_probes)
    anchor_count = min(len(probe_ids), max(1, max_probes // 2)) if max_probes else len(probe_ids)
    anchor_ids = set(probe_ids[:anchor_count])
    selected = []
    for probe_id in probe_ids:
        group = grouped[probe_id]
        candidates = [group[0]]
        candidates.extend(sorted(
            group[1:],
            key=lambda payload: (
                _probe_visual_metric(payload, "quality_composite"),
                -_probe_visual_metric(payload, "floating_artifact_fraction_tau2"),
            ),
        ))
        unique_prompt_sets = set()
        for payload in candidates:
            prompt_set_id = str(payload["prompt_set_id"])
            if prompt_set_id in unique_prompt_sets:
                continue
            selected.append(payload)
            payload["_visual_selection_role"] = "anchor" if probe_id in anchor_ids else "worst"
            unique_prompt_sets.add(prompt_set_id)
            if len(unique_prompt_sets) >= max_per_probe:
                break
    return selected


def _probe_visual_dir_name(payload: dict[str, Any]) -> str:
    role = str(payload.get("_visual_selection_role", "probe"))
    return f"{role}_{_safe_path_token(payload['probe_id'])}"


def _probe_visual_max_sheets(probe_cfg: dict[str, Any], key: str) -> int | None:
    value = probe_cfg.get(key)
    if value is None:
        return None
    return max(0, int(value))


def _select_probe_ids_for_visualization(
    grouped: dict[str, list[dict[str, Any]]],
    *,
    max_probes: int | None,
) -> list[str]:
    """Keep stable anchors while exposing the worst current probe failures."""

    probe_ids = list(grouped)
    if max_probes is None or max_probes >= len(probe_ids):
        return probe_ids
    if max_probes <= 0:
        return []
    anchor_count = min(len(probe_ids), max(1, max_probes // 2))
    anchors = probe_ids[:anchor_count]
    anchor_ids = set(anchors)
    worst_first = sorted(
        (probe_id for probe_id in probe_ids if probe_id not in anchor_ids),
        key=lambda probe_id: _probe_visual_failure_key(probe_id, grouped[probe_id]),
    )
    return anchors + worst_first[:max_probes - len(anchors)]


def _probe_visual_failure_key(
    probe_id: str,
    payloads: list[dict[str, Any]],
) -> tuple[float, float, str]:
    quality = min(_probe_visual_metric(payload, "quality_composite") for payload in payloads)
    floating_artifact = max(
        _probe_visual_metric(payload, "floating_artifact_fraction_tau2")
        for payload in payloads
    )
    return quality, -floating_artifact, probe_id


def _probe_visual_metric(item: dict[str, Any], name: str) -> float:
    """Read a metric from either a rendered payload or a lightweight row."""

    metrics = item.get("metrics")
    source = metrics if isinstance(metrics, dict) else item
    return float(source.get(name, 0.0))


def _payload_visual_metrics(payload: dict[str, Any]) -> dict[str, float]:
    return {
        **payload["metrics"],
        **{
            f"ae_recon_{key}": value
            for key, value in (payload.get("ae_recon_metrics") or {}).items()
        },
    }


def _max_metric_tolerance(metrics_cfg: dict[str, Any]) -> int:
    tolerances = metrics_cfg.get("tolerance_voxels", [2])
    if isinstance(tolerances, (list, tuple)):
        return max(int(value) for value in tolerances) if tolerances else 0
    return int(tolerances)


def _safe_path_token(value: Any) -> str:
    text = str(value)
    return "".join(char if char.isalnum() or char in {"-", "_", "."} else "_" for char in text)


def _to_jsonable_points(points: Any) -> list[list[float]]:
    if points is None:
        return []
    return np.asarray(points, dtype=np.float32).tolist()


def _to_jsonable_labels(labels: Any) -> list[int]:
    if labels is None:
        return []
    return [int(value) for value in np.asarray(labels, dtype=np.int64).reshape(-1)]


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
