"""Aggregation helpers for fixed multi-prompt P2SD evaluation probes."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np

from vesuvius_p2sd.eval.metrics import dice


def aggregate_fixed_probe_rows(
    rows: list[dict[str, Any]],
    *,
    prediction_masks: dict[str, list[np.ndarray]] | None = None,
    pairwise_prediction_dice: dict[str, dict[str, float]] | None = None,
    tolerance_voxels: int = 2,
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    """Aggregate prompt variants into one reproducible score per fixed sheet.

    ``pairwise_prediction_dice`` lets a caller stream PS320 predictions: retain
    masks only until all prompt variants for one sheet have been compared, then
    pass the small scalar summary here instead of every full-resolution mask.
    """

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["probe_id"])].append(row)
    per_sheet = [
        _aggregate_sheet_rows(
            probe_id,
            group,
            prediction_masks=(prediction_masks or {}).get(probe_id, []),
            pairwise_prediction_dice=(pairwise_prediction_dice or {}).get(probe_id),
            tolerance_voxels=tolerance_voxels,
        )
        for probe_id, group in sorted(grouped.items())
    ]
    summary = _aggregate_probe_sheets(per_sheet, tolerance_voxels=tolerance_voxels)
    return per_sheet, summary


def _aggregate_sheet_rows(
    probe_id: str,
    rows: list[dict[str, Any]],
    *,
    prediction_masks: list[np.ndarray],
    pairwise_prediction_dice: dict[str, float] | None,
    tolerance_voxels: int,
) -> dict[str, Any]:
    first = rows[0]
    tau = int(tolerance_voxels)
    output: dict[str, Any] = {
        "probe_id": probe_id,
        "case_id": str(first["case_id"]),
        "component_id": int(first["component_id"]),
        "crop_start": [int(value) for value in first["crop_start"]],
        "target_voxels": float(first["target_voxels"]),
        "prompt_count": float(len(rows)),
        "prompt_set_ids": [str(row["prompt_set_id"]) for row in rows],
    }
    _append_high_is_good(output, rows, "quality_composite", prefix="quality_composite")
    _append_high_is_good(output, rows, f"tolerant_f1_tau{tau}", prefix=f"tolerant_f1_tau{tau}")
    _append_high_is_good(
        output,
        rows,
        f"prompt_component_gt_recall_tau{tau}",
        prefix=f"prompt_component_gt_recall_tau{tau}",
    )
    _append_low_is_good(
        output,
        rows,
        f"floating_artifact_fraction_tau{tau}",
        prefix=f"floating_artifact_fraction_tau{tau}",
    )
    _append_low_is_good(
        output,
        rows,
        f"floating_artifact_component_count_tau{tau}",
        prefix=f"floating_artifact_component_count_tau{tau}",
    )
    _append_low_is_good(output, rows, "component_num_components", prefix="component_num_components")
    _append_low_is_good(
        output,
        rows,
        "fast_topology_sheet_component_count",
        prefix="fast_topology_sheet_component_count",
    )
    _append_low_is_good(output, rows, "border_artifact_fraction", prefix="border_artifact_fraction")
    _append_low_is_good(output, rows, "max_missing_block_fraction", prefix="max_missing_block_fraction")
    _append_high_is_good(
        output,
        rows,
        "kaggle_surface_dice_tau2",
        prefix="kaggle_surface_dice_tau2",
    )
    _append_high_is_good(output, rows, "kaggle_voi_score", prefix="kaggle_voi_score")
    _append_high_is_good(
        output,
        rows,
        "kaggle_toposcore",
        prefix="kaggle_toposcore",
    )
    _append_high_is_good(
        output,
        rows,
        "kaggle_prompt_component_score",
        prefix="kaggle_prompt_component_score",
    )
    output.update(pairwise_prediction_dice or pairwise_prediction_dice_stats(prediction_masks))
    return output


def _append_high_is_good(
    output: dict[str, Any],
    rows: list[dict[str, Any]],
    metric: str,
    *,
    prefix: str,
) -> None:
    values = _metric_values(rows, metric)
    if values.size == 0:
        return
    output[f"{prefix}_mean"] = float(values.mean())
    output[f"{prefix}_prompt_p25"] = float(np.quantile(values, 0.25))
    output[f"{prefix}_prompt_min"] = float(values.min())
    output[f"{prefix}_prompt_std"] = float(values.std())


def _append_low_is_good(
    output: dict[str, Any],
    rows: list[dict[str, Any]],
    metric: str,
    *,
    prefix: str,
) -> None:
    values = _metric_values(rows, metric)
    if values.size == 0:
        return
    output[f"{prefix}_mean"] = float(values.mean())
    output[f"{prefix}_prompt_p75"] = float(np.quantile(values, 0.75))
    output[f"{prefix}_prompt_max"] = float(values.max())
    output[f"{prefix}_prompt_std"] = float(values.std())


def _metric_values(rows: list[dict[str, Any]], metric: str) -> np.ndarray:
    return np.asarray([float(row[metric]) for row in rows if metric in row], dtype=np.float64)


def pairwise_prediction_dice_stats(prediction_masks: list[np.ndarray]) -> dict[str, float]:
    """Summarize agreement between all prompt-conditioned masks for one sheet."""

    if len(prediction_masks) < 2:
        return {
            "prompt_prediction_pairwise_dice_mean": 1.0,
            "prompt_prediction_pairwise_dice_min": 1.0,
            "prompt_prediction_pairwise_dice_max": 1.0,
            "prompt_prediction_pairwise_dice_std": 0.0,
            "prompt_prediction_pairwise_dice_pair_count": 0.0,
        }
    values = []
    for left in range(len(prediction_masks)):
        for right in range(left + 1, len(prediction_masks)):
            values.append(dice(prediction_masks[left], prediction_masks[right]))
    values_array = np.asarray(values, dtype=np.float64)
    if values_array.size == 0:
        return {
            "prompt_prediction_pairwise_dice_mean": 1.0,
            "prompt_prediction_pairwise_dice_min": 1.0,
            "prompt_prediction_pairwise_dice_max": 1.0,
            "prompt_prediction_pairwise_dice_std": 0.0,
            "prompt_prediction_pairwise_dice_pair_count": 0.0,
        }
    return {
        "prompt_prediction_pairwise_dice_mean": float(values_array.mean()),
        "prompt_prediction_pairwise_dice_min": float(values_array.min()),
        "prompt_prediction_pairwise_dice_max": float(values_array.max()),
        "prompt_prediction_pairwise_dice_std": float(values_array.std()),
        "prompt_prediction_pairwise_dice_pair_count": float(values_array.size),
    }


def _aggregate_probe_sheets(
    per_sheet: list[dict[str, Any]],
    *,
    tolerance_voxels: int,
) -> dict[str, float]:
    tau = int(tolerance_voxels)
    summary: dict[str, float] = {
        "probe_sheet_count": float(len(per_sheet)),
        "probe_prompt_count": float(sum(row["prompt_count"] for row in per_sheet)),
    }
    _append_macro(summary, per_sheet, "quality_composite_mean")
    _append_macro(summary, per_sheet, "quality_composite_prompt_min")
    _append_macro(summary, per_sheet, f"tolerant_f1_tau{tau}_mean")
    _append_macro(summary, per_sheet, f"tolerant_f1_tau{tau}_prompt_min")
    _append_macro(summary, per_sheet, f"floating_artifact_fraction_tau{tau}_mean")
    _append_macro(summary, per_sheet, f"floating_artifact_fraction_tau{tau}_prompt_max")
    _append_macro(summary, per_sheet, f"floating_artifact_component_count_tau{tau}_mean")
    _append_macro(summary, per_sheet, f"floating_artifact_component_count_tau{tau}_prompt_max")
    _append_macro(summary, per_sheet, "fast_topology_sheet_component_count_mean")
    _append_macro(summary, per_sheet, "fast_topology_sheet_component_count_prompt_max")
    _append_macro(summary, per_sheet, "prompt_prediction_pairwise_dice_mean")
    _append_macro(summary, per_sheet, "prompt_prediction_pairwise_dice_min")
    _append_macro(summary, per_sheet, "kaggle_surface_dice_tau2_mean")
    _append_macro(summary, per_sheet, "kaggle_surface_dice_tau2_prompt_min")
    _append_macro(summary, per_sheet, "kaggle_voi_score_mean")
    _append_macro(summary, per_sheet, "kaggle_voi_score_prompt_min")
    _append_macro(summary, per_sheet, "kaggle_toposcore_mean")
    _append_macro(summary, per_sheet, "kaggle_toposcore_prompt_min")
    _append_macro(summary, per_sheet, "kaggle_prompt_component_score_mean")
    _append_macro(summary, per_sheet, "kaggle_prompt_component_score_prompt_min")
    return summary


def _append_macro(summary: dict[str, float], rows: list[dict[str, Any]], key: str) -> None:
    values = np.asarray([float(row[key]) for row in rows if key in row], dtype=np.float64)
    if values.size:
        summary[f"probe_macro_{key}"] = float(values.mean())
