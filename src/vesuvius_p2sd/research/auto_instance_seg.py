"""GT-free instance segmentation: auto-prompted P2SD over the binary foreground.

SAM's "segment everything" recipe with this repo's parts: the binary-seg model
(exp1) proposes foreground, seeds are the deepest points (distance-transform
argmax) of unclaimed foreground blobs -- exactly the on-sheet interior points
P2SD was trained on -- and the varK P2SD decodes one sheet per seed from a
single click. Contested voxels go to the highest predicted probability;
duplicate discoveries of the same sheet are merged by IoU; residual foreground
is re-seeded for several rounds so sheets missed by the first pass (typically
the far side of a merged blob) get their own seed.

Instances and metrics are computed on the LABELED region only (predictions in
the ignore region are neither instances nor errors -- the binary model is
unconstrained there). Scoring is ``eval/instance_matching``: AP@0.5, matched
IoU, and merge/split counts.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy import ndimage

from vesuvius_p2sd.data.build_ignore_masks import load_ignore_mask
from vesuvius_p2sd.data.dataset import load_jsonl
from vesuvius_p2sd.eval.instance_matching import match_instances
from vesuvius_p2sd.models.binary_seg import build_binary_seg_model
from vesuvius_p2sd.models.p2sd import build_p2sd
from vesuvius_p2sd.research.evaluate_p2sd_case_scenes import filter_small_components
from vesuvius_p2sd.research.instances_io import instances_file, load_instances, save_instances
from vesuvius_p2sd.research.run_status import write_experiment_provenance, write_json
from vesuvius_p2sd.research import tta as _tta
from vesuvius_p2sd.train.common import amp_dtype, autocast_context, get_device, load_ae_from_config, load_model_state
from vesuvius_p2sd.train.train_p2sd import build_static_latent_codec, decoded_foreground_probability
from vesuvius_p2sd.utils.config import load_config


def propose_seeds(
    unclaimed: np.ndarray,
    *,
    max_seeds: int,
    min_blob_voxels: int,
) -> list[tuple[int, int, int]]:
    """One seed per unclaimed blob: the deepest interior point.

    The distance-transform argmax of a blob is the point farthest from any
    boundary -- the highest-confidence single click available. Blobs are taken
    largest-first; sub-``min_blob_voxels`` blobs are noise and skipped.
    """

    labels, count = _label6(unclaimed)
    if count == 0:
        return []
    sizes = np.bincount(labels.ravel())
    order = np.argsort(-sizes[1:]) + 1
    chosen = [int(b) for b in order[: max(0, int(max_seeds))] if sizes[b] >= min_blob_voxels]
    if not chosen:
        return []
    # ONE distance transform for the whole unclaimed mask, then the per-blob
    # deepest point via labeled maximum_position -- a full-volume EDT per blob
    # was the dominant cost of the whole pipeline.
    distance = ndimage.distance_transform_edt(unclaimed)
    positions = ndimage.maximum_position(distance, labels, chosen)
    if isinstance(positions, tuple):
        positions = [positions]
    return [tuple(int(v) for v in position) for position in positions]


def spread_core_points(
    prob: np.ndarray,
    mask: np.ndarray,
    seed: tuple[int, int, int],
    *,
    count: int,
    core_threshold: float = 0.9,
    candidates: int = 2000,
    rng: np.random.Generator | None = None,
) -> list[tuple[int, int, int]]:
    """Extra self-prompt clicks from the mask's own high-confidence core.

    Greedy farthest-point selection over a random subset of voxels where the
    first-pass probability is >= ``core_threshold`` -- points the model itself
    is most certain belong to the sheet, spread to pin its extent. Falls back
    to the seed alone when the core is empty.
    """

    rng = rng or np.random.default_rng(0)
    core = np.argwhere((prob >= core_threshold) & mask)
    points = [tuple(int(v) for v in seed)]
    if core.shape[0] == 0 or count <= 0:
        return points
    if core.shape[0] > candidates:
        core = core[rng.choice(core.shape[0], candidates, replace=False)]
    chosen = np.array([seed], dtype=np.float64)
    for _ in range(count):
        distances = np.min(
            np.linalg.norm(core[:, None, :] - chosen[None, :, :], axis=2), axis=1)
        best = int(distances.argmax())
        points.append(tuple(int(v) for v in core[best]))
        chosen = np.vstack([chosen, core[best]])
    return points


def sample_foreground_points(
    foreground: np.ndarray,
    *,
    count: int,
    rng: np.random.Generator,
) -> list[tuple[int, int, int]]:
    coords = np.argwhere(foreground)
    if coords.shape[0] == 0:
        return []
    take = min(int(count), coords.shape[0])
    chosen = coords[rng.choice(coords.shape[0], take, replace=False)]
    return [tuple(int(v) for v in point) for point in chosen]


def sample_points_per_blob(
    foreground: np.ndarray,
    *,
    count: int,
    rng: np.random.Generator,
    blob_floor: int,
    blob_min_voxels: int,
    min_spacing: float = 0.0,
    oversample: int = 4,
) -> list[tuple[int, int, int]]:
    """Stratified clicks: proportional to blob volume, with a guaranteed floor.

    Uniform sampling starves small isolated sheets: of the 118 GT sheets the
    champion misses, the 32 with usable foreground expect a median ~2 of 256
    uniform clicks — below cluster_min_points, so they can never form a
    cluster. Here every 6-conn foreground blob >= ``blob_min_voxels`` gets
    max(proportional share, ``blob_floor``) clicks (capped at its voxel
    count); dust blobs keep only their proportional share. The total may
    exceed ``count`` by the floor top-ups — deliberately, so large sheets
    lose no click density. With ``min_spacing`` > 0 the blue-noise thinning
    runs per blob against that blob's quota.
    """
    labels, n_blobs = _label6(foreground)
    if n_blobs == 0:
        return []
    volumes = np.bincount(labels.ravel())[1:]
    total = int(volumes.sum())
    raw = np.asarray(count, dtype=np.float64) * volumes / total
    quotas = np.floor(raw).astype(int)
    for blob_index in np.argsort(raw - quotas)[::-1][: int(count) - int(quotas.sum())]:
        quotas[blob_index] += 1
    quotas[volumes >= int(blob_min_voxels)] = np.maximum(
        quotas[volumes >= int(blob_min_voxels)], int(blob_floor))
    quotas = np.minimum(quotas, volumes)

    coords = np.argwhere(labels > 0)
    blob_of = labels[tuple(coords.T)]
    order = np.argsort(blob_of, kind="stable")
    coords, blob_of = coords[order], blob_of[order]
    starts = np.searchsorted(blob_of, np.arange(1, n_blobs + 2))
    points: list[tuple[int, int, int]] = []
    for blob_index in range(n_blobs):
        quota = int(quotas[blob_index])
        if quota == 0:
            continue
        blob_coords = coords[starts[blob_index]: starts[blob_index + 1]]
        take = min(quota * (max(1, int(oversample)) if min_spacing > 0 else 1),
                   blob_coords.shape[0])
        chosen = blob_coords[rng.choice(blob_coords.shape[0], take, replace=False)]
        blob_points = [tuple(int(v) for v in point) for point in chosen]
        if min_spacing > 0:
            blob_points = thin_points_min_spacing(blob_points, min_spacing, quota)
        points.extend(blob_points)
    return points


def cluster_latents(
    latents: torch.Tensor,
    *,
    mse_threshold: float,
) -> list[list[int]]:
    """Leader clustering on per-point sheet latents.

    Same-sheet prompts land at latent MSE ~0.0003 and different sheets at
    >= 0.31 (measured on 0017), a three-orders-of-magnitude gap, so a
    mid-threshold separates cleanly. Clustering against cluster CENTROIDS
    rather than a thresholded pairwise graph prevents one ambiguous gap-click
    from chain-merging two sheets through transitive closure.
    """

    flat = latents.reshape(latents.shape[0], -1).float()
    dims = flat.shape[1]
    centroids: list[torch.Tensor] = []
    sums: list[torch.Tensor] = []
    members: list[list[int]] = []
    for index in range(flat.shape[0]):
        z = flat[index]
        best, best_d = -1, float("inf")
        for c, centroid in enumerate(centroids):
            d = float((z - centroid).pow(2).mean())
            if d < best_d:
                best, best_d = c, d
        if best >= 0 and best_d <= mse_threshold:
            members[best].append(index)
            sums[best] = sums[best] + z
            centroids[best] = sums[best] / len(members[best])
        else:
            centroids.append(z.clone())
            sums.append(z.clone())
            members.append([index])
    return members


def split_bimodal_clusters(
    clusters: list[list[int]],
    fingerprints: torch.Tensor,
    *,
    split_mse: float,
    min_points: int,
) -> tuple[list[list[int]], int]:
    """Two-means split of clusters whose fingerprints are bimodal.

    Leader clustering merges two touching sheets when one ambiguous early
    click drags the centroid between them. A genuinely mixed cluster shows
    two latent modes; if the two sub-centroids sit further apart than
    ``split_mse`` (same units as the clustering threshold) and both halves
    keep ``min_points`` members, the cluster decodes as two sheets. Returns
    the new cluster list and the number of splits applied.
    """

    flat = fingerprints.reshape(fingerprints.shape[0], -1).float()
    result: list[list[int]] = []
    splits = 0
    for members in clusters:
        if split_mse <= 0 or len(members) < 2 * min_points:
            result.append(members)
            continue
        z = flat[members]
        distances = torch.cdist(z, z).pow(2) / z.shape[1]
        seed_a, seed_b = divmod(int(distances.argmax()), len(members))
        centers = torch.stack([z[seed_a], z[seed_b]])
        for _ in range(5):
            assign = (torch.cdist(z, centers).pow(2) / z.shape[1]).argmin(dim=1)
            if int(assign.sum()) in (0, len(members)):
                break
            centers = torch.stack([z[assign == 0].mean(0), z[assign == 1].mean(0)])
        half_a = [members[i] for i in range(len(members)) if int(assign[i]) == 0]
        half_b = [members[i] for i in range(len(members)) if int(assign[i]) == 1]
        gap = float((centers[0] - centers[1]).pow(2).mean())
        if gap > split_mse and len(half_a) >= min_points and len(half_b) >= min_points:
            result.extend([half_a, half_b])
            splits += 1
        else:
            result.append(members)
    return result, splits


def purify_clusters(
    clusters: list[list[int]],
    fingerprints: torch.Tensor,
    *,
    margin_ratio: float,
    disagreement_ratio: float,
    model_dims: list[int],
    min_points: int,
    core_floor: float = 0.0,
) -> tuple[list[list[int]], int, int]:
    """Purge ambiguous members instead of letting them drag centroids.

    Purity over coverage (user directive 2026-08-17): a click that cannot be
    confidently attributed serves NO sheet -- decodes need only
    ``prompt_points_per_sheet`` clean clicks, so dropping doubtful ones costs
    nothing and stops boundary clicks from chain-merging sheets. A member is
    purged when

    - margin: its MSE to the nearest OTHER cluster centroid is less than
      ``margin_ratio`` times its MSE to its own centroid (boundary-ambiguous
      click), or
    - ensemble disagreement: per-model normalized MSEs to its own centroid
      (the ``model_dims`` blocks of the concatenated ensemble fingerprint)
      spread by more than ``disagreement_ratio`` (max/min over models) --
      the models dispute this click.

    One pass, drop-only (no reassignment); clusters below ``min_points``
    afterwards die entirely. Returns (clusters, purged_points, died).
    """

    if (margin_ratio <= 0 and disagreement_ratio <= 0) or len(clusters) == 0:
        return clusters, 0, 0
    flat = fingerprints.reshape(fingerprints.shape[0], -1).float()
    purged = 0
    current = [list(c) for c in clusters]
    # Pass 1 -- margin, against MEAN centroids (the clustering metric):
    # boundary clicks sitting between two sheets go first, so pass 2 sees
    # centroids they no longer contaminate.
    if margin_ratio > 0 and len(current) > 1:
        centroids = torch.stack([flat[c].mean(dim=0) for c in current])
        purged_pass: list[list[int]] = []
        for c, members in enumerate(current):
            keep: list[int] = []
            others = torch.cat([centroids[:c], centroids[c + 1:]])
            for i in members:
                z = flat[i]
                d_own = float((z - centroids[c]).pow(2).mean())
                d_other = float((z[None] - others).pow(2).mean(dim=1).min())
                if d_other < margin_ratio * d_own:
                    purged += 1
                else:
                    keep.append(i)
            purged_pass.append(keep)
        current = [c for c in purged_pass if c]
    # Pass 2 -- ensemble disagreement, against elementwise-MEDIAN centroids:
    # robust to a minority straddler, so core members are not flagged by a
    # centroid the straddler dragged. ``core_floor`` anchors the ratio's
    # denominator -- points every model sees within the floor are core.
    if disagreement_ratio > 0 and len(model_dims) > 1 and current:
        blocks = []
        start = 0
        for dim in model_dims:
            blocks.append((start, start + dim))
            start += dim
        purged_pass = []
        for members in current:
            center = flat[members].median(dim=0).values
            keep = []
            for i in members:
                per_model = [float((flat[i][a:b] - center[a:b]).pow(2).mean())
                             for a, b in blocks]
                if max(per_model) > disagreement_ratio * max(min(per_model), core_floor):
                    purged += 1
                else:
                    keep.append(i)
            purged_pass.append(keep)
        current = [c for c in purged_pass if c]
    result = [c for c in current if len(c) >= min_points]
    died = len(clusters) - len(result)
    return result, purged, died


def thin_points_min_spacing(
    points: list[tuple[int, int, int]],
    spacing: float,
    limit: int,
) -> list[tuple[int, int, int]]:
    """Greedy blue-noise thinning: keep points at least ``spacing`` apart.

    Near-coincident clicks carry correlated latents -- they inflate a
    cluster's apparent support without adding evidence. Iterates in the
    (already random) sampling order, so oversample the pool first and the
    survivors spread across ALL foreground, which also improves small-sheet
    representation over mass-weighted random sampling.
    """

    if spacing <= 0 or not points:
        return points[:limit]
    kept: list[tuple[int, int, int]] = []
    array = np.asarray(points, dtype=np.float64)
    kept_array = np.empty((0, 3))
    spacing2 = float(spacing) ** 2
    for row, point in zip(array, points):
        if len(kept) >= limit:
            break
        if len(kept) == 0 or float(((kept_array - row) ** 2).sum(axis=1).min()) >= spacing2:
            kept.append(point)
            kept_array = np.vstack([kept_array, row[None]])
    return kept


def spread_points_3d(points: list[tuple[int, int, int]], count: int) -> list[tuple[int, int, int]]:
    """Greedy farthest-point subset -- clicks spread across the sheet."""

    if len(points) <= count:
        return list(points)
    array = np.asarray(points, dtype=np.float64)
    chosen = [0]
    distances = np.linalg.norm(array - array[0], axis=1)
    for _ in range(count - 1):
        nxt = int(distances.argmax())
        chosen.append(nxt)
        distances = np.minimum(distances, np.linalg.norm(array - array[nxt], axis=1))
    return [points[i] for i in chosen]


def _label6(mask: np.ndarray) -> tuple[np.ndarray, int]:
    """6-connected labeling, cc3d-accelerated (~5x scipy on 320^3), scipy fallback.

    Drop-in for ``ndimage.label(mask)`` with the default 6-conn structure; label
    numbering may differ from scipy but every caller here consumes it via
    sizes/argmax/count, which are numbering-invariant.
    """
    try:
        import cc3d

        labels, n = cc3d.connected_components(
            np.ascontiguousarray(mask), connectivity=6, return_N=True)
        return labels, int(n)
    except Exception:
        return ndimage.label(mask)


def dilate_mask(mask: np.ndarray, voxels: int, device=None) -> np.ndarray:
    """Binary dilation by ``voxels`` (iterated 3^3 max-pool = 26-neighborhood).

    Runs as max_pool3d on the GPU when a cuda device is given (a 320^3 volume
    dilates in ~1 ms vs ~1 s for scipy); scipy fallback keeps tests and
    CPU-only paths working with identical semantics.
    """

    if voxels <= 0 or not mask.any():
        return mask
    if device is not None and torch.device(device).type == "cuda":
        t = torch.from_numpy(np.ascontiguousarray(mask)).to(device=device, dtype=torch.float16)
        t = t[None, None]
        for _ in range(int(voxels)):
            t = torch.nn.functional.max_pool3d(t, 3, stride=1, padding=1)
        return (t[0, 0] > 0).cpu().numpy()
    return ndimage.binary_dilation(
        mask, ndimage.generate_binary_structure(3, 3), iterations=int(voxels))


def claim_instances(
    instance_ids: np.ndarray,
    best_prob: np.ndarray,
    masks: list[np.ndarray],
    probs: list[np.ndarray],
    *,
    next_id: int,
    dedup_iou: float = 0.5,
    dedup_dilate_voxels: int = 0,
    dedup_cover: float = 0.5,
    device=None,
    mask_latents: list | None = None,
    instance_latents: dict | None = None,
    latent_gate_mse: float = 0.0,
    fuse: dict | None = None,
) -> tuple[int, list[int]]:
    """Fold new sheet predictions into the running instance map.

    A voxel belongs to whichever instance predicts it with the highest
    probability (sheets cannot physically overlap). A new mask whose claimed
    region mostly re-covers one existing instance (IoU >= ``dedup_iou`` against
    that instance's current territory) is the same sheet re-discovered and is
    merged into it instead of minting a duplicate id.

    ``dedup_dilate_voxels > 0`` additionally catches NEAR-duplicates: two
    ~3-voxel-thin decodes of the same sheet offset by 1-2 voxels barely
    overlap, so their IoU is low precisely because they are twins (the b0/void
    decomposition, 2026-07-30). The new mask is dilated and, if the dilated
    mask covers >= ``dedup_cover`` of min(new, existing), the new mask is
    treated as a re-discovery and SKIPPED -- not unioned -- because welding
    both shells into one id would keep the sealed air gap (the spurious-void
    dim2 failure) alive inside the union.

    ``latent_gate_mse > 0`` (with ``mask_latents`` per new mask and
    ``instance_latents`` per existing id) VETOES both dedup paths when the
    fingerprints disagree: geometric overlap alone can be two distinct sheets
    pressed together, and welding them is the merge failure mode. Disagreeing
    latents keep the new mask as its own instance; the voxel-level argmax
    still resolves the contested region. Newly minted ids record their latent
    so later candidates in the same call are gated too.
    """

    def latents_disagree(index: int, existing_id: int) -> bool:
        if latent_gate_mse <= 0 or mask_latents is None or instance_latents is None:
            return False
        new_latent = mask_latents[index]
        old_latent = instance_latents.get(existing_id)
        if new_latent is None or old_latent is None:
            return False
        return float((new_latent - old_latent).pow(2).mean()) > latent_gate_mse

    assigned: list[int] = []
    for index, (mask, prob) in enumerate(zip(masks, probs, strict=True)):
        if not mask.any():
            assigned.append(0)
            continue
        probe = dilate_mask(mask, dedup_dilate_voxels, device=device) \
            if dedup_dilate_voxels > 0 else mask
        overlap_ids = instance_ids[probe]
        overlap_ids = overlap_ids[overlap_ids > 0]
        target_id = 0
        if overlap_ids.size:
            candidate = int(np.bincount(overlap_ids).argmax())
            existing = instance_ids == candidate
            inter = int((existing & mask).sum())
            union = int((existing | mask).sum())
            if union and inter / union >= dedup_iou:
                if not latents_disagree(index, candidate):
                    target_id = candidate
            elif dedup_dilate_voxels > 0:
                inter_dilated = int((existing & probe).sum())
                smaller = min(int(mask.sum()), int(existing.sum()))
                if smaller and inter_dilated / smaller >= dedup_cover \
                        and not latents_disagree(index, candidate):
                    # Twin of an existing instance: keep the first shell only.
                    assigned.append(candidate)
                    continue
        if target_id == 0:
            target_id = next_id
            next_id += 1
            if instance_latents is not None and mask_latents is not None \
                    and mask_latents[index] is not None:
                instance_latents[target_id] = mask_latents[index]
        take = mask & (prob > best_prob)
        instance_ids[take] = target_id
        best_prob[take] = prob[take]
        if fuse is not None:
            # Per-voxel decode fusion accumulators (fusion probe, 2026-08-25):
            # the FULL probability field of every accepted mask, including the
            # sub-threshold tails the binary union discards. Twin-skipped and
            # empty masks never reach here, so the accumulators mirror exactly
            # the decodes that formed the instance union.
            p = np.asarray(prob, dtype=np.float32)
            fuse["sum"] += p
            fuse["nor"] *= (1.0 - p)
            np.maximum(fuse["max"], p, out=fuse["max"])
            fuse["n"] += mask.astype(np.uint8)
        assigned.append(target_id)
    return next_id, assigned


def _pad_prompt_sets(
    point_sets: list[list[list[int]]],
    label_sets: list[list[int]] | None,
) -> tuple[list[list[list[int]]], list[list[int]]]:
    """Pad sets to uniform K by repeating a POSITIVE point of each set (a
    repeated positive click is a no-op; repeating a negative would amplify
    it). Labels default to all-positive."""

    width = max(len(points) for points in point_sets)
    padded_points, padded_labels = [], []
    for index, points in enumerate(point_sets):
        labels = list(label_sets[index]) if label_sets else [1] * len(points)
        anchor = points[labels.index(1)] if 1 in labels else points[0]
        pad = width - len(points)
        padded_points.append(list(points) + [anchor] * pad)
        padded_labels.append(labels + [1] * pad)
    return padded_points, padded_labels


def _predict_latents(
    p2sd_model, tokens, coords, context, point_sets, shape, device, dtype,
    label_sets=None, latent_prompts=None,
) -> torch.Tensor:
    """Predicted (normalized) sheet latents for prompt sets, no AE decode.

    ``latent_prompts``: optional [len(point_sets), C] pooled cluster-centroid
    latents for a latent-prompt-trained decoder (one per prompt set).
    """
    padded, labels = _pad_prompt_sets(point_sets, label_sets)
    prompt_points = torch.tensor(padded, device=device, dtype=torch.float32)
    prompt_labels = torch.tensor(labels, device=device, dtype=torch.long)
    image_index = torch.zeros(len(padded), device=device, dtype=torch.long)
    with autocast_context(device, dtype):
        output = p2sd_model.forward_from_image_context(
            tokens, coords, context, prompt_points, prompt_labels,
            image_shape=shape, image_index=image_index,
            latent_prompt=(latent_prompts.to(device) if latent_prompts is not None else None))
    return output["latent"]


_DECODE_BATCH = 8  # decoder VRAM scales linearly with this; --decode_batch 1 fits a 15 GiB T4


def _decode_latents(target_ae, latent_codec, latents, device, dtype) -> np.ndarray:
    """AE-decode predicted latents to foreground probability volumes.

    Chunked by ``_DECODE_BATCH``: samples are independent through the decoder,
    so chunking is exact — only the VRAM peak changes.
    """
    outs = []
    for start in range(0, int(latents.shape[0]), _DECODE_BATCH):
        with autocast_context(device, dtype):
            decoded = target_ae.decode(
                latent_codec.raw_prediction(latents[start:start + _DECODE_BATCH]))
        outs.append(decoded_foreground_probability(decoded).float()[:, 0].cpu().numpy())
    return np.concatenate(outs, axis=0)


def _decode_prompt_sets(
    p2sd_model, target_ae, latent_codec, tokens, coords, context,
    point_sets, shape, device, dtype, label_sets=None, latent_prompts=None,
):
    """Decode one mask per prompt set (predict latents, then AE-decode)."""
    latents = _predict_latents(
        p2sd_model, tokens, coords, context, point_sets, shape, device, dtype,
        label_sets=label_sets, latent_prompts=latent_prompts)
    return _decode_latents(target_ae, latent_codec, latents, device, dtype)


def _truncated_sdf(mask: np.ndarray, radius: int, device) -> np.ndarray:
    """Signed distance to the mask boundary, truncated to [-radius, radius]: +k at depth k inside, -k at distance k
    outside (26-neighbourhood steps via max-pooling). float32 numpy."""
    m = torch.from_numpy(np.ascontiguousarray(mask)).to(device=device, dtype=torch.float16)[None, None]
    # outside: distance = smallest k with dilate_k(m) > 0
    out = torch.full_like(m, float(radius + 1)); cur = m
    for k in range(1, radius + 1):
        cur = torch.nn.functional.max_pool3d(cur, 3, stride=1, padding=1)
        out = torch.where((cur > 0) & (out > radius), torch.full_like(out, float(k)), out)
    # inside: depth = smallest k with erode_k(m) == 0 (erosion = 1 - dilate(1 - m))
    depth = torch.full_like(m, float(radius + 1)); cur = m
    for k in range(1, radius + 1):
        cur = 1 - torch.nn.functional.max_pool3d(1 - cur, 3, stride=1, padding=1)
        depth = torch.where((cur == 0) & (depth > radius), torch.full_like(depth, float(k)), depth)
    sdf = torch.where(m > 0, depth.clamp(max=radius), -out.clamp(max=radius))
    return sdf[0, 0].float().cpu().numpy()


def _largest_component(mask: np.ndarray) -> np.ndarray:
    labels_cc, count = _label6(mask)
    if count == 0:
        return mask
    sizes = np.bincount(labels_cc.ravel())
    sizes[0] = 0
    return labels_cc == int(sizes.argmax())


@torch.no_grad()
def run_case(
    *,
    binseg_model,
    p2sd_model,
    target_ae,
    latent_codec,
    image: np.ndarray,
    valid: np.ndarray,
    device,
    dtype,
    binary_threshold: float,
    sheet_threshold: float,
    max_rounds: int,
    seeds_per_round: int,
    min_blob_voxels: int,
    min_component_voxels: int,
    refine_points: int = 3,
) -> dict[str, Any]:
    """Binary foreground -> iterated seed/decode/claim -> instance label map."""

    tensor = torch.from_numpy(image[None, None].astype(np.float32)).to(device)
    with autocast_context(device, dtype):
        binary_logits = binseg_model(tensor)
        _, tokens, coords, context = p2sd_model.encode_image_context_from_image(tensor)
        if hasattr(target_ae, "set_image"):
            target_ae.set_image(tensor)
    foreground = (torch.sigmoid(binary_logits.float())[0, 0] >= binary_threshold).cpu().numpy()
    foreground &= valid

    shape = image.shape
    instance_ids = np.zeros(shape, dtype=np.int16)
    best_prob = np.zeros(shape, dtype=np.float32)
    next_id = 1
    seed_log: list[dict[str, Any]] = []
    for round_index in range(max_rounds):
        unclaimed = foreground & (instance_ids == 0)
        seeds = propose_seeds(unclaimed, max_seeds=seeds_per_round, min_blob_voxels=min_blob_voxels)
        if not seeds:
            break
        # Pass 1: one click per seed. Pass 2 (self-refinement): re-decode with
        # extra clicks sampled from each first-pass mask's high-confidence core.
        # Single-click masks assign the right sheet but run ~1.3-1.6x too fat
        # (measured); the varK decoder at K=4 is substantially tighter, and the
        # extra clicks come from the model's own certainty, no GT involved.
        first = _decode_prompt_sets(
            p2sd_model, target_ae, latent_codec, tokens, coords, context,
            [[list(seed)] for seed in seeds], shape, device, dtype)
        point_sets = []
        for seed, prob in zip(seeds, first, strict=True):
            core_mask = _largest_component(prob >= sheet_threshold)
            point_sets.append([list(point) for point in spread_core_points(
                prob, core_mask, seed, count=refine_points)])
        probabilities = (
            _decode_prompt_sets(
                p2sd_model, target_ae, latent_codec, tokens, coords, context,
                point_sets, shape, device, dtype)
            if refine_points > 0 else first)
        masks, probs = [], []
        for prob in probabilities:
            # Keep the LARGEST connected component -- P2SD is a single-sheet
            # decoder, so its largest component IS the sheet. Do not require
            # the mask to contain the seed voxel: seeds from over-thick binary
            # blobs often sit in the inter-sheet gap, and even on-sheet clicks
            # miss inclusion by the ~2.8-voxel placement fidelity; P2SD still
            # decodes the correct nearby sheet (measured, not assumed).
            mask = _largest_component(prob >= sheet_threshold)
            masks.append(mask & valid)
            probs.append(prob.astype(np.float32))
        before = next_id
        next_id, assigned = claim_instances(instance_ids, best_prob, masks, probs, next_id=next_id)
        seed_log.append({"round": round_index, "seeds": [list(s) for s in seeds],
                         "assigned": assigned, "new_instances": next_id - before})
        if next_id == before and not any(assigned):
            break

    # Small-instance cleanup mirrors the CC-500 convention.
    if min_component_voxels > 0:
        ids, counts = np.unique(instance_ids[instance_ids > 0], return_counts=True)
        for instance, size in zip(ids, counts):
            if size < min_component_voxels:
                instance_ids[instance_ids == instance] = 0
    return {"instance_ids": instance_ids, "foreground": foreground, "seed_log": seed_log}


def topology_reward(
    mask: np.ndarray,
    fragment_count: int,
    own_points: list[tuple[int, int, int]],
    foreign_points: list[tuple[int, int, int]],
    *,
    foreign_weight: float = 1.0,
    fragment_weight: float = 0.1,
) -> float:
    """GT-free candidate score: own clicks in, other sheets' clicks out, one piece.

    ``cover_own`` is the anti-empty anchor (a mask must contain its own
    cluster's clicks), ``cover_foreign`` is the anti-merge term (other
    clusters' clicks inside the mask signal a bridge into a neighbouring
    sheet), and the fragment penalty counts extra components BEFORE the
    largest-component selection.
    """

    if not mask.any():
        return -1.0
    cover_own = float(np.mean([mask[p] for p in own_points])) if own_points else 0.0
    cover_foreign = float(np.mean([mask[p] for p in foreign_points])) if foreign_points else 0.0
    return cover_own - foreign_weight * cover_foreign - fragment_weight * max(0, fragment_count - 1)


@torch.no_grad()
def run_case_cluster(
    *,
    binseg_model,
    p2sd_model,
    target_ae,
    latent_codec,
    image: np.ndarray,
    valid: np.ndarray,
    device,
    dtype,
    binary_threshold: float,
    sheet_threshold: float,
    min_component_voxels: int,
    min_component_connectivity: int = 0,
    seed_min_blob_voxels: int = -1,
    cluster_points: int = 512,
    cluster_mse_threshold: float = 0.02,
    cluster_min_points: int = 8,
    prompt_points_per_sheet: int,
    candidates_per_cluster: int = 1,
    latent_fusion_samples: int = 0,
    dedup_dilate_voxels: int = 0,
    dedup_cover: float = 0.5,
    verify_overlap_iou: float = 0.0,
    cluster_p2sd_model=None,
    neg_points_per_sheet: int = 0,
    binseg_tta: int = 0,
    latent_batch: int = 32,
    seed: int = 0,
    cluster_split_mse: float = 0.0,
    cluster_stability_min_iou: float = 0.0,
    cluster_stability_depth: int = 2,
    cluster_subset_knn: int = 0,
    cluster_subset_knn_neighbors: int = 3,
    cluster_subset_knn_fuse: int = 1,
    cluster_drop_mse: float = 0.008,
    cluster_rep_merge_mse: float = 0.0,
    click_self_decode_min_prob: float = 0.0,
    dedup_latent_gate_mse: float = 0.0,
    residual_rounds: int = 0,
    residual_novelty_mse: float = 0.0,
    residual_max_seeds: int = 8,
    residual_verify_iou: float = -1.0,
    latent_prompt_condition: bool = False,
    embed_model=None,
    embed_ball_radius: int = 0,
    embed_ball_points: int = 32,
    click_min_spacing: float = 0.0,
    click_oversample: int = 4,
    click_per_blob_floor: int = 0,
    click_blob_min_voxels: int = 500,
    cluster_purge_margin: float = 0.0,
    cluster_purge_disagreement: float = 0.0,
    external_mask: np.ndarray | None = None,
    image_context_cache=None,
    fuse_dump: bool = False,
    tta: int = 0,
    tta_fuse: str = "mean",
    tta_dilate: int = 2,
    tta_vote: float = 0.5,
    tta_thin: str = "medial",
) -> dict[str, Any]:
    """Latent-fingerprint clustering: M random clicks -> sheet clusters -> K-click decodes.

    ``tta`` (0/1 off, 8 flips, 16 = the full D4(y,x) x flip(z) training-symmetry group): the CHOSEN prompt subsets of
    the kNN path are re-decoded on every transformed image (prompts transformed alike) and the full-resolution
    probabilities are inverse-transformed and averaged (decoder-output TTA; the latent is not equivariant).
    ``binseg_tta``: 1/8 = the 8 axis flips (historic), 16 = the full group; sigmoids averaged.

    Each random foreground click is mapped to its predicted sheet LATENT
    (modulator + refiner only -- no AE decode), clicks are clustered by latent
    MSE, clusters bigger than ``cluster_min_points`` are treated as one sheet
    each, and ``prompt_points_per_sheet`` spread clicks from the cluster decode
    the sheet at the strong multi-click operating point.
    """

    rng = np.random.default_rng(seed)
    tensor = torch.from_numpy(image[None, None].astype(np.float32)).to(device)
    binseg_models = (binseg_model if isinstance(binseg_model, (list, tuple))
                     else [binseg_model])
    with autocast_context(device, dtype):
        if external_mask is not None:
            # Externally supplied foreground (e.g. a stronger third-party
            # binseg): clicks sample from it verbatim; binseg models are not
            # run and binary_threshold is unused.
            foreground_mask = torch.from_numpy(np.ascontiguousarray(external_mask > 0)).to(device)
        elif binseg_tta:
            # TTA over exact symmetries of the training augmentation (8 axis flips, or the full 16-element
            # D4(y,x) x flip(z) group when binseg_tta == 16): averaging their sigmoids denoises the foreground
            # the clicks are sampled from.
            tta_list = _tta.transforms(16 if int(binseg_tta) == 16 else 8)
            foreground_mask = None
            for model in binseg_models:
                binary_prob = None
                for t in tta_list:
                    prob = _tta.invert(torch.sigmoid(model(_tta.apply(tensor, t)).float()), t)
                    binary_prob = prob if binary_prob is None else binary_prob + prob
                mask = (binary_prob / len(tta_list))[0, 0] >= binary_threshold
                foreground_mask = mask if foreground_mask is None else foreground_mask & mask
        else:
            # Multiple proposers: bitwise AND of their thresholded masks --
            # clicks sample only from foreground EVERY decoder believes
            # (prominent sheet cores; suppresses one model's fringe).
            foreground_mask = None
            for model in binseg_models:
                mask = torch.sigmoid(model(tensor).float())[0, 0] >= binary_threshold
                foreground_mask = mask if foreground_mask is None else foreground_mask & mask
        # A co-trained foreground head can reuse the same unprompted context.
        # Callers supply the tuple from this model and this exact input volume.
        if image_context_cache is None:
            _, tokens, coords, context = p2sd_model.encode_image_context_from_image(tensor)
        else:
            _, tokens, coords, context = image_context_cache
        if hasattr(target_ae, "set_image"):
            target_ae.set_image(tensor)
    foreground = foreground_mask.cpu().numpy()
    foreground &= valid
    shape = image.shape

    if click_per_blob_floor > 0:
        points = sample_points_per_blob(
            foreground, count=cluster_points, rng=rng,
            blob_floor=click_per_blob_floor, blob_min_voxels=click_blob_min_voxels,
            min_spacing=click_min_spacing, oversample=click_oversample)
    else:
        oversample = max(1, int(click_oversample)) if click_min_spacing > 0 else 1
        points = sample_foreground_points(
            foreground, count=cluster_points * oversample, rng=rng)
        if click_min_spacing > 0:
            points = thin_points_min_spacing(points, click_min_spacing, cluster_points)
    if not points:
        return {"instance_ids": np.zeros(shape, dtype=np.int16), "foreground": foreground,
                "seed_log": [], "cluster_sizes": [],
                "points": {"clicks": [], "sheets": {}}}
    click_filter_log: dict[str, int] = {}
    if click_self_decode_min_prob > 0:
        # Seed self-consistency filter (calibration probe 2026-08-18): a click
        # whose own single-click decode does not cover the click itself
        # (decoded prob at the click < threshold) is dropped BEFORE
        # clustering. Measured: ~29% of clicks fail at 0.5, of which ~2/3 sit
        # on GT background; the decode is sharply bimodal so the threshold is
        # insensitive in [0.3, 0.7]. (The AE query-distance head is NOT
        # usable for this: it reports ~0.1 voxels for every click.)
        keep: list[int] = []
        with torch.no_grad():
            for start in range(0, len(points), 16):
                chunk = points[start:start + 16]
                latents = _predict_latents(
                    p2sd_model, tokens, coords, context,
                    [[list(p)] for p in chunk], shape, device, dtype)
                decs = _decode_latents(target_ae, latent_codec, latents, device, dtype)
                for k, prob in enumerate(decs):
                    pz, py, px = chunk[k]
                    if float(prob[pz, py, px]) >= click_self_decode_min_prob:
                        keep.append(start + k)
        click_filter_log = {"sampled": len(points), "kept": len(keep),
                            "culled": len(points) - len(keep)}
        points = [points[k] for k in keep]
        if not points:
            return {"instance_ids": np.zeros(shape, dtype=np.int16),
                    "foreground": foreground, "seed_log": [{"round": 0,
                    "click_filter": click_filter_log}], "cluster_sizes": [],
                    "points": {"clicks": [], "sheets": {}}}
    # Prompt provenance for visualization: every sampled click, and per final
    # instance id the positive prompt points of the decode that claimed it.
    sheet_points: dict[int, list[list[int]]] = {}
    # Fingerprint model(s): clustering may use DIFFERENT P2SDs (e.g. the
    # contrast-loss fine-tune, trained to push cross-sheet latents apart)
    # while the champion still decodes. Each needs its own image context.
    fingerprint_models = ([cluster_p2sd_model] if cluster_p2sd_model is not None
                          and not isinstance(cluster_p2sd_model, (list, tuple))
                          else list(cluster_p2sd_model or [p2sd_model]))
    model_contexts = []
    for f_model in fingerprint_models:
        if f_model is p2sd_model:
            model_contexts.append((f_model, tokens, coords, context))
        else:
            with autocast_context(device, dtype):
                _, f_tokens, f_coords, f_context = \
                    f_model.encode_image_context_from_image(tensor)
            model_contexts.append((f_model, f_tokens, f_coords, f_context))
    ensemble_scales: list[torch.Tensor] = []
    latent_grid_shape: list[tuple[int, ...]] = []
    model_flat_dims: list[int] = []

    def fingerprints_for(query_points) -> torch.Tensor:
        """Per-click fingerprints in the clustering metric space.

        Ensemble scales are computed once, on the initial click pool, and
        reused for later (residual-round) queries so distances stay
        comparable across the whole case.
        """
        per_model_latents = []
        for f_model, f_tokens, f_coords, f_context in model_contexts:
            latents = []
            for start in range(0, len(query_points), latent_batch):
                chunk = query_points[start:start + latent_batch]
                pts = torch.tensor([[list(p)] for p in chunk], device=device, dtype=torch.float32)
                lbl = torch.ones(len(chunk), 1, device=device, dtype=torch.long)
                idx = torch.zeros(len(chunk), device=device, dtype=torch.long)
                with autocast_context(device, dtype):
                    out = f_model.forward_from_image_context(
                        f_tokens, f_coords, f_context, pts, lbl, image_shape=shape, image_index=idx)
                if not latent_grid_shape:
                    latent_grid_shape.append(tuple(int(v) for v in out["latent"].shape[1:]))
                latents.append(out["latent"].float().cpu())
            per_model_latents.append(torch.cat(latents).flatten(1))
        if not model_flat_dims:
            model_flat_dims.extend(int(lat.shape[1]) for lat in per_model_latents)
        if len(per_model_latents) == 1:
            return per_model_latents[0]  # absolute units, v8-compatible
        # Ensemble: normalize each model's metric by its own median pairwise
        # MSE over this case's clicks (self-calibrating), then concatenate so
        # the combined MSE is the MEAN of normalized per-model distances --
        # two clicks cluster together only when the models AGREE. The
        # threshold is then a fraction of the median pairwise distance.
        # cluster_latents uses mean-over-dims MSE, so concatenating latents
        # that each have median pairwise MSE 1 yields a combined MSE that is
        # the MEAN of the models' normalized distances -- and the configured
        # threshold reads directly as "fraction of the median pairwise MSE"
        # (per-case adaptive).
        if not ensemble_scales:
            for lat in per_model_latents:
                d2 = torch.cdist(lat, lat).pow(2) / lat.shape[1]
                ensemble_scales.append(torch.sqrt(d2[d2 > 0].median()).clamp_min(1e-8))
        return torch.cat(
            [lat / scale for lat, scale in zip(per_model_latents, ensemble_scales)], dim=1)

    if embed_model is not None:
        if latent_prompt_condition:
            raise ValueError("embed fingerprints have no P2SD latent space for "
                             "latent_prompt_condition centroids")
        # Fingerprints from a ctx-attn voxel-embedding checkpoint (0039/0040
        # line) instead of P2SD click latents. The decode path is untouched;
        # clustering operates on normalized 64-d embeddings, so
        # cluster_mse_threshold must be given in THEIR units:
        # mean-over-dims MSE = (2 - 2cos)/64 (e.g. 0.015 ~ cos 0.5).
        embed_grid: list[torch.Tensor] = []

        def embed_fingerprints_for(query_points) -> torch.Tensor:
            if not embed_grid:
                with torch.no_grad(), autocast_context(device, dtype):
                    e_ctx = embed_model.refine_context(embed_model.encode_context(tensor))
                    embed_grid.append(embed_model.embed_context(e_ctx).float())
            emb = embed_grid[0]
            size = torch.tensor(shape, device=emb.device, dtype=torch.float32).clamp_min(1)

            def at(coords_np: np.ndarray) -> torch.Tensor:
                coords = torch.as_tensor(
                    np.asarray(coords_np, dtype=np.float32), device=emb.device)
                grid_n = ((coords + 0.5) / size) * 2 - 1
                grid_xyz = grid_n[:, [2, 1, 0]].view(1, -1, 1, 1, 3)
                with torch.no_grad():
                    sampled = torch.nn.functional.grid_sample(
                        emb, grid_xyz, mode="bilinear", align_corners=False)
                return torch.nn.functional.normalize(
                    sampled[0, :, :, 0, 0].transpose(0, 1), dim=1)

            points_np = np.asarray([list(p) for p in query_points], dtype=np.int64)
            if embed_ball_radius <= 0:
                return at(points_np).cpu()
            r = int(embed_ball_radius)
            offsets = np.stack(np.meshgrid(*([np.arange(-r, r + 1)] * 3),
                                           indexing="ij"), axis=-1).reshape(-1, 3)
            fingerprints = []
            for point in points_np:
                neighbors = point + offsets
                inb = np.all((neighbors >= 0) & (neighbors < np.array(shape)), axis=1)
                neighbors = neighbors[inb]
                neighbors = neighbors[
                    foreground[neighbors[:, 0], neighbors[:, 1], neighbors[:, 2]]]
                if len(neighbors) == 0:
                    neighbors = point[None]
                if len(neighbors) > embed_ball_points:
                    neighbors = neighbors[rng.choice(len(neighbors), embed_ball_points,
                                                     replace=False)]
                fingerprints.append(torch.nn.functional.normalize(
                    at(neighbors).mean(dim=0), dim=0))
            return torch.stack(fingerprints).cpu()

        fingerprints_for = embed_fingerprints_for

    def self_consistency_filter(candidate_masks: list[np.ndarray], min_iou: float) -> list[dict]:
        """Zero out masks that fail the re-decode self-consistency check.

        Re-decode each surviving mask from 8 random points OF THE MASK ITSELF
        and require the two decodes to agree. A stable sheet reproduces
        itself; an unstable decode (clicks straddling two sheets,
        hallucinated geometry) does not. No-op when min_iou <= 0.
        """
        checks = []
        if min_iou <= 0:
            return checks
        check_sets, check_owners = [], []
        for c, mask in enumerate(candidate_masks):
            if not mask.any():
                continue
            voxels = np.argwhere(mask)
            take = voxels[rng.choice(len(voxels), min(8, len(voxels)), replace=False)]
            check_sets.append([list(map(int, p)) for p in take])
            check_owners.append(c)
        redecoded = []
        for start in range(0, len(check_sets), 8):
            redecoded.extend(_decode_prompt_sets(
                p2sd_model, target_ae, latent_codec, tokens, coords, context,
                check_sets[start:start + 8], shape, device, dtype))
        for c, prob in zip(check_owners, redecoded):
            re_mask = _largest_component(prob >= sheet_threshold) & valid
            inter = int((candidate_masks[c] & re_mask).sum())
            union = int((candidate_masks[c] | re_mask).sum())
            iou = inter / union if union else 0.0
            checks.append({"cluster": c, "self_iou": round(iou, 4)})
            if iou < min_iou:
                candidate_masks[c] = np.zeros(shape, dtype=bool)
        return checks

    fingerprints = fingerprints_for(points)
    clusters = cluster_latents(fingerprints, mse_threshold=cluster_mse_threshold)
    purged_points = died_clusters = 0
    if cluster_purge_margin > 0 or cluster_purge_disagreement > 0:
        clusters, purged_points, died_clusters = purify_clusters(
            clusters, fingerprints,
            margin_ratio=cluster_purge_margin,
            disagreement_ratio=cluster_purge_disagreement,
            model_dims=(model_flat_dims
                        or [int(np.prod(fingerprints.shape[1:]))]),
            min_points=cluster_min_points + 1,
            core_floor=0.25 * cluster_mse_threshold)
    kept = [c for c in clusters if len(c) > cluster_min_points]
    split_count = 0
    if cluster_split_mse > 0 and kept:
        kept, split_count = split_bimodal_clusters(
            kept, fingerprints,
            split_mse=cluster_split_mse, min_points=cluster_min_points)

    instance_ids = np.zeros(shape, dtype=np.int16)
    best_prob = np.zeros(shape, dtype=np.float32)
    fuse = ({"sum": np.zeros(shape, dtype=np.float32), "nor": np.ones(shape, dtype=np.float32),
             "max": np.zeros(shape, dtype=np.float32), "n": np.zeros(shape, dtype=np.uint8)}
            if fuse_dump else None)
    next_id = 1
    log = []
    # Per-instance fingerprint centroids: consulted by the dedup latent gate
    # and by residual-round novelty checks (both no-ops unless enabled).
    flat_fingerprints = fingerprints.reshape(fingerprints.shape[0], -1).float()
    instance_latents: dict[int, torch.Tensor] = {}
    if kept:
        kept_centroids = [flat_fingerprints[cluster].mean(dim=0) for cluster in kept]
        pooled_centroids = None
        if latent_prompt_condition:
            # Pooled (spatial-mean) centroid per cluster for the
            # latent-prompt-trained decoder. Only meaningful in the raw
            # single-model fingerprint space -- the ensemble metric is
            # per-case renormalized and no longer the model's latent space.
            if len(model_contexts) != 1:
                raise ValueError("latent_prompt_condition requires a single fingerprint model")
            if getattr(p2sd_model, "latent_prompt_mlp", None) is None:
                raise ValueError("latent_prompt_condition requires a latent-prompt-trained p2sd model")
            channels = latent_grid_shape[0][0]
            pooled_centroids = torch.stack([
                centroid.view(channels, -1).mean(dim=1) for centroid in kept_centroids])
        cluster_points_3d = [[points[i] for i in cluster] for cluster in kept]
        n_sets = max(1, int(latent_fusion_samples) if latent_fusion_samples > 0
                     else int(candidates_per_cluster))
        # Set 0 is the deterministic farthest-spread subset; the rest are
        # random click subsets -- the jitter that makes best-of-N selection
        # (or latent fusion) meaningful. (Skipped in kNN-representative mode,
        # which draws its own subsets.)
        point_sets, owners = [], []
        if cluster_subset_knn <= 0:
            for c, cpoints in enumerate(cluster_points_3d):
                point_sets.append([list(p) for p in spread_points_3d(cpoints, prompt_points_per_sheet)])
                owners.append(c)
                for _ in range(n_sets - 1):
                    take = min(prompt_points_per_sheet, len(cpoints))
                    idx = rng.choice(len(cpoints), take, replace=False)
                    point_sets.append([list(cpoints[i]) for i in idx])
                    owners.append(c)
        label_sets = None
        if neg_points_per_sheet > 0:
            # Corrective negatives: the NEAREST foreign clusters' clicks are
            # the most confusable neighbors -- append them with label 0
            # ("that sheet is not mine"). Requires a negative-click-trained
            # decoder (e.g. 0030); an untrained label-0 pathway is noise.
            label_sets = []
            centroids_3d = [np.mean(np.array(cp, dtype=float), axis=0)
                            for cp in cluster_points_3d]
            for set_index, c in enumerate(owners):
                foreign = [p for c2, pts2 in enumerate(cluster_points_3d)
                           if c2 != c for p in pts2]
                labels = [1] * len(point_sets[set_index])
                if foreign:
                    foreign = sorted(
                        foreign, key=lambda p: float(np.sum((np.array(p, dtype=float)
                                                             - centroids_3d[c]) ** 2)))
                    for p in foreign[:neg_points_per_sheet]:
                        point_sets[set_index].append(list(p))
                        labels.append(0)
                label_sets.append(labels)
        masks, probs, chosen_log = [], [], []
        chosen_points: list[list[list[int]]] = []
        cluster_candidates: list[dict] = []
        if cluster_subset_knn > 0:
            # kNN-representative selection + cluster DROPPING (user design
            # 2026-08-18): M random subsets per cluster, pairwise latent MSE
            # (no decodes, no IoU), each subset scored by the mean of its N
            # nearest neighbors. The best subset represents the cluster IF its
            # score clears cluster_drop_mse — otherwise the whole cluster is
            # dropped: an unstable cluster re-decodes to the same attractor
            # (e18), so the only union-changing move is not to decode it.
            if neg_points_per_sheet or latent_prompt_condition:
                raise ValueError("cluster_subset_knn does not support negative "
                                 "clicks or latent_prompt_condition")
            m_sets = int(cluster_subset_knn)
            knn = max(1, min(int(cluster_subset_knn_neighbors), m_sets - 1))
            subset_sets: list[list[list[int]]] = []
            for cpoints in cluster_points_3d:
                for _ in range(m_sets):
                    take = min(prompt_points_per_sheet, len(cpoints))
                    idx = rng.choice(len(cpoints), take, replace=False)
                    subset_sets.append([list(cpoints[i]) for i in idx])
            pred_chunks = []
            for start in range(0, len(subset_sets), 8):
                pred_chunks.append(_predict_latents(
                    p2sd_model, tokens, coords, context,
                    subset_sets[start:start + 8], shape, device, dtype))
            latents_all = torch.cat(pred_chunks)
            flat = latents_all.float().cpu().flatten(1)
            best_rows, best_scores = [], []
            for c in range(len(cluster_points_3d)):
                block = flat[c * m_sets:(c + 1) * m_sets]
                d2 = torch.cdist(block, block).pow(2) / block.shape[1]
                d2.fill_diagonal_(float("inf"))
                scores = d2.topk(knn, largest=False).values.mean(dim=1)
                best = int(scores.argmin())
                best_rows.append(c * m_sets + best)
                best_scores.append(float(scores[best]))
            # Cross-cluster representative merge: two clusters whose BEST
            # subsets predict near-identical sheet latents are the same sheet
            # decoded twice (the flush parallel-slab duplicates voxel dedup
            # misses at cover 0.5). Union-find groups under the threshold;
            # each group keeps its most compact member (smallest best score).
            merged_into = {}
            rep_pairs = []
            if cluster_rep_merge_mse > 0 and len(best_rows) > 1:
                reps = flat[best_rows]
                parent = list(range(len(best_rows)))

                def find(x):
                    while parent[x] != x:
                        parent[x] = parent[parent[x]]
                        x = parent[x]
                    return x

                for i in range(len(best_rows)):
                    for j in range(i + 1, len(best_rows)):
                        mse = float((reps[i] - reps[j]).pow(2).mean())
                        rep_pairs.append([i, j, round(mse, 6)])
                        if mse < cluster_rep_merge_mse:
                            parent[find(i)] = find(j)
                groups: dict[int, list[int]] = {}
                for c in range(len(best_rows)):
                    groups.setdefault(find(c), []).append(c)
                for members in groups.values():
                    keeper = min(members, key=lambda c: best_scores[c])
                    for c in members:
                        if c != keeper:
                            merged_into[c] = keeper
            to_decode: list[tuple[int, int]] = []
            for c in range(len(cluster_points_3d)):
                best_score = best_scores[c]
                masks.append(np.zeros(shape, dtype=bool))
                probs.append(np.zeros(shape, dtype=np.float32))
                row = {"cluster": c, "members": len(cluster_points_3d[c]),
                       "knn_best_mse": round(best_score, 6)}
                if c in merged_into:
                    chosen_points.append([])
                    chosen_log.append({**row, "verdict": "merged",
                                       "merged_into": int(merged_into[c])})
                elif best_score > cluster_drop_mse:
                    chosen_points.append([])
                    chosen_log.append({**row, "verdict": "dropped"})
                else:
                    to_decode.append((c, best_rows[c]))
                    chosen_points.append(
                        [[int(v) for v in p] for p in subset_sets[best_rows[c]]])
                    chosen_log.append({**row, "verdict": "decode"})
            if rep_pairs:
                chosen_log.append({"rep_mse_pairs": rep_pairs})
            decode_rows = [row for _, row in to_decode]
            # rows whose latents are fused per decoded cluster (kNN core; the representative alone when fuse == 1)
            member_rows: list[list[int]] = []
            if int(cluster_subset_knn_fuse) > 1 and decode_rows:
                # kNN-core latent fusion (user proposal 2026-08-25): decode the
                # MEAN of the representative's latent and its (M-1) nearest
                # subsets' latents -- the cluster's consistent core -- with one
                # AE decode, instead of the representative alone. Outlier
                # subsets (a different sheet) stay out of the mean; M=1 is the
                # champion path.
                fuse_m = min(int(cluster_subset_knn_fuse), m_sets)
                for row in decode_rows:
                    c0 = row // m_sets
                    block = flat[c0 * m_sets:(c0 + 1) * m_sets]
                    near = (block - flat[row]).pow(2).mean(dim=1).topk(fuse_m, largest=False).indices
                    member_rows.append([c0 * m_sets + int(i) for i in near])
            else:
                member_rows = [[row] for row in decode_rows]
            tta_list = _tta.transforms(int(tta)) if int(tta) > 1 else [(0, 0, 0)]
            decoded = [None] * len(decode_rows); tta_probmean: dict = {}
            for t in tta_list:
                if t == (0, 0, 0):
                    lat_t = latents_all; tokens_t, coords_t, context_t = tokens, coords, context
                    if hasattr(target_ae, "set_image") and len(tta_list) > 1:
                        with autocast_context(device, dtype):
                            target_ae.set_image(tensor)
                else:
                    # decoder-output TTA: the same prompt subsets on the transformed image, latents predicted in
                    # that frame, decoded there, probabilities inverse-transformed below
                    tensor_t = _tta.apply(tensor, t)
                    with autocast_context(device, dtype):
                        _, tokens_t, coords_t, context_t = p2sd_model.encode_image_context_from_image(tensor_t)
                        if hasattr(target_ae, "set_image"):
                            target_ae.set_image(tensor_t)
                    need = sorted({r for rows_ in member_rows for r in rows_})
                    sets_t = [[[int(v) for v in q] for q in _tta.apply_points(subset_sets[r], t, shape)] for r in need]
                    pred_t = []
                    for start in range(0, len(sets_t), 8):
                        pred_t.append(_predict_latents(
                            p2sd_model, tokens_t, coords_t, context_t,
                            sets_t[start:start + 8], _tta.out_shape(shape, t), device, dtype))
                    pred_t = torch.cat(pred_t)
                    lat_t = {r: pred_t[i] for i, r in enumerate(need)}
                fused_t = torch.stack([
                    (lat_t[rows_] if t == (0, 0, 0) else torch.stack([lat_t[r] for r in rows_])).mean(dim=0)
                    for rows_ in member_rows])
                for start in range(0, len(decode_rows), 8):
                    chunk = _decode_latents(target_ae, latent_codec, fused_t[start:start + 8], device, dtype)
                    for j, prob in enumerate(chunk):
                        if t != (0, 0, 0):
                            prob = _tta.invert(torch.from_numpy(prob), t).numpy()
                        i = start + j
                        if tta_fuse == "max":   # union of the transformed decodes: no pinholes where they disagree
                            decoded[i] = prob if decoded[i] is None else np.maximum(decoded[i], prob)
                        elif tta_fuse == "dilthin":   # dilate each transformed mask so misaligned copies overlap; fused below
                            d = dilate_mask(prob >= sheet_threshold, int(tta_dilate), device=device).astype(np.float32) / len(tta_list)
                            decoded[i] = d if decoded[i] is None else decoded[i] + d
                            tta_probmean[i] = prob / len(tta_list) if tta_probmean.get(i) is None else tta_probmean[i] + prob / len(tta_list)
                        elif tta_fuse == "sdf":   # mean truncated signed distance -> the median surface, thin, no pinholes
                            R = 4
                            d = _truncated_sdf(prob >= sheet_threshold, R, device) / len(tta_list)
                            decoded[i] = d if decoded[i] is None else decoded[i] + d
                        else:
                            decoded[i] = prob / len(tta_list) if decoded[i] is None else decoded[i] + prob / len(tta_list)
            if tta_fuse == "dilthin" and len(tta_list) > 1:
                # majority of the dilated masks (hole-free slab), thinned back to its medial ~3 voxels (user proposal
                # 2026-09-06: "fuse at dilated version and then thin"); the mean probability is kept for argmax ties
                from vesuvius_p2sd.data.dataset import thin_sheets

                fused = []
                for i, d in enumerate(decoded):
                    slab = _largest_component(d >= float(tta_vote) - 1e-6)   # vote 0.5 = majority of the dilated masks, ~0 = union
                    if tta_thin == "erode":
                        # erode the slab by the dilation radius (= morphological closing of the fused mask): the majority
                        # mask with its sub-2r holes filled, thickness kept (the medial rule below thinned too far, 2026-09-06)
                        thin = ~dilate_mask(~slab, int(tta_dilate), device=device)
                    else:
                        thin = thin_sheets(slab.astype(np.int16), 3, 1.0 + int(tta_dilate)) > 0
                    pm = tta_probmean[i]
                    fused.append(np.where(thin, np.maximum(pm, sheet_threshold + 1e-3), np.minimum(pm, sheet_threshold - 1e-3)).astype(np.float32))
                decoded = fused
            if tta_fuse == "sdf" and len(tta_list) > 1:
                # mean SDF -> pseudo-probability: 0.5 at the mean surface, 1 at depth >= R inside, 0 at distance >= R outside
                decoded = [np.clip(0.5 + d / (2.0 * 4), 0.0, 1.0).astype(np.float32) for d in decoded]
                if sheet_threshold != 0.5:
                    decoded = [np.clip(p - 0.5 + sheet_threshold, 0.0, 1.0) for p in decoded]   # keep 'prob >= sheet_threshold' = mean surface
            if len(tta_list) > 1 and hasattr(target_ae, "set_image"):
                with autocast_context(device, dtype):
                    target_ae.set_image(tensor)   # leave the decoder on the untransformed image for later stages
            for (c, _), prob in zip(to_decode, decoded):
                masks[c] = _largest_component(prob >= sheet_threshold) & valid
                probs[c] = prob.astype(np.float32)
        elif latent_fusion_samples > 0:
            # Fuse in LATENT space: mean of the subsets' predicted latents,
            # one AE decode per cluster. The latent is the sheet fingerprint
            # (same-sheet MSE ~3e-4), so averaging denoises the prediction
            # instead of picking one candidate mask.
            latents = []
            for start in range(0, len(point_sets), 8):
                latents.append(_predict_latents(
                    p2sd_model, tokens, coords, context,
                    point_sets[start:start + 8], shape, device, dtype,
                    label_sets=label_sets[start:start + 8] if label_sets else None,
                    latent_prompts=(torch.stack([
                        pooled_centroids[owners[i]]
                        for i in range(start, min(start + 8, len(owners)))])
                        if pooled_centroids is not None else None)))
            latents = torch.cat(latents)
            owner_t = torch.tensor(owners, device=latents.device)
            fused = torch.stack([latents[owner_t == c].mean(dim=0)
                                 for c in range(len(cluster_points_3d))])
            raw = []
            for start in range(0, len(fused), 8):
                raw.extend(_decode_latents(
                    target_ae, latent_codec, fused[start:start + 8], device, dtype))
            for c, prob in enumerate(raw):
                masks.append(_largest_component(prob >= sheet_threshold) & valid)
                probs.append(prob.astype(np.float32))
                chosen_log.append({"cluster": c, "fused_samples": int(n_sets)})
                # All sets fused; record the deterministic spread set (set 0).
                chosen_points.append(point_sets[c * n_sets][:prompt_points_per_sheet])
        else:
            raw = []
            for start in range(0, len(point_sets), 8):
                raw.extend(_decode_prompt_sets(
                    p2sd_model, target_ae, latent_codec, tokens, coords, context,
                    point_sets[start:start + 8], shape, device, dtype,
                    label_sets=label_sets[start:start + 8] if label_sets else None,
                    latent_prompts=(torch.stack([
                        pooled_centroids[owners[i]]
                        for i in range(start, min(start + 8, len(owners)))])
                        if pooled_centroids is not None else None)))
            # Select the best candidate per cluster by the GT-free topology reward.
            for c, cpoints in enumerate(cluster_points_3d):
                foreign = [p for c2, pts2 in enumerate(cluster_points_3d) if c2 != c for p in pts2]
                best_r, best_mask, best_prob_vol, rewards = -1e9, None, None, []
                best_points: list[list[int]] = []
                cand_masks: list[np.ndarray] = []
                cand_sets: list[int] = []
                for set_index, (owner, prob) in enumerate(zip(owners, raw)):
                    if owner != c:
                        continue
                    thresholded = prob >= sheet_threshold
                    _, fragment_count = _label6(thresholded)
                    candidate = _largest_component(thresholded) & valid
                    cand_masks.append(candidate)
                    cand_sets.append(set_index)
                    r = topology_reward(candidate, fragment_count, cpoints, foreign)
                    rewards.append(round(float(r), 4))
                    if r > best_r:
                        best_r, best_mask, best_prob_vol = r, candidate, prob
                        set_labels = label_sets[set_index] if label_sets else None
                        best_points = [p for i, p in enumerate(point_sets[set_index])
                                       if set_labels is None or set_labels[i] == 1]
                chosen_points.append(best_points)
                cluster_candidates.append({"masks": cand_masks, "sets": cand_sets})
                masks.append(best_mask if best_mask is not None else np.zeros(shape, dtype=bool))
                probs.append((best_prob_vol if best_prob_vol is not None else np.zeros(shape)).astype(np.float32))
                chosen_log.append({"cluster": c, "rewards": rewards, "chosen_reward": round(float(best_r), 4)})
        stability_log: list[dict] = []
        if cluster_stability_min_iou > 0 and not latent_fusion_samples and kept \
                and cluster_candidates \
                and not neg_points_per_sheet and not latent_prompt_condition:
            # Multi-round clustering (promptset probe, 2026-08-18 01:20):
            # instability is a CLUSTER property — when the N candidate subsets
            # of one cluster decode different objects (min pairwise IoU below
            # threshold; probe separation at ~0.70), the cluster mixes sheets
            # and no single 8-point set is right. Split its members in the
            # validated superpoint space: seeds = the latents of the two most
            # disagreeing subsets, members assigned by fingerprint-to-seed
            # MSE; each big-enough half is re-decoded (side rng, so the
            # champion path stays byte-identical when this gate is off) and
            # re-gated up to cluster_stability_depth rounds.
            side_rng = np.random.default_rng((int(seed) + 1) * 1000003)

            def min_iou_pair(cand_masks):
                worst, pair = 2.0, None
                for i in range(len(cand_masks)):
                    if not cand_masks[i].any():
                        continue
                    for j in range(i + 1, len(cand_masks)):
                        if not cand_masks[j].any():
                            continue
                        union = int((cand_masks[i] | cand_masks[j]).sum())
                        iou = int((cand_masks[i] & cand_masks[j]).sum()) / union if union else 0.0
                        if iou < worst:
                            worst, pair = iou, (i, j)
                return worst, pair

            def decode_and_select(cpoints, foreign):
                sets = [[list(p) for p in spread_points_3d(cpoints, prompt_points_per_sheet)]]
                for _ in range(n_sets - 1):
                    take = min(prompt_points_per_sheet, len(cpoints))
                    idx = side_rng.choice(len(cpoints), take, replace=False)
                    sets.append([list(cpoints[i]) for i in idx])
                decoded = []
                for start in range(0, len(sets), 8):
                    decoded.extend(_decode_prompt_sets(
                        p2sd_model, target_ae, latent_codec, tokens, coords, context,
                        sets[start:start + 8], shape, device, dtype))
                best_r, best = -1e9, None
                cand_masks = []
                for set_index, prob in enumerate(decoded):
                    thresholded = prob >= sheet_threshold
                    _, fragment_count = _label6(thresholded)
                    candidate = _largest_component(thresholded) & valid
                    cand_masks.append(candidate)
                    r = topology_reward(candidate, fragment_count, cpoints, foreign)
                    if r > best_r:
                        best_r = r
                        best = (candidate, prob.astype(np.float32), sets[set_index])
                return best, cand_masks, sets

            final = {"masks": [], "probs": [], "points": [], "centroids": []}

            def accept(mask, prob, pts, centroid):
                final["masks"].append(mask)
                final["probs"].append(prob)
                final["points"].append(pts)
                final["centroids"].append(centroid)

            for c in range(len(cluster_points_3d)):
                worst, pair = min_iou_pair(cluster_candidates[c]["masks"])
                if pair is None or worst >= cluster_stability_min_iou:
                    accept(masks[c], probs[c], chosen_points[c], kept_centroids[c])
                    continue
                event = {"cluster": c, "min_iou": round(float(worst), 3), "halves": []}
                # Work queue of member-index lists (into kept[c]) with depth.
                seed_sets = [point_sets[cluster_candidates[c]["sets"][k]] for k in pair]
                queue = [(list(range(len(kept[c]))), 1, seed_sets)]
                accepted_any = False
                while queue:
                    members, depth, seeds = queue.pop()
                    seed_latents = _predict_latents(
                        p2sd_model, tokens, coords, context, seeds,
                        shape, device, dtype).float().cpu().flatten(1)
                    member_fps = fingerprints[[kept[c][m] for m in members]]
                    assign = (member_fps - seed_latents[0]).pow(2).mean(dim=1) > \
                             (member_fps - seed_latents[1]).pow(2).mean(dim=1)
                    halves = [[m for m, a in zip(members, assign.tolist()) if a == side]
                              for side in (False, True)]
                    sizes_all = [len(h) for h in halves]
                    halves = [h for h in halves if len(h) > cluster_min_points]
                    if not halves:
                        event["halves"].append({"depth": depth, "verdict": "unsplittable",
                                                "sizes": sizes_all})
                        continue
                    if len(halves) == 1:
                        # Outlier purge: the minority side is below
                        # cluster_min_points (it could never decode alone), so
                        # re-decoding only the majority core strictly refines
                        # the prompt pool.
                        event["halves"].append({"depth": depth, "verdict": "purged_minority",
                                                "sizes": sizes_all})
                    for half in halves:
                        half_set = set(half)
                        hpoints = [cluster_points_3d[c][m] for m in half]
                        foreign = [p for c2, pts2 in enumerate(cluster_points_3d)
                                   if c2 != c for p in pts2]
                        foreign += [cluster_points_3d[c][m] for m in range(len(kept[c]))
                                    if m not in half_set]
                        best, cand_masks, cand_sets_local = decode_and_select(hpoints, foreign)
                        if best is None:
                            event["halves"].append({"depth": depth, "size": len(half),
                                                    "verdict": "no_decode"})
                            continue
                        sub_worst, sub_pair = min_iou_pair(cand_masks)
                        if sub_pair is not None and sub_worst < cluster_stability_min_iou \
                                and depth < int(cluster_stability_depth):
                            # Re-seed from the half's own most-disagreeing pair.
                            event["halves"].append({"depth": depth, "size": len(half),
                                                    "verdict": f"resplit@{round(float(sub_worst), 3)}"})
                            queue.append((half, depth + 1,
                                          [cand_sets_local[sub_pair[0]],
                                           cand_sets_local[sub_pair[1]]]))
                            continue
                        centroid = fingerprints[[kept[c][m] for m in half]].mean(dim=0)
                        accept(best[0], best[1], [[int(v) for v in p] for p in best[2]], centroid)
                        accepted_any = True
                        event["halves"].append({"depth": depth, "size": len(half),
                                                "verdict": "accepted",
                                                "voxels": int(best[0].sum()),
                                                "min_iou": round(float(sub_worst), 3)
                                                if sub_pair is not None else 1.0})
                if not accepted_any:
                    # Splitting failed everywhere — keep the original decode.
                    accept(masks[c], probs[c], chosen_points[c], kept_centroids[c])
                    event["halves"].append({"verdict": "fallback_original"})
                stability_log.append(event)
            masks, probs = final["masks"], final["probs"]
            chosen_points, kept_centroids = final["points"], final["centroids"]
        verify_log = self_consistency_filter(masks, verify_overlap_iou)
        next_id, assigned = claim_instances(
            instance_ids, best_prob, masks, probs, next_id=next_id,
            dedup_dilate_voxels=dedup_dilate_voxels, dedup_cover=dedup_cover,
            device=device,
            mask_latents=kept_centroids,
            instance_latents=instance_latents,
            latent_gate_mse=dedup_latent_gate_mse, fuse=fuse)
        for c, claimed_id in enumerate(assigned):
            if claimed_id > 0 and claimed_id not in sheet_points:
                sheet_points[claimed_id] = [[int(v) for v in p] for p in chosen_points[c]]
        log.append({"round": 0, "clusters": len(kept),
                    "click_filter": click_filter_log,
                    "cluster_sizes": [len(c) for c in kept], "assigned": assigned,
                    "candidates_per_cluster": n_sets, "selection": chosen_log,
                    "latent_fusion_samples": int(latent_fusion_samples),
                    "dedup_dilate_voxels": int(dedup_dilate_voxels),
                    "cluster_splits": split_count,
                    "purged_points": int(purged_points),
                    "died_clusters": int(died_clusters),
                    "dedup_latent_gate_mse": float(dedup_latent_gate_mse),
                    "verification": verify_log,
                    "stability": {"min_iou": float(cluster_stability_min_iou),
                                  "events": stability_log},
                    "new_instances": next_id - 1})
    # Residual coverage rounds: the single-shot cluster pass drops every click
    # cluster below cluster_min_points, so sheets that caught few of the
    # random clicks (small or thin ones) never decode -- the persistent
    # ~1.2-sheets/case coverage gap. Re-seed the unclaimed foreground at its
    # deepest points, but only decode seeds whose fingerprint is NOVEL versus
    # every existing instance: re-decoding an already-claimed sheet costs a
    # duplicate or a merge, not coverage.
    novelty_gate = residual_novelty_mse if residual_novelty_mse > 0 else cluster_mse_threshold
    for round_index in range(int(residual_rounds)):
        unclaimed = foreground & (instance_ids == 0)
        seeds = propose_seeds(
            unclaimed,
            max_seeds=int(residual_max_seeds),
            min_blob_voxels=(int(seed_min_blob_voxels) if seed_min_blob_voxels >= 0
                             else max(int(min_component_voxels), 1)),
        )
        if not seeds:
            break
        seed_points = [tuple(int(v) for v in s) for s in seeds]
        seed_fps = fingerprints_for(seed_points).reshape(len(seed_points), -1).float()
        known = list(instance_latents.values())
        novelty_min = []
        for i in range(len(seed_points)):
            novelty_min.append(min(
                (float((seed_fps[i] - k).pow(2).mean()) for k in known),
                default=float("inf")))
        novel = [i for i in range(len(seed_points)) if novelty_min[i] > novelty_gate]
        row = {"round": round_index + 1, "residual_seeds": len(seed_points),
               "novel": len(novel),
               "novelty_min_mse": [round(v, 4) if v != float("inf") else -1.0
                                   for v in novelty_min]}
        if not novel:
            log.append({**row, "assigned": []})
            break
        prompt_sets = [[list(seed_points[i])] for i in novel]
        residual_probs: list[np.ndarray] = []
        for start in range(0, len(prompt_sets), 8):
            residual_probs.extend(_decode_prompt_sets(
                p2sd_model, target_ae, latent_codec, tokens, coords, context,
                prompt_sets[start:start + 8], shape, device, dtype))
        residual_masks = [_largest_component(p >= sheet_threshold) & valid
                          for p in residual_probs]
        # Residual seeds sit in foreground the cluster pass could NOT claim --
        # selection-biased toward ambiguous regions -- so their single-click
        # decodes need the self-consistency gate even more than cluster masks.
        row["verification"] = self_consistency_filter(
            residual_masks,
            residual_verify_iou if residual_verify_iou >= 0 else verify_overlap_iou)
        next_id, assigned = claim_instances(
            instance_ids, best_prob, residual_masks, residual_probs, next_id=next_id,
            dedup_dilate_voxels=dedup_dilate_voxels, dedup_cover=dedup_cover,
            device=device,
            mask_latents=[seed_fps[i] for i in novel],
            instance_latents=instance_latents,
            latent_gate_mse=dedup_latent_gate_mse, fuse=fuse)
        for j, claimed_id in enumerate(assigned):
            if claimed_id > 0 and claimed_id not in sheet_points:
                sheet_points[claimed_id] = [[int(v) for v in p] for p in prompt_sets[j]]
        log.append({**row, "assigned": assigned})
    if min_component_voxels > 0:
        if min_component_connectivity:
            # CC dusting per instance mask (fragments below the size gate drop; the
            # instance survives if anything remains), vs the default whole-instance drop.
            import cc3d
            for instance in [int(v) for v in np.unique(instance_ids[instance_ids > 0])]:
                m = instance_ids == instance
                cc = cc3d.connected_components(
                    np.ascontiguousarray(m), connectivity=int(min_component_connectivity))
                sizes = np.bincount(cc.ravel())
                small = np.where(sizes < min_component_voxels)[0]
                small = small[small > 0]
                if small.size:
                    instance_ids[m & np.isin(cc, small)] = 0
        else:
            ids, counts_ = np.unique(instance_ids[instance_ids > 0], return_counts=True)
            for instance, size in zip(ids, counts_):
                if size < min_component_voxels:
                    instance_ids[instance_ids == instance] = 0
    surviving = {int(v) for v in np.unique(instance_ids) if v > 0}
    return {"instance_ids": instance_ids, "foreground": foreground, "seed_log": log,
            "fused": fuse,
            "cluster_sizes": sorted((len(c) for c in clusters), reverse=True),
            "points": {"clicks": [[int(v) for v in p] for p in points],
                       "sheets": {int(k): v for k, v in sheet_points.items()
                                  if int(k) in surviving}}}


def run_case_voxel_embed(
    *,
    binseg_model,
    image: np.ndarray,
    valid: np.ndarray,
    device,
    dtype,
    binary_threshold: float,
    min_component_voxels: int,
    embed_points: int = 2048,
    embed_mse_threshold: float = 0.6,
    embed_min_points: int = 16,
    embed_assign_floor: float = 0.5,
    assign_chunk: int = 262144,
    seed: int = 0,
    foreground_models=None,
    embed_ball_radius: int = 0,
    embed_ball_points: int = 32,
) -> dict[str, Any]:
    """Instances directly from the 0024-style voxel embedding -- no P2SD.

    Sampled foreground voxels are embedded through the contrast head,
    leader-clustered by cosine (cluster_latents uses MEAN-over-dims MSE, so
    for normalized d-dim vectors the threshold reads (2 - 2cos)/d — e.g.
    cos 0.5 at 64-d is 0.0156, NOT 1.0), and every foreground voxel
    is assigned to its nearest centroid (cos >= ``embed_assign_floor``, else
    unlabeled). The union is exactly the binary foreground; what this mode
    tests is whether the embedding alone individuates sheets.

    ``foreground_models``: when given (list), foreground is the AND-consensus
    of THEIR decodes and ``binseg_model`` supplies only the embedding via
    embed_context (no decode) -- required for ctx-attn distill/contrast
    checkpoints (0039/0040), whose frozen-decoder logits are unconstrained.
    ``embed_ball_radius`` > 0 averages each CLUSTERING fingerprint over up to
    ``embed_ball_points`` foreground voxels within that Chebyshev radius
    (noisy-per-point embeddings, e.g. 0040 prototype contrast, become usable
    as mean fingerprints); voxel ASSIGNMENT stays per-voxel.
    """

    rng = np.random.default_rng(seed)
    tensor = torch.from_numpy(image[None, None].astype(np.float32)).to(device)
    with torch.no_grad(), autocast_context(device, dtype):
        context = binseg_model.refine_context(binseg_model.encode_context(tensor))
        if foreground_models:
            embedding = binseg_model.embed_context(context)
            foreground_mask = None
            for f_model in foreground_models:
                f_logits = f_model(tensor)
                mask = torch.sigmoid(f_logits.float())[0, 0] >= binary_threshold
                foreground_mask = mask if foreground_mask is None else foreground_mask & mask
            foreground = foreground_mask.cpu().numpy()
        else:
            logits, embedding = binseg_model.decode_context_with_embedding(context)
            foreground = (torch.sigmoid(logits.float())[0, 0] >= binary_threshold).cpu().numpy()
    foreground &= valid
    shape = image.shape
    result = {"instance_ids": np.zeros(shape, dtype=np.int16), "foreground": foreground,
              "seed_log": [], "cluster_sizes": []}
    coords_all = np.argwhere(foreground)
    if coords_all.shape[0] == 0:
        return result

    size = torch.tensor(shape, device=device, dtype=torch.float32).clamp_min(1)

    def embed_at(coords_np: np.ndarray) -> torch.Tensor:
        # Voxel-centre convention, matching train-time _embed_at_coords
        # (fixed 2026-08-14; the previous c/(N-1) + align_corners=True form
        # misaligned the stride-32 tap by up to ~16 voxels).
        coords = torch.from_numpy(coords_np).to(device).float()
        grid = ((coords + 0.5) / size) * 2 - 1
        grid_xyz = grid[:, [2, 1, 0]].view(1, -1, 1, 1, 3)
        with torch.no_grad():
            sampled = torch.nn.functional.grid_sample(
                embedding.float(), grid_xyz, mode="bilinear", align_corners=False)
        return torch.nn.functional.normalize(sampled[0, :, :, 0, 0].transpose(0, 1), dim=1)

    sample_idx = rng.choice(coords_all.shape[0], min(embed_points, coords_all.shape[0]),
                            replace=False)
    if embed_ball_radius > 0:
        # Mean fingerprint over a foreground ball around each seed.
        r = int(embed_ball_radius)
        offsets = np.stack(np.meshgrid(*([np.arange(-r, r + 1)] * 3),
                                       indexing="ij"), axis=-1).reshape(-1, 3)
        fingerprints = []
        for seed_coord in coords_all[sample_idx]:
            neighbors = seed_coord + offsets
            inb = np.all((neighbors >= 0) & (neighbors < np.array(shape)), axis=1)
            neighbors = neighbors[inb]
            neighbors = neighbors[foreground[neighbors[:, 0], neighbors[:, 1], neighbors[:, 2]]]
            if len(neighbors) > embed_ball_points:
                neighbors = neighbors[rng.choice(len(neighbors), embed_ball_points,
                                                 replace=False)]
            fingerprints.append(torch.nn.functional.normalize(
                embed_at(neighbors).mean(dim=0), dim=0))
        cluster_inputs = torch.stack(fingerprints).cpu()
    else:
        cluster_inputs = embed_at(coords_all[sample_idx]).cpu()
    clusters = cluster_latents(cluster_inputs, mse_threshold=embed_mse_threshold)
    kept = [c for c in clusters if len(c) >= embed_min_points]
    result["cluster_sizes"] = sorted((len(c) for c in clusters), reverse=True)
    if not kept:
        return result
    member_embeds = cluster_inputs.to(device)
    centroids = torch.nn.functional.normalize(
        torch.stack([member_embeds[list(c)].mean(dim=0) for c in kept]), dim=1)

    instance_ids = np.zeros(shape, dtype=np.int16)
    assigned_counts = np.zeros(len(kept), dtype=np.int64)
    for start in range(0, coords_all.shape[0], assign_chunk):
        chunk = coords_all[start:start + assign_chunk]
        cos = embed_at(chunk) @ centroids.T
        best_cos, best_id = cos.max(dim=1)
        keep_mask = (best_cos >= embed_assign_floor).cpu().numpy()
        ids = (best_id.cpu().numpy() + 1).astype(np.int16)
        sel = chunk[keep_mask]
        instance_ids[sel[:, 0], sel[:, 1], sel[:, 2]] = ids[keep_mask]
        np.add.at(assigned_counts, best_id.cpu().numpy()[keep_mask] - 0, 1)
    if min_component_voxels > 0:
        ids_u, counts_u = np.unique(instance_ids[instance_ids > 0], return_counts=True)
        for instance, size in zip(ids_u, counts_u):
            if size < min_component_voxels:
                instance_ids[instance_ids == instance] = 0
    result["instance_ids"] = instance_ids
    result["seed_log"] = [{"round": 0, "clusters": len(kept),
                           "cluster_sizes": [len(c) for c in kept],
                           "assigned_voxels": assigned_counts.tolist()}]
    return result


class _UnionAsBinseg(torch.nn.Module):
    """Adapter: makes a UnionMaskDecoder callable like a binseg model (image -> logits) for the
    foreground sites of the instance pipeline (0066)."""

    def __init__(self, union) -> None:
        super().__init__()
        self.union = union

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        self.union.set_image(image, self.union._p2sd)
        ae = self.union._ae
        ae = getattr(ae, "target_ae", ae)   # unwrap a refined/image-conditioned decoder to the plain AE helpers
        return self.union.logits(ae)


def load_p2sd_stack(run_dir: Path, device, checkpoint_path: Path | None = None):
    cfg = load_config(run_dir / "resolved_config.yaml")
    target_cfg = cfg.get("target_ae", {})
    target_ae, _ = load_ae_from_config(target_cfg.get("config_path"), target_cfg.get("checkpoint_path"))
    target_ae = target_ae.to(device).eval()
    model = build_p2sd(cfg, latent_channels=target_ae.latent_channels).to(device)
    load_model_state(model, checkpoint_path or run_dir / "last.pt")
    model.eval()
    codec = build_static_latent_codec(
        cfg.get("p2sd", {}).get("loss", {}).get("latent_normalization", {}),
        latent_channels=target_ae.latent_channels, device=device)
    return model, target_ae, codec, cfg


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binseg_run_dir", required=True, nargs="+",
                        help="Binseg proposer run dir(s). Multiple dirs = bitwise-AND ensemble of "
                             "their thresholded foregrounds (cluster mode only); the FIRST dir "
                             "supplies device/dataset config.")
    parser.add_argument("--p2sd_run_dir", default=None,
                        help="Required except in voxel_embed mode (which never decodes).")
    parser.add_argument("--split", default="val")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--purpose", default=None,
                        help="One sentence: what question this experiment answers and vs "
                             "what baseline. Recorded in the output dir's lineage.json.")
    parser.add_argument("--binary_threshold", type=float, default=0.5)
    parser.add_argument("--external_mask_dir", default=None,
                        help="Dir of instances_<case>.npy; voxels >0 replace the binseg "
                             "threshold mask as the click-seeding foreground (cluster mode).")
    parser.add_argument("--sheet_threshold", type=float, default=0.5)
    parser.add_argument("--save_fused_prob", type=int, default=0,
                        help="Cluster mode: also save per-case fused_<case>.npz with per-voxel "
                             "decode-fusion fields over all ACCEPTED masks -- sum (float16), "
                             "noisy-OR and max (uint8/255), claim count n (uint8). Champion "
                             "path is untouched when 0.")
    parser.add_argument("--max_rounds", type=int, default=6)
    parser.add_argument("--seeds_per_round", type=int, default=8)
    parser.add_argument("--min_blob_voxels", type=int, default=2000)
    parser.add_argument("--min_component_voxels", type=int, default=500)
    parser.add_argument("--min_component_connectivity", type=int, default=0, choices=[0, 6, 26],
                        help="0 (default): drop whole instances under --min_component_voxels (historic "
                             "behavior). 6/26: instead dust each instance mask's connected components "
                             "at that connectivity (cluster mode only).")
    parser.add_argument("--seed_min_blob_voxels", type=int, default=-1,
                        help="Residual-round seed proposal blob floor; -1 (default) keeps the historic "
                             "max(min_component_voxels, 1) so ablating the dust gate does not change seeding.")
    parser.add_argument("--mode", choices=("rounds", "cluster", "voxel_embed"), default="rounds",
                        help="rounds: seed/decode/re-seed iteration. cluster: latent-fingerprint "
                             "clustering of M random clicks, then one K-click decode per cluster. "
                             "voxel_embed: cluster the binseg contrast-head voxel embedding "
                             "directly (no P2SD; binseg_run_dir must be a contrast-head run).")
    parser.add_argument("--cluster_points", type=int, default=256)
    parser.add_argument("--cluster_mse_threshold", type=float, default=0.15)
    parser.add_argument("--cluster_min_points", type=int, default=8,
                        help="Keep clusters with MORE than this many clicks.")
    parser.add_argument("--prompt_points_per_sheet", type=int, default=8)
    parser.add_argument("--candidates_per_cluster", type=int, default=1,
                        help="Best-of-N: decode N click-subset candidates per cluster and keep "
                             "the best by the GT-free topology reward (1 = no selection).")
    parser.add_argument("--latent_fusion_samples", type=int, default=0,
                        help="Fuse N click-subsets per cluster in LATENT space (mean) and decode "
                             "once, instead of best-of-N mask selection (0 = off).")
    parser.add_argument("--dedup_dilate_voxels", type=int, default=0,
                        help="Dilate new masks by this many voxels (GPU max-pool) when checking for "
                             "near-duplicate instances; twins offset 1-2 voxels barely overlap so "
                             "plain IoU misses them (0 = IoU dedup only).")
    parser.add_argument("--dedup_cover", type=float, default=0.5,
                        help="Dilated-overlap fraction of min(new, existing) above which the new "
                             "mask is treated as a twin and skipped.")
    parser.add_argument("--cluster_p2sd_run_dir", default=None, nargs="+",
                        help="P2SD(s) used ONLY as latent fingerprints for clustering. One dir = "
                             "raw latents + absolute mse threshold (v8-compatible). Multiple dirs "
                             "= per-case-normalized ensemble; the threshold then reads as a "
                             "fraction of the median pairwise distance (try 0.05-0.2).")
    parser.add_argument("--verify_overlap_iou", type=float, default=0.0,
                        help="Self-consistency gate: re-decode each mask from 8 of its own random "
                             "points and discard the sheet if the two decodes' IoU is below this "
                             "(0 = off).")
    parser.add_argument("--click_min_spacing", type=float, default=0.0,
                        help="Blue-noise click pool: oversample foreground clicks then keep only "
                             "points at least this many voxels apart (0 = off). Near-coincident "
                             "clicks carry correlated latents.")
    parser.add_argument("--click_oversample", type=int, default=4,
                        help="Oversampling factor for the click pool before spacing thinning.")
    parser.add_argument("--click_per_blob_floor", type=int, default=0,
                        help="Stratified clicks: guarantee every 6-conn foreground blob >= "
                             "--click_blob_min_voxels at least this many clicks (0 = off, "
                             "uniform). Set to cluster_min_points so small isolated sheets "
                             "can form clusters; total clicks may exceed --cluster_points "
                             "by the floor top-ups.")
    parser.add_argument("--click_blob_min_voxels", type=int, default=500,
                        help="Blobs below this size get no floor (dust; 500 = official CC "
                             "filter).")
    parser.add_argument("--cluster_purge_margin", type=float, default=0.0,
                        help="Purity purge: drop a cluster member unless its MSE to the nearest "
                             "OTHER centroid is at least this ratio times its MSE to its own "
                             "centroid (0 = off; try 2-4). Ambiguous clicks serve no sheet.")
    parser.add_argument("--cluster_purge_disagreement", type=float, default=0.0,
                        help="Ensemble purity purge: drop a member whose per-model normalized "
                             "MSEs to its own centroid spread by more than this max/min ratio "
                             "(0 = off; needs >=2 --cluster_p2sd_run_dir).")
    parser.add_argument("--cluster_split_mse", type=float, default=0.0,
                        help="Two-means split of clusters whose fingerprint sub-centroids sit "
                             "further apart than this MSE (same units as --cluster_mse_threshold; "
                             "0 = off). Targets touching-sheet clusters welded by one gap click.")
    parser.add_argument("--cluster_stability_min_iou", type=float, default=0.0,
                        help="Multi-round clustering gate (0 = off). A cluster whose N candidate "
                             "decodes disagree (min pairwise IoU below this; promptset probe "
                             "2026-08-18 separates healthy/unstable at ~0.70) mixes sheets: split "
                             "its members in superpoint space (seeds = the two most-disagreeing "
                             "subsets' latents, members assigned by fingerprint-to-seed MSE), "
                             "re-decode each big-enough half with side-rng subsets, re-gate up to "
                             "--cluster_stability_depth rounds. Champion rng stream untouched.")
    parser.add_argument("--cluster_stability_depth", type=int, default=2,
                        help="Max split rounds per unstable cluster (2 = up to 4-way).")
    parser.add_argument("--cluster_subset_knn", type=int, default=0,
                        help="kNN-representative mode (0 = off): draw M random prompt subsets "
                             "per cluster, score each by the mean latent MSE to its "
                             "--cluster_subset_knn_neighbors nearest subsets (no decodes, no "
                             "IoU), decode ONLY the best subset — and DROP the whole cluster "
                             "when even the best score exceeds --cluster_drop_mse (unstable "
                             "clusters re-decode to the same attractor, e18; not decoding is "
                             "the only union-changing move). Replaces best-of-N selection.")
    parser.add_argument("--cluster_subset_knn_fuse", type=int, default=1,
                        help="kNN mode: decode the mean latent of the representative subset and "
                             "its M-1 nearest subsets (one AE decode per cluster). 1 = representative "
                             "only (champion path).")
    parser.add_argument("--cluster_subset_knn_neighbors", type=int, default=3,
                        help="N nearest subsets averaged into each subset's purity score.")
    parser.add_argument("--click_self_decode_min_prob", type=float, default=0.0,
                        help="Drop sampled clicks whose own single-click decode probability AT "
                             "the click is below this, before clustering (0 = off; calibrated "
                             "2026-08-18: 0.5 culls ~29%% of clicks at ~2:1 junk:real, decode "
                             "is bimodal so 0.3-0.7 behave alike).")
    parser.add_argument("--cluster_rep_merge_mse", type=float, default=0.0,
                        help="Cross-cluster latent dedup in kNN mode (0 = off): clusters whose "
                             "BEST subsets' latents agree within this MSE are one sheet; each "
                             "union-find group keeps its most compact member (smallest kNN "
                             "score) and the rest are suppressed before decoding. Catches the "
                             "flush parallel-slab duplicates voxel dedup misses at cover 0.5.")
    parser.add_argument("--cluster_drop_mse", type=float, default=0.008,
                        help="Drop threshold on the best subset's kNN score (metric-space "
                             "mean-over-dims MSE; promptset probe: healthy clusters "
                             "~1e-4..2e-3, catastrophic >2e-2).")
    parser.add_argument("--dedup_latent_gate_mse", type=float, default=0.0,
                        help="Veto IoU/twin dedup merges whose fingerprints differ by more than "
                             "this MSE (0 = off): geometric overlap of DISTINCT sheets stays two "
                             "instances.")
    parser.add_argument("--residual_rounds", type=int, default=0,
                        help="After the cluster pass, re-seed unclaimed foreground this many times "
                             "(deepest-point seeds), decoding only latent-novel seeds (0 = off).")
    parser.add_argument("--residual_novelty_mse", type=float, default=0.0,
                        help="A residual seed decodes only if its fingerprint MSE to every "
                             "existing instance exceeds this (default 0 = reuse "
                             "--cluster_mse_threshold).")
    parser.add_argument("--residual_max_seeds", type=int, default=8,
                        help="Seed cap per residual round.")
    parser.add_argument("--residual_verify_iou", type=float, default=-1.0,
                        help="Self-consistency gate applied to residual decodes only "
                             "(-1 = inherit --verify_overlap_iou).")
    parser.add_argument("--latent_prompt_condition", type=int, default=0,
                        help="Feed each cluster's pooled centroid latent to a "
                             "latent-prompt-trained decoder as an extra prompt token.")
    parser.add_argument("--neg_points_per_sheet", type=int, default=0,
                        help="Append this many NEAREST foreign-cluster clicks as label-0 prompts "
                             "per decode (needs a negative-click-trained decoder, e.g. 0030).")
    parser.add_argument("--tta", type=int, default=0,
                        help="Decoder-output TTA for the kNN decode path: 0/1 off, 8 axis flips, 16 = full "
                             "D4(y,x) x flip(z) training-symmetry group (probabilities averaged at full res).")
    parser.add_argument("--tta_fuse", choices=("mean", "max", "sdf", "dilthin"), default="mean",
                        help="How the transformed decodes are fused at full res: mean (pinholes where they disagree), max "
                             "(thick shells), sdf (mean truncated signed distance), or dilthin (majority of masks dilated by "
                             "--tta_dilate, thinned back to the medial ~3 voxels; user proposal 2026-09-06).")
    parser.add_argument("--tta_dilate", type=int, default=2, help="dilthin: dilation radius before fusing")
    parser.add_argument("--tta_vote", type=float, default=0.5, help="dilthin: fraction of transforms whose dilated mask must cover a voxel (0.5 majority, 0.1 ~ union)")
    parser.add_argument("--tta_thin", choices=("medial", "erode"), default="medial",
                        help="dilthin: how the fused slab is thinned back: medial (EDT band, ~3 voxels) or erode (by --tta_dilate = closing).")
    parser.add_argument("--binseg_tta", type=int, default=0,
                        help="8-flip TTA on the binseg proposer before click sampling.")
    parser.add_argument("--seed", type=int, default=0,
                        help="Pipeline RNG seed (click sampling / candidate jitter) -- vary for "
                             "multi-seed ensembling via merge_instance_runs.")
    parser.add_argument("--embed_points", type=int, default=2048,
                        help="voxel_embed mode: foreground voxels sampled for clustering.")
    parser.add_argument("--embed_mse_threshold", type=float, default=0.6,
                        help="voxel_embed mode: MSE threshold on normalized embeddings "
                             "(0.6 = cosine 0.7).")
    parser.add_argument("--embed_min_points", type=int, default=16)
    parser.add_argument("--embed_run_dir", default=None,
                        help="voxel_embed mode: binseg run supplying the EMBEDDING "
                             "(ctx-attn distill/contrast checkpoint). When set, "
                             "--binseg_run_dir keeps its usual role as foreground "
                             "provider (AND-consensus) and this model never decodes.")
    parser.add_argument("--embed_ball_radius", type=int, default=0,
                        help="voxel_embed mode: average each clustering fingerprint "
                             "over foreground voxels within this Chebyshev radius "
                             "(0 = single-point fingerprints).")
    parser.add_argument("--embed_ball_points", type=int, default=32)
    parser.add_argument("--embed_fingerprint_run_dir", default=None,
                        help="cluster mode: take click-clustering fingerprints from this "
                             "ctx-attn embedding run instead of P2SD click latents "
                             "(decode path unchanged). Threshold comes from "
                             "--embed_fp_mse_threshold; --embed_ball_radius applies.")
    parser.add_argument("--embed_fp_mse_threshold", type=float, default=0.015,
                        help="cluster threshold when --embed_fingerprint_run_dir is set "
                             "(mean-over-dims MSE on normalized 64-d: (2-2cos)/64).")
    parser.add_argument("--embed_assign_floor", type=float, default=0.5,
                        help="voxel_embed mode: minimum cosine to a centroid for a foreground "
                             "voxel to be assigned.")
    parser.add_argument("--refine_points", type=int, default=3,
                        help="Self-refinement clicks sampled from the first-pass mask core "
                             "(0 = single-click decode only).")
    parser.add_argument("--max_cases", type=int, default=None)
    parser.add_argument("--case_start", type=int, default=0,
                        help="Skip this many manifest rows first (multi-GPU sharding).")
    parser.add_argument("--clip_to_valid", type=int, default=1,
                        help="1 (default): restrict foreground/masks to labeled voxels (honest "
                             "eval). 0: predict everywhere (pseudo-label generation).")
    parser.add_argument("--manifest_path", default=None,
                        help="Explicit manifest (e.g. an unlabeled-cube manifest from "
                             "fetch_scroll_cubes). Rows without components/ignore paths get no "
                             "metrics and are treated as fully valid.")
    parser.add_argument("--visualize_worst_n", type=int, default=4)
    parser.add_argument("--save_pred_nifti", type=int, default=1,
                        help="Save every case's final instance map as NIfTI under pred_nifti/ (default on).")
    parser.add_argument("--device", default=None)
    parser.add_argument("--amp_dtype", default=None,
                        help="Override autocast dtype (bfloat16|float16|float32). Default: the "
                             "binseg config's training.amp_dtype. Needed on GPUs without bf16 "
                             "(e.g. Kaggle T4).")
    parser.add_argument("--skip_existing", action="store_true",
                        help="Skip cases whose instances file already exists in --output_dir (per-case resume after a "
                             "crash, e.g. the Blackwell Xid-31 faults of 2026-08-28).")
    parser.add_argument("--union_decoder_path", default=None,
                        help="UnionMaskDecoder checkpoint (union_last.pt, 0066): the foreground/seed logits come "
                             "from it (through the --p2sd_run_dir encoder + context) instead of the binseg model(s).")
    parser.add_argument("--refine_encoder_run_dir", default=None,
                        help="P2SD run whose (frozen) image encoder feeds the refinement head's image taps "
                             "(default: --p2sd_run_dir). Needed when decoding with a P2SD whose encoder differs "
                             "from the one the head was trained on (e.g. 0058 decode + 0062 head -> 0033).")
    parser.add_argument("--refine_head_path", default=None,
                        help="SheetRefineHead checkpoint (refine_last.pt): every P2SD decode goes "
                             "through the image-conditioned refinement (0061).")
    parser.add_argument("--decode_batch", type=int, default=8,
                        help="AE-decode chunk size (exact; VRAM-linear). 1 fits a 15 GiB T4.")
    args = parser.parse_args(argv)
    global _DECODE_BATCH
    _DECODE_BATCH = max(1, int(args.decode_batch))

    binseg_dirs = [Path(d) for d in (args.binseg_run_dir if isinstance(args.binseg_run_dir, list)
                                     else [args.binseg_run_dir])]
    binseg_dir = binseg_dirs[0]
    binseg_cfg = load_config(binseg_dir / "resolved_config.yaml")
    if args.device is not None:
        binseg_cfg["device"] = args.device
    device = get_device(binseg_cfg)
    dtype = amp_dtype(args.amp_dtype or binseg_cfg.get("training", {}).get("amp_dtype"))

    binseg_pool = []
    for d in binseg_dirs:
        cfg_d = load_config(d / "resolved_config.yaml")
        cfg_d["device"] = str(device)
        model_d = build_binary_seg_model(cfg_d).to(device)
        state = torch.load(d / "last.pt", map_location="cpu", weights_only=False)
        model_d.load_state_dict(state.get("model", state), strict=True)
        model_d.eval()
        binseg_pool.append(model_d)
    binseg_model = binseg_pool[0] if len(binseg_pool) == 1 else binseg_pool
    p2sd_model = target_ae = latent_codec = cluster_p2sd_model = None
    if args.mode != "voxel_embed":
        if not args.p2sd_run_dir:
            raise ValueError("--p2sd_run_dir is required outside voxel_embed mode")
        p2sd_model, target_ae, latent_codec, _ = load_p2sd_stack(Path(args.p2sd_run_dir), device)
        if args.refine_head_path:
            from vesuvius_p2sd.models.refine_head import load_refined_decoder

            enc_model = (load_p2sd_stack(Path(args.refine_encoder_run_dir), device)[0]
                         if args.refine_encoder_run_dir else p2sd_model)
            target_ae = load_refined_decoder(target_ae, enc_model.image_encoder, args.refine_head_path, device)
            print(f"[refine] decoding through {args.refine_head_path} (encoder taps from "
                  f"{args.refine_encoder_run_dir or args.p2sd_run_dir})", flush=True)
        if args.union_decoder_path:
            from vesuvius_p2sd.models.sheet_decoder import load_union_decoder

            union_model = load_union_decoder(args.union_decoder_path, device)
            union_model._p2sd = p2sd_model
            union_model._ae = target_ae
            print(f"[union] foreground from {args.union_decoder_path} (context from {args.p2sd_run_dir})", flush=True)
            binseg_model = _UnionAsBinseg(union_model)
        if args.cluster_p2sd_run_dir:
            dirs = (args.cluster_p2sd_run_dir if isinstance(args.cluster_p2sd_run_dir, list)
                    else [args.cluster_p2sd_run_dir])
            loaded = [load_p2sd_stack(Path(d), device)[0] for d in dirs]
            cluster_p2sd_model = loaded[0] if len(loaded) == 1 else loaded
    embed_model = None
    embed_dir = (args.embed_run_dir if args.mode == "voxel_embed"
                 else args.embed_fingerprint_run_dir)
    if embed_dir:
        e = Path(embed_dir)
        cfg_e = load_config(e / "resolved_config.yaml")
        cfg_e["device"] = str(device)
        embed_model = build_binary_seg_model(cfg_e).to(device)
        state = torch.load(e / "last.pt", map_location="cpu", weights_only=False)
        embed_model.load_state_dict(state.get("model", state), strict=True)
        embed_model.eval()
        if embed_model.contrast_head is None:
            raise ValueError("--embed_run_dir must be a run trained with binary_seg.contrast")
    elif args.mode == "voxel_embed" and binseg_model.contrast_head is None:
        raise ValueError("voxel_embed mode needs a binseg run trained with binary_seg.contrast "
                         "(or pass --embed_run_dir)")

    dataset_root = Path(binseg_cfg["data"]["dataset_root"])
    if args.manifest_path:
        rows = load_jsonl(Path(args.manifest_path))
    else:
        rows = load_jsonl(dataset_root / f"manifest_{args.split}_ignore.jsonl")
    rows = rows[int(args.case_start):]
    if args.max_cases is not None:
        rows = rows[: int(args.max_cases)]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    import sys as _sys
    write_experiment_provenance(
        output_dir, argv=_sys.argv, purpose=args.purpose,
        parents={"p2sd_run_dir": args.p2sd_run_dir, "refine_head_path": args.refine_head_path,
                 "binseg_run_dir": " ".join(args.binseg_run_dir)
                 if isinstance(args.binseg_run_dir, list) else args.binseg_run_dir,
                 "cluster_p2sd_run_dir": " ".join(args.cluster_p2sd_run_dir)
                 if args.cluster_p2sd_run_dir else None,
                 "embed_fingerprint_run_dir": args.embed_fingerprint_run_dir})

    case_rows: list[dict[str, Any]] = []
    start = time.monotonic()
    for case_index, row in enumerate(rows, start=1):
        if args.skip_existing and instances_file(output_dir, str(row["case_id"])) is not None:
            continue
        image = np.asarray(np.load(row["image_path"], mmap_mode="r"))
        shape = tuple(int(v) for v in row["shape"])
        has_labels = "components_path" in row
        components = (np.asarray(np.load(row["components_path"], mmap_mode="r"))
                      if has_labels else np.zeros(shape, dtype=np.int16))
        ignore = (load_ignore_mask(row["ignore_path"], shape)
                  if "ignore_path" in row else np.zeros(shape, dtype=bool))
        # Default contract: instances live on labeled voxels only (metrics are
        # honest there). --clip_to_valid 0 lifts that for pseudo-label
        # generation, where predictions INTO the ignore region are the point.
        valid = ~ignore if args.clip_to_valid else np.ones_like(ignore, dtype=bool)
        if args.mode == "voxel_embed":
            result = run_case_voxel_embed(
                binseg_model=embed_model if embed_model is not None else binseg_pool[0],
                image=image, valid=valid, device=device,
                dtype=dtype, binary_threshold=args.binary_threshold,
                min_component_voxels=args.min_component_voxels,
                embed_points=args.embed_points,
                embed_mse_threshold=args.embed_mse_threshold,
                embed_min_points=args.embed_min_points,
                embed_assign_floor=args.embed_assign_floor,
                foreground_models=binseg_pool if embed_model is not None else None,
                embed_ball_radius=args.embed_ball_radius,
                embed_ball_points=args.embed_ball_points)
        elif args.mode == "cluster":
            result = run_case_cluster(
                binseg_model=binseg_model, p2sd_model=p2sd_model, target_ae=target_ae,
                latent_codec=latent_codec, image=image, valid=valid, device=device, dtype=dtype,
                binary_threshold=args.binary_threshold, sheet_threshold=args.sheet_threshold,
                min_component_voxels=args.min_component_voxels,
                min_component_connectivity=args.min_component_connectivity,
                seed_min_blob_voxels=args.seed_min_blob_voxels,
                cluster_points=args.cluster_points,
                cluster_mse_threshold=(args.embed_fp_mse_threshold if embed_model is not None
                                       else args.cluster_mse_threshold),
                cluster_min_points=args.cluster_min_points,
                prompt_points_per_sheet=args.prompt_points_per_sheet,
                candidates_per_cluster=args.candidates_per_cluster,
                latent_fusion_samples=args.latent_fusion_samples,
                dedup_dilate_voxels=args.dedup_dilate_voxels,
                dedup_cover=args.dedup_cover,
                verify_overlap_iou=args.verify_overlap_iou,
                cluster_p2sd_model=cluster_p2sd_model,
                neg_points_per_sheet=args.neg_points_per_sheet,
                binseg_tta=int(args.binseg_tta),
                seed=int(args.seed),
                cluster_split_mse=args.cluster_split_mse,
                cluster_stability_min_iou=args.cluster_stability_min_iou,
                cluster_stability_depth=args.cluster_stability_depth,
                cluster_subset_knn=args.cluster_subset_knn,
                cluster_subset_knn_neighbors=args.cluster_subset_knn_neighbors,
                cluster_subset_knn_fuse=args.cluster_subset_knn_fuse,
                cluster_drop_mse=args.cluster_drop_mse,
                cluster_rep_merge_mse=args.cluster_rep_merge_mse,
                click_self_decode_min_prob=args.click_self_decode_min_prob,
                dedup_latent_gate_mse=args.dedup_latent_gate_mse,
                residual_rounds=args.residual_rounds,
                residual_novelty_mse=args.residual_novelty_mse,
                residual_max_seeds=args.residual_max_seeds,
                residual_verify_iou=args.residual_verify_iou,
                latent_prompt_condition=bool(args.latent_prompt_condition),
                embed_model=embed_model,
                embed_ball_radius=args.embed_ball_radius,
                embed_ball_points=args.embed_ball_points,
                click_min_spacing=args.click_min_spacing,
                click_oversample=args.click_oversample,
                click_per_blob_floor=args.click_per_blob_floor,
                click_blob_min_voxels=args.click_blob_min_voxels,
                cluster_purge_margin=args.cluster_purge_margin,
                cluster_purge_disagreement=args.cluster_purge_disagreement,
                external_mask=(np.asarray(load_instances(
                    Path(args.external_mask_dir), str(row["case_id"]))) > 0
                    if args.external_mask_dir else None),
                fuse_dump=bool(args.save_fused_prob),
                tta=int(args.tta), tta_fuse=str(args.tta_fuse), tta_dilate=int(args.tta_dilate), tta_vote=float(args.tta_vote),
                tta_thin=str(args.tta_thin))
        else:
            result = run_case(
                binseg_model=binseg_pool[0], p2sd_model=p2sd_model, target_ae=target_ae,
                latent_codec=latent_codec, image=image, valid=valid, device=device, dtype=dtype,
                binary_threshold=args.binary_threshold, sheet_threshold=args.sheet_threshold,
                max_rounds=args.max_rounds, seeds_per_round=args.seeds_per_round,
                min_blob_voxels=args.min_blob_voxels, min_component_voxels=args.min_component_voxels,
                refine_points=args.refine_points)
        gt = np.where(valid, components, 0)
        gt = np.where(filter_small_components(gt > 0, args.min_component_voxels), gt, 0)
        metrics = match_instances(result["instance_ids"], gt)
        case_rows.append({
            "case_id": str(row["case_id"]),
            **{k: v for k, v in metrics.items() if not isinstance(v, list)},
            "seed_rounds": len(result["seed_log"]),
            "seeds_total": int(sum(len(r.get("seeds", r.get("cluster_sizes", []))) for r in result["seed_log"])),
        })
        save_instances(output_dir, str(row["case_id"]), result["instance_ids"])
        if result.get("fused") is not None:
            fz = result["fused"]
            np.savez_compressed(
                output_dir / f"fused_{row['case_id']}.npz",
                sum=fz["sum"].astype(np.float16),
                nor=np.round(255.0 * (1.0 - fz["nor"])).astype(np.uint8),
                max=np.round(255.0 * fz["max"]).astype(np.uint8),
                n=fz["n"])
        (output_dir / f"seeds_{row['case_id']}.json").write_text(
            json.dumps(result["seed_log"]), encoding="utf-8")
        if "points" in result:
            # Click + per-sheet prompt provenance (consumed by the eval's
            # case_nifti visualization: clicks-on-binseg and prompt cubes).
            (output_dir / f"points_{row['case_id']}.json").write_text(
                json.dumps(result["points"]), encoding="utf-8")
        if args.save_pred_nifti:
            # The (possibly TTA/AND-ensembled) proposer foreground the clicks
            # sampled from -- saved per case alongside pred/gt.
            import nibabel as nib

            case_dir = output_dir / "case_nifti" / str(row["case_id"])
            case_dir.mkdir(parents=True, exist_ok=True)
            nib.save(nib.Nifti1Image(result["foreground"].astype(np.uint8),
                                     np.eye(4, dtype=np.float32)),
                     str(case_dir / "binseg_foreground.nii.gz"))
        if case_index % 5 == 0 or case_index == len(rows):
            print(f"[instance] {case_index}/{len(rows)} cases, {time.monotonic() - start:.0f}s", flush=True)

    def _mean(key: str) -> float:
        return float(np.mean([r[key] for r in case_rows]))

    summary = {
        "binseg_run_dir": str(binseg_dir),
        "p2sd_run_dir": str(args.p2sd_run_dir),
        "split": args.split,
        "case_count": len(case_rows),
        "settings": {k: getattr(args, k) for k in (
            "binary_threshold", "sheet_threshold", "max_rounds", "seeds_per_round",
            "min_blob_voxels", "min_component_voxels", "refine_points",
            "mode", "cluster_points", "cluster_mse_threshold", "cluster_min_points",
            "embed_points", "embed_mse_threshold", "embed_min_points",
            "embed_assign_floor", "embed_run_dir", "embed_ball_radius", "embed_ball_points",
            "embed_fingerprint_run_dir", "embed_fp_mse_threshold",
            "prompt_points_per_sheet", "candidates_per_cluster",
            "latent_fusion_samples", "dedup_dilate_voxels", "dedup_cover",
            "verify_overlap_iou", "cluster_p2sd_run_dir",
            "cluster_split_mse", "cluster_stability_min_iou", "cluster_stability_depth",
            "cluster_subset_knn", "cluster_subset_knn_neighbors", "cluster_subset_knn_fuse", "cluster_drop_mse",
            "cluster_rep_merge_mse", "click_self_decode_min_prob",
            "dedup_latent_gate_mse", "residual_rounds",
            "residual_novelty_mse", "residual_max_seeds", "residual_verify_iou",
            "latent_prompt_condition", "click_min_spacing", "click_oversample",
            "click_per_blob_floor", "click_blob_min_voxels",
            "cluster_purge_margin", "cluster_purge_disagreement", "save_fused_prob")},
        "macro_ap50": _mean("ap"),
        "macro_ap25": _mean("ap25"),
        "macro_mean_matched_iou": _mean("mean_matched_iou"),
        "macro_mean_gt_iou": _mean("mean_gt_iou"),
        "macro_merge_count": _mean("merge_count"),
        "macro_merged_gt_pairs": _mean("merged_gt_pairs"),
        "macro_split_count": _mean("split_count"),
        "macro_pred_count": _mean("pred_count"),
        "macro_gt_count": _mean("gt_count"),
        "total_true_positives": int(sum(r["true_positives"] for r in case_rows)),
        "total_false_positives": int(sum(r["false_positives"] for r in case_rows)),
        "total_false_negatives": int(sum(r["false_negatives"] for r in case_rows)),
        "elapsed_s": float(time.monotonic() - start),
    }
    with (output_dir / "case_metrics.jsonl").open("w", encoding="utf-8") as f:
        for r in case_rows:
            f.write(json.dumps(r, sort_keys=True) + "\n")
    write_json(output_dir / "summary.json", summary)

    # Every case's final prediction as NIfTI -- the deliverable people open in
    # a viewer. Prediction only: image and GT ship with the dataset, and the
    # worst-case bundles below carry all three for the cases worth studying.
    if args.save_pred_nifti:
        import nibabel as nib

        affine = np.eye(4, dtype=np.float32)
        rows_by_id = {str(r["case_id"]): r for r in rows}
        for r in case_rows:
            # One small folder per case with the pair people open together:
            # predicted instances + raw GT components (all sheets -- the
            # scorer's gt_count additionally applies valid-mask + CC-500).
            case_dir = output_dir / "case_nifti" / str(r["case_id"])
            case_dir.mkdir(parents=True, exist_ok=True)
            pred = load_instances(output_dir, str(r["case_id"]))
            nib.save(nib.Nifti1Image(pred.astype(np.int16), affine),
                     str(case_dir / "pred_instances.nii.gz"))
            comp_path = rows_by_id[str(r["case_id"])].get("components_path")
            if comp_path:   # label-free manifests (test mode) carry no GT
                components = np.asarray(np.load(comp_path, mmap_mode="r"))
                nib.save(nib.Nifti1Image(components.astype(np.int16), affine),
                         str(case_dir / "gt_components.nii.gz"))
            write_json(case_dir / "metrics.json", {
                **{k: r[k] for k in ("case_id", "pred_count", "gt_count", "ap25",
                                     "true_positives_25") if k in r},
                "gt_note": "gt_components is RAW (all sheets); metrics gt_count "
                           "applies valid-mask + CC-500 to it",
                "nifti_axis_order": "z,y,x (same as the source .npy arrays)",
            })

    # NIfTI for the worst cases by AP (instance ids as label volumes).
    if args.visualize_worst_n > 0:
        import nibabel as nib

        worst = sorted(case_rows, key=lambda r: r["ap"])[: args.visualize_worst_n]
        rows_by_id = {str(r["case_id"]): r for r in rows}
        affine = np.eye(4, dtype=np.float32)
        for rank, entry in enumerate(worst):
            row = rows_by_id[entry["case_id"]]
            directory = output_dir / "case_viz" / f"worst{rank:02d}_{entry['case_id']}"
            directory.mkdir(parents=True, exist_ok=True)
            pred = load_instances(output_dir, str(entry["case_id"]))
            components = np.asarray(np.load(row["components_path"], mmap_mode="r"))
            image = np.asarray(np.load(row["image_path"], mmap_mode="r"))
            nib.save(nib.Nifti1Image(np.ascontiguousarray(image).astype(np.uint8), affine),
                     str(directory / "image.nii.gz"))
            nib.save(nib.Nifti1Image(pred.astype(np.int16), affine),
                     str(directory / "pred_instances.nii.gz"))
            nib.save(nib.Nifti1Image(components.astype(np.int16), affine),
                     str(directory / "gt_components.nii.gz"))
            write_json(directory / "metrics.json", {
                **entry,
                "nifti_axis_order": "z,y,x (same as the source .npy arrays)",
                "nifti_label_maps": {
                    "pred_instances": "0 background, n = predicted instance n",
                    "gt_components": "0 background, n = GT sheet n",
                },
            })
    print(summary, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
