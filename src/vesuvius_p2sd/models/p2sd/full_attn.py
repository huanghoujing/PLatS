"""Full-attention point-to-sheet decoder."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from vesuvius_p2sd.models.ae.sheet_ae import ResBlock3d, norm_layer, patchify_start_stage


@dataclass(frozen=True)
class P2SDConfig:
    image_channels: int = 1
    latent_channels: int = 64
    encoder_channels: tuple[int, ...] = (8, 16, 24, 32, 48, 64)
    dim: int = 256
    depth: int = 4
    heads: int = 8
    mlp_ratio: float = 4.0
    use_rope: bool = True
    rope_mode: str = "axis_mixed"
    rope_max_freq: float | None = None  # None = derive N/2 from the grid
    image_context_depth: int = 4
    image_context_heads: int = 8
    image_context_mlp_ratio: float = 4.0
    image_context_use_rope: bool = True
    image_context_rope_mode: str = "axis_mixed"
    image_context_rope_max_freq: float | None = None
    # Task-12 (2026-08-14, user-directed): 1x1 head on image_context whose
    # point embeddings distill to JL-projected AE sheet anchors, trained
    # JOINTLY with the P2SD objective -- shapes the context the modulator
    # attends over toward instance-sheet features. 0 = off.
    context_distill_dim: int = 0
    # 2026-08-14 user-directed: fresh coordinate-distance head on the
    # PREDICTED latent, supervised by GT EDT directly (replaces distilling
    # through the frozen AE's weak query head, whose optimum-consistency came
    # at the cost of a 0.43-Dice teacher ceiling). 0 = off.
    latent_distance_hidden: int = 0
    prompt_composition_type: str = "point_feature_mlp"
    prompt_point_pe: str = "fourier"
    prompt_point_pe_num_bands: int = 8
    prompt_sampled_feature: str = "image_tokens_trilinear"
    prompt_fusion: str = "single_token"
    # Latent-conditioned decoding: accept a (noisy) sheet latent -- at
    # inference the fingerprint-cluster centroid -- as one extra prompt token,
    # so the modulator can refine an off-manifold averaged latent against the
    # image instead of decoding it blindly through the AE.
    latent_prompt_enabled: bool = False
    latent_prompt_channels: int = 64
    prompt_modulator_type: str = "broadcast_mlp"
    prompt_modulator_merge: str = "concat"
    prompt_modulator_depth: int = 1
    encoder_num_groups: int = 4
    encoder_dropout: float = 0.0
    encoder_stem: str = "conv"
    encoder_patch_size: int = 4
    # How the patchify stem is built (only used when encoder_stem == patchify):
    # plain = single stride-K conv (ViT patch embed); overlap = kernel 2K so
    # patches share context; early_conv = thin stack of stride-2 convs with one
    # nonlinearity, recovering early spatial detail the plain embed discards.
    encoder_patch_stem: str = "plain"
    freeze_image_encoder: bool = False
    freeze_image_encoder_through_stage: int = -1


class FullAttentionP2SD(nn.Module):
    """Predict an AE latent grid from an image crop and point prompts."""

    def __init__(self, cfg: P2SDConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.image_encoder = P2SDImageEncoder(
            in_channels=cfg.image_channels,
            channels=cfg.encoder_channels,
            num_groups=cfg.encoder_num_groups,
            dropout=cfg.encoder_dropout,
            stem=cfg.encoder_stem,
            patch_size=cfg.encoder_patch_size,
            patch_stem=cfg.encoder_patch_stem,
        )
        self.freeze_image_encoder = bool(cfg.freeze_image_encoder)
        self.freeze_image_encoder_through_stage = int(cfg.freeze_image_encoder_through_stage)
        if self.freeze_image_encoder and self.freeze_image_encoder_through_stage >= 0:
            raise ValueError(
                "freeze_image_encoder and freeze_image_encoder_through_stage are mutually exclusive"
            )
        if self.freeze_image_encoder:
            for parameter in self.image_encoder.parameters():
                parameter.requires_grad_(False)
        elif self.freeze_image_encoder_through_stage >= 0:
            self.image_encoder.freeze_through_stage(self.freeze_image_encoder_through_stage)
        self.image_feature_channels = self.image_encoder.out_channels
        self.grid_proj = nn.Conv3d(self.image_feature_channels, cfg.dim, 1)
        if cfg.image_context_depth < 0:
            raise ValueError("p2sd.image_context.depth must be >= 0")
        self.image_context_blocks = nn.ModuleList([
            AttentionBlock(
                cfg.dim,
                cfg.image_context_heads,
                cfg.image_context_mlp_ratio,
                use_rope=cfg.image_context_use_rope,
                rope_mode=cfg.image_context_rope_mode,
                rope_max_freq=cfg.image_context_rope_max_freq,
            )
            for _ in range(cfg.image_context_depth)
        ])
        self.context_distill_head = (
            nn.Conv3d(cfg.dim, cfg.context_distill_dim, 1)
            if cfg.context_distill_dim > 0 else None)
        self.latent_distance_head = (
            nn.Sequential(
                nn.Linear(cfg.latent_channels + 3, cfg.latent_distance_hidden),
                nn.GELU(),
                nn.Linear(cfg.latent_distance_hidden, 1),
            )
            if cfg.latent_distance_hidden > 0 else None)
        self.prompt_pe_dim = point_pe_dim(cfg.prompt_point_pe, cfg.prompt_point_pe_num_bands)
        self.prompt_feature_dim = (
            cfg.dim if cfg.prompt_sampled_feature != "none" else 0
        )
        self.prompt_mlp = nn.Sequential(
            nn.Linear(self.prompt_pe_dim + self.prompt_feature_dim + 1, cfg.dim),
            nn.SiLU(),
            nn.Linear(cfg.dim, cfg.dim),
        )
        self.latent_prompt_mlp = None
        if cfg.latent_prompt_enabled:
            # Spatially pooled sheet latent -> one prompt token. Fresh keys
            # live under this prefix so warm starts use
            # training.resume_strict=false with
            # resume_partial_prefixes=("latent_prompt_mlp",).
            self.latent_prompt_mlp = nn.Sequential(
                nn.Linear(int(cfg.latent_prompt_channels), cfg.dim),
                nn.SiLU(),
                nn.Linear(cfg.dim, cfg.dim),
            )
        if cfg.prompt_modulator_merge == "concat":
            self.cond_proj = nn.Linear(cfg.dim * 2, cfg.dim)
        elif cfg.prompt_modulator_merge == "addition":
            self.cond_proj = nn.Linear(cfg.dim, cfg.dim)
        else:
            raise ValueError(
                "prompt_modulator.merge must be addition or concat, "
                f"got {cfg.prompt_modulator_merge}"
            )
        if cfg.prompt_modulator_type == "two_way_attn":
            # SAM-style bidirectional prompt<->grid refinement. Different module
            # than the prefix_attn AttentionBlock, so a checkpoint trained with
            # prefix_attn loads everything EXCEPT these (they start fresh).
            self.modulator_blocks = nn.ModuleList([
                TwoWayAttentionBlock(cfg.dim, cfg.heads, cfg.mlp_ratio,
                                     rope_mode=cfg.rope_mode, rope_max_freq=cfg.rope_max_freq)
                for _ in range(cfg.prompt_modulator_depth)
            ])
        else:
            self.modulator_blocks = nn.ModuleList([
                AttentionBlock(cfg.dim, cfg.heads, cfg.mlp_ratio, use_rope=cfg.use_rope,
                               rope_mode=cfg.rope_mode, rope_max_freq=cfg.rope_max_freq)
                for _ in range(cfg.prompt_modulator_depth)
            ])
        self.blocks = nn.ModuleList([
            AttentionBlock(cfg.dim, cfg.heads, cfg.mlp_ratio, use_rope=cfg.use_rope,
                           rope_mode=cfg.rope_mode, rope_max_freq=cfg.rope_max_freq)
            for _ in range(cfg.depth)
        ])
        self.norm = nn.LayerNorm(cfg.dim)
        self.out = nn.Linear(cfg.dim, cfg.latent_channels)
        self.aux_out = nn.ModuleList([nn.Linear(cfg.dim, cfg.latent_channels) for _ in range(cfg.depth)])
        if cfg.prompt_modulator_type == "loop":
            # ONE head, shared across iterations, matching the tied blocks: every
            # iteration decodes the same thing (the running latent estimate), just
            # from more points. A head per iteration would also cap the point
            # count at build time, which defeats variable-K training.
            self.loop_aux_out = nn.Linear(cfg.dim, cfg.latent_channels)

    def forward(
        self,
        image: torch.Tensor,
        prompt_points: torch.Tensor,
        prompt_labels: torch.Tensor,
        image_index: torch.Tensor | None = None,
        latent_prompt: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor | list[torch.Tensor]]:
        feat, image_tokens, image_coords, image_context = self.encode_image_context_from_image(image)
        return self.forward_from_image_context(
            image_tokens,
            image_coords,
            image_context,
            prompt_points,
            prompt_labels,
            image_shape=tuple(int(value) for value in image.shape[-3:]),
            image_index=image_index,
            image_latent=feat,
            latent_prompt=latent_prompt,
        )

    def query_latent_distance(
        self,
        z: torch.Tensor,
        points_zyx: torch.Tensor,
        image_shape: tuple[int, int, int],
    ) -> torch.Tensor:
        """Predict unsigned normalized distance at points from a RAW latent.

        Same interface and feature construction as the AE's query head
        (trilinear latent sample + unit coords -> MLP), but this head is
        trained against GT EDT, not distilled through the AE.
        """
        from vesuvius_p2sd.models.ae.sheet_ae import (
            image_points_to_unit_coords,
            sample_latent_at_points,
        )

        if self.latent_distance_head is None:
            raise RuntimeError("model built without latent_distance head")
        latent_values = sample_latent_at_points(z, points_zyx, image_shape)
        coord_features = image_points_to_unit_coords(points_zyx, image_shape)
        features = torch.cat([latent_values, coord_features], dim=-1)
        return self.latent_distance_head(features).squeeze(-1)

    def encode_image_context_from_image(
        self,
        image: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode one or more images for reuse across multiple prompt batches."""
        if self.freeze_image_encoder:
            # A frozen encoder should not retain Conv3d activations for
            # backward. Keep it in eval mode in case a future config enables
            # encoder dropout; GroupNorm itself has no running state.
            self.image_encoder.eval()
            with torch.no_grad():
                feat = self.image_encoder.encode(image)
        else:
            feat = self.image_encoder.encode(image)
        image_tokens, image_coords, image_context = self.encode_image_context(feat, image.dtype)
        return feat, image_tokens, image_coords, image_context

    def forward_from_image_context(
        self,
        image_tokens: torch.Tensor,
        image_coords: torch.Tensor,
        image_context: torch.Tensor,
        prompt_points: torch.Tensor,
        prompt_labels: torch.Tensor,
        *,
        image_shape: tuple[int, int, int],
        image_index: torch.Tensor | None = None,
        image_latent: torch.Tensor | None = None,
        latent_prompt: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor | list[torch.Tensor]]:
        """Predict prompt-conditioned latents from a cached image context.

        ``image_tokens``, ``image_coords``, and ``image_context`` must come
        from :meth:`encode_image_context_from_image`. A caller may use one
        image row with many prompt rows by supplying a repeated ``image_index``.
        This preserves the normal forward path while avoiding repeated Conv3d
        image encoding during fixed multi-prompt evaluation.
        """
        if image_index is None:
            image_index = torch.arange(prompt_points.shape[0], device=image_context.device)
        image_index = image_index.to(device=image_context.device, dtype=torch.long)
        grid = image_tokens.index_select(0, image_index)
        grid_coords = image_coords.index_select(0, image_index)
        pair_context = image_context.index_select(0, image_index)
        m, _, dz, dy, dx = pair_context.shape
        prompt_tokens, prompt_coords = self.compose_prompts(
            pair_context,
            prompt_points,
            prompt_labels,
            image_shape=image_shape,
        )
        if latent_prompt is not None:
            if self.latent_prompt_mlp is None:
                raise RuntimeError(
                    "latent_prompt tensor passed but p2sd.latent_prompt is disabled")
            # Spatial mean-pool a latent grid; accept an already-pooled [M, C]
            # too (cluster centroids arrive flattened at inference). Appended
            # AFTER fusion so both single_token and none fusion see exactly
            # one extra token, anchored at the volume centre.
            pooled = (latent_prompt.float().mean(dim=(-3, -2, -1))
                      if latent_prompt.ndim == 5 else latent_prompt.float())
            latent_token = self.latent_prompt_mlp(
                pooled.to(prompt_tokens.dtype)).unsqueeze(1)
            prompt_tokens = torch.cat([prompt_tokens, latent_token], dim=1)
            prompt_coords = torch.cat(
                [prompt_coords, prompt_coords.new_full((m, 1, 3), 0.5)], dim=1)
        loop_states: list[torch.Tensor] | None = (
            [] if self.cfg.prompt_modulator_type == "loop" else None
        )
        grid = self.modulate_grid_tokens(
            grid, grid_coords, prompt_tokens, prompt_coords, trace=loop_states)
        aux_latents = []
        if loop_states:
            # Deep supervision on the running estimate after each click: "be right
            # with 1 point, better with 2, ...". Drops the last state, which is
            # what the refiner (or `out`, if refiner.depth is 0) already consumes,
            # so it is not supervised twice. The count varies with the point count,
            # so this needs loss.per_round_mse_weights_mode: uniform.
            aux_latents.extend(
                tokens_to_latent(self.loop_aux_out(state), (dz, dy, dx))
                for state in loop_states[:-1]
            )
        for block, aux in zip(self.blocks, self.aux_out):
            grid = block(grid, grid_coords)
            aux_latents.append(tokens_to_latent(aux(grid), (dz, dy, dx)))
        grid = self.norm(grid)
        latent = tokens_to_latent(self.out(grid), (dz, dy, dx))
        output: dict[str, torch.Tensor | list[torch.Tensor]] = {
            "latent": latent,
            "aux_latents": aux_latents,
        }
        if image_latent is not None:
            output["image_latent"] = image_latent
        # Pre-modulation context grid (b, dim, dz, dy, dx): the tensor the
        # context-distill head reads and the modulator attends over.
        output["image_context"] = image_context
        return output

    def encode_image_context(
        self,
        feat: torch.Tensor,
        coord_dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        b, _, dz, dy, dx = feat.shape
        tokens = self.grid_proj(feat).flatten(2).transpose(1, 2)
        coords = grid_coordinates(
            b,
            (dz, dy, dx),
            device=feat.device,
            dtype=coord_dtype,
        )
        for block in self.image_context_blocks:
            tokens = block(tokens, coords)
        context = tokens_to_latent(tokens, (dz, dy, dx))
        return tokens, coords, context

    def modulate_grid_tokens(
        self,
        grid: torch.Tensor,
        grid_coords: torch.Tensor,
        prompt_tokens: torch.Tensor,
        prompt_coords: torch.Tensor,
        trace: list[torch.Tensor] | None = None,
    ) -> torch.Tensor:
        if self.cfg.prompt_modulator_type == "loop":
            # One point at a time through the SAME broadcast + N blocks, so the
            # grid carries a running estimate that iteration k has conditioned on
            # points 0..k. Weight-tied: N blocks reused K times, giving depth NxK
            # for N blocks' worth of parameters. Cost scales with the POINT COUNT,
            # not with N -- K=8 at N=2 is 16 grid passes against the 5 (1
            # modulator + 4 refiner) a prefix_attn model spends.
            #
            # This is deliberately NOT permutation-invariant: the other modulators
            # see an unordered set, this one sees a click sequence. Shuffle the
            # points during training or it will overfit the sampler's ordering.
            grid_tokens = grid
            for index in range(prompt_tokens.shape[1]):
                point = prompt_tokens[:, index:index + 1]
                if self.cfg.prompt_modulator_merge == "concat":
                    grid_tokens = self.cond_proj(
                        torch.cat([grid_tokens, point.expand_as(grid_tokens)], dim=-1))
                else:
                    grid_tokens = grid_tokens + self.cond_proj(point)
                for block in self.modulator_blocks:
                    grid_tokens = block(grid_tokens, grid_coords)
                if trace is not None:
                    trace.append(grid_tokens)
            return grid_tokens
        if self.cfg.prompt_modulator_type == "broadcast_mlp":
            prompt_context = prompt_tokens.mean(dim=1, keepdim=True)
            if self.cfg.prompt_modulator_merge == "concat":
                return self.cond_proj(torch.cat([grid, prompt_context.expand_as(grid)], dim=-1))
            return grid + self.cond_proj(prompt_context)
        if self.cfg.prompt_modulator_type == "prefix_attn":
            # prompt_coords are center-frame ((p+0.5)/size); grid_coords are the
            # linspace(0,1,N) frame the grid tokens carry. Concatenating them raw
            # (as before) misaligns the shared RoPE by up to half a token (16
            # voxels at the edges) -- dormant under broadcast_mlp, which discards
            # prompt_coords, but a real defect here where prefix attention relies
            # on prompt/grid position matching. Map the prompt into the grid
            # frame first; grid_coords itself is untouched.
            prompt_coords = align_prompt_coords_to_grid(prompt_coords, grid_coords)
            tokens = torch.cat([prompt_tokens, grid], dim=1)
            coords = torch.cat([prompt_coords, grid_coords], dim=1)
            prefix = prompt_tokens.shape[1]
            for block in self.modulator_blocks:
                tokens = block(tokens, coords)
            return tokens[:, prefix:]
        if self.cfg.prompt_modulator_type == "two_way_attn":
            # SAM-style: prompt and grid mutually refine via bidirectional
            # cross-attention; return the refined grid tokens.
            prompt_coords = align_prompt_coords_to_grid(prompt_coords, grid_coords)
            prompt = prompt_tokens
            grid_tokens = grid
            for block in self.modulator_blocks:
                prompt, grid_tokens = block(prompt, grid_tokens, prompt_coords, grid_coords)
            return grid_tokens
        raise ValueError(f"Unsupported prompt_modulator.type: {self.cfg.prompt_modulator_type}")

    def compose_prompts(
        self,
        pair_feat: torch.Tensor,
        points_zyx: torch.Tensor,
        labels: torch.Tensor,
        *,
        image_shape: tuple[int, int, int],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if points_zyx.ndim != 3:
            raise ValueError(f"prompt_points must have shape [M, P, 3], got {points_zyx.shape}")
        coords = normalize_points(points_zyx, image_shape)
        pe = point_position_encoding(
            coords,
            mode=self.cfg.prompt_point_pe,
            num_bands=self.cfg.prompt_point_pe_num_bands,
        )
        parts = [pe]
        if self.cfg.prompt_sampled_feature != "none":
            parts.append(sample_trilinear_features(pair_feat, points_zyx, image_shape))
        label_value = labels.to(points_zyx.dtype).unsqueeze(-1)
        parts.append(label_value)
        prompt_tokens = self.prompt_mlp(torch.cat(parts, dim=-1))
        if self.cfg.prompt_fusion in {"single_token", "mean"}:
            prompt_tokens = prompt_tokens.mean(dim=1, keepdim=True)
            coords = coords.mean(dim=1, keepdim=True)
        elif self.cfg.prompt_fusion not in {"none", "keep"}:
            raise ValueError(f"Unsupported prompt_composition.fusion: {self.cfg.prompt_fusion}")
        return prompt_tokens, coords

    def encode_prompts(
        self,
        points_zyx: torch.Tensor,
        labels: torch.Tensor,
        *,
        image_shape: tuple[int, int, int],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Backward-compatible helper used by older tests; no sampled image feature.
        coords = normalize_points(points_zyx, image_shape)
        pe = point_position_encoding(
            coords,
            mode=self.cfg.prompt_point_pe,
            num_bands=self.cfg.prompt_point_pe_num_bands,
        )
        label_value = labels.to(points_zyx.dtype).unsqueeze(-1)
        pad = pe.new_zeros(*pe.shape[:-1], self.prompt_feature_dim)
        tokens = self.prompt_mlp(torch.cat([pe, pad, label_value], dim=-1))
        prompt_context = tokens.mean(dim=1, keepdim=True)
        if self.cfg.prompt_fusion in {"single_token", "mean"}:
            tokens = prompt_context
            coords = coords.mean(dim=1, keepdim=True)
        return tokens, coords


def _build_patch_stem(
    kind: str,
    in_channels: int,
    ch: tuple[int, ...],
    start_stage: int,
    patch_size: int,
    num_groups: int,
) -> nn.Module:
    """Build the patchify stem that maps the input to ch[start_stage] at
    1/patch_size resolution. All variants share the same output shape so the
    encoder body is unchanged; they differ only in how the tokenization is
    formed."""
    out_ch = ch[start_stage]
    if kind == "plain":
        # ViT-style non-overlapping patch embed (kernel == stride).
        return nn.Conv3d(in_channels, out_ch, patch_size, stride=patch_size)
    if kind == "overlap":
        # Overlapping patch embed: kernel twice the stride so neighboring
        # patches share context, reducing patch-boundary blocking and
        # thin-sheet aliasing for dense prediction (SegFormer / PVTv2).
        return nn.Conv3d(
            in_channels, out_ch, patch_size * 2, stride=patch_size, padding=patch_size // 2
        )
    if kind == "early_conv":
        # Thin stack of stride-2 convs with one nonlinearity: overlapping 3x3
        # receptive fields recover early spatial detail the plain embed
        # discards, at a fraction of the conv stem's cost because the interior
        # width stays at ch[0] and it never touches full resolution (Xiao
        # et al., "Early Convolutions Help Transformers See Better", 2021).
        layers: list[nn.Module] = []
        prev = in_channels
        for step in range(start_stage):
            last = step == start_stage - 1
            cur = out_ch if last else ch[0]
            layers.append(nn.Conv3d(prev, cur, 3, stride=2, padding=1))
            if not last:
                layers.append(norm_layer(cur, num_groups))
                layers.append(nn.SiLU())
            prev = cur
        return nn.Sequential(*layers)
    raise ValueError(f"encoder_patch_stem must be plain, overlap, or early_conv, got {kind!r}")


class P2SDImageEncoder(nn.Module):
    """Encoder-only 5-down image backbone for prompt-conditioned P2SD."""

    def __init__(
        self,
        *,
        in_channels: int,
        channels: tuple[int, ...],
        num_groups: int,
        dropout: float,
        stem: str = "conv",
        patch_size: int = 4,
        patch_stem: str = "plain",
    ) -> None:
        super().__init__()
        if len(channels) != 6:
            raise ValueError("P2SDImageEncoder expects 6 channel entries for 5 downs")
        ch = tuple(int(c) for c in channels)
        self.downsample_factor = 32
        self.out_channels = ch[-1]
        start_stage = patchify_start_stage(stem, patch_size, len(ch))
        if start_stage == 0:
            if patch_stem != "plain":
                raise ValueError("encoder_patch_stem only applies to the patchify stem")
            self.stem = nn.Conv3d(int(in_channels), ch[0], 3, padding=1)
        else:
            # The patch stem replaces the full-resolution shallow stages — by
            # far the dominant Conv3d cost at PS320 — and downsamples by
            # patch_size before the encoder body enters at start_stage.
            self.stem = _build_patch_stem(
                patch_stem, int(in_channels), ch, start_stage, patch_size, int(num_groups)
            )
        enc = []
        for i in range(start_stage, 5):
            enc.extend([
                ResBlock3d(ch[i], num_groups=int(num_groups), dropout=float(dropout)),
                nn.Conv3d(ch[i], ch[i + 1], 3, stride=2, padding=1),
            ])
        enc.append(ResBlock3d(ch[-1], num_groups=int(num_groups), dropout=float(dropout)))
        self.encoder = nn.Sequential(*enc)
        self.freeze_through_stage_index = -1

    def freeze_through_stage(self, stage_index: int) -> None:
        """Freeze the stem and encoder modules through one sequential stage."""

        stage_index = int(stage_index)
        if stage_index < 0 or stage_index >= len(self.encoder):
            raise ValueError(
                "P2SDImageEncoder freeze stage must be in "
                f"[0, {len(self.encoder) - 1}], got {stage_index}"
            )
        self.freeze_through_stage_index = stage_index
        for parameter in self.stem.parameters():
            parameter.requires_grad_(False)
        for index in range(stage_index + 1):
            for parameter in self.encoder[index].parameters():
                parameter.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.freeze_through_stage_index < 0:
            return self.encoder(self.stem(x))

        # Frozen shallow stages must not retain high-resolution activations.
        # Eval mode makes a future nonzero encoder dropout deterministic;
        # GroupNorm itself has no running state.
        self.stem.eval()
        with torch.no_grad():
            h = self.stem(x)
            for index in range(self.freeze_through_stage_index + 1):
                stage = self.encoder[index]
                stage.eval()
                h = stage(h)
        for index in range(self.freeze_through_stage_index + 1, len(self.encoder)):
            h = self.encoder[index](h)
        return h

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)

    def encode_with_pyramid(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[int, torch.Tensor]]:
        """Final 1/32 feature plus the deepest feature at every downsample
        factor on the way (keyed by factor: with the patchify stem
        {4: ResBlock output at 1/4, 8: ..., 16: ..., 32: final}). Same
        computation as ``encode``; freeze bookkeeping is the caller's job
        (run it under no_grad for a frozen encoder). Refinement-head input
        (0061, 2026-08-27)."""
        full = int(x.shape[-1])
        taps: dict[int, torch.Tensor] = {}
        h = self.stem(x)
        taps[full // int(h.shape[-1])] = h
        for layer in self.encoder:
            h = layer(h)
            taps[full // int(h.shape[-1])] = h
        return h, taps


def normalize_points(
    points_zyx: torch.Tensor,
    image_shape: tuple[int, int, int],
) -> torch.Tensor:
    denom = torch.tensor(
        [max(image_shape[0], 1), max(image_shape[1], 1), max(image_shape[2], 1)],
        device=points_zyx.device,
        dtype=points_zyx.dtype,
    )
    return ((points_zyx + 0.5) / denom).clamp(0.0, 1.0)


def point_pe_dim(mode: str, num_bands: int) -> int:
    if mode in {"none", "raw"}:
        return 3
    if mode == "fourier":
        return 3 + 3 * 2 * int(num_bands)
    raise ValueError(f"Unsupported point PE mode: {mode}")


def point_position_encoding(
    coords: torch.Tensor,
    *,
    mode: str,
    num_bands: int,
) -> torch.Tensor:
    if mode in {"none", "raw"}:
        return coords
    if mode != "fourier":
        raise ValueError(f"Unsupported point PE mode: {mode}")
    bands = torch.arange(num_bands, device=coords.device, dtype=coords.dtype)
    freq = (2.0 ** bands) * math.pi
    phase = coords[..., None] * freq
    sincos = torch.cat([phase.sin(), phase.cos()], dim=-1).flatten(-2)
    return torch.cat([coords, sincos], dim=-1)


def sample_trilinear_features(
    feat: torch.Tensor,
    points_zyx: torch.Tensor,
    image_shape: tuple[int, int, int],
) -> torch.Tensor:
    coords = normalize_points(points_zyx, image_shape).clamp(0, 1)
    grid_xyz = torch.stack([
        coords[..., 2] * 2 - 1,
        coords[..., 1] * 2 - 1,
        coords[..., 0] * 2 - 1,
    ], dim=-1)
    grid = grid_xyz.view(points_zyx.shape[0], points_zyx.shape[1], 1, 1, 3)
    sampled = F.grid_sample(
        feat.float(),
        grid.float(),
        mode="bilinear",
        padding_mode="border",
        align_corners=False,
    )
    return sampled[..., 0, 0].transpose(1, 2).to(dtype=feat.dtype)


class AttentionBlock(nn.Module):
    def __init__(self, dim: int, heads: int, mlp_ratio: float, *, use_rope: bool,
                 rope_mode: str = "axis_mixed", rope_max_freq: float | None = None) -> None:
        super().__init__()
        if dim % heads != 0:
            raise ValueError("dim must be divisible by heads")
        self.dim = dim
        self.heads = heads
        self.head_dim = dim // heads
        self.use_rope = use_rope
        self.rope_mode = rope_mode
        self.rope_max_freq = rope_max_freq
        self.norm1 = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio * 2 / 3)
        self.fc1 = nn.Linear(dim, hidden * 2)
        self.fc2 = nn.Linear(hidden, dim)

    def forward(self, x: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        b, n, _ = x.shape
        h = self.norm1(x)
        qkv = self.qkv(h).view(b, n, 3, self.heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        if self.use_rope:
            q, k = apply_simple_3d_rope(
                q, k, coords, mode=self.rope_mode, max_freq=self.rope_max_freq
            )
        attn = F.scaled_dot_product_attention(q, k, v)
        attn = attn.transpose(1, 2).reshape(b, n, self.dim)
        x = x + self.proj(attn)
        gate, value = self.fc1(self.norm2(x)).chunk(2, dim=-1)
        x = x + self.fc2(F.silu(gate) * value)
        return x


def _axis_mixed_rope_angles(coords: torch.Tensor, n_pairs: int) -> torch.Tensor:
    """Legacy angle construction: one scalar phase for all frequency pairs.

    Retained as the default so existing runs stay reproducible, but it is
    defective as a 3D encoding: every frequency is applied to the single scalar
    ``z + 1.7y + 2.3x``, collapsing position onto one diagonal. Because 1.7 and
    2.3 are small-denominator rationals the collisions are exact -- on a 10^3
    token grid there are only 363 distinct codes for 1000 tokens, 93% of tokens
    share a code with another, and positionally identical pairs sit a median of
    233 voxels apart. Prefer ``rope_mode: per_axis``.
    """

    idx = torch.arange(0, n_pairs, device=coords.device, dtype=torch.float32)
    inv_freq = 1.0 / (10000 ** (idx / max(1, n_pairs)))
    phase_base = coords[..., 0] + 1.7 * coords[..., 1] + 2.3 * coords[..., 2]
    return phase_base[..., None].float() * inv_freq * (2 * math.pi)


def _axis_token_count(axis_coords: torch.Tensor) -> int:
    """Recover N (tokens along this axis) from a normalized linspace grid.

    ``grid_coordinates`` lays each axis out as ``linspace(0, 1, N)``, so the
    number of distinct values is N. Reading it back here lets ``max_freq`` track
    the grid automatically instead of being pinned to a hand-set constant that
    silently mis-scales the moment the token resolution changes.
    """

    return int(torch.unique(axis_coords).numel())


def _per_axis_rope_angles(
    coords: torch.Tensor, n_pairs: int, max_freq: float | None
) -> torch.Tensor:
    """Give each spatial axis its own contiguous block of frequency pairs.

    Two axes can no longer alias onto each other, because a pair responds to
    exactly one coordinate. The ladder is geometric from 1 cycle across the
    volume up to ``max_freq``; with coords normalized to [0, 1] and N tokens per
    axis, ``max_freq = N/2`` puts the highest frequency at Nyquist for adjacent
    tokens.

    ``max_freq=None`` (the default) derives N/2 per axis from the grid, so the
    ladder is always correctly scaled at any token resolution. Passing a float
    overrides it for every axis.

    This replaces the standard 10000^(-i/d) ladder, which is built for integer
    token indices and leaves most dims positionally dead over a [0, 1] range.
    """

    counts = [n_pairs // 3] * 3
    for i in range(n_pairs - sum(counts)):
        counts[i] += 1
    blocks = []
    for axis, count in enumerate(counts):
        if count == 0:
            continue
        if max_freq is None:
            axis_top = max(_axis_token_count(coords[..., axis]) / 2.0, 1.0)
        else:
            axis_top = float(max_freq)
        idx = torch.arange(count, device=coords.device, dtype=torch.float32)
        exponent = idx / max(1, count - 1) if count > 1 else idx
        freqs = axis_top ** exponent
        blocks.append(coords[..., axis : axis + 1].float() * freqs * (2 * math.pi))
    return torch.cat(blocks, dim=-1)


def apply_simple_3d_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    coords: torch.Tensor,
    *,
    mode: str = "axis_mixed",
    max_freq: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply 3D RoPE using normalized z/y/x coordinates.

    ``max_freq=None`` (per_axis default) derives the per-axis Nyquist frequency
    from the token grid, so the ladder self-scales with resolution; pass a float
    only to override.

    Angles are always built in float32: under autocast the previous
    ``dtype=q.dtype`` quantized the frequency ladder to bfloat16's 8-bit
    mantissa, which silently coupled the positional basis to the AMP dtype.
    """

    head_dim = q.shape[-1]
    rope_dim = (head_dim // 2) * 2
    if rope_dim < 2:
        return q, k
    n_pairs = rope_dim // 2
    if mode == "axis_mixed":
        angles = _axis_mixed_rope_angles(coords, n_pairs)
    elif mode == "per_axis":
        angles = _per_axis_rope_angles(coords, n_pairs, max_freq)
    else:
        raise ValueError(f"rope_mode must be axis_mixed or per_axis, got {mode!r}")
    cos = angles.cos()[:, None].to(q.dtype)
    sin = angles.sin()[:, None].to(q.dtype)

    def rotate(x: torch.Tensor) -> torch.Tensor:
        main = x[..., :rope_dim]
        tail = x[..., rope_dim:]
        x1 = main[..., 0::2]
        x2 = main[..., 1::2]
        rot = torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1).flatten(-2)
        return torch.cat([rot, tail], dim=-1) if tail.numel() else rot

    return rotate(q), rotate(k)


def _rope_cos_sin(coords, n_pairs, mode, max_freq, dtype):
    """cos/sin for RoPE at `coords`, broadcastable over [B, heads, N, pairs]."""
    if mode == "axis_mixed":
        angles = _axis_mixed_rope_angles(coords, n_pairs)
    elif mode == "per_axis":
        angles = _per_axis_rope_angles(coords, n_pairs, max_freq)
    else:
        raise ValueError(f"rope_mode must be axis_mixed or per_axis, got {mode!r}")
    return angles.cos()[:, None].to(dtype), angles.sin()[:, None].to(dtype)


def _rope_rotate(x, cos, sin, rope_dim):
    main = x[..., :rope_dim]
    tail = x[..., rope_dim:]
    x1 = main[..., 0::2]
    x2 = main[..., 1::2]
    rot = torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1).flatten(-2)
    return torch.cat([rot, tail], dim=-1) if tail.numel() else rot


class CrossAttention(nn.Module):
    """Multi-head attention where q-tokens attend to kv-tokens, each carrying its
    own coordinates for RoPE. Used by the two-way (SAM-style) modulator; q and kv
    coords must already be in the same frame (align_prompt_coords_to_grid)."""

    def __init__(self, dim: int, heads: int, *, rope_mode: str, rope_max_freq: float | None) -> None:
        super().__init__()
        if dim % heads != 0:
            raise ValueError("dim must be divisible by heads")
        self.heads = heads
        self.head_dim = dim // heads
        self.dim = dim
        self.rope_mode = rope_mode
        self.rope_max_freq = rope_max_freq
        self.to_q = nn.Linear(dim, dim)
        self.to_k = nn.Linear(dim, dim)
        self.to_v = nn.Linear(dim, dim)
        self.proj = nn.Linear(dim, dim)

    def forward(self, q_tokens, kv_tokens, q_coords, kv_coords):
        b, nq, _ = q_tokens.shape
        nk = kv_tokens.shape[1]
        q = self.to_q(q_tokens).view(b, nq, self.heads, self.head_dim).transpose(1, 2)
        k = self.to_k(kv_tokens).view(b, nk, self.heads, self.head_dim).transpose(1, 2)
        v = self.to_v(kv_tokens).view(b, nk, self.heads, self.head_dim).transpose(1, 2)
        rope_dim = (self.head_dim // 2) * 2
        if rope_dim >= 2:
            n_pairs = rope_dim // 2
            cq, sq = _rope_cos_sin(q_coords, n_pairs, self.rope_mode, self.rope_max_freq, q.dtype)
            ck, sk = _rope_cos_sin(kv_coords, n_pairs, self.rope_mode, self.rope_max_freq, k.dtype)
            q = _rope_rotate(q, cq, sq, rope_dim)
            k = _rope_rotate(k, ck, sk, rope_dim)
        attn = F.scaled_dot_product_attention(q, k, v)
        attn = attn.transpose(1, 2).reshape(b, nq, self.dim)
        return self.proj(attn)


class TwoWayAttentionBlock(nn.Module):
    """SAM-style two-way transformer block over (prompt tokens, grid tokens):
    (1) prompt self-attention, (2) prompt->grid cross-attention, (3) prompt MLP,
    (4) grid->prompt cross-attention. Returns the updated (prompt, grid)."""

    def __init__(self, dim: int, heads: int, mlp_ratio: float, *,
                 rope_mode: str, rope_max_freq: float | None) -> None:
        super().__init__()
        rope = dict(rope_mode=rope_mode, rope_max_freq=rope_max_freq)
        self.norm1 = nn.LayerNorm(dim)
        self.self_attn = CrossAttention(dim, heads, **rope)
        self.norm2 = nn.LayerNorm(dim)
        self.prompt_to_grid = CrossAttention(dim, heads, **rope)
        self.norm3 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio * 2 / 3)
        self.fc1 = nn.Linear(dim, hidden * 2)
        self.fc2 = nn.Linear(hidden, dim)
        self.norm4 = nn.LayerNorm(dim)
        self.grid_to_prompt = CrossAttention(dim, heads, **rope)

    def forward(self, prompt, grid, prompt_coords, grid_coords):
        p = self.norm1(prompt)
        prompt = prompt + self.self_attn(p, p, prompt_coords, prompt_coords)
        p = self.norm2(prompt)
        prompt = prompt + self.prompt_to_grid(p, grid, prompt_coords, grid_coords)
        gate, value = self.fc1(self.norm3(prompt)).chunk(2, dim=-1)
        prompt = prompt + self.fc2(F.silu(gate) * value)
        g = self.norm4(grid)
        grid = grid + self.grid_to_prompt(g, prompt, grid_coords, prompt_coords)
        return prompt, grid


def align_prompt_coords_to_grid(
    prompt_coords: torch.Tensor,
    grid_coords: torch.Tensor,
) -> torch.Tensor:
    """Map center-frame prompt coords into the grid's linspace(0,1,N) frame.

    A grid token j sits at ``j/(N-1)``; a prompt at that token's physical centre
    has center-frac ``(j+0.5)/N``. Solving for the linspace value gives
    ``l = (c*N - 0.5)/(N-1)`` per axis. N is read from the grid itself (distinct
    coordinate values along each axis), so this is correct for any grid shape and
    leaves ``grid_coords`` unchanged.
    """

    out = torch.empty_like(prompt_coords)
    for axis in range(prompt_coords.shape[-1]):
        n = int(torch.unique(grid_coords[..., axis]).numel())
        if n > 1:
            out[..., axis] = ((prompt_coords[..., axis] * n - 0.5) / (n - 1)).clamp(0.0, 1.0)
        else:
            out[..., axis] = prompt_coords[..., axis]
    return out


def grid_coordinates(
    batch_size: int,
    shape: tuple[int, int, int],
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    dz, dy, dx = shape
    z = torch.linspace(0, 1, dz, device=device, dtype=dtype)
    y = torch.linspace(0, 1, dy, device=device, dtype=dtype)
    x = torch.linspace(0, 1, dx, device=device, dtype=dtype)
    zz, yy, xx = torch.meshgrid(z, y, x, indexing="ij")
    coords = torch.stack([zz, yy, xx], dim=-1).view(1, dz * dy * dx, 3)
    return coords.expand(batch_size, -1, -1)


def tokens_to_latent(tokens: torch.Tensor, shape: tuple[int, int, int]) -> torch.Tensor:
    b, _, c = tokens.shape
    dz, dy, dx = shape
    return tokens.transpose(1, 2).reshape(b, c, dz, dy, dx)


def build_p2sd(cfg: dict[str, Any], *, latent_channels: int) -> FullAttentionP2SD:
    p2sd = cfg.get("p2sd", cfg)
    reject_legacy_p2sd_keys(p2sd)
    prompt_comp = p2sd.get("prompt_composition", {})
    prompt_mod = p2sd.get("prompt_modulator", {})
    image_context = p2sd.get("image_context", {})
    refiner = p2sd.get("refiner", {})
    if str(image_context.get("type", "self_attn")) != "self_attn":
        raise ValueError(
            "p2sd.image_context.type must be self_attn, "
            f"got {image_context.get('type')}"
        )
    prompt_comp_type = str(prompt_comp.get("type", "point_feature_mlp"))
    sampled_feature = str(prompt_comp.get("sampled_feature", "image_tokens_trilinear"))
    if prompt_comp_type in {"point_tokens", "coord_label_mlp"} and "sampled_feature" not in prompt_comp:
        sampled_feature = "none"
    prompt_modulator_type = str(prompt_mod.get("type", "broadcast_mlp"))
    prompt_modulator_merge = resolve_prompt_modulator_merge(prompt_mod, prompt_modulator_type)
    model_cfg = P2SDConfig(
        image_channels=int(p2sd.get("image_channels", 1)),
        latent_channels=int(latent_channels),
        encoder_channels=tuple(int(v) for v in p2sd.get("encoder_channels", (8, 16, 24, 32, 48, 64))),
        dim=int(refiner.get("dim", p2sd.get("dim", 256))),
        depth=int(refiner.get("depth", p2sd.get("depth", 4))),
        heads=int(refiner.get("heads", p2sd.get("heads", 8))),
        mlp_ratio=float(refiner.get("mlp_ratio", p2sd.get("mlp_ratio", 4.0))),
        use_rope=str(refiner.get("pos_encoding", "rope")) == "rope",
        rope_mode=str(refiner.get("rope_mode", "axis_mixed")),
        rope_max_freq=(
            None if refiner.get("rope_max_freq") in (None, "auto")
            else float(refiner.get("rope_max_freq"))
        ),
        image_context_depth=int(image_context.get("depth", 4)),
        image_context_heads=int(image_context.get("heads", refiner.get("heads", p2sd.get("heads", 8)))),
        image_context_mlp_ratio=float(
            image_context.get("mlp_ratio", refiner.get("mlp_ratio", p2sd.get("mlp_ratio", 4.0)))
        ),
        image_context_use_rope=str(
            image_context.get("pos_encoding", refiner.get("pos_encoding", "rope"))
        ) == "rope",
        image_context_rope_mode=str(image_context.get("rope_mode", "axis_mixed")),
        image_context_rope_max_freq=(
            None if image_context.get("rope_max_freq") in (None, "auto")
            else float(image_context.get("rope_max_freq"))
        ),
        context_distill_dim=int((p2sd.get("context_distill", {}) or {}).get("dim", 0)),
        latent_distance_hidden=int((p2sd.get("latent_distance", {}) or {}).get("hidden_dim", 0)),
        encoder_num_groups=int(p2sd.get("encoder_num_groups", 4)),
        encoder_dropout=float(p2sd.get("encoder_dropout", 0.0)),
        encoder_stem=str(p2sd.get("encoder_stem", "conv")),
        encoder_patch_size=int(p2sd.get("encoder_patch_size", 4)),
        encoder_patch_stem=str(p2sd.get("encoder_patch_stem", "plain")),
        freeze_image_encoder=bool(p2sd.get("freeze_image_encoder", False)),
        freeze_image_encoder_through_stage=int(
            p2sd.get("freeze_image_encoder_through_stage", -1)
        ),
        prompt_composition_type=prompt_comp_type,
        prompt_point_pe=str(prompt_comp.get("point_pe", "fourier")),
        prompt_point_pe_num_bands=int(prompt_comp.get("point_pe_num_bands", 8)),
        prompt_sampled_feature=sampled_feature,
        prompt_fusion=str(prompt_comp.get("fusion", "single_token")),
        prompt_modulator_type=prompt_modulator_type,
        prompt_modulator_merge=prompt_modulator_merge,
        prompt_modulator_depth=int(prompt_mod.get("depth", 1)),
        latent_prompt_enabled=(
            str((p2sd.get("latent_prompt", {}) or {}).get("mode", "none")) != "none"
        ),
        latent_prompt_channels=int(latent_channels),
    )
    return FullAttentionP2SD(model_cfg)


def reject_legacy_p2sd_keys(p2sd: dict[str, Any]) -> None:
    legacy = [key for key in ("cond_mode", "prompt") if key in p2sd]
    if not legacy:
        return
    raise ValueError(
        "Legacy P2SD config key(s) are no longer supported: "
        f"{', '.join('p2sd.' + key for key in legacy)}. "
        "Use p2sd.prompt_composition, p2sd.prompt_modulator, and p2sd.refiner instead."
    )


def resolve_prompt_modulator_merge(prompt_mod: dict[str, Any], modulator_type: str) -> str:
    if modulator_type in {"broadcast_mlp", "loop"}:
        # `loop` broadcasts one point per iteration through the same cond_proj,
        # so merge is live for it too -- not the dead inherited key it is below.
        merge = str(prompt_mod.get("merge", "concat"))
        if merge not in {"concat", "addition"}:
            raise ValueError(
                "p2sd.prompt_modulator.merge must be concat or addition for "
                f"{modulator_type}, got {merge}"
            )
        return merge
    # merge only configures broadcast_mlp's cond_proj, which is dead for other
    # types. It is ignored here rather than rejected because it commonly arrives
    # via _base_ inheritance (e.g. a prefix_attn config extending a broadcast_mlp
    # base); erroring on an inherited no-op key would be a footgun.
    return "concat"
