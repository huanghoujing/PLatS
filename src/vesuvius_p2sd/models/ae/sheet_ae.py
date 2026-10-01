"""Native sheet autoencoder with dense or torchsparse encoders."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class SheetAEConfig:
    in_channels: int = 1
    out_channels: int = 1
    channels: tuple[int, ...] = (8, 16, 24, 32, 48, 64)
    latent_channels: int | None = None
    latent_projection: str = "learned"
    latent_noise_std: float = 0.0
    latent_noise_mode: str = "mul_uniform"
    num_groups: int = 4
    dropout: float = 0.0
    distance_head_enabled: bool = False
    query_head_enabled: bool = False
    query_hidden_dim: int = 128
    deep_supervision_enabled: bool = False
    deep_supervision_stages: tuple[int, ...] = (0, 1, 2, 3)
    dense_encoder_layout: str = "legacy"
    dense_enc_norm: str = "group"
    dense_enc_norm_grad_clamp: float | None = None
    dense_encoder_stem: str = "conv"
    dense_encoder_patch_stem: str = "plain"
    dense_encoder_patch_size: int = 4
    decoder_type: str = "dense"
    point_decoder_hidden_dim: int = 256
    point_decoder_depth: int = 3
    point_decoder_num_bands: int = 8
    point_decoder_coords: str = "global"
    point_decoder_head: str = "two_channel"
    point_decoder_neighborhood: str = "none"
    point_decoder_detach_occupancy: bool = False
    point_decoder_eval_chunk: int = 524_288
    dense_stage_res_blocks: tuple[int, ...] = (1, 1, 1, 1, 1, 1)
    dense_downsample_mode: str = "stride_conv"
    sparse_enc_norm: str = "rms"
    sparse_stage_res_blocks: tuple[int, ...] = (1, 1, 1, 1, 1, 1)


class _ClampBackwardGrad(torch.autograd.Function):
    """Identity in forward; replace NaN/Inf and bound magnitude in backward.

    RMSNorm backward multiplies gradients by up to 1/sqrt(eps) wherever the
    activation RMS is eps-dominated. The bias-free stage-matched encoder keeps
    empty mask regions at exactly zero activation, so that amplification
    compounds across the norm stack and overflows BF16 gradients. Clamping at
    every norm input caps the compounding; the affected voxels have zero
    activations, so their conv weight-gradient contributions are zero either way.
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor, limit: float) -> torch.Tensor:
        ctx.limit = limit
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad: torch.Tensor):
        limit = ctx.limit
        grad = torch.nan_to_num(grad, nan=0.0, posinf=limit, neginf=-limit)
        return grad.clamp(-limit, limit), None


class RMSNorm3d(nn.Module):
    def __init__(self, channels: int, eps: float = 1e-6, grad_clamp: float | None = None) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1, channels, 1, 1, 1))
        self.eps = eps
        self.grad_clamp = None if grad_clamp is None else float(grad_clamp)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.grad_clamp is not None and torch.is_grad_enabled() and x.requires_grad:
            x = _ClampBackwardGrad.apply(x, self.grad_clamp)
        x32 = x.float()
        rms = x32.pow(2).mean(dim=1, keepdim=True).add(self.eps).sqrt()
        return (x32 / rms).to(x.dtype) * self.weight


class ResBlock3d(nn.Module):
    def __init__(self, channels: int, *, num_groups: int, dropout: float) -> None:
        super().__init__()
        self.norm1 = norm_layer(channels, num_groups)
        self.conv1 = nn.Conv3d(channels, channels, 3, padding=1)
        self.norm2 = norm_layer(channels, num_groups)
        self.dropout = nn.Dropout3d(dropout) if dropout > 0 else nn.Identity()
        self.conv2 = nn.Conv3d(channels, channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = self.conv2(self.dropout(F.silu(self.norm2(h))))
        return x + h


def patchify_start_stage(stem: str, patch_size: int, num_stages: int) -> int:
    """Map a stem choice to the stage index the encoder body starts at.

    A patchify stem with stride ``patch_size`` replaces ``log2(patch_size)``
    downsampling stages, so the body enters at that stage's channel width and
    the total downsample factor is unchanged.
    """
    stem = str(stem or "conv")
    if stem == "conv":
        return 0
    if stem != "patchify":
        raise ValueError(f"encoder stem must be conv or patchify, got {stem!r}")
    patch_size = int(patch_size)
    start_stage = patch_size.bit_length() - 1
    if patch_size <= 1 or patch_size != 2 ** start_stage:
        raise ValueError(f"patch_size must be a power of two >= 2, got {patch_size}")
    if start_stage >= num_stages - 1:
        raise ValueError(
            f"patch_size {patch_size} consumes all downsampling stages")
    return start_stage


def dense_encoder_norm(
    kind: str,
    channels: int,
    num_groups: int,
    grad_clamp: float | None = None,
) -> nn.Module:
    kind = str(kind or "group")
    if kind == "none":
        return nn.Identity()
    if kind == "rms":
        return RMSNorm3d(channels, eps=1e-5, grad_clamp=grad_clamp) if int(channels) > 1 else nn.Identity()
    if kind == "batch":
        return nn.BatchNorm3d(int(channels))
    if kind == "group":
        return norm_layer(channels, num_groups)
    raise ValueError(f"dense encoder norm must be rms|batch|group|none, got {kind!r}")


class DenseStageResBlock3d(nn.Module):
    """Pre-activation dense residual block shaped like the sparse encoder block."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        num_groups: int,
        dropout: float,
        norm: str,
        norm_grad_clamp: float | None = None,
    ) -> None:
        super().__init__()
        self.norm1 = dense_encoder_norm(norm, in_channels, num_groups, grad_clamp=norm_grad_clamp)
        self.conv1 = nn.Conv3d(in_channels, out_channels, 3, padding=1, bias=False)
        self.norm2 = dense_encoder_norm(norm, out_channels, num_groups, grad_clamp=norm_grad_clamp)
        self.dropout = nn.Dropout3d(dropout) if dropout > 0 else nn.Identity()
        self.conv2 = nn.Conv3d(out_channels, out_channels, 3, padding=1, bias=False)
        self.skip = (
            nn.Conv3d(in_channels, out_channels, 1, bias=False)
            if int(in_channels) != int(out_channels)
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.skip(x)
        h = self.conv1(F.silu(self.norm1(x)))
        h = self.conv2(self.dropout(F.silu(self.norm2(h))))
        return h + identity


class DenseDown3d(nn.Module):
    """Dense downsample block matched to sparse conv + voxel mean-pool."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        num_groups: int,
        norm: str,
        mode: str,
        norm_grad_clamp: float | None = None,
    ) -> None:
        super().__init__()
        self.mode = str(mode or "conv_mean_pool")
        if self.mode not in {"conv_mean_pool", "stride_conv"}:
            raise ValueError("dense downsample mode must be conv_mean_pool or stride_conv")
        stride = 2 if self.mode == "stride_conv" else 1
        self.norm = dense_encoder_norm(norm, in_channels, num_groups, grad_clamp=norm_grad_clamp)
        self.conv = nn.Conv3d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv(F.silu(self.norm(x)))
        if self.mode == "conv_mean_pool":
            return F.avg_pool3d(h, kernel_size=2, stride=2)
        return h


def _build_ae_patch_stem(
    kind: str,
    *,
    in_channels: int,
    channels: tuple[int, ...],
    start_stage: int,
    patch_size: int,
    num_groups: int,
    norm: str,
    norm_grad_clamp: float | None,
) -> list[nn.Module]:
    """Build the AE patchify stem. Mirrors the P2SD's `_build_patch_stem`.

    Both variants emit ``channels[start_stage]`` at 1/patch_size, so the encoder
    body and the latent shape are unchanged and the two are directly A/B-able.

    ``early_conv`` won the P2SD stem ablation decisively (beat plain on 8/9
    scene metrics; reached plain's ep200 quality by ~ep120), but the AE -- which
    *defines* the latent the P2SD must regress -- was hardcoded to ``plain``. A
    stride-4 non-overlapping embed is the wrong first operation for a ~3-voxel
    sheet: adjacent patches share no input voxels.

    ``plain`` remains the default so the champion AE stays reproducible.
    """

    out_ch = int(channels[start_stage])
    if kind == "plain":
        return [nn.Conv3d(in_channels, out_ch, patch_size, stride=patch_size)]
    if kind == "early_conv":
        layers: list[nn.Module] = []
        prev = int(in_channels)
        for step in range(start_stage):
            last = step == start_stage - 1
            cur = out_ch if last else int(channels[0])
            # bias=True for the same reason as the plain embed: it keeps empty
            # mask regions off the exact-zero activation path that produced the
            # historical RMSNorm backward blowup.
            layers.append(nn.Conv3d(prev, cur, 3, stride=2, padding=1, bias=True))
            if not last:
                layers.append(dense_encoder_norm(norm, cur, num_groups, norm_grad_clamp))
                layers.append(nn.SiLU())
            prev = cur
        return layers
    raise ValueError(f"dense_encoder_patch_stem must be plain or early_conv, got {kind!r}")


class DenseStageMatchedEncoder(nn.Module):
    """Dense encoder with the same stage schedule as `SparseNoSkipEncoder`."""

    def __init__(
        self,
        *,
        in_channels: int,
        channels: tuple[int, ...],
        num_groups: int,
        dropout: float,
        norm: str,
        stage_res_blocks: tuple[int, ...],
        downsample_mode: str,
        norm_grad_clamp: float | None = None,
        stem: str = "conv",
        patch_size: int = 4,
        patch_stem: str = "plain",
    ) -> None:
        super().__init__()
        if len(channels) != 6:
            raise ValueError("DenseStageMatchedEncoder expects 6 channel entries for 5 downs")
        if len(stage_res_blocks) != 6:
            raise ValueError("dense_stage_res_blocks must contain 6 entries")
        start_stage = patchify_start_stage(stem, patch_size, len(channels))
        mods = []
        prev = int(in_channels)
        if start_stage > 0:
            # Patch embed replaces the full-resolution shallow stages (the
            # dominant conv cost). bias=True keeps empty mask regions at a
            # nonzero constant, removing the exact-zero activation pathway
            # behind the RMSNorm backward blowup at its source.
            mods.extend(_build_ae_patch_stem(
                patch_stem,
                in_channels=prev,
                channels=channels,
                start_stage=start_stage,
                patch_size=patch_size,
                num_groups=num_groups,
                norm=norm,
                norm_grad_clamp=norm_grad_clamp,
            ))
            prev = int(channels[start_stage])
        for stage_index in range(start_stage, len(channels)):
            ch = int(channels[stage_index])
            if stage_index > start_stage:
                mods.append(
                    DenseDown3d(
                        prev,
                        ch,
                        num_groups=num_groups,
                        norm=norm,
                        mode=downsample_mode,
                        norm_grad_clamp=norm_grad_clamp,
                    )
                )
                prev = ch
            for _ in range(int(stage_res_blocks[stage_index])):
                mods.append(
                    DenseStageResBlock3d(
                        prev,
                        ch,
                        num_groups=num_groups,
                        dropout=dropout,
                        norm=norm,
                        norm_grad_clamp=norm_grad_clamp,
                    )
                )
                prev = ch
        self.net = nn.Sequential(*mods)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        depth, height, width = x.shape[-3:]
        if depth % 32 or height % 32 or width % 32:
            raise ValueError(
                f"patch {(depth, height, width)} must be divisible by matched encoder stride 32"
            )
        return self.net(x)


def fourier_point_features(unit_coords: torch.Tensor, num_bands: int) -> torch.Tensor:
    """Fourier features of unit-cube coordinates: [coords, sin/cos octaves]."""
    if num_bands <= 0:
        return unit_coords
    frequencies = 2.0 ** torch.arange(
        num_bands, device=unit_coords.device, dtype=unit_coords.dtype)
    angles = unit_coords[..., None] * frequencies * torch.pi
    encoded = torch.cat([angles.sin(), angles.cos()], dim=-1)
    return torch.cat([unit_coords, encoded.flatten(-2)], dim=-1)


class PointDecoder(nn.Module):
    """Implicit decoder: occupancy logit + unsigned distance at query points.

    Query features are the trilinearly sampled latent plus Fourier-encoded
    unit coordinates — the same idiom as the proven coordinate query head,
    widened and given two output channels. Replaces the dense deconvolution
    decoder: training supervision moves to sampled points, and dense volumes
    are only materialized at evaluation via chunked queries.
    """

    def __init__(
        self,
        latent_channels: int,
        *,
        hidden_dim: int = 256,
        depth: int = 3,
        num_bands: int = 8,
        coords: str = "global",
        cell_voxels: int = 32,
        head: str = "two_channel",
        neighborhood: str = "none",
        detach_occupancy_from_sdf: bool = False,
    ) -> None:
        super().__init__()
        if coords not in {"global", "local"}:
            raise ValueError(f"point decoder coords must be global or local, got {coords!r}")
        if head not in {"two_channel", "signed_distance"}:
            raise ValueError(
                f"point decoder head must be two_channel or signed_distance, got {head!r}")
        if neighborhood not in {"none", "face6"}:
            raise ValueError(
                f"point decoder neighborhood must be none or face6, got {neighborhood!r}")
        self.coords = coords
        self.head = head
        self.neighborhood = neighborhood
        self.detach_occupancy_from_sdf = bool(detach_occupancy_from_sdf)
        self.cell_voxels = int(cell_voxels)
        self.num_bands = int(num_bands)
        if neighborhood == "face6":
            # Half-cell taps along each axis give the head a finite-difference
            # view of the latent around the query — a local feature gradient
            # to triangulate the surface inside the cell, which a single
            # trilinear sample cannot provide.
            half = self.cell_voxels / 2.0
            offsets = [[0.0, 0.0, 0.0]]
            for axis in range(3):
                for sign in (-1.0, 1.0):
                    offset = [0.0, 0.0, 0.0]
                    offset[axis] = sign * half
                    offsets.append(offset)
            self.register_buffer("tap_offsets", torch.tensor(offsets), persistent=False)
        else:
            self.tap_offsets = None
        taps = 1 if self.tap_offsets is None else int(self.tap_offsets.shape[0])
        in_dim = taps * int(latent_channels) + 3 * (1 + 2 * self.num_bands)
        layers: list[nn.Module] = [nn.Linear(in_dim, hidden_dim), nn.SiLU()]
        for _ in range(max(int(depth) - 1, 0)):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.SiLU()])
        self.mlp = nn.Sequential(*layers)
        if head == "signed_distance":
            # One geometric output: normalized signed distance to the sheet
            # boundary (negative inside). Occupancy is derived as
            # logit = -scale * sdf, so boundary sharpness inherits the
            # distance channel's accuracy instead of relying on an
            # independent, blurry occupancy channel.
            self.out = nn.Linear(hidden_dim, 1)
            self.logit_scale = nn.Parameter(torch.tensor(8.0))
        else:
            self.out = nn.Linear(hidden_dim, 2)

    def forward(
        self,
        z: torch.Tensor,
        points_zyx: torch.Tensor,
        image_shape: tuple[int, int, int],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.tap_offsets is None:
            latent_values = sample_latent_at_points(z, points_zyx, image_shape)
        else:
            taps = int(self.tap_offsets.shape[0])
            batch, count = points_zyx.shape[0], points_zyx.shape[1]
            tapped = points_zyx[:, :, None, :] + self.tap_offsets.to(points_zyx.dtype)
            latent_values = sample_latent_at_points(
                z, tapped.reshape(batch, count * taps, 3), image_shape,
            ).reshape(batch, count, -1)
        if self.coords == "local":
            # Offset within the latent cell only: translation-equivariant and
            # deliberately unable to express absolute-position shortcuts —
            # global Fourier coordinates let the head fit supervision points
            # without the latent and ring between sparse far-field samples.
            scaled = (points_zyx.float() + 0.5) / float(self.cell_voxels)
            coords = scaled - scaled.floor()
        else:
            coords = image_points_to_unit_coords(points_zyx, image_shape)
        features = torch.cat(
            [latent_values, fourier_point_features(coords.float(), self.num_bands)],
            dim=-1,
        )
        prediction = self.out(self.mlp(features))
        if self.head == "signed_distance":
            sdf = prediction[..., 0]
            # Optionally keep BCE a pure calibration of the scale: geometry is
            # then shaped only by the signed regression, and the occupancy
            # objective cannot inflate |sdf| toward saturated logits.
            logit_source = sdf.detach() if self.detach_occupancy_from_sdf else sdf
            return -self.logit_scale * logit_source, sdf
        return prediction[..., 0], prediction[..., 1]


class NoSkipSheetAE(nn.Module):
    """Compact 3D sheet AE with five downs and no encoder-decoder skips."""

    def __init__(self, cfg: SheetAEConfig) -> None:
        super().__init__()
        if len(cfg.channels) != 6:
            raise ValueError("NoSkipSheetAE expects 6 channel entries for 5 downs")
        self.cfg = cfg
        self.downsample_factor = 32
        ch = tuple(int(c) for c in cfg.channels)
        if str(cfg.dense_encoder_layout) == "stage_matched":
            self.stem = nn.Identity()
            self.encoder = DenseStageMatchedEncoder(
                in_channels=int(cfg.in_channels),
                channels=ch,
                num_groups=int(cfg.num_groups),
                dropout=float(cfg.dropout),
                norm=str(cfg.dense_enc_norm),
                stage_res_blocks=tuple(int(v) for v in cfg.dense_stage_res_blocks),
                downsample_mode=str(cfg.dense_downsample_mode),
                norm_grad_clamp=cfg.dense_enc_norm_grad_clamp,
                stem=str(cfg.dense_encoder_stem),
                patch_size=int(cfg.dense_encoder_patch_size),
                patch_stem=str(cfg.dense_encoder_patch_stem),
            )
        elif str(cfg.dense_encoder_layout) == "legacy":
            self.stem = nn.Conv3d(cfg.in_channels, ch[0], 3, padding=1)
            enc = []
            for i in range(5):
                enc.extend([
                    ResBlock3d(ch[i], num_groups=cfg.num_groups, dropout=cfg.dropout),
                    nn.Conv3d(ch[i], ch[i + 1], 3, stride=2, padding=1),
                ])
            enc.append(ResBlock3d(ch[-1], num_groups=cfg.num_groups, dropout=cfg.dropout))
            self.encoder = nn.Sequential(*enc)
        else:
            raise ValueError("dense_encoder_layout must be legacy or stage_matched")
        latent_channels = int(cfg.latent_channels or ch[-1])
        self._latent_channels = latent_channels
        latent_projection = str(cfg.latent_projection or "learned")
        if latent_projection == "learned":
            self.to_latent = nn.Conv3d(ch[-1], latent_channels, 1)
            self.from_latent = nn.Conv3d(latent_channels, ch[-1], 1)
        elif latent_projection == "identity":
            if latent_channels != ch[-1]:
                raise ValueError(
                    "identity latent projection requires latent_channels to match "
                    f"encoder channels, got {latent_channels} and {ch[-1]}"
                )
            self.to_latent = nn.Identity()
            self.from_latent = nn.Identity()
        else:
            raise ValueError("latent_projection must be learned or identity")
        self.point_decoder = None
        if str(cfg.decoder_type) == "point":
            # Implicit decoding: no dense decoder, heads, deep supervision, or
            # separate query head — the point decoder subsumes them all.
            self.point_decoder = PointDecoder(
                latent_channels,
                hidden_dim=int(cfg.point_decoder_hidden_dim),
                depth=int(cfg.point_decoder_depth),
                num_bands=int(cfg.point_decoder_num_bands),
                coords=str(cfg.point_decoder_coords),
                cell_voxels=self.downsample_factor,
                head=str(cfg.point_decoder_head),
                neighborhood=str(cfg.point_decoder_neighborhood),
                detach_occupancy_from_sdf=bool(cfg.point_decoder_detach_occupancy),
            )
            self.decoder = None
            self.head = None
            self.distance_head = None
            self.deep_supervision_indices = {}
            self.deep_supervision_heads = nn.ModuleDict()
            self.query_head = None
            return
        if str(cfg.decoder_type) != "dense":
            raise ValueError(f"decoder_type must be dense or point, got {cfg.decoder_type!r}")
        dec = []
        rev = list(reversed(ch))
        for i in range(5):
            dec.extend([
                ResBlock3d(rev[i], num_groups=cfg.num_groups, dropout=cfg.dropout),
                nn.ConvTranspose3d(rev[i], rev[i + 1], 2, stride=2),
            ])
        dec.append(ResBlock3d(ch[0], num_groups=cfg.num_groups, dropout=cfg.dropout))
        self.decoder = nn.Sequential(*dec)
        self.head = nn.Conv3d(ch[0], cfg.out_channels, 1)
        self.distance_head = nn.Conv3d(ch[0], cfg.out_channels, 1) if cfg.distance_head_enabled else None
        aux_stage_channels = {
            0: rev[1],
            1: rev[2],
            2: rev[3],
            3: rev[4],
            4: ch[0],
        }
        aux_stage_indices = {
            0: 2,
            1: 4,
            2: 6,
            3: 8,
            4: 10,
        }
        self.deep_supervision_indices: dict[int, int] = {}
        self.deep_supervision_heads = nn.ModuleDict()
        if cfg.deep_supervision_enabled:
            for stage in cfg.deep_supervision_stages:
                stage = int(stage)
                if stage not in aux_stage_channels:
                    raise ValueError("AE deep supervision stages must be in [0, 4]")
                self.deep_supervision_indices[aux_stage_indices[stage]] = stage
                self.deep_supervision_heads[str(stage)] = nn.Conv3d(
                    aux_stage_channels[stage],
                    cfg.out_channels,
                    1,
                )
        self.query_head = None
        if cfg.query_head_enabled:
            self.query_head = nn.Sequential(
                nn.Linear(latent_channels + 3, int(cfg.query_hidden_dim)),
                nn.SiLU(),
                nn.Linear(int(cfg.query_hidden_dim), int(cfg.query_hidden_dim)),
                nn.SiLU(),
                nn.Linear(int(cfg.query_hidden_dim), 1),
            )

    @property
    def latent_channels(self) -> int:
        return self._latent_channels

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.to_latent(self.encoder(self.stem(x)))

    def decode_features(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.from_latent(z))

    def decode_features_with_aux(self, z: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        h = self.from_latent(z)
        aux_logits = []
        for index, layer in enumerate(self.decoder):
            h = layer(h)
            stage = self.deep_supervision_indices.get(index)
            if stage is not None:
                aux_logits.append(self.deep_supervision_heads[str(stage)](h))
        return h, aux_logits

    def decode_aux_until(self, z: torch.Tensor, stages) -> list[torch.Tensor]:
        """Aux occupancy logits for the requested deep-supervision stages only,
        running the decoder just far enough to reach the last of them (the
        coarse-stage decoded loss for P2SD training, 2026-08-25). Returned in
        ascending stage order."""
        wanted = {int(s) for s in stages}
        missing = wanted - set(self.deep_supervision_indices.values())
        if missing:
            raise ValueError(f"AE has no deep-supervision heads for stages {sorted(missing)}")
        stop = max(i for i, s in self.deep_supervision_indices.items() if s in wanted)
        h = self.from_latent(z)
        out: dict[int, torch.Tensor] = {}
        for index, layer in enumerate(self.decoder):
            h = layer(h)
            stage = self.deep_supervision_indices.get(index)
            if stage is not None and stage in wanted:
                out[stage] = self.deep_supervision_heads[str(stage)](h)
            if index >= stop:
                break
        return [out[s] for s in sorted(out)]

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        if self.point_decoder is not None:
            return self.decode_dense_from_points(z)
        return self.head(self.decode_features(z))

    def decode_with_taps(self, z: torch.Tensor, factors) -> tuple[torch.Tensor, dict[int, torch.Tensor]]:
        """Dense decode returning the logits plus the deepest decoder feature at
        each requested downsample factor relative to the OUTPUT volume (1 = the
        final pre-head features, 2 = 1/2 resolution, 4 = 1/4). Refinement-head
        input (0061, 2026-08-27)."""
        if self.point_decoder is not None:
            raise RuntimeError("decode_with_taps requires the dense decoder")
        wanted = {int(f) for f in factors}
        full = int(z.shape[-1]) * int(self.downsample_factor)
        taps: dict[int, torch.Tensor] = {}
        h = self.from_latent(z)
        for layer in self.decoder:
            h = layer(h)
            factor = full // int(h.shape[-1])
            if factor in wanted:
                taps[factor] = h
        missing = wanted - set(taps)
        if missing:
            raise ValueError(f"decoder has no feature at factors {sorted(missing)}")
        return self.head(h), taps

    # ------------------------------------------------------------------
    # Tile helpers (2026-08-27) for block-sparse REFINEMENT (models/refine_head.py).
    # The AE decoder itself stays dense: its ResBlocks use GroupNorm, whose
    # statistics span the whole volume, so tiling its last stages is not
    # exact (measured: max |diff| 17.6 logits). The refinement head uses
    # per-voxel RMSNorm and a receptive field <= 7 voxels, so it is exact on
    # 32^3 tile centres gathered with an 8-voxel halo from the dense decode.
    # ------------------------------------------------------------------
    SPARSE_TILE = 32
    SPARSE_HALO4 = 2
    SPARSE_FILL = -20.0

    @staticmethod
    def active_tiles(active4: torch.Tensor, tile4: int = 8, dilate: int = 1) -> torch.Tensor:
        """(B,1,Q,Q,Q) bool at 1/4 res -> (B,nT,nT,nT) bool: tiles whose (dilated) region holds any active voxel."""
        m = active4.float()
        if dilate > 0:
            m = F.max_pool3d(m, kernel_size=2 * dilate + 1, stride=1, padding=dilate)
        return (F.max_pool3d(m, kernel_size=tile4, stride=tile4) > 0)[:, 0]

    @staticmethod
    def gather_tiles(x: torch.Tensor, tiles: torch.Tensor, tile: int, halo: int) -> tuple[torch.Tensor, torch.Tensor]:
        """x (B,C,S,S,S) with S = nT*tile -> (N,C,tile+2*halo,..) for the active tiles (zero outside the
        volume) and the tile index (N,4) = [b,i,j,k]."""
        xp = F.pad(x, (halo,) * 6)
        w = tile + 2 * halo
        idx = tiles.nonzero(as_tuple=False)
        if x.requires_grad:
            # Differentiable path (1/4 feature, 1/2 stem): unfold view + one advanced index. The
            # per-tile in-place loop below is NOT usable here — autograd turns it into N CopySlices
            # nodes that each clone the whole (N,C,w^3) stack in backward (3+ s per step).
            u = xp.unfold(2, w, tile).unfold(3, w, tile).unfold(4, w, tile)   # (B,C,nT,nT,nT,w,w,w)
            return u[idx[:, 0], :, idx[:, 1], idx[:, 2], idx[:, 3]], idx
        # No-grad path (full-res image/mask/valid stacks): explicit slicing per tile — the unfold
        # index would materialise the whole (B,C,nT^3,w^3) overlap tensor (11 GB at 320^3 x 13 ch).
        out = x.new_empty((int(idx.shape[0]), int(x.shape[1]), w, w, w))
        for n, (b, i, j, k) in enumerate(idx.tolist()):
            out[n] = xp[b, :, i * tile:i * tile + w, j * tile:j * tile + w, k * tile:k * tile + w]
        return out, idx

    def gather_tile_set(self, tiles: torch.Tensor, full: dict[int, torch.Tensor]) -> tuple[dict[int, torch.Tensor], dict[int, torch.Tensor], torch.Tensor]:
        """Gather every level of a {factor: (B,C,S/f,..)} pyramid into tiles with the matching halo, plus
        in-volume masks per level. Returns (tiles_by_factor, valid_by_factor, index)."""
        out, valid, idx = {}, {}, None
        for f, x in full.items():
            t, idx = self.gather_tiles(x, tiles, self.SPARSE_TILE // f, self.SPARSE_HALO4 * 4 // f)
            ones = torch.ones((x.shape[0], 1) + tuple(x.shape[-3:]), device=x.device, dtype=x.dtype)
            valid[f], _ = self.gather_tiles(ones, tiles, self.SPARSE_TILE // f, self.SPARSE_HALO4 * 4 // f)
            out[f] = t * valid[f]
        return out, valid, idx

    def scatter_tiles(self, tiles_out: torch.Tensor, idx: torch.Tensor, base: torch.Tensor) -> torch.Tensor:
        """Write the central 32^3 of per-tile outputs (N,C,48,48,48) into a copy of `base` (B,C,S,S,S)."""
        tile, halo1 = self.SPARSE_TILE, self.SPARSE_HALO4 * 4
        out = base.clone()
        batch, channels = int(out.shape[0]), int(out.shape[1])
        n_t = [int(v) // tile for v in out.shape[-3:]]
        view = out.reshape(batch, channels, n_t[0], tile, n_t[1], tile, n_t[2], tile).permute(0, 2, 4, 6, 1, 3, 5, 7)
        c = slice(halo1, halo1 + tile)
        view[idx[:, 0], idx[:, 1], idx[:, 2], idx[:, 3]] = tiles_out[..., c, c, c].to(out.dtype)
        return out

    def decode_dense_from_points(self, z: torch.Tensor) -> torch.Tensor:
        """Materialize a dense logits volume by chunked point queries.

        Evaluation-only path for metrics/visualization and for downstream
        consumers of ``decode``; training supervises sampled points directly.
        """
        if self.point_decoder is None:
            raise RuntimeError("decode_dense_from_points requires decoder_type=point")
        batch = z.shape[0]
        image_shape = tuple(int(v) * self.downsample_factor for v in z.shape[-3:])
        depth, height, width = image_shape
        rows_per_chunk = max(
            int(self.cfg.point_decoder_eval_chunk) // max(height * width, 1), 1)
        logits = z.new_empty((batch, 1, depth, height, width), dtype=torch.float32)
        for z_start in range(0, depth, rows_per_chunk):
            z_stop = min(z_start + rows_per_chunk, depth)
            grid_z, grid_y, grid_x = torch.meshgrid(
                torch.arange(z_start, z_stop, device=z.device, dtype=torch.float32),
                torch.arange(height, device=z.device, dtype=torch.float32),
                torch.arange(width, device=z.device, dtype=torch.float32),
                indexing="ij",
            )
            points = torch.stack([grid_z, grid_y, grid_x], dim=-1).reshape(1, -1, 3)
            occupancy, _ = self.point_decoder(z, points.expand(batch, -1, 3), image_shape)
            logits[:, 0, z_start:z_stop] = occupancy.float().reshape(
                batch, z_stop - z_start, height, width)
        return logits

    def forward(
        self,
        x: torch.Tensor,
        *,
        query_points: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        z = self.encode(x)
        z_used = apply_latent_noise(
            z,
            std=float(self.cfg.latent_noise_std),
            mode=str(self.cfg.latent_noise_mode),
            training=self.training,
        )
        if self.point_decoder is not None:
            out = {"latent": z}
            if query_points is not None and query_points.numel():
                occupancy_logits, distance = self.point_decoder(
                    z_used, query_points, x.shape[-3:])
                out["point_logits"] = occupancy_logits
                if self.point_decoder.head == "signed_distance":
                    out["point_signed_distance"] = distance
                    out["query_distance"] = distance.abs()
                else:
                    out["query_distance"] = distance
            return out
        if self.deep_supervision_heads:
            features, aux_logits = self.decode_features_with_aux(z_used)
        else:
            features = self.decode_features(z_used)
            aux_logits = []
        out = {
            "logits": self.head(features),
            "latent": z,
        }
        if aux_logits:
            out["aux_logits"] = aux_logits
        if self.distance_head is not None:
            out["distance"] = self.distance_head(features)
        if self.query_head is not None and query_points is not None:
            out["query_distance"] = self.query_distance(z_used, query_points, x.shape[-3:])
        return out

    def query_distance(
        self,
        z: torch.Tensor,
        points_zyx: torch.Tensor,
        image_shape: tuple[int, int, int],
    ) -> torch.Tensor:
        if points_zyx.numel() == 0:
            return points_zyx.new_zeros(points_zyx.shape[:-1])
        if self.point_decoder is not None:
            distance = self.point_decoder(z, points_zyx, image_shape)[1]
            return distance.abs() if self.point_decoder.head == "signed_distance" else distance
        if self.query_head is None:
            raise RuntimeError("coordinate distance query head is disabled")
        latent_values = sample_latent_at_points(z, points_zyx, image_shape)
        coord_features = image_points_to_unit_coords(points_zyx, image_shape)
        features = torch.cat([latent_values, coord_features], dim=-1)
        return self.query_head(features).squeeze(-1)


def apply_latent_noise(
    z: torch.Tensor,
    *,
    std: float,
    mode: str,
    training: bool,
) -> torch.Tensor:
    """Apply training-only latent noise using the label_diffusion AE semantics."""
    if not (training and std > 0):
        return z
    if mode == "mul_uniform":
        return z * (1.0 + (torch.rand_like(z) * 2 - 1) * float(std))
    if mode == "mul_gaussian":
        return z * (1.0 + torch.randn_like(z) * float(std))
    return z + torch.randn_like(z) * float(std)


def norm_layer(channels: int, num_groups: int) -> nn.Module:
    groups = min(int(num_groups), int(channels))
    while channels % groups != 0 and groups > 1:
        groups -= 1
    if groups > 1:
        return nn.GroupNorm(groups, channels)
    return RMSNorm3d(channels)


def sample_latent_at_points(
    z: torch.Tensor,
    points_zyx: torch.Tensor,
    image_shape: tuple[int, int, int],
) -> torch.Tensor:
    norm_zyx = image_points_to_unit_coords(points_zyx, image_shape).mul(2.0).sub(1.0)
    grid = torch.stack([norm_zyx[..., 2], norm_zyx[..., 1], norm_zyx[..., 0]], dim=-1)
    grid = grid.view(points_zyx.shape[0], points_zyx.shape[1], 1, 1, 3)
    sampled = F.grid_sample(
        z.float(),
        grid.float(),
        mode="bilinear",
        padding_mode="border",
        align_corners=False,
    )
    return sampled[:, :, :, 0, 0].transpose(1, 2)


def image_points_to_unit_coords(
    points_zyx: torch.Tensor,
    image_shape: tuple[int, int, int],
) -> torch.Tensor:
    shape = points_zyx.new_tensor([
        max(int(image_shape[0]), 1),
        max(int(image_shape[1]), 1),
        max(int(image_shape[2]), 1),
    ])
    return ((points_zyx.float() + 0.5) / shape).clamp(0.0, 1.0)


def build_sheet_ae(cfg: dict[str, Any]) -> NoSkipSheetAE:
    model_cfg = cfg.get("model", cfg)
    channels = model_cfg.get("channels") or model_cfg.get("enc_channels")
    if channels is None:
        channels = (8, 16, 24, 32, 48, 64)
    task_heads = model_cfg.get("task_heads", {})
    distance_head = task_heads.get("distance_field_decoder", task_heads.get("distance_decoder", {}))
    query_head = task_heads.get("coordinate_distance_query", {})
    deep_supervision = task_heads.get(
        "occupancy_deep_supervision",
        task_heads.get("deep_supervision", {}),
    )
    deep_supervision_stages = deep_supervision.get("stages", (0, 1, 2, 3))
    encoder = str(model_cfg.get("encoder", model_cfg.get("encoder_type", "dense_aniso_unet")))
    dense_encoder = model_cfg.get("dense_encoder", {})
    decoder_cfg = model_cfg.get("decoder", {}) or {}
    sparse_encoder = model_cfg.get("sparse_encoder", {})
    default_stage_res_blocks = model_cfg.get("encoder_stage_res_blocks", (1, 1, 1, 1, 1, 1))
    dense_stage_res_blocks = dense_encoder.get("stage_res_blocks", default_stage_res_blocks)
    sparse_stage_res_blocks = sparse_encoder.get("stage_res_blocks", default_stage_res_blocks)
    dense_encoder_layout = str(dense_encoder.get("layout", model_cfg.get("dense_encoder_layout", "legacy")))
    if encoder in {"dense_stage_matched_aniso_unet", "matched_dense_aniso_unet"}:
        dense_encoder_layout = "stage_matched"
    dense_downsample_mode = str(
        dense_encoder.get(
            "downsample_mode",
            model_cfg.get(
                "dense_downsample_mode",
                "conv_mean_pool" if dense_encoder_layout == "stage_matched" else "stride_conv",
            ),
        )
    )
    dense_enc_norm = str(dense_encoder.get("norm", model_cfg.get("dense_enc_norm", model_cfg.get("enc_norm", "group"))))
    dense_enc_norm_grad_clamp = dense_encoder.get(
        "norm_grad_clamp", model_cfg.get("dense_enc_norm_grad_clamp")
    )
    sparse_enc_norm = str(
        sparse_encoder.get("norm", model_cfg.get("sparse_enc_norm", model_cfg.get("enc_norm", "rms")))
    )
    ae_cfg = SheetAEConfig(
        in_channels=int(model_cfg.get("in_channels", model_cfg.get("in_ch", 1))),
        out_channels=int(model_cfg.get("out_channels", 1)),
        channels=tuple(int(v) for v in channels),
        latent_channels=(
            None if model_cfg.get("latent_channels") is None
            else int(model_cfg.get("latent_channels"))
        ),
        latent_projection=str(model_cfg.get("latent_projection", "learned")),
        latent_noise_std=float(model_cfg.get("latent_noise_std", 0.0)),
        latent_noise_mode=str(model_cfg.get("latent_noise_mode", "mul_uniform")),
        num_groups=int(model_cfg.get("num_groups", 4)),
        dropout=float(model_cfg.get("dropout", 0.0)),
        distance_head_enabled=bool(distance_head.get("enabled", False)),
        query_head_enabled=bool(query_head.get("enabled", False)),
        query_hidden_dim=int(query_head.get("hidden_dim", 128)),
        deep_supervision_enabled=bool(deep_supervision.get("enabled", False)),
        deep_supervision_stages=tuple(int(v) for v in deep_supervision_stages),
        dense_encoder_layout=dense_encoder_layout,
        dense_enc_norm=dense_enc_norm,
        dense_enc_norm_grad_clamp=(
            None if dense_enc_norm_grad_clamp is None else float(dense_enc_norm_grad_clamp)
        ),
        dense_encoder_stem=str(dense_encoder.get("stem", model_cfg.get("dense_encoder_stem", "conv"))),
        dense_encoder_patch_size=int(
            dense_encoder.get("patch_size", model_cfg.get("dense_encoder_patch_size", 4))
        ),
        dense_encoder_patch_stem=str(
            dense_encoder.get("patch_stem", model_cfg.get("dense_encoder_patch_stem", "plain"))
        ),
        decoder_type=str(decoder_cfg.get("type", model_cfg.get("decoder_type", "dense"))),
        point_decoder_hidden_dim=int(decoder_cfg.get("hidden_dim", 256)),
        point_decoder_depth=int(decoder_cfg.get("depth", 3)),
        point_decoder_num_bands=int(decoder_cfg.get("num_bands", 8)),
        point_decoder_coords=str(decoder_cfg.get("coords", "global")),
        point_decoder_head=str(decoder_cfg.get("head", "two_channel")),
        point_decoder_neighborhood=str(decoder_cfg.get("neighborhood", "none")),
        point_decoder_detach_occupancy=bool(decoder_cfg.get("detach_bce", False)),
        point_decoder_eval_chunk=int(decoder_cfg.get("eval_chunk", 524_288)),
        dense_stage_res_blocks=tuple(int(v) for v in dense_stage_res_blocks),
        dense_downsample_mode=dense_downsample_mode,
        sparse_enc_norm=sparse_enc_norm,
        sparse_stage_res_blocks=tuple(int(v) for v in sparse_stage_res_blocks),
    )
    if encoder in {"sparse_aniso_unet", "torchsparse_aniso_unet", "sparse_flex_aniso"}:
        from vesuvius_p2sd.models.ae.sparse_encoder import SparseNoSkipSheetAE

        return SparseNoSkipSheetAE(ae_cfg)
    return NoSkipSheetAE(ae_cfg)
