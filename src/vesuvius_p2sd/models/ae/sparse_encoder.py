"""torchsparse encoder for native sheet AEs.

The encoder mirrors the working sparse-encoder AE pattern from label_diffusion:
occupied voxels are encoded with torchsparse stride-1 convolutions, downsampling
uses differentiable voxel mean-pooling, and the coarse bottleneck is densified
for the existing dense decoder and task heads.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn

try:  # pragma: no cover - exercised in environments with torchsparse installed
    from torchsparse import SparseTensor
    from torchsparse import nn as spnn
except Exception:  # pragma: no cover - import error is raised at model construction
    SparseTensor = None
    spnn = None

from vesuvius_p2sd.models.ae.sheet_ae import NoSkipSheetAE, SheetAEConfig


class SparseNoSkipSheetAE(NoSkipSheetAE):
    """Native AE with a torchsparse occupied-voxel encoder and dense decoder."""

    def __init__(self, cfg: SheetAEConfig) -> None:
        if spnn is None or SparseTensor is None:
            raise ImportError(
                "torchsparse is required for model.encoder=torchsparse_aniso_unet"
            )
        super().__init__(cfg)
        self.stem = nn.Identity()
        self.encoder = SparseNoSkipEncoder(
            in_channels=int(cfg.in_channels),
            channels=tuple(int(c) for c in cfg.channels),
            num_groups=int(cfg.num_groups),
            dropout=float(cfg.dropout),
            norm=str(cfg.sparse_enc_norm),
            stage_res_blocks=tuple(int(v) for v in cfg.sparse_stage_res_blocks),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        if x.device.type != "cuda":
            raise RuntimeError("torchsparse encoder is CUDA-only; move inputs/model to cuda")
        h = self.encoder(x)
        return self.to_latent(h)


class SparseNoSkipEncoder(nn.Module):
    """Five-down torchsparse encoder that returns a dense bottleneck grid."""

    def __init__(
        self,
        *,
        in_channels: int,
        channels: tuple[int, ...],
        num_groups: int,
        dropout: float,
        norm: str,
        stage_res_blocks: Sequence[int],
    ) -> None:
        super().__init__()
        if len(channels) != 6:
            raise ValueError("SparseNoSkipEncoder expects 6 channel entries for 5 downs")
        if len(stage_res_blocks) != 6:
            raise ValueError("sparse_stage_res_blocks must contain 6 entries")
        self.downsample_factor = 32
        self.stride = (32, 32, 32)
        stages = []
        prev = int(in_channels)
        for stage_index, ch in enumerate(channels):
            mods: list[nn.Module] = []
            if stage_index > 0:
                mods.append(
                    SparseDown(
                        prev,
                        int(ch),
                        factor=(2, 2, 2),
                        kernel_size=(3, 3, 3),
                        num_groups=num_groups,
                        norm=norm,
                    )
                )
                prev = int(ch)
            for _ in range(int(stage_res_blocks[stage_index])):
                mods.append(
                    SparseResBlock(
                        prev,
                        int(ch),
                        kernel_size=(3, 3, 3),
                        num_groups=num_groups,
                        dropout=dropout,
                        norm=norm,
                    )
                )
                prev = int(ch)
            stages.append(nn.Sequential(*mods))
        self.stages = nn.ModuleList(stages)
        self.out_channels = int(channels[-1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, _, depth, height, width = x.shape
        if depth % 32 or height % 32 or width % 32:
            raise ValueError(
                f"patch {(depth, height, width)} must be divisible by sparse encoder stride 32"
            )
        st = dense_to_sparse_occupied(x)
        for stage in self.stages:
            st = stage(st)
        st.spatial_range = (batch, depth // 32, height // 32, width // 32)
        dense = st.dense().permute(0, 4, 1, 2, 3)
        if x.is_contiguous(memory_format=torch.channels_last_3d):
            return dense.contiguous(memory_format=torch.channels_last_3d)
        return dense.contiguous()


class SparseRMSNorm(nn.Module):
    """Per-point RMSNorm over sparse feature channels."""

    def __init__(self, channels: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(int(channels)))
        self.eps = float(eps)

    def forward(self, st: SparseTensor) -> SparseTensor:
        feats = st.feats
        feats32 = feats.float()
        feats32 = feats32 * torch.rsqrt(feats32.pow(2).mean(dim=1, keepdim=True) + self.eps)
        out = SparseTensor(
            feats=feats32.to(feats.dtype) * self.weight,
            coords=st.coords,
            stride=st.stride,
            spatial_range=st.spatial_range,
        )
        out._caches = st._caches
        return out


def sparse_norm(kind: str, channels: int, num_groups: int) -> nn.Module:
    kind = str(kind or "rms")
    if kind == "none":
        return nn.Identity()
    if kind == "rms":
        return SparseRMSNorm(channels) if int(channels) > 1 else nn.Identity()
    if kind == "batch":
        return spnn.BatchNorm(channels)
    if kind == "group":
        return spnn.GroupNorm(min(int(num_groups), int(channels)), int(channels))
    raise ValueError(f"sparse norm must be rms|batch|group|none, got {kind!r}")


class SparseResBlock(nn.Module):
    """Pre-activation sparse residual block with stride-1 convolutions."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        kernel_size: tuple[int, int, int],
        num_groups: int,
        dropout: float,
        norm: str,
    ) -> None:
        super().__init__()
        self.norm1 = sparse_norm(norm, in_channels, num_groups)
        self.act = spnn.SiLU(inplace=False)
        self.conv1 = spnn.Conv3d(in_channels, out_channels, kernel_size=list(kernel_size))
        self.norm2 = sparse_norm(norm, out_channels, num_groups)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None
        self.conv2 = spnn.Conv3d(out_channels, out_channels, kernel_size=list(kernel_size))
        self.skip = (
            spnn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False)
            if int(in_channels) != int(out_channels)
            else None
        )

    def forward(self, st: SparseTensor) -> SparseTensor:
        identity = st if self.skip is None else self.skip(st)
        h = self.conv1(self.act(self.norm1(st)))
        h = self.act(self.norm2(h))
        if self.dropout is not None:
            h.feats = self.dropout(h.feats)
        h = self.conv2(h)
        h.feats = h.feats + identity.feats
        return h


class SparseDown(nn.Module):
    """Stride-1 sparse conv followed by voxel mean-pooling downsample."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        factor: tuple[int, int, int],
        kernel_size: tuple[int, int, int],
        num_groups: int,
        norm: str,
    ) -> None:
        super().__init__()
        self.factor = tuple(int(v) for v in factor)
        self.norm = sparse_norm(norm, in_channels, num_groups)
        self.act = spnn.SiLU(inplace=False)
        self.conv = spnn.Conv3d(in_channels, out_channels, kernel_size=list(kernel_size), stride=1)

    def forward(self, st: SparseTensor) -> SparseTensor:
        h = self.conv(self.act(self.norm(st)))
        return sparse_voxel_pool(h, self.factor)


def sparse_voxel_pool(st: SparseTensor, factor: tuple[int, int, int]) -> SparseTensor:
    factor_t = torch.tensor(factor, device=st.coords.device)
    coords = st.coords.clone()
    coords[:, 1:] = torch.div(coords[:, 1:], factor_t, rounding_mode="floor")
    unique_coords, inverse = torch.unique(coords, dim=0, return_inverse=True)
    out = torch.zeros(
        unique_coords.shape[0],
        st.feats.shape[1],
        device=st.feats.device,
        dtype=st.feats.dtype,
    )
    counts = torch.zeros(unique_coords.shape[0], 1, device=st.feats.device, dtype=st.feats.dtype)
    out.index_add_(0, inverse, st.feats)
    counts.index_add_(0, inverse, torch.ones_like(st.feats[:, :1]))
    out = out / counts.clamp_min(1.0)
    stride = t3(st.stride)
    new_stride = tuple(int(s) * int(f) for s, f in zip(stride, factor))
    spatial_range = st.spatial_range
    new_range = None
    if spatial_range is not None:
        new_range = (spatial_range[0],) + tuple(
            (int(spatial_range[1 + i]) + int(factor[i]) - 1) // int(factor[i])
            for i in range(3)
        )
    return SparseTensor(
        feats=out,
        coords=unique_coords.int(),
        stride=new_stride,
        spatial_range=new_range,
    )


def dense_to_sparse_occupied(x: torch.Tensor) -> SparseTensor:
    batch, _, depth, height, width = x.shape
    coords_per_batch = []
    for b in range(batch):
        coords_zyx = x[b, 0].nonzero(as_tuple=False)
        if coords_zyx.numel() == 0:
            coords_zyx = torch.tensor(
                [[depth // 2, height // 2, width // 2]],
                device=x.device,
                dtype=torch.long,
            )
        batch_col = torch.full((coords_zyx.shape[0], 1), b, device=x.device, dtype=torch.int32)
        coords_per_batch.append(torch.cat([batch_col, coords_zyx.to(torch.int32)], dim=1))
    coords = torch.cat(coords_per_batch, dim=0)
    feats = torch.ones(coords.shape[0], 1, device=x.device, dtype=x.dtype)
    return SparseTensor(
        feats=feats,
        coords=coords,
        stride=1,
        spatial_range=(batch, depth, height, width),
    )


def t3(value) -> tuple[int, int, int]:
    if isinstance(value, int):
        return (value, value, value)
    if isinstance(value, Sequence):
        if len(value) == 3:
            return tuple(int(v) for v in value)
        if len(value) == 1:
            return (int(value[0]), int(value[0]), int(value[0]))
    return (int(value), int(value), int(value))
