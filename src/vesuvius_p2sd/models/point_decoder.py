"""0068 (2026-08-29): sparse two-level POINT decoder for the union (bin-seg) mask on the frozen
P2SD trunk — the user's direction after the 0067 conv head failed ("use some sparse points in
full-res and use global attention for them, achieve efficiency and strong shape capability").

Level A (cells): every 1/4-resolution cell of the encoder pyramid is a token (1/4 tap + upsampled
1/8 and 1/16 taps + 3D sinusoidal position). `depth` blocks of shifted 8^3-window self-attention
(local continuity of the thin sheets) + cross-attention to the 10^3 context tokens (the global
view the seed model needs; 0022's working design has 4 global attention layers there) + MLP.
Heads: cell occupancy logit and a `descriptor_channels` descriptor per cell.

Level B (points): a per-voxel MLP on [trilinearly interpolated descriptor, 3^3 raw-intensity
patch, sub-cell offset (Fourier)] for voxels inside positive cells (PointRend-style). No dense
full-resolution convolutions, no tiles; inference runs the MLP over the ~10-20M voxels of the
positive cells in chunks.

Training samples points per image: GT-positive voxels, hard negatives within `hard_radius` of the
GT, and random negatives (`SparsePointUnionDecoder.forward_train`). Inference:
`PointUnionSeedModel` mirrors the `UnionMaskDecoder` interface (`set_image(image, p2sd)`,
`logits(target_ae)`), so `--union_decoder_path` dispatches to it unchanged.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

SPARSE_FILL = -20.0
CELL = 4  # level-A cell size in voxels (the encoder's 1/4 tap)


@dataclass
class PointDecoderConfig:
    dim: int = 128
    heads: int = 8
    depth: int = 4
    window: int = 8
    context_channels: int = 512
    tap_channels: dict[int, int] = field(default_factory=lambda: {4: 96, 8: 192, 16: 384})
    descriptor_channels: int = 32
    patch: int = 3
    mlp_hidden: int = 128
    pe_freqs: int = 8
    fine_gate: float = 0.2
    fine_dilate: int = 1
    chunk: int = 2_000_000
    stem_channels: int = 0      # 0068b: 1/2-res intensity stem (Conv 5^3 s2 -> SiLU -> Conv 3^3) sampled per point; 0 = off (0068)
    context_refiner_depth: int = 0   # 0068e: trainable global self-attention layers over the 1/32 context tokens (0022's refiner)
    context_tokens: bool = False     # 0068e: add the (refined) context, upsampled to the 1/4 grid, to the level-A tokens
    use_taps: bool = True            # 0068e: False = level-A tokens from the context only (0022's information path)

    @classmethod
    def from_mapping(cls, cfg: dict[str, Any]) -> "PointDecoderConfig":
        known = {f for f in cls.__dataclass_fields__}
        kwargs = {k: v for k, v in (cfg or {}).items() if k in known}
        if "tap_channels" in kwargs:
            kwargs["tap_channels"] = {int(k): int(v) for k, v in kwargs["tap_channels"].items()}
        return cls(**kwargs)


def sinusoidal_pe_3d(shape: tuple[int, int, int], dim: int, device, dtype=torch.float32) -> torch.Tensor:
    """(1, D, H, W, dim) per-axis sinusoidal embedding (dim split over the 3 axes)."""
    per_axis = dim // 3
    freqs = per_axis // 2
    out = []
    for axis, size in enumerate(shape):
        pos = torch.arange(size, device=device, dtype=dtype)
        omega = torch.exp(-math.log(10000.0) * torch.arange(freqs, device=device, dtype=dtype) / max(freqs, 1))
        ang = pos[:, None] * omega[None, :]                                    # (size, freqs)
        emb = torch.cat([ang.sin(), ang.cos()], dim=1)                          # (size, 2*freqs)
        view = [1, 1, 1, 1, emb.shape[1]]
        view[axis + 1] = size
        expand = [1, *shape, emb.shape[1]]
        out.append(emb.view(*view).expand(*expand))
    pe = torch.cat(out, dim=-1)
    if pe.shape[-1] < dim:
        pe = F.pad(pe, (0, dim - pe.shape[-1]))
    return pe


class WindowAttention(nn.Module):
    """Shifted-window self-attention over a (B, D, H, W, C) token grid."""

    def __init__(self, dim: int, heads: int, window: int, shift: int) -> None:
        super().__init__()
        self.dim, self.heads, self.window, self.shift = dim, heads, window, shift
        self.norm = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        self.pos = nn.Parameter(torch.zeros(window ** 3, dim))  # intra-window position embedding
        nn.init.trunc_normal_(self.pos, std=0.02)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, d, h, w, c = x.shape
        ws = self.window
        y = self.norm(x)
        if self.shift:
            y = torch.roll(y, shifts=(-self.shift, -self.shift, -self.shift), dims=(1, 2, 3))
        nd, nh, nw = d // ws, h // ws, w // ws
        y = y.view(b, nd, ws, nh, ws, nw, ws, c).permute(0, 1, 3, 5, 2, 4, 6, 7).reshape(-1, ws ** 3, c)
        y = y + self.pos.to(y.dtype)
        qkv = self.qkv(y).view(y.shape[0], ws ** 3, 3, self.heads, c // self.heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        a = F.scaled_dot_product_attention(q, k, v)                             # (B*nW, heads, ws^3, hd)
        a = a.transpose(1, 2).reshape(-1, ws ** 3, c)
        a = self.proj(a)
        a = a.view(b, nd, nh, nw, ws, ws, ws, c).permute(0, 1, 4, 2, 5, 3, 6, 7).reshape(b, d, h, w, c)
        if self.shift:
            a = torch.roll(a, shifts=(self.shift, self.shift, self.shift), dims=(1, 2, 3))
        return x + a


class CrossAttention(nn.Module):
    """Tokens (B, N, C) attend to context tokens (B, M, C)."""

    def __init__(self, dim: int, heads: int) -> None:
        super().__init__()
        self.heads = heads
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(dim, 2 * dim)
        self.proj = nn.Linear(dim, dim)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor, ctx: torch.Tensor) -> torch.Tensor:
        b, n, c = x.shape
        hd = c // self.heads
        q = self.q(self.norm_q(x)).view(b, n, self.heads, hd).transpose(1, 2)
        kv = self.kv(self.norm_kv(ctx)).view(b, ctx.shape[1], 2, self.heads, hd).permute(2, 0, 3, 1, 4)
        a = F.scaled_dot_product_attention(q, kv[0], kv[1])
        return x + self.proj(a.transpose(1, 2).reshape(b, n, c))


class MLPBlock(nn.Module):
    def __init__(self, dim: int, ratio: float = 4.0) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, int(dim * ratio))
        self.fc2 = nn.Linear(int(dim * ratio), dim)
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.fc2(F.gelu(self.fc1(self.norm(x))))


class SparsePointUnionDecoder(nn.Module):
    def __init__(self, cfg: PointDecoderConfig) -> None:
        super().__init__()
        self.cfg = cfg
        c = cfg.dim
        self.tap_proj = nn.ModuleDict({str(f): nn.Conv3d(ch, c, 1) for f, ch in cfg.tap_channels.items()})
        self.in_norm = nn.LayerNorm(c)
        self.ctx_proj = nn.Linear(cfg.context_channels, c)
        self.blocks = nn.ModuleList()
        for i in range(cfg.depth):
            shift = 0 if i % 2 == 0 else cfg.window // 2
            self.blocks.append(nn.ModuleDict({
                "win": WindowAttention(c, cfg.heads, cfg.window, shift),
                "cross": CrossAttention(c, cfg.heads),
                "mlp": MLPBlock(c),
            }))
        self.ctx_refiner = nn.ModuleList()
        for _ in range(cfg.context_refiner_depth):
            self.ctx_refiner.append(nn.ModuleDict({"attn": CrossAttention(c, cfg.heads), "mlp": MLPBlock(c)}))
        self.ctx_up = nn.Linear(c, c) if cfg.context_tokens else None
        self.out_norm = nn.LayerNorm(c)
        self.cell_head = nn.Linear(c, 1)
        self.desc_head = nn.Linear(c, cfg.descriptor_channels)
        n_patch = cfg.patch ** 3
        n_off = 3 + 3 * 2 * 2                                                     # raw offset + 2 Fourier freqs
        self.stem = None
        if cfg.stem_channels > 0:
            sc = cfg.stem_channels
            self.stem = nn.Sequential(nn.Conv3d(1, sc, 5, stride=2, padding=2), nn.SiLU(), nn.Conv3d(sc, sc, 3, padding=1))
        fine_in = cfg.descriptor_channels + n_patch + n_off + max(cfg.stem_channels, 0)
        self.fine_norm = nn.LayerNorm(fine_in)
        self.fine_mlp = nn.Sequential(nn.Linear(fine_in, cfg.mlp_hidden), nn.SiLU(),
                                  nn.Linear(cfg.mlp_hidden, cfg.mlp_hidden), nn.SiLU(),
                                  nn.Linear(cfg.mlp_hidden, 1))
        self._pe_cache: dict[tuple, torch.Tensor] = {}

    # ---------------------------------------------------------------- level A
    def _pe(self, shape: tuple[int, int, int], device, dtype) -> torch.Tensor:
        key = (shape, str(device))
        if key not in self._pe_cache:
            self._pe_cache[key] = sinusoidal_pe_3d(shape, self.cfg.dim, device)
        return self._pe_cache[key].to(dtype)

    def coarse(self, taps: dict[int, torch.Tensor], context: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """taps: {factor: (B, C_f, D/f, H/f, W/f)}, context (B, C_ctx, D/32, H/32, W/32) ->
        cell logits (B, 1, D/4, H/4, W/4), descriptors (B, C_desc, D/4, H/4, W/4)."""
        base = taps[CELL]
        b, _, d4, h4, w4 = base.shape
        cshape = tuple(int(v) for v in context.shape[-3:])
        ctx = context.flatten(2).transpose(1, 2)                                 # (B, M, C_ctx)
        ctx = self.ctx_proj(ctx)
        ctx = ctx + self._pe(cshape, ctx.device, ctx.dtype).reshape(1, -1, self.cfg.dim)
        for blk in self.ctx_refiner:                                             # 0068e: global self-attention among context tokens
            ctx = blk["attn"](ctx, ctx)
            ctx = blk["mlp"](ctx)
        if self.cfg.use_taps:
            x = self.tap_proj[str(CELL)](base)
            for f, proj in self.tap_proj.items():
                f = int(f)
                if f == CELL:
                    continue
                x = x + F.interpolate(proj(taps[f]), size=(d4, h4, w4), mode="trilinear", align_corners=False)
        else:
            x = torch.zeros(b, self.cfg.dim, d4, h4, w4, device=base.device, dtype=ctx.dtype)
        if self.ctx_up is not None:                                              # 0068e: context broadcast to the 1/4 grid
            grid = self.ctx_up(ctx).transpose(1, 2).reshape(b, self.cfg.dim, *cshape)
            x = x + F.interpolate(grid, size=(d4, h4, w4), mode="trilinear", align_corners=False)
        x = x.permute(0, 2, 3, 4, 1)                                             # (B, D4, H4, W4, C)
        x = self.in_norm(x) + self._pe((d4, h4, w4), x.device, x.dtype)
        ws = self.cfg.window
        pad = [(-s) % ws for s in (d4, h4, w4)]
        if any(pad):
            x = F.pad(x, (0, 0, 0, pad[2], 0, pad[1], 0, pad[0]))
        bd, bh, bw = x.shape[1:4]
        for blk in self.blocks:
            x = blk["win"](x)
            x = blk["cross"](x.reshape(b, -1, self.cfg.dim), ctx).view(b, bd, bh, bw, self.cfg.dim)
            x = blk["mlp"](x)
        x = self.out_norm(x)[:, :d4, :h4, :w4]
        cell = self.cell_head(x).permute(0, 4, 1, 2, 3)
        desc = self.desc_head(x).permute(0, 4, 1, 2, 3)
        return cell, desc

    # ---------------------------------------------------------------- level B
    def stem_features(self, image: torch.Tensor) -> torch.Tensor | None:
        """1/2-resolution intensity features (B, stem_channels, D/2, H/2, W/2) or None when disabled."""
        if self.stem is None:
            return None
        return self.stem(((image.to(next(self.stem.parameters()).dtype) / 255.0) - 0.5) * 2.0)

    def fine(self, desc: torch.Tensor, image: torch.Tensor, coords: torch.Tensor,
             stem: torch.Tensor | None = None) -> torch.Tensor:
        """desc (B, C, D4, H4, W4), image (B, 1, D, H, W) in [0, 255], coords (B, N, 3) long (z, y, x),
        stem (B, C_s, D/2, H/2, W/2) from stem_features() when enabled -> point logits (B, N)."""
        b, _, d, h, w = image.shape
        n = coords.shape[1]
        size = torch.tensor([d, h, w], device=coords.device, dtype=torch.float32)
        # descriptor: trilinear sample at the voxel centre (grid_sample wants x, y, z order)
        u = (coords.float() + 0.5) / size * 2.0 - 1.0
        grid = u[..., [2, 1, 0]].view(b, n, 1, 1, 3)
        dsc = F.grid_sample(desc, grid.to(desc.dtype), mode="bilinear", padding_mode="border",
                            align_corners=False)[:, :, :, 0, 0].transpose(1, 2)  # (B, N, C)
        # raw-intensity patch
        p = self.cfg.patch
        r = p // 2
        off = torch.stack(torch.meshgrid(*[torch.arange(-r, r + 1, device=coords.device)] * 3, indexing="ij"), -1).view(-1, 3)
        nb = coords[:, :, None, :] + off[None, None]                            # (B, N, p^3, 3)
        nb = torch.minimum(torch.clamp(nb, min=0), (size - 1).long())
        flat = (nb[..., 0] * h + nb[..., 1]) * w + nb[..., 2]                    # (B, N, p^3)
        img = image.reshape(b, -1)
        patch = torch.gather(img, 1, flat.reshape(b, -1)).view(b, n, p ** 3) / 255.0
        patch = (patch - 0.5) * 2.0
        # sub-cell offset (voxel position inside its 4^3 cell), raw + Fourier
        o = ((coords % CELL).float() - (CELL - 1) / 2.0) / (CELL / 2.0)          # [-0.75, 0.75]
        fo = torch.cat([o, torch.sin(math.pi * o), torch.cos(math.pi * o),
                        torch.sin(2 * math.pi * o), torch.cos(2 * math.pi * o)], dim=-1)
        parts = [dsc.float(), patch.float(), fo]
        if self.stem is not None:
            if stem is None:
                raise ValueError("stem_channels > 0: pass stem=self.stem_features(image) to fine()")
            st = F.grid_sample(stem, grid.to(stem.dtype), mode="bilinear", padding_mode="border",
                               align_corners=False)[:, :, :, 0, 0].transpose(1, 2)
            parts.append(st.float())
        feat = torch.cat(parts, dim=-1).to(desc.dtype)
        return self.fine_mlp(self.fine_norm(feat)).squeeze(-1)

    # ---------------------------------------------------------------- training
    @staticmethod
    def sample_points(target: torch.Tensor, valid: torch.Tensor, *, n_pos: int, n_hard: int, n_rand: int,
                      hard_radius: int, generator=None) -> tuple[torch.Tensor, torch.Tensor]:
        """target/valid (B, 1, D, H, W) bool -> coords (B, N, 3) long, labels (B, N) float.
        Per image: up to n_pos GT voxels, n_hard negatives within hard_radius of the GT, n_rand random
        valid negatives (padded by repetition so every image yields the same N)."""
        b = target.shape[0]
        k = 2 * hard_radius + 1
        band = F.max_pool3d(target.float(), kernel_size=k, stride=1, padding=hard_radius) > 0.5
        coords_out, labels_out = [], []
        for i in range(b):
            t, v = target[i, 0], valid[i, 0]
            pos = torch.nonzero(t & v)
            hard = torch.nonzero(band[i, 0] & ~t & v)
            rnd = torch.nonzero(~band[i, 0] & v)
            pieces, labels = [], []
            for src, cap, lab in ((pos, n_pos, 1.0), (hard, n_hard, 0.0), (rnd, n_rand, 0.0)):
                if src.shape[0] == 0 or cap <= 0:
                    continue
                idx = torch.randint(src.shape[0], (cap,), device=src.device, generator=generator) \
                    if src.shape[0] < cap else torch.randperm(src.shape[0], device=src.device, generator=generator)[:cap]
                pieces.append(src[idx])
                labels.append(torch.full((idx.shape[0],), lab, device=src.device))
            c = torch.cat(pieces, 0)
            l = torch.cat(labels, 0)
            coords_out.append(c)
            labels_out.append(l)
        n = min(c.shape[0] for c in coords_out)
        return torch.stack([c[:n] for c in coords_out]), torch.stack([l[:n] for l in labels_out])

    def forward_train(self, taps, context, image, target, valid, *, n_pos, n_hard, n_rand, hard_radius) -> dict[str, torch.Tensor]:
        cell, desc = self.coarse(taps, context)
        coords, labels = self.sample_points(target, valid, n_pos=n_pos, n_hard=n_hard, n_rand=n_rand, hard_radius=hard_radius)
        logits = self.fine(desc, image, coords, stem=self.stem_features(image))
        return {"cell_logits": cell, "point_logits": logits, "point_labels": labels, "coords": coords}

    # ---------------------------------------------------------------- inference
    @torch.no_grad()
    def predict(self, taps, context, image) -> torch.Tensor:
        """Full-volume union logits (B=1): level A everywhere, level B inside cells with
        sigmoid(cell) > fine_gate (dilated fine_dilate cells); elsewhere SPARSE_FILL."""
        cell, desc = self.coarse(taps, context)
        b, _, d, h, w = image.shape
        assert b == 1
        active = torch.sigmoid(cell.float()) > self.cfg.fine_gate
        if self.cfg.fine_dilate > 0:
            k = 2 * self.cfg.fine_dilate + 1
            active = F.max_pool3d(active.float(), kernel_size=k, stride=1, padding=self.cfg.fine_dilate) > 0.5
        out = torch.full((1, 1, d, h, w), SPARSE_FILL, device=image.device, dtype=torch.float32)
        cells = torch.nonzero(active[0, 0])                                      # (K, 3) cell coords
        if cells.shape[0] == 0:
            return out
        sub = torch.stack(torch.meshgrid(*[torch.arange(CELL, device=image.device)] * 3, indexing="ij"), -1).view(-1, 3)
        vox = (cells[:, None, :] * CELL + sub[None]).reshape(-1, 3)              # (K*64, 3)
        vox = vox[(vox[:, 0] < d) & (vox[:, 1] < h) & (vox[:, 2] < w)]
        flat_out = out.view(-1)
        stem = self.stem_features(image)
        for s in range(0, vox.shape[0], self.cfg.chunk):
            c = vox[s:s + self.cfg.chunk]
            lg = self.fine(desc, image, c[None], stem=stem).float()[0]
            flat_out[(c[:, 0] * h + c[:, 1]) * w + c[:, 2]] = lg
        return out


def build_point_decoder(cfg: dict[str, Any]) -> SparsePointUnionDecoder:
    return SparsePointUnionDecoder(PointDecoderConfig.from_mapping(cfg))


class PointUnionSeedModel(nn.Module):
    """Inference wrapper with the `UnionMaskDecoder` interface (auto_instance_seg `_UnionAsBinseg`):
    `set_image(image, p2sd_model)` runs the frozen trunk; `logits(target_ae)` -> (1, 1, D, H, W)."""

    def __init__(self, decoder: SparsePointUnionDecoder) -> None:
        super().__init__()
        self.decoder = decoder
        self._image = self._taps = self._context = None

    @torch.no_grad()
    def set_image(self, image: torch.Tensor, p2sd_model: nn.Module) -> None:
        if image.ndim != 5 or image.shape[0] != 1:
            raise ValueError(f"set_image expects one (1, C, D, H, W) image, got {tuple(image.shape)}")
        enc = p2sd_model.image_encoder
        enc.eval()
        feat, taps = enc.encode_with_pyramid(image)
        _, _, context = p2sd_model.encode_image_context(feat, image.dtype)
        self._image, self._taps, self._context = image, taps, context

    @torch.no_grad()
    def logits(self, target_ae: nn.Module | None = None) -> torch.Tensor:
        if self._context is None:
            raise RuntimeError("PointUnionSeedModel.set_image() must be called before logits()")
        return self.decoder.predict(self._taps, self._context, self._image)


def load_point_union_decoder(path: str | Path, device) -> PointUnionSeedModel:
    state = torch.load(Path(path), map_location="cpu", weights_only=False)
    dec = build_point_decoder(state.get("point_decoder", {}) or {})
    dec.load_state_dict(state["model"], strict=True)
    return PointUnionSeedModel(dec.to(device).eval()).to(device).eval()
