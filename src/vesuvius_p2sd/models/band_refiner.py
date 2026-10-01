"""0069 (2026-08-29): SHEET BAND REFINER — the user's design: global self-attention among ALL
full-resolution voxels of a 5-voxel band around one decoded sheet, per 160^3 patch (no
Swin-style sub-windows), trained on random patches, run by sliding patches.

Per sheet (one P2SD latent z):
  1. the frozen 0065a LatentSheetDecoder decodes z (gated tiles, as in production) -> logits,
     and its 1/4 feature h4 is kept;
  2. band B = dilate(logits > 0, radius) via iterated 3^3 max-pools (also gives the distance
     0..radius of every band voxel to the decoded mask);
  3. every band voxel of a patch is a token: [intensity, frozen tap 1/4 (trilinear), frozen tap
     1/8 (trilinear), decoder h4 (trilinear), decoder stem 1/2 (trilinear), decoder logit, in-mask,
     distance, (0069c: frozen binseg sigmoid prob — the all-sheets foreground the p2sd decode does
     not carry), Fourier position inside the patch] -> LayerNorm -> Linear -> dim;
  4. `depth` blocks of [global self-attention over the patch's tokens (flash SDPA, O(N) memory)
     + MLP]; head -> delta; refined logit = logit + delta inside the band, unchanged outside.

`BandRefinedSheetDecoder` wraps an ImageConditionedSheetDecoder with the same interface
(`set_image`, `decode`, `forward`), so `--refine_head_path band_last.pt` drops into every chain.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class BandRefinerConfig:
    dim: int = 96
    heads: int = 6
    depth: int = 4
    radius: int = 5
    patch: int = 160
    stride: int = 128
    pe_freqs: int = 4
    tap_channels: dict[int, int] = field(default_factory=lambda: {4: 96, 8: 192})
    h4_channels: int = 48
    stem_channels: int = 8
    max_tokens: int = 400_000        # inference: larger sets are split into random groups
    mlp_ratio: float = 4.0
    residual: bool = True            # True (0069): refined = logit + head(x), zero-init head (identity at init)
                                     # False (0069b): refined = head(x) — the whole logit, not restricted by the input
    binseg_channels: int = 0         # 0069c: 1 = frozen binseg sigmoid prob (all-sheets foreground) per token

    @classmethod
    def from_mapping(cls, cfg: dict[str, Any]) -> "BandRefinerConfig":
        known = {f for f in cls.__dataclass_fields__}
        kwargs = {k: v for k, v in (cfg or {}).items() if k in known}
        if "tap_channels" in kwargs:
            kwargs["tap_channels"] = {int(k): int(v) for k, v in kwargs["tap_channels"].items()}
        return cls(**kwargs)

    @property
    def in_dim(self) -> int:
        return (1 + sum(self.tap_channels.values()) + self.h4_channels + self.stem_channels + 3
                + self.binseg_channels + 3 * 2 * self.pe_freqs)


class GlobalAttention(nn.Module):
    """Self-attention over one whole token set (B=1, N tokens). Uses SDPA so the flash /
    mem-efficient kernels keep memory O(N); head dim 16 for the kernels."""

    def __init__(self, dim: int, heads: int) -> None:
        super().__init__()
        self.heads = heads
        self.norm = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:       # (N, C)
        n, c = x.shape
        hd = c // self.heads
        qkv = self.qkv(self.norm(x)).view(1, n, 3, self.heads, hd).permute(2, 0, 3, 1, 4)
        a = F.scaled_dot_product_attention(qkv[0], qkv[1], qkv[2])   # (1, heads, N, hd)
        return x + self.proj(a.transpose(1, 2).reshape(n, c))


class MLPBlock(nn.Module):
    def __init__(self, dim: int, ratio: float) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, int(dim * ratio))
        self.fc2 = nn.Linear(int(dim * ratio), dim)
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.fc2(F.gelu(self.fc1(self.norm(x))))


def band_from_mask(mask: torch.Tensor, radius: int) -> tuple[torch.Tensor, torch.Tensor]:
    """mask (1,1,D,H,W) bool -> band (1,1,D,H,W) bool = dilate(mask, radius), dist (1,1,D,H,W) uint8
    = Chebyshev distance to the mask, 0..radius inside the band (radius+1 outside)."""
    cur = mask.float()
    dist = torch.full(mask.shape, radius + 1, device=mask.device, dtype=torch.uint8)
    dist[mask] = 0
    for r in range(1, radius + 1):
        nxt = F.max_pool3d(cur, kernel_size=3, stride=1, padding=1)
        ring = (nxt > 0.5) & ~(cur > 0.5)
        dist[ring] = r
        cur = nxt
    return cur > 0.5, dist


class BandRefiner(nn.Module):
    def __init__(self, cfg: BandRefinerConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.in_norm = nn.LayerNorm(cfg.in_dim)
        self.in_proj = nn.Linear(cfg.in_dim, cfg.dim)
        self.blocks = nn.ModuleList(nn.ModuleDict({"attn": GlobalAttention(cfg.dim, cfg.heads),
                                                   "mlp": MLPBlock(cfg.dim, cfg.mlp_ratio)}) for _ in range(cfg.depth))
        self.out_norm = nn.LayerNorm(cfg.dim)
        self.head = nn.Linear(cfg.dim, 1)
        if cfg.residual:
            nn.init.zeros_(self.head.weight)
            nn.init.zeros_(self.head.bias)

    # ------------------------------------------------------------------ features
    @staticmethod
    def _sample(vol: torch.Tensor, coords: torch.Tensor, size: torch.Tensor) -> torch.Tensor:
        """Trilinear sample of vol (1,C,d,h,w) at full-res voxel centres coords (N,3) long -> (N,C)."""
        u = (coords.float() + 0.5) / size * 2.0 - 1.0
        grid = u[:, [2, 1, 0]].view(1, -1, 1, 1, 3)
        return F.grid_sample(vol, grid.to(vol.dtype), mode="bilinear", padding_mode="border",
                             align_corners=False)[0, :, :, 0, 0].transpose(0, 1)

    @staticmethod
    def _gather(vol: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        """vol (1,1,D,H,W) -> values at coords (N,3) -> (N,)."""
        d, h, w = vol.shape[-3:]
        flat = (coords[:, 0] * h + coords[:, 1]) * w + coords[:, 2]
        return vol.reshape(-1)[flat]

    def features(self, coords: torch.Tensor, origin: torch.Tensor, *, image, taps, h4, stem2, logits, mask, dist,
                 binseg=None) -> torch.Tensor:
        """Token features (N, in_dim) for band voxels `coords` (N,3) of the patch at `origin` (3,)."""
        size = torch.tensor(image.shape[-3:], device=coords.device, dtype=torch.float32)
        parts = [self._gather(image, coords).float()[:, None] / 255.0]
        for f in sorted(self.cfg.tap_channels):
            parts.append(self._sample(taps[f], coords, size).float())
        parts.append(self._sample(h4, coords, size).float())
        parts.append(self._sample(stem2, coords, size).float())
        lg = self._gather(logits, coords).float().clamp(-20.0, 20.0)
        parts.append(torch.stack([lg / 10.0, self._gather(mask, coords).float(),
                                  self._gather(dist, coords).float() / float(self.cfg.radius)], dim=1))
        if self.cfg.binseg_channels:
            if binseg is None:
                raise RuntimeError("binseg_channels > 0 requires the binseg probability volume")
            parts.append(self._gather(binseg, coords).float()[:, None])
        rel = (coords - origin.view(1, 3)).float() / float(self.cfg.patch)      # [0, 1) inside the patch
        fo = [rel]
        for k in range(self.cfg.pe_freqs):
            fo += [torch.sin(math.pi * (2 ** k) * rel), torch.cos(math.pi * (2 ** k) * rel)]
        parts.append(torch.cat(fo[1:], dim=1))
        return torch.cat(parts, dim=1)

    # ------------------------------------------------------------------ network
    def forward_tokens(self, feat: torch.Tensor) -> torch.Tensor:
        """(N, in_dim) -> delta (N,) for one attention set."""
        x = self.in_proj(self.in_norm(feat.to(self.in_proj.weight.dtype)))
        for blk in self.blocks:
            x = blk["attn"](x)
            x = blk["mlp"](x)
        return self.head(self.out_norm(x)).squeeze(-1)

    def forward_tokens_split(self, feat: torch.Tensor, generator=None) -> torch.Tensor:
        """Inference guard: sets larger than max_tokens are split into random groups (each a set)."""
        n = feat.shape[0]
        if n <= self.cfg.max_tokens:
            return self.forward_tokens(feat)
        groups = int(math.ceil(n / self.cfg.max_tokens))
        perm = torch.randperm(n, device=feat.device, generator=generator)
        out = torch.empty(n, device=feat.device, dtype=torch.float32)
        for g in range(groups):
            idx = perm[g::groups]
            out[idx] = self.forward_tokens(feat[idx]).float()
        return out

    # ------------------------------------------------------------------ inference
    @torch.no_grad()
    def refine(self, logits: torch.Tensor, *, image, taps, h4, stem2, binseg=None) -> torch.Tensor:
        """logits (1,1,D,H,W) of one decoded sheet -> refined logits (same shape). Sliding patches of
        `patch` with `stride` over the band's bounding box; overlapping deltas are averaged."""
        mask = logits > 0
        if not bool(mask.any()):
            return logits
        band, dist = band_from_mask(mask, self.cfg.radius)
        d, h, w = logits.shape[-3:]
        p, s = self.cfg.patch, self.cfg.stride
        nz = torch.nonzero(band[0, 0])
        lo = nz.min(0).values.tolist()
        hi = nz.max(0).values.tolist()

        def starts(l, u, n):
            if n <= p:
                return [0]
            first = max(0, min(l, n - p))
            out = list(range(first, max(first, u - p) + 1, s))
            last = min(max(u - p + 1, 0), n - p)
            if out[-1] < last:
                out.append(last)
            return out

        delta_sum = torch.zeros_like(logits, dtype=torch.float32)
        delta_cnt = torch.zeros_like(logits, dtype=torch.float32)
        for z0 in starts(lo[0], hi[0], d):
            for y0 in starts(lo[1], hi[1], h):
                for x0 in starts(lo[2], hi[2], w):
                    sub = band[0, 0, z0:z0 + p, y0:y0 + p, x0:x0 + p]
                    if not bool(sub.any()):
                        continue
                    origin = torch.tensor([z0, y0, x0], device=logits.device)
                    coords = torch.nonzero(sub) + origin
                    feat = self.features(coords, origin, image=image, taps=taps, h4=h4, stem2=stem2,
                                         logits=logits, mask=mask, dist=dist, binseg=binseg)
                    delta = self.forward_tokens_split(feat).float()
                    flat = (coords[:, 0] * h + coords[:, 1]) * w + coords[:, 2]
                    delta_sum.view(-1).index_add_(0, flat, delta)
                    delta_cnt.view(-1).index_add_(0, flat, torch.ones_like(delta))
        out = logits.float().clone()
        hit = delta_cnt > 0
        if self.cfg.residual:
            out[hit] = out[hit] + delta_sum[hit] / delta_cnt[hit]
        else:                                   # direct prediction inside the band; outside it the decode stands
            out[hit] = delta_sum[hit] / delta_cnt[hit]
        return out.to(logits.dtype)


def build_band_refiner(cfg: dict[str, Any]) -> BandRefiner:
    return BandRefiner(BandRefinerConfig.from_mapping(cfg))


# ---------------------------------------------------------------------- decoder helper
def decode_one(dec, ae, z: torch.Tensor, skips: dict[int, torch.Tensor], stem2: torch.Tensor, image: torch.Tensor,
               *, gate_logit: float, gate_dilate: int, chunk: int = 256) -> tuple[torch.Tensor, torch.Tensor]:
    """Production decode of ONE latent through a LatentSheetDecoder (coarse dense, fine on gated tiles),
    returning (logits (1,1,D,H,W), h4 (1,C4,D/4,H/4,W/4)). Mirrors ImageConditionedSheetDecoder.decode."""
    from vesuvius_p2sd.models.sheet_decoder import fine_in_chunks

    h4, aux = dec.coarse(z, skips)
    tiles = ae.active_tiles(aux > gate_logit, tile4=ae.SPARSE_TILE // 4, dilate=gate_dilate)
    size = tuple(int(v) * int(ae.downsample_factor) for v in z.shape[-3:])
    out = torch.full((1, 1) + size, ae.SPARSE_FILL, device=z.device, dtype=h4.dtype)
    if not bool(tiles.any()):
        return out, h4
    full = {4: h4, 2: stem2, 1: image.to(h4.dtype)}
    t, valid, idx = ae.gather_tile_set(tiles, full)
    logits_t = fine_in_chunks(dec, t, valid, chunk)
    return ae.scatter_tiles(logits_t, idx, out), h4


class BandRefinedSheetDecoder(nn.Module):
    """ImageConditionedSheetDecoder + BandRefiner with the same interface: `set_image(image)`,
    `decode(z)` (decode, then refine each sheet's band), `forward(mask)` for aepp."""

    def __init__(self, inner, refiner: BandRefiner, binseg_model=None) -> None:
        super().__init__()
        self.inner = inner
        self.refiner = refiner
        self.binseg_model = binseg_model
        self._binseg = None

    @torch.no_grad()
    def set_image(self, image: torch.Tensor) -> None:
        self.inner.set_image(image)
        # 0069c: the refiner's binseg-prob feature is computed here from the RAW image, so every
        # consumer of the --refine_head_path dispatch (instseg decode, aepp, scene-eval shards)
        # works unchanged. One redundant binseg forward per case beside instseg's own foreground.
        if self.refiner.cfg.binseg_channels:
            if self.binseg_model is None:
                raise RuntimeError("band refiner has binseg_channels > 0 but no binseg model was loaded")
            self._binseg = torch.sigmoid(self.binseg_model(image).float())

    @torch.no_grad()
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        inner = self.inner
        if inner._image is None:
            raise RuntimeError("BandRefinedSheetDecoder.set_image() must be called for the case before decode()")
        outs = []
        for i in range(int(z.shape[0])):
            logits, h4 = decode_one(inner.decoder, inner.target_ae, z[i:i + 1], inner._skips, inner._stem2, inner._image,
                                    gate_logit=inner.gate_logit, gate_dilate=inner.gate_dilate, chunk=inner.tile_chunk)
            taps = {f: inner._skips[f] for f in self.refiner.cfg.tap_channels}
            outs.append(self.refiner.refine(logits, image=inner._image, taps=taps, h4=h4, stem2=inner._stem2,
                                            binseg=self._binseg))
        return torch.cat(outs, dim=0)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        return {"logits": self.decode(self.inner.target_ae.encode(x))}

    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(super().__getattr__("inner"), name)


def load_band_refined_decoder(target_ae: nn.Module, image_encoder: nn.Module, path: str | Path, device) -> BandRefinedSheetDecoder:
    from vesuvius_p2sd.models.sheet_decoder import load_image_conditioned_decoder

    state = torch.load(Path(path), map_location="cpu", weights_only=False)
    cfg = state.get("band_refiner", {}) or {}
    refiner = build_band_refiner(cfg)
    refiner.load_state_dict(state["model"], strict=True)
    inner = load_image_conditioned_decoder(target_ae, image_encoder, state["decoder_checkpoint"], device)
    binseg_model = None
    if refiner.cfg.binseg_channels:
        binseg_model = load_frozen_binseg(state["binseg_run_dir"], device)
    return BandRefinedSheetDecoder(inner, refiner.to(device).eval(), binseg_model).to(device).eval()


def load_frozen_binseg(run_dir: str | Path, device) -> nn.Module:
    """The binseg proposer exactly as auto_instance_seg loads it (resolved_config + last.pt), frozen."""
    from vesuvius_p2sd.models.binary_seg import build_binary_seg_model
    from vesuvius_p2sd.utils.config import load_config

    run_dir = Path(run_dir)
    cfg = load_config(run_dir / "resolved_config.yaml")
    cfg["device"] = str(device)
    model = build_binary_seg_model(cfg).to(device)
    state = torch.load(run_dir / "last.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(state.get("model", state), strict=True)
    for p in model.parameters():
        p.requires_grad_(False)
    return model.eval()
