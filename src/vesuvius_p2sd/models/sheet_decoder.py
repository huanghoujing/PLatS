"""Image-conditioned sheet decoder from the P2SD latent (0065, 2026-08-28).

Replaces the frozen 0032 AE decoder + refinement head: ONE decoder that maps the
P2SD latent (64 ch per 32^3 cell, AE latent space) to a full-resolution sheet
mask with skip connections from the frozen P2SD image encoder at 1/32, 1/16,
1/8 and 1/4, a learned stride-2 stem on the raw image at 1/2, and the raw image
at full resolution. Trained from scratch on the thin GT sheet masks (no AE
teacher). All normalisation is per-voxel RMSNorm, so the two fine stages (1/2,
full) are exactly tileable: their receptive field is 6 voxels, and 32^3 tiles
gathered with an 8-voxel halo from the dense 1/4 feature reproduce the dense
forward bit-for-bit (out-of-volume positions are re-zeroed after every op to
mirror dense zero padding). Tiles are gated by the decoder's own 1/4 aux head.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from vesuvius_p2sd.models.ae.sheet_ae import RMSNorm3d


class _ResBlock(nn.Module):
    def __init__(self, channels: int, grad_clamp: float | None) -> None:
        super().__init__()
        self.norm1 = RMSNorm3d(channels, eps=1e-5, grad_clamp=grad_clamp)
        self.conv1 = nn.Conv3d(channels, channels, 3, padding=1)
        self.norm2 = RMSNorm3d(channels, eps=1e-5, grad_clamp=grad_clamp)
        self.conv2 = nn.Conv3d(channels, channels, 3, padding=1)

    def forward(self, x: torch.Tensor, valid: torch.Tensor | None = None) -> torch.Tensor:
        if valid is None:
            h = self.conv1(F.silu(self.norm1(x)))
            h = self.conv2(F.silu(self.norm2(h)))
            return x + h
        x = x * valid
        h = self.conv1(F.silu(self.norm1(x))) * valid
        h = self.conv2(F.silu(self.norm2(h)))
        return (x + h) * valid


class LatentSheetDecoder(nn.Module):
    COARSE_FACTORS = (32, 16, 8, 4)

    def __init__(
        self,
        *,
        latent_channels: int = 64,
        skip_channels: dict[int, int] | None = None,
        widths: tuple[int, ...] = (128, 96, 64, 48, 16, 8),   # at 1/32, 1/16, 1/8, 1/4, 1/2, 1
        stem_channels: int = 8,
        grad_clamp: float | None = 10000.0,
    ) -> None:
        super().__init__()
        self.skip_channels = {int(k): int(v) for k, v in (skip_channels or {32: 512, 16: 384, 8: 192, 4: 96}).items()}
        self.widths = tuple(int(w) for w in widths)
        w32, w16, w8, w4, w2, w1 = self.widths
        self.z_norm = RMSNorm3d(latent_channels, eps=1e-5, grad_clamp=grad_clamp)
        self.skip_norm = nn.ModuleDict({str(f): RMSNorm3d(c, eps=1e-5, grad_clamp=grad_clamp) for f, c in self.skip_channels.items()})
        self.in32 = nn.Conv3d(latent_channels + self.skip_channels[32], w32, 1)
        self.block32 = _ResBlock(w32, grad_clamp)
        self.up32 = nn.ConvTranspose3d(w32, w16, 2, stride=2)
        self.in16 = nn.Conv3d(w16 + self.skip_channels[16], w16, 1)
        self.block16 = _ResBlock(w16, grad_clamp)
        self.up16 = nn.ConvTranspose3d(w16, w8, 2, stride=2)
        self.in8 = nn.Conv3d(w8 + self.skip_channels[8], w8, 1)
        self.block8 = _ResBlock(w8, grad_clamp)
        self.up8 = nn.ConvTranspose3d(w8, w4, 2, stride=2)
        self.in4 = nn.Conv3d(w4 + self.skip_channels[4], w4, 1)
        self.block4 = _ResBlock(w4, grad_clamp)
        self.aux4 = nn.Conv3d(w4, 1, 1)
        # fine (tileable) part
        self.stem_channels = int(stem_channels)
        self.image_stem = nn.Sequential(
            nn.Conv3d(1, self.stem_channels, 3, stride=2, padding=1),
            RMSNorm3d(self.stem_channels, eps=1e-5, grad_clamp=grad_clamp),
            nn.SiLU(),
        )
        self.up4 = nn.ConvTranspose3d(w4, w2, 2, stride=2)
        self.in2 = nn.Conv3d(w2 + self.stem_channels, w2, 1)
        self.block2 = _ResBlock(w2, grad_clamp)
        self.up2 = nn.ConvTranspose3d(w2, w1, 2, stride=2)
        self.in1 = nn.Conv3d(w1 + 1, w1, 1)
        self.block1 = _ResBlock(w1, grad_clamp)
        self.out = nn.Conv3d(w1, 1, 1)
        self.checkpoint_level1 = False

    # ---- coarse (dense) ----
    def coarse(self, z: torch.Tensor, skips: dict[int, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """z (S,64,g,g,g) + encoder skips {32,16,8,4} (S,C,..) -> (h4 (S,w4,4g,..), aux logits at 1/4)."""
        s = {f: self.skip_norm[str(f)](skips[f]) for f in self.COARSE_FACTORS}
        h = self.block32(self.in32(torch.cat([self.z_norm(z), s[32]], dim=1)))
        h = self.up32(h)
        h = self.block16(self.in16(torch.cat([h, s[16]], dim=1)))
        h = self.up16(h)
        h = self.block8(self.in8(torch.cat([h, s[8]], dim=1)))
        h = self.up8(h)
        h = self.block4(self.in4(torch.cat([h, s[4]], dim=1)))
        return h, self.aux4(h)

    # ---- fine (dense or tiled) ----
    def fine(self, h4: torch.Tensor, stem2: torch.Tensor, image: torch.Tensor,
             valid: dict[int, torch.Tensor] | None = None) -> torch.Tensor:
        """h4 (N,w4,a,..) at 1/4, stem2 (N,stem,2a,..) at 1/2, image (N,1,4a,..) -> logits (N,1,4a,..).
        With `valid` = {4,2,1} in-volume masks (tiles) the result on tile centres equals the dense forward."""
        v4 = valid[4] if valid else None
        v2 = valid[2] if valid else None
        v1 = valid[1] if valid else None
        h = h4 if v4 is None else h4 * v4
        h = self.up4(h)
        if v2 is not None:
            h = h * v2
            stem2 = stem2 * v2
        h = self.block2(self.in2(torch.cat([h, stem2.to(h.dtype)], dim=1)), v2)
        h = self.up2(h)
        if v1 is not None:
            h = h * v1
            image = image * v1
        x1 = torch.cat([h, image.to(h.dtype)], dim=1)
        if self.checkpoint_level1 and torch.is_grad_enabled() and any(p.requires_grad for p in self.block1.parameters()):
            from torch.utils.checkpoint import checkpoint
            h = checkpoint(self._level1, x1, v1, use_reentrant=False)
        else:
            h = self._level1(x1, v1)
        return self.out(h)

    def _level1(self, x1: torch.Tensor, v1: torch.Tensor | None) -> torch.Tensor:
        return self.block1(self.in1(x1), v1)

    def fine_rf(self) -> int:
        return 6   # +-2 at 1/2 (=4 full) + +-2 at full


def _fine_autocast():
    """The tiled fine stages under an fp16 autocast hit a Blackwell kernel bug (Xid 31 MMU faults in
    two fp16 instseg runs on 2026-08-28, none in ~2500 bf16 scene-eval decodes): run them in bf16
    where supported (fp32 otherwise) whenever the ambient autocast is fp16."""
    if torch.is_autocast_enabled() and torch.get_autocast_gpu_dtype() == torch.float16:
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32
        return torch.autocast("cuda", dtype=dtype, enabled=dtype != torch.float32) if dtype != torch.float32 \
            else torch.autocast("cuda", enabled=False)
    import contextlib
    return contextlib.nullcontext()


def fine_in_chunks(dec: LatentSheetDecoder, t: dict[int, torch.Tensor], valid: dict[int, torch.Tensor],
                   chunk: int) -> torch.Tensor:
    """Run the fine stages over tiles in chunks (inference memory bound: an untrained gate or a dense
    union can activate all 1000 tiles of a 320^3 case)."""
    n = int(t[4].shape[0])
    with _fine_autocast():
        if n <= chunk:
            return dec.fine(t[4], t[2], t[1], valid=valid).float()
        outs = [dec.fine(t[4][i:i + chunk], t[2][i:i + chunk], t[1][i:i + chunk],
                         valid={f: v[i:i + chunk] for f, v in valid.items()}).float() for i in range(0, n, chunk)]
        return torch.cat(outs, dim=0)


def build_sheet_decoder(cfg: dict[str, Any]) -> LatentSheetDecoder:
    return LatentSheetDecoder(
        latent_channels=int(cfg.get("latent_channels", 64)),
        skip_channels=cfg.get("skip_channels"),
        widths=tuple(int(w) for w in cfg.get("widths", (128, 96, 64, 48, 16, 8))),
        stem_channels=int(cfg.get("stem_channels", 8)),
        grad_clamp=cfg.get("grad_clamp", 10000.0),
    )


def load_sheet_decoder(path: str | Path, device) -> tuple[LatentSheetDecoder, dict[str, Any]]:
    state = torch.load(Path(path), map_location="cpu", weights_only=False)
    cfg = state.get("sheet_decoder", {}) or {}
    dec = build_sheet_decoder(cfg)
    dec.load_state_dict(state["model"], strict=True)
    return dec.to(device).eval(), cfg


class ImageConditionedSheetDecoder(nn.Module):
    """Drop-in for a sheet AE: `set_image(image)` once per case (encoder pyramid + stem), then
    `decode(z)` for any latent batch through the LatentSheetDecoder (coarse dense, fine tiled and
    gated by the aux head); `forward(mask)` = decode(ae.encode(mask)) for the aepp repair step.
    Everything else (latent_channels, downsample_factor, encode, ...) is forwarded to the AE."""

    def __init__(self, target_ae: nn.Module, decoder: LatentSheetDecoder, image_encoder: nn.Module,
                 gate_logit: float = -3.0, gate_dilate: int = 1) -> None:
        super().__init__()
        self.target_ae = target_ae
        self.decoder = decoder
        self.image_encoder = image_encoder
        self.gate_logit = float(gate_logit)
        self.gate_dilate = int(gate_dilate)
        self.tile_chunk = 256   # fine-stage tiles per forward at inference (memory: ~40 MB/tile in fp16+fp32 norms)
        self._image: torch.Tensor | None = None
        self._skips: dict[int, torch.Tensor] | None = None
        self._stem2: torch.Tensor | None = None

    @torch.no_grad()
    def set_image(self, image: torch.Tensor) -> None:
        if image.ndim != 5 or image.shape[0] != 1:
            raise ValueError(f"set_image expects one (1, C, D, H, W) image, got {tuple(image.shape)}")
        self.image_encoder.eval()
        _, taps = self.image_encoder.encode_with_pyramid(image)
        self._skips = {f: taps[f] for f in LatentSheetDecoder.COARSE_FACTORS}
        self._stem2 = self.decoder.image_stem(image)
        self._image = image

    def _expand(self, t: torch.Tensor, b: int) -> torch.Tensor:
        return t.expand(b, *t.shape[1:])

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        if self._image is None:
            raise RuntimeError("ImageConditionedSheetDecoder.set_image() must be called for the case before decode()")
        ae, dec = self.target_ae, self.decoder
        b = int(z.shape[0])
        h4, aux = dec.coarse(z, {f: self._expand(t, b) for f, t in self._skips.items()})
        tiles = ae.active_tiles(aux > self.gate_logit, tile4=ae.SPARSE_TILE // 4, dilate=self.gate_dilate)
        size = tuple(int(v) * int(ae.downsample_factor) for v in z.shape[-3:])
        out = torch.full((b, 1) + size, ae.SPARSE_FILL, device=z.device, dtype=h4.dtype)
        if not bool(tiles.any()):
            return out
        full = {4: h4, 2: self._expand(self._stem2, b), 1: self._expand(self._image, b).to(h4.dtype)}
        t, valid, idx = ae.gather_tile_set(tiles, full)
        logits_t = fine_in_chunks(dec, t, valid, self.tile_chunk)
        return ae.scatter_tiles(logits_t, idx, out)

    def decode_dense(self, z: torch.Tensor) -> torch.Tensor:
        """Ungated dense decode (verification / small volumes)."""
        b = int(z.shape[0])
        h4, _ = self.decoder.coarse(z, {f: self._expand(t, b) for f, t in self._skips.items()})
        return self.decoder.fine(h4, self._expand(self._stem2, b), self._expand(self._image, b).to(h4.dtype))

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        return {"logits": self.decode(self.target_ae.encode(x))}

    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(super().__getattr__("target_ae"), name)


def load_image_conditioned_decoder(target_ae: nn.Module, image_encoder: nn.Module, path: str | Path, device,
                                   gate_logit: float | None = None) -> ImageConditionedSheetDecoder:
    dec, cfg = load_sheet_decoder(path, device)
    gate = float(cfg.get("gate_logit", -3.0)) if gate_logit is None else float(gate_logit)
    return ImageConditionedSheetDecoder(target_ae, dec, image_encoder, gate_logit=gate,
                                        gate_dilate=int(cfg.get("gate_dilate", 1))).to(device).eval()


class UnionMaskDecoder(nn.Module):
    """Binary UNION-mask decoder (0066): a LatentSheetDecoder whose 'latent' is the P2SD image
    context (self-attention refined, C ch per 32^3 cell) instead of a per-sheet latent; same encoder
    skips, image stem, aux gate and exact tiling. Replaces the 0022 binseg as the seed model:
    `set_image(image, p2sd_model)` then `logits()` -> (1,1,D,H,W) foreground logits."""

    def __init__(self, decoder: LatentSheetDecoder, gate_logit: float = -3.0, gate_dilate: int = 1) -> None:
        super().__init__()
        self.decoder = decoder
        self.gate_logit = float(gate_logit)
        self.gate_dilate = int(gate_dilate)
        self.tile_chunk = 256
        self._image = self._context = self._skips = self._stem2 = None

    @torch.no_grad()
    def set_image(self, image: torch.Tensor, p2sd_model: nn.Module) -> None:
        if image.ndim != 5 or image.shape[0] != 1:
            raise ValueError(f"set_image expects one (1, C, D, H, W) image, got {tuple(image.shape)}")
        enc = p2sd_model.image_encoder
        enc.eval()
        feat, taps = enc.encode_with_pyramid(image)
        _, _, context = p2sd_model.encode_image_context(feat, image.dtype)
        self._context = context
        self._skips = {f: taps[f] for f in LatentSheetDecoder.COARSE_FACTORS}
        self._stem2 = self.decoder.image_stem(image)
        self._image = image

    @torch.no_grad()
    def logits(self, target_ae: nn.Module) -> torch.Tensor:
        """Full-volume union logits (tiles gated by the aux head; background = SPARSE_FILL)."""
        if self._context is None:
            raise RuntimeError("UnionMaskDecoder.set_image() must be called before logits()")
        h4, aux = self.decoder.coarse(self._context, self._skips)
        tiles = target_ae.active_tiles(aux > self.gate_logit, tile4=target_ae.SPARSE_TILE // 4, dilate=self.gate_dilate)
        size = tuple(int(v) * 32 for v in self._context.shape[-3:])
        out = torch.full((1, 1) + size, target_ae.SPARSE_FILL, device=h4.device, dtype=h4.dtype)
        if not bool(tiles.any()):
            return out
        t, valid, idx = target_ae.gather_tile_set(tiles, {4: h4, 2: self._stem2, 1: self._image.to(h4.dtype)})
        return target_ae.scatter_tiles(fine_in_chunks(self.decoder, t, valid, self.tile_chunk), idx, out)


def load_union_decoder(path: str | Path, device):
    state = torch.load(Path(path), map_location="cpu", weights_only=False)
    if "point_decoder" in state:   # 0068: sparse point union decoder (same set_image/logits interface)
        from vesuvius_p2sd.models.point_decoder import load_point_union_decoder
        return load_point_union_decoder(path, device)
    cfg = state.get("union_decoder", {}) or {}
    dec = build_sheet_decoder(cfg)
    dec.load_state_dict(state["model"], strict=True)
    return UnionMaskDecoder(dec.to(device).eval(), gate_logit=float(cfg.get("gate_logit", -3.0)),
                            gate_dilate=int(cfg.get("gate_dilate", 1))).to(device).eval()
