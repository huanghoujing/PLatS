"""Image-conditioned refinement of AE-decoded sheets (0061, 2026-08-27).

P2SD regresses a 64-d latent per 32^3 cell and the frozen sheet AE decodes it
with an image-blind decoder, so the only image information in a full-res
sheet is what the regressor squeezed into that latent. Its sub-cell placement
error caps the per-sheet dice at ~0.51 (the AE decoding the GT latent reaches
0.92; per-sheet surface dice at tau=2 is 0.87). ``SheetRefineHead`` fuses the
AE decoder's intermediate features (1/4, 1/2 and full resolution) with the
frozen P2SD image encoder's 1/4 feature and the raw image, and predicts a
RESIDUAL on the AE logits (zero-initialised output = identity at start). Only
per-voxel channel RMSNorm is used (no spatial statistics), so crop and
full-volume behaviour are identical and the head can be trained on the exact
inference path: full-volume decodes of P2SD's own predictions.

``RefinedSheetDecoder`` wraps a sheet AE so ``decode(z)`` returns refined
logits; callers set the case image once per case (``set_image``), which
computes the encoder pyramid the head needs. Everything else is forwarded to
the wrapped AE, so it is a drop-in for ``target_ae`` in the instance-seg and
scene-eval decode paths.
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = self.conv2(F.silu(self.norm2(h)))
        return x + h


def _int_keys(d: dict | None) -> dict[int, int]:
    return {int(k): int(v) for k, v in (d or {}).items()}


class SheetRefineHead(nn.Module):
    """Residual refinement of AE sheet logits from AE decoder taps + image features.

    Inputs (all for the same volume, batch-aligned):
      ae_logits   (B, 1, D, H, W)          AE head output
      ae_taps     {4: (B, C4, D/4, ..), 2: (B, C2, D/2, ..), 1: (B, C1, D, ..)}
      image_taps  {4: (B, I4, D/4, ..)}    frozen P2SD encoder feature(s)
      image       (B, 1, D, H, W)          the raw (normalised) image
    """

    def __init__(
        self,
        *,
        ae_channels: dict[int, int] | None = None,
        image_channels: dict[int, int] | None = None,
        widths: tuple[int, int, int] = (32, 16, 8),
        image_raw_channels: int = 1,
        grad_clamp: float | None = 10000.0,
        arch: str = "v1",
    ) -> None:
        super().__init__()
        # v1 (0061-0063): ResBlock at every level, receptive field ~15 voxels.
        # v2 (0064, tiled): no ResBlock at 1/4 -> receptive field <= 7 voxels at
        # full res, so the head is exact on 32^3 tile centres with the AE's
        # 8-voxel halo (see SheetAE.decode_sparse_stages).
        if arch not in {"v1", "v2"}:
            raise ValueError(f"arch must be v1|v2, got {arch!r}")
        self.arch = arch
        self.checkpoint_level1 = False   # set by the trainer (p2sd.refine.checkpoint_level1) to trade compute for memory
        self.ae_channels = _int_keys(ae_channels) or {4: 24, 2: 16, 1: 8}
        self.image_channels = _int_keys(image_channels) or {4: 96}
        w4, w2, w1 = (int(w) for w in widths)
        self.widths = (w4, w2, w1)
        in4 = self.ae_channels[4] + self.image_channels.get(4, 0)
        self.in4 = nn.Conv3d(in4, w4, 1)
        self.block4 = _ResBlock(w4, grad_clamp) if arch == "v1" else nn.Identity()
        self.up4 = nn.ConvTranspose3d(w4, w2, 2, stride=2)
        in2 = w2 + self.ae_channels[2] + self.image_channels.get(2, 0)
        self.in2 = nn.Conv3d(in2, w2, 1)
        self.block2 = _ResBlock(w2, grad_clamp)
        self.up2 = nn.ConvTranspose3d(w2, w1, 2, stride=2)
        in1 = w1 + self.ae_channels[1] + 1 + int(image_raw_channels) + self.image_channels.get(1, 0)
        self.in1 = nn.Conv3d(in1, w1, 3, padding=1)
        self.block1 = _ResBlock(w1, grad_clamp)
        self.out = nn.Conv3d(w1, 1, 1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    @property
    def ae_factors(self) -> tuple[int, ...]:
        return tuple(sorted(self.ae_channels, reverse=True))

    @property
    def image_factors(self) -> tuple[int, ...]:
        return tuple(sorted(self.image_channels, reverse=True))

    def forward(
        self,
        ae_logits: torch.Tensor,
        ae_taps: dict[int, torch.Tensor],
        image_taps: dict[int, torch.Tensor],
        image: torch.Tensor,
        valid: dict[int, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """Dense (valid=None) or per-tile forward; `valid` = {4,2,1} in-volume masks for tiles
        (reproduces dense zero-padding at the volume border, as in SheetAE.decode_sparse_stages)."""
        m4 = valid[4] if valid else None
        m2 = valid[2] if valid else None
        m1 = valid[1] if valid else None
        x4 = [ae_taps[4]] + ([image_taps[4]] if 4 in self.image_channels else [])
        h = self.in4(torch.cat(x4, dim=1))
        h = self.block4(h) if self.arch == "v1" else F.silu(h)
        if m4 is not None:
            h = h * m4
        h = self.up4(h)
        if m2 is not None:
            h = h * m2
        x2 = [h, ae_taps[2]] + ([image_taps[2]] if 2 in self.image_channels else [])
        h = self.in2(torch.cat(x2, dim=1))
        h = self._block_masked(self.block2, h, m2)
        h = self.up2(h)
        if m1 is not None:
            h = h * m1          # the 3x3 in1 must see zeros outside the volume, as the dense pass does
        x1 = [h, ae_taps[1], ae_logits.to(h.dtype), image.to(h.dtype)]
        if 1 in self.image_channels:
            x1.append(image_taps[1])
        x1 = torch.cat(x1, dim=1)
        if self.checkpoint_level1 and torch.is_grad_enabled() and any(p.requires_grad for p in self.block1.parameters()):
            from torch.utils.checkpoint import checkpoint
            h = checkpoint(self._level1, x1, m1, use_reentrant=False)
        else:
            h = self._level1(x1, m1)
        return ae_logits + self.out(h).to(ae_logits.dtype)

    def _level1(self, x1: torch.Tensor, m1: torch.Tensor | None) -> torch.Tensor:
        """Full-resolution level (the memory hog): recomputed in backward when checkpoint_level1 is set."""
        h = self.in1(x1)
        return self._block_masked(self.block1, h, m1)

    @staticmethod
    def _block_masked(block: "_ResBlock", x: torch.Tensor, valid: torch.Tensor | None) -> torch.Tensor:
        if valid is None:
            return block(x)
        x = x * valid
        h = block.conv1(F.silu(block.norm1(x))) * valid
        h = block.conv2(F.silu(block.norm2(h)))
        return (x + h) * valid


def build_refine_head(refine_cfg: dict[str, Any]) -> SheetRefineHead:
    return SheetRefineHead(
        ae_channels=refine_cfg.get("ae_channels"),
        image_channels=refine_cfg.get("image_channels"),
        widths=tuple(int(w) for w in refine_cfg.get("widths", (32, 16, 8))),
        image_raw_channels=int(refine_cfg.get("image_raw_channels", 1)),
        grad_clamp=refine_cfg.get("grad_clamp", 10000.0),
        arch=str(refine_cfg.get("arch", "v1")),
    )


def load_refine_head(path: str | Path, device) -> SheetRefineHead:
    state = torch.load(Path(path), map_location="cpu", weights_only=False)
    head = build_refine_head(state.get("refine", {}) or {})
    head.load_state_dict(state["model"], strict=True)
    return head.to(device).eval()


class RefinedSheetDecoder(nn.Module):
    """Drop-in for a sheet AE whose ``decode`` refines through a SheetRefineHead.

    ``set_image(image)`` must be called once per case with the SAME tensor the
    P2SD encoder consumes (B=1); ``decode(z)`` then works for any latent batch
    (the case's image taps are broadcast along the batch).
    """

    def __init__(self, target_ae: nn.Module, head: SheetRefineHead, image_encoder: nn.Module) -> None:
        super().__init__()
        self.target_ae = target_ae
        self.head = head
        self.image_encoder = image_encoder
        self.tile_threshold = -4.0   # stage-2 aux logit above which a 1/4 voxel activates its tile (v2 sparse path)
        self._image: torch.Tensor | None = None
        self._image_taps: dict[int, torch.Tensor] | None = None

    @torch.no_grad()
    def set_image(self, image: torch.Tensor) -> None:
        if image.ndim != 5 or image.shape[0] != 1:
            raise ValueError(f"set_image expects one (1, C, D, H, W) image, got {tuple(image.shape)}")
        self.image_encoder.eval()
        _, taps = self.image_encoder.encode_with_pyramid(image)
        self._image_taps = {f: taps[f] for f in self.head.image_factors}
        self._image = image

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        if self._image is None or self._image_taps is None:
            raise RuntimeError("RefinedSheetDecoder.set_image() must be called for the case before decode()")
        b = int(z.shape[0])
        if self.head.arch == "v2":
            return self.decode_sparse(z)
        logits, ae_taps = self.target_ae.decode_with_taps(z, self.head.ae_factors)
        image_taps = {f: t.expand(b, *t.shape[1:]) for f, t in self._image_taps.items()}
        image = self._image.expand(b, *self._image.shape[1:])
        return self.head(logits, ae_taps, image_taps, image)

    def decode_sparse(self, z: torch.Tensor) -> torch.Tensor:
        """Block-sparse path (v2 heads): dense AE decode (exact), head only on the tiles the base decode
        marks as possibly occupied (logit > tile_threshold, dilated by 4 voxels); base logits elsewhere."""
        ae = self.target_ae
        b = int(z.shape[0])
        logits, taps = ae.decode_with_taps(z, (4, 2, 1))
        active4 = F.max_pool3d((logits > self.tile_threshold).float(), kernel_size=4, stride=4) > 0
        tiles = ae.active_tiles(active4, tile4=ae.SPARSE_TILE // 4, dilate=1)
        if not bool(tiles.any()):
            return logits
        full = {4: taps[4], 2: taps[2], 1: torch.cat([taps[1], logits.to(taps[1].dtype),
                                                   self._image.expand(b, *self._image.shape[1:]).to(taps[1].dtype)], dim=1)}
        t, valid, idx = ae.gather_tile_set(tiles, full)
        c1 = int(taps[1].shape[1])
        ae_t = {4: t[4], 2: t[2], 1: t[1][:, :c1]}
        logits_t = t[1][:, c1:c1 + 1]
        image_t = t[1][:, c1 + 1:c1 + 2]
        img4 = self._image_taps[4].expand(b, *self._image_taps[4].shape[1:])
        img4_t, _ = ae.gather_tiles(img4, tiles, ae.SPARSE_TILE // 4, ae.SPARSE_HALO4)
        refined_t = self.head(logits_t, ae_t, {4: img4_t * valid[4]}, image_t, valid=valid)
        return ae.scatter_tiles(refined_t, idx, logits)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """AE-style forward (encode -> refined decode); the repair step calls the AE this way."""
        return {"logits": self.decode(self.target_ae.encode(x))}

    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(super().__getattr__("target_ae"), name)


def load_refined_decoder(target_ae: nn.Module, image_encoder: nn.Module, path: str | Path, device):
    """Dispatch on the checkpoint: a `sheet_decoder` checkpoint (0065, models/sheet_decoder.py) builds the
    image-conditioned LatentSheetDecoder wrapper; otherwise the SheetRefineHead wrapper."""
    state = torch.load(Path(path), map_location="cpu", weights_only=False)
    if "sheet_decoder" in state:
        from vesuvius_p2sd.models.sheet_decoder import load_image_conditioned_decoder

        return load_image_conditioned_decoder(target_ae, image_encoder, path, device)
    if "band_refiner" in state:   # 0069: band refiner on top of the 0065a decoder (models/band_refiner.py)
        from vesuvius_p2sd.models.band_refiner import load_band_refined_decoder

        return load_band_refined_decoder(target_ae, image_encoder, path, device)
    return RefinedSheetDecoder(target_ae, load_refine_head(path, device), image_encoder).to(device).eval()
