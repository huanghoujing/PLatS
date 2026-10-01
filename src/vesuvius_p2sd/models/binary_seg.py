"""Prompt-free binary sheet segmentation on a frozen P2SD trunk.

The model reuses a trained P2SD's image encoder + image-context attention
exactly as they run during prompted inference (``encode_image_context_from_image``)
and attaches a fresh AE-style conv-upsampling decoder to the contextualized
feature volume. Only the decoder trains; the trunk -- including the prompt
modulator and refiner, which never execute here -- is frozen wholesale.

The decoder mirrors the proven SheetAE dense decoder shape: five
ResBlock3d + ConvTranspose3d(2x) stages take the 10^3 context volume to the
320^3 input resolution, followed by one ResBlock and a 1-channel head.
"""

from __future__ import annotations

from typing import Any, Sequence

import torch
from torch import nn

from vesuvius_p2sd.models.ae.sheet_ae import ResBlock3d
from vesuvius_p2sd.models.p2sd import build_p2sd
from vesuvius_p2sd.models.p2sd.full_attn import (
    AttentionBlock,
    grid_coordinates,
    tokens_to_latent,
)


class BinarySegFromP2SD(nn.Module):
    def __init__(
        self,
        trunk: nn.Module,
        *,
        decoder_channels: Sequence[int] = (512, 256, 128, 64, 32, 16),
        num_groups: int = 8,
        dropout: float = 0.0,
        refiner_depth: int = 0,
        refiner_heads: int = 8,
        refiner_mlp_ratio: float = 4.0,
        refiner_rope_mode: str = "per_axis",
        freeze_decoder: bool = False,
        contrast_tap: str = "decoder_stage",
        contrast_dim: int = 0,
        contrast_tap_stage: int = 8,
    ) -> None:
        super().__init__()
        channels = [int(value) for value in decoder_channels]
        if len(channels) != 6:
            raise ValueError(f"decoder_channels needs 6 entries for 5 upsamples, got {channels}")
        if channels[0] != int(trunk.cfg.dim):
            raise ValueError(
                f"decoder_channels[0] must match the trunk token dim {trunk.cfg.dim}, got {channels[0]}")
        self.trunk = trunk
        for parameter in self.trunk.parameters():
            parameter.requires_grad = False
        stages: list[nn.Module] = []
        for index in range(5):
            stages.append(ResBlock3d(channels[index], num_groups=num_groups, dropout=dropout))
            stages.append(nn.ConvTranspose3d(channels[index], channels[index + 1], 2, stride=2))
        stages.append(ResBlock3d(channels[5], num_groups=num_groups, dropout=dropout))
        self.decoder = nn.Sequential(*stages)
        self.head = nn.Conv3d(channels[5], 1, 1)
        # Optional global self-attention over the 10^3 context tokens between
        # the frozen trunk and the conv decoder. Each block's output projections
        # (attn proj + SwiGLU fc2) are zero-initialized, so the stack is exactly
        # identity at step 0: a warm-started decoder sees unperturbed context
        # and the refiner learns a pure residual correction.
        self.context_refiner = nn.ModuleList(
            AttentionBlock(
                channels[0], refiner_heads, refiner_mlp_ratio,
                use_rope=True, rope_mode=refiner_rope_mode,
            )
            for _ in range(int(refiner_depth))
        )
        for block in self.context_refiner:
            for module in (block.proj, block.fc2):
                nn.init.zeros_(module.weight)
                nn.init.zeros_(module.bias)
        if freeze_decoder:
            for parameter in [*self.decoder.parameters(), *self.head.parameters()]:
                parameter.requires_grad = False
        # Optional voxel-embedding branch for the prototype-contrast loss. It
        # taps a MID decoder stage (default: after the ResBlock at 1/2
        # resolution, 32ch at PS320) so the final binseg-specific stages and
        # logits head stay task-pure -- the two objectives share early decoder
        # features but do not fight over the last layers.
        self.contrast_tap_stage = int(contrast_tap_stage)
        self.contrast_tap = str(contrast_tap)
        if self.contrast_tap not in {"decoder_stage", "context"}:
            raise ValueError(f"contrast.tap must be decoder_stage|context, got {self.contrast_tap!r}")
        if contrast_dim > 0:
            if self.contrast_tap == "context":
                # Head reads the (optionally attention-refined) 1/32 context
                # grid: instance identity needs the global view -- a conv
                # decoder tap cannot separate locally-identical voxels of
                # different sheets (0024/0025/0037 all failed there).
                tap_channels = None
                for module in self.decoder:
                    if hasattr(module, "in_channels"):
                        tap_channels = int(module.in_channels)
                        break
                if tap_channels is None:
                    raise ValueError("could not infer context channels for contrast tap")
            else:
                if self.contrast_tap_stage >= len(self.decoder):
                    raise ValueError(
                        f"contrast_tap_stage {contrast_tap_stage} out of range for "
                        f"a {len(self.decoder)}-module decoder")
                tap_channels = None
                for module in list(self.decoder)[self.contrast_tap_stage::-1]:
                    if hasattr(module, "out_channels"):
                        tap_channels = int(module.out_channels)
                        break
                    if hasattr(module, "channels"):
                        tap_channels = int(module.channels)
                        break
                if tap_channels is None:
                    tap_channels = channels[0]
            self.contrast_head = nn.Conv3d(tap_channels, int(contrast_dim), 1)
        else:
            self.contrast_head = None

    def encode_context(self, image: torch.Tensor) -> torch.Tensor:
        # The trunk is frozen: run it without autograd bookkeeping and in eval
        # mode so no Conv3d activations are retained for backward.
        self.trunk.eval()
        with torch.no_grad():
            _, _, _, context = self.trunk.encode_image_context_from_image(image)
        return context.detach()

    def refine_context(self, context: torch.Tensor) -> torch.Tensor:
        """Refine the FULL context grid with global self-attention (RoPE coords).

        Must run on the whole 10^3 grid, before any decode crop -- the point of
        the attention is cross-volume sheet structure, and RoPE coordinates are
        derived from the grid shape. No-op when refiner_depth is 0.
        """

        if len(self.context_refiner) == 0:
            return context
        b, _, dz, dy, dx = context.shape
        tokens = context.flatten(2).transpose(1, 2)
        coords = grid_coordinates(b, (dz, dy, dx), device=context.device, dtype=torch.float32)
        for block in self.context_refiner:
            tokens = block(tokens, coords)
        return tokens_to_latent(tokens, (dz, dy, dx))

    def decode_context(self, context: torch.Tensor) -> torch.Tensor:
        """Decode a context volume (or any aligned sub-crop of one) to logits.

        The decoder is pure convolution, so it is translation-equivariant and
        coordinate-independent: training it on grid-aligned context crops (one
        context voxel = 32^3 output voxels) and running it on the full grid at
        inference is the standard patch-training regime.
        """

        return self.head(self.decoder(context))

    def decode_context_with_embedding(
        self, context: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Logits plus the contrast-branch voxel embedding (at tap resolution).

        The embedding is the raw head output; normalize AFTER any spatial
        sampling (grid_sample of unit vectors is not a unit vector).
        """

        if self.contrast_head is None:
            raise RuntimeError("model was built without a contrast head (contrast_dim=0)")
        if self.contrast_tap == "context":
            return self.head(self.decoder(context)), self.contrast_head(context)
        feat = context
        embedding = None
        for index, module in enumerate(self.decoder):
            feat = module(feat)
            if index == self.contrast_tap_stage:
                embedding = self.contrast_head(feat)
        return self.head(feat), embedding

    def embed_context(self, context: torch.Tensor) -> torch.Tensor:
        """Embedding only, skipping the (frozen) seg decode entirely.

        The distill-only training regime (seg loss weights 0, tap=context)
        never needs logits; skipping the full-volume decode saves most of the
        step.
        """

        if self.contrast_head is None or self.contrast_tap != "context":
            raise RuntimeError("embed_context requires a contrast head with tap=context")
        return self.contrast_head(context)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.decode_context(self.refine_context(self.encode_context(image)))

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]


def build_binary_seg_model(cfg: dict[str, Any]) -> BinarySegFromP2SD:
    seg_cfg = cfg.get("binary_seg", {})
    # latent_channels only sizes the trunk's (frozen, unused) prediction heads;
    # 64 matches the checkpoints this model warm-starts from.
    trunk = build_p2sd(cfg, latent_channels=int(seg_cfg.get("trunk_latent_channels", 64)))
    refiner_cfg = seg_cfg.get("context_refiner", {})
    return BinarySegFromP2SD(
        trunk,
        decoder_channels=seg_cfg.get("decoder_channels", (512, 256, 128, 64, 32, 16)),
        num_groups=int(seg_cfg.get("num_groups", 8)),
        dropout=float(seg_cfg.get("dropout", 0.0)),
        refiner_depth=int(refiner_cfg.get("depth", 0)),
        refiner_heads=int(refiner_cfg.get("heads", 8)),
        refiner_mlp_ratio=float(refiner_cfg.get("mlp_ratio", 4.0)),
        refiner_rope_mode=str(refiner_cfg.get("rope_mode", "per_axis")),
        freeze_decoder=bool(seg_cfg.get("freeze_decoder", False)),
        contrast_dim=int(seg_cfg.get("contrast", {}).get("dim", 0)
                         if seg_cfg.get("contrast", {}).get("enabled", False) else 0),
        contrast_tap_stage=int(seg_cfg.get("contrast", {}).get("tap_stage", 8)),
        contrast_tap=str(seg_cfg.get("contrast", {}).get("tap", "decoder_stage")),
    )
