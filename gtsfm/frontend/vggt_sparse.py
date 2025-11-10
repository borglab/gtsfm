"""Utilities to add sparse attention support to VGGT at runtime."""

from __future__ import annotations

import types
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from gtsfm.utils.logger import get_logger

logger = get_logger()


@dataclass(frozen=True)
class SparseAttentionOptions:
    """Configuration controlling the sparse attention patch."""

    threshold: float = 0.7
    k_nearest: int = 10


class MegaLocMPS:
    """Lightweight descriptor extractor approximating MegaLoc behaviour.

    The original MegaLoc model is large and may require additional third-party
    weights. To keep the sparse attention patch lightweight and dependency
    free, we approximate the behaviour with a simple global descriptor based on
    adaptive avg pooling. This keeps the interface compatible with the original
    reference implementation while avoiding large downloads.
    """

    def __init__(self, device: str | torch.device = "cpu") -> None:
        self._device = torch.device(device)

    @torch.no_grad()
    def extract_features(self, image: torch.Tensor) -> torch.Tensor:
        """Return a normalized descriptor for a single image tensor."""

        if image.ndim != 4:
            raise ValueError("Expected image tensor shaped (1, 3, H, W).")

        image = image.to(self._device, dtype=torch.float32, copy=False)
        pooled = F.adaptive_avg_pool2d(image, output_size=(16, 16))
        descriptor = pooled.flatten(start_dim=1)
        descriptor = F.normalize(descriptor, dim=-1)
        return descriptor.squeeze(0).cpu()

    @torch.no_grad()
    def compute_covisibility_matrix(
        self,
        features: torch.Tensor,
        *,
        threshold: float,
        k_nearest: int,
    ) -> torch.Tensor:
        """Compute a binary covisibility mask from feature vectors."""

        if features.ndim != 2:
            raise ValueError("Expected feature tensor shaped (num_images, feat_dim).")

        if features.shape[0] == 0:
            return torch.zeros(0, 0, dtype=torch.bool)

        feats = F.normalize(features.to(dtype=torch.float32), dim=-1)
        similarity = torch.matmul(feats, feats.T)

        num_images = feats.shape[0]
        k = min(int(k_nearest) + 1, num_images)
        _, topk_indices = torch.topk(similarity, k=k, dim=-1)

        mask = torch.zeros_like(similarity, dtype=torch.bool)
        mask.scatter_(1, topk_indices, True)
        mask = mask | mask.T  # ensure symmetry

        if threshold is not None:
            mask &= similarity >= float(threshold)

        mask |= torch.eye(num_images, dtype=torch.bool)
        return mask


class _SparseAttentionManager:
    """Internal helper that patches VGGT's aggregator to use sparse attention."""

    def __init__(
        self,
        aggregator: nn.Module,
        feature_extractor: MegaLocMPS,
        options: SparseAttentionOptions,
    ) -> None:
        self._aggregator = aggregator
        self._feature_extractor = feature_extractor
        self._options = options

        self._orig_forward = aggregator.forward
        self._orig_process_global_attention = aggregator._process_global_attention  # type: ignore[attr-defined]

        self._frame_mask: Optional[torch.Tensor] = None

        self._patch_forward()

    # ------------------------------------------------------------------
    # Public API exposed on aggregator
    # ------------------------------------------------------------------
    def set_covisibility_mask(self, images: torch.Tensor) -> None:
        """Compute and cache the covisibility mask for the current batch."""

        if images.ndim == 4:
            images = images.unsqueeze(0)

        if images.ndim != 5:
            raise ValueError("Images must be shaped (B, S, 3, H, W) for VGGT.")

        batch, sequence = images.shape[:2]
        if sequence <= 1:
            self._frame_mask = None
            return

        masks = []
        for b in range(batch):
            descriptors = []
            for s in range(sequence):
                single = images[b, s].unsqueeze(0)
                descriptor = self._feature_extractor.extract_features(single)
                descriptors.append(descriptor)

            feat_tensor = torch.stack(descriptors)
            mask = self._feature_extractor.compute_covisibility_matrix(
                feat_tensor,
                threshold=self._options.threshold,
                k_nearest=self._options.k_nearest,
            )
            masks.append(mask)

        self._frame_mask = torch.stack(masks)

    # ------------------------------------------------------------------
    # Monkey patching helpers
    # ------------------------------------------------------------------
    def _patch_forward(self) -> None:
        aggregator = self._aggregator

        if hasattr(aggregator, "_sparse_attention_manager"):
            raise RuntimeError("Sparse attention already enabled on this aggregator.")

        aggregator._sparse_attention_manager = self  # type: ignore[attr-defined]

        aggregator.forward = types.MethodType(self._forward_with_sparse_attention, aggregator)
        aggregator.set_covisibility_mask = self.set_covisibility_mask  # type: ignore[attr-defined]
        aggregator._process_global_attention = types.MethodType(  # type: ignore[attr-defined]
            self._process_global_attention_with_mask,
            aggregator,
        )

    def _forward_with_sparse_attention(self, aggregator: nn.Module, images: torch.Tensor):
        # Pre-compute covisibility mask for current batch.
        try:
            aggregator.set_covisibility_mask(images)  # type: ignore[attr-defined]
        except Exception as exc:  # pragma: no cover - defensive: keep VGGT working
            logger.warning("Failed to compute sparse attention mask: %s", exc)
            self._frame_mask = None

        return self._orig_forward(images)

    def _process_global_attention_with_mask(
        self,
        aggregator: nn.Module,
        tokens: torch.Tensor,
        B: int,
        S: int,
        P: int,
        C: int,
        global_idx: int,
        pos: Optional[torch.Tensor] = None,
    ):
        """Wrap the original global attention to inject sparse masks."""

        # Fallback to the dense implementation if we do not have a mask or if
        # the module is running in training mode (checkpointing is unsupported).
        mask = self._frame_mask
        if mask is None or aggregator.training:
            return self._orig_process_global_attention(tokens, B, S, P, C, global_idx, pos=pos)

        if tokens.shape != (B, S * P, C):
            tokens = tokens.view(B, S, P, C).view(B, S * P, C)

        if pos is not None and pos.shape != (B, S * P, 2):
            pos = pos.view(B, S, P, 2).view(B, S * P, 2)

        token_mask = self._expand_frame_mask(mask.to(device=tokens.device), S, P)

        intermediates = []
        for _ in range(aggregator.aa_block_size):
            block = aggregator.global_blocks[global_idx]
            tokens = self._run_block_with_mask(block, tokens, pos, token_mask)
            global_idx += 1
            intermediates.append(tokens.view(B, S, P, C))

        return tokens, global_idx, intermediates

    # ------------------------------------------------------------------
    # Attention helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _expand_frame_mask(mask: torch.Tensor, frames: int, tokens_per_frame: int) -> torch.Tensor:
        """Expand a frame-level mask to the per-token representation."""

        batch = mask.shape[0]
        device = mask.device
        frame_indices = torch.arange(frames, device=device).repeat_interleave(tokens_per_frame)
        expanded = mask[:, frame_indices.unsqueeze(0), frame_indices.unsqueeze(1)]
        expanded = expanded | torch.eye(frame_indices.numel(), device=device, dtype=torch.bool)
        assert expanded.shape == (batch, frames * tokens_per_frame, frames * tokens_per_frame)
        return expanded

    def _run_block_with_mask(
        self,
        block: nn.Module,
        tokens: torch.Tensor,
        pos: Optional[torch.Tensor],
        token_mask: torch.Tensor,
    ) -> torch.Tensor:
        attn_module = block.attn  # type: ignore[attr-defined]
        original_forward = attn_module.forward

        def forward_with_mask(attn_self, x: torch.Tensor, pos=None):
            attn_bias = self._build_attention_bias(token_mask.to(device=x.device), attn_self, x)
            return self._attention_forward(attn_self, x, pos, attn_bias)

        attn_module.forward = types.MethodType(forward_with_mask, attn_module)
        try:
            return block(tokens, pos=pos)
        finally:
            attn_module.forward = original_forward

    @staticmethod
    def _build_attention_bias(
        token_mask: torch.Tensor,
        attn_module: nn.Module,
        tokens: torch.Tensor,
    ) -> torch.Tensor:
        batch, length = token_mask.shape[:2]
        allowed = token_mask.unsqueeze(1)
        bias = torch.zeros(
            batch,
            attn_module.num_heads,
            length,
            length,
            device=tokens.device,
            dtype=tokens.dtype,
        )
        bias.masked_fill_(~allowed, float("-inf"))
        return bias

    @staticmethod
    def _attention_forward(attn_module: nn.Module, x: torch.Tensor, pos, attn_bias: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = attn_module.qkv(x).reshape(B, N, 3, attn_module.num_heads, attn_module.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = attn_module.q_norm(q), attn_module.k_norm(k)

        if attn_module.rope is not None:
            q = attn_module.rope(q, pos)
            k = attn_module.rope(k, pos)

        dropout = attn_module.attn_drop.p if attn_module.training else 0.0

        if attn_module.fused_attn:
            x = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_bias, dropout_p=dropout)
        else:
            q = q * attn_module.scale
            attn = torch.matmul(q, k.transpose(-2, -1))
            attn = attn + attn_bias
            attn = attn.softmax(dim=-1)
            attn = attn_module.attn_drop(attn)
            x = torch.matmul(attn, v)

        x = x.transpose(1, 2).reshape(B, N, C)
        x = attn_module.proj(x)
        x = attn_module.proj_drop(x)
        return x


def make_vggt_sparse(
    vggt_model: nn.Module,
    *,
    device: str | torch.device = "cpu",
    options: Optional[SparseAttentionOptions] = None,
) -> nn.Module:
    """Patch a VGGT instance to use sparse global attention."""

    opts = options or SparseAttentionOptions()
    feature_extractor = MegaLocMPS(device=device)
    _SparseAttentionManager(vggt_model.aggregator, feature_extractor, opts)
    logger.info(
        "Enabled sparse attention for VGGT aggregator (threshold=%.2f, k=%d).",
        opts.threshold,
        opts.k_nearest,
    )
    return vggt_model


__all__ = [
    "SparseAttentionOptions",
    "MegaLocMPS",
    "make_vggt_sparse",
]
