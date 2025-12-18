"""
LoRA adaptation for Segment Anything (SAM).

This module injects lightweight low-rank adapters into the vision encoder
of HuggingFace `SamModel`. The goal is to fine-tune the image encoder
with a small number of additional parameters while keeping the original
weights frozen. The mask decoder is left trainable and can be fully
fine-tuned on the target dataset.

Typical usage:

    from transformers import SamModel
    from sam_LoRA import apply_lora_to_sam, freeze_sam_except_lora

    sam = SamModel.from_pretrained("facebook/sam-vit-base")
    sam = apply_lora_to_sam(sam, r=64, alpha=64, dropout=0.1)
    freeze_sam_except_lora(sam)

References:

1. https://github.com/MathieuNlp/Sam_LoRA/blob/main/src/lora.py
2. https://github.com/computational-cell-analytics/micro-sam/blob/master/micro_sam/models/peft_sam.py
3. https://github.com/JamesQFreeman/Sam_LoRA/blob/main/sam_lora.py


"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import math
import torch
from torch import nn


@dataclass
class LoRAConfig:
    """Configuration for LoRA in the SAM vision encoder.

    Args:
        r: Rank of the low-rank matrices (bottleneck size).
        alpha: Scaling factor applied to the LoRA update (effective scale alpha / r).
        dropout: Dropout probability on the input before the LoRA blocks.
    """

    r: int = 32
    alpha: float = 32.0
    dropout: float = 0.0


class LoRAQKV(nn.Module):
    """LoRA wrapper for a combined QKV projection.

    Expects a linear layer mapping d_model → 3 * d_model producing
    concatenated [q, k, v]. This wrapper adds a low-rank, additive update
    to the query and value projections:

        q' = q + Δq,  v' = v + Δv,  k' = k

    The original layer is kept inside and typically frozen. Only the
    LoRA parameters are trainable.
    """

    def __init__(self, qkv: nn.Linear, config: LoRAConfig) -> None:
        super().__init__()
        self.qkv = qkv
        self.config = config

        in_features = qkv.in_features
        out_features = qkv.out_features
        if out_features % 3 != 0:
            raise ValueError(
                f"Expected qkv.out_features to be divisible by 3, got {out_features}."
            )
        d_model = out_features // 3

        self.input_dropout = (
            nn.Dropout(config.dropout) if config.dropout > 0.0 else nn.Identity()
        )

        r = config.r
        # A: in_features → r,  B: r → d_model (for q and v)
        self.lora_A_q = nn.Linear(in_features, r, bias=False)
        self.lora_A_v = nn.Linear(in_features, r, bias=False)
        self.lora_B_q = nn.Linear(r, d_model, bias=False)
        self.lora_B_v = nn.Linear(r, d_model, bias=False)

        # Start from zero update: A random, B zero.
        nn.init.kaiming_uniform_(self.lora_A_q.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.lora_A_v.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B_q.weight)
        nn.init.zeros_(self.lora_B_v.weight)

        self.scale = config.alpha / float(config.r)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        qkv_out = self.qkv(x)
        q, k, v = qkv_out.chunk(3, dim=-1)

        x_dropped = self.input_dropout(x)

        delta_q = self.lora_B_q(self.lora_A_q(x_dropped)) * self.scale
        delta_v = self.lora_B_v(self.lora_A_v(x_dropped)) * self.scale

        q = q + delta_q
        v = v + delta_v

        return torch.cat([q, k, v], dim=-1)


def apply_lora_to_sam(
    model: "transformers.SamModel",
    *,
    r: int = 32,
    alpha: float = 32.0,
    dropout: float = 0.0,
) -> "transformers.SamModel":
    """Insert LoRA adapters into the vision encoder of a SAM model.

    LoRA is applied to the combined QKV projection in each transformer
    block of the vision encoder. Architectures with:

        model.vision_encoder.encoder.layers   (older)
        model.vision_encoder.layers           (newer)

    are supported.

    Args:
        model: A `SamModel` instance.
        r: Rank of the low-rank adapters.
        alpha: Scaling factor for the LoRA update.
        dropout: Dropout probability before the LoRA blocks.

    Returns:
        The same model instance with LoRAQKV wrappers attached.
    """
    if not hasattr(model, "vision_encoder"):
        raise ValueError("Expected SamModel with a `vision_encoder` attribute.")

    if hasattr(model.vision_encoder, "encoder") and hasattr(
        model.vision_encoder.encoder, "layers"
    ):
        layers = model.vision_encoder.encoder.layers
    elif hasattr(model.vision_encoder, "layers"):
        layers = model.vision_encoder.layers
    else:
        raise ValueError(
            "Could not locate transformer layers on `vision_encoder`. "
            "Check SAM version before applying LoRA."
        )

    config = LoRAConfig(r=r, alpha=alpha, dropout=dropout)

    for idx, layer in enumerate(layers):
        if hasattr(layer, "self_attn"):
            attn_module = layer.self_attn
        elif hasattr(layer, "attn"):
            attn_module = layer.attn
        else:
            # Not a transformer block; skip.
            continue

        if not hasattr(attn_module, "qkv"):
            raise ValueError(
                f"Layer {idx} attention module has no `qkv` attribute. "
                "This LoRA injection assumes a combined QKV projection."
            )

        original_qkv = attn_module.qkv
        if not isinstance(original_qkv, LoRAQKV):
            attn_module.qkv = LoRAQKV(original_qkv, config)

    return model


def freeze_sam_except_lora(model: "transformers.SamModel") -> None:
    """Freeze all SAM parameters except LoRA (vision encoder) and mask decoder.

    After calling this, gradients are enabled only for:

      * LoRA parameters inside `vision_encoder`
      * all parameters in `mask_decoder`

    The prompt encoder and original SAM weights remain frozen.
    """
    for p in model.parameters():
        p.requires_grad = False

    for name, p in model.named_parameters():
        if "vision_encoder" in name and "lora_" in name:
            p.requires_grad = True

    if hasattr(model, "mask_decoder"):
        for p in model.mask_decoder.parameters():
            p.requires_grad = True
