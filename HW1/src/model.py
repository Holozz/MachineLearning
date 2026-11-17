"""小型 Encoder–Decoder Transformer。"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

from .attention import padding_mask, subsequent_mask
from .layers import DecoderBlock, EncoderBlock
from .positional import sinusoidal_positional_encoding


class TinyTransformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 32,
        n_heads: int = 4,
        n_layers: int = 2,
        d_ff: int = 64,
        max_len: int = 64,
        dropout: float = 0.1,
        pad_idx: int = 0,
    ):
        super().__init__()
        self.d_model = d_model
        self.pad_idx = pad_idx
        self.src_embed = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.tgt_embed = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.register_buffer(
            "pe", sinusoidal_positional_encoding(max_len, d_model), persistent=True
        )
        self.encoders = nn.ModuleList(
            [EncoderBlock(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)]
        )
        self.decoders = nn.ModuleList(
            [DecoderBlock(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)]
        )
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(d_model, vocab_size)

    def forward(
        self,
        src: Tensor,
        tgt: Tensor,
        src_mask: Optional[Tensor] = None,
        tgt_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """src: (B, T_src)，tgt: (B, T_tgt)（已含 BOS 的 teacher-forcing 输入）。

        返回 logits，形状 (B, T_tgt, vocab_size)。

        建议步骤
        --------
        1. embedding * sqrt(d_model)（原论文；课件第 8 页只强调再加 PE）
        2. 加上 self.pe[:T]
        3. dropout
        4. 若未传入 mask：源端用 padding_mask(src)；目标端用 subsequent_mask 与 padding_mask 的逻辑与
        5. 依次通过 encoders、decoders
        6. out_proj
        """
        # TODO
        src_x = self.src_embed(src) * math.sqrt(self.d_model)
        tgt_x = self.tgt_embed(tgt) * math.sqrt(self.d_model)

        src_x = self.dropout(src_x + self.pe[: src.size(1)])
        tgt_x = self.dropout(tgt_x + self.pe[: tgt.size(1)])

        if src_mask is None:
            src_mask = padding_mask(src, self.pad_idx)
        if tgt_mask is None:
            tgt_mask = subsequent_mask(tgt.size(1), device=tgt.device) * padding_mask(tgt, self.pad_idx)

        memory = src_x
        for encoder in self.encoders:
            memory = encoder(memory, src_mask)

        out = tgt_x
        for decoder in self.decoders:
            out = decoder(out, memory, tgt_mask, src_mask)

        return self.out_proj(out)