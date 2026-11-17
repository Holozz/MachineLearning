"""Transformer 积木：与面试手撕清单对齐。

必须手写 forward 的模块
----------------------
Softmax / 缩放点积注意力见 attention.py
LayerNorm、FFN、Multi-Head Attention、EncoderBlock、DecoderBlock 见本文件
TinyTransformer.forward 见 model.py

允许使用 nn.Linear / nn.Embedding / nn.Dropout / 矩阵乘 / autograd。
禁止 nn.LayerNorm、nn.MultiheadAttention、nn.Transformer*、F.softmax。
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

from .attention import scaled_dot_product_attention


class LayerNorm(nn.Module):
    """对最后一个维度做标准化，再乘 gamma、加 beta。"""

    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))

    def forward(self, x: Tensor) -> Tensor:
        """x: (..., d_model) -> 同形状。

        mean / var 沿最后一维；用总体方差（unbiased=False）。
        """
        # TODO
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, unbiased=False, keepdim=True)

        stand_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.gamma * stand_x + self.beta
        raise NotImplementedError("请实现 LayerNorm.forward")


class MultiHeadAttention(nn.Module):
    """先线性投影，再拆成 H 个头。拆头辅助函数已给出，forward 必须自己写。"""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model 必须能被 n_heads 整除")
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.last_weights: Optional[Tensor] = None

    def _split_heads(self, x: Tensor) -> Tensor:
        # (B, T, d_model) -> (B, H, T, d_k)
        b, t, _ = x.shape
        x = x.view(b, t, self.n_heads, self.d_k)
        return x.transpose(1, 2)

    def _combine_heads(self, x: Tensor) -> Tensor:
        # (B, H, T, d_k) -> (B, T, d_model)
        b, h, t, d_k = x.shape
        x = x.transpose(1, 2).contiguous()
        return x.view(b, t, h * d_k)

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        """query/key/value: (B, T, d_model)。返回 (B, T_q, d_model)。

        步骤：W_q/k/v → split heads → scaled_dot_product_attention
        → dropout(weights) → combine → W_o。
        把权重存到 self.last_weights，形状 (B, H, T_q, T_k)。
        """
        # TODO
        q = self._split_heads(self.W_q(query))
        k = self._split_heads(self.W_k(key))
        v = self._split_heads(self.W_v(value))

        _, weights = scaled_dot_product_attention(q, k, v, mask)
        self.last_weights = weights

        o = self.W_o(self._combine_heads(self.dropout(weights) @ v))
        return o
        raise NotImplementedError("请实现 MultiHeadAttention.forward")


class PositionwiseFFN(nn.Module):
    """两层全连接，中间 ReLU。原论文 d_ff 通常为 4 * d_model。"""

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        # TODO: fc1 → ReLU → dropout → fc2。可用 torch.relu，不要换 GELU
        return self.fc2(self.dropout(torch.relu(self.fc1(x))))
        raise NotImplementedError("请实现 PositionwiseFFN.forward")


class EncoderBlock(nn.Module):
    """两个子层：Self-Attn、FFN。本作业用 Post-LN：LN(x + Dropout(Sublayer(x)))。"""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ffn = PositionwiseFFN(d_model, d_ff, dropout)
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        # TODO: self-attn(Q=K=V=x) + 残差 + LN，再 FFN + 残差 + LN
        attn = self.self_attn.forward(x, x, x, mask)
        x = self.norm1.forward(x + self.dropout(attn))
        x = self.norm2.forward(x + self.dropout(self.ffn.forward(x)))
        return x
        raise NotImplementedError("请实现 EncoderBlock.forward")


class DecoderBlock(nn.Module):
    """三个子层：Masked Self-Attn → Cross-Attn → FFN，均为 Post-LN。"""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ffn = PositionwiseFFN(d_model, d_ff, dropout)
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.norm3 = LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: Tensor,
        memory: Tensor,
        self_mask: Optional[Tensor] = None,
        cross_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """x: (B, T_tgt, d)；memory: Encoder 输出 (B, T_src, d)。

        Cross-attention：Q 来自 x，K/V 来自 memory。
        """
        # TODO
        attn = self.self_attn.forward(x, x, x, self_mask)
        q = self.norm1.forward(x + self.dropout(attn))
        cross_attn = self.cross_attn.forward(q, memory, memory, cross_mask)

        x = self.norm2.forward(q + self.dropout(cross_attn))
        x = self.norm3.forward(x + self.dropout(self.ffn.forward(x)))
        
        return x
        raise NotImplementedError("请实现 DecoderBlock.forward")
