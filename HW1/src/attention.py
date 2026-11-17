"""缩放点积注意力与 mask。面试手撕的最小原语。"""
from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
from torch import Tensor


def softmax(x: Tensor, dim: int = -1) -> Tensor:
    """数值稳定 Softmax：先减 max 再 exp。禁止调用 F.softmax / torch.softmax。"""
    # TODO
    shifted = x-x.max(dim, keepdim = True).values
    E = torch.exp(shifted)
    return E/E.sum(dim, keepdim=True)

    raise NotImplementedError("请实现 softmax")


def scaled_dot_product_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    mask: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor]:
    """返回 (output, attention_weights)。

    必须按课件 / 面试公式逐步写：
        scores = Q K^T / sqrt(d_k)
        对 mask==0 的位置在 softmax 前屏蔽（填 -1e9，不要填 0）
        weights = softmax(scores)   # 调用上面手写的 softmax
        output = weights V
    禁止调用 F.scaled_dot_product_attention。
    """
    # TODO
    d_k = query.size(-1)
    scores = query @ key.transpose(2, 3) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask==0, -1e9)
    weights = softmax(scores, dim=-1)
    outputs = weights @ value
    return outputs, weights
    raise NotImplementedError("请实现 scaled_dot_product_attention")


def subsequent_mask(size: int, device: Optional[torch.device] = None) -> Tensor:
    """因果 mask，形状 (1, 1, size, size)。下三角（含对角）为 1。"""
    # TODO: torch.tril
    return torch.ones(1, 1, size, size, device=device).tril()
    raise NotImplementedError("请实现 subsequent_mask")


def padding_mask(seq: Tensor, pad_idx: int = 0) -> Tensor:
    """seq: (B, T) -> (B, 1, 1, T)，padding 位置为 0，其余为 1。"""
    # TODO
    return (seq != pad_idx).float().view(seq.size(0), 1, 1, seq.size(1))
    raise NotImplementedError("请实现 padding_mask")
