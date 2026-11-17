"""正弦位置编码，对应 Lec8-Supplement2 第 8 页。"""
from __future__ import annotations

import math

import torch
from torch import Tensor


def sinusoidal_positional_encoding(max_len: int, d_model: int) -> Tensor:
    """返回 (max_len, d_model)，不需要梯度。

    PE(pos, 2i)     = sin(pos / 10000^{2i/d_model})
    PE(pos, 2i+1)   = cos(pos / 10000^{2i/d_model})
    """
    # TODO: 实现正弦 / 余弦位置编码
    pe = torch.zeros(max_len, d_model)
    pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)

    div = torch.pow(10000, torch.arange(0, d_model, 2, dtype=torch.float32)/d_model)

    pe[:, 0::2] = torch.sin(pos / div)
    pe[:, 1::2] = torch.cos(pos / div)
    return pe
    raise NotImplementedError("请实现 sinusoidal_positional_encoding")
