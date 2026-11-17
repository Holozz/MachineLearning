"""合成的固定长度数字序列反转数据。无需下载。

词表：PAD=0, BOS=1, EOS=2；数字 d=0..9 的 token 为 d+3。
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

PAD, BOS, EOS = 0, 1, 2
NUM_START = 3
VOCAB_SIZE = 13  # 0..12
SEQ_LEN = 8


@dataclass
class Batch:
    src: Tensor  # (B, L)
    tgt_in: Tensor  # (B, L+1)  BOS + reversed
    tgt_out: Tensor  # (B, L+1)  reversed + EOS


def generate_batch(
    batch_size: int,
    seq_len: int = SEQ_LEN,
    device: torch.device | str = "cpu",
    generator: torch.Generator | None = None,
) -> Batch:
    device = torch.device(device)
    if generator is None:
        digits = torch.randint(0, 10, (batch_size, seq_len), device=device)
    else:
        digits = torch.randint(0, 10, (batch_size, seq_len), generator=generator).to(device)
    src = digits + NUM_START
    reversed_src = src.flip(dims=[1])
    bos = torch.full((batch_size, 1), BOS, device=device, dtype=src.dtype)
    eos = torch.full((batch_size, 1), EOS, device=device, dtype=src.dtype)
    tgt = torch.cat([bos, reversed_src, eos], dim=1)
    return Batch(src=src, tgt_in=tgt[:, :-1], tgt_out=tgt[:, 1:])


def example_batch(device: torch.device | str = "cpu") -> Batch:
    """固定数字 0..SEQ_LEN-1，用于画注意力图。"""
    device = torch.device(device)
    digits = torch.arange(SEQ_LEN, device=device).unsqueeze(0)
    src = digits + NUM_START
    reversed_src = src.flip(dims=[1])
    bos = torch.full((1, 1), BOS, device=device, dtype=src.dtype)
    eos = torch.full((1, 1), EOS, device=device, dtype=src.dtype)
    tgt = torch.cat([bos, reversed_src, eos], dim=1)
    return Batch(src=src, tgt_in=tgt[:, :-1], tgt_out=tgt[:, 1:])


def tokens_to_digits(tokens: Tensor) -> Tensor:
    return tokens - NUM_START
