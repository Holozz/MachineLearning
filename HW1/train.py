#!/usr/bin/env python3
"""训练 TinyTransformer 做数字序列反转，并导出曲线与注意力图。"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor

from src.data import (
    BOS,
    EOS,
    NUM_START,
    SEQ_LEN,
    VOCAB_SIZE,
    Batch,
    example_batch,
    generate_batch,
)
from src.model import TinyTransformer


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


@torch.no_grad()
def greedy_decode(model: TinyTransformer, src: Tensor, max_new: int) -> Tensor:
    model.eval()
    ys = torch.full((src.size(0), 1), BOS, dtype=src.dtype, device=src.device)
    for _ in range(max_new):
        logits = model(src, ys)
        next_tok = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        ys = torch.cat([ys, next_tok], dim=1)
        if torch.all(next_tok.squeeze(-1) == EOS):
            break
    return ys


def sequence_metrics(pred_with_bos: Tensor, src: Tensor) -> tuple[float, float]:
    gold = src.flip(dims=[1])
    pred = pred_with_bos[:, 1 : 1 + src.size(1)]
    if pred.size(1) < gold.size(1):
        pad = torch.full(
            (pred.size(0), gold.size(1) - pred.size(1)),
            EOS,
            device=pred.device,
            dtype=pred.dtype,
        )
        pred = torch.cat([pred, pad], dim=1)
    pred = pred[:, : gold.size(1)]
    token_acc = (pred == gold).float().mean().item()
    exact = (pred == gold).all(dim=1).float().mean().item()
    return token_acc, exact


def batch_loss(model: TinyTransformer, batch: Batch) -> Tensor:
    logits = model(batch.src, batch.tgt_in)
    return F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        batch.tgt_out.reshape(-1),
        ignore_index=0,
    )


@torch.no_grad()
def evaluate(model: TinyTransformer, batches: list[Batch]) -> tuple[float, float, float]:
    model.eval()
    losses, token_accs, exacts = [], [], []
    for batch in batches:
        losses.append(batch_loss(model, batch).item())
        pred = greedy_decode(model, batch.src, max_new=SEQ_LEN + 1)
        t, e = sequence_metrics(pred, batch.src)
        token_accs.append(t)
        exacts.append(e)
    return (
        sum(losses) / len(losses),
        sum(token_accs) / len(token_accs),
        sum(exacts) / len(exacts),
    )


def plot_curves(history: dict, path: Path) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
    ax[0].plot(history["train_loss"], label="train")
    ax[0].plot(history["val_loss"], label="val")
    ax[0].set_title("Cross-entropy loss")
    ax[0].set_xlabel("epoch")
    ax[0].legend()
    ax[1].plot(history["val_token_acc"], label="token acc")
    ax[1].plot(history["val_exact"], label="exact match")
    ax[1].set_title("Validation accuracy")
    ax[1].set_xlabel("epoch")
    ax[1].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_attention(weights: Tensor, path: Path, title: str, xlabel: str, ylabel: str) -> None:
    import matplotlib.pyplot as plt

    w = weights.detach().cpu()
    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    im = ax.imshow(w.numpy(), cmap="viridis", vmin=0, vmax=max(1e-6, float(w.max())))
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_attention_maps(model: TinyTransformer, device: torch.device, out_dir: Path) -> None:
    model.eval()
    batch = example_batch(device=device)
    _ = model(batch.src, batch.tgt_in)
    self_w = model.encoders[-1].self_attn.last_weights
    cross_w = model.decoders[-1].cross_attn.last_weights
    if self_w is None or cross_w is None:
        print("未取到注意力权重：请在 MultiHeadAttention.forward 里设置 self.last_weights")
        return
    plot_attention(
        self_w[0].mean(dim=0),
        out_dir / "attn_self.png",
        title="Encoder self-attention (last layer, head avg)",
        xlabel="key position (source)",
        ylabel="query position (source)",
    )
    # 固定序列 0..7：第 0 行 BOS 应看源端最后一个位置（数字 7）
    plot_attention(
        cross_w[0].mean(dim=0),
        out_dir / "attn_cross.png",
        title="Decoder cross-attention (last layer, head avg)",
        xlabel="key position (source 0..7)",
        ylabel="query position (target, row0=BOS)",
    )
    print("已保存 attn_self.png / attn_cross.png（固定数字 0..7）")


def print_examples(model: TinyTransformer, device: torch.device, n: int = 4) -> None:
    batch = generate_batch(n, device=device)
    pred = greedy_decode(model, batch.src, max_new=SEQ_LEN + 1)
    print("解码样例（数字，不含 BOS/EOS）：")
    for i in range(n):
        src = (batch.src[i] - NUM_START).tolist()
        gold = list(reversed(src))
        pred_d = (pred[i, 1 : 1 + SEQ_LEN] - NUM_START).tolist()
        mark = "OK" if pred_d == gold else "XX"
        print(f"  [{mark}] src {src}  gold {gold}  pred {pred_d}")


def make_val_batches(n_batches: int, batch_size: int, device: torch.device, seed: int) -> list[Batch]:
    """用独立 RNG 生成固定验证集，避免每轮重采样造成曲线抖动。"""
    g = torch.Generator()
    g.manual_seed(seed)
    return [generate_batch(batch_size, device=device, generator=g) for _ in range(n_batches)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--d_model", type=int, default=32)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--n_layers", type=int, default=2)
    parser.add_argument("--d_ff", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--steps_per_epoch", type=int, default=80)
    parser.add_argument("--val_batches", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out_dir", type=str, default=".")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = TinyTransformer(
        vocab_size=VOCAB_SIZE,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        d_ff=args.d_ff,
        dropout=args.dropout,
    ).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=args.lr)
    val_batches = make_val_batches(args.val_batches, args.batch_size, device, args.seed + 10007)

    history = {
        "train_loss": [],
        "val_loss": [],
        "val_token_acc": [],
        "val_exact": [],
    }

    print(f"device={device}  params={sum(p.numel() for p in model.parameters())}")
    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        for _ in range(args.steps_per_epoch):
            batch = generate_batch(args.batch_size, device=device)
            loss = batch_loss(model, batch)
            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            running += loss.item()
        train_loss = running / args.steps_per_epoch
        val_loss, token_acc, exact = evaluate(model, val_batches)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_token_acc"].append(token_acc)
        history["val_exact"].append(exact)
        print(
            f"epoch {epoch:02d}  train {train_loss:.4f}  val {val_loss:.4f}  "
            f"token_acc {token_acc:.3f}  exact {exact:.3f}"
        )

    plot_curves(history, out_dir / "curves.png")
    save_attention_maps(model, device, out_dir)
    torch.save(
        {"model": model.state_dict(), "history": history, "args": vars(args)},
        out_dir / "best.pt",
    )
    print_examples(model, device)
    print("达标线：exact match >= 0.80 或 token_acc >= 0.95")
    print(f"曲线已保存到 {out_dir / 'curves.png'}")


if __name__ == "__main__":
    main()
