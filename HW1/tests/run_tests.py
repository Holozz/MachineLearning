"""第一次作业单测。在「学生发布包」目录下运行: python tests/run_tests.py"""
from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.attention import padding_mask, scaled_dot_product_attention, softmax, subsequent_mask
from src.layers import DecoderBlock, EncoderBlock, LayerNorm, MultiHeadAttention, PositionwiseFFN
from src.positional import sinusoidal_positional_encoding


class AttentionTests(unittest.TestCase):
    def test_softmax_stable(self):
        x = torch.tensor([[1000.0, 1000.0, 1000.0], [1.0, 2.0, 3.0]])
        y = softmax(x, dim=-1)
        self.assertTrue(torch.isfinite(y).all())
        self.assertTrue(torch.allclose(y.sum(dim=-1), torch.ones(2), atol=1e-5))
        self.assertTrue(torch.allclose(y[0], torch.full((3,), 1.0 / 3.0), atol=1e-5))

    def test_attention_weights_sum_to_one(self):
        torch.manual_seed(0)
        q = torch.randn(2, 3, 5, 8)
        k = torch.randn(2, 3, 6, 8)
        v = torch.randn(2, 3, 6, 4)
        out, w = scaled_dot_product_attention(q, k, v)
        self.assertEqual(tuple(out.shape), (2, 3, 5, 4))
        self.assertEqual(tuple(w.shape), (2, 3, 5, 6))
        self.assertTrue(torch.allclose(w.sum(dim=-1), torch.ones(2, 3, 5), atol=1e-5))

    def test_attention_numerical(self):
        q = torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]])
        k = q.clone()
        v = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]])
        out, w = scaled_dot_product_attention(q, k, v)
        inv = 1.0 / math.sqrt(2.0)
        e0, e1 = math.exp(inv), math.exp(0.0)
        p00, p01 = e0 / (e0 + e1), e1 / (e0 + e1)
        e2, e3 = math.exp(0.0), math.exp(inv)
        p10, p11 = e2 / (e2 + e3), e3 / (e2 + e3)
        expected_w = torch.tensor([[[[p00, p01], [p10, p11]]]])
        expected_out = expected_w @ v
        self.assertTrue(torch.allclose(w, expected_w, atol=1e-5))
        self.assertTrue(torch.allclose(out, expected_out, atol=1e-5))

    def test_causal_mask_upper_zero(self):
        m = subsequent_mask(4)
        self.assertEqual(tuple(m.shape), (1, 1, 4, 4))
        lower = torch.tril(torch.ones(4, 4))
        self.assertTrue(torch.allclose(m[0, 0].float(), lower))
        q = k = v = torch.randn(1, 1, 4, 5)
        _, w = scaled_dot_product_attention(q, k, v, mask=m)
        upper = torch.triu(torch.ones(4, 4), diagonal=1).bool()
        self.assertTrue(torch.all(w[0, 0][upper] < 1e-6))

    def test_padding_mask(self):
        seq = torch.tensor([[3, 4, 5, 0, 0], [7, 0, 0, 0, 0]])
        m = padding_mask(seq, pad_idx=0)
        self.assertEqual(tuple(m.shape), (2, 1, 1, 5))
        self.assertTrue(torch.allclose(m[0, 0, 0].float(), torch.tensor([1.0, 1.0, 1.0, 0.0, 0.0])))
        self.assertTrue(torch.allclose(m[1, 0, 0].float(), torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0])))

    def test_masked_fill_neg_inf(self):
        q = torch.ones(1, 1, 2, 2)
        k = torch.ones(1, 1, 2, 2)
        v = torch.tensor([[[[1.0, 0.0], [100.0, 100.0]]]])
        mask = torch.tensor([[[[1.0, 0.0], [1.0, 0.0]]]])
        out, w = scaled_dot_product_attention(q, k, v, mask=mask)
        self.assertTrue(torch.all(w[..., 1] < 1e-6))
        self.assertTrue(torch.allclose(out, v[:, :, :1, :].expand_as(out), atol=1e-5))


class PositionalTests(unittest.TestCase):
    def test_positional_sin_cos(self):
        pe = sinusoidal_positional_encoding(16, 8)
        self.assertEqual(tuple(pe.shape), (16, 8))
        pos, i = 5, 1
        div = 10000 ** (2 * i / 8)
        self.assertAlmostEqual(pe[pos, 2 * i].item(), math.sin(pos / div), places=5)
        self.assertAlmostEqual(pe[pos, 2 * i + 1].item(), math.cos(pos / div), places=5)
        self.assertFalse(pe.requires_grad)


class MHATests(unittest.TestCase):
    def test_mha_shape(self):
        mha = MultiHeadAttention(d_model=32, n_heads=4)
        x = torch.randn(2, 7, 32)
        y = torch.randn(2, 9, 32)
        out = mha(x, y, y)
        self.assertEqual(tuple(out.shape), (2, 7, 32))
        self.assertIsNotNone(mha.last_weights)
        self.assertEqual(tuple(mha.last_weights.shape), (2, 4, 7, 9))

    def test_mha_causal(self):
        mha = MultiHeadAttention(d_model=16, n_heads=4)
        x = torch.randn(2, 6, 16)
        mask = subsequent_mask(6)
        _ = mha(x, x, x, mask=mask)
        w = mha.last_weights
        upper = torch.triu(torch.ones(6, 6), diagonal=1).bool()
        self.assertTrue(torch.all(w[:, :, upper] < 1e-5))


class BlockTests(unittest.TestCase):
    def test_layer_norm(self):
        ln = LayerNorm(8)
        x = torch.linspace(-3, 3, 80).view(2, 5, 8)
        y = ln(x)
        self.assertEqual(tuple(y.shape), (2, 5, 8))
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, unbiased=False, keepdim=True)
        expected = (x - mean) / torch.sqrt(var + ln.eps)
        self.assertTrue(torch.allclose(y, expected, atol=1e-5))
        self.assertTrue(torch.allclose(y.mean(dim=-1), torch.zeros(2, 5), atol=1e-4))

    def test_ffn_and_encoder_shapes(self):
        ffn = PositionwiseFFN(32, 64, dropout=0.0)
        x = torch.randn(2, 8, 32)
        self.assertEqual(tuple(ffn(x).shape), (2, 8, 32))
        enc = EncoderBlock(32, 4, 64, dropout=0.0)
        y = enc(x)
        self.assertEqual(tuple(y.shape), (2, 8, 32))
        self.assertFalse(torch.allclose(y, x, atol=1e-5))
        dec = DecoderBlock(32, 4, 64, dropout=0.0)
        mem = torch.randn(2, 9, 32)
        z = dec(x, mem)
        self.assertEqual(tuple(z.shape), (2, 8, 32))


class ModelShapeTests(unittest.TestCase):
    def test_encoder_decoder_shapes(self):
        from src.data import VOCAB_SIZE, generate_batch
        from src.model import TinyTransformer

        model = TinyTransformer(vocab_size=VOCAB_SIZE, d_model=32, n_heads=4, n_layers=2)
        model.eval()
        batch = generate_batch(3)
        with torch.no_grad():
            logits = model(batch.src, batch.tgt_in)
        self.assertEqual(tuple(logits.shape), (3, batch.tgt_in.size(1), VOCAB_SIZE))
        src2 = batch.src.flip(dims=[1])
        with torch.no_grad():
            logits2 = model(src2, batch.tgt_in)
        self.assertFalse(torch.allclose(logits, logits2, atol=1e-5))

        tgt_changed = batch.tgt_in.clone()
        tgt_changed[:, -1] = (tgt_changed[:, -1] + 3) % VOCAB_SIZE
        with torch.no_grad():
            logits3 = model(batch.src, tgt_changed)
        self.assertTrue(
            torch.allclose(logits[:, :-1], logits3[:, :-1], atol=1e-5),
            "改 tgt 最后一位后，前面位置的 logits 不应变：请给 Decoder 自注意力加因果 mask",
        )


def main() -> None:
    loader = unittest.defaultTestLoader
    suite = loader.loadTestsFromModule(sys.modules[__name__])
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    passed = result.testsRun - len(result.failures) - len(result.errors) - len(result.skipped)
    print(
        f"\n通过 {passed}/{result.testsRun}  "
        f"(失败 {len(result.failures)}，错误 {len(result.errors)}，跳过 {len(result.skipped)})"
    )
    print("评分：12 个测试各 0.5 分，计入模块部分 6 分。不要依赖跳过。")
    sys.exit(0 if result.wasSuccessful() and not result.skipped else 1)


if __name__ == "__main__":
    main()
