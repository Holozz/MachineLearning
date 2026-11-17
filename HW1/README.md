# 第一次作业：Mini-Transformer

先读 `第一次作业说明.md`。独立完成。没有书面题。

建议 Python 3.9–3.12，先确认 `import torch` 成功再开始。数字 `d` 的 token 是 `d+3`（`1` 是 BOS，不是数字 1）。

手写范围对齐面试手撕：Softmax、缩放点积注意力、mask、MHA、LayerNorm、FFN、Encoder、Decoder、整网 forward。

```bash
pip install -r requirements.txt
python tests/run_tests.py
python train.py --epochs 15
```

要补全的文件：

- `src/attention.py`（softmax、注意力、两种 mask）
- `src/positional.py`
- `src/layers.py`（LayerNorm、MHA.forward、FFN.forward、EncoderBlock.forward、DecoderBlock.forward）
- `src/model.py`（TinyTransformer.forward）

`src/data.py`、`train.py`、MHA 的拆头辅助函数已经写好。
