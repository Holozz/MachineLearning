<div class="cover" style="text-align: center; margin: 0; padding: 0;">
  <h1 style="font-size: 22px; margin: 0 0 4px;">《机器学习与数据挖掘》实验报告</h1>
  <h2 style="font-size: 18px; margin: 0 0 4px;">Lecture 1 Mini-Transformer</h2>
  <div class="info">
    <p style="margin: 0;">姓名：韩宇</p>
    <p style="margin: 0;">学号：24319050</p>
    <p class="date" style="font-size: 1em; margin: 0;">日期：2026-10-07</p>
  </div>
</div>

## 实验目的

手写注意力与编码器、解码器的前向计算，在长度为 8 的数字序列上完成反转任务，并保存训练曲线和注意力图

## 实验设置

实现稳定 Softmax、缩放点积注意力、因果及填充 mask、正弦位置编码、多头注意力、LayerNorm、FFN 和整网前向计算，各子层使用 Post-LN

Python 3.12.7，PyTorch 2.10.0+cpu，CPU 核验，matplotlib 3.9.2
编码器、解码器各 2 层，d_model=32，4 个头，d_ff=64，Dropout=0.1，共 43,245 个参数
Adam 学习率 0.003，批大小 64，训练 15 轮，每轮 80 步，梯度裁剪阈值 1.0，训练种子 0，固定验证集 640 条，验证种子 10007

数字 d 的 token 为 d+3，PAD=0、BOS=1、EOS=2，源长 8，解码器输入为 BOS 加反转序列，标签为反转序列加 EOS

## 实验结果

### 1. 损失曲线与准确率

按注意力计算顺序补全模块，以 teacher forcing 训练，验证时从 BOS 开始贪心解码

补全后执行命令：

```bash
python tests/run_tests.py
python train.py --epochs 15 --d_model 32 --n_heads 4 --n_layers 2
```

结果：见 `res.png`

可见第 15 轮两项准确率均达标

执行结果图：

![1791365493269](image/report/1791365493269.png)

横轴 0–14 对应第 1–15 轮，训练与验证损失分别由训练模式和评估模式计算


<div style="page-break-after: always;"></div>

### 2. 注意力可视化

对固定数字序列 0 1 2 3 4 5 6 7 执行前向计算，提取最后一层的注意力权重并对 4 个头取平均

对应代码：

```python
self_w = model.encoders[-1].self_attn.last_weights[0].mean(dim=0)
cross_w = model.decoders[-1].cross_attn.last_weights[0].mean(dim=0)
```

执行结果图：见 `attn_self.png` 和 `attn_cross.png`

自注意力图大小为 8×8，行是源查询位置，列是源键位置；权重分布在主对角线两侧，编码器使用填充 mask，未使用因果 mask；可见重点关注相邻区域

交叉注意力图大小为 9×8，行是目标查询位置，列是源位置；前 8 行最大权重列依次为 7、6、5、4、3、2、1、0，呈反对角排列，呈现对角的关注，第i行关注L-i-1行，是因为目标端的各行要去关注源端的逆位置

第 0 行输入 BOS，最大权重在源位置 7，用于预测数字 7；第 7 行用于预测数字 0，第 8 行用于预测 EOS，其最大权重列为 0，不属于数字反转位置

固定序列的解码结果为数字 7 6 5 4 3 2 1 0，后接 EOS

## 实现中的一个坑

实现时差点直接对 `scaled_dot_product_attention` 返回的 `output` 直接做 Dropout，这样会再次计算 `weights @ v`
实际应该对注意力权重做 Dropout，再与 V 相乘；对 `output `做 Dropout 会丢弃输出特征，两种操作的计算结果不同

```python
_, weights = scaled_dot_product_attention(q, k, v, mask)
self.last_weights = weights

o = self.W_o(self._combine_heads(self.dropout(weights) @ v))
```

## 讨论与工具使用声明

未与同学讨论；
代码编写未使用生成式工具；
使用 Deepseek Harness 阅读实验要求、解释代码和报错
