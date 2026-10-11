---
title: "CS336 Lecture 3：架构与超参数"
description: "从原始 Transformer 到现代变体：Pre-Norm、RMSNorm、SwiGLU、RoPE 等架构选择，以及各类超参数的经验取值。"
tags:
  - "cs336"
  - "transformer"
  - "architecture"
category: "courses"
series: "CS336"
date: "2026-07-10"
updated: "2026-07-10"
---

> **CS336 — Tatsu H**  


---

## 1. 从原始 Transformer 到现代变体

### 1.1 原始 Transformer
- 位置编码：正弦余弦（sines and cosines）
- FFN：ReLU
- Norm 类型：**Post-Norm** + LayerNorm

### 1.2 简单现代变体
![CS336 作业中实现的现代 Transformer 变体：Pre-Norm、RoPE、SwiGLU，线性层不带 bias。](/notes-assets/cs336/pasted-20260707192956.png)

*CS336 作业中实现的现代 Transformer 变体：Pre-Norm、RoPE、SwiGLU，线性层不带 bias。*

**与现代标准的差异：**
- LayerNorm 放在 block **前面**（Pre-Norm）
- **RoPE**（旋转位置编码）
- FF 层使用 **SwiGLU**，而非 ReLU
- 线性层和 LayerNorm **没有偏置项（bias）**

### 1.3 架构的思考方式
- 2024-2025 年间发布了 **19+ 个新的密集（dense）模型**，大多有微小的架构调整
- 核心问题：
  - 这些模型有什么**共同点**？
  - 哪些部分**有差异**？
  - 我们能从中学到什么？

---

## 2. Pre-Norm vs Post-Norm

### 2.1 基本概念

| 类型            | 位置                          | 说明                           |
| ------------- | --------------------------- | ---------------------------- |
| **Post-Norm** | 每个子组件之后（残差连接之后）进行 LayerNorm | 原始 Transformer、BERT、OPT-350M |
| **Pre-Norm**  | LayerNorm **前置**到子组件之前      | **现代几乎所有 LM 的标准**            |

![Post-Norm（左）与 Pre-Norm（右）的残差结构对比，图源 Xiong et al., 2020。](/notes-assets/cs336/pasted-20260707193451.png)

*Post-Norm（左）与 Pre-Norm（右）的残差结构对比，图源 Xiong et al., 2020。*

### 2.2 为什么 Pre-Norm 胜出？
- 不影响主残差信号路径（residual signal path）
- **梯度衰减更小**
- **减少梯度尖峰（gradient spikes）**【Salazar & Nguyen】
- 原始优势：**移除 warmup** 的需求
- 现代优势：**训练更稳定**，允许更大的学习率

### 2.3 新进展：Double Norm / Non-Residual Post-Norm
- 如果把 LayerNorm 放在残差流中不好 → **为什么不放在残差流外面？**
- 代表模型：**Grok、Gemma 2、OLMo 2**

---

## 3. LayerNorm vs RMSNorm

### 3.1 公式对比

| 方法 | 公式 | 代表模型 |
|------|------|----------|
| **LayerNorm** | $\text{LN}(x) = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} \cdot \gamma + \beta$ | GPT-3/2/1, OPT, GPT-J, BLOOM |
| **RMSNorm** | $\text{RMSNorm}(x) = \frac{x}{\sqrt{\text{mean}(x^2) + \epsilon}} \cdot \gamma$ | LLaMA 系列, PaLM, Chinchilla, T5 |

### 3.2 为什么选择 RMSNorm？
**表面原因**：更快，效果一样好
- 更少的操作（无需计算均值）
- 更少的参数（无需存储 bias）

**深层原因**：**FLOPs ≠ 运行时！**
- 矩阵乘法占 FLOPs 和内存的绝大部分
- RMSNorm 的优势不在于 FLOPs 减少，而在于 **数据搬运（data movement）** 的减少
- 在 FLOP-to-memory 比率上表现更好


---

## 4. 去除偏置项（Bias-Free）

**观察**：大多数现代 Transformer 没有 bias 项。

**原始 Transformer**：
$$y = W_1x + b_1, \quad \text{ReLU}(y), \quad z = W_2y + b_2$$

**现代实现（无 bias）**：
$$y = W_1x, \quad \text{Act}(y), \quad z = W_2y$$

**原因**：
- **内存**：与 RMSNorm 类似的考虑，减少参数搬运
- **优化稳定性**

---

## 5. 激活函数

### 5.1 常见激活函数一览

| 函数 | 公式 | 代表模型 |
|------|------|----------|
| **ReLU** | $\max(0, xW_1)W_2$ | 原始 Transformer, T5, Gopher, Chinchilla, OPT |
| **GeLU** | $\text{GELU}(xW_1)W_2$ | GPT-1/2/3, GPT-J, GPT-NeoX, BLOOM |
| **SwiGLU** | $(\text{Swish}(xW_1) \odot xV)W_2$ | LLaMA 1/2/3, PaLM, Mistral, OlMo, **大多数 2023 后模型** |
| **GeGLU** | $(\text{GELU}(xW_1) \odot xV)W_2$ | T5 v1.1, mT5, LaMDA, Phi3, Gemma 2/3/4 |

### 5.2 Gated Activation（*GLU）

GLU 修改了 FF 层的"第一部分"：

- 原始：$\text{ReLU}(xW_1)W_2$
- GLU：$(\text{Act}(xW_1) \odot xV)W_2$
- 引入了**额外参数 $V$**（门控权重）

**关键细节**：门控模型将 $d_{ff}$ 缩小 **2/3**，以保持 FLOPs 不变。

### 5.3 GLU 有效吗？

**是的，相当一致地有效。**
- [Shazeer 2020]：首次提出并验证
- [Narang et al. 2020]：后续工作佐证
- *GLU 对于工作模型不是必需的（见 GPT-3），但现在已经很少看到非 GLU 的模型了

**例外**：Nemotron 340B 使用 Squared ReLU。

---

## 6. 串行 vs 并行层

### 6.1 标准串行 Transformer
```python
output = FFN(Attention(x))
```

### 6.2 并行层
```python
output = Attention(x) + FFN(x)  # 并行计算
```

- 如果实现得当，**LayerNorm 可以共享**，矩阵乘法可以融合
- 代表模型：**GPT-J, PaLM, GPT-NeoX, Cohere Command A, Falcon 2 11B**

**现状**：大多数模型现在使用**串行层**。

---

## 7. 位置编码

### 7.1 各种位置编码

| 方法 | 公式 | 说明 | 代表模型 |
|------|------|------|----------|
| **Sine Embeddings** | $PE_{(pos,2i)} = \sin(pos/10000^{2i/d})$ | 正弦余弦，可外推 | 原始 Transformer |
| **Absolute Embeddings** | $x_{pos} = x + p_{pos}$ | 将位置向量加到 embedding 上 | GPT-1/2/3, OPT |
| **Relative Embeddings** | 加到 attention 计算中 | $A_{ij}$ 依赖于 $i-j$ | T5, Gopher, Chinchilla |
| **RoPE** | 旋转矩阵乘以 query/key | 相对位置的自然表达 | GPT-J, PaLM, LLaMA, **大多数 2024+ 模型** |

### 7.2 RoPE（Rotary Position Embeddings）

**核心直觉**：
- 我们希望 embedding 对**绝对位置不变**
- 我们知道**内积对任意旋转是不变的**
- 旋转角度可以自然地编码**相对位置**

**数学实现**：
- $R_{\Theta, m}^T R_{\Theta, n} = R_{\Theta, m-n}$（旋转矩阵的差 = 相对位置旋转）
- 将坐标**两两配对**，在 2D 空间旋转（受复数乘法启发）
- 乘以正弦余弦矩阵，而非相加

**实现细节**：
- 在每次 attention 操作中对 query/key 输入进行旋转
- 确保位置不变性

---

## 8. 关键超参数

### 8.1 Feedforward 与 Model Dim 比率

**共识**：$d_{ff} / d_{model} \approx 4$（非 GLU）或 $8/3$（GLU）

| 模型 | $d_{ff}/d_{model}$ |
|------|:------------------:|
| PaLM | 4 |
| Mistral 7B | 3.5 |
| LLaMA-2 70B | 3.5 |
| LLaMA 70B | 2.68 |
| Qwen 14B | 2.67 |
| DeepSeek 67B | 2.68 |
| Yi 34B | 2.85 |
| T5 v1.1 | 2.5 |
| **T5 (11B)** | **64**（例外） |
| Gemma 2 | 8 |
| SmolLM / Gemma 3 / Gemma 4 | 4 |

**经验范围**：1-10 之间有个最优盆地 [Kaplan+ 2020]
- 默认的 $4$（非 GLU）和 $2.66$（GLU）对几乎所有现代 LLM 都适用
- T5 的 64 倍是个例外，T5 v1.1 已回退到 2.5

### 8.2 注意力头数与头维度

**共识**：`num_heads × head_dim = d_model`（大多数模型遵循）

| 模型 | Num Heads | Head Dim | Model Dim | Ratio |
|------|:---------:|:--------:|:---------:|:-----:|
| GPT-3 | 96 | 128 | 12288 | 1 |
| T5 | 128 | 128 | 1024 | 16 |
| T5 v1.1 | 64 | 64 | 4096 | 1 |
| LaMDA | 128 | 128 | 8192 | 2 |
| PaLM | 48 | 258 | 18432 | 1.48 |
| LLaMA-2 | 64 | 128 | 8192 | 1 |
| Qwen 3.5 (27B) | 24 | 256 | 5120 | 1.2 |

**权衡**：
- 头太多 → 每个头维度过小，难以捕捉复杂模式
- 头太少 → 牺牲并行度与多视角交互
- head_dim 一般在 **64–128** 之间

### 8.3 宽深比（Aspect Ratio）：模型该深还是宽？

| 模型 | $d_{model} / N_{layers}$ |
|------|:------------------------:|
| BLOOM | 205 |
| T5 v1.1 | 171 |
| PaLM (540B) | 156 |
| GPT-3 / OPT / Mistral / Qwen | 128 |
| LLaMA / LLaMA-2 | 102 |
| Gemma 3 | 87 |
| Gemma 4 | 61 |
| T5 (11B) | 33 |

**考虑因素**：
- 极深的模型更难并行化，延迟更高
- 有一个广泛的有效范围（100-200）
- **系统考量**决定具体取值

### 8.4 词表大小

| 类型 | 大小 | 代表模型 |
|------|:----:|----------|
| **单语言** | 30k–50k | 原始 Transformer (40k), GPT-2/3 (50k), LLaMA (32k), T5 (32k) |
| **多语言 / 生产系统** | 100k–250k | mT5 (250k), PaLM (256k), GPT-4 (100k), DeepSeek (100k), Qwen (152k), Gemma 4 (262k) |

---

## 9. 正则化

### 9.1 预训练需要正则化吗？

**反对理由**：
- 数据量巨大（万亿 token），超过参数量
- SGD 在语料上只做一个 epoch（很难记忆）

### 9.2 实践中的 Dropout 和 Weight Decay

| 模型 | Dropout | Weight Decay |
|------|:-------:|:------------:|
| 原始 Transformer | 0.1 | 0 |
| GPT-2 | 0.1 | 0.1 |
| T5 | 0.1 | 0 |
| GPT-3 | 0.1 | 0.1 |
| T5 v1.1 | 0 | 0 |
| PaLM | 0 | (variable) |
| OPT | 0.1 | 0.1 |
| LLaMA | 0 | 0.1 |
| Qwen 14B | 0.1 | 0.1 |

**趋势**：较新的模型（除 Qwen 外）在预训练时**不再使用 dropout**，仅依赖 weight decay。

### 9.3 为什么 LLM 要用 Weight Decay？
- **不是为了控制过拟合**
- Weight decay 与学习率（cosine schedule）产生交互，影响**优化动力学**[Andriushchenko et al. 2023]

---

## 10. 稳定性技巧

### 10.1 Softmax 的危险
Softmax 因为指数运算和除以零，可能表现不稳定。

### 10.2 Z-Loss（输出 Softmax 稳定性）
- 核心思想：在 softmax 计算中加入 $\log(\sum e^{x_i})$ 的正则项
- 来源：PaLM
- 其他使用模型：Baichuan 2 (2023), DCLM (2024), OLMo 2 (2025), OLMo 3 (2025)

### 10.3 QK-Norm（Attention Softmax 稳定性）
- 在进入 softmax 之前，对 query 和 key 进行 Layer/RMS Norm
- 来源：视觉和多模态模型 [Dehghani 2023, Idefics, Chameleon]
- 使用模型：DCLM, OLMo 2, Gemma 2, Qwen 3, OLMo 3, Gemma 4

### 10.4 Logit Soft-Capping
- 通过 Tanh 将 logits 软裁剪到某个最大值
- 防止 logits 爆炸，但可能有性能影响

---

## 11. 注意力机制变体

### 11.1 GQA / MQA

**动机**：在自回归生成中，需要逐步更新 KV Cache。推理时的算术强度不理想：
$$\text{Arithmetic Intensity} \approx \frac{1}{1 + \frac{n}{d} + \frac{1}{h-1}}$$

- MHA 的瓶颈：每次生成需要搬运很多 KV 数据
- MQA：多个 query 头共享**一组 key/value**
- GQA：介于 MHA 和 MQA 之间，**分组共享**

| 方法 | KV Cache 大小 | 质量影响 |
|------|:-------------:|:---------:|
| MHA | $h \times k \times n$ | 标准 |
| GQA | $g \times k \times n$（$g < h$） | **几乎无损**【Ainslie 2023】 |
| MQA | $1 \times k \times n$ | **轻微 PPL 损失**【Shazeer 2019】 |

### 11.2 稀疏 / 滑动窗口注意力（Sparse / Sliding Window Attention）
- 处理完整上下文是 $O(n^2)$ 的
- 构建稀疏/结构化的注意力模式，在表达力和运行时之间做权衡
- **当前标准技巧**：交错使用 full attention 和 local attention
  - Cohere Command A：**每 4 层做一次 full attention**
  - Long-range 信息通过 NoPE，short-range 通过 RoPE + SWA
  - 其他模型：LLaMA 4, Gemma 3, Gemma 4, OLMo 3

---

## 12. Architectures 总结

![近年主流大模型的架构选择汇总：归一化、并行层、位置编码、激活函数与 MLP 比例。](/notes-assets/cs336/pasted-20260707193154.png)

*近年主流大模型的架构选择汇总：归一化、并行层、位置编码、激活函数与 MLP 比例。*

**主要共同点：**
- Pre-Norm（非残差 norm）
- RMSNorm（有明确的计算优势）
- GLU 门控激活（现在是共识）
- 串行层（大多数模型）

**主要差异：**
- 位置编码（RoPE 是主流）
- 激活函数（SwiGLU/GeGLU）
- 分词器（tokenizer）
