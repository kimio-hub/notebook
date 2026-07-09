---
title: "MVFT-JI"
tags:
  - cir
  - multimodal
---

# MLLM-Guided VLM Fine-Tuning with Joint Inference for Zero-Shot Composed Image Retrieval

> 论文：MLLM-Guided VLM Fine-Tuning with Joint Inference for Zero-Shot Composed Image Retrieval  
> arXiv:2505.19707  
> 方法名：MVFT-JI  
> 任务：Zero-Shot Composed Image Retrieval, ZS-CIR  
> 核心关键词：MLLM-generated data, VLM fine-tuning, target text retrieval, text-to-image retrieval, joint inference

## 摘要

MVFT-JI 关注 zero-shot CIR 中 adapter/pseudo-token 方法的不足：这些方法通常只把 reference image 映射成 text encoder 可接受的伪文本 token，但没有直接优化 composed query representation 与目标语义的对齐，导致复杂或细粒度修改下表现受限。论文提出用 MLLM 从无标注图像中构造两类训练任务：一类是 **target text retrieval**，让 composed query 对齐 MLLM 生成的目标文本；另一类是 **text-to-image retrieval**，让 MLLM 生成的 caption 对齐对应图像。通过联合训练，VLM 在没有人工 CIR triplets 的情况下学到组合检索能力。推理时，模型同时计算 composed query 与候选图像的相似度，以及 MLLM 生成 target text 与候选图像的相似度，两者融合得到最终排序。

## 1. 问题背景

已有 ZS-CIR 方法常见路线：

```text
reference image -> pseudo-text tokens
pseudo-text tokens + modification text -> frozen VLM/LLM
```

问题是：

- adapter 主要学习让 token 兼容 text encoder；
- pseudo-token 不一定保留丰富视觉语义；
- 训练目标没有直接约束 composed query 和 target semantics；
- 复杂修改时，query representation 可能不够准确。

MVFT-JI 的核心想法是：**不要只训练 adapter 生成伪文本，而要直接 fine-tune VLM，让它学会组合查询的目标语义。**

## 2. 理论动机

对一个 composed query `(x_i, m_i)`，目标是检索 target image `x_i^t`：

```math
x_i^t = \arg\max_{x_c \in C} P(x_c | x_i, m_i; \theta)
```

论文假设存在一个 latent textual description `t_i`，它完整表达了 composed query 的目标语义。

于是：

```math
P(x_i^t | x_i, m_i; \theta)
=
\sum_{t_i} P(x_i^t | t_i; \theta) P(t_i | x_i, m_i; \theta)
```

如果 `t_i` 已经完整表达目标语义，则：

```math
P(x_i^t | x_i, m_i, t_i; \theta)
=
P(x_i^t | t_i; \theta)
```

所以优化 CIR 可以分解成两个任务：

1. **Target Text Retrieval**：学习 `P(t_i | x_i, m_i; theta)`  
   composed query 应该能找到目标文本。

2. **Text-to-Image Retrieval**：学习 `P(x_i^t | t_i; theta)`  
   目标文本应该能找到目标图像。

这就是 MVFT-JI 双任务训练的理论依据。

## 3. 训练数据构造

MVFT-JI 只使用无标注图像，通过 MLLM 自动生成训练监督。

### 3.1 Target Text Retrieval Data

给定图像 `x_i`，先让 MLLM 生成一个 modification text `m_i`，再让 MLLM 根据 `(x_i, m_i)` 生成修改后的 target text `t_i`：

```math
t_i = MLLM(x_i, m_i, P_{tt})
```

得到训练样本：

```text
(x_i, m_i, t_i)
```

训练目标：让 VLM 的 composed query representation 对齐 `t_i`。

### 3.2 Text-to-Image Retrieval Data

对每张图像 `x_i`，让 MLLM 生成详细 caption `c_i`：

```math
c_i = MLLM(x_i, P_c)
```

得到训练样本：

```text
(c_i, x_i)
```

训练目标：让文本 caption 和对应图像在 VLM 空间中对齐。

## 4. VLM Fine-Tuning

模型采用 Q-Former-based VLM，初始化自 BLIP-2。训练中使用共享 learnable query tokens 提取三类表示：

- composed query `(x_i, m_i)` 的 token features；
- target text `t_i` 的 token features；
- image caption `c_i` 和 image `x_i` 的 token features。

相似度不是简单单向量 cosine，而是 token-level matching：

```math
\hat{s}_{ij}
=
\frac{1}{k}
\sum_{z=1}^{k}
\max_r
\frac{(f^{Q_i}_z)^T f^{T_j}_r}
{\|f^{Q_i}_z\|_2 \|f^{T_j}_r\|_2}
```

caption-image 相似度同理：

```math
\tilde{s}_{ij}
=
\frac{1}{k}
\sum_{z=1}^{k}
\max_r
\frac{(f^{C_i}_z)^T f^{I_j}_r}
{\|f^{C_i}_z\|_2 \|f^{I_j}_r\|_2}
```

训练损失是两个 InfoNCE：

```math
L = L_t + L_c
```

其中：

- `L_t`：composed query -> target text；
- `L_c`：caption -> image。

## 5. Joint Inference

推理时给定 query `(x_q, m_q)`：

1. 用 fine-tuned VLM 得到 composed query features；
2. 与候选图像计算相似度 `hat{s}_{qj}`；
3. 同时让 MLLM 生成 imagined target text `t_q`；
4. 用 VLM 编码 `t_q`，与候选图像计算相似度 `tilde{s}_{qj}`；
5. 融合两种相似度：

```math
s_{qj}
=
\frac{1}{2}(\hat{s}_{qj} + \tilde{s}_{qj})
```

直觉：

- `hat{s}` 体现 fine-tuned VLM 的组合表示能力；
- `tilde{s}` 体现 MLLM 生成目标文本的语义推理能力；
- 融合后更稳。

## 6. 实验设置

训练数据：

- ImageNet1K subset，约 10K 无标注图像；
- MLLM：MiniCPM-V-2_6；
- VLM：BLIP-2 ViT-L/14 Q-Former；
- 优化器：AdamW；
- batch size：128；
- 单张 NVIDIA A100 40GB。

评估数据：

- FashionIQ；
- CIRR；
- CIRCO。

## 7. 实验结果

### 7.1 FashionIQ

| 方法 | Avg. R@10 | Avg. R@50 |
| --- | ---: | ---: |
| FTI4CIR | 29.39 | 50.88 |
| MLLM-I2W | 30.30 | 50.10 |
| MVFT-JI | 34.84 | 56.29 |

MVFT-JI 在三类服装上都有提升，尤其 R@10 提升明显。

### 7.2 CIRR

| 方法 | R@1 | R@5 | Avg |
| --- | ---: | ---: | ---: |
| MCL | 26.22 | 56.84 | 59.15 |
| MLLM-I2W | 28.30 | 57.90 | - |
| MVFT-JI | 39.30 | 69.49 | 68.71 |

论文强调 MVFT-JI 相比 MLLM-I2W 在 CIRR R@1 上有约 11% 绝对提升。

### 7.3 CIRCO

| 方法 | mAP@5 | mAP@10 |
| --- | ---: | ---: |
| CIReVL | 18.57 | 19.01 |
| MCL | 17.67 | 18.86 |
| MVFT-JI | 21.69 | 22.99 |

CIRCO 允许多个正确目标，mAP 更能体现排序质量。

## 8. 消融实验

| 变体 | FashionIQ R@10 | CIRCO mAP@5 | CIRR R@1 |
| --- | ---: | ---: | ---: |
| w/o generated text similarity `tilde{s}` | 32.52 | 17.53 | 30.34 |
| w/o VLM similarity `hat{s}` | 26.66 | 15.21 | 33.11 |
| w/o text-to-image loss `L_c` | 32.75 | 22.55 | 36.15 |
| w/o target-text loss `L_t` | 29.83 | 15.18 | 34.31 |
| InternVL2.5 data generation | 31.88 | 18.18 | 37.47 |
| MVFT-JI | 34.84 | 21.69 | 39.30 |

结论：

- joint inference 很重要，只用任一相似度都会下降；
- `L_t` 对 composed query understanding 更关键；
- `L_c` 对图文对齐也有帮助，但在 CIRCO 上有轻微数据分布冲突；
- MLLM 生成数据质量影响训练效果。

## 9. 我的理解

MVFT-JI 和 CoLLM 都不是纯 training-free 方法，而是用无标注图像合成监督来训练模型。但二者的侧重点不同：

- CoLLM 强调从 image-caption pairs 动态合成 triplets，并让 LLM 产出 composed embedding；
- MVFT-JI 强调用 MLLM 构造两个可解释的训练任务，把 CIR 分解成 composed query -> target text 和 text -> image。

我觉得 MVFT-JI 的核心价值在于：它把「MLLM 生成的目标文本」从一个推理时技巧，变成了训练目标的一部分。模型不只是临时生成 caption 来搜图，而是在训练阶段学会让 composed query 对齐目标语义文本。

## 10. 评审

### 优点

- **理论分解清晰。** 用 latent target text 把 CIR 概率分解成两个检索任务，解释了为什么双任务训练有效。
- **数据成本低。** 只用 10K 无标注 ImageNet 图像就能生成训练数据。
- **推理融合有效。** VLM composed feature 和 MLLM target text 是互补信号。
- **实验提升很大。** 尤其 CIRR R@1 提升明显。

### 局限

- **仍依赖 MLLM 生成质量。** 修改文本、目标文本、caption 任何一步噪声都会进入训练。
- **生成的 target text 不等于真实 target image。** 它是语义代理，可能缺少细粒度视觉差异。
- **需要 fine-tuning。** 相比 OSrCIR/CoRR 这类推理时方法，部署成本更高。
- **joint score 权重简单。** 论文用 1/2 平均，未来可探索自适应融合。

## 11. 一句话总结

MVFT-JI 的核心是：用 MLLM 从无标注图像中构造 target-text 与 image-text 两类监督，直接 fine-tune VLM 学会组合语义，并在推理时融合 VLM 表示和 MLLM 目标文本两种检索信号。

