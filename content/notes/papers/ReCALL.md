---
title: "ReCALL"
tags:
  - cir
  - multimodal
---

# ReCALL: Recalibrating Capability Degradation for MLLM-based Composed Image Retrieval

> 论文：Tianyu Yang et al., **ReCALL: Recalibrating Capability Degradation for MLLM-based Composed Image Retrieval**, arXiv:2602.01639v2, Accepted to CVPR 2026  
> 代码：https://github.com/RemRico/Recall  
> 任务：Composed Image Retrieval, CIR，即给定「参考图像 + 修改文本」，从图库中检索目标图像。

## 1. 核心问题

传统 dual-tower VLM 检索器在 CIR 上的问题是：图像和文本交互较浅，难以做细粒度组合推理。后来一些方法把生成式 MLLM 改造成检索器，用一个 embedding 表示「参考图像 + 修改文本」的组合查询，再和候选图像 embedding 做相似度匹配。

这篇文章指出：这种改造会带来 **Capability Degradation**。

直观理解：

- 原始 MLLM `F` 本来擅长逐步推理，比如观察参考图、理解修改文本、逐项比较候选图。
- 但把它压缩成单向量判别式检索器后，训练目标变成「让 query embedding 靠近 target embedding」。
- 这种范式转换会削弱原本的细粒度视觉-语义推理能力，尤其是属性、关系、朝向、局部细节等。

论文的关键证据：在 foundation MLLM `F` 原本能够通过 VQA 推理解决的 1k 子集上，检索微调后的 baseline `R_base` 的 R@1 只有：

| 数据集 | `F` 可解子集 | `R_base` R@1 |
| --- | ---: | ---: |
| CIRR | 100% | 62.33% |
| FashionIQ | 100% | 55.80% |

这说明问题不是 MLLM 没有知识，而是检索式适配后，原本的推理能力没有被很好地映射到 embedding 空间里。

## 2. 方法总览

ReCALL 的主线是：

```text
baseline adaptation -> diagnose -> generate -> refine
```

也可以理解为：

1. 先把 foundation MLLM `F` 微调成一个基础检索器 `R_base`。
2. 用 `R_base` 自己找出最容易混淆的失败样本。
3. 让原始 foundation MLLM `F` 对这些失败样本生成纠偏文本，构造新的 corrective triplets。
4. 用原始 triplets + corrective triplets 继续训练，得到 `R_refine`。

## 3. 符号定义

| 符号 | 含义 |
| --- | --- |
| `I_r` | reference image，参考图像 |
| `T` | modification text，修改文本 |
| `I_t` | target image，真实目标图像 |
| `I_h` | informative instance，被 `R_base` 错排在 ground truth 前面的高迷惑性图像 |
| `T_h` | 为 `I_h` 生成的纠偏修改文本 |
| `F` | foundation MLLM，原始多模态大模型 |
| `R_base` | 从 `F` 适配得到的基础检索器 |
| `R_refine` | 经过 ReCALL 校准后的最终检索器 |

原始训练样本是：

```text
(I_r, T, I_t)
```

ReCALL 生成的新样本是：

```text
(I_r, T_h, I_h)
```

其中 `T_h` 是在原始 `T` 的基础上做最小修改，让 `I_h` 也成为一个语义合法的目标。

## 4. Stage 1: Baseline Retrieval Model Adaptation

第一阶段从 foundation model `F` 初始化，使用 CIR triplets 训练基础检索器 `R_base`。

训练目标是标准 InfoNCE：

```math
L_{\text{InfoNCE}}
=
-\log
\frac{
  \exp(s(z_q, z_t^+) / \tau)
}{
  \sum_{z_t \in B} \exp(s(z_q, z_t) / \tau)
}
```

其中：

- `z_q`：组合查询 `(I_r, T)` 的表示；
- `z_t^+`：真实目标图像 `I_t` 的表示；
- `B`：batch 内所有候选图像表示；
- `s(u, v)`：余弦相似度；
- `tau`：温度系数。

这个阶段能得到一个可用的检索模型，但也会触发论文所说的 capability degradation。

## 5. Stage 2: Diagnose - Self-Guided Informative Instance Mining

第二阶段的目标是找出 `R_base` 的「认知盲点」。

流程：

1. 用 `R_base` 在训练集上执行检索。
2. 如果 ground truth `I_t` 已经排在第 1 位，说明这个样本当前不难，跳过。
3. 对失败样本，取排在 `I_t` 前面的 top-K 错误图像，作为 informative instances，记作 `{I_h}`。

这些 `I_h` 不是普通负样本，而是 hard negatives：

- 它们和目标图像很像；
- 它们足以骗过 `R_base`；
- 它们暴露了 `R_base` 对细节、属性、关系的判别边界不够清楚。

这一步很重要，因为 ReCALL 不是随机造数据，而是先定位模型真实失败处，再把生成预算集中用在这些失败处。

## 6. Stage 3: Generate - Generative Calibration

第三阶段利用 foundation MLLM `F` 的原生推理能力，给每个 informative instance 生成纠偏监督。

对每个失败样本：

```text
原始 triplet: (I_r, T, I_t)
迷惑图像: I_h
目标: 构造新 triplet (I_r, T_h, I_h)
```

关键思想：`I_h` 虽然不是原始 query 的正确答案，但它通常只和 `I_t` 差一些细节。因此可以把原始修改文本 `T` 做最小修改，得到 `T_h`，让 `I_h` 成为新 query 的正确答案。

生成过程分两步：

1. **Intent Decomposition & Verification**  
   让 `F` 把原始修改文本 `T` 拆成多个 atomic intents，然后检查这些 intent 在 `(I_r, I_h)` 上是否成立。

2. **Minimal Edit Synthesis**  
   保留在 `I_h` 上成立的 intent；对不成立的部分做最小文本修改，生成新的 `T_h`。

为了降低 LLM 生成噪声，论文还加入了 **VQA-Assisted Quality Control**：

- 对 `T_h` 中的关键属性生成 VQA 问题；
- 用 `F` 检查这些属性在图像中是否一致；
- 只保留高置信且内部一致的 corrective triplets。

## 7. Stage 4: Refine - Targeted Refinement

第四阶段从 `R_base` 继续训练，得到 `R_refine`。

训练数据包含：

- 原始 triplet：`(I_r, T, I_t)`
- 纠偏 triplet：`(I_r, T_h, I_h)`

ReCALL 构造 micro-group，把原始正样本和对应纠偏样本放在同一组里：

```text
[
  (I_r, T,   I_t),
  (I_r, T_h, I_h)
]
```

这样模型在一个 batch 内同时看到：

- 原始 query 应该靠近 `I_t`，远离 `I_h`；
- 纠偏 query 应该靠近 `I_h`；
- `T` 和 `T_h` 的差别通常很小，所以模型必须学习细粒度语义差别。

最终损失由两部分组成：

```math
L_{\text{total}}
=
L_{\text{InfoNCE}}
+
\lambda L_{\text{triplet}}
```

其中 triplet margin loss 用于显式拉开原始目标和 informative instance：

```math
L_{\text{triplet}}
=
\max(0,\ s(z_q, z_h) - s(z_q, z_t^+) + m)
```

含义是：如果 query 对 hard negative `I_h` 的相似度太高，甚至接近或超过真实目标 `I_t`，就施加惩罚。

## 8. 实验结果

### 8.1 主结果

CIRR test set：

| 模型 | R@1 | Avg. |
| --- | ---: | ---: |
| `R_base` | 51.23 | 79.86 |
| ReCALL `R_refine` | 55.52 | 82.81 |
| 相对提升 | +8.38% | +3.70% |

FashionIQ validation set：

| 模型 | Avg. R@10 | Avg. R@50 |
| --- | ---: | ---: |
| `R_base` | 53.23 | 74.37 |
| ReCALL `R_refine` | 57.04 | 76.42 |
| 相对提升 | +7.16% | +2.76% |

论文还在 CIRR 和 FashionIQ 上超过了多种已有方法，包括 CIR-LVLM、TME、QuRe 等，说明 ReCALL 不只是修补 baseline，也带来了 SOTA 级别的效果。

### 8.2 消融实验

**Self-Guided Mining vs Random Mining**

| 方法 | FashionIQ Avg. R@10 |
| --- | ---: |
| `R_base` | 53.23 |
| Random Mining | 53.80 |
| Self-Guided Mining | 57.04 |

结论：随机抽 hard/near negative 造数据收益很小，说明「诊断失败样本」比「盲目扩数据」更关键。

**核心组件消融**

| 组件 | Avg. R@10 | 说明 |
| --- | ---: | --- |
| baseline | 53.23 | 只做基础检索微调 |
| + CoT-assisted Generation | 55.41 | 生成纠偏文本有效 |
| + VQA Quality Control | 56.13 | 过滤噪声继续提升 |
| + Grouped Refinement | 57.04 | micro-group 对比进一步强化细粒度边界 |

## 9. 我的理解修正


1. **ReCALL 不是单纯的数据增强。**  
   它不是随便生成更多 triplets，而是先用 `R_base` 找失败样本，再只针对这些失败点生成 corrective triplets。

2. **`I_h` 不是普通负样本。**  
   它是被 baseline 错排在 ground truth 前面的迷惑样本，代表模型当前最需要修正的边界。

3. **`T_h` 的作用不是解释错误，而是把错误样本转成合法正样本。**  
   原始 query `(I_r, T)` 下，`I_h` 是 hard negative；但在新 query `(I_r, T_h)` 下，`I_h` 成为 positive target。

4. **核心目标是把 MLLM 的推理能力重新压回 embedding 空间。**  
   Foundation MLLM 能通过 VQA/CoT 判断细节差异，但检索器用单向量相似度时丢失了这种能力。ReCALL 用纠偏 triplets 和 grouped contrastive learning 让这些细粒度差异重新变成可学习的判别边界。

## 10. 评审

### 优点

- **问题定义有价值。** 论文不是只追求新结构或新 loss，而是指出 MLLM-to-retriever 适配中的范式冲突：生成式逐步推理能力会在单向量判别式训练中退化。
- **方法闭环清晰。** diagnose -> generate -> refine 对应「找错 -> 造针对性监督 -> 内化到检索器」，逻辑完整。
- **数据效率高。** 消融显示 self-guided mining 明显优于 random mining，说明它确实把训练信号集中在模型盲点上。
- **模型无关性较强。** 论文在 Qwen2.5-VL-7B 和 Qwen3-VL-8B 上都验证了增益，说明它不是只对弱 backbone 有效。

### 局限

- **计算和工程成本较高。** 需要先训练 `R_base`，再跑全训练集检索挖掘失败样本，还要调用 foundation MLLM 生成和 VQA 过滤。
- **纠偏质量依赖 `F`。** 如果 foundation MLLM 对细粒度属性本身判断不稳，生成的 `T_h` 可能引入噪声。
- **本质仍是离线校准。** ReCALL 修正的是训练集暴露出的失败边界，对开放域中新型组合关系的泛化仍取决于 mining 和生成覆盖度。
- **单向量检索范式没有被根本改变。** 它缓解了能力退化，但没有完全解决「逐步推理」与「一次性 embedding 匹配」之间的结构性张力。

## 11. 可以记住的一句话

ReCALL 的核心不是让 MLLM 学会检索，而是让已经被检索微调压扁的 MLLM，把原本会做的细粒度组合推理重新反映到 embedding 空间里。
