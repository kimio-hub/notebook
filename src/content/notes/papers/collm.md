---
title: "CoLLM"
tags:
  - cir
  - multimodal
  - llm
description: "> 论文：CoLLM: A Large Language Model for Composed Image Retrieval > 会议：CVPR 2025 > 项目页：collm-cvpr25.github.io > 任务：Composed Image Retrieval, CIR > 核心关键词：LLM embed…"
category: "papers"
date: "2026-07-09"
updated: "2026-07-09"
---

## CoLLM: A Large Language Model for Composed Image Retrieval

> 论文：CoLLM: A Large Language Model for Composed Image Retrieval  
> 会议：CVPR 2025  
> 项目页：collm-cvpr25.github.io  
> 任务：Composed Image Retrieval, CIR  
> 核心关键词：LLM embedding, synthetic triplet, MTCIR, on-the-fly synthesis, refined benchmark

## 摘要

CoLLM 研究的是如何在缺少人工标注 CIR triplets 的情况下训练一个强 CIR 检索模型。传统 CIR 需要 `<reference image, modification text, target image>` 三元组，但这类数据昂贵且规模有限；zero-shot 方法虽然避免标注，但常常依赖浅层 adapter、embedding 插值或中间 caption，难以处理复杂修改文本。CoLLM 的思路是：从普通 image-caption pairs 动态合成 CIR triplets，并用 LLM 直接生成组合 query embedding，从而把 LLM 的语义理解能力引入检索表示学习。

论文贡献有四个：第一，提出从 image-caption pairs 进行 **on-the-fly triplet synthesis** 的训练方式；第二，提出 CoLLM 架构，用 LLM/Large Language Embedding Model 处理「参考图像 embedding + 修改文本」并输出组合查询 embedding；第三，构建大规模合成数据集 **MTCIR**，包含约 3.4M 图像对和 17.7M 修改文本；第四，使用 MLLM 重写 CIRR 和 FashionIQ 中有歧义的样本，得到更可靠的 refined benchmarks。实验显示，CoLLM 在 CIRR、FashionIQ、CIRCO 等数据集上取得强结果，MTCIR 对其他模型也有明显提升。

## 1. 论文想解决什么问题

CIR 的标准输入是：

```text
reference image + modification text -> target image
```

难点是训练数据：

- 人工 triplet 很贵；
- 合成 triplet 可能缺乏多样性，修改文本不自然；
- 只用 image-caption pairs 的 zero-shot 方法，通常没有直接学习「组合 query -> 目标图像」；
- 现有评测数据有歧义，一个 query 可能对应多个合理目标，但只标一个 ground truth。

CoLLM 的核心判断是：**与其只做 zero-shot 推理，不如把丰富的 image-caption pairs 转化为可监督的 CIR 训练信号。**

## 2. 方法概览

CoLLM 的训练分两类数据来源：

1. **只有 image-caption pairs 时**：动态合成 reference embedding 和 modification text，把普通图文对变成 CIR triplet。
2. **有 synthetic CIR triplets 时**：直接用 reference image、modification text 和 target image 做对比学习。

模型结构包含：

- vision encoder `f(.)`：提取图像 embedding；
- adapter `g(.)`：把图像 embedding 转成 LLM 可接收的视觉 token；
- LLM / LLEM `Phi(.)`：负责组合 reference 和 modification text；
- projection/pooling `p(.)`：得到最终检索 embedding。

训练目标是让组合 query embedding 靠近 target image embedding。

## 3. On-the-Fly Triplet Synthesis

给定一个普通 image-caption pair：

```text
(v_i, w_i)
```

其中 `v_i` 是图像，`w_i` 是 caption。CoLLM 把 `v_i` 当作 target image，然后合成：

- reference image embedding；
- modification text。

### 3.1 Reference Image Embedding Synthesis

论文不真的生成一张 reference image，而是在 embedding 空间中合成 reference embedding：

1. 对 target image `v_i` 做数据增强，得到 `v_i'`；
2. 用 vision encoder 得到 embedding `h_i'`；
3. 在 batch 内找与 `h_i'` 最近的另一个图像 embedding `h_j'`；
4. 用 Slerp 在二者之间插值，得到 reference embedding `h_i*`。

直觉：

- `h_i*` 和 target `h_i` 接近，但不完全一样；
- 它模拟了一个和 target 相似、但需要文本修改才能到达 target 的 reference；
- 只在 embedding 层合成，比生成真实图像便宜很多。

### 3.2 Modification Text Synthesis

CoLLM 不直接把 target caption `w_i` 当修改文本，因为这样模型可能只看文本就能检索到 target，忽略 reference image。

它用目标 caption `w_i` 和近邻 caption `w_j` 构造修改文本，形式类似：

```text
Instead of <w_j>, <w_i>
```

这样修改文本同时包含：

- reference 与 target 的相似部分；
- 需要改变的差异部分。

这更接近真实 CIR 用户输入。

## 4. Query Composition with LLM

CoLLM 用 LLM 生成三类 embedding：

```math
c_i^v = p(\Phi(g(h_i^*)))
```

```math
c_i^w = p(\Phi(w_i))
```

```math
c_i = p(\Phi([g(h_i^*); w_i^*]))
```

含义：

- `c_i^v`：只看 reference embedding 的 unimodal query；
- `c_i^w`：只看文本的 unimodal query；
- `c_i`：真正的 composed query；
- target embedding 是 `z_i = f(v_i)`。

训练时对这些 query embedding 和 target image embedding 做 contrastive loss。消融表明，加入 unimodal queries 有帮助，因为它们给模型额外的单模态对齐约束。

## 5. MTCIR 数据集

MTCIR 是论文的重要贡献之一：

- 约 3.4M image pairs；
- 约 17.7M modification texts；
- 每个 image pair 对应多个短修改文本；
- 数据源来自 LLaVA-558k 等多样图像；
- 使用 MLLM 做详细 caption，再用 LLM 描述 caption 间差异；
- 去除人脸/生物特征相关敏感信息。

和已有合成数据相比，MTCIR 的重点不是只扩大规模，而是让修改文本更接近真实用户查询：短、多样、覆盖不同属性。

## 6. Refined Benchmarks

作者认为 CIRR 和 FashionIQ 中存在歧义样本：一个 query 可能匹配多个图像，但标注只承认一个 target。

他们用 Claude 3 Sonnet 对样本进行：

- 歧义检测；
- 修改文本重写；
- 无法修复的样本删除；
- 生物特征信息过滤。

结果是 refined CIRR 和 refined FashionIQ。论文认为这些新 benchmark 更适合评估模型是否真正理解细粒度修改。

## 7. 实验结果

### 7.1 MTCIR 的效果

在 synthetic CIR datasets 上训练：

| 方法 | 数据集 | CIRR R@1 | FashionIQ R@10 |
| --- | --- | ---: | ---: |
| BLIP-L | LaSCo | 36.6 | 24.8 |
| BLIP-L | WebCoVR | 39.3 | 26.7 |
| BLIP-L | MTCIR | 42.4 | 37.9 |
| CoLLM | LaSCo | 43.2 | 38.5 |
| CoLLM | MTCIR | 45.8 | 39.1 |

结论：

- MTCIR 不只对 CoLLM 有用，也能提升普通 BLIP-L；
- CoLLM + MTCIR 效果最好；
- 数据质量和修改文本风格很关键。

### 7.2 Refined Benchmark 上的结果

在 refined CIRR / FashionIQ 上：

| 方法 | 数据集 | Ref. CIRR R@1 | Ref. FashionIQ R@10 |
| --- | --- | ---: | ---: |
| MagicLens | MagicLens | 42.3 | 45.5 |
| CoLLM | MTCIR | 46.5 | 48.3 |
| BLIP-L | MTCIR | 53.8 | 54.8 |
| CoLLM | LaSCo | 57.3 | 56.9 |
| CoLLM | MTCIR | 60.4 | 57.2 |

作者强调 refined benchmark 更能放大模型对复杂修改文本的理解差异。

## 8. 消融实验

### 8.1 图像和文本插值

| Image interpolation | Text interpolation | CIRCO mAP sum | CIRR Recall sum |
| --- | --- | ---: | ---: |
| no | no | 14.7 | 170.1 |
| yes | no | 14.5 | 176.4 |
| no | yes | 39.6 | 183.3 |
| yes | yes | 52.8 | 194.5 |

结论：文本插值非常关键，图像 embedding 插值配合文本插值效果最好。

### 8.2 近邻选择

| Reference synthesis | CIRCO mAP sum | CIRR Recall sum |
| --- | ---: | ---: |
| random in-batch sample | 46.7 | 182.2 |
| nearest in-batch neighbor | 52.8 | 194.5 |

结论：reference 应该和 target 相似但不同，近邻比随机样本更适合模拟真实 CIR。

### 8.3 LLM vs LLEM

论文发现用于 embedding/retrieval 训练过的 LLEM 通常优于普通生成式 LLM，例如 SFR-Embedding-2 表现更好。这说明 CIR query composition 不只是语言生成任务，更是表示学习任务。

## 9. 我的理解

CoLLM 和前面几篇 training-free 方法不太一样。CIReVL、OSrCIR、CoRR 更像是在推理时生成或修正 target caption；CoLLM 则更像是在问：**怎么用便宜数据训练一个真正的 CIR 表示模型？**

它最巧妙的点是 reference image 不用真的生成，只在 embedding 空间合成。这让普通 image-caption pairs 可以变成可监督训练样本，而且成本低很多。

我觉得 CoLLM 的核心不是“用了 LLM”，而是把 LLM 当作 **composed query embedding generator**，让它参与检索向量生成，而不是只输出一段文本。

## 10. 评审

### 优点

- **数据视角很强。** MTCIR 和 on-the-fly synthesis 都在解决 CIR 最大痛点：triplet 数据稀缺。
- **避免中间 caption 误差。** CoLLM 直接输出 composed embedding，不一定依赖 target caption 做检索。
- **benchmark 反思有价值。** 指出 CIRR/FashionIQ 的歧义问题，并尝试修正。
- **消融扎实。** 证明近邻合成、文本插值、unimodal queries、LLEM 选择都影响明显。

### 局限

- **训练成本高于 training-free 方法。** 需要大规模图文对、LLM/LLEM 和对比训练。
- **embedding-level reference synthesis 仍是近似。** 合成 reference embedding 不一定对应真实图像分布。
- **refined benchmark 依赖 MLLM/Claude 判断。** 虽然减少歧义，但也可能引入模型偏好。
- **图像细节压缩成单视觉 token 可能不足。** 作者也承认未来可探索更丰富的 MLLM 表示。

## 11. 一句话总结

CoLLM 的核心是：用 image-caption pairs 动态合成 CIR triplets，并让 LLM 直接学习组合查询 embedding，从数据和模型两端缓解 CIR 标注稀缺与组合理解不足的问题。

