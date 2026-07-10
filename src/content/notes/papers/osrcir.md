---
title: "OSrCIR"
tags:
  - cir
  - multimodal
  - reasoning
description: "> 论文：Reason-before-Retrieve: One-Stage Reflective Chain-of-Thoughts for Training-Free Zero-Shot Composed Image Retrieval > 会议：CVPR 2025 > 方法名：OSrCIR > 代码：https:…"
category: "papers"
date: "2026-07-09"
updated: "2026-07-09"
---

## Reason-before-Retrieve: One-Stage Reflective Chain-of-Thoughts for Training-Free Zero-Shot Composed Image Retrieval

> 论文：Reason-before-Retrieve: One-Stage Reflective Chain-of-Thoughts for Training-Free Zero-Shot Composed Image Retrieval  
> 会议：CVPR 2025  
> 方法名：OSrCIR  
> 代码：https://github.com/microsoft/ACV/tree/main/OSrCIR  
> 任务：Training-Free Zero-Shot Composed Image Retrieval  
> 核心关键词：one-stage reasoning, reflective CoT, MLLM, target image description

## 摘要

OSrCIR 研究 training-free zero-shot CIR。已有训练免费方法通常采用两阶段流程：先用 image captioner 给参考图像生成 caption，再让 LLM 根据 caption 和 modification text 推理目标图像描述。这种流程有两个问题：第一，captioner 在不知道修改文本的情况下可能遗漏关键视觉细节；第二，LLM 只能基于已经压缩过的 caption 推理，无法充分利用原图信息。OSrCIR 提出 one-stage reasoning：直接把参考图像和修改文本一起输入 MLLM，让 MLLM 用 Reflective Chain-of-Thought 生成目标图像描述，再用 CLIP 做 text-to-image retrieval。实验显示，OSrCIR 在 CIRCO、CIRR、FashionIQ 等任务上超过已有 training-free 方法，并保持较高推理效率。

## 1. 论文想解决什么问题

典型 training-free ZS-CIR 过程：

```text
reference image -> image caption
image caption + modification text -> LLM target description
target description -> CLIP retrieval
```

论文认为这个 two-stage 过程会带来信息损失：

- image captioner 不知道 modification text，因此可能漏掉修改所需的关键对象；
- caption 一旦漏了细节，后面的 LLM 很难恢复；
- LLM 的推理能力被限制在文本 caption 上，没有直接看图；
- 简单 prompt 难以准确理解用户隐含意图。

OSrCIR 的核心思想是：**先推理，再检索；而且推理必须直接同时看参考图像和修改文本。**

## 2. 方法总览

OSrCIR 流程：

```text
reference image + modification text
        ↓
MLLM + Reflective CoT
        ↓
target image description
        ↓
CLIP text-to-image retrieval
```

它不训练任何模型，不生成 adapter，也不迭代多轮检索。

和 CoRR 的区别：

- OSrCIR 是 one-stage：一次 MLLM 推理后检索；
- CoRR 是 iterative：检索结果反过来作为下一轮 MLLM 的反馈。

## 3. Reflective Chain-of-Thought

OSrCIR 的 prompt 让 MLLM 分四步输出。

### 3.1 Original Image Description

要求 MLLM 描述参考图像中和 modification text 相关的视觉细节。

这一步不是泛泛描述整张图，而是带着修改文本去看图。例如用户说 “remove the human”，模型必须注意图里是否有人、人和物体是什么关系。

### 3.2 Thoughts

推理潜在用户意图：

- 用户想增加什么？
- 删除什么？
- 改变颜色、数量、姿态、背景还是视角？
- 哪些参考图像元素应该保留？

### 3.3 Reflections

对 Thoughts 进行反思和过滤：

- 去掉与用户意图无关的元素；
- 修正幻觉；
- 保留最关键的修改对象；
- 保证目标描述与原图上下文一致。

论文强调 Reflections 可以缓解 Thoughts 阶段的幻觉。

### 3.4 Target Image Description

最后生成简洁的目标图像描述，只保留用于检索的目标内容。

之后用 CLIP：

```math
I_t =
\arg\max_{I_c \in D}
\cos(\Psi_I(I_c), \Psi_T(T_t))
```

其中 `T_t` 是 MLLM 生成的 target image description。

## 4. Vision-by-Language In-Context Learning

为了让 MLLM 明白 Reflective CoT 每一步该怎么写，作者加入 in-context examples。

但为了保持 zero-shot，示例只用文本形式展示期望输出，不额外提供 reference image。这被称为 **vision-by-language ICL**。

作用：

- 规范输出格式；
- 引导模型学会“描述 -> 思考 -> 反思 -> 目标描述”的流程；
- 提高推理稳定性。

## 5. 实验结果

### 5.1 CIRCO 与 CIRR

| Backbone | 方法 | CIRCO mAP@5 | CIRR R@1 |
| --- | --- | ---: | ---: |
| ViT-B/32 | CIReVL | 14.94 | 23.94 |
| ViT-B/32 | OSrCIR | 18.04 | 25.42 |
| ViT-L/14 | CIReVL | 18.57 | 24.55 |
| ViT-L/14 | OSrCIR | 23.87 | 29.45 |
| ViT-G/14 | CIReVL | 26.77 | 34.65 |
| ViT-G/14 | OSrCIR | 30.47 | 37.26 |

结论：

- OSrCIR 在不同 CLIP backbone 上都提升；
- 对 CIRCO 提升尤其明显；
- CIRR 上也有稳定提升。

### 5.2 FashionIQ

论文表明 OSrCIR 在 FashionIQ 上也达到 training-free SOTA，但提升相对自然图像任务更受限制。

原因是 FashionIQ 中有很多服饰领域术语，例如 “sequined bodice”“less flowy”。MLLM 能生成这些词，但 CLIP 未必能很好理解，造成 reasoning module 和 retrieval module 的语义不匹配。

## 6. 消融实验

| 变体 | CIRCO mAP@5 | FashionIQ R@10 |
| --- | ---: | ---: |
| Full model, GPT-4o | 23.87 | 33.26 |
| w/o one-stage reasoning | 21.73 | 31.16 |
| w/o Reflective CoT | 20.86 | 30.27 |
| w/o Original Description | 22.56 | 32.37 |
| w/o Thoughts | 21.46 | 31.59 |
| w/o Reflections | 22.04 | 32.05 |
| w/o ICL | 22.97 | 32.03 |

结论：

- one-stage reasoning 比 two-stage 更好；
- Reflective CoT 是核心模块；
- Thoughts 和 Reflections 都重要；
- ICL 有稳定帮助。

### 6.1 不同 MLLM 的影响

| MLLM | CIRCO mAP@5 | FashionIQ R@10 |
| --- | ---: | ---: |
| LLaVA | 20.89 | 30.75 |
| MiniGPT-4 | 19.85 | 29.36 |
| GPT-4o-mini | 23.10 | 32.19 |
| GPT-4V | 22.15 | 31.55 |
| GPT-4o | 23.87 | 33.26 |

MLLM 越强，效果通常越好，但开源模型也能跑通这个框架。

## 7. 效率

论文报告 OSrCIR 每个 query 约 `0.6s`，比 CIReVL 快约 `66.67%`。

原因是：

- OSrCIR 省掉单独 captioner 阶段；
- 一次 MLLM prompt 完成所有 reasoning；
- 检索仍然是标准 CLIP text-to-image retrieval。

## 8. 我的理解

OSrCIR 的关键是把 MLLM 从“caption 后处理器”提升为“直接看图的意图推理器”。

CIReVL 这类两阶段方法的问题像是：先让一个人看图写一句话，再把这句话交给另一个人改写。但如果第一句话漏掉了“人抱着狗”这个关系，第二个人就很难推理出“remove the human” 后应该保留小狗。

OSrCIR 则让 MLLM 一开始就同时看到图和修改文本，所以它知道哪些视觉细节对修改任务重要。

## 9. 和 CoRR 的关系

OSrCIR 和 CoRR 都属于 MLLM-guided training-free ZS-CIR，但颗粒度不同：

| 维度 | OSrCIR | CoRR |
| --- | --- | --- |
| 推理次数 | 单轮 | 多轮 |
| 是否看检索结果 | 不看 | 看上一轮 top-K |
| 核心机制 | Reflective CoT 生成 target description | retrieval feedback 反思并更新 query |
| 风险 | 第一次推理错了仍可能错 | 多轮可能 query drift |
| 优势 | 快、简单、低成本 | 可利用检索结果自我纠错 |

可以理解为：OSrCIR 解决 “推理前别丢图像细节”，CoRR 解决 “检索后要能反思错误结果”。

## 10. 评审

### 优点

- **方法简单有效。** 不训练、不迭代，只换 prompt 和推理方式。
- **直击两阶段信息损失。** 让 MLLM 直接看图和修改文本，减少 caption bottleneck。
- **可解释性好。** Reflective CoT 输出能看到模型如何理解用户意图。
- **效率高。** 相比复杂多轮或 adapter 方法部署更轻。

### 局限

- **依赖 MLLM 能力。** 弱 MLLM 的视觉理解和推理会明显拉低效果。
- **仍受 CLIP 语言空间限制。** MLLM 生成的细粒度词语，CLIP 不一定能检索好。
- **没有检索反馈。** 如果第一轮 target description 偏了，OSrCIR 不会像 CoRR 那样根据 top-K 结果自纠。
- **prompt 工程敏感。** Reflective CoT 的步骤设计对效果影响明显。

## 11. 一句话总结

OSrCIR 的核心是：把 zero-shot CIR 从“两阶段 caption 后推理”改成“MLLM 直接看图和文本做单轮反思推理”，先生成更准确的目标图像描述，再交给 CLIP 检索。

