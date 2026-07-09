---
title: "CoRR"
tags:
  - cir
  - multimodal
  - reasoning
category: "papers"
date: "2026-07-09"
updated: "2026-07-09"
---

# Reflection from Retrieval: MLLM-Guided Iterative Reasoning for Zero-Shot Composed Image Retrieval

> 论文状态：ICLR 2026 withdrawn submission, OpenReview  
> 方法名：CoRR, Chain of Reflective Composed Image Retrieval  
> 任务：Zero-Shot Composed Image Retrieval, ZS-CIR  
> 核心关键词：retrieval feedback, MLLM reasoning, query refinement, Slerp, training-free

## 摘要

这篇论文研究 **zero-shot composed image retrieval**：给定一张参考图像和一段修改文本，在没有任务专门训练的情况下，从图像库中检索修改后的目标图像。已有方法通常先用 MLLM/LLM 把「参考图像 + 修改文本」改写成目标图像 caption，再用 CLIP 这类 embedding model 做 text-to-image retrieval。但这种单轮方法有一个明显缺陷：如果第一次理解错了用户意图，后续检索没有自我纠错机制。

论文提出 **CoRR**，把 ZS-CIR 重新定义为一个闭环过程：

```text
retrieval -> reflection -> refinement
```

它先用初始 query 检索 top-K 图像，然后让 MLLM 观察这些检索结果和对应 caption，分析当前结果哪里偏离了用户意图，再生成新的目标 caption 进入下一轮检索。为了避免多轮迭代造成 query drift，作者用 **Spherical Linear Interpolation, Slerp** 融合历史 query 向量和新生成 query 向量；为了让 MLLM 输出更适配 embedding model，作者还提出 **Retrieval-Driven Caption Optimization**，从检索库中为候选图像生成并筛选更符合检索模型偏好的 caption，作为 MLLM 反思时的上下文例子。实验显示，CoRR 在 CIRCO、CIRR、FashionIQ、GeneCIS 等数据集上能稳定提升不同 backbone 的表现，并且保持 training-free。

## 1. 研究背景

Composed Image Retrieval, CIR 的输入是：

```text
reference image I_r + modification text T
```

目标是从图库 `D = {I_1, I_2, ..., I_n}` 中找出符合修改意图的目标图像。

例如：

```text
参考图像：一只狗坐在雪地里
修改文本：让这只狗和朋友一起拉雪橇
目标图像：多只狗在雪地里拉雪橇
```

Zero-shot CIR 不使用 `<reference, text, target>` 三元组训练专门模型，而是依赖已有的 MLLM 和 embedding model。常见做法是：

1. MLLM 根据参考图像和修改文本生成目标 caption；
2. 用 CLIP/BLIP/MMRet 等 embedding model 编码 caption；
3. 在图库向量中最近邻检索。

## 2. 论文指出的问题

已有 zero-shot CIR 方法大多是 **single-shot query rewriting**：

```text
(I_r, T) -> generated caption -> retrieval result
```

问题在于：

- MLLM 生成的 caption 可能过宽，遗漏关键视觉约束；
- 也可能过窄，把参考图像中无关细节带进去；
- 一旦初始 caption 理解错了，检索器没有机会根据结果修正；
- 检索结果本身其实包含反馈信息，但过去方法没有充分利用。

作者的核心观点是：**检索结果不应该只是最终输出，也应该成为下一轮推理的证据。**

## 3. 方法总览

CoRR 是一个 training-free 的迭代框架，主要包含三个模块：

1. **MLLM-Guided Self-Reflection**  
   让 MLLM 根据上一轮检索结果反思当前 query 的错误，并生成更好的目标 caption。

2. **Historical Query Fusion based on Slerp**  
   用 Slerp 融合历史 query 和新 query，避免多轮反思后语义漂移。

3. **Retrieval-Driven Caption Optimization**  
   为检索到的候选图像生成并筛选高质量 caption，让 MLLM 学到当前 embedding space 偏好的表达方式。

整体流程：

```text
Step 0: 由 embedding model 得到初始 query vector v_0
Step 1: 用 v_t 检索 top-K 图像
Step 2: 给 top-K 图像配上 top-N 优化 caption
Step 3: MLLM 观察原始输入 + 检索图像 + caption，反思失败点
Step 4: MLLM 生成新 caption T_t
Step 5: 编码 T_t 得到 u_t
Step 6: 用 Slerp 融合 u_t 和历史向量 v_{t-1}，得到 v_t
Step 7: 进入下一轮检索
```

论文默认设置：

- top-K retrieved images：`K = 5`
- top-N synthetic captions：`N = 10`
- BLIP-2 每张候选图生成 caption 数：`M = 30`
- 额外反思轮数：`2 rounds`
- Slerp 插值权重：`alpha = 0.8`
- 主 MLLM：Qwen-VL-Max

## 4. MLLM-Guided Self-Reflection

每轮迭代中，MLLM 会执行一个 Chain-of-Thought 风格的反思流程。

### 4.1 Understand User Intent

先分析参考图像和修改文本，明确用户真正想要什么：

- 要保留哪些原图元素；
- 要修改哪些属性；
- 是否涉及数量变化、空间关系、动作变化、视角变化、否定等。

### 4.2 Analyze Retrieval Results

然后观察上一轮 top-K 检索结果：

- 它们共同出现了哪些视觉模式；
- 哪些元素符合用户意图；
- 哪些元素缺失、错误或被过度强调；
- 检索模型似乎偏好什么样的 caption 风格。

### 4.3 Problem Reflection

接着反思为什么当前结果失败：

- 是遗漏了参考图像里的关键实体？
- 是修改文本理解错了？
- 是把无关背景带进 query 了？
- 是 caption 太抽象或太具体，不适合 embedding model？

### 4.4 Caption Generation Strategy

最后生成新的目标 caption。论文强调输出应该：

- 简洁、事实性强；
- 用正向描述，避免否定表达；
- 避免 “this image / it” 这类自指；
- 突出最有区分度的视觉细节；
- 风格贴近检索模型喜欢的 caption。

## 5. Historical Query Fusion: 用 Slerp 防止 Query Drift

如果每一轮都直接用 MLLM 新生成的 caption 替换旧 query，容易出现 **query drift**：新 caption 引入噪声，导致原本有用的语义信息被覆盖。

因此作者不用直接替换，而是融合：

```math
v_t = \text{Slerp}(u_t, v_{t-1}; \alpha)
```

其中：

- `u_t = Psi_T(T_t)`：当前轮新 caption 的文本向量；
- `v_{t-1}`：上一轮历史 query 向量；
- `alpha`：插值权重，论文默认 `0.8`；
- `v_t`：融合后的新 query 向量。

Slerp 定义为：

```math
\text{Slerp}(u, v; \alpha)
=
\frac{\sin((1-\alpha)\theta)}{\sin(\theta)}u
+
\frac{\sin(\alpha\theta)}{\sin(\theta)}v
```

其中：

```math
\theta = \arccos(u \cdot v)
```

为什么用 Slerp：

- CLIP 类 embedding 通常是 unit-normalized，在超球面上比较余弦相似度；
- 线性插值可能偏离超球面几何；
- Slerp 可以让 query 在 embedding space 中平滑移动；
- 它保留历史语义，又允许新反思结果逐步改变方向。

## 6. Retrieval-Driven Caption Optimization

这部分是论文比较有意思的地方。

作者认为：好的 query caption 不仅要符合用户意图，还要符合 embedding model 的表达偏好。不同 caption 描述同一张图，检索效果可能差很多。

因此 CoRR 为每张 top-N 候选图像用 BLIP-2 生成 `M = 30` 个 caption，然后用检索模型自己来筛选 caption。

筛选标准是两级排序：

1. 用 caption `C_i` 作为 query 检索，看源图像 `I` 被排在第几名，rank 越靠前越好；
2. 如果 rank 相同，再比较 caption 和源图像的相似度 `sim(I, C_i)`，相似度越高越好。

形式化写法：

```math
(r_i, s_i) < (r_j, s_j)
\Longleftrightarrow
(r_i < r_j) \ \text{or} \ (r_i = r_j \ \text{and} \ s_i > s_j)
```

其中：

- `r_i = rank(I, C_i)`；
- `s_i = sim(I, C_i)`。

这样选出来的 caption 会作为 MLLM 的上下文例子，帮助它：

- 看懂候选图像中哪些元素重要；
- 学到 embedding model 偏好的描述粒度；
- 生成更容易被检索模型理解的目标 caption。

## 7. 实验结果

### 7.1 CIRCO 与 CIRR

在 CLIP-ViT-L/14 架构上：

| 方法 | CIRCO mAP@5 | CIRR R@1 |
| --- | ---: | ---: |
| Slerp | 16.40 | 19.28 |
| Ours + Slerp | 26.08 | 25.59 |
| MMRet-Large | 40.20 | 37.95 |
| Ours + MMRet-Large | 42.70 | 43.21 |

关键提升：

- `Ours + Slerp` 在 CIRCO mAP@5 从 `16.40` 提升到 `26.08`；
- `Ours + MMRet-Large` 在 CIRR R@1 从 `37.95` 提升到 `43.21`；
- 即使在强 baseline MMRet-Large 上仍有稳定增益。

### 7.2 FashionIQ

| 方法 | Avg. R@10 | Avg. R@50 |
| --- | ---: | ---: |
| MMRet-Large | 34.65 | 55.27 |
| Ours + MMRet-Large | 36.70 | 57.45 |

FashionIQ 更偏属性修改，比如衣服颜色、款式、图案等。CoRR 在这类细粒度属性任务上也有效。

### 7.3 GeneCIS

论文附录报告了 GeneCIS 结果：

| 方法 | Avg. R@1 |
| --- | ---: |
| OSrCIR | 17.90 |
| MMRet-Large | 17.13 |
| Ours + MMRet-Large | 19.53 |

说明 CoRR 不只适用于传统 CIR 数据集，也能迁移到更一般的 conditional image similarity 任务。

## 8. 消融实验

使用 MMRet-Large 作为 baseline：

| 方法 | CIRCO mAP@5 | FashionIQ Avg. R@10 |
| --- | ---: | ---: |
| baseline | 37.75 | 34.65 |
| reflection only | 36.66 | 30.82 |
| reflection + query fusion | 40.82 | 36.26 |
| + random caption | 41.85 | 36.35 |
| + optimized caption, full model | 42.77 | 36.70 |

结论：

- 单独 reflection 反而下降，说明 MLLM 直接改 query 很容易漂移；
- query fusion 是关键，没有它反思会不稳定；
- caption 上下文有帮助；
- optimized caption 比 random caption 更有效，说明 caption 要和检索模型偏好对齐。

### 8.1 Prompt 策略

| 方法 | CIRCO mAP@5 | FashionIQ Avg. R@10 |
| --- | ---: | ---: |
| full model | 42.77 | 36.70 |
| w/o CoT | 41.45 | 35.68 |
| w/o think process | 40.77 | 35.97 |

结构化 CoT prompt 有帮助，但不是全部收益来源；Slerp 和 caption optimization 也很关键。

### 8.2 Query Fusion 策略

FashionIQ validation：

| 方法 | Dress R@10 | Shirt R@10 | TopTee R@10 |
| --- | ---: | ---: | ---: |
| baseline | 29.84 | 37.04 | 37.07 |
| linear interpolation | 28.84 | 38.17 | 37.97 |
| pseudo-relevance feedback | 27.56 | 32.82 | 33.45 |
| proposed Slerp | 31.33 | 39.10 | 39.67 |

结论：传统 pseudo-relevance feedback 直接用 top retrieved images 的均值会被噪声污染；Slerp 更稳。

### 8.3 反思轮数与效率

论文主实验使用额外 2 轮反思。更多轮数收益变小，因此 2 轮是成本和性能的折中点。

平均每个 query 耗时：

| 方法 | Seconds |
| --- | ---: |
| CIReVL | 2.21 |
| LDRE | 14.26 |
| OSrCIR | 13.38 |
| Proposed | 6.44 |

CoRR 比单轮方法慢，但比一些复杂 reasoning 方法更快。

## 9. 和 ReCALL 的关系

这篇 CoRR 和前一篇 ReCALL 可以放在一起理解。

| 维度 | ReCALL | CoRR |
| --- | --- | --- |
| 目标 | 修复 MLLM 检索微调后的能力退化 | 修复 zero-shot query rewriting 的单轮不可纠错 |
| 是否训练 | 需要继续训练检索器 | training-free |
| 反馈来源 | `R_base` 在训练集上的失败 hard negatives | 测试时 top-K retrieval results |
| MLLM 作用 | 生成 corrective triplets | 观察检索结果并反思生成新 caption |
| 核心机制 | diagnose -> generate -> refine | retrieval -> reflection -> refinement |
| 解决问题 | 把细粒度推理能力压回 embedding space | 用检索反馈动态修正 query |

一句话区别：

> ReCALL 是训练阶段用 MLLM 帮检索器补课；CoRR 是推理阶段让 MLLM 看检索结果后自己改答案。

## 10. 我的理解

这篇论文的核心不是发明一个新检索器，而是改变 zero-shot CIR 的使用方式。

过去的方法像是：

```text
我猜用户想要什么 -> 生成 caption -> 检索
```

CoRR 变成：

```text
我先猜一次 -> 看检索结果错在哪里 -> 再改 query -> 再检索
```

这更接近人在搜索图片时的过程：第一次搜不到，就会看返回结果，意识到关键词缺了什么或多了什么，然后换一个更准确的搜索词。

我觉得最关键的设计是 **retrieval feedback**。检索结果虽然可能是错的，但它们能暴露当前 query 的偏差：如果 top results 都是“拉雪橇的狗”但没有“白色 Samoyed”和“森林背景”，那下一轮 query 就应该补上这些约束。

不过，论文也说明不能盲目相信检索结果。单独 reflection 会让性能下降，传统 pseudo-relevance feedback 也会下降。这意味着 retrieval feedback 是有噪声的，必须通过 query fusion 和 caption optimization 控制它。

## 11. 评审

### 优点

- **问题切入点自然。** 单轮 zero-shot CIR 的确容易“一错到底”，用检索结果做反馈很符合检索系统直觉。
- **框架是 plug-and-play。** 可以接在 Slerp、MMRet、CIReVL、OSrCIR 等不同 backbone 上。
- **不需要训练。** 对没有标注 triplets 的场景很有吸引力。
- **消融比较有说服力。** reflection only 下降这一点反而很重要，证明作者不是简单说“让 MLLM 多想想就行”，而是指出必须稳定融合历史 query。
- **和传统 IR 思想连接清楚。** 它像是多模态版本的 pseudo-relevance feedback，但用 MLLM 做语义级反思，而不是直接平均 top documents。

### 局限

- **推理成本更高。** 每个 query 需要多轮 MLLM 调用、caption generation 和检索，实时系统要考虑延迟。
- **依赖 MLLM 视觉理解质量。** 如果 MLLM 看错候选图，反思就会把 query 改坏。
- **依赖 captioner 和 embedding model 的配合。** BLIP-2 生成的 caption 如果质量差，或者检索模型偏好奇怪，optimized caption 可能带偏 MLLM。
- **withdrawn submission 状态需要注意。** 这篇目前不是正式接收论文，实验设置和结论还需要更谨慎对待。
- **多轮迭代没有理论收敛保证。** 作者用 2 轮作为经验折中，但复杂查询下什么时候该停仍是开放问题。

## 12. 可以记住的一句话

CoRR 的核心思想是：不要把检索结果只当答案，而要把它当作 MLLM 下一轮推理的反馈证据，让 zero-shot CIR 从一次性猜测变成可反思、可纠错的闭环检索过程。

