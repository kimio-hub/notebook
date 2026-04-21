---
title: SAGE Benchmarking and Improving Retrieval for Deep Research Agents
---

# SAGE: Benchmarking and Improving Retrieval for Deep Research Agents

## 论文要解决什么问题
这篇论文关注的是：在 deep research agent 场景里，检索模块到底是不是已经足够强？特别是，近年来 LLM-based retriever 很热门，于是一个自然问题是：它们是否真的比传统检索器更适合 deep research agent？

作者的答案并不简单。通过系统评测，他们发现当前 deep research agent 在 reasoning-intensive retrieval 上仍然很吃力，而且在某些设置下，传统 BM25 甚至明显优于 LLM-based retriever。

## 核心思路 / 方法
这篇工作的贡献大致分成两部分：

### 1. 构建 benchmark
作者提出了 **SAGE**，这是一个面向 scientific literature retrieval 的 benchmark。根据公开摘要，它包含：
- 1200 个查询；
- 4 个科学领域；
- 一个约 20 万篇论文的检索语料库。

这个 benchmark 的目的是专门考察 deep research agent 在复杂科研检索问题上的能力，而不是只看简单问答。

### 2. 发现问题并提出改进
作者先评测了多个 deep research agent，发现这些系统在需要较强推理能力的检索任务上普遍表现不佳。随后，他们以 **DR Tulu** 为 backbone，对比了多类检索器，包括：
- **BM25** 这类传统关键词检索器；
- **LLM-based retriever**，例如摘要中提到的 ReasonIR 和 gte-Qwen2-7B-instruct。

一个重要发现是：**BM25 显著优于 LLM-based retriever，差距大约在 30% 左右。**

## 为什么会出现这个结果
这篇论文最有意思的地方，在于作者没有停留在“谁分高谁分低”，而是进一步追问：为什么会这样？

根据你原先的笔记和公开摘要，原因主要是：
- 当前 deep research agent 生成的 sub-query 往往是**稀疏关键词式**的；
- 这种 query 形式更适合传统的关键词匹配检索；
- 反而没有充分发挥 LLM-based dense retriever 在语义理解上的优势。

也就是说，问题不一定出在 LLM retriever 本身，而是出在 **agent 发出的查询形式** 与 **retriever 的擅长方式** 不匹配。

## 作者提出的改进方向
针对这个问题，作者提出了一种 **corpus-level test-time scaling** 的思路。根据公开摘要，他们会用 LLM 对文档补充 metadata 和关键词，使现成检索器更容易命中相关文档。

从直觉上看，这相当于：
- 不是强行要求 agent 改掉自己现在的 query 风格；
- 而是先把文档表示增强，让检索器更容易“听懂”这些关键词式 query。

公开摘要给出的结果是：
- 在 short-form 问题上带来约 **8%** 提升；
- 在 open-ended 问题上带来约 **2%** 提升。

## 我的理解 / 易混点
我觉得这篇论文很值得注意的一点是：它提醒我们，检索效果不只是 retriever 本身的能力问题，还和 **query 分布**、**agent 行为模式**、**文档表示方式** 密切相关。

换句话说，不能简单地说“LLM retriever 一定比 BM25 强”。如果 agent 生成的查询本质上就是关键词集合，那么 BM25 这种方法反而可能更契合当前系统。

## 小结
SAGE 的意义主要有三点：
- 它提供了一个更贴近 deep research agent 的科研检索 benchmark；
- 它揭示了当前系统在 reasoning-intensive retrieval 上的真实短板；
- 它说明检索器表现取决于整个系统的匹配关系，而不只是模型本身是否“更先进”。

对我来说，这篇论文最核心的 takeaway 是：**retrieval 不是孤立模块，必须放在 agent 的 query 生成方式和任务分布里一起看。**
