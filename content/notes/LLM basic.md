---
title: LLM basic
---

# LLM 基础学习笔记

## 摘要
LLM 的输入和输出都不是直接处理自然语言字符串，而是处理 token 及其对应的向量表示。理解 token、embedding、encoder、decoder 以及生成过程，是继续学习 Transformer 和推理过程的基础。

## 核心概念
### Token
文本不能直接送进模型计算，通常要先被切分成 token。每个 token 会在词表中找到对应的编号，然后再映射成向量表示。

除了 token 本身的语义向量，模型还需要位置信息，因此通常会把 token embedding 与 position embedding 结合起来，作为模型的输入表示。

### Encoder 与 Decoder
- **Encoder（编码器）**：通常是双向的，能够同时利用前后文信息。它更适合理解输入内容。
- **Decoder（解码器）**：通常是单向的，只能看到当前位置之前的内容，更适合逐步生成输出。

在早期 Transformer 中，encoder-decoder 结构比较常见；而现在很多主流 LLM 更常采用 **decoder-only** 架构。

## 生成流程
1. 用户输入文本。
2. 文本被切分成 token。
3. token 编号和位置信息一起转换成输入向量。
4. 模型通过 Transformer 的 attention 机制计算上下文表示。
5. 模型输出下一个 token 的概率分布。
6. 根据解码策略选出下一个 token，并不断重复，直到生成结束。

当 temperature 接近 0 时，模型更倾向于选择当前概率最大的 token；temperature 更高时，采样会更随机。

## 常见补充概念
### Chain of Thought（CoT）
CoT 指的是让模型显式写出中间推理过程，从而提升复杂任务上的表现。它不是模型结构本身的一部分，而是一种提示和推理范式。

## 小结
LLM 可以理解为“基于上下文不断预测下一个 token”的模型。掌握 token 化、向量表示、Transformer 计算和解码流程后，再去看更复杂的 agent、reasoning 和 RLHF 方法会更顺畅。
