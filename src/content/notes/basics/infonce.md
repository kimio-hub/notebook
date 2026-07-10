---
title: InfoNCE：对比学习的核心损失
tags:
  - contrastive-learning
  - multimodal
description: "InfoNCE 是对比学习（contrastive learning）中最常用的损失函数，堪称多模态领域的\"统治级\"损失。核心就一句话： 拉近正样本，推远负样本。CLIP、MoCo、SimCLR 这些经典工作，以及组合图像检索（CIR）所依赖的图文对齐能力，底层都是它。"
category: "basics"
date: "2026-07-09"
updated: "2026-07-09"
---

## 摘要

InfoNCE 是对比学习（contrastive learning）中最常用的损失函数，堪称多模态领域的"统治级"损失。核心就一句话：**拉近正样本，推远负样本**。CLIP、MoCo、SimCLR 这些经典工作，以及组合图像检索（CIR）所依赖的图文对齐能力，底层都是它。

## 损失形式

给定一个 query $q$、一个正样本 $k^+$ 和 $K$ 个负样本 $k_i$：

$$\mathcal{L}_{\text{InfoNCE}} = -\log \frac{\exp\big(\mathrm{sim}(q, k^+)/\tau\big)}{\exp\big(\mathrm{sim}(q, k^+)/\tau\big) + \sum_{i=1}^{K}\exp\big(\mathrm{sim}(q, k_i)/\tau\big)}$$

其中 $\mathrm{sim}(\cdot,\cdot)$ 通常是余弦相似度，$\tau$ 是温度系数。

## 怎么理解

- 形式上就是一个 $(K+1)$ 类的 softmax 分类：把正样本从一堆负样本里"认出来"。分类分对了，代表表示空间里正样本确实离 query 最近。
- **温度 $\tau$ 很关键**：$\tau$ 越小，softmax 越尖锐，梯度越集中在最难的负样本（hard negatives）上；$\tau$ 越大，惩罚越平均。CLIP 直接把 $\tau$ 设成可学习参数。
- 名字里的 NCE 来自 noise-contrastive estimation。InfoNCE 出自 CPC（van den Oord et al., 2018），可以证明它是 $q$ 与 $k^+$ 之间**互信息的一个下界**——负样本数 $K$ 越大，下界越紧。这也解释了为什么对比学习普遍偏好大 batch / 大负样本队列。

## 在多模态里的用法

CLIP 用的是**对称 InfoNCE**：一个 batch 里 $N$ 对图文，图→文方向和文→图方向各算一次（batch 内其余样本互为负样本），再取平均。训练出来的联合嵌入空间，就是后续一大批检索工作的地基——[CoLLM](/notes/papers/collm/)、[OSrCIR](/notes/papers/osrcir/) 这类 CIR 方法能"零样本"工作，靠的正是这个对比空间。

## 相关笔记

- [Transformer](/notes/basics/transformer/)：编码器骨架
- [CoLLM](/notes/papers/collm/) / [OSrCIR](/notes/papers/osrcir/)：建立在 CLIP 对比空间上的组合图像检索
