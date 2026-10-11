---
title: "DPO：你的语言模型其实是一个奖励模型"
description: "把 KL 约束下的 RLHF 改写成偏好对上的直接优化：DPO 损失的推导、直觉、优势与局限。"
tags:
  - "alignment"
  - "preference"
  - "rlhf"
category: "papers"
date: "2026-06-11"
updated: "2026-06-11"
---

## 论文核心 idea

DPO（Direct Preference Optimization）的核心思想是：不再像传统 RLHF 那样显式训练一个 reward model，再用 PPO 去优化 policy，而是把 RLHF 中的“奖励最大化 + KL 约束”目标，通过数学变形，改写成一个可以直接在偏好数据上训练语言模型的 loss。

传统 RLHF 的流程大致是：

$$
\text{SFT model}
\rightarrow
\text{Reward Model}
\rightarrow
\text{PPO / RL optimization}
$$

DPO 的流程则更简单：

$$
\text{SFT model}
\rightarrow
\text{Preference pairs}
\rightarrow
\text{Direct policy optimization}
$$

也就是说，DPO 不需要显式训练 reward model，也不需要训练 critic model 或进行复杂的 RL rollout。它直接利用偏好数据中的 chosen / rejected response 来更新模型。

论文标题里说：

> Your Language Model is Secretly a Reward Model

意思是：在 KL 约束的 RLHF 目标下，最优 policy 和 reward function 之间存在一个可以互相转换的关系。因此，我们可以绕过显式 reward model，直接从偏好数据中优化 policy。

---

## DPO 的损失函数

DPO 的 loss 可以写成：

$$
\mathcal{L}_{DPO}(\pi_\theta;\pi_{ref})
=
-\mathbb{E}_{(x,y_w,y_l)\sim D}
\left[
\log \sigma
\left(
\beta
\log
\frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)}
-
\beta
\log
\frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)}
\right)
\right]
$$

其中：

- $x$ 是 prompt。
- $y_w$ 是 preferred / chosen response。
- $y_l$ 是 dispreferred / rejected response。
- $\pi_\theta$ 是正在训练的模型。
- $\pi_{ref}$ 是冻结的 reference model，通常是 SFT 后的模型。
- $\beta$ 控制模型偏离 reference model 的程度。

这个公式的直觉是：

> 相对于 reference model，提高 chosen response 的相对 log-prob，同时降低 rejected response 的相对 log-prob。

所以 DPO 不是简单地“增大 accept 和 reject 的散度差”，而是在 reference model 的约束下，让模型更倾向于人类偏好的回答。

reference model 在这里非常重要。它相当于一个锚点，避免模型为了迎合偏好数据而偏离原本的语言能力太远。

---

## 和 SFT 的区别

DPO 看起来有点像 SFT，因为它也是在离线数据上做监督式训练，不需要和环境交互，也不需要 PPO 那种在线 RL 过程。

但二者的监督信号不同：

- SFT 学的是单个标准答案：最大化 $\log \pi_\theta(y|x)$。
- DPO 学的是两个答案之间的偏好关系：让 chosen response 比 rejected response 更可能。

所以 DPO 可以理解成一种 pairwise preference fine-tuning，而不是普通的 next-token supervised fine-tuning。

---

## 我对 DPO 的理解

DPO 的本质不是“训练一个 reward model”，而是把 reward model 隐式地吸收到 policy optimization 里面。

它真正优化的是模型策略 $\pi_\theta$，只是这个优化目标等价于某种 KL 约束下的 reward maximization。

因此，DPO 在形式上更像监督学习，但在理论上继承了 RLHF 的目标。

可以这样理解：

> DPO 用偏好数据直接塑造模型的输出分布，让模型在不显式训练 reward model 和不使用 PPO 的情况下，更偏向 chosen response，而远离 rejected response。

---

## 关于 reasoning 和 exploration

DPO 的局限在于：它主要利用已有偏好数据中的信号，本身没有像 RL rollout 那样主动探索新策略、新路径的机制。

所以它更像是 refinement，而不是 exploration。

如果模型在 pretraining 或 SFT 之后已经具备某些能力，DPO 可以把这些能力按人类偏好重新排序、强化和对齐。例如，让模型更常输出被人类偏好的风格、格式、解释方式或推理路径。

但是，如果偏好数据中没有包含更强的 reasoning pattern，DPO 通常很难凭空带来新的推理能力。它更依赖数据中已经存在的高质量 chosen response。


> DPO 不太适合作为主动发现新 reasoning 能力的算法。它更适合在已有能力和已有偏好数据的基础上做对齐和分布重塑。

这也是它和 PPO / RL-style 方法的重要区别：PPO 可以通过 rollout 和 reward feedback 对模型行为进行更直接的策略搜索，而 DPO 主要是在静态偏好数据上学习。

---

## DPO 的价值

DPO 的价值主要在于：

1. 训练流程更简单。
2. 不需要显式 reward model。
3. 不需要 critic model。
4. 不需要复杂的 PPO 训练过程。
5. 训练更稳定，算力成本更低。
6. 可以很好地利用人类偏好数据进行对齐。

它可以放在 SFT 之后，作为偏好对齐步骤：

$$
\text{Pretraining}
\rightarrow
\text{SFT}
\rightarrow
\text{DPO}
$$

在一些场景下，也可以把 DPO 和其他 RL 方法结合使用。例如先用 DPO 做稳定的偏好对齐，再用 RL 方法进一步探索；或者在 RL 之后，用 DPO 做额外的偏好校准。

不过从本质上看，DPO 更适合做 alignment / preference tuning，而不是做强探索型的能力发现。

---

## 总结

DPO 的本质是：

> 把 KL 约束下的 RLHF policy optimization 改写成一个基于偏好对的直接优化问题。

它不显式训练 reward model，也不使用 PPO 进行在线 RL，而是通过 reference model 约束下的 chosen / rejected log-prob 差异，直接更新语言模型策略。

因此，DPO 的优势是简单、稳定、省算力；局限是缺少主动探索机制，效果高度依赖偏好数据质量。
