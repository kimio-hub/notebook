---
title: "SEED：面向 Agentic RL 的自演化 On-Policy 蒸馏"
description: "SEED 让模型自己生成 hindsight skill，用“看到技能的自己”当教师给出 token 级蒸馏权重，再与 GRPO 的轨迹级奖励联合训练。"
tags:
  - "agentic-rl"
  - "distillation"
  - "grpo"
category: "papers"
date: "2026-07-22"
updated: "2026-07-22"
---

论文链接:[arxiv.org/abs/2607.14777](https://t.co/d0ScMPt3NB)

一句话总结:SEED 首先使用 GLM-5.2 为 rollout 轨迹生成 hindsight skill 标注,再通过 SFT 让目标模型学会根据完整轨迹总结技能。进入强化学习阶段后,每轮冻结当前策略,用它生成 on-policy rollout,并分析完整轨迹得到技能。对于 rollout 中已经采样的同一个动作 token,当前策略在看到技能的特权上下文下作为 Teacher 打分(stop-gradient),在不看到技能的正常上下文下作为 Student 打分,两者的概率差被转换成 token 级蒸馏权重。这个细粒度信号与 GRPO 的轨迹级奖励联合训练模型。更新完成后,再冻结新策略、生成新的轨迹和技能,如此循环,使技能监督始终跟随当前策略不断演化。

个人认为的创新点:让模型自己生成 hindsight skill,从而在第一轮 GLM 冷启动之后实现完全的 self-evolving。

## 论文 pipeline

冷启动阶段:

1. 先用原本模型 rollout 出多条轨迹
2. 再用 GLM-5.2 针对这些轨迹生成 hindsight skill。skill 具体是对每一条 rollout 成功或失败原因的总结(类似于给模型一个答案让它变成 teacher,给模型一个 skill 起到同样的效果)
3. 把 rollout 和 hindsight skill 一起用于原本模型的 SFT,让新模型能够根据 rollout 自主生成 hindsight skill

这样模型就具备了两个功能:

- actor:产生 rollout
- analyzer:针对 rollout 生成 skill

接着进行迭代,以某一轮为例:

1. 冻结这轮的 policy $\pi_{\theta_{old}}$
2. 用 $\pi_{\theta_{old}}$ 对每个 query 生成一组 rollout $\{\tau^{(1)},\dots,\tau^{(N)}\}$(GRPO 需要组内对比)
3. 用 $\pi_{\theta_{old}}$ 自己当 analyzer,为每条 rollout 总结 skill $s_\tau$
4. 用当前策略 $\pi_\theta$ 对已采样的动作 token 在两种上下文下重新打分:
	- teacher(带技能的特权上下文 $\tilde h_t$,即把 $s_\tau$ 拼进历史):$\ell^{\mathrm{skill}}_{t,\ell} = \log \pi_\theta(a_{t,\ell} \mid \tilde h_t, a_{t,<\ell})$
	- student(正常上下文):$\ell^{\theta}_{t,\ell} = \log \pi_\theta(a_{t,\ell} \mid h_t, a_{t,<\ell})$
	- 两者之差衡量技能对这个 token 的影响(注意有 stop-gradient):$\Delta_{t,\ell} = \operatorname{sg}\left[\ell^{\mathrm{skill}}_{t,\ell} - \ell^{\theta}_{t,\ell}\right]$
5. 用 sigmoid 把 $\Delta$ 压成 $(0,1)$ 之间的 token 级门控权重:$g_{t,\ell} = \sigma(\beta_{\mathrm{opd}}\,\Delta_{t,\ell})$
6. 得到 on-policy distillation(OPD)损失,只在动作 token 上算($m_{t,\ell}$ 是动作 token 的 mask):

$$\mathcal{L}_{\mathrm{opd}}(\theta) = \mathbb{E}_{t,\ell}\left[m_{t,\ell}\cdot g_{t,\ell}\cdot\left(\operatorname{sg}\left[\ell^{\mathrm{skill}}_{t,\ell}\right] - \ell^{\theta}_{t,\ell}\right)\right]$$

由于 teacher 项被 sg 挡住,梯度其实就是一个加权的对数似然:

$$\nabla_\theta \mathcal{L}_{\mathrm{opd}} = -\mathbb{E}_{t,\ell}\left[m_{t,\ell}\cdot g_{t,\ell}\cdot \nabla_\theta\, \ell^{\theta}_{t,\ell}\right]$$

也就是:技能越能提升某个 token 的概率($\Delta$ 越大 → $g$ 越接近 1),就越用力把 student 在正常上下文下的概率往上推;技能帮不上忙的 token 权重趋近 0,基本不管。

7. RL 部分就是标准 GRPO:组内奖励归一化得到 advantage

$$A^{\mathrm{rl}}_{q,n} = \frac{R(\tau_q^{(n)}) - \mu_q}{\sigma_q + \epsilon}$$

配上 clip 的重要性采样比 $\rho_{t,\ell}(\theta) = \exp(\ell^\theta_{t,\ell} - \ell^{\mathrm{old}}_{t,\ell})$ 和 KL 正则,得到 $\mathcal{L}_{\mathrm{rl}}$。

8. 最终 loss 就是两者加权和:

$$\mathcal{L}_{\mathrm{SEED}}(\theta) = \mathcal{L}_{\mathrm{rl}}(\theta) + \lambda_{\mathrm{opd}}\,\mathcal{L}_{\mathrm{opd}}(\theta)$$

## 几个容易误解的点

- **teacher 不是 $\pi_{\theta_{old}}$**:rollout 和 skill 由冻结的 $\pi_{\theta_{old}}$ 产生,但两路打分用的都是当前正在更新的 $\pi_\theta$,teacher 只是多看了 skill 这个特权信息,再靠 stop-gradient 保证梯度只从 student 一侧流过。所以 teacher 和 student 是同一个模型,区别只在上下文。
- **冷启动 SFT 的数据是过滤过的**:$\mathcal{D}_{\mathrm{sft}} = \{(x_\tau, s_\tau) : v_\tau = 1\}$,只保留验证通过的 skill 标注。
- **OPD 的理论解释**:论文证明 $\nabla_\theta\mathcal{L}_{\mathrm{opd}}$ 等价于在最小化当前策略与一个重加权分布 $r_k(v\mid c) \propto \pi_k(v\mid c)\,w_k(c,v)$ 之间的 KL——直观上就是把概率质量往"技能会提升的 token"上搬,而且当 GRPO 的组内奖励全相同(advantage 全为 0、RL 梯度消失)时,这个信号依然存在,这是它比纯 outcome RL 样本效率高的来源。

## 实验

- 文本任务:ALFWorld、WebShop,底座为 Qwen2.5-3B/7B-Instruct 和 Qwen3-1.7B
- 视觉任务:Sokoban(6×6)、EZPoints,底座为 Qwen2.5-VL-3B-Instruct
- 结论:相比 GRPO 等 baseline,成功率和样本效率都稳定提升,且对未见过的场景泛化更好

代码:[github.com/jinyangwu/SEED](https://github.com/jinyangwu/SEED)
