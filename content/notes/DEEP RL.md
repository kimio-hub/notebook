---
title: DEEP RL
---

# Deep RL 学习笔记

## 摘要
Deep RL 的目标是学习一个策略（policy），让智能体在与环境交互时获得尽可能大的累计回报。常见方法大致可以分为 value-based、policy-based 和 actor-critic 三类；其中 [[PPO]] 和 [[GRPO]] 都可以看作围绕策略优化展开的方法。

## 核心概念
- **State / Observation（状态/观测）**：智能体当前能够感知到的信息。
- **Action（动作）**：智能体在当前状态下可执行的选择。
- **Policy（策略）**：从状态到动作的映射，记作 $\pi(a\mid s)$。
- **Trajectory（轨迹）**：智能体与环境交互形成的序列，例如 $(s_0,a_0,r_0,s_1,a_1,r_1,\dots)$。
- **Return（回报）**：从某一时刻开始未来奖励的累计值。

从任务角度看，强化学习就是选择一个合适的 policy，使 agent 按照该 policy 行动时得到的长期 reward 最大。

## 主要流派

### Value-based
基于价值的方法不直接学习策略，而是学习动作价值函数 $Q(s,a)$ 或状态价值函数 $V(s)$。如果我们已经得到了最优动作价值函数 $Q^*(s,a)$，那么策略就可以通过“在每个状态下选择 Q 值最大的动作”直接导出。

这类方法的优点是目标比较明确，但通常更适合离散动作空间；如果要表达显式的随机策略，会不如 policy-based 方法自然。

### Policy-based
基于策略的方法直接用参数化模型表示策略，记作 $\pi_\theta(a\mid s)$，然后直接优化期望回报：

$$
J(\theta)=\mathbb{E}_{\tau\sim\pi_\theta}[R(\tau)]
$$

常见的策略梯度形式可以写成：

$$
\nabla_\theta J(\theta)=\mathbb{E}_{\tau\sim\pi_\theta}\left[\sum_{t=0}^{T}\nabla_\theta \log \pi_\theta(a_t\mid s_t)\, G_t\right]
$$

这里的思想是：让带来更高回报的动作在未来出现得更频繁。

### Actor-Critic
Actor-Critic 结合了前两类方法的优点：
- **Actor**：策略网络 $\pi_\theta(a\mid s)$，负责决定动作。
- **Critic**：价值网络 $V_w(s)$ 或 $Q_w(s,a)$，负责评估当前策略的表现。

Critic 的作用主要是给 Actor 提供更稳定的学习信号，从而降低策略梯度的方差。[[PPO]] 就是典型的 Actor-Critic 方法。

## 方法之间的关系
- **Value-based**：先学“这个动作值不值”。
- **Policy-based**：直接学“下一步该怎么做”。
- **Actor-Critic**：一边学“怎么做”，一边学“做得怎么样”。

从这个角度看，PPO 可以理解为“带约束的小步策略优化”；GRPO 则更强调基于一组样本的相对比较来更新策略。

## 小结
Deep RL 的核心不是单纯拟合一个标签，而是在交互中不断更新策略，使长期回报最大。理解 value、policy 和 actor-critic 三种范式的差异，是继续学习 PPO、GAE、GRPO 等方法的基础。
