---
title: "强化学习初印象"
description: "从 MDP、策略、价值函数与 Bellman 方程，到蒙特卡洛、TD、函数近似与 Actor-Critic：强化学习的基本问题地图。"
tags:
  - "rl"
  - "basics"
category: "basics"
date: "2026-06-08"
updated: "2026-06-08"
---

## 1. 强化学习到底在学什么？

强化学习（Reinforcement Learning, RL）的核心，是让一个 **agent** 在某个 **environment** 中不断采取 **action**，环境根据当前状态和动作返回新的状态以及奖励 **reward**。agent 的目标不是只追求某一步的 reward 最大，而是让长期累计回报（return）尽可能大。

也就是说，RL 研究的是：

> 一个智能体如何通过与环境交互，学习出一套能带来更高长期收益的决策规则。

单步奖励记作 $R_{t+1}$，而长期回报通常写成：

$$
G_t = R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \cdots
$$

其中 $\gamma \in [0,1]$ 是折扣因子，用来控制 agent 对未来奖励的重视程度。

---

## 2. MDP：强化学习的基本数学框架

RL 通常被建模为一个马尔科夫决策过程（Markov Decision Process, **MDP**）。

一个 MDP 可以表示为五元组：

$$
\mathcal{M} = (\mathcal{S}, \mathcal{A}, P, R, \gamma)
$$

分别表示：

- $\mathcal{S}$：状态空间（state space）
- $\mathcal{A}$：动作空间（action space）
- $P(s'|s,a)$：状态转移概率（transition probability）
- $R(s,a,s')$：奖励函数（reward function）
- $\gamma$：折扣因子（discount factor）

MDP 的关键是 **马尔科夫性**：

$$
P(s_{t+1}|s_t,a_t)
$$

只依赖当前状态 $s_t$ 和当前动作 $a_t$，而不依赖更早之前的历史。也就是说，当前状态已经包含了做决策所需要的全部信息。

---

## 3. Policy：agent 怎么选择动作？

在一个状态 $s$ 下，agent 可以采取不同的动作 $a$。策略（policy）就是描述 agent 如何选择动作的函数，通常写作：

$$
\pi(a|s)
$$

它表示：在状态 $s$ 下选择动作 $a$ 的概率。

RL 的目标就是找到一个好的策略 $\pi$，使得 agent 在环境中获得的长期累计回报最大。

为了判断一个策略好不好，我们通常会引入两个价值函数：

### 状态价值函数

$$
V^\pi(s) = \mathbb{E}_\pi[G_t | S_t=s]
$$

它表示：在状态 $s$ 出发，并按照策略 $\pi$ 行动，未来能获得多少期望回报。

### 动作价值函数

$$
Q^\pi(s,a) = \mathbb{E}_\pi[G_t | S_t=s, A_t=a]
$$

它表示：在状态 $s$ 下先采取动作 $a$，之后继续按照策略 $\pi$ 行动，未来能获得多少期望回报。

简单来说：

- $V^\pi(s)$ 衡量“这个状态值不值得待”。
- $Q^\pi(s,a)$ 衡量“在这个状态下做这个动作值不值得”。

---

## 4. 强化学习的学习主线

强化学习的整体学习结构可以粗略归纳为：

$$
\text{MDP}
\rightarrow
\text{Bellman 方程}
\rightarrow
\text{动态规划}
\rightarrow
\text{采样估计}
\rightarrow
\text{TD 学习}
\rightarrow
\text{函数近似}
\rightarrow
\text{策略梯度}
\rightarrow
\text{Actor-Critic}
\rightarrow
\text{深度强化学习}
$$

这条线可以理解为：

1. 先用 MDP 描述强化学习问题。
2. 再用 Bellman 方程刻画价值函数的递归关系。
3. 如果已知环境模型，可以用动态规划求解。
4. 如果不知道环境模型，就通过采样估计价值。
5. TD 学习结合采样和 bootstrap，提升学习效率。
6. 当状态/动作空间太大时，用函数近似表示价值函数或策略。
7. 策略梯度直接优化策略。
8. Actor-Critic 同时结合策略方法和价值方法。
9. 深度强化学习用深度神经网络处理更复杂的环境。

---

## 5. Bellman 方程

Bellman 方程的核心思想是：

> 一个状态的价值 = 当前能得到的奖励 + 后续状态价值的折扣期望。

在给定策略 $\pi$ 的情况下，状态价值函数满足：

$$
V^\pi(s)
=
\sum_a \pi(a|s)
\sum_{s'} P(s'|s,a)
\left[ R(s,a,s') + \gamma V^\pi(s') \right]
$$

最优价值函数满足最优 Bellman 方程：

$$
V^*(s)
=
\max_a
\sum_{s'} P(s'|s,a)
\left[ R(s,a,s') + \gamma V^*(s') \right]
$$

Bellman 方程是强化学习里非常核心的递归结构。后面的动态规划、TD 学习、Q-learning、Actor-Critic 等方法，本质上都和这个递归关系有关。

---

## 6. 动态规划：已知环境模型时怎么求解？

动态规划（Dynamic Programming, DP）通常要求我们知道环境模型，也就是知道：

- 状态转移概率 $P(s'|s,a)$
- 奖励函数 $R(s,a,s')$

在这种情况下，可以利用 Bellman 方程对价值函数进行迭代优化。典型方法包括：

- policy evaluation：评估当前策略的价值。
- policy improvement：根据价值函数改进策略。
- policy iteration：策略评估和策略改进交替进行。
- value iteration：直接迭代最优价值函数。

不过，现实中很多环境的转移概率并不知道，所以就需要采样估计和 model-free 方法。

---

## 7. 采样估计：不知道概率怎么办？

如果无法知道具体的 $P(s'|s,a)$，就不能直接用动态规划计算期望。这时可以通过和环境交互得到样本，用样本平均来估计价值。

蒙特卡洛方法（Monte Carlo, MC）的思想是：

> 让 agent 跑完整条轨迹，然后用实际得到的 return 来估计状态或动作的价值。

例如，对某个状态 $s$，如果多次访问后得到的回报是 $G_1, G_2, \dots, G_n$，那么可以用平均值估计：

$$
V(s) \approx \frac{1}{n}\sum_{i=1}^n G_i
$$

MC 方法不需要知道环境模型，但通常要等一整个 episode 结束后才能更新。

---

## 8. TD 学习：采样 + bootstrap

TD 学习是时序差分学习（Temporal-Difference Learning）。它结合了蒙特卡洛的采样思想和动态规划的 bootstrap 思想。

典型的 TD(0) 更新公式是：

$$
V(S_t)
\leftarrow
V(S_t)
+
\alpha
\left[
R_{t+1} + \gamma V(S_{t+1}) - V(S_t)
\right]
$$

其中：

$$
\delta_t = R_{t+1} + \gamma V(S_{t+1}) - V(S_t)
$$

叫作 TD error。

TD 学习的关键优势是：不需要等整个 episode 结束，只要走一步，就可以用下一状态的估计值来更新当前状态的价值。

---

## 9. 函数近似：状态空间太大怎么办？

当状态空间或动作空间很小时，可以用表格保存每个状态或动作的价值。但在复杂任务中，状态数量可能极大，甚至是连续空间，这时就不能再维护一个完整表格。

函数近似就是用一个带参数的函数来表示价值函数或策略函数，例如：

$$
V_w(s) \approx V^\pi(s)
$$

$$
Q_w(s,a) \approx Q^\pi(s,a)
$$

$$
\pi_\theta(a|s) \approx \pi(a|s)
$$

这个函数可以是线性模型，也可以是神经网络。深度强化学习通常就是用深度神经网络来近似价值函数或策略函数。

---

## 10. 策略梯度：直接优化策略

价值方法通常是先学习 $V$ 或 $Q$，再根据价值函数选择动作。例如 Q-learning 会学习 $Q(s,a)$，然后倾向于选择 $Q$ 值更大的动作。

策略梯度方法则不同，它直接对策略 $\pi_\theta(a|s)$ 的参数 $\theta$ 进行优化，使期望回报最大：

$$
J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}[G(\tau)]
$$

策略梯度的基本形式可以写成：

$$
\nabla_\theta J(\theta)
=
\mathbb{E}
\left[
\nabla_\theta \log \pi_\theta(a|s) G_t
\right]
$$

加入优势函数后，常见形式是：

$$
\nabla_\theta J(\theta)
=
\mathbb{E}
\left[
\nabla_\theta \log \pi_\theta(a|s) A^\pi(s,a)
\right]
$$

优势函数定义为：

$$
A^\pi(s,a)=Q^\pi(s,a)-V^\pi(s)
$$

它表示：在状态 $s$ 下采取动作 $a$，比这个状态下的平均水平好多少。

所以策略梯度不是简单地“在每个状态下找概率最大的动作”，而是根据动作带来的长期回报，调整整个策略分布。

---

## 11. Actor-Critic：策略方法 + 价值方法

Actor-Critic 同时使用两个部分：

- Actor：负责选择动作，也就是策略 $\pi_\theta(a|s)$。
- Critic：负责评价动作或状态，也就是估计 $V(s)$、$Q(s,a)$ 或优势函数 $A(s,a)$。

Actor 根据 Critic 给出的评价来更新策略。常见更新形式是：

$$
\theta
\leftarrow
\theta
+
\alpha \nabla_\theta \log \pi_\theta(a|s) \hat{A}(s,a)
$$

Critic 通常可以用 TD error 来更新价值函数：

$$
\delta_t
=
r_{t+1}
+
\gamma V_w(s_{t+1})
-
V_w(s_t)
$$

简单来说：

> Actor 负责行动，Critic 负责评价；Critic 用价值估计指导 Actor，让 Actor 朝着更高长期回报的方向更新。

这也是很多现代强化学习算法的基础结构。

---

## 12. 深度强化学习与现代算法

深度强化学习（Deep Reinforcement Learning, DRL）就是把深度神经网络引入强化学习，用神经网络表示价值函数、策略函数或环境模型。

常见的深度强化学习算法包括：

- DQN：用深度网络近似 $Q$ 函数。
- DDPG / TD3：适合连续动作空间的 actor-critic 方法。
- SAC：带最大熵思想的 actor-critic 方法。
- TRPO：通过信赖域约束稳定策略更新。
- PPO：对 TRPO 做了更简单实用的近似，是非常常用的策略优化算法。

在大语言模型对齐中，也经常会看到这些名字：

- PPO：曾广泛用于 RLHF，通过 reward model 对语言模型策略进行强化学习优化。
- GRPO：一种面向大模型训练的策略优化方法，常见于推理模型训练语境。
- DPO：直接偏好优化方法，常用于偏好对齐，但它不属于传统意义上“通过环境交互采样 reward 再更新策略”的 RL 算法。


> PPO、TRPO、SAC 等是典型强化学习算法；DPO 更像是一种偏好优化方法；GRPO 则是大模型强化学习训练中常见的策略优化变体。

---

## 13. 总结

强化学习可以先抓住这几个核心问题：

1. 问题怎么表示？  
   用 MDP：$\mathcal{M} = (\mathcal{S}, \mathcal{A}, P, R, \gamma)$。

2. agent 怎么行动？  
   用策略：$\pi(a|s)$。

3. 怎么判断状态或动作好不好？  
   用价值函数：$V^\pi(s)$ 和 $Q^\pi(s,a)$。

4. 价值函数之间有什么递归关系？  
   用 Bellman 方程。

5. 不知道环境模型怎么办？  
   用采样估计、蒙特卡洛方法和 TD 学习。

6. 状态空间太大怎么办？  
   用函数近似，尤其是神经网络。

7. 如何直接优化策略？  
   用策略梯度和 Actor-Critic。
