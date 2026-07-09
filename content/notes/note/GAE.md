---
title: GAE
---

# GAE 学习笔记

## 摘要
GAE（Generalized Advantage Estimation）是一种用于估计优势函数的方法，核心目标是平衡偏差（bias）和方差（variance）。它常与 [[PPO]] 搭配使用，用来给策略更新提供更稳定的 advantage 信号。

## 核心思想
在策略优化中，我们希望知道某个动作到底“比平均水平好多少”，这就是 advantage。问题在于，直接用 Monte Carlo 回报虽然无偏，但方差很大；只用 1-step TD 又更稳定，但偏差会更明显。GAE 的作用就是在这两者之间做折中。

## 关键概念
- **Monte Carlo**：把整条轨迹的最终回报分配给当前动作，方差较大。
- **1-step TD**：只看一步奖励和下一时刻的价值估计，更稳定，但更依赖 Critic。
- **$\gamma$**：折扣因子，控制未来奖励的重要程度。
- **$\lambda$**：平滑系数，控制估计更接近 Monte Carlo 还是 TD。

## 关键公式
首先定义每一步的 TD 残差：

$$
\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)
$$

其中：
- $r_t$ 是第 $t$ 步的即时奖励；
- $V(s_t)$ 是当前状态的价值估计；
- $V(s_{t+1})$ 是下一状态的价值估计。

GAE 的优势函数写作：

$$
\hat A_t^{\mathrm{GAE}} = \sum_{l=0}^{\infty}(\gamma\lambda)^l \delta_{t+l}
$$

这表示：当前优势不仅看这一刻的 TD 残差，还会把未来若干步的残差按 $(\gamma\lambda)^l$ 递减加权后累加起来。

## 流程理解
1. 用 Critic 估计每个状态的价值 $V(s_t)$。
2. 计算每一步的 TD 残差 $\delta_t$。
3. 从后往前做折扣累加，得到每一步的 $\hat A_t^{\mathrm{GAE}}$。
4. 把这个 advantage 交给策略网络，用于更新 policy。

## 与其他方法的关系
- 当 $\lambda=0$ 时，GAE 更接近 1-step TD，方差更小，但偏差更大。
- 当 $\lambda=1$ 时，GAE 更接近 Monte Carlo，偏差更小，但方差更大。
- 在 [[PPO]] 中，GAE 常被用来构造 $\hat A_t$，让策略更新既稳定又有效。

## 小结
GAE 本质上是在“估计更准”和“波动更小”之间做权衡。理解它的关键，不是死记公式，而是明白它为什么能把多步 TD 信息平滑地整合成一个更适合策略优化的 advantage。
