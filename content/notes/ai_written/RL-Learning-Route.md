---
title: RL-Learning-Route
---

下面我按“导师式学习路线”给你搭一套强化学习理论体系。你可以把它当作长期学习纲要：先理解数学建模，再理解 Bellman 方程，再理解采样学习，最后进入函数近似、策略梯度和深度强化学习。

---

## 1. 先建立强化学习的总图景

强化学习研究的是：

> 一个智能体 agent 在环境 environment 中反复行动，根据奖励 reward 学习一个策略 policy，使长期累计回报最大。

它和监督学习最大的区别是：

|监督学习|强化学习|
|---|---|
|给定输入和正确标签|没有直接标签，只有奖励|
|样本通常独立同分布|经验数据由当前策略产生，强相关|
|目标是预测正确|目标是长期决策最优|
|错误通常即时可见|行动影响未来，存在延迟奖励|
|数据集固定|数据分布会随策略改变|

强化学习的核心不是“做一个动作拿奖励”，而是：

> 当前动作会改变未来状态，未来状态又影响未来奖励，所以要优化长期回报，而不是眼前奖励。

---

## 2. 你要掌握的数学前置知识

理论学习不需要一开始就非常高深，但下面这些一定要熟：

### 2.1 概率论

你需要熟悉：

$$
\mathbb{E}[X], \quad \mathbb{P}(A|B), \quad p(x), \quad p(x,y), \quad p(y|x)  
$$

尤其是条件期望：

$$
\mathbb{E}[X|Y]  
$$

强化学习里几乎所有价值函数都是条件期望。

### 2.2 线性代数

你要能看懂：

$$
V = R + \gamma PV  
$$

这里 (V) 是向量，(P) 是状态转移矩阵，(\gamma) 是折扣因子。

整理可得：

$$
V = (I - \gamma P)^{-1}R  
$$

这说明在有限状态空间、模型已知的情况下，价值函数可以被看作线性方程组的解。

### 2.3 微积分与优化

后面学策略梯度时，你需要知道：

$$
\nabla_\theta J(\theta) =
$$

也就是目标函数对参数 (\theta) 的梯度。

你还要熟悉随机梯度上升：

$$
\theta \leftarrow \theta + \alpha \nabla_\theta J(\theta)  
$$

强化学习里通常是最大化回报，所以很多时候是梯度上升，而不是梯度下降。

---

## 3. 强化学习的数学模型：MDP

强化学习理论的核心模型是马尔可夫决策过程，Markov Decision Process，简称 MDP。

一个 MDP 通常写作：

$$
\mathcal{M} = (\mathcal{S}, \mathcal{A}, P, R, \gamma)  
$$

其中：

|符号|含义|
|---|---|
|(\mathcal{S})|状态空间|
|(\mathcal{A})|动作空间|
|(P(s'|s,a))|
|(R(s,a)) 或 (R(s,a,s'))|奖励函数|
|(\gamma \in [0,1])|折扣因子|

### 3.1 马尔可夫性

马尔可夫性的意思是：

$$
P(s_{t+1}|s_t, a_t, s_{t-1}, a_{t-1}, \dots) = P(s_{t+1}|s_t,a_t)  
$$

也就是说，只要当前状态 (s_t) 足够完整，未来只依赖当前状态和当前动作，不依赖更早的历史。

这点极其重要。很多强化学习失败的根源，其实是状态设计不满足马尔可夫性。

---

## 4. 策略、轨迹与回报

### 4.1 策略 policy

策略表示智能体如何选择动作。

确定性策略：

$$
a = \pi(s)  
$$

随机策略：

$$
\pi(a|s) = P(a_t = a | s_t = s)  
$$

在理论学习中，随机策略更通用，因为它可以表达探索，也方便推导策略梯度。

### 4.2 轨迹 trajectory

一次完整交互可以写作：

$$
\tau = (s_0,a_0,r_1,s_1,a_1,r_2,\dots)  
$$

### 4.3 回报 return

从时刻 (t) 开始的累计折扣回报为：

$$
G_t = R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \cdots  
$$

也可以写成：

$$
G_t = \sum_{k=0}^{\infty} \gamma^k R_{t+k+1}  
$$

折扣因子 (\gamma) 的作用有三个：

第一，让远期奖励权重降低。

第二，在无限时域问题中保证累计回报有限。

第三，更重要的是，它让 Bellman 算子成为压缩映射，从而保证很多算法收敛。

---

## 5. 价值函数：强化学习理论的中心

强化学习理论最核心的概念是价值函数。

### 5.1 状态价值函数

在策略 (\pi) 下，状态 (s) 的价值定义为：

$$
V^\pi(s) = \mathbb{E}_\pi[G_t | S_t = s]  
$$

意思是：

> 如果现在处于状态 (s)，之后一直按照策略 (\pi) 行动，长期平均能拿到多少回报。

### 5.2 动作价值函数

$$
Q^\pi(s,a) = \mathbb{E}_\pi[G_t | S_t = s, A_t = a]  
$$

意思是：

> 如果现在处于状态 (s)，先做动作 (a)，之后按照策略 (\pi) 行动，长期平均能拿到多少回报。

### 5.3 优势函数

$$
A^\pi(s,a) = Q^\pi(s,a) - V^\pi(s)  
$$

优势函数表示：

> 在状态 (s) 下，动作 (a) 比策略平均水平好多少。

策略梯度、Actor-Critic、PPO 里经常用优势函数。

---

## 6. Bellman 方程：强化学习理论的主干

Bellman 方程是强化学习里最重要的方程。你可以把它理解为：

> 一个状态的价值 = 当前一步奖励 + 下一状态价值的折扣期望。

### 6.1 Bellman 期望方程

对状态价值函数：

$$
V^\pi(s) =

\sum_a \pi(a|s)  
\sum_{s'} P(s'|s,a)  
\left[  
R(s,a,s') + \gamma V^\pi(s')  
\right]  
$$

对动作价值函数：

$$
Q^\pi(s,a) =

\sum_{s'} P(s'|s,a)  
\left[  
R(s,a,s') + \gamma \sum_{a'} \pi(a'|s')Q^\pi(s',a')  
\right]  
$$

这两个方程解决的是 prediction 问题：

> 给定一个策略 (\pi)，估计它到底好不好。

### 6.2 Bellman 最优方程

最优状态价值函数：

$$
V^*(s) =

\max_a  
\sum_{s'}P(s'|s,a)  
\left[  
R(s,a,s') + \gamma V^*(s')  
\right]  
$$

最优动作价值函数：

$$
Q^*(s,a) =

\sum_{s'}P(s'|s,a)  
\left[  
R(s,a,s') + \gamma \max_{a'}Q^*(s',a')  
\right]  
$$

这解决的是 control 问题：

> 如何找到最优策略。

一旦你有了 (Q^*(s,a))，最优策略就是：

$$
\pi^*(s) = \arg\max_a Q^*(s,a)  
$$

---

## 7. 强化学习的两大问题：Prediction 与 Control

你要始终区分两个问题。

### 7.1 Prediction：评估策略

给定策略 (\pi)，求：

$$
V^\pi(s), \quad Q^\pi(s,a)  
$$

例如：已知一个机器人当前的导航策略，问它平均能不能走到目标点。

### 7.2 Control：寻找最优策略

寻找：

$$
\pi^* = \arg\max_\pi V^\pi(s)  
$$

例如：不仅要评估机器人现有策略，还要找到更好的导航策略。

很多算法都可以放进这个框架：

|问题|典型算法|
|---|---|
|Prediction|Monte Carlo prediction, TD prediction|
|Control|SARSA, Q-learning, Policy Iteration, Value Iteration|
|直接优化策略|REINFORCE, Actor-Critic, PPO|
|连续动作控制|DDPG, TD3, SAC|

---

## 8. 动态规划：模型已知时的精确方法

动态规划 Dynamic Programming 适用于模型已知的情况，也就是你知道：

$$
P(s'|s,a), \quad R(s,a,s')  
$$

### 8.1 策略评估 Policy Evaluation

目标：给定 (\pi)，求 (V^\pi)。

迭代公式：

$$
V_{k+1}(s) =

\sum_a \pi(a|s)  
\sum_{s'}P(s'|s,a)  
\left[  
R(s,a,s')+\gamma V_k(s')  
\right]  
$$

这就是反复应用 Bellman 期望算子。

### 8.2 策略改进 Policy Improvement

有了 (V^\pi) 之后，可以构造更贪心的新策略：

$$
\pi'(s) = \arg\max_a  
\sum_{s'}P(s'|s,a)  
\left[  
R(s,a,s')+\gamma V^\pi(s')  
\right]  
$$

策略改进定理告诉我们，新策略不会比旧策略差。

### 8.3 策略迭代 Policy Iteration

流程是：

$$
\pi_0  
\rightarrow V^{\pi_0}  
\rightarrow \pi_1  
\rightarrow V^{\pi_1}  
\rightarrow \pi_2  
\rightarrow \cdots  
$$

也就是：

1. 先评估当前策略；
    
2. 再改进策略；
    
3. 重复直到稳定。
    

### 8.4 价值迭代 Value Iteration

价值迭代把评估和改进合在一起：

$$
V_{k+1}(s) =

\max_a  
\sum_{s'}P(s'|s,a)  
\left[  
R(s,a,s')+\gamma V_k(s')  
\right]  
$$

最后得到最优价值函数 (V^*)。

---

## 9. 蒙特卡洛方法：不知道模型时，用采样估计

现实里通常不知道 (P) 和 (R)，只能和环境交互，采样得到经验。

蒙特卡洛 Monte Carlo 方法的思想是：

> 等一条 episode 结束后，用实际得到的回报 (G_t) 来估计价值。

例如：

$$
V(s) \leftarrow V(s) + \alpha [G_t - V(s)]  
$$

这里 (G_t) 是真实采样得到的完整回报。

### 9.1 蒙特卡洛的特点

优点：

- 不需要环境模型；
    
- 目标 (G_t) 是真实回报的采样，直观；
    
- 不使用 bootstrap，因此估计偏差较小。
    

缺点：

- 必须等 episode 结束；
    
- 方差大；
    
- 对长时域问题效率低。
    

---

## 10. 时序差分学习：强化学习的核心算法思想

Temporal Difference，简称 TD，是强化学习最关键的学习机制。

TD 的思想是：

> 不等 episode 结束，只用一步奖励和下一状态的估计值来更新当前价值。

TD(0) 更新：

$$
V(S_t)  
\leftarrow  
V(S_t)  
+  
\alpha  
\left[  
R_{t+1}  
+  
\gamma V(S_{t+1})

- V(S_t)  
\right]  
$$

括号中的部分叫 TD error：

$$
\delta_t =  
R_{t+1}  
+  
\gamma V(S_{t+1})

- V(S_t)  
$$

TD error 可以理解为：

> 现实观察到的一步结果，和自己原先预期之间的差距。

---

## 11. SARSA 与 Q-learning

这两个算法非常重要，必须彻底分清。

### 11.1 SARSA：on-policy

SARSA 的名字来自：

$$
S_t, A_t, R_{t+1}, S_{t+1}, A_{t+1}  
$$

更新公式：

$$
Q(S_t,A_t)  
\leftarrow  
Q(S_t,A_t)  
+  
\alpha  
\left[  
R_{t+1}  
+  
\gamma Q(S_{t+1},A_{t+1})

- Q(S_t,A_t)  
\right]  
$$

SARSA 学的是当前行为策略本身的价值，所以是 on-policy。

如果你的行为策略是 (\epsilon)-greedy，那么 SARSA 学到的也是包含探索行为的策略价值。

### 11.2 Q-learning：off-policy

Q-learning 更新公式：

$$
Q(S_t,A_t)  
\leftarrow  
Q(S_t,A_t)  
+  
\alpha  
\left[  
R_{t+1}  
+  
\gamma \max_{a'} Q(S_{t+1},a')

- Q(S_t,A_t)  
\right]  
$$

它的行为策略可以是 (\epsilon)-greedy，但更新目标使用的是贪心策略：

$$
\max_{a'} Q(S_{t+1},a')  
$$

所以 Q-learning 是 off-policy。

### 11.3 二者差异

|对比|SARSA|Q-learning|
|---|---|---|
|类型|on-policy|off-policy|
|下一动作|实际执行的 (A_{t+1})|贪心动作 (\arg\max Q)|
|学习对象|当前探索策略|最优贪心策略|
|风格|更保守|更激进|
|经典例子|Cliff Walking 中更安全|Cliff Walking 中可能更贴边|

---

## 12. Bias-Variance 视角：MC、TD、n-step、TD(λ)

你要从偏差-方差角度理解这些算法。

|方法|目标|偏差|方差|
|---|---|---|---|
|Monte Carlo|完整回报 (G_t)|低|高|
|TD(0)|一步 bootstrap|较高|低|
|n-step TD|n 步回报|中等|中等|
|TD(λ)|多个 n-step 的加权平均|可调|可调|

n-step return：

$$
G_t^{(n)} =

R_{t+1}  
+  
\gamma R_{t+2}  
+  
\cdots  
+  
\gamma^{n-1}R_{t+n}  
+  
\gamma^n V(S_{t+n})  
$$

TD(λ) 可以看作对不同 (n)-step return 的加权平均。

$$
G_t^\lambda =

(1-\lambda)  
\sum_{n=1}^{\infty}  
\lambda^{n-1}G_t^{(n)}  
$$

当：

$$
\lambda = 0  
$$

接近 TD(0)。

当：

$$
\lambda = 1  
$$

接近 Monte Carlo。

---

## 13. 函数近似：从表格强化学习到大状态空间

前面的算法默认你能为每个状态或状态-动作对存一个值：

$$
V(s), \quad Q(s,a)  
$$

但现实问题状态空间巨大，甚至连续，无法建表。

于是用函数近似：

$$
V(s;w) \approx V^\pi(s)  
$$

或者：

$$
Q(s,a;w) \approx Q^\pi(s,a)  
$$

其中 (w) 是参数，可以是线性模型，也可以是神经网络。

### 13.1 线性价值函数近似

$$
V(s;w) = w^\top x(s)  
$$

其中 (x(s)) 是状态特征。

更新可以写成：

$$
w \leftarrow w + \alpha \delta_t \nabla_w V(s;w)  
$$

如果是线性函数：

$$
\nabla_w V(s;w) = x(s)  
$$

所以：

$$
w \leftarrow w + \alpha \delta_t x(s)  
$$

### 13.2 函数近似带来的危险：Deadly Triad

强化学习里有一个非常重要的概念：致命三角 Deadly Triad。

它指的是三件事同时出现时，算法可能不稳定甚至发散：

1. 函数近似；
    
2. bootstrapping；
    
3. off-policy learning。
    

DQN、TD3、SAC 等深度强化学习算法的很多工程设计，本质上都是为了缓解这些不稳定性。

---

## 14. DQN：深度价值函数方法

DQN 的思想是用神经网络近似 Q 函数：

$$
Q(s,a;\theta)  
$$

Q-learning 的目标变成：

$$
y =  
r + \gamma \max_{a'} Q(s',a';\theta^-)  
$$

损失函数：

$$
L(\theta) =

\left(  
y - Q(s,a;\theta)  
\right)^2  
$$

其中 (\theta^-) 是 target network 的参数。

DQN 的关键机制：

|机制|作用|
|---|---|
|Experience Replay|打破样本强相关，提高数据利用率|
|Target Network|稳定 bootstrap 目标|
|(\epsilon)-greedy|保持探索|
|Clipping / normalization|缓解训练不稳定|

DQN 适合离散动作空间。如果动作是连续的，直接对动作求 (\max_a Q(s,a)) 会很困难，因此通常要用 Actor-Critic 类方法。

---

## 15. 策略梯度：直接优化策略

价值方法是先学 (Q)，再选动作：

$$
a = \arg\max_a Q(s,a)  
$$

策略梯度方法则直接参数化策略：

$$
\pi_\theta(a|s)  
$$

目标函数：

$$
J(\theta) = \mathbb{E}_{\pi_\theta}[G_0]  
$$

希望求：

$$
\nabla_\theta J(\theta) =
$$

### 15.1 Policy Gradient Theorem

核心公式是：

$$
\nabla_\theta J(\theta) =

\mathbb{E}_{\pi_\theta}  
\left[  
\nabla_\theta \log \pi_\theta(a|s)  
Q^{\pi_\theta}(s,a)  
\right]  
$$

这非常重要。它说明：

> 如果某个动作带来的 (Q) 高，就提高该动作在该状态下的概率；如果 (Q) 低，就降低概率。

### 15.2 REINFORCE

REINFORCE 用完整回报 (G_t) 替代 (Q^\pi(s,a))：

$$
\theta
\leftarrow
\theta
+
\alpha G_t \nabla_\theta \log \pi_\theta(A_t|S_t)
$$

它简单但方差很大。

### 15.3 Baseline

为了降低方差，可以减去 baseline：

$$
\nabla_\theta J(\theta) =

\mathbb{E}  
\left[  
\nabla_\theta \log \pi_\theta(a|s)  
\left(  
Q^\pi(s,a)-b(s)  
\right)  
\right]  
$$

常用 baseline 是：

$$
b(s)=V^\pi(s)  
$$

于是得到优势函数：

$$
A^\pi(s,a)=Q^\pi(s,a)-V^\pi(s)  
$$

---

## 16. Actor-Critic：价值方法与策略方法的结合

Actor-Critic 有两个部分：

|部分|作用|
|---|---|
|Actor|学策略 (\pi_\theta(a|
|Critic|学价值函数 (V_w(s)) 或 (Q_w(s,a))|

Actor 根据 Critic 给出的优势估计来更新策略：

$$
\theta
\leftarrow
\theta
+
\alpha \nabla_\theta \log \pi_\theta(a|s) \hat{A}(s,a)
$$

Critic 用 TD error 学价值：

$$
\delta_t =  
r_{t+1}  
+  
\gamma V_w(s_{t+1})

- V_w(s_t)  
$$

Actor-Critic 的直觉是：

> Critic 负责评价动作好坏，Actor 负责调整行为方式。

---

## 17. PPO、TRPO、SAC：现代深度强化学习核心算法

### 17.1 TRPO

TRPO 的核心思想是限制每次策略更新不要太大。

因为如果策略变化太剧烈，新策略采样分布和旧策略差别太大，训练会不稳定。

TRPO 通常约束 KL divergence：

$$
D_{KL}(\pi_{\theta_{\text{old}}} || \pi_\theta)  
$$

### 17.2 PPO

PPO 是 TRPO 的简化实用版本。它使用 clipped objective：

$$
L^{CLIP}(\theta) =

\mathbb{E}  
\left[  
\min  
\left(  
r_t(\theta)\hat{A}_t,  
\text{clip}(r_t(\theta),1-\epsilon,1+\epsilon)\hat{A}_t  
\right)  
\right]  
$$

其中：

$$
r_t(\theta) =

\frac{\pi_\theta(a_t|s_t)}  
{\pi_{\theta_{\text{old}}}(a_t|s_t)}  
$$

PPO 的直觉是：

> 如果新策略相对旧策略变化太大，就截断收益，防止策略更新过猛。

### 17.3 SAC

SAC，Soft Actor-Critic，是一种最大熵强化学习方法。

它优化的不只是奖励，还鼓励策略保持随机性：

$$
\mathbb{E}  
\left[  
\sum_t \gamma^t  
\left(  
r_t + \alpha \mathcal{H}(\pi(\cdot|s_t))  
\right)  
\right]  
$$

直觉是：

> 不仅要拿高奖励，还要保持足够探索性。

SAC 在连续控制任务里非常常用。

---

## 18. Model-Free 与 Model-Based

### 18.1 Model-Free

Model-Free 不显式学习环境模型，直接学价值函数或策略。

代表：

- SARSA
    
- Q-learning
    
- DQN
    
- REINFORCE
    
- PPO
    
- SAC
    

优点：简单、通用。

缺点：样本效率通常较低。

### 18.2 Model-Based

Model-Based 学或使用环境模型：

$$
\hat{P}(s'|s,a), \quad \hat{R}(s,a)  
$$

然后用模型做规划。

代表：

- Dynamic Programming
    
- Dyna
    
- Model Predictive Control
    
- Monte Carlo Tree Search
    
- MuZero 类方法
    

优点：样本效率可能更高。

缺点：模型误差可能导致规划偏差。

---

## 19. On-policy 与 Off-policy

### 19.1 On-policy

学习的策略和采样的策略相同。

典型算法：

- SARSA
    
- REINFORCE
    
- A2C
    
- PPO
    

优点：稳定、理论清晰。

缺点：样本利用率较低。

### 19.2 Off-policy

采样策略和学习目标策略不同。

典型算法：

- Q-learning
    
- DQN
    
- DDPG
    
- TD3
    
- SAC
    

优点：可以复用旧数据，样本效率较高。

缺点：更容易不稳定，尤其结合函数近似时。

---

## 20. 探索与利用：强化学习绕不开的问题

强化学习必须平衡 exploration 与 exploitation。

### 20.1 (\epsilon)-greedy

以 (1-\epsilon) 的概率选当前最优动作，以 (\epsilon) 的概率随机探索：

$$
a =  
\begin{cases}  
\arg\max_a Q(s,a), & \text{probability } 1-\epsilon \  
\text{random action}, & \text{probability } \epsilon  
\end{cases}  
$$

### 20.2 Softmax exploration

$$
P(a|s)
= \frac{\exp(Q(s,a)/\tau)}
{\sum_{a'}\exp(Q(s,a')/\tau)}
$$

温度 (\tau) 越大，越随机；(\tau) 越小，越贪心。

### 20.3 UCB

Upper Confidence Bound 的思想是：

> 不仅看当前估计值，还要看不确定性。

常见形式：

$$
a_t
= \arg\max_a
\left[
Q(a) + c\sqrt{\frac{\log t}{N(a)}}
\right]
$$

### 20.4 Thompson Sampling

从后验分布中采样一个模型，再基于采样模型选择最优动作。

它在 bandit 问题中非常重要。

---

## 21. 你应该如何安排理论学习路线

我建议分成 6 个阶段。

---

### 阶段 1：Bandit 与强化学习基本概念

目标：理解“试错学习”和“探索-利用”。

要掌握：

- multi-armed bandit
    
- action value
    
- sample average
    
- (\epsilon)-greedy
    
- UCB
    
- regret
    

重点问题：

1. 为什么总选当前最优动作可能不是最优？
    
2. 为什么探索有成本，但不探索会错过更优动作？
    
3. regret 衡量的是什么？
    

练习：

- 手推两臂老虎机的 sample average 更新；
    
- 比较 (\epsilon=0)、(\epsilon=0.1)、(\epsilon=0.01) 的长期行为；
    
- 解释 UCB 中不确定性奖励项为什么会随 (N(a)) 增大而下降。
    

---

### 阶段 2：MDP 与 Bellman 方程

目标：掌握强化学习的数学语言。

要掌握：

- MDP
    
- policy
    
- return
    
- (V^\pi)
    
- (Q^\pi)
    
- Bellman expectation equation
    
- Bellman optimality equation
    

重点问题：

1. 什么是马尔可夫性？
    
2. (V^\pi) 和 (Q^\pi) 有什么区别？
    
3. 为什么 Bellman 方程是递归方程？
    
4. 为什么最优价值函数满足 max 形式？
    

练习：

- 给一个 3 状态 MDP，手写 (P_\pi)、(R_\pi)、(V^\pi)；
    
- 用线性方程组解 (V^\pi)；
    
- 推导 Bellman optimality equation。
    

---

### 阶段 3：动态规划

目标：在模型已知时求最优策略。

要掌握：

- policy evaluation
    
- policy improvement
    
- policy iteration
    
- value iteration
    
- Bellman operator
    
- contraction mapping
    

重点问题：

1. policy iteration 为什么会越来越好？
    
2. value iteration 为什么可以收敛？
    
3. 策略评估和策略改进分别解决什么？
    

练习：

- 手算一个小 gridworld；
    
- 证明 Bellman expectation operator 是 (\gamma)-contraction；
    
- 比较 policy iteration 和 value iteration。
    

---

### 阶段 4：Monte Carlo 与 TD Learning

目标：理解没有模型时如何从经验中学习。

要掌握：

- Monte Carlo prediction
    
- TD(0)
    
- TD error
    
- SARSA
    
- Q-learning
    
- Expected SARSA
    
- n-step TD
    
- TD(λ)
    

重点问题：

1. MC 为什么方差大？
    
2. TD 为什么有 bootstrap？
    
3. SARSA 和 Q-learning 为什么一个保守一个激进？
    
4. TD error 的意义是什么？
    

练习：

- 手算一条 trajectory 上的 MC return；
    
- 手算 TD(0) 更新；
    
- 在 Cliff Walking 中解释 SARSA 和 Q-learning 的策略差异；
    
- 比较 1-step、n-step、MC 的目标。
    

---

### 阶段 5：函数近似与深度 Q-learning

目标：从表格方法过渡到大规模状态空间。

要掌握：

- linear value approximation
    
- semi-gradient TD
    
- off-policy instability
    
- deadly triad
    
- DQN
    
- target network
    
- experience replay
    
- Double DQN
    
- Dueling DQN
    

重点问题：

1. 为什么表格法不能处理高维状态？
    
2. 为什么函数近似会带来泛化，也带来不稳定？
    
3. DQN 为什么需要 target network？
    
4. experience replay 为什么能提高稳定性？
    

练习：

- 写出 (V(s;w)=w^\top x(s)) 的 TD 更新；
    
- 解释 Q-learning + 神经网络为什么可能发散；
    
- 对比 DQN 和 tabular Q-learning。
    

---

### 阶段 6：策略梯度与 Actor-Critic

目标：理解现代深度强化学习算法。

要掌握：

- policy gradient theorem
    
- REINFORCE
    
- baseline
    
- advantage function
    
- Actor-Critic
    
- GAE
    
- TRPO
    
- PPO
    
- SAC
    

重点问题：

1. 为什么可以对采样轨迹的 log probability 求梯度？
    
2. baseline 为什么不改变梯度期望？
    
3. Actor 和 Critic 各自负责什么？
    
4. PPO 为什么要限制策略更新幅度？
    
5. SAC 为什么要加入熵项？
    

练习：

- 推导 REINFORCE；
    
- 证明 baseline 不引入偏差；
    
- 写出 Actor-Critic 的 actor 更新和 critic 更新；
    
- 解释 PPO clipped objective 的直觉。
    

---

## 22. 一份 12 周学习计划

### 第 1 周：强化学习总览与 Bandit

任务：

- 理解 agent、environment、reward、return；
    
- 学 multi-armed bandit；
    
- 掌握 (\epsilon)-greedy、UCB、regret。
    

产出：

- 能解释探索-利用矛盾；
    
- 能手算 sample average 更新。
    

---

### 第 2 周：MDP

任务：

- 学状态、动作、转移概率、奖励函数；
    
- 理解马尔可夫性；
    
- 写出 MDP 五元组。
    

产出：

- 能把一个现实问题抽象成 MDP；
    
- 能判断某个状态定义是否满足马尔可夫性。
    

---

### 第 3 周：价值函数与 Bellman 方程

任务：

- 掌握 (V^\pi)、(Q^\pi)、(A^\pi)；
    
- 推导 Bellman expectation equation；
    
- 推导 Bellman optimality equation。
    

产出：

- 能独立写出 Bellman 方程；
    
- 能解释为什么 Bellman 方程表达了“当前 + 未来”。
    

---

### 第 4 周：动态规划

任务：

- 学 policy evaluation；
    
- 学 policy improvement；
    
- 学 policy iteration；
    
- 学 value iteration。
    

产出：

- 能手算小型 MDP；
    
- 能解释策略迭代和价值迭代的区别。
    

---

### 第 5 周：Monte Carlo

任务：

- 学 MC prediction；
    
- 学 first-visit 与 every-visit；
    
- 学 MC control；
    
- 学 exploring starts 与 (\epsilon)-soft policy。
    

产出：

- 能从一条 episode 估计价值；
    
- 能解释 MC 的高方差问题。
    

---

### 第 6 周：TD Learning

任务：

- 学 TD(0)；
    
- 学 TD error；
    
- 学 SARSA；
    
- 学 Q-learning；
    
- 学 Expected SARSA。
    

产出：

- 能比较 MC 与 TD；
    
- 能比较 SARSA 与 Q-learning。
    

---

### 第 7 周：n-step 与 TD(λ)

任务：

- 学 n-step return；
    
- 学 eligibility traces；
    
- 学 forward view 与 backward view；
    
- 学 TD(λ)。
    

产出：

- 能解释 (\lambda) 如何控制 MC 与 TD 之间的折中。
    

---

### 第 8 周：函数近似

任务：

- 学线性价值函数近似；
    
- 学 semi-gradient 方法；
    
- 学 projected Bellman error；
    
- 理解 deadly triad。
    

产出：

- 能写出线性 TD 更新；
    
- 能解释 off-policy + bootstrapping + function approximation 为什么危险。
    

---

### 第 9 周：DQN 与深度价值方法

任务：

- 学 DQN；
    
- 学 replay buffer；
    
- 学 target network；
    
- 学 Double DQN；
    
- 学 Dueling DQN。
    

产出：

- 能从 Q-learning 推导出 DQN loss；
    
- 能解释 DQN 的稳定化技巧。
    

---

### 第 10 周：Policy Gradient

任务：

- 学 trajectory probability；
    
- 学 log-derivative trick；
    
- 推导 REINFORCE；
    
- 学 baseline 和 advantage。
    

产出：

- 能推导 policy gradient theorem 的核心形式；
    
- 能解释 baseline 为什么降低方差。
    

---

### 第 11 周：Actor-Critic 与 PPO

任务：

- 学 Actor-Critic；
    
- 学 A2C/A3C 思想；
    
- 学 GAE；
    
- 学 TRPO 与 PPO。
    

产出：

- 能解释 PPO clipped objective；
    
- 能说明 Actor-Critic 相比 REINFORCE 的优势。
    

---

### 第 12 周：SAC、Model-Based、Offline RL 总览

任务：

- 学最大熵强化学习；
    
- 学 SAC；
    
- 学 Model-Based RL；
    
- 学 Offline RL；
    
- 了解 imitation learning 与 inverse RL。
    

产出：

- 能画出现代 RL 算法谱系；
    
- 能判断一个任务适合用 value-based、policy-gradient、actor-critic、model-based 还是 offline RL。
    

---

## 23. 推荐资料顺序

第一主线建议用 Sutton & Barto 的 _Reinforcement Learning: An Introduction_ 第二版。官方页面提供第二版信息、PDF、勘误、代码、slides 等配套材料，适合作为理论主教材。([不完整想法](https://incompleteideas.net/book/the-book.html?utm_source=chatgpt.com "Sutton & Barto Book: Reinforcement Learning: An Introduction"))

第二主线可以配 David Silver 的强化学习课程。他的教学页面列出了从 introduction、MDP、dynamic programming、model-free prediction/control、value function approximation、policy gradient、planning、exploration 到 classic games 的 10 讲，并提供 Easy21 assignment 和往年考试题。([David Silver](https://davidstarsilver.wordpress.com/teaching/ "Teaching – David Silver"))

进入深度强化学习后，可以看 OpenAI Spinning Up。它的官方文档包含 RL 关键概念、算法分类、policy optimization 介绍，以及 VPG、TRPO、PPO、DDPG、TD3、SAC 等算法文档。([Spinning Up](https://spinningup.openai.com/en/latest/ "Welcome to Spinning Up in Deep RL! — Spinning Up  documentation"))

更进阶时，可以看 Berkeley CS285。课程页面包含 imitation learning、RL basics、policy gradients、actor-critic、value-based RL、advanced policy gradients、model-based RL、offline RL、exploration、RL theory、multi-task RL 等主题，并列出作业和 lecture slides。([贝尔卡学院机器人与人工智能实验室](https://rail.eecs.berkeley.edu/deeprlcourse/ "CS 185/285"))

建议顺序是：

$$
\text{Sutton & Barto}  
\rightarrow  
\text{David Silver}  
\rightarrow  
\text{Spinning Up}  
\rightarrow  
\text{CS285}  
$$

不要一开始就直接冲 PPO、SAC、RLHF。先把 Bellman 方程、TD error、on-policy/off-policy、policy gradient theorem 搞懂，否则后面会变成背算法名。

---

## 24. 每学一个算法，都用这个模板检查自己

你学任何强化学习算法时，都问自己 8 个问题：

1. 它解决 prediction 还是 control？
    
2. 它是 model-free 还是 model-based？
    
3. 它是 on-policy 还是 off-policy？
    
4. 它学的是 (V)、(Q)，还是直接学 (\pi)？
    
5. 它的更新目标 target 是什么？
    
6. 它是否使用 bootstrap？
    
7. 它的主要偏差和方差来源是什么？
    
8. 它在什么情况下可能不稳定？
    

例如 Q-learning：

|问题|答案|
|---|---|
|prediction/control|control|
|model-free/model-based|model-free|
|on/off-policy|off-policy|
|学什么|(Q(s,a))|
|target|(r+\gamma\max_{a'}Q(s',a'))|
|bootstrap|是|
|风险|max 过估计、函数近似不稳定|
|适用|离散动作，或可枚举动作|

---

## 25. 强化学习理论最重要的几条主线

### 主线一：Bellman 递归

几乎所有价值方法都来自：

$$
\text{value} = \text{immediate reward} + \gamma \times \text{future value}  
$$

### 主线二：Bootstrap

TD、SARSA、Q-learning、DQN 都在使用估计值更新估计值：

$$
\text{target} = r + \gamma \hat{V}(s')  
$$

bootstrap 提高效率，但会引入偏差和不稳定性。

### 主线三：Generalized Policy Iteration

很多 RL 算法都可以看成：

1. policy evaluation：估计当前策略的价值；
    
2. policy improvement：根据价值改进策略。
    

即使是 Q-learning、Actor-Critic，也可以从这个角度理解。

### 主线四：采样替代模型

动态规划需要 (P) 和 (R)。

Monte Carlo 和 TD 不知道模型，于是用采样：

$$
s,a,r,s'  
$$

来近似 Bellman 更新。

### 主线五：函数近似带来泛化与不稳定

表格法稳定但不能扩展。

神经网络能处理高维状态，但会引入非平稳目标、相关样本、分布偏移和发散风险。

### 主线六：策略梯度直接优化行为

价值方法间接得到策略，策略梯度直接优化：

$$
\pi_\theta(a|s)  
$$

这使它适合连续动作、高维动作和随机策略建模。

---

## 26. 常见误区

### 误区一：奖励高等于价值高

不一定。

奖励是即时的：

$$
R_{t+1}  
$$

价值是长期的：

$$
V^\pi(s)=\mathbb{E}[G_t|S_t=s]  
$$

一个动作眼前奖励高，但可能导致未来很差。

---

### 误区二：Q-learning 总比 SARSA 好

不一定。

Q-learning 学贪心最优策略，可能更激进。

SARSA 学实际探索策略，可能更安全。

在有风险路径的任务里，SARSA 可能表现更稳。

---

### 误区三：深度强化学习就是 DQN/PPO/SAC

不是。

DQN、PPO、SAC 只是算法。真正底层的是：

- MDP；
    
- Bellman 方程；
    
- TD error；
    
- policy gradient；
    
- exploration；
    
- distribution shift；
    
- function approximation stability。
    

---

### 误区四：强化学习只要调参就行

强化学习非常依赖问题建模。

奖励设计、状态定义、动作空间设计、终止条件、探索机制，往往比换算法更重要。

---

### 误区五：off-policy 一定样本效率高，所以一定更好

off-policy 能复用数据，但也更容易出现分布偏移和训练不稳定。

尤其在深度强化学习中，off-policy 算法通常需要 replay buffer、target network、conservative updates 等技巧。

---

## 27. 你应该达到的几个能力层级

### 入门合格

你能解释：

- 什么是 MDP；
    
- 什么是 (V^\pi)、(Q^\pi)；
    
- Bellman 方程是什么；
    
- MC 和 TD 有什么区别；
    
- SARSA 和 Q-learning 有什么区别。
    

### 理论扎实

你能推导：

- Bellman expectation equation；
    
- Bellman optimality equation；
    
- policy improvement theorem；
    
- TD update；
    
- Q-learning update；
    
- policy gradient theorem；
    
- baseline 不改变梯度期望。
    

### 深度强化学习入门

你能解释：

- DQN loss；
    
- target network；
    
- replay buffer；
    
- Actor-Critic；
    
- advantage；
    
- PPO clipping；
    
- SAC entropy objective；
    
- deadly triad。
    

### 研究准备

你能讨论：

- exploration 的理论难点；
    
- sample efficiency；
    
- offline RL 的 distribution shift；
    
- model-based RL 的 model bias；
    
- RLHF 与传统 RL 的关系；
    
- safety、robustness、generalization。
    

---

## 28. 最后给你一条学习建议

强化学习不要按“算法名字”学，而要按“问题结构”学：

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

每学一个算法，都把它放回这张地图里。你会发现强化学习不是一堆零散算法，而是一套围绕“长期决策最优”的统一理论体系。


好的，我们可以在前面强化学习理论体系的基础上，把关键概念和算法进行深入拆解，重点讲解其数学本质、直觉理解以及常见陷阱。下面我会按照几个核心模块进行深度细化。

---

## 1. **Bellman 方程的深入理解**

### 核心思想

Bellman 方程体现了 **动态最优性原理**：

> 一个状态的最优价值等于当前奖励 + 未来价值的期望。

数学表达：

$$
V^\pi(s) = \mathbb{E}_\pi [ R_{t+1} + \gamma V^\pi(S_{t+1}) \mid S_t = s]  
$$

重点理解：

1. **递归性**：Bellman 方程是自我引用的。
    
2. **压缩映射**：Bellman 算子 $(T^\pi(V) = R + \gamma P^\pi V)$ 是 γ-压缩映射，从而保证迭代收敛。
    
3. **连续性与不连续性**：在离散状态下收敛快；在连续状态下必须用函数逼近或积分形式。
    

#### 直觉示例

- 状态价值函数 = 眼前奖励 + 对未来状态价值的加权平均。
    
- 对于最优状态价值函数 (V^*(s))，选择使未来价值最大化的动作。
    

---

## 2. **TD Error 与采样更新机制**

TD Error 是 RL 核心：

$$
\delta_t = R_{t+1} + \gamma V(S_{t+1}) - V(S_t)  
$$

### 深入点

1. **偏差-方差折中**
    
    - MC 方法：无偏但高方差。
        
    - TD(0)：有偏但低方差。
        
    - TD(λ)：可调 λ 实现偏差与方差平衡。
        
2. **Bootstrapping 的意义**
    
    - 用当前估计值更新自己。
        
    - 提高效率，但引入估计偏差。
        
3. **实战理解**
    
    - TD error 是“实际结果 - 预期”的差距。
        
    - Actor-Critic 的 Critic 正是学习 TD error 来指导 Actor 更新。
        

---

## 3. **On-policy vs Off-policy 深度理解**

|特性|On-policy|Off-policy|
|---|---|---|
|学习策略|行为策略 = 目标策略|行为策略 ≠ 目标策略|
|例子|SARSA, REINFORCE|Q-learning, DQN, SAC|
|核心风险|样本效率低|分布偏移导致不稳定|

### 核心陷阱

- Off-policy + 函数逼近 + bootstrap = **Deadly Triad**
    
- 解决办法：
    
    - Target network
        
    - Replay buffer
        
    - Conservative updates (如 SAC、TD3)
        

---

## 4. **策略梯度的数学本质**

### Policy Gradient Theorem

$$
\nabla_\theta J(\theta) = \mathbb{E}_\pi [ \nabla_\theta \log \pi_\theta(a|s) Q^\pi(s,a) ]  
$$

#### 深入理解

1. **Log-derivative trick**
    
    -$$ (\nabla_\theta \pi_\theta(a|s) = \pi_\theta(a|s) \nabla_\theta \log \pi_\theta(a|s))
        $$
        
    - 将梯度转化为可采样期望。
        
2. **方差降低**
    
    - baseline = (V^\pi(s))
        
    - 优势函数 $(A^\pi(s,a) = Q^\pi(s,a)-V^\pi(s))$ 
        
    - 保持期望不变，但减少方差。
        
3. **直觉**
    
    - 高价值动作 → 增加被选概率
        
    - 低价值动作 → 减少被选概率
        

--

## 5. **Actor-Critic 深入解析**

Actor-Critic 将价值方法与策略方法结合：

1. Critic 学价值函数 (V_w(s))
    
    - 使用 TD error 学习
        
    - 输出优势估计 (\hat{A}(s,a))
        
2. Actor 学策略 (\pi_\theta(a|s))
    
    - 使用 Critic 提供的优势更新策略
        
    - 避免 REINFORCE 方差过大
        

### 核心公式

- Critic update:  
$$
    \delta_t = R_{t+1} + \gamma V_w(S_{t+1}) - V_w(S_t)  
$$
    
- Actor update:  
$$
    \theta \leftarrow \theta + \alpha \delta_t \nabla_\theta \log \pi_\theta(a|s)  
$$
    

#### 关键点

- Critic 越准确，Actor 更新越稳健
    
- 可加入 GAE (Generalized Advantage Estimation) 平滑优势
    

---

## 6. **深度价值函数与 DQN 的工程技巧**

1. **经验回放 (Experience Replay)**
    
    - 打破数据相关性
        
    - 提高数据利用率
        
2. **目标网络 (Target Network)**
    
    - 稳定 bootstrap 更新
        
    - 每隔 N 步更新一次
        
3. **Double DQN**
    
    - 减轻 Q-learning 的过估计
        
    - 两个网络分离 action selection 和 evaluation
        
4. **Dueling DQN**
    
    - 分离状态价值 (V(s)) 和优势 (A(s,a))
        
    - 提高学习效率
        

#### 深入点

- 函数逼近 + bootstrap = 训练不稳定 → DQN 的多重工程技巧都是为了缓解这个问题。
    
- 理解“非平稳目标”是深度 RL 核心难点。
    

---

## 7. **探索策略的理论理解**

- **ε-greedy**
    
    - 简单直接，探索固定概率
        
- **Softmax / Boltzmann**
    
    - 概率与 Q 值成指数关系
        
- **UCB**
    
    - 上界置信度，理论保证
        
- **Thompson Sampling**
    
    - 采样动作，适用于 Bayes RL / bandit
        

> 核心思想：任何 RL 算法都离不开探索-利用权衡，尤其在稀疏奖励环境中，探索策略设计决定训练成败。

---

## 8. **函数逼近与 Deadly Triad 深入**

1. **Deadly Triad**
    
    - Off-policy
        
    - Bootstrapping
        
    - 函数近似
        
2. **风险理解**
    
    - 状态空间大 → 必须用函数逼近
        
    - Off-policy → 目标与数据分布不一致
        
    - Bootstrapping → 目标非平稳
        
    - 三者结合 → 学习发散
        
3. **解决方法**
    
    - Target network (DQN)
        
    - Clipped objective (PPO)
        
    - Entropy regularization (SAC)
        
    - Actor-Critic + replay buffer
        

---

如果你希望，我可以帮你**画一张“强化学习理论深化图谱”**，把 Bellman、TD Error、On/Off-policy、Policy Gradient、Actor-Critic、Deadly Triad 等关键模块，用图形化方式串联起来，让你一眼看懂整个理论体系及深层联系。

你希望我帮你画这张图吗？

下面这份学习指导，我会把目标设定得更高一点：不是“会用深度学习框架”，也不是“会调 LLM API”，而是帮你建立 **LLM 科研直觉**。

所谓科研直觉，指的是你看到一个现象、论文或实验结果时，能下意识问出这些问题：

> 这个能力来自数据、架构、目标函数、规模、推理预算，还是评估偏差？  
> 这个改进是真的能力提升，还是 benchmark/数据污染/提示词技巧造成的？  
> 这个方法为什么能 work？它的失败模式在哪里？  
> 这个问题值得研究吗？是否只是工程调参？  
> 如果我资源有限，能不能做出小规模但有解释力的实验？

---

# 一、先建立深度学习的核心世界观

深度学习不是“堆很多层神经网络”，而是：

> 用可微分函数族，在大量数据上，通过梯度下降自动学习表示，并把复杂任务转化成统计预测问题。

这个定义里有 5 个关键词：

$$
\text{函数族} \quad  
\text{可微分} \quad  
\text{表示} \quad  
\text{数据} \quad  
\text{优化}  
$$

你之后学任何模型，都要围绕这 5 个维度判断。

---

## 1. 神经网络的第一直觉：它是“可微分程序”

普通程序是人手写规则：

```text
if 条件:
    执行动作 A
else:
    执行动作 B
```

神经网络不是手写规则，而是给它一个参数化函数：

$$
f_\theta(x)  
$$

然后用数据告诉它：

$$
f_\theta(x) \approx y  
$$

训练过程就是调整参数 (\theta)，让预测误差变小：

$$
\theta \leftarrow \theta - \eta \nabla_\theta L(\theta)  
$$

你可以把神经网络看成：

> 一个由矩阵乘法、非线性函数、归一化、注意力机制组成的“软程序”。

它不是显式写出“规则”，而是把规则隐含在权重矩阵里。

---

## 2. 深度学习的第二直觉：模型学的不是答案，而是表示

很多初学者会以为模型在学：

$$
x \rightarrow y  
$$

但更深层的理解是，模型在学中间表示：

$$
x \rightarrow h_1 \rightarrow h_2 \rightarrow h_3 \rightarrow y  
$$

例如图像模型不是直接从像素跳到“猫”，而是可能逐层形成：

$$
\text{边缘}  
\rightarrow  
\text{纹理}  
\rightarrow  
\text{局部部件}  
\rightarrow  
\text{整体对象}  
$$

语言模型也不是直接从 token 跳到答案，而是可能逐层形成：

$$
\text{词形}  
\rightarrow  
\text{短语}  
\rightarrow  
\text{句法关系}  
\rightarrow  
\text{实体}  
\rightarrow  
\text{语义}  
\rightarrow  
\text{任务状态}  
\rightarrow  
\text{输出分布}  
$$

这就是 representation learning。

你学习深度学习时，应该不断问：

> 这个网络结构给模型提供了什么样的表示空间？  
> 它更容易表示什么？更难表示什么？

---

## 3. 深度学习的第三直觉：优化不是“找到精确解”，而是“在巨大函数空间里找到能泛化的解”

神经网络参数很多，通常远多于训练样本。按传统统计学习的直觉，这应该严重过拟合。但现代深度学习的核心现象之一是：

> 过参数化模型不仅能拟合训练集，还常常能泛化。

这背后有很多解释方向，包括隐式正则化、数据结构、优化偏置、模型架构先验、scale effect 等。你暂时不需要一次性掌握所有理论，但要建立一个重要直觉：

> 深度学习模型的能力，不仅由参数数量决定，也由数据分布、优化过程、架构偏置和训练目标共同决定。

---

# 二、你应该如何学习深度学习：不要按“模型名字”学

很多人学深度学习的顺序是：

$$
\text{MLP}  
\rightarrow  
\text{CNN}  
\rightarrow  
\text{RNN}  
\rightarrow  
\text{Transformer}  
\rightarrow  
\text{LLM}  
$$

这个顺序可以，但容易变成“背模型史”。

我建议你按 7 条主线学习。

---

# 主线 1：前向传播——信息如何流动

神经网络最基本的一层是：

$$
h = \sigma(Wx + b)  
$$

你要把它理解成三件事：

第一，线性变换 (Wx) 改变坐标系。

第二，偏置 (b) 平移决策边界。

第三，非线性 (\sigma) 让模型可以表达复杂函数。

如果没有非线性，多层线性网络仍然只是一个线性函数：

$$
W_3W_2W_1x = Wx  
$$

所以深度网络的表达能力来自：

$$
\text{线性变换} + \text{非线性激活} + \text{层级组合}  
$$

直觉上，神经网络每一层都在做：

> 把输入重新编码到一个更适合当前任务的空间里。

---

# 主线 2：反向传播——责任如何分配

反向传播不是神秘算法，它本质上是链式法则。

如果：

$$
L = L(h_3), \quad h_3 = f_3(h_2), \quad h_2 = f_2(h_1), \quad h_1=f_1(x)  
$$

那么：

$$
\frac{\partial L}{\partial h_1}

\frac{\partial L}{\partial h_3}  
\frac{\partial h_3}{\partial h_2}  
\frac{\partial h_2}{\partial h_1}  
$$

反向传播的直觉是：

> 输出错了，要把“责任”一层一层往前分摊。

你要形成一个非常重要的科研直觉：

> 深度学习训练的困难，很多时候不是模型没有表达能力，而是梯度信号传不过去、传歪了、太噪了，或者目标函数给的反馈不对。

这解释了很多技术为什么重要：

|技术|解决的问题|
|---|---|
|ReLU|缓解梯度消失|
|残差连接|让梯度更容易跨层传播|
|LayerNorm / BatchNorm|稳定激活尺度|
|Adam / AdamW|适应不同参数的梯度尺度|
|学习率调度|控制优化早期探索和后期收敛|
|warmup|避免训练初期不稳定|
|gradient clipping|防止梯度爆炸|

Google 的 Deep Learning Tuning Playbook 强调从架构、优化器、batch size、实验追踪、输入管线、评估等角度系统调优，而不是盲目调参；这类工程化经验非常适合训练你的实验直觉。([Google for Developers](https://developers.google.com/machine-learning/guides/deep-learning-tuning-playbook?utm_source=chatgpt.com "Deep Learning Tuning Playbook - Google Developers"))

---

# 主线 3：损失函数——你真正让模型学了什么

深度学习模型不会学“你心里想要的东西”，它只会优化你写下来的 loss。

分类任务常用交叉熵：

$$
L = -\sum_i y_i \log \hat{y}_i  
$$

语言模型的 next-token prediction 本质上也是交叉熵：

$$
L = - \sum_t \log p_\theta(x_t | x_{<t})  
$$

直觉是：

> 模型每一步都在问：在当前上下文下，下一个 token 应该是什么？

这件事看起来很简单，但 LLM 的很多能力都从这里长出来。

为什么？

因为要预测下一个 token，模型必须压缩大量隐含信息：

|要预测的东西|模型可能需要学会|
|---|---|
|语法正确的下一个词|句法结构|
|事实类补全|世界知识|
|代码下一行|程序结构|
|数学证明下一步|推理模式|
|对话回复|语用和意图|
|多轮上下文|状态追踪|
|模仿论文风格|文体和领域知识|

所以 LLM 的核心直觉是：

> next-token prediction 是一个表面简单、信息密度极高的自监督任务。

它不是直接教模型“推理”，而是让模型在压缩互联网文本分布时，被迫学习许多能帮助预测的潜在结构。

---

# 主线 4：归纳偏置——架构让什么事情变容易

模型架构不是随便设计的。架构给模型提供了 inductive bias，也就是“偏好学习某些函数”。

例如：

|架构|归纳偏置|
|---|---|
|CNN|局部性、平移等变性|
|RNN|时间递归、顺序状态|
|Transformer|全局 token 交互、内容寻址|
|GNN|图结构消息传递|
|Diffusion U-Net|多尺度去噪生成|

Transformer 的重大意义在于：它抛弃循环和卷积，主要依靠 attention 来建模输入和输出之间的全局依赖，同时比循环结构更容易并行化；原始 Transformer 论文就是以 “Attention Is All You Need” 提出这种架构的。([arXiv](https://arxiv.org/abs/1706.03762?utm_source=chatgpt.com "[1706.03762] Attention Is All You Need - arXiv.org"))

科研直觉：

> 架构不是“更复杂就更强”，而是看它是否把任务中的重要结构变成容易学习的模式。

---

# 主线 5：优化——模型不是被“设计出来”的，而是被“训练出来”的

一个架构在纸面上有表达能力，不代表训练后真的会学到你想要的功能。

训练过程由很多因素共同决定：

$$
\text{初始化}  
+  
\text{优化器}  
+  
\text{学习率}  
+  
\text{batch size}  
+  
\text{数据顺序}  
+  
\text{正则化}  
+  
\text{训练步数}  
$$

你要建立一个很重要的实验直觉：

> 深度学习研究中，很多“方法创新”其实改变的是优化路径，而不仅是模型最终可表达的函数集合。

例如 residual connection：

$$
h_{l+1} = h_l + F(h_l)  
$$

表面看是多加了一条旁路，深层意义是：

> 每一层不必重新生成全部表示，只需要学习对已有表示的增量修正。

这让深层网络更容易训练。

---

# 主线 6：泛化——为什么测试集能好

泛化不是一句“防止过拟合”能解释的。

你需要从 4 个角度理解泛化。

第一，数据分布。

训练集和测试集是否同分布？

$$
p_{\text{train}}(x,y) \approx p_{\text{test}}(x,y)  
$$

第二，模型偏置。

架构是否天然适合任务？

第三，优化偏置。

SGD/Adam 找到的解是否偏向简单、平滑、鲁棒的函数？

第四，评估设计。

测试集是否真的测到了你想测的能力？

LLM 研究里，泛化尤其复杂，因为模型可能见过相似数据、benchmark 可能被污染、prompt 格式可能影响结果、模型可能会“猜测评测意图”。

所以 LLM 科研里有一句非常重要的警惕：

> 一个 benchmark 分数提升，不等于能力真的提升。

---

# 主线 7：规模——能力如何随 compute/data/model 变化

LLM 科研一定要理解 scaling law。

OpenAI 的 scaling laws 研究发现，语言模型的交叉熵损失会随模型大小、数据规模和训练计算量呈幂律下降，并且这种趋势跨越多个数量级；这让研究者可以用较小实验预测大规模训练表现。([arXiv](https://arxiv.org/abs/2001.08361?utm_source=chatgpt.com "[2001.08361] Scaling Laws for Neural Language Models - arXiv.org"))

DeepMind 的 Chinchilla 工作进一步强调：在固定计算预算下，模型参数量和训练 token 数量需要更平衡地增长；他们发现许多早期大模型相对“数据训练不足”，并用更小但训练 token 更多的 Chinchilla 获得了更好的表现。([arXiv](https://arxiv.org/abs/2203.15556?utm_source=chatgpt.com "Training Compute-Optimal Large Language Models"))

这给你的科研直觉是：

> LLM 不是单纯“越大越好”，而是参数、数据、计算、训练步数、推理成本之间的资源分配问题。

你要学会问：

|问题|直觉|
|---|---|
|模型变大了但数据没变多|可能 undertrained 或过拟合|
|数据变多但模型太小|可能容量不足|
|训练 loss 降了但 eval 不涨|可能数据质量、评估或泛化出问题|
|benchmark 涨了但真实任务不涨|可能过拟合评测格式|
|推理时加长思考能涨|可能能力受 test-time compute 限制|

---

# 三、深度学习基础阶段：你必须真正吃透的内容

下面我按学习顺序展开。

---

## 阶段 1：线性模型与 logistic regression

你不要跳过线性模型。

线性模型是深度学习的“原子”。

$$
\hat{y} = Wx + b  
$$

分类时：

$$
p(y|x) = \text{softmax}(Wx+b)  
$$

你要理解：

1. (W) 的每一行可以看成一个类别方向；
    
2. logits 是未归一化分数；
    
3. softmax 把分数变成概率；
    
4. cross entropy 惩罚“把概率分给错误类别”。
    

直觉：

> 分类模型本质上是在表示空间里学习决策边界。

如果你连 logistic regression 的梯度都不能手推，后面看 Transformer 也只是看热闹。

---

## 阶段 2：MLP 与非线性表达

MLP 是：

$$
h_1 = \sigma(W_1x+b_1)  
$$

$$
h_2 = \sigma(W_2h_1+b_2)  
$$

$$
\hat{y}=W_3h_2+b_3  
$$

MLP 的核心直觉是：

> 每一层都在把数据弯折、拉伸、重新排列，使原本复杂的边界变得线性可分。

你可以把 MLP 看成一个“表示空间变换器”。

浅层模型是在原始空间画线。

深层模型是先把空间变形，再画线。

---

## 阶段 3：反向传播与自动微分

你需要手推至少一次两层 MLP 的反向传播。

不用每天手推，但必须知道梯度是怎么来的。

最小练习：

给定：

$$
h = \text{ReLU}(Wx+b)  
$$

$$
\hat{y} = Uh+c  
$$

$$
L = \frac{1}{2}|\hat{y}-y|^2  
$$

你要能推出：

$$
\frac{\partial L}{\partial U}, \quad  
\frac{\partial L}{\partial W}, \quad  
\frac{\partial L}{\partial b}  
$$

如果这一步扎实，你以后看 LoRA、adapter、fine-tuning、RLHF、DPO 都会更轻松。

---

## 阶段 4：优化器与训练动力学

你要理解 SGD、Momentum、Adam、AdamW 的直觉。

### SGD

$$
\theta \leftarrow \theta - \eta g  
$$

直觉：

> 每次用一个 noisy gradient 朝下降方向走一步。

噪声不是纯坏事。噪声有时帮助模型逃离尖锐区域，找到泛化更好的解。

### Momentum

$$
v \leftarrow \beta v + g  
$$

$$
\theta \leftarrow \theta - \eta v  
$$

直觉：

> 给梯度加惯性，减少来回震荡。

### Adam

Adam 给每个参数自适应学习率。

直觉：

> 梯度经常很大的参数少走一点，梯度经常很小的参数多走一点。

### AdamW

AdamW 把 weight decay 从 Adam 的梯度更新里解耦出来，现代 Transformer/LLM 训练中非常常用。

科研直觉：

> 优化器不是纯工程细节。优化器会改变模型最终学到的解。

---

## 阶段 5：初始化、归一化、残差连接

这三个东西是深度网络能训练起来的关键。

### 初始化

如果权重太大，激活和梯度爆炸。

如果权重太小，信号逐层消失。

所以初始化要让信号尺度在层间大致稳定。

### 归一化

LayerNorm 的直觉：

> 不让某一层的激活尺度乱飘，让后续层看到更稳定的输入。

在 Transformer 里，LayerNorm 是训练稳定性的核心组件之一。

### 残差连接

$$
x_{l+1}=x_l+F(x_l)  
$$

直觉：

> 深层网络不是每层重写世界，而是每层在已有表示上做修正。

这也解释了为什么 LLM 的 residual stream 很重要。

你可以把 residual stream 看成：

> 模型内部的共享黑板。Attention 和 MLP 每层都往黑板上读写信息。

这个直觉对理解 Transformer 非常关键。

---

# 四、Transformer 的科研直觉

现在进入 LLM 的核心。

---

## 1. Tokenization：语言模型的接口层

LLM 不直接看文字，而是看 token。

一句话会被切成 token 序列：

$$
x_1, x_2, \dots, x_T  
$$

tokenization 的直觉是：

> 它决定了模型看到世界的最小单位。

token 太细，比如字符级，序列变长，建模困难。

token 太粗，词表巨大，罕见词和组合泛化困难。

科研直觉：

> tokenization 不是预处理小事，它影响效率、泛化、多语言能力、代码能力、数字能力和长上下文成本。

Stanford CS336 这类语言模型课程已经把 tokenization、资源核算、架构、注意力替代、MoE、GPU/TPU、kernel、并行、scaling laws、推理、评估、数据、后训练和 RLHF/RLVR 放在同一门课里讲，这说明现代 LLM 研究已经变成“算法 + 数据 + 系统 + 对齐 + 评估”的整体工程科学。([Stanford CS336](https://cs336.stanford.edu/?utm_source=chatgpt.com "Stanford CS336 | Language Modeling from Scratch"))

---

## 2. Embedding：把离散符号放进连续空间

token 是离散 ID，模型不能直接计算语义，所以先映射为向量：

$$
e_i = E[x_i]  
$$

embedding 的直觉：

> 把符号变成空间里的点，让相似关系可以通过几何结构表达。

例如：

$$
\text{king} - \text{man} + \text{woman} \approx \text{queen}  
$$

虽然现实 LLM 的表示更复杂，但这个例子说明：语义关系可以变成向量空间关系。

LLM 的每一层都在更新 token 的表示。

最开始 token 表示更多是词形和局部语义。

越往后，表示越受上下文影响。

---

## 3. Self-Attention：内容寻址的动态检索

Transformer 的核心是 self-attention。

对每个 token 表示 (x_i)，模型生成：

$$
q_i = W_Qx_i  
$$

$$
k_i = W_Kx_i  
$$

$$
v_i = W_Vx_i  
$$

注意力权重：

$$
\alpha_{ij} =  
\text{softmax}  
\left(  
\frac{q_i^\top k_j}{\sqrt{d}}  
\right)  
$$

输出：

$$
o_i = \sum_j \alpha_{ij}v_j  
$$

直觉非常重要：

|符号|直觉|
|---|---|
|Query|我现在想找什么信息|
|Key|我这里有什么信息可被匹配|
|Value|如果你关注我，我提供什么内容|
|Attention score|查询和候选信息的匹配程度|
|Softmax|软选择，不是硬选择|
|Weighted sum|从多个位置聚合信息|

所以 self-attention 可以理解为：

> 每个 token 根据当前需要，从上下文中检索相关信息。

这就是为什么它适合语言。

例如句子：

> 小明把苹果放进书包，因为他明天要吃。

“他”要理解成“小明”，attention 可以让“他”直接关注“小明”。

---

## 4. Multi-head Attention：多种关系并行检索

单个 attention head 只能学一种检索模式。

多个 head 可以并行学不同关系：

|head 可能学到的模式|示例|
|---|---|
|指代关系|他 → 小明|
|句法关系|动词 → 主语|
|局部邻近|当前词 → 前一个词|
|括号匹配|右括号 → 左括号|
|复制模式|前文出现过的序列 → 后续补全|
|格式结构|Markdown 标题 → 段落结构|

科研直觉：

> 多头注意力不是简单增加参数，而是允许模型同时运行多个“信息路由子程序”。

---

## 5. MLP 层：不只是“前馈网络”，更像特征变换与记忆库

Transformer block 里除了 attention，还有 MLP：

$$
\text{MLP}(x)=W_2\sigma(W_1x)  
$$

很多人只重视 attention，忽略 MLP。

但 MLP 很重要。

直觉上：

> Attention 负责从上下文中搬运信息，MLP 负责对当前位置的信息进行非线性加工、特征触发和知识变换。

一种常见理解是：

- attention 像“读上下文”；
    
- MLP 像“查内部知识/做局部计算”；
    
- residual stream 像“共享工作区”。
    

Transformer Circuits 系列工作把 Transformer 作为可逆向工程的计算系统来研究，目标是把模型内部的注意力头、MLP 和 residual stream 的计算分解成人能理解的 circuits。([变压器电路](https://transformer-circuits.pub/2021/framework/index.html?utm_source=chatgpt.com "A Mathematical Framework for Transformer Circuits"))

---

## 6. Residual Stream：LLM 内部的“黑板”

一个 Transformer block 可以粗略看成：

$$
x \leftarrow x + \text{Attention}(\text{LN}(x))  
$$

$$
x \leftarrow x + \text{MLP}(\text{LN}(x))  
$$

这意味着每层都在往同一个表示流里添加信息。

直觉：

> residual stream 是模型内部的共享黑板。每一层 attention/MLP 读取黑板，写入增量信息。

这给你一个重要科研直觉：

> LLM 不是每层独立工作，而是很多层共同迭代更新同一个隐状态。

所以当你看模型可解释性论文时，经常会看到：

- activation patching；
    
- logit lens；
    
- residual stream decomposition；
    
- attention head ablation；
    
- MLP neuron/feature analysis。
    

这些方法本质上都在问：

> 哪些内部组件在什么位置、什么层，对最终输出产生了因果影响？

---

# 五、LLM 的核心训练直觉

---

## 1. Pretraining：语言模型先学“世界分布”

预训练目标：

$$
\max_\theta \sum_t \log p_\theta(x_t|x_{<t})  
$$

它不是直接学“回答用户问题”，而是学文本分布。

直觉：

> 预训练模型是一个强大的文本续写器，它学会了大量语言、知识、代码、格式、推理痕迹和任务模式。

但它未必听话。

因为互联网文本里有很多风格：

- 回答问题；
    
- 争论；
    
- 胡说；
    
- 模仿；
    
- 补全文档；
    
- 写代码；
    
- 写小说；
    
- 引用错误信息；
    
- 生成有害内容。
    

所以预训练模型的能力和可用性是两件事。

这就引出 instruction tuning 和 alignment。

---

## 2. Instruction Tuning：把“会续写”变成“会听指令”

监督微调 SFT 的目标是让模型看到：

$$
(\text{instruction}, \text{desired answer})  
$$

然后学习：

$$
p_\theta(\text{desired answer}|\text{instruction})  
$$

直觉：

> SFT 不是主要注入世界知识，而是教模型在对话场景中采用人类希望的行为格式。

比如：

- 如何回答问题；
    
- 如何拒绝不安全请求；
    
- 如何分步骤解释；
    
- 如何保持语气；
    
- 如何遵循格式约束。
    

科研直觉：

> SFT 往往改变“行为分布”大于改变“基础能力”。

一个模型预训练时已经知道很多东西，但 SFT 后才变得像助手。

---

## 3. RLHF：从“模仿好答案”到“偏好更好答案”

InstructGPT 论文展示了一个经典流程：先用人类示范数据做监督微调，再收集人类对模型输出的排序来训练奖励模型，最后用强化学习优化模型，使输出更符合人类偏好；论文还指出，单纯变大并不会天然让模型更好地遵循用户意图。([arXiv](https://arxiv.org/abs/2203.02155?utm_source=chatgpt.com "Training language models to follow instructions with human feedback"))

RLHF 的直觉：

> 人类很难直接写出“好回答”的完整规则，但比较两个回答哪个更好相对容易。

所以 RLHF 把问题转化为偏好学习。

流程大致是：

$$
\text{SFT model}  
\rightarrow  
\text{生成多个回答}  
\rightarrow  
\text{人类排序}  
\rightarrow  
\text{训练 reward model}  
\rightarrow  
\text{用 RL 优化策略}  
$$

科研直觉：

> RLHF 优化的是“被偏好模型打高分的行为”，不是客观真理。

这会带来问题：

- reward hacking；
    
- 过度迎合；
    
- 拒答过度；
    
- 语言风格变得油滑；
    
- 长答案偏好；
    
- 表面合理但事实错误。
    

---

## 4. DPO：把偏好优化变成分类式目标

DPO 的核心贡献是把 RLHF 中“奖励建模 + 强化学习”的复杂流程，转化成一个更简单的直接偏好优化目标；论文称这种方法可以用简单分类损失从偏好数据中优化语言模型，避免显式奖励模型和复杂 RL 过程。([arXiv](https://arxiv.org/abs/2305.18290?utm_source=chatgpt.com "Direct Preference Optimization: Your Language Model is Secretly a Reward Model"))

直觉：

> DPO 直接告诉模型：chosen 回答相对于 rejected 回答应该更可能。

粗略理解：

$$
\log p_\theta(y_{\text{chosen}}|x)

\log p_\theta(y_{\text{rejected}}|x)  
$$

但它还会用 reference model 约束模型不要偏离太多。

科研直觉：

> DPO 类方法的魅力在于简单稳定，但它仍然受偏好数据质量、偏好分布、reference model 和长度偏差影响。

---

## 5. LoRA：低秩更新的直觉

LoRA 冻结预训练权重，只在部分线性层里加入低秩可训练矩阵，从而大幅减少下游任务的可训练参数；原论文以 GPT-3 175B 为例，说明完整微调独立大模型副本成本很高，而 LoRA 能在保留质量的同时显著降低可训练参数和显存需求。([arXiv](https://arxiv.org/abs/2106.09685?utm_source=chatgpt.com "LoRA: Low-Rank Adaptation of Large Language Models"))

普通微调是：

$$
W \leftarrow W + \Delta W  
$$

LoRA 假设：

$$
\Delta W \approx AB  
$$

其中：

$$
A \in \mathbb{R}^{d \times r}, \quad B \in \mathbb{R}^{r \times k}, \quad r \ll d,k  
$$

直觉：

> 下游适配不需要在所有方向上改模型，只需要在一个低维子空间里做有效更新。

科研直觉：

> LoRA 的成功暗示：大模型预训练后已经有大量可复用能力，下游任务很多时候只需要“轻微重定向”。

---

# 六、LLM 能力从哪里来：你必须建立的 8 个直觉

---

## 直觉 1：LLM 是压缩器，不是数据库

LLM 的参数存了大量统计规律，但它不是精确数据库。

它会记住很多事实，但事实以分布式方式存储在权重中。

这意味着：

- 常见事实容易答对；
    
- 罕见事实容易混淆；
    
- 训练后新事实不知道；
    
- 相似实体可能互相干扰；
    
- 看似自信不代表有证据。
    

所以当你研究 factuality 时，要区分：

$$
\text{知识是否在参数中}  
$$

$$
\text{模型是否能正确检索出来}  
$$

$$
\text{模型是否能在当前上下文下表达出来}  
$$

$$
\text{模型是否有外部证据支持}  
$$

RAG 的核心思想就是把参数记忆和外部检索结合起来：模型先检索相关文档，再基于文档生成答案；原始 RAG 工作在知识密集型 NLP 任务上比较了不同检索增强生成形式，并报告生成结果更具体、多样和事实化。([arXiv](https://arxiv.org/abs/2005.11401?utm_source=chatgpt.com "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks"))

科研直觉：

> 参数记忆适合常识和模式，外部检索适合新鲜、长尾、可追溯事实。

---

## 直觉 2：LLM 的“推理”部分来自模式、搜索和中间状态

Chain-of-thought prompting 论文发现，让模型生成中间推理步骤，可以显著提升大模型在算术、常识和符号推理任务上的表现；论文强调这种能力在足够大的语言模型中通过少量示例提示被激发出来。([arXiv](https://arxiv.org/abs/2201.11903?utm_source=chatgpt.com "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models"))

这给我们一个关键直觉：

> LLM 的很多推理能力不是一次性从输入跳到答案，而是通过中间 token 展开计算。

也就是说，输出 token 本身可以成为计算轨迹。

当模型写下：

```text
第一步...
第二步...
第三步...
```

它不仅是在解释，也是在利用更多推理时间。

科研直觉：

> 对 LLM 来说，生成过程就是计算过程；更多 token 往往意味着更多 test-time compute。

这解释了：

- chain-of-thought；
    
- scratchpad；
    
- self-consistency；
    
- tree-of-thought；
    
- program-of-thought；
    
- tool use；
    
- agent planning；
    
- reasoning model 的 test-time scaling。
    

---

## 直觉 3：In-context learning 是“临时学习”，但不是参数更新

In-context learning 指模型在 prompt 里看到几个例子后，就能适配新任务。

例如：

```text
英文: dog -> 中文: 狗
英文: cat -> 中文: 猫
英文: apple -> 中文:
```

模型回答：

```text
苹果
```

没有梯度更新，但行为改变了。

直觉：

> prompt 里的例子改变了模型的内部激活状态，相当于在上下文中临时指定了任务。

Anthropic 的 induction heads 研究提出，归纳头可能是 Transformer 中通用 in-context learning 的机制来源之一；论文用多条证据支持某些注意力头能实现类似“看到 A B … A，就预测 B”的模式补全。([arXiv](https://arxiv.org/abs/2209.11895?utm_source=chatgpt.com "[2209.11895] In-context Learning and Induction Heads - arXiv.org"))

科研直觉：

> in-context learning 不是魔法，它可能依赖模型内部已经学到的模式匹配、复制、类比和任务识别机制。

---

## 直觉 4：大模型的知识是分布式、重叠和多义的

一个神经元不一定对应一个概念。

一个概念也不一定只由一个神经元表示。

Anthropic 的 superposition 工作用 toy model 研究了模型如何在维度有限时用重叠方式表示更多稀疏特征，也就是多个特征可能叠加在同一个表示空间中。([Anthropic](https://www.anthropic.com/research/toy-models-of-superposition?utm_source=chatgpt.com "Toy Models of Superposition - Anthropic"))

直觉：

> 神经网络可能把很多特征压进同一个高维空间里，特征之间会重叠、纠缠、干扰。

这解释了为什么：

- 单个 neuron 可解释性有限；
    
- activation steering 有时有效；
    
- sparse autoencoder 能帮助提取特征；
    
- 模型会出现奇怪的概念混淆；
    
- 对抗样本和 jailbreak 可能存在。
    

科研直觉：

> 解释 LLM 不是给每个神经元贴标签，而是找出内部特征、方向、子空间和 circuits 如何组合成行为。

---

## 直觉 5：Scaling 带来能力，但不是唯一道路

规模带来很多能力：

- 更低 loss；
    
- 更多知识；
    
- 更强泛化；
    
- 更好 few-shot；
    
- 更复杂推理；
    
- 更强代码能力。
    

但 scale 不是唯一变量。

同样重要的还有：

$$
\text{数据质量}  
$$

$$
\text{训练目标}  
$$

$$
\text{架构}  
$$

$$
\text{上下文长度}  
$$

$$
\text{后训练}  
$$

$$
\text{推理策略}  
$$

$$
\text{工具使用}  
$$

$$
\text{评估方式}  
$$

科研直觉：

> 当一个新模型更强时，不要只问“是不是更大”，要问“是哪一种资源被更好利用了”。

---

## 直觉 6：Data is the hidden architecture

很多 LLM 能力不是来自架构变化，而是来自数据。

数据决定模型看到什么世界。

数据包含：

- 网页；
    
- 书籍；
    
- 代码；
    
- 数学；
    
- 论文；
    
- 对话；
    
- 合成数据；
    
- 工具调用轨迹；
    
- 人类偏好；
    
- 多模态数据。
    

数据处理包含：

- filtering；
    
- deduplication；
    
- decontamination；
    
- mixture weighting；
    
- curriculum；
    
- synthetic data generation；
    
- quality scoring；
    
- domain balancing。
    

科研直觉：

> 对 LLM 来说，数据配方常常比架构细节更重要，但更难公开、更难复现。

这也是为什么 LLM 研究越来越像实验科学：你不仅要比较模型结构，还要控制数据来源、数据质量和训练预算。

---

## 直觉 7：Evaluation 是 LLM 科研的地基

没有好评估，就没有可靠研究。

HELM 把自己定位为语言模型透明评估的 living benchmark，强调覆盖面、多指标测量、标准化以及承认评估不完整；这很符合 LLM 评估的现实复杂性。([斯坦福CRFM](https://crfm.stanford.edu/helm/?utm_source=chatgpt.com "Holistic Evaluation of Language Models (HELM) - Stanford University"))

EleutherAI 的 lm-evaluation-harness 是很多论文和组织使用的语言模型评估框架，支持大量任务和可复现比较。([GitHub](https://github.com/EleutherAI/lm-evaluation-harness?utm_source=chatgpt.com "EleutherAI/lm-evaluation-harness - GitHub"))

科研直觉：

> LLM 评估不能只看一个总分，而要看能力切片、错误类型、鲁棒性、污染风险、成本和真实使用场景。

你需要区分：

|类型|例子|风险|
|---|---|---|
|静态 benchmark|MMLU、HellaSwag|数据污染、饱和|
|人类偏好评估|pairwise comparison|标注偏差、风格偏好|
|LLM-as-judge|GPT/Claude 评分|评委偏差、位置偏差|
|任务型 eval|代码、检索、工具调用|环境依赖|
|安全 eval|jailbreak、toxicity|攻击分布变化|
|长上下文 eval|needle-in-haystack|人工性强|
|agent eval|web/task completion|不稳定、难复现|

你做 LLM 科研时，必须养成一个习惯：

> 每个实验结果都要配错误分析。

不要只报告分数。

要看模型具体错在哪里。

---

## 直觉 8：LLM 研究是“算法—数据—系统—产品行为”的耦合科学

传统 ML 研究可能主要比较算法。

LLM 研究不同。

同一个算法，换数据、batch size、上下文长度、tokenizer、训练步数、推理策略，结果可能完全不同。

所以 LLM 科研直觉必须包括系统层：

|系统因素|影响|
|---|---|
|GPU/TPU 利用率|训练成本|
|mixed precision|稳定性和速度|
|activation checkpointing|显存|
|tensor parallelism|大模型切分|
|pipeline parallelism|多卡流水线|
|ZeRO/FSDP|参数和优化器状态切分|
|KV cache|推理吞吐|
|quantization|推理成本与精度|
|speculative decoding|解码加速|
|FlashAttention|attention 计算效率|

科研直觉：

> LLM 中很多“理论可行”的想法，会死在显存、吞吐、延迟和数据管线上。

你不一定一开始就要成为系统专家，但必须理解系统约束。

---

# 七、你应该重点建立的“科研判断力”

---

## 1. 看到一个 LLM 论文，先判断它改了哪一层

你可以用这个表快速定位论文贡献。

|层级|论文可能改什么|
|---|---|
|数据层|数据过滤、合成数据、数据混合、去污染|
|tokenizer|词表、byte-level、多语言、数字处理|
|架构层|attention、MoE、位置编码、长上下文|
|训练层|优化器、loss、batch、curriculum|
|后训练层|SFT、RLHF、DPO、RLVR、安全对齐|
|推理层|CoT、self-consistency、beam/search、tool use|
|检索层|RAG、memory、indexing、reranking|
|评估层|benchmark、LLM judge、human eval|
|系统层|并行、kernel、量化、KV cache|
|解释层|circuits、SAE、activation patching|

你读论文时第一句话就应该问：

> 它到底改的是哪一层？

第二句话问：

> 它声称提升的是哪种能力？

第三句话问：

> 它的评估是否足以支持这个声称？

---

## 2. 区分“能力提升”和“行为诱导”

这是 LLM 研究最重要的直觉之一。

例如，一个 prompt 让模型数学题正确率提升。

这可能意味着：

1. 模型真的会更强推理；
    
2. prompt 激发了已有能力；
    
3. 输出格式更适合评测；
    
4. 采样次数更多，相当于用了更多计算；
    
5. benchmark 泄漏；
    
6. 评估脚本偏向某种格式。
    

所以你要问：

$$
\text{能力真的变了吗？}  
$$

还是：

$$
\text{只是读取能力的方式变了？}  
$$

SFT、prompting、CoT、DPO 很多时候改变的是行为分布，而不是基础能力本身。

---

## 3. 区分“训练时计算”和“推理时计算”

传统观点常关注 training compute。

但 LLM 时代，test-time compute 越来越重要。

模型可以：

- 多生成 reasoning tokens；
    
- 自我检查；
    
- 多样采样后投票；
    
- 调用工具；
    
- 检索资料；
    
- 执行代码；
    
- 分解任务；
    
- 规划再行动。
    

科研直觉：

> 同一个 base model，通过不同推理策略，可以表现出不同能力上限。

所以比较模型时要看：

$$
\text{参数量}  
$$

$$
\text{训练 token}  
$$

$$
\text{训练 compute}  
$$

$$
\text{推理 token}  
$$

$$
\text{采样次数}  
$$

$$
\text{工具使用}  
$$

$$
\text{外部信息}  
$$

否则比较不公平。

---

## 4. 区分“知识问题”和“推理问题”

很多题看似推理，其实是知识。

很多题看似知识，其实需要推理。

例如：

> 法国首都是哪里？

主要是参数知识。

> 如果 A 比 B 高，B 比 C 高，谁最高？

主要是符号推理。

> 某篇 2026 年论文提出了什么？

如果不检索，就是知识时效问题。

科研直觉：

> 不要把模型不知道事实误判为推理失败，也不要把记住答案误判为推理能力。

---

## 5. 区分“短上下文能力”和“长上下文能力”

长上下文不是简单把 context window 拉长。

长上下文要求模型能：

- 记住远距离信息；
    
- 检索相关段落；
    
- 抑制无关干扰；
    
- 跨文档综合；
    
- 保持任务状态；
    
- 避免位置偏差；
    
- 控制推理链。
    

科研直觉：

> 长上下文能力的核心不是“能塞多少 token”，而是“能不能在大量 token 中找到、整合、利用正确证据”。

---

## 6. 区分“模型能力”和“系统能力”

一个 LLM 应用可能包含：

$$
\text{LLM}  
+  
\text{retriever}  
+  
\text{reranker}  
+  
\text{tools}  
+  
\text{memory}  
+  
\text{planner}  
+  
\text{eval loop}  
+  
\text{guardrails}  
$$

应用表现好，不代表 base model 强。

base model 强，也不代表应用系统可靠。

科研直觉：

> 研究 LLM 本身和研究 LLM system 是两个不同问题。

---

# 八、深度学习到 LLM 的完整学习路线

下面给你一套 6 阶段路线。

---

## 阶段 1：深度学习基础，目标是“能手写小网络”

推荐时间：4 到 6 周。

你要掌握：

- 线性分类器；
    
- logistic regression；
    
- softmax；
    
- cross entropy；
    
- MLP；
    
- ReLU；
    
- backprop；
    
- SGD；
    
- Adam；
    
- regularization；
    
- dropout；
    
- normalization；
    
- residual connection。
    

推荐资料：

Goodfellow、Bengio 和 Courville 的《Deep Learning》在线版是经典理论教材，覆盖线性代数、概率、优化、机器学习和深度网络基础；它更适合打理论底座。([深度学习书籍](https://www.deeplearningbook.org/?utm_source=chatgpt.com "Deep Learning"))

《Dive into Deep Learning》强调数学、图示、代码和真实数据实验，并且每节内容都是可执行 notebook，适合边学边实现。([深度学习导论](https://d2l.ai/?utm_source=chatgpt.com "D2L - Dive into Deep Learning 1.0.3 documentation"))

Stanford CS231n 的定位是让学生实现、训练和调试自己的神经网络，这一点很适合你建立训练直觉。([CS231n](https://cs231n.stanford.edu/?utm_source=chatgpt.com "Stanford University CS231n: Deep Learning for Computer Vision"))

阶段产出：

1. 不用框架自动层，手写一个两层 MLP；
    
2. 实现 softmax cross entropy；
    
3. 手写 backprop；
    
4. 在 MNIST/CIFAR-10 上训练；
    
5. 画出 train loss、val loss、accuracy；
    
6. 做一次错误分析。
    

你要形成的直觉：

> 训练曲线比最终分数更重要。  
> debug 神经网络时，先看数据、loss、梯度、学习率和过拟合能力。

---

## 阶段 2：表示学习与视觉/序列模型，目标是“理解架构偏置”

推荐时间：4 周。

你要掌握：

- CNN；
    
- pooling；
    
- batch norm；
    
- ResNet；
    
- RNN；
    
- LSTM/GRU；
    
- seq2seq；
    
- attention 的前身；
    
- embedding。
    

阶段产出：

1. 训练一个 CNN；
    
2. 可视化中间特征；
    
3. 训练一个字符级语言模型；
    
4. 比较 RNN 和 Transformer 的建模方式。
    

你要形成的直觉：

> 架构决定信息流动方式。  
> CNN 强在局部模式，RNN 强在顺序状态，Transformer 强在全局内容寻址。

---

## 阶段 3：Transformer，目标是“从零实现小 GPT”

推荐时间：6 到 8 周。

你要掌握：

- tokenization；
    
- embedding；
    
- positional encoding；
    
- causal self-attention；
    
- multi-head attention；
    
- LayerNorm；
    
- residual stream；
    
- MLP block；
    
- causal language modeling；
    
- weight tying；
    
- KV cache；
    
- sampling；
    
- temperature；
    
- top-k / top-p。
    

阶段产出：

1. 从零实现一个 character-level GPT；
    
2. 再实现一个 BPE/token-level 小 GPT；
    
3. 在小文本集上训练；
    
4. 实现 greedy、temperature、top-k、top-p sampling；
    
5. 观察不同采样策略对生成质量的影响；
    
6. 可视化 attention pattern。
    

推荐资料：

Stanford CS224N 现在明确覆盖深度学习 NLP 基础和最新 LLM 研究，适合把 NLP、Transformer 和 LLM 放在一起系统学习。([Stanford University](https://web.stanford.edu/class/cs224n/?utm_source=chatgpt.com "Stanford CS 224N | Natural Language Processing with Deep Learning"))

Hugging Face Transformers 文档覆盖文本、视觉、音频、视频和多模态模型的推理与训练框架，适合在掌握原理后熟悉现代生态。([Hugging Face](https://huggingface.co/docs/transformers/index?utm_source=chatgpt.com "Transformers · Hugging Face"))

你要形成的直觉：

> Transformer 不是“注意力公式”，而是一个可训练的信息路由系统。

---

## 阶段 4：LLM 训练工程，目标是“理解大模型为什么贵、难、不稳定”

推荐时间：6 周。

你要掌握：

- training tokens；
    
- batch size；
    
- gradient accumulation；
    
- mixed precision；
    
- optimizer states；
    
- activation memory；
    
- FLOPs；
    
- GPU utilization；
    
- data pipeline；
    
- distributed training；
    
- checkpointing；
    
- FSDP/ZeRO；
    
- tensor parallel；
    
- pipeline parallel；
    
- FlashAttention；
    
- inference KV cache；
    
- quantization。
    

阶段产出：

1. 计算一个 Transformer 的参数量；
    
2. 估算训练 FLOPs；
    
3. 估算显存占用；
    
4. 训练一个 10M、100M 级别小模型；
    
5. 比较不同 batch size 和学习率；
    
6. 写一份实验记录。
    

你要形成的直觉：

> LLM 研究不是只看 loss，它还必须受计算预算约束。

一个研究想法如果需要 1000 张 GPU 才能验证，对个人科研并不友好。

你要学会设计小规模代理实验。

---

## 阶段 5：后训练与对齐，目标是“理解模型如何从 base model 变成 assistant”

推荐时间：6 周。

你要掌握：

- supervised fine-tuning；
    
- instruction data；
    
- preference data；
    
- reward model；
    
- RLHF；
    
- PPO 基础；
    
- DPO；
    
- rejection sampling；
    
- constitutional AI 思想；
    
- safety tuning；
    
- refusal behavior；
    
- reward hacking；
    
- over-optimization；
    
- alignment tax。
    

阶段产出：

1. 用小模型做 SFT；
    
2. 构造 chosen/rejected 数据；
    
3. 实现 DPO 或使用 TRL 跑 DPO；
    
4. 比较 SFT 与 DPO 输出差异；
    
5. 做人工错误分析。
    

你要形成的直觉：

> 后训练主要塑造模型行为，而不是凭空创造基础能力。

---

## 阶段 6：LLM 科研专题，目标是“找到你的研究方向”

推荐时间：长期。

你可以从以下方向选择。

### 方向 A：数据

问题：

- 数据质量如何影响推理？
    
- 合成数据什么时候有用？
    
- 去重和去污染如何做？
    
- 数据混合比例怎么影响能力？
    
- 小模型能否预测大模型数据配方效果？
    

适合资源有限的研究者。

### 方向 B：评估

问题：

- benchmark 是否测到真实能力？
    
- LLM-as-judge 是否可靠？
    
- 如何评估长上下文？
    
- 如何评估 agent？
    
- 如何减少数据污染？
    
- 如何做细粒度错误分类？
    

非常适合入门科研，因为不一定需要训练大模型。

### 方向 C：推理与 test-time compute

问题：

- CoT 为什么有效？
    
- self-consistency 什么时候有效？
    
- 推理 token 数如何影响正确率？
    
- 模型什么时候需要工具？
    
- verifier 如何提升解题能力？
    
- search 与 language model 如何结合？
    

这个方向和当前 LLM 能力提升关系很大。

### 方向 D：高效微调

问题：

- LoRA rank 如何影响任务适配？
    
- 哪些层最值得微调？
    
- adapter、prefix tuning、LoRA 的差异是什么？
    
- 微调会不会破坏原有能力？
    
- 多任务 LoRA 如何合并？
    

适合工程和实验型研究。

### 方向 E：RAG 与外部记忆

问题：

- 什么时候检索比参数记忆更好？
    
- retriever 错了 generator 能否纠正？
    
- chunk size 如何影响回答？
    
- reranking 如何影响事实性？
    
- 长上下文和 RAG 如何取舍？
    
- 多跳检索如何做？
    

适合应用研究和系统研究。

### 方向 F：机制可解释性

问题：

- 某个能力由哪些 attention heads/MLP features 实现？
    
- induction heads 如何形成？
    
- factual recall 的内部路径是什么？
    
- jailbreak 是否对应特定特征方向？
    
- sparse autoencoder 能否分解语义特征？
    

这个方向理论味更重，也很适合培养深层直觉。

### 方向 G：安全与对齐

问题：

- 模型为什么会 hallucinate？
    
- 为什么会 reward hacking？
    
- 拒答边界如何控制？
    
- 模型是否会隐藏推理？
    
- preference tuning 是否牺牲真实性？
    
- 如何评估危险能力？
    

这个方向重要但复杂，需要谨慎实验设计。

---

# 九、建立 LLM 科研直觉的训练方法

---

## 方法 1：每读一篇论文，写“5 句话卡片”

不要只做长笔记。

每篇论文读完写 5 句话：

1. 这篇论文解决什么问题？
    
2. 它的核心假设是什么？
    
3. 它改了 LLM 系统的哪一层？
    
4. 实验证据是否支持结论？
    
5. 它最可能的失败模式是什么？
    

例如 DPO：

1. 问题：RLHF 复杂且不稳定。
    
2. 假设：偏好优化可以通过重新参数化直接转成分类式目标。
    
3. 层级：后训练/偏好优化。
    
4. 证据：在若干偏好任务上匹配或超过 PPO-RLHF。
    
5. 失败模式：依赖偏好数据质量，可能受长度偏差和 reference model 影响。
    

---

## 方法 2：每个实验都做 ablation

你要养成 ablation 思维。

如果一个方法提升了分数，要问：

|Ablation|问题|
|---|---|
|去掉核心模块|提升还在吗？|
|换数据集|是否泛化？|
|换模型大小|是否 scale？|
|换随机种子|是否稳定？|
|控制 token budget|是否公平？|
|控制参数量|是否只是模型更大？|
|控制训练步数|是否只是训练更久？|
|控制 prompt|是否只是提示词更好？|

科研直觉：

> 没有 ablation 的 LLM 实验，很难说明因果。

---

## 方法 3：做小模型实验，不要只跑大模型 API

API 很方便，但科研直觉来自可控实验。

你至少应该训练过：

- 一个 tiny MLP；
    
- 一个 CNN；
    
- 一个 RNN/char LM；
    
- 一个 tiny GPT；
    
- 一个 LoRA fine-tuning；
    
- 一个 DPO/SFT 对比实验；
    
- 一个 RAG pipeline；
    
- 一个 evaluation harness。
    

小模型会暴露很多真实问题：

- loss 不降；
    
- 学习率爆炸；
    
- 数据格式错；
    
- tokenizer 错；
    
- padding mask 错；
    
- label shift 错；
    
- eval 泄漏；
    
- 训练集能过拟合但验证集不行；
    
- 生成质量和 loss 不一致。
    

这些问题会训练你的科研嗅觉。

---

## 方法 4：建立“错误分类本”

每次模型错了，不要只说“效果不好”。

把错误分类：

|错误类型|例子|
|---|---|
|知识缺失|不知道某事实|
|知识冲突|混淆两个实体|
|检索失败|没找到相关证据|
|证据忽略|找到了但没用|
|推理断裂|中间步骤错|
|指令不遵循|格式/约束错|
|过度拒答|安全边界太宽|
|幻觉|编造来源或事实|
|长上下文遗忘|忘记前文信息|
|工具使用错误|调错 API 或误读结果|
|评估误判|评分脚本/LLM judge 错|

科研直觉来自错误分析，而不是来自排行榜。

---

## 方法 5：每周复现一个小结果

不要一开始追求复现 GPT-4 级别论文。

你可以复现这些小现象：

1. 学习率太大导致 loss 爆炸；
    
2. residual connection 让深层 MLP 更容易训练；
    
3. LayerNorm 改善 Transformer 稳定性；
    
4. temperature 增大让生成更多样但更不稳定；
    
5. top-p 和 top-k 的差异；
    
6. LoRA rank 增大对小任务的影响；
    
7. SFT 让模型更听话；
    
8. DPO 改变输出偏好；
    
9. RAG 降低时效性问题；
    
10. CoT 提高部分数学题正确率；
    
11. self-consistency 提升但增加推理成本；
    
12. 长上下文中间位置的信息更容易被忽略。
    

---

# 十、LLM 研究入门的 12 周路线

---

## 第 1-2 周：深度学习基本功

目标：

- 手写 MLP；
    
- 理解 forward/backward；
    
- 熟悉 PyTorch；
    
- 会看训练曲线。
    

任务：

- 手写 softmax regression；
    
- 手写两层 MLP；
    
- 用 PyTorch 重写；
    
- 在 MNIST/CIFAR-10 上训练；
    
- 调学习率、batch size、weight decay。
    

验收标准：

> 你能解释 loss 为什么不降、为什么过拟合、为什么梯度爆炸。

---

## 第 3-4 周：优化与训练稳定性

目标：

- 理解初始化、归一化、残差；
    
- 掌握 AdamW 和学习率调度；
    
- 建立 debug checklist。
    

任务：

- 比较 SGD、Momentum、Adam；
    
- 比较有无 LayerNorm；
    
- 比较有无 residual；
    
- 记录梯度范数；
    
- 做一次系统调参实验。
    

验收标准：

> 你能根据训练曲线猜出可能问题，而不是盲目调参。

---

## 第 5-6 周：Transformer from scratch

目标：

- 从零实现 tiny GPT；
    
- 理解 attention、mask、position、sampling。
    

任务：

- 实现 causal self-attention；
    
- 实现 multi-head attention；
    
- 实现 Transformer block；
    
- 训练字符级语言模型；
    
- 做 attention 可视化；
    
- 比较 greedy、temperature、top-k、top-p。
    

验收标准：

> 你能不看代码解释 GPT 一次 forward pass 发生了什么。

---

## 第 7 周：Scaling 与数据

目标：

- 理解参数、token、compute 的关系；
    
- 认识数据质量和数据混合。
    

任务：

- 训练不同大小的小 GPT；
    
- 控制 token 数；
    
- 比较 loss 曲线；
    
- 做小型 scaling plot；
    
- 尝试不同数据混合。
    

验收标准：

> 你能解释为什么“更大模型 + 更少数据”未必更好。

---

## 第 8 周：SFT 与 LoRA

目标：

- 理解 base model 到 instruction model 的变化；
    
- 学会参数高效微调。
    

任务：

- 选一个开源小模型；
    
- 做 LoRA SFT；
    
- 比较微调前后；
    
- 分析遗忘和格式遵循。
    

验收标准：

> 你能区分“模型学会新知识”和“模型学会新行为”。

---

## 第 9 周：Preference tuning：DPO/RLHF

目标：

- 理解偏好数据如何塑造输出；
    
- 知道 RLHF 和 DPO 的差异。
    

任务：

- 构造 chosen/rejected 数据；
    
- 跑 DPO；
    
- 比较 SFT vs DPO；
    
- 检查长度偏差和风格偏差。
    

验收标准：

> 你能解释为什么“更讨人喜欢”不一定等于“更真实”。

---

## 第 10 周：RAG 与评估

目标：

- 理解参数记忆和外部记忆；
    
- 建立 eval 意识。
    

任务：

- 搭一个小 RAG；
    
- 比较无检索/有检索；
    
- 设计 50 条测试集；
    
- 做错误分类；
    
- 尝试 lm-evaluation-harness 或自定义 eval。
    

验收标准：

> 你能指出模型错误是检索错、生成错、还是评估错。

---

## 第 11 周：推理增强

目标：

- 理解 CoT、self-consistency、verifier、tool use。
    

任务：

- 在数学/逻辑小任务上比较 direct answer vs CoT；
    
- 做多次采样投票；
    
- 加一个简单 verifier；
    
- 比较 accuracy 与 token cost。
    

验收标准：

> 你能用 test-time compute 的角度解释推理增强。

---

## 第 12 周：选科研方向

目标：

- 从阅读者变成研究者。
    

任务：

- 选 1 个方向；
    
- 读 10 篇论文；
    
- 写 10 张 5 句话卡片；
    
- 复现一个小实验；
    
- 提出 3 个可验证假设；
    
- 做一个 ablation。
    

验收标准：

> 你能提出一个小但清晰的研究问题，并设计实验验证它。

---

# 十一、推荐学习资料组合

建议你按这个顺序学：

## 第一层：深度学习基础

1. 《Dive into Deep Learning》：边看边跑代码，适合建立实践直觉。([深度学习导论](https://d2l.ai/?utm_source=chatgpt.com "D2L - Dive into Deep Learning 1.0.3 documentation"))
    
2. 《Deep Learning》：补理论，尤其线性代数、概率、优化、深度网络基础。([深度学习书籍](https://www.deeplearningbook.org/?utm_source=chatgpt.com "Deep Learning"))
    
3. CS231n：训练和调试神经网络，建立实验基本功。([CS231n](https://cs231n.stanford.edu/?utm_source=chatgpt.com "Stanford University CS231n: Deep Learning for Computer Vision"))
    

## 第二层：NLP 与 Transformer

1. CS224N：系统理解深度学习 NLP 和 LLM。([Stanford University](https://web.stanford.edu/class/cs224n/?utm_source=chatgpt.com "Stanford CS 224N | Natural Language Processing with Deep Learning"))
    
2. Transformer 原论文：理解 attention-only 架构为什么重要。([arXiv](https://arxiv.org/abs/1706.03762?utm_source=chatgpt.com "[1706.03762] Attention Is All You Need - arXiv.org"))
    
3. Hugging Face Transformers：熟悉现代模型生态和实践工具。([Hugging Face](https://huggingface.co/docs/transformers/index?utm_source=chatgpt.com "Transformers · Hugging Face"))
    

## 第三层：LLM 全栈

1. Stanford CS336：从语言建模、tokenization、FLOPs、架构、attention alternatives、MoE、并行、scaling、inference、evaluation、data 到 post-training。([Stanford CS336](https://cs336.stanford.edu/?utm_source=chatgpt.com "Stanford CS336 | Language Modeling from Scratch"))
    
2. Scaling Laws / Chinchilla：理解数据、模型和计算预算的关系。([arXiv](https://arxiv.org/abs/2001.08361?utm_source=chatgpt.com "[2001.08361] Scaling Laws for Neural Language Models - arXiv.org"))
    
3. InstructGPT / DPO：理解从 base model 到 assistant 的后训练路径。([arXiv](https://arxiv.org/abs/2203.02155?utm_source=chatgpt.com "Training language models to follow instructions with human feedback"))
    

## 第四层：研究工具

1. lm-evaluation-harness：做可复现 benchmark。([GitHub](https://github.com/EleutherAI/lm-evaluation-harness?utm_source=chatgpt.com "EleutherAI/lm-evaluation-harness - GitHub"))
    
2. HELM：理解多维度、标准化、透明评估的重要性。([斯坦福CRFM](https://crfm.stanford.edu/helm/?utm_source=chatgpt.com "Holistic Evaluation of Language Models (HELM) - Stanford University"))
    
3. Transformer Circuits / TransformerLens：进入机制可解释性。([变压器电路](https://transformer-circuits.pub/?utm_source=chatgpt.com "Transformer Circuits Thread"))
    

---

# 十二、你最终应该形成的 LLM 科研直觉清单

你可以把下面当成自测表。

看到一个 LLM 现象时，你应该能问：

1. 这是 pretraining 能力，还是 post-training 行为？
    
2. 是参数知识，还是上下文检索？
    
3. 是模型真的会推理，还是 prompt 激发了模式？
    
4. 提升是否来自更多 test-time compute？
    
5. 是否控制了模型大小、数据量、训练步数和 token budget？
    
6. benchmark 是否可能污染？
    
7. 是否做了错误分析？
    
8. 是否有 ablation？
    
9. 是否跨模型规模成立？
    
10. 是否跨数据集成立？
    
11. 是否只是输出风格更像标准答案？
    
12. 失败案例是什么？
    
13. 这个方法是否增加推理成本？
    
14. 是否牺牲其他能力？
    
15. 小模型实验能否验证机制？
    
16. 这个结果能否复现？
    
17. 数据质量是否比算法更可能解释提升？
    
18. 系统瓶颈在哪里？
    
19. 安全和鲁棒性影响是什么？
    
20. 这个问题是否值得研究？
    

如果你能自然问出这些问题，你已经开始具备 LLM 科研直觉。

---

# 十三、最后给你一个总纲

深度学习的底层逻辑是：

$$
\text{可微分函数}  
+  
\text{数据}  
+  
\text{目标函数}  
+  
\text{优化}  
+  
\text{归纳偏置}  
$$

LLM 的底层逻辑是：

$$
\text{Transformer}  
+  
\text{大规模文本}  
+  
\text{next-token prediction}  
+  
\text{scaling}  
+  
\text{post-training}  
+  
\text{test-time compute}  
+  
\text{evaluation}  
$$

LLM 科研的底层判断是：

$$
\text{能力从哪里来？}  
$$

$$
\text{证据是否支持？}  
$$

$$
\text{成本是什么？}  
$$

$$
\text{失败模式在哪里？}  
$$

你接下来学习时，不要只是追论文标题。每学一个模型、算法或技术，都把它放回这张图里：

$$
\text{数据}  
\rightarrow  
\text{token}  
\rightarrow  
\text{表示}  
\rightarrow  
\text{attention/MLP}  
\rightarrow  
\text{loss}  
\rightarrow  
\text{优化}  
\rightarrow  
\text{能力}  
\rightarrow  
\text{后训练}  
\rightarrow  
\text{推理}  
\rightarrow  
\text{评估}  
$$

真正的 LLM 科研直觉，就是能在这条链路上定位问题、提出假设、设计实验、解释结果。

下面这套直觉可以先记成一句话：

**LLM 里的 RL 大多不是在“灌新知识”，而是在把模型已有或可探索到的行为重新排序：让高奖励轨迹的概率上升，让低奖励轨迹的概率下降。Agentic RL 再往前走一步：不仅排序答案，还排序“观察—思考—行动—工具—反馈—再行动”的整条交互策略。**

---

## 1. 先建立一个坐标系：RL 到底在学哪一维？

把 LLM 能力拆成几个“知识维度”：

|维度|它在模型里表现成什么|典型训练信号|
|---|---|---|
|**事实知识**|知道某个概念、API、定理、世界事实|预训练、检索、工具反馈|
|**程序性知识**|会一步步解题、写代码、规划任务|SFT、CoT 数据、RLVR、过程奖励|
|**偏好知识**|知道人类更喜欢哪种回答|RLHF、DPO、RLAIF|
|**验证知识**|知道什么答案能通过单测、判题器、格式检查|RLVR、execution reward|
|**交互知识**|知道什么时候搜索、调用工具、读环境状态|tool-use RL、agentic RL|
|**信用分配知识**|知道“前面哪一步导致最后成功/失败”|PPO/GRPO/actor-critic、过程奖励、trajectory-level RL|
|**元控制知识**|知道什么时候多想、什么时候停止、什么时候换策略|RLVR、agentic RL、search-based training|

很多误解来自把这些维度混在一起。比如：

**RLHF 主要学偏好知识，不保证学会数学。**  
**RLVR 主要学可验证任务里的成功策略，不保证回答更有礼貌。**  
**Agentic RL 主要学交互控制，不等于自动获得更多事实知识。**

---

## 2. LLM-RL 的核心直觉：概率质量搬运

给定 prompt `x`，模型会生成回答 `y`。RL 做的事可以粗略看成：

[  
$max_\pi \mathbb{E}_{y \sim \pi(\cdot|x)}[R(x,y)] - \beta , KL(\pi || \pi_{ref})$  
]

也就是：

**奖励越高的回答，概率上升；偏离原模型太远，会被 KL 惩罚拉回来。**

PPO 之所以常用于 RLHF，是因为它在“采样—优化 surrogate objective”的过程中限制单次策略更新不要太激进；原始 PPO 论文强调它在稳定性、样本效率和实现复杂度之间做了折中。([arXiv](https://arxiv.org/abs/1707.06347 "[1707.06347] Proximal Policy Optimization Algorithms")) InstructGPT/RLHF 的经典流程是：先用人工示范做 SFT，再收集模型输出排序，训练 reward model，最后用 RLHF 微调模型；论文报告 InstructGPT 在人类偏好、truthfulness 和 toxicity 上都有改善。([arXiv](https://arxiv.org/abs/2203.02155 "[2203.02155] Training language models to follow instructions with human feedback"))

所以你读任何 LLM-RL 方法时，可以先问：

> 它的 reward 到底奖励了哪类轨迹？  
> 它的 KL 或 reference model 又在阻止模型忘掉什么？

这两个问题通常比算法名更重要。

---

## 3. 各种策略到底学什么、忽略什么？

### 1）SFT：模仿“应该怎么答”

**学的维度：** 表达格式、任务模板、示范里的解题路径、基本对话风格。  
**忽略的维度：** 它不知道示范之外还有哪些更优答案；也不会主动探索。SFT 是“老师怎么写，我就怎么写”。

直觉：SFT 像临摹字帖。它能让模型像专家，但不能保证模型知道为什么这样做是最优。

---

### 2）RLHF / PPO：学习“人类更喜欢哪种整体回答”

**学的维度：** helpfulness、harmlessness、honesty、礼貌程度、结构化表达、拒答风格、偏好对齐。  
**忽略的维度：** 如果 reward model 不懂数学、不懂代码、不懂真实世界，它就可能奖励“看起来好”的错答案。RLHF 学的是 reward model 里的偏好，而不是宇宙真理。

InstructGPT 这类 RLHF 流程用人工排序训练 reward model，再用 RL 优化模型输出，使其更符合人类意图。([arXiv](https://arxiv.org/abs/2203.02155 "[2203.02155] Training language models to follow instructions with human feedback")) 这解释了为什么 RLHF 很适合“助手化”，但单独靠它不一定解决深层推理。

直觉：RLHF 像训练销售/客服。它让模型更会“对人说话”，但不一定让模型更懂数学。

---

### 3）DPO：离线偏好学习，不显式跑 RL

**学的维度：** 成对偏好里的相对选择：chosen 比 rejected 更好。  
**忽略的维度：** 没有在线探索；看不到当前策略会产生哪些新坏样本；通常也缺少多步环境反馈。

DPO 的核心贡献是把 RLHF 的 KL-constrained reward maximization 重新参数化，用一个分类式 loss 直接优化偏好数据，避免显式 reward model 和 RL 采样训练。([arXiv](https://arxiv.org/abs/2305.18290 "[2305.18290] Direct Preference Optimization: Your Language Model is Secretly a Reward Model"))

直觉：DPO 像刷选择题：“A 比 B 好”。它稳定、便宜，但它只在已有比较数据的边界内学习。

---

### 4）RLAIF / Constitutional AI：学习“原则下的偏好”

**学的维度：** 安全规范、拒答边界、原则一致性。  
**忽略的维度：** 原则本身可能不完整；AI judge 可能有盲点；有些价值判断依然需要人类语境。

Constitutional AI 的思路是用一组原则减少人工 harm labels，并包含监督学习与 RL from AI Feedback 阶段。([arXiv](https://arxiv.org/abs/2212.08073?utm_source=chatgpt.com "Constitutional AI: Harmlessness from AI Feedback"))

直觉：RLAIF 像给模型一本“行为宪法”。它擅长塑造边界，但边界之外的复杂判断仍可能出错。

---

### 5）RLVR：用可验证奖励学“做对”

RLVR = Reinforcement Learning with Verifiable Rewards。典型例子是：

数学题：答案对/错。  
代码题：单测 pass/fail。  
格式任务：是否满足格式约束。  
工具任务：环境是否返回成功。

**学的维度：** 可验证任务里的成功策略、答案校验意识、长 CoT、试错倾向。  
**忽略的维度：** 不可验证质量，比如审美、帮助性、语气、开放问题的深度；也可能只学会钻 verifier 的空子。

GRPO 是 PPO 的变体。DeepSeekMath 论文提出 GRPO，并称其在增强数学推理能力的同时减少 PPO 的内存开销。([arXiv](https://arxiv.org/abs/2402.03300 "[2402.03300] DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models")) 后续 RLVR 分析把 GRPO 理解为用 verifiable/binary rewards 放大成功概率：对同一个 prompt 采样一组回答，用组内 reward 均值和方差估计 advantage，而不是训练单独 critic。([arXiv](https://arxiv.org/html/2503.06639v1 "Reinforcement Learning with Verifiable Rewards: GRPO’s Effective Loss, Dynamics, and Success Amplification"))

GRPO 的直觉公式可以记成：

[  
$$A_i \approx \frac{R_i - mean(R_{group})}{std(R_{group})}$$

]

同一个题采样 8 个答案：  
如果有的对、有的错，模型就知道“这类轨迹比那类轨迹好”。  
如果 8 个全错，几乎没信号。  
如果 8 个全对，也几乎没信号。

所以 RLVR 最喜欢的训练题不是“太难全错”或“太简单全对”，而是**模型有一定概率做对、但还不稳定**的题。

关于 RLVR 是否只是提高采样效率，还是确实增强 reasoning，近期有争论。一篇 2025/2026 的分析总结了这个争论，并提出用 CoT-Pass@K 等指标区分“最终答案碰巧对”和“推理路径也对”；作者认为 RLVR 可以在某些条件下隐式激励正确推理。([arXiv](https://arxiv.org/html/2506.14245v2 "Reinforcement Learning with Verifiable Rewards Implicitly Incentivizes Correct Reasoning in Base LLMs"))

直觉：RLVR 像刷有标准答案的竞赛题。它不一定让你更会聊天，但会逼你把“能得分的解法”概率提上去。

---

### 6）过程奖励 / PRM：学习“每一步对不对”

**学的维度：** 中间推理步骤的局部正确性、信用分配、错误定位。  
**忽略的维度：** 标注成本高；局部步骤看起来合理，不一定保证全局最优；reward model 本身也会错。

“Let’s Verify Step by Step” 对比了 outcome supervision 和 process supervision，发现对 MATH 这类多步推理任务，过程监督显著优于只看最终答案，并发布了 PRM800K 这类步骤级反馈数据。([arXiv](https://arxiv.org/abs/2305.20050 "[2305.20050] Let's Verify Step by Step"))

直觉：Outcome reward 是“最后答案对了给分”；process reward 是“每一步老师都批改”。后者更会教推理，但更贵。

---

### 7）Agentic RL：学习“在环境中怎么行动”

普通 LLM-RL 通常是：

```text
prompt -> response -> reward
```

Agentic RL 是：

```text
observation_t -> thought/action/tool_call_t -> environment -> observation_{t+1} -> ... -> reward
```

这里模型不只是写答案，而是在动态环境里做决策。Agentic RL survey 把这个转变描述为：从被动 sequence generator 变成嵌入复杂动态世界的 autonomous decision-making agent，并用 temporally extended POMDP 而不是单步 MDP 来刻画。([arXiv](https://arxiv.org/abs/2509.02547 "[2509.02547] The Landscape of Agentic Reinforcement Learning for LLMs: A Survey"))

Agent Lightning 这类近期工作也强调把 agent execution 形式化为 MDP，并加入 trajectory decomposition / credit assignment，让 RL 可以处理 text-to-SQL、RAG、math tool-use、多智能体、动态 workflow 等复杂交互。([arXiv](https://arxiv.org/abs/2508.03680 "[2508.03680] Agent Lightning: Train ANY AI Agents with Reinforcement Learning"))

**学的维度：**  
什么时候搜索、查哪条信息、调用哪个工具、是否重试、如何利用观察、如何规划多步任务。

**忽略或困难的维度：**  
长程信用分配很难；环境非平稳；工具失败会污染 reward；模型可能学会 reward hacking；轨迹越长，实验越难复现。

直觉：Agentic RL 不像“写作文得分”，更像“玩密室逃脱”。奖励可能只在最后出现，但成败取决于前面一串动作。

---

## 4. 一张总表：策略 vs 学习维度 vs 盲区

|方法|主要学习什么|它不太管什么|典型风险|
|---|---|---|---|
|Pretraining|语言、事实、世界模型、隐式技能|用户偏好、任务目标|会说但不一定听话|
|SFT|模仿专家答案|探索、最优性、负例|学到表面模板|
|RLHF/PPO|人类偏好、助手风格|真正确性、复杂推理|reward hacking、过度迎合|
|DPO|离线 pairwise preference|在线探索、环境反馈|数据边界外泛化弱|
|RLAIF/CAI|原则、安全边界|人类细腻偏好|judge bias、原则空洞|
|RLVR/GRPO|可验证成功、数学/代码/格式|开放质量、语气、不可验证目标|钻 verifier、答案对但过程假|
|PRM/process reward|步骤级推理质量|标注成本、全局目标|局部合理但全局错|
|Tool-use RL|工具选择、调用时机|工具外的语言质量|乱调用工具、成本升高|
|Agentic RL|多步规划、环境交互、记忆控制|稳定性、安全边界、长程 credit|轨迹变长后训练不稳|

---

# 5. 怎么读 RL 实验曲线？

你读曲线时不要先看 loss。先问这五个问题：

1. **最终 eval success 有没有涨？**
    
2. **train reward 涨，是不是 eval 也涨？**
    
3. **KL、entropy、response length 有没有异常？**
    
4. **奖励来自哪里：reward model、verifier、human eval、unit test，还是 LLM judge？**
    
5. **训练集涨，未见题/未见环境涨不涨？**
    

RL 里最危险的情况是：**优化指标很好看，真实能力没变强。**

---

## 6. LLM-RL 最常见曲线及含义

### A. Reward curve

理想情况：

```text
train reward ↑
eval reward / pass@1 ↑
train-eval gap 不大
```

危险情况：

```text
train reward ↑
eval success 不涨甚至下降
```

这通常意味着：reward 被 hack 了，或者模型过拟合训练 prompt / verifier / judge。

在 RLHF 里，这可能是 reward model 喜欢“长、礼貌、自信”的回答。  
在 RLVR 里，这可能是模型学会利用格式漏洞、测试漏洞、答案猜测。  
在 agentic RL 里，这可能是模型学会触发环境的奖励漏洞，而不是真正完成任务。

---

### B. KL curve

KL 表示当前 policy 离 reference model 多远。

**KL 太低：**

```text
reward 不涨
KL 接近 0
entropy 变化小
```

说明模型几乎没动。可能是 learning rate 太小、KL penalty 太强、reward 太弱、advantage 全是 0。

**KL 稳定中等：**

```text
reward ↑
KL 缓慢 ↑ 或稳定在目标区间
eval ↑
```

这是健康训练。

**KL 突然爆炸：**

```text
reward spike
KL spike
entropy cliff
eval crash
```

通常是策略更新太猛。优先检查 learning rate、clip range、batch size、reward scale、KL coefficient。

---

### C. Entropy curve

Entropy 是模型输出分布的多样性。

**慢慢下降**是正常的：模型越来越确定哪些轨迹好。  
**突然掉到很低**要警惕：可能 mode collapse。  
**一直很高**也可能不对：模型还在乱采样，没有形成稳定策略。

在 RLVR 中，entropy 下降常常伴随 pass@1 上升，但 pass@K 可能下降。也就是说，模型更会稳定地产生某类答案，但探索空间变窄。近期 RLVR 讨论里，pass@1、pass@K 和 CoT-Pass@K 的差异正是判断“只是采样效率提升”还是“推理边界扩展”的关键。([arXiv](https://arxiv.org/html/2506.14245v2 "Reinforcement Learning with Verifiable Rewards Implicitly Incentivizes Correct Reasoning in Base LLMs"))

---

### D. Response length / CoT length

这条曲线非常重要。

**好情况：**

```text
pass@1 ↑
length 小幅 ↑
reasoning 更充分
```

**坏情况：**

```text
reward ↑
length ↑↑
pass@1 不涨
```

说明模型可能学会了“写得更长看起来更聪明”。RLHF、LLM judge、PRM 都容易奖励冗长。数学 RLVR 也可能出现“长 CoT 但逻辑质量没变”。

读这条曲线时要配合：

```text
final accuracy
CoT correctness
token cost
latency
```

不要看到长 CoT 就默认 reasoning 变强。

---

### E. PPO / GRPO 的 clip fraction

clip fraction 高，说明很多 token 的 policy ratio 被截断。

**clip fraction 长期很高：** 更新太猛，policy 想大幅偏离 old policy。  
**clip fraction 接近 0 且 reward 不涨：** 更新太保守，学不动。  
**clip fraction 周期性尖峰：** 可能 batch reward 方差很大，或者某些 batch 特别容易被 reward hack。

---

### F. Advantage 分布

PPO/GRPO 真正更新靠 advantage。

对 GRPO 来说，最关键的是组内 reward 差异：

```text
同一 prompt 采样 G 个答案
有对有错 -> 有学习信号
全对/全错 -> advantage 接近没信息
```

所以你读 RLVR 实验时，要特别看：

```text
per-prompt reward variance
non-zero advantage ratio
all-correct group ratio
all-wrong group ratio
```

如果训练早期大量 all-wrong，说明题太难或 base model 太弱。  
如果训练后期大量 all-correct，说明训练集已饱和，继续训容易过拟合或压缩探索。

---

### G. Value loss / explained variance

这主要用于 PPO with critic。

**value loss 爆炸：** critic 学不会 reward，advantage 很噪。  
**explained variance 很低：** critic 对回报预测没用。  
**policy reward 振荡且 value loss 振荡：** 可能 reward scale、GAE、batch size、critic lr 有问题。

GRPO 的一个直觉优势就是不用单独训练 critic，而用同一 prompt 的 group rewards 做相对 baseline；这也是它在 LLM reasoning RL 中受欢迎的原因之一。([arXiv](https://arxiv.org/html/2503.06639v1 "Reinforcement Learning with Verifiable Rewards: GRPO’s Effective Loss, Dynamics, and Success Amplification"))

---

## 7. 一套“曲线诊断表”

|曲线现象|第一解释|优先排查|
|---|---|---|
|reward ↑，eval ↑，KL 稳|健康学习|继续看泛化和成本|
|reward ↑，eval 不涨|reward hacking / 过拟合|换 verifier、人工抽检、OOD eval|
|train ↑，test ↓|训练题记忆|数据泄漏、prompt overlap|
|KL ≈ 0，reward 平|学不动|KL penalty、lr、reward scale|
|KL spike，eval crash|更新过大|lr、clip range、batch size|
|entropy cliff|mode collapse|KL、temperature、reward 过强|
|length ↑↑，accuracy 平|verbosity hacking|length penalty、分开评估简洁性|
|pass@1 ↑，pass@K ↓|策略变窄|entropy、diversity、sampling eval|
|all reward = 0|探索失败|降低难度、curriculum、SFT warmup|
|all reward = 1|任务太简单|换更难数据|
|invalid tool calls ↓，success ↑|agent 学会工具协议|继续看是否泛化|
|tool calls ↑，success 不涨|乱用工具|tool cost penalty、action mask|
|trajectory length ↑，success 不涨|agent 绕路|step penalty、subgoal reward|
|judge score ↑，human score ↓|judge 被 hack|多 judge、blind human eval|

---

## 8. Agentic RL 的曲线要额外看什么？

普通 LLM-RL 看 response-level metrics 就够一半。Agentic RL 不够，因为最终 success 掩盖了中间过程。

Agentic RL 至少要拆这些曲线：

```text
episode success rate
average episode length
tool call count
invalid action rate
environment error rate
subgoal success rate
retrieval precision
memory read/write count
backtracking rate
cost per success
latency per success
```

一个 agent 可能 final success 提高，但 tool call 增加 5 倍、latency 增加 10 倍。那它不是“更聪明”，可能只是“更贵地蒙对”。

我建议你读 agentic RL 曲线时用这个顺序：

```text
1. 成功率涨了吗？
2. 成本涨了多少？
3. 无效动作少了吗？
4. 轨迹更短还是更长？
5. 工具使用更精准还是更频繁？
6. 换环境/换任务还有效吗？
7. 是否出现奖励漏洞？
```

---

## 9. 一个很实用的实验直觉：RL 不是看一条曲线，而是看“曲线组合”

### 健康 RLVR 训练长这样

```text
pass@1 ↑
train reward ↑
heldout reward ↑
KL 缓慢 ↑
entropy 缓慢 ↓
length 小幅 ↑ 或稳定
all-wrong group ratio ↓
all-correct group ratio 逐渐 ↑
```

解释：模型正在把正确轨迹概率提上去，而且没有明显崩。

---

### Reward hacking 长这样

```text
train reward ↑↑
eval pass@1 平 / ↓
KL ↑
length ↑↑
human eval ↓
```

解释：模型学会了骗 reward，而不是做任务。

---

### 学不动长这样

```text
reward 平
KL ≈ 0
entropy 平
advantage variance 很低
```

解释：要么 reward 没信号，要么 KL 太强，要么任务太难/太易。

---

### 过度收缩长这样

```text
pass@1 ↑
pass@K ↓
entropy ↓↓
response diversity ↓
```

解释：模型变得更稳定，但探索空间变小。数学里可能 pass@1 变好，但多样解法减少；agent 里可能固定使用某个套路，换环境就坏。

---

### Agent 学会“绕路”长这样

```text
success ↑ 小幅
trajectory length ↑↑
tool calls ↑↑
cost ↑↑
invalid actions 不降
```

解释：agent 不是学会规划，而是学会堆工具、堆步骤、堆重试。

---

## 10. 读论文时的一套模板

以后你看任何 RL in LLM / agentic RL 论文，可以按这几句拆：

```text
1. Policy 是什么？
   是 token policy、response policy，还是 agent action policy？

2. State / observation 是什么？
   只有 prompt？还是有工具返回、网页、代码执行结果、memory？

3. Action 是什么？
   生成 token、生成完整回答、调用工具、写代码、点击 UI？

4. Reward 是什么？
   人类偏好、AI judge、单测、数学答案、环境成功、过程奖励？

5. Credit assignment 怎么做？
   整个 response 一个 reward？每一步 reward？critic？GRPO group baseline？

6. Regularization 是什么？
   KL to SFT model？clip？length penalty？tool cost penalty？

7. 它真正优化了哪一维？
   偏好、正确性、格式、工具调用、规划、记忆、成本？

8. 它没评价哪一维？
   泛化、安全、真实性、成本、长程稳定性、OOD 环境？
```

你只要这样问，很多论文的本质会立刻清楚。

---

## 11. 给你一个学习路线

先学普通 RL 的最小核心：

```text
policy
reward
return
advantage
baseline
entropy
KL
credit assignment
exploration vs exploitation
```

然后学 LLM-RL：

```text
SFT -> reward model -> PPO/RLHF -> DPO -> RLVR/GRPO -> PRM
```

最后学 agentic RL：

```text
MDP/POMDP
trajectory
tool action
environment feedback
long-horizon credit assignment
cost-aware reward
multi-turn evaluation
```

最重要的心法是：

> 看到一个算法，不要先问“它是不是 RL”。  
> 先问：它把哪个行为的概率提高了？奖励来自哪里？谁给信用？哪些质量维度没有被奖励？

如果你能稳定回答这四个问题，就已经开始具备读 RL in LLM 和 agentic RL 论文的直觉了。