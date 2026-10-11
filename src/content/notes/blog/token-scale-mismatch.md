---
title: "同一把尺子，量不同的 token：后训练中的更新尺度与概率失配"
description: "从 DFT 与 RIPO 出发，用四种“变化量”统一梳理 SFT 与 RL 中更新尺度与 token 概率失配的问题，以及各方法的修正思路。"
tags:
  - "post-training"
  - "rl"
  - "sft"
  - "survey"
category: "blog"
date: "2026-07-23"
updated: "2026-07-23"
---

之前读了 **On the Generalization of SFT: A Reinforcement Learning Perspective with Reward Rectification**，它用 RL 视角重写 SFT，发现标准 SFT 隐含了对低概率 token 的逆概率放大，并提出 DFT。最近又读了 **Beyond Euclidean Clipping: Overcoming Exploration Collapse in LLM RL via Riemannian Isometric Policy Optimization**，它指出 PPO 式 clipping 用欧氏距离约束 ratio，对高概率与低概率 token 造成几何失配，提出 RIPO。

这两篇工作有一个共同的内核：**更新尺度与 token 概率的失配**——同一个名义上的"更新幅度"（ratio 区间、损失权重、梯度范数），落在不同概率的 token 上，实际含义完全不同。索性把这一类问题做个系统梳理。

## 统一框架：四种"变化量"

考虑旧策略下某个已采样 token 的概率

$$p=\pi_{\mathrm{old}}(a\mid s),$$

以及新旧策略比率

$$r=\frac{\pi_{\theta}(a\mid s)}{\pi_{\mathrm{old}}(a\mid s)}.$$

围绕"一次更新到底改变了多少"，至少有四种不同的度量，它们并不等价：

**1. 概率的绝对变化量**

$$\Delta p = p\,(r-1).$$

若固定 $|r-1|\le\epsilon$，则 $|\Delta p|\le p\,\epsilon$：低概率 token 的绝对概率变化被压得很小。这是 DAPO、DCPO、DPPO、BandPO、RIPO 关注的问题——**低概率但有价值的 token 无法快速提升概率**。

**2. 策略分布的局部距离**

对单个动作的扰动，KL 散度的二阶（局部）近似为

$$D_{\mathrm{KL}}\approx\frac{1}{2}\,p\,(r-1)^2.$$

相同的 ratio 变化，对不同 $p$ 的 token 对应的分布距离并不相同。这是 RIPO、DPPO、BandPO 关注的问题——**固定的 ratio clipping 并不是真正统一的 trust region**。

**3. 参数（logit）空间的梯度范数**

一个被采样到的低概率 token 可能有很大的

$$\bigl\|\nabla_\theta\log\pi_\theta(a\mid s)\bigr\|.$$

尤其当整条回答共享同一个 outcome reward 时，某些偶然出现、对推理没有贡献的稀有 token 也会继承整条轨迹的 advantage，从而产生很大的参数梯度。这是 **Do Not Let Low-Probability Tokens Over-Dominate**（Lopti）和 STAPO 关注的问题——**低概率 token 有时不是学不到，而是反过来支配了参数更新**。

**4. SFT 的隐式奖励权重**

DFT 把 SFT 损失从策略梯度角度重写后，得到一个与 $1/p$ 成比例的隐式奖励（重要性权重）。当专家 token 在当前模型下概率极低时，这个重写中的方差可能无界。DFT 因此用 token 概率重新缩放 SFT 损失。

> 这四种量分别活在**概率空间**、**分布几何空间**、**参数空间**和**隐式奖励空间**。很多表面上"互相矛盾"的结论（比如"低概率 token 学不动" vs "低概率 token 更新过强"），其实只是在不同空间里测量同一件事。

---

## 第一条线：SFT 中的概率相关重加权

## DFT：Dynamic Fine-Tuning

**On the Generalization of SFT: A Reinforcement Learning Perspective with Reward Rectification**

DFT 指出，标准 SFT 的 RL 重写隐含了不合理的逆概率奖励结构：

$$r_{\mathrm{implicit}}\propto\frac{1}{p_t}.$$

于是把标准损失 $-\log p_t$ 改为

$$\mathcal{L}_{\mathrm{DFT}}=-\operatorname{sg}(p_t)\,\log p_t,$$

其中 $\operatorname{sg}(\cdot)$ 为 stop-gradient。这样梯度被 $p_t$ 缩放，中和了隐式的逆概率权重，也缓解了对应的无界方差问题，降低模型原本极不认可的专家 token 的影响。

DFT 的直觉是：**单条 off-policy 示范并不足以证明模型极不认可的 token 必须被强制学会。**

## ASFT：给 DFT 加一个分布锚点

**Anchored Supervised Fine-Tuning**

后续工作指出，DFT 的权重来自正在训练的在线模型，权重本身随训练变化，可能造成自增强式的分布漂移：

$$p_t\uparrow\;\Rightarrow\;w_t\uparrow\;\Rightarrow\;p_t\ \text{进一步上升}.$$

ASFT 在 DFT 之上增加相对固定参考模型的 KL 正则：

$$\mathcal{L}_{\mathrm{ASFT}}=\mathcal{L}_{\mathrm{DFT}}+\beta\,D_{\mathrm{KL}}\!\left(\pi_\theta\,\|\,\pi_{\mathrm{ref}}\right),$$

目标是既保留 DFT 的概率重加权，又防止策略不断偏离初始化模型。

## PriFT：不要用在线概率算权重

**Prior-Support Guided Supervised Fine-Tuning**

PriFT 对 DFT 的批评更直接：在线模型的概率是一个不断移动的目标，会产生自增强偏差。因此它改用**冻结的预训练模型**计算 token 权重：

$$w_t=f\!\left(\pi_{\mathrm{pretrain}}(y_t\mid x,y_{<t})\right),$$

再训练在线模型：

$$\mathcal{L}=-\sum_t w_t\log\pi_\theta(y_t\mid x,y_{<t}).$$

这样权重衡量的是"该 token 在微调之前是否受到预训练知识的支持"，而不是"它在微调过程中是否已经被学会"。PriFT 的分析还显示，DFT 可能使模型变得过度确定（over-confident），冻结的先验权重更有利于保留采样多样性。

## ProFit：直接屏蔽低概率 token

**ProFit: Leveraging High-Value Signals in SFT via Probability-Guided Fine-Tuning**

ProFit 的经验假设更强：

- 高概率 token 更多承载解题的逻辑框架；
- 低概率 token 更多是可替换的措辞或表面形式。

因此它选择性地 mask 掉低概率 token，避免 SFT 在措辞和轨迹细节上过拟合。它与 DFT 同属一个方向——

$$\boxed{\text{低概率的离线标签应该降权}}$$

——但 ProFit 是离散筛选，DFT 是连续加权。

## CADFT：从 token 失配扩展到样本失配

**Compatibility-Aware Dynamic Fine-Tuning**

CADFT 认为仅在 token 级做 DFT 还不够：不同示范与当前模型之间还存在**样本级**的兼容性差异。它在 DFT 的 token 重加权之外，进一步控制 demonstration–policy mismatch 引起的样本级优化方差。

---

## 第二条线：固定 clipping 压制低概率探索

## DAPO：最早明确提出 Clip-Higher

**DAPO: An Open-Source LLM Reinforcement Learning System at Scale**

DAPO 指出固定的上裁剪边界会限制低概率 token 的增长。由

$$p_{\mathrm{new}}\le p_{\mathrm{old}}\,(1+\epsilon_{\mathrm{high}}),$$

当 $p_{\mathrm{old}}=0.01,\ \epsilon_{\mathrm{high}}=0.2$ 时，一步更新后最多只能到 $0.012$。

DAPO 把上下 clipping 解耦（Clip-Higher），调大正向上界，给低概率探索 token 更多增长空间。其训练系统还结合了动态采样、token-level loss 和过长回答的奖励塑形。

**局限**：DAPO 对所有 token 仍使用同一个更高的上界 $r\le 1+\epsilon_{\mathrm{high}}$，是**全局放宽**，不是概率自适应。

## DCPO：按 token 先验概率动态裁剪

**Dynamic Clipping Policy Optimization**

DCPO 进一步主张不同 token 应有不同的 clipping 范围：

$$\epsilon_t=f(p_{\mathrm{old},t}),$$

低先验概率 token 获得更宽的裁剪边界，避免固定 clipping 大量制造零梯度。它还设计了平滑的 advantage 标准化，减少一组 rollout 奖励完全相同时的无效更新。

与 RIPO 相比：DCPO 主要从零梯度和数据利用率出发；RIPO 从 Fisher/KL 黎曼几何推导裁剪尺度。

## DPPO：不要近似 trust region，直接限制 divergence

**Rethinking the Trust Region in LLM Reinforcement Learning**

DPPO 认为 PPO ratio clipping 的根本问题是：对低概率 token 过度惩罚，对高概率 token 又约束不足。作者主张直接约束策略 divergence，而不是用固定 ratio 区间去近似。由于完整词表上的 divergence 计算和存储代价高，DPPO 采用 binary divergence、top-$K$ 等近似。

它与 RIPO 的问题诊断非常接近：

$$\boxed{\text{ratio 相同}\ \neq\ \text{分布变化相同}}$$

区别在于：DPPO 直接近似 divergence；RIPO 把 divergence 的局部几何投影为 token-specific 的 ratio bound。

## BandPO：从任意 f-divergence 推导动态裁剪带

**BandPO: Bridging Trust Regions and Ratio Clipping via Probability-Aware Bounds**

BandPO 是与 RIPO 最值得对读的工作之一。它提出统一的 **Band operator**：

1. 先指定一个 $f$-divergence trust region；
2. 把该 trust region 投影成每个 token 的动态 ratio 区间；
3. 对 KL、TV、Pearson $\chi^2$ 等 divergence 可得数值解或闭式解。

BandPO 同样指出，固定 clipping 下

$$-\epsilon_-\,p\ \le\ \Delta p\ \le\ \epsilon_+\,p,$$

低概率动作的增长空间随 $p$ 线性缩小。

## RIPO：局部黎曼等距裁剪

RIPO 从局部二阶近似

$$D_{\mathrm{KL}}\approx\frac{1}{2}\,p\,(r-1)^2$$

出发，要求每个 token 的更新满足统一的分布距离预算：

$$\frac{1}{2}\,p\,(r-1)^2\le\delta
\quad\Longrightarrow\quad
|r-1|\le\sqrt{\frac{2\delta}{p}}.$$

因此 $p\downarrow\ \Rightarrow\ \epsilon_t\uparrow$，即 $\epsilon_t\propto 1/\sqrt{p}$。

与 DAPO 的启发式调高上界相比，RIPO 给出了明确的概率依赖尺度；与 DPPO 相比，它不需要计算完整分布的 divergence；与 BandPO 相比，它更集中在 Fisher–Riemannian 局部几何及其 bias–variance 解释。

---

## 第三条线：熵坍塌到底由哪些 token 驱动

## The Entropy Mechanism of RL for Reasoning LMs

这篇论文推导出，策略熵的变化近似由

$$\operatorname{Cov}\bigl(\text{action probability},\ \text{logit change}\bigr)$$

驱动（协方差为正时熵下降）；而在策略梯度算法中，logit change 又正比于 advantage。作者观察到该协方差长期为正，因此熵呈单调下降趋势，并提出：

- **Clip-Cov**：裁剪高协方差 token 的更新；
- **KL-Cov**：对高协方差 token 施加更强的 KL 约束。

这篇论文的价值在于，它不是笼统地说"熵在降"，而是定位：

$$\boxed{\text{哪些 token 的更新正在系统性地降低熵}}$$

## Reasoning with Exploration: An Entropy Perspective

这项工作发现，高熵位置通常更容易出现关键推理分叉、反思/回溯行为和罕见推理动作。它将截断且 stop-gradient 的 entropy 信号加到 advantage 中，鼓励更长、更有探索性的推理，同时尽量保留原始策略梯度方向。

## Clip-Low / Clip-High 的熵偏置

**Clip-Low Increases Entropy and Clip-High Decreases Entropy**

这篇论文把 clipping 与熵直接关联：

- lower clip 一侧倾向**增加**熵；
- upper clip 一侧倾向**降低**熵；
- 常见超参设置下上侧效应占优，因此即使 reward 没有任何结构，clipping 机制本身也会造成熵下降。

这意味着 PPO clipping 不只是稳定化工具，它自带

$$\boxed{\text{隐式的熵方向偏置}}$$

——上下 clipping 参数不能只从"更新幅度"角度理解。

## GPPO / CE-GPPO：不要让 clipped token 完全零梯度

PPO 一旦进入 clipping 区域，相应 token 的梯度直接变为零。GPPO 与 CE-GPPO 的思路不是无限放宽 clipping，而是**对越界 token 保留一个较小、受控、连续衰减的梯度**。CE-GPPO 特别指出，被裁剪的 token 对熵演化有重要作用，因此在 clipping 区间外以温和、有界的形式重新引入梯度。

这类方法解决的是

$$\text{hard clipping}\ \Rightarrow\ \text{information deletion},$$

而 RIPO 解决的是

$$\text{fixed clipping}\ \Rightarrow\ \text{probability-dependent geometric unfairness}.$$

## Lp-Reg：低概率 token 也分"火花"和"噪声"

**Low-probability Tokens Sustain Exploration in RLVR**

这篇论文提出了很重要的 **reasoning sparks** 概念：有价值的低概率 token 往往是 "wait" / "however" / "perhaps" / "alternatively" 这类可能开启反思、转折和新推理路径的词。

但不是所有低概率 token 都有价值——极低概率、语义无关的 token 可能只是噪声。单纯提高 entropy 会同时放大

$$\text{reasoning sparks}+\text{irrelevant noise}.$$

Lp-Reg 先通过概率阈值过滤疑似噪声，再对剩余分布重新归一化，并用 KL 正则保护剩下的低概率探索 token。

这是对 RIPO 的重要补充：**RIPO 解决"能不能更新"，Lp-Reg 讨论"哪些低概率 token 值得保护"。**

---

## 第四条线：低概率 token 反而可能更新过强

## Do Not Let Low-Probability Tokens Over-Dominate in RL for LLMs

这篇论文的结论表面上与 DAPO、RIPO 相反：**低概率 token 的梯度范数可能过大，从而支配参数更新并干扰高概率 token 的学习。** 它提出 Advantage Reweighting 和 Low-Probability Token Isolation（Lopti）来削弱低概率 token 的梯度影响。

## 为什么并不矛盾？

因为二者测量的是不同空间里的量。RIPO 讨论的是**概率空间中允许的移动量**：

$$\Delta p=p\,(r-1);$$

Lopti 讨论的是**共享参数空间中的梯度范数**：

$$\bigl\|A_t\,\nabla_\theta\log\pi_\theta(a_t\mid s_t)\bigr\|.$$

完全可能同时出现：

$$\boxed{\Delta p\ \text{很小，但参数梯度很大}}$$

原因在于神经网络参数被所有 token 共享：一个稀有 token 的梯度未必能把自身概率提高多少，却可能强烈扰动其他 token 的 logit 和隐藏表示。

此外，RIPO 主要想保护的是 $A_t>0$ 的罕见有价值动作；Lopti 更担心的是 outcome reward 被粗粒度地摊给所有 token 后，一些偶然出现的稀有 token 获得不合理的大梯度。合起来是一个更完整的原则：

$$\boxed{\text{低概率}+\text{正奖励}\ \neq\ \text{因果上有价值的推理 token}}$$

---

## 第五条线：重要性采样的对象到底应该是什么？

这条线与 RIPO 不完全同题，但应该放在一起看：前几条线讨论"边界该多宽"，这条线讨论"ratio 本身该定义在什么粒度上"。

## CISPO：裁剪权重，不裁掉梯度

MiniMax-M1 提出 CISPO。它不使用 PPO 式的 hard-clipped surrogate，而是裁剪重要性采样**权重**本身，同时保留相应梯度，旨在提高 RL 训练效率——与 GPPO/CE-GPPO 的"不要删信息"精神相通。

## GSPO：token ratio 与 sequence-level reward 粒度不符

GSPO 认为，GRPO 使用 token-level importance ratio，却给整条序列分配 outcome reward，粒度不一致。因此 GSPO 使用基于整条序列似然（几何平均/长度归一化）的 importance ratio，并在组内使用序列级 advantage，减少长回答中 token ratio 噪声累积造成的不稳定。

## CTPO：用前缀累计 ratio

**Rethinking Importance Sampling in LLM Policy Optimization: A Cumulative Token Perspective**

CTPO 的分析是：

- 单 token ratio 忽略了 prefix state distribution mismatch，有偏；
- 完整序列 ratio 无偏，但连乘方差巨大；
- GSPO 的长度归一化序列 ratio 更稳定，但不是精确的 IS 修正。

因此它提出前缀累计 ratio

$$\rho_t^{\mathrm{cum}}=\prod_{j=1}^{t}r_j,$$

用截止位置 $t$ 的累计 ratio 修正第 $t$ 个 token，并根据累计 log-ratio 方差随位置增长的规律，使用位置相关的 clipping。

它与 RIPO 解决的是两个正交问题：RIPO——边界如何随 token 概率变化；CTPO——ratio 应该覆盖单 token、前缀还是整条序列。

## ASPO：正 advantage 下 ratio 权重的方向不合理

ASPO 认为，在 outcome-supervised RL 中，importance ratio 已不只是分布校正，实际上成了 token 的梯度权重。对正 advantage token，标准 ratio 会给已经高概率的 token 更大权重、给低概率 token 更小权重——方向恰好反了。ASPO 因此翻转正 advantage token 的 ratio 作用方向，并加上 soft dual clipping，使低概率正向 token 获得更强更新，同时在极端区域保留梯度。

它与 RIPO 很接近但角度不同：ASPO 从 outcome supervision 下的 credit weighting 出发；RIPO 从策略分布几何出发。

---

## 核心论文对比表

| 论文 | 场景 | 低概率 token 的问题 | 方法方向 |
|---|---|---|---|
| DFT | SFT | 离线专家 token 的隐式逆概率权重过大 | 低概率 token 降权 |
| ASFT | SFT | DFT 在线权重造成分布漂移 | DFT + KL 锚定 |
| PriFT | SFT | 在线概率权重自增强、不稳定 | 用冻结预训练模型算权重 |
| ProFit | SFT | 低概率 token 可能只是表面措辞 | 选择性 mask |
| DAPO | RL | 固定上界让稀有好 token 增长太慢 | 全局提高 upper clip |
| DCPO | RL | 每个 token 应有不同 clipping | 概率相关动态边界 |
| DPPO | RL | ratio clip 不能代表真实分布距离 | 直接约束 divergence |
| BandPO | RL | 固定 band 不对应 $f$-divergence trust region | divergence 投影成动态 band |
| RIPO | RL | 欧氏 ratio 距离与 Fisher 几何失配 | $\epsilon_t\propto 1/\sqrt{p}$ |
| Entropy Mechanism | RL | 特定 token 更新驱动熵单调下降 | Clip-Cov / KL-Cov |
| GPPO / CE-GPPO | RL | hard clipping 删除梯度 | 有界恢复 clipped gradient |
| Lp-Reg | RL | 有价值低概率 token 与噪声混淆 | 选择性保护 reasoning sparks |
| Lopti | RL | 某些低概率 token 梯度支配训练 | 隔离或降低稀有 token 梯度 |
| ASPO | RL | 正 advantage ratio 错配学习权重 | 翻转正向 ratio 权重 |
| CTPO | RL | 单 token IS 忽略 prefix mismatch | 前缀累计 IS ratio |

---
后续可能方向：

1. **四个空间尚未统一。** 概率变化量、分布几何距离、参数梯度范数、隐式奖励权重，目前每篇工作只在一个空间里做修正。是否存在一个同时约束多个空间的统一目标（例如在 Fisher 几何下同时考虑 $\Delta p$ 与参数空间投影），还没有工作给出。
2. **"有价值的低概率"与"噪声的低概率"缺少因果判据。** Lp-Reg 用概率阈值、Lopti 用梯度范数、Entropy Mechanism 用协方差，本质上都是相关性代理。真正需要的是对单个 token 的因果贡献估计（类似 token-level credit assignment），这与过程奖励模型（PRM）的研究可能会合流。
3. **SFT 线与 RL 线的对偶尚未被显式建立。** DFT 说"SFT 隐含 $1/p$ 权重，要乘回 $p$"；RIPO 说"RL clipping 隐含固定欧氏尺度，要按 $1/\sqrt p$ 放宽"。两者都在做"概率相关的尺度校正"，但方向相反（SFT 压低低概率、RL 放开低概率）。这不矛盾——SFT 的低概率 token 是 off-policy 的外来标签，RL 的低概率 token 是 on-policy 采样出的自身探索——但这一区分值得被形式化：**更新尺度应依赖于"信号来源是否 on-policy"与"token 概率"两个变量的联合。**
