---
title: "On-Policy Distillation 的脉络梳理"
description: "用统一的状态分布框架串起 GKD、MiniLLM、OPSD、G-OPD 到 Agentic OPD：OPD 的演化主线不是换一种 KL，而是状态分布、教师构造与 credit 的连续修正。"
tags:
  - "distillation"
  - "post-training"
  - "rl"
category: "blog"
feature: true
date: "2026-08-03"
updated: "2026-08-03"
---


OPD 应该是现在 Post-Training 最热门的范式之一了。比如 Kimi K3 将多个 expert model merge 到最终模型时也用到了 OPD，因此我准备系统性梳理一下它的发展脉络，尽量讲清这个范式的前世今生。
## 先建立一个统一的数学框架

给定输入 $x\sim\mathcal D$，模型生成长度为 $T$ 的 trajectory：

$$
y=(y_1,\ldots,y_T).
$$

第 $t$ 个 token 之前的状态为：

$$
s_t=(x,y_{<t}).
$$

对于 Agent，可以进一步写成：

$$
s_t=(x,y_{<t},o_{\leq t},a^{\text{tool}}_{<t},h_t),
$$

其中 $o_t$ 是环境观测，$a^{\text{tool}}$ 是工具调用，$h_t$ 是交互历史或外部 memory。

学生策略记为 $p_\theta(a\mid s)$，教师策略记为 $q(a\mid s,c)$。这里的 $c$ 可以为空，也可以是标准答案、skill、memory等额外信息。
学生的完整序列分布为：
$$
p_\theta(y\mid x)=\prod_{t=1}^{T}p_\theta(y_t\mid s_t).
$$
### 状态分布

令策略 $\mu$ 在第 $t$ 步诱导出的状态分布为：

$$
d_t^\mu(s)
=
\Pr\left(
s_t=s
\mid
x\sim\mathcal D,\,
y_{<t}\sim\mu
\right).
$$

不同训练范式的核心区别之一，就是训练时的状态到底来自哪里：

$$
\begin{aligned}
\text{SFT/KD}:&\quad s_t\sim d_t^{\text{data}},\\
\text{Sequence KD}:&\quad s_t\sim d_t^q,\\
\text{OPD}:&\quad s_t\sim d_t^{p_\theta}.
\end{aligned}
$$

普通 KD(knowledge distillation) 在固定数据或教师轨迹上训练，但部署时学生访问的是自己的状态。一旦学生前面犯了错，后面看到的 prefix 就可能完全偏离训练分布。

OPD 的第一个意义，就是让学生在训练时暴露自己的错误状态，再让教师在这些状态上给 correction。

### OPD 的两种基本数学形态

现在的大多数 OPD 方法，可以分为两类。

第一类是在学生状态上直接做 distribution matching：

$$
\mathcal L_{\text{direct}}
=
\sum_{t=1}^{T}
\mathbb E_{s_t\sim d_t^{\mu_k}}
\left[
m_t
D_{\Omega_t}
\left(
p_\theta(\cdot\mid s_t),
q(\cdot\mid s_t,c)
\right)
\right].
$$

其中：
- $\mu_k$ 是生成 rollout 的行为策略，理想情况下就是当前学生 $p_{\theta_k}$；
- $m_t$ 是位置、置信度、熵或正确性相关的权重；
- $\Omega_t$ 可以是 full vocabulary、top-$k$ 或 sampled token；
- $D$ 可以是 forward KL、reverse KL、JSD、skew KL 或其他 $f$-divergence。

这类方法一般把 rollout 出来的 prefix 当成固定数据，也就是对 $s_t$ 做 stop-gradient。模型会在学生状态上训练，但不会追问“学生为什么会走到这个状态”。GKD、很多 full-vocabulary OPD、原始 OPSD 和 OPCD 都更接近这种形式。

第二类是把 OPD 写成 sequence-level policy gradient。以 sequence reverse KL 为例：

$$
\mathcal J_{\text{seq}}
=
D_{\mathrm{KL}}
\left(
p_\theta(y\mid x)
\parallel
q(y\mid x)
\right).
$$

因为采样轨迹本身依赖 $\theta$，这里不仅要更新当前状态下的 token distribution，还要考虑策略变化以后状态分布也会变化，也就是 $\nabla_\theta d_t^{p_\theta}$。MiniLLM 和后来的 return-to-go 类方法属于这条路线。

所以 on-policy 其实有两个层次：

1. distributionally on-policy：状态由学生策略生成；
2. optimization-wise on-policy：梯度还考虑了策略对未来状态分布的影响。

##  OPD 的前身：从 Behavior Cloning 到 ImitKD

OPD 并不是突然出现的，它最直接的理论祖先其实是 imitation learning。

### Behavior Cloning 的 distribution shift

传统 Behavior Cloning 在专家状态上训练：

$$
\min_\theta
\mathbb E_{s\sim d^q}
\left[
-\log p_\theta(a^q\mid s)
\right].
$$

但部署时状态来自 $d^{p_\theta}$。学生只要在某一步犯错，就可能进入专家数据没有覆盖过的状态，之后的错误还会继续累积。

在经典有限 horizon 的分析里，如果单步错误率为 $\epsilon$，普通 Behavior Cloning 的累积误差可能达到 $O(T^2\epsilon)$；而在学习者自己访问的状态上反复查询专家，可以把量级改善到 $O(T\epsilon)$，具体常数依赖损失和 no-regret 假设。

###  DAgger：OPD 的抽象原型

DAgger 的流程：

1. 用当前学生策略 rollout；
2. 在学生实际访问的状态上查询专家；
3. 收集 $(s,a^q)$；
4. 更新学生；
5. 重复。

把环境状态换成 token prefix，把专家动作换成教师的 next-token distribution，这几乎就是今天 OPD 的抽象原型。

DAgger 真正留下来的核心思想是：

> 监督必须覆盖学习者自己会访问到的状态。

[DAgger, 2011](https://proceedings.mlr.press/v15/ross11a.html)

### 2.3 Sequence KD 还不是真正的 OPD

Sequence-Level Knowledge Distillation 不再只匹配单个 token，而是先让教师生成完整序列 $y^q\sim q(\cdot\mid x)$，再训练学生：

$$
\mathcal L_{\text{SeqKD}}
=
-\mathbb E_{y^q\sim q}
\sum_t
\log p_\theta(y_t^q\mid x,y_{<t}^q).
$$

它缓解了 token-level teacher forcing 与 sequence generation 的不一致，但状态仍然来自教师轨迹 $d_t^q$，而不是学生部署时的 $d_t^{p_\theta}$。所以它更准确地说是 teacher-occupancy distillation，而不是严格意义上的 OPD。

[Sequence KD, 2016](https://arxiv.org/abs/1606.07947)

### 2.4 ImitKD 开始把学生状态带进训练

ImitKD 显式地把自回归 KD 看成 imitation learning，并逐渐增加 student-generated states 的比例：

$$
d_t^{\text{mix}}
=
(1-\lambda_k)d_t^{\text{data}}
+
\lambda_k d_t^{p_{\theta_k}}.
$$

随着 $\lambda_k$ 增大，训练分布从固定数据慢慢过渡到学生分布。它已经非常接近现代 OPD，只是当时主要讨论的还是 exposure bias 和 sequence modeling，还没有进入今天 reasoning post-training 的时代。

##  现代 OPD 的两条主线：GKD 和 MiniLLM

2023 年前后，现代 OPD 沿两条不同的数学路线成型：

- GKD：在 student prefix 上做直接分布匹配；
- MiniLLM：把 sequence reverse KL 写成 policy gradient。

它们经常被放在同一个框架里讨论，但优化对象并不完全一样。

### 1. GKD：先解决 student-state mismatch

GKD 先用学生生成轨迹：

$$
y\sim p_{\theta_k}(\cdot\mid x),
$$

然后在每个学生 prefix 上匹配教师分布：

$$
\mathcal L_{\text{GKD}}
=
\mathbb E_{x\sim\mathcal D}
\mathbb E_{y\sim p_{\theta_k}}
\left[
\frac{1}{T}
\sum_{t=1}^{T}
D\left(
q(\cdot\mid s_t),
p_\theta(\cdot\mid s_t)
\right)
\right].
$$

这里的 $\theta_k$ 是生成 rollout 时的策略。训练时通常不对采样过程反向传播，因此更像 DAgger style 的 semi-gradient。

GKD 还可以混合固定数据与 student rollout：

$$
\mathcal L
=
(1-\lambda)\mathcal L_{\text{fixed}}
+
\lambda\mathcal L_{\text{student-rollout}}.
$$

当 $\lambda=0$ 时接近传统 KD，当 $\lambda=1$ 时就是完全基于学生 rollout 的 OPD。

它解决的核心问题是：

$$
d_t^{\text{train}}
\neq
d_t^{\text{deploy}}
\quad\Longrightarrow\quad
d_t^{\text{train}}
\approx
d_t^{p_\theta}.
$$

学生会在训练时暴露自己的错误分支、措辞习惯、长度模式和格式问题，教师不再只演示理想答案，而是在学生真正会走到的地方做局部纠错。

但 GKD 没有自动解决 temporal credit。假如第 $t$ 步的错误导致第 $t+20$ 步失败，普通 GKD 通常只在第 $t$ 步优化 $D(q_t,p_t)$，不会把未来的 disagreement 回传给早期 token。

 它主要修正 state-distribution mismatch，而不是完整解决 long-horizon credit assignment。

[GKD](https://arxiv.org/html/2306.13649v3)

### 2. MiniLLM：把教师差异变成 sequence reward

MiniLLM 从 sequence-level reverse KL 出发：

$$
\mathcal J_{\text{seq}}(\theta)
=
D_{\mathrm{KL}}
\left(
p_\theta(y\mid x)
\parallel
q(y\mid x)
\right).
$$

展开以后：

$$
\mathcal J_{\text{seq}}
=
\mathbb E_{y\sim p_\theta}
\left[
\sum_{t=1}^{T}
\log
\frac{p_\theta(y_t\mid s_t)}
{q(y_t\mid s_t)}
\right].
$$

定义 token reward：

$$
r_t
=
\log
\frac{q(y_t\mid s_t)}
{p_\theta(y_t\mid s_t)},
$$

以及 return-to-go：

$$
G_t=\sum_{k=t}^{T}r_k.
$$

那么负梯度的核心形式就是：

$$
-\nabla_\theta\mathcal J_{\text{seq}}
\approx
\mathbb E
\left[
\sum_{t=1}^{T}
G_t
\nabla_\theta\log p_\theta(y_t\mid s_t)
\right],
$$

这已经是标准的 policy gradient 形式。

局部 OPD 只用即时 $r_t$ 更新第 $t$ 个 token，而 sequence-level OPD 使用 $G_t$。两者的差别正是未来 teacher-student discrepancy 对早期 token 的 credit。

因此：

- GKD 更像 student-state 上的局部纠错；
- MiniLLM 更像以 teacher log-ratio 为 dense reward 的 sequence RL。

这两条路线构成了后续 OPD 的两个数学源头。

[MiniLLM](https://arxiv.org/html/2306.08543v6)

## 为什么大家一直在换 divergence

在同一个状态 $s$ 上，记：

$$
p_v=p_\theta(v\mid s),\qquad q_v=q(v\mid s).
$$

###  Forward KL：尽量覆盖教师

Forward KL 为：

$$
D_{\mathrm{KL}}(q\parallel p)
=
\sum_v q_v\log\frac{q_v}{p_v}.
$$

它对学生 logit 的梯度是：

$$
\frac{\partial D_{\mathrm{KL}}(q\parallel p)}
{\partial z_j}
=
p_j-q_j.
$$

这意味着它会根据教师的完整分布重新分配 probability mass，具有 mode-covering 的特点。优点是能保留教师认为合理的多个 token；缺点是学生容量较小时，可能被迫在多个 mode 之间摊分概率。

###  Reverse KL：集中到教师的高概率区域

Reverse KL 为：

$$
D_{\mathrm{KL}}(p\parallel q)
=
\sum_v p_v\log\frac{p_v}{q_v}.
$$

它主要关注学生自己已经放置概率的区域，因此更 mode-seeking，也更适合让小学生集中到教师的高质量 mode。

但当 $p_v>0$、$q_v\approx0$ 时：

$$
\log\frac{p_v}{q_v}\rightarrow+\infty.
$$

这会带来 support mismatch、极端梯度、rollout 后段不稳定以及 diversity collapse。更麻烦的是，教师在异常 student prefix 上给出极低概率，并不一定代表这个 token 真错了，也可能只是教师自己已经离开了熟悉的分布。

###  JSD、skew KL 和 adaptive divergence

JSD 在 teacher coverage 与 student mode-seeking 之间折中。DistiLLM 则进一步使用 skew reverse KL：

$$
D_{\mathrm{KL}}
\left(
p
\parallel
(1-\alpha)q+\alpha p
\right).
$$

令 $m=(1-\alpha)q+\alpha p$，因为 $m_v\geq\alpha p_v$，所以：

$$
\log\frac{p_v}{m_v}
\leq
-\log\alpha.
$$

也就是说，skew reverse KL 给 log-ratio 加了天然上界，可以缓解普通 reverse KL 的 support mismatch。DistiLLM 同时引入 adaptive replay，开始正面处理 on-policy 数据昂贵的问题。

[DistiLLM](https://arxiv.org/html/2402.03898v1)

###  对一般 $f$-divergence，不能随手把 loss 当 reward

令：

$$
u(a)=\frac{q(a)}{p_\theta(a)}.
$$

一般 $f$-divergence 可以写成：

$$
D_f(q\parallel p_\theta)
=
\mathbb E_{a\sim p_\theta}f(u(a)).
$$

因为 $u=q/p_\theta$ 本身也依赖 $\theta$，完整梯度中的 policy-gradient 权重应该是：

$$
w_f(u)=-f(u)+u f'(u),
$$

而不是简单把 $-f(u)$ stop-gradient 后当 reward。

Reverse KL 是一个比较特殊的例子，它对应的权重只比 $\log(q/p)$ 多一个不影响期望梯度的常数 baseline，所以朴素 log-ratio reward 恰好成立。对其他 divergence，这种写法一般会产生偏差。OPD+ 一类工作主要是在澄清这个问题。

[OPD+](https://arxiv.org/html/2606.01039v1)

Sampled-token OPD 还有一个实际问题是方差。对 reverse KL，令：

$$
r(a,s)=\log\frac{q(a\mid s)}{p_\theta(a\mid s)},
$$

它在当前学生分布下的状态期望为：

$$
V(s)
=
\mathbb E_{a\sim p_\theta}r(a,s)
=
-D_{\mathrm{KL}}(p_\theta\parallel q).
$$

因为 $V(s)$ 与本次采样动作无关，所以可以用 $A(a,s)=r(a,s)-V(s)$ 作为 control variate，在不改变期望梯度的情况下减少方差。这就是 vOPD 的核心思路。

[vOPD](https://arxiv.org/abs/2605.07865)

###  区分两种完全不同的 token credit

讨论 token-level credit 时，经常会混在一起的其实是两个问题。

第一个是 temporal credit：未来结果要不要影响早期 token。

$$
A_t=r_t
\quad\text{vs.}\quad
A_t=G_t=\sum_{k=t}^{T}r_k.
$$

第二个是 vocabulary routing：释放出来的概率到底应该流向哪个 token。

Full-vocabulary forward KL 的梯度是 $p_v-q_v$，它明确知道要把概率推向哪些 teacher-supported tokens；sampled-token policy gradient 只知道应该提高或压低当前采到的 token，剩余概率会按照学生自己的分布重新分配。

所以：

> temporal credit 和 vocabulary credit 是两个正交问题，解决一个并不代表另一个也解决了。

##  学生 on-policy，教师却可能 off-policy

学生 prefix：

$$
s_t\sim d_t^{p_\theta}
$$

对学生而言当然是 on-policy。但教师通常只在自己的正常轨迹上训练过，因此同一个状态对教师可能是：

$$
s_t\not\sim d_t^q.
$$

也就是说，一个状态可以同时满足 student-on-policy 和 teacher-off-policy。

这也是 DAgger 假设在 LLM 场景中的薄弱点：专家未必能够可靠标注学习者访问的任意状态。随着 rollout 变长，学生错误不断累积，teacher 面对的 prefix 越来越异常，它的 correction 质量可能持续下降。

可以把 teacher decay 粗略写成：

$$
\Delta_{\text{decay}}(t)
=
A_T(x)
-
A_T(x,y_{<t}^S),
$$

其中 $A_T(x)$ 是教师从正常输入开始的成功率，$A_T(x,y_{<t}^S)$ 是教师从学生 prefix 接着生成的成功率。通常 $t$ 越大，decay 越明显。

因此，teacher logits 很 dense，不等于 teacher signal 很 reliable。

2026 年围绕 teacher reliability 的方法，本质上都在把“是否蒸馏”从固定规则改成 state-dependent decision：

- EOPD 根据教师熵在 forward KL 与 reverse KL 之间切换：低熵时更相信 mode-seeking，高熵时保留更多 coverage；
- TrOPD 用 trust region、mask、clip 或安全 target 限制异常状态上的更新；
- Early-Stopping Rollout 在教师明显失效以后直接截断监督；
- 其他方法使用 teacher veto、first-error mask、target reformulation 或 disagreement threshold。

研究问题也因此从“forward KL 还是 reverse KL”变成了：

> 在哪个状态、哪个 token、以多大强度、通过什么 target 相信教师？

[Rethinking OPD](https://arxiv.org/abs/2604.13016)  
[EOPD](https://arxiv.org/abs/2603.07079)  
[TrOPD](https://arxiv.org/abs/2606.01249)  
[Early-Stopping Rollout](https://arxiv.org/abs/2605.27028)

## OPD 与 RL 为什么会逐渐合流

从 reverse KL 的推导可以看到，teacher-student log-ratio 天然可以写成 dense reward。再引入 reference policy $\pi_0$，可以把教师相对 reference 的方向写成：

$$
r_T(y,x)
=
\log\frac{q(y\mid x)}{\pi_0(y\mid x)}.
$$

考虑 KL-regularized objective：

$$
\mathcal J_\lambda(p)
=
\mathbb E_{y\sim p}
\left[
\lambda
\log\frac{q(y\mid x)}{\pi_0(y\mid x)}
\right]
-
D_{\mathrm{KL}}(p\parallel\pi_0).
$$

它的最优策略为：

$$
p_\lambda^*(y\mid x)
\propto
q(y\mid x)^\lambda
\pi_0(y\mid x)^{1-\lambda}.
$$

这个式子很有解释力：

- $\lambda=0$：不学习教师，保持 reference；
- $0<\lambda<1$：在 reference 和 teacher 之间做几何插值；
- $\lambda=1$：最优点就是教师；
- $\lambda>1$：沿教师相对 reference 的方向做 extrapolation。

这也解释了为什么纯 OPD 一般不能保证超过教师。标准目标 $\min_p D_{\mathrm{KL}}(p\parallel q)$ 的最优解就是 $p^*=q$；学生容量受限时，得到的只是教师在学生策略空间中的投影。

要真正 beyond teacher，通常还需要额外信息，比如 verifier、outcome reward、环境反馈、多样本搜索、successful sibling rollout、multi-teacher delta 或 $\lambda>1$ 的能力外推。

实践中更合理的统一目标是把 outcome RL 和 OPD 放在一起：

$$
\mathcal J_{\text{hybrid}}
=
\mathbb E_{y\sim p_\theta}
\left[
R_{\text{outcome}}(x,y)
+
\eta\sum_t r_t^{\text{teacher}}
\right]
-
\beta D_{\mathrm{KL}}(p_\theta\parallel\pi_0).
$$

Outcome reward 回答整条轨迹最终是否成功，OPD reward 则提供当前 prefix 上的局部 policy shaping。一个给方向，一个给密集反馈，二者是互补的。

[G-OPD / reward extrapolation](https://arxiv.org/html/2602.12125v1)

不过这里还有一个理论上的限制：teacher log-ratio 并不自动等价于任务 advantage。只有当教师近似是某个 entropy-regularized MDP 的最优策略，并满足：

$$
q^*(a\mid s)
\propto
\pi_0(a\mid s)
\exp\left(Q^*(s,a)/\beta\right),
$$

才有：

$$
\beta\log\frac{q^*(a\mid s)}{\pi_0(a\mid s)}
=
Q^*(s,a)-V^*(s).
$$

一般 LLM teacher 并不满足这个条件。因此 teacher log-ratio 更准确地说是 policy-shaping signal，而不是对最终任务回报的无偏因果 credit。

##  从外部大教师到 self teacher 和 context teacher

OPD 早期默认教师是一个更强的大模型，但 2026 年以后，“教师是谁”开始迅速扩展。

###  OPSD：同一个模型也可以自己教自己

OPSD 中，学生和教师可以来自同一个模型，只是教师额外看到 privileged context：

$$
p_S(a\mid x,y_{<t}),
\qquad
p_T(a\mid x,c,y_{<t}).
$$

学生先 rollout，教师再带着标准答案、verified solution、环境反馈或正确轨迹，在学生 prefix 上提供分布。

它的 token reward 可以写成：

$$
r_t^c
=
\log
\frac{
p(y_t\mid x,c,y_{<t})
}{
p(y_t\mid x,y_{<t})
}.
$$

利用 Bayes 公式：

$$
r_t^c
=
\log
\frac{
p(c\mid x,y_{\leq t})
}{
p(c\mid x,y_{<t})
}.
$$

所以这个 reward 衡量的是：看到当前 token 以后，对 privileged context $c$ 的认知增加了多少。对整条轨迹求和以后，它对应 pointwise conditional mutual information。

这个解释也直接暴露了 OPSD 的风险：它奖励的是“哪些 token 最能反映 privileged context”，但不一定是“哪些 token 对最终正确性有最大的因果贡献”。

当教师已经知道答案时，它可能不再探索、不再验证、不再 backtrack，而是快速收敛到已知答案。学生测试时却没有这些 privileged information。于是模型可能把“知道答案以后怎么表达”，错误地学成“不知道答案时应该怎么搜索”。

因此后续修正主要围绕：

- entropy 或 uncertainty gating；
- 只蒸馏 first error、reasoning pivot、critical span；
- 用 contrastive context 去掉与问题无关的 style component；
- 只有 correction 与 verifier outcome 一致时才使用；
- 在 backtracking、verification 和高熵分叉处保护 exploration。

[OPSD](https://arxiv.org/pdf/2601.18734)  
[OPSD 的信息论与负面结果讨论](https://arxiv.org/html/2605.11613)  
[Privileged teacher 对 reasoning 的影响](https://arxiv.org/abs/2603.24472)

###  β-OPSD：重新把未来 credit 接回来

很多 sampled-token OPSD 只用即时 reward：

$$
r_t=\log\frac{q_t(y_t)}{p_t(y_t)}.
$$

β-OPSD 认为这种更新过于 myopic，因此重新使用：

$$
G_t^\gamma
=
\sum_{k=t}^{T}
\gamma^{k-t}r_k.
$$

当 $\gamma=0$ 时退化为即时 token correction；当 $\gamma=1$ 时恢复完整 sequence-level return-to-go。这条路线实际上重新连接了 OPSD、MiniLLM 和 policy gradient。

[β-OPSD](https://arxiv.org/abs/2607.28582)

###  OPCD：把 context 里的能力内化进参数

OPCD 不要求教师一定更大。教师的优势也可以来自额外 context：

$$
q(a\mid x,c,y_{<t}),
$$

而学生只能看到：

$$
p_\theta(a\mid x,y_{<t}).
$$

训练完成以后，学生推理时不再需要 $c$，因为原本存放在 prompt、memory、skill 或经验中的能力被内化到了参数中。

Skill-SD 又把这件事做成一个循环：

$$
\text{rollout}
\rightarrow
\text{总结成功经验}
\rightarrow
\text{构造 skill context}
\rightarrow
\text{OPD 内化}
\rightarrow
\text{更强 rollout}.
$$

从这个角度看，OPSD 和 OPCD 数学形式很接近，区别主要在语义重点：OPSD 更强调 self-improvement，OPCD 更强调 context internalization。

[OPCD](https://arxiv.org/html/2602.12275v1)  
[Skill-SD](https://arxiv.org/abs/2604.10674)

##  Long-horizon Agent 把问题进一步放大

Agent trajectory 不只有 token，还包含 reasoning step、tool call、environment turn、observation 和 subgoal。直接在完整 horizon 上做 OPD 会遇到几个问题：

- teacher decay 更严重；
- 前期错误会污染很长的后缀；
- trajectory importance ratio 方差极高；
- 工具调用和环境执行成本很高；
- token-level correction 很难对应真实的任务结构。

一种思路是 horizon curriculum：

$$
H_1<H_2<\cdots<H_K=T.
$$

先训练短程轨迹，再逐步增加 horizon，避免一开始就让 student 和 teacher 在极长的异常轨迹上交互。

另一种思路是 step-wise OPD。把 trajectory 按 reasoning step、tool call 或 subgoal 切成：

$$
I_1,I_2,\ldots,I_M,
$$

然后给每个 segment 分配权重：

$$
\mathcal L_{\text{step}}
=
\sum_{j=1}^{M}
w_j
\frac{1}{|I_j|}
\sum_{t\in I_j}
D(p_t,q_t).
$$

SOD 一类方法会进一步把 step-level OPD 与 GRPO 风格的 outcome optimization 结合，让 verifier 决定哪些 step 值得强化。

[TCOD](https://arxiv.org/html/2604.24005v1)  
[SOD](https://arxiv.org/abs/2605.07725)

不过 step-wise 并不等于 causal。普通 OPD 使用：

$$
r_t^{\text{OPD}}
=
\log\frac{q(a_t\mid s_t)}{p(a_t\mid s_t)},
$$

真正的反事实 credit 则应该回答：

$$
\mathbb E[R\mid s_t,\operatorname{do}(a_t)]
-
\mathbb E_{a\sim p}
\mathbb E[R\mid s_t,\operatorname{do}(a)].
$$

前者告诉学生“教师在当前状态更喜欢什么”，后者回答“如果早期换一个动作，最终结果会改变多少”。要估计后者，通常还需要 sibling branches、Monte Carlo continuation、world model、verifier 或真实环境 rollout。

所以 OPD 的准确定位仍然是 dense policy shaping，而不是已经完整解决了 long-horizon causal credit assignment。

##  最新趋势：不再复制绝对教师，只迁移 capability delta

传统 OPD 直接学习教师的完整分布，但这个分布里同时混着：

- 基础语言能力；
- reasoning 增量；
- 表达风格；
- 格式和长度偏好；
- 真正与任务有关的能力。

因此 2026 年后半段一个非常明显的趋势是：不再复制教师的绝对行为，而只迁移教师相对某个 anchor 的能力变化。

可以统一写成：

$$
q_\Delta(a\mid s)
\propto
p_{\text{anchor}}(a\mid s)
\exp\left(\alpha\Delta(a,s)\right),
$$

其中 $\Delta$ 表示真正想迁移的 capability 或 context direction。

###  OPD²：学习教师 post-training 前后的变化

设 $q_{\text{base}}$ 是教师 post-training 前的分布，$q_{\text{post}}$ 是 post-training 后的分布，则：

$$
\Delta_T(a,s)
=
\log q_{\text{post}}(a\mid s)
-
\log q_{\text{base}}(a\mid s).
$$

普通 OPD 学习 $q_{\text{post}}$，OPD² 更关心 $\Delta_T$，也就是“教师经过 reasoning post-training 以后究竟改变了什么”，从而减少对教师原有语言和风格特征的复制。

[OPD²](https://arxiv.org/abs/2607.15161)

###  Weak-to-Strong OPD：弱模型只提供方向

假设有两个弱模型 $q_+$ 和 $q_-$，它们在目标能力上有好坏差异。可以用：

$$
\Delta_{\text{weak}}=z_+-z_-
$$

表示能力方向，再加到强学生的基础 logits 上：

$$
z_{\text{proxy}}
=
z_{\text{strong-base}}
+
\alpha(z_+-z_-).
$$

弱模型不需要直接成为强学生的行为上界，只要它们的差分能够提供一个有意义的方向。

[Weak-to-Strong OPD](https://arxiv.org/abs/2607.26246)

###  Flux-OPD：context 在变化时，学习稳定 residual

如果不同 context 对应不同教师 $q_c(a\mid s)$，直接平均这些教师可能把互相冲突的 context 混在一起。

Flux-OPD 引入 context-free anchor $q_0$，定义：

$$
\Delta_c(a,s)
=
\log q_c(a\mid s)
-
\log q_0(a\mid s),
$$

只学习 context 相对稳定 anchor 带来的 correction，并根据不同 context 之间的冲突程度调节更新强度。

[Flux-OPD](https://arxiv.org/abs/2607.28022)

###  Lightning OPD 2.0：把 style bias 从能力信号中减掉

Cross-teacher 的 logit difference 可以粗略分成：

$$
d_{it}
=
b_{it}^{\text{style}}
+
v_{it}^{\text{capability}}
+
\epsilon_{it}.
$$

Lightning OPD 2.0 试图估计稳定的 style component，再只蒸馏 residual。这样学生就不会把教师固定的模板、标点、reasoning cadence 和长度偏好都当成能力。

[Lightning OPD 2.0](https://arxiv.org/abs/2607.28449)

这些方法虽然名字不同，但共同结构都是：

$$
\text{anchor policy}
+
\text{capability/context residual}
-
\text{style/conflict/noise}.
$$

这说明 OPD 正在从 absolute imitation 走向 directional policy improvement。

##  系统效率：on-policy 不是非黑即白

严格 on-policy 的代价很高。每次学生更新以后，旧 rollout 都会逐渐 stale；如果每个 gradient step 都重新生成完整 trajectory，再运行 teacher forward，成本通常无法接受。

假设 rollout 由旧策略 $\mu$ 生成，当前策略为 $p_\theta$。单步 importance ratio 为：

$$
\rho_t
=
\frac{p_\theta(y_t\mid s_t)}
{\mu(y_t\mid s_t)}.
$$

但它只能修正给定状态下的 action distribution。要修正第 $t$ 步的 state occupancy，需要前缀乘积：

$$
\rho_{1:t-1}
=
\prod_{k=1}^{t-1}
\frac{p_\theta(y_k\mid s_k)}
{\mu(y_k\mid s_k)},
$$

在长序列中方差会非常高。

所以实际系统通常位于一个连续谱上：

$$
\text{current-policy}
\rightarrow
\text{recent-policy}
\rightarrow
\text{replay}
\rightarrow
\text{cached rollout}
\rightarrow
\text{offline approximation}.
$$

DistiLLM 使用 adaptive replay；Lightning OPD 类方法缓存 teacher signal；Agent 场景还会使用 prefix replay、environment state caching、异步 actor-teacher-learner、selective teacher querying 和 early termination。

因此实践中的 on-policy，通常指 $d^{\mu_k}\approx d^{p_{\theta_k}}$，并不一定要求每次更新前都重新生成全部轨迹。这里始终存在一个无法消失的 bias-cost trade-off：数据越新越贵，数据复用越多偏差越大。

##  用一个统一视角判断 OPD 论文的新意

现在的OPD论文基本都是在以下维度来进行一些修改，可以通过这个来帮助快速浏览一篇论文

|维度|需要问的问题|
|---|---|
|Rollout policy $\mu_k$|状态由 teacher、current student、stale student 还是 mixed policy 生成？|
|Teacher $q_t$|教师是大模型、self teacher、context teacher、delta teacher 还是 multi-teacher？|
|Divergence $D$|使用 forward KL、reverse KL、JSD、skew KL 还是一般 $f$-divergence？|
|Vocabulary support $\Omega_t$|使用 full vocabulary、top-$k$ 还是 sampled token？|
|Credit $A_t$|即时 log-ratio、return-to-go、segment reward 还是 verifier advantage？|
|Trust weight $m_t$|什么时候相信教师，什么时候 mask、clip 或 early stop？|
|Policy staleness $\rho$|rollout 有多旧，是否 replay，是否做 importance correction？|
|Extra objective $\mathcal R$|是否加入 outcome reward、reference KL、entropy 或 style residual？|

Full-vocabulary OPD 可以统一写成：

$$
\mathcal L_{\text{direct}}
=
\mathbb E_{s_t\sim d_t^{\mu_k}}
\sum_t
m_t
D_{\Omega_t}
\left(
p_\theta(\cdot\mid s_t),
q_t(\cdot\mid s_t)
\right).
$$

Sampled-token OPD 则可以粗略写成：

$$
\widehat g
=
\mathbb E_{x\sim\mathcal D,\,y\sim\mu_k}
\left[
\sum_t
\rho_{1:t}
m_t
A_t
\nabla_\theta
\log p_\theta(y_t\mid s_t)
\right]
+
\nabla_\theta\mathcal R.
$$

几乎所有新方法，都是在重新设计 $\mu_k$、$q_t$、$D$、$\Omega_t$、$A_t$、$m_t$、$\rho$ 或 $\mathcal R$ 中的某几项。

##  timeline

|时间|代表工作|真正改变了什么|
|---|---|---|
|2011|DAgger|开始在学习者访问的状态上查询专家|
|2016|Sequence KD|从 token label 转向教师完整序列|
|2020|ImitKD|显式引入 student-generated states|
|2023—2024|GKD|student rollout + full-vocabulary teacher distribution|
|2023—2024|MiniLLM|sequence reverse KL + policy gradient|
|2024|DistiLLM|skew KL + adaptive replay|
|2025|DistiLLM-2 等|根据 teacher data 和 student data 选择不同几何|
|2026 年初|OPSD、OPCD、G-OPD|self teacher、context internalization、OPD-RL 统一|
|2026 年春|EOPD、TrOPD、vOPD、OPD+|teacher reliability、方差与梯度正确性|
|2026 年春夏|TCOD、SOD、Skill-SD|long-horizon、step segmentation、经验内化|
|2026 年 7 月|OPD²、W2S-OPD|从绝对教师转向 capability delta|
|2026 年 7 月末|Flux-OPD、Lightning OPD 2.0、β-OPSD|evolving context、style residual、未来 credit|

如果再抽象一层，这条脉络其实经历了七次变化：

1. 在教师轨迹上学习 $\rightarrow$ 在学生轨迹上学习；
2. 局部 token matching $\rightarrow$ sequence-level policy gradient；
3. 固定 reverse KL $\rightarrow$ skew、entropy、trust 和 replay；
4. distribution matching $\rightarrow$ teacher log-ratio as dense reward；
5. 外部大教师 $\rightarrow$ self、context 和 skill teacher；
6. token correction $\rightarrow$ step、segment、turn 和 horizon credit；
7. 复制绝对教师 $\rightarrow$ 迁移 capability/context delta。

##  OPD 目前还没有解决的问题

最后列一下我认为这个方向最核心的几个未解问题。

### 1. 教师是否真的理解学生的异常状态

OPD 修正了学生的 state distribution shift，但把问题转移成了 teacher off-policy。越到 rollout 后段，教师 correction 越可能失真。

### 2. Dense signal 不等于 causal credit

每个 token 都有 $\log q-\log p$，只代表反馈很密集，不代表它一定正确、无偏，或者与最终任务价值一致。

### 3. 局部最合理的 continuation 不等于全局可恢复

教师在错误 prefix 上给出当前最合理的 next token，并不能保证轨迹还可恢复，也不能证明当前 token 就是真正的问题来源。

### 4. Privileged teacher 与学生不在同一个 information state

知道答案后的教师和不知道答案时的学生，不只是输入分布不同，而是 epistemic state 不同。对 reasoning 来说，这种 information-state mismatch 比普通 covariate shift 更麻烦。

### 5. Capability 和 style 仍然纠缠

教师分布的差异里同时包含能力、风格、长度、格式和确定性。Delta/residual OPD 已经在处理这个问题，但目前还很难彻底分离。

### 6. Long-horizon 的切分粒度没有统一答案

Credit 可以按 token、sentence、reasoning span、first error、tool step、environment turn 或 subgoal 切分，但还没有统一理论说明哪个边界最接近真实因果结构。

### 7. 超越教师仍然依赖额外信息

标准 OPD 的固定点就是教师。真正 beyond teacher 往往来自教师方向以外的搜索、环境反馈、verifier、多样本选择或 capability extrapolation。

### 8. On-policy 成本与 off-policy 偏差无法同时消失

数据越新，rollout 和 teacher inference 越贵；数据复用越多，state occupancy 越 stale。这个系统层面的 bias-cost trade-off 不是换一个 loss 就能解决的。

##  总结

最早我们可以把 OPD 理解成“学生用自己生成的数据模仿教师”，但这个定义现在已经不够了。

更准确地说：

> OPD 是在 student-induced states 上，利用教师、特权上下文、能力差分或环境反馈构造局部策略方向，并据此持续进行 policy improvement 的 Post-Training 范式。

所以 OPD 的进化并不是 forward KL 换成 reverse KL，再换成另一个 KL。它真正的主线是：

$$
\text{状态分布修正}
\rightarrow
\text{序列级优化}
\rightarrow
\text{teacher reliability}
\rightarrow
\text{OPD-RL 统一}
\rightarrow
\text{self/context teacher}
\rightarrow
\text{long-horizon credit}
\rightarrow
\text{capability delta}.
$$

从这个框架出发，GKD、MiniLLM、OPSD、OPCD、G-OPD、Agentic OPD 和最新的 delta/residual OPD 就不再是互不相关的方法，而是同一个范式在状态分布、教师构造、分布几何、时间 credit 和能力方向几个维度上的连续演化。
