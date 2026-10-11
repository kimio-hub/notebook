---
title: "Frontier RL 算法学习：FlashREINFORCE、RPG、BPO、GPM"
description: "从 REINFORCE 出发，梳理 FlashREINFORCE、RPG、BPO 与 GPM 各自在解决什么问题，并区分推导中严格成立与开始近似的部分。"
tags:
  - "rl"
  - "policy-gradient"
  - "async-rl"
  - "preference"
category: "blog"
feature: true
date: "2026-09-23"
updated: "2026-09-23"
---

最近看见很多关于 OpenAI 等 frontier AI lab 的 RL 算法的讨论，正好开个 RL 的专题总结，汇总一些之前没怎么学习过的算法。

这次读了四篇：[FlashREINFORCE](https://www.researchgate.net/publication/414274571_FLASHREINFORCE_CRITIC-FREE_SINGLE-ROLLOUT_ASYNCHRONOUS_RL_FOR_AGENTIC_LANGUAGE_MODELS)、[RPG](https://arxiv.org/pdf/2505.17508)、[BPO](https://arxiv.org/pdf/2609.15987)、[GPM / GPO](https://arxiv.org/pdf/2410.02197)。网上有说法认为这几篇合起来就是某家 lab 的内部配方，我查了一圈，没有找到能直接支持的公开证据。比较接近的是 [DeepSeek-V3.2 技术报告](https://arxiv.org/pdf/2512.02556)，里面写了用重要性权重修正 KL 估计、屏蔽偏离当前策略过大的负优势序列，分别和 RPG、FlashREINFORCE 讨论的问题相近；另外 Yifan Zhang 在 [How to Train Frontier Models Effectively](https://yifanzhang-pro.github.io/how-to-train-frontier-models/) 里把 RPG、GPM 放进了他推荐的训练流程。

所以本文不去猜谁在用什么，只记录每篇在解决什么问题，以及推导里哪些地方是严格的、哪些地方开始是近似。

## 从 REINFORCE 开始

首先是最基本的 REINFORCE。题目为 $x$，模型逐 token 生成回答 $y=(y_1,\dots,y_T)$，生成第 $t$ 个 token 时看到的前缀是 $h_t=(x,y_{<t})$。最后验证器给整条回答一个奖励 $R(x,y)$，比如答对为 1、答错为 0，下文都默认这种终局奖励设定。更新方向公式：

$$
\nabla_\theta J = \mathbb{E}_{y \sim \pi_\theta} \left[ R(x, y) \sum_t \nabla_\theta \log \pi_\theta(y_t \mid h_t) \right].
$$

简单理解就是获得好结果的生成过程更加容易出现。同时为了让反馈更加精准，通常会减去基线 $A=R-b$，高于基线的回答得到正反馈，低于基线的回答得到负反馈。只要基线不依赖当前动作，它只影响方差，不改变梯度的期望。

我们很容易提出更进一步的问题：奖励是否可靠？基线怎么估计？rollout 和更新的模型是否是同一个模型？

最后一个问题要先把符号分清。记 $\pi_\theta$ 表示当前正在训练的策略，$\mu$ 表示实际生成这批 rollout 的行为策略，$\pi_{ref}$ 表示 KL 正则化所参照的策略。

一条 agent 轨迹开始的时候用的是旧的模型策略 $\mu$，等它调用工具、完成任务之后，训练器已经把策略更新成了 $\pi_\theta$，而 KL 正则化可能参照另一个固定或者定期更新的 $\pi_{ref}$。异步训练、策略版本滞后、训推引擎的数值差异，都会让 $\mu$ 与 $\pi_\theta$ 不一致。因此后面反复出现的两个比值要分开看：$\pi_\theta/\mu$ 用来修正采样分布，$\pi_\theta/\pi_{ref}$ 用来衡量离参考策略有多远。

带着这几个问题，四篇大致可以这样对应：

|论文|这一篇要回答的问题|
|---|---|
|**FlashREINFORCE**|不训练 critic、每题只采样一次，异步 RL 怎样稳定运行？|
|**RPG**|代码里 KL loss 算出来的梯度，是不是目标函数想要的梯度？|
|**BPO**|能否借助 Bellman 结构，绕过中间状态的价值估计？|
|**GPM / GPO**|"好坏"一定能用一个固定的标量 reward 表达吗？|

## FlashREINFORCE：每题只要一条 rollout

这篇工作相比 GRPO 提出了更加节省 rollout 的一个 trick：每条 prompt 只生成一条 rollout，再把不同 prompt 已经完成的轨迹组成训练 batch。

算一笔账：预算是 1024 条轨迹，GRPO 每题采 8 条只能看 128 道题，每题 1 条可以看 1024 道。牺牲的是同题内部的比较，换来的是题目覆盖和异步吞吐。对工具型 agent 来说后者更重要，轨迹长短差别很大，跑得快的不必等同一道题的其他回答。

### 基线：batch 均值

没有同题分组，也没有 critic，它直接使用 batch 级奖励均值：

$$
\bar{R} = \frac{1}{B} \sum_{j=1}^B R_j, \qquad A_i = R_i - \bar{R}.
$$

比如四条轨迹奖励是 $[1,0,0,1]$，均值 0.5，优势就是 $[+0.5,-0.5,-0.5,+0.5]$。

这组奖励来自不同 prompt，而不是同一道题，公式里也没有项在校正题目难度，一道难题答错和一道简单题答错拿到的惩罚一样。所以这可能需要同一个 batch 的题目难度差异不要太大，对于任务的分类有一定要求。相当于用一个很便宜的公共基线，替掉了同题多采样或者 value model。

### 过期数据怎么处理

异步训练中，生成轨迹的策略已经落后于当前策略。它的做法是生成时保存每个 token 的真实采样概率，训练时计算 token 级的重要性比值：

$$
\rho_{i,t} = \frac{\pi_\theta(y_{i,t}\mid h_{i,t})}{\mu_i(y_{i,t}\mid h_{i,t})}.
$$

这个比值修正的是"在这个前缀下选哪个 token"。但前缀本身也是旧策略走出来的：如果旧策略在前面进入了一条错误的推理路线，后面几千个 token 都建立在上面，逐 token 修正并不能把它变成当前策略会走的路线。

所以它又加了一层序列级过滤（Sequence Trust Region）：对每条轨迹算平均偏移，超过阈值 $\delta$ 的整条不参与更新。

$$
\bar D_i = \frac1{T_i} \sum_t D_{\mathrm{Bern}}(\mu_{i,t}\Vert\pi_{i,t}), \qquad m_i=\mathbf 1[\bar D_i\leq\delta].
$$

这里的 $D_{\mathrm{Bern}}$ 把词表压成"采到的 token"和"其他 token"两类再算 KL，它是完整 KL 的下界。因此这个值大时可以确定轨迹偏得厉害，值小时却不能保证轨迹真的接近当前策略，是一个偏宽松的筛子。

### 长度归一化

最后是 sample-mean。一个 batch 里成功的回答 100 tokens、失败的 1000 tokens，如果所有 token 混在一起平均，失败回答在 loss 里就占了 10 倍的权重。FlashREINFORCE 先在每条轨迹内部平均，再对轨迹平均，得到核心更新：

$$
\widehat g = \frac1B\sum_i \frac{m_iA_i}{T_i} \sum_t \rho_{i,t}\nabla_\theta\log\pi_\theta(y_{i,t}\mid h_{i,t}).
$$

四个因子分别负责：$A_i$ 给出好坏，$m_i$ 判断轨迹是否可信，$\rho_{i,t}$ 做局部采样修正，$1/T_i$ 做长度重加权。长度重加权改变了样本权重，严格来说它已经不是原始期望回报的无偏估计；每批数据只用一次就丢，也只是减少了过期，并不是严格的 on-policy。

读下来我觉得这篇的重点在训练组织：等不到数据、数据过期、长失败主导更新，这几个问题都能让训练直接失效，它逐个做了处理，而没有去碰更难的长程 credit assignment。

## RPG：KL loss 的梯度对不对

RPG 讨论的是在更新参数的时候加入 KL：

$$
J(\theta) = \mathbb{E}_{y \sim \pi_\theta} [R(y)] - \beta D_{\text{KL}}(\pi_\theta \| \pi_{\text{ref}}).
$$

第一项鼓励高奖励，第二项限制相对于参考策略的变化。论文把正向/反向 KL、归一化/非归一化、完全可微/stop-gradient 几种写法对应的梯度整理了一遍（[项目页](https://complex-reasoning.github.io/RPG/)）。我读下来最大的收获是：**目标函数、采样分布、梯度估计、autograd 代码，这四件事要逐一对上。**

### 数值估计对，梯度不一定对

先看一个求导恒等式：

$$
\nabla_\theta\mathbb E_{y\sim\pi_\theta}[f_\theta(y)] = \mathbb E_{\pi_\theta} \left[ f_\theta(y)\nabla_\theta\log\pi_\theta(y) + \nabla_\theta f_\theta(y) \right].
$$

梯度由两部分组成：一部分来自"$y$ 出现的概率变了"，一部分来自"固定 $y$ 之后 $f_\theta(y)$ 的值变了"。训练时手里是已经采好的样本，直接对样本上的 $f_\theta(y)$ 反传，只能得到第二部分。所以一个表达式的期望等于 KL，不代表对它反传就能得到 KL 的梯度。

### reverse KL 的正确更新

假设策略归一化、奖励和参考策略固定，可以推出：

$$
\nabla_\theta J = \mathbb E_{\pi_\theta} \left[ \left( R-b-\beta\log\frac{\pi_\theta}{\pi_{\mathrm{ref}}} \right) \nabla_\theta\log\pi_\theta \right].
$$

推导中多出的常数项乘上 score 后期望为零，已经消掉。括号里相当于扣掉 KL 之后的有效反馈。如果数据来自 $\mu$，需要乘上完整回答的重要性比值：

$$
\nabla_\theta J = \mathbb E_\mu \left[ \frac{\pi_\theta(y)}{\mu(y)} \left( R-b-\beta\log\frac{\pi_\theta(y)}{\pi_{\mathrm{ref}}(y)} \right) \nabla_\theta\log\pi_\theta(y) \right].
$$

注意这里是整条回答的比值，换成 token 级比值需要重新推导对应的目标。

落到代码上，一种写法是把系数整体 stop-gradient：

$$
L_{\mathrm{SG}} = -\mathbb E_\mu \left[ \operatorname{sg} \left( w\left[R-b-\beta\log\frac{\pi_\theta}{\pi_{\mathrm{ref}}}\right] \right) \log\pi_\theta \right], \qquad w=\frac{\pi_\theta}{\mu}.
$$

这里 detach 的理由是更新方向已经手推好了，不希望 autograd 再额外加项，并不是因为这些量与参数无关。也可以写成完全可微的重要性加权目标，两种都可以，只是公式和 detach 的方式要配套。

### $k_3$ 估计器

令 $z=\pi_{ref}/\pi_\theta$，$k_3(z)=z-1-\log z$。样本来自 $\pi_\theta$ 时，$\mathbb E_{\pi_\theta}[k_3(z)]$ 正好等于 $D_{\mathrm{KL}}(\pi_\theta\Vert\pi_{\mathrm{ref}})$。样本来自 $\mu$ 时这一点就不成立了，需要补上比值：

$$
\mathbb E_\mu\left[ \frac{\pi_\theta}{\mu}k_3(z) \right].
$$

在完全可微的写法里，外面这个权重也要参与求导，不能 detach。DeepSeek-V3.2 披露的就是对 $k_3$ 加重要性权重的这个修正。

所以以后看 RL 代码，我会多问三句：KL 是对谁取期望？数据实际来自谁？哪些项参与求导？

### 其他几个细节

KL 的方向不能混用：$D_{\mathrm{KL}}(\pi_\theta\Vert\pi_{\mathrm{ref}})$ 在当前策略的分布下衡量偏移，$D_{\mathrm{KL}}(\pi_{\mathrm{ref}}\Vert\pi_\theta)$ 在参考策略的分布下衡量，两者给回答空间的权重不同，梯度估计也不同。

非归一化 KL 写作：

$$
D_{\mathrm{UKL}}(p\Vert q) = \sum_y \left[ p(y)\log\frac{p(y)}{q(y)}-p(y)+q(y) \right].
$$

$p,q$ 都归一化时后两项求和抵消，数值和 KL 一样，但展开成采样损失时表达式不同，这里并不是说 softmax 不归一化了。

RPG 实际训练时还对重要性权重做了裁剪。未裁剪时推导对应原始目标，裁剪后变成有偏但更稳定的更新，这两种情况的理论保证要分开看。

这篇给我的感觉是一个审计 RL loss 的框架。很多实现看起来都只是"某个系数 $\times\log p$"，区别全在系数里放了什么、样本从哪来、哪里断了梯度。

## BPO：为什么可以不训 critic

这是四篇里和"为什么可以不训练 critic"联系最紧的一篇。它利用了 LLM 生成的特殊性：状态转移是确定的，而且只有终局奖励。问题是，在这个设定下，能不能把原本需要中间 advantage 的策略改进，改写成只依赖整条轨迹的目标？

### 理想的一步更新

对某个前缀 $h$，理想目标是在 KL 约束下提高优势：

$$
\max_\pi \left\{ \mathbb E_{a\sim\pi}[A^\mu(h,a)] - \frac1\eta D_{\mathrm{KL}}(\pi\Vert\mu) \right\}.
$$

它有解析解：

$$
\pi^+(a\mid h) = \frac{\mu(a\mid h)\exp(\eta A^\mu(h,a))}{Z(h)}.
$$

也就是在旧概率的基础上，按动作优势做指数加权，$\eta$ 越大偏好越强。难点在于每个中间 token 的 $A^\mu(h,a)$ 从哪里来。

### Bellman 方程让中间 value 抵消

记 $V^\mu(h)$ 为从前缀 $h$ 出发、继续按 $\mu$ 生成的期望奖励。选定一个 token 后下一个前缀就确定了，没有中间奖励时：

$$
A^\mu(h_t,y_t) = V^\mu(h_{t+1})-V^\mu(h_t).
$$

沿整条回答求和，中间项全部抵消：

$$
\sum_t A^\mu(h_t,y_t) = R(x,y)-V^\mu(x).
$$

举个例子，价值依次是 $0.2\to0.3\to0.8\to1.0$，每步增量 0.1、0.5、0.2，加起来正好是 $1.0-0.2=0.8$。只要知道起点和终点，就不需要分别估计每个中间值。这一步是我读这篇时觉得最巧妙的地方。

### 把 advantage 换成概率关系

光有总和还不够决定每一步怎么更新，需要和上面的解析解接起来。对理想更新取对数得到 $\log\frac{\pi^+}{\mu} = \eta A^\mu-\log Z$，再利用 $\mathbb E_{a\sim\mu}[A^\mu(h,a)]=0$，可以得到 $\log Z(h)=D_{\mathrm{KL}}(\mu\Vert\pi^+)$，于是：

$$
\eta A^\mu = \log\frac{\pi^+}{\mu} + D_{\mathrm{KL}}(\mu\Vert\pi^+).
$$

沿轨迹求和：

$$
\sum_t \left[ \log\frac{\pi^+(y_t\mid h_t)}{\mu(y_t\mid h_t)} + D_{\mathrm{KL}}(\mu(\cdot\mid h_t)\Vert\pi^+(\cdot\mid h_t)) \right] = \eta[R-V^\mu(x)].
$$

左边只有策略概率，右边只剩终局奖励和起点价值。把两边之差当作残差，最小化残差平方即可。论文证明在其设定下，这个重写与原 Policy Mirror Descent 问题有相同的唯一最优解。需要注意保证的只是最优解相同，训练过程和后面的近似并不在这个定理的覆盖范围内。

### 实现中出现的互补概率比值

实际实现做了几处近似：对残差梯度线性化，用同题分组估计起点价值，把完整词表 KL 换成二元 KL。最后出现的权重是：

$$
\omega_t = \frac{1+\epsilon-\mu(y_t\mid h_t)}{1+\epsilon-\pi_\theta(y_t\mid h_t)},
$$

再配合上限截断和 mask，构造加权的 $\log\pi_\theta$ 损失。忽略 $\epsilon$ 后和普通的重要性比值对比一下：

|token 概率变化|$\pi/\mu$|$(1-\mu)/(1-\pi)$|
|---|--:|--:|
|$0.001\to0.002$|2|约 1.001|
|$0.8\to0.9$|1.125|2|

普通比值对低概率 token 翻倍很敏感，互补比值则对高概率 token 进一步接近 1 更敏感。这个权重是从 BPO 的目标求导得到的，并不是 $\mu$ 到 $\pi$ 的密度比，不能当作 IS 来理解。另外 BPO 虽然不需要 learned critic，但仍然依赖同题多条回答来估计优势，这一点和 FlashREINFORCE 不同。

### 和 Score Centering 的联系

这一段是我自己把公式放在一起推的，论文里没有这么说。BPO 在尚未把完整 KL 换成二元 KL 时，梯度里包含：

$$
\nabla_\theta \left[ \log\pi_\theta(a\mid h) + D_{\mathrm{KL}}(\mu(\cdot\mid h)\Vert\pi_\theta(\cdot\mid h)) \right].
$$

固定 $\mu$ 时 $\nabla_\theta D_{\mathrm{KL}}(\mu\Vert\pi_\theta) = -\mathbb E_{v\sim\mu}[\nabla_\theta\log\pi_\theta(v\mid h)]$，代回得到：

$$
\nabla_\theta\log\pi_\theta(a\mid h) - \mathbb E_{v\sim\mu}[\nabla_\theta\log\pi_\theta(v\mid h)],
$$

即当前 token 的 score 减去行为策略下的平均 score，正是 Score Centering 的形式。不过最终实现用了二元 KL、平滑和裁剪，完整词表上的平均 score 不再被精确扣除，所以两者只是在中间形式上有联系。

最后说一下 credit assignment。BPO 绕开了显式的中间价值估计，但最终实现里同一条回答的所有 token 仍然共用一个分组优势，只是按概率关系调整了权重。它并不能额外告诉我们"第 438 个 token 才是真正导致失败的那一步"。

## GPM / GPO：一个分数够不够

前三篇关心怎样更新策略，这篇往上游走了一层，问的是判断好坏的奖励模型本身有没有限制。这里的 GPO 是 General Preference Optimization，和 GRPO、GSPO 都不是一回事。（[项目页](https://general-preference.github.io/general-preference-model/)）

### Bradley–Terry 的限制

BT 奖励模型给每个回答一个标量 $r(x,y)$，偏好概率由分数差得到：

$$
P(y_i\succ y_j\mid x) = \sigma(r_i-r_j).
$$

它输出的是概率，处理随机偏好没有问题。限制在于所有比较都要由同一套标量分数差来解释。如果数据中 $P(A\succ B)$、$P(B\succ C)$、$P(C\succ A)$ 都大于 $\frac12$，BT 就需要 $r_A>r_B>r_C>r_A$，这不可能成立，模型再大也没用。当然，这只说明 BT 有一个表达能力的边界，并不说明人类偏好普遍是循环的。

### GPM 的做法

GPM 为每个回答生成一个向量 $v(x,y)\in\mathbb R^d$，两个回答的偏好分数为：

$$
s_{ij}=v_i^\top Jv_j, \qquad J^\top=-J.
$$

$J$ 是反对称矩阵，所以 $s_{ji}=-s_{ij}$，再通过 $P(y_i\succ y_j)=\sigma(s_{ij})$ 得到偏好概率，这样循环偏好也能表示。向量的各个维度是学出来的潜在表示，不一定对应"严谨""简洁"这类可解释标签。

计算上也比较划算。$K$ 个候选回答两两交给大模型比较需要约 $K^2$ 次调用，GPM 只需要每个回答算一次向量，剩下的都是便宜的矩阵运算。如果只需要一个回答对整组的平均得分，还可以先求平均向量 $\bar v=\frac1K\sum_jv_j$，再算 $v_i^\top J\bar v$。

### 没有总分怎么做 RL

引入一个对手分布 $\nu$，定义回答相对于对手的平均偏好分数，然后在 KL 约束下优化它：

$$
\widetilde r_\nu(x,y) = \mathbb E_{y'\sim\nu}[s(x,y,y')], \qquad \max_\theta\ \mathbb E_{y\sim\pi_\theta}[\widetilde r_\nu(x,y)] - \beta D_{\mathrm{KL}}(\pi_\theta\Vert\pi_{\mathrm{ref}}).
$$

这看起来还是一个标量 reward，区别在于它依赖对手分布。可以类比石头剪刀布：石头没有固定的总分，对手里剪刀多时它好，布多时它差。GPM 去掉的是"每个回答都有一个与比较对象无关的总分"这个假设。

反过来，如果偏好分数退化成 BT 形式 $s(y,y')=r(y)-r(y')$，那么 $\widetilde r_\nu(y) = r(y)-\mathbb E_{y'\sim\nu}[r(y')]$，对手固定时后一项是常数，优化它和优化 $r(y)$ 等价。也就是说传统的标量 reward 是它的一个特例。

实际训练中，GPO 用当前一组回答的平均偏好分数来推动新旧策略的 log-probability 差，损失是概率比值的回归形式，归一化项做了近似。

我觉得它更适合评价维度相互冲突、相对比较比绝对打分更自然的任务，比如开放式写作和对话。数学题这类可以直接判对错的任务，0/1 奖励已经够清楚，不太需要它。

## 实验结果

四篇的模型、预算、反馈类型和评测协议都不同，不能拿各自的最高分互相比较。下面只记录每篇论文内部的代表性结果。

|论文|代表性结果|阅读时要注意|
|---|---|---|
|**FlashREINFORCE**|Qwen2.5-Math-1.5B 五项平均 38.0，引用的 GRPO 基线 36.3；rollout 数 256k 对 512k|基线来自已发表结果，不是同条件重跑；rollout 减半不等于时间或算力减半。[论文](https://yifanzhang-pro.github.io/FlashREINFORCE/FlashREINFORCE.pdf)|
|**RPG**|8K 实验中 RPG-REINFORCE-URKL 的 AIME25 "Best" 为 52.08%，同表 DAPO 为 47.40%|"Best" 是训练过程中最好的 checkpoint，不是最终结果|
|**BPO**|Qwen3-30B-A3B-Base 上 AIME24–26 平均峰值 50.5%，CISPO 47.4%，GRPO-ClipHigher 39.5%|同样按峰值选 checkpoint，只在数学训练设置上验证|
|**GPM**|Gemma-2B-it 上 RewardBench 平均分从 BT 的 74.85 提升到 82.29|衡量的是偏好判断能力，不是训练出的策略变强了多少|

GPM 的 AlpacaEval 2.0 表格值得细看：8B 反馈模型、GPO 第三轮时，GPM 的普通胜率高于 BT，但长度控制胜率反而更低（38.98 对 40.37），回答也明显更长。只看摘要里的胜率提升，很容易漏掉这一点。

## 放在一起看

四篇大致对应 RL 训练链条上的四个环节：

$$
\underbrace{\text{判断好坏}}_{\text{GPM/GPO}} \to \underbrace{\text{策略怎样改进}}_{\text{BPO}} \to \underbrace{\text{梯度怎样算对}}_{\text{RPG}} \to \underbrace{\text{系统里怎样稳定更新}}_{\text{FlashREINFORCE}}
$$

它们不是互斥的选项，但也不能不经推导直接拼成一个 loss。比如用 GPM 给出相对于固定对手的分数，再接一个 policy gradient，概念上是通的；而把 BPO 的互补概率比值当成 IS 比值塞进 FlashREINFORCE，或者以为加一个 KL 项就能解决 off-policy，就都有问题。

我把读的过程中踩到的几个误区整理成一张表，方便以后看代码时对照：

|看到的做法|先检查什么|并没有意味着|
|---|---|---|
|代码里有 KL penalty|KL 对谁取期望、数据来自谁、哪些项 detach|优化的就是目标里那个 KL|
|用了 token 级重要性比值|前缀是否也来自旧策略，是否需要序列级过滤|off-policy 偏差已被完整修正|
|用 batch 均值做基线|batch 内题目难度和奖励尺度是否接近|得到了可靠的 value 估计|
|对轨迹做长度归一化|样本权重怎样改变|仍是原目标的无偏估计|
|不训练 critic|优势从哪来：同题分组、batch 均值还是结构抵消|已经解决 credit assignment|
|换了新的 reward / preference 模型|长度控制胜率、固定任务、reward hacking|策略能力一定变强|

## 总结：这次能学到哪些做 RL 的技巧？

**1. 先分清 $\mu$、$\pi_\theta$、$\pi_{ref}$。**

异步训练、版本滞后、训推数值差异都会让 $\mu\neq\pi_\theta$。写 loss 之前先明确每一项的期望对谁取，哪个比值在修正采样，哪个比值在衡量偏移。

**2. 从目标函数一路推到 autograd 代码。**

数值估计对不代表梯度对。每个 detach 都要有理由，完全可微写法和 stop-gradient 写法要配套；用 $k_3$ 这类估计器时，检查数据来源是否需要重要性权重。

**3. 训练组织本身就是算法的一部分。**

FlashREINFORCE 处理的等待、过期、长度问题都不涉及新的理论，但每一个都足以让训练失效。先保证反馈能及时、干净地进入训练，再考虑更复杂的优化器。

**4. 遇到难估计的中间量，先看问题结构。**

BPO 利用确定性转移和终局奖励让中间 value 抵消，没有再训一个模型。但理论上的最优解等价，和近似之后的实际算法是两个层面的事。

**5. 改优化器之前，先看 reward 的表示够不够。**

一个固定标量表达不了循环偏好。评价维度冲突的任务，相对比较可能比绝对打分更合适，验证时也要单独看长度控制这类指标。

**6. 读实验表时看清口径。**

"Best" checkpoint、引用的基线、单一任务设置、长度效应，都会影响结论的强度。

如果以后有机会自己做 LLM 的 RL 训练，我会先确定反馈形式和评测协议；写 loss 时逐项核对期望分布和 detach；系统上记录每个 token 的采样概率和策略版本，用序列级过滤和长度重加权处理异步数据；最后用固定任务和长度控制指标验收，而不是只看训练 reward 有没有上涨。
