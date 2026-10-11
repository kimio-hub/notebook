---
title: "CS336 Scaling Laws 完整笔记"
description: "覆盖 Lecture 9 与 11：从幂律是什么，到怎样用小实验决定大模型的参数量、数据量、学习率和 batch size。"
tags:
  - "cs336"
  - "scaling-laws"
category: "courses"
series: "CS336"
date: "2026-07-11"
updated: "2026-07-11"
---

> 覆盖 Stanford CS336 Spring 2026 Lecture 9 与 Lecture 11。目标：不看课程也能从“幂律是什么”一路理解到“怎样用小实验决定大模型的参数量、数据量、学习率和 batch size”。

## 0. Scaling Laws 到底解决什么

假设你有一万张 B200、一个月时间和一批预训练数据。真正的问题不是“能不能训练”，而是：

- 模型做多大？训练多少 token？
- 深一点还是宽一点？Dense 还是 MoE？
- Adam、Muon 或别的 optimizer？
- batch size、learning rate、warmup、decay 怎样设置？
- 小模型上的结论能否转移到最终大模型？

在大模型上逐个试错极其昂贵。Scaling law 的目标是：在小规模、低成本实验上拟合简单而稳定的规律，再外推到大规模。

它不是自然定律，而是特定模型族、数据分布、训练方法和指标下的经验规律。正确态度是：**把 scaling law 当成带误差条的预测模型，而不是宇宙常数。**

---

## 1. 幂律、log-log 图与 loss floor

最基本的数据 scaling law 写成：

$$
E(D)=A D^{-\beta}+E_\infty,
$$

其中：

- $D$：训练数据量，常用 token 数；
- $E(D)$：验证 loss 或超额误差；
- $\beta>0$：数据 scaling exponent；
- $A$：与数据质量、任务、模型族有关的常数；
- $E_\infty$：不可约 loss / 这套模型和数据下的渐近下限。

若先忽略 $E_\infty$：

$$
\log E=\log A-\beta\log D.
$$

所以在横轴 $\log D$、纵轴 $\log E$ 的图上，幂律是一条斜率为 $-\beta$ 的直线。“数据翻倍后 loss 减少多少”由 $\beta$ 决定。

### 1.1 为什么必须考虑 loss floor

如果直接拟合 $L=aD^{-b}$，模型会预测无限数据使 loss 趋于 0。但语言本身有条件熵，数据也有噪声，架构和优化还有不可约误差。更合理的是拟合：

$$
L(D)=L_\infty+A D^{-\beta}.
$$

在接近 $L_\infty$ 时，原始 loss 的 log-log 图会弯曲；减去正确 floor 后，超额 loss 才可能接近直线。错误估计 floor 会显著扭曲 exponent。

### 1.2 Scaling law 与 logistic/S-curve 不矛盾

Accuracy 有上下界，常呈 S 形；cross-entropy loss 或 error 的可缩放区间却可能表现为幂律。指标变换会改变曲线形状，因此不能把某个指标上的 exponent 直接搬到另一个指标。

---

## 2. 为什么幂律会出现

课程用均值估计说明“多项式收敛率”很自然。设：

$$
x_i\sim\mathcal N(\mu,\sigma^2),\qquad \hat\mu=\frac1n\sum_i x_i.
$$

均方误差为：

$$
\mathbb E[(\hat\mu-\mu)^2]=\frac{\sigma^2}{n}.
$$

取对数后：

$$
\log \operatorname{Error}=-\log n+2\log\sigma,
$$

这就是 exponent 为 1 的 scaling law。

但神经网络中观测到的 exponent 往往远小于 1。一个直觉来自非参数估计：若要学习高维空间中的复杂函数，数据要同时覆盖许多自由方向。课程中的简化例子给出 $d$ 维问题可能出现：

$$
E(n)\propto n^{-1/d}.
$$

因此有人把 exponent 与数据的 intrinsic dimension 联系起来。这个解释有启发性，但不是严密定论：真实语言分布、优化过程和神经网络归纳偏置都比这个 toy model 复杂，intrinsic dimension 的估计本身也不稳定。

真正可靠的结论只有：许多学习问题呈多项式改善，所以 log-log 图上的近似线性并不神秘；具体 exponent 为什么是某个值仍是开放问题。

---

## 3. Scaling Laws 的历史直觉

早期统计学习理论关注 sample complexity 上界，但上界不等于实际达到的 loss。1990s–2010s 的实证研究逐渐发现：数据规模与 NLP、机器翻译、语音、视觉性能之间存在可预测关系。

Hestness 等人在 2017 年系统展示多个神经任务的幂律行为，并提前提出几个后来很重要的观点：

- 更大计算往往平滑地换取更高准确率；
- 某些能力在基础 loss 连续改善时会跨过可见阈值，表现得像“涌现”；
- 算法进步和算力增长都可以折算成达到某一性能所需资源的变化。

Kaplan 2020 把参数量、数据量和 compute scaling 系统化；Hoffmann/Chinchilla 2022 重新校准了固定计算预算下模型大小与数据量的最优分配。

---

## 4. 数据 Scaling：数量、组成、重复与分布偏移

### 4.1 数据数量

在模型容量足够、训练设置稳定、尚未触及其他瓶颈时：

$$
L(D)\approx L_\infty+A D^{-\beta}.
$$

这能回答：继续采集/训练更多 token 还值不值？若 $\beta$ 很小，数据翻倍带来的改善有限，但在巨大的商业规模上仍可能值得。

### 4.2 数据质量和 distribution shift

课程强调一种常见现象：数据组成改变往往主要改变曲线的 offset，而 slope 可能相近。高质量、匹配目标域的数据使同样 token 数取得更低 loss，相当于提高“每 token 的有效价值”。

但这不是普适保证。若数据混合改变了任务本身、tokenizer、难度分布或模型进入不同 regime，slope 也可能变化。

### 4.3 为什么用小模型挑数据 mixture 很难

“在小模型上选择最优 mixture，然后放大”可能失败：

- 不同数据源的 exponent 不同，曲线会交叉；
- 小模型容量不足，偏爱简单/重复数据；
- 大模型能从代码、数学、多语言等复杂数据获得不同收益；
- 训练 token 数增加后，某数据源先耗尽并开始重复。

因此 mixture 应看完整 scaling curve，而不是只比较一个小规模点。

### 4.4 重复数据与 effective data

同一 token 第二次出现的价值通常小于第一次。可以定义有效数据量：

$$
D_{\text{eff}}<D_{\text{raw}}
$$

并让重复次数越高，边际有效 token 越少。具体函数依论文和数据而定，核心结论是：**不能把 10 次重复的 1T unique tokens 当成 10T 独立 tokens。**

这也意味着数据筛选策略随训练规模变化：小预算时只用最高质量子集可能最好；大预算时若高质量子集被过度重复，引入更多样、稍低质量的 unique data 反而更好。

### 4.5 Scaling law 可能失效

- 数据耗尽并大量重复；
- 数据分布改变；
- 优化没有收敛或训练不稳定；
- 模型容量成为瓶颈；
- 评测集污染；
- loss 已接近 floor；
- 架构/optimizer/tokenizer 在 sweep 中不一致。

经验曲线描述已观测 regime，不保证无限外推。

---

## 5. 模型参数 Scaling

固定足够数据和良好优化时，模型大小 $N$ 与 loss 常写为：

$$
L(N)=L_\infty+A_N N^{-\alpha}.
$$

Scaling analysis 可以低成本比较模型家族：分别给 Transformer/LSTM、不同 attention、不同 optimizer 拟合曲线，不只比较一个预算点。

### 5.1 为什么单点比较会误导

两种方法可能有不同 offset 和 slope：

$$
L_1(N)=a_1N^{-\alpha_1}+L_{\infty,1},\qquad
L_2(N)=a_2N^{-\alpha_2}+L_{\infty,2}.
$$

方法 1 在小模型领先，方法 2 slope 更陡，放大后反超。需要多规模点判断趋势及交叉点。

### 5.2 参数不是等价的

- Embedding 参数主要随 vocabulary 增长，与 Transformer block 参数的作用不同；
- MoE 的 total parameters 与每 token active parameters 不同；
- 增加深度、宽度、heads 的效果不同；
- 共享参数和低秩参数不能仅用 raw parameter count 比较。

报告 scaling law 必须说明参数口径：total、non-embedding、active 或 trainable parameters。

### 5.3 深度与宽度

从极浅变为合理深度收益明显；在正常 Transformer 范围内，aspect ratio 的影响往往小于总 scale，但不是完全无关。实践可先固定成熟宽深比，再整体放大；若要比较新架构，仍需专门 sweep。

---

## 6. Critical Batch Size

增大 batch 最初能减少梯度噪声并并行处理更多样本，但超过某点后边际收益下降。

一种操作性定义：固定目标 loss，对多个 batch size 训练，记录：

- 达到目标需要的 optimizer steps $S(B)$；
- 消耗的 examples/tokens $E(B)=B\,S(B)$。

小 batch：steps 多但每步样本少；大 batch：steps 下降，但总样本可能浪费。Critical batch 在“最少 steps”和“最少 examples”的权衡附近。课程还指出它与梯度协方差迹和梯度范数平方之比有关，可理解为 gradient noise scale。

经验上目标 loss 越低，critical batch 往往越大。重要区别：

- global batch = microbatch × data-parallel degree × accumulation steps；
- batch 扩大不等于自动提高统计效率；
- 超过 critical batch 后继续增大主要换硬件吞吐，不再等比例减少 steps。

---

## 7. 联合 Model–Data Scaling Law

常用形式：

$$
L(N,D)=L_\infty+A N^{-\alpha}+B D^{-\beta}.
$$

三项分别代表：不可约误差、模型容量不足、数据有限。它能回答固定训练计算下“更大模型”还是“更多数据”。

Transformer 训练 FLOPs 常粗略估计为：

$$
C\approx 6ND,
$$

其中 $N$ 是非 embedding 或 active parameters，$D$ 是训练 token 数。常数 6 来自 forward 约 $2ND$、backward 约为 forward 的两倍；具体架构、attention、MoE 和实现会改变常数。

### 7.1 Compute-optimal 推导

令 $C'=C/6$，约束 $ND=C'$，所以 $D=C'/N$。代回：

$$
L(N)=L_\infty+A N^{-\alpha}+B(C'/N)^{-\beta}
=L_\infty+A N^{-\alpha}+B C'^{-\beta}N^\beta.
$$

求导并令其为 0：

$$
\alpha A N^{-\alpha}=\beta B C'^{-\beta}N^\beta.
$$

因此：

$$
N_{\text{opt}}\propto C^{\frac{\beta}{\alpha+\beta}},\qquad
D_{\text{opt}}\propto C^{\frac{\alpha}{\alpha+\beta}}.
$$

如果 $\alpha\approx\beta$，模型参数和 token 数都近似按 $C^{1/2}$ 增长，因此 tokens/parameter 近似保持常数。这就是 Chinchilla 风格结论的数学来源。

注意：常见“约 20 tokens/parameter”是特定论文、数据、参数定义和优化方案下的经验值，不是所有模型的永久常数。

---

## 8. Kaplan 与 Chinchilla 为什么差很多

Kaplan 的拟合倾向于随着 compute 增长更快地增大模型、较慢增加数据；Chinchilla 得到更平衡的模型/数据增长，认为许多早期大模型训练不足。

差异来源包括：

- 是否计入 embedding/output 参数；
- 小 compute runs 的 warmup 占比过高；
- learning-rate schedule 与 early stopping；
- 拟合函数和 loss floor；
- 选取的模型/data grid；
- 原始 Chinchilla method 3 数据/拟合本身后来被指出存在问题。

所以不要只背 exponent；要审查参数口径、训练是否充分、拟合方法和数据点。

---

## 9. Chinchilla 的三种方法

### 9.1 Method 1：所有训练曲线的 lower envelope

训练多种模型，每条曲线随 token/compute 延伸。对每个 compute budget，从所有已到达的点中取最低 loss，得到最优前沿，再拟合前沿的幂律。

优点：直观，能利用训练曲线中间点。缺点：不同 run 的 schedule 阶段未必可比；lower envelope 对噪声和离群点敏感。

### 9.2 Method 2：IsoFLOPs

选择多个固定 FLOP budgets。对每个 budget，训练不同参数量的模型，并让 token 数满足预算：

$$
D=\frac{C}{6N}.
$$

在同一 $C$ 下，loss 对 $N$ 通常呈 U/凸形：模型太小则 capacity-limited，模型太大则数据不足。找到每个预算的最优 $N^*(C)$，再拟合 $N^*$、$D^*$ 与 $C$ 的幂律。

这是课程最推荐的干净方法之一。缺点是每个预算要训练多个完整 run，成本高。

### 9.3 Method 3：联合拟合

在 $(N,D)$ 网格上训练模型，直接拟合：

$$
L(N,D)=L_\infty+A N^{-\alpha}+B D^{-\beta}.
$$

优点：参数含义清楚，能利用全部点，拟合后可直接做 constrained optimization。缺点：非线性拟合对初始化、loss floor、权重和异常点敏感；若 grid 覆盖不足，不同参数组合会互相补偿，得到看似漂亮但错误的外推。

### 9.4 实践建议

- 优先画原始散点和 residual，不只报 $R^2$；
- 至少留出最大的若干 run 做 out-of-sample 验证；
- 用 bootstrap/不同 seed 给 exponent 置信区间；
- 同时比较 Method 1/2/3 是否大致一致；
- 不要让大量廉价小模型点在 least squares 中压过少量大模型点；
- 在 log-loss 空间还是 raw-loss 空间拟合要明确。

---

## 10. Train-optimal 不等于 Deployment-optimal

Chinchilla 优化的是固定**训练 FLOPs**下最低 loss。但部署总成本是：

$$
C_{\text{total}}=C_{\text{train}}+Q\,C_{\text{inference/request}},
$$

其中 $Q$ 是未来请求或生成 token 数。大模型即使训练更省 token，推理时每个 token 都更贵。若模型会被调用很多次，通常值得训练一个更小模型更久：增加一次性训练成本，降低长期推理成本。

因此现代模型常远超早期的 tokens/parameter：课程列举从 GPT-3 的约 2，到 Chinchilla 约 20，再到 Mistral、Llama 3 的上百。这里“overtrain”是相对 training-compute-optimal 而言，不是对验证 loss 过拟合。

决策时还应考虑显存、KV cache、latency SLA、吞吐、硬件价格和模型使用寿命。

---

## 11. 怎样降低 Scaling Sweep 成本：WSD

Chinchilla grid 看似需要每个 $(N,D)$ 都从头训练到不同终点，成本可能从 $O(k)$ 变成 $O(k^2)$。

**WSD（Warmup–Stable–Decay）** learning-rate schedule：

1. Warmup：学习率升到目标值；
2. Stable：长时间保持近似稳定；
3. Decay：最后约 10% 快速衰减。

从 stable 阶段的 checkpoint 可以分叉多个不同终点长度，各自加 decay，得到多种 token budgets，避免每个 budget 从头训练。MiniCPM 和 DeepSeek 都使用类似 piecewise schedule 做低成本 scaling analysis。

注意：stable 阶段的即时 loss 可能比 cosine 慢，但加 decay 后最终 loss 快速下降；比较 run 时必须处于可比 schedule 阶段。

---

## 12. 公开 Scaling Recipe 案例

### 12.1 MiniCPM

- 使用 muP 稳定宽度扩展；
- 固定 aspect ratio，整体放大模型；
- 在 9M、30M、170M 等小模型上拟合 optimal LR/batch；
- 使用 WSD 复用 stable checkpoint；
- 用 lower envelope 和联合拟合估计 model–data tradeoff；
- 最终训练规模与最大 sweep 模型仍有约 5× 外推距离。

课程强调其结果有趣但不应盲从：其 joint fit 得到很高 data/model ratio，曲线是否真正线性、fit 是否稳健仍需检查。

### 12.2 DeepSeek

- 不依赖 muP，而是直接在小规模上搜索 optimal LR/batch；
- 收集接近最优的超参数点再拟合趋势；
- 使用 WSD 风格 schedule；
- 用 IsoFLOPs 选择模型大小与数据量；
- scaling 模型能较准确预测最终 7B/67B loss。

其学习率趋势并非总是十分干净，这恰好说明：batch/LR 的 scaling 比 loss scaling 更脆弱。

### 12.3 近期模型的共同做法

- Qwen：对 LR/batch 做小规模拟合；
- Kimi K2、Hunyuan：为 MoE 拟合 sparsity、active/total parameters 与数据比例；
- Llama 3：IsoFLOPs 选择模型和 token 规模，并检查 downstream scaling；
- MiniMax：用 scaling curves 比较架构方案。

共同模式是：成熟 Transformer 超参数多直接固定，只重点搜索 LR、batch、模型/data 分配和 MoE sparsity。

---

## 13. Learning Rate、Optimizer 与尺度

预训练 loss 在局部的 $(\text{LR},\text{batch})$ 网格上常呈较干净的凸形，可以找最优点。但趋势受以下因素影响：

- 模型宽度/深度；
- token budget 和目标 loss；
- schedule 类型；
- optimizer 参数化；
- 数据集与 MoE/dense 架构；
- warmup 和 weight decay。

比较 optimizer 时，必须为每个 optimizer 独立调参。用 Adam 的 LR 直接跑 Muon/SGD，再断言 optimizer 不好，是无效比较。还要在多个 compute 和 Chinchilla ratios 上检查，避免一个小规模点的胜利在放大后消失。

Muon 对矩阵参数的更新做近似正交化，在小模型 speedrun 与 Kimi K2 等规模上显示可用，但其真实 scaling gain 很难与调参、compute 和架构变化完全分离。

---

## 14. muP：为什么小模型超参数可能转移

Standard parametrization 下，宽度变化会改变 activation 和更新的尺度，导致 optimal LR 随宽度漂移。**Maximum Update Parametrization（muP）**希望满足：

1. 初始化 activation 保持 $\Theta(1)$；
2. 一步更新引起的 activation 变化也保持 $\Theta(1)$。

这里每个分量是 $O(1)$，宽度为 $n_l$ 的 activation 向量范数约为 $\Theta(\sqrt{n_l})$。

对线性层：

$$
h_l=W_lh_{l-1},\qquad
\Delta W_l=-\eta_l\nabla_{h_l}\ell\,h_{l-1}^\top.
$$

muP 根据 fan-in/fan-out 为不同参数类型选择初始化方差、learning-rate multiplier 和输出缩放，使宽度增大时 feature learning 的更新仍为有限量，而不是消失或爆炸。

### 14.1 muP 的实际价值

- 宽度 sweep 中 optimal LR 更稳定；
- 小 proxy model 的调参更可能转移；
- CerebrasGPT、MiniCPM 等显示更稳定的 scaling；
- 相比 standard parametrization，失败和漂移往往更少。

### 14.2 muP 不是魔法

理论假设与现代 LM 有差距：SwiGLU、RMSNorm gain、attention、weight decay、sign-based optimizer 等都会改变尺度。课程列举：learnable RMSNorm gain、exotic optimizer、较强 weight decay 可能破坏转移；有些问题可通过移除 norm gain 或重新参数化缓解。

muP 主要处理**宽度**扩展，不自动解决深度、batch、sequence length、数据分布和架构变化。

---

## 15. Downstream Scaling 与“涌现”

预训练 loss 往往平滑、可预测；downstream accuracy 可能不平滑，因为：

- accuracy 是离散阈值指标；
- prompting/evaluation format 引入额外噪声；
- 某能力只有超过 loss 阈值才显现；
- benchmark 小，置信区间大；
- contamination 或 memorization；
- 不同任务曲线斜率不同。

因此“预训练 loss 外推准确”不等于“所有 benchmark 都能准确外推”。最好同时拟合 loss，并研究 downstream metric 相对 loss 的关系。

---

## 16. 一套可执行的 Scaling 实验流程

### Step 1：定义固定不变的实验族

固定 tokenizer、数据 mixture、context length、架构比例、optimizer family、schedule 定义和参数统计口径。

### Step 2：选对数均匀的规模点

例如参数量按约 2–4× 递增；不要把所有点挤在同一量级。至少跨 1–2 个数量级才容易识别 slope。

### Step 3：先做稳定性与 LR/batch sweep

对几个代表规模搜索 LR 和 batch，判断是否保持不变、呈幂律变化，或需要 muP。

### Step 4：做多个 IsoFLOP budgets

每个 budget 选择多个 $N$，令 $D=C/(6N)$；保证 U 形最小值两侧都有点，而不是最优点落在搜索边界。

### Step 5：拟合与诊断

- 在 log 空间检查线性；
- 画 residual 对 $N,D,C$；
- 检查小/大规模是否系统偏离；
- bootstrap seeds；
- 留出最大模型验证外推；
- 比较多个 functional forms 和参数口径。

### Step 6：加入现实约束

把训练预算、推理需求、GPU memory、并行效率、数据上限和 deadline 纳入，而不是只最小化理论 loss。

### Step 7：大训练前做一次中等规模复核

外推距离过大时，先训练一个介于 sweep 与目标之间的 anchor model，验证 loss、稳定性和系统吞吐。

---

## 17. 常见致命错误

1. 用训练 FLOPs 代替实际 wall-clock，却忽略不同模型 MFU 不同；
2. sweep 中不同模型没有调好 LR/batch；
3. 把未收敛或仍在 warmup 的点混入；
4. 参数统计一会儿含 embedding、一会儿不含；
5. 只用 seed=1，拟合噪声为规律；
6. 用同一数据重复很多遍，却按 raw tokens 计；
7. 只看 $R^2$，不看外推误差和 residual；
8. 从很小模型外推过多数量级；
9. 用预训练 loss 规律保证 downstream 能力；
10. 忽略未来 inference 成本，机械采用 train-optimal Chinchilla 点。

---

## 18. 自测题

1. 为什么幂律在 log-log 图上是直线？斜率代表什么？
2. 为什么 loss floor 估计错误会改变 exponent？
3. 数据重复为什么不能按 raw token 数计算？
4. 联合 scaling law 三项分别代表什么瓶颈？
5. 从 $C\approx6ND$ 怎样推出 $N_{opt}$ 和 $D_{opt}$ 的 compute exponent？
6. IsoFLOPs 曲线为什么通常呈 U 形？
7. Chinchilla Method 1、2、3 的数据需求和风险分别是什么？
8. 为什么部署量很大时应训练更小模型更久？
9. WSD 怎样降低 sweep 成本？
10. muP 保持哪两个量随宽度为 $\Theta(1)$？它不能解决什么？

## 19. 最应该记住的结论

1. Scaling law 是经验预测工具，不是永远成立的定律。
2. 幂律意味着资源增加带来稳定但递减的收益。
3. 固定 compute 下，模型容量和数据不足必须平衡。
4. IsoFLOPs 是最直观、稳健的 compute-optimal 实验方法之一。
5. 超参数、参数口径、schedule 和数据重复会严重污染拟合。
6. Train-optimal 与 deployment-optimal 是不同目标。
7. 小规模外推必须用中间 anchor、残差和留出大模型验证。

## 资料

- [课程官方 Lecture 9：Scaling Laws Basics](https://github.com/stanford-cs336/lectures/blob/main/lecture_09.pdf)
- [课程官方 Lecture 11：Scaling Case Study and Details](https://github.com/stanford-cs336/lectures/blob/main/lecture_11.pdf)
- [Kaplan et al., Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361)
- [Hoffmann et al., Training Compute-Optimal Large Language Models](https://arxiv.org/abs/2203.15556)
- [muP / Tensor Programs V](https://arxiv.org/abs/2203.03466)
