---
title: "把一次 12.7T tokens 的训练读懂：Marin 8B 开源训练复盘"
description: "沿着 Marin 8B 的六个训练阶段，学习怎样阅读训练曲线、设计冷却、使用 EMA，并从失败分支中定位训练问题。"
tags:
  - llm
  - pretraining
  - optimization
  - marin
category: "blog"
feature: true
date: "2026-09-14"
updated: "2026-09-14"
---
最近关注到 Percy Liang 团队的 [Marin 训练项目](https://mtracker.oa.dev/hero-run-535b)，我注意到之前还公开了 Marin 8B 和 Marin 32B 的模型训练过程，因此准备对这两次训练进行一些学习，尽力弥补目前没算力、无法进行大规模实验的问题。

这次先学习 Marin 8B，主要对照 [官方复盘 Marin 8B Retro](https://marin.readthedocs.io/en/latest/reports/marin-8b-retro/)、[W&B 训练曲线](https://wandb.ai/marin-community/marin/reports/Tootsie-8B---VmlldzoxMTY3MzU3OA) 和[历史训练配置](https://github.com/marin-community/marin/blob/ee163702c5bc71c9bbba3238db84b6ee86e826a7/experiments/tootsie/exp600_tootsie.py)。本文只记录 8B，重点是每个阶段为什么改配置，以及曲线能支持什么判断。

## 模型配置与训练地图

先讲讲这个模型的基本参数以及 architecture。

|**Parameter**|**Value**|
|---|---|
|`hidden_dim`|4096|
|`intermediate_dim`|14336|
|`num_heads`|32|
|`num_kv_heads`|8|
|`num_layers`|32|
|`activation_function`|`silu`|

架构选择的是 Llama，以上主要尺寸与 Llama 3.1 8B 一致，但这是 Marin 自己从头预训练的模型，不是从 Llama 3.1 权重开始微调。原始 [LLaMA 论文](https://arxiv.org/abs/2302.13971) 可以用来学习架构背景，具体配置要看 Marin 的记录。

从这些参数还可以算出：每个 attention head 的维度为 $4096/32=128$；32 个 Q heads 对应 8 个 KV heads，即每组 4 个 Q heads 共享一组 K/V，这是 GQA（Grouped-Query Attention）。

每个训练 sample 的 sequence length 是 4096 tokens；sample 是训练序列，不一定恰好对应一篇原始文档。Tokenizer 使用 Llama 3 tokenizer 的变体，模型卡给出的名称是 `stanford-crfm/marin-tokenizer`。[模型卡](https://huggingface.co/marin-community/marin-8b-base#model-details)

精度是 **FP32 / BF16 混合精度**：模型参数和优化器状态使用 FP32，主要计算使用 BF16（bfloat16），最终词表 softmax 是例外，不能笼统写成所有运算都是 BF16。这里原先的 `bfp16`、`bfp32` 是笔误。训练使用 Levanter / JAX，优化器选择 AdamW；输入 embedding 和 layer norm 参数不做 weight decay。[配置说明](https://marin.readthedocs.io/en/latest/reports/marin-8b-retro/#architecture)

![Marin 8B 的训练主线：六个阶段沿同一组权重继续，诊断实验从中途检查点分叉。](/notes-assets/marin-8b/training-map.png)

*Marin 8B 的训练主线：六个阶段沿同一组权重继续，诊断实验从中途检查点分叉。*

这是训练的主线阶段，累计 tokens 使用模型卡中的约数。

|阶段|累计训练量，约数|这一站要理解的问题|
|---|--:|---|
|**Kestrel**|0 → 2.7T|如何开始预训练？为什么周期性降低学习率？|
|**Ocelot**|2.7T → 3.78T|换算力、增大 batch 后怎么调整？如何用 EMA 观察模型？|
|**Jellyfish**|3.78T → 4.78T|怎样结束一轮训练？如何选择冷却阶段的数据？|
|**Phoenix**|4.78T → 11.1T|已经冷却的模型，能不能重新升温继续训练？|
|**Starling**|11.1T → 约 12.4T|如何把前面发现的经验用于最终冷却？|
|**Deeper Starling**|约 12.4T → 12.7T|固定在低学习率继续训练，还得到什么？|

这些阶段是一条连续续训的主线，Raccoon / Spoonbill 是从中间 checkpoint 分出的诊断实验，不把它们的训练量加进最终主线。复盘中的约 12.75T 和模型卡中的约 12.7T 是不同取整口径，不应拿这些约数反推精确边界。[检查点列表](https://huggingface.co/marin-community/marin-8b-base#base-model-checkpoints)

### 读图前先统一单位

一个 step 指一次优化器更新，这次更新使用的是一个全局 batch，而不是只取一段文本。全局 batch 可以分散在多个设备上，也可以通过梯度累积完成。

| 阶段 | 全局 batch 中的序列数 | 每步 tokens（序列长 4096） |
|---|---:|---:|
| Kestrel | 1024 | 4,194,304，约 419 万 |
| Ocelot / Jellyfish / Phoenix | 3072 | 12,582,912，约 1258 万 |
| Starling / Deeper Starling | 4096 | 16,777,216，约 1678 万 |

因此累计训练量应按各阶段的 batch 分段相加：$T=\sum_s B_s S$。同样是 10,000 steps，Ocelot 处理的 tokens 是 Kestrel 的 3 倍，不能只用对 step 的斜率比较学习效率。这里的 tokens 是训练处理量，可能包含重复采样，不等于互不重复的数据量。

纵轴也要区分：以自然对数计算的 token 平均交叉熵为 $L_{\mathrm{CE}}$ 时，$\mathrm{PPL}=e^{L_{\mathrm{CE}}}$；BPB（bits per byte）则是总负对数似然换算成 bits 后，再除以对应文本的字节数：

$$
\mathrm{BPB}=\frac{-\sum_i\ln p_\theta(x_i\mid x_{<i})}{N_{\mathrm{bytes}}\ln 2}.
$$

所以 BPB 为 0.75，不代表 loss 或 PPL 为 0.75。比较时还要固定计分位置、文本预处理和上下文设置。[Paloma 评估论文](https://arxiv.org/html/2312.10523v2)

## Phase 1：Kestrel

在训练一个 step 时，我们取出一个 batch 的训练序列，模型对每个可计分位置预测下一个 token，计算交叉熵，反向传播得到梯度，优化器更新参数。

这里有两种需要区分的 loss：

- **Train loss**：当前参数在当前 batch 上的预测误差，用它计算梯度，再进行这一步更新。因此它通常是这一步更新之前的误差，受到模型状态和当前 batch 的共同影响。
- **Eval loss**：隔一段时间，用固定验证集重新测试，不用这次验证结果反向传播更新参数，更适合观测模型随训练的变化。

Kestrel 的初始配置是 AdamW，峰值 LR $10^{-3}$，weight decay 0.05，约 1000 steps 的初始 warmup，每步 $1024\times4096$ tokens。[历史配置中的 `tootsie_phase1_config`](https://github.com/marin-community/marin/blob/ee163702c5bc71c9bbba3238db84b6ee86e826a7/experiments/tootsie/exp600_tootsie.py)

Warmup 是从较低学习率逐渐进入目标学习率，然后在高学习率平台继续学习。Weight decay 给适用的参数施加额外收缩；在 AdamW 中，这部分与自适应梯度更新分开处理。

数据从 DCLM Baseline、StarCoder Data 和 Proofpile 2 的混合开始。这里要同时记住“模型怎样更新”和“模型正在读什么”：只抄 LR，不能还原训练方案。[阶段数据说明](https://marin.readthedocs.io/en/latest/reports/marin-8b-retro/#data-mix)

![Kestrel 的学习率周期性冷却与恢复，为长训练提供可评估的检查点。](/notes-assets/marin-8b/kestrel-learning-rate.png)

*Kestrel 的学习率周期性冷却与恢复，为长训练提供可评估的检查点。*

图中 `optim/learning_rate` 长时间停在高位，周期性下降，然后重新回到高位。这是 WSD-S（Warmup-Stable-Decay-Simplified）：初始 warmup 之后，反复进行 stable 和 decay，并从冷却后的 checkpoint 接着训练。[WSD-S 论文](https://arxiv.org/html/2410.05192v2)

为什么这样安排？我的理解是：最终训练预算未必一开始就确定，高 LR 阶段可以持续推进，周期性冷却则提供相对适合评估和使用的 checkpoint。降低 LR 可以减少随机更新带来的波动，使已有训练进展更明显地体现在 loss 上。**这里不能直接说“已经收敛到局部最优”——曲线下降没有证明这一点。**

![训练损失随学习率周期下凹；比较时需要对齐相似的学习率状态。](/notes-assets/marin-8b/kestrel-loss.png)

*训练损失随学习率周期下凹；比较时需要对齐相似的学习率状态。*

这里蓝色 train loss 有周期性下凹；橙色 C4 eval loss 的采样更稀疏，周期细节不如训练曲线明显。结合 LR 计划和约 200k steps 的转折，可以把主要变化理解为预期的冷却响应。不是 train loss 的下降“导致”了 eval loss 下降，而是模型参数状态的变化同时影响了两者。

当 LR 提高时，loss 有一定反弹，也不能立刻叫作模型遗忘。还要比较相似的训练状态：高 LR 对高 LR、冷却结束对冷却结束，并确认评测集和评测方式相同。

再看 LR 的周期变化：官方记录前 200k steps 是每 10k steps 安排 1k steps decay，之后变成每 20k steps 安排 2k steps decay。冷却步数占比仍然是 10%，改变的是每次冷却的连续长度。图上调整后有进一步下降，但还要检查不同评估域是否都受益。

![分域 BPB 并不同步变化，局部异常仍需检查格式与逐 token 误差。](/notes-assets/marin-8b/domain-bpb.png)

*分域 BPB 并不同步变化，局部异常仍需检查格式与逐 token 误差。*
再看分域评估图，部分网页、论文域的 **BPB** 下降，而社交文本域在转折点反而上升，`macro_bpb` 也出现了跳升。不能仅据此判断“模型丧失了社交文本能力”。

**团队的后续分析**：一些 Paloma 子集的文本固定以空格结尾，与训练数据的格式不同；更深冷却后，模型对这些末尾空格赋予更低概率，导致相关评估变差。这一现象提醒我，分域异常既可能来自能力取舍，也可能来自少数格式 token，需要下钻到逐文档、逐 token 的误差。[官方分析说明](https://marin.readthedocs.io/en/latest/reports/marin-8b-retro/#wsd-cycle-change)

## Phase 2：Ocelot

Ocelot 换到 TPU v4-2048，将全局 batch 从 1024 个样本增加到 3072 个，LR 也从 $10^{-3}$ 调到 $1.7\times10^{-3}$（$1.7\approx\sqrt{3}$）。这是这次训练采用的 batch / LR 缩放选择，不能直接当作所有 AdamW 训练都适用的规则。

同时从周期性的 WSD-S 切换到 WSD，取消反复短冷却，在高 LR 阶段用参数 EMA（Exponential Moving Average，指数移动平均）监控表现，$\beta=0.995$：[Ocelot 配置](https://github.com/marin-community/marin/blob/ee163702c5bc71c9bbba3238db84b6ee86e826a7/experiments/tootsie/exp600_tootsie.py)

$$
\bar\theta_t=0.995\bar\theta_{t-1}+0.005\theta_t.
$$

相当于维护一个跟着训练模型走的影子模型：每步保留旧 EMA 权重的 99.5%，吸收当前权重的 0.5%。**平均的是参数，再用平均后的模型重新评估；不是把 W&B 曲线做显示平滑，也不是直接平滑 AdamW 的更新。** 若每步更新一次 EMA，其历史权重的特征时间尺度约为 $1/(1-\beta)=200$ steps，但不是只保留最近 200 步的硬窗口。

![普通权重与 EMA 权重在高学习率阶段保持差距，最终冷却后逐渐靠近。](/notes-assets/marin-8b/ema-vs-weights.png)

*普通权重与 EMA 权重在高学习率阶段保持差距，最终冷却后逐渐靠近。*

图的横轴是 step，纵轴是 C4 BPB；`eval/paloma/...` 和 `eval/ema/paloma/...` 分别对应普通权重与 EMA 权重。Ocelot 和 Phoenix 的高 LR 阶段，EMA 线更低，两者维持相对稳定的距离，后面进入 Starling 冷却阶段，距离逐渐减少。

怎么读？看上下：同一评测集上 EMA 更低，说明平均权重模型在这个指标上更好；看共同趋势：两者都下降，说明都在改善；看距离随 LR 怎么变：冷却后两条线靠近，与普通参数的波动减小相符。

这里还有两个工程细节：团队起初写反了 EMA 的平均方向，早期 EMA 评估因此没有明显优势；这一阶段也把早期误用的 Llama 2 RoPE 设置改为 Llama 3 风格。所以“主要尺寸与 Llama 3.1 一致”不代表整个训练过程的所有架构设置都没变。[官方 Ocelot 记录](https://marin.readthedocs.io/en/latest/reports/marin-8b-retro/#adjusted-hyperparameters)

### 补充：冷却数据不是凭“高质量”三个字选出来的

进入 Jellyfish 前，还应补上 microannealing（短程退火实验）这一环：从已经预训练较久的 checkpoint 分出短实验，固定预算和学习率计划，比较加入不同数据源后的效果。实验问题是“这份数据能否改善目标任务”，而不仅是“它看起来是否像高质量文本”。[数据实验 #784](https://github.com/marin-community/marin/issues/784)、[候选数据评估 #820](https://github.com/marin-community/marin/issues/820)

官方复盘记录：单纯提高 HQ 数据占比，可以改善相近领域的验证 loss，却未必改善下游任务；在这组短实验中，70% 预训练数据 / 15% FLAN / 15% HQ 的组合效果较好，而 70% 预训练数据 / 30% FLAN 并没有优于纯预训练数据对照。[Microannealing 结果](https://marin.readthedocs.io/en/latest/reports/marin-8b-retro/#interlude-microannealing)

我的理解：数据的价值取决于它与现有数据的互补、配比和评测目标，不能简单给每个数据集排一个通用的“质量高低”。而且短实验里有效的比例，也不能未经验证就外推到长程训练。


## Phase 3：Jellyfish

这个阶段做了约 1T tokens 的线性冷却，LR 从 $1.7\times10^{-3}$ 到 $1.7\times10^{-4}$，对应约 79,500 steps，同时采用新的 cooldown 数据混合。

这里同时做了两件事：改善优化状态，让模型从高 LR 进入更适合收尾的状态；调整数据分布，让剩余预算服务于希望强化的能力。主要结构是约 70% DCLM HQ 与 StarCoder，另约 30% 为论文、数学等来源。它不是直接照搬上面 70/15/15 的短实验组合，Jellyfish 当时没有纳入 FLAN，并排除了部分合成数学数据。[Jellyfish 数据记录](https://marin.readthedocs.io/en/latest/reports/marin-8b-retro/#phase-3-jellyfish-first-cooldown)

回到之前 EMA 总览图的 Jellyfish 段，C4 BPB 在阶段开始附近上升，之后回落。Marin 团队的解释是：可能与 DCLM HQ 和 C4 之间更明显的格式差异有关，但明确表示没有深入验证。[C4 现象与假说](https://marin.readthedocs.io/en/latest/reports/marin-8b-retro/#c4-en-perplexity)

这与 Kestrel 的末尾空格分析要分开：不能把已经分析过的一次格式问题，当作后面所有 C4 异常的已证实原因。因为 LR 和数据一起变了，这一阶段也不能只凭一条曲线拆出两者各自贡献。

## 诊断分支：Raccoon / Spoonbill

这是从 Jellyfish checkpoint 分出的实验，和 Phoenix 主线并行探索，放在这里是为了接着理解第一次冷却留下的问题。

Raccoon 的动机是研究 Jellyfish 再冷却一些之后，是否更容易进行 SFT，于是继续把 LR 从 $1.7\times10^{-4}$ 降到 $1.7\times10^{-5}$，预算约 100B tokens。[Raccoon issue #898](https://github.com/marin-community/marin/issues/898)

![更深冷却中的损失先降后升，触发后续诊断。](/notes-assets/marin-8b/deep-cooldown-failure.png)

*更深冷却中的损失先降后升，触发后续诊断。*

看 loss 图，进一步冷却开始时 loss 下降，但后面转为上升，而且不是孤立尖峰，而是持续的趋势变化。Raccoon 的 issue 记录了去掉 weight decay、重置 AdamW 状态等尝试，都没有解决这个问题。

我的理解：这两个实验分别在检查参数收缩和历史优化器状态是否是可干预因素。失败结果只能说明这些具体干预没有救回这条 run，不能逻辑上排除它们与问题存在任何关联。

随后，Spoonbill 又尝试不降得那么深，将**冷却终点**设为 $3\times10^{-5}$，不是从头固定在这个 LR；还加入少量 Tulu 3 和 FLAN。但问题仍然出现，因此它也不是只改变一个 LR 变量的严格对照。[Spoonbill issue #916](https://github.com/marin-community/marin/issues/916)

![分模块参数 norm 暴露出输出头的异常增长。](/notes-assets/marin-8b/parameter-norms.png)

*分模块参数 norm 暴露出输出头的异常增长。*

这里记录了分模块参数 norm 的变化：某些模块在下降，另一些在上升，而 `lm_head.weight` 的 norm 明显增长，并在后段加速。`lm_head` 把最后一层隐藏状态映射成词表 logits，因此值得进一步检查输出侧的数值行为。不同面板纵轴范围不同，不能只按曲线的视觉斜率比较异常程度。

### z-loss 具体约束了什么？

设第 $i$ 个计分位置的词表 logits 为 $a_{i,v}$，定义归一化常数 $Z_i=\sum_v\exp(a_{i,v})$。常见的 logit z-loss 写作：

$$
L=L_{\mathrm{CE}}+\lambda_z\frac{1}{N}\sum_{i=1}^{N}(\log Z_i)^2.
$$

实现时用 `logsumexp` 稳定计算 $\log Z_i$，实际平均还要遵守 loss mask。这里 $10^{-4}$ 是正则项的**系数**，不代表 z-loss 的实际数值恒等于 $10^{-4}$。[T5X 的公式与实现](https://t5x.readthedocs.io/en/latest/_modules/t5x/losses.html#cross_entropy_with_logits)

为什么它可能有帮助？Softmax 对 logits 的共同平移不敏感：$\operatorname{softmax}(a+c\mathbf1)=\operatorname{softmax}(a)$。所以 CE 不直接约束这种整体偏移，而 z-loss 会随 $\log Z$ 变化，增加了这方面的约束。它不是直接对 `lm_head` 权重做 L2 正则，也不保证所有参数 norm 都变小。这解释了它为什么可能有用，但不是对这次异常完整根因的证明。

![加入 z-loss 后，绿色分支越过原故障区间继续下降。](/notes-assets/marin-8b/z-loss-recovery.png)

*加入 z-loss 后，绿色分支越过原故障区间继续下降。*

加入 z-loss 后，绿色分支在原先红色分支抬头的区间继续下降，支持“这次干预有效抑制了观测到的异常”。官方报告将输出头的异常作为关键线索，并采用 $10^{-4}$ 的 z-loss 系数修复。[Spoonbill 修复记录](https://marin.readthedocs.io/en/latest/reports/marin-8b-retro/#spoonbill-z-loss-to-the-rescue)


## Phase 4：Phoenix

从 Jellyfish 继续，重新提高 LR，并过渡到 Nemotron-CC 加代码数据。配置安排约 2000 steps 的过渡，把 LR 从 $1.7\times10^{-4}$ 线性拉回 $1.7\times10^{-3}$，同时使用桥接数据混合，之后进入目标配比并保持高 LR。按当前 batch 算，这段过渡约处理 25.2B tokens。[Phoenix 记录](https://marin.readthedocs.io/en/latest/reports/marin-8b-retro/#phase-4-phoenix-reheated)

![Phoenix 把冷却后的模型重新升温，同时切换训练数据。](/notes-assets/marin-8b/phoenix-reheat.png)

*Phoenix 把冷却后的模型重新升温，同时切换训练数据。*

图题是 `log(train/loss) vs log(tokens)`，横轴对应累计 tokens，不是 step。Jellyfish 的绿色曲线逐渐下降到冷却低点，之后 Phoenix 的青绿色曲线在切换时向上跳，再回到一个新平台。

这里有三个位置：

- A：第一次冷却之前的高 LR 水平。
- B：Jellyfish 冷却结束时的低点。
- C：Phoenix 回温后重新稳定的水平。

从图上看，C 低于此前的高 LR 水平 A，但没有立刻低于冷却低点 B。回温后 loss 又上升了，但这次同时换了训练数据，因此 train loss 的比较还混入了文本难度与分布变化，不能把 $C<A$ 单独解释为“回温提高了能力”。

对我更有用的经验是：冷却不是不可逆的训练终点。已有 checkpoint 可以继续升温训练；验收时要联合固定验证集和任务分数，判断过渡后是否稳定，而不是要求 loss 从冷却低点开始永不反弹。

## Phase 5：Starling

把全局 batch 调到 4096 个样本，使用系数为 $10^{-4}$ 的 z-loss，在约 80,000 steps 内把 LR 从 $1.7\times10^{-3}$ 线性降到 $1.7\times10^{-5}$。数据采用约 70% Nemotron-CC、30% 其他高质量来源的混合，包含代码、数学、FLAN 以及 Marin 新处理的数据。[Starling 配置](https://github.com/marin-community/marin/blob/ee163702c5bc71c9bbba3238db84b6ee86e826a7/experiments/tootsie/exp600_tootsie.py)

这说明诊断分支的价值：Raccoon / Spoonbill 不必成为最终模型的直接祖先，它们的发现仍能改变主线后续方案。但这里 batch、LR、数据和正则化都有变化，不能把最终改善全部归功于其中一项。

![Starling 冷却期间 C4 BPB 明显下降，但无法由单图拆分各项贡献。](/notes-assets/marin-8b/starling-cooldown.png)

*Starling 冷却期间 C4 BPB 明显下降，但无法由单图拆分各项贡献。*

图题是 `c4en/bpb vs log(gflops)`，横轴是对数计算量，纵轴是 BPB。冷却时 C4 BPB 明显下降，与继续训练取得进展、降低 LR 改善优化状态的解释相符；数据变化也可能影响这一特定评估集的结果。**这张图不能把各因素的贡献分别量化，也不能仅凭这一段更陡就断言最后的 tokens 学习效率最高。**

## Phase 6：Deeper Starling

Starling 的 80,000 steps 对应：

$$
80{,}000\times4096\times4096\approx1.34\mathrm{T\ tokens}.
$$

之后 Deeper Starling 又继续约 20,000 steps，即约 0.336T tokens，但 LR **固定在 $1.7\times10^{-5}$**。代码中的 `learning_rate=1.7e-5`、`min_lr_ratio=1.0`、`decay=0` 支持这一点，不能因名字里有 deeper 就写成进一步降低 LR。[Deeper Starling 历史配置](https://github.com/marin-community/marin/blob/ee163702c5bc71c9bbba3238db84b6ee86e826a7/experiments/tootsie/exp600_tootsie.py)

前面的 C4 图最右侧棕色段对应这一阶段，下降明显变慢。我的理解是：冷却时同时有继续训练和减少优化波动的收益；LR 停止下降后，后者不再以原来的方式持续贡献。这与 WSD 的 river / hill 分析框架相符，但不能据此证明模型“已经学完”。

决定是否停止训练，更实际的问题是：继续消耗 tokens 和时间，固定验证集及目标任务还能改善多少？预算耗尽、边际收益变小和达到数学最优，是不同的停止理由。

## 最终产物与评估：训练结束后还要核对什么？

模型卡将最终 base checkpoint 标为 `deeper-starling`，约 12.7T tokens。复现时应记录具体 revision，而不是只写可能变化的 `main`。发布的 instruct 版本另外进行了约 5.3B tokens 的 SFT，这部分与上面的 base 预训练量分开记录。[模型检查点与 SFT 记录](https://huggingface.co/marin-community/marin-8b-base#checkpoints)

模型卡给出的最终 base 结果包括 MMLU 5-shot 67.6%、GSM8K 61.3%、MMLU Pro 36.5%。这些是其列明的 LM Eval Harness 评测设置下的结果；跨模型或跨阶段比较时，还要对齐任务版本、few-shot 数量、提示模板和计分方式，不能拿不同协议的分数直接相减。[最终 Base 评估表](https://huggingface.co/marin-community/marin-8b-base#base-model-results)


## 我从这次 8B 学习中整理出的检查顺序

| 看到的现象              | 先检查什么                          | 并没有意味着      |
| ------------------ | ------------------------------ | ----------- |
| Loss 周期性下降、再反弹     | LR 是否同步冷却和恢复；比较点是否处于相同状态       | 模型周期性遗忘     |
| 换数据时 train loss 跳变 | 数据配比、计分方式、固定验证集                | 模型能力一定变差    |
| 个别评估域反向变化          | 文档末尾、格式 token、预处理和逐 token loss | 该领域能力一定退化   |
| 低 LR 时 loss 持续抬升   | 分模块 norm、输出 logits、优化器状态和验证集   | LR 更低就必然更稳定 |
| EMA 与普通权重有差距       | 参数平均实现、评测口径、当前 LR              | 把曲线平滑就有同样效果 |
| 冷却后的曲线变平           | LR 是否停止下降，任务分数的边际改善            | 模型已经达到最优    |

这些是排查入口，不是看到某种形状就自动执行某种修复。每次解释曲线时，我都要补问：还有什么同样合理的解释？哪个额外观测或对照实验能把它们区分开？

## 总结：这次能学到哪些训练模型的技巧？

回顾 Marin 8B，我可以把学到的技巧整理了一下，分别对应训练安排、数据选择、稳定性和实验管理。
**1. 训练预算不确定时，给学习率计划留出继续训练的空间。**

Kestrel 的 WSD-S 用高学习率平台持续训练，再周期性冷却，获得适合评估和使用的 checkpoint；Phoenix 则说明冷却后仍可以回温续训。实际可以学的是：根据“还要继续学多久”和“何时需要交付模型”安排平台与冷却，并保存足够的恢复状态。拿到额外预算后，可以从已有 checkpoint 做续训过渡实验，观察回温后能否稳定，再决定是否扩大投入。[WSD-S 机制](https://arxiv.org/html/2410.05192v2)

**2. 高学习率阶段，同时维护 EMA 权重，增加一个评估视角。**

Ocelot 的普通权重与 EMA 权重之间存在稳定的评估差距。实际训练中，可以对两套权重运行相同的固定验证集，帮助判断高学习率下的参数波动是否掩盖了部分进展。实现时要检查平均方向、初始化、更新频率，以及评估到底加载了哪套权重；还要考虑额外的权重存储和评估开销。EMA 是否值得保留，用验证结果和目标任务表现判断。

**3. Batch、学习率和训练进度单位要一起调整。**

Ocelot 为利用新硬件增大 batch，并相应提高学习率。自己的实验中，换 batch 后需要重新检查吞吐量、稳定性和相同 tokens 下的效果，同时换算 warmup、冷却、评估间隔各自覆盖多少 tokens。否则，原来“每隔一万步评估”的设置，可能已经意味着多等几倍的数据量。Marin 采用的平方根 LR 缩放可以作为候选设置，效果仍要实测。

**4. 用短程退火实验筛数据，并保留原始配比作为对照。**

Microannealing 最值得借鉴的是实验方法：从同一个成熟 checkpoint 出发，给不同候选数据相同的短训练预算，保留一条使用原始数据配比的对照，比较固定验证集和目标任务。筛选时同时关注“领域拟合是否改善”和“任务能力是否改善”；论文、百科等来源带来的低 loss，不自动等于更好的答题能力。小实验筛出的候选，还要检查长程训练中的重复采样和收益变化。[数据筛选实验入口](https://github.com/marin-community/marin/issues/784)

**5. 把最终冷却当作一段需要设计的数据与优化过程。**

Jellyfish 和 Starling 都在降 LR 时调整数据。可以借鉴的做法是：提前为收尾留预算，用经过实验筛选的数据强化目标能力，同时保留足够的主体数据覆盖；到阶段结束时，用固定任务评估是否实现了预期。若想知道收益究竟来自数据还是冷却，可以从相同 checkpoint 分别做“原配比冷却”和“新配比冷却”，让其他条件尽量一致。70/30 是 Marin 的具体选择，自己的配比应由对照结果决定。

**6. 训练失稳时，把观测精细到模块，再选择干预。**

Raccoon / Spoonbill 的经验是：总 loss 缓慢上升时，低 LR 本身并不能保证安全。排查需要结合分模块参数 norm、梯度、优化器统计和输出 logits；发现输出侧异常后，再测试 z-loss 这类对应的干预。日志中最好分开记录 CE、正则项和总 loss。修复的验收条件也要提前写清：越过原故障区间，内部异常得到缓解，固定验证集和目标任务没有明显退化。[诊断实验 #898](https://github.com/marin-community/marin/issues/898)、[后续实验 #916](https://github.com/marin-community/marin/issues/916)

**7. 用固定评估和短分支，降低错误决策的成本。**

保留一套固定验证数据和固定任务协议，并记录每次 LR、数据、batch、代码的变更位置。这样遇到拐点时，才能先检查是否与主动切换对齐。需要诊断时，从异常前的 checkpoint 分出短实验，尽量一次检验一个假说；遇到分域评估异常，再检查文本格式和逐 token 误差。诊断分支的价值，是用较少算力回答问题，并把有效改动带回主线。

**8. 用单位成本带来的收益，决定何时停止训练。**

Deeper Starling 在固定低 LR 下继续训练，但收益变慢。自己的训练可以按固定 tokens 或固定时间窗口，记录验证集和目标任务的增益，再判断继续预训练是否值得。停止标准应结合目标能力、预算和改进幅度；微小变化还要考虑评测波动，不能只追逐最后一点 train loss 的下降。

如果以后有机会自己训练模型，我会把这些技巧落实为一个流程：先确定基线配置、数据和评测协议，用短跑检查稳定性与吞吐；主训练期间记录变更并保存恢复状态，用短分支筛数据和诊断异常；最后安排冷却，比较候选 checkpoint，并根据目标任务的边际收益决定是否继续。
