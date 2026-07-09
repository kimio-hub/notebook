---
title: On-Policy Distillation：让学生在自己的错误上学习
tags:
  - distillation
  - rl
  - llm
---

OPD 的"正典"是 **Agarwal et al., _On-Policy Distillation of Language Models: Learning from Self-Generated Mistakes_, ICLR 2024**。
作者来自 Google DeepMind、Mila、University of Toronto；论文提出的方法叫 **Generalized Knowledge Distillation, GKD**，其中最核心的一种特例就是 **on-policy distillation**。这篇之所以经典，是因为它直接把自回归语言模型的蒸馏问题形式化为"学生在自己生成的轨迹上接受教师反馈"，并系统比较了不同 KL/JSD 损失、不同 on-policy 数据比例，以及和 RL 微调结合的方式。([ICLR 会议记录](https://proceedings.iclr.cc/paper_files/paper/2024/hash/5be69a584901a26c521c2b51e40a4c20-Abstract-Conference.html "On-Policy Distillation of Language Models: Learning from Self-Generated Mistakes"))

---

# 1. 这篇论文一句话在讲什么？

**传统蒸馏是：学生看老师写好的答案来学。**
**OPD 是：学生自己先写答案，老师再针对学生写出来的每一步进行批改。**

对语言模型来说，"每一步"就是每个 token 位置。学生生成一个回答：

$$y = (y_1, y_2, \ldots, y_T)$$

在第 $t$ 步，学生已经写出了前缀：

$$y_{<t} = (y_1, \ldots, y_{t-1})$$

教师模型看到同一个 prompt $x$ 和学生当前前缀 $y_{<t}$，给出下一个 token 的概率分布：

$$p_T(\cdot \mid x, y_{<t})$$

学生也给出自己的概率分布：

$$p_S(\cdot \mid x, y_{<t})$$

然后训练学生，让学生在**自己真实会走到的前缀状态**上，模仿教师的 token 分布。论文摘要里明确说，GKD 不只依赖固定输出序列，而是在学生自生成序列上利用教师反馈训练学生，从而缓解训练和推理时输出分布不匹配的问题。

最简单的类比是：

- **SFT / 离线蒸馏**：你看老师做题的标准答案。

- **RL**：你自己做题，最后只知道对错。

- **OPD**：你自己做题，老师逐步批改你每一步哪里像、哪里不像高手。

Thinking Machines Lab 后来把这点概括成：SFT 是 off-policy + dense，RL 是 on-policy + sparse，OPD 是 on-policy + dense。([Thinking Machines Lab](https://thinkingmachines.ai/blog/on-policy-distillation/ "On-Policy Distillation - Thinking Machines Lab"))

---

# 2. 论文要解决的核心痛点：distribution mismatch

自回归语言模型有一个非常麻烦的问题：**当前 token 的预测依赖之前所有 token。**

传统监督微调或传统 KD 通常在固定数据 $(x, y)$ 上训练。这里的 $y$ 可能是人工答案，也可能是教师模型生成的答案。训练时，学生看到的是"标准前缀"：

$$y_{<t}^{\text{teacher/gold}}$$

但推理时，学生看到的是自己前面生成的内容：

$$y_{<t}^{\text{student}}$$

如果学生前面写错一步，后面的上下文就进入了训练时很少见过的区域。于是错误会连锁放大。这就是论文说的 **train-inference distribution mismatch**：训练时看到的输出序列，和学生部署时自己生成的序列不一致。论文指出，因为自回归模型每一步预测依赖前面的步骤，所以早期错误会级联影响后续生成质量。

举个数学推理例子。

题目：
"12 个苹果，吃掉 5 个，还剩几个？"

老师标准答案可能是：

> 12 - 5 = 7，所以还剩 7 个。

传统蒸馏让学生模仿这个答案。但学生真实推理时可能第一步写成：

> 12 + 5 = 17

这时它已经进入了一个"老师标准答案里没有出现过的错误前缀"。传统 SFT/KD 没教过它：**如果你已经走错到这里，下一步该怎么修正？**

OPD 的做法是：让学生先自己生成。假设学生写出：

> 12 + 5 = 17, so the answer is ...

然后教师在这个错误前缀上给出下一个 token 的分布。教师可能会更倾向生成：

> Wait, this should be subtraction...

这就变成了"从自己的错误状态中学习"。

这也是论文标题里 **Learning from Self-Generated Mistakes** 的含义。

---

# 3. OPD 的核心算法：GKD

论文的方法叫 **Generalized Knowledge Distillation, GKD**。它不是只提出一个固定算法，而是提出一个统一框架：

$$\text{GKD} = \text{固定数据蒸馏} + \text{学生自生成数据蒸馏}$$

论文里的算法非常简单。每一步训练时：

1. 抽一个 prompt $x$。

2. 以概率 $\lambda$，让学生自己生成回答 $y \sim p_S(\cdot \mid x)$。

3. 以概率 $1-\lambda$，从固定数据集里取已有回答 $y$。

4. 对这个 $(x,y)$，逐 token 比较教师和学生的 next-token 分布。

5. 最小化某种 divergence，比如 forward KL、reverse KL 或 JSD。

论文算法 1 里，$\lambda$ 被称为 **student data fraction**，也就是训练 batch 中来自学生自生成输出的比例；$\lambda=1$ 是纯 on-policy，$\lambda=0$ 退化为传统 supervised KD。

核心损失可以写成：

$$L_{\text{GKD}}(\theta) = (1-\lambda)\,\mathbb{E}_{(x,y)\sim \mathcal{D}}\Big[ D(p_T \Vert p_S^\theta)(y \mid x) \Big] + \lambda\,\mathbb{E}_{x\sim X}\,\mathbb{E}_{y\sim p_S(\cdot \mid x)}\Big[ D(p_T \Vert p_S^\theta)(y \mid x) \Big]$$

这里：

$$D(p_T \Vert p_S)(y \mid x) = \frac{1}{|y|} \sum_{t=1}^{|y|} D\Big( p_T(\cdot \mid x, y_{<t}) \,\Big\Vert\, p_S(\cdot \mid x, y_{<t}) \Big)$$

这句话翻译成人话就是：

> 对学生生成或数据集中取出的完整回答 $y$，在每个 token 位置，把"教师看到此前缀后对下一个 token 的概率分布"和"学生看到此前缀后对下一个 token 的概率分布"拉近。

论文特别强调：**不要对学生采样过程本身反向传播。**也就是说，虽然 $y$ 是学生采样出来的，但训练时把这个 $y$ 当成一条固定轨迹，只在轨迹上的每个 token 位置计算 teacher/student 分布差异并更新学生。论文说，这样做更稳定，也更计算高效。

你可以把它理解成下面的伪代码：

```python
for batch in training:
    x = sample_prompts()

    if random() < lambda_on_policy:
        # on-policy: 学生自己生成
        y = student.generate(x)
    else:
        # off-policy: 用固定数据
        y = dataset_outputs[x]

    # 注意：这里 y detach，不对 generate 过程做 policy gradient
    loss = 0
    for t in range(len(y)):
        prefix = y[:t]

        teacher_probs = teacher.next_token_distribution(x, prefix)
        student_probs = student.next_token_distribution(x, prefix)

        loss += divergence(teacher_probs, student_probs)

    loss.backward()
    optimizer.step()
```

这就是 OPD 的工程本质：
**student rollout + teacher token-level grading + supervised-style divergence update。**

---

# 4. 为什么它叫 "on-policy"？

在强化学习里，**policy** 指当前模型的行为策略。对语言模型来说，policy 就是：

$$p_S(y_t \mid x, y_{<t})$$

也就是"在当前上下文下，学生会怎么选下一个 token"。

如果训练数据来自教师生成的答案，那么数据分布是教师的：

$$y \sim p_T(\cdot \mid x)$$

这就是 off-policy，因为学生训练时看到的是老师会走到的状态，不是自己会走到的状态。

如果训练数据来自学生当前模型自己生成的答案：

$$y \sim p_S(\cdot \mid x)$$

这就是 on-policy，因为训练发生在学生自己当前策略真实访问到的状态分布上。论文明确说，on-policy KD 使用学生生成的输出序列，学生会在自己可能生成的序列上接受教师的 token-specific feedback。

这点非常关键。

OPD 不是简单地"用老师打标签"。它真正有价值的地方是：

> 教师不是在自己的舒适区教学生，而是在学生真实犯错的地方教学生。

---

# 5. Forward KL、Reverse KL、JSD：三种损失怎么理解？

论文的另一个重要贡献是：它不只说"用 KL 就完了"，而是系统讨论不同 divergence 的效果。

## 5.1 Forward KL

Forward KL 通常写成：

$$D_{\mathrm{KL}}(p_T \Vert p_S)$$

它的直觉是：

> 老师认为有可能的 token，学生也都要覆盖到。

所以 forward KL 是 **mode-covering** 的。它鼓励学生覆盖教师分布的多个可能模式。

优点：
生成更丰富，不容易过早塌缩。

缺点：
如果学生容量远小于老师，学生可能为了覆盖老师的多个模式，把概率质量分散到一些低质量 token 上。论文指出，当学生容量远低于教师时，forward KL 可能让学生给教师低概率 token 分配概率，从而导致幻觉或低质量生成。

## 5.2 Reverse KL

Reverse KL 写成：

$$D_{\mathrm{KL}}(p_S \Vert p_T)$$

它的直觉是：

> 学生只要集中模仿老师最认可的高概率选择就行。

所以 reverse KL 是 **mode-seeking** 的。它更像"选老师最确定的路线"。

优点：
更保守，更集中，低质量 token 更少。

缺点：
多样性下降。论文指出，mode-seeking divergence 会优先关注教师高概率 token，可以避免低质量生成，但代价是生成多样性下降。

## 5.3 JSD

JSD 可以看成 forward KL 和 reverse KL 之间的折中。论文用了多个 $\mathrm{JSD}(\beta)$，例如 JSD(0.1)、JSD(0.5)、JSD(0.9)，并指出不同任务的最佳 divergence 不一样。

你可以这样记：

|损失|性格|适合场景|
|---|---|---|
|Forward KL|覆盖老师的各种可能性|需要多样性、学生容量足够|
|Reverse KL|追随老师最强模式|需要稳、准、少幻觉|
|JSD|折中|不确定时可调|

论文在 XSum 上还观察到，从 forward KL 过渡到 reverse KL，生成多样性降低，但高温采样时某些 mode-seeking divergence 质量更好。

---

# 6. 这篇论文的实验设计

论文实验主要用 T5 系列模型。教师是大模型 **T5-XL，约 3B 参数**；学生是更小的 T5-small、T5-base、T5-large，分别约 77M、250M、800M 参数，也就是比教师小约 38 倍、12 倍、3.8 倍。任务包括摘要、翻译、算术推理，以及指令泛化。

论文比较了几类方法：

1. **Supervised FT**：普通监督微调。

2. **Supervised KD**：在固定数据上模仿教师 token 分布。

3. **SeqKD**：用教师生成完整答案，再让学生模仿这些答案。

4. **GKD / OPD**：学生自己生成，教师逐 token 反馈。

5. **Mixed GKD**：部分学生自生成数据，部分固定数据。

整体结论非常清楚：**on-policy GKD 通常优于传统 KD 基线。**

论文在 Figure 1 中报告，on-policy GKD 相比基线 KD 方法带来的性能提升，在摘要、翻译、算术推理任务上的平均相对增益分别约为 **2.1×、1.7×、1.9×**；在任务无关蒸馏上，GKD 也带来了 MMLU 和 BBH 的绝对准确率提升。

更细的几个实验结论：

- 在 **XSum 摘要**任务上，on-policy GKD 变体通常优于基线；而且 GKD 的数据效率更高。论文甚至提到，用 5% 子采样数据做 on-policy GKD、且不用 ground-truth summaries，也能超过用完整训练集的 supervised KD 和 ImitKD。

- 在 **WMT 英德翻译**上，论文观察到只使用学生自生成输出样本的 GKD 变体优于其他 GKD 变体。

- 在 **GSM8K 算术推理**上，随着学生生成数据比例超过 25%，性能通常会提升。

- 在 **RL + GKD** 组合上，论文把 on-policy GKD 和 RLAIF 结合，用文本蕴含分数作为 reward；结果显示，相比 RLEF 基线，on-policy GKD + RL 有更高 ROUGE-2，同时生成比教师更事实一致的摘要。

---

# 7. 这篇论文真正教会我们的 5 个原则

## 原则一：蒸馏不该只在老师的轨迹上发生

传统 KD 的问题不是"教师信号不好"，而是**教师信号出现在错误的数据分布上**。

老师生成的答案很好，但学生推理时不一定会走到老师那些完美前缀上。学生最需要被教的，恰恰是自己犯错之后的状态。OPD 把监督信号搬到了学生自己会访问的状态上。

## 原则二：OPD 不是 RL，但吸收了 RL 最重要的优点

RL 的优点是 on-policy：模型从自己的行为后果中学习。
蒸馏的优点是 dense：每个 token 都有监督信号。
OPD 把这两个结合起来。

但 OPD 又不是标准 RL。它没有对采样过程做 policy gradient，而是在学生轨迹上做 supervised-style 的 divergence minimization。论文也强调，不对学生采样过程反向传播，这让训练更稳定、更便宜。

所以你可以记成：

$$\text{OPD} \approx \text{on-policy data collection} + \text{supervised distillation loss}$$

## 原则三：学生不能太烂

论文特别提醒，他们不是从随机初始化学生开始，而是从已经经过 supervised fine-tuning、能够生成"足够质量序列"的学生开始。原因很简单：如果学生生成的前缀完全乱套，教师在这些垃圾前缀上的反馈价值也会下降。

这在工程上非常重要。

OPD 常见训练流程应该是：

$$\text{pretrain} \rightarrow \text{SFT / off-policy distill warmup} \rightarrow \text{OPD}$$

而不是：

$$\text{random student} \rightarrow \text{OPD}$$

## 原则四：loss 选错会影响质量和多样性

Forward KL、reverse KL、JSD 不是数学洁癖，而是会改变模型行为。

如果你希望学生保留多样性，forward KL 或较 forward 的 JSD 可能更合适。
如果你希望学生更稳、更贴近老师的主模式，reverse KL 或较 reverse 的 JSD 更合适。
论文结论是：最优 divergence 具有任务依赖性，不能一刀切。

## 原则五：OPD 需要教师概率，不只是教师答案

这点经常被忽略。

如果你只有教师生成的文本答案，那更像 SeqKD 或 SFT。
OPD 的强监督来自：

$$p_T(\cdot \mid x, y_{<t})$$

也就是教师在学生每个前缀上的 token 分布。论文明确说，学生在自生成输出序列的错误 token 上接收来自教师 logits 的 token-specific feedback。

所以，理想 OPD 需要教师模型能返回 logits 或 logprobs。只有黑盒文本 API 时，也可以做近似变体，但那已经不是这篇论文里最标准的 GKD 形式。

---

# 8. 这篇论文的局限性

第一，**成本不是零**。OPD 要让学生采样，还要让教师在学生轨迹上计算概率。论文在 GSM8K 上报告，相比从固定输出数据采样，学生采样带来的计算开销大约是 1.8×、2×、2.2×，具体取决于学生/教师尺寸比。

第二，**教师错，学生也会学错**。蒸馏本质上是在模仿教师分布。如果教师在某些状态下给出糟糕分布，学生也会被带偏。论文在摘要任务中也指出，单纯蒸馏不一定改善事实一致性，因为大模型本身也会幻觉；这也是他们尝试 RL + GKD 的原因。

第三，**学生初始能力要够**。如果学生的策略分布离教师太远，OPD 会在很多低质量状态上训练，效率会下降。论文实验从 SFT 后的学生开始，这一点不是偶然，而是方法成立的重要前提。

第四，**divergence 和 $\lambda$ 都要调**。论文把 $\lambda$ 作为 student data fraction，并测试纯 on-policy、mixed、supervised 三类设置；不同任务中，最佳 divergence 和最佳 on-policy 比例并不完全一样。

---

# 9. 怎么检验自己真的掌握了？

能回答下面 7 个问题，就基本掌握了。

**问题 1：OPD 和传统 KD 最大区别是什么？**
传统 KD 在固定数据或教师轨迹上训练；OPD 在学生当前策略生成的轨迹上训练。

**问题 2：为什么 OPD 能缓解 exposure bias？**
因为训练时的前缀分布更接近推理时学生自己会遇到的前缀分布。

**问题 3：OPD 的监督信号来自哪里？**
来自教师在学生每个自生成前缀上的 next-token 概率分布。

**问题 4：OPD 是不是 RL？**
不是标准 RL。它用 on-policy 采样，但更新方式通常是 supervised-style divergence loss，而不是对最终 reward 做 policy gradient。

**问题 5：为什么不对采样过程反传？**
因为把学生采样轨迹当作固定数据可以让训练更稳定、更高效；论文明确采用这种做法。

**问题 6：$\lambda$ 是什么？**
$\lambda$ 是学生自生成数据比例。
$\lambda=0$：传统 supervised KD。
$\lambda=1$：纯 on-policy KD。
中间值：mixed GKD。

**问题 7：Forward KL 和 Reverse KL 怎么选？**
Forward KL 更覆盖、多样；reverse KL 更聚焦、保守；JSD 折中。论文结论是任务依赖。

---

# 10. 最终浓缩版

这篇论文的核心可以压缩成一句公式和一句话。

公式：

$$L_{\text{OPD}} = \mathbb{E}_{x}\,\mathbb{E}_{y\sim p_S(\cdot\mid x)}\left[ \frac{1}{|y|} \sum_t D\Big( p_T(\cdot\mid x,y_{<t}) \,\Big\Vert\, p_S(\cdot\mid x,y_{<t}) \Big) \right]$$

一句话：

> **让学生自己生成答案，然后让教师在学生真实走到的每个前缀状态上给 token-level 分布反馈；这样既保留了 on-policy 的相关性，又保留了蒸馏的密集监督。**

这就是 _On-Policy Distillation of Language Models: Learning from Self-Generated Mistakes_ 这篇论文最值得掌握的东西。

---

# 附：DeepSeek-V4 把 OPD 用到了什么程度？

> 本节整理自社区二手资料（技术报告转述与解读文章），细节未逐一核对原始报告，仅作延伸阅读。

据社区解读，DeepSeek-V4（2026）在后训练阶段用 OPD 大规模替代了此前的混合 RL 路线，大致流程是：

1. **预训练**：在大规模多样化数据上预训练（>32T tokens）；
2. **领域专家训练**：数学、代码、agent、指令追随等方向分别训练专家模型，每个专家先 SFT，再用 GRPO 这类 RL 方法强化到极致；
3. **OPD 融合专家**：统一的学生模型自己 rollout，在自己走到的前缀分布上对照各领域教师做逐 token 对齐（以 reverse KL 为主），把多个专家的能力融进一个模型。

之所以用 OPD 替代 mixed RL，理由和本文第 7 节的原则一脉相承：

- 多任务的 reward 难以设计和调和，OPD 直接绕开 reward 设计；
- supervised-style 的稠密监督比高方差的 RL 目标更稳定、更易收敛；
- 多个单领域教师可以在学生自己的轨迹上逐 token 融合，而不是做单一 reward 下的混合最优化。

工程上，由于教师又多又大（10+ 个领域专家），据称 V4 采用了教师权重分布式存储按需加载、只缓存教师 hidden states、mini-batch 内按教师分组排序等手段来控制显存与 IO 开销。

对照阅读：[[DeepSeek-V3]]（上一代的架构与训练笔记）。
