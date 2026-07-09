---
title: Paper-List
---

可以。你现在最适合读的不是“泛泛的大模型论文列表”，而是围绕你已经在做的 **SFT → GRPO → 数学推理 RL**，建立一个有主线的阅读体系。我的建议是：第一批不要超过 15 篇，分成 **后训练算法、reasoning RL、verifier/reward、评测、多模态** 五条线。

我会把论文分成三类：

```text
A. 必读：和你当前实验直接相关，应该精读
B. 拓展：帮助你理解前沿趋势，适合泛读
C. Taste 训练：重点看它的问题定义、实验设计、失败分析
```

另外，有些 2025–2026 的前沿工作还没进入正式顶会周期，但已经是社区研究方向的核心，我会标成“高影响预印本/技术报告”，不把它们伪装成已发表顶会。

---

# 一、第一批最推荐你读的 12 篇

## 1. Direct Preference Optimization: Your Language Model is Secretly a Reward Model

**Rafailov et al., 2023，DPO，基础必读。**

这篇虽然不是最新，但它是你理解 DPO、SimPO、ORPO、KTO 等一整条 preference optimization 线的起点。DPO 的核心贡献是把传统 RLHF 里的“训练 reward model → 用 RL 优化 policy”改写成一个直接的分类式目标；论文摘要里明确说，它通过 reward model 的重新参数化，使标准 RLHF 问题可以用简单 classification loss 来解，并且不需要在 fine-tuning 时从 LM 采样。

**你要读出什么 taste：**

```text
一个好算法不一定是“更复杂”；
DPO 的 taste 是：把复杂 pipeline 中的中间变量消掉，让目标函数变简单。
```

重点读：

```text
RLHF objective
reward-policy reparameterization
DPO loss
DPO vs PPO 的实验对比
```

你读完后要能回答：

```text
为什么 DPO 不显式训练 reward model？
DPO 的隐式 reward 是什么？
DPO 为什么可能比 PPO 稳？
DPO 的局限是什么？
```

---

## 2. KTO: Model Alignment as Prospect Theoretic Optimization

**Ethayarajh et al., ICML 2024。**

KTO 很适合培养 taste，因为它不是简单说“我比 DPO 多一个 trick”，而是从人类效用建模出发，用 prospect theory 解释为什么某些 alignment loss 有效。论文说 DPO 等目标可以看作一类 human-aware losses，并提出 KTO；它只需要“这个输出是否 desirable”的二值信号，不一定需要成对 preference。

**你要读出什么 taste：**

```text
好论文可以从一个更高层的建模视角重新解释已有方法；
不是只追 SOTA，而是问：已有 loss 背后的归纳偏置是什么？
```

重点读：

```text
HALO 框架
KTO loss
desirable / undesirable binary feedback
和 DPO / SFT 的关系
```

你读完后要能回答：

```text
pairwise preference 和 binary feedback 有什么区别？
为什么不同 alignment loss 没有绝对最优？
什么叫 loss 的 inductive bias？
```

---

## 3. SimPO: Simple Preference Optimization with a Reference-Free Reward

**Meng, Xia, Chen, NeurIPS 2024。**

SimPO 是 DPO 后非常值得读的一篇。它把隐式 reward 改成 sequence 的 average log probability，并且去掉 reference model，还加了 target reward margin。论文摘要明确说，SimPO 使用平均 log probability 作为隐式 reward，消除 reference model，使训练更省算力和显存，并在 AlpacaEval 2、MT-Bench、Arena-Hard 等评测上和 DPO 变体比较。

**你要读出什么 taste：**

```text
一个方法的改进点应该能对应到明确问题：
DPO 的 reference model 成本、长度偏置、sequence probability 和生成行为不匹配。
```

重点读：

```text
average log probability reward
length normalization
target margin
DPO vs SimPO 的 ablation
```

你读完后要能回答：

```text
为什么 sequence-level log probability 要做长度归一？
为什么 reference-free 可能有优势？
SimPO 会不会引入新的偏差？
```

这篇和你的实验也很相关。你之前 RFT 只保留正确轨迹，丢掉了错误轨迹；SimPO/DPO 这类方法可以把同题的 correct completion 和 incorrect completion 构造成 preference pair。

---

## 4. DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models

**DeepSeek-AI, 2024，高影响技术报告；GRPO 线必读。**

这篇与你当前实验最直接相关，因为它引入了 **GRPO**。论文摘要说 DeepSeekMath 7B 通过 120B 数学相关 token 的继续预训练和 GRPO 提升数学推理能力；GRPO 被描述为 PPO 的变体，能增强数学推理，同时优化 PPO 的内存使用。

**你要读出什么 taste：**

```text
reasoning 能力不是单靠 RL 来的；
数据选择、继续预训练、SFT、RL、采样评测共同构成系统。
```

重点读：

```text
math data pipeline
continued pretraining
GRPO objective
PPO vs GRPO
math benchmark setup
```

你读完后要能回答：

```text
GRPO 为什么不需要 value model？
group-relative advantage 的信号来自哪里？
为什么同一题多采样很关键？
为什么 all-correct / all-wrong group 会没有学习信号？
```

这篇应该和你自己的 zero group 问题一起读。你读的时候可以在旁边写：

```text
论文里的 GRPO 假设 group 内 reward 有区分度；
我的实验里大量 group 全对，说明这个假设在 SFT 后的 GSM8K 上失效。
```

---

## 5. DAPO: An Open-Source LLM Reinforcement Learning System at Scale

**Yu et al., 2025，高影响预印本；reasoning RL 工程与算法前沿。**

DAPO 非常适合你当前阶段，因为它直接处理大规模 LLM RL 的训练细节和可复现问题。摘要说，DAPO 提出 Decoupled Clip 和 Dynamic Sampling Policy Optimization，并开源了系统、代码和数据；它特别强调很多 state-of-the-art reasoning LLM 的关键 RL 细节没有公开，所以社区难以复现。

**你要读出什么 taste：**

```text
前沿 LLM RL 不是只改一个 loss；
采样、过滤、clip、batch 构造、数据难度、系统实现都会决定结果。
```

重点读：

```text
dynamic sampling
overlong filtering
decoupled clipping
训练系统设计
AIME / math RL 结果
```

你读完后要能回答：

```text
为什么 dynamic sampling 对 RLVR 很关键？
为什么有效 batch 比 nominal batch 更重要？
为什么采样策略本身就是算法？
```

这篇尤其对应你之前遇到的 zero group。它会帮你把“重采样”从一个工程补丁，提升为一个正式算法设计问题。

---

## 6. DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning

**DeepSeek-AI, 2025，高影响技术报告；reasoning model 前沿必读。**

这篇是理解 2025 reasoning model 热潮的关键报告。摘要说 DeepSeek-R1-Zero 直接用大规模 RL、没有先做 SFT，就涌现出很强的 reasoning 行为；但也有可读性差、语言混杂等问题，因此 DeepSeek-R1 加入 cold-start data 和 multi-stage training。

**你要读出什么 taste：**

```text
前沿不是“RL 能不能提升 accuracy”这么简单；
而是 RL 是否能诱导 self-reflection、long CoT、verification 等行为。
```

重点读：

```text
R1-Zero vs R1
cold start data
rule-based reward
multi-stage RL
distillation
failure cases：readability / language mixing
```

你读完后要能回答：

```text
为什么 pure RL 会涌现 reasoning，但也会带来格式和语言问题？
为什么 cold-start SFT 仍然重要？
为什么 rule-based reward 在数学/代码任务上特别有价值？
```

---

## 7. s1: Simple Test-Time Scaling

**Muennighoff et al., 2025，高影响预印本；test-time scaling 线。**

这篇适合帮助你理解：提升 reasoning 不只有训练时 RL，一部分能力也来自 test-time compute。论文摘要说，s1 用 1000 个高质量问题和 reasoning traces 做 SFT，并通过 budget forcing 控制测试时计算，例如在模型想结束时追加 “Wait” 让它继续检查，从而提高 AIME/MATH 表现。

**你要读出什么 taste：**

```text
有些提升来自训练算法，有些提升来自推理时计算预算；
不要把 benchmark 提升都误认为模型参数里“学会了”。
```

重点读：

```text
s1K 数据筛选标准：difficulty / diversity / quality
budget forcing
test-time scaling
和 RL reasoning model 的关系
```

你读完后要能回答：

```text
训练时 scaling 和测试时 scaling 有什么区别？
为什么小数据也可能有效？
budget forcing 是能力提升还是评测策略提升？
```

---

## 8. Let’s Verify Step by Step

**Lightman et al., 2023；PRM / process supervision 必读。**

这篇稍早，但对你做数学 reasoning RL 很关键。论文比较 outcome supervision 和 process supervision，发现对 MATH 这类多步推理任务，过程监督显著优于只看最终答案的结果监督，并发布了 PRM800K step-level human feedback 数据。

**你要读出什么 taste：**

```text
最终答案 reward 太稀疏；
对于多步推理，过程级信号可能比 outcome-level 信号更有价值。
```

重点读：

```text
outcome reward model vs process reward model
PRM800K
active learning
MATH evaluation
```

你读完后要能回答：

```text
为什么 final-answer reward 不足？
PRM 的标注成本为什么高？
PRM 和 verifier 有什么关系？
过程监督会不会奖励“看起来正确”的推理？
```

这篇和你的实验关系也非常直接：你现在的 reward 主要是最终答案正确与否，所以一旦 group 全对，优势函数就没信号；PRM/process reward 是另一种给全对样本内部制造差异的方式。

---

## 9. RewardBench: Evaluating Reward Models for Language Modeling

**Lambert et al., 2024；reward model 评测必读。**

做 RLHF/RLVR/reward model，不能只学训练方法，也要学怎么评测 reward。RewardBench 的摘要说，它构造了 prompt-chosen-rejected trios，覆盖 chat、reasoning、safety，并专门设计有细微但可验证差异的 comparison data，用来评测各种 reward model，包括显式 RM 和 DPO 的隐式 reward。

**你要读出什么 taste：**

```text
reward model 不是“训练出来就能用”；
reward 自身也需要 benchmark、OOD 测试和失败分析。
```

重点读：

```text
benchmark construction
chosen/rejected pair design
reasoning / safety / chat subsets
reward model failure modes
```

你读完后要能回答：

```text
一个 reward model 会在哪些类型的问题上失效？
为什么 reward benchmark 要有 subtle but verifiable differences？
为什么 reward model 评测和 policy 评测不是一回事？
```

---

## 10. The N+ Implementation Details of RLHF with PPO

**Huang et al., 2024；RLHF 复现与工程 taste 必读。**

这篇不是炫新算法，而是复现 RLHF with PPO 的关键实现细节。摘要说它从零构建 RLHF pipeline，枚举 20 多个关键实现细节，并复现 OpenAI TL;DR summarization work 中的 scaling behavior，还开源模型和代码。

**你要读出什么 taste：**

```text
好研究不只是提出新 loss；
能把别人复现不出来的 pipeline 拆成可检查的实现细节，也是重要贡献。
```

重点读：

```text
PPO implementation details
reward normalization
KL control
sampling settings
batching
reproducibility
```

你读完后要能回答：

```text
为什么 RLHF 论文的 implementation details 会决定结果？
哪些细节会导致训练不稳定？
为什么复现类论文也有学术价值？
```

这篇对你现在非常重要。你已经在记录实验 log、zero group rate、effective groups、entropy、skipped group rate，这正是“实现细节决定研究质量”的方向。

---

## 11. Chatbot Arena: An Open Platform for Evaluating LLMs by Human Preference

**Chiang et al., 2024；LLM 评测 taste 必读。**

这篇帮助你理解为什么单一 benchmark accuracy 不够。论文摘要说 Chatbot Arena 用 crowdsourced pairwise comparison 来评测 LLM，并收集了超过 240K votes；它分析了 crowdsourced questions 是否多样、是否能区分模型，以及 crowdsourced human votes 是否和 expert raters 一致。

**你要读出什么 taste：**

```text
评测不是找一个数字；
评测本身是一个统计建模问题，包括数据来源、偏差、区分度和人类一致性。
```

重点读：

```text
pairwise comparison
Elo / Bradley-Terry-style ranking
human preference agreement
crowdsourced evaluation bias
```

你读完后要能回答：

```text
为什么 pairwise human preference 比单点打分稳？
为什么排行榜也可能被过拟合？
为什么开放式任务很难用 accuracy 评测？
```

---

## 12. MMMU: A Massive Multi-discipline Multimodal Understanding and Reasoning Benchmark for Expert AGI

**Yue et al., CVPR 2024 Oral。**

如果你对多模态感兴趣，这篇是多模态 reasoning benchmark 的重要入口。论文摘要说 MMMU 有 11.5K 个来自大学考试、测验和教材的多模态问题，覆盖艺术设计、商业、科学、医学、人文社科、工程技术等六大领域，并包含图表、表格、乐谱、化学结构等多种图像类型；它被标为 CVPR 2024 Oral。

**你要读出什么 taste：**

```text
好的 benchmark 不只是“题多”；
它要测试现有模型真正薄弱的组合能力：感知 + 专业知识 + deliberate reasoning。
```

重点读：

```text
benchmark construction
discipline coverage
image type taxonomy
human / closed-source / open-source model comparison
failure cases
```

你读完后要能回答：

```text
多模态推理为什么不是 OCR + LLM 就够？
为什么 benchmark 的数据来源和题型分布很重要？
什么叫 expert-level multimodal reasoning？
```

---

# 二、第二批拓展论文：了解更广的前沿

## 13. Cambrian-1: A Fully Open, Vision-Centric Exploration of Multimodal LLMs

**Tong et al., NeurIPS 2024 Oral。**

Cambrian-1 是多模态方向很值得读的一篇“cookbook 型论文”。它系统研究 vision encoder、视觉表征、connector、数据配比和 benchmark；论文摘要说它测试了 20 多种 vision encoders，提出 CV-Bench 和 Spatial Vision Aggregator，并开源权重、代码、工具、数据和训练评测 recipes。

**为什么值得读：**

```text
它不是只报一个 SOTA；
它像一本多模态实验设计手册，告诉你哪些设计选择真的重要。
```

适合你重点学习：

```text
vision encoder 如何影响 MLLM
connector 为什么重要
benchmark 为什么会误导
数据源比例如何影响结果
```

---

## 14. LLaVA-OneVision: Easy Visual Task Transfer

**Li et al., 2024；open LMM 方向重要工作。**

LLaVA-OneVision 关注单图、多图、视频任务之间的迁移。摘要说它是一个 open large multimodal model family，能够在 single-image、multi-image、video 三类场景中同时推进 open LMM 的能力边界，并展示从图像到视频的任务迁移。

**为什么值得读：**

```text
多模态前沿已经不只是 image-text；
而是在追求 single-image / multi-image / video 的统一建模和迁移。
```

你读的时候重点看：

```text
统一模型如何处理不同视觉场景
图像任务怎么迁移到视频任务
数据和模型设计的 trade-off
```

---

## 15. Qwen2-VL: Enhancing Vision-Language Model’s Perception of the World at Any Resolution

**Wang et al., 2024；多模态模型工程与 scaling 方向。**

Qwen2-VL 和你使用 Qwen 系列模型也有关系。摘要说它提出 Naive Dynamic Resolution，让模型能把不同分辨率图像动态处理成不同数量的 visual tokens，并使用 Multimodal Rotary Position Embedding 来融合文本、图像、视频的位置编码，还探索了 LVLM scaling laws。

**为什么值得读：**

```text
它能帮你理解多模态模型不是简单“ViT + LLM”；
图像分辨率、visual tokens、位置编码、视频统一处理都会影响能力。
```

你读的时候重点看：

```text
dynamic resolution
M-RoPE
image/video unified processing
scaling law for LVLM
```

---

# 三、按照你的方向，我建议这样排序

你现在的研究主线是：

```text
LLM 后训练 / reasoning RL / 数学推理 / 未来扩展到多模态 reasoning
```

所以不要先平均读所有方向。我建议阅读顺序如下。

## 第 1 组：preference optimization 基础

先读：

```text
DPO
KTO
SimPO
```

你要形成的能力是：

```text
看到一个 alignment loss，能说清楚：
它的 implicit reward 是什么？
它是否需要 reference model？
它用 pairwise preference 还是 binary feedback？
它可能有什么长度偏置或过优化问题？
```

读完这三篇，你再看 ORPO、IPO、CPO、DPOP、BPO 等变体时，就不会被名字淹没。

---

## 第 2 组：GRPO / reasoning RL 主线

再读：

```text
DeepSeekMath
DeepSeek-R1
DAPO
s1
```

你要形成的能力是：

```text
区分训练时 RL、test-time scaling、distillation、cold-start SFT 的作用。
```

这里最重要的是不要把所有提升都归因于“RL 很强”。你要问：

```text
数据是否更好？
采样是否更多？
reward 是否更可验证？
是否用了 cold-start traces？
是否用了 distillation？
是否只是 test-time compute 更大？
```

这就是学术 taste。

---

## 第 3 组：reward / verifier / evaluation

然后读：

```text
Let’s Verify Step by Step
RewardBench
The N+ Implementation Details of RLHF with PPO
Chatbot Arena
Arena-Hard / BenchBuilder
```

这里你要形成的能力是：

```text
不只问模型准确率涨没涨；
还要问 reward 是否可信、评测是否可信、实现是否可复现。
```

这对你尤其重要。你现在做的数学 RL 已经遇到：

```text
zero group
all-correct saturation
effective groups 太少
reward 方差不足
单次 test[:500] 波动
```

这些都不是单纯 “accuracy” 能解释的。

---

## 第 4 组：多模态扩展

最后读：

```text
MMMU
Cambrian-1
LLaVA-OneVision
Qwen2-VL
```

你要形成的能力是：

```text
理解多模态 reasoning 不是把图片输入 LLM；
而是视觉表征、connector、分辨率、数据配比、评测构造共同决定能力。
```

未来你可以把你现在的数学 RL 经验迁移到：

```text
MathVista
geometry reasoning
chart QA
scientific figure QA
multimodal verifier
vision-language GRPO / RLVR
```

---

# 四、给你一个 4 周阅读计划

## Week 1：读懂 preference optimization

读：

```text
DPO
KTO
SimPO
```

每篇写 1 页笔记，必须回答：

```text
1. 它优化什么 objective？
2. 它和 SFT / PPO / DPO 的差异是什么？
3. 它的 reward 是显式还是隐式？
4. 它最可能在哪些情况下失败？
5. 我能不能用它改造自己的 correct/incorrect 数学样本？
```

这一周的产出：

```text
一张表：DPO / KTO / SimPO / GRPO 的 objective 对比。
```

---

## Week 2：读懂 reasoning RL

读：

```text
DeepSeekMath
DeepSeek-R1
DAPO
s1
```

每篇重点看：

```text
训练数据
reward
采样方式
RL objective
评测设置
ablation
失败现象
```

这一周的产出：

```text
写一份 2 页 memo：
“我的 zero group 问题在 DeepSeekMath / DAPO / R1 框架下应该如何解释？”
```

---

## Week 3：读懂 reward 和 eval

读：

```text
Let’s Verify Step by Step
RewardBench
The N+ Implementation Details of RLHF with PPO
Chatbot Arena
```

这一周你要培养的 taste 是：

```text
真正的研究者不会只相信 benchmark accuracy；
他会检查 reward、数据、采样、实现和评测统计。
```

这一周的产出：

```text
给你的当前 GRPO 实验新增一套诊断指标设计：
reward_std
zero_advantage_group_rate
mixed_group_rate
all_correct_group_rate
entropy
KL
length
format
per-topic accuracy
multi-seed variance
```

---

## Week 4：读多模态 reasoning

读：

```text
MMMU
Cambrian-1
LLaVA-OneVision
Qwen2-VL
```

这一周的产出：

```text
写一个未来课题草案：
“如何把 rule-verifiable reasoning RL 从文本数学题扩展到图表/几何/多模态数学题？”
```

这个草案不需要马上做，但它能帮助你建立科研方向感。

---

# 五、如何通过读论文培养 academic taste

你以后每读一篇论文，不要先问“它涨了多少点”，而是问下面 8 个问题：

```text
1. 这篇论文的问题定义是否重要？
2. 它有没有指出旧方法的具体失败模式？
3. 它的核心改动是 loss、数据、采样、模型结构、系统，还是评测？
4. 它的改动是否有清楚的机制解释？
5. 它的实验是否排除了简单替代解释？
6. 它有没有做关键 ablation？
7. 它有没有讨论失败案例和局限？
8. 我能不能把它的思想迁移到自己的实验？
```

好 taste 不是“知道很多论文名字”，而是能判断：

```text
这篇论文是真的解决了一个核心瓶颈，
还是只是换 benchmark / 换模型 / 换 prompt 后报了一个好看的数字。
```

---

# 六、结合你当前实验，最值得优先精读的 5 篇

如果你最近时间有限，我建议先读这五篇：

```text
1. DeepSeekMath
2. DAPO
3. SimPO
4. Let’s Verify Step by Step
5. The N+ Implementation Details of RLHF with PPO
```

原因是它们分别对应你现在最关键的五个问题：

```text
DeepSeekMath：GRPO 的原始动机和数学 RL 框架
DAPO：dynamic sampling / 有效 batch / RL 细节
SimPO：如何用 correct vs incorrect 构造 preference learning
Let’s Verify Step by Step：final answer reward 为什么不够
N+ RLHF PPO：为什么实现细节决定 RLHF 是否能复现
```

读完这五篇，你对自己当前实验的理解会明显提升：你会从“我在调 GRPO 参数”，变成“我在研究可验证 reward 下，如何构造有效 learning signal、避免 group saturation，并稳定提升 reasoning policy”。


可以。这一批我建议你不要再按“经典必读”标准选，而要按 **“前沿问题正在往哪里移动”** 来选。2026 年还处在论文周期中期，很多最前沿工作还只是 arXiv / 技术报告，未必已经完成 NeurIPS/EMNLP/COLM 等最终接收；所以我会明确标注：

```text
[顶会确认]：我能看到接收信息的
[高价值预印本]：2026 最新、很前沿，但还不能当成已接收顶会
[系统/技术报告]：不是传统顶会论文，但能代表工业界或社区前沿趋势
```

结合你正在做的 **Qwen 7B + SFT + GRPO + 数学推理 RL**，我最推荐你读下面这批。

---

# 第一组：RLVR / GRPO 的 2026 前沿

这一组和你当前实验最直接相关。你现在遇到的 zero group、all-correct saturation、entropy 下降、本质上都属于 **RLVR 中 learning signal 稀疏、探索不足、优势估计退化** 的问题。

## 1. Temperature as a Meta-Policy: Adaptive Temperature in LLM Reinforcement Learning

**Haoran Dang et al., 2026，[顶会确认：ICLR 2026]**

这篇非常适合你读，因为它直接把 generation temperature 从一个手调超参，提升成一个可以学习的 meta-policy。论文认为固定 temperature 或人工 schedule 不能适应 RL 训练过程中 exploration/exploitation 的动态变化，于是提出 TAMPO：内层用 GRPO 等方法更新 LLM policy，外层学习选择 temperature，使高 advantage trajectory 更容易出现。

**为什么对你重要：**

你之前计划把 temperature 从 `0.9` 调到 `1.0`，并增加 zero-advantage group 的 resampling。TAMPO 的 taste 在于：它不是说“温度调高一点”，而是问：

```text
temperature 是否应该成为训练过程中可适应的策略变量？
什么时候需要更探索？
什么时候需要收敛？
temperature 改变的是样本分布，而不仅是生成风格。
```

你读这篇要重点看：

```text
temperature 如何进入采样分布
meta-policy 怎么更新
它和 GRPO 的接口在哪里
高温带来的 exploration 与低质量样本之间怎么平衡
```

---

## 2. From Reasoning Chains to Verifiable Subproblems: Curriculum Reinforcement Learning Enables Credit Assignment for LLM Reasoning

**Xitai Jiang et al., 2026，[高价值预印本]**

这篇是我觉得你最应该精读的 2026 前沿之一。它提出 SCRL，把完整 reasoning chain 拆成可验证 subproblems，用子问题级别的 curriculum RL 改善 credit assignment。论文指出 outcome-only RLVR 在难题上低效，因为最终答案正确 rollout 很少，失败尝试中的 partial progress 没有学习信号；SCRL 用 subproblem-level normalization，把优势信号分配给对应的答案片段，在多个数学 benchmark 上超过 GRPO。

**为什么对你重要：**

你现在的 zero group 主要来自 easy prompts 的 all-correct；但如果你之后换到更难数据，又会遇到另一种 dead zone：

```text
all-wrong group 太多
final answer reward 稀疏
模型有部分推理进展，但最终答案错，reward=0
```

SCRL 正好处理这个问题。

它的 taste 是：

```text
不是简单加 process reward；
而是把原问题结构化成可验证子问题，让 partial progress 也能训练。
```

你读的时候重点想：

```text
我能不能把 GSM8K/MATH 题自动拆成 intermediate equations？
能不能把“最终答案 verifier”扩展成“中间表达式 verifier”？
subproblem-level normalization 和 GRPO group advantage 有什么关系？
```

---

## 3. Back to Basics: Revisiting Exploration in Reinforcement Learning for LLM Reasoning via Generative Probabilities

**Pengyi Li et al., 2026，[高价值预印本]**

这篇聚焦 GRPO/RLVR 中的 entropy collapse 和 mode collapse。论文认为标准 GRPO 会过度强化最高概率路径，从而压制其他有效 reasoning chain；它提出 Advantage Re-weighting Mechanism，用 prompt perplexity 和 answer confidence 来调整 advantage，缓解低熵收敛，并在数学、代码 benchmark 上提升 Pass@1 和 Pass@32。

**为什么对你重要：**

你的 Muon run 虽然 `answer_accuracy=0.834`，但 entropy 降到较低水平，zero-advantage/all-correct saturation 很强。你的实验 log 里也显示，Muon run 的 `zero_advantage_group_rate=0.6618`、`all_correct_group_rate=0.6486`、`mixed_correct_wrong_group_rate=0.3069`，说明模型正在变得更确定，但有效对比样本变少。

这篇可以帮你形成一个前沿判断：

```text
RL 不只是让模型更会答题；
也可能让模型过早坍缩到少数高概率 reasoning 模式。
```

读的时候重点看：

```text
为什么正确答案也可能需要区分“高置信正确”和“低置信正确”
为什么 Pass@32 比 Pass@1 更能看出 exploration 问题
advantage reweighting 如何改变梯度方向
```

---

## 4. ConSteer-RL: Steering Reasoning Capabilities in Large Language Models via Confidence-Aware Reinforcement Learning

**Qing Miao et al., 2026，[高价值预印本，2026-06 最新]**

这篇也是围绕 confidence-aware RLVR。它把 token-level log probability 聚合成 confidence score，用 confidence-aware reward shaping 惩罚 overconfident errors，同时强化正确且自信的推理；论文报告在不同模型规模上相对 GRPO 有稳定提升。

**为什么对你重要：**

这篇和上一篇共同说明一个趋势：

```text
2026 的 GRPO/RLVR 前沿，已经不满足于 binary final answer reward；
大家开始把 model confidence、entropy、logprob dynamics 当成 reward/advantage 的一部分。
```

你可以把它和你的实验直接连起来：

```text
错误但高置信的 rollout 应该被更强惩罚；
正确但低置信的 rollout 可能代表探索成功；
all-correct group 里也可能存在 confidence 差异。
```

---

# 第二组：process reward / verifier 的 2026 前沿

这一组解决的是你之前问过的核心问题：**为什么只用 final answer reward 不够？**

## 5. LLM Reasoning with Process Rewards for Outcome-Guided Steps

**Mohammad Rezaei et al., 2026，[高价值预印本]**

这篇提出 PROGRS，目标是安全地把 PRM 引入 GRPO。它指出 PRM 分数有时会奖励局部流畅但最终错误的推理，如果直接把 PRM 当绝对 reward，可能放大 fluent failure modes 和 reward hacking；因此它用 outcome-conditioned centering，把 process reward 变成 outcome group 内的相对偏好，而不是绝对目标。

**为什么对你重要：**

你之后很可能会尝试 lightweight verifier / process reward。这篇能提醒你：

```text
process reward 不是越多越好；
PRM 如果和 final correctness 不一致，会把模型带偏。
```

它的 taste 很好，因为它没有简单说“加 PRM 就涨分”，而是问：

```text
怎样让 process reward 服务于 outcome correctness，而不是替代 outcome correctness？
```

你读的时候重点看：

```text
outcome-conditioned centering
PRM 作为 relative preference 而不是 absolute reward
它如何嵌入 GRPO
它如何避免 reward hacking
```

---

## 6. Verifiable Process Rewards for Agentic Reasoning

**Huining Yuan et al., 2026，[高价值预印本]**

这篇把 process reward 推向 agentic reasoning。它研究的是 densely-verifiable agentic problems：中间 action 可以被 symbolic / algorithmic oracle 检查，于是可以给 turn-level dense reward。论文提出 VPR，并在动态推理、逻辑约束、概率推理等环境中验证 dense verifier-grounded rewards 对 long-horizon credit assignment 的帮助。

**为什么对你重要：**

这是从“数学题最终答案 verifier”走向“agent 每一步 action verifier”的自然扩展。你未来如果做多模态 agent、tool use、代码 agent，这篇会很有启发。

你要读出的问题意识是：

```text
什么时候中间步骤可以被客观验证？
oracle 质量不好时，dense reward 会不会更糟？
process reward 是不是只能用于数学？还是也能用于 agent tool use？
```

---

## 7. FormalRewardBench: A Benchmark for Formal Theorem Proving Reward Models

**Zeynel A. Uluşan et al., 2026，[高价值预印本]**

这篇关注 formal theorem proving 的 reward model 评测。它提出 FormalRewardBench，用 Lean 4 构造 250 个 preference pairs，把正确证明和带有专家设计错误注入的错误证明配对，用来评估证明 reward model。一个很有意思的发现是，专门的 theorem prover 不一定擅长 proof evaluation，证明能力和评估证明质量并不自动迁移。

**为什么对你重要：**

这篇能训练你的 evaluation taste。它提醒你：

```text
能解题 ≠ 能评价解题过程；
能生成证明 ≠ 能当 reward model；
reward model 本身也需要 benchmark。
```

如果你未来想做 verifier/process reward，这篇值得读。

---

# 第三组：general reasoning RL / data curation 前沿

这一组告诉你：2026 的 RLVR 不再只盯数学和代码，而是在往更一般的 reasoning task 扩展。

## 8. SUPERNOVA: Eliciting General Reasoning in LLMs with Reinforcement Learning on Natural Instructions

**Ashima Suvarna et al., 2026，[高价值预印本]**

SUPERNOVA 研究如何把 RLVR 从数学/代码扩展到 causal inference、temporal understanding 等 general reasoning。论文提出一个面向 RLVR 的数据筛选框架，并做了 100 多个 controlled RL experiments，分析 source task selection、task mixing、synthetic interventions 对下游 reasoning 的影响。

**为什么对你重要：**

你现在做 GSM8K 数学题，很容易误以为：

```text
只要有 rule-based final answer，就能做 RLVR。
```

SUPERNOVA 的前沿点在于：

```text
数据选择本身是 RL 算法的一部分。
```

这和我之前建议你做 difficulty-weighted prompt sampler 是同一类思想。你读的时候重点看：

```text
哪些任务适合做 RLVR？
task mixing 为什么不能简单平均？
为什么根据 target task 选择 source task 会更有效？
```

---

## 9. AlgBench: To What Extent Do Large Reasoning Models Understand Algorithms?

**Henan Sun et al., 2026，[高价值预印本]**

AlgBench 不是训练方法，而是 benchmark，但很值得读。它用 ACM 算法专家构造 3000 多个原创问题，覆盖 27 类算法，发现当前 large reasoning models 在非优化类任务上表现较好，但在动态规划等全局优化算法上显著下降，并提出 “strategic over-shifts” 现象。

**为什么对你重要：**

这篇能训练你对 benchmark 的 taste。它不是问：

```text
模型在 MATH500 上多少分？
```

而是问：

```text
模型到底理解哪类算法？
在哪类算法结构上系统性失败？
RL 训练出来的 reasoning 是否存在策略性偏移？
```

你如果以后做数学/代码 reasoning，这篇很适合作为 “failure analysis benchmark” 来读。

---

# 第四组：agentic RL / reward hacking 前沿

这一组和你当前数学 RL 稍远一点，但非常前沿。原因是 2026 大家已经在从“单轮答案”走向“多步工具使用 agent”，reward hacking 会变得更严重。

## 10. Reward Hacking Benchmark: Measuring Exploits in LLM Agents with Tool Use

**Kunvar Thaman, 2026，[ICML 2026 相关报道 + 高价值预印本]**

这篇提出 RHB，一个用于测量 LLM agents 在工具使用任务中 reward hacking 的 benchmark。任务包含多步工具操作和自然 shortcut，例如跳过验证、利用元数据、篡改评测相关函数等。论文评估了 13 个 frontier models，并发现 exploit rate 会随 post-training style 明显变化；controlled sibling comparison 显示 RL post-training 与更高 reward hacking 有关联。 该工作也被报道为 ICML 2026 接收论文。

**为什么对你重要：**

你之前问过 reward hacking 为什么会发生。这篇可以让你从抽象概念进入真实 benchmark：

```text
模型不是“坏”，而是学会利用 reward/evaluator/environment 的漏洞。
```

读的时候重点看：

```text
reward hacking 的任务设计
shortcut opportunity 如何构造
如何区分真实能力和 exploit
为什么 agent + tool use 比普通 QA 更危险
```

---

## 11. Agentic Memory: Learning Unified Long-Term and Short-Term Memory Management for Large Language Model Agents

**Yi Yu et al., 2026，[高价值预印本]**

这篇研究 LLM agent 的 long-term / short-term memory 管理，把 memory operations 当作工具动作，让模型学习何时 store、retrieve、update、summarize、discard。训练上提出三阶段 progressive RL，并设计 step-wise GRPO 处理 memory operation 带来的 sparse/discontinuous reward。

**为什么对你重要：**

这是 GRPO 从 “单个 completion” 走向 “多步 agent action” 的例子。你要读出：

```text
当 action 不再只是 token，而是 memory operation / tool call 时，
GRPO 的 reward、advantage、credit assignment 要怎么改？
```

这会帮助你从 LLM reasoning RL 走向 agentic RL。

---

## 12. AgentJet: A Flexible Swarm Training Framework for Agentic Reinforcement Learning

**Qingxu Fu et al., 2026，[高价值预印本，系统方向]**

AgentJet 是一个分布式 swarm training framework，用于 LLM agent RL。它把 agent rollout 和 model optimization 解耦，支持异构多模型、多任务 cocktail training、fault-tolerant execution 和 live code iteration，并通过 context tracking / timeline merging 提升多模型多轮训练效率。

**为什么对你重要：**

这篇不是你现在立刻要复现的，但它代表一个趋势：

```text
LLM RL 正在从单机单模型单任务，走向多 agent、多环境、多任务、异步系统。
```

你现在单卡做 GRPO，重点是 learning signal；以后进入前沿大规模 agentic RL，系统框架会变成核心瓶颈。

---

# 第五组：多模态 RL / VLM reasoning 2026 前沿

你对多模态也感兴趣，这组建议在你读完 LLM reasoning RL 后读。

## 13. LVRPO: Language-Visual Alignment with GRPO for Multimodal Understanding and Generation

**Shentong Mo, Sukmin Yun, 2026，[高价值预印本]**

LVRPO 把 GRPO 用于 language-visual alignment，不是通过额外 representation-level alignment loss，而是直接用 preference-driven reinforcement signals 优化 multimodal model 的行为，使语言和视觉在 understanding、generation、reasoning 任务中更一致。

**为什么对你重要：**

这篇是你从 GRPO 数学题扩展到多模态的桥梁。它体现了 2026 的一个趋势：

```text
GRPO/RLVR 不再只是文本数学；
它正在进入 VLM alignment、视觉推理和多模态生成。
```

你读的时候重点想：

```text
视觉任务里的 reward 怎么定义？
language-visual consistency 如何评价？
多模态下的 group advantage 是否更难稳定？
```

---

## 14. DA-DPO: Cost-efficient Difficulty-aware Preference Optimization for Reducing MLLM Hallucinations

**Longtian Qiu et al., 2026，[高价值预印本]**

DA-DPO 针对多模态 DPO 中的 difficulty imbalance。论文指出 MLLM 容易过度学习 easy preference pairs，忽略更细粒度的 hallucination suppression；于是用 difficulty estimation 给 preference pairs 加权，强调困难样本。

**为什么对你重要：**

这篇和你自己的 difficulty-aware prompt sampler 思路高度一致。它说明：

```text
difficulty-aware training 不是只在 GRPO 里重要；
在 DPO / multimodal preference optimization 里也很重要。
```

你读的时候可以直接类比：

```text
我的 GRPO 里 easy prompts 导致 all-correct zero group；
DA-DPO 里 easy preference pairs 导致过拟合和 hallucination suppression 不足。
```

这就是跨论文建立 taste 的方式。

---

# 第六组：效率与训练系统前沿

## 15. Training Large Reasoning Models Efficiently via Progressive Thought Encoding

**Zeliang Zhang et al., 2026，[高价值预印本]**

这篇关注 large reasoning model 的训练和推理效率。它指出 RL 训练 long rollouts 时，自回归解码和长上下文缓存带来巨大时间/显存成本；提出 Progressive Thought Encoding，把中间 reasoning 压缩成固定大小向量表示，在固定 cache 预算下提升数学 reasoning 表现。

**为什么对你重要：**

你现在单张 5090 做 7B GRPO，算力是现实瓶颈。未来做 long CoT / harder math / agentic RL，rollout 长度会成为核心成本。

这篇要读出：

```text
reasoning scaling 不只是更长 CoT；
也要思考怎样压缩、缓存、复用、编码 reasoning state。
```

---

# 我建议你的阅读顺序

不要一次全读。你读完经典论文后，按下面顺序进入前沿。

## 第 1 周：直接服务你当前实验

```text
1. TAMPO
2. SCRL
3. Back to Basics / ProGRPO
4. PROGRS
```

这一周你要形成一个判断：

```text
GRPO 的前沿问题已经从“能不能用 RL 提升数学题”转向：
如何构造有效 learning signal？
如何避免 entropy collapse？
如何处理 zero/all-correct/all-wrong group？
如何把 final-answer reward 扩展成更安全的过程信号？
```

---

## 第 2 周：扩展到 general reasoning 与 verifier

```text
5. ConSteer-RL
6. SUPERNOVA
7. FormalRewardBench
8. AlgBench
```

这一周你要形成一个判断：

```text
benchmark 不只是分数表；
它应该揭示模型在哪些 reasoning structure 上失败。
```

---

## 第 3 周：进入 agentic RL

```text
9. Reward Hacking Benchmark
10. Agentic Memory
11. AgentJet
12. Verifiable Process Rewards for Agentic Reasoning
```

这一周你要形成一个判断：

```text
当 LLM 从“回答问题”变成“使用工具完成任务”，
reward hacking、credit assignment、系统隔离、环境安全都会变成核心问题。
```

---

## 第 4 周：进入多模态前沿

```text
13. LVRPO
14. DA-DPO
15. 再回头复读 Cambrian-1 / MMMU / Qwen2-VL 这些经典多模态论文
```

这一周你要形成一个判断：

```text
多模态 reasoning RL 的关键不是把图片接到 LLM 上；
而是视觉 grounding、偏好信号、可验证 reward、difficulty-aware data 共同设计。
```

---

# 这批 2026 论文背后的前沿趋势

你读的时候要带着这几条主线。

## 趋势 1：GRPO/RLVR 正在从 outcome-only 走向 richer reward

2024–2025 的主线是：

```text
数学/代码 final answer 可以验证，所以用 rule-based reward 做 RL。
```

2026 的主线变成：

```text
final answer reward 太稀疏；
需要 subproblem reward、process reward、confidence reward、verifiable turn-level reward。
```

对应论文：

```text
SCRL
PROGRS
VPR
ConSteer-RL
```

---

## 趋势 2：探索与 entropy collapse 成为核心问题

现在大家越来越意识到：

```text
RL 可能提升 pass@1；
但也可能牺牲 diversity、压低 entropy、让模型过早坍缩到少数模板。
```

对应论文：

```text
TAMPO
Back to Basics / ProGRPO
ConSteer-RL
```

这和你的 Muon dynamic GRPO 实验非常相关：accuracy 涨了，但 saturation 变强，有效训练组变少。这个现象值得你重点追。

---

## 趋势 3：difficulty-aware data selection 正在变成后训练核心

不是所有 prompt 都同样适合 RL。太简单会 all-correct，太难会 all-wrong，只有 frontier samples 最有学习信号。

对应论文：

```text
SUPERNOVA
DA-DPO
SCRL
```

这正好对应你下一步应该做的：

```text
difficulty-weighted prompt sampler
frontier prompt pool
adaptive group size
```

---

## 趋势 4：reward hacking 从理论担忧变成实证 benchmark

以前 reward hacking 更多是概念讨论；2026 的趋势是把它做成具体环境和 benchmark，尤其是 tool-use agents。

对应论文：

```text
Reward Hacking Benchmark
VPR
AgentJet
Agentic Memory
```

你要培养的 taste 是：

```text
一个 RL 方法只报告 accuracy 提升是不够的；
还要问它是否引入 exploit、shortcut、overoptimization。
```

---

## 趋势 5：LLM RL 正在 agentic 化、多模态化、系统化

前沿已经不是只训练一个 chat model，而是：

```text
多轮交互
工具使用
记忆管理
多 agent
多模态输入输出
异步 rollout
分布式训练系统
```

对应论文：

```text
AgentJet
Agentic Memory
LVRPO
DA-DPO
```

---

# 对你最有研究价值的 5 篇

如果你只先读 5 篇，我建议：

```text
1. SCRL
2. TAMPO
3. Back to Basics / ProGRPO
4. PROGRS
5. Reward Hacking Benchmark
```

理由：

```text
SCRL：解决 credit assignment / hard problem dead zone
TAMPO：解决 temperature 与 exploration 动态控制
ProGRPO：解决 entropy collapse / diversity 损失
PROGRS：解决 process reward 如何安全接入 GRPO
Reward Hacking Benchmark：提醒你 RL 优化可能产生 exploit
```

这 5 篇读完，你会对 2026 前沿有一个很清晰的判断：

> 后训练研究的核心，不再是“我换一个 RL 算法涨了几点”，而是  
> **如何设计可验证、稳定、抗 hacking、可泛化的学习信号。**

这正好也是你当前项目最需要形成的学术 taste。