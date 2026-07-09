---
title: CoT-Paper-Report
---

# CoT 论文小报告

## Chain-of-Thought Prompting Elicits Reasoning in Large Language Models

---

## 1. 论文基本信息

**论文标题：** Chain-of-Thought Prompting Elicits Reasoning in Large Language Models  
**中文理解：** 链式思维提示可以激发大语言模型的推理能力  
**核心关键词：** Chain-of-Thought, CoT, reasoning trace, prompting, emergent ability, large language model  
**论文定位：** LLM reasoning 方向的奠基论文之一。

这篇论文的核心贡献不是训练了一个新模型，也不是提出了复杂架构，而是发现：

> 只要在 prompt 中加入带有中间推理步骤的示例，大模型就会倾向于先生成推理链，再输出答案，并在复杂推理任务上显著提升表现。

---

## 2. 研究背景

在 CoT 之前，大语言模型通常被用来直接从问题预测答案：

```text
Question -> Answer
```

例如：

```text
Q: 小明有 3 个苹果，又买了 2 个，一共有几个？
A: 5
```

这种方式对简单问题没问题，但面对多步数学、逻辑、符号推理时，模型很容易直接猜错。

早期也有人关注模型的“解释”或 “rationale”，但很多时候解释是：

```text
Question -> Answer -> Explanation
```

也就是说，模型先给答案，再补充一个解释。这种解释更像是 **post-hoc explanation**，主要给人看，不一定真的参与了答案生成。

CoT 的关键变化是：

```text
Question -> Reasoning -> Answer
```

它让模型在答案之前生成中间推理步骤。由于 LLM 是自回归模型，前面生成的 reasoning token 会进入上下文，影响后续答案生成。因此，推理链不只是解释，而是参与了模型的计算过程。

---

## 3. CoT 的核心方法

原始 CoT 论文主要使用的是 **few-shot CoT prompting**。

也就是：

> 不训练模型，不修改参数，只在 prompt 中放几个带有推理过程的示例。

普通 prompting：

```text
Q: Roger has 5 tennis balls. He buys 2 cans. Each can has 3 balls. How many balls does he have?
A: 11
```

CoT prompting：

```text
Q: Roger has 5 tennis balls. He buys 2 cans. Each can has 3 balls. How many balls does he have?
A: Roger starts with 5 balls. Two cans have 2*3=6 balls. So he has 5+6=11 balls. The answer is 11.
```

然后给新问题时，模型会模仿这种格式，先生成推理过程，再输出答案。

后来又发展出了 **Zero-shot CoT**，只需要在 prompt 后面加一句：

```text
Let's think step by step.
```

或者中文：

```text
让我们一步一步思考。
```

模型也会更倾向于输出推理链。

---

## 4. CoT 的机制理解

可以这样理解 CoT 的机制：

> CoT prompt 会改变模型当前上下文下的条件生成分布，让“推理型 token 序列”的概率变高。模型一旦开始生成推理链，后续 token 又会基于前面的推理 token 继续生成，从而形成完整的链式推理过程。

普通直接回答时，模型近似在做：

```text
P(answer | question)
```

CoT 时，模型变成：

```text
P(reasoning_1, reasoning_2, ..., reasoning_n, answer | question, cot_examples)
```

自回归展开就是：

```text
P(r1 | context)
× P(r2 | context, r1)
× P(r3 | context, r1, r2)
× ...
× P(answer | context, r1, r2, ..., rn)
```

所以，中间推理 token 不是装饰，而是真的进入了后续预测的条件上下文。

这也是 CoT 有效的关键：它把一个困难映射：

```text
复杂问题 -> 最终答案
```

拆成多个简单映射：

```text
复杂问题 -> 第一步
第一步 -> 第二步
第二步 -> 第三步
第三步 -> 最终答案
```

---

## 5. CoT 和“解释”的区别

CoT 之前，模型输出解释更多是：

```text
答案之后的解释
```

这种解释可能只是事后合理化。

例如：

```text
Q: 小明有 3 个苹果，又买了 2 个，一共有几个？
A: 5。因为 3+2=5。
```

这里解释在答案之后，不一定参与模型得出答案。

CoT 则是：

```text
答案之前的推理
```

例如：

```text
Q: 小明有 3 个苹果，又买了 2 个，一共有几个？
A: 小明一开始有 3 个苹果，又买了 2 个，所以一共有 3+2=5 个。答案是 5。
```

这里的 `3+2=5` 在最终答案之前生成，会作为后续上下文，直接影响“答案是 5”的生成。

因此可以区分：

```text
explanation: 给人看的解释
reasoning trace: 参与模型生成答案的推理轨迹
```

CoT 的贡献在于把解释从 **post-hoc explanation** 转变成 **pre-answer reasoning trace**。

---

## 6. 论文的实验设计

这篇论文的实验流程很值得学习。

### 6.1 Baseline 设计

论文主要比较两种 prompt：

```text
Standard Prompting vs Chain-of-Thought Prompting
```

区别只在于：

- **Standard Prompting：** 示例只给最终答案
- **CoT Prompting：** 示例给中间推理步骤 + 最终答案

其他条件尽量保持一致。

这体现了很重要的实验原则：

> 控制变量，让读者清楚性能提升到底来自哪里。

---

### 6.2 任务选择

论文没有只在一个数据集上测试，而是选择了多个任务族。

主要包括：

1. **数学推理**
   - GSM8K
   - MultiArith
   - AddSub
   - AQUA-RAT

2. **常识推理**
   - StrategyQA
   - Date Understanding
   - Sports Understanding

3. **符号推理**
   - Last Letter Concatenation
   - Coin Flip

这些任务覆盖了不同类型的推理能力。

数学任务需要多步计算，常识任务需要隐含知识和判断，符号任务则更强调严格步骤执行。

这种任务设计说明：

> 好论文不应该只在一个 benchmark 上证明有效，而应该跨任务族展示方法的泛化性。

---

### 6.3 模型规模对比

论文还有一个重要实验设计：比较不同规模模型上的效果。

它发现：

> CoT 不是所有模型都有效，而是在大模型上更明显有效。

小模型即使给了 CoT prompt，也可能无法生成高质量推理链，甚至生成错误废话。

大模型则能够更稳定地模仿示例中的推理格式，并利用中间步骤提升答案正确率。

这说明 CoT 具有明显的 **scale-dependent** 特征，也就是和模型规模相关。

---

## 7. 论文的重要发现

### 7.1 CoT 可以显著提升复杂推理任务表现

尤其是在多步数学推理任务上，CoT 相比直接回答有明显提升。

这说明：

> 对于复杂推理任务，让模型先生成中间步骤再给答案，比直接预测答案更有效。

---

### 7.2 CoT 是大模型上的涌现能力

论文的重要发现之一是：

> CoT 的效果随着模型规模增大而变得明显。

这不是简单的 prompt trick，而是揭示了一种规模相关现象：

```text
小模型：CoT 效果弱，甚至无效
大模型：CoT 效果明显
```

这后来和 **emergent abilities** 的讨论联系很紧。

---

### 7.3 推理链让错误分析更容易

直接回答时：

```text
A: 17
```

如果错了，很难知道模型错在哪里。

CoT 输出时：

```text
第一步...
第二步...
所以答案是 17
```

我们可以分析：

- 是理解错了？
- 是中间计算错了？
- 是最后汇总错了？
- 是推理步骤正确但答案抄错了？

这对后续研究非常重要，因为它引出了 **process supervision** 和 **PRM**。

---

## 8. 论文真正有价值的创新点

我认为这篇论文最值得学习的创新不只是“加一个 prompt”。

更深层的价值有三点。

### 8.1 把 reasoning trace 变成研究对象

CoT 让大家意识到：

> 研究 LLM 推理不能只看最终答案，还要看中间推理轨迹。

这直接影响了后续很多方向：

- **Self-Consistency：** 生成多条推理链并投票
- **Verifier：** 判断哪条推理链更可靠
- **PRM：** 给每一步推理打分
- **R1 / RLVR：** 用奖励训练模型产生更有效的推理
- **Test-time Scaling：** 用更多 token / 更多采样换更高正确率

---

### 8.2 发现推理能力和模型规模相关

CoT 论文不是只说“我的 prompt 有用”，而是展示了一个重要现象：

> 当模型足够大时，prompt 中的推理示例能够激发出强得多的推理能力。

这让论文从普通技巧变成了现象发现型工作。

---

### 8.3 简单方法 + 系统实验

这篇论文的方法很简单，但实验非常系统。

它的科研方法论值得学习：

```text
重要问题
+ 简单清晰的方法
+ 严格控制变量
+ 多任务验证
+ 不同模型规模比较
+ 定性案例分析
= 有影响力的论文
```

这对初学科研很有启发。

---

## 9. 局限性

### 9.1 CoT 不保证推理真实可靠

模型可能生成看似合理但实际错误的推理链。

甚至可能出现：

```text
答案是对的，但推理过程是错的
```

或者：

```text
推理过程看起来很自然，但其实是事后编造
```

这就是 CoT 的 **faithfulness** 问题。

---

### 9.2 中间步骤错误会传播

如果模型中间一步算错，后面可能会基于错误结果继续推理。

例如：

```text
10 - 4 = 7
7 + 3 = 10
答案是 10
```

这种 **error propagation** 是 CoT 的重要风险。

---

### 9.3 CoT 会增加 token 成本

CoT 输出更长，推理成本更高，延迟更大。

这引出后来的问题：

> 什么时候值得让模型多想？什么时候直接回答就够了？

也就是 **test-time compute / adaptive reasoning** 的研究方向。

---

### 9.4 依赖 prompt 示例质量

Few-shot CoT 的效果受到示例选择、示例数量、推理格式的影响。

不同 prompt 可能导致不同结果，因此稳定性也是问题。

---

## 10. 和后续工作的关系

CoT 可以看成 LLM reasoning 主线的起点。

后续发展大致是：

```text
CoT Prompting
  ↓
Zero-shot CoT
  ↓
Self-Consistency
  ↓
Verifier / PRM
  ↓
Reasoning SFT
  ↓
RLVR / DeepSeek-R1
  ↓
Test-time Scaling / o1 类模型
```

CoT 解决的是：

```text
如何让模型生成一条推理链？
```

但马上会遇到新问题：

```text
如果这条推理链错了怎么办？
```

于是出现了 Self-Consistency：

```text
生成多条推理链，让答案投票。
```

再进一步，如果多条推理链质量不同，就需要 verifier 或 PRM 判断哪条更可靠。

再后来，R1/o1 类模型开始用 SFT/RL 训练模型更稳定地产生有效推理。

---

## 11. 当前理解总结

通过这次学习，可以把 CoT 总结为：

> CoT Prompting 是一种 inference-time prompting 方法。它通过在 prompt 中加入带中间推理步骤的示例，诱导大模型先生成 reasoning trace 再输出答案。由于 LLM 是自回归生成的，中间推理 token 会成为后续预测的上下文，从而将复杂推理问题拆解为一系列局部生成步骤。论文的主要价值不仅在于提出了 CoT 方法，还在于系统证明了 CoT 在大模型上的 scale-dependent 推理增强效果，并开启了后续围绕 reasoning trace 的研究范式。

更简洁地说：

```text
CoT 的核心不是让模型“解释答案”，
而是让模型“通过生成中间 token 来完成推理”。
```

---

## 12. 后续学习方向

学完这篇后，下一篇自然应该看：

**Self-Consistency Improves Chain of Thought Reasoning in Language Models**

因为它接着解决 CoT 的一个核心问题：

```text
单条推理链可能错。
```

Self-Consistency 的方法是：

```text
采样多条推理链
得到多个答案
用多数投票选择最终答案
```

它是 CoT 之后最自然、也最重要的扩展之一。
