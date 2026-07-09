---
title: Self-Consistency-Paper-Report
---

# Self-Consistency 论文小报告

## Self-Consistency Improves Chain of Thought Reasoning in Language Models

---

## 1. 论文基本信息

**论文标题：** Self-Consistency Improves Chain of Thought Reasoning in Language Models  
**中文理解：** 自洽性可以提升语言模型的链式思维推理能力  
**核心关键词：** Self-Consistency, Chain-of-Thought, sampling decoding, majority voting, reasoning paths, test-time compute  
**论文定位：** CoT 之后的重要扩展工作，是早期 test-time scaling / 多路径推理思想的代表。

这篇论文接着 CoT 解决了一个很自然的问题：

> CoT 可以让模型生成一条推理链，但这一条推理链可能是错的。那能不能让模型生成多条推理链，然后选择最一致的答案？

Self-Consistency 的答案是：可以。

---

## 2. 研究背景

第一篇 CoT 论文解决的是：

```text
如何让模型生成一条推理链？
```

CoT 的基本流程是：

```text
Question -> Reasoning Path -> Answer
```

但是单条推理链并不稳定。

同一道题，模型可能生成不同推理路径：

```text
Path 1: 正确理解题目，正确计算 -> 答案 25
Path 2: 中间计算错误 -> 答案 30
Path 3: 理解错题目 -> 答案 18
Path 4: 换一种正确解法 -> 答案 25
```

如果只生成一次，就可能刚好抽到错误路径。

因此 Self-Consistency 试图解决：

```text
如何从多条可能的 CoT 推理路径中得到更可靠的最终答案？
```

---

## 3. 核心方法

Self-Consistency 的核心流程是：

```text
输入问题
  ↓
使用 CoT prompt
  ↓
通过 sampling decoding 生成多条 reasoning paths
  ↓
抽取每条 path 的 final answer
  ↓
对 final answers 做 majority voting
  ↓
选择票数最多的答案作为最终输出
```

也就是说：

```text
CoT：让模型想一遍。
Self-Consistency：让模型想很多遍，然后投票。
```

它不是训练方法，不改模型参数，也不改模型结构。

它主要改的是：

```text
inference-time decoding strategy + answer aggregation
```

---

## 4. 它是怎么实现的？

Self-Consistency 仍然基于 CoT prompt。

但普通 CoT 和 Self-Consistency 的 decoding 策略不同。

### 4.1 普通 CoT

普通 CoT 通常使用 greedy decoding 或者单次生成：

```text
同一个 prompt
  ↓
greedy decoding / 单次生成
  ↓
一条推理链
  ↓
一个答案
```

### 4.2 Self-Consistency

Self-Consistency 使用 sampling decoding 多次生成：

```text
同一个 prompt
  ↓
sampling decoding 多次生成
  ↓
多条推理链
  ↓
多个答案
  ↓
多数投票
```

伪代码如下：

```python
def self_consistency(model, cot_prompt, question, K=20):
    answers = []

    for _ in range(K):
        output = model.generate(
            cot_prompt + question,
            do_sample=True,
            temperature=0.7,
            top_p=0.95,
        )
        answer = extract_final_answer(output)
        answer = normalize(answer)
        answers.append(answer)

    return majority_vote(answers)
```

---

## 5. Greedy Decoding 和 Sampling Decoding 的区别

模型本体做的是：

```text
context -> Transformer forward -> logits / probability distribution
```

Decoding 做的是：

```text
logits / probability distribution -> 选择下一个 token
```

因此，decoding 是在模型网络输出之后的 token 选择策略，不属于模型结构修改。

### 5.1 Greedy Decoding

Greedy decoding 每一步都选择概率最高的 token。

例如模型输出：

```text
token_A: 0.60
token_B: 0.25
token_C: 0.15
```

greedy decoding 永远选择：

```text
token_A
```

所以它比较确定，同一个 prompt 往往生成同一条路径。

### 5.2 Sampling Decoding

Sampling decoding 按照概率分布随机抽样。

例如：

```text
token_A: 0.60
token_B: 0.25
token_C: 0.15
```

sampling 大多数时候会选 token_A，但也有概率选 token_B 或 token_C。

所以同一个 prompt，多次生成可能得到不同的推理链。

这就是 Self-Consistency 能获得多条 reasoning paths 的基础。

---

## 6. 为什么这不是改模型结构？

因为模型的这些部分都没有变化：

```text
参数没变
层数没变
attention 没变
MLP 没变
vocab 没变
训练方式没变
```

变的只是：

```text
从模型输出的概率分布里选择 token 的规则
```

可以理解为：

> 模型负责给每个 token 打分，decoding 负责根据这些分数选哪个 token。

Self-Consistency 就是在这个推理阶段做文章：

```text
多次采样 token 序列 -> 得到多条 CoT -> 对答案投票
```

---

## 7. Majority Voting 的细节

Self-Consistency 不是对完整 CoT 文本投票，而是对最终答案投票。

例如：

```text
Path 1: 18 + 12 - 5 = 25 -> 答案 25
Path 2: 用容斥原理，18 + 12 - 5 = 25 -> 答案 25
Path 3: 18 + 12 = 30 -> 答案 30
Path 4: 先算总数再去重 -> 答案 25
```

投票结果：

```text
25: 3 票
30: 1 票
```

最终选择：

```text
25
```

需要注意：

> 原论文通常不是投票出答案后，再根据这个答案重新生成一条完整 CoT。

更准确的流程是：

```text
采样多条 CoT
  ↓
抽取最终答案
  ↓
投票
  ↓
直接选择票数最多的最终答案
```

如果需要展示解释，可以从已经采样出的多条 CoT 中，挑一条导向该答案的路径作为解释，但重新生成 CoT 不是 Self-Consistency 的核心步骤。

---

## 8. 核心思想：Marginalization over Reasoning Paths

Self-Consistency 的核心不只是“投票”，更深层是：

> 不执着于单条最可能推理路径，而是看所有可能推理路径整体上支持哪个答案。

普通 CoT 更像：

```text
选择一条最可能的 reasoning path
```

Self-Consistency 更像：

```text
对多个 reasoning paths 进行 marginalization，选择总支持度最高的 answer
```

可以写成：

```text
answer = argmax_a Σ P(reasoning_path, answer=a | question)
```

也就是说：

> 正确答案往往会被多条不同推理路径支持，而随机错误更可能被投票稀释。

这个思想是这篇论文最值得学习的地方之一。

---

## 9. 实验设计

这篇论文的实验设计可以拆成几个关键部分。

### 9.1 Baseline 对照

主要比较：

```text
CoT + greedy decoding / 单次生成
vs
CoT + sampling decoding + majority voting
```

控制变量：

```text
模型相同
prompt 相同
任务相同
只改变 decoding 和 aggregation 策略
```

如果 Self-Consistency 提升准确率，就说明多路径采样和答案投票是有效的。

---

### 9.2 多任务验证

论文在多种 reasoning benchmark 上测试，包括数学推理、常识推理和符号推理。

常见任务包括：

- GSM8K
- SVAMP
- AQuA
- StrategyQA
- ARC-Challenge
- Coin Flip
- Last Letter Concatenation

这说明它不是只在单一任务上有效，而是对多类推理任务都有帮助。

---

### 9.3 采样数量 K 的影响

论文还会分析采样数量 K 对准确率的影响。

一般趋势是：

```text
K 增大 -> 准确率提升
但提升逐渐变小
```

例如：

```text
K = 1: 普通 CoT
K = 5: 有明显提升
K = 20: 继续提升
K = 40: 边际收益变小
```

这说明 test-time compute 有用，但不是无限有用。

---

## 10. 它为什么有效？

### 10.1 多条正确路径可能收敛到同一答案

同一道题可能有多种正确解法。

如果某个答案是正确的，它可能被多条不同推理路径支持。

### 10.2 随机错误容易被稀释

单条 CoT 可能因为偶然错误得到错误答案。

但多条路径投票后，随机错误不一定稳定复现，因此会被多数投票抵消。

### 10.3 Sampling 提供了路径多样性

Greedy decoding 只走一条最可能的路径。

Sampling decoding 可以探索不同 reasoning paths。

这让模型有机会从多个角度解决同一个问题。

---

## 11. 和普通 Ensemble 的关系

Self-Consistency 很像 ensemble，但不是多个模型的 ensemble。

普通 ensemble：

```text
多个模型 -> 多个答案 -> 投票
```

Self-Consistency：

```text
同一个模型 -> 多次采样 -> 多个答案 -> 投票
```

所以它可以理解为：

```text
single-model sampling ensemble
```

或者：

```text
inference-time ensemble
```

这也是它优雅的地方：不训练多个模型，只利用同一个模型的随机采样获得多样化推理路径。

---

## 12. 和 Test-time Compute 的关系

Self-Consistency 是早期 test-time scaling 思想的代表。

它说明：

> 同一个模型，在推理时多花计算，也能获得更强推理能力。

可以对比：

```text
Training-time scaling: 更大模型，更多数据，更多训练算力
Test-time scaling: 更多采样，更长思考，更多搜索，更多推理时算力
```

Self-Consistency 属于：

```text
多采样 + 答案聚合
```

它为后来的：

- verifier-guided search
- best-of-N
- Tree-of-Thought
- MCTS-style reasoning
- o1 / R1 类长推理模型

提供了早期思想基础。

---

## 13. 实现细节和注意事项

### 13.1 答案抽取

模型输出是一整段自然语言，需要抽取 final answer。

例如：

```text
Therefore, the answer is 25.
```

需要抽取：

```text
25
```

如果是选择题，需要抽取：

```text
A / B / C / D
```

---

### 13.2 答案归一化

不同形式可能表示同一个答案：

```text
25
25.0
twenty-five
25 people
```

投票前需要 normalize，否则同一答案会被拆成不同类别。

---

### 13.3 Temperature / Top-p 控制

Sampling 需要控制多样性。

- temperature 太低：多条路径太相似，投票意义小
- temperature 太高：路径质量下降，噪声过多
- top-p / top-k：限制采样范围，避免采到太离谱的 token

这体现了一个重要权衡：

```text
diversity vs quality
```

---

### 13.4 K 的选择

K 越大，采样越多，准确率可能越高，但成本也越大。

```text
K = 1: 普通 CoT
K = 5/10: 低成本提升
K = 20/40: 更高准确率，但成本明显增加
```

实际系统需要在准确率和成本之间权衡。

---

## 14. 局限性

### 14.1 无法解决系统性错误

如果模型对某类问题整体理解错误，那么多数路径可能都会导向错误答案。

这种情况下，投票反而会更自信地选错。

所以：

```text
投票能缓解随机错误
但不能解决系统性偏差
```

---

### 14.2 只看最终答案，不判断推理过程质量

Self-Consistency 不判断每条 reasoning path 是否合理。

只要最终答案一样，就算一票。

因此可能出现：

```text
错误推理 -> 正确答案
乱猜 -> 正确答案
正确推理 -> 正确答案
```

它们在 voting 中权重相同。

这也是后来 verifier / PRM 要解决的问题。

---

### 14.3 推理成本显著增加

如果 K=40，就相当于同一道题生成 40 条 CoT。

成本大概变成：

```text
40 倍输出 token
```

这引出了后续研究问题：

```text
能不能自适应采样？
简单题少采样，难题多采样？
能不能早停？
能不能用 verifier 减少采样？
```

---

## 15. 最值得学习的点

### 15.1 从单路径推理到多路径推理

CoT 关注的是：

```text
如何生成一条推理链？
```

Self-Consistency 关注的是：

```text
多个推理路径的答案分布是什么？
```

这让 LLM reasoning 从单路径生成走向了多路径搜索/采样。

---

### 15.2 不改模型，也能通过 inference strategy 提升能力

这篇论文说明：

> 模型能力不只由参数决定，也受到推理时策略影响。

同一个模型，在不同 decoding / aggregation 策略下，表现可能明显不同。

---

### 15.3 简单方法背后的深刻思想

方法本身很简单：

```text
采样多条 CoT + 答案投票
```

但它揭示了一个重要思想：

```text
推理答案的可靠性可以通过多个 reasoning paths 的一致性来估计。
```

---

## 16. 当前理解总结

可以把 Self-Consistency 总结为：

> Self-Consistency 是 CoT 的 inference-time 增强方法。它不修改模型结构，也不训练模型，而是在 CoT prompt 基础上使用 sampling decoding 生成多条推理路径，再对这些路径的最终答案做 majority voting。其核心思想是对推理路径进行 marginalization：正确答案往往会被多条不同推理路径支持，而随机错误会被投票稀释。该方法展示了 test-time compute 对推理能力的提升作用，但代价是更高推理成本，且无法解决模型的系统性错误。

更短一点：

```text
CoT：让模型想一遍。
Self-Consistency：让模型想很多遍，然后看哪个答案最一致。
```

---

## 17. 和后续工作的关系

Self-Consistency 之后，LLM reasoning 进一步发展出很多方向：

```text
Self-Consistency
  ↓
Best-of-N
  ↓
Verifier / Reward Model
  ↓
Process Reward Model, PRM
  ↓
Tree-of-Thought / 搜索式推理
  ↓
Test-time scaling / o1 / R1 类模型
```

它自然引出下一批问题：

```text
如果多条推理链质量不同，能不能给它们打分？
如果投票还不够，能不能用 verifier 判断哪条更可靠？
能不能用更少采样达到同样准确率？
能不能根据题目难度动态决定采样次数？
```

这些问题会连接到后续的 verifier、PRM 和 test-time scaling 方向。
