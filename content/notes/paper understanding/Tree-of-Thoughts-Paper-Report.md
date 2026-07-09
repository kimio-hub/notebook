---
title: Tree-of-Thoughts-Paper-Report
---

# Tree of Thoughts 论文小报告

## Tree of Thoughts: Deliberate Problem Solving with Large Language Models

---

## 1. 论文基本信息

**论文标题：** Tree of Thoughts: Deliberate Problem Solving with Large Language Models  
**中文理解：** 思维树：用大语言模型进行审慎的问题求解  
**核心关键词：** Tree of Thoughts, ToT, search, thought, state, evaluator, BFS, DFS, beam search, deliberate reasoning  
**论文定位：** CoT 与 Self-Consistency 之后的重要扩展，将 LLM 推理从“单链生成/多链投票”推进到“搜索式推理”。

这篇论文的核心思想是：

> 不要让模型一次性生成完整推理链，而是把推理过程拆成多个 thought step，在每一步生成多个候选 thought，评估这些 thought/state 的前景，再用搜索算法保留好分支、剪掉差分支，最终找到可行解。

---

## 2. 和前两篇论文的关系

可以把前三篇论文串成一条线：

```text
CoT：一条思维链
Self-Consistency：多条完整思维链，最后投票
Tree of Thoughts：一棵思维树，中间就评估、剪枝和回溯
```

### 2.1 CoT 的问题

CoT 让模型生成一条推理链：

```text
Question -> Reasoning Path -> Answer
```

但问题是：

```text
单条链可能中途走错，而且很难回退。
```

---

### 2.2 Self-Consistency 的问题

Self-Consistency 生成多条完整推理链，然后对最终答案投票：

```text
Path 1 -> Answer A
Path 2 -> Answer B
Path 3 -> Answer A
最终选 Answer A
```

但它的问题是：

```text
只在完整路径结束后投票，中间过程没有真正搜索和纠错。
```

如果某条路径第一步就错了，它仍然会完整生成到底，比较浪费。

---

### 2.3 ToT 的改进

ToT 把推理过程建模成搜索树：

```text
当前状态
  ↓
生成多个候选 thought
  ↓
评估每个 thought / state
  ↓
保留高价值分支
  ↓
继续扩展
```

它的关键改进是：

```text
在中间节点就进行评估、选择、剪枝和回溯。
```

---

## 3. 什么是 Thought？

**Thought** 是 ToT 里的核心概念。

它不是单个 token，也不是完整推理链，而是一个中间推理单元。

可以理解为：

```text
thought = 当前问题求解过程中的一个中间步骤 / 局部方案 / 候选操作
```

例如 24 点游戏：

```text
初始 state: [4, 7, 8, 8]

thought: 8 - 4 = 4

新 state: [4, 7, 8]
```

这里：

```text
thought 是一步操作
state 是操作后的中间状态
```

更一般地：

```text
thought = 一步候选想法 / 操作 / 局部推理
state = 应用了 thought 后形成的中间状态
```

---

## 4. ToT 的核心框架

ToT 可以拆成三个核心组件：

```text
Generator + Evaluator + Search Algorithm
```

---

### 4.1 Thought Generator

Generator 负责生成候选 thought。

给定当前状态，让 LLM 提出几个下一步想法。

例如：

```text
当前状态：[4, 7, 8, 8]
任务：用这四个数通过加减乘除得到 24

请提出 3 个可能的下一步操作。
```

模型可能输出：

```text
1. 8 - 4 = 4，剩下 [4, 7, 8]
2. 8 / 4 = 2，剩下 [2, 7, 8]
3. 7 + 8 = 15，剩下 [4, 8, 15]
```

---

### 4.2 State / Thought Evaluator

Evaluator 负责评价某个 thought 或中间 state 是否有前途。

例如：

```text
当前状态：[2, 7, 8]
这个状态是否有可能继续得到 24？
请评价为 sure / likely / impossible。
```

模型可能输出：

```text
likely
```

也可以转换成分数：

```text
sure = 1.0
likely = 0.5
impossible = 0.0
```

这一步让 ToT 能在中间阶段剪枝，而不是等完整答案生成完才发现错误。

---

### 4.3 Search Algorithm

Search Algorithm 决定如何探索 thought tree。

常见方式包括：

- **BFS:** 广度优先搜索，每层扩展一批节点
- **DFS:** 深度优先搜索，一条路走到底，失败后回溯
- **Beam Search:** 每层只保留评分最高的 top-k 个状态

实际使用中，beam search 很常见，因为完整树搜索太贵。

---

## 5. ToT 是否修改模型结构？

不修改。

ToT 不改变：

```text
模型参数
Transformer 层数
Attention 结构
MLP 结构
词表
训练方式
```

它是一个外置的 inference-time search framework。

可以理解为：

```text
LLM = 负责生成和评价 thought 的黑盒函数
外部程序 = 负责维护搜索树、状态、分数、剪枝和终止条件的控制器
```

所以 ToT 更像一个外置 harness / reasoning framework。

---

## 6. 具体实现方式

一个 ToT 系统大致如下：

```python
def tree_of_thoughts(initial_state, max_depth, beam_size):
    states = [initial_state]

    for depth in range(max_depth):
        candidates = []

        # 1. 扩展当前状态
        for state in states:
            thoughts = llm_generate_thoughts(state)

            for thought in thoughts:
                new_state = apply_thought(state, thought)
                candidates.append(new_state)

        # 2. 评估候选状态
        scored_candidates = []
        for state in candidates:
            score = llm_evaluate_state(state)
            scored_candidates.append((state, score))

        # 3. 选择高分状态继续扩展
        states = select_top_k(scored_candidates, k=beam_size)

        # 4. 检查是否完成
        for state in states:
            if is_solution(state):
                return extract_solution(state)

    return best_state(states)
```

里面的：

```text
llm_generate_thoughts
llm_evaluate_state
```

是 LLM 调用。

而：

```text
states
candidates
scores
select_top_k
search tree
termination condition
```

由外部程序维护。

---

## 7. 和 Self-Consistency 的实现差别

Self-Consistency：

```text
调用模型 K 次
每次生成一条完整 CoT
最后对答案投票
```

ToT：

```text
第 1 层：生成候选 thought -> 评估 -> 保留
第 2 层：继续扩展 -> 评估 -> 保留
第 3 层：继续扩展 -> 评估 -> 保留
直到找到答案
```

所以 ToT 的 LLM 调用是交替进行的：

```text
generate -> evaluate -> generate -> evaluate -> ...
```

Self-Consistency 则是：

```text
generate full path -> generate full path -> ... -> vote
```

核心差别：

```text
Self-Consistency 只在终点投票
ToT 在中间节点就评估和搜索
```

---

## 8. 为什么 ToT 像 harness 工程？

ToT 很像一个外部 reasoning harness。

它不改变模型，而是在模型外面搭一个控制框架：

```text
模型负责“想”和“评价”
框架负责“组织、记录、筛选、剪枝、回溯”
```

这和 AI agent / coding agent 的系统设计很像。

例如 coding agent 里：

```text
state: 当前代码 + 报错信息
thought: 一个修复假设 / 一个修改方案
evaluate: 运行测试或让模型评价方案
search: 保留有效方案，放弃失败方案
```

所以 ToT 可以看作 agentic reasoning 的早期雏形。

---

## 9. 实验任务

ToT 论文选择了几个很能体现搜索、规划、回溯能力的任务。

主要包括：

1. **Game of 24**
2. **Creative Writing**
3. **Mini Crosswords**

它们分别对应：

```text
数学/符号搜索
开放式生成规划
约束满足问题
```

---

## 10. Game of 24

### 10.1 任务是什么？

给 4 个数字，通过加减乘除得到 24。

例如：

```text
输入：[4, 7, 8, 8]
目标：用这四个数字各一次，得到 24
```

---

### 10.2 为什么适合 ToT？

因为它有非常清楚的中间状态。

```text
初始 state: [4, 7, 8, 8]
thought: 8 - 4 = 4
new state: [4, 7, 8]
```

每次操作都会让 state 发生变化，直到剩下一个数：

```text
[24]
```

这天然是一棵搜索树。

---

### 10.3 ToT 怎么做？

每一步生成多个可能操作：

```text
当前数字：[4, 7, 8, 8]

候选 thought:
1. 8 - 4 = 4 -> [4, 7, 8]
2. 8 / 4 = 2 -> [2, 7, 8]
3. 7 + 8 = 15 -> [4, 8, 15]
```

然后评估：

```text
[4, 7, 8] 是否可能得到 24？ maybe
[2, 7, 8] 是否可能得到 24？ sure
[4, 8, 15] 是否可能得到 24？ unlikely
```

保留高分分支继续搜索。

这个任务体现了 ToT 的核心优势：

```text
明确 state
明确 thought/action
中间状态可评价
需要搜索和回溯
```

---

## 11. Creative Writing

### 11.1 任务是什么？

给模型几个随机句子，让它写一段连贯文章，并自然地融入这些句子。

例如：

```text
1. The clock struck midnight.
2. She found a key under the mat.
3. The garden smelled of rain.
4. Nobody answered the phone.
```

要求写出一篇连贯短文，包含这些句子。

---

### 11.2 为什么适合 ToT？

这个任务不是数学题，没有唯一答案。

它测试的是：

```text
开放式生成中的规划能力
```

如果模型直接开始写，可能写到后面发现某些句子很难自然插入，文章变得生硬。

---

### 11.3 ToT 怎么做？

ToT 可以先生成多个 high-level plan：

```text
Plan A: 写悬疑故事，钥匙是进入老宅的线索。
Plan B: 写家庭回忆，雨后的花园触发记忆。
Plan C: 写科幻故事，午夜电话来自未来。
```

然后评价这些 plan：

```text
哪个 plan 更能自然包含所有给定句子？
哪个故事更连贯？
哪个更有创造性？
```

保留最好的 plan，再扩展成完整文章。

这个任务说明：

```text
ToT 不只适合数学搜索，也适合开放式规划和生成。
```

---

## 12. Mini Crosswords

### 12.1 任务是什么？

小型填字游戏。

给出横纵线索，让模型填一个小网格。

例如：

```text
Across:
1. A domestic animal
2. Opposite of yes

Down:
1. ...
```

模型需要填出符合所有横纵约束的单词。

---

### 12.2 为什么适合 ToT？

填字游戏是典型约束满足问题。

一个格子填错，会影响横向和纵向多个词。

它需要：

```text
尝试
检查约束
发现冲突
回溯
换词
```

这正是 ToT 适合的场景。

---

### 12.3 ToT 怎么做？

每次填一个词可以看作 thought：

```text
Thought 1: Across 1 填 CAT
Thought 2: Across 1 填 DOG
Thought 3: Down 1 填 ...
```

每填一步得到一个新 state：

```text
当前网格状态
```

Evaluator 判断：

```text
当前网格是否和已有 clues 一致？
是否还有可能完成？
```

如果冲突，就剪枝或回溯。

这个任务体现了 ToT 对约束传播和回溯的能力。

---

## 13. Baseline 对照

ToT 的实验通常和这些方法比较：

### 13.1 Input-Output Prompting

```text
问题 -> 答案
```

不显式推理。

---

### 13.2 CoT Prompting

```text
问题 -> 一条推理链 -> 答案
```

---

### 13.3 CoT Self-Consistency

```text
多条完整推理链 -> 最终答案投票
```

---

### 13.4 Tree of Thoughts

```text
中间状态生成
中间状态评价
搜索/剪枝/回溯
```

这个对照非常清楚：

```text
直接回答
vs 单链推理
vs 多链投票
vs 树搜索推理
```

---

## 14. 论文最值得学习的亮点

### 14.1 把 LLM 从“回答器”变成“搜索组件”

在 CoT / Self-Consistency 里，LLM 主要是 solver：

```text
输入问题 -> 输出解法/答案
```

在 ToT 里，LLM 被拆成多个角色：

```text
generator: 提出候选 thought
evaluator: 评估中间状态
solver: 最终构造答案
```

LLM 不再只是一次性回答问题，而是被嵌入到一个搜索系统中。

这对后来的 Agent / Tool-use / Planning 都很重要。

---

### 14.2 引入中间状态评估

Self-Consistency 只看最终答案。

ToT 的亮点是：

```text
partial solution 也可以被评价
```

这让系统能提前剪掉没希望的路径，避免浪费完整生成。

---

### 14.3 把 prompt engineering 升级为 search engineering

CoT 更像 prompt engineering：

```text
怎么写 prompt 让模型一步步想？
```

Self-Consistency 是 decoding / sampling engineering：

```text
怎么采样多条链并投票？
```

ToT 更像 search engineering：

```text
thought 粒度怎么定义？
每步生成几个候选？
怎么评价 state？
保留多少分支？
用 BFS 还是 DFS？
什么时候停止？
```

这说明 ToT 已经不只是一个 prompt trick，而是系统设计。

---

### 14.4 连接 LLM 与经典 AI Search

ToT 借鉴了经典 AI 中的：

```text
state
action
transition
evaluation
search
backtracking
```

只不过 action/thought 和 evaluation 都由 LLM 通过语言完成。

这说明一个重要科研思路：

> 把经典算法思想迁移到 LLM 新范式中，可能产生有价值的方法。

---

## 15. ToT 的关键设计点

### 15.1 Thought 粒度

thought 粒度非常关键。

如果 thought 太小：

```text
一个 token 一个 thought
```

搜索空间会爆炸。

如果 thought 太大：

```text
一整条 CoT 一个 thought
```

又退化成 Self-Consistency。

合适的 thought 应该是：

```text
比 token 大
比完整解法小
可生成
可评价
可组合
```

例如：

- 24 点：一次算术操作
- 写作：一个故事大纲 / 一个段落方向
- 编程：一个调试假设 / 一个修改步骤
- 数学证明：一个 lemma / 一个变形步骤

---

### 15.2 State 表示

state 也需要根据任务设计。

例如：

```text
24 点：当前剩余数字
写作：当前大纲 / 已有段落
填字游戏：当前网格状态
代码调试：当前代码 + 错误信息 + 修复假设
```

ToT 不是固定 prompt，而是一个框架。具体任务要设计合适的 state 和 thought。

---

### 15.3 Evaluator 质量

ToT 很依赖 evaluator。

如果 evaluator 判断错了，就可能剪掉正确分支，保留错误分支。

因此 ToT 的质量受 evaluator 质量限制。

后续可以改进为：

- 用规则 evaluator
- 用 verifier model
- 用 reward model
- 用工具执行结果作为评价
- 用多个 evaluator 投票

这也自然连接到后续 verifier / PRM 方向。

---

## 16. 局限性

### 16.1 计算成本高

ToT 每一层都要：

```text
生成多个候选
评估多个候选
保留部分状态
继续扩展
```

如果深度大、候选多，调用次数会迅速增加。

例如：

```text
深度 = 4
每个状态生成 5 个候选
每层保留 3 个状态
```

模型调用成本会明显高于普通 CoT 和 Self-Consistency。

---

### 16.2 不适合所有任务

ToT 适合：

```text
可以拆成中间步骤
中间状态可以评价
存在搜索空间
可能需要回溯
```

不太适合：

```text
简单事实问答
一步就能完成的问题
中间状态难以评价的问题
```

因此关键问题是：

> 什么时候值得建树？什么时候 CoT 就够了？

---

### 16.3 Prompt 和任务设计成本高

不同任务需要设计不同的：

```text
thought 表示
state 表示
generator prompt
evaluator prompt
搜索策略
终止条件
```

所以 ToT 比 CoT 更复杂，更像系统工程。

---

## 17. 当前理解总结

可以把 Tree of Thoughts 总结为：

> Tree of Thoughts 是一种不修改模型参数的 inference-time search framework。它把 LLM 推理过程从单条 CoT 链扩展为一棵搜索树：每个 thought 表示一个中间推理步骤或候选操作，应用 thought 后得到新的 state；系统通过 LLM 生成候选 thought，再用 LLM 或规则评估 state 的前景，并使用 BFS/DFS/beam search 保留高价值分支、剪枝低价值分支，最终找到可行解。它的核心价值在于引入中间状态评估和搜索/回溯机制，使 LLM 能处理更需要规划和探索的问题。

更短一点：

```text
CoT 是一条链，
Self-Consistency 是多条链投票，
ToT 是把推理变成一棵可搜索、可评估、可回溯的树。
```

---

## 18. 和后续工作的关系

ToT 自然连接到后续多个方向：

```text
Tree-of-Thought
  ↓
Graph-of-Thought
  ↓
Verifier-guided search
  ↓
MCTS-style reasoning
  ↓
Agentic planning
  ↓
Tool-use / coding agent / environment feedback
```

它的重要性在于：

> 它展示了 LLM 可以不只是文本生成器，而是可以作为搜索系统中的生成器、评价器和规划组件。

这对 AI Agent、代码智能体、复杂任务规划都有启发。
