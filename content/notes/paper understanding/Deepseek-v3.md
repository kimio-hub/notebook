---
title: Deepseek-v3
---

# DeepSeek-V3 学习笔记

DeepSeek-V3 可以看作是在 Transformer 基础上，对注意力机制、MoE 结构和训练目标做了进一步改造。

## 1. 在 Transformer 基础上的主要改动
DeepSeek-V3 的几个核心关键词是：

- **MLA（Multi-head Latent Attention）**
- **DeepSeekMoE**
- **Auxiliary-Loss-Free Load Balancing**
- **MTP（Multi-Token Prediction）**

下面分别整理。

## 2. Multi-head Latent Attention（MLA）
传统 Multi-Head Attention 中，每个 token 都要显式维护完整的 $K$ 和 $V$ 表示。在长序列推理里，这会让 KV cache 占用很大。

DeepSeek-V3 的 MLA 思路是：
- 先把 Key/Value 压缩到一个低秩的 latent 表示；
- 再从 latent 表示恢复出各个 head 需要的表示；
- 从而减少 KV cache 的存储成本。


1. 原始隐藏状态先投影到一个较小的潜变量空间；
2. 这个潜变量承担“压缩后的 KV 信息”；
3. 真正做注意力时，再恢复出每个头所需的 Key/Value。

如果用抽象形式表示，可以写成：

$$
c_t^{KV} = W^{DKV} h_t
$$

其中：
- $h_t$ 是第 $t$ 个位置的隐藏状态；
- $W^{DKV}$ 是降维矩阵；
- $c_t^{KV}$ 是压缩后的 latent KV 表示。

之后再恢复出各头使用的 Key 和 Value：

$$
K_t = W^{UK} c_t^{KV}, \quad V_t = W^{UV} c_t^{KV}
$$

其中 $W^{UK}, W^{UV}$ 可以理解为“上投影矩阵”。

### 为什么要单独处理位置编码

> 对于位置编码，单独使用了极小的位置向量，从而防止时刻在变的位置信息无法被压缩。

原因是：
- 内容信息更适合被低秩压缩；
- 位置相关信息如果直接和内容混在一起压缩，可能会损失精度；
- 所以 DeepSeek-V3 会把位置相关部分拆出来，单独保留。

这本质上是在兼顾：
- **压缩率**；
- **位置建模能力**。

### 吸收权重

这里的直觉是：如果某些线性变换可以在训练后合并，那么推理时就不一定需要显式执行完整的“先解压、再乘权重”过程，而是可以把部分矩阵乘法吸收到一起，减少推理开销。

## 3. DeepSeekMoE
DeepSeek-V3 在 FFN 部分采用了 MoE（Mixture of Experts）结构，并做了两点重要设计。

### 3.1 更细粒度专家
- **finer-grained experts（更细粒度专家）**
- 把专家拆得更细；
- 单个 expert 参数量更小；
- token 可以更灵活地路由到合适专家。

这样做通常能提高专家分工的精细程度。

### 3.2 shared experts 与 routed experts 共存
DeepSeekMoE 里同时存在两类专家：

- **shared experts**：所有 token 都会经过，提供通用的 FFN 能力；
- **routed experts**：只对部分 token 激活，提供更专业、更稀疏的建模能力。

可以理解为：
- shared experts 负责“公共基础能力”；
- routed experts 负责“按 token 定制能力”。

如果写成形式化表达，MoE 层可以粗略写成：

$$
\text{MoE}(x) = \sum_{i \in \mathcal{S}} E_i(x) + \sum_{j \in \mathcal{T}(x)} g_j(x) E_j(x)
$$

其中：
- $\mathcal{S}$ 表示共享专家集合；
- $\mathcal{T}(x)$ 表示对输入 $x$ 选中的 routed experts；
- $g_j(x)$ 表示 gating/router 给出的权重；
- $E_i(x)$ 表示第 $i$ 个 expert 的输出。

## 4. Auxiliary-Loss-Free Load Balancing
传统 MoE 常用辅助损失（auxiliary loss）去平衡各个 expert 的负载，避免部分专家特别忙、部分专家几乎不用。

DeepSeek-V3 采用的是 **Auxiliary-Loss-Free Load Balancing**，即不额外引入辅助损失，而是直接通过 bias 调整路由倾向。

### 核心思路
给每个 expert 加一个 bias 项 $b_i$，影响 top-K 路由分数：

$$
s_i = g_i(x) + b_i
$$

然后根据 $s_i$ 选择 top-K experts。

在每个训练 step 结束后：
- 如果某个 expert 过载（overloaded），就把它的 bias 降低一个步长；
- 如果某个 expert 负载不足（underloaded），就把它的 bias 提高一个步长。

可以抽象写成：

$$
b_i \leftarrow b_i - \gamma \quad \text{if expert } i \text{ is overloaded}
$$

$$
b_i \leftarrow b_i + \gamma \quad \text{if expert } i \text{ is underloaded}
$$

其中 $\gamma$ 是调节步长。

### 直觉理解
这样做的好处是：
- 不需要额外加一个 balancing loss 去干扰主训练目标；
- 直接在路由分数层面调节专家使用频率；
- 让训练更稳定，也让目标函数更干净。

## 5. Multi-Token Prediction（MTP）
DeepSeek-V3 还引入了 **Multi-Token Prediction**。

传统语言模型通常只预测下一个 token：

$$
P(x_{t+1} \mid x_{\le t})
$$

而 MTP 会顺序地预测未来多个 token，并保持完整因果链。

### 5.1 核心思想
原文里的关键点是：
- **顺序地（sequentially）** 预测未来多个 token；
- 保持完整的因果链（complete causal chain）。

这意味着它不是简单并行地猜多个位置，而是让更深层的预测模块继续基于前面的预测深度进行建模。

### 5.2 MTP 模块组成
对于第 $k$ 个预测深度，作者引入一个 MTP module，通常包含：

- 共享 embedding layer；
- 共享 output head；
- 一个 Transformer block；
- 一个 projection matrix。

### 5.3 公式整理
原文中的核心公式可以整理成：

先把当前深度表示与未来 token 的 embedding 拼接，再做线性映射：

$$
h_{i}^{\prime,k} = M_k\big[\operatorname{RMSNorm}(h_i^{k-1});\operatorname{RMSNorm}(\operatorname{Emb}(t_{i+k}))\big]
$$

其中：
- $h_i^{k-1}$ 表示第 $k-1$ 个深度下位置 $i$ 的表示；
- $\operatorname{Emb}(t_{i+k})$ 表示未来第 $k$ 个 token 的 embedding；
- $M_k$ 是第 $k$ 个深度的投影矩阵。

然后经过当前深度的 Transformer 模块：

$$
h_{1:T-k}^{k} = \operatorname{TRM}_k\big(h_{1:T-k}^{\prime,k}\big)
$$

最后输出第 $k$ 个未来 token 的概率分布：

$$
P(t_{i+k} \mid t_{\le i}, \text{depth } k) = \operatorname{Softmax}(W_o h_i^k)
$$

这里 $W_o$ 是共享输出头。

### 5.4 MTP 的作用
MTP 的直觉收益主要有两点：

1. **增强训练信号**：一次前向传播能监督多个未来位置；
2. **提升表示能力**：模型不仅学“下一个 token 是什么”，也学“更远未来如何展开”。

## 6. 小结
DeepSeek-V3 的关键改动可以概括为：

- 用 **MLA** 压缩 KV 表示，降低长序列推理的 cache 成本；
- 用 **DeepSeekMoE** 提升 FFN 的稀疏表达能力；
- 用 **无辅助损失的负载均衡** 改善 MoE 路由分布；
- 用 **MTP** 增强训练目标，让模型顺序预测多个未来 token。

如果一句话总结：

> DeepSeek-V3 的设计目标，是在保持 Transformer 主体框架的同时，同时优化 **推理效率、专家路由质量和训练信号密度**。
