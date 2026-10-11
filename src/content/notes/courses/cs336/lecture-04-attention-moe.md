---
title: "CS336 Lecture 4：注意力替代方案与混合专家"
description: "长上下文下的线性注意力、状态空间模型与混合架构，以及 MoE 的路由、负载均衡与 DeepSeek 系列设计。"
tags:
  - "cs336"
  - "attention"
  - "moe"
category: "courses"
series: "CS336"
date: "2026-07-10"
updated: "2026-07-10"
---

> **CS336 — Tatsu H**

---

## Part 1: Attention Alternatives

### 1.1 背景：长上下文的成本

随着上下文长度的增加，标准注意力的 $O(n^2)$ 成本变得难以承受。

**基本工具箱：**
- 结合局部（local）+ 全局（global）注意力
- 系统工程优化（Flash Attention 等）

---

### 1.2 Linear Attention（线性注意力）

#### 核心思想
标准注意力：$\text{Attention}(Q,K,V) = \text{softmax}(QK^T)V$

如果去掉 softmax（即 $\phi$ 为 identity）：$\text{Attention}(Q,K,V) = (QK^T)V$

利用矩阵乘法的结合律重排计算顺序：
$$\text{Attention}(Q,K,V) = Q(K^TV)$$

这从 $O(n^2 + n^2)$ 降到了 $O(n^2)$（实际上 $Q(K^TV)$ 是 $O(n)$ 的，因为 $K^TV$ 是固定大小的）。

#### 循环形式（RNN 对偶性）
线性注意力的优美之处在于它可以写成 RNN 形式：

$$S_t = S_{t-1} + k_t v_t^T, \quad o_t = S_t^T q_t$$

- **训练**：使用并行的二次形式（快）
- **推理**：使用串行的线性形式（高效）
- **扩展**：如果给 $S_{t-1}$ 加一个权重 $\gamma$，就得到 **RetNet**

---

### 1.3 从 Linear Attention 到 Mamba-2

在 Linear Attention 基础上添加**逐位置权重**：

| 方法 | 状态更新公式 |
|------|-------------|
| **Linear Attention** | $S_t = S_{t-1} + k_t v_t^T$ |
| **Mamba-2** | $S_t = A_t S_{t-1} + k_t v_t^T$，$o_t = S_t^T q_t + d_t \cdot x_t$，其中 $A_t = \exp(a_t)$ |

- 通过**门控（gating）** 使线性注意力更具表达力
- 继续保持对偶性（并行计算 $S$，串行推理）

**代表模型**：
- **Nemotron 3**：Mamba-Attention 混合（约 3:1），性能与类似模型相当或更好

### 1.4 Gated Delta Net（门控 Delta 网络）

进一步泛化：**门控输入 + 选择性擦除状态**

$$S_t = (1 - g_t) S_{t-1} + k_t v_t^T, \quad o_t = S_t^T q_t$$

其中 $g_t = \sigma(\cdot)$，$f_t = \text{sigmoid}(\cdot)$

Gated Delta Net 新增特性：
- **"无输入操作"门**（$f_t = 0$）
- 擦除当前 key 方向上的任何信息 $(1 - f_t)$

**代表模型**：**Qwen 3.5 / Qwen Next**（3:1 GDN/Attention 混合）

### 1.5 MiniMax M1
- 使用 **7:1 混合**（7 层 linear attention，1 层 full attention）
- 性能强劲，且上下文长度呈**线性缩放**

### 1.6 混合架构的性能分析
- 虽然有严格对照实验的论文不多
- 但有证据表明：**小的混合比例就能达到较低损失**

### 1.7 Sparse Attention / DeepSeek Sparse Attention (DSA)

**替代混合架构的思路**：不 attend 到每个 token，而是做**稀疏注意力**。

**DSA 设计**：
- 使用一个轻量的 **indexer** 来筛选需要 attend 的 token
- 可以获得显著的效率提升
- 可以在密集短上下文预训练后**事后适配（post-hoc adaptation）**
- 代表：DeepSeek v3.2, GLM-5

### 1.8 混合注意力变体总结

| 方法 | 核心思路 | 代表模型 |
|------|----------|----------|
| **Linear Attention** | $Q(K^TV)$ 重排，RNN 对偶 | - |
| **Mamba-2** | Linear Attention + 门控 + 状态矩阵 A | Nemotron 3 |
| **Gated Delta Net** | 门控 + 选择性擦除状态 | Qwen 3.5 |
| **MiniMax Hybrid** | 7 linear : 1 full | MiniMax M1 |
| **DSA** | 稀疏 indexer 筛选 token | DeepSeek v3.2 |

---

## Part 2: Mixture of Experts (MoE)

### 2.1 什么是 MoE？

把大的 FFN 替换成**多个 FFN（专家）** + 一个**门控选择器（router）**：

```
输入 → Router → 选择 Top-K 个 Expert → 加权合并 → 输出
```

- 可以增加 expert 数量而不增加 FLOPs（稀疏激活）
- 每个 token 只激活一部分 expert

> **GPT-4 据推测也使用了 MoE 架构**

### 2.2 为什么 MoE 现在越来越流行？

**1. 同样 FLOPs，更多参数 → 更好性能**
- ST-MoE [Fedus et al. 2022]：MoE 在相同计算量下表现更好

**2. 训练更快**
- OlMoE：MoE 可以更快地训练到同等质量

**3. 与密集模型高度竞争**
- 许多开源 MoE 模型的性能已达到甚至超过同等规模的密集模型

**4. 天然可并行**
- 每个 expert 可以放在不同的设备上

### 2.3 MoE 研究成果概览

| 来源 | 代表模型 | 特点 |
|------|----------|------|
| **西方** | Mixtral 8x7B, DBRX, Grok | 高性能开源 MoE |
| **中国（Qwen）** | Qwen MoE | 中小规模验证 |
| **中国（DeepSeek）** | DeepSeek MoE v1/v2/v3 | 系统性消融实验 |
| **DeepSeek v3** | 671B (37B active) | 当前最强开源 MoE 之一 |

### 2.4 为什么 MoE 没有更早流行起来？

**原因 1：基础设施复杂**
- All-to-All 通信在多节点上是瓶颈
- 需要**负载均衡**

**原因 2：训练目标不太明确（heuristic）**
- 离散路由不可微
- 有时训练不稳定

---

### 2.5 MoE 的通用结构

```
Transformer Layer (标准)           Transformer Layer (MoE)
┌───────────────────┐            ┌─────────────────────────┐
│    Self-Attention  │            │    Self-Attention        │
│         ↓          │            │         ↓               │
│       Norm         │            │       Norm              │
│         ↓          │            │         ↓               │
│    FFN (单层)      │    →       │    Router + Top-K       │
│         ↓          │            │      ↓    ↓    ↓        │
│       Norm         │            │   Expert₁ ... Expertₙ   │
└───────────────────┘            │      ↓    ↓    ↓        │
                                 │   加权合并 (Combine)     │
                                 │         ↓               │
                                 │       Norm              │
                                 └─────────────────────────┘
```

**典型做法**：将 MLP 替换为 MoE 层
**较少见**：对 attention 头做 MoE（ModuleFormer, JetMoE）

---

### 2.6 MoE 中变化的要素

1. **路由函数（Routing Function）**
2. **Expert 大小**
3. **训练目标**

---

### 2.7 路由算法

#### 主流：Top-K 路由（Token Choice）

| 变体 | 说明 | 使用模型 |
|------|------|----------|
| **Top-1** | Switch Transformer | Switch Transformer |
| **Top-2** | Grok, Mixtral | Grok, Mixtral |
| **Top-4** | Qwen, DBRX | Qwen, DBRX |
| **Top-7** | DeepSeek MoE | DeepSeek |

#### 其他路由方法

| 方法 | 说明 | 现状 |
|------|------|------|
| **Hashing** | 确定性路由 | 常用 baseline |
| **RL 学习路由** | 强化学习优化门控策略 | 早期工作，现在不常用 |
| **线性分配** | 求解匹配问题 | Clark et al. '22 等论文使用 |

#### Top-K 路由的细节

```
Gating:   g_i(x) = softmax(TopK(x · W_g, k))
```

- DeepSeek (V1-2) / Grok / Qwen：用 logistic regressor 选门
- Mixtral / DBRX / DeepSeek v3：softmax 在 TopK 之后

#### DeepSeek 风格的变体
- **更小、更多的 expert** + **少量共享 expert（始终激活）**
- 共享 expert 处理所有 token 都需要的通用知识
- 来源：DeepSpeed MoE
- 使用模型：DeepSeek / Qwen

#### 消融实验发现
- DeepSeek 论文：**更多 expert + 共享 expert 通常有帮助**
- OlMoE 论文：**细粒度 expert 有效**，共享 expert 无额外收益

#### 各 MoE 的路由配置

| 模型 | Routed Experts | Active | Shared | Fine-grained ratio |
|------|:--------------:|:------:|:------:|:------------------:|
| GShard | 2048 | 2 | 0 | - |
| Switch Transformer | 64 | 1 | 0 | - |
| ST-MoE | 64 | 2 | 0 | - |
| Mixtral | 8 | 2 | 0 | - |
| DBRX | 16 | 4 | 0 | - |
| Grok | 8 | 2 | 0 | - |
| DeepSeek v1 | 64 | 6 | 2 | 1/4 |
| Qwen 1.5 | 60 | 4 | 4 | 1/8 |
| DeepSeek v3 | 256 | 8 | 1 | 1/14 |
| OlMoE | 64 | 8 | 0 | 1/8 |
| MiniMax | 32 | 2 | 0 | ~1/4 |
| LLaMA 4 (Maverick) | 128 | 1 | 1 | 1/2 |

---

### 2.8 如何训练 MoE？

**核心挑战**：稀疏门控决策**不可微**！

#### 解决方法 1：RL（REINFORCE）
- 确实有效 [Clark et al. 2020]
- 但梯度方差大，复杂度高 → **不广泛使用**

#### 解决方法 2：随机近似（Stochastic Approximation）
- Shazeer et al. 2017：在路由决策中加入高斯扰动
- Fedus et al. 2022：均匀乘性扰动（后移除）

#### 解决方法 3（最常用）：启发式平衡损失

**负载均衡损失（Auxiliary Balancing Loss）**：
$$\mathcal{L}_{\text{balance}} = \alpha \cdot \sum_{i=1}^{E} f_i \cdot P_i$$

其中 $f_i$ 是分配给 expert $i$ 的 token 比例，$P_i$ 是 router 给 expert $i$ 的平均概率。

- 更频繁使用的 expert 会得到更强的降权
- 来自 **Switch Transformer**

#### DeepSeek 的负载均衡策略

**V1-2**：
- Per-expert balancing（同 Switch Transformer）
- Per-device balancing（按设备聚合）

**V3：Auxiliary-Loss-Free Balancing**
- 设置每个 expert 的 bias，通过在线学习动态调整
- 使被选中的 expert 更易再次被选中
- 不过并不是完全无 aux loss

**如果去掉负载均衡损失会怎样？**
- 训练不稳定，部分 expert 死亡（未被分配到 token）

---

### 2.9 MoE 的系统层面

#### 并行化
- 每个 FFN 可以放在一个设备上
- 支持**额外类型的并行**（Expert Parallelism）

#### 通信挑战
- All-to-All 通信是瓶颈
- 现代库如 **MegaBlocks** 使用智能的稀疏矩阵乘法

#### 新思路
- **Nemotron 3**：下投影（down-projecting）激活值以减少通信量

### 2.10 MoE 的额外随机性

MoE 比普通模型有更多的随机性，原因：
- Token 丢弃在**批次层面**进行路由
- **其他人的 query 可能让你的 token 被丢弃！**

### 2.11 MoE 的稳定性问题

- 路由器的 softmax 可能不稳定
- **解决方案**：路由器使用 **Float32**（有时加上 aux z-loss）
- 移除了 z-loss 后，训练可能崩溃

### 2.12 MoE 的微调问题

- **稀疏 MoE 在小规模微调数据上容易过拟合**
- Zoph et al. 的解决方案：只微调**非 MoE 的 MLP**
- DeepSeek 的解决方案：使用**大量数据**（1.4M SFT 样本）

---

### 2.13 升级（Upcycling）

**问题**：能否用预训练好的密集模型初始化 MoE？

**方法**：将预训练模型的 FFN 复制多份作为 expert 的初始化。

**成功案例**：

**MiniCPM**：
- 基于 MiniCPM 模型（Top-2, 8 experts, ~4B active params）
- 训练约 520B tokens 后，相比基础模型有明显提升

**Qwen MoE**：
- 从 Qwen 1.8B 模型初始化
- Top-4, 60 experts + 4 shared
- 首批确认成功的 upcycling 案例之一

---

### 2.14 DeepSeek MoE 架构演进

#### V1（16B, 2.8B active）
- Shared (2) + Fine-grained (64/4) experts
- 标准 Top-K 路由
- 标准 Aux-loss balancing

#### V2（236B, 21B active）
- Shared (2) + Fine-grained (160/10) experts, 6 active
- **新增**：
  - 通信平衡损失（balancing both communication in and out）
  - Top-M device routing

#### V3（671B, 37B active）
- Shared (1) + Fine-grained (258) experts, 8 active
- **新增**：
  - Aux-loss-free + sequence-wise aux
  - Sigmoid + Softmax Top-K + Top-M

---

### 2.15 DeepSeek V3 的配套技术

#### MLA（Multi-head Latent Attention）
- 将 Q, K, V 表示为低维**潜在激活**的函数
- KV 缓存只需存储压缩后的潜在向量，大大减小缓存
- 挑战：RoPE 与 MLA 式缓存冲突
  - 无 RoPE：$k_c = W_{UK}h, k_c = W_{UK}h$
  - 有 RoPE：$k_c = [k_c^{(1)}; R_{\Theta} k_c^{(2)}]$
  - **解决方案**：保留少量可旋转的非潜在 key 维度

#### MTP（Multi-Token Prediction）
- 用轻量级模型预测**未来的多个 token**
- 但实际只预测一个 token 前向
- 灵感来自 EAGLE

---

### 2.16 MoE 总结

| 方面 | 要点 |
|------|------|
| **核心优势** | 利用稀疏性——不是所有输入都需要完整模型 |
| **路由** | 离散路由困难，但 Top-K 启发式方法有效 |
| **实证** | 大量证据表明 MoE 有效且成本效益好 |
| **挑战** | 基础设施复杂、训练不稳定、微调容易过拟合 |
| **趋势** | 更细粒度的 expert + 共享 expert + 无 aux loss 路由 |
