---
title: "Kimi K3 技术报告：沿 length、depth 与 width 扩展信息流"
description: "拆解 Kimi K3 的 2.8T MoE 架构：KDA–Gated MLA 混合注意力、Attention Residuals、Stable LatentMoE、原生视觉与训练基础设施。"
tags:
  - "llm"
  - "moe"
  - "architecture"
  - "tech-report"
category: "blog"
date: "2026-07-28"
updated: "2026-07-28"
---

> 资料来源：[Kimi K3: Open Frontier Intelligence](https://arxiv.org/abs/2607.24653)，Kimi Team，2026-07-27

K3模型信息

Kimi K3 是一个原生多模态稀疏 MoE 模型。它拥有约 2.8T 总参数，每个 token 激活约 104B 参数，最大上下文长度为 1M tokens。

| 项目 | Kimi K3 配置 |
|---|---:|
| 总参数量 | 2.78T |
| 单 token 激活参数量 | 104.2B |
| Transformer 层数 | 93 |
| 隐藏维度 | 7,168 |
| 上下文长度 | 最高 1M tokens |
| 注意力组成 | 69 层 KDA + 24 层 MLA |
| Routed Experts | 896，每个 token 激活 16 个 |
| Shared Experts | 每层 2 个 |
| Latent MoE 维度 | 3,584，即主干宽度的 0.5 倍 |
| 激活函数 | SiTU-GLU |
| 视觉编码器 | MoonViT-V2，401M 参数、27 层 |

如果只用一句话概括其架构思想：

 Kimi K3 不只是继续增加参数量，而是分别沿序列长度、网络深度和模型宽度三个方向，扩大信息能够被读取、传递和存储的范围。

三个方向分别对应：

- **Sequence length**：用 3:1 的 KDA–Gated MLA 混合注意力扩展跨 token 信息流。
- **Network depth**：用 Attention Residuals 扩展跨层信息流。
- **Model width**：用 Stable LatentMoE 扩展稀疏通道容量。

以及
- **Native vision**：用 MoonViT-V2 将图像和视频映射进同一个 token space。

报告认为，架构、数据和训练 recipe 的共同改进，使 Kimi K3 相比 Kimi K2 获得约 **2.5 倍整体 scaling efficiency**。

## 整体架构

Kimi K3 的基本 block 由以下结构组成：

1. 连续 3 层 Kimi Delta Attention（KDA）。
2. 接 1 层 Gated Multi-head Latent Attention（Gated MLA）。
3. 每个注意力层后都接一层 Stable LatentMoE。
4. Attention Residuals（AttnRes）让当前层从 embedding 和之前 block 的输出中选择信息。

主干末尾还额外放置一层 Gated MLA，确保最后一层一定执行一次全局注意力。

![Kimi K3 整体架构：三层 KDA 接一层 Gated MLA，每个注意力层后接 Stable LatentMoE；右侧粉色连线为跨 block 的 Attention Residuals。](/notes-assets/kimi-k3/pasted-20260728150250.png)

*Kimi K3 整体架构：三层 KDA 接一层 Gated MLA，每个注意力层后接 Stable LatentMoE；右侧粉色连线为跨 block 的 Attention Residuals。*

上图可以从三个方向阅读：

- 横向的 token mixing：KDA 与 MLA 决定不同位置的 token 如何交换信息。
- 纵向的 layer mixing：AttnRes 决定当前层应该读取哪些早期层表示。
- 通道方向的 channel mixing：Stable LatentMoE 决定 token 进入哪些专家，以及如何增加模型容量。

##  第一部分 Sequence length：Hybrid Attention

###  为什么采用 3:1 的 KDA–MLA 混合结构

标准全注意力允许任意两个 token 直接交互，但计算量和中间状态会随序列长度快速增长。在 1M token 场景中，如果 93 层全部使用全注意力，训练与推理成本会非常高。

Kimi K3 因此采用混合注意力：

- **KDA** 占大部分层，以线性递归状态高效处理长序列。
- **Gated MLA** 周期性出现，补充不受固定递归状态压缩限制的全局 token-to-token 交互。
- 3:1 的比例相当于用三层低成本长序列混合，搭配一层高容量全局信息校正。

这种结构并不是简单地用线性注意力替换全注意力，而是在效率与表达能力之间分工：KDA 负责“持续记忆”，MLA 负责“全局检索”。

### Kimi Delta Attention（KDA）

KDA 是一种带遗忘门的 delta-rule 线性注意力。对单个注意力头，其状态更新为：

$$
S_t = \left(I-\beta_t k_tk_t^\top\right)\operatorname{Diag}(\alpha_t)S_{t-1}
+\beta_t k_tv_t^\top,\qquad
\tilde{o}_t=S_t^\top q_t.
$$

其中：

- $S_t$ 是压缩历史信息的递归状态。
- $\alpha_t\in(0,1)^{d_k}$ 是逐通道保留系数，决定旧信息忘记多少。
- $\beta_t\in(0,1)$ 控制当前 token 的写入强度。
- $k_t$ 指定写入或修正的地址，$v_t$ 是写入内容，$q_t$ 用于读取状态。

直觉上，它像一个可学习的高速缓存：先按通道衰减旧记忆，再用 delta rule 做纠错式写入，最后用 query 从更新后的状态中读取。与标量遗忘门相比，channel-wise forget gate 允许不同通道拥有不同记忆时间尺度。

#### Chunkwise 并行

纯递归计算无法充分利用 GPU，因此 KDA 采用“chunk 间递归、chunk 内并行”：

- chunk 之间通过状态 $S$ 传递历史信息；
- 同一 chunk 内用下三角矩阵并行计算因果交互；
- 输出由 inter-chunk 历史状态和 intra-chunk 当前块交互两部分组成。

这样既保留线性递归的长上下文复杂度，又能在 chunk 内使用高吞吐矩阵乘法。

#### Lower-bounded decay

Kimi Linear 的 log-decay 没有下界，累计衰减可能变得极小，倒数缩放因子可能溢出，并迫使对角 tile 使用低效的逐位置 kernel。

Kimi K3 改用有下界的 scaled sigmoid：

$$
g_t=g_{\min}\operatorname{Sigmoid}(e^A z_t),\qquad
\alpha_t=e^{g_t},\qquad g_{\min}=-5.
$$

因此单步保留系数始终大于 $e^{-5}\approx6.7\times10^{-3}$。在 16-token tile 内，累计 log-decay 被限制在 $(-80,0)$，倒数缩放仍处于 BF16 动态范围。对角和非对角 causal tile 都可以直接使用 Tensor Core dense matrix multiplication，消除低效的 position-pair 路径。

#### Full-rank output gate

KDA 的递归输出经过逐通道门控：

$$
y_t=W_o\left[\operatorname{Sigmoid}(W_gx_t)\odot
\operatorname{RMSNorm}(\tilde{o}_t)\right].
$$

Kimi K3 将 Kimi Linear 的低秩 gate 改为输入相关的 full-rank projection，使不同 token 可以决定从递归状态中读取哪些通道。

### Gated MLA

MLA 把每个 token 的 Key 和 Value 压缩成低维 latent vector $c_t=W_cx_t$。推理时只缓存 $c_t$，计算注意力时再通过上投影恢复不同 head 所需的 K/V，因此能减少 KV cache，同时保留全局注意力能力。

Kimi K3 的 Gated MLA 有三个关键点：

1. **NoPE**：MLA 的 query 和 key 不使用显式位置编码；位置信息由相邻 KDA 层的递归门控与衰减隐式提供。
2. **Full-rank output gate**：当前 token 可以逐通道控制从全局注意力读取多少信息。
3. **FP32 attention output**：为修正 Flash Attention 中有偏的舍入误差，训练时 attention output 保持 FP32，并通过 kernel 重设计缓解片上存储压力。

KDA 与 MLA 的互补关系如下：

| 机制 | 主要作用 | 优点 | 局限 |
|---|---|---|---|
| KDA | 递归式长序列混合 | 序列扩展成本低，适合 1M context | 历史信息需压缩进固定状态 |
| Gated MLA | 全局 token-to-token 注意力 | 可直接检索任意位置，表达能力强 | 长序列计算和缓存成本更高 |

## 第二部分 Network depth：Attention Residuals

###  标准残差连接的深度瓶颈

标准残差连接把早期信息不断相加并压缩进单一状态。虽然它有利于梯度传播，但较深的层不能显式选择“我要读取哪一层的表示”。从信息流角度看，这与 RNN 把所有历史 token 压缩进一个隐状态很相似。

Transformer 用 attention 解决了序列维度的这一问题；AttnRes 则把同样的思想应用到深度维度。

###  Full Attention Residuals

第 $l$ 层使用可学习的 layer-specific pseudo-query $q_l=w_l$，并把 embedding 与之前各层输出作为 key/value：

$$
\alpha_{i\rightarrow l}
=\frac{\exp(q_l^\top\operatorname{RMSNorm}(k_i))}
{\sum_{j=0}^{l-1}\exp(q_l^\top\operatorname{RMSNorm}(k_j))},
\qquad
h_l=\sum_{i=0}^{l-1}\alpha_{i\rightarrow l}v_i.
$$

每个新层可以学习自己需要哪些深度的表示。RMSNorm 避免大幅值层仅凭数值尺度占据过高权重。

### Block Attention Residuals

完整 AttnRes 的算术量并不高，因为模型深度小于 100；真正的成本是保存所有层输出，以及在 pipeline-parallel stages 之间传输它们。

Kimi K3 使用 Block AttnRes：

- 将 93 层主干按约 12 层组成 block；
- block 内用 partial sum 表示当前 block 的累积状态；
- block 之间只保存一个聚合表示，并在 block-level representations 上做 attention；
- 主干使用 8 个层块；计入 embedding 这个独立信息源后，最终形成 9 个可选择的深度级表示。

保存和通信开销因此从 $O(Ld)$ 降到 $O(Nd)$，其中 $L$ 是层数，$N$ 是 block 数。当前层仍可以直接回看 embedding、历史 block 和当前 block 的 partial sum。

AttnRes 的本质可以概括为：

> 普通残差连接是“把过去全部相加”，AttnRes 是“从过去按需检索”。

## 第三部分 Model width：Stable LatentMoE

### LatentMoE

扩大专家池能够提高模型容量和专家分工的细粒度，但普通 MoE 会把完整的 $d$ 维 token 表示发送给所有被选中的专家。Top-$k$ 增大时，通信、专家计算与权重读取成本也会增加。

LatentMoE 把公共能力与专业能力分成两条路径：

- Shared experts 在完整隐藏维度 $d=7168$ 上处理通用变换。
- Routed experts 只在 latent space $\ell=3584$ 上处理专业变换。

其计算可简化为：

$$
u=\sum_{i\in T_k(x)}p_iE_i^{\text{routed}}(W_\downarrow x),
$$

$$
y=\sum_{j=1}^{N_s}E_j^{\text{shared}}(x)
+W_\uparrow\operatorname{RMSNorm}(u).
$$

Kimi K3 每层有 2 个 full-width shared experts、896 个 routed experts，每个 token 选择 16 个 routed experts。由于 routed experts 只接收半宽表示，它可以在不过度增加通信量的情况下扩大专家池。稀疏度为 $896/16=56$，即单 token 只使用全部 routed experts 的 $1/56$。

### Normalized LatentMoE

专家聚合结果的尺度会随专家选择和路由权重改变。Kimi K3 在 routed aggregation 与上投影之间加入 RMSNorm，使 routed branch 在与 shared branch 相加前具有更稳定的尺度。报告指出，这不仅提升训练稳定性，也持续改善验证损失和下游表现。

###  SiTU-GLU

SwiGLU 的两个乘法分支都没有上界；当两个分支同时出现大值时，乘积会形成 activation outlier，并增加低精度训练的溢出风险。Kimi K3 对 gate branch 和 up branch 分别使用 smooth cap：

$$
\operatorname{SiTU\text{-}GLU}(x)=
\left[
\beta_1\tanh\left(\frac{W_gx}{\beta_1}\right)
\odot\operatorname{Sigmoid}(W_gx)
\right]
\odot
\left[
\beta_2\tanh\left(\frac{W_ux}{\beta_2}\right)
\right].
$$

其中 $\beta_1=4$、$\beta_2=25$。它在原点附近近似线性，保留 SwiGLU 的局部响应；在大输入下逐渐饱和，将乘积绝对值限制在约 $\beta_1\beta_2=100$，从而抑制 activation explosion。

### Quantile Balancing

896 个专家会放大负载不均衡问题：过热专家拖慢 expert-parallel 训练，长期收不到 token 的专家则可能成为 dying expert。

Kimi K3 采用 auxiliary-loss-free routing。专家偏置 $b$ 只用于 Top-$k$ 选择，不进入最终 mixture weight，因此可以调节负载而不直接改变路由权重的梯度目标。

如果 batch 有 $m$ 个 token、$n$ 个专家、每个 token 选择 $k$ 个专家，则每个专家的目标负载为 $q=mk/n$。Quantile Balancing 根据专家相对 Top-$k$ cutoff 的 margin 分布，直接估计能够达到目标 token 数的 quantile bias。

## Native Vision：MoonViT-V2

Kimi K3 是原生多模态模型，而不是先完成纯文本预训练，再外挂视觉编码器。文本、图片和视频从训练开始就在同一个 next-token prediction 目标下联合优化。

视觉路径为：图像或视频 → MoonViT-V2 → 2×2 Pixel Shuffle → 轻量 MLP Projector → Kimi K3 共享 token space。

MoonViT-V2 的主要设计包括：

- 约 401M 参数、27 层、12 个注意力头、patch size 为 14；
- 使用 RMSNorm，并移除 linear 与 attention projection 中的 bias；
- 图像与视频完全共享参数；
- 视频 attention 拆为帧内空间 attention 和帧间时间 attention，并做 temporal pooling；
- projector 前使用 2×2 pixel shuffle，使视觉 token 数减少 4 倍；
- 支持最高约 3584×3584 像素输入，同时控制 1M context 中的视觉 token 成本。

与 Kimi K2.5 不同，MoonViT-V2 没有从 SigLIP 初始化，而是从零开始用 next-token prediction 联合训练。报告显示，从零训练的视觉编码器梯度范数更低、尖峰更少，同时视觉评测可追平 SigLIP 初始化基线。

这也服务于长程视觉 Agent：模型能够写代码、观察渲染截图或视频帧、发现问题并继续修改，无需在独立语言模型与视觉模型之间切换。
## Per-Head Muon

Kimi K3 延续 Kimi K2，使用 Muon 优化器。对于 Q、K、V attention projection，它进一步采用 Per-Head Muon：

- 普通 Muon 对完整投影矩阵的 momentum 一起做 Newton–Schulz orthogonalization。
- Per-Head Muon 按 attention head 切分矩阵，并对每个 head 单独正交化。

如果所有 head 一起处理，梯度或动量尺度更大的 head 可能主导共享更新方向，较小的 head 得不到充分归一化。逐 head 处理能够平衡更新尺度，提高大模型训练稳定性，同时略微降低正交化开销。

## Kimi K2 与 Kimi K3 的关键变化

| 项目              | Kimi K2 |        Kimi K3 | 变化含义                    |
| --------------- | ------: | -------------: | ----------------------- |
| 层数              |      61 |             93 | 增加深度，并由 AttnRes 改善跨层信息流 |
| 总参数             |   1.04T |          2.78T | 大幅增加模型容量                |
| 激活参数            |   32.6B |         104.2B | 提高单 token 实际计算容量        |
| 隐藏维度            |   7,168 |          7,168 | 不靠扩大主干 hidden size 完成扩展 |
| Routed Experts  |     384 |            896 | 更细粒度的专家分工               |
| 激活专家数           |       8 |             16 | 每个 token 组合更多专业变换       |
| Shared Experts  |       1 |              2 | 加强 full-width 公共路径      |
| Attention Heads |      64 |             96 | 增强多头表示能力                |
| 注意力             |     MLA | Hybrid KDA–MLA | 兼顾 1M 长上下文效率与全局交互       |
| 激活函数            |  SwiGLU |       SiTU-GLU | 控制极端激活，提高低精度稳定性         |
| 训练上下文           |    128K |             1M | 扩展 8 倍                  |
| 原生视觉            |       无 |     MoonViT-V2 | 从训练开始联合建模文本、图像和视频       |

一个值得注意的点是：Kimi K3 的隐藏维度仍为 7,168。它主要通过增加层数、注意力头、专家数和激活专家数进行扩展，而不是简单把所有 dense matrix 做得更宽。LatentMoE 与 AttnRes 分别缓解了宽度和深度扩展带来的通信与信息流问题。

## 1M 长上下文是如何实现的

Kimi K3 的 1M context 是架构、数据和训练课程共同作用的结果。

### 1 NoPE 与长度外推

Kimi K3 不使用显式位置嵌入。位置信息主要通过 KDA 的递归状态、门控和衰减隐式编码。因此，将上下文扩展到 1M 时，不需要重新调整 RoPE base，也不需要 YaRN 或位置插值。

### 2 Progressive context extension

上下文采用四阶段逐步扩展：

**8K（预训练起点）→ 64K（预训练后期）→ 256K（Cooldown）→ 1M（Cooldown）**

昂贵的超长序列训练只占总训练预算的一小部分，从而兼顾成本与长程适应能力。

### 3 长上下文数据

报告强调，单纯把短文本拼到 1M 长度并不会自然产生长程能力。Kimi K3 使用：

- 长文档和视频的去重、质量过滤与结构检查；
- 对稀缺的真实长文档和长视频进行上采样；
- 合成长程多模态任务，把解题信息分散在完整上下文的不同位置。

这样可以迫使模型真正跨越长距离寻找证据，而不是把 1M context 退化为局部窗口。

##  补充训练流程

### pre train 

- 数据覆盖 Web、代码、数学、知识和大规模视觉语料。
- 文本和视觉从训练开始共同使用 next-token prediction。
- 使用 Per-Head Muon、weight clipping、Quantile Balancing 和 cosine learning-rate decay。
- 学习率先进行 1% linear warmup，weight decay 为 0.1。

### post train

后训练采用三阶段流程：

1. SFT 建立基础指令遵循、工具调用和 Agent 能力。
2. 在 general tasks、general agents、coding agents 三个领域分别做 RL。
3. 每个领域训练 low、high、max 三种 reasoning effort，共得到 9 个专家策略。
4. 使用 Multi-Teacher On-Policy Distillation（MOPD）把能力整合进统一模型。

从 SFT 阶段开始，模型还进行 MXFP4 权重与 MXFP8 激活的量化感知训练。

### infra

2.8T 参数、896 个专家和 1M-token Agent trajectory 需要专门的系统协同设计：

- KDA fused kernels、KDA Context Parallelism 和 state-aware prefix caching；
- MoonEP 提供平衡的专家并行执行、静态计算形状和 zero-copy communication；
- Agentic RL 系统保存外部 KV cache、partial rollout 和可恢复的 microVM sandbox 状态；
- 部署侧通过专用 kernel 以及 cache-aware、budget-aware scheduling 控制线上成本。


 总结

通过混合线性/全局注意力扩展序列，通过 Attention Residuals 扩展深度信息访问，通过 Stable LatentMoE 扩展稀疏通道容量，再用原生视觉、Per-Head Muon 和训练基础设施把这些组件连接起来。
