---
title: "CS336 Lecture 8：并行基础与大规模训练"
description: "网络与 collective、ZeRO/FSDP、PP/TP/SP/CP/EP，一路到 3D/4D 并行与真实大模型的训练配置。"
tags:
  - "cs336"
  - "distributed"
  - "training"
category: "courses"
series: "CS336"
date: "2026-07-11"
updated: "2026-07-11"
---

> Stanford CS336 Spring 2026 · 73 页课件完整主线：网络与 collective → ZeRO/FSDP → PP/TP/SP/CP/EP → 3D/4D 并行与真实大模型。

## 0. 本讲要解决的问题

单 GPU 有两堵墙：计算量有限、显存有限。进入大模型训练后，“计算机”不再是一张 GPU，而是整个数据中心。我们希望 GPU 数增加时同时得到：

- **线性计算扩展**：总 FLOPs 近似随 GPU 数增长；
- **线性内存扩展**：可训练参数/状态/activation 总量随 GPU 数增长；
- **高利用率**：新增 GPU 不是大部分时间等待通信。

没有一种并行方法同时在显存、带宽、延迟、batch size 和工程复杂度上最优，所以真实训练采用多维并行。

---

## Part 1. 网络与 collective 基础

### 1.1 关键 collective

课件集中比较 all-reduce、broadcast、reduce、all-gather、reduce-scatter。最关键等式是：

$$
\text{AllReduce}=\text{ReduceScatter}+\text{AllGather}.
$$

在大消息、带宽受限场景，这个分解已经接近最优数据移动量。ZeRO 的根本洞察是：既然 reduce-scatter 后结果天然分片，就不必立刻把所有状态恢复成每卡完整副本。

### 1.2 拓扑不是细节

TPU 传统上偏 mesh/torus，通信结构规则且成本可控；GPU 数据中心常有节点内全互联/NVSwitch 与节点间 switched network。Mesh 便宜且适合结构化邻居通信；tree/全交换拓扑对 expert routing 等不规则通信更友好。新 TPU 也逐渐引入更接近 switched network 的扩展结构，反映 MoE/all-to-all 的需求。

不能把所有设备无限全连接，因为交换机端口、线缆、功耗、成本、物理距离和故障域都会爆炸。因此系统被划成不同 **communication domains**，并行策略必须匹配域大小。

---

## Part 2. 数据并行与 ZeRO

## 2. Naive Data Parallel / DDP

全局 batch $B$ 分到 $M$ 张 GPU，每卡计算 $B/M$ 个样本，随后同步梯度：

$$
\theta_{t+1}=\theta_t-\eta\sum_{i=1}^{B}\nabla f(x_i).
$$

优点是计算随 GPU 数扩展；通信每 step 与参数量同阶；最大问题是每卡复制完整训练状态，显存没有参数级扩展。

### 2.1 每参数到底占多少内存

经典混合精度 Adam 可能包含：

- BF16/FP16 参数：2 bytes；
- BF16/FP16 梯度：2 bytes；
- FP32 master weights：4 bytes；
- Adam first moment：4 bytes；
- Adam second moment：4 bytes。

合计可达 16 bytes/param，尚未计算 activation、临时 buffer、allocator fragmentation。纯 BF16、Kahan summation 或低精度 optimizer 可降低常数，但不会改变“完整复制”的结构问题。

## 3. ZeRO Stage 1：切 optimizer state

每卡仍有完整参数和梯度，但只保存 $1/P$ 的 optimizer state，并负责更新对应参数 shard。

一步流程：

1. 各卡对本地数据算完整梯度；
2. reduce-scatter 梯度，每卡得到负责 shard 的聚合梯度；
3. 每卡用本地 optimizer state 更新本地参数 shard；
4. all-gather 更新后的参数，使每卡重新获得完整模型。

DDP 的 all-reduce 数据量约 $2N$；ZeRO-1 的 reduce-scatter + all-gather 也约 $2N$。因此在 bandwidth-bound 近似下，ZeRO-1 用相同通信换来 optimizer-state 显存下降，几乎是“免费”的内存收益。

若参数与梯度固定部分按课件记为 4 bytes/param，optimizer state 为 $K$ bytes/param：

$$
M_{\text{ZeRO-1}}\approx \left(4+\frac{K}{P}\right)N.
$$

## 4. ZeRO Stage 2：再切 gradient

参数仍完整复制，梯度与 optimizer state 分片。Backward 不能长期实例化完整梯度：

1. 某层梯度一产生就 reduce-scatter 到负责 rank；
2. 本地不再需要的完整梯度立即释放；
3. 各 rank 更新参数 shard；
4. all-gather 参数。

通信仍约 $2N$，额外节省完整梯度副本。难点在于 hook、bucket、梯度生命周期和通信计算 overlap。

## 5. ZeRO Stage 3 / FSDP：全部切分

参数、梯度、optimizer state 都只保留 $1/P$。执行某个 FSDP block 时按需 all-gather 参数，用完立即 reshard/free；backward 再次按需获取参数，梯度通过 reduce-scatter 分片保存。

Baby version 的参数通信：

- forward 参数 all-gather：$N$；
- backward 参数 all-gather：$N$；
- backward gradient reduce-scatter：$N$。

总计约 $3N$，相对 DDP 的 $2N$ 是 1.5 倍，但参数相关状态近似降到 $1/P$。实际 FSDP 通过 prefetch 下一 block 的参数，把 all-gather 隐藏在当前 block 计算后面。

### 5.1 为什么不能一次 gather 整个模型

那会瞬间恢复完整模型显存，失去分片意义。正确做法是以 module/block 为单位：gather → compute → reshard，并限制同时 materialize 的参数量。

### 5.2 课件中的容量直觉

假设每参数训练状态共 12 bytes，在 8×A100 80GB 上忽略 activation/碎片时，课件估算：baseline 约 6.7B 参数，ZeRO-1 约 16B，ZeRO-2 约 24.6B，ZeRO-3 约 53.3B。真实可用规模更小，因为 activation 和通信 buffer 也占显存。

### 5.3 ZeRO 仍未解决什么

- ZeRO-1/2 不分片参数，超大模型仍可能放不下；
- ZeRO-3 分片参数但不自动降低 activation memory；
- Data parallel 需要把 global batch 随 DP degree 扩大，超过 critical batch 后收益递减；
- 小 batch 时每卡计算少，参数通信难以摊薄。

---

## Part 3. Model Parallelism

ZeRO-3 的哲学是“传参数，算完整层”；model parallel 的哲学是“参数固定分片，传 activation”。模型可按深度、宽度、序列或 experts 切开。

## 6. Pipeline Parallelism：沿深度切

层分配给不同 stages，边界传 activation，反向传 activation gradient。没有 microbatch 时，$P$ 张卡中每张约 $1/P$ 时间工作。

把 batch 切为 $M$ 个 microbatches 后形成流水线，简单调度的 bubble/有效计算比约：

$$
\frac{P-1}{M}.
$$

所以 PP 对 batch/microbatch 数敏感。它的优势是：

- 参数显存约缩到 $1/P$；
- 通信是相邻 stage 的点对点 activation，大小约 $bsh$；
- 相比 FSDP 频繁传参数，更适合较慢的跨节点链路。

代价是调度复杂、bubble、stage 负载不均和 activation 缓存。更复杂的 interleaved schedule 可提高利用率，但增加通信和实现难度。

### 6.1 Zero-bubble 的直觉

Backward 可拆成：

1. 对输入 activation 的梯度，必须尽快传给前一 stage；
2. 对权重的梯度，只要 optimizer step 前算完即可。

第二部分调度更自由，可以塞进原本的空泡，从而接近 zero-bubble。

## 7. Tensor Parallelism：沿宽度切

对连续两层 MLP，可把第一层按列切、第二层按行切：每卡持有子矩阵，局部 matmul 后在必要边界做 all-reduce/reduce-scatter。Transformer 常见：

- Column parallel：QKV、MLP up/gate projection；
- Row parallel：attention output、MLP down projection；
- Replicated：norm、router 等小操作。

TP 无 pipeline bubble，不要求很大 batch，包装线性层即可表达；但每个 block 都有 activation collective。课件给出的比较：PP 每 microbatch 传约 $bsh$ 的点对点数据，TP 每层可能传约

$$
8bsh\frac{P-1}{P}
$$

量级的 collective 数据。因此 TP 必须放在低延迟、高带宽域内，实践中常先用满单节点 8 卡，而不是跨大量节点继续增大。

## 8. Activation memory 与重计算

训练显存不仅有静态状态，还有为 backward 保存的 activation。普通 attention 中与 $s^2$ 有关的 attention matrix/dropout 项可非常大；FlashAttention 或 recomputation 能避免保存这些二次项。

**Activation checkpointing/recomputation**：forward 不保存全部中间结果，backward 时重算。它用额外 FLOPs 换显存，且释放的显存有时允许增大 microbatch，使 matmul 更高效，最终吞吐甚至可能提高。

但 TP 只切了 attention/MLP 的矩阵乘，LayerNorm、dropout、residual 和 block 输入等逐 token activation 仍可能每卡复制，不能获得完整线性显存缩放。

## 9. Sequence Parallelism

观察：LayerNorm、dropout 等点运算对不同 token 独立，可以沿 sequence 维切。进入需要 hidden-width 分片的 TP 线性层前做 all-gather，离开时用 reduce-scatter 恢复 sequence shard；backward 中方向相反。

SP 与 TP 配合后，相关 activation 也能近似按 TP degree 缩小。它处理的是普通长度下逐 token activation，并不等同于为超长上下文设计的 context parallel。

## 10. Context Parallelism / Ring Attention

沿长序列切 token，并让不同设备协作完成 attention。每卡只保留一段 query/KV 或 KV cache，通过 ring/collective 交换必要块，降低长上下文 activation/KV 显存。代价是 attention 层新增跨卡通信，且 causal mask、load balance 与通信重叠更复杂。

## 11. Expert Parallelism

MoE 不把一个 matmul 切细，而是把不同 experts 放到不同 GPU：router 决定 token → expert，all-to-all dispatch token，expert 本地计算，再 all-to-all combine 返回。

为什么可能优于 TP：完整 expert matmul 保持较大形状，计算效率好；但要求每个 expert 有足够 token，并且 routing 平衡。EP degree 可以很大，但网络和软件挑战也更大。

### 11.1 EP 与其他并行的交互

- EP 只作用于 MLP，attention 仍需 TP/CP/DP；
- attention 可能希望高 TP，expert MLP 更希望低 TP + 高 EP；
- 因而现代框架允许 attention 使用 TP/CP/DP，而 MoE 部分使用 ETP/EP/EDP；
- DP replica 与 EP group 的划分必须仔细设计，否则 token 太少或 matmul 太碎，利用率下降。

---

## 12. 方法对照表

| 方法 | 参数/状态显存 | Activation/KV | 主通信 | Batch 约束 | 常放在哪里 |
|---|---|---|---|---|---|
| DDP / ZeRO-1 | 参数不缩；ZeRO-1 切 optimizer | 不缩 | 每 step 梯度 all-reduce | global batch 随 DP 增 | 任意，最好高速 |
| FSDP / ZeRO-3 | 参数/梯度/state 约 $1/DP$ | 不自动缩 | 按层参数 all-gather + grad reduce-scatter | DP 扩 batch | 高带宽域优先 |
| Pipeline | 参数约 $1/PP$ | 与调度/buffer 有关 | stage 间 activation P2P | 需足够 microbatches | 可跨节点 |
| Tensor | TP 权重约 $1/TP$ | 配 SP 可下降 | 每 block activation collective | 不要求大 batch | 节点内 NVLink |
| Sequence | 参数不变 | 逐 token activation 约 $1/SP$ | per-layer gather/scatter | 无 | 常与 TP 同组 |
| Context | 参数不变 | 长序列 activation/KV 约 $1/CP$ | attention KV/块交换 | 长上下文 | 高带宽组 |
| Expert | expert 权重约 $1/EP$ | token buffer | MoE all-to-all | 每 expert 要足够 token | 可多节点但很难 |

---

## Part 4. 3D/4D 并行组合

总 GPU 数常近似分解为：

$$
N_{GPU}=DP\times PP\times TP
$$

MoE/长上下文时再加入 EP、CP，但这些 group 可能与 DP/TP 重叠，不能总是简单全部相乘。

### 13.1 实用规则

1. **先让模型放得下**：节点内用 TP/EP，跨节点用 PP，或在带宽足够时用 FSDP；
2. **再扩大吞吐**：剩余 GPU 用 DP；
3. 小 microbatch 时用 gradient accumulation 增加 global batch、摊薄通信；
4. 高频 blocking collective 置于最快链路；
5. TP 通常先增到每节点 GPU 数后封顶；
6. PP stage 要按真实层耗时和显存均衡，不只是平均分层数；
7. 用 activation recomputation 换显存，可能允许更大高效 batch。

Narayanan/Megatron 的实验展示，精心选择 TP×PP×DP 后，GPU 数增加仍能保持近似平坦的利用率。课件中 64 设备场景常以 8×8 的节点内 TP 与节点间维度组合为优，强调“拓扑感知”。

---

## 14. 真实模型采用什么

课件列出的模式比具体数字更重要：

- **DeepSeek / DeepSeek-V3**：ZeRO-1，V3 使用约 PP=16、EP=64，并对 all-to-all 做 1F1B overlap；
- **Yi / Yi-Lightning**：ZeRO-1 + TP + PP；后续 MoE 更偏 EP 替代部分 TP；
- **Llama 3 405B**：约 DP=128、TP=8、PP=16，训练分常规预训练与长上下文等阶段；
- **Gemma 2**：ZeRO-3 + model parallel + DP；
- **Mixtral 8×22B**：示例 TP/PP/CP/EP = 4/4/1/8，另有 DP；
- **Nemotron 3 长上下文**：较小 TP、很大 EP 与 CP；
- **Qwen 3 MoE**：以 EP 为主，大模型示例含 TP=2、EP=32、PP=8。

共性：TP 通常 $\le 8$；EP 可以很大但实现困难；进入长上下文阶段会显著增加 CP；并行配置会随训练阶段变化，而非从头到尾固定。

## 15. 大规模训练的故障现实

数千 GPU 长时间训练时，GPU、NIC、链路、节点和软件进程故障是常态。大规模效率不仅是单 step 吞吐，还包括：

- checkpoint 频率与保存带宽；
- 故障检测和自动重启；
- 从最近 checkpoint 恢复的时间；
- straggler 检测；
- 数据加载状态和随机数状态的一致恢复；
- silent data corruption 与 loss spike 监控。

设备越多，平均无故障运行时间越短；因此 resilience 是并行系统的一部分。

---

## 16. 怎么为新模型选并行策略

按以下顺序做资源核算：

1. 参数、梯度、optimizer state 各多少 bytes/param？
2. 给定 $b,s,h,L$，activation 与 attention/KV 占多少？
3. 单卡是否放得下？若不能，瓶颈是参数状态还是 activation？
4. 节点内/节点间带宽与延迟分别多少？
5. global batch 是否能继续增大？
6. 模型是 dense、MoE 还是超长上下文？

对应决策：

- 状态大：ZeRO/FSDP；
- 单层矩阵也太大：TP；
- 层很多且跨节点：PP；
- LayerNorm/residual activation 大：SP；
- 序列/KV 大：CP；
- experts 大：EP；
- 模型放下后还有 GPU：DP。

最后必须用真实 shape benchmark，因为理论通信量相同并不代表延迟、kernel 效率、overlap 和拓扑相同。

## 17. 易错理解

- ZeRO-3 与 TP 都切参数，但 ZeRO-3 临时 gather 完整层、TP 始终计算分片层。
- ZeRO stage 越高，内存越省，但通信次数、延迟敏感性和实现复杂度也上升。
- PP 通信字节少不代表一定快；microbatch 太少时 bubble 会压倒优势。
- TP 没有 bubble 不代表可无限扩；本地 matmul 变小和 collective 延迟会使扩展饱和。
- SP 与 CP 都沿序列切，但解决的算子与通信模式不同。
- EP 的参数扩展很漂亮，但 token imbalance 会让部分 GPU 空闲、部分过载。
- “线性 scaling”应说明是 strong scaling、weak scaling、memory scaling 还是 throughput scaling。

## 18. 自测

1. ZeRO-1 为什么通信量与 DDP 近似相同，却节省显存？
2. ZeRO-3 一次训练 step 为什么通常需要两次参数 all-gather？
3. FSDP 与 TP 的“切参数”有何本质差别？
4. PP 为什么适合慢链路，为什么又依赖大量 microbatch？
5. TP 后为什么还需要 SP 才能让 activation 显存更线性？
6. 超长上下文为什么需要 CP/Ring Attention？
7. MoE 为什么把系统瓶颈从 matmul 部分转向 all-to-all 与 load balance？
8. 为什么常见配置先把 TP 开到 8，再增加 PP/DP？

## 19. 最终总结

1. 数据中心是大模型训练的新计算单元。
2. DDP 扩计算，ZeRO/FSDP 扩训练状态显存。
3. PP 切深度、TP 切宽度、SP/CP 切序列、EP 切 experts。
4. 真实系统用多维并行，并把不同通信放到匹配的网络层次。
5. 目标不是最少通信或最省显存中的单项最优，而是模型能放下、GPU 持续忙、训练可恢复的整体最优。

## 资料

- [课程官方 Lecture 8 PDF](https://github.com/stanford-cs336/lectures/blob/main/lecture_08.pdf)
- [ZeRO 论文](https://arxiv.org/abs/1910.02054)
- [PyTorch FSDP 教程](https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html)
- [Megatron-Core MoE 指南](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/features/moe.html)



---

## 20. 逐页覆盖版：Lecture 8 的 73 页分别讲了什么

### 20.1 第 1–13 页：动机、网络与 collective

| 页 | 内容 | 必须理解的结论 |
|---:|---|---|
| 1 | Parallelism Basics | 本讲讨论多 GPU/多机训练 |
| 2 | Goals | 理解大模型训练复杂性、并行范式及组合 |
| 3 | 三部分结构 | 网络基础→并行方法→大模型案例 |
| 4 | 单 GPU compute 极限 | 超算级训练必须聚合大量设备 |
| 5 | 单 GPU memory 极限 | 参数只是显存的一部分，超大模型无法单卡容纳 |
| 6 | Multi-GPU/Multi-machine | 用高速互联分摊 memory 和 compute |
| 7 | Collective 总览 | all-reduce、broadcast、reduce、all-gather、reduce-scatter |
| 8 | All-reduce 分解 | bandwidth-bound 下 reduce-scatter+all-gather 接近最优 |
| 9 | TPU mesh vs GPU switched topology | 拓扑决定适合的通信模式 |
| 10 | Mesh/tree 比较 | mesh 便宜规则；tree/A2A 适合不规则 expert routing |
| 11 | TPU8i/t、Virgo | TPU 也向 switched scale-out 网络演进 |
| 12 | 为什么不全连接 | 端口、线缆、成本、功耗和故障域限制 domain size |
| 13 | Part 1 recap | 数据中心成为新的计算单元；目标是线性 memory/compute scaling |

### 20.2 第 14–30 页：DDP 与 ZeRO 1–3

| 页 | 内容 | 必须理解的结论 |
|---:|---|---|
| 14 | 方法总览 | DP、ZeRO、PP、TP、activation/sequence parallel |
| 15 | Naive DP | batch 切分，梯度同步；compute 扩展、memory 不扩展 |
| 16 | DP 的问题 | 每卡复制完整训练状态 |
| 17 | 16 bytes/param | BF16 param/grad + FP32 master + Adam m/v |
| 18 | ZeRO 核心 | 分片昂贵 state，并利用 collective 分解 |
| 19 | ZeRO-1 定义 | 只切 optimizer state |
| 20 | ZeRO-1 流程 | full grad→reduce-scatter→local update→all-gather params |
| 21 | ZeRO-1 vs DDP | 同约 $2P$ 通信，显存更低 |
| 22 | ZeRO-2 定义 | optimizer state + gradient 分片 |
| 23 | ZeRO-2 流程 | layer grad ready 后立即 reduce/free |
| 24 | ZeRO-3/FSDP | 参数、梯度、optimizer 全分片 |
| 25 | ZeRO-3 baby timeline | forward/backward 按需 gather，gradient reduce-scatter |
| 26 | 实际 FSDP | incremental materialization + comm/compute overlap |
| 27 | 通信比较 | DDP/ZeRO1/2 约 $2P$；ZeRO3 约 $3P$ |
| 28 | 8×A100 容量示例 | baseline 6.7B→ZeRO3 理论 53.3B，未计 activation |
| 29 | DP compute 上限 | DP degree 受 global batch/critical batch 限制 |
| 30 | 仍放不下 activation | 需要真正的 model parallel/activation 分片 |

这里 $P$ 在“$2P$ communication”中表示参数字节量，不是 world size；为避免混淆，下面统一用 $\Psi$ 表示模型参数量、$d$ 表示 DP degree。

### 20.3 第 31–43 页：Pipeline 与 Tensor Parallel

| 页 | 内容 | 必须理解的结论 |
|---:|---|---|
| 31 | Model parallel 定义 | 参数固定分片、主要传 activation |
| 32 | Layer-wise parallel | 沿深度分层，前向/反向传 activation/gradient |
| 33 | 朴素 layer-wise 利用率 | $n$ GPUs 时每卡约 $1/n$ 时间工作 |
| 34 | Pipeline + microbatch | bubble 比约 $(p-1)/m$ |
| 35 | 为什么 PP 有用 | 参数显存缩小；只传 $bsh$ activation；适合慢链路 |
| 36 | Batch 依赖 | microbatch 太少，bubble 迅速恶化 |
| 37 | 复杂 schedule | 用更多通信换更高 utilization |
| 38 | Zero-bubble | activation-gradient 必须及时，weight-gradient 可延后填泡 |
| 39 | 沿 width 切 | matmul 可拆成子矩阵和 partial sums |
| 40 | TP 基本结构 | column/row split；forward/backward 的 collective 互换 |
| 41 | Transformer 分法 | QKV/up 为 column；out/down 为 row；norm/router replicate |
| 42 | TP 放置 | GPU 上通常限制在节点内高速互联 |
| 43 | TP vs PP | TP 无 bubble、不需大 batch，但每层有大 collective |

### 20.4 第 44–55 页：Activation、SP、EP、CP

| 页 | 内容 | 必须理解的结论 |
|---:|---|---|
| 44 | Memory 是动态的 | activation 随 batch/sequence/layers 增长 |
| 45 | Activation memory | TP/PP 切参数不等于切所有 activation |
| 46 | 每层 activation 公式 | attention 的 $s^2$ 项可通过 recomputation 去掉 |
| 47 | TP 后残留 | norm/dropout/block input 等 $10sbh$ 项仍复制 |
| 48 | Sequence parallel | 点运算沿 sequence 切；gather/reduce-scatter 接 TP |
| 49 | 完整线性显存扩展 | TP+SP 共同缩小 parameter 和 activation |
| 50 | Expert parallel | 不切 matmul，而切 experts 并路由 token |
| 51 | EP 的理由 | 保持完整 expert GEMM，可能比细切 TP 高效 |
| 52 | EP 与其他方法组合 | EP/DP group 和 token 数影响 utilization |
| 53 | Attention/MLP 解耦 | attention 喜欢 TP/CP，MoE MLP 喜欢 EP/ETP/EDP |
| 54 | Context/Ring attention | 超长序列沿 token/KV 维分片 |
| 55 | 总对照表 | 六类方法的 memory、communication、batch、复杂度 |

### 20.5 第 56–73 页：组合规则与真实模型

| 页 | 内容 | 必须理解的结论 |
|---:|---|---|
| 56 | Model vs tensor parallel 图 | global batch 与每 GPU workload 是关键量 |
| 57 | 3D/4D 规则 | 先 TP/EP 节点内、PP 跨节点让模型放下，再 DP |
| 58 | Megatron 推荐 | 基础并行外还有 overlap、sequence/context/expert 技术 |
| 59 | Narayanan 2021 | TP 先到 8，PP 解决容量，剩余做 DP |
| 60 | 线性 gains | 拓扑感知组合可维持近似平坦 utilization |
| 61 | TP=8 常最优 | 64 设备常用 8×8 结构匹配节点 |
| 62 | Recomputation 可能增吞吐 | 省显存后可增 batch，使 GEMM 更高效 |
| 63 | Dolma 7B | 用 FSDP，规模可在较小域内处理 |
| 64 | DeepSeek/V3 | ZeRO1；V3 PP16、EP64，A2A overlap |
| 65 | Yi | ZeRO1+TP+PP；MoE 版本更多用 EP |
| 66 | Llama3 训练阶段 | 小 batch、预训练、长上下文阶段配置可变 |
| 67 | Llama3 故障 | 大规模训练硬件故障是常态 |
| 68 | Gemma2 | ZeRO3 + MP(TP/SP) + DP |
| 69 | Mixtral 8×22B | 示例 TP/PP/CP/EP=4/4/1/8 |
| 70 | Nemotron3 long context | TP2、CP64、EP64 |
| 71 | Qwen3 MoE | EP 主导；大模型示例 TP2/EP32/PP8 |
| 72 | 模型配置汇总 | 共性：TP≤8、EP 可大、长上下文 CP 大 |
| 73 | Recap | 没有单一答案，组合多个维度 |

## 21. ZeRO 内存公式的完整版本

设参数量为 $\Psi$，混合精度 Adam 每参数：

- BF16 parameter：2 bytes；
- BF16 gradient：2 bytes；
- FP32 master parameter：4 bytes；
- FP32 first moment：4 bytes；
- FP32 second moment：4 bytes。

总计 16 bytes/param。令 data-parallel degree 为 $d$，忽略 activation/buffer：

| 方法 | 每卡模型状态显存 |
|---|---:|
| DDP | $16\Psi$ |
| ZeRO-1 | $4\Psi+12\Psi/d$ |
| ZeRO-2 | $2\Psi+14\Psi/d$ |
| ZeRO-3 | $16\Psi/d$ |

课件第 28 页采用“纯 BF16 状态较轻、只有 master 为 FP32”的 12 bytes/param 假设，所以数字与上表的经典 16-byte accounting 不完全相同。公式必须和 dtype 假设一起报告。

### 21.1 ZeRO-1 为什么通信不增加

DDP ring all-reduce gradient 可分解为：

$$gradient\ reduce\text{-}scatter\ (\Psi)+all\text{-}gather\ (\Psi).$$

ZeRO-1 在 reduce-scatter 后不立即 gather gradient，而由每 rank 更新负责的 parameter shard，再 all-gather updated parameters。因此仍约 $2\Psi$ 通信，却把 optimizer state 分片。

### 21.2 ZeRO-2 的 gradient 生命周期

Backward 按层逆序产生 gradient。某 FSDP/ZeRO bucket ready 后立刻 reduce-scatter；完整 bucket 不再需要就 free。若等整个 backward 完成后才处理，会瞬间保留完整 gradients，失去 ZeRO-2 的峰值显存收益。

### 21.3 ZeRO-3/FSDP 时间线

对每个 wrapped block：

```text
Forward:
prefetch next params ─┐
all-gather current ───┼→ compute current → reshard/free
Backward:
all-gather current → recompute/grad compute → reduce-scatter grads → free
```

两次 parameter all-gather 分别服务 forward 与 backward；若 forward 后一直保留参数可少一次通信，但会失去显存收益。通信约 $3\Psi$：两次 all-gather + 一次 reduce-scatter。实际是否能隐藏取决于 block compute 时间、bucket 大小、network 和 prefetch 距离。

FSDP wrap 太细会产生许多 latency-heavy collectives；太粗会在 gather 时 materialize 太多参数。Wrap granularity 是重要超参数。

## 22. Pipeline Parallel 的完整性能模型

设 $p$ stages、$m$ microbatches，每个 stage 单个 microbatch 计算时间近似相等。只看简单流水线，填充/排空 bubble 为 $p-1$ 个 slot：

$$bubble\ fraction\approx\frac{p-1}{m+p-1},$$

课件用相对 useful compute 的近似：

$$bubble/useful\approx\frac{p-1}{m}.$$

两者表达同一趋势：$m\gg p$ 才能小 bubble。

GPipe 常先做全部 forwards 再 backwards，activation memory 大；1F1B 在 warmup 后交替一次 forward/一次 backward，减少在途 activation。Interleaved pipeline 让每设备持多个非连续 virtual stages，可改善平衡和 bubble，但会增加通信。

Zero-bubble 利用 backward 的两部分：

- dgrad：对 input activation 的梯度，必须立即发给前 stage；
- wgrad：parameter gradient，只需 optimizer step 前完成，可延后填 bubble。

PP 通信每 stage boundary、每 microbatch 约为 $bsh$ elements 的 activation，反向再传相同量级 gradient。相比逐层 TP collective，它频率低、点对点，因此更适合 inter-node。

## 23. Tensor Parallel 的 column/row 数学

对 $Y=XW$。

**Column parallel**：按输出列切 $W=[W_1|\cdots|W_t]$：

$$Y=[XW_1|\cdots|XW_t].$$

每 rank 得 output shard。逐元素 activation 可在 shard 上本地执行。

**Row parallel**：按输入行切
$$X=[X_1|\cdots|X_t],\qquad W=[W_1;\cdots;W_t],$$

则：

$$Y=\sum_{r=1}^t X_rW_r.$$

每 rank 产生 partial output，需 all-reduce 或 reduce-scatter 求和。

Transformer 组合：

- Q/K/V projections、MLP gate/up：column parallel；
- attention output、MLP down：row parallel；
- RMSNorm、router、小参数：replicated 或 sequence parallel。

这样 column layer 的 shard 可直接作为 row layer输入，collective 只放在成对线性层的边界。

课件第 43 页给 TP 每层通信量级约：

$$8bsh\frac{t-1}{t},$$

系数 8 概括 forward/backward 中多次 activation collective 和低精度字节；具体实现随是否用 SP、reduce-scatter、GQA/MoE 改变。核心是它随**每层**发生，所以只能放快链路。

## 24. Activation Memory 与 Sequence Parallel 公式

令：

- $s$：sequence length；
- $b$：microbatch size；
- $h$：hidden size；
- $a$：attention heads；
- $t$：tensor-parallel degree。

课程引用的 Megatron activation accounting 中，单层未分片 activation 量级可写为：

$$sbh\left(34+5\frac{as}{h}\right)\ bytes.$$

$5as/h$ 对应 attention matrix/dropout 等随 $s^2$ 增长的项；selective recomputation/FlashAttention 可不保存它们。

只做 TP 后，matmul 相关 activation 被切，但 norm、dropout、residual、block inputs 等仍有约 $10sbh$ replicated 项，形式近似：

$$sbh\left(10+\frac{24}{t}+5\frac{as}{ht}\right).$$

加入 sequence parallel，把 pointwise operations 沿 sequence 切，得到更接近：

$$\frac{sbh}{t}\left(34+5\frac{as}{h}\right).$$

不同实现、dtype、checkpoint 策略会改变常数，但“不加 SP 时有不随 TP 缩小的 10 项”是课件重点。

TP↔SP layout 转换：

- forward：一侧 all-gather，另一侧 reduce-scatter；
- backward：方向反转。

这不是额外复制完整 activation 后永久保存，而是在算子边界改变 shard 轴。

## 25. Expert 与 Context Parallel 的深入理解

### 25.1 Expert Parallel

每个 MoE layer：

1. router 为 token 计算 expert scores；
2. 选择 top-k experts；
3. all-to-all dispatch token；
4. 本地 experts 执行完整 GEMM；
5. all-to-all combine outputs；
6. 按 router weights 加权。

相比 TP 把一个 GEMM 切小，EP 保留完整 expert matrix shape，可能有更高计算效率。但每 expert tokens 太少时 GEMM 仍小；负载不平衡会产生 capacity overflow 或 straggler。

Attention 没有 experts，因此同一 block 的 attention 可能用 TP/CP/DP，MLP 用 ETP/EP/EDP。Megatron 的复杂 group 划分正是为解耦两部分。

### 25.2 Context Parallel / Ring Attention

每 rank 保存一段 query/KV。为让每段 query 看到所需历史 K/V，KV blocks 沿 ring 轮转，每轮用本地 Q 对当前 KV block 做 online softmax accumulation。显存随 CP degree 降低，但每层产生环形通信。Causal mask 会使不同 rank 的有效工作量不均，需要负载平衡或 zig-zag 分配。

SP 主要切 LayerNorm/dropout 等普通 activation；CP 主要切 attention 的长上下文/KV，二者不能简单视为同义词。

## 26. 3D/4D 并行选择的决策顺序

Dense 模型常有：

$$N_{GPU}=DP\times TP\times PP.$$

加入 CP/EP 时 group 可能重叠，不能总把所有 degree 直接相乘。

实用顺序：

1. 单层参数/GEMM 是否单卡放得下？不行则节点内 TP；
2. MoE experts 是否是主要参数？是则优先 EP，避免过高 ETP；
3. 全模型是否仍放不下？跨节点增加 PP 或高速域内 FSDP；
4. activation/KV 是否主导？加入 recomputation、SP、CP；
5. 模型放下后，剩余 GPU 用 DP 提吞吐；
6. global batch 太大时用更小 DP、更多 model parallel 或改变 accumulation；
7. 将高频 blocking collective 放最快 topology。

### 26.1 真实模型配置表

| 模型 | 课件给出的要点 |
|---|---|
| Dolma 7B | FSDP，模型规模较小 |
| DeepSeek 7/67B | ZeRO-1 + model parallel |
| DeepSeek-V3 | PP=16、EP=64（8 nodes）、ZeRO-1；1F1B all-to-all overlap |
| Yi | ZeRO-1 + TP + PP |
| Yi-Lightning | 更多以 EP 替代 TP |
| Llama3 405B | DP=128、TP=8、PP=16、CP=1；阶段性训练 |
| Gemma2 | ZeRO-3 + TP/SP + DP |
| Mixtral 8×22B 示例 | TP/PP/CP/EP=4/4/1/8，DP 约 2 |
| Nemotron3 120B-A12B long context | TP=2、CP=64、EP=64 |
| Qwen3 MoE 示例 | TP=2、EP=32、PP=8、CP=1 |

这些数字是特定硬件和训练阶段的结果，不是通用默认值。真正可迁移的模式是 TP 通常不超过节点内 8，EP 可以更大，长上下文阶段 CP 显著增加。

## 27. 大规模训练可靠性

Llama3 405B 案例提醒：数千/上万 GPU 中，GPU、NIC、link、host、filesystem 或进程故障是常态。有效训练吞吐应考虑：

$$effective\ throughput=raw\ throughput\times(1-failure/recovery\ overhead).$$

系统需要定期异步 checkpoint、保存 dataloader/RNG/optimizer 状态、自动检测 hanging collectives/stragglers、重启并验证 silent corruption。GPU 数越多，集群整体 mean time between failures 越短。

## 28. Lecture 8 全覆盖核对

- 1–13：动机、collective、TPU/GPU topology；
- 14–30：DDP、16-byte accounting、ZeRO 1/2/3、容量和 batch 限制；
- 31–43：model parallel、pipeline、zero-bubble、tensor parallel；
- 44–55：activation formula、recomputation、SP、EP、CP、总表；
- 56–62：3D parallel、TP=8、linear scaling、recomputation；
- 63–72：Dolma、DeepSeek、Yi、Llama3、Gemma2、Mixtral、Nemotron、Qwen；
- 73：组合并行总结。
