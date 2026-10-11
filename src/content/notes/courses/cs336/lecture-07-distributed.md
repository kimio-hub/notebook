---
title: "CS336 Lecture 7：分布式并行"
description: "从 collective communication 出发，用最小 PyTorch 代码搭出数据并行、张量并行与流水线并行。"
tags:
  - "cs336"
  - "distributed"
  - "systems"
category: "courses"
series: "CS336"
date: "2026-07-11"
updated: "2026-07-11"
---

> Stanford CS336 Spring 2026 · 本讲从 collective communication 出发，用最小 PyTorch 代码搭出 data、tensor、pipeline parallelism。

## 0. 统一主线：计算离数据很远

单 GPU 内要避免 HBM 往返；多 GPU 时要避免跨 GPU、跨节点通信。速度层次大致是：

$$
\text{register/shared memory} > \text{HBM} > \text{NVLink/NVSwitch} > \text{InfiniBand/RoCE} > \text{普通 Ethernet}.
$$

多 GPU 的两个原因：

1. 模型参数、梯度、优化器状态和 activation 放不进一张卡；
2. 希望利用更多 FLOPs 缩短训练时间。

并行策略本质是在三个选项间交换：**重新计算、占用本地显存、跨设备通信。**

---

## 1. 分布式基本概念

- **Rank**：一个进程/设备的编号，如 0 到 3；
- **World size**：参与通信的 rank 总数；
- **Process group**：参与某组 collective 的 ranks；
- **Backend**：CPU 常用 Gloo，NVIDIA GPU 常用 NCCL；
- **Barrier**：所有 rank 到齐后才继续，主要用于同步阶段或可靠计时。

`torch.distributed` 通常是一卡一进程。`MASTER_ADDR`/`MASTER_PORT` 用于 rendezvous 和建立进程组；大块训练数据并非都绕 rank 0 转发。

---

## 2. Collective communication

Collective 用一个整体通信模式描述多设备间数据流，比手写点对点 send/recv 更易优化。

### 2.1 基础操作

| 操作        | 输入到输出                    | 典型用途      |
| --------- | ------------------------ | --------- |
| Broadcast | root 的数据复制给所有 rank       | 下发初始权重/配置 |
| Scatter   | root 的大张量切开分给各 rank      | 分发 shard  |
| Gather    | 各 rank shard 汇总到 root    | 收集结果      |
| Reduce    | 汇总并执行 sum/min/max 到 root | 聚合统计量     |

### 2.2 三个训练主力

**All-gather**：每个 rank 持有一个 shard，操作后每个 rank 都得到完整拼接结果。FSDP 前向按需收集参数会使用它。

**Reduce-scatter**：先对不同 rank 的对应位置做 reduction，再把结果 shard 分给各 rank。它适合“梯度需要求和，但结果本来就分片保存”。

**All-reduce**：每个 rank 最终都得到完整 reduction 结果。

$$
\text{all-reduce}=\text{reduce-scatter}+\text{all-gather}.
$$

这条等价关系很重要：DDP 需要完整同步梯度，用 all-reduce；ZeRO/FSDP 则拆开两步，让梯度/状态保持分片。

### 2.3 All-to-all

每个 rank 都向每个其他 rank 发送一部分数据。平衡切分时像矩阵转置。MoE 中每张卡既持有 token shard 又持有部分 experts，需要把 token 路由到对应 expert，因此 all-to-all 是核心通信。

负载不均会让某些 rank 收到更多 token，最慢 rank 决定整体速度，这就是 MoE load balancing 的系统意义。

### 2.4 记忆方法

- Reduce：做可结合/可交换的聚合；
- Scatter 与 gather 互为方向相反；
- All：结果目的地是所有 rank；
- All-to-all：每个人都给每个人发不同内容。

---

## 3. 硬件拓扑与 NCCL

典型数据中心节点有 8 张 GPU，经 NVLink 接到 NVSwitch；节点间经 PCIe 到 HCA/NIC，再走 InfiniBand 或 RoCE；跨 pod/机房可能走更慢的 Ethernet。

课程给出的量级直觉：B200 NVLink 5.0 聚合带宽约 1.8 TB/s，而节点间 InfiniBand 约 0.05 TB/s，差距巨大。因此：

- 高频 collective 的 tensor parallel 通常留在节点内；
- 只传 stage 边界 activation 的 pipeline parallel 可以跨较慢链路；
- rank 的物理放置必须匹配通信模式。

普通网络路径常需 CPU、内核 socket buffer 和 NIC buffer。**RDMA** 允许设备直接读写远端设备内存，绕过 CPU 数据拷贝；InfiniBand 原生支持，RoCE 在以太网上提供类似能力。

**NCCL** 做三件事：探测 GPU/交换机/链路拓扑；选择 ring/tree 等算法和路径；启动负责收发的 GPU kernel。应用调用的是 `all_reduce`，NCCL 决定如何把它变成低层数据包。

---

## 4. 正确 benchmark collective

通信测时同样要 warmup、GPU synchronize、barrier。否则某个 rank 先开始/先结束，测到的不是同一通信区间。

区分：

- **algorithm bandwidth**：从算法视角处理了多少数据；
- **bus bandwidth**：换算成物理链路承担的数据量，便于跨 collective 比较。

Ring all-reduce 中，每个 rank 大致经历 $P-1$ 步 reduce-scatter 和 $P-1$ 步 all-gather；总移动量趋近 $2\frac{P-1}{P}N$。大消息处于 bandwidth-bound，小消息更受 latency 和同步开销影响。

---

## 5. Data Parallelism / DDP

### 5.1 切分方式

沿 batch 维切分：每个 rank 有完整模型，处理不同样本。

训练一步：

1. 每 rank 在本地 batch shard 上前向；
2. 本地反向得到不同梯度；
3. 对每个参数梯度做平均 all-reduce；
4. 各 rank 用相同梯度执行 optimizer step；
5. 因初始参数和更新相同，模型副本始终一致。

公式上，全局 batch 梯度被拆成各 rank 局部和：

$$
g=\frac1P\sum_{r=1}^{P}g_r.
$$

### 5.2 优缺点

优点：实现简单、计算并行自然、通常扩展最好。缺点：每张卡保留完整参数/梯度/优化器状态，显存不随 GPU 数下降；通信量与参数量相关；world size 受到有效 global batch size 限制。

真实 DDP 不会等所有 backward 结束才一次同步，而是把参数分 bucket，某 bucket 的梯度就绪后异步 all-reduce，与剩余 backward 重叠。

---

## 6. Tensor Parallelism

### 6.1 切分方式

沿层的宽度切参数。以 $Y=XW$ 为例，按列切 $W=[W_1,\ldots,W_P]$：

$$
Y=[XW_1,\ldots,XW_P].
$$

每张卡算一个输出 shard，若下一层需要完整激活，就 all-gather。另一种按行切时，各 rank 计算部分和，再 all-reduce/reduce-scatter 得到结果。

课程的最小实现把每层输出维切开：

1. 所有 rank 都有完整输入 $X$；
2. rank $r$ 只持 $W_r$；
3. 计算局部 $XW_r$ 和 GeLU；
4. all-gather 局部激活并拼成完整宽度；
5. 进入下一层。

### 6.2 Transformer 中的组合

常见 Megatron 风格：QKV 与 MLP up projection 用 column parallel；attention output 与 down projection 用 row parallel。正确配对可以让两次线性层之间保持 sharded，减少不必要 gather。

### 6.3 优缺点

- 参数和部分 activation 随 TP degree 分片；
- 没有 pipeline bubble，不依赖大 batch；
- 每层都有 blocking collective，对低延迟、高带宽要求极高；
- 切得过细会让本地 matmul 变小，GPU 利用率下降。

所以 TP 通常限制在一个 NVLink 域内，常见 degree 不超过每节点 GPU 数。

---

## 7. Pipeline Parallelism

### 7.1 切分方式

沿深度切层：rank 0 放前几层，rank 1 放后几层。前向传 activation，反向传 activation gradient。

若直接整 batch 依次过 stages，在 $P$ 个 stage 中每张卡约只有 $1/P$ 时间工作。解决办法是把 batch 切成多个 **microbatches**，形成流水线。

课程代码的 baby version：

1. rank 0 将 batch `chunk` 成多个 microbatch；
2. 每个 stage 接收上一 stage 的 activation；
3. 执行本地层；
4. 用 `dist.send` 发给下一 stage；
5. 多个 microbatch 连续流动，填充流水线。

### 7.2 Bubble

理想化 forward pipeline 的 bubble 比例约为：

$$
\frac{P-1}{M},
$$

其中 $P$ 为 stages，$M$ 为 microbatches。microbatch 越多，填充/排空成本占比越低；但更多 microbatch 也改变 activation 缓存、调度和有效 batch。

### 7.3 为什么仍然使用

- 每张卡只放部分层，参数显存随 stage 数下降；
- 只在相邻 stages 间点对点传 activation，通信量往往比逐层 TP collective 小；
- 因此适合跨节点慢链路。

困难包括 stage 负载均衡、bubble、activation 生命周期、forward/backward 调度和通信计算重叠。课程代码只演示 forward，真实系统还需 1F1B 等调度。

---

## 8. 四个切分轴的统一视图

| 轴 | 并行方法 | 切什么 | 主要通信 |
|---|---|---|---|
| Batch | Data parallel | 样本 | 梯度 all-reduce |
| Width | Tensor / Expert parallel | hidden 维或 experts | 激活 collective / all-to-all |
| Depth | Pipeline parallel | layers | 相邻 stage activation |
| Sequence | Sequence/context parallel | token 位置 | activation/KV 交换 |

实际大模型通常组合这些轴，而不是四选一。基本原则：把最频繁、最重的通信放到最快链路；先解决“能否放下”，再用剩余 GPU 做 data parallel 提升吞吐。

## 9. 易混点与工程陷阱

- `barrier` 不搬训练张量，只同步进度；滥用会破坏 overlap。
- All-reduce 后若求的是梯度和而损失按全局均值定义，还要正确除以 world size。
- 一卡一进程时必须把 rank 映射到正确 CUDA device。
- TP 不是简单把权重任意切开；切分方向决定 forward/backward 所需 collective。
- PP 的 microbatch 不等于 optimizer 的 global batch；还可能有 gradient accumulation。
- 通信 API 的 `async_op=True` 只有在延迟 `wait()` 且中间确有可执行计算时，才实现真正 overlap。
- 多节点性能取决于拓扑映射，逻辑 ranks 相邻不代表物理链路快。

## 10. 自测

1. 为什么 all-reduce 可以拆成 reduce-scatter 与 all-gather？
2. DDP 为什么每个 rank 的 loss 不同，但参数更新后相同？
3. TP 为什么适合 NVLink 域，而 PP 更能容忍跨节点链路？
4. Pipeline bubble 与 stage 数、microbatch 数怎样变化？
5. MoE 为什么需要 all-to-all，负载不均为什么会拖慢全局？
6. 如果模型已经能放进单卡，增加 GPU 最简单的策略是什么？

## 11. 本讲结论

Collective 是分布式训练的汇编语言；DDP、FSDP、TP、PP 都只是对这些 primitives 的不同编排。性能优化的核心仍与单 GPU 相同：理解数据在哪里、需要搬多少、链路有多快，以及通信能否与计算重叠。

## 资料

- [课程官方 Lecture 7 源码](https://github.com/stanford-cs336/lectures/blob/main/lecture_07.py)
- [PyTorch Distributed 文档](https://pytorch.org/docs/stable/distributed.html)
- [NCCL Tests 的带宽解释](https://github.com/NVIDIA/nccl-tests/blob/master/doc/PERFORMANCE.md#allreduce)



---

## 12. 逐源码详细补充：Collective、通信计量与三种并行

### 12.1 从单 GPU 到多 GPU 的统一视角

Lecture 6 用 fusion/tiling 减少 HBM 数据移动；Lecture 7 用 replication/sharding 减少 NVLink、InfiniBand、Ethernet 上的数据移动。层次为：

$$register/shared > HBM > NVLink/NVSwitch > InfiniBand/RoCE > Ethernet.$$

多 GPU 的两类目标必须区分：

- capacity scaling：单卡放不下参数、梯度、optimizer state、activation；
- speed scaling：模型能放下，但要更多 FLOPs 缩短 wall-clock。

任何 tensor 有三种选择：保存在本卡显存、在需要时重算、保存在其他 GPU 并通信取回。

### 12.2 八种 collective 的 4-rank 具体结果

设 world size 为 4。

**Broadcast**：rank 0 的 `[0,1,2,3]` 复制给 ranks 0–3。用于 rank 0 load checkpoint 后同步，但大型训练通常让各 rank 并行读取分片，避免 root bottleneck。

**Scatter**：root 的 `[0,1,2,3]` 分成四片，rank $r$ 得到 `[r]`。

**Gather**：各 rank 的 `[0]`、`[1]`、`[2]`、`[3]` 只在 root 拼成 `[0,1,2,3]`。

**Reduce**：各 rank 输入经过 sum/min/max 等 reduction，结果只在 root。例如 $0+1+2+3=6$。

**All-gather**：每个 rank 最终都有 `[0,1,2,3]`。FSDP 前向可用它把 parameter shards 临时 materialize 为完整层。

**Reduce-scatter**：若四个 rank 分别输入：
```text
r0 [0,1,2,3]
r1 [1,2,3,4]
r2 [2,3,4,5]
r3 [3,4,5,6]
```
先按列求和为 `[6,10,14,18]`，再分片：r0 得 6、r1 得 10、r2 得 14、r3 得 18。它适合“梯度需要聚合，但聚合后本来就只想保存 shard”。

**All-reduce**：每个 rank 都得到完整 `[6,10,14,18]`。

$$AllReduce=ReduceScatter+AllGather.$$

**All-to-all**：每个 rank 把不同片发给不同目的地。若 r0 输入 0–3、r1 4–7、r2 8–11、r3 12–15，输出为：
```text
r0 [0,4,8,12]
r1 [1,5,9,13]
r2 [2,6,10,14]
r3 [3,7,11,15]
```
平衡切分像矩阵转置。MoE 用它把各 rank 的 token 发送给各 rank 持有的 experts；不平衡时最慢 rank 决定 step time。

术语记忆：reduce 表示结合/交换运算；scatter 与 gather 方向相反；all 表示所有 rank 都是目的地。

### 12.3 拓扑与课程数字

传统家用环境：同节点 GPU 走 PCIe；不同节点可能走慢 Ethernet。课件给出 PCIe 7.0×16 理论约 242 GB/s、传统 Ethernet 示例约 200 MB/s，用来强调数量级，不代表所有机器。

现代数据中心典型：

- 8 GPUs/node，经 NVLink 接 NVSwitch；
- B200 NVLink 5.0 聚合约 1.8 TB/s，而 HBM 约 8 TB/s；
- 约 256 nodes/pod，经 PCIe→HCA/NIC→InfiniBand，示例约 0.05 TB/s；
- pods 间可能再走 Ethernet。

普通 TCP 路径涉及 GPU/host copy、kernel socket buffer、packet construction、NIC ring buffer。RDMA 允许设备直接访问远端 memory，绕过 CPU 数据路径；InfiniBand 支持 RDMA，RoCE 在 Ethernet 上提供类似能力。

GB200/GB300 NVL72 把 8 GPUs/tray、9 trays/rack 共 72 GPUs 放进一个 NVLink domain，扩大可高效做 TP/EP 的通信域。

### 12.4 NCCL 做了什么

应用只说 `all_reduce`，NCCL：

1. 探测 GPU、PCIe、NVLink、switch、NIC 拓扑；
2. 选择 ring/tree/channel 和路由；
3. chunk tensor；
4. 启动 GPU communication kernels 完成 send/recv/reduce。

因此 logical collective 相同，物理数据路径和速度可能完全不同。Rank placement 不匹配 topology 会让本应走 NVLink 的流量跨节点。

### 12.5 PyTorch distributed 初始化与 API

`setup` 设置：

```python
MASTER_ADDR=localhost
MASTER_PORT=15623
dist.init_process_group(
    backend="nccl" if cuda else "gloo",
    rank=rank,
    world_size=world_size,
)
```

Master 地址用于 rendezvous/协调，不表示所有训练 tensor 经 rank 0。NCCL 用于 GPU，Gloo 用于 CPU fallback。

`dist.all_reduce(tensor=data,...)` **in-place** 修改 `data`；`reduce_scatter_tensor` 需要预分配 output；`all_gather_into_tensor` 也需要目标 buffer。`async_op=False` 表示调用完成前阻塞；`async_op=True` 只有延迟 `wait()` 且中间有独立计算时才产生 overlap。

`barrier()` 不传训练 tensor，只确保所有 rank 到齐。讲义用它整理 print 和测时；训练热路径滥用会把本可重叠的工作强制串行。

### 12.6 代码验证 AllReduce 分解

每 rank 构造：

$$data_r=[0,1,2,3]+r.$$

All-reduce sum 后各 rank 得 `[6,10,14,18]`。随后 reduce-scatter 让每 rank 得其中一个标量，再 all-gather 回完整向量，直接验证等式。

### 12.7 Collective benchmark 与带宽公式

All-reduce benchmark 对约 $100\times1024^2$ elements：

1. 分配 tensor；
2. warmup collective；
3. GPU synchronize；
4. barrier；
5. CPU wall-clock 包围正式 collective；
6. synchronize + barrier 后结束。

同步不可少，因为 NCCL kernel 也异步。源码对 ring-style all-reduce 估算：

```python
size_bytes = element_size * numel
sent_bytes = size_bytes * 2 * (P-1)
total_duration = P * duration
bandwidth = sent_bytes / total_duration
```

即：

$$BW_{alg}\approx\frac{2(P-1)}P\frac{N}{t}.$$

当 $P$ 大时约 $2N/t$。Reduce-scatter 只含一半阶段，数据量/耗时都近似 all-reduce 的一半，因此测得链路 bandwidth 相似。这里是课程的 normalized/effective bandwidth，不能直接与 NIC 标称单向带宽混为一谈；NCCL tests 还区分 algorithm bandwidth 与 bus bandwidth。

小消息更 latency-bound，大消息更 bandwidth-bound；benchmark 100M elements 是为了进入后者。

### 12.8 Data Parallel 教学代码逐步执行

数据 shape 为 $128\times1024$，world size 4，所以每 rank 得 $32\times1024$ batch shard。四层 MLP 每层参数都是 $1024\times1024$，每 rank 持完整副本和自己的 AdamW state。

流程：

```python
x = local_data
for W in params:
    x = gelu(x @ W)
loss = mean(x**2)
loss.backward()
for W in params:
    all_reduce(W.grad, AVG)
optimizer.step()
```

各 rank 因数据不同而 loss/本地 gradient 不同；all-reduce AVG 后 gradient 相同。初始参数通过相同 seed 构造，optimizer state 初始相同，所以更新后参数继续相同。

这段只展示 DDP 的本质，生产代码还必须：

- 每 step `optimizer.zero_grad()`，否则梯度累积；源码只跑一步所以没暴露；
- 用 DDP gradient hooks/buckets 在 backward 中自动同步，而非 Python 循环；
- overlap gradient all-reduce 与剩余 backward；
- 分布式 sampler 让各 rank 只加载自己的数据，而非先复制整数据再 slice；
- 正确处理 mixed precision、gradient clipping、checkpoint 和 failure。

内存没有参数级缩放：参数、gradient、optimizer state 都完整复制。Lecture 8 的 ZeRO/FSDP 正是对这里的状态分片。

### 12.9 Tensor Parallel 教学代码

数据在所有 rank 上完整复制。$D=1024,P=4$，所以 `local_num_dim=256`。每层每 rank 持 $W_r\in R^{1024\times256}$，计算：

$$X_r=XW_r\in R^{B\times256}.$$

然后 all-gather 四个局部 activation 并沿 hidden dimension concat，恢复 $B\times1024$，供下一层使用。

这相当于 column-parallel linear 的最小例子。参数显存约 $1/P$，但每层都发送 activation，且 collective blocking，因此需要 NVLink 等高速链路。

源码把 GeLU 放在 local shard 后是合法的，因为 GeLU 是逐元素的：
$$GELU([XW_0|\cdots|XW_{P-1}])=[GELU(XW_0)|\cdots].$$

源码未实现 backward。Backward 需要根据切分方向同步 input gradients，并让 parameter gradients 保持本地 shard。真实 Transformer 会把 column-parallel QKV/up projection 与 row-parallel output/down projection配对，避免每个 linear 都恢复完整 activation。

### 12.10 Pipeline Parallel 教学代码

$L=4$ layers、$P=2$ stages，所以每 rank 持 2 层。Batch 128 切成 4 microbatches，每个 $32\times1024$。

- rank 0 从原数据获得 microbatches；
- rank 1 预分配同 shape receive buffers；
- 对每个 microbatch，非首 stage `recv`；
- 运行本地 layers；
- 非末 stage `send` 给下一 rank。

只传 stage boundary activation，参数不跨 stage。通信是 point-to-point，不是 collective，适合较慢跨节点链路。

但源码使用 blocking send/recv，只有 forward，且没有精细 schedule；“把 batch 切块”并不自动实现完整高效 pipeline。生产系统需要：

- backward activation-gradient 反向传输；
- GPipe 或 1F1B schedule；
- communication/computation overlap；
- stage load balance；
- activation checkpoint；
- microbatch bookkeeping；
- 防止 send/recv 顺序不一致造成 deadlock。

简单 forward pipeline 的 bubble 比约：
$$\frac{P-1}{M},$$
其中 $M$ 是 microbatches。M 越大 bubble 越小，但 activation buffer 和调度更复杂。

### 12.11 讲义运行辅助代码

`get_init_params` 每次设 seed 0，并按 $1/\sqrt{num\_outputs}$ 缩放，使 ranks 初始参数一致。`int_divide` 断言能整除，教学代码不处理 ragged batch/layers。`summarize_tensor` 打印 shape 和首元素；`render_duration` 自动显示 μs/ms/s。

`spawn` 正常情况下用 `torch.multiprocessing.spawn` 启动 world-size 个进程。Edtrace 调试时无法追踪 multiprocessing，于是 `DisableDistributed` 临时把 distributed 函数替换为 no-op，只执行 rank 0 来展示控制流；这时输出不能用于验证真实通信。

### 12.12 完整覆盖清单

| Lecture 7 内容 | 覆盖 |
|---|---|
| 统一数据移动层次 | 0、12.1 |
| broadcast/scatter/gather/reduce | 2、12.2 |
| all-gather/reduce-scatter/all-reduce | 2、12.2、12.6 |
| all-to-all/MoE | 2.3、12.2 |
| PCIe/NVLink/IB/RDMA/RoCE/NVL72 | 3、12.3 |
| NCCL | 3、12.4 |
| torch.distributed/Gloo/NCCL | 1、12.5 |
| collective benchmark | 4、12.7 |
| data parallel code | 5、12.8 |
| tensor parallel code | 6、12.9 |
| pipeline parallel code | 7、12.10 |
| setup/spawn/helpers | 12.11 |
| 未覆盖的 overlap、attention、SP/EP/JAX | 原讲 59–64 行及 8、9 |


### 12.13 Function-by-function 精确索引

- `main`：从数据移动层次进入 collective、硬件、PyTorch、DP/TP/PP。
- `collective_operations`：八类 collective 的概念输入输出。
- `hardware`：PCIe、NVLink/NVSwitch、IB、RDMA、RoCE、NCCL。
- `torch_distributed`：backend 与 collective API 入口。
- `collective_operations_main`：用真实 tensors 验证 all-reduce 分解。
- `benchmarking`、`all_reduce`、`reduce_scatter`：warmup、同步、带宽核算。
- `data_parallelism`、`generate_sample_data`、`data_parallelism_main`：batch shard 和 gradient AVG。
- `tensor_parallelism`、`tensor_parallelism_main`：width shard、activation all-gather。
- `pipeline_parallelism`、`pipeline_parallelism_main`：layer shard、microbatch、send/recv。
- `setup`、`cleanup`：初始化/销毁 process group。
- `DisableDistributed`、`spawn`：真实 multiprocessing 与 edtrace no-op 路径。
- `get_init_params`：相同 seed 和尺度初始化。
- `int_divide`：教学代码要求整除。
- `summarize_tensor`：打印 shape 与首元素。
- `render_duration`：μs/ms/s 单位格式化。
