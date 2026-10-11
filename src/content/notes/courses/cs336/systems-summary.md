---
title: "CS336 System 部分纪要"
description: "单卡把数据搬近计算，多卡把通信放在快链路上：重算、本地显存与跨设备通信的三选一。"
tags:
  - "cs336"
  - "systems"
  - "gpu"
category: "courses"
series: "CS336"
date: "2026-07-11"
updated: "2026-07-11"
---

> 单卡把数据搬近计算；多卡把通信放在快链路上。  
> 统一哲学：**重算 / 占本地显存 / 跨设备通信**，三选一，没有免费午餐。

带宽层次（从快到慢）：

$$
\text{register / shared memory} > \text{HBM} > \text{NVLink / NVSwitch} > \text{InfiniBand / RoCE} > \text{Ethernet}
$$

---

## PART 1  GPU

## 1.1 为什么要管 GPU

语言模型性能大致随 compute 可预测地涨（scaling laws）。  
没有 GPU 的 parallel scaling，就没有现在的 LLM scale。

| | CPU | GPU |
|--|-----|-----|
| 优化目标 | 延迟 latency | 吞吐 throughput |
| 线程 | 少量强线程 | 海量轻量线程 |
| 哲学 | 单任务尽快做完 | 单位时间尽量多干活 |

## 1.2 硬件树

```
GPU
├── 许多 SM（Streaming Multiprocessor）  ← 独立执行 block
│     ├── 控制逻辑
│     ├── 许多 SP（Streaming Processor）← 并行执行 thread
│     ├── Tensor Cores（专用 matmul，吃低精度）
│     ├── Registers
│     └── Shared Memory / L1
├── L2 Cache（on-die）
└── Global Memory / HBM（片外 DRAM，最慢）
```

编程抽象 ↔ 硬件：

| 抽象 | 硬件/存储 | 直觉 |
|------|-----------|------|
| Grid | 整块 GPU / HBM | 一次 kernel launch 的全部工作 |
| Block / CTA | 一个 SM / shared memory | 能协作的一组线程 |
| Thread | 寄存器 + 执行单元 | 处理少量元素 |
| Warp | SM 调度单位 | **32 个 thread 锁步执行** |

调度：kernel 启动许多 block → block 分配到 SM → block 内按 warp 调度到 SP 上执行。

访问规则：
- Device 代码可读：每 thread 的 register（最快）、每 block 的 shared memory、每 grid 的 global memory（最慢）、constant memory（只读）
- Host(CPU) 主要负责 global/constant 的数据搬运
- **跨 block 交换信息基本要走 global memory**  
  理想情况：thread 在 register / shared 上完成大部分计算，只把最终结果写回 HBM

## 1.3 先会算账：Roofline

Arithmetic intensity ≈ FLOPs / 搬运字节数  

| 区域 | 瓶颈 | 含义 |
|------|------|------|
| 左侧 memory-bound | 带宽 | 算得再快也在等数据 |
| 右侧 compute-bound | 峰值算力 | 数据够，算力打满 |

目标：提高 intensity，尽量爬出 memory-bound。  
现代 ML 很多 kernel 其实是 **Memory is King**——优先减 HBM 访问，而不是死磕 FLOPs。

Matmul 吞吐随矩阵尺寸 $N$ 不平滑：大体呈 roofline（先升后平台），细节上有波浪/阶跃——后面用 tiling + wave quantization 解释。

## 1.4 加速技巧

### 控制流：warp divergence

Warp 内 32 thread **必须同一条指令**。  
`if/else` 分叉 → 两支路径串行执行，没走该路径的 thread 空转。  
热路径上少分歧；尾部 mask 通常可接受。

### Trick 1  低精度 low precision

位数更少 → 搬运更少 bit → 等效提高带宽。

例子 elementwise ReLU：
- FP32：1R+1W 各 4B → ~8 bytes/FLOP 量级
- FP16：各 2B → ~4 bytes/FLOP

硬件侧 Tensor Core：常见 16-bit 输入、32-bit 累加、再写回 16-bit。  
训练里还要记 mixed precision：参数/激活用 BF16/FP16，optimizer 侧常有 FP32 master weights；FP16 有时需要 loss scaling，BF16 动态范围更接近 FP32 更省心。  
更低还有 FP8 / MXFP 等，scale 分组、转置不再 trivial，实践里不是所有 weight 都同一格式。

### Trick 2  算子融合 operator fusion

朴素：每步写回 HBM，下一步再读出。  
融合：中间结果留在 register / shared memory，只写最终结果。

![TorchInductor 算子融合：sin、cos、pow、add 等多个逐元素算子被合成一个 kernel。](/notes-assets/cs336/pasted-20260711153205.png)

*TorchInductor 算子融合：sin、cos、pow、add 等多个逐元素算子被合成一个 kernel。*

工厂隐喻：仓库 = HBM，工厂 = SM。反复出货入库很傻。  
fusion 的主收益通常是 **消中间张量 + 少 launch**，不是少做数学。  
简单 fusion 编译器也能干（`torch.compile`）。

### Trick 3  重计算 recomputation / checkpointing

反向需要前向激活。全存：
- 激活显存爆炸
- 反向大量读 HBM → intensity 极差

丢掉中间激活，反向时重算：多算一点，换更少 HBM 读写 + 省显存。  
算力过剩、带宽稀缺时，**故意多算可以更快**。  
gradient checkpointing 同源；FlashAttention 反向是 tile 级 recompute。

### Trick 4  memory coalescing 与 DRAM burst

DRAM：打开一行贵，一旦打开可廉价读一整段 **burst**。  

定义：warp 内所有 thread 的访存落在同一 burst → 一次高效读服务多个 thread。  
未合并：每个 thread 打到不同 burst → 极慢。

注意两层：
- **coalescing**：HBM/global 上连续访问
- **bank conflict**：shared memory 内部，32 bank 撞同一 bank 会串行化（padding / swizzle 缓解）

布局 + thread 映射决定能不能 coalescing。

### Trick 5  tiling（重点）

思想：分组线程、重排计算顺序，**最小化 global memory 访问**。

朴素 matmul：访存未合并 + 同一输入元素被重复 global 读很多次。

![分块矩阵乘（tiling）：每个输出块只需读入对应的行块与列块。](/notes-assets/cs336/pasted-20260711153824.png)

*分块矩阵乘（tiling）：每个输出块只需读入对应的行块与列块。*

Tiled matmul 流程：
1. 大矩阵切小 tile
2. $M$、$N$ 的 tile 加载进 shared memory
3. 片上算部分积，累加
4. 沿 reduction 维换下一组 tile
5. 最终结果写回 global

量级：非 tiling 每个输入约读 $N$ 次 global；tile 边长 $T$ 时约读 $N/T$ 次 global（复用在 shared 上）。  
Global 访问约减 $T$ 倍。

代价与坑：
- **tile quantization**：维度不能整除 → 尾部 tile 利用率低
- shared memory 容量限制最大 tile
- 加载要对齐才能 coalescing；必要时 padding
- **wave quantization**：tile 数跳变导致多一波、SM 吃不满  
  例：A100 108 SM，$N=1792$ 刚好一轮，$N=1793$ tile 数暴涨 → 吞吐断崖  
  实践：词表 $50257\to50304$（凑对齐）能白捡加速

技巧归类：
| 类别 | 手段 |
|------|------|
| 减少访存 | coalescing、fusion |
| 数据搬近计算 | tiling → shared |
| 用算力/精度换带宽 | low precision、recomputation |

---

## 1.5 FlashAttention

标准 attention：

$$
\text{Attention}(Q,K,V)=\mathrm{softmax}\!\left(\frac{QK^\top}{\sqrt{d}}\right)V
$$

本质是 matmul + 中间 softmax。  
朴素实现物化完整 $N\times N$ 注意力矩阵 → HBM 读写巨大 → memory-bound，长上下文更惨。

### 机制 1：对 QKV matmul 做 tiling

把 $Q,K,V$ 切成能放进 shared memory 的 block，tile-wise 算 $S=QK^\top$ 局部块，**不在 HBM 写满整个 $N\times N$**。

![GPU 存储层次与 FlashAttention 的分块计算：注意力块在 SRAM 中完成，N×N 矩阵不写回 HBM。](/notes-assets/cs336/pasted-20260711154736.png)

*GPU 存储层次与 FlashAttention 的分块计算：注意力块在 SRAM 中完成，N×N 矩阵不写回 HBM。*

问题：softmax 要全局归一化，似乎破坏 tile 局部性？

### 机制 2：online softmax

普通 softmax：先整行 max，再 $\sum e^{x_i-m}$ —— 需要整行一次可见。  
Online softmax：维护 running max / running sum，读新 tile 时更新，用 rescale（望远镜求和）修正旧统计量 → **可 tile-by-tile 流式算**，最终仍正确。

![在线 softmax：按块累积分母并重新缩放已有输出，从而无需物化完整的注意力矩阵。](/notes-assets/cs336/pasted-20260711154909.png)

*在线 softmax：按块累积分母并重新缩放已有输出，从而无需物化完整的注意力矩阵。*

### 合在一起：前向

1. tile-wise 算内积块 $S$
2. exp / softmax 与 matmul 融在同一 kernel，中间大矩阵不落 HBM
3. online softmax 跨 tile 正确归一化
4. 增量累加输出 $O$（对 $V$ 的加权）

**反向（本课不细讲）：** 按 tile 重算需要的中间量，避免存完整 $N\times N$。典型 recomputation 用法。

效果直觉：更少 HBM traffic → 数倍加速 + 显存大降 → 更长上下文可行。

| FlashAttention 组件 | 对应技巧 |
|---------------------|----------|
| blocked $QK^\top$、$PV$ | tiling |
| 不物化 $N\times N$ | 减 global 访存 |
| softmax+exp 与 matmul 同 kernel | fusion |
| online softmax | 让 tiling 与全局归一化兼容 |
| 反向不存完整 attention | recomputation |

---

## PART 2  Kernels and Triton

上一讲回答「GPU 为什么能快」；这一讲回答「程序怎么真的跑快」。  
循环：benchmark 测 → profiler 找瓶颈 → 减搬运/增复用 → 再测。  
敌人通常不是 FLOPs 不够，是 HBM 搬得太多或搬法不对。

## 2.1 编程模型再贴一层

Triton 关键抽象：描述 **一个 program/block 处理什么**；  
CUDA 更接近描述 **每个 thread 做什么**。

Occupancy = SM 上实际驻留 warp 数 / 上限。  
每线程多用寄存器 → 驻留 warp 变少；但 thread coarsening（每线程多干一点）可能更好。  
**Occupancy 是约束，不是目标。**

Benchmark 纪律：warmup（JIT/内核选择）、`torch.cuda.synchronize` 或 CUDA event（异步）、多次测、固定 shape/dtype/layout。  
小矩阵时间可能几乎不随 $n$ 变（launch 开销主导）；够大后才显现 $O(n^3)$ 计算缩放。

## 2.2 Triton 最小骨架

```python
pid = tl.program_id(0)
offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
mask = offsets < n
x = tl.load(x_ptr + offsets, mask=mask)
y = f(x)
tl.store(y_ptr + offsets, y, mask=mask)
```

- `program_id`：当前 Triton program/block
- `tl.arange`：block 内向量索引
- mask：防尾部越界
- 编译器把向量程序映射到 threads/warps → PTX → 机器码

## 2.3 四类 kernel 模式

### GeLU：elementwise

每个输入只影响对应输出，不需线程间通信。  
重点：连续访问、mask、融合、合适 block size。  
朴素多算子 GeLU 会多次 HBM 读写；fused / `torch.compile` 理想路径是 1 读 + 寄存器算完 + 1 写。

### Softmax：行内 reduction

$$
\operatorname{softmax}(x_i)=\frac{e^{x_i}}{\sum_j e^{x_j}}
$$

数值稳定：

$$
m=\max_j x_j,\qquad y_i=\frac{e^{x_i-m}}{\sum_j e^{x_j-m}}
$$

整行能塞进一个 block：一个 program 管一行，load → max → exp → sum → 除 → store。  
列数不是 2 的幂：block size 取下一个 2 的幂 + mask；mask 位在 max 当 $-\infty$，在 sum 当 0。  
朴素分算子 softmax 多个中间张量；融合只需一读一写——FlashAttention 的思想基础。

### Row sum：一行放不进一个 block

1. 行切多个 tile  
2. 每个 lane 循环跨 tile 累加  
3. block 内 accumulator 做最终 reduction  
4. 写出一个标量  

最小型 tiling：循环装载 + 局部累计 + 最后归约。

### Matmul + ReLU：二维 tiling

1. 每个 block 负责 $C$ 的一个 $B_M\times B_N$ 输出 tile  
2. 沿 $K$ 迭代，加载 $A$ 的 $B_M\times B_K$ 与 $B$ 的 $B_K\times B_N$  
3. 片上 `tl.dot` 累加  
4. accumulator 上做 ReLU  
5. 只把最终输出写回 HBM  

同一块 $A/B$ 被多次乘加复用 → intensity 随 tile 尺寸上升。  
地址：$\mathrm{addr}(i,j)=\mathrm{base}+i\cdot\mathrm{stride}_0+j\cdot\mathrm{stride}_1$。  
边界 tile 用 mask，越界输入填 0。

实用顺序：先写对 PyTorch reference → 边界/非整除 shape 验正确性 → warmup 测 → profiler 看 kernel 数与访存 → 优先 fusion/tiling → 再看 coalescing / bank / divergence / 寄存器压力。

---

## PART 3  Distributed Parallelism

单 GPU 内躲 HBM；多 GPU 躲慢链路。  
多 GPU 两个原因：
1. 参数/梯度/optimizer/activation **放不下**（capacity）
2. 想用更多 FLOPs **缩短时间**（speed）

任何 tensor 三条路：放本卡显存 / 需要时重算 / 放别的 GPU 再通信取回。

## 3.1 基本概念

| 词 | 含义 |
|----|------|
| rank | 进程/设备编号 |
| world size | 参与通信的 rank 总数 |
| process group | 参与某组 collective 的 ranks |
| backend | CPU 常用 gloo，NVIDIA GPU 常用 nccl |
| barrier | 所有 rank 到齐才继续（同步进度，**不搬训练张量**） |

`torch.distributed` 通常一卡一进程；`MASTER_ADDR`/`MASTER_PORT` 用于 rendezvous，不是所有数据都绕 rank 0 转发。

## 3.2 Collective communication

用整体通信模式描述多设备数据流，比手写 send/recv 好优化。NCCL 负责：探拓扑 → 选 ring/tree 等 → 起 GPU 通信 kernel。应用只说 `all_reduce`，物理路径可能完全不同。

| 操作 | 输入→输出 | 典型用途 |
|------|-----------|----------|
| Broadcast | root 数据复制给所有 rank | 下发初始权重/配置 |
| Scatter | root 大切开分给各 rank | 分发 shard |
| Gather | 各 shard 汇总到 root | 收集结果 |
| Reduce | 汇总 sum/min/max 到 root | 聚合统计量 |
| All-gather | 每 rank 持 shard → 每 rank 都有完整拼接 | FSDP 前向按需集参数 |
| Reduce-scatter | 先 reduction 再分片 | 梯度要求和且结果本身分片存 |
| All-reduce | 每 rank 都得到完整 reduction | DDP 同步梯度 |
| All-to-all | 每人给每人发不同片（像转置） | MoE token↔expert 路由 |

关键等式：

$$
\text{all-reduce}=\text{reduce-scatter}+\text{all-gather}
$$

DDP 要完整同步梯度 → all-reduce；ZeRO/FSDP 拆开两步，让状态保持分片。

Ring all-reduce 量级：大消息带宽受限时，每 rank 移动量趋近 $2\frac{P-1}{P}N$（$N$ 为张量大小）。小消息更吃 latency + 同步。

All-to-all 在 MoE 里关键：token 路由到持有对应 expert 的 rank；负载不均时 **最慢 rank 决定 step time**。

## 3.3 拓扑直觉（决定并行放哪）

典型节点：8 GPU + NVLink/NVSwitch；节点间 InfiniBand/RoCE，量级可差一个数量级以上。  
粗规则：
- **高频、重 collective 的 TP** 放节点内 NVLink 域
- **只在 stage 边界传 activation 的 PP** 更能忍跨节点慢链路
- rank 物理放置必须匹配通信模式；逻辑相邻 ≠ 链路快

RDMA：设备直接读写远端内存，少绕 CPU 拷贝。

## 3.4 Data Parallelism / DDP

切分：沿 **batch** 维。每 rank 完整模型，不同样本。

一步：
1. 本地 batch 前向  
2. 本地反向得不同梯度  
3. 对参数梯度 all-reduce 平均  
4. 各 rank 相同梯度做 optimizer step  
5. 初始参数一致 + 更新相同 → 副本始终 **一致**

$$
g=\frac1P\sum_{r=1}^{P}g_r
$$

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

优点：实现简单、算力扩展自然。  
缺点：每卡完整参数/梯度/optimizer，显存不随 GPU 数掉；通信量与参数量相关；global batch 随 world size 涨，超过 critical batch 收益递减。

真实 DDP：参数分 bucket，某 bucket 梯度就绪就异步 all-reduce，与剩余 backward 重叠——`async_op` 只有延迟 wait 且中间真有计算才算 overlap。

混合精度 Adam 粗算可达 ~16 bytes/param（BF16 参数+梯度 + FP32 master + 两个 moment），还没算 activation。

## 3.5 Tensor Parallelism

切分：沿层的 **宽度** 切参数。

列切 $W=[W_1,\ldots,W_P]$：

$$
Y=[XW_1,\ldots,XW_P]
$$

课程 baby 版：
1. 各 rank 都有完整输入 $X$  
2. rank $r$ 只持 $W_r$  
3. 局部 $XW_r$ + GeLU  
4. all-gather 激活拼成完整宽度  
5. 进下一层  

行切则各 rank 部分和，再 all-reduce / reduce-scatter。

Megatron 常见配对：QKV / MLP up 用 column parallel；attention out / down 用 row parallel → 两层之间可保持 sharded，少 gather。

特点：
- 参数和部分 activation 随分片  
- 无 pipeline bubble，不太依赖大 batch  
- **每层 blocking collective**，极吃低延迟高带宽  
- 切太细 → 本地 matmul 变小 → 利用率掉  
→ TP degree 通常 ≤ 节点内 GPU 数

## 3.6 Pipeline Parallelism

切分：沿 **深度** 切层。前向传 activation，反向传 activation gradient。

整 batch 串 stages：每卡大约只有 $1/P$ 时间在干活 → bubble 巨大。  
解决：batch 切成 $M$ 个 **microbatch** 流水填充。

Baby 版流程：
1. rank 0 把 batch `chunk` 成多个 microbatch  
2. stage 收上一 stage 的 activation  
3. 跑本地层  
4. `dist.send` 给下一 stage  
5. 多个 microbatch 连续流动  

理想化 bubble 比例约：

$$
\frac{P-1}{M}
$$

$M$ 越大填充/排空占比越低，但 activation 缓存、调度、有效 batch 都变。  
真实系统还要 1F1B 等 schedule；课程代码多只 demo forward。

优点：参数显存随 stage 降；相邻 stage 点对点，通信量常比逐层 TP collective 小 → 适合跨节点。  
难点：负载均衡、bubble、activation 生命周期、F/B 调度与 overlap。

## 3.7 ZeRO / FSDP：数据并行上的状态分片

洞察：all-reduce = reduce-scatter + all-gather；reduce-scatter 后结果本就分片，不必立刻恢复成每卡完整副本。

### Stage 1：切 optimizer state

每卡仍有完整参数和梯度，只存 $1/P$ optimizer state。

1. 各卡算完整梯度  
2. reduce-scatter 梯度 → 每卡拿到自己负责的 shard  
3. 用本地 optimizer state 更新参数 shard  
4. all-gather 参数 → 每卡重新有完整模型  

通信体积 bandwidth-bound 下仍约 $2N$（和 DDP all-reduce 同阶），几乎「同通信换 optimizer 显存」。

### Stage 2：再切 gradient

参数仍完整；梯度 + optimizer 分片。  
某层梯度一出来就 reduce-scatter，本地完整梯度立刻释放 → 更新 shard → all-gather 参数。

### Stage 3 / FSDP：参数也切

参数、梯度、optimizer 都只留 $1/P$。  
按 block：all-gather 参数 → 算 → 立刻 reshard/free；backward 再按需 gather；梯度 reduce-scatter 分片存。

通信粗算约 $3N$（前向 gather $N$ + 反向 gather $N$ + 梯度 reduce-scatter $N$），相对 DDP 约 1.5×，换参数状态约 $1/P$。  
不能一次 gather 整个模型——会瞬间恢复完整显存；正确粒度是 module/block，并可 prefetch 藏通信。

ZeRO 没解决的：
- Stage 1/2 参数仍整份，超大模型可能仍放不下  
- Stage 3 不自动降 activation memory  
- DP 仍要够大的 global batch  

## 3.8 多轴怎么叠（统一视图）

| 轴 | 方法 | 切什么 | 主要通信 |
|----|------|--------|----------|
| Batch | DP / ZeRO | 样本 | 梯度 all-reduce 或 RS+AG |
| Width | TP / EP | hidden 或 experts | 激活 collective / all-to-all |
| Depth | PP | layers | 相邻 stage activation |
| Sequence | SP / CP | token 位置 | activation / KV 交换 |

真实大模型是组合，不是四选一。  
原则：最频繁最重的通信放最快链路；先解决「放得下」，再用剩余 GPU 做 DP 提吞吐。

| 策略         | 一句话                         |
| ---------- | --------------------------- |
| DDP        | 复制模型，同步梯度；简单，显存不扩展          |
| ZeRO-1/2/3 | 仍是数据并行，但状态分片；3≈FSDP，通信略涨换容量 |
| TP         | 层内切开，每层重通信；吃 NVLink         |
| PP         | 层间切开，边界通信；能跨节点，有 bubble     |
| EP         | expert 维 + all-to-all；怕负载不均 |

---

## 收束

单卡与多卡是同一故事的两端：

1. **数据离计算有多远** —— register/shared vs HBM vs 跨卡链路  
2. **搬多少、搬几次** —— fusion / tiling / 分片 / 重算  
3. **能不能和计算重叠** —— bucket、async collective、prefetch  

Collective 是分布式训练的「汇编」；DDP / FSDP / TP / PP 只是对这些 primitive 的不同编排。  
会做快算法 ≈ 会 coalescing + tiling + fusion，并在需要时用 recompute 和低精度换带宽；  
会做大规模训练 ≈ 会按拓扑选并行轴，并会算通信体积与 bubble。
