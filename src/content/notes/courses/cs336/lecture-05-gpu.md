---
title: "CS336 Lecture 5：GPU"
description: "为什么 GPU 有时莫名变慢：存储层次、roofline、coalescing、tiling 与算子融合，以及 FlashAttention 为什么快。"
tags:
  - "cs336"
  - "gpu"
  - "systems"
category: "courses"
series: "CS336"
date: "2026-07-11"
updated: "2026-07-11"
---

> **CS336 — Tatsu H**

---

## 0. 本讲目标与大纲

**目标：**
1. **祛魅 CUDA / GPU**：理解为何 GPU 有时会莫名其妙地慢（例如 matmul 吞吐随尺寸呈波浪形）
2. **学会做快算法**：掌握 coalescing、tiling、fusion 等原语，能理解 Flash Attention 为何快

![Matmul 吞吐的“神秘波浪”示例](/notes-assets/cs336/lecture5/p02_img2_xref27.png)

**结构：**
| Part | 内容 |
|------|------|
| **Part 1** | GPU 原理：硬件结构、执行模型、内存层次 |
| **Part 2** | 如何让 ML workload 在 GPU 上变快 |
| **Part 3** | 综合应用：拆解 Flash Attention |

---

## Part 1: GPU 深度解剖

### 1.1 为什么关心 GPU？

- 语言模型性能往往随 **compute** 可预测地提升（Kaplan 等 scaling laws）
- 更快的硬件、更高利用率、更好的并行，本身就能推动进展

#### Dennard Scaling → Parallel Scaling

| 阶段 | 时间 | 机制 | 现状 |
|------|------|------|------|
| **Dennard Scaling** | 1980s–2000s | 晶体管更小 → 更高时钟 + 更低功耗 → 单线程更快 | 已基本结束 |
| **Parallel Scaling** | 2010s–至今 | 用 GPU 做海量并行 | 过去约 10 年 GPU 性能 > **1000×** |

![Dennard scaling 已到尽头](/notes-assets/cs336/lecture5/p06_img1_xref55.jpeg)


> **没有 GPU scaling，就没有 LLM scaling。**

并行 scaling 的驱动力（Bill Dally HotChips）：
- 数值表示：FP32 → FP16 / BF16 / INT8
- 专用指令：Tensor Core 等
- 工艺节点、稀疏性等

---

### 1.2 CPU vs GPU

| 维度 | **CPU** | **GPU** |
|------|---------|---------|
| 优化目标 | **延迟（latency）** | **吞吐（throughput）** |
| 线程数 | 少量快速线程 | 海量轻量线程 |
| 控制逻辑 | 复杂分支预测、大缓存 | 控制简单，少支持分支 |
| 计算单元 | 少量强大核心 | 大量小 ALU |
| 设计哲学 | 每个任务尽快完成 | 单位时间处理尽量多数据 |

![CPU 优化延迟 vs GPU 优化吞吐](/notes-assets/cs336/lecture5/p08_img1_xref62.png)


- CPU：少数强工人 + 复杂管理员 + 大本地仓库
- GPU：大量简单工人 + 极简管理 → 总产量优先

---

### 1.3 GPU 执行单元解剖

```
GPU
├── 许多 SM（Streaming Multiprocessor）  ← 独立执行 block
│     ├── 控制逻辑
│     ├── 许多 SP（Streaming Processor）← 并行执行 thread
│     ├── Tensor Cores（专用 matmul）
│     ├── Registers
│     └── Shared Memory / L1
├── L2 Cache（on-die）
└── Global Memory / HBM（片外 DRAM，最慢）
```

**要点：**
- **SM**：GPU 上的原子执行单元，可独立跑不同 block
- **SP**：SM 内的线程执行单元
- **SIMT**（Single Instruction, Multiple Threads）：同一指令施加到不同数据上

---

### 1.4 GPU 内存层次

| 层级                           | 位置   | 速度                | 成本     | 作用域         |
| ---------------------------- | ---- | ----------------- | ------ | ----------- |
| **Registers**                | SM 内 | 最快                | 最贵     | 单 thread    |
| **Shared Memory / L1**       | SM 内 | 很快（~20–33 cycles） | 贵      | 同 block 内共享 |
| **L2 Cache**                 | 片上   | 中等（~200 cycles）   | 中      | 全 GPU       |
| **Global Memory (HBM/DRAM)** | 片外   | 最慢（~290+ cycles）  | 便宜、容量大 | 全 grid      |

![内存越靠近 SM 越快](/notes-assets/cs336/lecture5/p10_img1_xref69.jpeg)

![Shared / L1 / L2 / Global 层次](/notes-assets/cs336/lecture5/p10_img2_xref70.jpeg)


- **跨 block 通信必须走 global memory（慢）**
- 核心原则：让数据尽量待在靠近 SM 的快速内存里

---

### 1.5 执行模型：Thread / Block / Warp

三个关键角色：

| 概念 | 定义 | 关键性质 |
|------|------|----------|
| **Thread** | 真正干活的单元 | 同一指令、不同数据（SIMT） |
| **Block** | 一组 thread | 绑定到一个 **SM**，共享 **shared memory** |
| **Warp** | **32** 个连续编号的 thread | 一起调度、一起执行同一指令 |

**调度流程：**
1. Kernel 启动许多 **block**
2. 每个 block 分配到某个 **SM**
3. Block 内按 **warp（32 threads）** 调度到 SP 上执行

---

### 1.6 内存模型（访问规则）

**Device（GPU）代码可访问：**
- 每 thread 的 **register**（读写，最快）
- 每 block 的 **shared memory**（读写）
- 每 grid 的 **global memory**（读写，最慢）
- **constant memory**（只读）

**Host（CPU）：** 主要负责 global / constant memory 的数据搬运

> 跨 block 的信息交换 → 必须经 global memory。理想情况是：thread 在 shared memory / register 上完成大部分计算。

---

### 1.7 旁支：TPU 呢？

在高层，**GPU、TPU 等加速器结构类似**：
- 轻量控制
- 巨大/快速的 **matmul 单元**
- 快速片上内存 + HBM

![GPU / TPU 高层结构相似](/notes-assets/cs336/lecture5/p13_img1_xref81.png)

| 对比   | GPU          | TPU                      |
| ---- | ------------ | ------------------------ |
| 并行单元 | 更多 SM        | 更少但更强的 Tensor Core       |
| 执行模型 | Warp / Block | 无 warp 概念，更“为 matmul 而生” |
| 网络拓扑 | 另讲并行时讨论      | 不同（XLA / ICI 等）          |
|      |              |                          |

---
### 1.8 GPU 模型的优势

1. **易横向扩展**：加更多 SM 即可放大算力
2. **SIMT 编程相对自然**：对矩阵/张量数据并行友好
3. **线程轻量**：可快速切换，用并发隐藏访存延迟（latency hiding）

---

### 1.9 GPU 作为“快速矩阵乘法器”

| 时代 | 特征 |
|------|------|
| 早期 | 可编程 shader 被“黑”来做 matmul |
| 现代 | **Tensor Core**（Volta / Turing 起）= 专用 matmul 电路 |

- Matmul 可比其他浮点运算快 **>10×**
- **架构设计启示**：应尽量把工作量组织成 matmul；非 matmul 算子往往成为瓶颈

---

### 1.10 Compute 增速远快于 Memory

| 资源 | 约 20 年增长量级 | 每 2 年倍率（约） |
|------|------------------|-------------------|
| **HW FLOPs** | ~60,000× | ~3.0× |
| **DRAM 带宽** | ~100× | ~1.2× |
| **Host–GPU 互联** | ~20× | ~1.1× |

![Memory Wall：FLOPs 远快于内存带宽](/notes-assets/cs336/lecture5/p18_img1_xref99.jpeg)

> **Memory wall**：算力喂不饱——现代 ML 往往是 **memory-bound**，而非单纯 FLOP-bound。  
> 高效算法必须优先 **减少数据搬运**。

---

### 1.11 Part 1 小结

- GPU = 大规模并行：同一指令作用在大量 worker 上
- **Compute（尤其 matmul）比 memory 增长更快**
- 必须尊重 **memory hierarchy**，才能真正跑满硬件

---

## Part 2: 如何让 ML Workload 在 GPU 上变快

### 2.1 现象：Matmul 吞吐的“神秘波浪”

对正方形 matmul，吞吐（TFLOP/s）随矩阵尺寸 **N** 并不平滑上升，而是：
- 总体呈 roofline 形态（先升后平台）
- 细节上出现 **波浪 / 周期跳变**

![简单方阵 matmul 也会出现奇怪性能曲线](/notes-assets/cs336/lecture5/p20_img1_xref107.jpeg)

后面用 **arithmetic intensity、tiling、wave quantization、对齐** 解释。

---

### 2.2 Roofline 模型

| 区域 | 瓶颈 | 含义 |
|------|------|------|
| **左侧 Memory-bound** | 内存带宽 | 算得再快也等数据 |
| **右侧 Compute-bound** | 峰值算力 | 数据够，算力打满 |

![Roofline：如何避免 memory-bound](/notes-assets/cs336/lecture5/p21_img1_xref110.jpeg)

**Arithmetic intensity**（算术强度）≈ FLOPs / 搬运字节数  
目标：**提高 intensity，尽量摆脱 memory-bound**。

---

### 2.3 加速技巧总览

| # | 技巧 | 本质 |
|---|------|------|
| 1 | Control divergence | 控制流问题（非 memory，但伤吞吐） |
| 2 | Low precision | 少搬 bit → 等效提高带宽 |
| 3 | Operator fusion | 中间结果不落 HBM |
| 4 | Recomputation | 用算力换访存 |
| 5 | Memory coalescing | 利用 DRAM burst |
| 6 | **Tiling** | 把数据搬进 shared memory 复用 |

---

### 2.4 Control Divergence（控制发散）

- Warp 内 32 个 thread **必须执行同一指令**
- 若出现 `if/else` 分叉：两支路径 **串行执行**，未走该路径的 thread 空转
- 条件分支可以写，但在热路径上代价很高

![SIMT 下的 control divergence](/notes-assets/cs336/lecture5/p23_img1_xref115.jpeg)

**启示：** 性能关键路径上尽量减少 warp 内分歧。

---

### 2.5 Trick 1：低精度计算（Low Precision）

**核心：位数更少 → 搬运更少。**

![低精度：更少 bit 需要搬运](/notes-assets/cs336/lecture5/p24_img1_xref119.jpeg)

#### 例子：Elementwise ReLU，$x = \max(0, x)$

| 精度 | 访存 | 运算 | Intensity（字节/FLOP 量级） |
|------|------|------|---------------------------|
| **FP32** | 1 read + 1 write，各 4B | 1 compare + 1 FLOP | ~8 bytes / FLOP |
| **FP16** | 同样 1R+1W，各 2B | 同上 | ~4 bytes / FLOP |

→ 精度减半，**访存减半，等效带宽提升**。

#### 硬件侧：Tensor Core
- 现代 GPU 用 Tensor Core 加速低精度 / 混合精度 matmul
- 常见：16-bit 输入、32-bit 累加、再写回 16-bit

#### 前沿：更低精度

| 格式 | 要点 |
|------|------|
| **FP8** | 不同动态范围 / 尾数权衡 |
| **MXFP8（Blackwell）** | 多用 E4M3；scale 本身为 FP8（E8M0），约每 32 个一组；**转置不再 trivial** |
| **MXFP4** | 可表示值极少；约 1/16 的 scale，E4M3 scale |

![FP8 与多种 scale 权衡](/notes-assets/cs336/lecture5/p27_img1_xref153.png)

实践中（MXFP8 训练）：
- 并非所有 weight 都是 MXFP8
- transpose 往往 **单独再量化**

---

### 2.6 Trick 2：算子融合（Operator Fusion）

**工厂隐喻（Horace He）：**
- 仓库 = global memory
- 工厂 = SM 上的计算
- 反复“出货–入库–出货”很傻

![工厂隐喻：compute 涨得比 memory 快](/notes-assets/cs336/lecture5/p30_img1_xref169.jpeg)

![Compute 与 memory 的失衡](/notes-assets/cs336/lecture5/p30_img2_xref170.png)

| 方式 | 行为 |
|------|------|
| **Naive（未融合）** | 每步写回 HBM，下一步再读出 |
| **Fused kernel** | 中间结果留在 register / shared memory，只写最终结果 |

![Naive vs Fused kernel](/notes-assets/cs336/lecture5/p31_img1_xref178.jpeg)

![融合后中间结果不再来回搬运](/notes-assets/cs336/lecture5/p31_img2_xref179.jpeg)

#### 例子：$\sin^2 x + \cos^2 x$
- 朴素实现：可能启动 **5 个** CUDA kernel（多次往返 HBM）
- 融合后：**1 个** kernel 完成全部 pointwise 运算

![sin²+cos² 朴素实现要 5 个 kernel](/notes-assets/cs336/lecture5/p32_img1_xref182.jpeg)

![5 个 pointwise 融合成 1 个 kernel](/notes-assets/cs336/lecture5/p33_img1_xref217.jpeg)

**自动化：** 简单 fusion 可由编译器完成（如 `torch.compile`）。

---

### 2.7 Trick 3：重计算（Recomputation / Checkpointing）

#### 问题
反向传播需要前向激活。若全部存盘：
- 占用大量激活内存
- 反向时大量 **读 HBM** → 低 arithmetic intensity

![反向需要存激活并算 Jacobian](/notes-assets/cs336/lecture5/p34_img1_xref222.jpeg)

#### 例子：叠 3 个 sigmoid
- 正常：约 **8 次** mem read/write，强度极差
- **丢掉中间激活，反向时重算**：访存可降到约 **5/8**

![全存激活：访存很差](/notes-assets/cs336/lecture5/p35_img1_xref225.jpeg)

![丢掉再算：访存可降到约 5/8](/notes-assets/cs336/lecture5/p36_img1_xref228.jpeg)

| 策略 | 访存 | 计算 |
|------|------|------|
| 全存激活 | 多 | 少 |
| 重计算 | **更少** | 更多 |

> 在 **算力过剩、带宽稀缺** 时，**故意多算可以更快**（也省显存）。  
> 与 gradient checkpointing 同源；Flash Attention 反向也用 tile 级重计算。

---

### 2.8 Trick 4：Memory Coalescing 与 DRAM Burst

#### DRAM 为何按“突发”读？
- 打开一行、送到 sense amplifier 很慢
- 一旦打开，可廉价读出 **一整段 burst**（很多字节）

![DRAM burst mode](/notes-assets/cs336/lecture5/p37_img1_xref231.png)

![Sense amplifier 与按行读取](/notes-assets/cs336/lecture5/p37_img2_xref232.jpeg)

#### Coalesced access 定义
> Warp 内所有 thread 的访存落在 **同一 burst** 内 → 一次高效读服务多个 thread。

![Coalescing：warp 落在同一 burst](/notes-assets/cs336/lecture5/p38_img1_xref235.png)

**提醒：** Warp = 32 个连续 thread，访存一起发生。

#### 对 matmul 的含义（row-major）
- 若 thread 沿错误方向跨步：每个 thread 打到不同 burst → **未合并，极慢**
- 若布局使连续 thread 访问连续地址：一次 burst 喂满 warp → **合并，快**

![Matmul 中未 coalesced vs coalesced 访问](/notes-assets/cs336/lecture5/p39_img1_xref239.jpeg)

![按正确方向读取可一次吃满向量](/notes-assets/cs336/lecture5/p39_img2_xref240.jpeg)

> 内存布局 + thread 映射决定能不能 coalescing。

---

### 2.9 Trick 5（重头戏）：Tiling

**思想：** 把线程分组、重排计算顺序，**最小化 global memory 访问**。

#### 朴素 matmul 的问题
- 访存未合并
- 同一输入元素被 **重复从 global 读** 很多次

![朴素 matmul：重复读 global、且未合并](/notes-assets/cs336/lecture5/p40_img1_xref243.jpeg)

#### Tiled matmul 流程
1. 把大矩阵切成小 **tile**
2. 把 $M$、$N$ 的 tile 加载进 **shared memory**
3. 在 shared memory 上算部分积，累加到 $P$
4. 换下一组 tile（沿 reduction 维推进）
5. 最终结果写回 global

![Tiling：把 tile 放进 shared memory 复用](/notes-assets/cs336/lecture5/p41_img1_xref247.jpeg)

**优点：**
- 重复读取发生在 **shared**，不是 global
- 更容易做 **coalesced** 加载

#### Tiling 数学
| 方式 | Global 读次数（量级） |
|------|----------------------|
| 非 tiling | 每个输入约读 **$N$** 次 |
| Tiling（tile 边长 $T$） | 每个输入约读 **$N/T$** 次 global + tile 内 $T$ 次 shared |

![Tiling 将 global 访问约减少 T 倍](/notes-assets/cs336/lecture5/p42_img1_xref250.png)

→ Global 访问约 **减少 $T$ 倍**。

#### Tiling 的复杂性

| 问题 | 说明 |
|------|------|
| **Tile 与维度整除** | 不能整除 → 尾部 tile 利用率低（**tile quantization**） |
| **Shared memory 容量** | 限制最大 tile 尺寸 |
| **Coalescing 约束** | tile 加载模式需对齐 burst |
| **Memory alignment** | 维度不合适时无法合并，需 **padding** |

![Tile 尺寸与利用率（tile quantization）](/notes-assets/cs336/lecture5/p43_img1_xref254.jpeg)

![对齐与 burst：维度不对就要 padding](/notes-assets/cs336/lecture5/p44_img1_xref266.jpeg)

**选择 tile 大小的因素：** coalescing、shared mem 大小、矩阵维度可整除性。

---

### 2.10 用这些知识解释 “Matmul 之谜”

![为什么更大矩阵有时反而更快？](/notes-assets/cs336/lecture5/p45_img1_xref279.jpeg)

综合 **compute intensity + tiling + 对齐** 后，仍有周期性行为：

#### Part 1：Tiling / 对齐的影响

![Tiling 与对齐对吞吐曲线的影响](/notes-assets/cs336/lecture5/p47_img1_xref294.jpeg)

#### Part 2：Wave quantization

- Tile 尺寸例如 $256 \times 128$
- $N = 1792$：$\frac{1792}{256} \times \frac{1792}{128} = 7 \times 14 = 98$ tiles
- $N = 1793$：变成 $8 \times 15 = 120$ tiles
- A100 有 **108 SM**，无法一轮跑完 120 tiles → 产生额外 wave，利用率骤降

![1792 → 1793：tile 数跳变与 wave quantization](/notes-assets/cs336/lecture5/p48_img1_xref298.jpeg)

→ 吞吐随 $N$ 呈 **波浪 / 阶跃**，而非单调。

**实践例子（Karpathy / nanoGPT）：**  
词表 $50257 \to 50304$（凑到 64 的倍数）→ 更好对齐与占用 → 约 **25%** 加速。

---

### 2.11 Part 2 小结：加速手段归类

| 类别 | 手段 |
|------|------|
| **减少访存** | Coalescing、Fusion |
| **把数据搬近计算** | Tiling（→ shared memory） |
| **用算力 / 精度换带宽** | Quantization、Recomputation |

---

## Part 3: 综合 — 理解 Flash Attention

### 3.1 标准 Attention 的计算

$$\text{Attention}(Q,K,V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d}}\right)V$$

本质：
1. 若干 **matmul**（$QK^\top$，再 $\times V$）
2. 中间夹一个 **softmax**

![Flash Attention 大幅加速 attention](/notes-assets/cs336/lecture5/p50_img1_xref305.png)

![标准 attention：3 次 matmul + 中间 softmax](/notes-assets/cs336/lecture5/p51_img1_xref310.jpeg)

朴素实现会 **物化完整 $N \times N$ 注意力矩阵** → HBM 读写巨大 → **memory-bound**，长上下文更惨。

---

### 3.2 技术 1：对 KQV Matmul 做 Tiling

论文 Figure 1 本质上就是：
- 把 $Q, K, V$ 切成能放进 **shared memory** 的 block
- **tile-wise** 计算内积 $S = QK^\top$（的局部块）
- 不在 HBM 上写满整个 $N \times N$ 矩阵

![Flash Attention Figure 1：本质是 KQV matmul 的 tiling](/notes-assets/cs336/lecture5/p52_img1_xref314.jpeg)

但问题：**softmax 是全局归一化**，一行要看所有 score——似乎破坏 tile 局部性？

---

### 3.3 技术 2：Online Softmax（增量 Softmax）

来源：Mikailov & Gimelshein 2018

| 方式 | 做法 | 问题 / 优点 |
|------|------|-------------|
| **Normal softmax** | 先扫整行找 max，再算 $\sum e^{x_i-m}$ | 需要整行一次性可见 |
| **Online softmax** | 维护 running max，增量更新；用 **望远镜求和（telescoping sum）** 修正旧统计量 | 可 **按 tile 流式** 计算 |

![Normal softmax vs Online softmax](/notes-assets/cs336/lecture5/p53_img1_xref317.png)

![Online softmax 允许 tile-by-tile 计算](/notes-assets/cs336/lecture5/p53_img2_xref318.png)

核心：不必一次拿到 $x_1,\ldots,x_n$；边读边更新 max 与归一化分母，最终仍得到正确 softmax。

→ Softmax 也能 **tile-by-tile** 完成。

---

### 3.4 合在一起：Flash Attention 前向

（Dao 2023 视角）

1. **Tile-wise** 计算内积块 $S$
2. **融合** 指数 / softmax 相关算子（不把中间大矩阵写回 HBM）
3. 用 **online / telescoping softmax** 跨 tile 正确归一化
4. 增量累加输出 $O$（对 $V$ 的加权）

![Flash Attention 前向：tiling + fusion + online softmax](/notes-assets/cs336/lecture5/p54_img1_xref321.jpeg)

**反向：** 本讲不细讲；思路是 **按 tile 重计算** 所需中间量，避免存完整 $N\times N$ 注意力（recompute 换 HBM）。

**效果（量级）：** 更少 HBM 流量 → 通常 **数倍加速** + 显存占用大降 → 更长上下文可行。

---

### 3.5 Flash Attention 与本讲技巧的对应

| Flash Attention 组件 | 对应技巧 |
|----------------------|----------|
| Blocked $QK^\top$、$PV$ | **Tiling** |
| 中间不物化 $N\times N$ | **减少 global 访存** |
| Softmax + exp 与 matmul 同 kernel | **Operator fusion** |
| Online softmax | 使 tiling 与 softmax 兼容 |
| 反向不存完整 attention | **Recomputation** |

---

## 全课 Recap

1. **硬件决定 scale**；底层细节决定什么能 scale、什么不能  
2. 当前 GPU 计算范式强烈鼓励：  
   - 把工作量做成 **matmul**  
   - 精心管理 **data movement**  
3. 认真对待 GPU：  
   - **Coalescing**  
   - **Tiling**  
   - **Fusion**  
   → 才能做出像 Flash Attention 这样的快算法  

![全课回顾：硬件 + matmul/data movement + coalescing/tiling/fusion](/notes-assets/cs336/lecture5/p55_img1_xref324.jpeg)

---

## 实用 Takeaways

| 原则 | 含义 |
|------|------|
| **Memory is King** | 现代 ML 常受带宽限制，优先减 HBM 访问 |
| **尊重内存层次** | 数据尽量留在 register / shared memory |
| **访问模式很重要** | 布局与 thread 映射决定能否 coalescing |
| **低精度有红利** | FP16/BF16/FP8… 既省显存又减搬运 |
| **该融合就融合** | 中间结果别无谓落盘 |
| **该重算就重算** | 带宽比算力更贵时，多算可更快 |
| **维度要对齐硬件** | padding 到 tile / warp / 64 等倍数常能白捡加速 |

---

## 延伸阅读

- Horace He: [Making Deep Learning Go Brrrr](https://horace.io/brrr_intro.html)
- [What shapes do matrix multiplications like?](https://www.thonking.ai/p/what-shapes-do-matrix-multiplications)
- Dao et al., Flash Attention / FlashAttention-2
- NVIDIA: DL performance — matrix multiplication / tile quantization
- Google: [Scaling book — GPUs](https://jax-ml.github.io/scaling-book/gpus/)
- AI and Memory Wall（compute vs memory 增速）
