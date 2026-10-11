---
title: "CS336 Lecture 6：Kernel、性能分析与 Triton"
description: "先 benchmark，再 profile，最后用 Triton 写 kernel：把对 GPU 硬件的理解变成真正跑得更快的程序。"
tags:
  - "cs336"
  - "triton"
  - "systems"
category: "courses"
series: "CS336"
date: "2026-07-11"
updated: "2026-07-11"
---

> Stanford CS336 Spring 2026 · 本讲主线：理解 GPU 硬件约束，测出瓶颈，再用 Triton 写出高效 kernel。

## 0. 一句话地图

上一讲回答“GPU 为什么快”；这一讲回答“怎样让程序真的跑快”。核心循环是：

1. 用 **benchmark** 测端到端时间；
2. 用 **profiler** 找时间花在哪里；
3. 根据硬件层次减少数据搬运、增加复用；
4. 修改 kernel 后重新测量。

真正的敌人通常不是 FLOPs 不够，而是数据在 HBM、shared memory、register 之间搬得太多或方式不对。

---

## 1. GPU 编程模型与硬件的对应关系

| 编程抽象               | 硬件/存储                 | 作用域         | 直觉                     |
| ------------------ | --------------------- | ----------- | ---------------------- |
| Grid               | 整块 GPU / HBM          | 全部 block    | 一次 kernel launch 的全部工作 |
| Thread block / CTA | 一个 SM / shared memory | block 内线程共享 | 能协作的一组线程               |
| Thread             | 寄存器与执行单元              | 单线程         | 处理少量元素                 |
| Warp               | SM 调度单位               | 32 threads  | 32 个线程锁步执行             |

元素级 GeLU 可以让每个线程独立处理元素；softmax、求和、矩阵乘法需要线程交换或复用数据，所以必须以 block 为协作单位。Triton 的关键抽象正是“描述一个 program/block 处理什么”，而 CUDA 更接近“描述每个 thread 做什么”。

### 1.1 A100、H100、B200 的层次感

课程给出的量级：A100/H100/B200 分别约有 108/132/148 个 SM；每个 SM 的寄存器约 256 KB，共享内存加 L1 为约 192/256/256 KB；HBM 带宽约 2/3.35/8 TB/s。寄存器带宽却可以达到数百 TB/s。

结论不是背数字，而是记住：**离计算单元越近，容量越小、速度越快。** 高性能 kernel 的任务就是让从 HBM 读进来的数据被尽量多用几次。

### 1.2 Warp divergence

一个 warp 的 32 个线程执行同一条指令。如果一部分进入 `if`，另一部分进入 `else`，硬件通常要把两条分支分时执行并屏蔽不相关线程。

因此：

- 数据相关的复杂分支会降低吞吐；
- 边界 mask 通常问题不大，因为只发生在少量尾部线程；
- 优化时应关注“同一 warp 内是否走不同控制流”。

### 1.3 Occupancy 不是越高越好

Occupancy 表示一个 SM 上实际驻留的 warp 数相对上限的比例。每线程使用更多寄存器，会减少能同时驻留的 warp；但每线程多做一些工作，即 **thread coarsening**，可能减少调度和访存，反而更快。

所以 occupancy 是约束，不是目标。要结合：

- 寄存器数量；
- 每 block 的线程数；
- 同时驻留的 block/warp；
- 是否有足够 warp 隐藏 HBM 延迟；
- 单线程做更多工作带来的复用收益。

### 1.4 Shared-memory bank conflict

Shared memory 通常分成 32 个 bank，每个 bank 每周期服务一个地址。若一个 warp 的多个线程访问同一 bank 的不同地址，访问会串行化，称为 bank conflict。

矩阵按行存储时，32 个线程访问连续元素通常落在不同 bank；若访问跨距恰好是 32 的列，就可能全部撞到同一个 bank。常见解决思路是 padding 或 **swizzling**，例如用行列索引的 XOR 重排地址。

### 1.5 HBM memory coalescing

一个 warp 的访存会合并成缓存行事务。32 个线程各读相邻的 4 字节元素，刚好形成 128 字节连续事务，是理想情况；若地址散乱，则会产生多个事务，浪费带宽。

区分两个概念：

- coalescing 管 HBM/global memory 的连续访问；
- bank conflict 管 shared memory 内部的并行访问。

### 1.6 Wave quantization

Block 以“波次”分配到 SM。B200 有 148 个 SM，若只有 160 个 block，第一波占满 148 个 SM，第二波仅 12 个 block，大部分 SM 空闲。小问题规模或不合适的 tile 数会造成尾波利用率很差。

---

## 2. Benchmark：到底有多快

Benchmark 测端到端耗时，回答“哪个实现更快”和“规模变化时怎样增长”。可靠 GPU benchmark 必须注意：

1. **Warmup**：首次调用可能含 JIT、内核选择、缓存建立等成本；
2. **异步执行**：CPU 发出 kernel 后立即返回，计时前后要用 CUDA event 或 `torch.cuda.synchronize()`；
3. **多次试验**：报告均值、中位数和波动，不能只跑一次；
4. **固定问题定义**：设备、dtype、shape、布局、warmup 次数要一致；
5. **区分 steady-state 与首次延迟**：训练重视稳态，在线服务也可能在意冷启动。

课程中的 matmul 随维度增长，小矩阵阶段时间近似不变，因为 launch overhead、固定开销占主导；矩阵足够大后，理论计算量按 $O(n^3)$ 增长，才显现计算缩放。

---

## 3. Profiler：时间花在哪里

Profiler 不只给总时间，还展示实际启动了哪些 CUDA kernel、各自耗时及调用次数。课程对比 `add(2048)`、`matmul(2048)`、`matmul(128)`，说明框架会根据 shape 选择不同 kernel。

例如名字：

```text
cutlass3x_sm100_simt_sgemm_..._64x64x16_...
```

可以读出：CUTLASS 库、Blackwell `sm100`、数据类型、tile 形状等。Profiler 的思考顺序：

- 是一个 kernel 慢，还是 kernel 太多？
- 时间在 compute 还是 memory copy？
- 是否出现意外 dtype/layout 转换？
- 是否因 shape 改变选到了不同实现？
- CPU launch 是否成为小 kernel 的瓶颈？

---

## 4. GeLU 案例：为什么 fusion 有效

近似 GeLU：

$$
\operatorname{GeLU}(x) \approx \frac{x}{2}\left(1+\tanh\left[\sqrt{\frac{2}{\pi}}\left(x+0.044715x^3\right)\right]\right).
$$

朴素 PyTorch 表达式包含乘、加、幂、tanh 等多个算子。如果每个算子单独启动 kernel，中间结果会反复写回 HBM、再读出。

三种实现：

- naive PyTorch：多个 kernel，多次 HBM 读写；
- `torch.nn.functional.gelu`：库内融合实现；
- `torch.compile(naive_gelu)`：编译器把表达式融合成 Triton kernel。

融合后的理想路径是：从 HBM 读一次 $x$，在寄存器中完成全部算术，再写一次 $y$。这说明 **kernel fusion 的主要收益通常是消除中间张量和 launch，而不是减少数学运算。**

---

## 5. Triton 基础

一个典型 Triton kernel 分四步：

```python
pid = tl.program_id(0)
offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
mask = offsets < n
x = tl.load(x_ptr + offsets, mask=mask)
y = f(x)
tl.store(y_ptr + offsets, y, mask=mask)
```

- `program_id` 标识当前 Triton program/block；
- `tl.arange` 生成一个 block 内的向量索引；
- mask 防止尾部越界；
- 编译器负责把向量程序映射到 threads/warps；
- Triton 最终生成 PTX，再由驱动生成机器码。

PTX 中常见：`ld.global`/`st.global` 是 global memory 读写，`%ctaid.x` 类似 block id，`%tid.x` 类似 thread id，`%f`/`%r` 是浮点/整数寄存器。生成代码显示一个物理线程可能处理多个元素，这就是 thread coarsening。

---

## 6. 四类 kernel 模式

### 6.1 GeLU：elementwise

每个输入只影响对应输出，不需线程间通信。重点是连续访问、mask、融合和合适 block size。

### 6.2 Softmax：行内 reduction

对每一行：

$$
\operatorname{softmax}(x_i)=\frac{e^{x_i}}{\sum_j e^{x_j}}.
$$

数值稳定形式应先减最大值：

$$
m=\max_j x_j,\qquad y_i=\frac{e^{x_i-m}}{\sum_j e^{x_j-m}}.
$$

若整行能放进一个 block：一个 program 负责一行，先 load 到片上存储，做 max、exp、sum、除法，再一次性 store。列数不是 2 的幂时，block size 常取下一个 2 的幂并用 mask；被 mask 的位置在 max 中要当 $-\infty$，在 sum 中要当 0。

朴素分算子 softmax 会读写多个中间张量；融合 softmax 只需一读一写，这也是 FlashAttention 的思想基础。

### 6.3 Row sum：一行放不进一个 block

若一行 4096 个元素而 block 只处理 1024 个位置：

1. 把行切成多个 tile；
2. 每个 lane 在循环中跨 tile 累加；
3. 对 block 内的 accumulator 做最终 reduction；
4. 写出一个标量。

这是最小型的 tiling 例子，展示了“循环装载 tile + 局部累计 + 最后归约”。

### 6.4 Matmul + ReLU：二维 tiling

对 $C=A B$，朴素算法对每个 $C_{mn}$ 都从 HBM 读取整行/整列，读次数为 $O(MKN)$，算术强度近似 $O(1)$。

理想情况把整个 $A,B$ 放入 shared memory，但容量不够，因此切 tile：

1. 每个 block 负责 $C$ 的一个 $B_M\times B_N$ 输出 tile；
2. 沿 $K$ 维迭代，每次加载 $A$ 的 $B_M\times B_K$ tile 与 $B$ 的 $B_K\times B_N$ tile；
3. 片上执行 `tl.dot` 并累加；
4. 最后在 accumulator 上做 ReLU；
5. 只把最终输出写回 HBM。

算术强度提高到与 tile size 同阶，因为同一块 $A/B$ 数据被多个乘加复用。实现地址时必须理解二维张量被线性化：

$$
\text{address}(i,j)=\text{base}+i\cdot \text{stride}_0+j\cdot \text{stride}_1.
$$

边界 tile 对 $M,N,K$ 分别使用 mask，越界的乘法输入填 0。ReLU 与 matmul 融合后避免额外 HBM 往返。

---

## 7. 一套实用优化流程

1. 先写正确、可对照的 PyTorch reference；
2. 用随机 shape、边界 shape、非整除 shape 做 correctness check；
3. warmup 后 benchmark，多次测量；
4. profiler 判断 kernel 数、计算/访存比例和意外开销；
5. 优先减少 HBM traffic：fusion、tiling、recomputation；
6. 检查 coalescing、bank conflict、warp divergence；
7. 调 block/tile/warp 数，观察寄存器压力与 occupancy；
8. 覆盖多种 shape，防止只优化一个点；
9. 每次修改后重新验证正确性与性能。

## 8. 易混点

- 高 FLOPs 利用率不等于低延迟；小 kernel 可能由 launch latency 主导。
- 高 occupancy 不等于高性能；寄存器复用和较低 occupancy 可能更好。
- Shared memory 不是自动更快；有 bank conflict 或搬运开销时也会变慢。
- Fusion 不总是免费；大融合可能增加寄存器压力和编译复杂度。
- `torch.cuda.synchronize()` 是测时必须，不应随意插进真实训练热路径。
- Triton block 是程序抽象，不应机械等同于固定数量 CUDA threads。

## 9. 自测

1. 为什么朴素 GeLU 数学运算不多，却可能比融合版本慢很多？
2. coalescing 与 bank conflict 分别发生在哪一级内存？
3. 为什么 softmax 要减去行最大值？mask 位置应分别怎样参与 max 和 sum？
4. 矩阵乘法 tiling 为什么提高算术强度？
5. 为什么 occupancy 只有 50% 的 kernel 也可能优于 100%？
6. GPU benchmark 忘记同步会测到什么？

## 10. 本讲最重要的五句话

1. 先测量，后优化。
2. GPU 性能经常受数据移动而非算术限制。
3. Fusion 减少中间张量，tiling 增加数据复用。
4. Triton 以 block 为中心表达 elementwise、reduction 和 matmul。
5. 正确性由编程模型保证，性能由硬件细节决定。

## 资料

- [课程官方 Lecture 6 源码](https://github.com/stanford-cs336/lectures/blob/main/lecture_06.py)
- 本地配套源码：`lecture_06.py`



---

## 11. 逐源码详细补充

> 本节严格按 `lecture_06.py` 的执行顺序，补齐代码、公式和数量关系。

### 11.1 硬件与 occupancy

| 项目 | A100 | H100 | B200 | kernel 意义 |
|---|---:|---:|---:|---|
| SM | 108 | 132 | 148 | grid 要有足够 blocks |
| Register/SM | 256 KB | 256 KB | 256 KB | 限制 resident warps |
| L1+shared/SM | 192 KB | 256 KB | 256 KB | 限制单 CTA tile |
| L2 | 40 MB | 50 MB | 96–126 MB | 支持跨 SM 复用 |
| HBM | 80 GB | 80 GB | 192 GB | 决定容量 |
| Register BW | ~116 | ~401 | ~447 TB/s | accumulator 尽量留在 register |
| L1/shared BW | ~19 | ~33 | ~19 TB/s | tile 复用 |
| L2 BW | ~5–8 | ~12 | ~9 TB/s | 中间层 |
| HBM BW | 2 | 3.35 | 8 TB/s | memory-bound 上限 |

B200 的 TMEM 是 tensor-core 专用存储；H100/B200 的 thread-block cluster 可用 distributed shared memory，但本讲示例未使用。

源码实际取 128 threads/block、160 registers/thread、65536 registers/SM、最多 64 warps：

$$R_{block}=128\times160=20480,$$
$$blocks/SM=\lfloor65536/20480\rfloor=3,$$
$$warps/SM=3\times128/32=12,$$
$$occupancy=12/64=18.75\%.$$

讲义文字写过 64 threads，但计算代码用 128。低 occupancy 不必然慢；若额外 registers 带来充分复用，收益可能更大。Occupancy 的价值是让其他 warp 在某 warp 等待 HBM 时继续执行。

Shared-memory bank 粗略为 $bank(address)=(address/4)\bmod32$。32 threads 访问 $32\times32$ row-major tile 的同一列会形成 32-way conflict；padding 为 33 列或 swizzle 可打破周期。同地址 broadcast 不等于 conflict。

HBM coalescing 关注 global addresses：32 threads 连续读 FP32 为 128 bytes，可合并成少量 transaction；大 stride 会触及多条 cache line。

148 个 SM 发射 160 blocks 要 148+12 两波，第二波大部分 SM 空闲，即 wave quantization。实际还要权衡 tile、register、shared memory 与每 block 工作量。

### 11.2 `benchmark()` 逐行解释

`run_operation1/2` 在计时前创建输入并返回闭包，所以计时不含 allocation 和随机数。Warmup 排除 CUDA context、library kernel selection、Triton/`torch.compile` 编译和冷 cache；warmup 后同步避免混入正式试验。

CUDA events 位于 GPU stream：

```python
start.record(); run(); end.record()
torch.cuda.synchronize()
elapsed_ms = start.elapsed_time(end)
```

源码跑 3 trials 取均值；严谨实验还应报 median/P95、控制 clocks 和其他负载。

$d\times d$ matmul FLOPs 为 $2d^3$。扫描 256–8192 时，小矩阵由 launch/固定开销主导；大矩阵才接近立方增长。不同 shape 还会选择不同 CUTLASS kernel。

### 11.3 `profile()`

流程：warmup→同步→CUDA profiler→运行→同步→按 `cuda_time_total` 聚合同名 kernel，显示前 10 项，并追加到 `var/profiles.txt`。

- add(2048)：elementwise、低 arithmetic intensity；
- matmul(2048)：大 GEMM；
- matmul(128)：小 GEMM，可能选择不同实现。

`cutlass3x_sm100_simt_sgemm_f32_..._64x64x16` 可读出 CUTLASS、SM100、dtype 和 tile。完整 CPU launch/streams overlap 要用 timeline/Nsight。

### 11.4 GeLU 三种实现

$$GELU(x)\approx\frac{x}{2}[1+\tanh(0.79788456(x+0.044715x^3))].$$

Naive PyTorch 的各算子可能分别启动 kernel，中间量反复进出 HBM；builtin 已融合；`torch.compile` 生成 fused Triton kernel。`check_equal_1d` 用 2048 随机值和 `allclose(atol=1e-6)` 检查语义。真实开发还应覆盖极值、NaN/Inf、低精度、非连续 layout 和尾部长度。

Fusion 理想上每元素一次 load/store，但也可能扩大 register live range、造成 spill 或降低 occupancy。

### 11.5 Triton GeLU

Wrapper 检查 CUDA/contiguous，分配输出，令 `BLOCK_SIZE=1024`，grid 为 `cdiv(n,BLOCK_SIZE)`。最后一个 program 用：

```python
pid = tl.program_id(0)
offsets = pid*BLOCK_SIZE + tl.arange(0,BLOCK_SIZE)
mask = offsets < n
```

`BLOCK_SIZE` 是 compile-time meta-parameter；`tl.arange` 是向量工作集合，不等于 1024 个物理 threads。

源码以 $\tanh(a)=(e^{2a}-1)/(e^{2a}+1)$ 实现 tanh。PTX 中 `ld.global/st.global` 是 HBM，`%ctaid.x/%tid.x` 是 block/thread id，`%f/%r` 是寄存器。生成代码显示一个物理 thread 处理 8 元素，即 thread coarsening。`output_ptx` 从 `kernel.asm["ptx"]` 导出 PTX。

### 11.6 Softmax 精确 HBM 流量

对 $M\times N$ naive softmax：

1. max：$MN$ reads、$M$ writes；
2. subtract：$MN+M$ reads、$MN$ writes；
3. exp：$MN$ reads、$MN$ writes；
4. sum：$MN$ reads、$M$ writes；
5. divide：$MN+M$ reads、$MN$ writes。

严格合计 $5MN+2M$ reads、$3MN+2M$ writes。源码注释少计一个 $M$ broadcast read；$N$ 大时可忽略。Fused 理想为 $MN$ reads + $MN$ writes，总流量约降至四分之一。

Triton 一行一个 program，`block_size=next_power_of_2(N)`。多余 lanes load `other=-inf`：不影响 max，exp 后为 0，也不影响 sum；store 仍 mask。地址使用 `row_idx*x_row_stride`，而非写死 $N$。

数值稳定形式：
$$m=\max_jx_j,\qquad y_i=e^{x_i-m}/\sum_je^{x_j-m}.$$

### 11.7 Row sum multi-tile reduction

若 $N=12,BLOCK=4$：

```text
lane0: x0+x4+x8
lane1: x1+x5+x9
lane2: x2+x6+x10
lane3: x3+x7+x11
```

`acc` 为 FP32。循环逐 tile load，越界用 0，再 `tl.sum(acc)`。模式是“每 lane 跨 tile 累加→block 内归约”。更大问题可能需要多级 kernel 或 atomic。

### 11.8 Matmul + ReLU 地址推导

$$A\in R^{M\times K},\quad B\in R^{K\times N},\quad C\in R^{M\times N}.$$

教学 tile 为 $(64,64,32)$，grid 为 $(\lceil M/64\rceil,\lceil N/64\rceil)$。地址：
$$ptr(i,j)=base+i\,stride_0+j\,stride_1.$$

```python
a_ptrs = base + m[:,None]*stride_am + k[None,:]*stride_ak  # [BM,BK]
b_ptrs = base + k[:,None]*stride_bk + n[None,:]*stride_bn  # [BK,BN]
```

Accumulator 是 FP32 $[BM,BN]$。每轮 load $[BM,BK]$ 与 $[BK,BN]$，越界补 0，`tl.dot` 后沿 K 前进 BLOCK_K。

每轮做 $2B_MB_NB_K$ FLOPs，读取 $B_MB_K+B_KB_N$ 元素。若 $B_M=B_N=T$，arithmetic intensity 与 $T$ 成正比。K-loop 后直接 `maximum(acc,0)`，融合 ReLU，省一次 $MN$ 中间写和读。

生产 GEMM 还要 autotune `num_warps`、pipeline stages、tensor-core layout、swizzle、program ordering、L2 locality 和 split-K。

### 11.9 源码覆盖表

| 函数 | 内容 |
|---|---|
| `main/review_of_gpus` | 路线、硬件、编程模型 |
| `benchmark/benchmarking` | CUDA-event 测时 |
| `profile/profiling` | kernel 归因 |
| `naive_vs_builtin_vs_compiled_gelu` | fusion |
| `triton_gelu*` | elementwise、PTX |
| `naive_softmax/triton_softmax*` | HBM accounting、row reduction |
| `triton_row_sum*` | 跨 tiles reduction |
| `triton_matmul_relu*` | 2D tiling、strides、fusion |
| `run_operation1/2/check_equal_*/mean/output_ptx` | 测量与验证辅助函数 |


### 11.10 Function-by-function 精确索引

- `main`：串联 GPU review、benchmark/profile、GeLU、Triton 四例。
- `review_of_gpus`：硬件表、grid/block/thread、warp、occupancy、bank、coalescing、wave。
- `benchmarking_and_profiling`：执行 benchmark→profile 的优化闭环。
- `benchmarking`、`benchmark`：shape sweep、warmup、CUDA events、均值。
- `profiling`、`profile`：add/大 matmul/小 matmul 和 profiler 日志。
- `naive_vs_builtin_vs_compiled_gelu`：三实现 correctness、benchmark、profile。
- `triton_introduction`：CUDA thread-centric 与 Triton block-centric 对比。
- `triton_gelu_example`、`triton_gelu`、`triton_gelu_kernel`：launch、mask、公式、PTX。
- `triton_softmax_example`、`naive_softmax`、`triton_softmax`、`triton_softmax_kernel`：流量计数和行归约。
- `triton_row_sum_example`、`builtin_row_sum`、`triton_row_sum`、`row_sum_kernel`：跨 tile 累加。
- `triton_matmul_relu_example`、`naive_matmul_relu`、`triton_matmul_relu`、`matmul_relu_kernel`：二维 tile 与融合。
- `run_operation1`、`run_operation2`：创建固定输入闭包。
- `naive_gelu`、`builtin_gelu`、`pytorch_softmax`：reference/builtin 实现。
- `check_equal_1d`、`check_equal_2d`、`check_equal_2d_2d`：一输入/两输入 correctness harness。
- `mean`：trial 算术均值。
- `output_ptx`：写出 compiled kernel PTX。
