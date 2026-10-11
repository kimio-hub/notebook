---
title: "CS336 Inference 完整笔记"
description: "从一次请求进入服务开始：推理为什么慢、显存花在哪里、KV cache 与 vLLM 类系统如何提高吞吐。"
tags:
  - "cs336"
  - "inference"
  - "systems"
category: "courses"
series: "CS336"
date: "2026-07-11"
updated: "2026-07-11"
---

> 以 Stanford CS336 Spring 2026 Lecture 10 为主，并补齐标准 decoding/sampling 基础。目标：从一次请求进入服务开始，理解它如何执行、为什么慢、显存花在哪里，以及 vLLM 类系统怎样提高吞吐。

## 0. Inference 为什么是独立的系统问题

训练通常是一次性成本，inference 会为每个用户、每次评测和每次 RL rollout 重复发生。聊天机器人输出供人阅读，速度要求有限；agent 的内部轨迹可能生成海量 token，推理成本甚至超过训练。

Inference 还用于：

- 在线聊天、代码补全、agent；
- 离线数据生成与标注；
- benchmark evaluation；
- RLHF/RL 的 rollout；
- test-time compute 和多样本搜索。

训练能同时看到整段 token 并沿 sequence 并行；自回归 inference 必须先生成 $x_t$，才能知道 $x_{t+1}$ 的输入，因此 decode 天生串行。

---

## 1. 衡量“快”的四组指标

### 1.1 Time to First Token（TTFT）

请求到达至第一个输出 token 的时间，主要由：

$$
\text{TTFT}=\text{queueing}+\text{prefill}+\text{first decode step}.
$$

交互应用很敏感。长 prompt、大 prefill batch、排队都会增大 TTFT。

### 1.2 Inter-Token Latency（ITL）/ Time per Output Token（TPOT）

首 token 之后相邻输出 token 的平均时间，单位 ms/token。它决定文字“流出来”的速度。

### 1.3 End-to-End Latency

完成整个请求的时间：

$$
T_{e2e}\approx \text{TTFT}+N_{out}\times\text{TPOT}.
$$

实际还有调度抖动、EOS 提前停止和网络传输。

### 1.4 Throughput

- output tokens/s：单位时间生成多少输出 token；
- total tokens/s：prompt + output 的处理量；
- requests/s：受请求长度分布影响大；
- goodput：在满足 TTFT/TPOT SLO 的前提下完成的有效吞吐。

只报 tokens/s 容易误导：把 batch 无限增大可提升吞吐，却可能让单用户延迟不可接受。

---

## 2. 一次请求的完整生命周期

1. 文本经 tokenizer 变成 token ids；
2. scheduler 等待并把请求放入 prefill batch；
3. **Prefill**：并行处理全部 prompt tokens，建立每层 KV cache，并得到下一 token logits；
4. 根据 decoding policy 从 logits 选第一个 token；
5. 请求加入 decode batch；
6. 每一步只输入最新 token，读取历史 KV cache，生成一个新 token并把新 K/V 追加到 cache；
7. 遇到 EOS、stop string 或 max tokens 后释放 KV blocks；
8. detokenize 并流式返回文本。

Prefill 与 decode 的硬件特征不同，因此现代 serving 系统可能把两者分开调度，甚至放在不同 GPU 池上。

---

## 3. Transformer 记号与参数量

课程使用：

- $B$：并发序列数；
- $S$：已有上下文/缓存长度；
- $T$：本次并行处理的新 token 数；
- $D$：model/hidden dimension；
- $F$：MLP intermediate dimension，常约 $4D$；
- $N$：query heads 数；
- $K$：KV heads 数；
- $H$：head dimension，$D=NH$；
- $L$：layers；
- $V$：vocabulary size。

以含 gate/up/down 的 MLP 和 GQA attention 为例，课程给出的粗略参数量：

$$
P\approx 2VD+3DFL+(2DNH+2DKH)L.
$$

不同实现是否绑 input/output embeddings、是否有 bias、MoE 等会改变公式。BF16 参数显存约 $2P$ bytes；推理不存梯度和 optimizer state，但要存 KV cache 和 runtime workspace。

---

## 4. Roofline 与 Arithmetic Intensity

算术强度定义：

$$
I=\frac{\text{FLOPs}}{\text{bytes moved from HBM}}.
$$

硬件的 machine balance：

$$
I_{hw}=\frac{\text{peak FLOPs/s}}{\text{memory bandwidth}}.
$$

若 $I>I_{hw}$，倾向 compute-bound；若 $I<I_{hw}$，倾向 memory-bound。课程以 H100 BF16 tensor-core 峰值约 989 TFLOP/s、HBM 带宽约 3.35 TB/s，得到阈值约 295 FLOPs/byte。

### 4.1 一个 MLP matmul

$$
X\in\mathbb R^{B\times D},\quad W\in\mathbb R^{D\times F},\quad Y=XW.
$$

BF16 下：

- FLOPs：$2BDF$；
- bytes：$2BD+2DF+2BF$。

当 $B\ll D,F$，权重读取 $2DF$ 主导：

$$
I\approx B.
$$

矩阵-向量乘 $B=1$ 每读一个权重只做约一次乘加，极度 memory-bound；batch 增大后，同一权重被多个序列复用，算术强度提高。

---

## 5. 为什么需要 KV Cache

没有 cache 时，为生成第 $t$ 个 token，每一步都重新对整个前缀做 Transformer forward。单次 attention 对长度 $t$ 是 $O(t^2)$，生成 $T$ 个 token 的累计复杂度约 $O(T^3)$。

每层历史 token 的 key/value 不会改变，因此把它们存下来。下一步只计算新 token 的 Q/K/V，并让新 Q attend 已缓存 K/V。累计 attention 计算降到约 $O(T^2)$，但付出 KV cache 显存和带宽。

每序列 BF16 KV cache：

$$
M_{KV}=S\times L\times K\times H\times 2_{K,V}\times 2_{bytes}.
$$

整个 batch：

$$
M_{total}\approx 2P+B M_{KV}+M_{workspace}.
$$

上下文长度、layers、KV heads、head dimension 和 batch 都线性增加 KV 显存。

---

## 6. Prefill 与 Decode 的根本差异

### 6.1 Prefill

一次输入整个 prompt，$T=S$。所有 prompt positions 的线性层可并行，形成大矩阵乘，通常 compute-bound。

课程对 gated MLP 的 accounting：

$$
\text{FLOPs}_{MLP}=6BTDF,
$$

当权重读取主导时：

$$
I_{MLP}\approx BT.
$$

长 prompt 或较大 prefill batch 很容易超过硬件强度阈值。

使用 FlashAttention 的 attention 部分：

$$
\text{FLOPs}_{attn}=4BSTD,
$$

读写量约 $4BSD+4BTD$ bytes，因而：

$$
I_{attn}=\frac{ST}{S+T}.
$$

Prefill 时 $T=S$：

$$
I_{attn}=\frac S2.
$$

长序列 attention 也可以 compute-bound，但计算量随 $S^2$ 增长。

### 6.2 Decode

每一步只处理一个新 token，$T=1$：

- MLP intensity 约 $B$；
- attention intensity 为 $S/(S+1)<1$。

MLP 可以靠增加并发 batch 复用模型权重；attention 的 KV cache 每个序列不同，batch 增大也不能复用其他请求的 cache，因此仍极度 memory-bound。

结论：

| 阶段 | 并行 token | 典型瓶颈 | 主要指标 |
|---|---:|---|---|
| Prefill | $B\times S$ | compute，长上下文 attention | TTFT |
| Decode | 每序列每步 1 token | 参数 + KV 的 HBM 带宽 | TPOT、tokens/s |

---

## 7. Latency–Throughput–Memory 三角

在 memory-bound 的简化模型中，每步需要读模型参数和 batch 的 KV：

$$
t_{step}\gtrsim \frac{M_{params}+B M_{KV/seq}}{BW_{HBM}}.
$$

每步生成 $B$ 个 token，所以：

$$
\text{throughput}\lesssim
\frac{B\,BW_{HBM}}{M_{params}+B M_{KV/seq}}.
$$

增大 $B$：

- 权重读取被更多 token 摊薄，throughput 上升；
- KV cache 线性增长，step latency 上升；
- 最终显存耗尽；
- 当 KV 流量主导后，throughput 增益趋于饱和。

因此：小 batch 适合低延迟，大 batch 适合高吞吐。TTFT 通常用较小 prefill batch；decode 可用更大 continuous batch。

### 7.1 Replication 与 Sharding

- **复制 M 个完整模型实例**：单请求 latency 基本不变，总吞吐约乘 M，前提是每实例能放下；
- **模型分片（TP/PP）**：让大模型放下或降低单卡权重读取，但每层增加通信；
- **KV/Context 分片**：解决超长上下文 KV 显存，但 attention 需要跨卡协作。

Inference 的 parallelism 应按 latency SLO、模型大小和 topology 选择，不应直接复制训练配置。

---

## 8. 减少 KV Cache：架构级捷径

Inference memory-bound 时，减少每 token 缓存维度会同时：增加可容纳 batch、降低 HBM 流量、改善吞吐/延迟。

### 8.1 MHA、MQA 与 GQA

- MHA：$K=N$，每个 query head 有独立 KV head；
- MQA：$K=1$，所有 query heads 共用一组 K/V；
- GQA：$1<K<N$，每 $N/K$ 个 query heads 共用一个 KV head。

相对 MHA，GQA 的 KV cache 缩小约：

$$
\frac{N}{K}.
$$

MQA 最省但更可能损失质量；GQA 是常见折中。节省出的显存可以用于更长 context 或更大 batch。

### 8.2 Multi-head Latent Attention（MLA）

普通 attention 缓存 $K=W_Kh$、$V=W_Vh$。MLA 缓存低维 latent：

$$
c=W_ch,qquad K=W_Kc,qquad V=W_Vc.
$$

需要时从 $c$ 上投影恢复 K/V。DeepSeek-V2 示例把原本 $NH=16384$ 维压到 latent 512 维；为兼容 RoPE 另存约 64 维位置部分，总约 576 维。

它用额外投影计算换巨大 cache/带宽节省。在 memory-bound decode 中，这通常划算。

### 8.3 Cross-Layer Attention（CLA）

GQA 在 heads 间共享 K/V；CLA 在 layers 间共享 K/V。若若干层复用一组 cache，显存按共享层数下降。代价是各层表达能力约束，需要训练时就采用。

### 8.4 Local / Sliding-window Attention

每层只保留最近 $W$ 个 token：

$$
M_{KV}=O(W),quad \text{不再随总序列长度无限增长}.
$$

多层传播使有效感受野扩大，但纯 local 可能丢失远程信息，常与少量 global/full-attention layers 交错。

### 8.5 Sparse/Compressed Attention

对超长上下文可压缩旧 token、选择 top-k 历史位置或组合多级压缩。它们降低 cache/attention 复杂度，但索引器质量、训练一致性和不规则 kernel 都是新成本。

### 8.6 Linear Attention / State-space Models

将全部历史 K/V 压成固定大小状态，使 decode state 与序列长度无关。代价是不同的建模假设和可能的记忆能力损失。这类优化必须在训练架构阶段决定。

---

## 9. Quantization

量化把浮点值映射到低精度整数或浮点格式。仿射量化：

$$
q=\operatorname{round}(x/s)+z,qquad
\hat x=(q-z)s,
$$

其中 $s$ 是 scale，$z$ 是 zero point。

常见格式：

| 格式 | bytes/value | 特点 |
|---|---:|---|
| FP32 | 4 | 推理通常没必要全用 |
| BF16/FP16 | 2 | 常规基线 |
| FP8 / INT8 | 1 | 带宽和容量约减半 |
| INT4 | 0.5 | 更省，但误差和 kernel 要求更高 |

真正速度取决于硬件是否支持该格式、dequant 是否融合、group size、scale metadata 和 kernel；文件小 4× 不保证端到端快 4×。

### 9.1 Weight-only 与 W/A/KV 量化

- Weight-only：主要减少每步参数读取，decode 常获益大；
- Weight + Activation：可用低精度 tensor cores，但 activation outlier 更难处理；
- KV-cache quantization：对长上下文/大 batch 很重要，但误差会跨所有后续 token 持续影响 attention。

### 9.2 QAT 与 PTQ

**Quantization-Aware Training（QAT）**：训练 forward 中模拟量化/反量化，让权重适应误差；质量好但需昂贵训练。

**Post-Training Quantization（PTQ）**：训练后用 calibration data 确定 scale/zero point；便宜但更依赖校准分布。

### 9.3 GPTQ 与 AWQ

- GPTQ：利用二阶/Hessian 近似，在逐块量化时调整其他权重以补偿误差；
- AWQ：观察少数 activation channels 很大，与它们相连的 weights 更重要，保留/缩放这些敏感权重，其他权重低比特化。

量化评估必须覆盖 perplexity、下游任务、长上下文、不同语言、吞吐、TTFT/TPOT 和实际显存。

---

## 10. Pruning 与 Distillation

Pruning 直接删除层、attention heads、hidden dimensions 或 experts，使模型结构变小。课程给出的流程：

1. 用小 calibration set 估计 layer/head/dimension 重要性；
2. 删除不重要结构；
3. 用原模型作为 teacher，对 pruned student 做 distillation 修复。

Distillation 可最小化 teacher/student logits 的 KL：

$$
\mathcal L_{KD}=T^2\operatorname{KL}left(
\operatorname{softmax}(z_t/T)\,\|\,
\operatorname{softmax}(z_s/T)
\right),
$$

再与 ground-truth cross-entropy 混合。结构化 pruning 才容易得到真实 kernel 加速；非结构化稀疏若硬件/库不支持，参数虽为 0 仍不会更快。

课程总结的两条路线：

- From scratch：定义更快架构，直接训练；
- Distillation：定义更快架构，从大模型知识初始化/修复。

---

## 11. Speculative Sampling / Decoding

关键非对称：target model 逐 token 生成很慢，但一次 prefill 式地**验证多个已知 candidate tokens**可以并行且 compute-efficient。

流程：

1. 小 draft model $p$ 连续提出 $k$ 个 tokens；
2. 大 target model $q$ 一次 forward 得到这些位置的条件概率；
3. 逐个接受 candidate；第一次拒绝时从 residual distribution 采样；
4. 若全部接受，再由 target 多产生一个 token。

对 draft token $x$，接受概率：

$$
a(x)=\min\left(1,\frac{q(x)}{p(x)}\right).
$$

若拒绝，从：

$$
r(x)=\frac{\max(q(x)-p(x),0)}{\sum_y\max(q(y)-p(y),0)}
$$

采样。这个修正保证最终序列**精确服从 target model $q$**，不是近似 greedy 替代。

### 11.1 为什么会加速

一次昂贵 target pass 若平均接受多个 token，就把多次 memory-bound decode 合并成一次并行验证。加速取决于：

- draft/target 的速度比；
- acceptance rate；
- speculation length $k$；
- batch 和服务负载；
- 验证 kernel 的效率。

Draft 太小则分布差、拒绝多；太大则 draft 自己昂贵。常用 distillation 让 draft 接近 target。Medusa 用多头并行提出候选树；EAGLE 利用 target 高层 feature 改善 draft。

---

## 12. Continuous Batching

静态 batching 等一批请求全部结束再换下一批。生成长度不同导致短请求完成后留下空位，GPU 利用率低且新请求等待。

**Iteration-level scheduling / continuous batching**：每个 decode iteration 后重新组 batch：

1. 移除已完成请求；
2. 立即加入等待队列中的新请求；
3. 为当前 active sequences 各生成一个 token。

这显著提高吞吐和降低 queueing，但调度器必须处理 prefill 与 decode 的竞争。大 prefill 会阻塞 decode，造成 ITL spike，因此可使用 chunked prefill：把长 prompt 切块，穿插 decode iterations。

### 12.1 Selective Batching / Ragged Tensors

不同请求长度不同，不能简单 padding 到最大长度：

- attention：每个序列有独立 causal 区间/KV，按 sequence metadata 单独处理；
- MLP、norm 等 token-wise 操作：把所有 active tokens 拼成 $[\sum_i T_i,D]$ 的大矩阵统一计算。

这避免 padding，同时保留大 matmul 效率。

---

## 13. PagedAttention

传统做法按“prompt + 最大输出长度”给每请求预留连续 KV 区域，产生：

- internal fragmentation：实际提前 EOS，预留尾部没用；
- external fragmentation：空洞分散，虽总空间够却找不到大连续区域。

PagedAttention 借鉴操作系统虚拟内存：

1. KV cache 切成固定大小 physical blocks；
2. 每序列维护 logical block table；
3. logical 连续 token 可映射到非连续 physical blocks；
4. token 增长时按需分配新块；
5. 完成后块立即回收到 free pool。

Attention kernel 根据 block table gather K/V。代价是地址间接与更复杂 kernel，换来接近按需分配的显存利用率。

### 13.1 Prefix Sharing 与 Copy-on-Write

系统 prompt、few-shot prefix、同一 prompt 的多个采样可以共享相同 KV blocks。分支产生不同 token 时，只复制被修改的最后 block，即 copy-on-write。

这既节省显存，也跳过重复 prefill。Agent workloads 有大量公共前缀，因此 prefix/radix caching 特别重要。

### 13.2 vLLM 类系统的其他优化

- 融合 block 读取与 attention；
- FlashAttention / FlashDecoding；
- CUDA Graphs 降低小 decode kernel 的 CPU launch overhead；
- optimized sampling kernels；
- tensor/pipeline parallel serving；
- prefix cache 与分布式 KV 管理。

---

## 14. 标准 Decoding 与 Sampling（课程衔接补充）

模型每步输出 logits $z_i$，先用 temperature：

$$
p_i=\operatorname{softmax}(z_i/T).
$$

- $T\to0$：更接近 greedy；
- $T=1$：原分布；
- $T>1$：更平坦、更多样，也更易出错。

### 14.1 Greedy

$$
x_t=\arg\max_i p_i.
$$

确定、便宜，但可能重复或陷入局部最优。

### 14.2 Top-k

仅保留概率最高的 $k$ 个 token，重新归一化采样。固定 $k$ 不适应分布尖锐程度。

### 14.3 Top-p / Nucleus

选择最小集合 $V_p$，使：

$$
\sum_{i\in V_p}p_i\ge p,
$$

再归一化采样。分布尖锐时集合小，分布平坦时集合大。

### 14.4 Repetition/Frequency Penalty

根据已出现 token 次数降低 logits，可减少循环，但改变目标分布。与 speculative sampling 结合时，draft 和 target 必须使用一致的实际 sampling distribution。

### 14.5 Beam Search

保留累计 log-probability 最高的多个序列，适合翻译等目标明确任务；开放式生成中常导致模板化，且 beam 数线性增加 KV/计算。

---

## 15. 怎样优化一个真实 Serving 系统

### Step 1：先定义 workload 与 SLO

记录 prompt/output length 分布、到达率、并发、共享 prefix、TTFT/TPOT 上限、质量要求，而不是用固定 synthetic batch 代表线上。

### Step 2：建立资源模型

- 参数显存；
- 每 token KV bytes；
- 最大 blocks/并发；
- prefill FLOPs；
- decode 每步参数/KV 读取；
- 节点内外通信。

### Step 3：分别 benchmark prefill 与 decode

Prefill 测不同 prompt tokens/batch；decode 测不同 batch、context length、KV dtype。混合 benchmark 还要重放真实到达过程。

### Step 4：优先解决主瓶颈

- 权重带宽：batch、weight quantization、replicas；
- KV 带宽/容量：GQA/MLA、KV quant、PagedAttention、prefix cache；
- TTFT：小/分离 prefill batch、chunked prefill；
- decode latency：较小 batch、模型分片、speculative decoding；
- 吞吐：continuous batching、larger batch、replicas；
- 超长上下文：local/sparse/context parallel。

### Step 5：同时验证质量和尾延迟

必须报告 P50/P95/P99 TTFT、TPOT、throughput、OOM rate、cache hit rate、功耗/成本，以及量化/架构改变后的准确率。

---

## 16. 常见错误

1. 把 prefill 和 decode 混成一个平均 tokens/s；
2. 只测 batch=1 后断言硬件慢；
3. 只追吞吐，不约束 TTFT/TPOT；
4. 只算模型权重，忘记 KV cache 与 workspace；
5. 预分配 max sequence，造成严重碎片；
6. 认为量化 4× 就必然加速 4×；
7. 在不支持稀疏的 kernel 上做非结构化 pruning；
8. speculative decoding 没有 residual correction，却声称 exact；
9. prefix cache 不考虑 tokenizer、position、adapter、sampling context 的 cache key 一致性；
10. 用平均请求长度做容量规划，忽略长尾导致 OOM 和 P99 抖动。

---

## 17. 自测题

1. TTFT、TPOT、throughput 分别由哪一阶段主导？
2. 为什么无 KV cache 的累计生成复杂度会到 $O(T^3)$？
3. 为什么 decode MLP 能靠 batch 提高 intensity，而 attention 仍接近 1 以下？
4. 推导 BF16 GQA KV cache 的 bytes 公式。
5. 为什么更大 batch 提高吞吐却恶化 latency？
6. GQA、MLA、CLA 分别在哪个维度共享/压缩 KV？
7. QAT、PTQ、GPTQ、AWQ 的区别是什么？
8. Speculative sampling 为什么仍精确服从 target distribution？
9. Continuous batching 如何处理请求不同到达和结束的问题？
10. PagedAttention 如何消除 internal/external fragmentation？
11. 为什么共享 prefix 需要 copy-on-write？
12. 什么时候复制模型比把一个模型分片更合适？

## 18. 最应该记住的结论

1. Prefill 通常 compute-bound；decode 通常 memory-bound 且串行。
2. KV cache 把重复计算换成显存和带宽，是 inference 的核心状态。
3. Batch 复用模型权重，形成 latency 与 throughput 的基本权衡。
4. GQA/MLA/CLA、量化和 local/sparse attention 都在减少每 token 数据移动。
5. Speculative decoding 利用“并行验证比串行生成便宜”，且可保持 exact distribution。
6. Continuous batching 和 PagedAttention 解决的是动态、ragged 在线 workload。
7. Serving 优化必须同时看质量、P99 延迟、吞吐和真实成本。

## 资料

- [课程官方 Lecture 10：Inference](https://github.com/stanford-cs336/lectures/blob/main/lecture_10.py)
- [How to Scale Your Model - Inference](https://jax-ml.github.io/scaling-book/inference/)
- [vLLM / PagedAttention](https://arxiv.org/abs/2309.06180)
- [Speculative Decoding](https://arxiv.org/abs/2211.17192)
- [Grouped-Query Attention](https://arxiv.org/abs/2305.13245)
- [DeepSeek-V2 / MLA](https://arxiv.org/abs/2405.04434)
