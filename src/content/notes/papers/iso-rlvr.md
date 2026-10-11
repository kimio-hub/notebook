---
title: "谱不动，方向转：RLVR 到底改了权重的什么？"
description: "ISO 论文阅读：RLVR 的权重更新几乎只旋转奇异向量、不改变奇异值谱，作者据此提出 ISO-Optimizer 与免数据的 ISO-Merger。"
tags:
  - "rlvr"
  - "optimization"
  - "model-merging"
category: "papers"
date: "2026-07-25"
updated: "2026-07-25"
---

论文：[ISO: An RLVR-Native Optimization Stack (arXiv:2607.19331)](https://arxiv.org/abs/2607.19331)


**一句话总结**：RLVR 对权重的有效修改几乎全部体现在左右奇异向量（frame）的旋转上，奇异值谱基本不动，而且功能上可以直接沿用基础模型的谱。作者把这个观察做成了一整套方法：在线的 **ISO-Optimizer**——冻结 $\Sigma_0$，只在 Stiefel 流形上优化 $U,V$；离线的 **ISO-Merger**——把同一 base 训出的多个 RL 专家的 frame 变化在切空间里合并，再用 $\Sigma_0$ 重建，全程不需要数据、rollout 或梯度更新。
## 背景：RLVR 缺一个自己的优化层

RLVR 这两年的进展基本集中在数据、目标函数（GRPO/DAPO 一族）和训练系统上，唯独"reward 信号如何转成权重空间的更新"这一层，是从预训练原封不动搬来的——AdamW、欧氏坐标、全参数更新。但已有不少工作发现 RLVR 的参数更新和预训练、SFT 长得很不一样：稀疏、偏离主成分方向、不同 run 之间高度重叠。这篇论文想回答的问题是：RLVR 的更新到底有什么结构？如果有，优化器是不是应该顺着这个结构来设计？

## 观察：谱几乎不动

对每个权重矩阵 $W\in\mathbb{R}^{d_\mathrm{out}\times d_\mathrm{in}}$ 做 thin SVD：

$$W=U\Sigma V^\top,\qquad U\in\mathrm{St}(d_\mathrm{out},q),\quad V\in\mathrm{St}(d_\mathrm{in},q),\quad q=\min\{d_\mathrm{out},d_\mathrm{in}\},$$

其中 $\mathrm{St}(d,r)=\{X\in\mathbb{R}^{d\times r}:X^\top X=I_r\}$ 是 Stiefel 流形（列正交矩阵的集合）。直观上：

- $V$：在输入空间里选出一组重要方向；
- $\Sigma$：每个方向的放大倍数；
- $U$：把放大后的信号映射到输出空间的哪些方向。

给定基础模型的权重 $W_0=U_0\Sigma_0V_0^\top$，定义它的**固定谱族**——所有与 $W_0$ 奇异值相同的矩阵：

$$\mathcal{F}(W_0)=\{Z:\sigma(Z)=\sigma(W_0)\}=\bigl\{U\Sigma_0V^\top: U\in\mathrm{St}(d_\mathrm{out},q),\ V\in\mathrm{St}(d_\mathrm{in},q)\bigr\}.$$

论文有个很干净的命题：任意矩阵 $W=U\Sigma V^\top$ 到这个族的 Frobenius 距离就是两组奇异值之差的范数，

$$\operatorname{dist}_F\bigl(W,\mathcal{F}(W_0)\bigr)=\|\sigma(W)-\sigma(W_0)\|_2,$$

而且最近点就是"换谱不换向"的 $U\Sigma_0V^\top$。于是"谱动了多少"可以被精确量化。定义相对残差

$$\rho_\Sigma=\frac{\|\sigma(W_1)-\sigma(W_0)\|_2}{\|W_1-W_0\|_F},$$

即谱变化占整个 checkpoint 位移的比例。实测下来：RLVR 训完 3000+ 步之后 $\rho_\Sigma$ 平均只有 **3%** 左右（沿训练轨迹逐 checkpoint 测约 1.3%），而 SFT 大约是 **35%**。也就是说，RLVR 把权重挪了不小的距离，但这段位移几乎完全落在"保谱"的方向上。

有人可能会问：高维空间里随机挑一个方向，本来就几乎不改奇异值——在 $d_\mathrm{out}d_\mathrm{in}$ 维的矩阵空间里，一阶意义上能改谱的坐标只有 $\operatorname{diag}(U_0^\top\Delta W\,V_0)$ 这 $q$ 个。所以作者定义了维度归一化的"改谱能量"

$$\kappa_\mathrm{spec}=\frac{d_\mathrm{out}d_\mathrm{in}}{q}\cdot\frac{\bigl\|\operatorname{diag}(U_0^\top\Delta W\,V_0)\bigr\|_2^2}{\|\Delta W\|_F^2},$$

各向同性的随机位移在这个归一化下期望值为 1。结果 RLVR 的 $\kappa_\mathrm{spec}$ 在 1.0–1.35 之间——**RLVR 并没有"额外回避"改谱方向，它的保谱主要是高维几何送的**。但对比依然成立：SFT 的 $\kappa_\mathrm{spec}$ 高达 89–1364，比维度基准高出两三个数量级。所以"谱不动"不是所有后训练的通性，而是 RLVR 区别于 SFT 的真实特征。

## 从"谱没动"到"谱可以不动"

谱变化小，不代表这一点点变化不重要。论文用两个干预实验把观察升级成了功能性结论（他们称之为 **spectral inheritance**，谱继承）：

**训完之后把谱换回去。** 保留 RL 训出来的 frame，只在谱上插值：

$$\widetilde{W}(\alpha)=U_\mathrm{RL}\bigl[(1-\alpha)\Sigma_0+\alpha\Sigma_\mathrm{RL}\bigr]V_\mathrm{RL}^\top,\qquad\alpha\in[0,1].$$

$\alpha=0$ 时完全用回基础模型的谱，性能几乎不掉。反过来，把 RL 的谱装进基础模型的 frame 里，则毫无提升——**能力全在 frame 里，不在谱里**。

**从第一步起就冻结谱。** 训练全程固定 $\Sigma_0$、只优化 frame，照样能学出很强的推理能力（这一版后来就成了 ISO-Optimizer）；对照组冻结 frame、只训对角谱，几乎学不动——当然这组每个矩阵只有 $q$ 个自由度，容量本身就受限。

那么 frame 里面还能再省吗？论文测了四种受限重构对 checkpoint 更新的解释程度：允许在 base 的输入输出子空间内部任意"重混"、只保留输入子空间、只保留输出子空间，以及只固定谱但允许两侧 frame 都动（$\widehat{W}_\mathrm{iso}=U_i\Sigma_j V_i^\top$）。用未解释比例 $u_h=\|W_i-\widehat{W}_h\|_F/\|W_i-W_j\|_F$ 衡量，中位数分别是 **87% / 45% / 42% / 1.8%**。结论很清楚：谱可以冻结，但**左右两个 frame 都必须保持可动**，一个都省不掉。这个模式在两阶段 VLA RL（目标完全不同的连续两个 RL 阶段）上同样复现。

## ISO-Optimizer：把 AdamW 搬到 Stiefel 流形上

有了上面的结论，在线算法就很自然了。把每个权重矩阵重参数化为

$$W(U,V)=U\Sigma_0V^\top,$$

$\Sigma_0$ 从基础模型继承并全程冻结，可训练变量只有两个正交 frame。给定对重建权重的梯度 $G_W=\nabla_W\mathcal{L}$，链式法则给出 frame 的梯度：

$$G_U=G_W\,V\Sigma_0,\qquad G_V=G_W^\top U\Sigma_0.$$

$\Sigma_0$ 出现在梯度里是有物理含义的：同样大小的 frame 位移，落在大奇异值的模式上对权重的影响更大。

然后任选一个基础优化器 $\mathsf{Opt}$（AdamW、Muon 都行），直接作用在 $U,V$ 上：

$$(\bar{U},s_U^+)=\mathsf{Opt}(U,G_U,s_U),\qquad(\bar{V},s_V^+)=\mathsf{Opt}(V,G_V,s_V).$$

更新后的 $\bar{U},\bar{V}$ 一般不再正交，用 polar retraction 拉回 Stiefel 流形：

$$\operatorname{polar}(X):=P_XQ_X^\top\quad(X=P_XS_XQ_X^\top\ \text{为 thin SVD}),$$

满秩时等价于 $X(X^\top X)^{-1/2}$，即离 $X$ 最近的列正交矩阵。最终

$$U^+=\operatorname{polar}(\bar{U}),\qquad V^+=\operatorname{polar}(\bar{V}),\qquad W^+=U^+\Sigma_0(V^+)^\top.$$

这样每一步迭代都精确落在 $\mathcal{F}(W_0)$ 里。要注意这**不是**"先更新 $W$ 再投影回固定谱族"：动量、二阶矩这些优化器状态直接活在 frame 坐标里，走出来的轨迹和先更新再投影完全不同。这个构造对基础优化器是透明的，$\mathsf{Opt}\mapsto\operatorname{ISO}[\mathsf{Opt}]$，于是有 ISO-AdamW、ISO-Muon。

- **Qwen3-8B-Base** 数学 RLVR：AdamW 训到 270 步达到 0.495 后平台；ISO-AdamW 100 步就到同样水平（**2.7× 省步数**），210 步升到 0.509。
- **Qwen3-4B-Base**：最终 43.46 vs AdamW 41.69，追平最强 AdamW 只需约 1/2.2 的步数；ISO-Muon 同样领先 Muon（0.428 vs 0.422）。
- 值得玩味的是，ISO-AdamW 全程只用一个学习率 $7.5\times10^{-7}$，对面的 AdamW/Muon 是扫了一圈学习率之后取最好的。

开销方面，polar retraction 用 FP64 的 SVD 实现（数值稳定性需要），按 DION 的方式分布到多卡，实测约占端到端 RL 步时的 **7%**——RLVR 的时间大头在 rollout 生成，优化器这点开销可以接受，异步 RL 系统里还能跟 rollout 重叠掉。代价是要存 $U,V$ 两个因子和各自的优化器状态，显存占用比单个稠密矩阵大，论文靠全分片 + offload 降下来。

## ISO-Merger：不训练的专家合并

同一个思路的离线版本更有意思。现在流行的模块化 RLVR 流程是：从同一个 base 分头训出数学、代码、工具调用等专家，再想办法合成一个模型——通常要再做一轮 on-policy distillation，又是数据又是 rollout。传统的免训练合并（Task Arithmetic、TIES 等）合的是欧氏空间里的任务向量 $\Delta W_i=W_i-W_0$。而在 ISO 的视角下，每个专家学到的东西本来就不在 $\Delta W$ 里，而在 frame 的旋转里，那就应该在流形坐标下合并：

1. 对 base 和每个专家做 SVD：$W_0=U_0\Sigma_0V_0^\top$，$W_i=U_i\Sigma_iV_i^\top$。先做符号对齐消掉 SVD 的规范自由度（$(u_k,v_k)$ 和 $(-u_k,-v_k)$ 表示同一个矩阵）：$s_{i,k}=\operatorname{sign}\langle u_{0,k},u_{i,k}\rangle$，同时乘到 $u_{i,k},v_{i,k}$ 上。
2. 计算 frame 位移 $\Delta U_i=U_i-U_0$，$\Delta V_i=V_i-V_0$，并投影到 base 处的 Stiefel 切空间：
$$\xi_{U,i}=\Pi_{U_0}(\Delta U_i)=\Delta U_i-U_0\operatorname{sym}(U_0^\top\Delta U_i),\qquad\operatorname{sym}(A)=\tfrac{1}{2}(A+A^\top).$$
3. 尾部小奇异值对应的模式在不同专家之间噪声大、互相打架，留着反而降分。于是只保留前 $k_\mathrm{keep}=\operatorname{round}(\rho_\mathrm{keep}q)$ 列（默认 $\rho_\mathrm{keep}=0.9$）：$\widetilde{\xi}_{U,i}=\xi_{U,i}D_\mathrm{keep}$。
4. 用一阶效应代理衡量专家之间的干涉。每个专家的 frame 位移对权重的一阶影响是
$$g_i=\widetilde{\xi}_{U,i}\Sigma_0V_0^\top+U_0\Sigma_0\widetilde{\xi}_{V,i}^\top,$$
由此构造 Gram 矩阵 $\Gamma_{ij}=\langle g_i,g_j\rangle_F$。
5. 合并系数 $c$ 的目标是让每个专家在合并结果中的"自保留率"都为 1：$\operatorname{ret}_i(c)=(\Gamma c)_i/\Gamma_{ii}=1$，即解 $\Gamma c=\operatorname{diag}(\Gamma)$。实际用 ridge 稳定化再裁剪：
$$(\Gamma+\lambda_\mathrm{ridge}I_K)\bar{c}=\operatorname{diag}(\Gamma),\qquad c^\star=\operatorname{clip}(\bar{c};0,1.5).$$
专家方向彼此正交时 $c^\star\to\mathbf{1}$ 退化为直接相加；方向重叠时系数自动缩小，抵消干涉。
6. 聚合、重新投影到切空间、polar retraction 拉回流形，最后用 base 的谱重建：
$$U_\star=\operatorname{polar}\Bigl(U_0+\Pi_{U_0}\Bigl(\sum_i c_i^\star\widetilde{\xi}_{U,i}\Bigr)\Bigr),\qquad V_\star\ \text{同理},\qquad W_\star=U_\star\Sigma_0V_\star^\top.$$

一维参数（norm 的 scale、bias）直接任务向量平均。整个过程只碰 checkpoint，没有任何前向传播。

结果：在 Qwen2.5-7B-Instruct 上合并代码（CURE）、工具调用（ToolRL）、长上下文记忆（MemAgent）三个 RL 专家，综合均分 63.80，超过最强免数据baseline（RAM，62.88）；在 DeepSeek-R1-Distill-Qwen-1.5B 上合并代码和数学两个专家，44.38 vs 最强基线 TIES 的 43.52。提升不算爆炸，但胜在各项能力恢复得比较均衡，而且 worst-case 表现（worst@4）比baseline稳定不少。

## 一点感想

这篇论文的 idea 我很喜欢，不只在算法本身，更在它的路线：先观察（谱不动）→ 再校准（用 $\kappa_\mathrm{spec}$ 承认这主要是高维几何的产物，而不是急着宣称"RLVR 偏好保谱"）→ 再做功能性干预（换谱、冻谱、冻 frame）→ 最后才落到算法设计。中间那步校准尤其难得，换一篇卷 novelty 的论文大概率会把"$\rho_\Sigma$ 只有 3%"直接吹成结论。而且同一个坐标系同时催生了在线优化器和离线合并器，说明这个分解确实切中了某种真实结构，而不是为了讲故事硬凑的。

当然也有没讲完的部分。为什么 RLVR 的能力恰好装得进"旋转 frame"这么窄的一族变换里，论文只给了现象和可用性，没有给机制解释——这可能和 RLVR 本质上是对已有能力的 KL 邻近重加权有关，和"SFT 重写谱、RL 只转方向"这个对比连起来想会很有意思。另外实验止步于 8B，固定谱这个约束在更长训练、更大模型、能力需要真正"长出来"（而非重组）的场景下会不会变成瓶颈，是个开放问题。相关工作里提到的同期论文 Pion 从另一条路（李代数上的保谱更新）走到了几乎同一个地方，加上 Muon、POET、Modular Manifolds 这一串工作，权重的矩阵几何正在从分析工具变成后训练的设计变量，这个方向值得继续。
