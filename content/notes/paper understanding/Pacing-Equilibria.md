---
title: Pacing-Equilibria
---

下面这一版可以直接当作你的课堂讲稿来用。核心目标是：即使学生没看论文，也能知道这篇论文**研究什么问题、模型怎么定义、主结果是什么、为什么难、证明怎么构造、还能往哪里做**。

---

# 0. 先用一句话说清楚这篇论文

这篇论文研究的是：

> 在预算受限的二价广告拍卖中，广告主用 pacing multiplier 缩放出价；这种系统的稳定状态叫 pacing equilibrium。论文证明：即使只想找一个**常数精度的近似 pacing equilibrium**，在一般情形下也是 **PPAD-hard** 的。

更具体地说，Chen 和 Li 证明：对任意常数 $\gamma<1/3$，在二价 pacing game 中计算一个 $\gamma$-approximate pacing equilibrium 是 PPAD-hard，而且这个困难性即使在每个 bidder 最多只对 4 个 item 有非零价值时也成立。论文最早在 arXiv 上发布，后来作为 WINE 2025 论文出现。([arXiv](https://arxiv.org/abs/2501.15295 "[2501.15295] Constant Inapproximability of Pacing Equilibria in Second-Price Auctions"))

这句话里有三个关键词：

1. **Second-price auction**：最高出价者赢，但支付第二高价。
    
2. **Pacing equilibrium**：广告主用一个乘数 $\alpha_i\in[0,1]$ 缩放所有出价，以控制预算。
    
3. **Constant inapproximability**：不是精确求解难，而是连某种常数级近似也难。
    

---

# 1. 为什么会有 pacing 这个东西？

先从广告拍卖背景讲起。

在线广告里，一个广告主不是只参加一次拍卖，而是一天内参加成千上万次广告展示机会的拍卖。每个展示机会可以看成一个 item。广告主 (i) 对 item (j) 有价值 (v_{ij})，但广告主还有一个总预算 (B_i)。

如果没有预算，二价拍卖有一个经典性质：真实出价是弱占优策略。但预算一加入，事情就变复杂了。广告主如果每次都按照真实价值出价，可能早早把预算花光，后面更好的展示机会就没钱买了。于是平台或广告主会使用 pacing：

$$
\text{实际出价}=\alpha_i v_{ij},  
$$

其中

$$
\alpha_i\in[0,1]  
$$

叫做 pacing multiplier。

直觉是：

- $\alpha_i=1$：不压价，正常出价；
    
- $\alpha_i<1$：预算紧张，所以把所有出价按比例压低；
    
- $\alpha_i$ 越小，花钱速度越慢。
    

论文研究的就是：当所有广告主都这样做时，系统会不会达到某种稳定状态？这个稳定状态能不能被高效计算出来？

---

# 2. 二价 pacing game 的正式模型

论文定义的 second-price pacing game 可以写成

$$
G=(n,m,(v_{ij}),(B_i)).  
$$

其中：

- (n)：买家，也就是广告主数量；
    
- (m)：商品，也就是广告展示机会数量；
    
- (v_{ij})：买家 (i) 对商品 (j) 的价值；
    
- (B_i)：买家 (i) 的预算；
    
- 每个商品 (j) 单独通过一个二价拍卖出售。
    

每个买家选择一个 pacing multiplier：

$$
\alpha_i\in[0,1].  
$$

于是买家 (i) 对商品 (j) 的实际出价是

$$
\alpha_i v_{ij}.  
$$

对每个商品 (j)，定义最高出价：

$$
h_j(\alpha)=\max_i \alpha_i v_{ij}.  
$$

定义第二高价：

$$
p_j(\alpha)=\text{商品 }j\text{ 上的第二高出价}.  
$$

如果有多个买家并列最高价，那么第二高价也等于最高价。分配变量 $x_{ij}\in[0,1]$ 表示商品 (j) 分给买家 (i) 的比例，也可以理解成在不可分商品下分给买家 (i) 的概率。论文的模型和 equilibrium 定义见其 Section 2。([arXiv](https://arxiv.org/pdf/2501.15295 "Constant Inapproximability of Pacing Equilibria in Second-Price Auctions"))

---

# 3. 什么是 exact pacing equilibrium？

一个 pacing equilibrium 是一组

$$
(\alpha,x)  
$$

满足四类条件。

## 条件 1：只有最高出价者能赢

如果买家 (i) 得到商品 (j) 的正比例，即

$$
x_{ij}>0,  
$$

那么他必须是最高出价者：

$$
\alpha_i v_{ij}=h_j(\alpha).  
$$

也就是说，不能把商品分给不是最高出价的人。

---

## 条件 2：有正出价的商品必须全部卖掉

如果某个商品 (j) 的最高出价是正的：

$$
h_j(\alpha)>0,  
$$

那么这个商品必须完全分配：

$$
\sum_i x_{ij}=1.  
$$

这表示市场不会故意浪费有买家愿意出价的商品。

---

## 条件 3：不能超预算

买家 (i) 的总支付不能超过预算：

$$
\sum_j x_{ij}p_j(\alpha)\le B_i.  
$$

注意二价拍卖里，赢家支付的是第二高价 (p_j$\alpha$)，不是自己的出价。

---

## 条件 4：没有不必要的 pacing

如果买家 (i) 没有花完预算：

$$
\sum_j x_{ij}p_j(\alpha)<B_i,  
$$

那么他就不应该被压价，必须有

$$
\alpha_i=1.  
$$

这条非常重要。它的意思是：

> 只有预算真的紧张的人才应该 pacing。  
> 如果一个人预算没花完，还把出价压低，那就是“不必要的 pacing”。

所以 exact pacing equilibrium 的经济含义是：

> 分配遵守二价拍卖规则，所有人不超预算，而且任何被压价的人都必须预算正好绑定。

---

# 4. 一个小例子帮助理解 pacing equilibrium

假设有两个广告主 (A,B)，两个相同广告展示机会 (g_1,g_2)。

广告主价值：

$$
v_{A1}=v_{A2}=10,\qquad v_{B1}=v_{B2}=4.  
$$

预算：

$$
B_A=5,  
$$

而 (B) 的预算很大。

如果 (A) 不 pacing，即 $\alpha_A=1$，他对每个商品都出价 10，会赢两个商品。每个商品支付第二高价 4，总支付 8，超过预算 5，所以不可行。

现在令

$$
\alpha_A=0.4,\qquad \alpha_B=1.  
$$

那么两人的出价都是

$$
A:0.4\times 10=4,\qquad B:1\times 4=4.  
$$

两人并列最高价，每个商品价格为 4。可以把两个商品的总量中 (5/4) 分给 (A)，例如

$$
x_{A1}=1,\qquad x_{A2}=0.25.  
$$

这样 (A) 的支付是

$$
4\times 1.25=5,  
$$

刚好花完预算。因为 (A) 的预算绑定，所以允许 $\alpha_A<1$。而 (B) 没有预算压力，所以可以保持 $\alpha_B=1$。

这个例子说明 pacing equilibrium 不是简单地“谁价值高谁拿走全部”，而是要同时满足：

- 出价排序；
    
- 二价支付；
    
- 预算；
    
- 没有不必要压价。
    

---

# 5. 什么是 $\gamma$-approximate pacing equilibrium？

论文研究的不是 exact equilibrium，而是近似 equilibrium。

它保留了前面大部分条件：分配仍然只能给最高出价者，商品仍然要完全分配，预算仍然不能超过。真正放松的是“没有不必要 pacing”这一条。论文的近似定义是：

$$
\sum_j x_{ij}p_j(\alpha)<(1-\gamma)B_i  
\quad\Longrightarrow\quad  
\alpha_i=1.  
$$

这句话要仔细理解。

exact equilibrium 要求：

$$
\text{只要没花满预算，就必须 }\alpha_i=1.  
$$

$\gamma$-approximate equilibrium 要求弱一些：

$$
\text{如果花的钱明显少于预算，少到低于 }(1-\gamma)B_i,\text{ 才必须 }\alpha_i=1.  
$$

也就是说，近似 equilibrium 容许一种情况：

$$
(1-\gamma)B_i\le \text{支出}<B_i  
$$

时，买家可以有 $\alpha_i<1$。这表示他虽然没有完全花光预算，但已经花得“差不多”了，所以允许一点不必要 pacing。

这里要特别提醒初学者：

> $\gamma$ 越大，近似条件越弱。  
> $\gamma=0$ 基本回到 exact no-unnecessary-pacing 条件。  
> $\gamma$ 接近 1 时，要求非常松。

论文证明的是：即使允许任意固定常数 $\gamma<1/3$ 的这种松弛，计算仍然 PPAD-hard。([arXiv](https://arxiv.org/pdf/2501.15295 "Constant Inapproximability of Pacing Equilibria in Second-Price Auctions"))

---

# 6. 这篇论文的主定理到底说了什么？

主定理是：

$$
\text{For any constant }\gamma<1/3,\text{ computing a }\gamma\text{-approximate pacing equilibrium is PPAD-hard.}  
$$

而且成立于一个很强的限制下：

> 每个 bidder 最多只对 4 个 goods 有非零价值。

这意味着困难性不是因为一个广告主参与了非常多商品、输入结构非常复杂；即使每个广告主只“关心”很少几个商品，问题仍然难。论文还指出，这排除了 pacing equilibrium 的 PTAS，除非 $\text{PPAD}=\text{FP}$。([arXiv](https://arxiv.org/pdf/2501.15295 "Constant Inapproximability of Pacing Equilibria in Second-Price Auctions"))

这里的 **FP** 可以理解成“多项式时间可计算的函数问题类”。如果一个 PPAD-hard 问题有多项式时间算法，那么很多由不动点、路径端点保证存在的均衡类问题都会有多项式时间算法。算法博弈论里通常不认为这是可能的。

---

# 7. PPAD-hard 在这里是什么意思？

给初学者讲时，不要把 PPAD-hard 和 NP-hard 混在一起。

NP-hard 常见于判定问题，比如“是否存在一个满足条件的解”。但 pacing equilibrium 一类问题通常是：

> 解保证存在，但问题是能不能高效找到一个解。

这类问题属于搜索问题。PPAD 是 TFNP 的一个子类，用来描述“解存在性由某种奇偶性 / 路径端点 / 不动点结构保证”的搜索问题。

经典例子包括：

- Brouwer 不动点；
    
- Sperner 引理；
    
- Nash equilibrium；
    
- 某些市场均衡；
    
- 这里的 pacing equilibrium。
    

所以这里的 PPAD-hard 意思是：

> 如果你能高效计算这篇论文要求的近似 pacing equilibrium，那么你也能高效解决 PPAD 中一大类困难问题。

这不是说现实广告系统完全不能运行，而是说：

> 在最坏情形下，不应期待一个通用多项式时间算法能保证找到这种近似 equilibrium。

---

# 8. 论文为什么难？核心证明思路概览

论文的证明是一个 **reduction**。

Reduction 的结构是：

$$
\text{Pure-Circuit}  
;\longrightarrow;  
\text{Second-price pacing game}.  
$$

Pure-Circuit 是一个已知 PPAD-complete 的问题。论文把任意一个 Pure-Circuit 实例转换成一个二价 pacing game，使得：

> 只要你能找到这个 pacing game 的 $\gamma$-approximate pacing equilibrium，就能从里面读出原 Pure-Circuit 的一个解。

于是，如果 pacing equilibrium 可以高效近似，就能高效解 Pure-Circuit。因为 Pure-Circuit 是 PPAD-complete，所以 pacing equilibrium 近似计算是 PPAD-hard。论文使用 Pure-Circuit 来证明强常数不可逼近性，这是近年 PPAD 常数不可逼近结果中常见的技术路线。([arXiv](https://arxiv.org/pdf/2501.15295 "Constant Inapproximability of Pacing Equilibria in Second-Price Auctions"))

---

# 9. Pure-Circuit 是什么？

Pure-Circuit 可以理解成一种“带未知值的逻辑电路”。

普通布尔电路里，每个变量是 0 或 1。Pure-Circuit 里，每个节点可以取三个值：

$$
x[v]\in{0,1,\bot}.  
$$

其中：

- (0)：false；
    
- (1)：true；
    
- $\bot$：未知、未纯化、垃圾值。
    

Pure-Circuit 有几类门：

## NOT 门

如果输入是布尔值，则输出必须是取反：

$$
x[u]\in{0,1}  
\quad\Longrightarrow\quad  
x[v]=\neg x[u].  
$$

如果输入是 $\bot$，则约束可以放松。

---

## NOR 门

NOR 是 “not OR”。

如果两个输入都是 0：

$$
x[u]=0,\quad x[v]=0,  
$$

则输出为 1：

$$
x[w]=1.  
$$

如果至少一个输入是 1：

$$
x[u]=1\ \text{or}\ x[v]=1,  
$$

则输出为 0：

$$
x[w]=0.  
$$

---

## PURIFY / NPURIFY 门

PURIFY 门的作用是保证系统里总有一些变量被“推回”到布尔值 (0/1)，不能全都逃到 $\bot$。

论文为了构造方便，使用的是 NPURIFY，也就是 negated purification。它大致要求：

- 至少一个输出是布尔值；
    
- 如果输入是布尔值，那么两个输出都等于输入的反值。
    

论文说明，NPURIFY 可以由 PURIFY 加 NOT 模拟，因此使用 NOT、NOR、NPURIFY 的 Pure-Circuit 仍然 PPAD-complete。([arXiv](https://arxiv.org/pdf/2501.15295 "Constant Inapproximability of Pacing Equilibria in Second-Price Auctions"))

---

# 10. 变量如何编码成 pacing multiplier？

这是整篇论文最关键的想法。

对 Pure-Circuit 中每个变量 (v)，构造一个买家 (b_v)。用这个买家的 pacing multiplier

$$
\alpha_{b_v}  
$$

来编码变量 (x[v])。

论文选择参数：

$$
\delta=1/3-\gamma>0,  
$$

$$
\kappa=\frac{3\delta}{2}.  
$$

然后设计拍卖实例，使得在任何 $\gamma$-approximate pacing equilibrium 中，都有

$$
\alpha_{b_v}\in[\kappa,1].  
$$

最后用下面规则解码：

$$
x[v]=  
\begin{cases}  
0, & \alpha_{b_v}=\kappa,\  
1, & \alpha_{b_v}=1,\  
\bot, & \text{otherwise}.  
\end{cases}  
$$

这一步非常巧妙。它不需要强迫每个 multiplier 都必须精确等于 $\kappa$ 或 (1)。如果 multiplier 落在中间，就解码成 $\bot$。Pure-Circuit 本来就允许 $\bot$，所以 reduction 可以容忍中间值。

这就是为什么 Pure-Circuit 很适合做这种常数不可逼近证明：它天然允许“模糊区域”。

---

# 11. 第一个关键 gadget：强迫 $\alpha_{b_v}\ge \kappa$

论文的 technical overview 里先解释了一个很重要的小技巧：通过加入一个辅助买家和一个辅助商品，可以强迫某个买家的 multiplier 不能太小。([arXiv](https://arxiv.org/pdf/2501.15295 "Constant Inapproximability of Pacing Equilibria in Second-Price Auctions"))

直觉如下。

对变量买家 (b_v)，加入一个辅助买家 (c_v) 和一个辅助商品 (g_v)。

辅助买家 (c_v) 很有钱，预算非常大，因此它不会因为预算问题而 pacing，于是

$$
\alpha_{c_v}=1.  
$$

设置 (b_v) 和 (c_v) 对 (g_v) 的价值，使得：

- 当 $\alpha_{b_v}<\kappa$ 时，(b_v) 的出价低于 (c_v)，所以 (b_v) 买不到 (g_v)；
    
- 当 $\alpha_{b_v}=\kappa$ 时，(b_v) 和 (c_v) 刚好打平；
    
- 当 $\alpha_{b_v}>\kappa$ 时，(b_v) 赢下整个 (g_v)。
    

如果 $\alpha_{b_v}<\kappa$，那么 (b_v) 买不到这个辅助商品，花钱太少。论文把预算和价值设计成：这时 (b_v) 的总支出会低于

$$
(1-\gamma)B_{b_v}.  
$$

根据 $\gamma$-approximate pacing equilibrium 的 no unnecessary pacing 条件，支出这么少就必须有

$$
\alpha_{b_v}=1.  
$$

这和 $\alpha_{b_v}<\kappa$ 矛盾。因此必然：

$$
\alpha_{b_v}\ge\kappa.  
$$

这个 gadget 的讲解重点是：

> 如果 multiplier 太小，买家花钱太少；  
> 花钱太少就不能被 pacing；  
> 所以 multiplier 不能太小。

这就是 Lemma 1 的核心。

---

# 12. 第二个关键 gadget：让输出买家“支付输入 multiplier”

接下来要模拟逻辑门。论文设计一种商品 (g_{(u,v)})，连接输入变量 (u) 和输出变量 (v)。

构造方式是：

- 输入买家 (b_u) 对 (g_{(u,v)}) 的价值为 1；
    
- 输出买家 (b_v) 对 (g_{(u,v)}) 的价值很大，约为 $1/\kappa+1$。
    

因为已经知道

$$
\alpha_{b_v}\ge\kappa,  
$$

所以输出买家 (b_v) 对该商品的出价至少大于输入买家 (b_u) 的最高可能出价。于是：

> (b_v) 一定赢得这个商品。

但这是二价拍卖，赢家支付第二高价。第二高价来自 (b_u)，而 (b_u) 的出价是

$$
\alpha_{b_u}\cdot 1=\alpha_{b_u}.  
$$

所以 (b_v) 赢下商品 (g_{(u,v)})，但支付的是：

$$
\alpha_{b_u}.  
$$

这就是 Lemma 2 的核心：

> 用二价拍卖可以让一个买家的支出精确携带另一个买家的 multiplier。

这一步是整篇 reduction 的灵魂。它把“逻辑输入”变成了“输出买家的支付金额”。

---

# 13. NOT 门怎么模拟？

假设 Pure-Circuit 中有一个 NOT 门：

$$
u\to v.  
$$

也就是说：

$$
x[v]=\neg x[u].  
$$

在 pacing game 中，对应输入买家 (b_u)，输出买家 (b_v)。

论文设置 (b_v) 的预算为：

$$
B_{b_v}=2.  
$$

并给 (b_v) 两类商品：

1. 输入商品 (g_{(u,v)})：(b_v) 一定赢，并支付 $\alpha_{b_u}$；
    
2. 辅助商品 (g_v)：用于判断 (b_v) 是否超过阈值 $\kappa$。
    

辅助商品的价格设计成大约 $1+\delta$。

于是 (b_v) 的支出结构大致是：

$$
\text{支出}=\alpha_{b_u}+\text{辅助商品支付}.  
$$

现在看两种布尔输入情况。

---

## 情况 1：输入是 0

根据解码规则，输入是 0 表示：

$$
\alpha_{b_u}=\kappa.  
$$

这时 (b_v) 的支出最多是：

$$
\kappa+(1+\delta).  
$$

论文选择

$$
\kappa=\frac{3\delta}{2}  
$$

正是为了保证这个支出低于：

$$
(1-\gamma)B_{b_v}=2(1-\gamma).  
$$

所以 (b_v) 花钱太少。根据 no unnecessary pacing 条件，必须：

$$
\alpha_{b_v}=1.  
$$

解码后：

$$
x[v]=1.  
$$

这正是 NOT：输入 0，输出 1。

---

## 情况 2：输入是 1

输入是 1 表示：

$$
\alpha_{b_u}=1.  
$$

如果输出买家 (b_v) 试图选择

$$
\alpha_{b_v}>\kappa,  
$$

那么它会赢下整个辅助商品，总支付变成：

$$
1+(1+\delta)=2+\delta.  
$$

但预算是 2，因此超预算，违反 budget constraint。

所以不能有 $\alpha_{b_v}>\kappa$。又因为 Lemma 1 已经保证

$$
\alpha_{b_v}\ge\kappa,  
$$

所以只能：

$$
\alpha_{b_v}=\kappa.  
$$

解码后：

$$
x[v]=0.  
$$

这正是 NOT：输入 1，输出 0。

因此 NOT 门被成功模拟。

---

# 14. NOR 门怎么模拟？

NOR 门有两个输入 (u,v)，一个输出 (w)：

$$
w=\operatorname{NOR}(u,v).  
$$

也就是：

- 输入都是 0，则输出 1；
    
- 只要有一个输入是 1，则输出 0。
    

在 pacing game 中，输出买家是 (b_w)。论文设置：

$$
B_{b_w}=3.  
$$

构造两个输入商品：

$$
g_{(u,w)},\qquad g_{(v,w)}.  
$$

根据 Lemma 2，(b_w) 一定赢下这两个商品，并分别支付：

$$
\alpha_{b_u},\qquad \alpha_{b_v}.  
$$

再加一个辅助商品 (g_w)，其辅助支付大约是：

$$
2-\delta.  
$$

于是 (b_w) 的支出结构大致为：

$$
\alpha_{b_u}+\alpha_{b_v}+(2-\delta).  
$$

---

## 情况 1：两个输入都是 0

输入都是 0 表示：

$$
\alpha_{b_u}=\kappa,\qquad \alpha_{b_v}=\kappa.  
$$

此时支出最多：

$$
2\kappa+(2-\delta).  
$$

论文参数保证：

$$
2\kappa+2-\delta < 3(1-\gamma).  
$$

所以 (b_w) 花钱太少，必须 no unnecessary pacing：

$$
\alpha_{b_w}=1.  
$$

解码得到：

$$
x[w]=1.  
$$

这符合 NOR。

---

## 情况 2：至少一个输入是 1

比如

$$
\alpha_{b_u}=1.  
$$

如果 (b_w) 试图选择

$$
\alpha_{b_w}>\kappa,  
$$

它会赢下辅助商品，支付至少：

$$
1+\kappa+(2-\delta)=3+\kappa-\delta.  
$$

因为

$$
\kappa=\frac{3\delta}{2}>\delta,  
$$

所以

$$
3+\kappa-\delta>3.  
$$

但预算是 3，超预算。因此不能有 $\alpha_{b_w}>\kappa$。结合 Lemma 1 的

$$
\alpha_{b_w}\ge\kappa,  
$$

只能得到：

$$
\alpha_{b_w}=\kappa.  
$$

解码得到：

$$
x[w]=0.  
$$

这也符合 NOR。

---

# 15. NPURIFY 门为什么需要？

如果只有 NOT 和 NOR，而变量可以取 $\bot$，很多约束在输入为 $\bot$ 时会变得松。为了保证 reduction 有足够的“布尔信息”，Pure-Circuit 使用 PURIFY 类型的门。

论文使用 NPURIFY。直觉上它做两件事：

1. 如果输入是布尔值，两个输出都变成输入的反值；
    
2. 无论输入是不是布尔值，至少保证某个输出是布尔值。
    

这避免所有变量都落在中间区域 $\bot$。

---

# 16. NPURIFY 门怎么模拟？

假设有 NPURIFY 门：

$$
u\to (v,w).  
$$

输入是 (u)，两个输出是 (v,w)。

论文给 (b_v) 和 (b_w) 都设置预算：

$$
B_{b_v}=B_{b_w}=3/2.  
$$

两人都各自有一个输入商品：

$$
g_{(u,v)},\qquad g_{(u,w)}.  
$$

根据 Lemma 2：

- (b_v) 赢 (g_{(u,v)})，支付 $\alpha_{b_u}$；
    
- (b_w) 赢 (g_{(u,w)})，支付 $\alpha_{b_u}$。
    

此外，它们分别有不同辅助商品，辅助价格设计为：

$$
1-\delta/2  
$$

和

$$
1/2+\delta/2.  
$$

这个不对称设计用于保证不同区间下至少一个输出被推到布尔端点。

论文证明四个关键结论：

1. 如果
    

$$
\alpha_{b_u}=\kappa,  
$$

则

$$
\alpha_{b_v}=1.  
$$

2. 如果
    

$$
\alpha_{b_u}>1/2+\delta/2,  
$$

则

$$
\alpha_{b_v}=\kappa.  
$$

3. 如果
    

$$
\alpha_{b_u}\le 1/2+\delta/2,  
$$

则

$$
\alpha_{b_w}=1.  
$$

4. 如果
    

$$
\alpha_{b_u}=1,  
$$

则

$$
\alpha_{b_w}=\kappa.  
$$

于是：

- 如果输入是 0，即 $\alpha_{b_u}=\kappa$，则两个输出都会被推成 1；
    
- 如果输入是 1，即 $\alpha_{b_u}=1$，则两个输出都会被推成 0；
    
- 如果输入是中间值 $\bot$，至少能保证某个输出落在布尔端点。
    

这正是 NPURIFY 的作用。

---

# 17. 整个 reduction 的正确性总结

证明可以压缩成下面这条链：

$$
\text{Pure-Circuit 实例}  
\Rightarrow  
\text{构造 pacing game }G  
\Rightarrow  
\text{任意 }\gamma\text{-approx PE}  
\Rightarrow  
\text{读出 Pure-Circuit 解}.  
$$

具体分四步：

## 第一步：变量买家不会低于 $\kappa$

辅助商品 gadget 保证：

$$
\alpha_{b_v}\ge\kappa.  
$$

否则买家花钱太少，违反 no unnecessary pacing。

---

## 第二步：输入商品把 multiplier 变成支付

对每条逻辑依赖 $u\to v$，构造商品 (g_{(u,v)})，让 (b_v) 一定赢，但支付

$$
\alpha_{b_u}.  
$$

这一步利用的是二价拍卖的“赢家付第二高价”。

---

## 第三步：预算和辅助商品强迫输出 multiplier

通过精确设置预算和辅助商品价格：

- 如果输出买家支出太少，则它必须 $\alpha=1$；
    
- 如果输出买家 $\alpha>\kappa$ 会超预算，则它只能 $\alpha=\kappa$。
    

于是 $\kappa$ 和 (1) 模拟了布尔值 0 和 1。

---

## 第四步：NOT、NOR、NPURIFY 都被正确模拟

因此从任意 approximate pacing equilibrium 中解码出的

$$
x[v]=  
\begin{cases}  
0, & \alpha_{b_v}=\kappa,\  
1, & \alpha_{b_v}=1,\  
\bot, & \text{otherwise}  
\end{cases}  
$$

一定满足所有 Pure-Circuit 门。

所以，如果有算法能算 pacing equilibrium 的 $\gamma$-approximation，就能算 Pure-Circuit。由于 Pure-Circuit 是 PPAD-complete，所以该 pacing equilibrium 近似问题 PPAD-hard。

---

# 18. 为什么是 (1/3) 这个常数？

论文设：

$$
\delta=1/3-\gamma.  
$$

只要

$$
\gamma<1/3,  
$$

就有

$$
\delta>0.  
$$

再令

$$
\kappa=\frac{3\delta}{2}.  
$$

所有 gadget 的预算不等式都依赖于这些正 margin。比如 NOT 门需要保证某些支出低于 ($1-\gamma$B)，NOR 和 NPURIFY 也类似。

所以 (1/3) 不是随便来的，而是这组 gadget 中预算 gap 能同时成立的临界设计。论文也明确把“(1/3) 是否 tight”列为开放问题：他们不知道是否存在针对其构造实例族的 (1/3)-approximation 算法，也不知道在每个 bidder 最多 bid 4 个 item 的情形下是否能做到 (1/3)-approximation。([arXiv](https://arxiv.org/pdf/2501.15295 "Constant Inapproximability of Pacing Equilibria in Second-Price Auctions"))

---

# 19. 每个 bidder 最多对 4 个 item 出价为什么成立？

这点很重要，因为它说明 hardness 不是靠“一个买家连接海量商品”堆出来的。

Pure-Circuit 使用的实例可以保证每个节点的总交互度很小。论文的构造里，每个变量买家 (b_v) 主要参与：

1. 自己作为输出时的辅助商品 (g_v)；
    
2. 自己作为某个门输出时的输入连接商品；
    
3. 自己作为其他门输入时对应的少数连接商品。
    

由于 Pure-Circuit 的节点度数被限制，每个变量买家最终只会对常数个商品有非零价值。论文主定理进一步给出强结论：每个 bidder 最多 bid on 4 items。([arXiv](https://arxiv.org/pdf/2501.15295 "Constant Inapproximability of Pacing Equilibria in Second-Price Auctions"))

课堂上可以这样说：

> reduction 不靠高度连接的复杂市场，而是在非常稀疏的拍卖图上就能编码 PPAD-hard 的电路结构。

---

# 20. Appendix 里的 weaker approximation 是什么？

论文还证明了一个更鲁棒的结果。

前面 Definition 2 的近似 equilibrium 只放松了 no unnecessary pacing 条件。Appendix 里考虑更弱的 ($\sigma,\gamma,\tau$)-approximate pacing equilibrium：

1. 赢者不一定必须是最高出价者，只要接近最高出价；
    
2. 预算仍然不能超；
    
3. 如果支出明显低于预算，不再要求 $\alpha_i=1$，只要求
    

$$
\alpha_i\ge 1-\tau.  
$$

论文证明：对常数

$$
\sigma\le 1/20,\qquad \gamma\le 1/20,\qquad \tau\le 1/20,  
$$

计算这种更弱的 approximate pacing equilibrium 仍然 PPAD-hard，而且每个 buyer 仍最多只对 4 个 goods 有非零价值。([arXiv](https://arxiv.org/pdf/2501.15295 "Constant Inapproximability of Pacing Equilibria in Second-Price Auctions"))

这说明论文的困难性不是非常脆弱的：即使把 winner condition 和 no-unnecessary-pacing condition 都进一步放松，问题仍然有常数级困难性。

---

# 21. 这篇论文的贡献应该怎么评价？

它的贡献可以分成三层。

## 第一层：补上二价 pacing equilibrium 的常数不可逼近缺口

之前 Chen、Kroer、Kumar 已经证明过，二价 pacing equilibrium 的近似计算在 inverse-polynomial precision 下是 PPAD-hard / PPAD-complete。但 inverse-polynomial precision 是非常高精度的要求。

这篇论文把结果强化为：

> 常数精度下也 PPAD-hard。

这更强，因为实际算法通常不追求极高精度，而是希望有个常数误差保证。如果常数误差都难，说明理论障碍更根本。

---

## 第二层：说明二价拍卖和一价拍卖在 pacing 上差异很大

一价拍卖中，赢家支付自己的出价，因此 pacing multiplier 与支付之间关系比较直接，很多 dynamics 或 tâtonnement-style 方法更容易工作。二价拍卖中，赢家支付的是别人出的第二价，导致一个人的 multiplier 会通过价格强烈影响另一个人的预算消耗。这种交叉依赖正是 reduction 能编码逻辑门的原因。

相关工作中也指出，一价拍卖下 pacing / throttling equilibrium 有多项式时间算法，而二价拍卖的 pacing equilibrium 在一般情形下计算困难。([ar5iv](https://ar5iv.org/pdf/2501.15295 "[2501.15295] Constant Inapproximability of Pacing Equilibria in Second-Price Auctions"))

---

## 第三层：展示 Pure-Circuit 技术在算法博弈论不可逼近中的威力

这篇论文不是单独发明了 Pure-Circuit，而是把它成功应用到 pacing equilibrium 上。Pure-Circuit 的三值逻辑特别适合 equilibrium reductions，因为 equilibrium 变量经常很难被强迫成精确二值，但可以自然分成：

- 一个端点代表 0；
    
- 另一个端点代表 1；
    
- 中间区域代表 $\bot$。
    

这比传统从 Brouwer 或 End-of-Line 直接做几何归约更适合讲给学生。

---

# 22. 初学者最容易误解的地方

## 误解 1：这是在近似 welfare 或 revenue 吗？

不是。

这里的 approximation 不是说社会福利达到最优的多少倍，也不是 revenue approximation。

这里近似的是 equilibrium 条件，尤其是 no unnecessary pacing 条件：

$$
\text{支出低于 }(1-\gamma)B_i  
\Rightarrow  
\alpha_i=1.  
$$

所以 “constant inapproximability” 指的是 equilibrium notion 的常数松弛仍然难算。

---

## 误解 2：PPAD-hard 是否说明现实广告平台做不了 pacing？

不是。

PPAD-hard 是最坏情形复杂性结论。现实市场可能有特殊结构、随机性、低维特征、可聚合商品类型、经验 dynamics 等。因此平台可以有实用启发式或在特定结构下可证明有效的算法。

论文告诉我们的是：

> 不加结构限制时，不应期待通用多项式时间算法有理论保证。

---

## 误解 3：$\gamma=0.99$ 是否也被证明难？

没有。

论文主结果覆盖的是任意常数

$$
\gamma<1/3.  
$$

当 $\gamma$ 很大，比如 0.99，approximate equilibrium 条件非常弱。论文把是否存在非平凡近似算法，例如 0.99-approximate pacing equilibrium 算法，列为开放方向。([arXiv](https://arxiv.org/pdf/2501.15295 "Constant Inapproximability of Pacing Equilibria in Second-Price Auctions"))

---

## 误解 4：每个 multiplier 都会被强迫成 0 或 1 吗？

不是。

变量编码用的是：

$$
\kappa \mapsto 0,\qquad 1\mapsto 1,\qquad \text{中间值}\mapsto\bot.  
$$

所以中间值是允许存在的。Pure-Circuit 正是为了处理这种中间值而设计的。

---

# 23. 一堂课可以怎么讲？

你可以按下面 6 个模块组织。

## 模块 1：从广告预算讲 pacing

先讲二价拍卖，再讲预算导致真实出价不够用，引入

$$
\alpha_i v_{ij}.  
$$

目标是让学生知道 pacing 是现实广告系统中的预算控制手段。

---

## 模块 2：定义 pacing equilibrium

板书四条：

1. 只有最高出价者能赢；
    
2. 有正出价就全部分配；
    
3. 不超预算；
    
4. 没花完预算就不能 pacing。
    

然后解释 approximate version 只放松第 4 条。

---

## 模块 3：主定理

写：

$$
\forall \gamma<1/3,\quad  
\gamma\text{-approx PE is PPAD-hard}.  
$$

强调：

- 常数近似；
    
- 稀疏实例；
    
- 排除 PTAS，除非 PPAD = FP。
    

---

## 模块 4：证明路线

画一条 reduction：

$$
\text{Pure-Circuit}  
\to  
\text{Pacing game}  
\to  
\text{Approx PE}  
\to  
\text{Circuit solution}.  
$$

---

## 模块 5：两个 gadget

重点讲两个 gadget：

1. **lower-bound gadget**：强迫 $\alpha\ge\kappa$；
    
2. **payment-transfer gadget**：让输出买家赢商品，但支付输入买家的 multiplier。
    

只要学生理解这两个 gadget，整篇论文就基本通了。

---

## 模块 6：三个门

最后讲：

- NOT：低输入导致输出花太少，所以输出 $\alpha=1$；高输入导致输出若太高会超预算，所以输出 $\alpha=\kappa$。
    
- NOR：两个低输入输出高；任一高输入输出低。
    
- NPURIFY：保证至少产生布尔输出，避免全是 $\bot$。
    

---

# 24. 可以延伸的研究方向

## 方向 1：tight approximability

论文证明 $\gamma<1/3$ 时难，但 $\gamma\ge 1/3$ 的精确边界不清楚。最直接的问题是：

$$
1/3  
$$

是不是 tight？

更广义地问：

> 二价 pacing equilibrium 最好的多项式时间近似保证到底是多少？

论文也明确提出是否存在非平凡常数近似算法，例如 0.99-approximate pacing equilibrium。([arXiv](https://arxiv.org/pdf/2501.15295 "Constant Inapproximability of Pacing Equilibria in Second-Price Auctions"))

---

## 方向 2：特殊结构下的正算法

虽然一般情形 PPAD-hard，但限制市场结构后可能可解。

论文中曾把 “商品数 (m) 为常数时是否可解” 列为开放问题。后续已有新结果研究常数买家数和常数商品数的情形：AAAI 2026 的 few buyers 工作给出常数 buyer 数下精确计算 SPPE 的多项式时间算法；另有 2026 年 arXiv 预印本给出常数 goods 数下精确计算 SPPE 的多项式时间算法，并把问题化为几何 cell decomposition 与线性可行性问题。([AAAI出版物](https://ojs.aaai.org/index.php/AAAI/article/view/38782/42744 "Pacing Equilibria in Second-Price Auctions with Few Buyers"))

这类方向很适合作为课程项目：

> 一般情形难，但低维结构可解。

---

## 方向 3：参数化复杂性

主定理说明即使每个 bidder 最多 bid 4 个 items 也难，但还有很多参数可以研究：

- 商品数 (m)；
    
- 买家数 (n)；
    
- 估值矩阵的 rank；
    
- 预算类型数量；
    
- 商品类型数量；
    
- bidder-item 图的 treewidth；
    
- 每个商品竞争者数量；
    
- 每个 bidder 参与商品数量更小的情形，比如最多 2 或 3。
    

可以问：

> 哪些参数固定后，SPPE 可以精确或近似多项式时间计算？

---

## 方向 4：first-price vs second-price 的复杂性差异

一价 pacing equilibrium 更容易计算，而二价 pacing equilibrium 很难。一个非常适合继续研究的问题是：

> 到底是 second-price 的哪个结构导致 PPAD-hard？

直觉上，关键在于二价拍卖中：

$$
\text{赢家支付别人出的价}.  
$$

这让 reduction 可以把输入变量的 multiplier 变成输出买家的支付。而一价拍卖中，支付主要由自己的 multiplier 决定，逻辑编码难度不同。

可以延伸研究 hybrid auction、reserve price、soft second price、generalized second price 等机制下的 pacing equilibrium 复杂性。

---

## 方向 5：动态过程是否收敛

现实系统不会一次性“求解 equilibrium”，而是通过反复调节 multiplier 来逼近稳定状态。

可以研究：

- best-response dynamics；
    
- tâtonnement dynamics；
    
- no-regret learning；
    
- mirror descent；
    
- bandit feedback；
    
- online budget pacing。
    

主问题是：

> 即使 equilibrium 难算，实际动态是否在典型实例上收敛？  
> 收敛到的点满足什么近似 equilibrium 保证？

---

## 方向 6：平均情形和 smoothed analysis

PPAD-hard 是 worst-case。现实广告估值可能有随机结构，因此可以问：

- 随机估值矩阵下是否容易？
    
- 加一点噪声后是否容易？
    
- budget 分布满足某些 regularity 时是否容易？
    
- gadget 构造是否对噪声敏感？
    

这可以把理论复杂性和实证广告系统连接起来。

---

## 方向 7：多 equilibrium 的选择问题

Pacing equilibrium 可能不唯一。即使找到一个 equilibrium，还可以问：

- 哪个 equilibrium revenue 最大？
    
- 哪个 welfare 最大？
    
- 哪个对广告主最公平？
    
- 哪个最稳定？
    
- 哪个更容易由动态过程达到？
    

few buyers 的后续工作已经提到，在其可解结构下可以识别优化 revenue 或 social welfare 的 equilibrium。([AAAI出版物](https://ojs.aaai.org/index.php/AAAI/article/view/38782/42744 "Pacing Equilibria in Second-Price Auctions with Few Buyers"))

---

## 方向 8：更弱近似定义的 tight bound

Appendix 证明 ($\sigma,\gamma,\tau$)-approximation 在常数 (1/20) 附近也难，但这些常数没有优化。自然问题是：

> 对更弱的 approximate pacing equilibrium，最优 hardness threshold 是多少？

这可以作为更技术性的延伸。

---

# 25. 最后用三句话总结给学生

第一句话：

> Pacing equilibrium 描述的是预算受限广告主在二价拍卖中通过统一乘数压低出价后形成的稳定状态。

第二句话：

> 论文证明，这个稳定状态不仅精确计算难，连 $\gamma<1/3$ 的常数近似也 PPAD-hard，而且在每个 bidder 只参与很少商品时仍然难。

第三句话：

> 证明的核心是把 Pure-Circuit 的逻辑门编码进预算和二价支付：输出买家赢商品但支付输入买家的 multiplier，再用预算是否绑定强迫输出 multiplier 变成逻辑 0 或 1。