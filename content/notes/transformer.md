---
title: transformer
---

# Transformer 学习笔记

## 输入表示
Transformer 的输入不是直接处理词本身，而是先把每个 token 映射成向量表示，再加入位置信息。

- **词向量（word embedding）**：把输入 token 映射为向量，记作 $X$
- **位置编码（positional encoding）**：给每个位置注入顺序信息，记作 $PE$

因此输入表示可以写成：

$$
H = X + PE
$$

经典 Transformer 使用正弦与余弦位置编码：

$$
PE_{(pos,2i)} = \sin\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right)
$$

$$
PE_{(pos,2i+1)} = \cos\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right)
$$

其中：
- $pos$ 表示位置；
- $i$ 表示维度索引；
- $d_{\text{model}}$ 表示模型隐藏维度。

## 缩放点积注意力
对于上一层输入 $X$，通过三个可学习参数矩阵把它映射为 Query、Key、Value：

$$
Q = XW_Q, \quad K = XW_K, \quad V = XW_V
$$

然后计算注意力分数：

$$
\text{score} = QK^T
$$

为了防止点积随维度增大而数值过大，需要做缩放：

$$
\text{score}_{\text{scaled}} = \frac{QK^T}{\sqrt{d_k}}
$$

再经过 softmax 得到注意力权重：

$$
A = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)
$$

最后得到注意力输出：

$$
\text{Attention}(Q,K,V) = A V = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

## 多头注意力
多头注意力的核心思想是：让模型在不同的表示子空间中并行学习信息，例如语法关系、语义关系、长程依赖等。

对于第 $i$ 个头：

$$
\text{head}_i = \text{Attention}(Q_i, K_i, V_i)
$$

其中：

$$
Q_i = XW_i^Q, \quad K_i = XW_i^K, \quad V_i = XW_i^V
$$

把所有头拼接后，再经过一次线性变换：

$$
\text{MultiHead}(X) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h)W^O
$$

## 残差连接与层归一化
Transformer 在每个子层后都使用残差连接与层归一化，以稳定训练并保留原始信息。

可以写成：

$$
Y = \text{LayerNorm}(X + \text{Sublayer}(X))
$$

这里的子层可以是多头注意力，也可以是前馈神经网络。

## 前馈神经网络
在注意力层之后，每个位置的表示都会独立通过一个两层前馈网络。

原始论文中常写为：

$$
\text{FFN}(x) = \max(0, xW_1 + b_1)W_2 + b_2
$$

论文原版使用 ReLU，现代大模型里更常见的是 GELU。

## Encoder 结构
一个 Transformer Encoder block 通常包含两部分：

1. 多头自注意力；
2. 前馈神经网络。

并且每部分后面都带有：
- 残差连接；
- LayerNorm。

因此可以理解为：

$$
H' = \text{LayerNorm}(H + \text{MultiHead}(H))
$$

$$
H'' = \text{LayerNorm}(H' + \text{FFN}(H'))
$$

多个 Encoder block 堆叠后，得到最终的编码表示。

## Decoder 结构
Decoder 比 Encoder 多了一个关键机制：**掩码自注意力（masked self-attention）**。

### 1. 掩码自注意力
Decoder 在生成第 $t$ 个 token 时，不能看到未来位置的信息，因此需要加上因果 mask：

$$
\text{MaskedAttention}(Q,K,V) = \text{softmax}\left(\frac{QK^T + M}{\sqrt{d_k}}\right)V
$$

其中 $M$ 是掩码矩阵，对未来位置赋予 $-\infty$，使 softmax 后这些位置权重为 0。

### 2. 交叉注意力
Decoder 还会使用 Encoder 的输出作为 Key 和 Value，让生成过程能够关注输入序列：

$$
\text{CrossAttention}(Q, K_{enc}, V_{enc})
$$

其中：
- $Q$ 来自 Decoder 当前状态；
- $K_{enc}, V_{enc}$ 来自 Encoder 输出。

## KV Cache
在 Decoder 推理时，生成每一个新 token 都需要用到前面所有 token 的 Key 和 Value。

如果不做缓存，那么每生成一个 token，都要把之前所有位置重新计算一遍，代价很高。

因此推理时通常会把历史 token 的 Key 和 Value 缓存起来：

- 第一次生成时，计算并保存历史位置的 $K, V$；
- 后续生成新 token 时，只需要计算新 token 对应的 $Q, K, V$；
- 再把新 token 的 $K, V$ 拼接到缓存中。

这样做可以把重复计算降到最低，显著提升自回归生成速度。

## 小结
Transformer 的核心可以概括为：

- 用 embedding 和位置编码表示输入；
- 用自注意力捕捉全局依赖；
- 用多头机制学习不同子空间的信息；
- 用残差连接和层归一化稳定训练；
- 用 FFN 提升非线性表示能力；
- 在 Decoder 中通过 mask 保证因果性；
- 在推理阶段通过 KV cache 提升生成效率。
