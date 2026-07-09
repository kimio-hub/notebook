---
title: ReCALL-Analysis
---

我读完你的设计文档后，理解你说的“跑不起来多轮”的意思是：**Naive Iterative ReCALL 表面上每轮都在挖 hard negatives、生成新 corrective triplets、继续训练，但第二轮检索时，原始 `(I_r, T)` 仍然被同一张错误图或同一类错误图吸引。** 这说明多轮并没有形成新的有效监督，而是在重复消费同一批错误样本。你的文档已经把这个问题定位为 **persistent errors**：ReCALL 一轮校准后，后续仍可能反复犯同一种 hard-negative 错误，因此需要显式追踪 repeated hard negatives / repeated patterns，并把原始 query、错误模式 query、澄清 query 放进同一个 tri-query micro-group 中训练，让监督直接作用到原始 `(I_r, T)` 的排序上。

我认为你这个方向是对的，但还需要再往前推一步：**不仅要提出 PE-ReCALL，还要解释为什么普通 ReCALL 多轮会卡死，并在算法上强制“第二轮监督必须和第一轮不同”。**

---

## 1. 先判断：你遇到的可能有两个层面的原因

### 第一类是方法层面的真实 persistent error

ReCALL 原始机制大致是：找到排在 ground truth 前面的 hard negative `I_h`，让 MLLM 生成一个 `T_h^+`，使 `I_h` 在新文本下成为合法正样本，然后继续训练。ReCALL 论文也将其描述为 diagnose-generate-refine 流程：先诊断 retriever 的 blind spots，再由 foundation MLLM 生成 corrective instructions/triplets，最后用 grouped contrastive scheme 继续训练 retriever。([CVF开放获取](https://openaccess.thecvf.com/content/CVPR2026/html/Yang_ReCALL_Recalibrating_Capability_Degradation_for_MLLM-based_Composed_Image_Retrieval_CVPR_2026_paper.html "CVPR 2026 Open Access Repository"))

问题在于：这个 corrective triplet 主要教模型 **“`I_h` 在另一个文本 `T_h^+` 下应该是正样本”**，但它不一定强制教模型 **“原始文本 `T` 下，`I_h` 必须被 `I_t` 压下去”**。你的文档也明确指出，如果只训练新生成文本，原始 query 可能仍然检索错误，所以必须验证把原始 `T`、错误模式 `T_pattern^+`、澄清文本 `T_pattern^-` 放进同一个 group 后，原始 query 排序是否直接改善。

所以，同一张错图第二轮还出来，不一定说明训练没生效；更准确地说，是 **新增样本没有对原始 query 的局部排序边界产生足够梯度**。

### 第二类是工程层面的“伪多轮”

这个也非常值得查。ReCALL 官方实现的 README 里说，迭代训练流程是每轮 encode training images & queries、mine hard negatives、generate augmented captions、再训练 original + augmented data。([GitHub](https://github.com/RemRico/Recall/tree/main "GitHub - RemRico/Recall: A composed retrieval project · GitHub")) 但从代码看，target embeddings 的 cache key 主要由 target database 的路径列表决定；cache validation 检查的是 `target_paths` 和 embedding 数量，而不是当前模型 checkpoint / iteration。([GitHub](https://raw.githubusercontent.com/RemRico/Recall/main/src/retrieval/embedding_cache.py "raw.githubusercontent.com")) 这意味着，如果你的实现类似，**第二轮可能在复用第一轮模型编码出来的 gallery embeddings**。这种情况下即使 query encoder 更新了，gallery embedding 还是旧的，甚至 query/gallery embedding space 会错配，top-k 很容易稳定地重复错图。

另外，hard negative miner 也会在同一 iteration_round 的 json 文件存在时直接加载已有 hard negatives。([GitHub](https://raw.githubusercontent.com/RemRico/Recall/main/src/mining/hard_negative.py "raw.githubusercontent.com")) 如果你反复调试同一个 round，没有删掉旧的 `hard_negatives_iter_k.json`，也可能看起来“第二轮还是同一批错图”，但其实根本没重新挖。

**所以第一步建议你先做一个 sanity check：**

```text
每一轮开始前：
1. 强制删除 / 禁用 target embedding cache；
2. 强制重新 encode 当前 R_{k-1} 的 gallery images；
3. 强制重新 encode 当前 R_{k-1} 的 original queries；
4. 删除当前 iteration 的 hard_negatives_iter_k.json；
5. 保存每个 query 的：
   - gt_rank
   - top-10 image ids
   - sim(q0, It)
   - sim(q0, Ih)
   - margin = sim(q0, It) - sim(q0, Ih)
```

如果 cache 修正后 top-k 仍重复，才是论文里真正值得研究的 **persistent error**；如果修正后不重复，那你的瓶颈首先是工程 cache / mining pipeline，而不是方法本身。

---

## 2. 我建议把 PE-ReCALL 的核心从“三文本生成”再升级为“原始 query 锚定的排序修复”

你现在文档里的 tri-query micro-group 是必要的：`q_0 = encode(I_r, T)`、`q_- = encode(I_r, T_pattern^-)`、`q_+ = encode(I_r, T_pattern^+)`，并要求 `q_0` 和 `q_-` 指向 `I_t`，`q_+` 指向 `H_pattern`。 这一点正好解决“只训练新文本不影响原始 query”的问题。

但我建议你把 loss 写得更直接一些。不要只说展开成 triplet，而是明确加入一个 **Original-query Rank Repair Loss**：

```text
L_oqr =
Σ_{h in H_persistent}
softplus((s(q0, h) - s(q0, It) + m_p) / τ)
```

其中：

```text
q0 = encode(Ir, T)
h  = repeated hard negative
It = ground-truth target
m_p = margin，随 repeat_count 增大
```

这个 loss 的意义非常明确：**只要同一张 hard negative 反复出现在 ground truth 前面，就直接惩罚 `s(q0, h) >= s(q0, It)`。**

然后 tri-query group 可以写成：

```text
L_group =
  L_rank(q0, It, H_pattern)
+ L_rank(q-, It, H_pattern)
+ β L_rank(q+, H_pattern, It)
+ γ L_align(q0, q-)
+ η L_sep(q0, q+)
```

其中：

```text
L_rank(q0, It, H_pattern):
  原始 query 必须把 ground truth 排到 repeated hard negatives 前面。

L_rank(q-, It, H_pattern):
  澄清 query 强化正确语义边界。

L_rank(q+, H_pattern, It):
  错误模式 query 把 repeated hard negatives 视为合法正样本。

L_align(q0, q-):
  让原始 query 和澄清 query 在局部语义上靠近。

L_sep(q0, q+):
  让原始 query 和错误模式 query 分开。
```

这里最重要的是第一项。你的文档里也写到，`q_0 = encode(I_r, T)` 必须直接进入 group 内排序约束，模型才会显式学习“原始 T 下，`I_t` 必须排在 repeated hard negatives 前面”。

我建议你在论文里把这件事提升成一个命名模块，例如：

```text
Original-query Anchored Pattern Recalibration
```

这样贡献点会更清楚：PE-ReCALL 不是“又生成了一些文本”，而是 **用 repeated error pattern 反向修复原始 query 的局部排序边界**。

---

## 3. 第二轮不能再给同一张错图生成同一种文本，要有“升级策略”

你文档里已经有 novelty check：新 corrective text 和历史文本相似度高于阈值时丢弃或重新生成，并检查新一轮的 `H_pattern / T_pattern^+ / T_pattern^-` 是否和历史 group 高度相似。 但我建议把它做得更强一点：**如果同一错误第二轮仍然出现，不要再普通生成 ReCALL-style `T_h^+`，而是进入 escalation mode。**

可以设计三种错误状态：

```text
state = new_error:
    普通 ReCALL 样本即可。
    生成 T_h^+，构造普通 corrective triplet。

state = repeated_once:
    进入 pattern-level correction。
    聚合同类 hard negatives，生成 T_pattern^+ 和 T_pattern^-。

state = repeated_twice_or_more:
    进入 rank-repair mode。
    不再主要依赖新文本，而是强制 q0 rank repair：
    q0 -> It, q0 !-> H_persistent。
```

也就是说，第三轮不要继续问 MLLM：“请再给这张错图生成一个描述”。这很可能只会生成和第一轮差不多的文本。第三轮应该问：

```text
Why did the previous correction fail to move the original query away from this hard negative?
What missing distinction must be directly attached to the original query?
```

这会把 MLLM 从“caption generator”变成“error boundary explainer”。

---

## 4. Hard negative mining 也要改：不要每轮只拿 deterministic top-k

你现在的问题是每轮检索出来的错图一样。除了模型没学会，还有一个原因是 mining 太贪心：每轮只看 top-k，天然会反复看到同一批最强吸引子。

我建议训练时的 mining 不要只用：

$$
H_i^k = \{\text{images ranked before } I_t\}
$$

而是分成四类：

```text
H_before:
  排在 It 前面的 top hard negatives。

H_persistent:
  和上一轮重复的 hard negatives。

H_pattern:
  与 persistent errors 属于同一语义簇的 negatives。

H_boundary:
  排在 It 附近的 boundary negatives，比如 It 前后各若干个。
```

尤其是 `H_boundary` 很重要。因为如果只训练最前面的同一张错图，模型可能只是把那一张压下去，但又会被附近另一张错图吸引。你的文档里也提到，如果 PER 降低但主指标不升，可能是模型避开旧错误后被新错误吸引，需要报告 newly emerged error rate 检查 error shifting。

所以我建议新增一个指标：

$$
\mathrm{NER}_k = \text{Newly Emerged Error Rate}

\mathrm{NER}_k
= \frac{\#\text{ new hard negatives before } I_t \text{ in round } k}
{\#\text{ hard negatives before } I_t \text{ in round } k}
$$

最终你希望看到的是：

```text
PER 下降，
NER 不显著上升，
OQRR 上升，
R@1/R@5 上升或至少不下降。
```

否则模型只是把旧错图换成了新错图。

---

## 5. Semantic pattern 不要完全交给 MLLM，先聚类再让 MLLM 总结

文档里把 semantic-level persistent error 定义为：不同 hard negatives 不是同一张图，但共享同一错误语义，例如把“局部眼周白斑”误解成“整体变白”。 这个定义很好，但实现时如果每个 query 都调用 MLLM 总结 pattern，成本和稳定性都会比较差。

我建议采用“两阶段 pattern discovery”：

```text
Stage A: embedding-based grouping
  对每个 hard negative 计算：
  d_h = image_emb(I_h) - image_emb(I_r)
  d_t = image_emb(I_t) - image_emb(I_r)

  然后用：
  sim(d_h, d_t)
  sim(image_emb(I_h), image_emb(I_t))
  sim(text_emb(T), caption_diff_h)
  做初步聚类。

Stage B: MLLM pattern reflection
  只对每个 cluster 的 representative samples 调 MLLM。
```

这样有几个好处：

```text
1. 降低 MLLM 调用次数；
2. 避免把语义不一致的 hard negatives 强行放进同一个 H_pattern；
3. 可以给每个 pattern 一个 confidence；
4. 低置信 pattern 不生成 tri-query group，只退化为 image-level rank repair。
```

这也能减少 pattern-positive text 的噪声。否则如果 `H_pattern` 里混了几种不同错因，`T_pattern^+` 会变成一个含糊文本，训练反而会污染 embedding space。

---

## 6. Tri-query group 还可以加一个“局部 listwise loss”，比普通 triplet 更适合排序修复

你现在文档里建议把 tri-query micro-group 展开成：

```text
(q0, positive = It, negatives = H_pattern)
(q-, positive = It, negatives = H_pattern)
(q+, positive = each Ih in H_pattern, negative = It)
```

这工程上确实容易实现。 但我建议再加一个局部 listwise objective，因为你的问题本质是 **排序列表里 It 和 H_pattern 的相对顺序**，不是单个 pair 的相似度。

例如对 `q0`：

$$
L_{\text{list}}(q_0)
= -\log \frac{\exp(s(q_0, I_t)/\tau)}
{\exp(s(q_0, I_t)/\tau) + \sum_h \exp(s(q_0, I_h)/\tau)}
$$

对 `q-` 同理。对 `q+` 可以用 multi-positive softmax：

$$
L_{\text{list}}(q_+)
= -\log \frac{\sum_h \exp(s(q_+, I_h)/\tau)}
{\sum_h \exp(s(q_+, I_h)/\tau) + \exp(s(q_+, I_t)/\tau)}
$$

这样模型会在一个局部候选集合 `{It} ∪ H_pattern` 内学习完整排序，比若干独立 triplet 更稳定。

---

## 7. 训练采样上要让 persistent groups 真正被看见

即使你构造了 PE-ReCALL group，如果训练时它在 batch 里出现频率太低，也不会明显改变第二轮排序。ReCALL 官方实现里配置默认 `max_iterations=2`、`hard_neg_top_k=10`、`hard_neg_per_query=5`、每轮训练若干 steps。([GitHub](https://github.com/RemRico/Recall/tree/main "GitHub - RemRico/Recall: A composed retrieval project · GitHub")) 如果你的 persistent groups 只占 augmented data 的很小部分，原始 InfoNCE 会把它淹没。

我建议采样比例明确写成：

```text
每个 iteration 的 batch 组成：

50% original CIR triplets
25% ordinary ReCALL corrective samples for new errors
25% PE-ReCALL persistent micro-groups
```

或者更简单：

```text
persistent group oversampling ratio = 2 or 3
```

同时 persistent group 的权重不要无限增长。你文档里建议：

$$
\text{weight} = 1 + \min(\text{repeat\_count}, \text{max\_repeat\_weight})

\text{max\_repeat\_weight} = 2
$$

这个是合理的。 我建议保留，但配合 sampling ratio 使用，而不是只依赖 loss weight。

---

## 8. 我会把你的方法改成这个版本

可以命名为：

```text
PE-ReCALL v2:
Persistent-Error Aware Original-Query Anchored Recalibration
```

核心流程：

```text
For round k:

1. Recompute embeddings with R_{k-1}
   不使用旧 target embedding cache。
   保存 gt_rank、top-k、similarity margin。

2. Mine hard negatives
   收集 before-GT negatives、persistent negatives、boundary negatives。

3. Update memory
   对每个 query 记录：
   repeated_images
   repeated_patterns
   newly_emerged_errors
   resolved_errors
   previous corrective texts
   previous local rank margins

4. Classify error state
   new_error:
      ordinary ReCALL correction

   repeated_once:
      pattern-positive + pattern-contrastive + tri-query group

   repeated_twice_or_more:
      original-query rank repair + stronger margin
      不再重复生成同质文本

5. Construct losses
   L = L_InfoNCE
     + λ1 L_tri_query_listwise
     + λ2 L_original_query_rank_repair
     + λ3 L_query_alignment
     + λ4 L_query_separation

6. Novelty and quality control
   文本重复则 regenerate；
   pattern 低置信则降级为 image-level repair；
   MLLM 必须说明 compared with previous correction 的新增区别。

7. Evaluate
   R@1 / R@5
   PER
   OQRR
   PRR
   NER
   rank margin improvement
```

这里最关键的是第 4 步：**多轮不是重复生成数据，而是根据错误是否重复来改变监督类型。**

---

## 9. 最小可行实验我建议这样做

你文档里已经给了最小实验：CIRR validation、2 rounds、Top-K hard negatives = 5，对比 ReCALL、Naive Iterative ReCALL、PE-ReCALL independent pattern triplets、PE-ReCALL tri-query full，并报告 R@1、R@5、PER、OQRR、CNS。 我建议在此基础上加两个 ablation，能更直接证明“多轮真的跑起来了”。

### 实验 A：排除工程伪影

```text
A1: Naive Iterative ReCALL with cache
A2: Naive Iterative ReCALL with forced re-encoding
A3: PE-ReCALL with forced re-encoding
```

如果 A1 和 A2 差异很大，说明你之前的重复错图部分来自 cache / stale retrieval。

### 实验 B：证明 q0 repair 是关键

```text
B1: only T_pattern^+ -> H_pattern
B2: T_pattern^+ + T_pattern^- independent triplets
B3: tri-query group without q0 rank repair
B4: tri-query group + q0 rank repair
```

你真正希望看到：

```text
B4 的 OQRR 明显高于 B1/B2/B3。
```

因为 OQRR 正是验证 original query 是否被修复的指标；你的文档也指出，如果只训练新文本而不训练 tri-query group，OQRR 可能不高。

---

## 10. 论文表述上，我建议你这样强调贡献

你现在的 PE-ReCALL 已经有五个模块：persistent error detection、semantic pattern reflection、pattern-level corrective text generation、tri-query micro-group training、PER-based stop criterion。 我建议最终写成三个更有冲击力的贡献：

**贡献 1：发现并量化 ReCALL 的多轮退化现象。**  
Naive Iterative ReCALL 并不必然持续提升，因为 repeated hard negatives 会在多轮中保持稳定，说明模型没有学到原始 query 与错误模式之间的语义边界。

**贡献 2：提出 persistent-error memory，把 repeated hard negatives 从“单个错图”提升为“错误模式”。**  
包括 image-level persistent error 和 semantic-level persistent error，并报告 PER / PRR / NER。

**贡献 3：提出 original-query anchored tri-query recalibration。**  
不仅生成 `T_pattern^+` 和 `T_pattern^-`，还强制 `q0 = encode(I_r, T)` 参与局部排序修复，从而提升 OQRR。

这样比单纯说“我们做了 multi-round ReCALL”更强，因为你的核心创新不是多轮，而是：

```text
当错误重复出现时，训练目标必须从“生成更多 corrective texts”
转向“直接修复原始 query 的局部排序边界”。
```

---

## 11. 我对你当前问题的直接建议

你现在马上可以做三件事：

第一，先排查 cache。每一轮强制重新 encode gallery，cache key 里加入 `iteration_id + model_checkpoint_hash`，并删除当前 round 的 hard negative json。否则你可能看到的是伪重复。

第二，在第二轮开始，只要发现同一 hard negative 再次出现在 `I_t` 前面，不要再只生成 `T_h^+`。直接加入：

```text
(q0, positive = It, negative = Ih)
```

并给它 persistent weight。这个是让多轮真正动起来的最小改动。

第三，把 repeated hard negatives 聚成 pattern 后，再构造完整 tri-query group：

```text
q0 -> It > H_pattern
q- -> It > H_pattern
q+ -> H_pattern > It
```

其中 `q0 -> It > H_pattern` 必须保留，否则第二轮还是可能只学会新文本，原始 `(I_r, T)` 继续检索同一张错图。

我认为你这篇工作最有潜力的点不是“ReCALL + memory”，而是可以把问题定义成一个更清楚的新现象：**persistent hard-negative attractors**。也就是，某些错误图像或错误语义模式在 embedding space 中会成为原始 composed query 的稳定吸引子。PE-ReCALL 的目标就是把这些 attractors 从普通 hard negatives 显式转化为可追踪、可解释、可加权、可排序修复的训练单元。

下面这版我建议你把它当成 **PE-ReCALL v2 / Persistent-Error Aware Original-Query Recalibration**。核心判断是：你现在遇到的“第二轮仍然检索同一张错图”不是简单的数据量问题，而是 **多轮之后监督没有改变原始 query `(I_r, T)` 对这些 persistent hard negatives 的局部排序边界**。你的文档已经把方向定得很准：显式追踪 repeated hard negatives / repeated patterns，并用包含原始 query、错误模式 query、澄清 query 的 tri-query micro-group 来让新监督直接作用到原始 `(I_r, T)`。

---

## 1. 先把问题重新定义得更“论文化”

我建议把现象命名为：

> **Persistent Hard-Negative Attractor**  
> 在多轮 ReCALL 迭代中，某些错误图像或错误语义模式持续位于 ground-truth target 前面，说明它们在当前 embedding space 中成为原始 composed query 的稳定吸引子。

这比“多轮没跑起来”更像一个可发表的问题定义。ReCALL 原论文的主线是 diagnose–generate–refine：先用 self-guided informative instance mining 找到排在 ground truth 前面的 informative instances，再由 MLLM 生成 minimal-edit corrective instruction，最后通过 grouped contrastive refinement 和 InfoNCE + in-group triplet margin loss 进行 targeted refinement。([arXiv](https://arxiv.org/html/2602.01639v2 "ReCALL: Recalibrating Capability Degradation for MLLM-based Composed Image Retrieval"))

但 ReCALL 的设计主要是一轮式 recalibration。它把 hard negative `I_h` 转化为另一个 query 下的 positive，即 `(I_r, T_h^+, I_h)`，这很聪明，因为它避免了“盲目把视觉相似图都推远”的问题；ReCALL 论文也明确指出，直接 hard-negative mining without textual refinement 会带来矛盾梯度并扭曲 learned manifold。([arXiv](https://arxiv.org/html/2602.01639v2 "ReCALL: Recalibrating Capability Degradation for MLLM-based Composed Image Retrieval"))

你的问题恰恰出现在这里：  
**一轮 ReCALL 能解释某个 hard negative 为什么在另一个文本下合理，但不保证第二轮原始 query `q_0 = encode(I_r, T)` 已经学会把 `I_t` 稳定排在这个 repeated hard negative 前面。**

所以 PE-ReCALL 的核心贡献应该不是“多跑几轮 ReCALL”，而是：

> 当同一错误图像或错误模式重复出现时，训练目标从 **positive-conversion** 升级为 **original-query anchored rank repair**。

---

## 2. 先排除一个很可能的工程坑：你可能在“伪多轮”

在正式改方法前，强烈建议你先做一个工程 sanity check。公开 ReCALL 仓库现在确实支持 iterative training，并描述了 progressive hard negative mining、多轮独立 optimizer/scheduler 等特性。([GitHub](https://github.com/RemRico/Recall/tree/main "GitHub - RemRico/Recall: A composed retrieval project · GitHub")) 但代码层面有两个地方非常容易导致“第二轮还是同一批错图”。

第一，target embedding cache 的文件名主要由 `target_database` 的 sorted hash 决定，cache validation 也主要检查 `target_paths` 和 embedding 数量，没有把当前 checkpoint hash / LoRA adapter hash / iteration id 纳入 cache key。也就是说，若你的实现沿用这个逻辑，第二轮可能复用第一轮 gallery embeddings。([GitHub](https://raw.githubusercontent.com/RemRico/Recall/main/src/retrieval/embedding_cache.py "raw.githubusercontent.com"))

第二，hard negative miner 在发现 `hard_negatives_iter_{iteration_round}.json` 已存在时会直接加载已有 hard negatives，而不是强制重新挖；如果你反复调试同一个 iteration id，可能看起来“训练了第二轮”，但 mining 文件根本没更新。([GitHub](https://raw.githubusercontent.com/RemRico/Recall/main/src/mining/hard_negative.py "raw.githubusercontent.com"))

因此你要先做这个检查：

```text
每一轮 mining 前强制记录：

model_checkpoint_hash
lora_adapter_hash
target_embedding_cache_hash
hard_negative_json_hash
gallery_embedding_mean/std
query_embedding_mean/std
top10 image ids
gt_rank
sim(q0, It)
sim(q0, Ih)
margin = sim(q0, It) - sim(q0, Ih)
```

并改 cache key：

```text
cache_key =
hash(
  sorted(target_paths)
  + model_checkpoint_hash
  + lora_adapter_hash
  + iteration_id
  + prompt_template_version
  + model_backbone
)
```

判断标准：

```text
如果第二轮 top-k 完全一样，
但 sim(q0, It), sim(q0, Ih), margin 几乎不变：
    大概率是 stale embedding / stale mining / checkpoint 没正确加载。

如果第二轮 top-k 一样，
但 margin 已明显变化：
    这是方法层面的 persistent attractor，值得继续做 PE-ReCALL。
```

---

## 3. 一个很关键的实现细节：不要直接把 tri-query 塞进 ReCALL 原 triplet loss

这是我认为最容易踩坑、也最值得你写进实验设计的地方。

ReCALL 原始 group 可以是：

```text
q_original   -> I_t
q_corrective -> I_h
```

一个 group 内两个 query 的 target 不同，因此 in-group triplet margin 可以把另一个 target 当作 hard negative。公开 trainer 里的 `_compute_intra_reference_triplet` 也是按同一 reference/group 内的 off-diagonal target 作为 negatives 来做 triplet。([GitHub](https://raw.githubusercontent.com/RemRico/Recall/main/src/trainer_iterative_.py "raw.githubusercontent.com"))

但你的 tri-query group 是：

```text
q_0 -> I_t
q_- -> I_t
q_+ -> H_pattern
```

这里 `q_0` 和 `q_-` 共享同一个 positive `I_t`。如果你简单展开成三行样本，然后沿用原来的 off-diagonal triplet loss，`q_0` 的 negative 里可能会包含 `q_-` 那一行的 `I_t`，这会制造自相矛盾的 triplet：

```text
q_0 的 positive = I_t
q_0 的 negative = 另一个 row 里的 I_t
```

这会导致 loss 无法正确表达你的设计意图。

所以 PE-ReCALL 不能只改 dataset；必须改 loss。推荐把 micro-group 表示成 **query set + image set + positive mask / negative mask**：

```text
Queries:
  q0 = encode(Ir, T)
  qm = encode(Ir, T_pattern^-)
  qp = encode(Ir, T_pattern^+)

Images:
  V = {It} ∪ H_pattern

Positive mask:
  q0: positive = {It}
  qm: positive = {It}
  qp: positive = H_pattern

Negative mask:
  q0: negatives = H_pattern
  qm: negatives = H_pattern
  qp: negatives = {It}
```

这比 triplet row 展开更安全，也天然支持 `q_+` 的 multi-positive。

---

## 4. Persistent error 的定义需要再细化成 5 类

你文档里已经定义了 image-level persistent error 和 semantic-level persistent error：同一 hard negative 连续多轮出现，或者不同 hard negatives 共享同一种错误语义。 我建议再细化成下面五类，这样分析和 ablation 会更强。

### 4.1 Image Persistent Error

```text
I_h ∈ H_i^{k-1} and I_h ∈ H_i^k
```

同一张错图连续出现。

### 4.2 Near-Duplicate Persistent Error

图像 id 不同，但视觉近重复，例如同一商品不同裁剪、同一场景不同帧。

判定方式：

```text
same cluster if:
  cosine(image_emb(h_a), image_emb(h_b)) > δ_img
or
  pHash distance < δ_hash
```

建议先设：

```text
δ_img = 0.90
```

### 4.3 Semantic Persistent Error

错图不同，但错误语义相同。例如都满足“颜色变浅”，但都忽略“less flowy”。你的文档里 FashionIQ 例子正适合这个定义：模型只捕捉 lighter color，却忽略 less flowy structure。

### 4.4 Margin Persistent Error

错图不一定仍在 top-k，但原始 query 的 margin 没有变好：

```text
Δ_i,h^k = s(q0^k, It) - s(q0^k, h)

margin_gain =
Δ_i,h^k - Δ_i,h^{k-1}
```

如果：

```text
margin_gain < ε
```

说明模型并没有真正学会边界，只是排序被别的样本扰动了。

建议：

```text
ε = 0.02 或按 similarity 标准差归一化后设 0.1
```

### 4.5 Error Shifting

PER 降了，但新错图又冒出来：

```text
H_i^{k-1} 中的旧错图消失，
但 H_i^k 中出现新的同模式错图。
```

这类特别重要，因为你可能看到“同一张错图没了”，但模型只是换了一张同类错图。你的文档里已经提醒：PER 降低但主指标不升，可能是模型避开旧错误后被新错误吸引，需要报告 newly emerged error rate。

---

## 5. Mining 不要再只取 deterministic top-k

ReCALL 原始 self-guided mining 会关注 failure cases，并隔离出排在 ground truth 前面的 top-k informative instances。([arXiv](https://arxiv.org/html/2602.01639v2 "ReCALL: Recalibrating Capability Degradation for MLLM-based Composed Image Retrieval")) 这在一轮 ReCALL 中合理，但多轮时如果每轮都 deterministic top-k，就会不断消费同一批 attractors。

我建议每轮 mining 生成 5 个集合：

```text
H_before:
  当前排在 It 前面的 hard negatives。

H_persistent:
  和上一轮重复的 image-level / near-duplicate / semantic-level errors。

H_boundary:
  It 附近的边界样本，例如 rank(It) 前后各 b 个。
  这些样本能防止模型只把最强错图压下去，却被相邻错图替代。

H_emerged:
  本轮新出现、上一轮没出现的错图。

H_diverse:
  从 top-M 中用 MMR 或聚类选出的多样化错图。
```

最终候选：

```text
C_i^k =
H_before ∪ H_persistent ∪ H_boundary ∪ H_emerged ∪ H_diverse
```

然后给每个候选打分：

```text
hardness(h) =
σ((s(q0, h) - s(q0, It)) / τ)

persistence(h) =
log(1 + repeat_count(h))

novelty(h) =
1 - max_sim(pattern(h), historical_patterns)

false_negative_risk(h) =
MLLM 或 reward model 判断 h 也满足原始 T 的概率

score(h) =
hardness(h)
* (1 + α * persistence(h))
* novelty(h)
* (1 - false_negative_risk(h))
```

这样做的原因是：hard negative 不是越硬越好。CIR 本身存在 false negative / alternative relevant image 的风险。QuRe 就指出，CIR 中只把 target 当 positive、batch 其他样本当 negatives 会引入 false negatives，并提出通过 hard negative sampling 减少这类问题。([arXiv](https://arxiv.org/abs/2507.12416 "[2507.12416] QuRe: Query-Relevant Retrieval through Hard Negative Sampling in Composed Image Retrieval")) FALCON 也从 VLP 角度指出，false negatives 会给 embedding space 引入冲突监督，并削弱 hard negative sampling 的有效性。([arXiv](https://arxiv.org/abs/2505.11192 "[2505.11192] FALCON: False-Negative Aware Learning of Contrastive Negatives in Vision-Language Alignment"))

所以 PE-ReCALL 要避免“把所有 repeated hard negatives 都粗暴推远”。正确策略是：

```text
如果 h 是明确错误：
    用 OQR loss 把 q0 下的 h 压到 It 后面。

如果 h 也是合理目标 / ambiguous：
    不作为 q0 的负样本；
    可以作为 weak positive 或从 OQR 中剔除。

如果 h 是同一错误模式：
    进入 pattern-level q+ multi-positive。
```

CIRCO 尤其适合验证这个问题，因为它是多 ground-truth CIR 数据集，目标就是缓解现有数据集中的 false-negative 问题，并用 mAP@K 评价多目标检索。([GitHub](https://github.com/miccunifi/CIRCO "GitHub - miccunifi/CIRCO: [ICCV 2023] - Composed Image Retrieval on Common Objects in context (CIRCO) dataset · GitHub"))

---

## 6. Pattern discovery 建议用“两阶段”，不要完全依赖 MLLM

你文档里目前是直接让 MLLM 总结 repeated error patterns。这个方向对，但为了稳定性，建议改成：

```text
Stage A: embedding/metadata pre-clustering
Stage B: MLLM reflection only for cluster representatives
```

### 6.1 Feature for clustering

对每个 query 和 hard negative 计算：

```text
z_r = image_emb(I_r)
z_t = image_emb(I_t)
z_h = image_emb(I_h)

d_t = normalize(z_t - z_r)
d_h = normalize(z_h - z_r)
```

其中：

```text
d_t 表示真实修改方向
d_h 表示错误图相对 reference 的视觉变化方向
```

聚类时使用：

```text
sim_delta = cos(d_h_a, d_h_b)
sim_target = cos(z_h, z_t)
sim_query = cos(text_emb(T), caption_diff_h)
```

一个 hard-negative cluster 可以先按：

```text
sim_delta > 0.60
or
sim_image > 0.80
```

粗分，再交给 MLLM 判断是否真的是同一错误模式。

### 6.2 Pattern confidence

每个 pattern 给一个置信度：

$$
C_{\mathrm{pattern}}
= 0.30 \cdot \text{visual\_cluster\_purity}
+ 0.25 \cdot \text{MLLM\_consistency}
+ 0.20 \cdot \text{VQA\_agreement}
+ 0.15 \cdot \text{support\_size\_score}
+ 0.10 \cdot \text{historical\_repeat\_score}
$$

建议保留：

$$
C_{\mathrm{pattern}} \ge 0.65
$$

低于阈值时不要生成 pattern-level group，退化为 image-level rank repair：

```text
(q0, positive = It, negative = Ih)
```

这样可以降低 MLLM pattern 总结错误带来的噪声。ReCALL 本身也使用 VQA-assisted quality control 来过滤生成信号，只保留高置信且内部一致的 triplets。([arXiv](https://arxiv.org/html/2602.01639v2 "ReCALL: Recalibrating Capability Degradation for MLLM-based Composed Image Retrieval"))

---

## 7. Loss 设计：核心是 OQR + Tri-query Masked Listwise

下面是我建议最终写进方法部分的训练目标。

记：

```text
q0 = encode(Ir, T)
qm = encode(Ir, T_pattern^-)
qp = encode(Ir, T_pattern^+)

v_t = encode_img(It)
V_H = {v_h | h ∈ H_pattern}

s(q, v) = cosine(q, v)
```

### 7.1 Global retrieval loss

保留原始 InfoNCE：

$$
L_{\mathrm{global}} = \mathrm{InfoNCE}(\text{batch})
$$

ReCALL 也是用 InfoNCE 保持全局结构，并叠加 in-group margin loss 做 targeted refinement。([arXiv](https://arxiv.org/html/2602.01639v2 "ReCALL: Recalibrating Capability Degradation for MLLM-based Composed Image Retrieval"))

### 7.2 Original-query Rank Repair Loss

这是 PE-ReCALL 最关键的新项：

```text
L_OQR =
Σ_{h ∈ H_persistent}
w_h * softplus((s(q0, v_h) - s(q0, v_t) + m_h) / τ)
```

其中：

```text
w_h = 1 + min(repeat_count(h), c_max)

m_h = m0 + α * min(repeat_count(h), c_max)
```

建议：

```text
m0 = 0.20
α = 0.05
c_max = 3
τ = 0.02 或沿用模型 InfoNCE temperature
```

这项的意义非常直接：

```text
只要同一张错图 / 同一错误模式反复排在 It 前面，
就显式惩罚 q0 下 s(q0, h) >= s(q0, It)。
```

这正是普通 ReCALL 多轮容易缺失的地方。

### 7.3 Tri-query Masked Listwise Loss

对 group 内图像集合：

$$
V_G = \{v_t\} \cup V_H
$$

原始 query：

$$
L_0
= -\log \frac{\exp(s(q_0, v_t)/\tau)}
{\exp(s(q_0, v_t)/\tau) + \sum_h \exp(s(q_0, v_h)/\tau)}
$$

澄清 query：

$$
L_m
= -\log \frac{\exp(s(q_m, v_t)/\tau)}
{\exp(s(q_m, v_t)/\tau) + \sum_h \exp(s(q_m, v_h)/\tau)}
$$

错误模式 query，多 positive：

$$
L_p
= -\log \frac{\sum_h \rho_h \exp(s(q_p, v_h)/\tau)}
{\sum_h \rho_h \exp(s(q_p, v_h)/\tau) + \exp(s(q_p, v_t)/\tau)}
$$

其中：

$$
\rho_h = \text{normalized pattern confidence / hardness weight}
$$

最终：

$$
L_{\mathrm{tri\_list}} = L_0 + L_m + L_p
$$

这比把 tri-query 展开成普通 triplets 更稳，因为它明确表达了：

```text
q0, qm 的唯一正样本是 It；
qp 的正样本是一组 H_pattern；
不会把另一个 It 行误当成负样本。
```

### 7.4 Query relation loss

让澄清 query 成为原始 query 的“局部语义老师”，但不要权重太大：

$$
L_{\mathrm{align}} = 1 - \cos(q_0, q_m)
$$

同时让错误模式 query 与原始 query 分开：

$$
L_{\mathrm{sep}}
= \max\left(0, \cos(q_0, q_p) - \cos(q_0, q_m) + m_q\right)
$$

建议：

```text
m_q = 0.1
λ_align 很小，例如 0.05
λ_sep 也小，例如 0.05
```

避免把原始文本过度拉向澄清文本，导致泛化下降。

### 7.5 总损失

```text
L_total =
L_global
+ λ_tri * L_tri_list
+ λ_oqr * L_OQR
+ λ_rel * (L_align + L_sep)
+ λ_replay * L_replay
```

建议起始权重：

```text
λ_tri = 1.0
λ_oqr = 0.5
λ_rel = 0.05
λ_replay = 0.1
```

如果你的 persistent errors 很顽固，可以把 `λ_oqr` 从 0.5 提到 1.0，但不要一开始就太大，因为 ReCALL 论文已经提示：盲目惩罚视觉相似 hard negatives 可能损害 broader metrics。([arXiv](https://arxiv.org/html/2602.01639v2 "ReCALL: Recalibrating Capability Degradation for MLLM-based Composed Image Retrieval"))

---

## 8. 多轮策略要做成状态机，而不是重复生成文本

我建议每个 query-pattern 维护状态：

```text
new_error
repeated_once
stubborn
resolved
ambiguous
```

### 8.1 new_error

首次出现：

```text
使用普通 ReCALL:
  生成 T_h^+
  构造 (Ir, T_h^+, Ih)
```

### 8.2 repeated_once

第二轮仍出现：

```text
进入 PE-ReCALL:
  生成 T_pattern^+
  生成 T_pattern^-
  构造 tri-query masked listwise group
  加 L_OQR
```

### 8.3 stubborn

第三轮仍出现，或者 margin 没改善：

```text
不要继续生成同质 T_h^+。
进入 rank-repair escalation:

1. 不再主要依赖新文本；
2. 提升 L_OQR 权重；
3. 扩展 H_pattern 到同簇 boundary negatives；
4. 要求 MLLM 解释 previous correction why failed；
5. 生成更强的 boundary clarification text。
```

给 MLLM 的问题要变成：

```text
The previous correction did not move the original query away from these hard negatives.
Identify the missing visual distinction that must be attached to the original query.
Do not repeat previous corrective texts.
```

### 8.4 resolved

如果：

```text
I_h 不再排在 It 前面
and
margin_gain > ε
```

进入 resolved memory。resolved 样本不要完全丢掉，而是低频 replay，防止下一轮回弹。

### 8.5 ambiguous

如果 MLLM/VQA 判断 hard negative 也满足原始 query：

```text
不要作为 q0 negative；
从 OQR 中剔除；
可作为 weak positive 或 pattern-positive。
```

这个状态能让你避免 false-negative 训练污染。

---

## 9. PE-ReCALL v2 算法伪代码

```text
Input:
  S = {(Ir_i, T_i, It_i)}
  gallery D
  foundation MLLM F
  retriever R_0
  max rounds K

Initialize:
  memory M_i = {}
  group pool G = {}

For k = 1 ... K:

  # 0. Engineering safety
  load checkpoint R_{k-1}
  invalidate target embedding cache unless checkpoint hash matches
  recompute gallery embeddings
  recompute original query embeddings

  # 1. Retrieval diagnosis
  For each query i:
    retrieve full ranking over D
    record:
      gt_rank_i^k
      topK_i^k
      sim(q0, It)
      sim(q0, h)
      margin_i,h^k

    build:
      H_before
      H_boundary
      H_emerged
      H_persistent

  # 2. Memory update
  compare H_i^k with M_i:
    detect image persistent errors
    detect near-duplicates
    detect semantic persistent patterns
    detect margin-persistent errors
    detect error shifting

  # 3. False-negative / ambiguity gate
  For each candidate h:
    ask F/VQA:
      does h satisfy the original query?
      is h a valid alternative target?
    if ambiguous:
      exclude from q0 negatives

  # 4. Pattern discovery
  cluster candidates by:
    image embedding
    modification direction embedding
    historical pattern label
  run MLLM reflection on representatives
  compute C_pattern

  # 5. Group construction
  For new errors:
    generate ReCALL-style T_h^+
    add ordinary ReCALL group

  For repeated patterns:
    generate:
      T_pattern^+
      T_pattern^-
    run novelty check
    construct masked tri-query group:
      queries = {T, T_pattern^-, T_pattern^+}
      images = {It} ∪ H_pattern
      masks = role-specific positive/negative masks

  For stubborn patterns:
    skip repetitive generation
    strengthen OQR
    add boundary negatives
    ask MLLM for failure-aware clarification

  # 6. Training
  train R_{k-1} -> R_k using:
    original triplets
    ordinary ReCALL groups
    PE-ReCALL groups
    resolved-error replay

  # 7. Evaluation
  compute:
    R@K / mAP@K
    PER_img
    PER_pattern
    OQRR
    PMG
    NER / ESR
    CNS
    duplicate group rate

  # 8. Stop
  stop if:
    validation retrieval metric plateaus
    and PER/OQRR no longer improve
```

---

## 10. 指标需要扩充，不然很难证明“多轮真的跑起来了”

你文档里已经有 PER、OQRR、PRR、CNS。 我建议增加下面几个，让审稿人更容易相信你的 claim。

### 10.1 Image Persistent Error Rate

```text
PER_img^k =
Σ_i |H_i^k ∩ H_i^{k-1}|
/
Σ_i |H_i^{k-1}|
```

### 10.2 Pattern Persistent Error Rate

$$
\mathrm{PER}_{\mathrm{pattern}}^k
= \frac{\#\text{ repeated semantic patterns in round } k}
{\#\text{ semantic patterns in round } k-1}
$$

### 10.3 Original Query Recovery Rate

你已有 OQRR，但建议定义得更严格：

```text
OQRR^k =
# persistent samples where all repeated hard negatives move behind It under q0
/
# persistent samples
```

也可以给一个宽松版：

```text
OQRR-margin^k =
# persistent pairs with margin gain > ε
/
# persistent pairs
```

### 10.4 Persistent Margin Gain

$$
\mathrm{PMG}^k
= \operatorname{mean}_{i,h \in P_i^k}
\left[
  \bigl(s_{k-1}(q_0, I_t) - s_{k-1}(q_0, I_h)\bigr)
  -
  \bigl(s_k(q_0, I_t) - s_k(q_0, I_h)\bigr)
\right]
$$

更直观地写成：

$$
\mathrm{PMG}^k = \operatorname{mean}(\text{margin}_{\text{after}} - \text{margin}_{\text{before}})
$$

目标：

$$
\mathrm{PMG} > 0
$$

这能证明即使 top-k 没完全变，局部边界也在移动。

### 10.5 Newly Emerged Error Rate

```text
NER^k =
Σ_i |H_i^k \ H_i^{k-1}|
/
Σ_i |H_i^k|
```

这个指标要和 PER 一起看：

```text
好情况:
  PER 下降，NER 不暴涨，R@K 上升。

坏情况:
  PER 下降，NER 暴涨，R@K 不升。
  说明只是 error shifting。
```

### 10.6 Duplicate Group Rate

$$
\mathrm{DGR}^k
= \frac{\#\text{ generated groups whose text/pattern similarity} > \text{threshold}}
{\#\text{ generated groups}}
$$

目标：

```text
PE-ReCALL with novelty check 的 DGR 更低。
```

### 10.7 False-Negative Exclusion Rate

$$
\mathrm{FNER}^k
= \frac{\#\text{ candidate hard negatives excluded as ambiguous / valid alternatives}}
{\#\text{ candidate hard negatives}}
$$

这能回应 hard negative 相关文献中的 false-negative 风险。NV-Retriever 也强调 hard-negative mining 中需要以 positive relevance score 作为 anchor 来去除 false negatives。([arXiv](https://arxiv.org/abs/2407.15831 "[2407.15831] NV-Retriever: Improving text embedding models with effective hard-negative mining"))

---

## 11. 实验矩阵建议

你的文档中已有最小实验：CIRR validation、2 rounds、top-k hard negatives = 5，对比 ReCALL、Naive Iterative ReCALL、PE-ReCALL independent pattern triplets、PE-ReCALL tri-query full，并报告 R@1、R@5、PER、OQRR、CNS。 我建议扩展成三层实验。

### 11.1 工程 sanity 实验

|方法|目的|
|---|---|
|Iter-ReCALL-cache-on|复现你当前问题|
|Iter-ReCALL-cache-off|排除 stale gallery embeddings|
|Iter-ReCALL-cache-off + re-mine|排除 hard-negative json 复用|
|Iter-ReCALL-cache-off + checkpoint-hash cache|最干净的 naive iterative baseline|

必须报告：

```text
Top10 Overlap between rounds
GT Rank Change
Margin Change
Hard-negative JSON Hash Change
Embedding Cache Hash Change
```

如果 cache-off 后问题明显缓解，论文里要把工程问题和方法问题分开。

### 11.2 方法主实验

|方法|说明|
|---|---|
|R_base|只训练原始 CIR triplets|
|ReCALL-1R|原始一轮 ReCALL|
|Naive Iterative ReCALL|多轮 ReCALL，无 memory、无 novelty|
|ReCALL + More Data|控制生成数据量|
|ReCALL + Diverse Mining|只改 mining，不做 PE memory|
|PE-ReCALL independent|只加入 `T_pattern^+ / T_pattern^-` 独立 triplets|
|PE-ReCALL tri-query triplet|错误实现风险版本，可选|
|PE-ReCALL masked-listwise|推荐主方法|
|PE-ReCALL full|masked-listwise + OQR + novelty + false-negative gate|

关键比较：

```text
PE-ReCALL full > Naive Iterative ReCALL:
  证明不是多跑几轮。

PE-ReCALL full > ReCALL + More Data:
  证明不是数据量。

PE-ReCALL full > PE independent:
  证明 tri-query / q0 anchoring 有效。

PE-ReCALL masked-listwise > tri-query triplet:
  证明需要 mask loss，不能直接复用原 triplet。
```

### 11.3 消融实验

|消融|证明什么|
|---|---|
|w/o OQR|原始 query 排序修复是否关键|
|w/o q0 in group|是否只是学会新文本|
|w/o `T_pattern^-`|澄清文本是否帮助边界|
|w/o `T_pattern^+`|错误模式 positive conversion 是否必要|
|w/o novelty|多轮是否生成重复监督|
|w/o false-negative gate|是否存在错误推远 valid alternatives|
|w/o semantic memory|只追踪同图是否不够|
|w/o boundary negatives|是否发生 error shifting|
|triplet loss instead of masked listwise|验证实现层核心改动|

---

## 12. 数据集顺序

建议顺序：

```text
1. CIRR
2. FashionIQ
3. CIRCO
```

理由：

CIRR 是 real-life open-domain composed image retrieval 数据集，适合先观察 generic hard negative 和局部混淆。([GitHub](https://github.com/Cuberick-Orion/CIRR "GitHub - Cuberick-Orion/CIRR: Official repository of ICCV 2021 - Image Retrieval on Real-life Images with Pre-trained Vision-and-Language Models · GitHub")) FashionIQ 是 natural language based interactive fashion image retrieval 数据集，适合观察颜色、版型、袖长、松紧等属性级 persistent errors。([GitHub](https://github.com/XiaoxiaoGuo/fashion-iq "GitHub - XiaoxiaoGuo/fashion-iq · GitHub")) CIRCO 是多 ground-truth 的 CIR benchmark，适合分析 false negatives、ambiguous positives 和 mAP@K。([GitHub](https://github.com/miccunifi/CIRCO "GitHub - miccunifi/CIRCO: [ICCV 2023] - Composed Image Retrieval on Common Objects in context (CIRCO) dataset · GitHub"))

最小可行顺序：

```text
Phase 1:
  CIRR val
  rounds = 2
  topK = 5
  只做 image-level persistent + OQR + masked tri-query

Phase 2:
  FashionIQ val
  rounds = 2-3
  加 semantic-level pattern

Phase 3:
  CIRCO val
  验证 false-negative gate 和 multi-positive q+
```

---

## 13. Prompt 也需要升级成 failure-aware

你文档已有三个 prompt：Semantic Pattern Reflection、Pattern-positive Generation、Pattern-contrastive Generation。 我建议改成 JSON 输出，并加入 false-negative 判断和 novelty 约束。

### 13.1 Reflection Prompt

```text
You are analyzing persistent retrieval errors in composed image retrieval.

Inputs:
- Reference image
- Original modification text
- Ground-truth target image
- Current hard negative images ranked before the ground truth
- Historical hard negatives
- Historical corrective texts
- Historical error patterns
- Previous round margins and ranks

Tasks:
1. Decide whether each hard negative is:
   A. true error
   B. valid alternative target / false negative
   C. ambiguous
2. Identify satisfied constraints:
   Which parts of the original query are satisfied by the hard negatives?
3. Identify missing constraints:
   Which required visual details in the ground truth are missing?
4. Identify persistent pattern:
   Is this the same error pattern as previous rounds?
5. Explain why previous correction failed:
   What distinction was still not learned by the original query?
6. Propose new correction axis:
   What new visual boundary should be emphasized?

Return JSON:
{
  "false_negative_risk": 0.0-1.0,
  "satisfied_constraints": [...],
  "missing_constraints": [...],
  "persistent_pattern": "...",
  "is_repeated_pattern": true/false,
  "why_previous_failed": "...",
  "new_correction_axis": "...",
  "confidence": 0.0-1.0
}
```

### 13.2 Pattern-positive Prompt

```text
Generate a concise CIR modification text that makes the repeated hard negatives valid targets.

Requirements:
- Describe the shared visual change of the hard negatives.
- Do not copy previous corrective texts.
- Do not describe the ground-truth-only detail.
- The text should make H_pattern positive under the new query.
- Keep it natural and short.

Return JSON:
{
  "T_pattern_positive": "...",
  "difference_from_previous": "...",
  "covered_hard_negative_attributes": [...]
}
```

### 13.3 Pattern-contrastive Prompt

```text
Generate a clarified modification text that still targets the original ground-truth image
and explicitly excludes the repeated hard-negative pattern.

Requirements:
- Preserve the original user intent.
- Emphasize the visual attribute present in the ground truth but missing in H_pattern.
- Explicitly avoid the repeated wrong interpretation.
- Do not become overly long.
- This text is not a replacement for the original query; it is used as a semantic teacher.

Return JSON:
{
  "T_pattern_contrastive": "...",
  "ground_truth_specific_details": [...],
  "excluded_wrong_pattern": "...",
  "difference_from_previous": "..."
}
```

### 13.4 Stubborn Error Prompt

```text
The previous correction did not fix the original query ranking.
The same hard negative or same error pattern is still ranked before the ground truth.

Analyze:
1. Which previous corrective text was redundant?
2. Which distinction did not transfer to the original query?
3. What boundary should be optimized directly?
4. Should this hard negative be treated as:
   - true negative under original query
   - alternative positive
   - pattern-positive under another query

Return JSON only.
```

---

## 14. 最小实现路线

为了最快验证，我建议你不要一开始做全量 semantic clustering。先做一个最小但能打中问题的版本。

### Step 1：修工程

```text
- cache key 加 checkpoint hash
- 每轮强制重新 encode gallery
- hard negatives json 加 checkpoint hash
- mining 前检查 checkpoint 是否为上一轮训练结果
- log top-k overlap 和 margin change
```

### Step 2：只做 image-level persistent

```text
P_i^k = H_i^k ∩ H_i^{k-1}
```

对这些重复错图加入：

$$
L_{\mathrm{OQR}}(q_0, I_t, P_i^k)
$$

不用 MLLM、不用 pattern，先看 OQRR 是否上升。

### Step 3：加 masked tri-query

对 repeated images 生成：

```text
T_pattern^+
T_pattern^-
```

然后用 mask loss：

```text
q0 -> It > H
qm -> It > H
qp -> H > It
```

### Step 4：加 semantic-level pattern

把不同图但同类错因聚成 H_pattern。

### Step 5：加 novelty 和 false-negative gate

这一步主要提升稳定性和可解释性。

---

## 15. 你现在最应该做的 3 个实验

### 实验 1：是不是 stale cache / stale mining

```text
A1: 当前 naive iterative
A2: 禁用 gallery embedding cache
A3: 禁用 cache + 删除 hard_negative json + checkpoint hash 命名
```

报告：

```text
Top10 overlap
GT rank
margin
R@1 / R@5
```

### 实验 2：OQR 是否能让第二轮动起来

```text
B1: Naive ReCALL round 2
B2: Naive ReCALL + image-level OQR
B3: Naive ReCALL + image-level OQR + boundary negatives
```

如果 B2 的 OQRR 明显高，说明你的瓶颈就是原始 query 没被修复。

### 实验 3：masked tri-query 是否优于 independent triplets

```text
C1: T_pattern^+ / T_pattern^- independent triplets
C2: tri-query but ordinary triplet loss
C3: tri-query masked listwise
C4: tri-query masked listwise + OQR
```

预期最关键：

```text
C4 > C3 > C1
```

如果 C2 很差，可能正好证明我上面说的 duplicate `I_t` target 会污染原始 triplet loss。

---

## 16. 最后给你一个可以写进论文的方法摘要

可以这样写：

> We identify a persistent hard-negative attractor phenomenon in iterative ReCALL: after one round of corrective triplet generation, some hard negatives or semantic error patterns remain ranked above the ground-truth target under the original composed query. To address this, we propose PE-ReCALL, a persistent-error aware iterative recalibration framework. PE-ReCALL maintains a query-level error memory, detects repeated image-level and semantic-level hard negatives, and converts them into pattern-aware micro-groups. Unlike standard ReCALL, which mainly transforms each hard negative into a positive under a newly edited instruction, PE-ReCALL directly anchors the original query in the local ranking objective. Specifically, we introduce a masked tri-query listwise loss over the original query, a pattern-positive query, and a pattern-contrastive query, together with an original-query rank repair loss that explicitly pushes persistent hard negatives behind the ground truth under the original query. This design ensures that each iteration provides non-redundant supervision and directly repairs the ranking boundary that failed in previous rounds.

这版的中心创新非常清楚：

```text
ReCALL:
  hard negative -> another query's positive

PE-ReCALL:
  repeated hard negative -> persistent pattern memory
  persistent pattern -> tri-query masked group
  tri-query group -> original-query rank repair
```

也就是一句话：

> **多轮 ReCALL 不能只是不断生成新的 corrective text；当错误重复出现时，必须直接修复原始 query 与 repeated error pattern 之间的局部排序边界。**


下面我把改进建议整理成一个**可以直接执行的版本**。整体目标是：把你的 PE-ReCALL 从“多轮 ReCALL 增强”进一步收紧成一个更有 taste 的研究问题：

> **当 hard negatives 或错误语义模式跨轮反复出现时，不能继续把它们当作新的独立样本处理，而应该显式建模 persistent error，并直接修复原始 query `(I_r, T)` 与这些错误模式之间的局部排序边界。**

---

# 0. 总体方向：你现在最应该强化的主线

你文档里的核心思路已经很清楚：显式追踪多轮中重复出现的 hard negatives，把它们抽象成 persistent error patterns，并用包含原始 query、错误模式 query、澄清 query 的 tri-query micro-group 来训练，使新监督直接影响原始 `(I_r, T)` 的检索。

我建议你把论文主线压成一句话：

> **PE-ReCALL studies persistent hard-negative attractors in iterative ReCALL and repairs them through original-query anchored tri-query recalibration.**

中文可以写成：

> **PE-ReCALL 研究多轮 ReCALL 中持续存在的 hard-negative attractors，并通过原始 query 锚定的 tri-query 局部排序修复来消除这些错误吸引子。**

这比“我们做了 ReCALL 的多轮版本”强很多。

---

# 1. 叙事层面：不要把 novelty 写成“多轮 ReCALL”

这是第一优先级。

ReCALL 官方实现已经明确支持 iterative training，包括 progressive hard negative mining across multiple training rounds；README 还写了每轮流程：encode training images & queries、mine hard negatives、generate augmented captions、train on original + augmented data、save checkpoint。([GitHub](https://github.com/RemRico/Recall/tree/main "GitHub - RemRico/Recall: A composed retrieval project · GitHub"))

所以你不能把贡献写成：

> We extend ReCALL to multiple iterations.

这个 novelty 会很弱。

你应该写成：

> Existing iterative ReCALL mines hard negatives in each round but treats them as fresh independent errors. We instead ask whether the same images or semantic error patterns persist across rounds, and convert these persistent errors into original-query anchored local ranking repair groups.

中文：

> 现有 iterative ReCALL 每轮都重新挖 hard negatives，但默认这些错误是独立的新错误；PE-ReCALL 显式追踪哪些错误跨轮持续存在，并把这些 repeated errors 转化为针对原始 query 的局部排序修复监督。

这会让你的工作从 **engineering extension** 变成 **phenomenon-driven research**。

---

# 2. 方法层面最重要的改进：把 tri-query 从“数据展开”升级成 masked local ranking loss

你现在文档里说 tri-query 可以展开成：

```text
(q_0, positive = I_t, negatives = H_pattern)
(q_-, positive = I_t, negatives = H_pattern)
(q_+, positive = each I_h in H_pattern, negative = I_t)
```

这个设计逻辑是对的，因为你想让：

```text
q_0 -> I_t > H_pattern
q_- -> I_t > H_pattern
q_+ -> H_pattern > I_t
```

也就是原始 query 和澄清 query 指向 ground truth，错误模式 query 指向 repeated hard negatives。

但实现上不要简单复用 ReCALL 原来的 group triplet。风险是：`q_0` 和 `q_-` 都对应同一个 positive `I_t`，如果你把它们展开成多行样本，再用 group 内 off-diagonal target 当 negative，可能会把同一个 `I_t` 同时当成正样本和负样本。

我建议你改成 **masked local listwise loss**。

定义 group：

```text
Queries:
  q_0 = encode(I_r, T)
  q_- = encode(I_r, T_pattern^-)
  q_+ = encode(I_r, T_pattern^+)

Images:
  V_G = {I_t} ∪ H_pattern
```

正负 mask：

```text
q_0:
  positive = {I_t}
  negatives = H_pattern

q_-:
  positive = {I_t}
  negatives = H_pattern

q_+:
  positives = H_pattern
  negative = {I_t}
```

对应 loss：

```text
L(q_0) =
-log exp(s(q_0, I_t) / τ)
/
[exp(s(q_0, I_t) / τ) + Σ_h exp(s(q_0, I_h) / τ)]

L(q_-) 同理

L(q_+) =
-log Σ_h exp(s(q_+, I_h) / τ)
/
[Σ_h exp(s(q_+, I_h) / τ) + exp(s(q_+, I_t) / τ)]
```

这样你的方法就不只是“构造新 triplets”，而是一个清楚的新训练目标：

> **masked tri-query local ranking objective**

这会显著提升方法 taste。

---

# 3. 再加一个核心 loss：Original Query Rank Repair Loss

我建议你给原始 query 单独加一个明确的排序修复项：

```text
L_OQR =
Σ_{h ∈ H_persistent}
softplus((s(q_0, h) - s(q_0, I_t) + m) / τ)
```

含义很直接：

```text
只要某个 hard negative h 跨轮持续出现在 I_t 前面，
就显式惩罚原始 query q_0 下：
s(q_0, h) >= s(q_0, I_t)
```

这个 loss 能把你的核心 claim 落到数学形式上：

> PE-ReCALL 不只是生成更多 corrective texts，而是直接修复原始 query 对 persistent errors 的局部排序边界。

建议初始设置：

```text
m = 0.2
τ = 沿用原模型 temperature
λ_OQR = 0.5
```

如果 persistent error 很顽固，再试：

```text
λ_OQR = 1.0
```

但不要一开始过大，否则可能把局部 hard negatives 推得太狠，破坏全局 embedding structure。

---

# 4. 工程层面必须先排查：确认不是“伪多轮”

这个非常重要。

你要先排除 stale cache / stale mining 的可能。公开代码里 target embedding cache 的文件名由 target database 的排序路径 hash 生成，cache validation 主要检查 `target_paths` 和 embedding 数量，而不是 checkpoint hash 或 iteration id。([GitHub](https://raw.githubusercontent.com/RemRico/Recall/main/src/retrieval/embedding_cache.py "raw.githubusercontent.com"))

同时 hard negative miner 如果发现 `hard_negatives_iter_{iteration_round}.json` 已存在，会直接加载已有 hard negatives。([GitHub](https://raw.githubusercontent.com/RemRico/Recall/main/src/mining/hard_negative.py "raw.githubusercontent.com"))

所以你要做一个 sanity check，否则你看到的 repeated hard negatives 可能不是方法问题，而是旧 gallery embedding 或旧 mining 文件导致的。

每轮 mining 前记录：

```text
model_checkpoint_hash
lora_adapter_hash
target_embedding_cache_hash
hard_negative_json_hash
gallery_embedding_mean/std
query_embedding_mean/std
top10 image ids
gt_rank
sim(q_0, I_t)
sim(q_0, I_h)
margin = sim(q_0, I_t) - sim(q_0, I_h)
```

并且修改 cache key：

```text
cache_key =
hash(
  sorted(target_paths)
  + model_checkpoint_hash
  + lora_adapter_hash
  + iteration_id
  + model_backbone
  + prompt_template_version
)
```

最小 sanity 实验：

```text
A1: Naive Iterative ReCALL with cache
A2: Naive Iterative ReCALL with forced re-encoding
A3: Naive Iterative ReCALL with forced re-encoding + forced re-mining
```

观察：

```text
top-k overlap
gt rank change
persistent margin change
R@1 / R@5
```

如果 cache-off 后 repeated errors 明显减少，那说明之前主要是工程伪影；如果 cache-off 后仍然存在 repeated errors，那才是 PE-ReCALL 真正要解决的 scientific phenomenon。

---

# 5. Persistent error 定义建议进一步细化

你文档目前定义了两类：

```text
image-level persistent error:
同一 hard negative 连续多轮出现。

semantic-level persistent error:
不同 hard negatives 共享同一种错误语义。
```

这个已经很好。

我建议你进一步扩成 5 类，方便做 error analysis。

## 5.1 Image Persistent Error

```text
I_h ∈ H_i^{k-1}
and
I_h ∈ H_i^k
```

同一张图连续出现。

## 5.2 Near-Duplicate Persistent Error

图像 id 不同，但视觉上几乎一样，比如同一商品不同 crop、同一场景不同帧。

可以用：

```text
cos(image_emb(h_a), image_emb(h_b)) > 0.90
```

或者 pHash 判断。

## 5.3 Semantic Persistent Error

不同图像，但错误模式一致。

比如你文档里的例子：

```text
原始 query:
give the bird a white patch around its eye

重复错误:
overall whiter body / white face / pale bird
```

模型不是反复选同一张图，而是反复把“局部眼周白斑”误解成“整体变白”。

## 5.4 Margin Persistent Error

这类很关键。

有时 hard negative 不再出现在 top-k，但原始 query 对它的 margin 没有真正改善。

定义：

```text
margin_k = s_k(q_0, I_t) - s_k(q_0, I_h)

margin_gain = margin_k - margin_{k-1}
```

如果：

```text
margin_gain < ε
```

说明边界没有被真正修复。

## 5.5 Error Shifting

PER 降低了，但新错误冒出来。

比如：

```text
旧的整体变白鸟不再出现，
但另一张白脸鸟又排到了 I_t 前面。
```

这说明模型只是绕开旧错图，而没有学会语义边界。你文档里也提到 PER 降低但主指标不升时，要警惕模型被新错误吸引。

---

# 6. 指标建议：保留你的指标，再加两个关键指标

你已有指标：

```text
PER: Persistent Error Rate
OQRR: Original Query Recovery Rate
PRR: Pattern Repetition Rate
CNS: Corrective Novelty Score
```

这些都和你的 claim 对齐。

我建议新增两个。

## 6.1 PMG：Persistent Margin Gain

```text
PMG =
mean [
  (s_after(q_0, I_t) - s_after(q_0, I_h))
  -
  (s_before(q_0, I_t) - s_before(q_0, I_h))
]
```

目标：

```text
PMG > 0
```

它回答：

> 即使 rank 没完全改变，原始 query 对 persistent hard negative 的局部 margin 是否真的变大了？

OQRR 是 rank-level，PMG 是 similarity-level。两个一起报更有说服力。

## 6.2 NER：Newly Emerged Error Rate

```text
NER_k =
# 本轮新出现且排在 I_t 前面的 hard negatives
/
# 本轮所有 hard negatives
```

你需要和 PER 一起看：

```text
好情况:
PER 下降，NER 不明显上升，R@K 上升。

坏情况:
PER 下降，NER 暴涨，R@K 不升。
```

坏情况说明方法只是把旧错误换成了新错误。

---

# 7. Semantic pattern 不要完全交给 MLLM

你文档里目前主要让 MLLM 做 Semantic Pattern Reflection，总结 hard negatives 共享的错误模式。

这个方向是对的，但如果完全依赖 MLLM，reviewer 会质疑：

```text
pattern 是否稳定？
prompt 换一下会不会变？
MLLM 总结错了怎么办？
```

我建议改成两阶段：

## Stage A：先做 embedding / attribute 粗聚类

对每个 query 计算：

```text
z_r = image_emb(I_r)
z_t = image_emb(I_t)
z_h = image_emb(I_h)

d_t = normalize(z_t - z_r)
d_h = normalize(z_h - z_r)
```

其中：

```text
d_t = ground-truth 修改方向
d_h = hard negative 相对 reference 的变化方向
```

聚类特征可以用：

```text
cos(d_h_a, d_h_b)
cos(z_h, z_t)
caption-diff similarity
attribute type: color / shape / texture / locality / count / background / pose
```

## Stage B：MLLM 只解释 cluster representative

也就是说，MLLM 不负责“凭空分组”，而是负责解释：

```text
这一组 hard negatives 共享什么错误语义？
它们满足了原 query 的哪部分？
它们缺失了 ground truth 的哪部分？
```

这样 semantic-level persistent error 会更客观、更稳。

---

# 8. Novelty constraint 建议保留，但不要当主贡献

你文档里设计了文本去重、group 去重、语义新颖性检查，比如新文本和历史文本相似度超过阈值就丢弃或重新生成。

这个模块是有用的，但不要把它写成主贡献。它更像是一个 training hygiene / quality control 机制。

建议定位：

```text
主贡献:
persistent error memory + original-query anchored tri-query recalibration

辅助机制:
novelty check + VQA quality filtering
```

否则论文会显得 pipeline-heavy。

你可以保留：

```text
text_sim threshold = 0.85
```

但最好报告一个简单指标：

```text
Duplicate Group Rate =
# 与历史 group 高度相似的新 group
/
# 所有新生成 group
```

这样 novelty check 有可量化结果。

---

# 9. 多轮策略建议改成状态机，而不是每轮重复生成文本

你现在的流程是：

```text
每轮发现 hard negatives
和 memory 对比
生成 T_pattern^+ / T_pattern^-
构造 tri-query group
继续训练
```

我建议进一步改成 error state machine。

## 9.1 new_error

第一次出现：

```text
使用普通 ReCALL-style correction:
生成 T_h^+
构造普通 corrective triplet
```

## 9.2 repeated_once

第二次出现：

```text
进入 PE-ReCALL:
聚合 repeated images / repeated pattern
生成 T_pattern^+
生成 T_pattern^-
构造 tri-query masked group
加入 L_OQR
```

## 9.3 stubborn

第三次仍然出现，或者 PMG 没有改善：

```text
不要继续生成类似 T_h^+
升级为 rank-repair mode
提高 L_OQR 权重
加入 boundary negatives
要求 MLLM 分析 previous correction why failed
```

这一步很有 taste，因为它体现了：

> 多轮不是重复生成数据，而是根据错误是否持续存在来改变监督类型。

## 9.4 resolved

如果错误不再排在 `I_t` 前面，且 margin 提升：

```text
进入 resolved memory
低频 replay，防止下一轮回弹
```

## 9.5 ambiguous

如果 hard negative 可能也是合理目标：

```text
不要把它作为 q_0 的负样本
从 OQR 中剔除
可以作为 weak positive 或跳过
```

这能减少 false negative 训练污染。

---

# 10. Mining 不要只取 deterministic top-k

如果每轮只取 top-k hard negatives，模型很容易反复看到同一批最强 attractors。你可以把候选 hard negatives 分成几类：

```text
H_before:
当前排在 I_t 前面的 negatives

H_persistent:
跨轮重复出现的 negatives

H_boundary:
I_t 附近的边界样本，比如 rank(I_t) 前后各 b 个

H_emerged:
本轮新出现的错误

H_diverse:
top-M 中经过聚类或 MMR 选出的多样化错误
```

然后训练时不要只修最强错图，也要修边界附近的错图。

这样可以减少：

```text
旧错图被压下去，
但另一张同类新错图又冒出来
```

也就是 error shifting。

---

# 11. 实验优先级：不要一开始做完整大系统

我建议你分三步做 MVP。

## Step 1：先证明 persistent error 现象真实存在

跑：

```text
R_0
ReCALL round 1
Naive Iterative ReCALL round 2
Naive Iterative ReCALL round 3
```

报告：

```text
R@1 / R@5
PER
Top-k overlap
OQRR
PMG
NER
```

核心问题：

> Naive Iterative ReCALL 是否真的会出现收益递减和 repeated hard negatives？

如果这个现象不明显，PE-ReCALL 的动机会变弱。

---

## Step 2：只做 image-level PE-ReCALL

先不要做 semantic pattern。

定义：

```text
H_persistent = H_i^k ∩ H_i^{k-1}
```

构造：

```text
q_0 -> I_t > H_persistent
q_- -> I_t > H_persistent
q_+ -> H_persistent > I_t
```

先验证：

```text
OQRR 是否提升？
PMG 是否提升？
PER 是否下降？
R@1/R@5 是否不下降？
```

这一步工程成本最低，最容易验证核心机制。

---

## Step 3：再扩展到 semantic-level pattern

当 image-level 版本证明有效后，再加入：

```text
embedding clustering
MLLM reflection
T_pattern^+
T_pattern^-
novelty check
pattern-level memory
```

这样风险更低。

---

# 12. 最关键的 baseline 和 ablation

你文档里的 baseline 已经比较完整，包括 `R_base`、`ReCALL`、`Naive Iterative ReCALL`、`ReCALL + More Data`、`PE-ReCALL without tri-query`、`PE-ReCALL without pattern-contrastive`、`PE-ReCALL without novelty check`、完整 PE-ReCALL。

我建议你最终主表至少保留这些：

|方法|目的|
|---|---|
|`R_base`|基础检索器|
|`ReCALL-1R`|原始一轮 ReCALL|
|`Naive Iterative ReCALL`|证明多轮 baseline|
|`ReCALL + More Data`|控制数据量|
|`PE-ReCALL independent triplets`|只有新文本，无 q0 anchored group|
|`PE-ReCALL w/o q0`|验证原始 query 是否关键|
|`PE-ReCALL triplet version`|直接展开 triplet 的版本|
|`PE-ReCALL masked-listwise`|推荐主方法|
|`PE-ReCALL full`|masked-listwise + OQR + novelty + quality filtering|

最关键的比较是：

```text
PE-ReCALL full > Naive Iterative ReCALL
```

证明不是“多跑几轮”。

```text
PE-ReCALL full > ReCALL + More Data
```

证明不是“多加数据”。

```text
PE-ReCALL full > PE-ReCALL independent triplets
```

证明不是“新文本本身有用”，而是 tri-query group 有用。

```text
PE-ReCALL masked-listwise > PE-ReCALL triplet version
```

证明你的 masked objective 不是实现细节，而是必要设计。

---

# 13. 论文贡献建议最终写成三点

不要写五六个模块。建议最终贡献写成三点。

## Contribution 1：Persistent error phenomenon

> We identify and quantify persistent hard-negative errors in iterative MLLM-based composed image retrieval.

对应指标：

```text
PER
PRR
top-k overlap
PMG
NER
```

## Contribution 2：Persistent error memory

> We maintain a query-level memory to track repeated images, repeated semantic patterns, resolved errors, newly emerged errors, and historical corrective groups.

这对应你文档里的 memory 设计，包括 repeated_images、repeated_patterns、resolved_errors、newly_emerged_errors、historical_corrective_texts 和 historical_micro_groups。

## Contribution 3：Original-query anchored tri-query recalibration

> We propose a masked tri-query local ranking objective that directly repairs the original query’s boundary against persistent error patterns.

这是你真正的方法核心。

---

# 14. 你现在文档里哪些东西应该降级为 appendix / implementation details

我建议这样处理：

|内容|建议定位|
|---|---|
|Persistent Error Detection|主方法|
|Semantic Pattern Reflection|主方法，但第二阶段再上|
|Pattern-level Corrective Text Generation|主方法|
|Tri-query Micro-group|核心主方法|
|PER-based Stop Criterion|appendix / training detail|
|Novelty Check|quality control / implementation detail|
|Persistent-error Weighting|ablation / implementation detail|
|CNS|辅助指标|
|Prompt 模板|appendix|
|Case study|主文保留 2 个，更多放 appendix|

这样论文不会显得太散。

---

# 15. 我建议你下一版文档直接改成这个结构

```text
1. Motivation
   - ReCALL solves one-round capability degradation
   - But iterative recalibration may suffer persistent hard-negative attractors

2. Observation
   - Define image-level PER
   - Define semantic-level PER
   - Show naive iterative ReCALL has repeated errors

3. Method
   3.1 Persistent Error Memory
   3.2 Pattern Discovery and Reflection
   3.3 Original-query Anchored Tri-query Group
   3.4 Masked Local Listwise Loss
   3.5 Original Query Rank Repair Loss

4. Training and Mining
   - error state machine
   - novelty and quality filtering
   - boundary negatives

5. Experiments
   - sanity check: cache / mining
   - main comparison
   - ablation
   - metrics: R@K, PER, OQRR, PMG, NER, CNS

6. Analysis
   - repeated error cases
   - error shifting
   - semantic pattern precision
   - failure cases
```

---

# 16. 最小可执行版本

你现在最应该先做这个：

```text
Dataset:
  CIRR validation

Rounds:
  2

Hard negatives:
  top-5 before ground truth

Methods:
  1. ReCALL-1R
  2. Naive Iterative ReCALL
  3. Naive Iterative ReCALL + OQR
  4. PE-ReCALL independent triplets
  5. PE-ReCALL masked tri-query + OQR

Metrics:
  R@1
  R@5
  PER
  OQRR
  PMG
  NER
```

第一阶段不要做太复杂的 semantic pattern。先只用：

```text
H_persistent = H_i^k ∩ H_i^{k-1}
```

如果这个版本已经能提高 OQRR 和 PMG，你的核心 idea 就基本站住了。

---

# 17. 最后给你一个“最终版 claim”

我建议你把 PE-ReCALL 的 claim 写成：

> Naive iterative ReCALL often provides redundant supervision because it does not distinguish newly mined hard negatives from errors that persist across rounds. We show that such persistent hard-negative attractors indicate an unrepaired local ranking boundary under the original composed query. PE-ReCALL addresses this by tracking persistent errors and converting them into masked tri-query micro-groups, where the original query, a pattern-contrastive query, and a pattern-positive query jointly recalibrate the target-vs-error boundary.

中文版本：

> Naive iterative ReCALL 的问题不是没有继续生成数据，而是没有区分新出现的 hard negatives 和跨轮持续存在的 persistent errors。我们发现这些 persistent hard-negative attractors 表明原始 composed query 的局部排序边界并没有被真正修复。PE-ReCALL 通过追踪 persistent errors，并把它们转化为 masked tri-query micro-groups，使原始 query、澄清 query 和错误模式 query 共同校准 ground truth 与重复错误模式之间的边界。

这个版本的 taste 比“ReCALL + 多轮 memory”强很多。

---

我的最终建议很明确：**不要继续加模块；现在要收紧核心机制。**  
你最应该优先完成三件事：

```text
1. 排除 cache / stale mining 伪影。
2. 把 tri-query 改成 masked local listwise loss。
3. 加 OQR loss，并用 OQRR + PMG 证明原始 query 的边界真的被修复。
```

做好这三件事，你这个 idea 会从 8/10 提升到 8.5/10 左右。