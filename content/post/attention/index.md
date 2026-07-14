---
title: "Attention, Parallelism, and Collective Communication in MoE Serving"
date: 2026-07-13
tags:
- Machine Learning
- Transformers
- Deep Learning
- Distributed Systems
- Mixture of Experts
- Attention
- Parallelism
summary: "A technical walkthrough of the transformer attention block, the parallelism strategies used to distribute it (DP, TP, CP, PP), mixture-of-experts dispatch and combine, and the collective communication primitives and cost models underneath all of it."
---

*Note: Significant portions of this article, including diagrams and code, were generated with the assistance of Claude (Anthropic). I wanted to be upfront about that, since I'd want to know before spending my time on something, and I imagine you might feel the same. If you find any inaccuracies feel free to point them out to me.*

## 1. Introduction

This article covers the machinery that modern mixture-of-experts (MoE) serving systems are built on: the attention block, the parallelism strategies used to distribute it across devices, the routing and communication that MoE layers require, and the collective communication primitives underneath all of it. It was written while reading two recent systems papers, NanoCP and UltraEP, both of which assume this material as background.

The organizing fact is the following. Almost every operation in a transformer acts on each token independently. Two do not:

1. The attention score computation, which mixes information across token positions.
2. MoE dispatch, which sends tokens to experts that may reside on other devices.

Every communication cost discussed below arises from one of these two operations. The axis that a parallelism strategy partitions determines whether that strategy collides with one of them, and therefore whether it requires communication at all. Section 2 identifies the exception in attention, Section 3 works out its consequences for parallelism, Section 4 identifies the exception in MoE, and Section 5 gives the cost model that prices both.

**Table 1: Notation used throughout.**

| Symbol | Meaning |
|---|---|
| `B` | Batch size (number of sequences processed together) |
| `T` | Sequence length (tokens per sequence) |
| `d_model` | Width of the residual stream, that is, the size of one token's vector |
| `n_heads` | Number of attention heads |
| `d_head` | Per-head width, equal to `d_model / n_heads` |
| `d_ff` | Inner width of the feed-forward network |
| `E` | Number of experts in an MoE layer |
| `k` | Number of experts activated per token (top-k routing) |
| `P` | Number of ranks participating in a collective operation |
| `n` | Message size in a collective operation, in bytes |

## 2. The transformer layer

The input to an attention block is a tensor `x` of shape `(B, T, d_model)`. Each of the `B × T` token positions holds a vector of length `d_model`, which is that token's current representation at this depth in the network.

### 2.1 Projections

Three learned matrices, `W_Q`, `W_K`, and `W_V`, each of shape `(d_model, d_model)`, produce the query, key, and value tensors:

```
Q = x @ W_Q,    K = x @ W_K,    V = x @ W_V
```

Each result has the same shape as the input, `(B, T, d_model)`. None of these matrices is head-aware. The head structure does not exist yet at this point in the computation.

This step is per-token. Writing the matrix product elementwise,

```
(x @ W)[i, j] = Σ_c x[i, c] · W[c, j]
```

the row index `i` appears on both sides and is never summed over. Output row `i` depends on input row `i` and on `W`. Row `i'` of `x`, belonging to any other token, is never referenced. The independence holds because `W` is a fixed weight matrix, shared identically across all rows, so any operation of the form `x @ W` has the same property. This fact recurs throughout the article.

### 2.2 Head partitioning

Since `d_model = n_heads × d_head` by construction, the last axis of `Q`, `K`, and `V` can be reinterpreted as two axes, `(n_heads, d_head)`. If `d_model = 8` and `n_heads = 2`, a token's eight-element key vector

```
[1.2, -0.4, 0.7, 2.1, 0.3, -1.5, 0.9, 1.8]
```

is reinterpreted as two four-element vectors, one per head:

```
head 0:  [1.2, -0.4, 0.7, 2.1]
head 1:  [0.3, -1.5, 0.9, 1.8]
```

No arithmetic occurs. The reshape is a reindexing of the same numbers, and in practice it does not move memory. The location of the boundary is a convention. What matters is that the same convention is applied to `Q`, `K`, and `V`, so that head 0's query, key, and value slices correspond to one another.

### 2.3 The score computation

Within a single head, attention computes a matrix of scores between every query position and every key position:

```
S = Q Kᵀ / √d_head        shape (T, T)
```

A causal mask sets `S[i, j] = -∞` for `j > i`, so that a query cannot attend to future positions. A softmax over the last axis converts each row into a distribution, and the output for each query is the corresponding weighted average of value vectors:

```
head_output = softmax(mask(S)) @ V        shape (T, d_head)
```

The head outputs are concatenated back to width `d_model`, multiplied by an output projection `W_O`, and added into the residual stream.

<svg width="100%" viewBox="0 0 680 420" role="img" xmlns="http://www.w3.org/2000/svg">
<title>Entry point of an attention block</title>
<defs><marker id="a1" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M2 1L8 5L2 9" fill="none" stroke="#888780" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/></marker></defs>
<rect x="190" y="40" width="300" height="44" rx="8" fill="#F1EFE8" stroke="#888780" stroke-width="0.5"/>
<text x="340" y="62" text-anchor="middle" dominant-baseline="central" font-family="sans-serif" font-size="14" font-weight="500" fill="#444441">x: (B,T,d_model) = (2,4,8)</text>
<line x1="340" y1="84" x2="340" y2="104" stroke="#888780" stroke-width="1.5" marker-end="url(#a1)"/>
<rect x="140" y="104" width="400" height="56" rx="8" fill="#F1EFE8" stroke="#888780" stroke-width="0.5"/>
<text x="340" y="122" text-anchor="middle" dominant-baseline="central" font-family="sans-serif" font-size="14" font-weight="500" fill="#444441">project + reshape (per head)</text>
<text x="340" y="140" text-anchor="middle" dominant-baseline="central" font-family="sans-serif" font-size="12" fill="#5F5E5A">Q,K,V each (T,d_head)=(4,4), head 0</text>
<line x1="340" y1="160" x2="340" y2="184" stroke="#888780" stroke-width="1.5" marker-end="url(#a1)"/>
<rect x="100" y="184" width="480" height="70" rx="8" fill="#FAEEDA" stroke="#BA7517" stroke-width="0.5"/>
<text x="340" y="208" text-anchor="middle" dominant-baseline="central" font-family="sans-serif" font-size="14" font-weight="500" fill="#633806">Q @ Kᵀ → scores, shape (T,T) = (4,4)</text>
<text x="340" y="230" text-anchor="middle" dominant-baseline="central" font-family="sans-serif" font-size="12" fill="#633806">every query row meets every key column</text>
<line x1="340" y1="254" x2="340" y2="278" stroke="#888780" stroke-width="1.5" marker-end="url(#a1)"/>
<rect x="140" y="278" width="400" height="56" rx="8" fill="#E1F5EE" stroke="#1D9E75" stroke-width="0.5"/>
<text x="340" y="296" text-anchor="middle" dominant-baseline="central" font-family="sans-serif" font-size="14" font-weight="500" fill="#085041">softmax(scores) → weights (4,4)</text>
<text x="340" y="314" text-anchor="middle" dominant-baseline="central" font-family="sans-serif" font-size="12" fill="#085041">weights @ V → head output (T,d_head)=(4,4)</text>
<line x1="340" y1="334" x2="340" y2="358" stroke="#888780" stroke-width="1.5" marker-end="url(#a1)"/>
<rect x="140" y="358" width="400" height="56" rx="8" fill="#F1EFE8" stroke="#888780" stroke-width="0.5"/>
<text x="340" y="376" text-anchor="middle" dominant-baseline="central" font-family="sans-serif" font-size="14" font-weight="500" fill="#444441">concat all heads + × W_O</text>
<text x="340" y="394" text-anchor="middle" dominant-baseline="central" font-family="sans-serif" font-size="12" fill="#5F5E5A">→ final output (T,d_model) = (4,8)</text>
</svg>

*Figure 1: The attention block. Only the highlighted step mixes information across token positions.*

### 2.4 The locus of cross-token interaction

The score computation is the only step in the block in which two distinct token positions interact. The difference lies in the operands:

```
x @ W        one operand (W) is shared by every row
Q @ Kᵀ       both operands are token-dependent
```

In the first product, the second operand is a fixed weight matrix, so the rows are independent. In the second, entry `(i, j)` of the result requires row `i` of `Q` and row `j` of `K`, and `K` is itself derived from every token's hidden state. Computing the score row for query `i` therefore requires the key vectors of all positions `j ≤ i`.

Whether an operation's operands are all token-independent determines every communication requirement in the rest of this article.

### 2.5 The feed-forward network

Each layer follows the attention block with a position-wise feed-forward network (FFN), which expands each token's vector to a wider inner dimension, applies a nonlinearity, and contracts it back:

```
ffn_out = GELU(x @ W_up) @ W_down
```

with `W_up` of shape `(d_model, d_ff)` and `W_down` of shape `(d_ff, d_model)`. The inner width `d_ff` is a capacity parameter, commonly a small multiple of `d_model`. The output width is fixed by the residual connection: `ffn_out` is added back into `x`, and addition requires matching shapes, so the FFN must return to width `d_model`.

<svg width="100%" viewBox="0 0 680 400" role="img" xmlns="http://www.w3.org/2000/svg">
<title>FFN shape flow: narrow, wide, narrow again</title>
<defs><marker id="a6" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M2 1L8 5L2 9" fill="none" stroke="#888780" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/></marker></defs>
<rect x="250" y="40" width="180" height="56" rx="8" fill="#F1EFE8" stroke="#888780" stroke-width="0.5"/>
<text x="340" y="58" text-anchor="middle" dominant-baseline="central" font-family="sans-serif" font-size="14" font-weight="500" fill="#444441">x_ln2</text>
<text x="340" y="76" text-anchor="middle" dominant-baseline="central" font-family="sans-serif" font-size="12" fill="#5F5E5A">(T,d_model) = (4,8)</text>
<line x1="340" y1="96" x2="340" y2="120" stroke="#888780" stroke-width="1.5" marker-end="url(#a6)"/>
<rect x="150" y="120" width="380" height="56" rx="8" fill="#E1F5EE" stroke="#1D9E75" stroke-width="0.5"/>
<text x="340" y="138" text-anchor="middle" dominant-baseline="central" font-family="sans-serif" font-size="14" font-weight="500" fill="#085041">hidden = GELU(x_ln2 @ W_up)</text>
<text x="340" y="156" text-anchor="middle" dominant-baseline="central" font-family="sans-serif" font-size="12" fill="#085041">(T,d_ff) = (4,16), nonlinearity here</text>
<line x1="340" y1="176" x2="340" y2="200" stroke="#888780" stroke-width="1.5" marker-end="url(#a6)"/>
<rect x="250" y="200" width="180" height="56" rx="8" fill="#F1EFE8" stroke="#888780" stroke-width="0.5"/>
<text x="340" y="218" text-anchor="middle" dominant-baseline="central" font-family="sans-serif" font-size="14" font-weight="500" fill="#444441">ffn_out</text>
<text x="340" y="236" text-anchor="middle" dominant-baseline="central" font-family="sans-serif" font-size="12" fill="#5F5E5A">hidden @ W_down → (4,8)</text>
<line x1="340" y1="256" x2="340" y2="280" stroke="#888780" stroke-width="1.5" marker-end="url(#a6)"/>
<rect x="180" y="280" width="320" height="56" rx="8" fill="#FAEEDA" stroke="#BA7517" stroke-width="0.5"/>
<text x="340" y="298" text-anchor="middle" dominant-baseline="central" font-family="sans-serif" font-size="14" font-weight="500" fill="#633806">x = x + ffn_out</text>
<text x="340" y="316" text-anchor="middle" dominant-baseline="central" font-family="sans-serif" font-size="12" fill="#633806">residual add, shapes must match: (4,8)</text>
<text x="340" y="360" text-anchor="middle" font-family="sans-serif" font-size="12" fill="#5F5E5A">Without GELU, the two linear maps compose into a single linear map</text>
<text x="340" y="380" text-anchor="middle" font-family="sans-serif" font-size="12" fill="#5F5E5A">The residual add requires ffn_out to return to width d_model</text>
</svg>

*Figure 2: The FFN expands, applies a nonlinearity, and contracts back to the residual width.*

The nonlinearity is required for the wide inner layer to contribute anything. Matrix multiplication is associative, so without it

```
(x @ W_up) @ W_down  =  x @ (W_up @ W_down)
```

and the right side is a single `(d_model, d_model)` matrix. This is straightforward to confirm numerically: with GELU removed, the two-layer computation and the collapsed single-matrix computation agree to floating-point tolerance, and with GELU restored they do not.

Both `x @ W_up` and `hidden @ W_down` have the form `x @ W`, so by Section 2.4 the FFN is per-token and requires no communication across positions. What the FFN computes, as opposed to how it is shaped, is covered in Appendix B. Section 4 replaces this component with a mixture of experts.

## 3. Parallelism as partitioning of a tensor axis

After the head reshape, the activation tensor has four axes: `(B, T, n_heads, d_head)`. Each parallelism strategy corresponds to partitioning one of these axes across devices. Communication is required exactly when the computation must combine information across slices of the partitioned axis.

<svg width="100%" viewBox="0 0 680 200" role="img" xmlns="http://www.w3.org/2000/svg">
<title>Which parallelism strategy splits which tensor axis</title>
<text x="340" y="36" text-anchor="middle" font-family="sans-serif" font-size="14" font-weight="500" fill="#444441">Tensor shape (B, T, n_heads, d_head): one axis per strategy</text>
<rect x="40" y="60" width="130" height="56" rx="8" fill="#E6F1FB" stroke="#378ADD" stroke-width="0.5"/>
<text x="105" y="80" text-anchor="middle" dominant-baseline="central" font-family="sans-serif" font-size="14" font-weight="500" fill="#0C447C">B (batch)</text>
<text x="105" y="100" text-anchor="middle" dominant-baseline="central" font-family="sans-serif" font-size="12" fill="#0C447C">→ split by DP</text>
<rect x="186" y="60" width="130" height="56" rx="8" fill="#E1F5EE" stroke="#1D9E75" stroke-width="0.5"/>
<text x="251" y="80" text-anchor="middle" dominant-baseline="central" font-family="sans-serif" font-size="14" font-weight="500" fill="#085041">T (sequence)</text>
<text x="251" y="100" text-anchor="middle" dominant-baseline="central" font-family="sans-serif" font-size="12" fill="#085041">→ split by CP</text>
<rect x="332" y="60" width="130" height="56" rx="8" fill="#EEEDFE" stroke="#7F77DD" stroke-width="0.5"/>
<text x="397" y="80" text-anchor="middle" dominant-baseline="central" font-family="sans-serif" font-size="14" font-weight="500" fill="#3C3489">n_heads</text>
<text x="397" y="100" text-anchor="middle" dominant-baseline="central" font-family="sans-serif" font-size="12" fill="#3C3489">→ split by TP</text>
<rect x="478" y="60" width="130" height="56" rx="8" fill="#F1EFE8" stroke="#888780" stroke-width="0.5"/>
<text x="543" y="80" text-anchor="middle" dominant-baseline="central" font-family="sans-serif" font-size="14" font-weight="500" fill="#444441">d_head</text>
<text x="543" y="100" text-anchor="middle" dominant-baseline="central" font-family="sans-serif" font-size="12" fill="#444441">→ never split</text>
<text x="340" y="150" text-anchor="middle" font-family="sans-serif" font-size="12" fill="#5F5E5A">d_head cannot be split without splitting a single dot product</text>
<text x="340" y="170" text-anchor="middle" font-family="sans-serif" font-size="12" fill="#5F5E5A">mid-computation, so it is not partitioned in practice</text>
</svg>

*Figure 3: The four axes of the attention activation tensor and the parallelism strategy associated with each.*

Ordering the axes by the communication they require gives `B < n_heads < T < d_head`. The ordering does not follow the position of the axes in the tensor shape. It follows from whether attention must combine information across the axis.

### 3.1 Data parallelism partitions the batch

Attention is defined strictly within a sequence. No operation in the block combines information from two different batch elements. Partitioning `B` across devices therefore requires no communication: each device runs the complete attention computation, over all heads, for its own subset of sequences.

### 3.2 Tensor parallelism partitions the heads

Each head is computed in isolation from the others until the final output projection, so the head axis can be partitioned across devices. The mechanism operates on the weight matrices.

The columns of `W_Q`, `W_K`, and `W_V` are grouped by head: the first `d_head` output columns produce head 0, the next `d_head` produce head 1, and so on. Splitting these matrices column-wise assigns whole heads to devices. Each device computes its own heads' queries, keys, and values, runs the entire score, softmax, and weighted-sum sequence locally, and never requires data from another device.

<svg width="100%" viewBox="0 0 680 440" role="img" xmlns="http://www.w3.org/2000/svg">
<title>Tensor parallelism: column-wise head split</title>
<defs><marker id="a2" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M2 1L8 5L2 9" fill="none" stroke="#888780" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/></marker></defs>
<rect x="200" y="44" width="280" height="24" rx="4" fill="#F1EFE8" stroke="#888780" stroke-width="0.5"/>
<text x="340" y="56" text-anchor="middle" dominant-baseline="central" font-family="sans-serif" font-size="12" fill="#444441">h (d_model)</text>
<line x1="340" y1="68" x2="340" y2="98" stroke="#888780" stroke-width="1.5" marker-end="url(#a2)"/>
<text x="356" y="88" font-family="sans-serif" font-size="12" fill="#5F5E5A">× W_K</text>
<rect x="180" y="104" width="80" height="86" fill="#EEEDFE" stroke="#7F77DD" stroke-width="0.5"/>
<text x="220" y="142" text-anchor="middle" dominant-baseline="central" font-family="sans-serif" font-size="14" font-weight="500" fill="#3C3489">H0</text>
<rect x="260" y="104" width="80" height="86" fill="#EEEDFE" stroke="#7F77DD" stroke-width="0.5"/>
<text x="300" y="142" text-anchor="middle" dominant-baseline="central" font-family="sans-serif" font-size="14" font-weight="500" fill="#3C3489">H1</text>
<rect x="340" y="104" width="80" height="86" fill="#E1F5EE" stroke="#1D9E75" stroke-width="0.5"/>
<text x="380" y="142" text-anchor="middle" dominant-baseline="central" font-family="sans-serif" font-size="14" font-weight="500" fill="#085041">H2</text>
<rect x="420" y="104" width="80" height="86" fill="#E1F5EE" stroke="#1D9E75" stroke-width="0.5"/>
<text x="460" y="142" text-anchor="middle" dominant-baseline="central" font-family="sans-serif" font-size="14" font-weight="500" fill="#085041">H3</text>
<line x1="340" y1="96" x2="340" y2="252" stroke="#888780" stroke-width="1.5" stroke-dasharray="4 3"/>
<text x="525" y="132" font-family="sans-serif" font-size="12" fill="#5F5E5A">← column-wise</text>
<text x="525" y="148" font-family="sans-serif" font-size="12" fill="#5F5E5A">TP split</text>
<text x="260" y="204" text-anchor="middle" font-family="sans-serif" font-size="12" fill="#5F5E5A">GPU 0</text>
<text x="420" y="204" text-anchor="middle" font-family="sans-serif" font-size="12" fill="#5F5E5A">GPU 1</text>
<text x="340" y="270" text-anchor="middle" font-family="sans-serif" font-size="12" fill="#5F5E5A">KV cache per GPU (per token)</text>
<rect x="70" y="282" width="230" height="72" rx="8" fill="none" stroke="#888780" stroke-width="0.5" stroke-dasharray="4 3"/>
<text x="185" y="300" text-anchor="middle" font-family="sans-serif" font-size="12" fill="#5F5E5A">GPU 0</text>
<rect x="90" y="310" width="180" height="24" rx="4" fill="#EEEDFE" stroke="#7F77DD" stroke-width="0.5"/>
<text x="180" y="322" text-anchor="middle" dominant-baseline="central" font-family="sans-serif" font-size="12" fill="#3C3489">K, V: H0, H1</text>
<rect x="380" y="282" width="230" height="72" rx="8" fill="none" stroke="#888780" stroke-width="0.5" stroke-dasharray="4 3"/>
<text x="495" y="300" text-anchor="middle" font-family="sans-serif" font-size="12" fill="#5F5E5A">GPU 1</text>
<rect x="400" y="310" width="180" height="24" rx="4" fill="#E1F5EE" stroke="#1D9E75" stroke-width="0.5"/>
<text x="490" y="322" text-anchor="middle" dominant-baseline="central" font-family="sans-serif" font-size="12" fill="#085041">K, V: H2, H3</text>
<text x="340" y="400" text-anchor="middle" font-family="sans-serif" font-size="12" fill="#5F5E5A">Each GPU caches only its own heads: no duplication</text>
</svg>

*Figure 4: Column-wise partitioning of the projection weights assigns whole heads to devices. The KV cache partitions along with them.*

The combine step requires an addition. Recombining per-head outputs looks like a concatenation, which across devices would be an all-gather. Megatron-style tensor parallelism instead folds the concatenation into the output projection: `W_O` is partitioned by rows, so each device multiplies its own head output by its own row block. Each device produces a tensor of full width `d_model` that represents a partial contribution to the correct answer. Splitting a matrix product along its contraction dimension makes the true result the sum of the partial products, so the combine is an addition, which across devices is an **all-reduce**.

Two consequences follow. TP requires the input `x` to be replicated across the group before the layer begins, which is typically already satisfied because the previous layer's all-reduce broadcast it. And the same column-then-row pattern applies to the feed-forward network, with `W_up` partitioned by columns and `W_down` by rows, so a transformer layer under TP costs one all-reduce for attention and one for the FFN.

### 3.3 The exception: multi-head latent attention

The head-partitioning argument depends on each head owning a private slice of the KV cache. Multi-head latent attention (MLA), used by the DeepSeek model family, violates this assumption by construction.

Instead of caching per-head keys and values, MLA compresses them into a single shared latent vector per token, from which every head reconstructs its own keys and values via a per-head up-projection. This compression is what makes MLA's KV cache small: DeepSeek-V3 caches a latent of width 512, plus a decoupled positional component of width 64, for 576 values per token, in place of full per-head keys and values.

Under head-wise TP, every device requires the entire latent to reconstruct even its own heads. There is no per-head slice to distribute, because the compression collapsed the head structure into one shared object. The cache therefore replicates instead of partitioning, and at a TP degree of 8 the model stores eight identical copies, which negates the compression that motivated MLA.

<svg width="100%" viewBox="0 0 680 480" role="img" xmlns="http://www.w3.org/2000/svg">
<title>MLA under tensor parallelism: cache duplication</title>
<defs><marker id="a3" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M2 1L8 5L2 9" fill="none" stroke="#888780" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/></marker></defs>
<rect x="200" y="44" width="280" height="24" rx="4" fill="#F1EFE8" stroke="#888780" stroke-width="0.5"/>
<text x="340" y="56" text-anchor="middle" dominant-baseline="central" font-family="sans-serif" font-size="12" fill="#444441">h (d_model)</text>
<line x1="340" y1="68" x2="340" y2="96" stroke="#888780" stroke-width="1.5" marker-end="url(#a3)"/>
<text x="356" y="86" font-family="sans-serif" font-size="12" fill="#5F5E5A">× W_DKV (compress)</text>
<rect x="270" y="102" width="140" height="36" rx="6" fill="#FAEEDA" stroke="#BA7517" stroke-width="1"/>
<text x="340" y="120" text-anchor="middle" dominant-baseline="central" font-family="sans-serif" font-size="14" font-weight="500" fill="#633806">c_KV</text>
<text x="424" y="112" font-family="sans-serif" font-size="12" fill="#5F5E5A">dim 512, no head divisions</text>
<text x="424" y="128" font-family="sans-serif" font-size="12" fill="#5F5E5A">one shared vector for ALL heads</text>
<text x="256" y="120" text-anchor="end" font-family="sans-serif" font-size="14" font-weight="500" fill="#444441">cached</text>
<line x1="340" y1="138" x2="340" y2="170" stroke="#888780" stroke-width="1.5" marker-end="url(#a3)"/>
<text x="356" y="158" font-family="sans-serif" font-size="12" fill="#5F5E5A">× W_UK, W_UV (decompress)</text>
<rect x="180" y="176" width="80" height="22" fill="#EEEDFE" stroke="#7F77DD" stroke-width="0.5"/>
<text x="220" y="187" text-anchor="middle" dominant-baseline="central" font-family="sans-serif" font-size="12" fill="#3C3489">H0</text>
<rect x="260" y="176" width="80" height="22" fill="#EEEDFE" stroke="#7F77DD" stroke-width="0.5"/>
<text x="300" y="187" text-anchor="middle" dominant-baseline="central" font-family="sans-serif" font-size="12" fill="#3C3489">H1</text>
<rect x="340" y="176" width="80" height="22" fill="#E1F5EE" stroke="#1D9E75" stroke-width="0.5"/>
<text x="380" y="187" text-anchor="middle" dominant-baseline="central" font-family="sans-serif" font-size="12" fill="#085041">H2</text>
<rect x="420" y="176" width="80" height="22" fill="#E1F5EE" stroke="#1D9E75" stroke-width="0.5"/>
<text x="460" y="187" text-anchor="middle" dominant-baseline="central" font-family="sans-serif" font-size="12" fill="#085041">H3</text>
<text x="340" y="238" text-anchor="middle" font-family="sans-serif" font-size="14" font-weight="500" fill="#444441">Every GPU needs full c_KV to decompress its heads</text>
<text x="340" y="272" text-anchor="middle" font-family="sans-serif" font-size="14" font-weight="500" fill="#444441">KV cache per GPU (per token)</text>
<rect x="70" y="284" width="230" height="68" rx="8" fill="none" stroke="#888780" stroke-width="0.5" stroke-dasharray="4 3"/>
<text x="185" y="302" text-anchor="middle" font-family="sans-serif" font-size="12" fill="#5F5E5A">GPU 0</text>
<rect x="95" y="312" width="180" height="24" rx="4" fill="#FAEEDA" stroke="#BA7517" stroke-width="0.5"/>
<text x="185" y="324" text-anchor="middle" dominant-baseline="central" font-family="sans-serif" font-size="12" fill="#633806">full c_KV (dim 512)</text>
<rect x="380" y="284" width="230" height="68" rx="8" fill="none" stroke="#888780" stroke-width="0.5" stroke-dasharray="4 3"/>
<text x="495" y="302" text-anchor="middle" font-family="sans-serif" font-size="12" fill="#5F5E5A">GPU 1</text>
<rect x="405" y="312" width="180" height="24" rx="4" fill="#FAEEDA" stroke="#BA7517" stroke-width="0.5"/>
<text x="495" y="324" text-anchor="middle" dominant-baseline="central" font-family="sans-serif" font-size="12" fill="#633806">full c_KV (dim 512)</text>
<text x="340" y="384" text-anchor="middle" font-family="sans-serif" font-size="12" fill="#5F5E5A">Identical copies: TP does not partition the cache</text>
<text x="340" y="404" text-anchor="middle" font-family="sans-serif" font-size="12" fill="#5F5E5A">With TP=8, eight full copies of c_KV, negating the compression</text>
<text x="340" y="428" text-anchor="middle" font-family="sans-serif" font-size="12" fill="#5F5E5A">Consequence: MLA-based models use DP, not TP, for attention</text>
</svg>

*Figure 5: MLA's shared latent has no head structure, so head-wise partitioning replicates the cache instead of dividing it.*

MoE serving stacks built on MLA models, including both systems that motivated this article, therefore use data parallelism for attention, where each instance holds the small latent cache for its own requests with no sharing and no duplication. Expert parallelism, a different axis, is reserved for the feed-forward layers. This is the DP-EP configuration that recurs throughout the MoE serving literature.

### 3.4 Context parallelism partitions the sequence

Context parallelism divides a single sequence's tokens across devices, where data parallelism divides whole sequences. The motivation is capacity: when one request's KV cache approaches the memory of a single device, that cache must be distributed regardless of how the compute is arranged.

For the projection step, partitioning `T` is as free as partitioning `B`. The projections are per-token, so a device holding tokens 0 and 1 can compute their queries, keys, and values using the full, unpartitioned weight matrices with no communication, and likewise for a device holding tokens 2 and 3.

<svg width="100%" viewBox="0 0 680 290" role="img" xmlns="http://www.w3.org/2000/svg">
<title>CP is silent like DP until attention needs to look across tokens</title>
<defs><marker id="a4" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M2 1L8 5L2 9" fill="none" stroke="#888780" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/></marker></defs>
<text x="340" y="30" text-anchor="middle" font-family="sans-serif" font-size="14" font-weight="500" fill="#444441">CP splits T: local until attention must look across it</text>
<rect x="55" y="50" width="260" height="56" rx="8" fill="#E6F1FB" stroke="#378ADD" stroke-width="0.5"/>
<text x="185" y="68" text-anchor="middle" dominant-baseline="central" font-family="sans-serif" font-size="14" font-weight="500" fill="#0C447C">x[:, 0:2, :]</text>
<text x="185" y="86" text-anchor="middle" dominant-baseline="central" font-family="sans-serif" font-size="12" fill="#0C447C">(2,2,8), tokens 0,1</text>
<rect x="365" y="50" width="260" height="56" rx="8" fill="#E1F5EE" stroke="#1D9E75" stroke-width="0.5"/>
<text x="495" y="68" text-anchor="middle" dominant-baseline="central" font-family="sans-serif" font-size="14" font-weight="500" fill="#085041">x[:, 2:4, :]</text>
<text x="495" y="86" text-anchor="middle" dominant-baseline="central" font-family="sans-serif" font-size="12" fill="#085041">(2,2,8), tokens 2,3</text>
<line x1="185" y1="106" x2="185" y2="130" stroke="#888780" stroke-width="1.5" marker-end="url(#a4)"/>
<line x1="495" y1="106" x2="495" y2="130" stroke="#888780" stroke-width="1.5" marker-end="url(#a4)"/>
<rect x="55" y="130" width="260" height="56" rx="8" fill="#E6F1FB" stroke="#378ADD" stroke-width="0.5"/>
<text x="185" y="148" text-anchor="middle" dominant-baseline="central" font-family="sans-serif" font-size="14" font-weight="500" fill="#0C447C">× W_Q, W_K, W_V (full)</text>
<text x="185" y="166" text-anchor="middle" dominant-baseline="central" font-family="sans-serif" font-size="12" fill="#0C447C">→ Q,K,V for tokens 0,1</text>
<rect x="365" y="130" width="260" height="56" rx="8" fill="#E1F5EE" stroke="#1D9E75" stroke-width="0.5"/>
<text x="495" y="148" text-anchor="middle" dominant-baseline="central" font-family="sans-serif" font-size="14" font-weight="500" fill="#085041">× W_Q, W_K, W_V (full)</text>
<text x="495" y="166" text-anchor="middle" dominant-baseline="central" font-family="sans-serif" font-size="12" fill="#085041">→ Q,K,V for tokens 2,3</text>
<line x1="40" y1="210" x2="640" y2="210" stroke="#888780" stroke-width="1" stroke-dasharray="4 3"/>
<text x="340" y="200" text-anchor="middle" font-family="sans-serif" font-size="12" fill="#5F5E5A">identical to DP so far: no communication</text>
<text x="340" y="228" text-anchor="middle" font-family="sans-serif" font-size="12" fill="#5F5E5A">the score computation needs every token's K,V, not only the local shard</text>
<text x="340" y="262" text-anchor="middle" font-family="sans-serif" font-size="12" fill="#5F5E5A">an exchange across shards is therefore required</text>
</svg>

*Figure 6: Under context parallelism the projections remain local. The score computation does not.*

The score computation does not remain local. The query at position 3 must attend to keys at positions 0 through 3, half of which reside on the other device, so an exchange is unavoidable. Context parallelism is more expensive than the other two axes for this reason: its communication sits in the middle of the computation, where tensor parallelism's sits only at the end.

Both implementations rely on the same property, which also underlies FlashAttention's tiling and Flash-Decoding's split reductions. A softmax-weighted average can be computed over disjoint blocks of keys and merged exactly, provided each partial result carries a running maximum and normalizer, equivalently a log-sum-exp term, alongside its unnormalized output. Given these, the partials recombine to the value the unpartitioned softmax would have produced.

The two implementations use the same merge and differ in topology:

- **Query routing.** The device holding the query sends it to whichever devices hold the relevant KV shards. Each computes a partial attention output and its log-sum-exp locally, and returns both. The originating device merges them. Helix and NanoCP both use this form.
- **Ring rotation.** KV chunks are rotated around a cycle of devices. At each hop, a device merges the arriving chunk into its running partial result and forwards its own chunk onward. After `P` hops every device has seen the entire sequence. This is Ring Attention.

<svg width="100%" viewBox="0 0 680 420" role="img" xmlns="http://www.w3.org/2000/svg">
<title>Ring Attention communication topology</title>
<defs><marker id="a5" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M2 1L8 5L2 9" fill="none" stroke="#888780" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/></marker></defs>
<text x="340" y="24" text-anchor="middle" font-family="sans-serif" font-size="14" font-weight="500" fill="#444441">Ring Attention: chunks rotate, partials accumulate</text>
<line x1="340" y1="70" x2="520" y2="190" stroke="#888780" stroke-width="1.5" marker-end="url(#a5)"/>
<line x1="520" y1="190" x2="340" y2="310" stroke="#888780" stroke-width="1.5" marker-end="url(#a5)"/>
<line x1="340" y1="310" x2="160" y2="190" stroke="#888780" stroke-width="1.5" marker-end="url(#a5)"/>
<line x1="160" y1="190" x2="340" y2="70" stroke="#888780" stroke-width="1.5" marker-end="url(#a5)"/>
<text x="430" y="118" text-anchor="middle" font-family="sans-serif" font-size="12" fill="#5F5E5A">KV chunks rotate this way</text>
<rect x="270" y="42" width="140" height="56" rx="8" fill="#E6F1FB" stroke="#378ADD" stroke-width="0.5"/>
<text x="340" y="60" text-anchor="middle" dominant-baseline="central" font-family="sans-serif" font-size="14" font-weight="500" fill="#0C447C">GPU 0</text>
<text x="340" y="78" text-anchor="middle" dominant-baseline="central" font-family="sans-serif" font-size="12" fill="#0C447C">own KV + partial</text>
<rect x="450" y="162" width="140" height="56" rx="8" fill="#E6F1FB" stroke="#378ADD" stroke-width="0.5"/>
<text x="520" y="180" text-anchor="middle" dominant-baseline="central" font-family="sans-serif" font-size="14" font-weight="500" fill="#0C447C">GPU 1</text>
<text x="520" y="198" text-anchor="middle" dominant-baseline="central" font-family="sans-serif" font-size="12" fill="#0C447C">own KV + partial</text>
<rect x="270" y="282" width="140" height="56" rx="8" fill="#E6F1FB" stroke="#378ADD" stroke-width="0.5"/>
<text x="340" y="300" text-anchor="middle" dominant-baseline="central" font-family="sans-serif" font-size="14" font-weight="500" fill="#0C447C">GPU 2</text>
<text x="340" y="318" text-anchor="middle" dominant-baseline="central" font-family="sans-serif" font-size="12" fill="#0C447C">own KV + partial</text>
<rect x="90" y="162" width="140" height="56" rx="8" fill="#E6F1FB" stroke="#378ADD" stroke-width="0.5"/>
<text x="160" y="180" text-anchor="middle" dominant-baseline="central" font-family="sans-serif" font-size="14" font-weight="500" fill="#0C447C">GPU 3</text>
<text x="160" y="198" text-anchor="middle" dominant-baseline="central" font-family="sans-serif" font-size="12" fill="#0C447C">own KV + partial</text>
<text x="340" y="362" text-anchor="middle" font-family="sans-serif" font-size="12" fill="#5F5E5A">Each hop: merge new chunk into running partial (LSE)</text>
<text x="340" y="380" text-anchor="middle" font-family="sans-serif" font-size="12" fill="#5F5E5A">then forward your own previous chunk to the next GPU</text>
<text x="340" y="398" text-anchor="middle" font-family="sans-serif" font-size="12" fill="#5F5E5A">After N hops, every GPU has seen the whole sequence</text>
</svg>

*Figure 7: Ring Attention rotates KV chunks around a fixed cycle instead of routing queries to fixed destinations.*

Ring passes fixed-size chunks through a predictable, bandwidth-friendly pattern, which suits the large groups typical of training. Query routing sends data directly to its destination and avoids intermediate hops, which suits the tighter latency budgets and the smaller, more dynamic groups of decode-time serving.

### 3.5 The head dimension is not partitioned

Splitting `d_head` would divide a single dot product across devices, requiring an exchange of partial sums before the softmax could be applied to any score. The synchronization granularity is too fine to be practical, and the axis is left intact. Research on partitioning MLA's latent dimension exists, motivated by the tensor-parallel limitation in Section 3.3, but it remains an open problem.

### 3.6 Pipeline parallelism partitions the layer stack

Pipeline parallelism does not partition the activation tensor. It assigns different layers to different devices, so one device holds layers 1 through 20 and passes activations onward to a device holding layers 21 through 40.

PP's communication profile differs from TP's. TP requires an all-reduce in every layer, which is frequent, latency-sensitive, synchronous traffic. PP requires one activation transfer per stage boundary. PP therefore tolerates slow interconnects far better, and it is the preferred choice once a deployment spans more devices than fit in a single high-bandwidth domain. Sarathi-Serve reports roughly a factor of two lower median latency for PP over TP when serving across nodes connected by commodity Ethernet.

PP pays for this with pipeline bubbles: idle stages waiting for work to arrive from upstream, which is most acute under low request load. The common production configuration follows from these two facts. TP is used within a fast interconnect domain, PP across domains, with DP and EP layered on top.

## 4. Mixture of experts

An MoE layer replaces the single shared FFN of Section 2.5 with `E` smaller ones, called experts, together with a gate that selects, per token, which `k` of them will process it.

### 4.1 Gating

The gate is a linear map from the token's hidden state to `E` scores, followed by a top-`k` selection and a softmax restricted to the selected experts. Like every other operation of the form `x @ W`, it is per-token and requires no communication.

<svg width="100%" viewBox="0 0 680 380" role="img" xmlns="http://www.w3.org/2000/svg">
<title>MoE gate routing decision</title>
<text x="340" y="30" text-anchor="middle" font-family="sans-serif" font-size="14" font-weight="500" fill="#444441">Gate routing: top-2 experts per token (no communication yet)</text>
<text x="247.5" y="75" text-anchor="middle" font-family="sans-serif" font-size="12" fill="#5F5E5A">e0</text>
<text x="302.5" y="75" text-anchor="middle" font-family="sans-serif" font-size="12" fill="#5F5E5A">e1</text>
<text x="357.5" y="75" text-anchor="middle" font-family="sans-serif" font-size="12" fill="#5F5E5A">e2</text>
<text x="412.5" y="75" text-anchor="middle" font-family="sans-serif" font-size="12" fill="#5F5E5A">e3</text>
<text x="205" y="117.5" text-anchor="end" dominant-baseline="central" font-family="sans-serif" font-size="12" fill="#5F5E5A">t0</text>
<text x="205" y="172.5" text-anchor="end" dominant-baseline="central" font-family="sans-serif" font-size="12" fill="#5F5E5A">t1</text>
<text x="205" y="227.5" text-anchor="end" dominant-baseline="central" font-family="sans-serif" font-size="12" fill="#5F5E5A">t2</text>
<text x="205" y="282.5" text-anchor="end" dominant-baseline="central" font-family="sans-serif" font-size="12" fill="#5F5E5A">t3</text>
<rect x="220" y="90" width="55" height="55" fill="#EEEDFE" stroke="#7F77DD" stroke-width="0.5"/>
<rect x="275" y="90" width="55" height="55" fill="#EEEDFE" stroke="#7F77DD" stroke-width="0.5"/>
<rect x="330" y="90" width="55" height="55" fill="none" stroke="#888780" stroke-width="0.5" stroke-dasharray="3 3"/>
<rect x="385" y="90" width="55" height="55" fill="none" stroke="#888780" stroke-width="0.5" stroke-dasharray="3 3"/>
<rect x="220" y="145" width="55" height="55" fill="#EEEDFE" stroke="#7F77DD" stroke-width="0.5"/>
<rect x="275" y="145" width="55" height="55" fill="#EEEDFE" stroke="#7F77DD" stroke-width="0.5"/>
<rect x="330" y="145" width="55" height="55" fill="none" stroke="#888780" stroke-width="0.5" stroke-dasharray="3 3"/>
<rect x="385" y="145" width="55" height="55" fill="none" stroke="#888780" stroke-width="0.5" stroke-dasharray="3 3"/>
<rect x="220" y="200" width="55" height="55" fill="#EEEDFE" stroke="#7F77DD" stroke-width="0.5"/>
<rect x="275" y="200" width="55" height="55" fill="none" stroke="#888780" stroke-width="0.5" stroke-dasharray="3 3"/>
<rect x="330" y="200" width="55" height="55" fill="#EEEDFE" stroke="#7F77DD" stroke-width="0.5"/>
<rect x="385" y="200" width="55" height="55" fill="none" stroke="#888780" stroke-width="0.5" stroke-dasharray="3 3"/>
<rect x="220" y="255" width="55" height="55" fill="#EEEDFE" stroke="#7F77DD" stroke-width="0.5"/>
<rect x="275" y="255" width="55" height="55" fill="none" stroke="#888780" stroke-width="0.5" stroke-dasharray="3 3"/>
<rect x="330" y="255" width="55" height="55" fill="none" stroke="#888780" stroke-width="0.5" stroke-dasharray="3 3"/>
<rect x="385" y="255" width="55" height="55" fill="#EEEDFE" stroke="#7F77DD" stroke-width="0.5"/>
<text x="340" y="336" text-anchor="middle" font-family="sans-serif" font-size="12" fill="#5F5E5A">Every token computes this independently and locally</text>
<text x="340" y="356" text-anchor="middle" font-family="sans-serif" font-size="12" fill="#5F5E5A">e0 is selected by all four tokens: load is skewed before any GPU is involved</text>
</svg>

*Figure 8: A routing table for four tokens under top-2 routing. Column e0 is already overloaded relative to e2 and e3, as a consequence of the router's learned scores alone.*

Load imbalance therefore originates in the model, not in the system. The routing table above is skewed before any question of device placement arises.

### 4.2 Dispatch and combine

Expert weights are large and are not moved. The tokens are moved instead. If a token's selected experts reside on other devices, its hidden state must travel to them, and this transfer is called **dispatch**. Each expert then runs its own FFN over whatever tokens arrived, grouped by expert so that one matrix multiplication, a grouped GEMM, serves all tokens routed to it. The results travel back to the tokens' origins, which is called **combine**, and are summed there.

<svg width="100%" viewBox="0 0 680 400" role="img" xmlns="http://www.w3.org/2000/svg">
<title>MoE dispatch and combine for a single token</title>
<defs><marker id="a8" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M2 1L8 5L2 9" fill="none" stroke="#888780" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/></marker></defs>
<rect x="190" y="40" width="300" height="44" rx="8" fill="#F1EFE8" stroke="#888780" stroke-width="0.5"/>
<text x="340" y="62" text-anchor="middle" dominant-baseline="central" font-family="sans-serif" font-size="14" font-weight="500" fill="#444441">t0 hidden state (after attention)</text>
<line x1="340" y1="84" x2="185" y2="104" stroke="#888780" stroke-width="1.5" marker-end="url(#a8)"/>
<line x1="340" y1="84" x2="495" y2="104" stroke="#888780" stroke-width="1.5" marker-end="url(#a8)"/>
<text x="220" y="98" text-anchor="middle" font-family="sans-serif" font-size="12" fill="#5F5E5A">dispatch</text>
<text x="460" y="98" text-anchor="middle" font-family="sans-serif" font-size="12" fill="#5F5E5A">dispatch</text>
<rect x="55" y="104" width="260" height="56" rx="8" fill="#E6F1FB" stroke="#378ADD" stroke-width="0.5"/>
<text x="185" y="122" text-anchor="middle" dominant-baseline="central" font-family="sans-serif" font-size="14" font-weight="500" fill="#0C447C">GPU 0: expert e0</text>
<text x="185" y="140" text-anchor="middle" dominant-baseline="central" font-family="sans-serif" font-size="12" fill="#0C447C">FFN(t0) → output_e0</text>
<rect x="365" y="104" width="260" height="56" rx="8" fill="#FAECE7" stroke="#D85A30" stroke-width="0.5"/>
<text x="495" y="122" text-anchor="middle" dominant-baseline="central" font-family="sans-serif" font-size="14" font-weight="500" fill="#712B13">GPU 1: expert e1</text>
<text x="495" y="140" text-anchor="middle" dominant-baseline="central" font-family="sans-serif" font-size="12" fill="#712B13">FFN(t0) → output_e1</text>
<line x1="185" y1="160" x2="340" y2="200" stroke="#888780" stroke-width="1.5" marker-end="url(#a8)"/>
<line x1="495" y1="160" x2="340" y2="200" stroke="#888780" stroke-width="1.5" marker-end="url(#a8)"/>
<text x="240" y="188" text-anchor="middle" font-family="sans-serif" font-size="12" fill="#5F5E5A">combine</text>
<text x="440" y="188" text-anchor="middle" font-family="sans-serif" font-size="12" fill="#5F5E5A">combine</text>
<rect x="190" y="200" width="300" height="56" rx="8" fill="#FAEEDA" stroke="#BA7517" stroke-width="0.5"/>
<text x="340" y="218" text-anchor="middle" dominant-baseline="central" font-family="sans-serif" font-size="14" font-weight="500" fill="#633806">combine: w0·out_e0 + w1·out_e1</text>
<text x="340" y="236" text-anchor="middle" dominant-baseline="central" font-family="sans-serif" font-size="12" fill="#633806">w0, w1 = softmax over t0's top-2 scores</text>
<line x1="340" y1="256" x2="340" y2="280" stroke="#888780" stroke-width="1.5" marker-end="url(#a8)"/>
<rect x="190" y="280" width="300" height="44" rx="8" fill="#F1EFE8" stroke="#888780" stroke-width="0.5"/>
<text x="340" y="302" text-anchor="middle" dominant-baseline="central" font-family="sans-serif" font-size="14" font-weight="500" fill="#444441">final MoE output for t0</text>
<text x="340" y="356" text-anchor="middle" font-family="sans-serif" font-size="12" fill="#5F5E5A">This repeats independently for every token, simultaneously</text>
<text x="340" y="376" text-anchor="middle" font-family="sans-serif" font-size="12" fill="#5F5E5A">No LSE rescaling: the top-2 weights are normalized before dispatch</text>
</svg>

*Figure 9: Dispatch and combine for a single token routed to two experts on different devices.*

Combine is simpler than the context-parallel merge of Section 3.4. That merge required log-sum-exp rescaling because a softmax normalizer cannot be computed correctly from a partial view of the keys. The MoE gate has already computed a full softmax over the selected experts locally, before dispatch, so the weights are correct on arrival and combine is a plain weighted sum.

### 4.3 Stragglers

Dispatch and combine are not per-token operations. Every token in the current batch is packed into a single collective over the expert-parallel group, and that collective does not complete for any participant until it has completed for all of them.

<svg width="100%" viewBox="0 0 680 330" role="img" xmlns="http://www.w3.org/2000/svg">
<title>Dispatch timeline showing GPU wait times</title>
<text x="598" y="46" text-anchor="end" font-family="sans-serif" font-size="12" fill="#5F5E5A">Barrier: all ranks must arrive</text>
<line x1="600" y1="58" x2="600" y2="250" stroke="#5F5E5A" stroke-width="1.5" stroke-dasharray="4 3"/>
<text x="598" y="256" text-anchor="end" font-family="sans-serif" font-size="12" fill="#5F5E5A">Only then does the next layer start</text>
<text x="128" y="74" text-anchor="end" dominant-baseline="central" font-family="sans-serif" font-size="12" fill="#5F5E5A">GPU 0 (10)</text>
<rect x="135" y="62" width="52" height="24" rx="4" fill="#FAEEDA" stroke="#BA7517" stroke-width="0.5"/>
<rect x="187" y="62" width="413" height="24" rx="4" fill="none" stroke="#5F5E5A" stroke-width="1" stroke-dasharray="4 3"/>
<text x="128" y="124" text-anchor="end" dominant-baseline="central" font-family="sans-serif" font-size="12" fill="#5F5E5A">GPU 1 (90)</text>
<rect x="135" y="112" width="465" height="24" rx="4" fill="#FAEEDA" stroke="#BA7517" stroke-width="0.5"/>
<text x="590" y="124" text-anchor="end" dominant-baseline="central" font-family="sans-serif" font-size="12" fill="#633806">straggler</text>
<text x="128" y="174" text-anchor="end" dominant-baseline="central" font-family="sans-serif" font-size="12" fill="#5F5E5A">GPU 2 (15)</text>
<rect x="135" y="162" width="78" height="24" rx="4" fill="#FAEEDA" stroke="#BA7517" stroke-width="0.5"/>
<rect x="213" y="162" width="387" height="24" rx="4" fill="none" stroke="#5F5E5A" stroke-width="1" stroke-dasharray="4 3"/>
<text x="128" y="224" text-anchor="end" dominant-baseline="central" font-family="sans-serif" font-size="12" fill="#5F5E5A">GPU 3 (20)</text>
<rect x="135" y="212" width="103" height="24" rx="4" fill="#FAEEDA" stroke="#BA7517" stroke-width="0.5"/>
<rect x="238" y="212" width="362" height="24" rx="4" fill="none" stroke="#5F5E5A" stroke-width="1" stroke-dasharray="4 3"/>
<rect x="135" y="272" width="16" height="12" rx="3" fill="#FAEEDA" stroke="#BA7517" stroke-width="0.5"/>
<text x="157" y="281" font-family="sans-serif" font-size="12" fill="#5F5E5A">Moving data (dispatch)</text>
<rect x="340" y="272" width="16" height="12" rx="3" fill="none" stroke="#5F5E5A" stroke-width="1" stroke-dasharray="4 3"/>
<text x="362" y="281" font-family="sans-serif" font-size="12" fill="#5F5E5A">Idle: waiting at the barrier</text>
</svg>

*Figure 10: Token counts per rank, in parentheses, translate directly into transfer time. The lightly loaded ranks idle at the barrier until the heaviest one completes.*

Consider a rank hosting an expert that the router favors heavily in the current batch. That rank pays for the same imbalance three times in succession, at three separate cost centers, each scaling with the same token count:

1. **Dispatch receive.** More tokens routed to its experts means more bytes arriving over the interconnect.
2. **Compute.** The grouped GEMM over those tokens is proportionally larger.
3. **Combine send.** The results, one vector per token, travel back out.

The lightly loaded ranks idle twice: once waiting for the heavy rank's dispatch to complete, and again waiting for its compute and combine. Their hardware is free during both intervals, but the next layer cannot begin, because it requires every rank's tokens to be assembled first.

This is the second exception announced in Section 1, and the target of the expert load balancing literature. EPLB replicates hot experts on periodically recomputed placements. UltraEP recomputes the replication plan from the exact post-gating load, on the critical path, every layer.

Continuous batching addresses a different problem. It governs which requests are admitted to or retired from the batch between iterations. Within a single iteration, every admitted token is packed into the same kernels and the same collectives, and stragglers arise at that level.

## 5. Collective communication

Both exceptions reduce to a small set of named collective operations. This section defines them, gives their costs, and identifies which ones the preceding sections have been invoking.

### 5.1 The primitives

**Table 2: The standard collective operations, for `P` ranks.**

| Operation | Result |
|---|---|
| Broadcast | One rank's buffer is copied identically to every rank |
| Scatter | One rank's buffer is partitioned; slice `i` goes to rank `i` |
| Gather | Every rank's buffer is collected onto one rank |
| All-gather | Every rank's buffer is collected onto every rank |
| Reduce | Elementwise reduction (typically a sum) of all buffers, result on one rank |
| All-reduce | The same reduction, result on every rank |
| Reduce-scatter | The same reduction, but rank `i` retains only slice `i` of the result |
| All-to-all | Rank `i`'s slice `j` is sent to rank `j`, for all `i, j` |

Broadcast and scatter share a fan-out shape and differ in what travels: identical copies in the first case, distinct pieces in the second. Gather, all-gather, reduce, and all-reduce share a converge shape and differ along two dimensions: whether the center performs a computation (reduce) or concatenates (gather), and whether the result is delivered to one rank or to all.

<svg width="100%" viewBox="0 0 680 460" role="img" xmlns="http://www.w3.org/2000/svg">
<title>Reduce versus all-reduce</title>
<defs><marker id="a9" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M2 1L8 5L2 9" fill="none" stroke="#888780" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/></marker></defs>
<text x="340" y="26" text-anchor="middle" font-family="sans-serif" font-size="14" font-weight="500" fill="#444441">Reduce (sum)</text>
<text x="105" y="42" text-anchor="middle" font-family="sans-serif" font-size="12" fill="#5F5E5A">R0</text><text x="252" y="42" text-anchor="middle" font-family="sans-serif" font-size="12" fill="#5F5E5A">R1</text><text x="399" y="42" text-anchor="middle" font-family="sans-serif" font-size="12" fill="#5F5E5A">R2</text><text x="546" y="42" text-anchor="middle" font-family="sans-serif" font-size="12" fill="#5F5E5A">R3</text>
<circle cx="105" cy="70" r="20" fill="#E6F1FB" stroke="#378ADD" stroke-width="0.5"/><text x="105" y="70" text-anchor="middle" dominant-baseline="central" font-family="sans-serif" font-size="14" font-weight="500" fill="#0C447C">a</text>
<circle cx="252" cy="70" r="20" fill="#E6F1FB" stroke="#378ADD" stroke-width="0.5"/><text x="252" y="70" text-anchor="middle" dominant-baseline="central" font-family="sans-serif" font-size="14" font-weight="500" fill="#0C447C">b</text>
<circle cx="399" cy="70" r="20" fill="#E6F1FB" stroke="#378ADD" stroke-width="0.5"/><text x="399" y="70" text-anchor="middle" dominant-baseline="central" font-family="sans-serif" font-size="14" font-weight="500" fill="#0C447C">c</text>
<circle cx="546" cy="70" r="20" fill="#E6F1FB" stroke="#378ADD" stroke-width="0.5"/><text x="546" y="70" text-anchor="middle" dominant-baseline="central" font-family="sans-serif" font-size="14" font-weight="500" fill="#0C447C">d</text>
<line x1="105" y1="90" x2="105" y2="140" stroke="#888780" stroke-width="1.5" marker-end="url(#a9)"/>
<line x1="252" y1="90" x2="105" y2="140" stroke="#888780" stroke-width="1.5" marker-end="url(#a9)"/>
<line x1="399" y1="90" x2="105" y2="140" stroke="#888780" stroke-width="1.5" marker-end="url(#a9)"/>
<line x1="546" y1="90" x2="105" y2="140" stroke="#888780" stroke-width="1.5" marker-end="url(#a9)"/>
<circle cx="105" cy="160" r="20" fill="#FAEEDA" stroke="#BA7517" stroke-width="0.5"/><text x="105" y="160" text-anchor="middle" dominant-baseline="central" font-family="sans-serif" font-size="14" font-weight="500" fill="#633806">Σ</text>
<circle cx="252" cy="160" r="20" fill="#F1EFE8" stroke="#888780" stroke-width="0.5"/><text x="252" y="160" text-anchor="middle" dominant-baseline="central" font-family="sans-serif" font-size="14" font-weight="500" fill="#444441">-</text>
<circle cx="399" cy="160" r="20" fill="#F1EFE8" stroke="#888780" stroke-width="0.5"/><text x="399" y="160" text-anchor="middle" dominant-baseline="central" font-family="sans-serif" font-size="14" font-weight="500" fill="#444441">-</text>
<circle cx="546" cy="160" r="20" fill="#F1EFE8" stroke="#888780" stroke-width="0.5"/><text x="546" y="160" text-anchor="middle" dominant-baseline="central" font-family="sans-serif" font-size="14" font-weight="500" fill="#444441">-</text>
<text x="340" y="200" text-anchor="middle" font-family="sans-serif" font-size="12" fill="#5F5E5A">Only R0 receives the summed result</text>
<text x="340" y="240" text-anchor="middle" font-family="sans-serif" font-size="14" font-weight="500" fill="#444441">All-Reduce (sum)</text>
<text x="105" y="256" text-anchor="middle" font-family="sans-serif" font-size="12" fill="#5F5E5A">R0</text><text x="252" y="256" text-anchor="middle" font-family="sans-serif" font-size="12" fill="#5F5E5A">R1</text><text x="399" y="256" text-anchor="middle" font-family="sans-serif" font-size="12" fill="#5F5E5A">R2</text><text x="546" y="256" text-anchor="middle" font-family="sans-serif" font-size="12" fill="#5F5E5A">R3</text>
<circle cx="105" cy="284" r="20" fill="#E6F1FB" stroke="#378ADD" stroke-width="0.5"/><text x="105" y="284" text-anchor="middle" dominant-baseline="central" font-family="sans-serif" font-size="14" font-weight="500" fill="#0C447C">a</text>
<circle cx="252" cy="284" r="20" fill="#E6F1FB" stroke="#378ADD" stroke-width="0.5"/><text x="252" y="284" text-anchor="middle" dominant-baseline="central" font-family="sans-serif" font-size="14" font-weight="500" fill="#0C447C">b</text>
<circle cx="399" cy="284" r="20" fill="#E6F1FB" stroke="#378ADD" stroke-width="0.5"/><text x="399" y="284" text-anchor="middle" dominant-baseline="central" font-family="sans-serif" font-size="14" font-weight="500" fill="#0C447C">c</text>
<circle cx="546" cy="284" r="20" fill="#E6F1FB" stroke="#378ADD" stroke-width="0.5"/><text x="546" y="284" text-anchor="middle" dominant-baseline="central" font-family="sans-serif" font-size="14" font-weight="500" fill="#0C447C">d</text>
<line x1="105" y1="304" x2="340" y2="328" stroke="#888780" stroke-width="1.5" marker-end="url(#a9)"/>
<line x1="252" y1="304" x2="340" y2="328" stroke="#888780" stroke-width="1.5" marker-end="url(#a9)"/>
<line x1="399" y1="304" x2="340" y2="328" stroke="#888780" stroke-width="1.5" marker-end="url(#a9)"/>
<line x1="546" y1="304" x2="340" y2="328" stroke="#888780" stroke-width="1.5" marker-end="url(#a9)"/>
<rect x="300" y="328" width="80" height="24" rx="6" fill="#FAEEDA" stroke="#BA7517" stroke-width="0.5"/>
<text x="340" y="340" text-anchor="middle" dominant-baseline="central" font-family="sans-serif" font-size="14" font-weight="500" fill="#633806">Σ sum</text>
<line x1="340" y1="352" x2="105" y2="380" stroke="#888780" stroke-width="1.5" marker-end="url(#a9)"/>
<line x1="340" y1="352" x2="252" y2="380" stroke="#888780" stroke-width="1.5" marker-end="url(#a9)"/>
<line x1="340" y1="352" x2="399" y2="380" stroke="#888780" stroke-width="1.5" marker-end="url(#a9)"/>
<line x1="340" y1="352" x2="546" y2="380" stroke="#888780" stroke-width="1.5" marker-end="url(#a9)"/>
<circle cx="105" cy="400" r="20" fill="#FAEEDA" stroke="#BA7517" stroke-width="0.5"/><text x="105" y="400" text-anchor="middle" dominant-baseline="central" font-family="sans-serif" font-size="14" font-weight="500" fill="#633806">Σ</text>
<circle cx="252" cy="400" r="20" fill="#FAEEDA" stroke="#BA7517" stroke-width="0.5"/><text x="252" y="400" text-anchor="middle" dominant-baseline="central" font-family="sans-serif" font-size="14" font-weight="500" fill="#633806">Σ</text>
<circle cx="399" cy="400" r="20" fill="#FAEEDA" stroke="#BA7517" stroke-width="0.5"/><text x="399" y="400" text-anchor="middle" dominant-baseline="central" font-family="sans-serif" font-size="14" font-weight="500" fill="#633806">Σ</text>
<circle cx="546" cy="400" r="20" fill="#FAEEDA" stroke="#BA7517" stroke-width="0.5"/><text x="546" y="400" text-anchor="middle" dominant-baseline="central" font-family="sans-serif" font-size="14" font-weight="500" fill="#633806">Σ</text>
<text x="340" y="440" text-anchor="middle" font-family="sans-serif" font-size="12" fill="#5F5E5A">Every rank ends up holding the identical sum</text>
</svg>

*Figure 11: Reduce and all-reduce differ only in whether the result is returned to one rank or to all of them.*

All-reduce is the operation that combines the two partial outputs in tensor parallelism (Section 3.2). Reduce-scatter shares the converge step but distributes slices of the result instead of copies of it. In practice all-reduce is not implemented as a distinct primitive: it is a reduce-scatter followed by an all-gather.

All-to-all does not fit the converge-diverge shape. Every rank sends distinct data to every other rank simultaneously, and the operation is a transpose of the block matrix whose entry `(i, j)` is the payload rank `i` owes rank `j`.

<svg width="100%" viewBox="0 0 680 300" role="img" xmlns="http://www.w3.org/2000/svg">
<title>All-to-all as a matrix transpose</title>
<defs><marker id="a10" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M2 1L8 5L2 9" fill="none" stroke="#888780" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/></marker></defs>
<text x="150" y="26" text-anchor="middle" font-family="sans-serif" font-size="14" font-weight="500" fill="#444441">before (row = sender)</text>
<g fill="#EEEDFE" stroke="#7F77DD" stroke-width="0.5">
<rect x="80" y="40" width="36" height="36"/><rect x="116" y="40" width="36" height="36"/><rect x="152" y="40" width="36" height="36"/><rect x="188" y="40" width="36" height="36"/>
<rect x="80" y="76" width="36" height="36"/><rect x="116" y="76" width="36" height="36"/><rect x="152" y="76" width="36" height="36"/><rect x="188" y="76" width="36" height="36"/>
<rect x="80" y="112" width="36" height="36"/><rect x="116" y="112" width="36" height="36"/><rect x="152" y="112" width="36" height="36"/><rect x="188" y="112" width="36" height="36"/>
<rect x="80" y="148" width="36" height="36"/><rect x="116" y="148" width="36" height="36"/><rect x="152" y="148" width="36" height="36"/><rect x="188" y="148" width="36" height="36"/>
</g>
<g font-family="sans-serif" font-size="12" fill="#3C3489" text-anchor="middle">
<text x="98" y="58" dominant-baseline="central">00</text><text x="134" y="58" dominant-baseline="central">01</text><text x="170" y="58" dominant-baseline="central">02</text><text x="206" y="58" dominant-baseline="central">03</text>
<text x="98" y="94" dominant-baseline="central">10</text><text x="134" y="94" dominant-baseline="central">11</text><text x="170" y="94" dominant-baseline="central">12</text><text x="206" y="94" dominant-baseline="central">13</text>
<text x="98" y="130" dominant-baseline="central">20</text><text x="134" y="130" dominant-baseline="central">21</text><text x="170" y="130" dominant-baseline="central">22</text><text x="206" y="130" dominant-baseline="central">23</text>
<text x="98" y="166" dominant-baseline="central">30</text><text x="134" y="166" dominant-baseline="central">31</text><text x="170" y="166" dominant-baseline="central">32</text><text x="206" y="166" dominant-baseline="central">33</text>
</g>
<text x="152" y="200" text-anchor="middle" font-family="sans-serif" font-size="12" fill="#5F5E5A">rank i's row j = piece sent to rank j</text>
<line x1="260" y1="112" x2="420" y2="112" stroke="#888780" stroke-width="1.5" marker-end="url(#a10)"/>
<text x="340" y="98" text-anchor="middle" font-family="sans-serif" font-size="12" fill="#5F5E5A">all-to-all</text>
<text x="530" y="26" text-anchor="middle" font-family="sans-serif" font-size="14" font-weight="500" fill="#444441">after (row = receiver)</text>
<g fill="#E1F5EE" stroke="#1D9E75" stroke-width="0.5">
<rect x="460" y="40" width="36" height="36"/><rect x="496" y="40" width="36" height="36"/><rect x="532" y="40" width="36" height="36"/><rect x="568" y="40" width="36" height="36"/>
<rect x="460" y="76" width="36" height="36"/><rect x="496" y="76" width="36" height="36"/><rect x="532" y="76" width="36" height="36"/><rect x="568" y="76" width="36" height="36"/>
<rect x="460" y="112" width="36" height="36"/><rect x="496" y="112" width="36" height="36"/><rect x="532" y="112" width="36" height="36"/><rect x="568" y="112" width="36" height="36"/>
<rect x="460" y="148" width="36" height="36"/><rect x="496" y="148" width="36" height="36"/><rect x="532" y="148" width="36" height="36"/><rect x="568" y="148" width="36" height="36"/>
</g>
<g font-family="sans-serif" font-size="12" fill="#085041" text-anchor="middle">
<text x="478" y="58" dominant-baseline="central">00</text><text x="514" y="58" dominant-baseline="central">10</text><text x="550" y="58" dominant-baseline="central">20</text><text x="586" y="58" dominant-baseline="central">30</text>
<text x="478" y="94" dominant-baseline="central">01</text><text x="514" y="94" dominant-baseline="central">11</text><text x="550" y="94" dominant-baseline="central">21</text><text x="586" y="94" dominant-baseline="central">31</text>
<text x="478" y="130" dominant-baseline="central">02</text><text x="514" y="130" dominant-baseline="central">12</text><text x="550" y="130" dominant-baseline="central">22</text><text x="586" y="130" dominant-baseline="central">32</text>
<text x="478" y="166" dominant-baseline="central">03</text><text x="514" y="166" dominant-baseline="central">13</text><text x="550" y="166" dominant-baseline="central">23</text><text x="586" y="166" dominant-baseline="central">33</text>
</g>
<text x="530" y="200" text-anchor="middle" font-family="sans-serif" font-size="12" fill="#5F5E5A">rank i's row j = piece received from rank j</text>
<text x="340" y="250" text-anchor="middle" font-family="sans-serif" font-size="12" fill="#5F5E5A">Columns of the before-grid become rows of the after-grid</text>
<text x="340" y="270" text-anchor="middle" font-family="sans-serif" font-size="12" fill="#5F5E5A">In MoE dispatch, cell (i,j) = "how many of rank i's tokens go to rank j's expert"</text>
</svg>

*Figure 12: All-to-all is a transpose of the send matrix. MoE dispatch is this operation, with token payloads in place of the indices shown.*

### 5.2 The cost model

The standard model decomposes the time of a communication step into two terms that behave differently:

```
time  ≈  (number of messages) · α  +  (bytes moved) · β
```

where `α` is a fixed per-message latency, incurred regardless of payload size, and `β` is the inverse of bandwidth, the cost per byte. An operation is latency-bound when the first term dominates and bandwidth-bound when the second does. Which regime an operation falls into depends on the algorithm used to implement it, and not only on the collective chosen.

### 5.3 Ring all-reduce

All-reduce shows how much the algorithm matters. Two implementations produce identical results at very different cost.

A naive implementation reduces onto a root rank and broadcasts the result back. The root receives a full `n`-byte contribution from each of the other `P - 1` ranks, so the busiest rank moves `(P - 1) · n` bytes, growing linearly in `P`.

The ring algorithm passes data around a cycle. It proceeds in two phases, a reduce-scatter followed by an all-gather, each costing `(P - 1)/P · n` bytes per rank, for a total of

```
2 (P - 1) / P · n
```

which approaches `2n` from below as `P` grows and is effectively independent of the number of ranks.

<svg width="100%" viewBox="0 0 680 390" role="img" xmlns="http://www.w3.org/2000/svg">
<title>Naive versus ring all-reduce cost</title>
<text x="340" y="26" text-anchor="middle" font-family="sans-serif" font-size="14" font-weight="500" fill="#444441">Bytes moved per rank (× message size n)</text>
<rect x="108" y="276.5" width="28" height="3.5" rx="2" fill="#FAECE7" stroke="#D85A30" stroke-width="0.5"/>
<rect x="144" y="276.5" width="28" height="3.5" rx="2" fill="#E1F5EE" stroke="#1D9E75" stroke-width="0.5"/>
<text x="122" y="266" text-anchor="middle" font-family="sans-serif" font-size="12" fill="#5F5E5A">1n</text>
<text x="158" y="266" text-anchor="middle" font-family="sans-serif" font-size="12" fill="#5F5E5A">1n</text>
<rect x="258" y="255.5" width="28" height="24.5" rx="2" fill="#FAECE7" stroke="#D85A30" stroke-width="0.5"/>
<rect x="294" y="273.9" width="28" height="6.1" rx="2" fill="#E1F5EE" stroke="#1D9E75" stroke-width="0.5"/>
<text x="272" y="247" text-anchor="middle" font-family="sans-serif" font-size="12" fill="#5F5E5A">7n</text>
<text x="308" y="265" text-anchor="middle" font-family="sans-serif" font-size="12" fill="#5F5E5A">1.75n</text>
<rect x="408" y="171.5" width="28" height="108.5" rx="2" fill="#FAECE7" stroke="#D85A30" stroke-width="0.5"/>
<rect x="444" y="273.2" width="28" height="6.8" rx="2" fill="#E1F5EE" stroke="#1D9E75" stroke-width="0.5"/>
<text x="422" y="163" text-anchor="middle" font-family="sans-serif" font-size="12" fill="#5F5E5A">31n</text>
<text x="458" y="265" text-anchor="middle" font-family="sans-serif" font-size="12" fill="#5F5E5A">1.94n</text>
<rect x="558" y="59.5" width="28" height="220.5" rx="2" fill="#FAECE7" stroke="#D85A30" stroke-width="0.5"/>
<rect x="594" y="273.1" width="28" height="6.9" rx="2" fill="#E1F5EE" stroke="#1D9E75" stroke-width="0.5"/>
<text x="572" y="51" text-anchor="middle" font-family="sans-serif" font-size="12" fill="#5F5E5A">63n</text>
<text x="608" y="265" text-anchor="middle" font-family="sans-serif" font-size="12" fill="#5F5E5A">1.97n</text>
<text x="140" y="300" text-anchor="middle" font-family="sans-serif" font-size="12" fill="#5F5E5A">P=2</text>
<text x="290" y="300" text-anchor="middle" font-family="sans-serif" font-size="12" fill="#5F5E5A">P=8</text>
<text x="440" y="300" text-anchor="middle" font-family="sans-serif" font-size="12" fill="#5F5E5A">P=32</text>
<text x="590" y="300" text-anchor="middle" font-family="sans-serif" font-size="12" fill="#5F5E5A">P=64</text>
<rect x="220" y="330" width="16" height="12" rx="3" fill="#FAECE7" stroke="#D85A30" stroke-width="0.5"/>
<text x="242" y="339" font-family="sans-serif" font-size="12" fill="#5F5E5A">naive</text>
<rect x="340" y="330" width="16" height="12" rx="3" fill="#E1F5EE" stroke="#1D9E75" stroke-width="0.5"/>
<text x="362" y="339" font-family="sans-serif" font-size="12" fill="#5F5E5A">ring</text>
<text x="340" y="368" text-anchor="middle" font-family="sans-serif" font-size="12" fill="#5F5E5A">Ring cost is nearly flat in P; naive cost grows linearly</text>
</svg>

*Figure 13: Bytes moved per rank under naive and ring all-reduce, as the group size grows.*

Ring all-reduce is the default in most distributed training frameworks for this reason. The ring requires `2(P - 1)` sequential steps, so the accumulated `α` term can dominate for small messages, where a tree-based algorithm with `O(log P)` depth is faster. Production implementations such as NCCL select an algorithm based on message size.

### 5.4 Cost summary

**Table 3: Bandwidth cost per rank under bandwidth-optimal algorithms. Here `n` denotes the size of the full buffer being reduced or gathered.**

| Operation | Bytes moved per rank | Messages |
|---|---|---|
| Broadcast, Reduce | approximately `n` | `O(log P)` for tree algorithms |
| Scatter, Gather | approximately `n` | `O(log P)` for tree algorithms |
| All-gather | `(P - 1)/P · n` | `P - 1` in ring form |
| Reduce-scatter | `(P - 1)/P · n` | `P - 1` in ring form |
| All-reduce (ring) | `2 (P - 1)/P · n` | `2(P - 1)` |
| All-reduce (naive) | `(P - 1) · n` on the root | `P - 1` |
| All-to-all | `(P - 1)/P · n` | `P - 1` distinct messages |

All-gather is the one operation whose cost grows with `P` unavoidably, since the combined result is `P` times a single rank's contribution and no algorithm can avoid delivering that much new information.

All-to-all is bandwidth-comparable to reduce-scatter, but it requires `P - 1` separate messages. In MoE dispatch many of those messages carry small payloads, since a given rank may route only a few tokens to a given distant expert. The fixed `α` term is then charged `P - 1` times against a small `β` term, and the operation becomes latency-bound.

NanoCP's routing-based communication backend targets this term. A standard collective imposes the full `P × P` message structure regardless of how much real traffic each pair carries. A routing table that issues transfers only for the pairs with genuine payloads reduces the message count directly, which a general-purpose collective library cannot do on the application's behalf.

## 6. Conclusion

The two systems that motivated this article each attack one of the exceptions identified in Section 1.

NanoCP attacks the sequence-axis exception. A uniform context-parallel degree forces short requests to pay for a cross-device attention exchange they do not need, and binding a request's KV cache to the same instance that performs its MoE dispatch makes it impossible to balance attention load and dispatch load at the same time. NanoCP decouples those two bindings and sizes the context-parallel degree per request.

UltraEP attacks the dispatch exception. Expert popularity is non-stationary at the granularity of a single microbatch, so any placement computed from historical statistics is already stale when it is used. UltraEP solves a replication and rerouting plan from the exact post-gating load, on the critical path, every layer.

Both systems reduce to decisions about which collective to invoke, over which group, and how sparsely. The cost model of Section 5 prices those decisions. NanoCP's routing backend reduces the message count of the all-to-all, and UltraEP's replication reduces the imbalance that makes the slowest rank, rather than the average rank, determine the cost of the collective.

## Appendix A: A reference implementation

The following is a complete forward pass, from token identifiers to next-token logits, through two transformer layers, using the toy dimensions carried through this article (`d_model = 8`, `n_heads = 2`, `T = 4`, `d_ff = 16`). It uses NumPy only, and prints the shape and value of every intermediate quantity, including the masked score matrix of Section 2.3.

```python
import numpy as np

np.random.seed(42)
np.set_printoptions(precision=2, suppress=True)

vocab_size = 6
d_model    = 8
n_heads    = 2
d_head     = d_model // n_heads
d_ff       = 16
T          = 4      # sequence length
num_layers = 2

def section(title):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)

def layernorm(x, eps=1e-5):
    mu  = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    return (x - mu) / np.sqrt(var + eps)

def softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)

def gelu(x):
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x ** 3)))

# causal mask: query position i may only see key position j <= i
causal_mask = np.triu(np.ones((T, T)), k=1).astype(bool)   # True = masked out

# ---------------------------------------------------------------------
# STAGE 0: token ids -> embeddings -> + positional embedding
# ---------------------------------------------------------------------
section("STAGE 0: token ids -> embedding -> + positional embedding")

token_ids = np.array([2, 0, 4, 1])          # toy input sequence, length T=4
print("token_ids:", token_ids, " shape:", token_ids.shape)

embed_table = np.random.randn(vocab_size, d_model) * 0.5
x = embed_table[token_ids]                  # (T, d_model)
print("\nx = embedding lookup, shape:", x.shape)
print(x)

pos_embed = np.random.randn(T, d_model) * 0.1
x = x + pos_embed
print("\nx after + positional embedding, shape:", x.shape)
print(x)

# ---------------------------------------------------------------------
# Transformer layers
# ---------------------------------------------------------------------
for layer in range(num_layers):

    section(f"LAYER {layer} -- pre-attention LayerNorm")
    x_ln = layernorm(x)
    print("x_ln shape:", x_ln.shape)
    print(x_ln)

    section(f"LAYER {layer} -- project to Q, K, V  (x_ln @ W_Q / W_K / W_V)")
    W_Q = np.random.randn(d_model, d_model) * 0.3
    W_K = np.random.randn(d_model, d_model) * 0.3
    W_V = np.random.randn(d_model, d_model) * 0.3
    W_O = np.random.randn(d_model, d_model) * 0.3

    Q = x_ln @ W_Q     # (T, d_model)
    K = x_ln @ W_K
    V = x_ln @ W_V
    print("Q shape:", Q.shape, "\n", Q)
    print("\nK shape:", K.shape, "\n", K)
    print("\nV shape:", V.shape, "\n", V)

    section(f"LAYER {layer} -- reshape last dim into (n_heads, d_head)")
    Qh = Q.reshape(T, n_heads, d_head).transpose(1, 0, 2)   # (n_heads, T, d_head)
    Kh = K.reshape(T, n_heads, d_head).transpose(1, 0, 2)
    Vh = V.reshape(T, n_heads, d_head).transpose(1, 0, 2)
    print("Qh shape:", Qh.shape, " = (n_heads, T, d_head)")
    print("Head 0 Q (T,d_head):\n", Qh[0])
    print("Head 1 Q (T,d_head):\n", Qh[1])

    section(f"LAYER {layer} -- Q @ Kᵀ -> scores (T,T), causal masked, softmax")
    head_outputs = []
    for h in range(n_heads):
        scores = (Qh[h] @ Kh[h].T) / np.sqrt(d_head)        # (T, T)
        scores_masked = np.where(causal_mask, -1e9, scores)
        print(f"\n-- Head {h} --")
        print("raw scores (T,T):\n", scores)
        print("causal-masked scores:\n", scores_masked)

        weights = softmax(scores_masked, axis=-1)
        print("softmax weights (each row sums to 1):\n", weights)
        print("row sums:", weights.sum(axis=-1))

        out_h = weights @ Vh[h]                              # (T, d_head)
        print(f"head {h} output = weights @ V, shape {out_h.shape}:\n", out_h)
        head_outputs.append(out_h)

    section(f"LAYER {layer} -- concat heads, x W_O, residual add")
    concat = np.concatenate(head_outputs, axis=-1)           # (T, d_model)
    print("concat heads shape:", concat.shape, "\n", concat)
    attn_out = concat @ W_O
    print("\nattn_out = concat @ W_O, shape:", attn_out.shape, "\n", attn_out)

    x = x + attn_out
    print("\nx after residual add, shape:", x.shape, "\n", x)

    section(f"LAYER {layer} -- pre-FFN LayerNorm -> FFN (up, GELU, down) -> residual")
    x_ln2 = layernorm(x)
    W_up   = np.random.randn(d_model, d_ff) * 0.3
    W_down = np.random.randn(d_ff, d_model) * 0.3

    hidden = gelu(x_ln2 @ W_up)
    print("hidden = gelu(x_ln2 @ W_up), shape:", hidden.shape, "\n", hidden)

    ffn_out = hidden @ W_down
    print("\nffn_out = hidden @ W_down, shape:", ffn_out.shape, "\n", ffn_out)

    x = x + ffn_out
    print("\nx after FFN residual add, shape:", x.shape, "\n", x)

# ---------------------------------------------------------------------
# Final: LayerNorm -> unembed -> logits -> next-token probabilities
# ---------------------------------------------------------------------
section("FINAL: LayerNorm -> unembed -> logits -> next-token probabilities")

x_final = layernorm(x)
W_unembed = np.random.randn(d_model, vocab_size) * 0.3
logits = x_final @ W_unembed                                 # (T, vocab_size)
print("logits shape:", logits.shape, "\n", logits)

probs = softmax(logits, axis=-1)
print("\nnext-token probabilities per position (rows sum to 1):\n", probs)
print("row sums:", probs.sum(axis=-1))

next_token_pred = probs.argmax(axis=-1)
print("\nargmax predicted token id at every position:", next_token_pred)
print("only the LAST position's row is the actual 'next token' prediction:",
      next_token_pred[-1])
```

Two details in the output correspond to Section 2.3. Row 0 of the softmax weights collapses to `[1, 0, 0, 0]`, since the first token can attend only to itself under the causal mask. Row 3 is the only row with a nonzero weight in all four columns, since the last query is the only one permitted to see the entire sequence.

## Appendix B: What the feed-forward network computes

Section 2.5 gives the structural facts about the FFN that the main argument requires. This appendix covers what the component computes, which is a separate question and one the interpretability literature has answered in some detail.

### B.1 The key-value memory interpretation

Geva et al. (EMNLP 2021) show that each of the `d_ff` inner units behaves like one entry of a learned associative memory. The columns of `W_up` act as keys, pattern detectors that a token's vector is compared against by dot product. The activation function determines how strongly each pattern fires. The rows of `W_down` act as values, the vectors added to the residual stream when the corresponding pattern fires. The authors report that the learned patterns are human-interpretable, with lower layers capturing shallower patterns and upper layers capturing more semantic ones.

<svg width="100%" viewBox="0 0 680 380" role="img" xmlns="http://www.w3.org/2000/svg">
<title>The FFN as a key-value memory</title>
<defs><marker id="a7" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M2 1L8 5L2 9" fill="none" stroke="#888780" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/></marker></defs>
<rect x="240" y="40" width="200" height="44" rx="8" fill="#F1EFE8" stroke="#888780" stroke-width="0.5"/>
<text x="340" y="62" text-anchor="middle" dominant-baseline="central" font-family="sans-serif" font-size="14" font-weight="500" fill="#444441">token: "Paris"</text>
<line x1="340" y1="84" x2="110" y2="110" stroke="#888780" stroke-width="1.5" marker-end="url(#a7)"/>
<line x1="340" y1="84" x2="263" y2="110" stroke="#888780" stroke-width="1.5" marker-end="url(#a7)"/>
<line x1="340" y1="84" x2="416" y2="110" stroke="#888780" stroke-width="1.5" marker-end="url(#a7)"/>
<line x1="340" y1="84" x2="569" y2="110" stroke="#888780" stroke-width="1.5" marker-end="url(#a7)"/>
<rect x="40" y="110" width="140" height="56" rx="8" fill="#FAEEDA" stroke="#BA7517" stroke-width="0.5"/>
<text x="110" y="128" text-anchor="middle" dominant-baseline="central" font-family="sans-serif" font-size="14" font-weight="500" fill="#633806">capital cities</text>
<text x="110" y="146" text-anchor="middle" dominant-baseline="central" font-family="sans-serif" font-size="12" fill="#633806">fires strongly</text>
<rect x="193" y="110" width="140" height="56" rx="8" fill="#F1EFE8" stroke="#888780" stroke-width="0.5"/>
<text x="263" y="128" text-anchor="middle" dominant-baseline="central" font-family="sans-serif" font-size="14" font-weight="500" fill="#444441">past tense</text>
<text x="263" y="146" text-anchor="middle" dominant-baseline="central" font-family="sans-serif" font-size="12" fill="#5F5E5A">barely fires</text>
<rect x="346" y="110" width="140" height="56" rx="8" fill="#F1EFE8" stroke="#888780" stroke-width="0.5"/>
<text x="416" y="128" text-anchor="middle" dominant-baseline="central" font-family="sans-serif" font-size="14" font-weight="500" fill="#444441">numbers</text>
<text x="416" y="146" text-anchor="middle" dominant-baseline="central" font-family="sans-serif" font-size="12" fill="#5F5E5A">barely fires</text>
<rect x="499" y="110" width="140" height="56" rx="8" fill="#FAEEDA" stroke="#BA7517" stroke-width="0.5"/>
<text x="569" y="128" text-anchor="middle" dominant-baseline="central" font-family="sans-serif" font-size="14" font-weight="500" fill="#633806">place names</text>
<text x="569" y="146" text-anchor="middle" dominant-baseline="central" font-family="sans-serif" font-size="12" fill="#633806">fires strongly</text>
<line x1="110" y1="166" x2="340" y2="230" stroke="#BA7517" stroke-width="3" opacity="0.85"/>
<line x1="569" y1="166" x2="340" y2="230" stroke="#BA7517" stroke-width="3" opacity="0.85"/>
<line x1="263" y1="166" x2="340" y2="230" stroke="#888780" stroke-width="1" opacity="0.35"/>
<line x1="416" y1="166" x2="340" y2="230" stroke="#888780" stroke-width="1" opacity="0.35"/>
<rect x="190" y="230" width="300" height="56" rx="8" fill="#E1F5EE" stroke="#1D9E75" stroke-width="0.5"/>
<text x="340" y="248" text-anchor="middle" dominant-baseline="central" font-family="sans-serif" font-size="14" font-weight="500" fill="#085041">weighted sum of active values</text>
<text x="340" y="266" text-anchor="middle" dominant-baseline="central" font-family="sans-serif" font-size="12" fill="#085041">added into the residual stream</text>
<text x="340" y="318" text-anchor="middle" font-family="sans-serif" font-size="12" fill="#5F5E5A">W_up supplies the keys (patterns), W_down the values (what to add)</text>
<text x="340" y="338" text-anchor="middle" font-family="sans-serif" font-size="12" fill="#5F5E5A">All d_ff units are evaluated in parallel; the patterns are learned, not designed</text>
<text x="340" y="358" text-anchor="middle" font-family="sans-serif" font-size="12" fill="#5F5E5A">After Geva et al., EMNLP 2021 and 2022; labels illustrative</text>
</svg>

*Figure 14: The FFN as a bank of learned pattern detectors. The labels are illustrative. No unit is designed to detect a given concept.*

This interpretation applies only to trained weights. In the reference implementation of Appendix A, `W_up` and `W_down` are random and the inner units mean nothing. The claim is that gradient descent discovers such detectors, because they are an effective way of reducing next-token prediction loss.

### B.2 What the value vectors do

A follow-up (Geva et al., EMNLP 2022) makes the function of the value vectors concrete. Projecting an individual value vector through the model's unembedding matrix, the same matrix that converts a final hidden state into vocabulary logits, shows that it promotes a coherent cluster of related tokens. One value vector may raise the probability of a group of geographically related words, another the probability of a group of arithmetic-related words.

The FFN's contribution to the residual stream is therefore a set of additive, direction-specific votes over the output vocabulary, weighted by how strongly each corresponding pattern matched the token. The authors demonstrate the causal force of this reading by suppressing value vectors associated with undesired token clusters and observing the corresponding drop in those outputs.

### B.3 The residual stream

Each layer reads from the residual stream and writes back by addition. Elhage et al. (2021) describe the residual stream as a communication channel between layers, into which each block writes a linear projection of its output while leaving prior contributions intact.

Given the reading in B.2, addition preserves the accumulated votes of every previous layer. Under replacement, layer 40 could not build on anything layer 3 established, and intermediate representations would carry no interpretable signal. The additivity is also what makes the logit lens possible: applying the unembedding matrix to an intermediate residual state yields a meaningful, if rough, prediction, because that state already contains the summed contributions of all preceding layers.

### B.4 Relation to attention

Geva et al. write the FFN formula to mirror attention's, and the two follow the same match, weight, blend template:

```
attention:  output_i = Σ_j  softmax(Q_i · K_j)      · V_j
FFN:        output_i = Σ_c  GELU(x_i · W_up[:, c])  · W_down[c, :]
```

Two differences matter. Attention's keys are dynamic: `K = x @ W_K` is recomputed from the surrounding context on every forward pass. The FFN's keys are static, fixed at the end of training. This is why the FFN requires no cross-token communication while attention's score step does, and it restates the criterion of Section 2.4 in a second setting. Attention's weights are also normalized by a softmax and must sum to one across positions, where the FFN's activations are unconstrained: all `d_ff` units may fire strongly, or none may.

## References

The two papers this article was written alongside:

- Chen, J. et al. "NanoCP: Request-Level Dynamic Context Parallelism for Data-Expert Parallel Decoding." [arXiv:2605.21100](https://arxiv.org/abs/2605.21100)
- Wei, X. et al. "UltraEP: Unleash MoE Training and Inference on Rack-Scale Nodes with Near-Optimal Load Balancing." [arXiv:2606.04101](https://arxiv.org/abs/2606.04101)

Attention, tensor parallelism, and MLA:

- Shoeybi, M. et al. "Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism." [arXiv:2104.04473](https://arxiv.org/pdf/2104.04473)
- Brenndoerfer, M. "Tensor Parallelism: Column, Row, and Megatron Patterns." [mbrenndoerfer.com](https://mbrenndoerfer.com/writing/tensor-parallelism-column-row-megatron-communication-patterns)
- Jang, I. "Analyzing Parallelization of Attention." [insujang.github.io](https://insujang.github.io/2022-08-03/analyzing-parallelization-of-attention/)
- DeepSeek-AI. "DeepSeek-V2 Technical Report." [arXiv:2405.04434](https://arxiv.org/pdf/2405.04434)
- Vizuara. "Decoding Multi-Head Latent Attention," Parts 1 and 2. [vizuara.substack.com](https://vizuara.substack.com/p/decoding-multi-head-latent-attention)
- Towards Data Science. "DeepSeek-V3 Explained 1: Multi-head Latent Attention." [towardsdatascience.com](https://towardsdatascience.com/deepseek-v3-explained-1-multi-head-latent-attention-ed6bee2a67c4/)
- Bhatia, N. et al. "Helix Parallelism: Rethinking Sharding Strategies for Interactive Multi-Million-Token LLM Decoding." [arXiv:2507.07120](https://arxiv.org/html/2507.07120v1)
- vLLM GitHub. RFC for Helix Parallelism implementation. [github.com/vllm-project/vllm/issues/34018](https://github.com/vllm-project/vllm/issues/34018)
- ROCm. "The vLLM MoE Playbook." [rocm.blogs.amd.com](https://rocm.blogs.amd.com/software-tools-optimization/vllm-moe-guide/README.html)
- vLLM Docs. "Data Parallel Deployment." [docs.vllm.ai](https://docs.vllm.ai/en/latest/serving/data_parallel_deployment/)
- Jarvislabs. "Scaling LLM Inference: DP, PP, TP." [docs.jarvislabs.ai](https://docs.jarvislabs.ai/blog/scaling-llm-inference-dp-pp-tp)

Context parallelism and the online-softmax merge:

- Dao, T. et al. "Flash-Decoding for long-context inference." [pytorch.org/blog](https://pytorch.org/blog/flash-decoding/)
- Meta AI. "Context Parallelism for Scalable Million-Token Inference." [arXiv:2411.01783](https://arxiv.org/pdf/2411.01783)
- HuggingFace Blog. "FlashAttention: Online Softmax." [huggingface.co/blog](https://huggingface.co/blog/atharv6f/flash-attention-online-softmax)

Serving architecture, memory, and scheduling:

- Kwon, W. et al. "Efficient Memory Management for Large Language Model Serving with PagedAttention," explained by Brenndoerfer, M. [mbrenndoerfer.com](https://mbrenndoerfer.com/writing/paged-attention-vllm-kv-cache-memory-management)
- Zhong, Y. et al. "DistServe: Disaggregating Prefill and Decoding for Goodput-Optimized Large Language Model Serving." [arXiv:2401.09670](https://arxiv.org/abs/2401.09670)
- Agrawal, A. et al. "Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve." OSDI 2024.
- NVIDIA. "GB200 NVL72." [nvidia.com](https://www.nvidia.com/en-us/data-center/gb200-nvl72/)
- Introl. "NVLink and Scale-Up Networking." [introl.com](https://introl.com/blog/nvlink-scale-up-networking-gpu-interconnect-infrastructure-2025)

Stragglers and MoE communication:

- Ho, Q. et al. "Solving the Straggler Problem for Iterative Convergent Parallel ML." CMU-PDL Technical Report. [pdl.cmu.edu](https://www.pdl.cmu.edu/PDL-FTP/BigLearning/CMU-PDL-15-102.pdf)
- Zuo, P. et al. "Serving Large Language Models on Huawei CloudMatrix384." [arXiv:2508.02520](https://arxiv.org/pdf/2508.02520)
- DeepSeek-AI. "DeepEP: an efficient expert-parallel communication library." [github.com/deepseek-ai/DeepEP](https://github.com/deepseek-ai/DeepEP/blob/main/README.md)
- DeepSeek-AI. "EPLB: Expert Parallelism Load Balancer." [github.com/deepseek-ai/EPLB](https://github.com/deepseek-ai/EPLB/blob/main/README.md)
- ROCm. "Dropless MoE Training with Primus-Turbo." [rocm.blogs.amd.com](https://rocm.blogs.amd.com/software-tools-optimization/maxtext-dropless-moe/README.html)

FFN interpretability and the residual stream:

- Geva, M., Schuster, R., Berant, J., Levy, O. "Transformer Feed-Forward Layers Are Key-Value Memories." EMNLP 2021. [aclanthology.org](https://aclanthology.org/2021.emnlp-main.446/)
- Geva, M., Caciularu, A., Wang, K., Goldberg, Y. "Transformer Feed-Forward Layers Build Predictions by Promoting Concepts in the Vocabulary Space." EMNLP 2022. [aclanthology.org](https://aclanthology.org/2022.emnlp-main.3/)
- Elhage, N. et al. "A Mathematical Framework for Transformer Circuits." Transformer Circuits Thread, 2021. [transformer-circuits.pub](https://transformer-circuits.pub/2021/framework/index.html)
- "MLPs in Transformers." Learn Mechanistic Interpretability. [learnmechinterp.com](https://learnmechinterp.com/topics/mlps-in-transformers/)
