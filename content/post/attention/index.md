---
title: "Attention, Parallelism, and MoE Communication: A Ground-Up Walkthrough"
date: 2026-07-13
tags:
- Machine Learning
- Transformers
- Deep Learning
- Distributed Systems
- Mixture of Experts
- Attention
- Parallelism
summary: "A question-driven, ground-up walkthrough of transformer internals, DP/TP/CP/PP parallelism, MoE dispatch and combine, and the collective operations underneath it all, written while reading the NanoCP and UltraEP papers."
---


## Part 1: Inside one attention block

### What is `W_K`, really? Where does it sit in the pipeline?

A token's "hidden state" going into an attention block is a vector of length `d_model`. A batch of `B` sequences, each `T` tokens long, gives a tensor of shape `(B, T, d_model)`. `W_K` is just a plain, non-head-aware `(d_model, d_model)` matrix. Multiplying `x @ W_K` gives back a flat `(B, T, d_model)` tensor with no visible head structure yet. Q and V are produced the same way, each with their own separate weight matrix.

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
<text x="340" y="230" text-anchor="middle" dominant-baseline="central" font-family="sans-serif" font-size="12" fill="#633806">every query row meets every key column, HERE</text>
<line x1="340" y1="254" x2="340" y2="278" stroke="#888780" stroke-width="1.5" marker-end="url(#a1)"/>
<rect x="140" y="278" width="400" height="56" rx="8" fill="#E1F5EE" stroke="#1D9E75" stroke-width="0.5"/>
<text x="340" y="296" text-anchor="middle" dominant-baseline="central" font-family="sans-serif" font-size="14" font-weight="500" fill="#085041">softmax(scores) → weights (4,4)</text>
<text x="340" y="314" text-anchor="middle" dominant-baseline="central" font-family="sans-serif" font-size="12" fill="#085041">weights @ V → head output (T,d_head)=(4,4)</text>
<line x1="340" y1="334" x2="340" y2="358" stroke="#888780" stroke-width="1.5" marker-end="url(#a1)"/>
<rect x="140" y="358" width="400" height="56" rx="8" fill="#F1EFE8" stroke="#888780" stroke-width="0.5"/>
<text x="340" y="376" text-anchor="middle" dominant-baseline="central" font-family="sans-serif" font-size="14" font-weight="500" fill="#444441">concat all heads + × W_O</text>
<text x="340" y="394" text-anchor="middle" dominant-baseline="central" font-family="sans-serif" font-size="12" fill="#5F5E5A">→ final output (T,d_model) = (4,8)</text>
</svg>

The only step where token positions actually interact is the amber box: `Q @ Kᵀ`. Everything else (projections, softmax normalization, output projection) is per-token or per-head local. That single fact turns out to explain almost everything else in this post.

### How does "splitting heads" actually work mechanically?

Reshaping `d_model = n_heads × d_head` is pure relabeling, no computation. Take one token's 8-number `K`-vector: chopping it into two groups of 4 consecutive numbers *is* "head 0" and "head 1." Nothing forces that split point; it's just a convention that has to be applied consistently to Q, K, and V so head 0's pieces line up across all three.

Because the split happens on the **output columns of the weight matrix itself**, you can literally hand GPU 0 the left half of `W_K`'s columns and GPU 1 the right half. Each GPU computes its own head from scratch, with zero communication needed until the very end.

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
<text x="340" y="400" text-anchor="middle" font-family="sans-serif" font-size="12" fill="#5F5E5A">Each GPU caches only its own heads, no duplication</text>
</svg>

For TP's combine step: each GPU multiplies its head output by its own **row**-slice of `W_O` (not the full matrix), producing a partial `d_model`-sized output, and the true answer is the **sum** of those partials, not a concatenation. So the collective that combines TP's two GPUs is an **all-reduce**, not an all-gather: a distinction that matters later.

### Why doesn't MLA (DeepSeek's attention variant) shard cleanly this way?

Standard multi-head attention has a natural per-head cache: head 1's K/V lives in its own slice. MLA throws that away: it compresses K and V into **one shared latent vector per token**, and every head reconstructs its own K/V from that *same* shared vector via a per-head up-projection. That compression is the entire reason MLA's KV cache is so small.

The problem: if you TP-shard by head, every GPU still needs the *entire* shared latent to reconstruct even its own heads. There's no per-head slice to hand out, because the compression collapsed it into one shared thing.

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
<text x="340" y="384" text-anchor="middle" font-family="sans-serif" font-size="12" fill="#5F5E5A">Identical copies. TP didn't split the cache at all</text>
<text x="340" y="404" text-anchor="middle" font-family="sans-serif" font-size="12" fill="#5F5E5A">With TP=8, that's 8 full copies of c_KV, and MLA's savings are erased</text>
<text x="340" y="428" text-anchor="middle" font-family="sans-serif" font-size="12" fill="#5F5E5A">This is why DeepSeek-style models use DP, not TP, for attention</text>
</svg>

This is why modern MoE serving stacks (and both papers this post is built around) run **DP-EP**, not TP-EP: DP replicates attention entirely (no sharing, no duplication problem), and EP is a completely different axis (sharding *experts*, not KV cache) used for the FFN.

---

## Part 2: DP, TP, CP, PP: which axis does each one cut?

Once Q/K/V are reshaped to `(B, T, n_heads, d_head)`, every axis of that tensor is a candidate for a different parallelism strategy:

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
<text x="340" y="150" text-anchor="middle" font-family="sans-serif" font-size="12" fill="#5F5E5A">d_head can't be split without splitting a single dot product</text>
<text x="340" y="170" text-anchor="middle" font-family="sans-serif" font-size="12" fill="#5F5E5A">mid-computation, that's the axis that needs communication</text>
</svg>

The rule that decides all four: an axis needs communication only when the computation must **combine information across** different slices of it. `B` and `n_heads` are axes attention treats as fully independent (batch elements never interact; heads never interact until the very last step). `T` is the axis attention is *defined* to reduce over: a query needs to see keys from potentially every position, so cutting `T` genuinely separates data the computation needs together. `d_head` would split a single dot product mid-sum, essentially never attempted.

Ranking these by actual communication cost gives `B < n_heads < T < d_head`, **not** simply "deeper in the tensor shape = more communication." `n_heads` (TP) needs one cheap combine step at the very end; `T` (CP) needs genuine ongoing exchange, which is exactly why CP is a whole research subfield and TP mostly isn't.

### How does CP's communication actually work?

Splitting `T` for the Q/K/V *projection* is exactly as free as DP splitting `B`: the projection is per-token, so a GPU can compute Q/K/V for its own local tokens with zero communication. The break happens one step later, at the score computation, because that's the one operation whose entire job is "let every token look at every other token":

<svg width="100%" viewBox="0 0 680 290" role="img" xmlns="http://www.w3.org/2000/svg">
<title>CP is silent like DP until attention needs to look across tokens</title>
<defs><marker id="a4" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M2 1L8 5L2 9" fill="none" stroke="#888780" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/></marker></defs>
<text x="340" y="30" text-anchor="middle" font-family="sans-serif" font-size="14" font-weight="500" fill="#444441">CP splits T: silent until attention needs to look across it</text>
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
<text x="340" y="200" text-anchor="middle" font-family="sans-serif" font-size="12" fill="#5F5E5A">same as DP so far, zero communication</text>
<text x="340" y="228" text-anchor="middle" font-family="sans-serif" font-size="12" fill="#5F5E5A">attention itself needs every token's K,V, not just the local shard</text>
<text x="340" y="262" text-anchor="middle" font-family="sans-serif" font-size="12" fill="#5F5E5A">→ this is the query-out / partial-back / merge step below</text>
</svg>

Two real implementations exist for that cross-boundary exchange. NanoCP/Helix route the query directly to whichever remote GPU holds the needed KV shard, get a partial result + a scaling factor back, and merge (the same online-softmax trick FlashAttention uses internally). Ring Attention does it differently: no fixed destination, just a **rotating relay**:

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

Ring trades a longer, bandwidth-friendly relay for training-scale CP groups; the routed/point-to-point style is lower-latency for the smaller, more dynamic groups decode-time serving needs, which is exactly the design choice NanoCP makes.

**On PP:** worth a note since it's easy to assume it's training-only. It's genuinely preferred over TP once GPUs span more than one fast-interconnect domain, because TP needs an all-reduce *every layer* (brutal over slow links) while PP only passes activations across one boundary per stage. Sarathi-Serve reports roughly 2× lower median latency for PP vs. TP once crossing nodes over ordinary Ethernet. This is exactly why UltraEP's DeepSeek-V3 config uses `EP64-PP4` rather than adding TP.

---

## Part 3: The toy transformer

Rather than keep everything abstract, here's a complete, runnable forward pass, embeddings through two full transformer layers to next-token logits, with the same toy dimensions used throughout (`d_model=8`, `n_heads=2`, `T=4`). Every stage prints its actual shape and values.

```python
import numpy as np

np.random.seed(42)
np.set_printoptions(precision=2, suppress=True)

# ---------------------------------------------------------------------
# Toy dimensions -- same numbers used throughout the conversation
# ---------------------------------------------------------------------
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

The masked score matrix is the part worth staring at: row 0 collapses to `[1,0,0,0]` after softmax (the first token can only attend to itself), while row 3 gets a genuine distribution across all four positions: proof the causal mask is doing exactly what it should.

---

## Part 4: What is the FFN actually *for*?

### What do `W_up` and `W_down` do, intuitively?

Shape-wise: `W_up` is `(d_model, d_ff)`: expansion. `W_down` is `(d_ff, d_model)`: contraction back to fit the residual add. But the *functional* answer comes from real interpretability research (Geva et al., EMNLP 2021, *"Transformer Feed-Forward Layers Are Key-Value Memories"*): each of the `d_ff` hidden units behaves like one entry in a lookup table. `W_up`'s columns are **keys**: learned pattern detectors a token gets dot-producted against. GELU is the gate deciding how strongly each pattern fired. `W_down`'s rows are **values**: what gets added if that pattern fired.

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
<text x="340" y="360" text-anchor="middle" font-family="sans-serif" font-size="12" fill="#5F5E5A">Without GELU, two linear layers collapse into one, no extra power</text>
<text x="340" y="380" text-anchor="middle" font-family="sans-serif" font-size="12" fill="#5F5E5A">The residual add is why ffn_out must return to d_model, not stay at d_ff</text>
</svg>

I proved the "collapse without GELU" claim rather than just asserting it: `(x @ W_up) @ W_down` computed directly is numerically identical to `x @ (W_up @ W_down)` collapsed into a single `(d_model, d_model)` matrix, and `np.allclose` returns `True`. Add GELU back in and the two stop matching entirely. The nonlinearity is the *entire* reason the wide middle layer buys any extra capacity.

### What does the "value" actually promote?

This is the part that made it click for me. A follow-up paper (Geva et al., EMNLP 2022) projects each value-vector through the model's **unembedding matrix**, the same matrix used at the very end to turn a hidden state into vocabulary logits, and finds each one promotes a coherent cluster of *actual words*:

> "A value vector might promote the cluster {Paris, France, French, European, Seine}, or the cluster {multiply, divide, arithmetic, calculate}."

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
<text x="340" y="318" text-anchor="middle" font-family="sans-serif" font-size="12" fill="#5F5E5A">W_up learns the "keys" (patterns), W_down learns the "values" (what to add)</text>
<text x="340" y="338" text-anchor="middle" font-family="sans-serif" font-size="12" fill="#5F5E5A">Every one of the d_ff neurons does this at once, learned from training data</text>
<text x="340" y="358" text-anchor="middle" font-family="sans-serif" font-size="12" fill="#5F5E5A">Source: Geva et al., EMNLP 2021 &amp; 2022</text>
</svg>

Important honesty check: this story only exists *after* training on real data. In the toy script above, `W_up`/`W_down` are random noise. There are no real "capital cities" detectors in there. The finding is that gradient descent *discovers* useful pattern-detectors on its own during training, purely because they're a useful building block for predicting the next token well.

### Why add to the residual stream instead of overwriting it?

Because the value vectors are literally votes toward specific output words, and **addition is the only operation that lets those votes accumulate instead of erasing each other.** This is Anthropic's own framing from *"A Mathematical Framework for Transformer Circuits"* (Elhage et al., 2021):

> "Rather than overwriting the results from previous layers, each layer of the residual block 'reads' its input from the residual stream... and then 'writes' its result to the residual stream by adding a linear projection back in."

If FFN layers overwrote instead of added, deep networks would be pointless: layer 40 couldn't build on anything layer 3 discovered.

### Is `W_down` doing something similar to `V` in Q/K/V?

Yes, genuinely, not just an analogy. Geva et al. deliberately wrote the FFN formula to mirror attention's: **match → weight → blend**, in both cases. Attention: `output = Σ softmax(Q·K) × V`. FFN: `output = Σ GELU(x·W_up) × W_down`. Same template.

The one real, load-bearing difference: attention's keys are **dynamic**: `K = x @ W_K`, computed fresh from whatever's in context. The FFN's keys are **static, learned parameters**, fixed the moment training ends. That's precisely why the FFN needs zero cross-token communication while attention's score step is the one place that genuinely does. (Second difference, more of a footnote: attention's weights are forced to sum to 1 via softmax; GELU has no such constraint, all `d_ff` keys can fire independently, unconstrained.)

---

## Part 5: MoE: dispatch and combine

MoE replaces the single shared FFN with many smaller, specialized ones ("experts"), plus a **gate** that decides, per token, which top-k experts should process it. The gate itself is a per-token linear layer, no communication needed yet.

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
<text x="340" y="356" text-anchor="middle" font-family="sans-serif" font-size="12" fill="#5F5E5A">e0 was picked by all 4 tokens, already a hot expert before any GPU is involved</text>
</svg>

If `e0` and `e1` live on one GPU while `e2`/`e3` live on another, each token's hidden state physically has to travel to wherever its chosen experts are (**dispatch**), get processed there, and travel back to be blended (**combine**):

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
<text x="340" y="376" text-anchor="middle" font-family="sans-serif" font-size="12" fill="#5F5E5A">No LSE needed, top-2 weights are already normalized before dispatch</text>
</svg>

Combine here is simpler than CP's merge: the gate already normalized `w0, w1` locally before dispatch, so it's a plain weighted sum, no rescaling trick required.

### Why does this create a straggler?

All 4 tokens' dispatch calls are packed into **one shared collective**, not 4 independent sends. If `e0` gets picked by every token and `e1` gets picked by fewer, the GPU hosting `e0` ends up with far more total traffic than the others, and the collective *as a whole* only completes once the busiest GPU finishes:

<svg width="100%" viewBox="0 0 680 330" role="img" xmlns="http://www.w3.org/2000/svg">
<title>Dispatch timeline showing GPU wait times</title>
<text x="598" y="46" text-anchor="end" font-family="sans-serif" font-size="12" fill="#5F5E5A">Barrier, all must arrive here</text>
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
<text x="362" y="281" font-family="sans-serif" font-size="12" fill="#5F5E5A">Idle, waiting at barrier</text>
</svg>

The idle GPUs' hardware is fully free during the dashed span. There's just nothing useful for them to do, because the next layer's computation needs *every* GPU's dispatched tokens assembled together before it can proceed at all. This is exactly the problem UltraEP solves by replicating a hot expert's weights onto a spare GPU in real time and splitting the token load across the replicas, reacting to the *exact*, current-iteration load rather than periodic, stale statistics the way its predecessor (EPLB) does.

One clarification worth keeping: **continuous batching solves a different problem than this.** It changes which requests are included *between* iterations. Within one iteration, every included token is glued into the same batched kernel calls and the same collective, that's the level the straggler problem actually lives at.

---

## Part 6: The collective operations underneath everything

Everything above eventually reduces to a handful of named collective operations. Worth knowing them by name and cost, not just by what we've called them informally.

**Broadcast / Scatter**: one rank's data either copied identically to everyone (broadcast), or cut into pieces with one distinct piece per rank (scatter). **Gather / All-Gather**: the reverse; everyone's data converges onto one rank (gather) or onto every rank (all-gather). **Reduce / All-Reduce**: same converge shape, but the center step does real work (a sum), landing on one rank or all of them:

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

**All-reduce is exactly what combined TP's two head outputs earlier.** Reduce-scatter is the same converge step, but instead of broadcasting the whole sum back, each rank keeps only its own slice, and in real implementations, **all-reduce isn't a separate primitive at all, it's reduce-scatter followed by all-gather**, chained together.

**All-to-all** is the odd one out: no converge/diverge shape at all, just every rank sending something different to every other rank simultaneously. It's literally a matrix transpose:

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

### How much does each of these actually cost?

The standard tool is the **α-β model**: total time ≈ (number of messages) × α (fixed latency per message) + (bytes moved) × β (per-byte cost). Whether an operation is latency-bound or bandwidth-bound depends on which term dominates, and that's where the choice of *algorithm*, not just which collective, matters enormously.

The single most important concrete result here: **naive all-reduce** (everyone sends to rank 0, it sums, it broadcasts back) costs `(P-1) × n` bytes on the busiest rank, linear in the number of ranks. **Ring all-reduce** (pass data around a ring instead) costs `2(P-1)/P × n`, which *approaches 2n and flatlines* as P grows, essentially independent of rank count:

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
<text x="340" y="368" text-anchor="middle" font-family="sans-serif" font-size="12" fill="#5F5E5A">Ring's cost barely grows as P scales, naive grows linearly, unbounded</text>
</svg>

That flat curve is why ring all-reduce became the default in every serious distributed training framework. The tradeoff: ring needs `2(P-1)` sequential steps, so for *tiny* messages the fixed per-message latency can dominate anyway. Real implementations (NCCL) switch strategy based on message size.

The rest, briefly: **broadcast / scatter / gather** move `~n` bytes per rank, roughly `P`-independent. **All-gather genuinely does scale with `P`**: the combined result is `P·n`, so each rank must receive `(P-1)·n` new bytes; no algorithm avoids this, since the information itself grows. **Reduce-scatter** is exactly half of ring all-reduce (the first phase). **All-to-all** is bandwidth-comparable to reduce-scatter, but its real cost is *latency*: it needs `P-1` distinct messages, and if each one carries only a handful of tokens (the common MoE case), the fixed per-message overhead `α` dominates over the actual bytes. That's the precise reason NanoCP builds a routing-based backend instead of calling a generic all-to-all: it attacks the message-count term directly, skipping pairs that have nothing real to send, rather than paying for a dense `P×P` mesh every single request.

---

## Closing thread

Zooming out: NanoCP decouples *where attention runs* from *where MoE dispatch runs* (dynamic CP degree instead of a fixed uniform group), and UltraEP reacts to *exact*, current-iteration expert load rather than periodic historical statistics, replicating a hot expert's weights and splitting its tokens across replicas before dispatch even starts. Both papers are, underneath everything, just careful choices about which of these collective primitives to use, when, and how sparsely. Once you can see that layer, the rest of the systems literature in this space reads a lot faster.
