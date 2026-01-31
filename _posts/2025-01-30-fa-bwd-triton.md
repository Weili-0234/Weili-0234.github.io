---
layout: distill
title: "Implementing Flash Attention: Backward Pass in Triton"
date: 2025-01-30 12:00:00
description: "In this follow-up post to Nathan Chen's <a href='https://nathanchen.me/public/Triton-Flash-Attention-Kernel-Walkthrough.html'>Triton Flash Attention Kernel Walkthrough: The Forward Pass</a>, we dive into gradient computation for queries, keys, and values in the backward pass."
tags: triton gpu attention mlsys
categories: technical

toc:
  - name: The Data Layout
  - name: The Big Picture - Autograd
  - name: The Launcher
  - name: Step 1 - Preprocessing Delta 
  - name: Step 2 - Computing dQ
    subsections:
      - name: Set Up
      - name: Recomputation along the main loop
      - name: The Gradient Math
  - name: Step 3 - Computing dK and dV
  - name: Acknowledgments
---

<!-- Readers are assumed to be familiar with the forward pass, including basic ideas like tiling and the `make_block_ptr` syntax. -->

So you’ve seen the forward pass in FlashAttention, and know how to compute the attention output without ever materializing the big $N \times N$ score matrix using tiling and online softmax.

But in deep learning, computing the output is only half the battle. To train the model, we need gradients. We need the **backward pass**.

Implementing a backward pass for FlashAttention is notoriously more difficult than the forward pass. Why? Because we didn't save the massive attention score matrix—it was too big. To get gradients, we effectively have to **recompute** the attention scores on the fly using the saved LogSumExp (`lse`) from the forward pass, while simultaneously propagating gradients backward.

In this walkthrough, we will dissect the backward pass in FLA's excellent [implementation](https://github.com/fla-org/flash-linear-attention/blob/f0de028bf3a4a6bb7c39f9c9585c4f03a7b72641/fla/ops/attn/parallel.py) in 3 phases.

<!-- Why 3 phases rather than 2? Why not have dqkv together? -->

## The Data Layout

Before we jump into the code, let's explicitly define what we are working with. The kernel is designed to handle Grouped Query Attention (GQA), which complicates the shapes slightly.

We are operating on the following tensors:

*   **`q` (Queries):** Shape `[B, T, HQ, K]`.
    *   `B`: Batch size.
    *   `T`: Sequence length.
    *   `HQ`: Number of Query Heads.
    *   `K`: Head dimension for keys/queries.
*   **`k` (Keys):** Shape `[B, T, H, K]`.
    *   `H`: Number of Key/Value Heads. Note that `HQ` must be a multiple of `H`.
*   **`v` (Values):** Shape `[B, T, H, V]`.
    *   `V`: Head dimension for values.
*   **`do` (Output's gradient):** Shape `[B, T, HQ, V]`.
    *   This is the incoming gradient from the next layer, with the same shape as the forward output.

Our goal is to produce `dq`, `dk`, and `dv`, matching the shapes of `q`, `k`, and `v` respectively.

## The Big Picture: Autograd

Inside the `ParallelAttentionFunction`, the `backward` method is automatically called by PyTorch when we call `optimizer.step()`. 

```python
@torch.compile
class ParallelAttentionFunction(torch.autograd.Function):

    @staticmethod
    @contiguous
    @autocast_custom_bwd
    def backward(ctx, do):
        q, k, v, o, g_cumsum, lse = ctx.saved_tensors
        dq, dk, dv, dg = parallel_attn_bwd(
            q=q,
            k=k,
            v=v,
            o=o,
            g_cumsum=g_cumsum,
            lse=lse,
            do=do,
            scale=ctx.scale,
            cu_seqlens=ctx.cu_seqlens,
        )
        if dg is not None:
            dg = chunk_global_cumsum(dg, cu_seqlens=ctx.cu_seqlens, reverse=True)

        return dq.to(q), dk.to(k), dv.to(v), dg, None, None
```

The context object `ctx` gives us back the tensors we saved during the forward pass. In particular, we have `o` (the forward output) and `lse` (the Log-Sum-Exp statistics). We pass these, along with the incoming gradient `do`, into our launcher `parallel_attn_bwd`.

## The Launcher

The launcher function `parallel_attn_bwd` handles the dispatch. Unlike the forward pass which is usually one big kernel, the backward pass here is split into **three** distinct operations.

1.  **Preprocessing:** Calculate a helper term called `delta`.
2.  **dQ Kernel:** Compute gradients for Queries (`dq`) and the log-decay `g` (`dg`).
3.  **dKV Kernel:** Compute gradients for Keys (`dk`) and Values (`dv`).

```python
def parallel_attn_bwd(
    # ... args ...
):
    # ... Shape extraction ...
    B, T, H, K, V = *k.shape, v.shape[-1]
    HQ = q.shape[2]
    G = HQ // H  # The GQA group size

    # 1. Tuning Block Sizes
    # Backward pass requires tuning based on hardware (Hopper vs Ampere)
    # to balance register usage.
    if check_shared_mem('hopper'):
        BT = 128
        BS = 64
        # ...
    
    # 2. Preprocess to get Delta
    delta = parallel_attn_bwd_preprocess(o, do)

    # 3. Prepare output tensors
    # Note: dq is created with shape [B, T, HQ, K]
    # But dk/dv are initially created with [B, T, HQ, K/V] to handle GQA expansion
    dq = torch.empty(..., device=q.device)
    dk = torch.empty(..., device=q.device) 
    dv = torch.empty(..., device=q.device)
    
    # 4. Launch separate kernels for dQ and dKV
    grid = (NV, NT, B * HQ)
    
    parallel_attn_bwd_kernel_dq[grid](
        q=q, k=k, v=v, lse=lse, delta=delta, do=do, dq=dq, ...
    )
    
    parallel_attn_bwd_kernel_dkv[grid](
        q=q, k=k, v=v, lse=lse, delta=delta, do=do, dk=dk, dv=dv, ...
    )
    
    # 5. Handle GQA reduction
    # We computed gradients for replicated keys/values. Now we sum them up.
    dk = reduce(dk, 'b t (h g) k -> b t h k', g=G, reduction='sum')
    dv = reduce(dv, 'b t (h g) v -> b t h v', g=G, reduction='sum')
    
    return dq, dk, dv, dg_cumsum
```

The design choice to split `dq` and `dkv` calculation into two kernels (unlike some implementations that fuse them) simplifies the logic regarding causal masking loops and reduces the amount of SRAM needed per thread block.

## Step 1: Preprocessing Delta

Before calculating gradients, we need a term `delta`. To derive the Softmax gradient, there is a row-wise summation term $\sum (O \cdot dO)$, which we pre-compute to avoid constantly re-computing it in the main loop below.

```python
@triton.jit
def parallel_attn_bwd_kernel_preprocess(
    o, do, delta,
    B: tl.constexpr, V: tl.constexpr,
):
    i_n = tl.program_id(0)
    o_d = tl.arange(0, B) # B is block size for V dimension
    m_d = o_d < V

    b_o = tl.load(o + i_n * V + o_d, mask=m_d, other=0)
    b_do = tl.load(do + i_n * V + o_d, mask=m_d, other=0).to(tl.float32)
    
    # delta = sum(output * grad_output)
    b_delta = tl.sum(b_o * b_do)

    tl.store(delta + i_n, b_delta.to(delta.dtype.element_ty))
```

This kernel computes the dot product of `o` and `do` for every token and stores it in `delta`. This term is critical for correcting the gradient of the Softmax function.

## Step 2: Computing dQ

Now comes the heavy lifting. `parallel_attn_bwd_kernel_dq` computes gradients with respect to `q`.

### Set Up
The kernel loads a block of queries `q` (`BT x BK`), the incoming gradient `do` (`BT x BV`), the denominator `lse`, and our precomputed `delta` into SRAM. These stay in registers/SRAM for the duration of the inner loop because we are computing `dq` for this specific block of time `BT`.

```python
    # [BT, BK] - Query Block
    b_q = tl.load(p_q, boundary_check=(0, 1))
    # [BT, BV] - Gradient of Output Block
    b_do = tl.load(p_do, boundary_check=(0, 1))
    # [BT] - LogSumExp and Delta
    b_lse = tl.load(p_lse, boundary_check=(0,))
    b_delta = tl.load(p_delta, boundary_check=(0,))

    # Initialize accumulator for dq
    b_dq = tl.zeros([BT, BK], dtype=tl.float32)
```

### Recomputation along the main loop

We iterate over blocks of Keys (`k`) and Values (`v`). This looks very similar to the forward pass, but we are doing it to recover the attention probability matrix $P$.

```python
    for i_s in range(0, i_t * BT, BS):
        # ... Load k [BK, BS] and v [BV, BS] ...
        
        # 1. Recompute Attention Scores [BT, BS]
        b_s = tl.dot(b_q, b_k) * scale * RCP_LN2
        
        # ... Apply log-decay g logic if USE_G is True ...

        # 2. Recompute Probability P = exp(Score - LSE)
        # We mask out future tokens if causal
        b_s = tl.where((o_q[:, None] >= o_k[None, :]) & m_k[None, :], b_s, float('-inf'))
        b_p = exp2(b_s - b_lse[:, None])
        
        # 3. Compute Gradient of Attention Scores (dS)
        # [BT, BV] @ [BV, BS] -> [BT, BS]
        b_dp = tl.dot(b_do, b_v)
        
        # dS = P * (dP - delta)
        b_ds = b_p * (b_dp.to(tl.float32) - b_delta[:, None])
        
        # 4. Accumulate Gradient for Query (dQ)
        # [BT, BS] @ [BS, BK] -> [BT, BK]
        b_dq += tl.dot(b_ds.to(b_k.dtype), tl.trans(b_k))
```

### The Gradient Math

The code concisely implements the backward pass of Softmax-Attention.
1.  **Recompute `b_p`**: We recover the attention probabilities using `q`, `k`, and `lse`. We didn't read this from HBM; we computed it on the fly.
2.  **Compute `b_ds`**: This represents $\frac{\partial L}{\partial S}$ (the gradient w.r.t the attention scores). The formula `P * (dot(do, v) - delta)` is the standard efficient softmax gradient implementation.[^1]
3.  **Compute `b_dq`**: Since $S = QK^T$, the gradient flow tells us $dQ = dS \cdot K$. In Triton, this is `tl.dot(b_ds, tl.trans(b_k))`.

## Step 3: Computing dK and dV

The second kernel, `parallel_attn_bwd_kernel_dkv`, is the "transpose" of the first.

In `dq`, we fixed a block of **Queries** (rows) and iterated over columns. Here, we fix a block of **Keys/Values** (columns) and iterate over the **Queries** (rows) that attended to them.

```python
    # Load the Key and Value blocks we want gradients for
    # [BT, BK]
    b_k = tl.load(p_k, boundary_check=(0, 1))
    # [BT, BV]
    b_v = tl.load(p_v, boundary_check=(0, 1))
    
    # Initialize accumulators
    b_dk = tl.zeros([BT, BK], dtype=tl.float32)
    b_dv = tl.zeros([BT, BV], dtype=tl.float32)
    
    # Iterate over Query blocks (i_s loop)
    # Note: We only loop over queries that come AFTER the current key (causal masking)
    # If I am a Key at time 10, only Queries at time >= 10 cared about me.
    for i_s in range(i_t * BT, min((i_t + 1) * BT, T), BS):
        # ... Load q, do, lse, delta ...

        # 1. Recompute Score (Transposed)
        # Note: We compute dot(K, Q^T) effectively
        b_s = tl.dot(b_k, tl.trans(b_q)) * scale * RCP_LN2
        
        # 2. Recompute P 
        b_p = tl.where((o_k[:, None] <= o_q[None, :]) & m_q[None, :], exp2(b_s - b_lse[None, :]), 0)
        
        # 3. Compute dV
        # dV += P^T * dO
        # [BT, BS] @ [BS, BV] -> [BT, BV]
        b_dv += tl.dot(b_p.to(b_do.dtype), b_do)
        
        # 4. Compute dS
        # We need dP = dO * V^T. 
        # In code: b_dp = dot(b_v, trans(b_do))
        b_dp = tl.dot(b_v, tl.trans(b_do))
        
        b_ds = b_p * (b_dp - b_delta[None, :])
        
        # 5. Compute dK
        # dK += dS^T * Q
        # [BT, BS] @ [BS, BK] -> [BT, BK]
        b_dk += tl.dot(b_ds.to(b_q.dtype), b_q)
```

The logic mirrors the `dq` kernel, but the matrices are transposed. For example, to get `dv`, we are essentially performing $P^T \cdot dO$.

By keeping `k` and `v` in SRAM and streaming `q`, `do`, and `delta` from HBM, we efficiently compute the gradients for the Key and Value matrices. Finally, the launcher script will take these results and `reduce` them if we are using GQA, summing the gradients across the replicated heads.

## Acknowledgments

This is inspired by our very amazing [Nathan Chen](https://nathanchen.me/)'s [Triton Flash Attention Kernel Walkthrough: The Forward Pass](https://nathanchen.me/public/Triton-Flash-Attention-Kernel-Walkthrough.html). Big shoutout to him! 

---

### The Math Corner

[^1]: **The Delta Term derivation:**
    If we define attention output $O_i = \sum_j P_{ij} V_j$, the gradient of the loss function $L$ with respect to the pre-softmax scores $S_{ij}$ involves the Jacobian of the Softmax.
    Specifically, $\frac{\partial L}{\partial S_{ij}} = P_{ij} (\frac{\partial L}{\partial P_{ij}} - \sum_k P_{ik} \frac{\partial L}{\partial P_{ik}})$.
    
    Since $\frac{\partial L}{\partial P_{ij}} = dO_i \cdot V_j$, we can substitute:
    $\frac{\partial L}{\partial S_{ij}} = P_{ij} ( (dO_i \cdot V_j) - \sum_k P_{ik} (dO_i \cdot V_k) )$.
    
    The term $\sum_k P_{ik} (dO_i \cdot V_k)$ can be rewritten by factoring out $dO_i$: 
    $dO_i \cdot (\sum_k P_{ik} V_k) = dO_i \cdot O_i$.
    
    This term $dO_i \cdot O_i$ is exactly what we calculate in the `preprocess` kernel and store as `delta`.
    So, $dS_{ij} = P_{ij} ( (dO_i \cdot V_j) - \text{delta}_i )$. This matches the code: `b_ds = b_p * (b_dp - b_delta)`.