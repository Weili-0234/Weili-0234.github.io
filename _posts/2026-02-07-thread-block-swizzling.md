---
layout: post
title: "How Thread Block Swizzling boosts L2 Cache Hit Rate in Matrix Multiplication"
date: 2026-02-07 00:00:00
description: "A visual aid to Triton's official matmul tutorial, explaining how thread block level swizzling increases L2 cache hit rate."
tags: triton gpu performance visual
categories: technical
---

Understanding how data access patterns affect GPU cache performance is critical for writing efficient kernels. This interactive simulation demonstrates the impact of **Thread Block Swizzling** on L2 Cache hit rates in Matrix Multiplication.

[Launch Interactive Simulation](https://how-thread-block-swizzling-boosts-l2-cache-hit-ra-952909165134.us-west1.run.app/)

### The Problem: Row-Major Access in Computing $C = A B$
In a standard **Row-Major** block execution order, the GPU processes the output matrix $C$ row-by-row. For each new row of tiles in $C$, the kernel potentially needs to reload the entire $B$ matrix. If matrix $B$ is larger than the L2 cache, this leads to cache thrashing, where useful data is evicted before it can be reused by subsequent blocks.

### The Solution: Swizzling (Grouped Access to $B$)
Triton and other high-performance kernels allows **Swizzling** (or Grouped Access) to reorder the execution of thread blocks. By processing blocks in groups (e.g., essentially iterating in a column-major order within a small distinct group of rows), multiple blocks can reuse the same tiles of Matrix $B$ while they are still resident in the L2 cache.

This simulation allows you to:
- Toggle between **Row-Major** and **Grouped** access patterns.
- Adjust the **Group Size** and **L2 Cache Size** to see the direct impact on hit rates.
- Watch the simulation step-by-step to see exactly when cache hits and DRAM loads occur.

This serves as a companion to [Triton's Official Matrix Multiplication Tutorial](https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html), helping to visualize the `GROUP_SIZE_M` parameter implementation.
