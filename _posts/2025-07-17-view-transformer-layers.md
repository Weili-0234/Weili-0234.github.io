---
layout: distill
title: "View Transformer Layers from Online Optimization Perspective"
date: 2025-07-17 12:00:00
description: "In this blog post co-authored with <a href='https://wenhaochai.com/'>Wenhao Chai</a>, we revisit the landscape of efficient Transformer variants from a unified view of fast weight programming."
tags: transformer sequence-modeling llm-architecture optimization
categories: technical
---



[Read the full PDF](https://wenhaochai.com/assets/file/blog/mesa.pdf)

### The Fast Weight Perspective

The core idea is to view the **hidden state** of linear attention models not just as a memory buffer, but as a set of **Fast Weights** that are dynamically updated during the forward pass. This contrasts with the "Slow Weights" (the model parameters) learned during training.

From this **Online Optimization Perspective**, the state update rule in architectures like **Linear Attention**, **DeltaNet**, and **TTT (Test-Time Training)** can be interpreted as a step of gradient descent on a specific objective function. For example, standard linear attention effectively minimizes the alignment error between the projected query and the target value stored in the state.

### Unifying Architectures

We provide a framework that connects several recently emerged architectures:
*   **Linear Attention Transformers** (e.g. GLA, DeltaNet)
*   **State-Space Models** (e.g. Mamba)
*   **RNN variants** (e.g. RWKV)

By understanding these models as a family of parametric systems that update their internal representation based on the input stream, we can better understand their capabilities and how this mechanism is related to their continual learning capabilities.
