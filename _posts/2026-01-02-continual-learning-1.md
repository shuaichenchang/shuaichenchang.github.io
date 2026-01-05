---
title: 'Continual Learning and Memory (1): Titans and End-to-End Test-Time Training'
date: 2026-1-4
permalink: /posts/2026/test-time-learning-1/
tags:
  - continual-learning
  - test-time training
  - memory
---

## Introduction

We all share the dream that powerful AI models could learn by themselves, continuously updating and improving from their daily tasks (that is, at test time). Humans do this naturally. In fact, we have no choice but to update our memory every second, and we have not found a way to reset ourselves to a moment in the past (building a time machine is not discussed in this post). Compared with humans, many believe that the future of AI lies in the ability to continually learn at test time, rather than waiting to be retrained periodically. That said, continual learning and memory are essential capabilities for future AI systems.

This post is the first in a series of blog posts where I share my notes and learnings on recent progress in continual learning research. In this post, we recap the core ideas from Google's paper [Titans: Learning to Memorize at Test Time](https://arxiv.org/pdf/2501.00663) (January 2025) which went viral at the end of 2025 as well as a more recent paper [End-to-End Test-Time Training for Long Context](https://arxiv.org/pdf/2512.23675) (December 2025). Both papers formulate continual learning as a long-context problem where a language model continues working on new tasks while keeping previous tasks in its context and learns from them.
While traditional attention has enabled in-context learning, it is limited by its quadratic growth in time complexity. These two papers address it with test-time training.

## Background: Full Attention and Linear Attention

[Transformer-based](https://arxiv.org/pdf/1706.03762) language models capture the dependency between the current token and previous tokens using the attention mechanism, which is based on a softmax over the previous tokens. Formally, the attention output $o_t$ can be written as:

$$
\begin{aligned}
q&=xW_q, \quad k=xW_k, \quad v=xW_v, \\
o_t &=
\sum_{j=1}^{t}
\frac{
\exp\!\left( q_t^\top k_j / \sqrt{d_{\text{in}}} \right)
}{
\sum_{l=1}^{t}
\exp\!\left( q_t^\top k_l / \sqrt{d_{\text{in}}} \right)
}
v_j,
\end{aligned}
$$

where $q, k, v \in \mathbb{R}^{d_{\text{in}}}$, and $W_q, W_k, W_v \in \mathbb{R}^{d_{\text{in}}\times d_{\text{in}}}$.

The [Linear Attention paper](https://arxiv.org/pdf/2006.16236) points out that for any non-negative similarity function $\text{sim}(q_t, k_j)$, including softmax, there exists a feature map $\phi$ (potentially in infinite dimensions) such that $\text{sim}(q_t, k_j)=\phi(q_t)^\top\phi(k_j)$. Under this formulation, attention can be rewritten as: 

$$
\begin{aligned}
o_t &=
\sum_{j=1}^{t}
\frac{
\text{sim}\!\left(q_t,k_j\right)
}{
\sum_{l=1}^{t}
\text{sim}\!\left(q_t, k_l\right)
}
v_j \\
&=
\sum_{j=1}^{t}
\frac{
\phi(q_t)^\top \phi(k_j)
}{
\sum_{l=1}^{t}
\phi(q_t)^\top \phi(k_l)
}
v_j \\
&=
\frac{
\phi(q_t)^\top \sum_{j=1}^{t} \phi(k_j) v_j
}{
\phi(q_t)^\top \sum_{l=1}^{t} \phi(k_l)
}
\end{aligned}
$$

The key insight is that $\sum_{j=1}^{t} \phi(k_j) v_j^\top$ can be computed recurrently, which can be written in a recurrent format.

$$
M_t = M_{t−1} + \phi(k_t) v_t
$$

From this perspective, the goal of linear attention is to compress the keys and values into $M$, which serves as a form of fast, associative memory.

## Titans
Since memory can be represented in a recurrent form, Titans uses a more general formulation:

$$
\begin{aligned}
M_t &= f(M_{t-1}, x_t),\\
\tilde{y}_t &= g(M_t,x_t),
\end{aligned}
$$

where the functions $f$ and $g$ can be viewed as memory write and memory read operations, respectively. Here, $M_t$ itself can be a neural network (e.g., an MLP), rather than a simple vector or matrix.

The next question is how to learn and update the memory unit $M_t$. The Titans paper motivates this from a perspective inspired by human memory: events that violate expectations (i.e., are surprising) are more memorable for humans. Accordingly, Titans uses a surprise signal, measured via prediction error, to update the memory. Concretely, the memory is updated using gradient descent with momentum and weight decay[^1]:

$$
M_t = M_{t−1} − \eta_t \nabla\ell(M_{t−1}; x_t),
$$

where $\eta_t$ is the learning rate at timestep $t$ and the surprise score $\ell$ is defined as:

$$
\ell(M_{t-1}; x_t) = ||M_{t-1}(k_t)-v_t||_2^2
$$


We can also understand this mechanism from a more traditional machine learning perspective. After the memory write, $M_t$ should contain the key $k_t$ and its corresponding value $v_t$, such that the value can be retrieved given the key. In other words, we want:

$$
M_t(k_t) = v_t
$$

If we treat $M$ as a set of model parameters, then updating $M$ so that the prediction $M(k_t)$ matches the target $v_t$ is simply a standard supervised learning problem. The update from $M_{t-1}$ to $M_t$ naturally follows from gradient descent on the loss:

$$
\ell(M_{t-1}; x_t) = ||M_{t-1}(k_t)-v_t||_2^2,
$$

which yields:

$$
M_t = M_{t−1} − \eta_t \nabla\ell(M_{t−1}; x_t)
$$


This means that we can update the memory $M$ at each timestep via gradient descent at test time. But how can we train such a memory mechanism when the overall model is trained with the standard language modeling objective (e.g., next-token prediction)?

Recall that the memory update itself can be viewed as a function $f$ in the recurrence $M_t = f(M_{t-1}, x_t)$. The training follows a meta-learning formulation: in the **inner loop**, the model learns how to update the memory parameters $M$ using the gradient-based rule above; in the **outer loop**, it trains the remaining parameters (e.g., the parameters in $g(M_t,x_t)$ and the projections for $k, v$) to optimize the language modeling objective, given the updated memory from the inner loop.  At test time, only the inner loop (memory updates) runs, while the outer loop parameters remain fixed.

## TTT-E2E
While Titans learns the memory $M_t$ by associating $K_{<t}$ and $V_{<t}$, TTT-E2E argues that since the goal of memorizing past knowledge is to improve future predictions, a more straightforward objective can be used:

$$
\ell(M_{t-1}; x_t) = \text{CE}(g(M_{t-1},x_{t-1}), x_t),
$$

which is the cross-entropy loss for next-token prediction, where $g(M_{t-1}, x_{t-1})$ produces the logits for predicting token $x_t$. As a result, the same loss function is used to train the memory during both training and test time. Furthermore, TTT-E2E uses some of the MLP layers in Transformers as the memory $M$, achieving continual learning without changing the Transformer architecture.

## Parallel Training
Both Titans and TTT-E2E rely on a recurrently updated memory unit, which requires $O(T)$ FLOPs during both training and test time. While it is efficient at test time, the recurrent nature of the memory does not naturally support token-level parallelization during training, making it less efficient than standard Transformers in practice. To address this issue, both papers adopt chunking/batching, partitioning the input token sequence to enable parallel computation within chunks.

For simplicity, assume the total number of tokens $T$ is divisible by a chunk size $b$. In Titans, the memory $M_t$ is updated with respect to $M_{t'}$ instead of $M_{t-1}$, where $t' = t- (t \mod b)$ denotes the timestep at the end of the previous chunk. Under this approach, all $M_t$ values within the same chunk can be computed in parallel. The update can be written as:

$$
M_t = M_{t'} - \eta_{t'+1} \frac{1}{t-t'} \sum_{i=t'+1}^t \nabla\ell(M_{t'},x_i)
$$

This is parallelizable because all gradients $\nabla\ell(M_{t'},x_i)$ depend on the same $M_{t'}$ and can be computed simultaneously. The partial sums can then be accumulated efficiently using [parallel prefix sum algorithms](https://www.cs.cmu.edu/~guyb/papers/Ble93.pdf).

TTT-E2E uses an even simpler formulation. Within each chunk, the memory unit $M$ remains constant and is updated only after the entire chunk is processed. As a result, all tokens within the chunk can be computed in parallel. For $i = 1, ..., T/b$, we have:

$$
M_i = M_{i−1} − \eta_i \frac{1}{b} \sum_{t=(i-1)b+1}^{ib} \nabla\ell(M_{i−1}; 𝑥_t)
$$

Both methods enable parallelization within each chunk. However, this introduces another issue: $M_t$ becomes stale or imprecise within the chunk, as it does not reflect the most recent tokens and memory. This can affect the output prediction, which originally takes the form $\tilde{y}_t = g(M_t, x_t)$. To mitigate this issue, both Titans and TTT-E2E retain full self-attention within each chunk, modifying the output prediction to:

$$
\tilde{y}_t = g(M_t,x_t, K[t'+1:t], V[t'+1:t])
$$


[^1]: Titans paper also leverages memory forgetting rate and surprise momentum, which we omit here for simplification.


## References

**Titans: Learning to Memorize at Test Time** [[PDF]](https://arxiv.org/pdf/2501.00663)
Behrouz, Ali, Zhong, Peilin, and Mirrokni, Vahab, 2024.

**End-to-End Test-Time Training for Long Context** [[PDF]](https://arxiv.org/pdf/2512.23675)
Tandon, Arnuv, Dalal, Karan, Li, Xinhao, Koceja, Daniel, Rød, Marcel, Buchanan, Sam, Wang, Xiaolong, et al., 2025.

**Attention Is All You Need** [[PDF]](https://arxiv.org/pdf/1706.03762)
Vaswani, Ashish, Shazeer, Noam, Parmar, Niki, Uszkoreit, Jakob, Jones, Llion, Gomez, Aidan N., Kaiser, Łukasz, and Polosukhin, Illia, 2017.

**Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention** [[PDF]](https://arxiv.org/pdf/2006.16236)
Katharopoulos, Angelos, Vyas, Apoorv, Pappas, Nikolaos, and Fleuret, François, 2020.

**Prefix Sums and Their Applications** [[PDF]](https://www.cs.cmu.edu/~guyb/papers/Ble93.pdf)
Blelloch, Guy E., 1990.

 