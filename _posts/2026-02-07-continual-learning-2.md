---
title: 'Continual Learning and Memory (2): Memory Architecture'
date: 2026-2-7
permalink: /posts/2026/continual-learning-2/
author: Shuaichen Chang
tags:
  - continual-learning
  - test-time training
  - memory
---

In [our first post](https://shuaichenchang.github.io/posts/2026/continual-learning-1/), we recapped the formulations of two continual learning papers. Both approaches conduct test-time learning to compress raw input into model parameters, effectively using those parameters as a dynamic working memory. In this post, we will step back to review the evolution of memory in neural models, from the older recurrent architectures to the latest research.

## Recurrent Hidden Vector as Memory
The concept of "memory" in neural networks is far from new. Recurrent Neural Networks (RNNs) have long utilized a hidden state vector to carry information over to new timesteps (or tokens, in modern terminology). The Long Short-Term Memory ([LSTM](https://deeplearning.cs.cmu.edu/S23/document/readings/LSTM.pdf)) architecture explicitly formalized this by distinguishing between a "hidden state" (output) and a "cell state" (memory).

![LSTM](/images/blogs/2025-12-31-continual-learning-memory/lstm.svg)

<p align="center">
<img src="/images/blogs/2025-12-31-continual-learning-memory/lstm.svg" style="width: 400px; max-width: 100%;">
</p>


At a high level, we can view the memory update as $c_t=f(x_t, c_{t-1})$, where $c_t$ and $c_{t-1}$ represent the memory at the current and previous timesteps, and $x_t$ is the current input. LSTMs introduce a forget gate and an input gate to regulate how much of the past memory $c_{t-1}$ is retained and how much new information from $x_t$ is added.

In theory, this recurrent memory allows information to be carried over indefinitely—from previous tasks to new ones. Aha! It turns out we had long-context language models for continual learning over 30 years ago.

However, it has two practical issues: 

(1) Capacity: A fixed-length hidden vector is easily overloaded; it simply cannot losslessly store the vast amount of information contained in a long sequence. 

(2) Permanence of Loss: Once information is discarded via the forget mechanism, it is gone forever. The model cannot "look back" to retrieve it later.

## Attention-based Memory

The first modern [attention paper](https://arxiv.org/pdf/1409.0473), proposed a solution: keep the RNN hidden vectors for all encoded tokens and use an attention mechanism to search over them during decoding. (Note: This was originally an encoder-decoder framework for machine translation, distinct from today's decoder-only LLMs). The RNN + Attention This architecture effectively utilizes two types of memory: (1) RNN hidden state, which is a fixed-size vector representing compressed context, (2) token activations which is growing buffer of states with a size linear to the input length.

<p align="center">
<img src="/images/blogs/2025-12-31-continual-learning-memory/rnn_attention.png" style="width: 250px; max-width: 100%;">
</p>

This mechanism laid the groundwork for the standard attention found in [Transformers](https://arxiv.org/pdf/1706.03762), which relies exclusively on this retrieved history as its working memory.

$$
\begin{aligned}
q&=xW_q, \quad k=xW_k, \quad v=xW_v, \\
o_t &=
\sum_{j=1}^{t}
\frac{
\exp\!\left( q_t^\top k_j / \sqrt{d} \right)
}{
\sum_{l=1}^{t}
\exp\!\left( q_t^\top k_l / \sqrt{d} \right)
}
v_j,
\end{aligned}
$$

where $q,k \in \mathbb{R}^{d_k}$, $v \in \mathbb{R}^{d_v}$, and $W_q, W_k, W_v$ are projection matrices.

Because Transformer attention provides a direct, lossless view of all past tokens, the recurrent hidden state became obsolete as a working memory. However, this capability comes at a steep price: quadratic time complexity with respect to sequence length.

### Recurrent Hidden Matrix as Memory

The [Linear Attention paper](https://arxiv.org/pdf/2006.16236) addresses this bottleneck by removing the softmax normalization from the attention mechanism. It observes that for any non-negative similarity function $\text{sim}(q_t, k_j)$, including softmax, there exists a feature map $\phi$ (potentially in infinite dimensions) such that $\text{sim}(q_t, k_j)=\phi(q_t)^\top\phi(k_j)$. Under this formulation, attention can be rewritten as: 

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
\phi(q_t)^\top \sum_{j=1}^{t} \phi(k_j) v_j^\top
}{
\phi(q_t)^\top \sum_{l=1}^{t} \phi(k_l)
}
\end{aligned}
$$

For simplicity, if we use the identity function as the feature map $\phi$ and omit the denominator normalization, the equation simplifies to:

$$
\begin{aligned}
o_t &=
q_t^\top \sum_{j=1}^{t} k_j v_j^\top
\end{aligned}
$$

Here, the term $\sum_{j=1}^{t} k_j v_j^\top$ can be denoted as a matrix $S_t \in \mathbb{R}^{d_k \times d_v}$. Crucially, this matrix can be computed recurrently:

$$
S_t = S_{t−1} + k_tv_t^\top
$$

This brings us full circle: $S_t$ acts as a recurrent memory, much like the RNN hidden vector. However, we are now using a matrix rather than a vector. While significantly more powerful than a simple vector, this compression is still lossy compared to standard Transformers.

Consider retrieving a specific value $v_i$ using its key $k_i$:

$$
k_i^\top S_t = (k_i^\top k_i) v_i^T +  \sum_{j\neq i} (k_i^\top k_j) v_j^\top
$$

If all keys are normalized to unit length, this becomes:

$$
k_i^\top S_t = v_i^T +  \underbrace{\sum_{j\neq i} (k_i^\top k_j) v_j^\top}_{\text{Noise}}
$$

To minimize retrieval error (i.e., reduce the noise term to zero), all keys must be orthogonal. This implies that a matrix of dimension $d_k$ can only losslessly store up to $d_k$ distinct items.

#### Gated DeltaNet

To mitigate the lossy nature of Linear Attention, esearchers revisited the LSTM's forget gate, leading to [Gated Linear Attention](https://arxiv.org/pdf/2312.06635). By introducing an input-dependent gate $G_t$, the model can selectively "clear up space" in $S_t$ for new information.
$$
S_t = G_t \odot  S_{t−1} + k_tv_t^\top
$$


Furthermore, [DeltaNet](https://arxiv.org/pdf/2406.06484) argues that the update rule should be mindful of what is already stored. Instead of blindly adding $k_t v_t^\top$, we should only add the difference (or delta) between the new information and the existing memory. Conceptually, we first "erase" the old value associated with $k_t$ and then write the new value[^1]:

$$
S_t = S_{t−1} - \beta k_t v_{old}^\top + \beta k_tv_t^\top,
$$
where $v_{old}^\top=k_t^\top S_{t-1}$. Expanding this term yields:
$$
\begin{aligned}
S_t
&= S_{t−1} - \beta k_t k_t^\top S_{t-1} + \beta k_tv_t^\top \\
&= S_{t−1} + \beta k_t (v_t^\top - k_t^\top S_{t-1})
\end{aligned}
$$

Interestingly, this update rule is equivalent to one step of gradient descent on an L2 loss function that measures the reconstruction error of the key-value pair:

$$
\begin{aligned}
\ell(S_{t-1}; k_t) &= ||v_t-k_t^\top S_{t-1}||_2^2 \\
\nabla\ell_S(S_{t−1}; k_t) &=  - k_t (v_t^\top - k_t^\top S_{t-1})
\end{aligned}
$$

From this perspective, Linear Attention is effectively a closed-form solution for test-time training of the memory matrix:
$$
\begin{aligned}
S_t &= S_{t-1} - \beta \nabla\ell_S(S_{t−1}; k_t)
\end{aligned}
$$

Moreover, the Gated Linear Attention can be viewed as a weight decay and combined into this optimization. The combined [Gated DeltaNets](https://arxiv.org/pdf/2412.06464) are then used in modern LLMs such as Kimi, Qwen3-next.

$$
S_t = G_t \odot S_{t−1} - G_t \odot \beta k_t k_t^\top S_{t-1} + \beta k_tv_t^\top
$$

If we view the matrix $S_t$ as memory and matrix multiplication as retrieval, we can extend this analogy further: every parameter in a Transformer, including the Feed-Forward Networks (FFNs), can be seen as memory.Typically, we view FFN weights as static storage for pre-training knowledge, which are not updated at test time. However, recent works like Titans and TTT-E2E propose that FFNs can also serve as dynamic working memory, provided we have an efficient mechanism to update them at test time.


## External Memory Pool

While innovations in Linear Attention allow for more efficient reading and writing of the memory state $S$, these methods still consolidate all knowledge into a single, superimposed representation rather than maintained in individual, discrete memory records. An alternative approach is to utilize an external memory pool and iteracte with it in the similar fashion as attention. This line of research dates back over a decade; let's briefly recap its evolution.


Following the rise of attention mechanisms in machine translation, researchers began to conceptualize neural memory as analogous to RAM in a computer, which is an external component that the CPU (the neural network) can read from and write to. Guided by this philosophy, pioneering works like [Memory Networks](https://arxiv.org/pdf/1410.3916) and [Neural Turing Machines](https://arxiv.org/pdf/1410.5401) implemented a memory pool $M \in \mathbb{R}^{N \times d}$ containing $N$ slots, each storing a $d$-dimensional vector. The model is then trained to learn specific read/write operations to manipulate this external pool for task-specific goals.


More recently, architectures like [MemoryLLM](https://arxiv.org/pdf/2402.04624) have augmented the standard Transformer attention mechanism by maintaining a set of external memory vectors.

<p align="center">
<img src="/images/blogs/2025-12-31-continual-learning-memory/memoryllm.png" style="width: 250; max-width: 100%;">
</p>

During generation, MemoryLLM attends to both the local context and the global memory pool, extending the standard formulation $Attention(Q, K, V)$ to:
$$
Attention(Q_X, [K_M;K_X], [V_M;V_X]),
$$
where $X$, $M$ represent the local context and global memory respectively.

Unlike the standard Attention, which updates the activations at every token, the MemoryLLM pool is updated only after processing a complete text segment (e.g., a paragraph). During this update phase, the model concatenates the hidden states of the new text chunk with the last $k$ records ($k \ll N$) from the existing memory pool. These are processed together, and the final $k$ hidden states are written back into the pool, replacing randomly selected vectors.

Conceptually, this acts as a "k-gram" conditional memory write: the model uses the previous $k$ memory records to condition the compression of the current text chunk into the latent memory space, generating $k$ updated vectors.


## Sparse Memory Layers

Recently, the authors of the [Memory layers paper](https://arxiv.org/pdf/2412.09764) argued that memory access is inherently sparse: only a few relevant pieces of information need to be retrieved at any given time, while the vast majority of stored knowledge remains irrelevant to the current context. Consequently, the dense matrix multiplication used in standard Transformer Feed-Forward Network (FFN) layers is an inefficient architecture for storage and retrieval.

They propose replacing the dense FFN layers in Transformer blocks with a Memory Lookup Layer. This sparse architecture allows for storing millions of memory slots, orders of magnitude more than a standard FFN, while maintaining efficient retrieval.

<p align="center">
<img src="/images/blogs/2025-12-31-continual-learning-memory/memory_layer.png" style="width: 400px; max-width: 100%;">
</p>

The memory layer contains a set of trainable parameters: keys $K \in \mathbb{R}^{d \times N}$ and values $V \in \mathbb{R}^{d \times N}$. Unlike the dynamic activations in attention, these parameters store static memory from pre-training data. At test time, a query $q \in \mathbb{R}^d$ is used to retrieve only the top-$k$ relevant keys ($k \ll N$), followed by a standard attention operation over just those $k$ slots:

$$
\begin{aligned}
I &= TopkIndices(Kq), \\
s &= Softmax(K_Iq), \\
y &= sV_I
\end{aligned}
$$


<!-- 
## Pluggable Memory

MemDecoder and MLPMem
 -->


## Wrap-up
Now we have explored several directions of memory representation in LLMs. In the next post, we will discuss more about the memory learning algorithms that make these architectures effective.


[^1]: The DeltaNet explanation is inspired by Songlin Yang's blog post: [Understanding DeltaNet](https://sustcsonglin.github.io/blog/2024/deltanet-1/)

## References

1. **Long Short-Term Memory** [[PDF]](https://deeplearning.cs.cmu.edu/S23/document/readings/LSTM.pdf)
   Hochreiter, Sepp and Schmidhuber, Jürgen, 1997.

2. **Neural Machine Translation by Jointly Learning to Align and Translate** [[PDF]](https://arxiv.org/pdf/1409.0473)
   Bahdanau, Dzmitry, Cho, Kyunghyun, and Bengio, Yoshua, 2014.

3. **Attention Is All You Need** [[PDF]](https://arxiv.org/pdf/1706.03762)
   Vaswani, Ashish, Shazeer, Noam, Parmar, Niki, Uszkoreit, Jakob, Jones, Llion, Gomez, Aidan N., Kaiser, Łukasz, and Polosukhin, Illia, 2017.

4. **Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention** [[PDF]](https://arxiv.org/pdf/2006.16236)
   Katharopoulos, Angelos, Vyas, Apoorv, Pappas, Nikolaos, and Fleuret, François, 2020.

5. **Gated Linear Attention Transformers with Hardware-Efficient Training** [[PDF]](https://arxiv.org/pdf/2312.06635)
   Yang, Songlin, Wang, Bailin, Shen, Yikang, Panda, Rameswar, and Kim, Yoon, 2024.

6. **Parallelizing Linear Transformers with the Delta Rule over Sequence Length** [[PDF]](https://arxiv.org/pdf/2406.06484)
   Yang, Songlin, Wang, Bailin, Zhang, Yu, Shen, Yikang, and Kim, Yoon, 2024.

7. **Gated Delta Networks: Improving Mamba2 with Delta Rule** [[PDF]](https://arxiv.org/pdf/2412.06464)
   Yang, Songlin, Kautz, Jan, and Hatamizadeh, Ali, 2024.

8. **Titans: Learning to Memorize at Test Time** [[PDF]](https://arxiv.org/pdf/2501.00663)
   Behrouz, Ali, Zhong, Peilin, and Mirrokni, Vahab, 2024.

9. **End-to-End Test-Time Training for Long Context** [[PDF]](https://arxiv.org/pdf/2512.23675)
   Tandon, Arnuv, Dalal, Karan, Li, Xinhao, Koceja, Daniel, Rød, Marcel, Buchanan, Sam, Wang, Xiaolong, et al., 2025.

10. **Memory Networks** [[PDF]](https://arxiv.org/pdf/1410.3916)
    Weston, Jason, Chopra, Sumit, and Bordes, Antoine, 2014.

11. **Neural Turing Machines** [[PDF]](https://arxiv.org/pdf/1410.5401)
    Graves, Alex, Wayne, Greg, and Danihelka, Ivo, 2014.

12. **MemoryLLM: Towards Self-Updatable Large Language Models** [[PDF]](https://arxiv.org/pdf/2402.04624)
    Wang, Yu, Dong, Yifan, Zeng, Zhuoyi, Ko, Shangchao, Li, Zhe, and Xiong, Wenhan, 2024.

13. **Memory Layers at Scale** [[PDF]](https://arxiv.org/pdf/2412.09764)
    Mu, Zhihang, Qiu, Shun, Lin, Xien, Huang, Po-Yao, Yan, Yingwei, and Lei, Tao, 2024.