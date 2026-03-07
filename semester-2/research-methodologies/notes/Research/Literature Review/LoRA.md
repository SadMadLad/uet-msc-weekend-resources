# Title
**LoRA: Low-Rank Adaptation of Large Language Models**
# Abstract
Freezes the existing weights of the model and inject **trainable rank decomposition matrices** into the transformer architecture, greatly reducing the number of trainable parameters for downstream tasks. [Github](https://github.com/microsoft/LoRA)
# Introduction
Earlier on, fine tuning usually required increasing the length of the model or reducing the sequence length, both of these approaches had trade-offs when it comes to model quality and inference latency (for example, modifying sequence length can degrade model quality and deepening the model can introduce inference lag)
***Intuitive Analogy**: Think of learning to play chess. A beginner needs to consciously think about thousands of rules and possibilities (high dimensional). A grandmaster has compressed all that knowledge into a handful of deep strategic principles (low intrinsic dimension). Pre-trained models are like grandmasters — they've already compressed the world, so adapting to a new task only requires a few "strategic adjustments."*
Benefits of LoRA:
- No inference delay introduced.
- Lowers hardware barrier, as trainable parameters are of the matrices only.
- Can create multiple LoRA modules for different tasks.
- Does not necessitate an entirely separate infrastructure for fine-tuning. Other methods for fine-tuning are still compatible.
## Problem Statement
For each new specific downstream task when it comes to fine-tuning, you need to learn new sets of parameters and store those parameters, which can quickly become infeasible. Instead, we can use much smaller set of parameters and learn those for every downstream task.
## Aren't Existing Solutions Good Enough
- **Adaptive Layers** introduce inference latency.
- **Prefix tuning** reduces sequence length, reducing available space for downstream tasks and performance gains are non-monotonic (i.e. not predictable in one direction).
# Our Method
For a pre-trained model, the weight matrix looks like $W_0 \in \mathbb{R}^{d \times k}$, with low rank decomposition, it will look like $W_0 + \Delta W = W_0 + BA$ where $B \in \mathbb{R}^{d \times r}$ and $A \in \mathbb{R}^{r \times k}$ where $r << min(d, k)$. During training $W_0$ is frozen and only $A$ and $B$ are trainable. For each input, the output is going:
$$
	h = W_0x + \Delta Wx = W_0 + BAx
$$
- $A$ is initialized by Gaussian Initialization
- $B$ is initialized by zeros
- $BA$ at the very start is 0.
- $\Delta W$ is scaled by $\frac{\alpha}{r}$. $\alpha$ is tuned the same way learning rate is tuned.
- We can switch between downstream tasks by subtracting $BA$ from $W_0$ and then later adding a different set of weights for other task $B'A'$ - hence, no delay at inference.
The paper specifically applies LoRA to attention weights, and not the feed forward layer.
# Glossary
- Trainable Rank Decomposition Matrices
- **Transformer Architecture:** Architecture introduced in "Attention is all you Need" paper.
- **Inference Latency:** Time it takes to produce an output for the LLM. Typically consists of **prefill** (prompt processing) and **decode** (token generation)
- **Over-parameterized models:** Models which have way more parameters than their training data points - typically BERT, GPT and other huge models with lots of parameters.
- **GPT:** Generative Pretrained Transformer
- **Adaptive Layers:** Adding layers to transformer block for fine-tuning purposes.
- **Prefix Tuning:** Instead of changing the architecture, we add trainable tokens at the input.
- **Gaussian Initialization:** Setting the initial weights of sampled from normal distribution. The idea being with initial weights being random allows neurons to learn variety of features.