## Paper 1: **LoRA: Low-Rank Adaptation of Large Language Models**
### Summary
The paper introduces a new method to fine-tune Large Language Models (and other models in general, as the method itself is model agnostic). The model introduces injection of low rank decomposed matrices with trainable weights in a pretrained model. The pretrained weights are frozen. The benefit over other fine-tuning methods (like prefix tuning and adapter layers) is that:
- There is practically no inference delay.
- There is no compromise on sequence length.
- Saving minimal weights for different downstream tasks is still the better way to go.
$$
	h = W_ox + \Delta Wx = W_ox + BAx
$$
### Objectives
The goals of the paper were to:
- Reduce cost of fine-tuning for Large Language Models.
- Introduce no to minimal inference delay
- Testing the hypothesis that major updates lie in the low-rank subspace.
- Compare LoRA with other fine-tuning mechanisms (adapter layers and prefix tuning)
- Avoid storing different weights for fully fine-tuned large model for different tasks. Instead, introduce lightweight, low rank, trained weight matrices and use those for different downstream tasks.
### Key Findings
- Parameter fine-tuning works extremely well.
- Low-rank weights are expressive enough.
- No loss in inference efficiency if weights are merged.
- LoRA works over different models (like GPT-2, BERT etc.)
- Good multi-task potential
### Limitations
- The paper highlights that some inference delay is introduced if there are multiple sets of weights for different downstream tasks. Merging is not easy if that is the case which eventually introduced some inference delay.
- Eventually, if the downstream task is entirely different in its domain relative to the original pretrained model learnt dataset, full fine-tuning will be required.
### Random Queries I had (and Answers)
- For the low rank $BA$ matrix, why is $B$ initialized with zeros and $A$ is with Gaussian?
	- This is to ensure training stability. At the very start, $BA$ will be zero, reciprocating the original, pretrained model. And slowly, new weights will be learnt (fine-tuned) based on the new inputs.
- Wouldn't adapter + LoRA hybrid models work?
	- They work, but with trade-offs. If the task is multi-modal or the domain is quite different, hybrid approaches can be quite beneficial. But of course, it will introduce complexities like inference delay, more parameters, sensitive learning rate (to manage both LoRA as well as the new adapter layers)

## Summary List
- **LoRA: Low-Rank Adaptation of Large Language Models***
	- **Authors:** Edward Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen
	- **Problem Addressed:** 
	- **Methodology:** Introduction of Low Rank decomposition matrices with trainable weights, while the pretrained model weights will be frozen.
	- **Dataset:** GLUE (General Language Understanding Evaluation) primarily used for benchmarking.
	- **Key Results:**
		- Substantially faster fine-tuning without sacrificing inference latency.
		- Multi-task trained matrices for different downstream tasks.
		- Low rank subspace is quite efficient for multitude of tasks within the domain, and achieve quite good results.
	- **Research Gaps:**
		- Which layers should be LoRA'ed? Where is performance the best?
		- Some inference delay is introduced if multiple downstream tasks are fine-tuned for the model. Switching between the matrices is where the inference delay happens.
		- If the learning domain entirely changes, fully fine-tuned training is eventually necessary - something highlighted in "LoRA Learns Less and Forgets Less" (Biderman et al., 2024)

## Random Paper Queries and Ideas
1. Which layers, which weight matrices, and which tasks actually require full-rank updates?
2. Would a hybrid approach work, where some layers may need full-rank updates, while others can work with LoRA?
	- This question is very close to the first one. Basically, combination of the my first question with LoRA. Which layers to be fine-tuned and which can utilize LoRA?
3. Applying different activation functions over LoRA, is this something to investigate?
