# Title
**QLoRA: Efficient Finetuning of Quantized LLMs**
# Abstract
Introduction of multiple innovations to performant fine-tuning:
- NF4 (4-bit Normal Float) - Theoretically optimal for normal distributed weights
- Double Quantization to reduce average memory footprint
- Paged Optimizers for memory spikes
Links:
- [QLoRA](https://github.com/artidoro/qlora)
- [bitsandbytes](https://github.com/bitsandbytes-foundation/bitsandbytes)
# Introduction
Fine-tuning an LLM is quite a tall task with heavy computation required. The paper claims to fine-tune a big model (65B) with a single GPU with a 4-bit quantization.
# Glossary
- **NF4**
- **Double Quantization**
- **Paged Optimizers**
- **4-bit Model:** Model weights are stored and computed using 4-bit numerical precision instead of 16-bit or 32-bit. Greatly reduces the memory footprint for a model.