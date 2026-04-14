# Building a Frontier LLM from Scratch

A single-page technical guide covering every component of a frontier large language model — from byte-pair encoding to RLHF, with full PyTorch implementations.

**[Read the guide](https://spirituslab.github.io/llm-from-scratch/)**

## Why this exists

I wanted to deeply understand how modern LLMs work, but the learning materials I found online were either too long (spread across dozens of blog posts, papers, and videos) or too scattered (each source covering one piece without connecting it to the whole picture). So I used Claude Code to help me build a single, self-contained guide that covers everything in one place — from tokenization all the way to RLHF and inference optimization.

## Prerequisites

- Basic linear algebra (matrix multiplication, dot products, transpose)
- Single-variable calculus (derivatives, chain rule)
- Basic probability (distributions, conditional probability)
- Python (functions, classes, loops)

No prior PyTorch, machine learning, or transformer experience needed — the guide covers all of that from scratch.

## What's covered

| Section | Topic |
|---------|-------|
| 0 | Preface & Prerequisites |
| 0b | Foundations: Neural Networks, Language Models, PyTorch |
| 1 | Tokenization (BPE) |
| 2 | Embeddings & Positional Encoding (RoPE) |
| 3 | Attention (Scaled Dot-Product, Multi-Head, GQA, Flash Attention) |
| 4 | Feed-Forward Networks (SwiGLU) |
| 5 | Normalization (RMSNorm, Pre-Norm) |
| 6 | The Complete Model (Transformer Block, Full Architecture, MoE) |
| 7 | Training Data (Sources, Quality Pipeline, Data Loading) |
| 8 | Optimizer & Schedule (AdamW, Cosine LR, Scaling Laws) |
| 9 | Distributed Training (BF16, FSDP/ZeRO, Tensor/Pipeline Parallelism) |
| 10 | Post-Training: SFT |
| 11 | Post-Training: RLHF (Reward Modeling, PPO) |
| 12 | Post-Training: DPO |
| 12b | 2025 Frontier: RLVR, Process Reward Models, Constitutional AI, Test-Time Compute |
| 13 | Inference (KV-Cache, Sampling, Quantization, Speculative Decoding) |
| 14 | Full Implementation (runnable training + generation scripts) |
| 15 | What Makes It Frontier |

## Run it yourself

```bash
pip install torch numpy
python prepare_data.py   # Download and tokenize sample data
python train.py          # Train a ~10M parameter toy model
python generate.py       # Generate text from the trained model
```

The code in Section 14 is copy-pasteable and runs on a laptop CPU (10-30 min) or GPU (2-5 min).

## Recommended reading

Sebastian Raschka's [*Build a Large Language Model (From Scratch)*](https://www.manning.com/books/build-a-large-language-model-from-scratch) is an excellent companion to this guide. It walks through building a GPT-style model step by step with clear code and explanations — highly recommended if you want a book-length treatment alongside this condensed reference.

## Disclaimer

This is a personal self-learning guide, not a definitive reference. It was built with the help of AI and may contain errors or inaccuracies. If something looks off, it probably is — please don't treat it as ground truth. Cross-check with primary sources (papers, official documentation) when in doubt.

## Suggestions welcome

If you find errors, unclear explanations, or missing context, please open an issue or PR — I'd appreciate it.
