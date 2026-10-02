# Transformer Pretraining with Self-Attention

A from-scratch Transformer language modeling project implemented in PyTorch, covering self-attention, multi-head attention, masking, positional representations, data batching, optimization, and pretraining.

> **Portfolio Project**  |  **NLP • Transformers • Self-Attention • PyTorch**

## Overview

Transformers replaced recurrence with attention-based sequence modeling, making it possible to compute token interactions in parallel and capture long-range dependencies more directly.

This project implements the main building blocks of a Transformer-style pretraining pipeline and connects the model to dataset preparation, training, validation, and evaluation. The focus is on understanding the mechanics of attention rather than treating a pretrained Transformer as a black box.

## What This Project Demonstrates

- Transformer architecture fundamentals
- Scaled dot-product self-attention
- Multi-head attention
- Query, key, and value projections
- Attention masking for autoregressive prediction
- Position-aware sequence representations
- Feed-forward Transformer blocks
- Language-model pretraining
- PyTorch training loops and optimization
- Dataset and batch construction

## Architecture

```text
Token IDs
   ↓
Token embeddings + positional information
   ↓
Transformer block
   ├── Multi-head self-attention
   ├── Residual connection + normalization
   └── Position-wise feed-forward network
   ↓
Transformer representation
   ↓
Language-model output layer
   ↓
Next-token probabilities
```

## Attention Intuition

For a sequence of hidden states, self-attention computes relationships between tokens by comparing queries with keys and using the resulting weights to aggregate values:

```text
Attention(Q, K, V) = softmax(QKᵀ / √dₖ) V
```

Multi-head attention repeats this operation across multiple learned projections, allowing the model to capture different interaction patterns within the same sequence.

## Repository Structure

```text
transformer-pretraining/
├── README.md
├── .gitignore
├── src/
│   ├── attention.py
│   ├── dataset.py
│   ├── helper.py
│   ├── model.py
│   ├── trainer.py
│   └── utils.py
└── results/
    └── README.md
```

## Core Components

### `attention.py`
Contains the attention computations used by the Transformer, including the core tensor operations behind self-attention and multi-head attention.

### `dataset.py`
Handles the data representation and batching required for language-model pretraining.

### `model.py`
Defines the Transformer model and its neural components.

### `trainer.py`
Implements the training and evaluation loop, including optimization and model updates.

### `helper.py`
Provides supporting functions used by the project workflow.

### `utils.py`
Contains reusable utility functions for data and training operations.

## Technology Stack

| Area | Technology |
| --- | --- |
| Language | Python |
| Deep learning | PyTorch |
| Architecture | Transformer |
| Core mechanism | Multi-head self-attention |
| Training objective | Autoregressive language modeling |

## Engineering Focus

The project emphasizes the implementation details that make Transformer models work in practice: tensor dimensions, attention masks, sequence batching, residual pathways, normalization, and stable training.

## Learning Outcomes

By implementing the Transformer components directly, the project builds a concrete understanding of how modern language models process context, how attention maps information across a sequence, and how these components fit into a complete pretraining pipeline.

## Course Context

Developed as part of **Stanford XCS224N: Natural Language Processing with Deep Learning** coursework.

## Notes on Reproducibility

Trained parameter files, caches, logs, and temporary outputs are excluded from version control by default. This keeps the repository focused on the reusable implementation rather than generated binaries.

## Status

**Completed coursework implementation - portfolio packaged**

## Author

**Saniya Mirza**
