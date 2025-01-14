# ViTHash

ViTHash is a Python library that leverages the **[ViT-L/14 model](https://huggingface.co/openai/clip-vit-large-patch14)** (Vision Transformer from OpenAI's CLIP) to compute unique, feature-rich embeddings (hashes) for images. It also provides utilities for calculating the similarity or distance between these hashes, enabling use cases such as image similarity, deduplication, and retrieval.

---

## Features

- **Compute Image Hashes**: Generate robust embeddings (hashes) for images using the ViT-L/14 model.
- **Measure Similarity**: Compare two hashes using metrics like cosine similarity.
- **Hugging Face Integration**: Powered by [Hugging Face Transformers](https://huggingface.co/), allowing easy model usage.
- **GPU Support**: Automatically utilizes GPU acceleration when available.

---

## Installation

### Prerequisites

- Python 3.8+
- PyTorch (with GPU support recommended)

### Install ViTHash

You can install ViTHash from PyPI (after publishing):

```bash
pip install vithash