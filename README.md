# Probing Language Models for Pre-training Data Detection
Official Implementation of ACL-2024 main conference paper: "Probing Language Models for Pre-training Data Detection".

## Overview
In this study, we propose to **utilize the probing technique for pre-training data detection** by examining the model's internal activations. Our method is simple and effective and leads to more trustworthy pre-training data detection. Additionally, we propose **ArxivMIA**, a new challenging benchmark comprising arxiv abstracts from Computer Science and Mathematics categories.
<p align="center">
  <img src="frame_work.png" width="90%" height="90%">
</p>

## ArxivMIA

To evaluate various pre-training data detection methods in a more challenging scenario, we introduce ArxivMIA, a new benchmark comprising abstracts from the fields of Computer Science (CS) and Mathematics (Math) sourced from Arxiv.

- ArxivMIA is avaliable in `data/arxiv_mia.jsonl`. 
- You also can access ArxivMIA directly on [Hugging Face](https://huggingface.co/datasets/zhliu/ArxivMIA).

```python
from datasets import load_dataset

dataset = load_dataset("zhliu/ArxivMIA")
```

## Probing LMs for Pre-training Data Detection

### Setup

Following commands to create an environment and install the dependencies:
```bash
conda create -n probing python=3.10
pip install -r requirements.txt
```

### Run
```bash
# TinyLLaMA
bash scripts/tinyllama_probe.sh > logs/tinyllama.log

# OpenLLaMA
bash scripts/openllama_probe.sh > logs/openllama.log
```

## Citation
TODO
```bibtex

```