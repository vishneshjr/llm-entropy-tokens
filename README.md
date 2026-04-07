# llm-entropy-tokens

## Setup
Requires Python 3.11+ and [`uv`](https://github.com/astral-sh/uv)

```bash
git clone
cd llm-entropy-tokens
uv sync
```

## Downloading MATH-500
We use [MATH-500](https://huggingface.co/datasets/HuggingFaceH4/MATH-500) (500 competition math problems) for both training and evaluation

```bash
.venv/bin/python src/data.py
```
This pulls the dataset from the Hugging Face Hub into `~/.cache/huggingface/`