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

## Computing Token Entropy

`compute_entropy.py` recreates Figure 2(a) from the "Beyond the 80/20 Rule" paper. It generates responses to MATH-500 problems using Qwen3-8B, computes the Shannon entropy of the model's output distribution at every generated token, and plots the resulting distribution.

### Basic usage

```bash
# Default: 50 questions, 4 responses each
.venv/bin/python compute_entropy.py

# Full MATH-500 run with 16 responses per question
.venv/bin/python compute_entropy.py --n_questions 500 --n_responses 16
```

### Options

| Flag | Default | Description |
|---|---|---|
| `--model` | `Qwen/Qwen3-8B` | Hugging Face model ID |
| `--n_questions` | `50` | Number of MATH-500 problems to use |
| `--n_responses` | `4` | Responses to generate per question |
| `--max_new_tokens` | `4096` | Max tokens per response |
| `--temperature` | `1.0` | Sampling temperature |
| `--output_dir` | `entropy_results` | Directory for all outputs |
| `--plot_only` | None | Path to an existing `.npy` file — skips generation, just re-plots |

### Outputs

All results are saved to `--output_dir` (default `entropy_results/`):

- `all_entropies.npy` — flat array of per-token entropy values
- `stats.json` — summary statistics (mean, median, percentiles, fraction of near-zero entropy tokens)
- `entropy_distribution.png` — histogram of token entropies with log-scale Y-axis and 80th percentile marked

### Re-plotting from saved data

If you already have entropy data from a previous run:

```bash
.venv/bin/python compute_entropy.py --plot_only entropy_results/all_entropies.npy
```