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

## Analyzing Per-Token Entropy

`analyze_tokens.py` recreates Figures 2(a), 2(b), and 2(c) from the paper. It identifies which specific tokens tend to be high-entropy ("forking" tokens where the model is uncertain) vs low-entropy ("path-following" tokens the model predicts confidently). It has two stages — **generate** and **analyze** — that can be run independently or together.

### Subcommands

```bash
# 1. Generate responses and collect per-token entropy + token IDs (requires GPU)
.venv/bin/python analyze_tokens.py generate --n_questions 50 --n_responses 4

# 2. Analyze saved data — rank tokens by average entropy (CPU only)
.venv/bin/python analyze_tokens.py analyze --min_freq 20 --top_n 100

# 3. Run both stages back-to-back
.venv/bin/python analyze_tokens.py all --n_questions 50 --n_responses 4
```

### Options

**`generate`** options:

| Flag | Default | Description |
|---|---|---|
| `--model` | `Qwen/Qwen3-8B` | Hugging Face model ID |
| `--n_questions` | `50` | Number of MATH-500 problems to use |
| `--n_responses` | `4` | Responses to generate per question |
| `--batch_size` | `4` | Responses per forward pass (lower if OOM) |
| `--max_new_tokens` | `4096` | Max tokens per response |
| `--temperature` | `1.0` | Sampling temperature |
| `--output_dir` | `entropy_results` | Directory for all outputs |

**`analyze`** options:

| Flag | Default | Description |
|---|---|---|
| `--model` | `Qwen/Qwen3-8B` | Tokenizer used for decoding token IDs |
| `--output_dir` | `entropy_results` | Directory containing saved `.npy` files |
| `--min_freq` | `20` | Minimum token frequency to include in ranking |
| `--top_n` | `100` | Number of highest/lowest entropy tokens to display |

The `all` subcommand accepts all options from both stages.

### Outputs

All results are saved to `--output_dir` (default `entropy_results/`):

- `token_ids.npy` — flat array of generated token IDs
- `all_entropies.npy` — flat array of per-token entropy values (paired with token IDs)
- `generation_config.json` — generation hyperparameters for reproducibility
- `token_entropy_ranking.json` — every frequent token ranked by average entropy
- `stats.json` — summary statistics (mean, median, p80, % near-zero entropy)
- `fig2a_entropy_distribution.png` — histogram of token entropies (log-scale y-axis, 80th percentile marked)
- `fig2bc_token_entropy_bars.png` — horizontal bar charts of the highest and lowest entropy tokens