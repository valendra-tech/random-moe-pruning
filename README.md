# random-moe-pruning

[![License](https://img.shields.io/badge/license-Apache_2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](pyproject.toml)

Prune a Hugging Face Mixture-of-Experts checkpoint by selecting the experts
that win a set of **random Gaussian probes** through the model's router.

The procedure is intentionally simple and dataset-free:

1. Download the original safetensors snapshot.
2. Read the per-layer router weights + pre-MoE RMSNorm scale.
3. Score each expert against `num_examples` random hidden states routed
   through the same RMSNorm and top-k softmax the model would use.
4. Keep the top `(1 - prune_level) * num_experts` experts per layer.
5. Rewrite every safetensors shard slicing the expert dimension, update
   `config.json` and the weight map, and copy the tokenizer / chat template.

It ships with a `random-moe-pruning` console script and a small Python API
(`random_moe_pruning.prune`).

## Installation

```bash
uv pip install random-moe-pruning
```

Or to hack on it locally:

```bash
git clone https://github.com/valendra-tech/random-moe-pruning-
cd random-moe-pruning-
uv sync
```

## Quick start

```bash
random-moe-pruning \
  --hf-repo openai/gpt-oss-20b \
  --prune-level 0.5 \
  --output-dir pruned_model \
  --workdir .workspace/
```

The same workflow as a library:

```python
from random_moe_pruning import PrunerConfig, prune

config = PrunerConfig(
    hf_repo="openai/gpt-oss-20b",
    prune_level=0.5,
    output_dir="pruned_model",
    workdir=".workspace/",
)
prune(config)
```

## Configuration

| Flag                      | Description                                                   |
|---------------------------|---------------------------------------------------------------|
| `--hf-repo`               | Hugging Face repository id (required).                        |
| `--prune-level`           | Fraction of experts to drop per layer (`0.0–1.0`).            |
| `--output-dir`            | Where the pruned checkpoint is written.                       |
| `--workdir`               | Where the original snapshot is downloaded.                    |
| `--random-seed`           | Seed for the random probes (default `42`).                    |
| `--num-examples`          | Number of probes per layer (default `1000`).                  |
| `--top-k`                 | Activated experts per token (default `4`).                    |
| `--router-pattern`        | Regex for the router tensors.                                 |
| `--rmsnorm-pattern`       | Regex for the pre-MoE RMSNorm tensor.                         |
| `--expert-pattern`        | Regex for the expert tensors.                                 |
| `--experts-per-token-key` | Config key holding the experts-per-token value.               |

The default regexes target the GPT-OSS layout, but other MoE checkpoints
can be pruned by overriding `--router-pattern`, `--rmsnorm-pattern` and
`--expert-pattern` together.

## Released checkpoints

A first pruned artefact of [`openai/gpt-oss-20b`](https://huggingface.co/openai/gpt-oss-20b)
(50% of the experts per layer) is published as an experimental
research checkpoint on the Hugging Face Hub:

- [`valendra/gpt-oss-11b-mini`](https://huggingface.co/valendra/gpt-oss-11b-mini)

It is generated with the default configuration of this CLI and is
intended as a hands-on example of **expert pruning**, the technique
of dropping a fraction of the experts from every MoE layer while
leaving the rest of the architecture (router, attention, embeddings,
tokenizer, chat template) untouched.

## Development

```bash
uv sync
uv run pytest
uv run ruff check
uv run random-moe-pruning --help
```

## License

Apache 2.0. See [LICENSE](LICENSE) for details.
