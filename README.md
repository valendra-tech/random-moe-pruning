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

For quality checks after pruning, it also ships with
`random-moe-pruning-eval-http`, a separate CLI that compares a baseline model
and a candidate model through OpenAI-compatible HTTP endpoints.

The current implementation is optimized for the GPT-OSS family, especially
`openai/gpt-oss-120b`, and for checkpoints that follow the same score-routed
top-k MoE layout.

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

## Evaluate a pruned model over HTTP

If the original or pruned model is only available behind a remote inference API,
you can compare both over HTTP instead of running them locally.

Create a JSONL file with one test case per line:

```jsonl
{"name":"capital-france","prompt":"Capital of France?","expected":"Paris"}
{"name":"parity","prompt":"Reply with yes or no: is 14 even?","expected":"yes","match":"contains"}
```

Then run:

```bash
random-moe-pruning-eval-http \
   --cases examples/http_eval_cases.jsonl \
   --baseline-url https://your-reference-endpoint.example.com \
   --baseline-model original-model \
   --candidate-url https://your-pruned-endpoint.example.com \
   --candidate-model pruned-model \
   --output-json eval-report.json \
   --fail-on-regression
```

The evaluator sends the same prompts to both endpoints, then reports:

- pass/fail accuracy against the expected answer,
- average token-level F1 against the expected answer,
- regressions where the baseline passes but the pruned model fails.

The HTTP API is intentionally simple: it expects an OpenAI-style
`/v1/chat/completions` endpoint with `model`, `messages`, `temperature`, and
`max_tokens` in the request and `choices[0].message.content` in the response.
You can override the path and add per-endpoint headers or API keys via env vars.

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

The HTTP evaluator accepts these main options:

| Flag                      | Description                                                   |
|---------------------------|---------------------------------------------------------------|
| `--cases`                 | JSONL test set with `prompt` or `messages` and `expected`.    |
| `--baseline-url`          | Base URL of the reference endpoint.                           |
| `--baseline-model`        | Model id sent to the reference endpoint.                      |
| `--candidate-url`         | Base URL of the pruned/candidate endpoint.                    |
| `--candidate-model`       | Model id sent to the candidate endpoint.                      |
| `--output-json`           | Optional path to persist the full comparison report.          |
| `--fail-on-regression`    | Exit non-zero if candidate regresses on a baseline pass case. |

The default regexes target the GPT-OSS layout, but other MoE checkpoints
can be pruned by overriding `--router-pattern`, `--rmsnorm-pattern` and
`--expert-pattern` together.

That naming flexibility is necessary but not sufficient for arbitrary MoE
families. The pruning logic still assumes a linear score-based router, a
uniform expert count across layers, and expert tensors that can be filtered or
sliced consistently with the router rows.

In practice, this makes the CLI a good fit for GPT-OSS models, especially
`openai/gpt-oss-120b`, and for close layout-compatible variants. Other MoE
checkpoints can require model-specific support even if their tensor names can
be matched with custom regexes.

Hash-routed MoE layouts are not supported by the current pruning algorithm.
For example, DeepSeek-V4-Flash uses `gate.tid2eid` in its first hash-routed
layers, so this CLI now rejects that checkpoint instead of emitting a degraded
model.

Other known caveats:

- DeepSeek-V3.2 uses grouped routing plus `e_score_correction_bias`, so the
   current random-probe scorer does not match the model's real expert selection
   path.
- MiniMax-M2.7 is closer to the supported GPT-OSS layout, but still needs
   model-specific handling for its gate tensors and scoring path before results
   should be trusted.

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

## Authors

José Carlos García Ortega and Marta Barea Sepúlveda, PhD, Valendra Tech SL.

## Future work

- Add SFT after pruning using the full model as a teacher.
- Review and support additional architectures.

## Development

```bash
uv sync
uv run pytest
uv run ruff check
uv run random-moe-pruning --help
```

## License

Apache 2.0. See [LICENSE](LICENSE) for details.
