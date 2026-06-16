# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project follows [Semantic Versioning](https://semver.org/).

## [0.1.0] - 2026-06-16

### Added

- Initial public release.
- `random-moe-pruning` console script and `random_moe_pruning.prune` API.
- Random-probe expert selection for MoE checkpoints (GPT-OSS layout by default).
- Safetensors rewrite that preserves tokenizer, chat template and metadata.
- `pyproject.toml` powered by [uv](https://docs.astral.sh/uv/) and
  [hatchling](https://hatch.pypa.io/).
- `pytest` + `ruff` tooling.
