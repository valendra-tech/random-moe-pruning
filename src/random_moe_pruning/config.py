"""Configuration dataclass for the pruner."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from re import Pattern

DEFAULT_ROUTER_RE = r"^model\.layers\.(\d+)\.mlp\.router\.(weight|bias)$"
DEFAULT_RMSNORM_RE = r"^model\.layers\.(\d+)\.post_attention_layernorm\.weight$"
DEFAULT_EXPERT_RE = r"^model\.layers\.(\d+)\.mlp\.experts\..*$"

ALLOW_PATTERNS: list[str] = [
    "*.safetensors",
    "model.safetensors.index.json",
    "config.json",
    "generation_config.json",
    "tokenizer*",
    "special_tokens_map.json",
    "chat_template.jinja",
    "*.py",
    "*.txt",
    "*.json",
    "*.model",
]

IGNORE_PATTERNS: list[str] = [
    "original/*",
    "metal/*",
]

ALLOW_PATTERNS = [
    "*.safetensors",
    "model.safetensors.index.json",
    "config.json",
    "generation_config.json",
    "tokenizer*",
    "special_tokens_map.json",
    "chat_template.jinja",
    "*.py",
    "*.txt",
    "*.json",
    "*.model",
]


@dataclass(frozen=True)
class PrunerConfig:
    """Parameters controlling a single pruning run."""

    hf_repo: str = "openai/gpt-oss-20b"
    prune_level: float = 0.5
    output_dir: str = "pruned_model"
    workdir: str = ".workspace/"
    random_seed: int = 42
    num_examples: int = 1000
    top_k: int | None = None
    experts_per_token_key: str = "num_experts_per_tok"
    num_experts_key: str | None = None
    router_pattern: str = DEFAULT_ROUTER_RE
    rmsnorm_pattern: str = DEFAULT_RMSNORM_RE
    expert_pattern: str = DEFAULT_EXPERT_RE

    router_re: Pattern[str] = field(init=False)
    rmsnorm_re: Pattern[str] = field(init=False)
    expert_re: Pattern[str] = field(init=False)

    def __post_init__(self) -> None:
        if not 0.0 <= self.prune_level < 1.0:
            raise ValueError(
                f"prune_level must be in [0.0, 1.0), got {self.prune_level}"
            )
        if self.num_examples <= 0:
            raise ValueError("num_examples must be positive")
        if self.top_k is not None and self.top_k <= 0:
            raise ValueError("top_k must be positive")
        object.__setattr__(self, "router_re", re.compile(self.router_pattern))
        object.__setattr__(self, "rmsnorm_re", re.compile(self.rmsnorm_pattern))
        object.__setattr__(self, "expert_re", re.compile(self.expert_pattern))
