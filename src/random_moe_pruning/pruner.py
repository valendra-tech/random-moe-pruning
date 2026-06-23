"""Public entry point that orchestrates the full pruning pipeline."""

from __future__ import annotations

import json
import logging
import os

from huggingface_hub import snapshot_download

from random_moe_pruning.config import ALLOW_PATTERNS, IGNORE_PATTERNS, PrunerConfig
from random_moe_pruning.loader import load_layer_tensors
from random_moe_pruning.probes import ProbeConfig, select_experts
from random_moe_pruning.writer import write_pruned_model

log = logging.getLogger(__name__)


def _setup_workspace(workdir: str) -> None:
    os.makedirs(workdir, exist_ok=True)


def _download_model(hf_repo: str, workdir: str) -> str:
    model_dir = os.path.join(workdir, hf_repo.replace("/", "__"))
    local_path = snapshot_download(
        repo_id=hf_repo,
        local_dir=model_dir,
        local_dir_use_symlinks=False,
        resume_download=True,
        allow_patterns=ALLOW_PATTERNS,
        ignore_patterns=IGNORE_PATTERNS,
    )
    log.info("Model snapshot ready at %s", local_path)
    return local_path


def _load_config_field(model_path: str, key: str) -> int:
    config_path = os.path.join(model_path, "config.json")
    with open(config_path, encoding="utf-8") as handle:
        config = json.load(handle)
    if key not in config:
        raise KeyError(f"{key} not found in model configuration.")
    return config[key]


def prune(config: PrunerConfig) -> str:
    """Run the full pruning pipeline and return the output directory."""
    log.info(
        "Pruning model=%s prune_level=%s output_dir=%s seed=%s",
        config.hf_repo,
        config.prune_level,
        config.output_dir,
        config.random_seed,
    )

    _setup_workspace(config.workdir)
    model_path = _download_model(config.hf_repo, config.workdir)

    layers = load_layer_tensors(model_path, config.router_re, config.rmsnorm_re)
    hidden_size = layers[0].router_weight.shape[1]
    experts_per_token = _load_config_field(model_path, config.experts_per_token_key)
    if config.num_experts_key:
        _load_config_field(model_path, config.num_experts_key)

    effective_top_k = config.top_k if config.top_k is not None else experts_per_token
    log.info("Using top_k=%d for expert scoring", effective_top_k)

    probe = ProbeConfig(
        num_examples=config.num_examples,
        top_k=effective_top_k,
        prune_level=config.prune_level,
        random_seed=config.random_seed,
    )
    experts_keep, _ = select_experts(layers, hidden_size, probe)

    return write_pruned_model(
        model_path=model_path,
        output_dir=config.output_dir,
        experts_keep=experts_keep,
        router_pattern=config.router_re,
        expert_pattern=config.expert_re,
        top_k=config.top_k,
        experts_per_token_key=config.experts_per_token_key,
        num_experts_key=config.num_experts_key,
    )
