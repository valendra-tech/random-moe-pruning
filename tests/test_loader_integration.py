"""Regression tests that touch real safetensors files."""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from random_moe_pruning.config import DEFAULT_RMSNORM_RE, DEFAULT_ROUTER_RE
from random_moe_pruning.loader import load_layer_tensors

pytestmark = pytest.mark.integration


def _write_minimal_shard(path: Path) -> None:
    """Create a shard with router + RMSNorm tensors for two layers."""
    hidden, num_experts = 4, 6
    tensors: dict[str, torch.Tensor] = {
        "model.layers.0.mlp.router.weight": torch.zeros(num_experts, hidden),
        "model.layers.0.mlp.router.bias": torch.zeros(num_experts),
        "model.layers.0.post_attention_layernorm.weight": torch.ones(hidden),
        "model.layers.1.mlp.router.weight": torch.ones(num_experts, hidden),
        "model.layers.1.mlp.router.bias": torch.ones(num_experts),
        "model.layers.1.post_attention_layernorm.weight": torch.ones(hidden),
        "model.layers.0.mlp.experts.gate_up_proj": torch.zeros(num_experts, 2, hidden),
    }
    save_file(tensors, str(path))


def test_load_layer_tensors_reads_safetensors_shard(tmp_path: Path) -> None:
    shard = tmp_path / "model-00000-of-00001.safetensors"
    _write_minimal_shard(shard)
    layers = load_layer_tensors(
        str(tmp_path),
        re.compile(DEFAULT_ROUTER_RE),
        re.compile(DEFAULT_RMSNORM_RE),
    )
    assert len(layers) == 2
    assert layers[0].router_weight.shape == (6, 4)
    assert layers[1].router_weight.shape == (6, 4)
    assert layers[0].router_bias is not None
    assert layers[0].rmsnorm_weight.shape == (4,)


def test_load_layer_tensors_allows_missing_router_bias(tmp_path: Path) -> None:
    shard = tmp_path / "model-00000-of-00001.safetensors"
    hidden, num_experts = 4, 6
    tensors: dict[str, torch.Tensor] = {
        "model.layers.0.mlp.router.weight": torch.zeros(num_experts, hidden),
        "model.layers.0.post_attention_layernorm.weight": torch.ones(hidden),
    }
    save_file(tensors, str(shard))

    layers = load_layer_tensors(
        str(tmp_path),
        re.compile(DEFAULT_ROUTER_RE),
        re.compile(DEFAULT_RMSNORM_RE),
    )

    assert len(layers) == 1
    assert layers[0].layer_id == 0
    assert layers[0].router_bias is None
