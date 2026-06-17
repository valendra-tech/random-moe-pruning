"""Tests for the pure (CPU-only) helpers."""

from __future__ import annotations

import re

import numpy as np
import pytest
import torch

from random_moe_pruning.config import PrunerConfig
from random_moe_pruning.loader import LayerTensors
from random_moe_pruning.probes import (
    EPS,
    ProbeConfig,
    rms_norm,
    score_experts,
    select_experts,
)
from random_moe_pruning.writer import (
    prune_tensor,
    update_config_for_pruned_experts,
    validate_uniform_keep,
)


def test_rms_norm_matches_manual_formula():
    x = np.array([[3.0, 4.0]], dtype=np.float32)
    weight = np.array([1.0, 1.0], dtype=np.float32)
    expected = x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + EPS)
    np.testing.assert_allclose(rms_norm(x, weight), expected, rtol=1e-6)


def test_pruner_config_rejects_invalid_prune_level():
    with pytest.raises(ValueError):
        PrunerConfig(prune_level=1.0)
    with pytest.raises(ValueError):
        PrunerConfig(prune_level=-0.1)


def test_pruner_config_compiles_patterns():
    cfg = PrunerConfig()
    assert isinstance(cfg.router_re, re.Pattern)
    assert cfg.router_re.pattern == cfg.router_pattern


def test_score_experts_aggregates_softmax_weights():
    hidden = 8
    num_experts = 5
    layer = LayerTensors(
        layer_id=0,
        router_weight=np.zeros((num_experts, hidden), dtype=np.float32),
        router_bias=None,
        rmsnorm_weight=np.ones(hidden, dtype=np.float32),
    )
    layer.router_weight[2] = 10.0  # expert 2 dominates the top-k
    probe = ProbeConfig(num_examples=64, top_k=2, random_seed=0)
    scores = score_experts(layer, hidden, probe)
    assert scores[2] == scores.max()
    assert scores[2] > scores.sum() / num_experts
    assert scores.sum() == pytest.approx(64.0, rel=1e-3)


def test_select_experts_keeps_expected_count():
    hidden = 4
    layers = [
        LayerTensors(
            layer_id=0,
            router_weight=np.eye(8, hidden, dtype=np.float32),
            router_bias=None,
            rmsnorm_weight=np.ones(hidden, dtype=np.float32),
        )
    ]
    probe = ProbeConfig(num_examples=200, top_k=2, prune_level=0.5, random_seed=1)
    keep, _ = select_experts(layers, hidden, probe)
    assert set(keep.keys()) == {0}
    assert len(keep[0]) == 4
    assert len(set(keep[0])) == 4


def test_select_experts_rejects_unsafe_prune_level():
    hidden = 4
    layers = [
        LayerTensors(
            layer_id=0,
            router_weight=np.eye(2, hidden, dtype=np.float32),
            router_bias=None,
            rmsnorm_weight=np.ones(hidden, dtype=np.float32),
        )
    ]
    probe = ProbeConfig(num_examples=10, top_k=4, prune_level=0.5, random_seed=0)
    with pytest.raises(ValueError):
        select_experts(layers, hidden, probe)


def test_validate_uniform_keep_rejects_mixed_sizes():
    with pytest.raises(ValueError):
        validate_uniform_keep({0: [0, 1], 1: [0, 1, 2]})


def test_validate_uniform_keep_rejects_duplicates():
    with pytest.raises(ValueError):
        validate_uniform_keep({0: [0, 0, 1]})


def test_prune_tensor_uses_router_pattern():
    cfg = PrunerConfig()
    tensor = torch.arange(6).reshape(3, 2)
    pruned = prune_tensor(
        "model.layers.0.mlp.router.weight",
        tensor,
        {0: [0, 2]},
        cfg.router_re,
        cfg.expert_re,
    )
    assert pruned.tolist() == [[0, 1], [4, 5]]


def test_prune_tensor_uses_expert_pattern():
    cfg = PrunerConfig()
    tensor = torch.zeros(4, 3, 2)
    tensor[1] = 1
    tensor[3] = 9
    pruned = prune_tensor(
        "model.layers.1.mlp.experts.gate_up_proj",
        tensor,
        {1: [1, 3]},
        cfg.router_re,
        cfg.expert_re,
    )
    assert pruned.shape == (2, 3, 2)
    assert pruned[0].eq(1).all()
    assert pruned[1].eq(9).all()


def test_prune_tensor_keeps_named_expert_tensor_when_id_is_retained():
    cfg = PrunerConfig()
    tensor = torch.arange(6).reshape(3, 2)
    pruned = prune_tensor(
        "model.layers.1.mlp.experts.3.gate_up_proj.weight",
        tensor,
        {1: [1, 3]},
        cfg.router_re,
        cfg.expert_re,
    )
    assert pruned is tensor


def test_prune_tensor_drops_named_expert_tensor_when_id_is_pruned():
    cfg = PrunerConfig()
    tensor = torch.arange(6).reshape(3, 2)
    pruned = prune_tensor(
        "model.layers.1.mlp.experts.2.gate_up_proj.weight",
        tensor,
        {1: [1, 3]},
        cfg.router_re,
        cfg.expert_re,
    )
    assert pruned is None


def test_prune_tensor_passes_through_unmatched():
    cfg = PrunerConfig()
    tensor = torch.ones(2, 2)
    out = prune_tensor(
        "model.layers.0.mlp.other.weight",
        tensor,
        {0: [0]},
        cfg.router_re,
        cfg.expert_re,
    )
    assert out is tensor


def test_update_config_uses_cli_top_k_and_key(tmp_path):
    model_dir = tmp_path / "model"
    out_dir = tmp_path / "out"
    model_dir.mkdir()
    out_dir.mkdir()

    config_path = model_dir / "config.json"
    config_path.write_text(
        """
{
  "num_local_experts": 16,
  "num_experts_per_tok": 8
}
""".strip(),
        encoding="utf-8",
    )

    update_config_for_pruned_experts(
        model_path=str(model_dir),
        output_dir=str(out_dir),
        keep_experts=8,
        top_k=4,
        experts_per_token_key="num_experts_per_tok",
    )

    updated = (out_dir / "config.json").read_text(encoding="utf-8")
    assert '"num_local_experts": 8' in updated
    assert '"num_experts_per_tok": 4' in updated


def test_update_config_uses_cli_num_experts_key(tmp_path):
    model_dir = tmp_path / "model"
    out_dir = tmp_path / "out"
    model_dir.mkdir()
    out_dir.mkdir()

    config_path = model_dir / "config.json"
    config_path.write_text(
        """
{
  "custom_total_experts": 20,
  "num_experts_per_tok": 4
}
""".strip(),
        encoding="utf-8",
    )

    update_config_for_pruned_experts(
        model_path=str(model_dir),
        output_dir=str(out_dir),
        keep_experts=10,
        top_k=4,
        experts_per_token_key="num_experts_per_tok",
        num_experts_key="custom_total_experts",
    )

    updated = (out_dir / "config.json").read_text(encoding="utf-8")
    assert '"custom_total_experts": 10' in updated


def test_update_config_supports_deepseek_expert_count_key(tmp_path):
    model_dir = tmp_path / "model"
    out_dir = tmp_path / "out"
    model_dir.mkdir()
    out_dir.mkdir()

    config_path = model_dir / "config.json"
    config_path.write_text(
        """
{
  "n_routed_experts": 256,
  "num_experts_per_tok": 6
}
""".strip(),
        encoding="utf-8",
    )

    update_config_for_pruned_experts(
        model_path=str(model_dir),
        output_dir=str(out_dir),
        keep_experts=128,
        top_k=6,
        experts_per_token_key="num_experts_per_tok",
    )

    updated = (out_dir / "config.json").read_text(encoding="utf-8")
    assert '"n_routed_experts": 128' in updated
    assert '"num_experts_per_tok": 6' in updated


def test_update_config_does_not_write_top_k_into_expert_count_key(tmp_path):
    model_dir = tmp_path / "model"
    out_dir = tmp_path / "out"
    model_dir.mkdir()
    out_dir.mkdir()

    config_path = model_dir / "config.json"
    config_path.write_text(
        """
{
  "n_routed_experts": 256,
  "num_experts_per_tok": 6
}
""".strip(),
        encoding="utf-8",
    )

    update_config_for_pruned_experts(
        model_path=str(model_dir),
        output_dir=str(out_dir),
        keep_experts=128,
        top_k=6,
        experts_per_token_key="n_routed_experts",
    )

    updated = (out_dir / "config.json").read_text(encoding="utf-8")
    assert '"n_routed_experts": 128' in updated
    assert '"num_experts_per_tok": 6' in updated
