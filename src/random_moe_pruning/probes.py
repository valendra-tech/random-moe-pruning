"""Random-probe expert selection for MoE models."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from random_moe_pruning.loader import LayerTensors

EPS = 1e-5


@dataclass(frozen=True)
class ProbeConfig:
    """Hyper-parameters for the random-probe scoring step."""

    num_examples: int = 1000
    top_k: int = 4
    prune_level: float = 0.5
    random_seed: int = 42
    eps: float = EPS


def rms_norm(x: np.ndarray, weight: np.ndarray, eps: float = EPS) -> np.ndarray:
    """Apply the same scaling used by the model's pre-MoE RMSNorm."""
    rms = np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps)
    return (x / rms) * weight


def score_experts(
    layer: LayerTensors,
    hidden_size: int,
    probe: ProbeConfig,
) -> np.ndarray:
    """Return accumulated router weights per expert for ``layer``."""
    rng = np.random.default_rng(probe.random_seed)
    router_weight = layer.router_weight.astype(np.float32)
    router_bias = layer.router_bias
    rms_weight = layer.rmsnorm_weight.astype(np.float32)

    probes = rng.standard_normal(
        size=(probe.num_examples, hidden_size), dtype=np.float32
    )
    probes = rms_norm(probes, rms_weight, eps=probe.eps)
    logits = probes @ router_weight.T
    if router_bias is not None:
        logits += router_bias.astype(np.float32)

    top_k = probe.top_k
    topk_idx = np.argpartition(logits, -top_k, axis=-1)[:, -top_k:]
    topk_logits = np.take_along_axis(logits, topk_idx, axis=-1)
    topk_logits = topk_logits - np.max(topk_logits, axis=-1, keepdims=True)
    weights = np.exp(topk_logits)
    weights = weights / np.sum(weights, axis=-1, keepdims=True)

    scores = np.zeros(router_weight.shape[0], dtype=np.float64)
    for slot in range(top_k):
        expert_ids = topk_idx[:, slot]
        np.add.at(scores, expert_ids, weights[:, slot])
    return scores


def select_experts(
    layers: list[LayerTensors],
    hidden_size: int,
    probe: ProbeConfig,
) -> tuple[dict[int, list[int]], dict[int, np.ndarray]]:
    """Pick the experts to keep per layer based on random-probe scores."""
    import logging
    log = logging.getLogger(__name__)
    experts_keep: dict[int, list[int]] = {}
    expert_scores: dict[int, np.ndarray] = {}

    for layer_index, layer in enumerate(layers):
        num_experts = layer.router_weight.shape[0]
        keep_count = int(np.ceil(num_experts * (1.0 - probe.prune_level)))
        if layer_index == 0:
            log.info("Pruning config: num_experts=%d, prune_level=%.2f → keep_count=%d", num_experts, probe.prune_level, keep_count)
        if keep_count < probe.top_k:
            raise ValueError(
                f"Layer {layer_index}: keep_count={keep_count} < top_k={probe.top_k}."
                " Unsafe pruning level."
            )
        scores = score_experts(layer, hidden_size, probe)
        keep_idx = np.sort(np.argsort(scores)[-keep_count:])
        layer_id = layer.layer_id if layer.layer_id >= 0 else layer_index
        experts_keep[layer_id] = keep_idx.tolist()
        expert_scores[layer_id] = scores
    return experts_keep, expert_scores
