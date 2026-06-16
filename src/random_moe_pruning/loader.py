"""Utilities for loading router/RMSNorm weights from a safetensors checkpoint."""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass

from safetensors import safe_open

log = logging.getLogger(__name__)

LayerStats = dict[int, dict[str, object]]


@dataclass
class LayerTensors:
    """Per-layer tensors needed for expert selection."""

    router_weight: object
    router_bias: object | None
    rmsnorm_weight: object


def _scan_tensor_keys(model_path: str, patterns: list[re.Pattern[str]]) -> LayerStats:
    """Return one ``LayerTensors`` per matched layer id."""
    safetensors_files = sorted(
        name for name in os.listdir(model_path) if name.endswith(".safetensors")
    )
    if not safetensors_files:
        raise FileNotFoundError(f"No .safetensors files found in {model_path}")

    layers: dict[int, dict[str, object]] = {}
    for shard in safetensors_files:
        shard_path = os.path.join(model_path, shard)
        log.info("Scanning %s", shard_path)
        with safe_open(shard_path, framework="pt", device="cpu") as handle:
            for key in handle:
                for pattern in patterns:
                    match = pattern.match(key)
                    if match is None:
                        continue
                    layer_id = int(match.group(1))
                    layers.setdefault(layer_id, {})
                    if pattern.pattern.startswith(
                        r"^model\.layers\.(\d+)\.mlp\.router"
                    ):
                        field_name = match.group(2)
                        layers[layer_id].setdefault("router", {})[field_name] = (
                            handle.get_tensor(key).float().numpy()
                        )
                    else:
                        layers[layer_id]["rmsnorm"] = (
                            handle.get_tensor(key).float().numpy()
                        )
                    break
    return layers


def _validate_layers(layers: LayerStats) -> list[LayerTensors]:
    """Convert raw dicts to typed ``LayerTensors`` and assert required fields."""
    missing: list[tuple[int, str]] = []
    typed: list[LayerTensors] = []
    for layer_id in sorted(layers):
        data = layers[layer_id]
        router = data.get("router") or {}
        if "weight" not in router:
            missing.append((layer_id, "router.weight"))
        if "bias" not in router:
            missing.append((layer_id, "router.bias"))
        if "rmsnorm" not in data:
            missing.append((layer_id, "rmsnorm"))
        if missing and missing[-1][0] == layer_id:
            continue
        typed.append(
            LayerTensors(
                router_weight=router["weight"],
                router_bias=router.get("bias"),
                rmsnorm_weight=data["rmsnorm"],
            )
        )
    if missing:
        log.warning("Missing expected tensors: %s", missing[:20])
    if not typed:
        raise ValueError("No layers with complete router/RMSNorm tensors found")
    return typed


def load_layer_tensors(
    model_path: str,
    router_pattern: re.Pattern[str],
    rmsnorm_pattern: re.Pattern[str],
) -> list[LayerTensors]:
    """Load the router + RMSNorm tensors used to drive expert selection."""
    raw = _scan_tensor_keys(model_path, [router_pattern, rmsnorm_pattern])
    return _validate_layers(raw)
