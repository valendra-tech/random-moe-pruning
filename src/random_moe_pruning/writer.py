"""Rewrite a safetensors checkpoint keeping only the selected experts per layer."""

from __future__ import annotations

import json
import logging
import os
import re
import shutil

import torch
from safetensors import safe_open
from safetensors.torch import save_file

log = logging.getLogger(__name__)

SKIP_SUFFIXES = (".safetensors", ".bin", ".pt", ".pth")
SKIP_NAMES = {
    "model.safetensors.index.json",
    "pytorch_model.bin.index.json",
}


def copy_non_weight_files(model_path: str, output_dir: str) -> None:
    """Copy tokenizer, config and code files, skipping the heavy shards."""
    os.makedirs(output_dir, exist_ok=True)
    for name in os.listdir(model_path):
        src = os.path.join(model_path, name)
        dst = os.path.join(output_dir, name)
        if name in SKIP_NAMES or name.endswith(SKIP_SUFFIXES):
            continue
        if os.path.isdir(src):
            if os.path.exists(dst):
                shutil.rmtree(dst)
            shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)


def update_config_for_pruned_experts(
    model_path: str,
    output_dir: str,
    keep_experts: int,
) -> None:
    """Patch ``config.json`` with the new expert count."""
    config_path = os.path.join(model_path, "config.json")
    out_config_path = os.path.join(output_dir, "config.json")
    with open(config_path, encoding="utf-8") as handle:
        config = json.load(handle)

    old_num_experts = (
        config.get("num_local_experts")
        or config.get("num_experts")
        or config.get("n_experts")
    )
    if old_num_experts is None:
        raise ValueError("Could not find num_local_experts/num_experts in config.json")

    top_k = (
        config.get("num_experts_per_tok")
        or config.get("num_experts_per_token")
        or config.get("top_k")
        or 1
    )
    if keep_experts < top_k:
        raise ValueError(
            f"Invalid pruned config: keep_experts={keep_experts} < top_k={top_k}"
        )

    for key in ("num_local_experts", "num_experts", "n_experts"):
        if key in config:
            config[key] = keep_experts

    config["pruned_from_num_experts"] = old_num_experts
    config["pruned_to_num_experts"] = keep_experts

    with open(out_config_path, "w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2, ensure_ascii=False)

    log.info(
        "Updated config: num_experts %s -> %s, top_k=%s",
        old_num_experts,
        keep_experts,
        top_k,
    )


def load_index(model_path: str) -> dict:
    """Read the original safetensors index file."""
    index_path = os.path.join(model_path, "model.safetensors.index.json")
    if not os.path.exists(index_path):
        raise FileNotFoundError(f"Missing safetensors index: {index_path}")
    with open(index_path, encoding="utf-8") as handle:
        return json.load(handle)


def prune_tensor(
    name: str,
    tensor: torch.Tensor,
    experts_keep: dict[int, list[int]],
    router_pattern: re.Pattern[str],
    expert_pattern: re.Pattern[str],
) -> torch.Tensor:
    """Slice the expert dimension of router or expert tensors."""
    router_match = router_pattern.match(name)
    if router_match is not None:
        layer_id = int(router_match.group(1))
        idx = torch.tensor(experts_keep[layer_id], dtype=torch.long)
        return tensor.index_select(0, idx)

    expert_match = expert_pattern.match(name)
    if expert_match is not None:
        layer_id = int(expert_match.group(1))
        idx = torch.tensor(experts_keep[layer_id], dtype=torch.long)
        return tensor.index_select(0, idx)

    return tensor


def validate_uniform_keep(experts_keep: dict[int, list[int]]) -> int:
    """Ensure every layer keeps the same number of experts."""
    if not experts_keep:
        raise ValueError("experts_keep is empty")
    sizes = {layer_id: len(values) for layer_id, values in experts_keep.items()}
    if len(set(sizes.values())) != 1:
        raise ValueError(
            "This writer requires the same number of kept experts per layer. "
            f"Got sizes: {sizes}"
        )
    keep_experts = next(iter(sizes.values()))
    for layer_id, ids in experts_keep.items():
        if len(set(ids)) != len(ids):
            raise ValueError(f"Duplicate expert ids in layer {layer_id}: {ids}")
        if min(ids) < 0:
            raise ValueError(f"Negative expert id in layer {layer_id}: {ids}")
    return keep_experts


def write_pruned_model(
    model_path: str,
    output_dir: str,
    experts_keep: dict[int, list[int]],
    router_pattern: re.Pattern[str],
    expert_pattern: re.Pattern[str],
) -> str:
    """Materialize the pruned model on disk and return ``output_dir``."""
    keep_experts = validate_uniform_keep(experts_keep)
    log.info(
        "Writing pruned model with %s experts/layer to %s", keep_experts, output_dir
    )

    os.makedirs(output_dir, exist_ok=True)
    copy_non_weight_files(model_path, output_dir)
    update_config_for_pruned_experts(model_path, output_dir, keep_experts)

    index = load_index(model_path)
    weight_map = index["weight_map"]
    shards = sorted(set(weight_map.values()))

    new_weight_map: dict[str, str] = {}
    total_tensors = 0
    pruned_tensors = 0

    for shard_name in shards:
        in_path = os.path.join(model_path, shard_name)
        out_path = os.path.join(output_dir, shard_name)
        log.info("Rewriting shard: %s", shard_name)
        tensors_out: dict[str, torch.Tensor] = {}
        with safe_open(in_path, framework="pt", device="cpu") as handle:
            for tensor_name in handle.keys():
                tensor = handle.get_tensor(tensor_name)
                is_prunable = bool(
                    router_pattern.match(tensor_name)
                    or expert_pattern.match(tensor_name)
                )
                tensors_out[tensor_name] = prune_tensor(
                    name=tensor_name,
                    tensor=tensor,
                    experts_keep=experts_keep,
                    router_pattern=router_pattern,
                    expert_pattern=expert_pattern,
                )
                new_weight_map[tensor_name] = shard_name
                total_tensors += 1
                if is_prunable:
                    pruned_tensors += 1
        save_file(tensors_out, out_path)

    out_index_path = os.path.join(output_dir, "model.safetensors.index.json")
    with open(out_index_path, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "metadata": index.get("metadata", {}),
                "weight_map": new_weight_map,
            },
            handle,
            indent=2,
            ensure_ascii=False,
        )

    log.info("Rewrote %s tensors (%s pruned)", total_tensors, pruned_tensors)
    return output_dir
