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

EXPERT_COUNT_KEYS = (
    "num_local_experts",
    "num_experts",
    "n_experts",
    "n_routed_experts",
    "num_routed_experts",
)

TOP_K_KEYS = (
    "num_experts_per_tok",
    "num_experts_per_token",
    "top_k",
    "n_activated_experts",
    "num_activated_experts",
)

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
    top_k: int | None = None,
    experts_per_token_key: str | None = None,
    num_experts_key: str | None = None,
) -> None:
    """Patch ``config.json`` with the new expert count."""
    config_path = os.path.join(model_path, "config.json")
    out_config_path = os.path.join(output_dir, "config.json")
    with open(config_path, encoding="utf-8") as handle:
        config = json.load(handle)

    expert_count_keys = [*EXPERT_COUNT_KEYS]
    if num_experts_key and num_experts_key not in expert_count_keys:
        expert_count_keys.insert(0, num_experts_key)

    old_num_experts = next((config[key] for key in expert_count_keys if key in config), None)
    if old_num_experts is None:
        raise ValueError(
            "Could not find expert-count key in config.json. "
            f"Tried: {', '.join(expert_count_keys)}"
        )

    inferred_top_k = next((config[key] for key in TOP_K_KEYS if key in config), 1)
    effective_top_k = top_k if top_k is not None else inferred_top_k
    if top_k is not None:
        if experts_per_token_key:
            if experts_per_token_key in expert_count_keys:
                log.warning(
                    "Skipping update of %s with top_k=%s because it is an expert-count key.",
                    experts_per_token_key,
                    top_k,
                )
            else:
                config[experts_per_token_key] = top_k
        for key in TOP_K_KEYS:
            if key in config:
                config[key] = top_k
    if keep_experts < effective_top_k:
        raise ValueError(
            "Invalid pruned config: "
            f"keep_experts={keep_experts} < top_k={effective_top_k}"
        )

    if num_experts_key:
        config[num_experts_key] = keep_experts
    for key in expert_count_keys:
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
        effective_top_k,
    )


def load_index(model_path: str) -> dict:
    """Read the original safetensors index file."""
    index_path = os.path.join(model_path, "model.safetensors.index.json")
    if not os.path.exists(index_path):
        raise FileNotFoundError(f"Missing safetensors index: {index_path}")
    with open(index_path, encoding="utf-8") as handle:
        return json.load(handle)


def _match_expert_tensor_name(
    name: str,
    expert_pattern: re.Pattern[str],
) -> tuple[int, str, str, str] | None:
    """Return layer id and expert-name components for expert tensors."""
    expert_match = expert_pattern.match(name)
    if expert_match is None or ".experts." not in name:
        return None

    layer_id = int(expert_match.group(1))
    expert_prefix, expert_suffix = name.split(".experts.", 1)
    expert_id_text, dot, remainder = expert_suffix.partition(".")
    return layer_id, expert_prefix, expert_id_text, remainder if dot else ""


def prune_tensor(
    name: str,
    tensor: torch.Tensor,
    experts_keep: dict[int, list[int]],
    router_pattern: re.Pattern[str],
    expert_pattern: re.Pattern[str],
) -> torch.Tensor | None:
    """Prune router tensors or drop per-expert tensors that are not retained."""
    router_match = router_pattern.match(name)
    if router_match is not None:
        layer_id = int(router_match.group(1))
        idx = torch.tensor(experts_keep[layer_id], dtype=torch.long)
        return tensor.index_select(0, idx)

    expert_info = _match_expert_tensor_name(name, expert_pattern)
    if expert_info is not None:
        layer_id, _, expert_id_text, _ = expert_info
        if expert_id_text.isdigit():
            expert_id = int(expert_id_text)
            return tensor if expert_id in experts_keep[layer_id] else None
        idx = torch.tensor(experts_keep[layer_id], dtype=torch.long)
        return tensor.index_select(0, idx)

    return tensor


def _rename_pruned_tensor(
    name: str,
    expert_pattern: re.Pattern[str],
    expert_id_remap: dict[int, dict[int, int]],
) -> str:
    """Compact retained named experts so checkpoint ids match pruned router rows."""
    expert_info = _match_expert_tensor_name(name, expert_pattern)
    if expert_info is None:
        return name

    layer_id, expert_prefix, expert_id_text, remainder = expert_info
    if not expert_id_text.isdigit():
        return name

    new_expert_id = expert_id_remap[layer_id][int(expert_id_text)]
    if remainder:
        return f"{expert_prefix}.experts.{new_expert_id}.{remainder}"
    return f"{expert_prefix}.experts.{new_expert_id}"


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
    top_k: int | None = None,
    experts_per_token_key: str | None = None,
    num_experts_key: str | None = None,
) -> str:
    """Materialize the pruned model on disk and return ``output_dir``."""
    keep_experts = validate_uniform_keep(experts_keep)
    log.info(
        "Writing pruned model with %s experts/layer to %s", keep_experts, output_dir
    )
    expert_id_remap = {
        layer_id: {expert_id: new_id for new_id, expert_id in enumerate(keep_ids)}
        for layer_id, keep_ids in experts_keep.items()
    }

    os.makedirs(output_dir, exist_ok=True)
    copy_non_weight_files(model_path, output_dir)
    update_config_for_pruned_experts(
        model_path=model_path,
        output_dir=output_dir,
        keep_experts=keep_experts,
        top_k=top_k,
        experts_per_token_key=experts_per_token_key,
        num_experts_key=num_experts_key,
    )

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
                pruned_tensor = prune_tensor(
                    name=tensor_name,
                    tensor=tensor,
                    experts_keep=experts_keep,
                    router_pattern=router_pattern,
                    expert_pattern=expert_pattern,
                )
                if pruned_tensor is None:
                    total_tensors += 1
                    if is_prunable:
                        pruned_tensors += 1
                    continue
                output_name = _rename_pruned_tensor(
                    tensor_name,
                    expert_pattern=expert_pattern,
                    expert_id_remap=expert_id_remap,
                )
                tensors_out[output_name] = pruned_tensor
                new_weight_map[output_name] = shard_name
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
