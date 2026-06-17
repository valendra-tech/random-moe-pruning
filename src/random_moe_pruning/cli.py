"""Command-line interface for ``random-moe-pruning``."""

from __future__ import annotations

import logging
import sys

import click

from random_moe_pruning import __version__
from random_moe_pruning.config import (
    DEFAULT_EXPERT_RE,
    DEFAULT_RMSNORM_RE,
    DEFAULT_ROUTER_RE,
    PrunerConfig,
)
from random_moe_pruning.pruner import prune


def _configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.version_option(__version__, prog_name="random-moe-pruning")
@click.option(
    "--hf-repo",
    required=True,
    help="Hugging Face repository id to prune.",
)
@click.option(
    "--prune-level",
    default=0.5,
    show_default=True,
    type=click.FloatRange(0.0, 1.0, max_open=True),
    help="Fraction of experts to drop per layer (0 keeps all, <1 keeps at least one).",
)
@click.option(
    "--output-dir",
    default="pruned_model",
    show_default=True,
    help="Directory to write the pruned checkpoint.",
)
@click.option(
    "--workdir",
    default=".workspace/",
    show_default=True,
    help="Working directory for downloaded snapshots.",
)
@click.option(
    "--router-pattern",
    default=DEFAULT_ROUTER_RE,
    show_default=True,
    help="Regex matching the router tensors.",
)
@click.option(
    "--rmsnorm-pattern",
    default=DEFAULT_RMSNORM_RE,
    show_default=True,
    help="Regex matching the pre-MoE RMSNorm tensor.",
)
@click.option(
    "--expert-pattern",
    default=DEFAULT_EXPERT_RE,
    show_default=True,
    help="Regex matching expert tensors.",
)
@click.option(
    "--random-seed",
    default=42,
    show_default=True,
    type=int,
    help="Seed for the random probes.",
)
@click.option(
    "--num-examples",
    default=1000,
    show_default=True,
    type=int,
    help="Number of random probes per layer.",
)
@click.option(
    "--top-k",
    default=4,
    show_default=True,
    type=int,
    help="Activated experts per token (router top-k).",
)
@click.option(
    "--experts-per-token-key",
    default="num_experts_per_tok",
    show_default=True,
    help="Config key holding the experts-per-token value.",
)
@click.option(
    "--num-experts-key",
    default=None,
    help="Config key holding the total routed experts (overrides auto-detection).",
)
@click.option(
    "-v",
    "--verbose",
    is_flag=True,
    help="Emit debug-level logs.",
)
def main(
    hf_repo: str,
    prune_level: float,
    output_dir: str,
    workdir: str,
    router_pattern: str,
    rmsnorm_pattern: str,
    expert_pattern: str,
    random_seed: int,
    num_examples: int,
    top_k: int,
    experts_per_token_key: str,
    num_experts_key: str | None,
    verbose: bool,
) -> None:
    """Prune an MoE Hugging Face checkpoint by random probe selection."""
    _configure_logging(verbose)
    config = PrunerConfig(
        hf_repo=hf_repo,
        prune_level=prune_level,
        output_dir=output_dir,
        workdir=workdir,
        router_pattern=router_pattern,
        rmsnorm_pattern=rmsnorm_pattern,
        expert_pattern=expert_pattern,
        random_seed=random_seed,
        num_examples=num_examples,
        top_k=top_k,
        experts_per_token_key=experts_per_token_key,
        num_experts_key=num_experts_key,
    )
    prune(config)
    return None


if __name__ == "__main__":
    sys.exit(main())
