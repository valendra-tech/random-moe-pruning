"""Random MoE Pruning: drop experts from a Hugging Face MoE checkpoint."""

from random_moe_pruning.config import PrunerConfig
from random_moe_pruning.pruner import prune

__all__ = ["PrunerConfig", "prune"]
__version__ = "0.1.0"
