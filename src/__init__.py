"""
MMABA-PSEUDO: Mamba 2 Neural Memory Benchmark Suite

This package contains the core modules for training and evaluating
memory-augmented reinforcement learning agents.
"""

from .mamba_compat import patch_mamba_for_cpu
from .mem_actor_critic_mamba import MemActorCritic, PseudoModeMemory, PseudoModeState
from .neural_memory_final import make_env
from .wandb_integration import init_wandb, log_metrics, finish_wandb

__all__ = [
    "patch_mamba_for_cpu",
    "MemActorCritic",
    "PseudoModeMemory",
    "PseudoModeState",
    "make_env",
    "init_wandb",
    "log_metrics",
    "finish_wandb",
]
