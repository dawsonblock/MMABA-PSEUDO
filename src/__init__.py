"""
MMABA-PSEUDO: Mamba 2 Neural Memory Benchmark Suite

This package contains the core modules for training and evaluating
memory-augmented reinforcement learning agents.

Note: Mamba imports are lazy-loaded to avoid errors when mamba_ssm is not installed.
Use controller="gru" if you don't have Mamba installed.
"""

# Import lightweight modules that don't trigger Mamba loading
from .neural_memory_final import make_env
from .wandb_integration import init_wandb, log_metrics, finish_wandb

# Lazy imports for MemActorCritic to avoid triggering Mamba errors at package import
# Users should import directly: from src.mem_actor_critic_mamba import MemActorCritic
def get_actor_critic():
    """
    Get MemActorCritic class (lazy import to avoid Mamba loading at package import).
    
    Returns:
        MemActorCritic class
    """
    from .mem_actor_critic_mamba import MemActorCritic
    return MemActorCritic


def get_pseudo_mode_memory():
    """
    Get PseudoModeMemory and related classes (lazy import).
    
    Returns:
        Tuple of (PseudoModeMemory, PseudoModeState)
    """
    from .mem_actor_critic_mamba import PseudoModeMemory, PseudoModeState
    return PseudoModeMemory, PseudoModeState


def patch_mamba_for_cpu():
    """
    Wrapper for mamba_compat.patch_mamba_for_cpu().
    
    Safe to call even if mamba_ssm is not installed.
    """
    from .mamba_compat import patch_mamba_for_cpu as _patch
    _patch()


__all__ = [
    # Environment
    "make_env",
    # WandB
    "init_wandb",
    "log_metrics", 
    "finish_wandb",
    # Lazy accessors
    "get_actor_critic",
    "get_pseudo_mode_memory",
    "patch_mamba_for_cpu",
]
