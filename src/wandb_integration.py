#!/usr/bin/env python3
"""
wandb_integration.py

Thin wrapper around Weights & Biases (wandb) logging.

Goals:
    - If args.track is False → do nothing (no dependency on wandb).
    - If wandb is not installed → print a warning once, then no-op.
    - If enabled → standard init/log/finish functions.

Exposes:
    - init_wandb(args) -> run or None
    - log_metrics(step, metrics, run=None)
    - finish_wandb(run)
"""

from __future__ import annotations

from typing import Any, Dict, Optional

_wandb_available = False
_wandb_import_error_logged = False

try:
    import wandb  # type: ignore

    _wandb_available = True
except Exception:
    _wandb_available = False
    wandb = None  # type: ignore


def init_wandb(args: Any) -> Optional[Any]:
    """
    Initialize wandb Run if tracking is enabled.

    Returns:
        wandb.Run object or None.
    """
    global _wandb_import_error_logged

    # If tracking not requested, do nothing.
    if not getattr(args, "track", False):
        return None

    if not _wandb_available:
        if not _wandb_import_error_logged:
            print("[wandb_integration] WARNING: wandb not installed; disabling tracking.")
            _wandb_import_error_logged = True
        return None

    project = getattr(args, "wandb_project", "neural-memory-suite")
    entity = getattr(args, "wandb_entity", None)
    run_name = getattr(args, "run_name", None)

    config = {k: v for k, v in vars(args).items() if not k.startswith("_")}

    run = wandb.init(
        project=project,
        entity=entity,
        name=run_name,
        config=config,
    )
    return run


def log_metrics(step: int, metrics: Dict[str, float], run: Optional[Any] = None) -> None:
    """
    Log metrics to wandb if a run is active.
    """
    if run is None:
        return
    if not _wandb_available:
        return

    try:
        run.log(metrics, step=step)
    except Exception as e:
        print(f"[wandb_integration] WARNING: failed to log metrics: {e}")


def finish_wandb(run: Optional[Any]) -> None:
    """
    Finish wandb run if active.
    """
    if run is None:
        return
    if not _wandb_available:
        return

    try:
        run.finish()
    except Exception as e:
        print(f"[wandb_integration] WARNING: failed to finish wandb run: {e}")
