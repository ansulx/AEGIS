"""Checkpoint save/load utilities with atomic writes and backward compatibility."""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.optim as optim

logger = logging.getLogger(__name__)

_CHECKPOINT_PATTERN = re.compile(r"checkpoint_epoch_(\d+)\.pt$")


def save_checkpoint(
    path: Path | str,
    model: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: Any,
    epoch: int,
    best_dice: float,
    config_dict: dict[str, Any],
) -> None:
    """Save training state atomically (write to .tmp, then rename)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "epoch": epoch,
        "best_dice": best_dice,
        "config": config_dict,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
    }

    tmp_path = path.with_suffix(".tmp")
    try:
        torch.save(payload, tmp_path, _use_new_zipfile_serialization=False)
        os.replace(tmp_path, path)
    except BaseException:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
        raise


def load_checkpoint(
    path: Path | str,
    model: nn.Module,
    optimizer: Optional[optim.Optimizer] = None,
    scheduler: Any = None,
    device: str | torch.device = "cpu",
) -> dict[str, Any]:
    """Load a checkpoint and restore model/optimizer/scheduler state.

    Returns a metadata dict with keys: epoch, best_dice, config.
    Missing keys default to safe values for backward compatibility.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    checkpoint = torch.load(path, map_location=device, weights_only=False)

    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if scheduler is not None and checkpoint.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    return {
        "epoch": checkpoint.get("epoch", 0),
        "best_dice": checkpoint.get("best_dice", 0.0),
        "config": checkpoint.get("config", {}),
    }


def find_latest_checkpoint(checkpoint_dir: Path | str) -> Optional[Path]:
    """Return the checkpoint with the highest epoch number, or None."""
    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.is_dir():
        return None

    best_epoch = -1
    best_path: Optional[Path] = None

    for entry in checkpoint_dir.iterdir():
        match = _CHECKPOINT_PATTERN.search(entry.name)
        if match:
            epoch = int(match.group(1))
            if epoch > best_epoch:
                best_epoch = epoch
                best_path = entry

    return best_path
