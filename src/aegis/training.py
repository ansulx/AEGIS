"""Training loop for 3D cerebrovascular aneurysm segmentation on ADAM sessions."""

from __future__ import annotations

import dataclasses
import logging
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import nibabel as nib
import numpy as np
import torch
import torch.nn as nn
from monai.data import CacheDataset, Dataset, DataLoader
from monai.losses import DiceFocalLoss
from monai.metrics import DiceMetric
from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    RandCropByPosNegLabeld,
    RandFlipd,
    RandGaussianNoised,
    RandGaussianSmoothd,
    RandRotate90d,
    RandScaleIntensityd,
    RandShiftIntensityd,
    SpatialPadd,
    ToTensord,
)

from aegis.checkpoint import find_latest_checkpoint, load_checkpoint, save_checkpoint
from aegis.data_loading import SessionPair

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TrainingConfig:
    """Immutable training hyper-parameters with validation."""

    num_epochs: int = 200
    batch_size: int = 2
    learning_rate: float = 2e-4
    weight_decay: float = 1e-5
    patch_size: tuple[int, int, int] = (96, 96, 96)
    samples_per_volume: int = 4
    val_interval: int = 5
    early_stopping_patience: int = 30
    gradient_clip_max_norm: float = 1.0
    warmup_epochs: int = 10

    def __post_init__(self) -> None:
        if self.num_epochs <= 0:
            raise ValueError(f"num_epochs must be positive, got {self.num_epochs}")
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}")
        if self.learning_rate <= 0:
            raise ValueError(f"learning_rate must be positive, got {self.learning_rate}")
        if self.weight_decay <= 0:
            raise ValueError(f"weight_decay must be positive, got {self.weight_decay}")
        if self.samples_per_volume <= 0:
            raise ValueError(f"samples_per_volume must be positive, got {self.samples_per_volume}")
        if self.val_interval <= 0:
            raise ValueError(f"val_interval must be positive, got {self.val_interval}")
        if self.early_stopping_patience <= 0:
            raise ValueError(
                f"early_stopping_patience must be positive, got {self.early_stopping_patience}"
            )
        if self.gradient_clip_max_norm <= 0:
            raise ValueError(
                f"gradient_clip_max_norm must be positive, got {self.gradient_clip_max_norm}"
            )
        for i, dim in enumerate(self.patch_size):
            if dim <= 0:
                raise ValueError(f"patch_size[{i}] must be positive, got {dim}")
            if dim % 32 != 0:
                raise ValueError(
                    f"patch_size[{i}] must be a multiple of 32, got {dim}"
                )


def load_and_normalize_nifti(
    path: Path, device: torch.device
) -> tuple[torch.Tensor, np.ndarray]:
    """Load a NIfTI volume, percentile-clip (1-99), z-score normalize.

    Returns (tensor on *device* with shape (1, D, H, W), affine).
    """
    img = nib.load(path)
    affine: np.ndarray = np.array(img.affine, dtype=np.float64)
    data = np.asarray(img.dataobj, dtype=np.float32)

    if not np.all(np.isfinite(data)):
        logger.warning("Non-finite values in %s — replacing with 0", path.name)
        data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)

    p1, p99 = float(np.percentile(data, 1)), float(np.percentile(data, 99))
    data = np.clip(data, p1, p99)

    std = float(data.std())
    if std < 1e-6:
        logger.warning("Near-constant volume %s (std=%.2e) — returning zeros", path.name, std)
        data = np.zeros_like(data)
    else:
        mean = float(data.mean())
        data = (data - mean) / std

    tensor = torch.from_numpy(data).unsqueeze(0).to(device)  # (1, D, H, W)
    return tensor, affine


def _load_session_dict(session: SessionPair) -> dict[str, np.ndarray]:
    """Load a session as raw 3D numpy arrays (D, H, W) — no channel dim.

    MONAI transforms (EnsureChannelFirstd) will add the channel later.
    """
    img = nib.load(session.image_path)
    data = np.asarray(img.dataobj, dtype=np.float32)
    if not np.all(np.isfinite(data)):
        data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
    p1, p99 = float(np.percentile(data, 1)), float(np.percentile(data, 99))
    data = np.clip(data, p1, p99)
    std = float(data.std())
    if std < 1e-6:
        data = np.zeros_like(data)
    else:
        data = (data - float(data.mean())) / std

    mask_img = nib.load(session.mask_path)
    mask = np.asarray(mask_img.dataobj, dtype=np.float32)
    mask = (mask > 0.5).astype(np.float32)

    return {"image": data, "label": mask}


def _parallel_load_sessions(sessions: list[SessionPair]) -> list[dict[str, np.ndarray]]:
    """Load all sessions in parallel using a thread pool."""
    from concurrent.futures import ThreadPoolExecutor, as_completed

    data_dicts: list[dict[str, np.ndarray]] = [{}] * len(sessions)
    n_workers = min(8, len(sessions))

    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        future_to_idx = {
            pool.submit(_load_session_dict, session): i
            for i, session in enumerate(sessions)
        }
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            data_dicts[idx] = future.result()

    logger.info("Loaded %d sessions using %d threads.", len(sessions), n_workers)
    return data_dicts


def create_patch_dataset(
    sessions: list[SessionPair],
    patch_size: tuple[int, int, int],
    samples_per_volume: int,
    is_train: bool,
) -> CacheDataset | Dataset:
    """Build a MONAI dataset of patches from session pairs.

    Training uses a regular Dataset so random augmentations are freshly
    sampled every epoch (CacheDataset would freeze them after init).
    Validation uses CacheDataset since transforms are deterministic.
    """
    if not sessions:
        raise ValueError("sessions list must not be empty")

    data_dicts = _parallel_load_sessions(sessions)

    if is_train:
        n_before = len(data_dicts)
        positive_dicts = [d for d in data_dicts if d["label"].sum() > 0]
        n_empty = n_before - len(positive_dicts)
        if positive_dicts:
            logger.info(
                "Training filter: %d/%d sessions have foreground; "
                "dropping %d empty-mask sessions.",
                len(positive_dicts), n_before, n_empty,
            )
            data_dicts = positive_dicts
        else:
            logger.warning(
                "All %d training masks are empty! Keeping all sessions.", n_before
            )

    keys = ["image", "label"]

    if is_train:
        transforms = Compose([
            EnsureChannelFirstd(keys=keys, channel_dim="no_channel"),
            SpatialPadd(keys=keys, spatial_size=patch_size),
            RandCropByPosNegLabeld(
                keys=keys,
                label_key="label",
                spatial_size=patch_size,
                pos=5.0,
                neg=1.0,
                num_samples=samples_per_volume,
            ),
            RandFlipd(keys=keys, prob=0.5, spatial_axis=0),
            RandFlipd(keys=keys, prob=0.5, spatial_axis=1),
            RandFlipd(keys=keys, prob=0.5, spatial_axis=2),
            RandRotate90d(keys=keys, prob=0.5, max_k=3, spatial_axes=(0, 1)),
            RandScaleIntensityd(keys=["image"], factors=0.1, prob=0.5),
            RandShiftIntensityd(keys=["image"], offsets=0.1, prob=0.5),
            RandGaussianNoised(keys=["image"], prob=0.2, mean=0.0, std=0.1),
            RandGaussianSmoothd(keys=["image"], prob=0.2, sigma_x=(0.5, 1.0), sigma_y=(0.5, 1.0), sigma_z=(0.5, 1.0)),
            ToTensord(keys=keys),
        ])
        return Dataset(data=data_dicts, transform=transforms)
    else:
        transforms = Compose([
            EnsureChannelFirstd(keys=keys, channel_dim="no_channel"),
            SpatialPadd(keys=keys, spatial_size=patch_size),
            ToTensord(keys=keys),
        ])
        return CacheDataset(data=data_dicts, transform=transforms, cache_rate=1.0, num_workers=4)


def _compute_val_dice(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    use_amp: bool,
    patch_size: tuple[int, int, int] = (96, 96, 96),
) -> float:
    """Run validation with sliding window inference and return mean Dice score."""
    from monai.inferers import sliding_window_inference

    model.eval()
    dice_metric = DiceMetric(include_background=False, reduction="mean", get_not_nans=True)

    with torch.no_grad():
        for batch in val_loader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)

            if use_amp:
                with torch.amp.autocast("cuda"):
                    logits = sliding_window_inference(
                        images, roi_size=patch_size, sw_batch_size=2,
                        predictor=model, overlap=0.25, mode="gaussian",
                    )
            else:
                logits = sliding_window_inference(
                    images, roi_size=patch_size, sw_batch_size=2,
                    predictor=model, overlap=0.25, mode="gaussian",
                )
            preds = (torch.sigmoid(logits) > 0.5).float()
            dice_metric(y_pred=preds, y=labels)

    result, not_nans = dice_metric.aggregate()
    dice_metric.reset()
    if not_nans.item() < 1:
        logger.warning("No valid Dice samples in validation; returning 0.0")
        return 0.0
    mean_dice = result.item()
    if not np.isfinite(mean_dice):
        return 0.0
    return mean_dice


def train_model(
    model: nn.Module,
    train_sessions: list[SessionPair],
    val_sessions: list[SessionPair],
    config: TrainingConfig,
    device: torch.device,
    checkpoint_dir: Path,
    resume_from: Optional[Path] = None,
) -> Path:
    """Full training loop with mixed precision, early stopping, and checkpointing.

    Returns the path to the best checkpoint.
    """
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    train_ds = create_patch_dataset(
        train_sessions, config.patch_size, config.samples_per_volume, is_train=True
    )
    val_ds = create_patch_dataset(
        val_sessions, config.patch_size, config.samples_per_volume, is_train=False
    )

    use_cuda = device.type == "cuda"
    train_loader = DataLoader(
        train_ds, batch_size=config.batch_size, shuffle=True,
        num_workers=4, pin_memory=use_cuda, persistent_workers=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=1, shuffle=False,
        num_workers=2, pin_memory=use_cuda, persistent_workers=True,
    )

    model = model.to(device)
    loss_fn = DiceFocalLoss(sigmoid=True, lambda_dice=1.0, lambda_focal=1.0, gamma=2.0)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=5, min_lr=1e-7,
    )

    start_epoch = 0
    best_dice = 0.0
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    if resume_from is not None:
        meta = load_checkpoint(resume_from, model, optimizer, scheduler, device=device)
        start_epoch = meta["epoch"] + 1
        best_dice = meta["best_dice"]
        logger.info(
            "Resumed from %s — epoch %d, best_dice=%.4f", resume_from.name, start_epoch, best_dice
        )

    epochs_without_improvement = 0
    best_checkpoint_path = checkpoint_dir / "best_model.pt"
    config_dict = dataclasses.asdict(config)

    warnings.filterwarnings("ignore", message=".*Num foregrounds.*unable to generate class balanced.*")
    warnings.filterwarnings("ignore", message=".*non-tuple sequence for multidimensional indexing.*")

    for epoch in range(start_epoch, config.num_epochs):
        t0 = time.monotonic()

        # --- LR warmup ---
        if epoch < config.warmup_epochs:
            warmup_lr = config.learning_rate * (epoch + 1) / config.warmup_epochs
            for pg in optimizer.param_groups:
                pg["lr"] = warmup_lr

        model.train()
        epoch_loss = 0.0
        batches_processed = 0

        for batch in train_loader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)

            try:
                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda", enabled=use_amp):
                    logits = model(images)
                    loss = loss_fn(logits, labels)

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_max_norm)
                scaler.step(optimizer)
                scaler.update()

                epoch_loss += loss.item()
                batches_processed += 1

            except torch.cuda.OutOfMemoryError:
                logger.warning(
                    "CUDA OOM at epoch %d — skipping batch, clearing cache", epoch
                )
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                continue

        elapsed = time.monotonic() - t0
        avg_loss = epoch_loss / max(batches_processed, 1)
        lr_now = optimizer.param_groups[0]["lr"]
        logger.info(
            "Epoch %03d/%03d  loss=%.4f  lr=%.2e  batches=%d  time=%.1fs",
            epoch + 1,
            config.num_epochs,
            avg_loss,
            lr_now,
            batches_processed,
            elapsed,
        )

        # --- Validation & checkpointing ---
        if (epoch + 1) % config.val_interval == 0:
            val_dice = _compute_val_dice(model, val_loader, device, use_amp, config.patch_size)
            logger.info("  val_dice=%.4f  (best=%.4f)", val_dice, best_dice)

            if epoch >= config.warmup_epochs:
                scheduler.step(val_dice)

            if val_dice > best_dice:
                best_dice = val_dice
                epochs_without_improvement = 0
                save_checkpoint(
                    best_checkpoint_path, model, optimizer, scheduler, epoch, best_dice, config_dict
                )
                logger.info("  -> new best model saved")
            else:
                epochs_without_improvement += config.val_interval

            if epochs_without_improvement >= config.early_stopping_patience:
                logger.info(
                    "Early stopping at epoch %d (no improvement for %d epochs)",
                    epoch + 1,
                    epochs_without_improvement,
                )
                break

            latest_path = checkpoint_dir / "latest_model.pt"
            save_checkpoint(latest_path, model, optimizer, scheduler, epoch, best_dice, config_dict)

    if not best_checkpoint_path.exists():
        save_checkpoint(
            best_checkpoint_path, model, optimizer, scheduler,
            config.num_epochs - 1, best_dice, config_dict,
        )
        logger.info("No validation improvement recorded; saved final model as best checkpoint")

    return best_checkpoint_path
