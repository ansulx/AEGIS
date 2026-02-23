"""
Strict research pipeline:
- Train on ADAM dataset only
- Run inference on ADAM and Indian datasets
- Estimate trustworthiness via Monte Carlo dropout
- Generate publication figures and failure-detection analysis

This module keeps metric definitions fixed and validates configuration bounds.
"""

from __future__ import annotations

import csv
import json
import logging
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

import nibabel as nib
import numpy as np
import torch

from .checkpoint import load_checkpoint
from .data_loading import DatasetBundle, DatasetLoadError, SessionPair, load_dataset_bundle
from .failure_detection import run_full_failure_analysis
from .mc_inference import MCResult, mc_dropout_inference
from .models.swin_unetr import AegisSwinUNETR
from .training import TrainingConfig, load_and_normalize_nifti, train_model
from .visualization import generate_all_figures

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PipelineConfig:
    """Validated configuration for the strict ADAM/Indian pipeline."""

    adam_train_fraction: float = 0.8
    min_adam_train_cases: int = 5
    seed: int = 2026
    mc_samples: int = 20
    save_nifti_outputs: bool = True
    save_qualitative_npz: bool = True
    require_gpu: bool = False

    num_epochs: int = 100
    batch_size: int = 1
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    patch_size: tuple[int, int, int] = (96, 96, 96)
    samples_per_volume: int = 4
    val_interval: int = 5
    early_stopping_patience: int = 15
    gradient_clip_max_norm: float = 1.0

    img_size: tuple[int, int, int] = (96, 96, 96)
    feature_size: int = 48
    drop_rate: float = 0.1
    attn_drop_rate: float = 0.1
    dropout_path_rate: float = 0.1

    failure_dice_threshold: float = 0.5
    generate_figures: bool = True
    run_failure_analysis: bool = True

    def __post_init__(self) -> None:
        if not 0.5 <= self.adam_train_fraction < 1.0:
            raise ValueError("adam_train_fraction must be in [0.5, 1.0).")
        if self.min_adam_train_cases < 1:
            raise ValueError("min_adam_train_cases must be >= 1.")
        if self.mc_samples < 2:
            raise ValueError("mc_samples must be >= 2.")
        if not isinstance(self.require_gpu, bool):
            raise ValueError("require_gpu must be a boolean.")
        if self.num_epochs < 1:
            raise ValueError("num_epochs must be >= 1.")
        if self.batch_size < 1:
            raise ValueError("batch_size must be >= 1.")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive.")
        if self.feature_size < 1:
            raise ValueError("feature_size must be >= 1.")
        for i, dim in enumerate(self.patch_size):
            if dim % 32 != 0:
                raise ValueError(f"patch_size[{i}] must be a multiple of 32, got {dim}.")


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _resolve_device(require_gpu: bool) -> torch.device:
    if require_gpu and not torch.cuda.is_available():
        raise RuntimeError(
            "GPU is required but CUDA is not available. "
            "Use a CUDA-enabled environment (e.g., RunPod GPU worker)."
        )
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _load_nifti(path: Path) -> tuple[np.ndarray, np.ndarray]:
    image = nib.load(str(path))
    data = np.asarray(image.get_fdata(dtype=np.float32), dtype=np.float32)
    if data.ndim != 3:
        raise DatasetLoadError(f"Expected 3D volume for '{path}', got shape {data.shape}.")
    return data, np.asarray(image.affine, dtype=np.float32)


def _load_binary_mask(path: Path) -> np.ndarray:
    mask, _ = _load_nifti(path)
    return (mask > 0.5).astype(np.uint8)


def compute_segmentation_metrics(pred_mask: np.ndarray, gt_mask: np.ndarray) -> dict[str, float]:
    """Fixed metric definitions for quantitative reporting."""
    pred = pred_mask.astype(bool)
    gt = gt_mask.astype(bool)
    tp = int(np.logical_and(pred, gt).sum())
    fp = int(np.logical_and(pred, np.logical_not(gt)).sum())
    tn = int(np.logical_and(np.logical_not(pred), np.logical_not(gt)).sum())
    fn = int(np.logical_and(np.logical_not(pred), gt).sum())

    eps = 1e-8
    dice = (2.0 * tp) / (2.0 * tp + fp + fn + eps)
    iou = tp / (tp + fp + fn + eps)
    sensitivity = tp / (tp + fn + eps)
    specificity = tn / (tn + fp + eps)
    precision = tp / (tp + fp + eps)

    return {
        "dice": float(dice),
        "iou": float(iou),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "precision": float(precision),
    }


def _split_adam_sessions(
    sessions: list[SessionPair],
    train_fraction: float,
    min_train_cases: int,
    seed: int,
) -> tuple[list[SessionPair], list[SessionPair]]:
    if len(sessions) < (min_train_cases + 1):
        raise DatasetLoadError(
            "ADAM dataset is too small for strict split: "
            f"{len(sessions)} found, need at least {min_train_cases + 1}."
        )
    ordered = sorted(sessions, key=lambda item: item.session_id)
    shuffled = ordered.copy()
    random.Random(seed).shuffle(shuffled)
    train_count = max(min_train_cases, int(round(train_fraction * len(shuffled))))
    train_count = min(train_count, len(shuffled) - 1)
    return shuffled[:train_count], shuffled[train_count:]


def _save_nifti(data: np.ndarray, affine: np.ndarray, target_path: Path) -> None:
    image = nib.Nifti1Image(data.astype(np.float32), affine)
    nib.save(image, str(target_path))


def _save_qualitative_npz(
    image: np.ndarray,
    pred_mask: np.ndarray,
    uncertainty: np.ndarray,
    out_path: Path,
    gt_mask: Optional[np.ndarray] = None,
) -> int:
    per_slice_uncertainty = uncertainty.sum(axis=(0, 1))
    slice_idx = int(np.argmax(per_slice_uncertainty))
    payload: dict[str, Any] = {
        "slice_index": slice_idx,
        "image_slice": image[:, :, slice_idx].astype(np.float32),
        "pred_slice": pred_mask[:, :, slice_idx].astype(np.uint8),
        "uncertainty_slice": uncertainty[:, :, slice_idx].astype(np.float32),
    }
    if gt_mask is not None:
        payload["gt_slice"] = gt_mask[:, :, slice_idx].astype(np.uint8)
    np.savez_compressed(out_path, **payload)
    return slice_idx


def _run_mc_inference_on_session(
    model: AegisSwinUNETR,
    session: SessionPair,
    cohort_name: str,
    config: PipelineConfig,
    device: torch.device,
    outputs_dir: Path,
    rows: list[dict[str, Any]],
    has_gt: bool = True,
) -> None:
    image_np, affine = _load_nifti(session.image_path)
    volume_tensor, _ = load_and_normalize_nifti(session.image_path, device)
    volume_3d = volume_tensor.squeeze(0)

    result: MCResult = mc_dropout_inference(
        model=model,
        volume_tensor=volume_3d,
        mc_samples=config.mc_samples,
        device=device,
        patch_size=tuple(config.patch_size),
    )

    gt_mask = None
    metrics: dict[str, Any] = {}
    if has_gt:
        gt_mask = _load_binary_mask(session.mask_path)
        if gt_mask.shape != result.pred_mask.shape:
            raise DatasetLoadError(
                f"Shape mismatch for {cohort_name} session '{session.session_id}': "
                f"mask {gt_mask.shape} vs pred {result.pred_mask.shape}."
            )
        metrics = compute_segmentation_metrics(result.pred_mask, gt_mask)

    predictions_dir = _ensure_dir(outputs_dir / "predictions" / cohort_name)
    uncertainty_dir = _ensure_dir(outputs_dir / "uncertainty" / cohort_name)
    qualitative_dir = _ensure_dir(outputs_dir / "qualitative" / cohort_name)

    if config.save_nifti_outputs:
        _save_nifti(result.pred_mask, affine, predictions_dir / f"{session.session_id}_pred.nii.gz")
        _save_nifti(result.entropy, affine, uncertainty_dir / f"{session.session_id}_uncertainty.nii.gz")

    slice_idx = None
    if config.save_qualitative_npz:
        slice_idx = _save_qualitative_npz(
            image=image_np,
            pred_mask=result.pred_mask,
            uncertainty=result.entropy,
            gt_mask=gt_mask,
            out_path=qualitative_dir / f"{session.session_id}_qualitative.npz",
        )

    row: dict[str, Any] = {
        "cohort": cohort_name,
        "session_id": session.session_id,
        "has_ground_truth": has_gt,
        "dice": metrics.get("dice"),
        "iou": metrics.get("iou"),
        "sensitivity": metrics.get("sensitivity"),
        "specificity": metrics.get("specificity"),
        "precision": metrics.get("precision"),
        "trust_score": result.trust_score,
        "volume_fraction_std": result.volume_fraction_std,
        "qualitative_slice_index": slice_idx,
    }
    rows.append(row)


def _run_mc_inference_on_image_only(
    model: AegisSwinUNETR,
    image_path: Path,
    session_id: str,
    cohort_name: str,
    config: PipelineConfig,
    device: torch.device,
    outputs_dir: Path,
    rows: list[dict[str, Any]],
) -> None:
    image_np, affine = _load_nifti(image_path)
    volume_tensor, _ = load_and_normalize_nifti(image_path, device)
    volume_3d = volume_tensor.squeeze(0)

    result: MCResult = mc_dropout_inference(
        model=model,
        volume_tensor=volume_3d,
        mc_samples=config.mc_samples,
        device=device,
        patch_size=tuple(config.patch_size),
    )

    predictions_dir = _ensure_dir(outputs_dir / "predictions" / cohort_name)
    uncertainty_dir = _ensure_dir(outputs_dir / "uncertainty" / cohort_name)
    qualitative_dir = _ensure_dir(outputs_dir / "qualitative" / cohort_name)

    if config.save_nifti_outputs:
        _save_nifti(result.pred_mask, affine, predictions_dir / f"{session_id}_pred.nii.gz")
        _save_nifti(result.entropy, affine, uncertainty_dir / f"{session_id}_uncertainty.nii.gz")

    slice_idx = None
    if config.save_qualitative_npz:
        slice_idx = _save_qualitative_npz(
            image=image_np,
            pred_mask=result.pred_mask,
            uncertainty=result.entropy,
            out_path=qualitative_dir / f"{session_id}_qualitative.npz",
        )

    rows.append({
        "cohort": cohort_name,
        "session_id": session_id,
        "has_ground_truth": False,
        "dice": None, "iou": None, "sensitivity": None,
        "specificity": None, "precision": None,
        "trust_score": result.trust_score,
        "volume_fraction_std": result.volume_fraction_std,
        "qualitative_slice_index": slice_idx,
    })


def _aggregate_metric_rows(rows: list[dict[str, Any]], cohort: str) -> dict[str, Any]:
    cohort_rows = [row for row in rows if row["cohort"] == cohort and row["has_ground_truth"]]
    if not cohort_rows:
        return {"cohort": cohort, "cases_with_gt": 0}
    metric_names = ("dice", "iou", "sensitivity", "specificity", "precision", "trust_score")
    summary: dict[str, Any] = {"cohort": cohort, "cases_with_gt": len(cohort_rows)}
    for metric in metric_names:
        values = [float(row[metric]) for row in cohort_rows if row.get(metric) is not None]
        if values:
            summary[f"{metric}_mean"] = float(np.mean(values))
            summary[f"{metric}_std"] = float(np.std(values))
    return summary


def _write_case_rows_csv(rows: list[dict[str, Any]], path: Path) -> None:
    fieldnames = [
        "cohort", "session_id", "has_ground_truth",
        "dice", "iou", "sensitivity", "specificity", "precision",
        "trust_score", "volume_fraction_std", "qualitative_slice_index",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_research_pipeline(
    output_dir: Path | str,
    base_dir: Optional[Path | str] = None,
    config: Optional[PipelineConfig] = None,
    resume_from: Optional[Path | str] = None,
) -> dict[str, Any]:
    """Execute strict ADAM-train/Indian-inference pipeline with real SwinUNETR model."""
    cfg = config or PipelineConfig()
    out = _ensure_dir(Path(output_dir).expanduser().resolve())
    reports_dir = _ensure_dir(out / "reports")
    checkpoint_dir = _ensure_dir(out / "checkpoints")
    device = _resolve_device(cfg.require_gpu)

    torch.manual_seed(cfg.seed)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(cfg.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    logger.info("Device: %s", device)
    if device.type == "cuda":
        logger.info("GPU: %s", torch.cuda.get_device_name(0))

    bundle: DatasetBundle = load_dataset_bundle(base_dir=base_dir)
    adam_train, adam_holdout = _split_adam_sessions(
        sessions=bundle.adam_sessions,
        train_fraction=cfg.adam_train_fraction,
        min_train_cases=cfg.min_adam_train_cases,
        seed=cfg.seed,
    )
    logger.info(
        "Split: %d ADAM train, %d ADAM holdout, %d Indian CAR, %d Indian NCAR",
        len(adam_train), len(adam_holdout),
        len(bundle.indian_car_sessions), len(bundle.indian_ncar_images),
    )

    model = AegisSwinUNETR(
        img_size=cfg.img_size,
        feature_size=cfg.feature_size,
        drop_rate=cfg.drop_rate,
        attn_drop_rate=cfg.attn_drop_rate,
        dropout_path_rate=cfg.dropout_path_rate,
    ).to(device)

    training_cfg = TrainingConfig(
        num_epochs=cfg.num_epochs,
        batch_size=cfg.batch_size,
        learning_rate=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        patch_size=cfg.patch_size,
        samples_per_volume=cfg.samples_per_volume,
        val_interval=cfg.val_interval,
        early_stopping_patience=cfg.early_stopping_patience,
        gradient_clip_max_norm=cfg.gradient_clip_max_norm,
    )

    resume_path = Path(resume_from) if resume_from else None
    best_ckpt = train_model(
        model=model,
        train_sessions=adam_train,
        val_sessions=adam_holdout,
        config=training_cfg,
        device=device,
        checkpoint_dir=checkpoint_dir,
        resume_from=resume_path,
    )
    logger.info("Training complete. Best checkpoint: %s", best_ckpt)

    load_checkpoint(best_ckpt, model, device=device)
    model.to(device)
    model.eval()

    case_rows: list[dict[str, Any]] = []

    logger.info("Running MC inference on ADAM train (%d sessions)...", len(adam_train))
    for session in sorted(adam_train, key=lambda s: s.session_id):
        _run_mc_inference_on_session(model, session, "adam_train", cfg, device, out, case_rows)

    logger.info("Running MC inference on ADAM holdout (%d sessions)...", len(adam_holdout))
    for session in sorted(adam_holdout, key=lambda s: s.session_id):
        _run_mc_inference_on_session(model, session, "adam_holdout", cfg, device, out, case_rows)

    logger.info("Running MC inference on Indian CAR (%d sessions)...", len(bundle.indian_car_sessions))
    for session in sorted(bundle.indian_car_sessions, key=lambda s: s.session_id):
        _run_mc_inference_on_session(
            model, session, "indian_car_inference_only", cfg, device, out, case_rows
        )

    logger.info("Running MC inference on Indian NCAR (%d images)...", len(bundle.indian_ncar_images))
    for image_path in sorted(bundle.indian_ncar_images):
        session_id = image_path.stem.replace(".nii", "")
        _run_mc_inference_on_image_only(
            model, image_path, session_id,
            "indian_ncar_inference_only", cfg, device, out, case_rows,
        )

    per_case_csv = reports_dir / "per_case_metrics.csv"
    _write_case_rows_csv(case_rows, per_case_csv)

    cohort_summaries = [
        _aggregate_metric_rows(case_rows, "adam_train"),
        _aggregate_metric_rows(case_rows, "adam_holdout"),
        _aggregate_metric_rows(case_rows, "indian_car_inference_only"),
        _aggregate_metric_rows(case_rows, "indian_ncar_inference_only"),
    ]

    summary: dict[str, Any] = {
        "policy": {
            "adam_training_used": True,
            "indian_training_used": False,
            "indian_inference_only": True,
        },
        "dataset_counts": {
            "adam_total": len(bundle.adam_sessions),
            "adam_train": len(adam_train),
            "adam_holdout": len(adam_holdout),
            "indian_car": len(bundle.indian_car_sessions),
            "indian_ncar": len(bundle.indian_ncar_images),
        },
        "model": {
            "type": "SwinUNETR_v2_MC_Dropout",
            "feature_size": cfg.feature_size,
            "mc_samples": cfg.mc_samples,
            "best_checkpoint": str(best_ckpt),
            "device": str(device),
        },
        "cohort_summaries": cohort_summaries,
        "config": asdict(cfg),
        "artifacts": {
            "per_case_csv": str(per_case_csv),
            "predictions_dir": str(out / "predictions"),
            "uncertainty_dir": str(out / "uncertainty"),
            "qualitative_dir": str(out / "qualitative"),
            "checkpoints_dir": str(checkpoint_dir),
        },
    }

    if cfg.generate_figures:
        try:
            figure_paths = generate_all_figures(out)
            summary["artifacts"]["figures"] = [str(p) for p in figure_paths]
            logger.info("Generated %d publication figures.", len(figure_paths))
        except Exception:
            logger.warning("Figure generation failed.", exc_info=True)

    if cfg.run_failure_analysis:
        try:
            fa_result = run_full_failure_analysis(
                per_case_csv, out, failure_dice_threshold=cfg.failure_dice_threshold
            )
            summary["failure_analysis"] = {
                "auroc_trust_vs_failure": fa_result.auroc_trust_vs_failure,
                "optimal_trust_threshold": fa_result.optimal_trust_threshold,
                "workload_reduction_fraction": fa_result.workload_reduction_fraction,
                "missed_failure_rate": fa_result.missed_failure_rate,
            }
            logger.info("Failure analysis complete. AUROC=%.4f", fa_result.auroc_trust_vs_failure)
        except Exception:
            logger.warning("Failure analysis failed.", exc_info=True)

    summary_path = reports_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    logger.info("Pipeline complete. Summary: %s", summary_path)
    return summary
