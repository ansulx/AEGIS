"""Tests for strict research pipeline policy and outputs."""

from pathlib import Path

import numpy as np
import pytest
import torch

from aegis.research_pipeline import PipelineConfig, _resolve_device, compute_segmentation_metrics


def test_pipeline_config_validation() -> None:
    cfg = PipelineConfig(mc_samples=5, patch_size=(32, 32, 32), img_size=(32, 32, 32))
    assert cfg.mc_samples == 5
    assert cfg.patch_size == (32, 32, 32)

    with pytest.raises(ValueError):
        PipelineConfig(adam_train_fraction=0.3)

    with pytest.raises(ValueError):
        PipelineConfig(mc_samples=1)

    with pytest.raises(ValueError):
        PipelineConfig(patch_size=(31, 32, 32))


def test_compute_segmentation_metrics_perfect() -> None:
    pred = np.zeros((10, 10, 10), dtype=np.uint8)
    pred[3:7, 3:7, 3:7] = 1
    gt = pred.copy()
    metrics = compute_segmentation_metrics(pred, gt)
    assert metrics["dice"] > 0.99
    assert metrics["iou"] > 0.99
    assert metrics["sensitivity"] > 0.99
    assert metrics["specificity"] > 0.99
    assert metrics["precision"] > 0.99


def test_compute_segmentation_metrics_empty() -> None:
    pred = np.zeros((10, 10, 10), dtype=np.uint8)
    gt = np.ones((10, 10, 10), dtype=np.uint8)
    metrics = compute_segmentation_metrics(pred, gt)
    assert metrics["dice"] < 0.01
    assert metrics["sensitivity"] < 0.01


def test_require_gpu_raises_when_cuda_unavailable() -> None:
    if torch.cuda.is_available():
        pytest.skip("CUDA is available in this environment; skipping negative test.")
    with pytest.raises(RuntimeError):
        _resolve_device(require_gpu=True)


def test_resolve_device_cpu_fallback() -> None:
    device = _resolve_device(require_gpu=False)
    assert device.type in ("cpu", "cuda")
