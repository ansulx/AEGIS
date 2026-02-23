"""Monte Carlo Dropout inference engine for uncertainty-aware 3D segmentation."""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from monai.inferers import sliding_window_inference

logger = logging.getLogger(__name__)


@dataclass
class MCResult:
    """Aggregated result from MC Dropout inference."""

    pred_mask: np.ndarray
    mean_prob: np.ndarray
    entropy: np.ndarray
    trust_score: float
    volume_fraction_std: float


def mc_dropout_inference(
    model: nn.Module,
    volume_tensor: torch.Tensor,
    mc_samples: int = 20,
    device: torch.device | str = "cpu",
    patch_size: tuple[int, int, int] = (96, 96, 96),
    sw_batch_size: int = 2,
    overlap: float = 0.25,
) -> MCResult:
    """Run MC Dropout inference on a single 3D volume using sliding window.

    Args:
        model: Network with ``enable_mc_dropout`` / ``disable_mc_dropout`` methods.
        volume_tensor: Float32 tensor of shape (D, H, W) — already normalized.
        mc_samples: Number of stochastic forward passes (>= 2).
        device: Target compute device.
        patch_size: Sliding window patch size (must match training patch size).
        sw_batch_size: Number of patches processed in parallel during sliding window.
        overlap: Fraction of overlap between adjacent sliding window patches.

    Returns:
        MCResult with binary mask, mean probability, entropy, trust score,
        and volume-fraction standard deviation.
    """
    if mc_samples < 2:
        raise ValueError(f"mc_samples must be >= 2, got {mc_samples}")
    if volume_tensor.ndim != 3:
        raise ValueError(
            f"volume_tensor must be 3D (D, H, W), got {volume_tensor.ndim}D "
            f"with shape {volume_tensor.shape}"
        )

    device = torch.device(device) if isinstance(device, str) else device
    x = volume_tensor.unsqueeze(0).unsqueeze(0).to(device=device, dtype=torch.float32)

    model.to(device)
    model.enable_mc_dropout()

    prob_sum = torch.zeros(1, *volume_tensor.shape, device=device, dtype=torch.float32)
    volume_fractions: list[float] = []
    completed = 0
    use_amp = device.type == "cuda"

    try:
        with torch.no_grad():
            for _ in range(mc_samples):
                try:
                    def _predictor(patch: torch.Tensor) -> torch.Tensor:
                        if use_amp:
                            with torch.amp.autocast("cuda"):
                                return torch.sigmoid(model(patch))
                        return torch.sigmoid(model(patch))

                    probs = sliding_window_inference(
                        inputs=x,
                        roi_size=patch_size,
                        sw_batch_size=sw_batch_size,
                        predictor=_predictor,
                        overlap=overlap,
                        mode="gaussian",
                    )
                    probs_squeezed = probs[:, 0].float()
                    prob_sum += probs_squeezed
                    volume_fractions.append(float(probs_squeezed.mean().item()))
                    completed += 1
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    if completed >= 2:
                        warnings.warn(
                            f"CUDA OOM after {completed}/{mc_samples} MC passes; "
                            f"using partial results.",
                            RuntimeWarning,
                            stacklevel=2,
                        )
                        break
                    raise
    finally:
        model.disable_mc_dropout()

    mean_prob = prob_sum / completed
    eps = 1e-7
    p = torch.clamp(mean_prob, eps, 1.0 - eps)
    entropy = -(p * torch.log2(p) + (1.0 - p) * torch.log2(1.0 - p))
    pred_mask = (mean_prob >= 0.5).to(torch.uint8)

    trust_score = float(1.0 - entropy.mean().item())
    vf_std = float(np.std(volume_fractions, ddof=0))

    return MCResult(
        pred_mask=pred_mask.squeeze(0).cpu().numpy(),
        mean_prob=mean_prob.squeeze(0).cpu().numpy().astype(np.float32),
        entropy=entropy.squeeze(0).cpu().numpy().astype(np.float32),
        trust_score=trust_score,
        volume_fraction_std=vf_std,
    )
