"""SwinUNETR wrapper with MC Dropout support for 3D aneurysm segmentation."""

from __future__ import annotations

import inspect

import torch
import torch.nn as nn
from monai.networks.nets import SwinUNETR


class AegisSwinUNETR(nn.Module):
    """MONAI SwinUNETR wrapped with MC Dropout activation for uncertainty estimation.

    Outputs raw logits (no sigmoid). Call enable_mc_dropout() before MC sampling
    to keep dropout active during inference while leaving BatchNorm in eval mode.
    """

    def __init__(
        self,
        img_size: tuple[int, int, int] = (96, 96, 96),
        in_channels: int = 1,
        out_channels: int = 1,
        feature_size: int = 48,
        drop_rate: float = 0.1,
        attn_drop_rate: float = 0.1,
        dropout_path_rate: float = 0.1,
        spatial_dims: int = 3,
    ) -> None:
        super().__init__()
        if in_channels < 1:
            raise ValueError(f"in_channels must be >= 1, got {in_channels}")
        if out_channels < 1:
            raise ValueError(f"out_channels must be >= 1, got {out_channels}")
        if spatial_dims != 3:
            raise ValueError(f"spatial_dims must be 3 for 3D volumes, got {spatial_dims}")
        if feature_size < 1:
            raise ValueError(f"feature_size must be >= 1, got {feature_size}")
        for i, s in enumerate(img_size):
            if s < 1:
                raise ValueError(f"img_size[{i}] must be >= 1, got {s}")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.img_size = img_size

        sig = inspect.signature(SwinUNETR.__init__)
        params = set(sig.parameters.keys())

        kwargs: dict = {
            "in_channels": in_channels,
            "out_channels": out_channels,
            "feature_size": feature_size,
            "drop_rate": drop_rate,
            "attn_drop_rate": attn_drop_rate,
            "dropout_path_rate": dropout_path_rate,
            "spatial_dims": spatial_dims,
        }
        if "img_size" in params:
            kwargs["img_size"] = img_size
        if "use_v2" in params:
            kwargs["use_v2"] = True

        self.net = SwinUNETR(**kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 5:
            raise ValueError(
                f"Input must be 5D (B, C, D, H, W), got {x.ndim}D with shape {x.shape}"
            )
        if x.shape[1] != self.in_channels:
            raise ValueError(
                f"Expected {self.in_channels} input channel(s), got {x.shape[1]}"
            )
        return self.net(x)

    def enable_mc_dropout(self) -> None:
        """Activate dropout layers for MC sampling while keeping BatchNorm in eval."""
        self.eval()
        for module in self.modules():
            if isinstance(module, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
                module.train()

    def disable_mc_dropout(self) -> None:
        """Restore standard eval mode for all layers."""
        self.eval()
