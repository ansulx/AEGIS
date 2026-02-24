"""SwinUNETR wrapper with MC Dropout support for 3D aneurysm segmentation."""

from __future__ import annotations

import inspect
import logging
from pathlib import Path

import torch
import torch.nn as nn
from monai.networks.nets import SwinUNETR

logger = logging.getLogger(__name__)

PRETRAINED_URL = (
    "https://github.com/Project-MONAI/MONAI-extra-test-data/releases/"
    "download/0.8.1/model_swinvit.pt"
)


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
        use_pretrained: bool = False,
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

        if use_pretrained:
            self._load_pretrained_backbone()

    def _load_pretrained_backbone(self) -> None:
        """Load self-supervised pre-trained ViT backbone weights from MONAI."""
        cache_dir = Path.home() / ".cache" / "aegis"
        cache_dir.mkdir(parents=True, exist_ok=True)
        weight_path = cache_dir / "model_swinvit.pt"

        if not weight_path.exists():
            logger.info("Downloading pre-trained SwinViT weights...")
            torch.hub.download_url_to_file(PRETRAINED_URL, str(weight_path))
            logger.info("Downloaded to %s", weight_path)

        pretrained = torch.load(str(weight_path), map_location="cpu", weights_only=False)

        if isinstance(pretrained, dict):
            for nested_key in ("state_dict", "model", "model_state_dict"):
                if nested_key in pretrained:
                    pretrained = pretrained[nested_key]
                    break

        model_dict = self.net.state_dict()
        loaded = 0
        skipped = 0
        for key, value in pretrained.items():
            for mapped_key in _candidate_keys(key):
                if mapped_key in model_dict and model_dict[mapped_key].shape == value.shape:
                    model_dict[mapped_key] = value
                    loaded += 1
                    break
            else:
                skipped += 1

        self.net.load_state_dict(model_dict)
        logger.info(
            "Loaded %d/%d pre-trained backbone parameters (skipped %d)",
            loaded, loaded + skipped, skipped,
        )

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


def _candidate_keys(key: str) -> list[str]:
    """Generate candidate mapped keys for a pre-trained checkpoint key."""
    base = key
    if base.startswith("module."):
        base = base[len("module."):]

    candidates = [
        base,
        "swinViT." + base,
    ]
    if base.startswith("swinViT."):
        candidates.append(base[len("swinViT."):])

    return candidates
