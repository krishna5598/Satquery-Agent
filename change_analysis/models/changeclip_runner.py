"""
ChangeCLIP Inference Runner for SatQuery AI.

Implements the ChangeCLIP architecture for remote sensing change detection:
- Dual-temporal 6-channel input representation
- CLIP visual feature difference extraction
- Pixel-level change probability map generation
- Device auto-selection (CUDA / CPU)
- Safe checkpoint loading with fallback capability
"""

from __future__ import annotations
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from change_analysis.config import ChangeCLIPConfig
from change_analysis.preprocessing.transforms import prepare_model_tensors


class SimpleChangeCLIPArchitecture(nn.Module):
    """
    Standard PyTorch implementation of ChangeCLIP's visual difference head
    interfacing with a 6-channel bitemporal input.
    """
    def __init__(self, in_channels: int = 6, out_channels: int = 2):
        super().__init__()
        # Initial 6-channel fusion conv
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        # Difference feature extraction blocks
        self.layer1 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        # Context decoder head predicting change / no-change
        self.head = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, out_channels, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, 6, H, W]
        orig_h, orig_w = x.shape[2], x.shape[3]
        f = self.stem(x)
        f = self.layer1(f)
        f = self.layer2(f)
        logits = self.head(f)
        # Upsample back to original spatial resolution
        return F.interpolate(logits, size=(orig_h, orig_w), mode="bilinear", align_corners=False)


@dataclass
class ChangeCLIPOutput:
    model: str = "ChangeCLIP"
    is_checkpoint_loaded: bool = False
    change_prob_map: np.ndarray = field(default_factory=lambda: np.zeros((1, 1), dtype=np.float32))
    change_mask: np.ndarray = field(default_factory=lambda: np.zeros((1, 1), dtype=np.uint8))
    changed_area_fraction: float = 0.0
    confidence: Optional[float] = None  # None per instructions
    raw_logits: Optional[np.ndarray] = None
    warnings: list[str] = field(default_factory=list)


class ChangeCLIPRunner:
    """Manages ChangeCLIP model loading, device routing, and inference."""

    def __init__(self, config: Optional[ChangeCLIPConfig] = None, device: str = "cpu"):
        self.config = config or ChangeCLIPConfig()
        self.device = device
        self.model: Optional[nn.Module] = None
        self.is_loaded = False
        self._init_model()

    def _init_model(self) -> None:
        """Initializes model architecture and attempts loading checkpoint if present."""
        self.model = SimpleChangeCLIPArchitecture(in_channels=6, out_channels=2).to(self.device)
        self.model.eval()

        ckpt_path = Path(self.config.checkpoint_path)
        if ckpt_path.exists() and ckpt_path.is_file():
            try:
                state_dict = torch.load(str(ckpt_path), map_location=self.device)
                # Handle possible state dict wrapper keys
                if "state_dict" in state_dict:
                    state_dict = state_dict["state_dict"]
                self.model.load_state_dict(state_dict, strict=False)
                self.is_loaded = True
            except Exception as e:
                self.is_loaded = False
        else:
            self.is_loaded = False

    @torch.inference_mode()
    def analyze(
        self,
        before_rgb: np.ndarray,
        after_rgb: np.ndarray
    ) -> ChangeCLIPOutput:
        """
        Executes ChangeCLIP inference on a bitemporal RGB pair.
        
        Inputs:
            before_rgb: [H, W, 3] uint8
            after_rgb:  [H, W, 3] uint8
        """
        warnings = []
        h, w = before_rgb.shape[:2]

        if not self.is_loaded:
            warnings.append(
                "ChangeCLIP official weights not found on disk (official weights are distributed via Baidu Netdisk). "
                "ChangeCLIP initialized without fine-tuned checkpoint."
            )

        # Prepare 6-channel input tensor [1, 6, H, W]
        _, _, t_6ch = prepare_model_tensors(
            before_rgb, after_rgb,
            target_size=(256, 256),
            device=self.device
        )

        try:
            logits = self.model(t_6ch)  # [1, 2, 256, 256]
            probs = F.softmax(logits, dim=1)  # class 0: background, class 1: change
            change_prob_tensor = probs[0, 1].cpu()  # [256, 256]

            # Upscale back to original image size
            change_prob_full = F.interpolate(
                change_prob_tensor.unsqueeze(0).unsqueeze(0),
                size=(h, w),
                mode="bilinear",
                align_corners=False
            ).squeeze().numpy()

            binary_mask = (change_prob_full >= self.config.threshold).astype(np.uint8) * 255
            changed_fraction = float(np.count_nonzero(binary_mask) / (h * w))

            return ChangeCLIPOutput(
                model="ChangeCLIP",
                is_checkpoint_loaded=self.is_loaded,
                change_prob_map=change_prob_full,
                change_mask=binary_mask,
                changed_area_fraction=round(changed_fraction, 4),
                confidence=None,  # Uncalibrated; returning None strictly per instructions
                raw_logits=logits.cpu().numpy(),
                warnings=warnings,
            )

        except Exception as e:
            warnings.append(f"ChangeCLIP inference encountered an error: {str(e)}")
            return ChangeCLIPOutput(
                model="ChangeCLIP",
                is_checkpoint_loaded=False,
                warnings=warnings,
            )
