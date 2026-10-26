"""
Tensor Transformations for ChangeCLIP and Multimodal Change Models.
"""

from __future__ import annotations
from typing import Tuple, List, Optional
import numpy as np
import torch
import torch.nn.functional as F


# Standard OpenAI CLIP normalization constants
CLIP_DEFAULT_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_DEFAULT_STD = [0.26862954, 0.26130258, 0.27577711]


def to_tensor(
    image: np.ndarray,
    target_size: Optional[Tuple[int, int]] = (256, 256),
    mean: List[float] = CLIP_DEFAULT_MEAN,
    std: List[float] = CLIP_DEFAULT_STD,
    device: str = "cpu"
) -> torch.Tensor:
    """
    Converts a normalized float32 [0.0, 1.0] HxWxC image into a normalized PyTorch tensor [1, C, H, W].
    """
    if image.ndim == 2:
        image = np.stack([image] * 3, axis=-1)

    # Convert to float32 if not already
    if image.dtype == np.uint8:
        tensor = torch.from_numpy(image.astype(np.float32) / 255.0)
    else:
        tensor = torch.from_numpy(image.astype(np.float32))

    # [H, W, C] -> [C, H, W]
    tensor = tensor.permute(2, 0, 1)

    # Resize if requested
    if target_size is not None:
        tensor = tensor.unsqueeze(0)  # [1, C, H, W]
        tensor = F.interpolate(tensor, size=target_size, mode="bilinear", align_corners=False)
        tensor = tensor.squeeze(0)    # [C, H, W]

    # Apply channel-wise normalization
    mean_t = torch.tensor(mean, dtype=torch.float32).view(-1, 1, 1)
    std_t = torch.tensor(std, dtype=torch.float32).view(-1, 1, 1)
    tensor = (tensor - mean_t) / std_t

    return tensor.unsqueeze(0).to(device)  # [1, C, H, W]


def prepare_model_tensors(
    before_img: np.ndarray,
    after_img: np.ndarray,
    target_size: Tuple[int, int] = (256, 256),
    device: str = "cpu"
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Produces model-ready tensors:
    - `t_before`: [1, 3, H, W]
    - `t_after`: [1, 3, H, W]
    - `t_bitemporal_6ch`: [1, 6, H, W] (Concatenated channels as required by ChangeCLIP SegDataPreProcessor)
    """
    t_before = to_tensor(before_img, target_size=target_size, device=device)
    t_after = to_tensor(after_img, target_size=target_size, device=device)
    t_bitemporal_6ch = torch.cat([t_before, t_after], dim=1)  # [1, 6, H, W]

    return t_before, t_after, t_bitemporal_6ch
