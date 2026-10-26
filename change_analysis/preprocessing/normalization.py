"""
Image Normalization and Radiometric Calibration for Bi-temporal Satellite Imagery.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, Optional
import numpy as np
from PIL import Image
import cv2


@dataclass
class ImageContainer:
    """Encapsulates raw, display-ready, and normalized representations of an image."""
    raw: np.ndarray             # Original raw array as loaded
    display_rgb: np.ndarray     # Standardized uint8 [0, 255] RGB for visual display
    normalized: np.ndarray      # Normalized float32 [0.0, 1.0] RGB
    height: int
    width: int
    channels: int
    dtype: str


def to_uint8_rgb(array: np.ndarray) -> np.ndarray:
    """Converts any 2D/3D numerical image array into standard 3-channel uint8 RGB."""
    arr = np.nan_to_num(array.copy(), nan=0.0, posinf=255.0, neginf=0.0)

    # Convert single channel to 3 channels
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    elif arr.ndim == 3 and arr.shape[2] == 1:
        arr = np.repeat(arr, 3, axis=2)
    elif arr.ndim == 3 and arr.shape[2] > 3:
        # If RGBA or multi-band, select first 3 bands (RGB)
        arr = arr[:, :, :3]

    # Normalize intensity to uint8 [0, 255]
    if arr.dtype == np.uint8:
        return arr
    elif arr.dtype == np.uint16:
        arr = (arr / 257.0).astype(np.uint8)
    elif np.issubdtype(arr.dtype, np.floating):
        min_v, max_v = float(np.min(arr)), float(np.max(arr))
        if max_v <= 1.0 and min_v >= 0.0:
            arr = (arr * 255.0).astype(np.uint8)
        elif max_v > min_v:
            # 2% - 98% percentile linear stretch (standard remote sensing display stretch)
            p2, p98 = np.percentile(arr, (2, 98))
            if p98 > p2:
                arr = np.clip((arr - p2) / (p98 - p2) * 255.0, 0, 255).astype(np.uint8)
            else:
                arr = np.clip(arr, 0, 255).astype(np.uint8)
        else:
            arr = np.zeros_like(arr, dtype=np.uint8)
    else:
        arr = cv2.normalize(arr, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    return arr


def match_radiometry(source: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """
    Applies histogram matching to balance sensor exposure and atmospheric illumination
    differences between bitemporal acquisitions without destroying true physical change.
    """
    matched = np.zeros_like(source)
    for c in range(3):
        s_chan = source[:, :, c]
        r_chan = reference[:, :, c]
        s_hist, _ = np.histogram(s_chan, 256, [0, 256])
        r_hist, _ = np.histogram(r_chan, 256, [0, 256])

        s_cdf = s_hist.cumsum()
        s_cdf = s_cdf / (s_cdf[-1] + 1e-7)

        r_cdf = r_hist.cumsum()
        r_cdf = r_cdf / (r_cdf[-1] + 1e-7)

        lut = np.interp(s_cdf, r_cdf, np.arange(256)).astype(np.uint8)
        matched[:, :, c] = cv2.LUT(s_chan, lut)
    return matched


def normalize_image_pair(
    before_raw: np.ndarray,
    after_raw: np.ndarray,
    balance_radiometry: bool = False
) -> Tuple[ImageContainer, ImageContainer]:
    """
    Normalizes a pair of before/after images into unified ImageContainers.
    Ensures both have consistent 3-channel uint8 RGB and float32 [0.0, 1.0] representations.
    """
    before_rgb = to_uint8_rgb(before_raw)
    after_rgb = to_uint8_rgb(after_raw)

    if balance_radiometry:
        after_rgb = match_radiometry(after_rgb, before_rgb)

    before_norm = before_rgb.astype(np.float32) / 255.0
    after_norm = after_rgb.astype(np.float32) / 255.0

    before_c = ImageContainer(
        raw=before_raw,
        display_rgb=before_rgb,
        normalized=before_norm,
        height=before_rgb.shape[0],
        width=before_rgb.shape[1],
        channels=before_rgb.shape[2],
        dtype=str(before_raw.dtype),
    )

    after_c = ImageContainer(
        raw=after_raw,
        display_rgb=after_rgb,
        normalized=after_norm,
        height=after_rgb.shape[0],
        width=after_rgb.shape[1],
        channels=after_rgb.shape[2],
        dtype=str(after_raw.dtype),
    )

    return before_c, after_c
