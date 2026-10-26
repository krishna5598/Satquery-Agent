"""
Spatial Alignment and Geometric Co-registration for Bi-temporal Satellite Imagery.
"""

from __future__ import annotations
from typing import Tuple
import numpy as np
import cv2


def align_image_pair(
    before_img: np.ndarray,
    after_img: np.ndarray,
    enable_feature_registration: bool = False
) -> Tuple[np.ndarray, np.ndarray, bool]:
    """
    Ensures spatial co-registration between before and after images.

    1. Checks if dimension resampling is required.
    2. Resamples `after_img` to exactly match `before_img` dimensions.
    3. Optionally runs sub-pixel ECC/homography alignment if sensors have small positional drifts.
    
    Returns:
        (aligned_before, aligned_after, registration_applied)
    """
    h1, w1 = before_img.shape[:2]
    h2, w2 = after_img.shape[:2]

    aligned_after = after_img.copy()
    aligned_before = before_img.copy()
    was_modified = False

    # 1. Resample to match dimensions if different
    if (w1, h1) != (w2, h2):
        interp = cv2.INTER_AREA if (w2 > w1 or h2 > h1) else cv2.INTER_CUBIC
        aligned_after = cv2.resize(after_img, (w1, h1), interpolation=interp)
        was_modified = True

    # 2. Optional sub-pixel feature registration
    if enable_feature_registration:
        try:
            gray1 = cv2.cvtColor(aligned_before, cv2.COLOR_RGB2GRAY)
            gray2 = cv2.cvtColor(aligned_after, cv2.COLOR_RGB2GRAY)

            warp_matrix = np.eye(2, 3, dtype=np.float32)
            criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 1e-4)
            _, warp_matrix = cv2.findTransformECC(
                gray1, gray2, warp_matrix, cv2.MOTION_TRANSLATION, criteria
            )
            aligned_after = cv2.warpAffine(
                aligned_after, warp_matrix, (w1, h1),
                flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
                borderMode=cv2.BORDER_REFLECT
            )
            was_modified = True
        except Exception:
            # If ECC fails (e.g. huge scene change), retain resized image
            pass

    return aligned_before, aligned_after, was_modified
