"""
Evaluation Runner for Comparing Change Masks with Ground Truth Annotations.
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, Any, Union
import numpy as np
from PIL import Image

from change_analysis.evaluation.metrics import calculate_change_metrics


def evaluate_change_prediction(
    prediction: Union[str, Path, np.ndarray],
    ground_truth: Union[str, Path, np.ndarray]
) -> Dict[str, Any]:
    """
    Compares predicted change mask with ground truth annotation.
    Accepts file paths or 2D numpy arrays.
    """
    if isinstance(prediction, (str, Path)):
        pred_arr = np.array(Image.open(prediction))
        if pred_arr.ndim == 3:
            pred_arr = pred_arr[:, :, 0]
    else:
        pred_arr = prediction

    if isinstance(ground_truth, (str, Path)):
        gt_arr = np.array(Image.open(ground_truth))
        if gt_arr.ndim == 3:
            gt_arr = gt_arr[:, :, 0]
    else:
        gt_arr = ground_truth

    # Ensure binary arrays
    pred_bin = (pred_arr > 0).astype(np.uint8)
    gt_bin = (gt_arr > 0).astype(np.uint8)

    metrics = calculate_change_metrics(pred_bin, gt_bin)
    return metrics.to_dict()
