"""
Statistical Metrics for Remote Sensing Change Detection.

Calculates IoU, F1-Score, Precision, Recall, Overall Accuracy, and Cohen's Kappa
strictly when genuine ground truth binary change masks are provided.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, Optional
import numpy as np


@dataclass
class EvaluationMetrics:
    iou: float
    f1_score: float
    precision: float
    recall: float
    accuracy: float
    cohens_kappa: float
    true_positives: int
    false_positives: int
    true_negatives: int
    false_negatives: int
    total_pixels: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "iou": round(self.iou, 4),
            "f1_score": round(self.f1_score, 4),
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "overall_accuracy": round(self.accuracy, 4),
            "cohens_kappa": round(self.cohens_kappa, 4),
            "confusion_matrix": {
                "TP": self.true_positives,
                "FP": self.false_positives,
                "TN": self.true_negatives,
                "FN": self.false_negatives,
            },
            "total_evaluated_pixels": self.total_pixels,
        }


def calculate_change_metrics(
    prediction_mask: np.ndarray,
    ground_truth_mask: np.ndarray
) -> EvaluationMetrics:
    """
    Computes statistical change detection metrics against ground truth.

    Inputs:
        prediction_mask: 2D binary array (values > 0 denote change)
        ground_truth_mask: 2D binary array (values > 0 denote change)
    """
    pred = (prediction_mask > 0).astype(bool)
    gt = (ground_truth_mask > 0).astype(bool)

    if pred.shape != gt.shape:
        raise ValueError(
            f"Shape mismatch: Prediction mask {pred.shape} vs Ground Truth {gt.shape}"
        )

    total_pixels = pred.size

    # Calculate Confusion Matrix
    tp = int(np.sum(pred & gt))
    fp = int(np.sum(pred & ~gt))
    tn = int(np.sum(~pred & ~gt))
    fn = int(np.sum(~pred & gt))

    # Precision, Recall, F1
    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    f1 = float(2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    # Intersection over Union (Jaccard Index)
    intersection = tp
    union = tp + fp + fn
    iou = float(intersection / union) if union > 0 else 1.0 if (tp == 0 and fp == 0 and fn == 0) else 0.0

    # Overall Accuracy
    accuracy = float((tp + tn) / total_pixels) if total_pixels > 0 else 0.0

    # Cohen's Kappa Coefficient
    # Expected agreement by chance Pe
    p_actual_pos = (tp + fn) / total_pixels
    p_actual_neg = (tn + fp) / total_pixels
    p_pred_pos = (tp + fp) / total_pixels
    p_pred_neg = (tn + fn) / total_pixels
    pe = (p_actual_pos * p_pred_pos) + (p_actual_neg * p_pred_neg)

    po = accuracy
    if pe >= 1.0 or (1.0 - pe) == 0:
        kappa = 1.0 if po == 1.0 else 0.0
    else:
        kappa = float((po - pe) / (1.0 - pe))

    return EvaluationMetrics(
        iou=iou,
        f1_score=f1,
        precision=precision,
        recall=recall,
        accuracy=accuracy,
        cohens_kappa=kappa,
        true_positives=tp,
        false_positives=fp,
        true_negatives=tn,
        false_negatives=fn,
        total_pixels=total_pixels,
    )
