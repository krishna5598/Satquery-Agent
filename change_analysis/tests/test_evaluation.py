"""
Unit tests for evaluation metrics (IoU, F1, Accuracy, Precision, Recall).
"""

from __future__ import annotations
import numpy as np
import pytest

from change_analysis.evaluation.metrics import calculate_change_metrics


def test_metrics_perfect_match():
    gt = np.zeros((50, 50), dtype=np.uint8)
    gt[10:30, 10:30] = 255
    pred = gt.copy()

    m = calculate_change_metrics(pred, gt)
    assert m.iou == 1.0
    assert m.f1_score == 1.0
    assert m.precision == 1.0
    assert m.recall == 1.0
    assert m.accuracy == 1.0
    assert m.cohens_kappa == 1.0
    assert m.true_positives == 20 * 20
    assert m.false_positives == 0
    assert m.false_negatives == 0


def test_metrics_completely_disjoint():
    gt = np.zeros((50, 50), dtype=np.uint8)
    gt[0:20, 0:20] = 255

    pred = np.zeros((50, 50), dtype=np.uint8)
    pred[30:50, 30:50] = 255

    m = calculate_change_metrics(pred, gt)
    assert m.iou == 0.0
    assert m.f1_score == 0.0
    assert m.precision == 0.0
    assert m.recall == 0.0
    assert m.true_positives == 0
    assert m.false_positives == 400
    assert m.false_negatives == 400


def test_metrics_partial_overlap():
    gt = np.zeros((10, 10), dtype=np.uint8)
    gt[:5, :] = 255  # 50 pixels

    pred = np.zeros((10, 10), dtype=np.uint8)
    pred[2:7, :] = 255  # 50 pixels, overlap is rows 2,3,4 (30 pixels)

    m = calculate_change_metrics(pred, gt)
    assert m.true_positives == 30
    assert m.false_positives == 20
    assert m.false_negatives == 20
    assert m.iou == pytest.approx(30 / 70)
    assert m.f1_score == pytest.approx(2 * 30 / (2 * 30 + 20 + 20))
