"""
Unit tests for deterministic fallback change analysis runner.
"""

from __future__ import annotations
import numpy as np
import pytest

from change_analysis.models.fallback_runner import DeterministicFallbackRunner


@pytest.fixture
def fallback_runner():
    return DeterministicFallbackRunner()


def test_fallback_identical_images(fallback_runner):
    # Identical scenes should yield 0.0 change detected
    b = np.full((128, 128, 3), 120, dtype=np.uint8)
    a = b.copy()

    res = fallback_runner.analyze(b, a)
    assert res.model == "deterministic_fallback"
    assert res.change_detected is False
    assert res.changed_area_fraction == 0.0
    assert res.changed_pixel_count == 0
    assert res.confidence is None
    assert "No significant" in res.answer


def test_fallback_high_contrast_change(fallback_runner):
    b = np.zeros((128, 128, 3), dtype=np.uint8)
    a = b.copy()
    # Add a large bright square in 'after'
    a[20:80, 20:80] = 230

    res = fallback_runner.analyze(b, a, question="What changed?")
    assert res.change_detected is True
    assert res.changed_area_fraction > 0.15
    assert res.changed_pixel_count > 1000
    assert len(res.regions) >= 1
    assert res.confidence is None


def test_fallback_spectral_proxy_vegetation(fallback_runner):
    # Before is dense vegetation (high green)
    b = np.full((100, 100, 3), [30, 150, 30], dtype=np.uint8)
    # After is cleared bare soil (high red, low green)
    a = np.full((100, 100, 3), [180, 130, 90], dtype=np.uint8)

    res = fallback_runner.analyze(b, a, question="Did vegetation decrease?")
    assert res.change_detected is True
    assert res.metrics["mean_delta_vegetation_index"] < -0.10
    assert "vegetation reduction" in res.answer.lower() or "clearing" in res.answer.lower()
