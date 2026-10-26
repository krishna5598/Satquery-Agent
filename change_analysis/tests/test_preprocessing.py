"""
Unit tests for alignment, normalization, and tensor preparation.
"""

from __future__ import annotations
import numpy as np
import pytest
import torch

from change_analysis.preprocessing.normalization import (
    to_uint8_rgb,
    normalize_image_pair,
    ImageContainer,
)
from change_analysis.preprocessing.alignment import align_image_pair
from change_analysis.preprocessing.transforms import prepare_model_tensors, to_tensor


def test_to_uint8_rgb_conversions():
    # 2D grayscale
    gray = np.array([[0, 128], [255, 64]], dtype=np.uint8)
    rgb = to_uint8_rgb(gray)
    assert rgb.shape == (2, 2, 3)
    assert rgb.dtype == np.uint8

    # Float [0.0, 1.0]
    flt = np.array([[0.0, 0.5], [1.0, 0.25]], dtype=np.float32)
    flt_rgb = to_uint8_rgb(flt)
    assert flt_rgb.shape == (2, 2, 3)
    assert flt_rgb.dtype == np.uint8
    assert flt_rgb[1, 0, 0] == 255

    # RGBA 4 channels to 3 channels
    rgba = np.zeros((10, 10, 4), dtype=np.uint8)
    rgba_rgb = to_uint8_rgb(rgba)
    assert rgba_rgb.shape == (10, 10, 3)


def test_normalize_image_pair():
    b_raw = np.full((64, 64, 3), 100, dtype=np.uint8)
    a_raw = np.full((64, 64, 3), 200, dtype=np.uint8)

    c_b, c_a = normalize_image_pair(b_raw, a_raw)
    assert isinstance(c_b, ImageContainer)
    assert c_b.normalized.shape == (64, 64, 3)
    assert c_b.normalized.dtype == np.float32
    assert c_b.normalized[0, 0, 0] == pytest.approx(100 / 255.0, abs=1e-3)
    assert c_a.normalized[0, 0, 0] == pytest.approx(200 / 255.0, abs=1e-3)


def test_align_image_pair_resampling():
    before = np.zeros((100, 120, 3), dtype=np.uint8)
    after = np.full((80, 60, 3), 150, dtype=np.uint8)

    al_b, al_a, modified = align_image_pair(before, after)
    assert modified is True
    assert al_b.shape == (100, 120, 3)
    assert al_a.shape == (100, 120, 3)


def test_prepare_model_tensors():
    b_img = np.zeros((128, 128, 3), dtype=np.uint8)
    a_img = np.zeros((128, 128, 3), dtype=np.uint8)

    t_b, t_a, t_6ch = prepare_model_tensors(b_img, a_img, target_size=(256, 256), device="cpu")
    assert t_b.shape == (1, 3, 256, 256)
    assert t_a.shape == (1, 3, 256, 256)
    assert t_6ch.shape == (1, 6, 256, 256)
    assert isinstance(t_6ch, torch.Tensor)
