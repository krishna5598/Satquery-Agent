"""
Change Map and Visual Evidence Generation for SatQuery AI.

Generates high-contrast binary change masks, continuous probability heatmaps,
bitemporal overlays with highlighted contours, and comprehensive 4-panel comparison figures.
"""

from __future__ import annotations
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont


def create_change_heatmap(change_prob: np.ndarray) -> np.ndarray:
    """Converts a continuous float [0.0, 1.0] change map into an RGB heatmap."""
    prob_scaled = np.clip(change_prob * 255.0, 0, 255).astype(np.uint8)
    heatmap_bgr = cv2.applyColorMap(prob_scaled, cv2.COLORMAP_TURBO)
    return cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)


def create_overlay(
    base_rgb: np.ndarray,
    change_mask: np.ndarray,
    color: Tuple[int, int, int] = (255, 30, 30),
    alpha: float = 0.45
) -> np.ndarray:
    """
    Overlays changed pixels on top of the base image with semi-transparency
    and distinct boundary contours.
    """
    overlay = base_rgb.copy()
    mask_bool = change_mask > 0

    if not np.any(mask_bool):
        return overlay

    # Create solid color layer
    color_layer = np.zeros_like(base_rgb)
    color_layer[:] = color

    # Blend changed pixels
    overlay[mask_bool] = (
        (1.0 - alpha) * base_rgb[mask_bool] + alpha * color_layer[mask_bool]
    ).astype(np.uint8)

    # Draw crisp outer boundary contours
    contours, _ = cv2.findContours((change_mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (255, 255, 255), 1)  # White border
    cv2.drawContours(overlay, contours, -1, color, 2)             # Colored outer contour

    return overlay


def draw_header_bar(image: np.ndarray, title: str, subtitle: Optional[str] = None) -> np.ndarray:
    """Adds a stylish dark header banner with text over an image panel."""
    h, w = image.shape[:2]
    header_h = 36 if not subtitle else 48
    header = np.zeros((header_h, w, 3), dtype=np.uint8)
    header[:] = (20, 24, 30)

    # Draw text with cv2
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(header, title, (10, 22), font, 0.55, (240, 240, 245), 1, cv2.LINE_AA)
    if subtitle:
        cv2.putText(header, subtitle, (10, 40), font, 0.38, (160, 170, 185), 1, cv2.LINE_AA)

    return np.vstack([header, image])


def create_comparison_figure(
    before_rgb: np.ndarray,
    after_rgb: np.ndarray,
    change_mask: np.ndarray,
    change_prob: Optional[np.ndarray] = None,
    overlay_rgb: Optional[np.ndarray] = None,
    title: str = "SatQuery AI - Bi-Temporal Change Detection",
    metadata_text: str = ""
) -> np.ndarray:
    """
    Assembles a demonstration-ready 4-panel comparison figure:
    Panel 1: Before (T1) | Panel 2: After (T2)
    Panel 3: Change Mask  | Panel 4: Change Overlay
    """
    h, w = before_rgb.shape[:2]

    # Ensure all components have matching spatial resolution
    p1 = draw_header_bar(before_rgb, "1. BEFORE ACQUISITION (T1)", "Pre-change optical baseline")
    p2 = draw_header_bar(after_rgb, "2. AFTER ACQUISITION (T2)", "Post-change observation")

    # Change Mask panel (binary with legend)
    mask_3ch = np.stack([change_mask] * 3, axis=-1)
    p3 = draw_header_bar(mask_3ch, "3. BINARY CHANGE MASK", "White: Detected Alterations")

    # Overlay panel
    if overlay_rgb is None:
        overlay_rgb = create_overlay(after_rgb, change_mask)
    p4 = draw_header_bar(overlay_rgb, "4. CHANGE OVERLAY", "Red highlight: Extracted change footprints")

    # Combine 2x2 grid
    row1 = np.hstack([p1, p2])
    row2 = np.hstack([p3, p4])
    grid = np.vstack([row1, row2])

    # Add top title banner
    grid_w = grid.shape[1]
    banner_h = 56
    banner = np.zeros((banner_h, grid_w, 3), dtype=np.uint8)
    banner[:] = (12, 16, 22)

    font = cv2.FONT_HERSHEY_DUPLEX
    cv2.putText(banner, title.upper(), (16, 28), font, 0.65, (0, 220, 255), 1, cv2.LINE_AA)
    if metadata_text:
        cv2.putText(banner, metadata_text, (16, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 190, 205), 1, cv2.LINE_AA)

    # Add bottom status bar
    footer_h = 24
    footer = np.zeros((footer_h, grid_w, 3), dtype=np.uint8)
    footer[:] = (15, 18, 24)
    cv2.putText(footer, "SatQuery AI Multimodal Remote-Sensing Specialist Engine | Member 3", (16, 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (110, 120, 135), 1, cv2.LINE_AA)

    final_figure = np.vstack([banner, grid, footer])
    return final_figure


def save_visual_artifacts(
    output_dir: str | Path,
    case_name: str,
    before_rgb: np.ndarray,
    after_rgb: np.ndarray,
    change_mask: np.ndarray,
    change_prob: Optional[np.ndarray] = None,
    metrics: Optional[Dict[str, Any]] = None
) -> Dict[str, str]:
    """
    Saves visual artifacts to disk following strict deterministic naming:
    - {case_name}_before.png
    - {case_name}_after.png
    - {case_name}_change_map.png
    - {case_name}_overlay.png
    - {case_name}_comparison.png
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    artifacts = {}

    # 1. Before Image
    before_path = out_path / f"{case_name}_before.png"
    Image.fromarray(before_rgb).save(before_path)
    artifacts["before"] = str(before_path.resolve())

    # 2. After Image
    after_path = out_path / f"{case_name}_after.png"
    Image.fromarray(after_rgb).save(after_path)
    artifacts["after"] = str(after_path.resolve())

    # 3. Change Mask / Binary Map
    mask_path = out_path / f"{case_name}_change_map.png"
    Image.fromarray(change_mask).save(mask_path)
    artifacts["change_map"] = str(mask_path.resolve())

    # 4. Change Overlay
    overlay = create_overlay(after_rgb, change_mask)
    overlay_path = out_path / f"{case_name}_overlay.png"
    Image.fromarray(overlay).save(overlay_path)
    artifacts["overlay"] = str(overlay_path.resolve())

    # 5. Continuous Probability Heatmap (if provided)
    if change_prob is not None:
        heatmap = create_change_heatmap(change_prob)
        heatmap_path = out_path / f"{case_name}_heatmap.png"
        Image.fromarray(heatmap).save(heatmap_path)
        artifacts["heatmap"] = str(heatmap_path.resolve())

    # 6. Full 4-Panel Comparison
    meta_str = ""
    if metrics:
        pct = metrics.get("changed_area_fraction", 0.0) * 100
        dom = metrics.get("dominant_change_type", "surface variation")
        meta_str = f"Changed Area: {pct:.2f}% | Dominant Signature: {dom.replace('_', ' ').title()} | Model: Deterministic Multimodal Pipeline"

    comp_figure = create_comparison_figure(
        before_rgb=before_rgb,
        after_rgb=after_rgb,
        change_mask=change_mask,
        overlay_rgb=overlay,
        title=f"SatQuery AI - {case_name.replace('_', ' ').title()}",
        metadata_text=meta_str,
    )
    comp_path = out_path / f"{case_name}_comparison.png"
    Image.fromarray(comp_figure).save(comp_path)
    artifacts["comparison"] = str(comp_path.resolve())

    return artifacts
