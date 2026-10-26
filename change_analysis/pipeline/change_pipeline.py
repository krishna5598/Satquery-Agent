"""
Unified Change Analysis Pipeline for SatQuery AI.

Orchestrates:
Validation -> Alignment & Preprocessing -> Model Execution (ChangeCLIP / ChangeChat / Fallback) ->
Visual Evidence Generation -> Contract Formatting.
"""

from __future__ import annotations
import os
import time
from pathlib import Path
from typing import Dict, Any, Optional, Union
import numpy as np
from PIL import Image

from change_analysis.config import AppConfig, load_config
from change_analysis.preprocessing.validator import validate_bitemporal_pair, extract_metadata
from change_analysis.preprocessing.normalization import normalize_image_pair
from change_analysis.preprocessing.alignment import align_image_pair
from change_analysis.models.fallback_runner import DeterministicFallbackRunner
from change_analysis.models.changeclip_runner import ChangeCLIPRunner
from change_analysis.models.changechat_runner import ChangeChatRunner
from change_analysis.visualization.change_map import save_visual_artifacts


class ChangePipeline:
    """End-to-end Change Analysis Pipeline."""

    def __init__(self, config: Optional[AppConfig] = None):
        self.config = config or load_config()
        self.device = self.config.effective_device
        self.mode = self.config.effective_mode

        # Initialize runners lazily or on startup
        self.fallback_runner = DeterministicFallbackRunner(config=self.config.fallback)
        self.changeclip_runner = ChangeCLIPRunner(config=self.config.changeclip, device=self.device)
        self.changechat_runner = ChangeChatRunner(config=self.config.changechat, device=self.device)

    def run(
        self,
        before_path: str | Path,
        after_path: str | Path,
        question: Optional[str] = None,
        case_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Executes end-to-end change analysis.

        Returns structured dictionary adhering to the SatQuery AI specialist tool schema.
        """
        start_time = time.time()
        before_p = Path(before_path)
        after_p = Path(after_path)
        q_str = question or "What changed between these two images?"
        prefix = case_name or f"change_{int(time.time())}"

        # 1. Validation Stage
        val_result = validate_bitemporal_pair(
            before_path=before_p,
            after_path=after_p,
            verify_crs_bounds=self.config.preprocessing.verify_crs_bounds
        )

        if not val_result.is_valid:
            return {
                "task": "bi_temporal_change",
                "model": "validation_failed",
                "answer": f"Validation failed: {'; '.join(val_result.errors)}",
                "confidence": None,
                "evidence": [],
                "metrics": {},
                "warnings": val_result.warnings,
                "parameters": {
                    "before": str(before_p),
                    "after": str(after_p),
                    "question": q_str,
                },
                "errors": val_result.errors,
            }

        # 2. Image Loading & Preprocessing
        _, _, raw_before = extract_metadata(before_p)
        _, _, raw_after = extract_metadata(after_p)

        assert raw_before is not None and raw_after is not None

        # Normalize raw arrays into standard 3-channel representations
        container_before, container_after = normalize_image_pair(raw_before, raw_after)

        # Spatial alignment & dimension harmonization
        before_aligned, after_aligned, was_aligned = align_image_pair(
            container_before.display_rgb,
            container_after.display_rgb,
            enable_feature_registration=False
        )

        warnings = list(val_result.warnings)
        if was_aligned:
            warnings.append("Dimensions differed; AFTER image was spatially resampled to match BEFORE image.")

        # 3. Model Inference Routing
        use_real = (self.mode == "real" and self.changeclip_runner.is_loaded)
        
        if use_real:
            model_name = "ChangeCLIP + ChangeChat"
            clip_res = self.changeclip_runner.analyze(before_aligned, after_aligned)
            change_prob = clip_res.change_prob_map
            change_mask = clip_res.change_mask
            warnings.extend(clip_res.warnings)

            # Extract basic metrics
            total_px = before_aligned.shape[0] * before_aligned.shape[1]
            changed_px = int(np.count_nonzero(change_mask > 0))
            frac = round(float(changed_px / total_px), 4)

            # Pass evidence to ChangeChat for multimodal QA
            evidence_ctx = {
                "changed_area_fraction": frac,
                "changed_pixel_count": changed_px,
                "dominant_change_type": "surface alteration",
                "changed_regions_count": 1,
            }
            chat_res = self.changechat_runner.analyze(
                before_aligned, after_aligned, question=q_str, evidence_context=evidence_ctx
            )
            answer = chat_res.answer
            warnings.extend(chat_res.warnings)

            metrics = {
                "changed_area_fraction": frac,
                "changed_pixel_count": changed_px,
                "total_pixel_count": total_px,
            }
            confidence = None

        else:
            # Deterministic Fallback Mode
            model_name = "deterministic_fallback"
            fb_res = self.fallback_runner.analyze(before_aligned, after_aligned, question=q_str)
            change_prob = fb_res.change_prob_map
            change_mask = fb_res.change_mask
            metrics = fb_res.metrics
            answer = fb_res.answer
            confidence = fb_res.confidence
            warnings.extend(fb_res.warnings)

        # 4. Save Visual Artifacts
        base_dir = Path(__file__).resolve().parents[1]
        out_dir = base_dir / self.config.paths.output_dir

        artifacts = save_visual_artifacts(
            output_dir=out_dir,
            case_name=prefix,
            before_rgb=before_aligned,
            after_rgb=after_aligned,
            change_mask=change_mask,
            change_prob=change_prob,
            metrics=metrics
        )

        # 5. Assemble Evidence List
        evidence = [
            {
                "type": "change_map",
                "path": artifacts["change_map"],
                "description": "Binary raster mask indicating detected pixel-level change footprints.",
            },
            {
                "type": "overlay",
                "path": artifacts["overlay"],
                "description": "Composite visualization with changes highlighted in red over the post-change acquisition.",
            },
            {
                "type": "comparison",
                "path": artifacts["comparison"],
                "description": "4-panel demonstration figure: Before (T1), After (T2), Change Mask, and Change Overlay.",
            },
        ]
        if "heatmap" in artifacts:
            evidence.append({
                "type": "heatmap",
                "path": artifacts["heatmap"],
                "description": "Continuous change probability heatmap.",
            })

        elapsed = round(time.time() - start_time, 3)

        # 6. Strict Machine-Readable Result Schema
        return {
            "task": "bi_temporal_change",
            "model": model_name,
            "answer": answer,
            "confidence": confidence,
            "evidence": evidence,
            "metrics": metrics,
            "warnings": warnings,
            "parameters": {
                "before": str(before_p),
                "after": str(after_p),
                "question": q_str,
                "device": self.device,
                "elapsed_seconds": elapsed,
            },
        }


def analyze_change(
    before: str | Path,
    after: str | Path,
    question: Optional[str] = None,
    case_name: Optional[str] = None
) -> Dict[str, Any]:
    """Convenience functional interface for Change Analysis."""
    pipeline = ChangePipeline()
    return pipeline.run(before_path=before, after_path=after, question=question, case_name=case_name)
