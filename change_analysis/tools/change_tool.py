"""
Specialist Tool Interface for SatQuery AI Agentic Architecture.

Provides the standardized `change_tool()` contract for multimodal routing:

USER QUERY -> ROUTER -> CHANGE ANALYSIS SPECIALIST -> STRUCTURED EVIDENCE -> FINAL ANSWER
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, Any, Optional, Union

from change_analysis.pipeline.change_pipeline import ChangePipeline, analyze_change


def change_tool(
    before_image: str | Path,
    after_image: str | Path,
    question: Optional[str] = None,
    case_name: Optional[str] = None
) -> Dict[str, Any]:
    """
    Standard SatQuery AI specialist tool interface for bi-temporal change analysis.

    Parameters:
        before_image: Path to the pre-change satellite image (TIFF, PNG, JPG).
        after_image:  Path to the post-change satellite image (TIFF, PNG, JPG).
        question:     Optional natural language question (e.g. 'Has construction increased?').
        case_name:    Optional identifier for saved visual artifacts.

    Returns:
        Standardized machine-readable dictionary following the shared agent contract:
        {
            "task": "bi_temporal_change",
            "model": "ChangeCLIP + ChangeChat" | "deterministic_fallback",
            "answer": "...",
            "confidence": null,
            "evidence": [
                {"type": "change_map", "path": "..."},
                {"type": "overlay", "path": "..."},
                {"type": "comparison", "path": "..."}
            ],
            "metrics": {
                "changed_area_fraction": float,
                "changed_pixel_count": int,
                ...
            },
            "warnings": [...],
            "parameters": {
                "before": "...",
                "after": "...",
                "question": "..."
            }
        }
    """
    return analyze_change(
        before=before_image,
        after=after_image,
        question=question,
        case_name=case_name
    )
