"""
FastAPI Local Microservice for SatQuery AI Change Analysis.

Provides HTTP REST endpoints:
- GET  /health          : Status, hardware specs, and model availability.
- POST /analyze-change  : Multipart image upload or path-based change detection.
"""

from __future__ import annotations
import os
import sys
import shutil
import tempfile
from pathlib import Path

# Ensure project root is in sys.path when executed directly
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from typing import Optional, Dict, Any
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from change_analysis.hardware_check import detect_hardware
from change_analysis.pipeline.change_pipeline import ChangePipeline
from change_analysis.tools.change_tool import change_tool


app = FastAPI(
    title="SatQuery AI - Change Analysis Specialist Service",
    description="Multimodal bi-temporal satellite image change detection & question answering.",
    version="1.0.0",
)

pipeline = ChangePipeline()


class PathAnalysisRequest(BaseModel):
    before_path: str
    after_path: str
    question: Optional[str] = "What changed between these two images?"
    case_name: Optional[str] = None


@app.get("/health")
def health_check() -> Dict[str, Any]:
    """Returns operational status, host hardware capabilities, and configured mode."""
    hw = detect_hardware()
    return {
        "status": "healthy",
        "service": "SatQuery AI Change Analysis Specialist",
        "device": pipeline.device,
        "mode": pipeline.mode,
        "hardware": {
            "os": hw["os"],
            "python": hw["python"],
            "pytorch": hw["pytorch"],
            "cuda_available": hw["cuda_available"],
            "gpu": hw["gpu"],
            "total_ram_gb": hw["total_ram_gb"],
            "free_ram_gb": hw["free_ram_gb"],
            "recommended_mode": hw["recommended_mode"],
        },
        "runners": {
            "changeclip_loaded": pipeline.changeclip_runner.is_loaded,
            "changechat_loaded": pipeline.changechat_runner.is_loaded,
            "fallback_available": True,
        }
    }


@app.post("/analyze-change")
async def analyze_change_endpoint(
    before_file: Optional[UploadFile] = File(None),
    after_file: Optional[UploadFile] = File(None),
    question: Optional[str] = Form("What changed between these two images?"),
    case_name: Optional[str] = Form(None)
) -> JSONResponse:
    """
    Analyzes bi-temporal imagery submitted via multipart file upload.
    """
    if before_file is None or after_file is None:
        raise HTTPException(status_code=400, detail="Both before_file and after_file are required.")

    # Save uploaded files temporarily
    temp_dir = Path(tempfile.mkdtemp(prefix="satquery_upload_"))
    try:
        before_ext = Path(before_file.filename or "before.png").suffix or ".png"
        after_ext = Path(after_file.filename or "after.png").suffix or ".png"

        b_path = temp_dir / f"before{before_ext}"
        a_path = temp_dir / f"after{after_ext}"

        with open(b_path, "wb") as fb:
            shutil.copyfileobj(before_file.file, fb)
        with open(a_path, "wb") as fa:
            shutil.copyfileobj(after_file.file, fa)

        result = pipeline.run(
            before_path=b_path,
            after_path=a_path,
            question=question,
            case_name=case_name or "api_upload"
        )
        return JSONResponse(content=result)

    finally:
        # Clean temporary uploads
        shutil.rmtree(temp_dir, ignore_errors=True)


@app.post("/analyze-change-by-path")
def analyze_change_by_path_endpoint(request: PathAnalysisRequest) -> Dict[str, Any]:
    """
    Analyzes bi-temporal imagery given existing server file paths.
    """
    b_path = Path(request.before_path)
    a_path = Path(request.after_path)

    if not b_path.exists():
        raise HTTPException(status_code=404, detail=f"Before image path not found: {request.before_path}")
    if not a_path.exists():
        raise HTTPException(status_code=404, detail=f"After image path not found: {request.after_path}")

    return pipeline.run(
        before_path=b_path,
        after_path=a_path,
        question=request.question,
        case_name=request.case_name
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
