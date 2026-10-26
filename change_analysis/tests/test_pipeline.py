"""
Integration tests for end-to-end ChangePipeline.
"""

from __future__ import annotations
from pathlib import Path
import pytest

from change_analysis.pipeline.change_pipeline import ChangePipeline


@pytest.fixture
def pipeline():
    return ChangePipeline()


def test_pipeline_case_01(pipeline):
    base = Path(__file__).resolve().parents[1]
    b = base / "examples/case_01/before.png"
    a = base / "examples/case_01/after.png"

    if not (b.exists() and a.exists()):
        pytest.skip("Case 01 images not found")

    res = pipeline.run(before_path=b, after_path=a, question="Has construction increased?", case_name="test_case_01")
    assert res["task"] == "bi_temporal_change"
    assert "model" in res
    assert "answer" in res
    assert len(res["answer"]) > 10
    assert res["confidence"] is None
    assert len(res["evidence"]) >= 3
    assert res["metrics"]["changed_area_fraction"] > 0.05

    # Check evidence file existence on disk
    for ev in res["evidence"]:
        p = Path(ev["path"])
        assert p.exists(), f"Evidence file missing: {p}"


def test_pipeline_case_no_change(pipeline):
    base = Path(__file__).resolve().parents[1]
    b = base / "examples/case_no_change/before.png"
    a = base / "examples/case_no_change/after.png"

    if not (b.exists() and a.exists()):
        pytest.skip("Case no change images not found")

    res = pipeline.run(before_path=b, after_path=a, question="What changed?", case_name="test_no_change")
    assert res["metrics"]["changed_area_fraction"] < 0.02
    assert "no significant" in res["answer"].lower()
