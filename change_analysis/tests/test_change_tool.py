"""
Contract tests for change_tool() specialist interface.
"""

from __future__ import annotations
from pathlib import Path
import pytest

from change_analysis.tools.change_tool import change_tool


def test_change_tool_schema_compliance():
    base = Path(__file__).resolve().parents[1]
    b = base / "examples/case_01/before.png"
    a = base / "examples/case_01/after.png"

    if not (b.exists() and a.exists()):
        pytest.skip("Case 01 images not found")

    output = change_tool(before_image=b, after_image=a, question="What changed?")

    # Required top-level keys
    required_keys = {"task", "model", "answer", "confidence", "evidence", "metrics", "warnings", "parameters"}
    assert required_keys.issubset(output.keys()), f"Missing keys in tool output: {required_keys - set(output.keys())}"

    assert output["task"] == "bi_temporal_change"
    assert isinstance(output["answer"], str)
    assert output["confidence"] is None  # Never fabricated
    assert isinstance(output["evidence"], list)
    assert isinstance(output["metrics"], dict)
    assert isinstance(output["warnings"], list)
    assert isinstance(output["parameters"], dict)

    # Check evidence structure
    for ev in output["evidence"]:
        assert "type" in ev
        assert "path" in ev
        assert Path(ev["path"]).exists()
