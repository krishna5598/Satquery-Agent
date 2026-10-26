"""
LangGraph Specialist Node Adapter for SatQuery AI.

Allows the Change Analysis module to be plugged directly as a node
inside a LangGraph state graph workflow without modifying core model logic.
"""

from __future__ import annotations
from typing import Dict, Any
from change_analysis.tools.change_tool import change_tool


def make_langgraph_change_node():
    """
    Returns a callable LangGraph node function compatible with standard StateGraph states:
    
    def change_analysis_node(state: Dict[str, Any]) -> Dict[str, Any]:
        ...
    """
    def change_analysis_node(state: Dict[str, Any]) -> Dict[str, Any]:
        # Extract inputs from state
        before = state.get("before_image") or state.get("before")
        after = state.get("after_image") or state.get("after")
        question = state.get("question") or state.get("query")
        case_name = state.get("case_name")

        if not before or not after:
            error_res = {
                "task": "bi_temporal_change",
                "error": "State missing 'before_image' or 'after_image' parameter."
            }
            return {**state, "change_analysis_result": error_res}

        # Invoke specialist tool
        tool_output = change_tool(
            before_image=before,
            after_image=after,
            question=question,
            case_name=case_name
        )

        # Update state with specialist evidence and answer
        return {
            **state,
            "change_analysis_result": tool_output,
            "evidence": state.get("evidence", []) + tool_output.get("evidence", []),
            "current_specialist": "change_analysis",
        }

    return change_analysis_node
