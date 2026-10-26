"""
Automated Demonstration Script for SatQuery AI Change Analysis.

Runs all four benchmark demo cases autonomously:
Case 1: Construction / Built-up Expansion (LEVIR-CD)
Case 2: Water Change (Hydrological Inundation)
Case 3: Vegetation Clearing (Deforestation)
Case 4: No Significant Change (Stability Baseline)

Outputs visual artifacts, JSON results, and prints an executive summary.
"""

from __future__ import annotations
import json
import sys
import time
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

from change_analysis.tools.change_tool import change_tool
from change_analysis.evaluation.evaluate import evaluate_change_prediction


DEMO_CASES = [
    {
        "id": "case_01",
        "title": "CASE 1: Construction / Built-up Expansion (LEVIR-CD)",
        "folder": "examples/case_01",
        "question": "Has construction increased in this area?",
    },
    {
        "id": "case_02",
        "title": "CASE 2: Water Body / Reservoir Inundation Change",
        "folder": "examples/case_02",
        "question": "Is there evidence of water expansion or flooding?",
    },
    {
        "id": "case_03",
        "title": "CASE 3: Vegetation Clearing / Forest Canopy Loss",
        "folder": "examples/case_03",
        "question": "Did vegetation or forest canopy decrease?",
    },
    {
        "id": "case_no_change",
        "title": "CASE 4: No Significant Change (Scene Stability Baseline)",
        "folder": "examples/case_no_change",
        "question": "What changed between these two images?",
    },
]


def run_demo() -> None:
    base_dir = Path(__file__).resolve().parent
    out_dir = base_dir / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 75)
    print(" SATQUERY AI - CHANGE ANALYSIS SPECIALIST DEMONSTRATION")
    print("=" * 75)
    print(" Running end-to-end evaluation across all 4 benchmark cases...\n")

    overall_start = time.time()
    summary_table = []

    for idx, case in enumerate(DEMO_CASES, 1):
        case_id = case["id"]
        case_folder = base_dir / case["folder"]
        before_img = case_folder / "before.png"
        after_img = case_folder / "after.png"
        gt_img = case_folder / "ground_truth.png"

        print(f"\n[{idx}/4] {case['title']}")
        print(f"  • Query: \"{case['question']}\"")

        # Execute change analysis tool
        res = change_tool(
            before_image=before_img,
            after_image=after_img,
            question=case["question"],
            case_name=case_id
        )

        # Evaluate against ground truth if present
        eval_metrics = None
        if gt_img.exists():
            mask_path = None
            for ev in res.get("evidence", []):
                if ev.get("type") == "change_map":
                    mask_path = ev.get("path")
                    break
            if mask_path:
                eval_metrics = evaluate_change_prediction(mask_path, gt_img)
                res["evaluation"] = eval_metrics

        # Save structured JSON
        json_path = out_dir / f"{case_id}_result.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(res, f, indent=2)

        # Print per-case summary
        metrics = res.get("metrics", {})
        pct = metrics.get("changed_area_fraction", 0.0) * 100
        iou = eval_metrics.get("iou", 0.0) if eval_metrics else "N/A"
        f1 = eval_metrics.get("f1_score", 0.0) if eval_metrics else "N/A"

        print(f"  • Model       : {res.get('model')}")
        print(f"  • Answer      : {res.get('answer')}")
        print(f"  • Change Area : {pct:.2f}% ({metrics.get('changed_pixel_count', 0):,} pixels)")
        if eval_metrics:
            print(f"  • IoU Score   : {iou:.4f} | F1 Score: {f1:.4f} | Accuracy: {eval_metrics.get('overall_accuracy'):.4f}")

        comp_path = ""
        for ev in res.get("evidence", []):
            if ev.get("type") == "comparison":
                comp_path = ev.get("path")
                print(f"  • Visual Quad : {comp_path}")
                break

        summary_table.append({
            "case": case_id,
            "title": case["title"].split(":")[1].strip(),
            "pct": f"{pct:.2f}%",
            "dominant": metrics.get("dominant_change_type", "none").title(),
            "iou": f"{iou:.4f}" if isinstance(iou, float) else iou,
            "f1": f"{f1:.4f}" if isinstance(f1, float) else f1,
        })

    total_time = round(time.time() - overall_start, 2)

    # Print Executive Summary Table
    print("\n" + "=" * 75)
    print(f" DEMONSTRATION COMPLETE ({total_time}s total elapsed)")
    print("=" * 75)
    print(f"{'Case':<16} | {'Scenario':<28} | {'Change %':<9} | {'IoU':<7} | {'F1':<7}")
    print("-" * 75)
    for row in summary_table:
        print(f"{row['case']:<16} | {row['title']:<28} | {row['pct']:<9} | {row['iou']:<7} | {row['f1']:<7}")
    print("=" * 75)
    print(f"All visual figures and JSON results saved to: {out_dir}\n")


if __name__ == "__main__":
    run_demo()
