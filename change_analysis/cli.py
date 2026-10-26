"""
Command-Line Interface (CLI) for SatQuery AI Change Analysis.

Usage example:
    python -m change_analysis.cli --before examples/case_01/before.png --after examples/case_01/after.png --question "Has construction increased?"
"""

from __future__ import annotations
import argparse
import json
import sys
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="SatQuery AI - Autonomous Remote Sensing Change Analysis Module"
    )
    parser.add_argument("--before", "-b", required=True, help="Path to before (pre-change) image")
    parser.add_argument("--after", "-a", required=True, help="Path to after (post-change) image")
    parser.add_argument("--question", "-q", default="What changed between these two images?", help="Question for the model")
    parser.add_argument("--case-name", "-n", default=None, help="Optional case prefix for saved outputs")
    parser.add_argument("--ground-truth", "-gt", default=None, help="Optional ground truth change mask for evaluation")
    parser.add_argument("--json", action="store_true", help="Print raw JSON response only")

    args = parser.parse_args()

    # Execute change analysis tool
    result = change_tool(
        before_image=args.before,
        after_image=args.after,
        question=args.question,
        case_name=args.case_name
    )

    # Optional evaluation against ground truth
    eval_metrics = None
    if args.ground_truth:
        gt_path = Path(args.ground_truth)
        if gt_path.exists():
            # Find the generated change mask from evidence
            mask_path = None
            for item in result.get("evidence", []):
                if item.get("type") == "change_map":
                    mask_path = item.get("path")
                    break
            if mask_path:
                eval_metrics = evaluate_change_prediction(mask_path, gt_path)
                result["evaluation"] = eval_metrics

    if args.json:
        print(json.dumps(result, indent=2))
        return

    # Print Clean Human-Readable Summary
    print("\n" + "=" * 65)
    print(" SATQUERY AI - CHANGE ANALYSIS SPECIALIST REPORT")
    print("=" * 65)
    print(f" Task        : {result.get('task')}")
    print(f" Model Engine: {result.get('model')}")
    print(f" Confidence  : {result.get('confidence') if result.get('confidence') is not None else 'null (Uncalibrated / Deterministic)'}")
    print("-" * 65)
    print(f" User Query  : {args.question}")
    print(f" Answer      : {result.get('answer')}")
    print("-" * 65)

    metrics = result.get("metrics", {})
    if metrics:
        print(" Genuine Calculated Metrics:")
        print(f"  • Changed Area Fraction : {metrics.get('changed_area_fraction', 0.0) * 100:.2f}%")
        print(f"  • Changed Pixels Count  : {metrics.get('changed_pixel_count', 0):,} / {metrics.get('total_pixel_count', 0):,}")
        print(f"  • Dominant Signature    : {metrics.get('dominant_change_type', 'N/A').replace('_', ' ').title()}")
        if "mean_ssim_difference" in metrics:
            print(f"  • Mean Structural Delta : {metrics.get('mean_ssim_difference'):.4f}")

    if eval_metrics:
        print("\n Ground-Truth Evaluation Metrics:")
        print(f"  • IoU (Jaccard Index)   : {eval_metrics.get('iou'):.4f}")
        print(f"  • F1-Score (Dice)       : {eval_metrics.get('f1_score'):.4f}")
        print(f"  • Precision             : {eval_metrics.get('precision'):.4f}")
        print(f"  • Recall                : {eval_metrics.get('recall'):.4f}")
        print(f"  • Overall Accuracy      : {eval_metrics.get('overall_accuracy'):.4f}")

    evidence = result.get("evidence", [])
    if evidence:
        print("\n Generated Visual Evidence Artifacts:")
        for ev in evidence:
            print(f"  • [{ev.get('type').upper()}]: {ev.get('path')}")

    warnings = result.get("warnings", [])
    if warnings:
        print("\n Warnings:")
        for w in warnings:
            print(f"  [!] {w}")

    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
