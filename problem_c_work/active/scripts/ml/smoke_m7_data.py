#!/usr/bin/env python3
"""
M7 smoke test: verify training data loads and target rectangles can be extracted.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ACTIVE_ROOT = Path(__file__).resolve().parents[2]
WORK_ROOT = Path(__file__).resolve().parents[3]
REPO_ROOT = Path(__file__).resolve().parents[4]
CONTEST_DIR = ACTIVE_ROOT / "contest"
if str(CONTEST_DIR) not in sys.path:
    sys.path.insert(0, str(CONTEST_DIR))

from iccad2026_evaluate import get_training_dataloader


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-path",
        default=str(REPO_ROOT),
        help="Repository root containing LiteTensorData/ and LiteTensorDataTest/.",
    )
    parser.add_argument("--num-samples", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=1)
    args = parser.parse_args()

    dataloader = get_training_dataloader(
        data_path=args.data_path,
        batch_size=args.batch_size,
        num_samples=args.num_samples,
        shuffle=False,
    )

    summary = {"samples": []}
    for batch_idx, batch in enumerate(dataloader):
        area_target, b2b_conn, p2b_conn, pins_pos, constraints, tree_sol, fp_sol, metrics = batch
        area_target = area_target.squeeze(0)
        fp_sol = fp_sol.squeeze(0)
        metrics = metrics.squeeze(0)

        block_count = int((area_target != -1).sum().item())
        fp_sol = fp_sol[:block_count]
        first_rect = None
        if block_count > 0:
            first_rect = [
                float(fp_sol[0, 2].item()),
                float(fp_sol[0, 3].item()),
                float(fp_sol[0, 0].item()),
                float(fp_sol[0, 1].item()),
            ]
        summary["samples"].append(
            {
                "batch_idx": batch_idx,
                "block_count": block_count,
                "first_rect": first_rect,
                "target_area_sum": float(area_target[:block_count].sum().item()),
                "baseline_bbox_area": float(metrics[0].item()) if len(metrics) > 0 else None,
            }
        )

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
