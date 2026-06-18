#!/usr/bin/env python3
"""
M4.7 main solver based on the M4 milestone with target_positions support.
"""

from pathlib import Path
import sys

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from common_solver import load_base_optimizer, repair_positions


BASE = load_base_optimizer(Path(__file__).resolve().parents[1] / "legacy" / "solver_m4.py")


class ContestOptimizer(BASE):
    def solve(self, block_count, area_targets, b2b_connectivity, p2b_connectivity, pins_pos, constraints, target_positions=None):
        base_positions = super().solve(block_count, area_targets, b2b_connectivity, p2b_connectivity, pins_pos, constraints)
        return repair_positions(base_positions, area_targets, constraints, target_positions)
