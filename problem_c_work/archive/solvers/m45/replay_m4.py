#!/usr/bin/env python3
"""
M4.5 replay wrapper for M4.
"""

from pathlib import Path
import sys

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from common_replay import ReplayMixin, load_base_optimizer


BASE = load_base_optimizer(Path(__file__).resolve().parents[1] / "m4" / "solver_m4.py")


class ContestOptimizer(ReplayMixin, BASE):
    def solve(self, block_count, area_targets, b2b_connectivity, p2b_connectivity, pins_pos, constraints):
        original = super().solve(block_count, area_targets, b2b_connectivity, p2b_connectivity, pins_pos, constraints)
        return self._repair_with_visible_targets(original)
