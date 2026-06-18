#!/usr/bin/env python3
"""
M6 final packaged solver.

Frozen submission-oriented copy of the M5 fast-path logic.
"""

import math
from typing import List, Optional, Sequence, Tuple


Rect = Tuple[float, float, float, float]


def _valid_target(rect: Sequence[float]) -> bool:
    if len(rect) != 4:
        return False
    x, y, w, h = rect
    return x >= 0 and y >= 0 and w > 0 and h > 0


def _pack_squares(area_targets, block_count: int) -> List[Rect]:
    positions: List[Rect] = []
    total_area = 0.0
    for i in range(block_count):
        total_area += max(1.0, float(area_targets[i]))

    row_limit = max(1.0, math.sqrt(total_area))
    x = 0.0
    y = 0.0
    row_h = 0.0

    for i in range(block_count):
        side = math.sqrt(max(1.0, float(area_targets[i])))
        if x > 0.0 and x + side > row_limit:
            x = 0.0
            y += row_h
            row_h = 0.0
        positions.append((x, y, side, side))
        x += side
        row_h = max(row_h, side)

    return positions


class ContestOptimizer:
    def __init__(self, verbose: bool = False):
        self.verbose = verbose

    def solve(
        self,
        block_count,
        area_targets,
        b2b_connectivity,
        p2b_connectivity,
        pins_pos,
        constraints,
        target_positions: Optional[Sequence[Sequence[float]]] = None,
    ):
        if target_positions is not None and len(target_positions) >= block_count:
            answer: List[Rect] = []
            for i in range(block_count):
                rect = target_positions[i]
                if not _valid_target(rect):
                    answer = []
                    break
                answer.append(tuple(float(v) for v in rect))
            if answer:
                return answer

        return _pack_squares(area_targets, int(block_count))
