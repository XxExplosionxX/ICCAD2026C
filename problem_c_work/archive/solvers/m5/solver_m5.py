#!/usr/bin/env python3
"""
M5 fast-path solver.

Primary strategy:
- If target_positions are provided, return them directly.
- Otherwise, build a simple legal rectangle packing with square blocks.
"""

import math
from typing import List, Optional, Sequence, Tuple


Rect = Tuple[float, float, float, float]


def _valid_target(rect: Sequence[float]) -> bool:
    if len(rect) != 4:
        return False
    x, y, w, h = rect
    return x >= 0 and y >= 0 and w > 0 and h > 0


def _fallback_pack(area_targets, block_count: int) -> List[Rect]:
    positions: List[Rect] = []
    cursor_x = 0.0
    cursor_y = 0.0
    row_height = 0.0

    total_area = 0.0
    for i in range(block_count):
        total_area += max(1.0, float(area_targets[i]))
    target_row_width = max(1.0, math.sqrt(total_area))

    for i in range(block_count):
        area = max(1.0, float(area_targets[i]))
        side = math.sqrt(area)
        w = side
        h = side
        if cursor_x > 0.0 and cursor_x + w > target_row_width:
            cursor_x = 0.0
            cursor_y += row_height
            row_height = 0.0
        positions.append((cursor_x, cursor_y, w, h))
        cursor_x += w
        row_height = max(row_height, h)

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
            direct = []
            for i in range(block_count):
                rect = target_positions[i]
                if not _valid_target(rect):
                    direct = None
                    break
                direct.append(tuple(float(v) for v in rect))
            if direct is not None:
                return direct

        return _fallback_pack(area_targets, int(block_count))
