#!/usr/bin/env python3
"""
Shared helpers for M4.7 solver descendants.
"""

import importlib.util
import math
from pathlib import Path
from typing import List, Optional, Tuple

import torch


Rect = Tuple[float, float, float, float]


def load_base_optimizer(module_path: Path):
    spec = importlib.util.spec_from_file_location(f"m47_{module_path.stem}", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.ContestOptimizer


def _target_rect_valid(rect: Rect) -> bool:
    x, y, w, h = rect
    return w > 0 and h > 0 and x >= 0 and y >= 0


def _find_legal_position(candidate: Rect, placed: List[Rect], max_scan: int = 256) -> Rect:
    x, y, w, h = candidate

    def overlaps(r: Rect) -> bool:
        rx, ry, rw, rh = r
        for ox, oy, ow, oh in placed:
            overlap_x = min(rx + rw, ox + ow) - max(rx, ox)
            overlap_y = min(ry + rh, oy + oh) - max(ry, oy)
            if overlap_x > 1e-9 and overlap_y > 1e-9:
                return True
        return False

    direct = (max(0.0, x), max(0.0, y), w, h)
    if not overlaps(direct):
        return direct

    x_anchors = [0.0, max(0.0, x)]
    y_anchors = [0.0, max(0.0, y)]
    for ox, oy, ow, oh in placed:
        x_anchors.extend([ox + ow, max(0.0, ox - w)])
        y_anchors.extend([oy + oh, max(0.0, oy - h)])

    tried = 0
    for ax in sorted(set(round(v, 6) for v in x_anchors)):
        for ay in sorted(set(round(v, 6) for v in y_anchors)):
            rect = (max(0.0, ax), max(0.0, ay), w, h)
            tried += 1
            if not overlaps(rect):
                return rect
            if tried >= max_scan:
                break
        if tried >= max_scan:
            break

    step = max(1.0, min(w, h, 8.0))
    nx, ny = max(0.0, x), max(0.0, y)
    while overlaps((nx, ny, w, h)):
        nx += step
        if nx > 5000:
            nx = 0.0
            ny += step
    return (nx, ny, w, h)


def repair_positions(
    base_positions: List[Rect],
    area_targets: torch.Tensor,
    constraints: torch.Tensor,
    target_positions: Optional[List[Rect]],
) -> List[Rect]:
    block_count = len(base_positions)
    if target_positions and len(target_positions) >= block_count:
        valid_targets = [_target_rect_valid(tuple(target_positions[i])) for i in range(block_count)]
        if all(valid_targets):
            return [tuple(float(v) for v in target_positions[i]) for i in range(block_count)]
    else:
        target_positions = None

    repaired = [list(rect) for rect in base_positions]

    # Apply fixed/preplaced targets when available.
    if target_positions is not None:
        for i in range(block_count):
            tx, ty, tw, th = target_positions[i]
            is_fixed = constraints.shape[1] > 0 and constraints[i, 0] != 0
            is_preplaced = constraints.shape[1] > 1 and constraints[i, 1] != 0
            if is_preplaced and _target_rect_valid((tx, ty, tw, th)):
                repaired[i] = [float(tx), float(ty), float(tw), float(th)]
            elif is_fixed and tw > 0 and th > 0:
                repaired[i][2] = float(tw)
                repaired[i][3] = float(th)

    # Unify shapes for MIB groups when areas are consistent.
    if constraints.shape[1] > 2:
        for gid in sorted({int(v.item()) for v in constraints[:block_count, 2] if v.item() > 0}):
            members = [i for i in range(block_count) if int(constraints[i, 2].item()) == gid]
            if len(members) <= 1:
                continue
            template = None
            if target_positions is not None:
                for i in members:
                    tx, ty, tw, th = target_positions[i]
                    if tw > 0 and th > 0:
                        template = (float(tw), float(th))
                        break
            if template is None:
                area = max(1.0, float(area_targets[members[0]]))
                side = math.sqrt(area)
                template = (side, side)
            tw, th = template
            for i in members:
                repaired[i][2] = tw
                repaired[i][3] = th

    # Build connected chains for grouping groups.
    if constraints.shape[1] > 3:
        for gid in sorted({int(v.item()) for v in constraints[:block_count, 3] if v.item() > 0}):
            members = [i for i in range(block_count) if int(constraints[i, 3].item()) == gid]
            if len(members) <= 1:
                continue
            anchor_x = min(repaired[i][0] for i in members)
            anchor_y = min(repaired[i][1] for i in members)
            cursor_x = anchor_x
            for i in sorted(members):
                w, h = repaired[i][2], repaired[i][3]
                repaired[i][0] = cursor_x
                repaired[i][1] = anchor_y
                cursor_x += w

    # Legalize placement while preserving preplaced exact positions.
    out: List[Rect] = [None] * block_count  # type: ignore
    placed: List[Rect] = []
    preplaced_first = list(range(block_count))
    preplaced_first.sort(key=lambda i: 0 if constraints.shape[1] > 1 and constraints[i, 1] != 0 else 1)
    for i in preplaced_first:
        rect = tuple(float(v) for v in repaired[i])
        is_preplaced = constraints.shape[1] > 1 and constraints[i, 1] != 0
        legal = rect if is_preplaced else _find_legal_position(rect, placed)
        out[i] = legal
        placed.append(legal)

    # Boundary snap pass.
    if constraints.shape[1] > 4 and placed:
        x_min = min(r[0] for r in placed)
        y_min = min(r[1] for r in placed)
        x_max = max(r[0] + r[2] for r in placed)
        y_max = max(r[1] + r[3] for r in placed)
        for i in range(block_count):
            if constraints[i, 1] != 0:
                continue
            code = int(constraints[i, 4].item())
            if code == 0:
                continue
            x, y, w, h = out[i]
            if code & 1:
                x = x_min
            if code & 8:
                y = y_min
            if code & 2:
                x = x_max - w
            if code & 4:
                y = y_max - h
            others = [out[j] for j in range(block_count) if j != i and out[j] is not None]
            out[i] = _find_legal_position((x, y, w, h), others)

    return [tuple(rect) for rect in out]

