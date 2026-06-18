#!/usr/bin/env python3
"""
Deterministic constructive solver for ICCAD 2026 Problem C.

This version stays within the simplified contest interface:
- exact area preservation
- overlap-free rectangular placement
- connectivity-guided initial targets
- corner-based legalization
- lightweight compaction
"""

import math
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import torch

CONTEST_DIR = Path(__file__).resolve().parent.parent / "contest"
if str(CONTEST_DIR) not in sys.path:
    sys.path.insert(0, str(CONTEST_DIR))

import iccad2026_evaluate as contest


Rect = Tuple[float, float, float, float]
Point = Tuple[float, float]


def _valid_edges(edges: torch.Tensor) -> List[Tuple[int, int, float]]:
    if edges is None:
        return []
    out: List[Tuple[int, int, float]] = []
    for edge in edges:
        if int(edge[0]) == -1:
            continue
        out.append((int(edge[0]), int(edge[1]), float(edge[2])))
    return out


def _rect_center(rect: Rect) -> Point:
    return (rect[0] + rect[2] / 2.0, rect[1] + rect[3] / 2.0)


def _overlaps(candidate: Rect, placed: Sequence[Rect], tol: float = 1e-9) -> bool:
    x, y, w, h = candidate
    for ox, oy, ow, oh in placed:
        overlap_x = min(x + w, ox + ow) - max(x, ox)
        overlap_y = min(y + h, oy + oh) - max(y, oy)
        if overlap_x > tol and overlap_y > tol:
            return True
    return False


def _bbox_after_add(placed: Sequence[Rect], candidate: Rect) -> Tuple[float, float]:
    if not placed:
        return candidate[2], candidate[3]
    x_min = min(min(r[0] for r in placed), candidate[0])
    y_min = min(min(r[1] for r in placed), candidate[1])
    x_max = max(max(r[0] + r[2] for r in placed), candidate[0] + candidate[2])
    y_max = max(max(r[1] + r[3] for r in placed), candidate[1] + candidate[3])
    return x_max - x_min, y_max - y_min


def _partial_hpwl(
    block_id: int,
    candidate: Rect,
    placed_map: Dict[int, Rect],
    b2b_by_block: Sequence[List[Tuple[int, float]]],
    p2b_by_block: Sequence[List[Tuple[float, float, float]]],
) -> float:
    cx, cy = _rect_center(candidate)
    cost = 0.0
    for other, weight in b2b_by_block[block_id]:
        if other not in placed_map:
            continue
        ocx, ocy = _rect_center(placed_map[other])
        cost += weight * (abs(cx - ocx) + abs(cy - ocy))
    for px, py, weight in p2b_by_block[block_id]:
        cost += weight * (abs(cx - px) + abs(cy - py))
    return cost


class ContestOptimizer(contest.FloorplanOptimizer):
    """
    Deterministic constructive placer with a cheap improvement pass.
    """

    def __init__(self, verbose: bool = False):
        super().__init__(verbose=verbose)
        self.force_iters = 50
        self.compaction_rounds = 2

    def solve(
        self,
        block_count: int,
        area_targets: torch.Tensor,
        b2b_connectivity: torch.Tensor,
        p2b_connectivity: torch.Tensor,
        pins_pos: torch.Tensor,
        constraints: torch.Tensor,
    ) -> List[Rect]:
        areas = [max(1.0, float(area_targets[i])) for i in range(block_count)]
        widths = [math.sqrt(a) for a in areas]
        heights = [a / w for a, w in zip(areas, widths)]

        b2b_edges = _valid_edges(b2b_connectivity)
        p2b_edges = _valid_edges(p2b_connectivity)

        b2b_by_block: List[List[Tuple[int, float]]] = [[] for _ in range(block_count)]
        for i, j, weight in b2b_edges:
            if i < block_count and j < block_count:
                b2b_by_block[i].append((j, weight))
                b2b_by_block[j].append((i, weight))

        p2b_by_block: List[List[Tuple[float, float, float]]] = [[] for _ in range(block_count)]
        for pin_idx, block_idx, weight in p2b_edges:
            if block_idx < block_count and pin_idx < len(pins_pos):
                px = float(pins_pos[pin_idx][0])
                py = float(pins_pos[pin_idx][1])
                p2b_by_block[block_idx].append((px, py, weight))

        target_centers = self._compute_target_centers(
            block_count,
            widths,
            heights,
            b2b_by_block,
            p2b_by_block,
        )

        order = list(range(block_count))
        order.sort(
            key=lambda idx: (
                -(sum(w for _, w in b2b_by_block[idx]) + sum(w for _, _, w in p2b_by_block[idx])),
                -areas[idx],
                idx,
            )
        )

        placed_map: Dict[int, Rect] = {}
        frontier: List[Point] = [(0.0, 0.0)]
        placed: List[Rect] = []

        for block_id in order:
            rect = self._place_block(
                block_id,
                widths[block_id],
                heights[block_id],
                target_centers[block_id],
                frontier,
                placed,
                placed_map,
                b2b_by_block,
                p2b_by_block,
            )
            placed_map[block_id] = rect
            placed.append(rect)
            frontier.append((rect[0] + rect[2], rect[1]))
            frontier.append((rect[0], rect[1] + rect[3]))
            frontier = self._dedupe_points(frontier)

        positions = [placed_map[i] for i in range(block_count)]
        positions = self._compact(positions)
        return positions

    def _compute_target_centers(
        self,
        block_count: int,
        widths: Sequence[float],
        heights: Sequence[float],
        b2b_by_block: Sequence[List[Tuple[int, float]]],
        p2b_by_block: Sequence[List[Tuple[float, float, float]]],
    ) -> List[Point]:
        total_area = sum(w * h for w, h in zip(widths, heights))
        radius = max(10.0, math.sqrt(total_area))

        centers: List[List[float]] = []
        for idx in range(block_count):
            pin_weight = sum(weight for _, _, weight in p2b_by_block[idx])
            if pin_weight > 0:
                tx = sum(px * weight for px, _, weight in p2b_by_block[idx]) / pin_weight
                ty = sum(py * weight for _, py, weight in p2b_by_block[idx]) / pin_weight
            else:
                angle = 2.0 * math.pi * idx / max(block_count, 1)
                tx = radius * (1.0 + 0.35 * math.cos(angle))
                ty = radius * (1.0 + 0.35 * math.sin(angle))
            centers.append([tx, ty])

        ideal_gap = max(1.0, math.sqrt(total_area / max(block_count, 1)) * 0.85)
        for _ in range(self.force_iters):
            new_centers = [[0.0, 0.0] for _ in range(block_count)]
            for i in range(block_count):
                weight_sum = 1.0
                ax = centers[i][0]
                ay = centers[i][1]

                pin_weight = sum(weight for _, _, weight in p2b_by_block[i])
                if pin_weight > 0:
                    ax += sum(px * weight for px, _, weight in p2b_by_block[i])
                    ay += sum(py * weight for _, py, weight in p2b_by_block[i])
                    weight_sum += pin_weight

                for nbr, weight in b2b_by_block[i]:
                    ax += centers[nbr][0] * weight
                    ay += centers[nbr][1] * weight
                    weight_sum += weight

                x = ax / weight_sum
                y = ay / weight_sum

                rx = 0.0
                ry = 0.0
                for j in range(block_count):
                    if i == j:
                        continue
                    dx = centers[i][0] - centers[j][0]
                    dy = centers[i][1] - centers[j][1]
                    dist2 = dx * dx + dy * dy + 1.0
                    force = (ideal_gap * ideal_gap) / dist2
                    rx += dx * force * 0.015
                    ry += dy * force * 0.015

                new_centers[i][0] = x + rx
                new_centers[i][1] = y + ry
            centers = new_centers

        min_x = min(c[0] - widths[i] / 2.0 for i, c in enumerate(centers))
        min_y = min(c[1] - heights[i] / 2.0 for i, c in enumerate(centers))
        shift_x = -min_x if min_x < 0 else 0.0
        shift_y = -min_y if min_y < 0 else 0.0
        return [(c[0] + shift_x, c[1] + shift_y) for c in centers]

    def _place_block(
        self,
        block_id: int,
        width: float,
        height: float,
        target_center: Point,
        frontier: Sequence[Point],
        placed: Sequence[Rect],
        placed_map: Dict[int, Rect],
        b2b_by_block: Sequence[List[Tuple[int, float]]],
        p2b_by_block: Sequence[List[Tuple[float, float, float]]],
    ) -> Rect:
        tx = max(0.0, target_center[0] - width / 2.0)
        ty = max(0.0, target_center[1] - height / 2.0)

        candidates: List[Point] = [(0.0, 0.0), (tx, ty)]
        candidates.extend(frontier)
        for ox, oy, ow, oh in placed:
            candidates.append((ox + ow, oy))
            candidates.append((ox, oy + oh))
            candidates.append((max(0.0, tx), oy))
            candidates.append((ox, max(0.0, ty)))
        candidates = self._dedupe_points(candidates)

        best_rect: Rect | None = None
        best_score = float("inf")
        for x, y in candidates:
            rect = (max(0.0, x), max(0.0, y), width, height)
            if _overlaps(rect, placed):
                continue

            bbox_w, bbox_h = _bbox_after_add(placed, rect)
            cx, cy = _rect_center(rect)
            target_cost = abs(cx - target_center[0]) + abs(cy - target_center[1])
            net_cost = _partial_hpwl(block_id, rect, placed_map, b2b_by_block, p2b_by_block)
            score = target_cost + 0.35 * net_cost + 0.08 * bbox_w * bbox_h
            if score < best_score:
                best_score = score
                best_rect = rect

        if best_rect is not None:
            return best_rect

        # Fallback: scan upward from the current bounding box frontier.
        if not placed:
            return (0.0, 0.0, width, height)
        bbox_w, bbox_h = _bbox_after_add(placed, (0.0, 0.0, 0.0, 0.0))
        trial_points = [(bbox_w, 0.0), (0.0, bbox_h)]
        for x, y in trial_points:
            rect = (x, y, width, height)
            if not _overlaps(rect, placed):
                return rect
        return (bbox_w, bbox_h, width, height)

    def _compact(self, positions: List[Rect]) -> List[Rect]:
        rects = list(positions)
        for _ in range(self.compaction_rounds):
            order_x = sorted(range(len(rects)), key=lambda idx: (rects[idx][0], rects[idx][1], idx))
            for idx in order_x:
                rects[idx] = self._shift_axis(rects, idx, axis=0)

            order_y = sorted(range(len(rects)), key=lambda idx: (rects[idx][1], rects[idx][0], idx))
            for idx in order_y:
                rects[idx] = self._shift_axis(rects, idx, axis=1)
        return rects

    def _shift_axis(self, rects: List[Rect], idx: int, axis: int) -> Rect:
        x, y, w, h = rects[idx]
        best = x if axis == 0 else y
        for other_idx, (ox, oy, ow, oh) in enumerate(rects):
            if other_idx == idx:
                continue
            if axis == 0:
                overlap_y = min(y + h, oy + oh) - max(y, oy)
                if overlap_y > 1e-9 and ox + ow <= x + 1e-9:
                    best = max(best if best < x else 0.0, ox + ow)
            else:
                overlap_x = min(x + w, ox + ow) - max(x, ox)
                if overlap_x > 1e-9 and oy + oh <= y + 1e-9:
                    best = max(best if best < y else 0.0, oy + oh)

        if axis == 0:
            candidate = (best if best < x else 0.0, y, w, h)
        else:
            candidate = (x, best if best < y else 0.0, w, h)

        others = [rect for j, rect in enumerate(rects) if j != idx]
        if not _overlaps(candidate, others):
            return candidate
        return rects[idx]

    def _dedupe_points(self, points: Sequence[Point]) -> List[Point]:
        seen = set()
        out: List[Point] = []
        for x, y in points:
            key = (round(max(0.0, x), 6), round(max(0.0, y), 6))
            if key in seen:
                continue
            seen.add(key)
            out.append((key[0], key[1]))
        return out
