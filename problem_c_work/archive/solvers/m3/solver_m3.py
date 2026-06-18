#!/usr/bin/env python3
"""
M3 solver for ICCAD 2026 Problem C.

This is an isolated evolution of M2:
- preserves exact area legality
- keeps discrete aspect-ratio search but prunes risky extremes
- strengthens constructive bbox control
- uses adaptive refinement gating
- adds targeted pair repair for strongly connected blocks
"""

import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch

CONTEST_DIR = Path(__file__).resolve().parents[2] / "contest"
if str(CONTEST_DIR) not in sys.path:
    sys.path.insert(0, str(CONTEST_DIR))

import iccad2026_evaluate as contest


Rect = Tuple[float, float, float, float]
Point = Tuple[float, float]


def _valid_edges(edges: torch.Tensor) -> List[Tuple[int, int, float]]:
    out: List[Tuple[int, int, float]] = []
    if edges is None:
        return out
    for edge in edges:
        if int(edge[0]) == -1:
            continue
        out.append((int(edge[0]), int(edge[1]), float(edge[2])))
    return out


def _rect_center(rect: Rect) -> Point:
    return (rect[0] + rect[2] / 2.0, rect[1] + rect[3] / 2.0)


def _overlaps(candidate: Rect, placed: Sequence[Rect], skip_idx: Optional[int] = None, tol: float = 1e-9) -> bool:
    x, y, w, h = candidate
    for idx, (ox, oy, ow, oh) in enumerate(placed):
        if skip_idx is not None and idx == skip_idx:
            continue
        overlap_x = min(x + w, ox + ow) - max(x, ox)
        overlap_y = min(y + h, oy + oh) - max(y, oy)
        if overlap_x > tol and overlap_y > tol:
            return True
    return False


def _bbox_of_positions(positions: Sequence[Rect]) -> Tuple[float, float, float]:
    if not positions:
        return 0.0, 0.0, 0.0
    x_min = min(r[0] for r in positions)
    y_min = min(r[1] for r in positions)
    x_max = max(r[0] + r[2] for r in positions)
    y_max = max(r[1] + r[3] for r in positions)
    width = x_max - x_min
    height = y_max - y_min
    return width, height, width * height


class ContestOptimizer(contest.FloorplanOptimizer):
    def __init__(self, verbose: bool = False):
        super().__init__(verbose=verbose)
        self.force_iters = 60
        self.compaction_rounds = 3
        self.aspect_ratios = (1.0, 1.25, 1.5, 2.0, 3.0)

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

        degrees = [
            sum(weight for _, weight in b2b_by_block[i]) + sum(weight for _, _, weight in p2b_by_block[i])
            for i in range(block_count)
        ]

        target_centers = self._compute_target_centers(block_count, areas, b2b_by_block, p2b_by_block)
        shape_candidates = self._build_shape_candidates(areas, target_centers, b2b_by_block, p2b_by_block)

        order = list(range(block_count))
        order.sort(key=lambda idx: (-degrees[idx], -areas[idx], idx))

        positions = self._construct_layout(
            order,
            target_centers,
            shape_candidates,
            b2b_by_block,
            p2b_by_block,
        )
        positions = self._compact(positions)
        base_positions = list(positions)
        refinement_profile = self._choose_refinement_profile(
            base_positions,
            block_count,
            degrees,
            b2b_by_block,
            p2b_by_block,
        )

        candidates = [base_positions]

        m2_positions = self._refine_layout_m2(
            list(base_positions),
            target_centers,
            shape_candidates,
            degrees,
            b2b_by_block,
            p2b_by_block,
        )
        m2_positions = self._compact(m2_positions)
        candidates.append(m2_positions)

        if refinement_profile["mode"] != "skip":
            adaptive_positions = self._refine_layout(
                list(base_positions),
                target_centers,
                shape_candidates,
                degrees,
                b2b_by_block,
                p2b_by_block,
                refinement_profile,
            )
            adaptive_positions = self._compact(adaptive_positions)
            candidates.append(adaptive_positions)

        positions = min(
            candidates,
            key=lambda cand: self._proxy_cost_full(cand, b2b_by_block, p2b_by_block),
        )
        positions = self._compact(positions)
        return positions

    def _compute_target_centers(
        self,
        block_count: int,
        areas: Sequence[float],
        b2b_by_block: Sequence[List[Tuple[int, float]]],
        p2b_by_block: Sequence[List[Tuple[float, float, float]]],
    ) -> List[Point]:
        total_area = sum(areas)
        base_gap = max(4.0, math.sqrt(total_area / max(block_count, 1)))
        radius = max(12.0, math.sqrt(total_area))

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

        for _ in range(self.force_iters):
            new_centers = [[0.0, 0.0] for _ in range(block_count)]
            for i in range(block_count):
                ax = centers[i][0]
                ay = centers[i][1]
                weight_sum = 1.0

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
                    repel = (base_gap * base_gap) / dist2
                    rx += dx * repel * 0.02
                    ry += dy * repel * 0.02

                new_centers[i][0] = x + rx
                new_centers[i][1] = y + ry
            centers = new_centers

        min_x = min(c[0] for c in centers)
        min_y = min(c[1] for c in centers)
        shift_x = -min_x if min_x < 0 else 0.0
        shift_y = -min_y if min_y < 0 else 0.0
        return [(c[0] + shift_x, c[1] + shift_y) for c in centers]

    def _build_shape_candidates(
        self,
        areas: Sequence[float],
        target_centers: Sequence[Point],
        b2b_by_block: Sequence[List[Tuple[int, float]]],
        p2b_by_block: Sequence[List[Tuple[float, float, float]]],
    ) -> List[List[Tuple[float, float]]]:
        candidates: List[List[Tuple[float, float]]] = []
        for idx, area in enumerate(areas):
            cx, cy = target_centers[idx]
            spread_x = 1.0
            spread_y = 1.0

            for nbr, weight in b2b_by_block[idx]:
                nx, ny = target_centers[nbr]
                spread_x += abs(nx - cx) * weight
                spread_y += abs(ny - cy) * weight

            for px, py, weight in p2b_by_block[idx]:
                spread_x += abs(px - cx) * weight
                spread_y += abs(py - cy) * weight

            preferred_ratio = max(1.0 / 3.0, min(3.0, spread_x / max(spread_y, 1e-6)))
            ordered_ratios = sorted(
                {ratio for base in self.aspect_ratios for ratio in (base, 1.0 / base)},
                key=lambda ratio: abs(math.log(ratio) - math.log(preferred_ratio)),
            )

            seen = set()
            block_shapes: List[Tuple[float, float]] = []
            for ratio in ordered_ratios[:5]:
                w = math.sqrt(area * ratio)
                h = area / w
                key = (round(w, 6), round(h, 6))
                if key in seen:
                    continue
                seen.add(key)
                block_shapes.append((w, h))
            candidates.append(block_shapes)
        return candidates

    def _construct_layout(
        self,
        order: Sequence[int],
        target_centers: Sequence[Point],
        shape_candidates: Sequence[List[Tuple[float, float]]],
        b2b_by_block: Sequence[List[Tuple[int, float]]],
        p2b_by_block: Sequence[List[Tuple[float, float, float]]],
    ) -> List[Rect]:
        positions: List[Optional[Rect]] = [None] * len(order)
        placed_map: Dict[int, Rect] = {}
        frontier: List[Point] = [(0.0, 0.0)]

        for block_id in order:
            rect = self._best_placement_for_block(
                block_id,
                positions,
                placed_map,
                frontier,
                target_centers,
                shape_candidates[block_id],
                b2b_by_block,
                p2b_by_block,
            )
            positions[block_id] = rect
            placed_map[block_id] = rect
            frontier.extend(
                [
                    (rect[0] + rect[2], rect[1]),
                    (rect[0], rect[1] + rect[3]),
                    (rect[0] + rect[2], rect[1] + rect[3]),
                ]
            )
            frontier = self._dedupe_points(frontier)

        return [rect for rect in positions if rect is not None]

    def _best_placement_for_block(
        self,
        block_id: int,
        positions: Sequence[Optional[Rect]],
        placed_map: Dict[int, Rect],
        frontier: Sequence[Point],
        target_centers: Sequence[Point],
        block_shapes: Sequence[Tuple[float, float]],
        b2b_by_block: Sequence[List[Tuple[int, float]]],
        p2b_by_block: Sequence[List[Tuple[float, float, float]]],
    ) -> Rect:
        current_positions = [rect for rect in positions if rect is not None]
        best_rect: Optional[Rect] = None
        best_score = float("inf")

        for width, height in block_shapes:
            anchors = self._generate_anchors(
                block_id,
                width,
                height,
                frontier,
                placed_map,
                target_centers,
                b2b_by_block,
            )
            for x, y in anchors:
                rect = (x, y, width, height)
                if _overlaps(rect, current_positions):
                    continue
                score = self._proxy_cost_for_candidate(
                    block_id,
                    rect,
                    current_positions,
                    placed_map,
                    target_centers[block_id],
                    b2b_by_block,
                    p2b_by_block,
                )
                if score < best_score:
                    best_score = score
                    best_rect = rect

        if best_rect is not None:
            return best_rect

        # Conservative fallback.
        width, height = block_shapes[0]
        if not current_positions:
            return (0.0, 0.0, width, height)
        bbox_w, bbox_h, _ = _bbox_of_positions(current_positions)
        fallback = (bbox_w, 0.0, width, height)
        if not _overlaps(fallback, current_positions):
            return fallback
        return (0.0, bbox_h, width, height)

    def _generate_anchors(
        self,
        block_id: int,
        width: float,
        height: float,
        frontier: Sequence[Point],
        placed_map: Dict[int, Rect],
        target_centers: Sequence[Point],
        b2b_by_block: Sequence[List[Tuple[int, float]]],
    ) -> List[Point]:
        target_x = max(0.0, target_centers[block_id][0] - width / 2.0)
        target_y = max(0.0, target_centers[block_id][1] - height / 2.0)

        anchors: List[Point] = [(0.0, 0.0), (target_x, target_y)]
        anchors.extend(frontier)

        for nbr, weight in sorted(b2b_by_block[block_id], key=lambda item: (-item[1], item[0]))[:8]:
            if nbr not in placed_map:
                continue
            ox, oy, ow, oh = placed_map[nbr]
            anchors.extend(
                [
                    (ox + ow, oy),
                    (ox - width, oy),
                    (ox, oy + oh),
                    (ox, oy - height),
                    (ox + ow, oy + oh - height),
                    (ox + ow - width, oy + oh),
                ]
            )

        return self._dedupe_points(anchors)

    def _proxy_cost_for_candidate(
        self,
        block_id: int,
        rect: Rect,
        current_positions: Sequence[Rect],
        placed_map: Dict[int, Rect],
        target_center: Point,
        b2b_by_block: Sequence[List[Tuple[int, float]]],
        p2b_by_block: Sequence[List[Tuple[float, float, float]]],
    ) -> float:
        cx, cy = _rect_center(rect)
        target_penalty = abs(cx - target_center[0]) + abs(cy - target_center[1])

        partial_hpwl = 0.0
        for nbr, weight in b2b_by_block[block_id]:
            if nbr not in placed_map:
                continue
            nx, ny = _rect_center(placed_map[nbr])
            partial_hpwl += weight * (abs(cx - nx) + abs(cy - ny))

        pin_cost = 0.0
        for px, py, weight in p2b_by_block[block_id]:
            pin_cost += weight * (abs(cx - px) + abs(cy - py))

        bbox_w, bbox_h, bbox_area = _bbox_of_positions(list(current_positions) + [rect])
        density_penalty = 0.0
        for ox, oy, ow, oh in current_positions:
            gap_x = max(0.0, max(ox - (rect[0] + rect[2]), rect[0] - (ox + ow)))
            gap_y = max(0.0, max(oy - (rect[1] + rect[3]), rect[1] - (oy + oh)))
            manhattan_gap = gap_x + gap_y
            if manhattan_gap < 6.0:
                density_penalty += 6.0 - manhattan_gap

        return partial_hpwl + pin_cost + 0.06 * bbox_area + 0.1 * (bbox_w + bbox_h) + 0.35 * target_penalty + 0.8 * density_penalty

    def _proxy_cost_full(
        self,
        positions: Sequence[Rect],
        b2b_by_block: Sequence[List[Tuple[int, float]]],
        p2b_by_block: Sequence[List[Tuple[float, float, float]]],
    ) -> float:
        hpwl_b2b = 0.0
        for i, rect in enumerate(positions):
            cx, cy = _rect_center(rect)
            for nbr, weight in b2b_by_block[i]:
                if nbr > i:
                    nx, ny = _rect_center(positions[nbr])
                    hpwl_b2b += weight * (abs(cx - nx) + abs(cy - ny))

        hpwl_p2b = 0.0
        for i, rect in enumerate(positions):
            cx, cy = _rect_center(rect)
            for px, py, weight in p2b_by_block[i]:
                hpwl_p2b += weight * (abs(cx - px) + abs(cy - py))

        _, _, bbox_area = _bbox_of_positions(positions)
        return hpwl_b2b + hpwl_p2b + 0.065 * bbox_area

    def _choose_refinement_profile(
        self,
        positions: Sequence[Rect],
        block_count: int,
        degrees: Sequence[float],
        b2b_by_block: Sequence[List[Tuple[int, float]]],
        p2b_by_block: Sequence[List[Tuple[float, float, float]]],
    ) -> Dict[str, float]:
        total_cost = self._proxy_cost_full(positions, b2b_by_block, p2b_by_block)
        locals_ = [self._local_cost(i, positions, b2b_by_block, p2b_by_block) for i in range(block_count)]
        avg_local = sum(locals_) / max(block_count, 1)
        peak_local = max(locals_) if locals_ else 0.0
        _, _, bbox_area = _bbox_of_positions(positions)
        total_area = sum(rect[2] * rect[3] for rect in positions)
        area_ratio = bbox_area / max(total_area, 1e-6)
        degree_scale = sum(degrees) / max(block_count, 1)
        risk = 0.55 * area_ratio + 0.25 * (peak_local / max(avg_local, 1e-6)) + 0.20 * (total_cost / max(degree_scale * 20.0, 1.0))

        if block_count <= 40:
            if risk < 2.0:
                return {"mode": "skip", "rounds": 0, "blocks": 0, "pairs": 0}
            if risk < 2.7:
                return {"mode": "light", "rounds": 1, "blocks": min(block_count, 6), "pairs": 0}
            return {"mode": "medium", "rounds": 2, "blocks": min(block_count, 8), "pairs": 0}
        if block_count <= 80:
            if risk < 2.1:
                return {"mode": "skip", "rounds": 0, "blocks": 0, "pairs": 0}
            if risk < 2.8:
                return {"mode": "light", "rounds": 1, "blocks": 5, "pairs": 0}
            return {"mode": "medium", "rounds": 2, "blocks": 7, "pairs": 0}
        if risk < 2.2:
            return {"mode": "skip", "rounds": 0, "blocks": 0, "pairs": 0}
        if risk < 2.9:
            return {"mode": "light", "rounds": 1, "blocks": 4, "pairs": 0}
        return {"mode": "medium", "rounds": 2, "blocks": 5, "pairs": 0}

    def _refine_layout(
        self,
        positions: List[Rect],
        target_centers: Sequence[Point],
        shape_candidates: Sequence[List[Tuple[float, float]]],
        degrees: Sequence[float],
        b2b_by_block: Sequence[List[Tuple[int, float]]],
        p2b_by_block: Sequence[List[Tuple[float, float, float]]],
        refinement_profile: Dict[str, float],
        ) -> List[Rect]:
        n = len(positions)
        if n <= 1:
            return positions

        rounds = int(refinement_profile["rounds"])
        blocks_per_round = int(refinement_profile["blocks"])
        pair_budget = int(refinement_profile["pairs"])

        current = list(positions)
        current_cost = self._proxy_cost_full(current, b2b_by_block, p2b_by_block)

        for _ in range(rounds):
            block_order = sorted(range(n), key=lambda idx: (-self._local_cost(idx, current, b2b_by_block, p2b_by_block), -degrees[idx], idx))
            improved_any = False
            for block_id in block_order[:blocks_per_round]:
                candidate = self._refine_block(
                    block_id,
                    current,
                    target_centers,
                    shape_candidates,
                    b2b_by_block,
                    p2b_by_block,
                )
                if candidate is None:
                    continue
                new_positions = list(current)
                new_positions[block_id] = candidate
                new_positions = self._compact(new_positions)
                new_cost = self._proxy_cost_full(new_positions, b2b_by_block, p2b_by_block)
                if new_cost + 1e-6 < current_cost:
                    current = new_positions
                    current_cost = new_cost
                    improved_any = True
            if pair_budget > 0:
                current, current_cost, pair_improved = self._pair_repair(
                    current,
                    target_centers,
                    shape_candidates,
                    b2b_by_block,
                    p2b_by_block,
                    pair_budget,
                    current_cost,
                )
                improved_any = improved_any or pair_improved
            if not improved_any:
                break
        return current

    def _refine_layout_m2(
        self,
        positions: List[Rect],
        target_centers: Sequence[Point],
        shape_candidates: Sequence[List[Tuple[float, float]]],
        degrees: Sequence[float],
        b2b_by_block: Sequence[List[Tuple[int, float]]],
        p2b_by_block: Sequence[List[Tuple[float, float, float]]],
    ) -> List[Rect]:
        n = len(positions)
        if n <= 1:
            return positions

        if n <= 40:
            rounds = 3
            blocks_per_round = min(n, 14)
        elif n <= 80:
            rounds = 2
            blocks_per_round = 10
        else:
            rounds = 2
            blocks_per_round = 7

        current = list(positions)
        current_cost = self._proxy_cost_full(current, b2b_by_block, p2b_by_block)

        for _ in range(rounds):
            block_order = sorted(range(n), key=lambda idx: (-self._local_cost(idx, current, b2b_by_block, p2b_by_block), -degrees[idx], idx))
            improved_any = False
            for block_id in block_order[:blocks_per_round]:
                candidate = self._refine_block(
                    block_id,
                    current,
                    target_centers,
                    shape_candidates,
                    b2b_by_block,
                    p2b_by_block,
                )
                if candidate is None:
                    continue
                new_positions = list(current)
                new_positions[block_id] = candidate
                new_positions = self._compact(new_positions)
                new_cost = self._proxy_cost_full(new_positions, b2b_by_block, p2b_by_block)
                if new_cost + 1e-6 < current_cost:
                    current = new_positions
                    current_cost = new_cost
                    improved_any = True
            if not improved_any:
                break
        return current

    def _pair_repair(
        self,
        positions: List[Rect],
        target_centers: Sequence[Point],
        shape_candidates: Sequence[List[Tuple[float, float]]],
        b2b_by_block: Sequence[List[Tuple[int, float]]],
        p2b_by_block: Sequence[List[Tuple[float, float, float]]],
        pair_budget: int,
        current_cost: float,
    ) -> Tuple[List[Rect], float, bool]:
        pairs: List[Tuple[float, int, int]] = []
        for i, nbrs in enumerate(b2b_by_block):
            for j, weight in nbrs:
                if j > i:
                    pairs.append((weight, i, j))
        pairs.sort(reverse=True)

        current = list(positions)
        improved = False
        for _, i, j in pairs[:pair_budget * 3]:
            trial = self._attempt_pair_swap(current, i, j, target_centers, shape_candidates)
            if trial is None:
                continue
            trial = self._compact(trial)
            trial_cost = self._proxy_cost_full(trial, b2b_by_block, p2b_by_block)
            if trial_cost + 1e-6 < current_cost:
                current = trial
                current_cost = trial_cost
                improved = True
                pair_budget -= 1
                if pair_budget <= 0:
                    break
        return current, current_cost, improved

    def _attempt_pair_swap(
        self,
        positions: Sequence[Rect],
        i: int,
        j: int,
        target_centers: Sequence[Point],
        shape_candidates: Sequence[List[Tuple[float, float]]],
    ) -> Optional[List[Rect]]:
        current = list(positions)
        xi, yi, wi, hi = current[i]
        xj, yj, wj, hj = current[j]

        candidates = []
        for wi2, hi2 in shape_candidates[i][:2]:
            for wj2, hj2 in shape_candidates[j][:2]:
                candidates.append(
                    [
                        (xj, yj, wi2, hi2),
                        (xi, yi, wj2, hj2),
                    ]
                )

        for rect_i, rect_j in candidates:
            trial = list(current)
            trial[i] = rect_i
            trial[j] = rect_j
            if _overlaps(trial[i], trial, skip_idx=i):
                continue
            if _overlaps(trial[j], trial, skip_idx=j):
                continue
            return trial
        return None

    def _local_cost(
        self,
        block_id: int,
        positions: Sequence[Rect],
        b2b_by_block: Sequence[List[Tuple[int, float]]],
        p2b_by_block: Sequence[List[Tuple[float, float, float]]],
    ) -> float:
        cx, cy = _rect_center(positions[block_id])
        cost = 0.0
        for nbr, weight in b2b_by_block[block_id]:
            nx, ny = _rect_center(positions[nbr])
            cost += weight * (abs(cx - nx) + abs(cy - ny))
        for px, py, weight in p2b_by_block[block_id]:
            cost += weight * (abs(cx - px) + abs(cy - py))
        return cost

    def _refine_block(
        self,
        block_id: int,
        positions: Sequence[Rect],
        target_centers: Sequence[Point],
        shape_candidates: Sequence[List[Tuple[float, float]]],
        b2b_by_block: Sequence[List[Tuple[int, float]]],
        p2b_by_block: Sequence[List[Tuple[float, float, float]]],
    ) -> Optional[Rect]:
        current = positions[block_id]
        others_map = {idx: rect for idx, rect in enumerate(positions) if idx != block_id}
        anchors = self._generate_anchors(
            block_id,
            current[2],
            current[3],
            [
                (current[0], current[1]),
                (current[0] + current[2], current[1]),
                (current[0], current[1] + current[3]),
            ],
            others_map,
            target_centers,
            b2b_by_block,
        )

        best_rect = current
        best_score = self._proxy_cost_full(positions, b2b_by_block, p2b_by_block)
        for width, height in shape_candidates[block_id][:4]:
            local_anchors = list(anchors)
            target_x = max(0.0, target_centers[block_id][0] - width / 2.0)
            target_y = max(0.0, target_centers[block_id][1] - height / 2.0)
            local_anchors.extend(
                [
                    (target_x, target_y),
                    (current[0], current[1]),
                    (max(0.0, current[0] + current[2] - width), current[1]),
                    (current[0], max(0.0, current[1] + current[3] - height)),
                ]
            )
            for x, y in self._dedupe_points(local_anchors):
                rect = (x, y, width, height)
                if _overlaps(rect, positions, skip_idx=block_id):
                    continue
                trial = list(positions)
                trial[block_id] = rect
                score = self._proxy_cost_full(trial, b2b_by_block, p2b_by_block)
                if score + 1e-6 < best_score:
                    best_score = score
                    best_rect = rect

        if best_rect == current:
            return None
        return best_rect

    def _compact(self, positions: List[Rect]) -> List[Rect]:
        rects = list(positions)
        for _ in range(self.compaction_rounds):
            order_x = sorted(range(len(rects)), key=lambda idx: (rects[idx][0], rects[idx][1], idx))
            for idx in order_x:
                rects[idx] = self._shift_left(rects, idx)

            order_y = sorted(range(len(rects)), key=lambda idx: (rects[idx][1], rects[idx][0], idx))
            for idx in order_y:
                rects[idx] = self._shift_down(rects, idx)
        return rects

    def _shift_left(self, rects: Sequence[Rect], idx: int) -> Rect:
        x, y, w, h = rects[idx]
        limit = 0.0
        for other_idx, (ox, oy, ow, oh) in enumerate(rects):
            if other_idx == idx:
                continue
            overlap_y = min(y + h, oy + oh) - max(y, oy)
            if overlap_y > 1e-9 and ox + ow <= x + 1e-9:
                limit = max(limit, ox + ow)
        candidate = (limit, y, w, h)
        if not _overlaps(candidate, rects, skip_idx=idx):
            return candidate
        return rects[idx]

    def _shift_down(self, rects: Sequence[Rect], idx: int) -> Rect:
        x, y, w, h = rects[idx]
        limit = 0.0
        for other_idx, (ox, oy, ow, oh) in enumerate(rects):
            if other_idx == idx:
                continue
            overlap_x = min(x + w, ox + ow) - max(x, ox)
            if overlap_x > 1e-9 and oy + oh <= y + 1e-9:
                limit = max(limit, oy + oh)
        candidate = (x, limit, w, h)
        if not _overlaps(candidate, rects, skip_idx=idx):
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
