#!/usr/bin/env python3
"""
M7.2 model for the example-structured training flow.

This milestone keeps width/height fixed to the visible training solution and
learns placement through predicted block centers.
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch
from torch import nn


def build_batch_block_features(
    area_targets: torch.Tensor,
    b2b_connectivity: torch.Tensor,
    p2b_connectivity: torch.Tensor,
    pins_pos: torch.Tensor,
    constraints: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    device = area_targets.device
    dtype = area_targets.dtype
    bsz, max_blocks = area_targets.shape
    valid_mask = area_targets > 0

    safe_area = torch.clamp(area_targets, min=1.0)
    log_area = torch.log1p(safe_area)
    sqrt_area = torch.sqrt(safe_area)

    b2b_degree = torch.zeros(bsz, max_blocks, device=device, dtype=dtype)
    b2b_weight = torch.zeros(bsz, max_blocks, device=device, dtype=dtype)
    p2b_degree = torch.zeros(bsz, max_blocks, device=device, dtype=dtype)
    p2b_weight = torch.zeros(bsz, max_blocks, device=device, dtype=dtype)
    pin_pull_x = torch.zeros(bsz, max_blocks, device=device, dtype=dtype)
    pin_pull_y = torch.zeros(bsz, max_blocks, device=device, dtype=dtype)

    for b in range(bsz):
        n = int(valid_mask[b].sum().item())

        valid_b2b = b2b_connectivity[b][b2b_connectivity[b][:, 0] >= 0]
        for edge in valid_b2b:
            i = int(edge[0].item())
            j = int(edge[1].item())
            w = edge[2].to(dtype)
            if 0 <= i < n and 0 <= j < n:
                b2b_degree[b, i] += 1.0
                b2b_degree[b, j] += 1.0
                b2b_weight[b, i] += w
                b2b_weight[b, j] += w

        valid_p2b = p2b_connectivity[b][p2b_connectivity[b][:, 0] >= 0]
        for edge in valid_p2b:
            pin_idx = int(edge[0].item())
            block_idx = int(edge[1].item())
            w = edge[2].to(dtype)
            if 0 <= block_idx < n and 0 <= pin_idx < pins_pos[b].shape[0]:
                p2b_degree[b, block_idx] += 1.0
                p2b_weight[b, block_idx] += w
                pin_pull_x[b, block_idx] += w * pins_pos[b, pin_idx, 0].to(dtype)
                pin_pull_y[b, block_idx] += w * pins_pos[b, pin_idx, 1].to(dtype)

    pin_weight_denom = torch.clamp(p2b_weight, min=1.0)
    pin_pull_x = pin_pull_x / pin_weight_denom
    pin_pull_y = pin_pull_y / pin_weight_denom

    fixed = (constraints[:, :, 0] != 0).to(dtype)
    preplaced = (constraints[:, :, 1] != 0).to(dtype)
    mib_gid = constraints[:, :, 2].to(dtype)
    group_gid = constraints[:, :, 3].to(dtype)
    boundary_code = constraints[:, :, 4].to(dtype) / 10.0

    max_mib = torch.clamp(mib_gid.amax(dim=1, keepdim=True), min=1.0)
    max_group = torch.clamp(group_gid.amax(dim=1, keepdim=True), min=1.0)

    total_area = (safe_area * valid_mask.to(dtype)).sum(dim=1)
    side = torch.sqrt(torch.clamp(total_area, min=1.0))
    scale_xy = torch.stack([side, side], dim=1)

    scale_x = torch.clamp(scale_xy[:, 0].unsqueeze(1), min=1.0)
    scale_y = torch.clamp(scale_xy[:, 1].unsqueeze(1), min=1.0)
    pin_pull_x = pin_pull_x / scale_x
    pin_pull_y = pin_pull_y / scale_y

    features = torch.stack(
        [
            log_area,
            sqrt_area,
            fixed,
            preplaced,
            mib_gid / max_mib,
            group_gid / max_group,
            boundary_code,
            b2b_degree,
            b2b_weight,
            p2b_degree,
            p2b_weight,
            pin_pull_x,
            pin_pull_y,
        ],
        dim=2,
    )
    return features, scale_xy, valid_mask


class M72ExamplePlacementModel(nn.Module):
    def __init__(self, input_dim: int = 13, hidden_dim: int = 192, depth: int = 4, dropout: float = 0.1):
        super().__init__()
        layers = []
        dim = input_dim
        for _ in range(max(1, depth)):
            layers.append(nn.Linear(dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            dim = hidden_dim
        layers.append(nn.Linear(dim, 2))
        self.net = nn.Sequential(*layers)

    def forward(
        self,
        area_targets: torch.Tensor,
        b2b_connectivity: torch.Tensor,
        p2b_connectivity: torch.Tensor,
        pins_pos: torch.Tensor,
        constraints: torch.Tensor,
        fp_sol: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        squeeze_output = area_targets.ndim == 1
        if squeeze_output:
            area_targets = area_targets.unsqueeze(0)
            b2b_connectivity = b2b_connectivity.unsqueeze(0)
            p2b_connectivity = p2b_connectivity.unsqueeze(0)
            pins_pos = pins_pos.unsqueeze(0)
            constraints = constraints.unsqueeze(0)
            fp_sol = fp_sol.unsqueeze(0)

        features, scale_xy, valid_mask = build_batch_block_features(
            area_targets,
            b2b_connectivity,
            p2b_connectivity,
            pins_pos,
            constraints,
        )
        pred_centers_norm = torch.sigmoid(self.net(features))
        pred_centers_norm = pred_centers_norm * valid_mask.unsqueeze(-1).to(pred_centers_norm.dtype)

        w = torch.where(valid_mask, fp_sol[:, :, 0], torch.zeros_like(fp_sol[:, :, 0]))
        h = torch.where(valid_mask, fp_sol[:, :, 1], torch.zeros_like(fp_sol[:, :, 1]))
        cx = pred_centers_norm[:, :, 0] * scale_xy[:, 0].unsqueeze(1)
        cy = pred_centers_norm[:, :, 1] * scale_xy[:, 1].unsqueeze(1)
        x = torch.where(valid_mask, cx - 0.5 * w, torch.zeros_like(cx))
        y = torch.where(valid_mask, cy - 0.5 * h, torch.zeros_like(cy))
        positions = torch.stack([x, y, w, h], dim=2)

        aux = {
            "pred_centers_norm": pred_centers_norm,
            "scale_xy": scale_xy,
            "valid_mask": valid_mask,
        }
        if squeeze_output:
            positions = positions.squeeze(0)
            aux = {key: value.squeeze(0) for key, value in aux.items()}
        return positions, aux
