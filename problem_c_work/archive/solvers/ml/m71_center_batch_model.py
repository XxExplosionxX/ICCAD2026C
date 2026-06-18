#!/usr/bin/env python3
"""
M7.1 batched center model with cost-aware training support.

Compared with M7:
- builds padded batch features [B, N, F]
- runs one forward pass per batch instead of per sample
- keeps a validity mask for masked losses
"""

from __future__ import annotations

from typing import Tuple

import torch
from torch import nn


def build_batch_block_features(
    area_targets: torch.Tensor,
    b2b_connectivity: torch.Tensor,
    p2b_connectivity: torch.Tensor,
    pins_pos: torch.Tensor,
    constraints: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Args:
        area_targets: [B, N]
        b2b_connectivity: [B, E1, 3]
        p2b_connectivity: [B, E2, 3]
        pins_pos: [B, P, 2]
        constraints: [B, N, 5]

    Returns:
        features: [B, N, F]
        scale_xy: [B, 2]
        valid_mask: [B, N]
    """
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

    denom = torch.clamp(p2b_weight, min=1.0)
    pin_pull_x = pin_pull_x / denom
    pin_pull_y = pin_pull_y / denom

    fixed = (constraints[:, :, 0] != 0).to(dtype)
    preplaced = (constraints[:, :, 1] != 0).to(dtype)
    mib_gid = constraints[:, :, 2].to(dtype)
    group_gid = constraints[:, :, 3].to(dtype)
    boundary_code = constraints[:, :, 4].to(dtype) / 10.0

    max_mib = torch.clamp(mib_gid.amax(dim=1, keepdim=True), min=1.0)
    max_group = torch.clamp(group_gid.amax(dim=1, keepdim=True), min=1.0)

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

    total_area = (safe_area * valid_mask.to(dtype)).sum(dim=1)
    side = torch.sqrt(torch.clamp(total_area, min=1.0))
    scale_xy = torch.stack([side, side], dim=1)
    return features, scale_xy, valid_mask


class M71CenterBatchModel(nn.Module):
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
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        features, scale_xy, valid_mask = build_batch_block_features(
            area_targets,
            b2b_connectivity,
            p2b_connectivity,
            pins_pos,
            constraints,
        )
        pred = self.net(features)
        pred_norm = torch.sigmoid(pred) * valid_mask.unsqueeze(-1).to(pred.dtype)
        return pred_norm, scale_xy, valid_mask
