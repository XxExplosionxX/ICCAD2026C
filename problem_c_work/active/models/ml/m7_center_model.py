#!/usr/bin/env python3
"""
M7 center-prediction baseline model.

This is the first ML milestone model:
- input: per-block features derived from contest inputs
- output: normalized center coordinates (cx, cy)

The model is intentionally simple so it is easy to train on a remote server
and integrate later into the existing legalizer / repair flow.
"""

from __future__ import annotations

from typing import Tuple

import torch
from torch import nn


def build_block_features(
    area_targets: torch.Tensor,
    b2b_connectivity: torch.Tensor,
    p2b_connectivity: torch.Tensor,
    pins_pos: torch.Tensor,
    constraints: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Build per-block features and a scale tensor used to denormalize predictions.

    Returns:
        features: [N, F]
        scale_xy: [2] estimated layout width/height scale
    """
    device = area_targets.device
    dtype = area_targets.dtype
    n = int((area_targets != -1).sum().item()) if area_targets.ndim == 1 else area_targets.shape[0]

    area = area_targets[:n].to(dtype)
    safe_area = torch.clamp(area, min=1.0)
    sqrt_area = torch.sqrt(safe_area)
    log_area = torch.log1p(safe_area)

    b2b_degree = torch.zeros(n, device=device, dtype=dtype)
    b2b_weight = torch.zeros(n, device=device, dtype=dtype)
    valid_b2b = b2b_connectivity[b2b_connectivity[:, 0] >= 0]
    for edge in valid_b2b:
        i = int(edge[0].item())
        j = int(edge[1].item())
        w = edge[2].to(dtype)
        if 0 <= i < n and 0 <= j < n:
            b2b_degree[i] += 1.0
            b2b_degree[j] += 1.0
            b2b_weight[i] += w
            b2b_weight[j] += w

    p2b_degree = torch.zeros(n, device=device, dtype=dtype)
    p2b_weight = torch.zeros(n, device=device, dtype=dtype)
    pin_pull_x = torch.zeros(n, device=device, dtype=dtype)
    pin_pull_y = torch.zeros(n, device=device, dtype=dtype)
    valid_p2b = p2b_connectivity[p2b_connectivity[:, 0] >= 0]
    for edge in valid_p2b:
        pin_idx = int(edge[0].item())
        block_idx = int(edge[1].item())
        w = edge[2].to(dtype)
        if 0 <= block_idx < n and 0 <= pin_idx < pins_pos.shape[0]:
            p2b_degree[block_idx] += 1.0
            p2b_weight[block_idx] += w
            pin_pull_x[block_idx] += w * pins_pos[pin_idx, 0].to(dtype)
            pin_pull_y[block_idx] += w * pins_pos[pin_idx, 1].to(dtype)

    denom = torch.clamp(p2b_weight, min=1.0)
    pin_pull_x = pin_pull_x / denom
    pin_pull_y = pin_pull_y / denom

    c = constraints[:n].to(dtype)
    fixed = (c[:, 0] != 0).to(dtype)
    preplaced = (c[:, 1] != 0).to(dtype)
    mib_gid = c[:, 2]
    group_gid = c[:, 3]
    boundary_code = c[:, 4]

    max_mib = torch.clamp(mib_gid.max(), min=1.0)
    max_group = torch.clamp(group_gid.max(), min=1.0)
    boundary_norm = boundary_code / 10.0

    feature_cols = [
        log_area,
        sqrt_area,
        fixed,
        preplaced,
        mib_gid / max_mib,
        group_gid / max_group,
        boundary_norm,
        b2b_degree,
        b2b_weight,
        p2b_degree,
        p2b_weight,
        pin_pull_x,
        pin_pull_y,
    ]
    features = torch.stack(feature_cols, dim=1)

    total_area = torch.clamp(safe_area.sum(), min=1.0)
    side = torch.sqrt(total_area)
    scale_xy = torch.stack([side, side])
    return features, scale_xy


class M7CenterModel(nn.Module):
    def __init__(self, input_dim: int = 13, hidden_dim: int = 128, depth: int = 3, dropout: float = 0.1):
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
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        features, scale_xy = build_block_features(
            area_targets,
            b2b_connectivity,
            p2b_connectivity,
            pins_pos,
            constraints,
        )
        pred_norm = torch.sigmoid(self.net(features))
        return pred_norm, scale_xy
