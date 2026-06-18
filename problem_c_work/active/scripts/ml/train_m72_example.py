#!/usr/bin/env python3
"""
M7.2 training script based on the contest training example.

This keeps the training flow structurally close to training_example.py while
adding a learned model, center supervision, checkpointing, and configurable
DataLoader performance knobs.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F

ACTIVE_ROOT = Path(__file__).resolve().parents[2]
WORK_ROOT = Path(__file__).resolve().parents[3]
REPO_ROOT = Path(__file__).resolve().parents[4]
CONTEST_DIR = ACTIVE_ROOT / "contest"
MODEL_DIR = ACTIVE_ROOT / "models"
if str(CONTEST_DIR) not in sys.path:
    sys.path.insert(0, str(CONTEST_DIR))
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))

from iccad2026_evaluate import compute_training_loss_differentiable, get_training_dataloader
from ml.m72_example_model import M72ExamplePlacementModel


def _build_target_centers(fp_sol: torch.Tensor, scale_xy: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
    w = fp_sol[:, :, 0]
    h = fp_sol[:, :, 1]
    x = fp_sol[:, :, 2]
    y = fp_sol[:, :, 3]
    cx = (x + 0.5 * w) / torch.clamp(scale_xy[:, 0].unsqueeze(1), min=1.0)
    cy = (y + 0.5 * h) / torch.clamp(scale_xy[:, 1].unsqueeze(1), min=1.0)
    target = torch.stack([cx, cy], dim=2)
    return target * valid_mask.unsqueeze(-1).to(target.dtype)


def _masked_center_loss(pred_norm: torch.Tensor, target_norm: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
    diff = F.smooth_l1_loss(pred_norm, target_norm, reduction="none")
    mask = valid_mask.unsqueeze(-1).to(diff.dtype)
    return (diff * mask).sum() / torch.clamp(mask.sum(), min=1.0)


def _json_default(value):
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _log_batch(
    *,
    stage_name: str,
    epoch_idx: int,
    batch_idx: int,
    total_loss: float,
    total_center_loss: float,
    total_cost_loss: float,
    total_samples: int,
    total_blocks: int,
    start_time: float,
    device: torch.device,
    use_amp: bool,
) -> None:
    elapsed = time.time() - start_time
    msg = {
        "stage": stage_name,
        "epoch": epoch_idx,
        "batch": batch_idx + 1,
        "avg_loss_so_far": total_loss / max(total_samples, 1),
        "avg_center_loss_so_far": total_center_loss / max(total_samples, 1),
        "avg_cost_loss_so_far": total_cost_loss / max(total_samples, 1),
        "samples_seen": total_samples,
        "blocks_seen": total_blocks,
        "elapsed_sec": elapsed,
        "samples_per_sec": total_samples / max(elapsed, 1e-6),
        "device": str(device),
        "amp": bool(use_amp and device.type == "cuda"),
    }
    print(json.dumps(msg), flush=True)


def _run_epoch(
    model: M72ExamplePlacementModel,
    dataloader,
    device: torch.device,
    center_weight: float,
    cost_weight: float,
    optimizer=None,
    scaler=None,
    use_amp: bool = False,
    epoch_idx: int = 0,
    stage_name: str = "train",
    log_interval: int = 0,
) -> Dict[str, float]:
    train_mode = optimizer is not None
    model.train(train_mode)
    total_loss = 0.0
    total_center_loss = 0.0
    total_cost_loss = 0.0
    total_samples = 0
    total_blocks = 0
    batch_counter = 0
    start_time = time.time()

    for batch_idx, batch in enumerate(dataloader):
        # Unpack batch - 8 tensors
        area_target, b2b_conn, p2b_conn, pins_pos, constraints, tree_sol, fp_sol, metrics = batch

        area_target = area_target.to(device)
        b2b_conn = b2b_conn.to(device)
        p2b_conn = p2b_conn.to(device)
        pins_pos = pins_pos.to(device)
        constraints = constraints.to(device)
        tree_sol = tree_sol.to(device)
        fp_sol = fp_sol.to(device)
        metrics = metrics.to(device)

        amp_context = (
            torch.autocast(device_type="cuda", dtype=torch.float16)
            if use_amp and device.type == "cuda"
            else nullcontext()
        )
        with amp_context:
            # =================================================================
            # YOUR NEURAL NETWORK HERE
            # =================================================================
            positions, aux = model(area_target, b2b_conn, p2b_conn, pins_pos, constraints, fp_sol)
            pred_centers_norm = aux["pred_centers_norm"]
            scale_xy = aux["scale_xy"]
            valid_mask = aux["valid_mask"]

            target_centers_norm = _build_target_centers(fp_sol, scale_xy, valid_mask)
            center_loss = _masked_center_loss(pred_centers_norm, target_centers_norm, valid_mask)

            cost_terms = []
            bsz = area_target.shape[0]
            for sample_idx in range(bsz):
                block_count = int(valid_mask[sample_idx].sum().item())
                pred_positions = positions[sample_idx, :block_count]
                areas = area_target[sample_idx, :block_count]
                cost = compute_training_loss_differentiable(
                    pred_positions,
                    b2b_conn[sample_idx],
                    p2b_conn[sample_idx],
                    pins_pos[sample_idx],
                    areas,
                    metrics[sample_idx],
                    constraints[sample_idx, :block_count],
                )
                cost_terms.append(cost)
                total_blocks += block_count

            cost_loss = torch.stack(cost_terms).mean()
            loss = center_weight * center_loss + cost_weight * cost_loss

        if train_mode:
            optimizer.zero_grad(set_to_none=True)
            if scaler is not None and use_amp and device.type == "cuda":
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

        batch_counter += 1
        batch_samples = int(area_target.shape[0])
        total_samples += batch_samples
        total_loss += float(loss.item()) * batch_samples
        total_center_loss += float(center_loss.item()) * batch_samples
        total_cost_loss += float(cost_loss.item()) * batch_samples

        if log_interval > 0 and ((batch_idx + 1) % log_interval == 0):
            _log_batch(
                stage_name=stage_name,
                epoch_idx=epoch_idx,
                batch_idx=batch_idx,
                total_loss=total_loss,
                total_center_loss=total_center_loss,
                total_cost_loss=total_cost_loss,
                total_samples=total_samples,
                total_blocks=total_blocks,
                start_time=start_time,
                device=device,
                use_amp=use_amp,
            )

    elapsed = time.time() - start_time
    return {
        "loss": total_loss / max(total_samples, 1),
        "center_loss": total_center_loss / max(total_samples, 1),
        "cost_loss": total_cost_loss / max(total_samples, 1),
        "samples": total_samples,
        "blocks": total_blocks,
        "batches": batch_counter,
        "elapsed_sec": elapsed,
        "samples_per_sec": total_samples / max(elapsed, 1e-6),
    }


def _resolve_bool_flag(explicit_value: Optional[bool], default_value: bool) -> bool:
    return default_value if explicit_value is None else explicit_value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-path",
        default=str(REPO_ROOT),
        help="Repository root containing LiteTensorData/ and LiteTensorDataTest/.",
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--train-samples", type=int, default=16384)
    parser.add_argument("--val-samples", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=192)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--center-weight", type=float, default=1.0)
    parser.add_argument("--cost-weight", type=float, default=0.2)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--cuda-index", type=int, default=0)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--pin-memory", dest="pin_memory", action="store_true")
    parser.add_argument("--no-pin-memory", dest="pin_memory", action="store_false")
    parser.add_argument("--persistent-workers", dest="persistent_workers", action="store_true")
    parser.add_argument("--no-persistent-workers", dest="persistent_workers", action="store_false")
    parser.add_argument("--prefetch-factor", type=int, default=None)
    parser.add_argument("--checkpoint-dir", default=str(WORK_ROOT / "artifacts" / "checkpoints" / "ml"))
    parser.add_argument("--result-dir", default=str(WORK_ROOT / "artifacts" / "results" / "ml"))
    parser.add_argument("--run-name", default="m72_example")
    parser.set_defaults(pin_memory=None, persistent_workers=None)
    args = parser.parse_args()

    checkpoint_dir = Path(args.checkpoint_dir)
    result_dir = Path(args.result_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)

    if args.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was explicitly requested but is not available.")
        device = torch.device(f"cuda:{args.cuda_index}")
    elif args.device == "cpu":
        device = torch.device("cpu")
    else:
        device = torch.device(f"cuda:{args.cuda_index}" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)

    use_amp = bool(args.amp and device.type == "cuda")
    pin_memory = _resolve_bool_flag(args.pin_memory, device.type == "cuda")
    persistent_workers = _resolve_bool_flag(args.persistent_workers, args.num_workers > 0)
    if args.num_workers <= 0:
        persistent_workers = False
        prefetch_factor = None
    else:
        prefetch_factor = args.prefetch_factor if args.prefetch_factor is not None else 2

    scaler = torch.cuda.amp.GradScaler(enabled=use_amp) if device.type == "cuda" else None
    model = M72ExamplePlacementModel(
        hidden_dim=args.hidden_dim,
        depth=args.depth,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    train_loader = get_training_dataloader(
        data_path=args.data_path,
        batch_size=args.batch_size,
        num_samples=args.train_samples,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
    )
    val_loader = get_training_dataloader(
        data_path=args.data_path,
        batch_size=args.batch_size,
        num_samples=args.val_samples,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
    )

    history: List[Dict[str, float]] = []
    best_val = float("inf")
    best_ckpt = checkpoint_dir / f"{args.run_name}_best.pt"

    for epoch in range(1, args.epochs + 1):
        train_stats = _run_epoch(
            model,
            train_loader,
            device,
            center_weight=args.center_weight,
            cost_weight=args.cost_weight,
            optimizer=optimizer,
            scaler=scaler,
            use_amp=use_amp,
            epoch_idx=epoch,
            stage_name="train",
            log_interval=args.log_interval,
        )
        val_stats = _run_epoch(
            model,
            val_loader,
            device,
            center_weight=args.center_weight,
            cost_weight=args.cost_weight,
            optimizer=None,
            scaler=None,
            use_amp=use_amp,
            epoch_idx=epoch,
            stage_name="val",
            log_interval=max(1, min(args.log_interval, 5)),
        )

        row = {
            "epoch": epoch,
            "train_loss": train_stats["loss"],
            "train_center_loss": train_stats["center_loss"],
            "train_cost_loss": train_stats["cost_loss"],
            "val_loss": val_stats["loss"],
            "val_center_loss": val_stats["center_loss"],
            "val_cost_loss": val_stats["cost_loss"],
            "train_samples": train_stats["samples"],
            "val_samples": val_stats["samples"],
            "train_batches": train_stats["batches"],
            "val_batches": val_stats["batches"],
            "train_blocks": train_stats["blocks"],
            "val_blocks": val_stats["blocks"],
            "train_elapsed_sec": train_stats["elapsed_sec"],
            "val_elapsed_sec": val_stats["elapsed_sec"],
            "train_samples_per_sec": train_stats["samples_per_sec"],
            "val_samples_per_sec": val_stats["samples_per_sec"],
        }
        history.append(row)
        print(json.dumps(row), flush=True)

        if val_stats["loss"] < best_val:
            best_val = val_stats["loss"]
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "config": vars(args),
                    "resolved_config": {
                        "device": str(device),
                        "amp": use_amp,
                        "num_workers": args.num_workers,
                        "pin_memory": pin_memory,
                        "persistent_workers": persistent_workers,
                        "prefetch_factor": prefetch_factor,
                    },
                    "best_val_loss": best_val,
                    "epoch": epoch,
                },
                best_ckpt,
            )

    summary = {
        "run_name": args.run_name,
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "cuda_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "amp": use_amp,
        "num_workers": args.num_workers,
        "pin_memory": pin_memory,
        "persistent_workers": persistent_workers,
        "prefetch_factor": prefetch_factor,
        "batch_size": args.batch_size,
        "train_samples_requested": args.train_samples,
        "val_samples_requested": args.val_samples,
        "best_val_loss": best_val,
        "epochs": args.epochs,
        "history": history,
        "best_checkpoint": str(best_ckpt),
        "config": vars(args),
    }
    summary_path = result_dir / f"{args.run_name}_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=_json_default)
    print(f"Saved summary to {summary_path}", flush=True)


if __name__ == "__main__":
    main()
