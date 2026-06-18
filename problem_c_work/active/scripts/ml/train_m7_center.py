#!/usr/bin/env python3
"""
M7 training script: supervised center prediction baseline.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Dict, List

import torch
from torch import nn

ACTIVE_ROOT = Path(__file__).resolve().parents[2]
WORK_ROOT = Path(__file__).resolve().parents[3]
REPO_ROOT = Path(__file__).resolve().parents[4]
CONTEST_DIR = ACTIVE_ROOT / "contest"
MODEL_DIR = ACTIVE_ROOT / "models"
if str(CONTEST_DIR) not in sys.path:
    sys.path.insert(0, str(CONTEST_DIR))
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))

from iccad2026_evaluate import get_training_dataloader
from ml.m7_center_model import M7CenterModel


def _normalize_centers_from_fp_sol(fp_sol: torch.Tensor, scale_xy: torch.Tensor, device: torch.device) -> torch.Tensor:
    target = []
    sx = float(scale_xy[0].item())
    sy = float(scale_xy[1].item())
    sx = max(sx, 1.0)
    sy = max(sy, 1.0)
    for row in fp_sol:
        w = float(row[0].item())
        h = float(row[1].item())
        x = float(row[2].item())
        y = float(row[3].item())
        cx = (x + 0.5 * w) / sx
        cy = (y + 0.5 * h) / sy
        target.append((cx, cy))
    return torch.tensor(target, dtype=torch.float32, device=device)


def _run_epoch(
    model: nn.Module,
    dataloader,
    device: torch.device,
    optimizer=None,
    scaler=None,
    use_amp: bool = False,
    max_batches: int | None = None,
    epoch_idx: int = 0,
    stage_name: str = "train",
    log_interval: int = 0,
) -> Dict[str, float]:
    train_mode = optimizer is not None
    model.train(train_mode)
    total_loss = 0.0
    total_samples = 0
    total_blocks = 0
    batch_counter = 0
    start_time = time.time()

    for batch_idx, batch in enumerate(dataloader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        area_target, b2b_conn, p2b_conn, pins_pos, constraints, _tree_sol, fp_sol, _metrics = batch

        batch_size = area_target.shape[0]
        batch_loss_total = 0.0
        batch_sample_count = 0
        batch_block_count = 0
        for sample_idx in range(batch_size):
            at = area_target[sample_idx]
            bc = b2b_conn[sample_idx]
            pc = p2b_conn[sample_idx]
            pp = pins_pos[sample_idx]
            cs = constraints[sample_idx]
            fp = fp_sol[sample_idx]

            block_count = int((at != -1).sum().item())
            at = at[:block_count].to(device)
            bc = bc.to(device)
            pc = pc.to(device)
            pp = pp.to(device)
            cs = cs[:block_count].to(device)
            fp = fp[:block_count].to(device)

            amp_context = (
                torch.autocast(device_type="cuda", dtype=torch.float16)
                if use_amp and device.type == "cuda"
                else nullcontext()
            )
            with amp_context:
                pred_norm, scale_xy = model(at, bc, pc, pp, cs)
                target_norm = _normalize_centers_from_fp_sol(fp, scale_xy.detach().cpu(), device)
                loss = nn.functional.smooth_l1_loss(pred_norm, target_norm)

            if train_mode:
                optimizer.zero_grad(set_to_none=True)
                if scaler is not None and use_amp and device.type == "cuda":
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

            loss_value = float(loss.item())
            total_loss += loss_value
            total_samples += 1
            total_blocks += block_count
            batch_loss_total += loss_value
            batch_sample_count += 1
            batch_block_count += block_count
        batch_counter += 1

        if log_interval > 0 and ((batch_idx + 1) % log_interval == 0):
            elapsed = time.time() - start_time
            msg = {
                "stage": stage_name,
                "epoch": epoch_idx,
                "batch": batch_idx + 1,
                "avg_batch_loss": batch_loss_total / max(batch_sample_count, 1),
                "avg_epoch_loss_so_far": total_loss / max(total_samples, 1),
                "batch_samples": batch_sample_count,
                "batch_blocks": batch_block_count,
                "samples_seen": total_samples,
                "elapsed_sec": elapsed,
                "device": str(device),
                "amp": bool(use_amp and device.type == "cuda"),
            }
            print(json.dumps(msg), flush=True)

    elapsed = time.time() - start_time
    return {
        "loss": total_loss / max(total_samples, 1),
        "samples": total_samples,
        "blocks": total_blocks,
        "batches": batch_counter,
        "elapsed_sec": elapsed,
        "samples_per_sec": total_samples / max(elapsed, 1e-6),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-path",
        default=str(REPO_ROOT),
        help="Repository root containing LiteTensorData/ and LiteTensorDataTest/.",
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--train-samples", type=int, default=2048)
    parser.add_argument("--val-samples", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--log-interval", type=int, default=20)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--cuda-index", type=int, default=0)
    parser.add_argument("--amp", action="store_true", help="Enable CUDA mixed precision training")
    parser.add_argument("--checkpoint-dir", default=str(WORK_ROOT / "artifacts" / "checkpoints" / "ml"))
    parser.add_argument("--result-dir", default=str(WORK_ROOT / "artifacts" / "results" / "ml"))
    parser.add_argument("--run-name", default="m7_center_baseline")
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
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp) if device.type == "cuda" else None
    model = M7CenterModel(hidden_dim=args.hidden_dim, depth=args.depth, dropout=args.dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    train_loader = get_training_dataloader(
        data_path=args.data_path,
        batch_size=args.batch_size,
        num_samples=args.train_samples,
        shuffle=False,
    )
    val_loader = get_training_dataloader(
        data_path=args.data_path,
        batch_size=1,
        num_samples=args.val_samples,
        shuffle=False,
    )

    history: List[Dict[str, float]] = []
    best_val = float("inf")
    best_ckpt = checkpoint_dir / f"{args.run_name}_best.pt"

    for epoch in range(1, args.epochs + 1):
        train_stats = _run_epoch(
            model,
            train_loader,
            device,
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
            optimizer=None,
            scaler=None,
            use_amp=use_amp,
            max_batches=args.val_samples,
            epoch_idx=epoch,
            stage_name="val",
            log_interval=max(1, min(args.log_interval, 10)),
        )
        row = {
            "epoch": epoch,
            "train_loss": train_stats["loss"],
            "val_loss": val_stats["loss"],
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
        "best_val_loss": best_val,
        "epochs": args.epochs,
        "history": history,
        "best_checkpoint": str(best_ckpt),
    }
    summary_path = result_dir / f"{args.run_name}_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary to {summary_path}")


if __name__ == "__main__":
    main()
