# M7.1 Milestone Plan

## Goal

Improve `M7` training throughput and make the objective more aligned with floorplanning cost.

Changes versus `M7`:
- batched model forward pass over the full mini-batch
- explicit differentiable contest cost term
- separate reporting of center loss and cost loss

## Files

- Model: [m71_center_batch_model.py](../../active/models/ml/m71_center_batch_model.py)
- Training: [train_m71_center_cost.py](../../active/scripts/ml/train_m71_center_cost.py)
- Config: [m71_center_cost.json](../../active/configs/ml/m71_center_cost.json)

## Training objective

`loss = center_weight * center_loss + cost_weight * differentiable_contest_cost`

Where:
- `center_loss` is masked Smooth L1 on normalized block centers
- `differentiable_contest_cost` comes from `compute_training_loss_differentiable(...)`

## Expected benefit

- better GPU utilization than `M7`
- more direct pressure toward wirelength/area-sensitive placements
- still keeps the target widths/heights from the dataset to stabilize training

## Suggested first server run

```bash
/path/to/python -u problem_c_work/active/scripts/ml/train_m71_center_cost.py --device cuda --cuda-index 0 --amp --data-path /path/to/repo-root --epochs 20 --batch-size 32 --train-samples 16384 --val-samples 256 --center-weight 1.0 --cost-weight 0.2 --checkpoint-dir problem_c_work/artifacts/checkpoints/ml --result-dir problem_c_work/artifacts/results/ml --log-interval 10 --run-name m71_center_cost
```
