# M7 Milestone Plan

## Goal

Start the ML track with a minimal supervised baseline that is easy to train on the GPU server and easy to sync back for local milestone development.

This milestone does **not** replace the current no-ML solver.
It trains a model to predict block center hints from the contest inputs.

## Files

- Model: [m7_center_model.py](../../active/models/ml/m7_center_model.py)
- Smoke script: [smoke_m7_data.py](../../active/scripts/ml/smoke_m7_data.py)
- Training script: [train_m7_center.py](../../active/scripts/ml/train_m7_center.py)
- Config: [m7_center_baseline.json](../../active/configs/ml/m7_center_baseline.json)

## Server Workflow

1. Push up:
   - `problem_c_work/active/models/ml/`
   - `problem_c_work/active/scripts/ml/`
   - `problem_c_work/active/configs/ml/`
   - `problem_c_work/docs/milestones/ml/`

2. On the server, run the smoke test first:

```bash
python problem_c_work/active/scripts/ml/smoke_m7_data.py --data-path /path/to/repo-root
```

3. Then run the first training baseline:

```bash
python problem_c_work/active/scripts/ml/train_m7_center.py \
  --data-path /path/to/repo-root \
  --epochs 5 \
  --batch-size 4 \
  --train-samples 128 \
  --val-samples 8 \
  --checkpoint-dir problem_c_work/artifacts/checkpoints/ml \
  --result-dir problem_c_work/artifacts/results/ml \
  --run-name m7_center_baseline
```

4. Pull back:
   - `problem_c_work/artifacts/checkpoints/ml/`
   - `problem_c_work/artifacts/results/ml/`

## Expected Outputs

- checkpoint:
  - `artifacts/checkpoints/ml/m7_center_baseline_best.pt`
- summary:
  - `artifacts/results/ml/m7_center_baseline_summary.json`

## What To Sync Back For Review

- best checkpoint
- training summary JSON
- console log if you saved one
- exact command used

## Next Step After First Server Run

Use the synced checkpoint and summary to decide:
- whether to improve the feature set
- whether to add shape prediction
- whether to integrate the model into a new ML-guided floorplanning solver milestone
