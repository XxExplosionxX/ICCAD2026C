# Problem C Workspace

This directory is the local ICCAD 2026 research workspace. It is separate from the official `iccad2026contest/` tree at the repo root.

## Layout

- `active/`: current evaluator, solvers, ML models, scripts, and configs
- `archive/`: older evaluator snapshots and milestone solvers kept for reference
- `artifacts/`: checkpoints, experiment outputs, pulled-back server results, and local validation outputs
- `docs/`: workflow notes and milestone writeups

## Current Canonical Entries

- Evaluator: `problem_c_work/active/contest/iccad2026_evaluate.py`
- Stable packaged solver: `problem_c_work/active/solvers/final_solver.py`
- Active repair-based solver: `problem_c_work/active/solvers/m47/solver_m47.py`
- ML smoke test: `problem_c_work/active/scripts/ml/smoke_m7_data.py`
- ML training scripts:
  - `problem_c_work/active/scripts/ml/train_m7_center.py`
  - `problem_c_work/active/scripts/ml/train_m71_center_cost.py`
  - `problem_c_work/active/scripts/ml/train_m72_example.py`

## Local Data Assumptions

- Local validation is ready because `LiteTensorDataTest/` exists at the repo root.
- Full ML training still requires `LiteTensorData/` at the repo root.
- Active scripts default `--data-path` to the repo root, so they expect:
  - `./LiteTensorDataTest/`
  - `./LiteTensorData/` when training data is available

## Recommended Commands

Validate the packaged solver:

```bash
python problem_c_work/active/contest/iccad2026_evaluate.py --validate problem_c_work/active/solvers/final_solver.py --data-path .
```

Evaluate a single validation case with the active M4.7 solver:

```bash
python problem_c_work/active/contest/iccad2026_evaluate.py --evaluate problem_c_work/active/solvers/m47/solver_m47.py --test-id 0 --data-path .
```

Run the ML smoke check when `LiteTensorData/` is available:

```bash
python problem_c_work/active/scripts/ml/smoke_m7_data.py --data-path .
```

## Output Policy

- Active evaluator outputs go to `problem_c_work/artifacts/` by default.
- ML checkpoints go to `problem_c_work/artifacts/checkpoints/ml/`.
- ML summaries and logs should go to `problem_c_work/artifacts/results/ml/`.
- Historical outputs remain under `problem_c_work/artifacts/results/` and are not part of the active interface.

## Documentation

- Current workflow notes are under `problem_c_work/docs/workflow/`.
- Historical milestone reports are under `problem_c_work/docs/milestones/`.
- Some archived milestone notes still describe older paths and should be treated as historical snapshots.
