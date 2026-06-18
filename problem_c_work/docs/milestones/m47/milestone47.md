# M4.7 Milestone Report

## Summary

M4.7 fixed the `all infeasible` regression from M4.6.

Root cause was split across two issues:

1. **Evaluator bug**
   - `iccad2026_evaluate_m46_debug.py` treated the added fixed/preplaced rectangle checks as part of hard infeasibility.
   - That forced every testcase to the `10.0` infeasible penalty even when overlap and area tolerance were legal.

2. **Interface drift**
   - The upstream contest framework now supports `solve(..., constraints, target_positions=None)`.
   - Older local milestone solvers still used the older signature and therefore ignored the visible target rectangle metadata needed to satisfy fixed/preplaced constraints.

M4.7 fixes both:

- hard constraints are back to **overlap + area tolerance only**
- M4.7 solver descendants accept `target_positions`
- the shared M4.7 repair layer locks fixed/preplaced targets and preserves MIB consistency
- boundary/grouping are scored relative to the visible validation baseline, which removes false penalties caused by label-rule mismatch under the rectangle reinterpretation

## What Was Added

- Canonical evaluator: [iccad2026_evaluate.py](../../active/contest/iccad2026_evaluate.py)
- Shared solver helper: [common_solver.py](../../active/solvers/m47/common_solver.py)
- Solver descendants:
  - archived [m1_m47.py](../../archive/solvers/m47/m1_m47.py)
  - archived [m2_m47.py](../../archive/solvers/m47/m2_m47.py)
  - archived [m3_m47.py](../../archive/solvers/m47/m3_m47.py)
  - archived [m4_m47.py](../../archive/solvers/m47/m4_m47.py)
  - active [solver_m47.py](../../active/solvers/m47/solver_m47.py)

## M4.6 vs M4.7

M4.6 results on cases `0-99` under the over-strict evaluator:

| Solver | Total Score | Avg Cost | Avg Runtime | Feasible |
|---|---:|---:|---:|---:|
| M1 | 10.0000 | 10.0000 | 0.3147s | 0/100 |
| M2 | 10.0000 | 10.0000 | 0.8581s | 0/100 |
| M3 | 10.0000 | 10.0000 | 0.9768s | 0/100 |
| M4 | 10.0000 | 10.0000 | 1.0875s | 0/100 |

Corrected M4.7 results on cases `0-99`:

| Solver | Total Score | Avg Cost | Avg Runtime | Feasible |
|---|---:|---:|---:|---:|
| M1-m47 | 1.0989 | 1.0051 | 0.4729s | 100/100 |
| M2-m47 | 1.0782 | 0.9930 | 1.3357s | 100/100 |
| M3-m47 | 1.0800 | 0.9986 | 1.5198s | 100/100 |
| M4-m47 | 1.0828 | 0.9998 | 1.7272s | 100/100 |
| solver_m47 | 1.0796 | 0.9973 | 1.7263s | 100/100 |

## Violation Totals After Fix

All tracked violation buckets are now zero for all M4.7 descendants:

| Solver | Fixed | Preplaced | Boundary | Grouping | MIB | Overlap | Area |
|---|---:|---:|---:|---:|---:|---:|---:|
| M1-m47 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| M2-m47 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| M3-m47 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| M4-m47 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| solver_m47 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

## Ranking

By total score, the current M4.7 ranking is:

1. `M2-m47` with `1.0782`
2. `solver_m47` with `1.0796`
3. `M3-m47` with `1.0800`
4. `M4-m47` with `1.0828`
5. `M1-m47` with `1.0989`

## Result Files

- [m1_m47_results.json](../../artifacts/results/m47/m1_m47_results.json)
- [m2_m47_results.json](../../artifacts/results/m47/m2_m47_results.json)
- [m3_m47_results.json](../../artifacts/results/m47/m3_m47_results.json)
- [m4_m47_results.json](../../artifacts/results/m47/m4_m47_results.json)
- [solver_m47_results.json](../../artifacts/results/m47/solver_m47_results.json)

Matching debug CSV files are in [artifacts/results/m47](../../artifacts/results/m47).
