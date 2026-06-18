# M4.6 Weighted Evaluation Comparison

Evaluator:
- `contest/iccad2026_evaluate_m46_debug.py`

Artifacts:
- [M1 JSON](/abs/path/E:/Project/iccad2026C/problem_c_work/results/m46/m1_m46_results.json)
- [M2 JSON](/abs/path/E:/Project/iccad2026C/problem_c_work/results/m46/m2_m46_results.json)
- [M3 JSON](/abs/path/E:/Project/iccad2026C/problem_c_work/results/m46/m3_m46_results.json)
- [M4 JSON](/abs/path/E:/Project/iccad2026C/problem_c_work/results/m46/m4_m46_results.json)
- [M1 CSV](/abs/path/E:/Project/iccad2026C/problem_c_work/results/m46/m1_m46_breakdown.csv)
- [M2 CSV](/abs/path/E:/Project/iccad2026C/problem_c_work/results/m46/m2_m46_breakdown.csv)
- [M3 CSV](/abs/path/E:/Project/iccad2026C/problem_c_work/results/m46/m3_m46_breakdown.csv)
- [M4 CSV](/abs/path/E:/Project/iccad2026C/problem_c_work/results/m46/m4_m46_breakdown.csv)

## Summary

All four milestone solvers are infeasible on all `100/100` validation cases under the current M4.6 exact-rectangle evaluator, so their weighted total score is identical:

| Milestone | Weighted Total Score | Feasible Cases | Avg Cost | Avg Runtime (s) |
|---|---:|---:|---:|---:|
| M1 | 10.0000 | 0 | 10.0000 | 0.3147 |
| M2 | 10.0000 | 0 | 10.0000 | 0.8581 |
| M3 | 10.0000 | 0 | 10.0000 | 0.9768 |
| M4 | 10.0000 | 0 | 10.0000 | 1.0875 |

Because `M = 10.0` is applied to every testcase for every solver, there is no weighted-score separation between `M1` through `M4` in this run.

## Weighted Comparison

By weighted total score:
1. `M1 = 10.0000`
2. `M2 = 10.0000`
3. `M3 = 10.0000`
4. `M4 = 10.0000`

Tie-break by runtime only:
1. `M1` at `0.3147s`
2. `M2` at `0.8581s`
3. `M3` at `0.9768s`
4. `M4` at `1.0875s`

## Aggregated Violation Totals

| Milestone | Fixed | Preplaced | Boundary | Grouping | MIB | Area | Overlap |
|---|---:|---:|---:|---:|---:|---:|---:|
| M1 | 668 | 259 | 1987 | 1273 | 0 | 882 | 0 |
| M2 | 699 | 259 | 2040 | 1235 | 250 | 923 | 0 |
| M3 | 699 | 259 | 2040 | 1235 | 251 | 923 | 0 |
| M4 | 699 | 259 | 2041 | 1234 | 250 | 923 | 0 |

## Interpretation

- The dominant result of this rerun is not score differentiation but infeasibility saturation.
- Under the current M4.6 exact-match evaluator:
  - all four solvers fail hard feasibility on every testcase
  - overlap is not the issue
  - the failures are driven by fixed/preplaced-dimension handling and resulting area infeasibility
- For debugging, the per-testcase breakdown CSVs are the primary next step; they expose the exact `HPWLgap`, `AREAgap`, per-constraint violation counts, and `p1/p2/p3` factors for each testcase.
