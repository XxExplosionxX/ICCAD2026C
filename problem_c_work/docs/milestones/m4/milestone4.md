# Milestone 4

Workspace isolation:
- M4 family solvers:
  - `src/m4/solver_m4.py`
  - `src/m4/solver_m4b.py`
  - `src/m4/solver_m4c.py`
- M4 family outputs:
  - `results/m4/solver_m4_results.json`
  - `results/m4/solver_m4b_results.json`
  - `results/m4/solver_m4c_results.json`
- Existing M1-M3 artifacts were not modified or overwritten.

Acceptance gate:
- required total score `< 1.8023` to beat M2
- required average runtime `<= 2.0s`
- required feasible count `100 / 100`

## M4 attempt
- idea: exact-scored `A/B/C` portfolio
- result:
  - total score: `1.7919`
  - average cost: `1.6309`
  - average runtime: `2.4982s`
  - feasible: `100 / 100`
- outcome: **passed score**, **failed runtime**

## M4b replan
- idea: exact-scored `A/B` only
- result:
  - total score: `1.9339`
  - average cost: `1.7719`
  - average runtime: `2.0944s`
  - feasible: `100 / 100`
- outcome: **failed score**, **failed runtime**

## M4c replan
- idea: keep `A/B/C` but trigger `C` much more selectively
- result:
  - total score: `1.8349`
  - average cost: `1.6813`
  - average runtime: `2.0873s`
  - feasible: `100 / 100`
- outcome: **failed score**, **failed runtime**

## Current frontier
- M1: score `1.8267`, runtime `0.2550s`
- M2: score `1.8023`, runtime `1.9561s`
- M3: score `1.8470`, runtime `2.2349s`
- M4: score `1.7919`, runtime `2.4982s`
- M4b: score `1.9339`, runtime `2.0944s`
- M4c: score `1.8349`, runtime `2.0873s`

Conclusion:
- No M4-family variant cleared the combined gate.
- `M2` remains the best accepted checkpoint.
- `M4` is the best score in the M4 family, but it is too slow for the imposed runtime gate.

Most likely reason:
- exact portfolio branch selection can buy score, but the extra branch cost pushes runtime above the gate.
- removing or heavily restricting branches lowers runtime but gives back too much score.

Best next direction beyond M4:
- move to a learned or rule-calibrated pre-selector that predicts when the extra branch is worth running, so the expensive branch is only evaluated on a much smaller subset of cases
- or relax the hard runtime gate if the true priority is best score rather than simultaneous score/runtime dominance
