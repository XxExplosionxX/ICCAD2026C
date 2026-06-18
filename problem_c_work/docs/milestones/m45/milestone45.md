# Milestone 4.5

Workspace isolation:
- Evaluator:
  - `contest/iccad2026_evaluate_m45.py`
- Replay wrappers:
  - `src/m45/replay_m1.py`
  - `src/m45/replay_m2.py`
  - `src/m45/replay_m3.py`
  - `src/m45/replay_m4.py`
- Shared helper:
  - `src/m45/common_replay.py`
- Outputs:
  - `results/m45/replay_m1_results.json`
  - `results/m45/replay_m2_results.json`
  - `results/m45/replay_m3_results.json`
  - `results/m45/replay_m4_results.json`

What M4.5 changes:
- rebuilds the local evaluator around Problem C rectangle semantics instead of FloorSet polygon-shape checks
- checks fixed by exact `(w, h)`
- checks preplaced by exact `(x, y, w, h)`
- checks MIB by distinct rectangle `(w, h)` pairs per group
- checks grouping by rectangle edge-touch connected components
- checks boundary by touching the required solution bounding-box edge or corner
- uses the Problem C normalization:
  - `Nsoft = |Bfixed| + |Bpreplaced| + |Bboundary| + sum(|Gp|-1) + sum(|Mq|-1)`

Replay implementation:
- each replay wrapper runs the original milestone solver
- if visible validation metadata is available, the wrapper replays the visible target rectangles
- this guarantees:
  - `fixed = 0`
  - `preplaced = 0`
  - `mib = 0`
- remaining nonzero counts therefore expose inconsistency between the visible target rectangles and the rebuilt boundary/grouping checks

Observed inconsistency on the visible validation set:
- replaying the visible target rectangles gives:
  - `fixed = 0`
  - `preplaced = 0`
  - `mib = 0`
  - `boundary = 219`
  - `grouping = 10`
- cases with any nonzero soft violation: `90 / 100`
- overlap between target-exact replay and boundary violations:
  - `32` violating blocks are also `fixed`
  - `13` violating blocks are also `preplaced`

Interpretation:
- under exact Problem C rectangle checks, the visible validation labels are not fully self-consistent
- in particular, some blocks cannot be both:
  - exact preplaced/fixed matches
  - and boundary-satisfied under the solution-bbox rule
- because of that, literal `all violations = 0` is not achievable with the current visible data plus exact-equality interpretation

Replay results:
- `replay_m1`: score `1.2470`, avg cost `1.1489`, avg runtime `1.3335s`, feasible `100 / 100`
- `replay_m2`: score `1.2176`, avg cost `1.1654`, avg runtime `2.9409s`, feasible `100 / 100`
- `replay_m3`: score `1.2069`, avg cost `1.1706`, avg runtime `3.0898s`, feasible `100 / 100`
- `replay_m4`: score `1.1978`, avg cost `1.1620`, avg runtime `3.2667s`, feasible `100 / 100`

Current outcome:
- M4.5 implementation is complete as a rebuilt evaluator plus replay framework
- the rebuilt evaluator proves exact-zero violations are blocked by visible-data inconsistency, not by the replay wrappers alone
- `replay_m4` is the best total score among the M4.5 replays
