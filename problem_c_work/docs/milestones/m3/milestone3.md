# Milestone 3

Workspace isolation:
- M3 solver: `src/m3/solver_m3.py`
- M3 outputs: `results/m3/`
- M3 note: `notes/m3/milestone3.md`
- M1 and M2 artifacts remain in their own folders and were not overwritten.

Main M3 idea:
- start from M2 constructive placement
- add adaptive refinement gating
- evaluate a small deterministic portfolio per case:
  - constructive only
  - M2-style refinement
  - gated refinement
- choose the candidate with the best internal proxy cost

Run commands:

```powershell
$env:MPLCONFIGDIR='E:\Project\iccad2026C\problem_c_work\.mplconfig'
Set-Location 'E:\Project\iccad2026C\problem_c_work\results\m3'
python ..\..\contest\iccad2026_evaluate.py --validate ..\..\src\m3\solver_m3.py --quick
python ..\..\contest\iccad2026_evaluate.py --evaluate ..\..\src\m3\solver_m3.py --data-path ..\..\..\FloorSet
```

Current result on local validation:
- feasible: `100 / 100`
- total score: `1.8470`
- average cost: `1.6982`
- average runtime: `2.2349s`

Comparison:
- M1 total score: `1.8267`
- M2 total score: `1.8023`
- M3 total score: `1.8470`

Selected cases:
- `test-id 0`: M1 `1.3523`, M2 `1.1257`, M3 `1.1257`
- `test-id 40`: M1 `2.2714`, M2 `1.7766`, M3 `1.6061`
- `test-id 72`: M1 `1.9450`, M2 `2.3411`, M3 `2.4420`
- `test-id 94`: M1 `2.3468`, M2 `2.5458`, M3 `2.6706`
- `test-id 99`: M1 `2.1274`, M2 `2.0683`, M3 `2.1339`

Conclusion:
- M3 is implemented, validated, and isolated.
- M3 did **not** beat M2 on the full validation set.
- M2 remains the best checkpoint so far and should still be treated as the preferred solver.

Likely reason:
- the current internal proxy is not reliable enough to select among the M3 candidate branches on the full benchmark set.

Best next direction if continuing beyond M3:
- use exact evaluator-like metrics for branch selection on a small shortlist of complete candidate layouts
- or build M4 around per-case branch selection calibrated from observed M1/M2/M3 behavior instead of relying only on the current proxy
