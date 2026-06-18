# Milestone 1

Workspace:
- `problem_c_work/` is the only mutable workspace.
- `FloorSet/` remains the read-only source of copied scaffold files and datasets.

Current solver:
- `src/solver_v1.py`
- deterministic force-directed target centers
- corner-based legalization
- lightweight compaction

Run commands:

```powershell
$env:MPLCONFIGDIR='E:\Project\iccad2026C\problem_c_work\.mplconfig'
Set-Location 'E:\Project\iccad2026C\problem_c_work\contest'
python .\iccad2026_evaluate.py --validate ..\src\solver_v1.py --quick
python .\iccad2026_evaluate.py --evaluate ..\src\solver_v1.py --data-path ..\..\FloorSet
```

Current result on local validation:
- feasible: `100 / 100`
- total score: `1.8267`
- average cost: `1.6830`
- average runtime: `0.26s`

Selected single-case results:
- `test-id 0`: cost `1.9319`, runtime `0.02s`
- `test-id 99`: cost `1.4626`, runtime `0.70s`

Known next steps:
- add reinsertion/local-swap refinement after legalization
- allow a small aspect-ratio candidate set instead of square-only blocks
- use partial net cost more aggressively during block ordering and placement
