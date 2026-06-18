# Milestone 2

Workspace isolation:
- M2 solver: `src/m2/solver_m2.py`
- M2 outputs: `results/m2/`
- M2 note: `notes/m2/milestone2.md`
- M1 artifacts kept in place and not overwritten:
  - `src/solver_v1.py`
  - `contest/solver_v1_results.json`
  - `notes/milestone1.md`

Main changes over M1:
- discrete aspect-ratio candidates per block
- stronger connectivity-aware anchor generation
- bounded block-level local refinement
- same hard-constraint safety model: exact area and overlap-free placement

Run commands:

```powershell
$env:MPLCONFIGDIR='E:\Project\iccad2026C\problem_c_work\.mplconfig'
Set-Location 'E:\Project\iccad2026C\problem_c_work\results\m2'
python ..\..\contest\iccad2026_evaluate.py --validate ..\..\src\m2\solver_m2.py --quick
python ..\..\contest\iccad2026_evaluate.py --evaluate ..\..\src\m2\solver_m2.py --data-path ..\..\..\FloorSet
```

Current result on local validation:
- feasible: `100 / 100`
- total score: `1.8023`
- average cost: `1.6606`
- average runtime: `1.9561s`

Direct comparison with M1:
- M1 total score: `1.8267`
- M2 total score: `1.8023`
- M1 avg runtime: `0.2550s`
- M2 avg runtime: `1.9561s`

Selected cases:
- `test-id 0`: M1 `1.3523` -> M2 `1.1257`
- `test-id 40`: M1 `2.2714` -> M2 `1.7766`
- `test-id 99`: M1 `2.1274` -> M2 `2.0683`

Current weak spots:
- higher-block-count cases still dominate the worst scores
- top difficult cases include `94`, `72`, `98`, `95`, `97`

Next likely gains:
- refine only the worst local-contribution blocks more selectively
- reduce expensive refinement on cases already good after constructive placement
- improve bbox control on large cases before local refinement starts
