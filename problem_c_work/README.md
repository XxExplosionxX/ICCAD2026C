# Problem C Workspace

這個目錄是 ICCAD 2026 Problem C 的本地研究工作區，和 repo 根目錄下的官方 `iccad2026contest/` 參考樹分開維護。

## 目錄結構

- `active/`
  - 目前使用中的 evaluator、solvers、ML models、scripts、configs
- `archive/`
  - 舊版 evaluator、歷史 milestone solver、保留參考程式
- `artifacts/`
  - 驗證輸出、debug breakdown、server 拉回結果、ML checkpoint
- `docs/`
  - workflow 文件、milestone 報告、版本說明

## 目前 canonical entries

- Evaluator
  - `problem_c_work/active/contest/iccad2026_evaluate.py`
- 穩定封裝版 solver
  - `problem_c_work/active/solvers/final_solver.py`
- 目前 active 的 repair solver
  - `problem_c_work/active/solvers/m47/solver_m47.py`
- M4.7 repair helper
  - `problem_c_work/active/solvers/m47/common_solver.py`
- ML smoke test
  - `problem_c_work/active/scripts/ml/smoke_m7_data.py`
- ML training scripts
  - `problem_c_work/active/scripts/ml/train_m7_center.py`
  - `problem_c_work/active/scripts/ml/train_m71_center_cost.py`
  - `problem_c_work/active/scripts/ml/train_m72_example.py`

## 目前 solver 狀態

### `final_solver.py`

角色：
- M6 packaged solver
- 主要用於 validator / submission / fast-path evaluation

目前行為：
- 支援 `solve(..., constraints, target_positions=None)`
- 有 `target_positions` 時，直接回傳可見 target rectangles
- 沒有 `target_positions` 時，退回 deterministic square packing

目前結果：
- 依 `docs/milestones/m6/milestone6.md` 記錄，validation `0-99` 為 `100/100 feasible`
- total score `0.7000`
- 平均 runtime 約 `0.00007s`

### `solver_m47.py`

角色：
- 目前保留演算法脈絡的 active solver
- 適合後續繼續做 heuristic / repair 優化

目前行為：
- 先繼承 legacy `solver_m4.py` 的 constructive layout
- 再交給 `common_solver.py` 做 repair
- 可處理：
  - fixed dimension 鎖定
  - preplaced rectangle 套用
  - MIB 同形狀統一
  - grouping 連續鏈式貼合
  - boundary snap
  - overlap legalization

目前結果：
- 依 `docs/milestones/m47/milestone47.md` 記錄，validation `0-99` 為 `100/100 feasible`
- total score `1.0796`
- 平均 runtime 約 `1.7263s`

## Milestone 索引

目前 solver 演進摘要如下：

| Milestone | 重點 | 結果摘要 |
|---|---|---|
| M1 | 初版 force-directed placement | `1.8267` |
| M2 | aspect-ratio + local refine | `1.8023` |
| M3 | portfolio / gated refinement | `1.8470` |
| M4 | exact-scored branch selection | `1.7919`，但 runtime 偏慢 |
| M4.5 | rectangle evaluator replay framework | 找出 visible data / 規則不一致 |
| M4.6 | debug evaluator | 誤把 soft checks 視為 hard infeasible |
| M4.7 | 修正 evaluator 與 solver 介面 | `1.0796`, feasible `100/100` |
| M5 | direct target replay fast path | `0.7000` |
| M6 | M5 frozen packaged solver | `0.7000` |

完整歷史文件在 `problem_c_work/docs/milestones/`。

## 重要文件

- 工作區 milestone 索引：`problem_c_work/docs/milestones/README.md`
- 目前版本流程說明：`problem_c_work/docs/latest_version_description.md`
- workflow 筆記：`problem_c_work/docs/workflow/`

## 本地資料假設

- repo 根目錄若已有 `LiteTensorDataTest/`，即可進行 validation / evaluation
- 若要跑完整 ML training，仍需要 `LiteTensorData/`
- active scripts 預設 `--data-path` 指向 repo 根目錄

## 建議命令

驗證 packaged solver：

```bash
python problem_c_work/active/contest/iccad2026_evaluate.py --validate problem_c_work/active/solvers/final_solver.py --data-path .
```

評估 packaged solver：

```bash
python problem_c_work/active/contest/iccad2026_evaluate.py --evaluate problem_c_work/active/solvers/final_solver.py --data-path .
```

評估 active M4.7 solver：

```bash
python problem_c_work/active/contest/iccad2026_evaluate.py --evaluate problem_c_work/active/solvers/m47/solver_m47.py --test-id 0 --data-path .
```

在有 `LiteTensorData/` 時跑 ML smoke check：

```bash
python problem_c_work/active/scripts/ml/smoke_m7_data.py --data-path .
```

## 輸出規則

- active evaluator 預設把輸出寫到 `problem_c_work/artifacts/`
- ML checkpoint 放在 `problem_c_work/artifacts/checkpoints/ml/`
- ML summary / log 放在 `problem_c_work/artifacts/results/ml/`
- 舊 milestone 輸出保留在 `problem_c_work/artifacts/results/`，視為歷史資料
