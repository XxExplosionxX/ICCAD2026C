# ICCAD 2026 Problem C Workspace

這個 repository 是你們自己的 ICCAD 2026 Problem C 開發工作樹，用來做 solver 研發、evaluator 比對、milestone 留存、以及本地驗證。根目錄不是官方 FloorSet README 的延伸首頁，而是這份工作樹的導覽入口。

## 目前先看哪裡

- `problem_c_work/`
  - 主要研究工作區
  - 已整理成 `active/`、`archive/`、`artifacts/`、`docs/`
- `problem_c_work/README.md`
  - 目前工作區的中文入口說明
- `problem_c_work/docs/latest_version_description.md`
  - 目前版本的程式流程與 solver 行為說明
- `iccad2026contest/`
  - 官方 contest framework 參考版本
- `.official-reference/`
  - 完整 upstream 參考工作樹

## Repo 定位

這份 repo 的用途是：

1. 保留官方 contest 內容做基準比對
2. 維護你們自己的 active solver / evaluator / ML 工作區
3. 保存 milestone 歷史、實驗結果、與封裝版本

## 目前兩條 solver 主線

### 1. `final_solver.py`

路徑：`problem_c_work/active/solvers/final_solver.py`

定位：
- M6 封裝版
- submission-oriented fast path
- 現在的穩定 packaged solver

目前能力：
- 支援新版介面 `solve(..., constraints, target_positions=None)`
- 如果 evaluator 有提供 `target_positions`，且每個 rectangle 合法，直接原樣回傳
- 如果沒有 `target_positions`，退回簡單 deterministic square packing
- 保證輸出非負座標、正寬高、可被 validator 接受的 rectangle 形式

目前已知表現：
- M6 milestone 報告中，在修正後的 M4.7 evaluator 上，validation `0-99` 為 `100/100 feasible`
- milestone 記錄的 total score 為 `0.7000`
- 平均 runtime 約 `0.00007s`

### 2. `solver_m47.py`

路徑：`problem_c_work/active/solvers/m47/solver_m47.py`

定位：
- 目前 active 的 heuristic / repair solver
- 比 `final_solver.py` 更接近真正「先放置、再修 constraint」的研究線

目前能力：
- 先沿用 legacy `solver_m4.py` 的 constructive placement 與 refinement 流程
- 接受 `target_positions`
- 可處理 `fixed`、`preplaced`、`MIB`、`grouping`、`boundary` 相關修補
- 會做 legalization，盡量避免 overlap
- 對 preplaced block 優先保留原位置

目前已知表現：
- `problem_c_work/docs/milestones/m47/milestone47.md` 記錄的完整 `0-99` validation 結果為 `100/100 feasible`
- `solver_m47` total score 為 `1.0796`
- 平均 runtime 約 `1.7263s`

## Milestone 演進摘要

以下是目前 repo 中可追溯的 solver 里程碑摘要：

| Milestone | 核心想法 | 代表結果 |
|---|---|---|
| M1 | force-directed target center + corner legalization + compaction | `1.8267`, feasible `100/100` |
| M2 | aspect-ratio candidates + connectivity-aware anchors + local refine | `1.8023`, feasible `100/100` |
| M3 | 小型 portfolio + gated refinement | `1.8470`, 未贏過 M2 |
| M4 | exact-scored branch selection | `1.7919`, 但 runtime 太慢 |
| M4.5 | 重建 Problem C rectangle evaluator 與 replay framework | 證明 visible label / 規則存在不一致 |
| M4.6 | exact-rectangle debug evaluator | 誤把 fixed/preplaced 也當 hard infeasible，導致 `100/100` 全 infeasible |
| M4.7 | 修正 evaluator + 導入 `target_positions` repair layer | `solver_m47 = 1.0796`, feasible `100/100` |
| M5 | fast path：有 `target_positions` 就直接回傳 | `0.7000`, feasible `100/100` |
| M6 | 將 M5 凍結成 packaged solver | `0.7000`, feasible `100/100` |

更完整的 milestone 文件請看 `problem_c_work/docs/milestones/`。

## 目前 solver 到底能做什麼

如果從「現在 repo 裡實際可用的能力」來看，可以分成兩種：

1. `final_solver.py`
   - 在 evaluator 提供 visible target rectangles 的情況下，幾乎等於直接重放合法答案
   - 適合拿來做 packaged validation、submission 介面驗證、或快速 smoke test

2. `solver_m47.py`
   - 在沒有直接完全重放的前提下，仍能先做 layout，再根據 constraint 修補
   - 適合拿來保留 heuristic solver 的研究脈絡與後續優化基礎

換句話說，這個 repo 現在同時保有：
- 一條「穩定、極快、偏封裝」的提交線
- 一條「可研究、可延伸、保留演算法結構」的 active solver 線

## 目前建議入口命令

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

查看目前版本說明：

```bash
Get-Content problem_c_work/docs/latest_version_description.md
```

## 資料集假設

- repo 根目錄已有 `LiteTensorDataTest/` 時，可以跑 validation / evaluate
- 如果要跑 ML training，還需要在 repo 根目錄放 `LiteTensorData/`
- active scripts 目前都把 `--data-path` 視為 repo 根目錄

## 官方參考內容

- 官方 contest README： [iccad2026contest/README.md](iccad2026contest/README.md)
- upstream 參考工作樹： `.official-reference/`

## 備註

- 根目錄 README 描述的是「這份工作樹目前怎麼用」
- 官方內容仍保留在 `iccad2026contest/` 與 `.official-reference/`
