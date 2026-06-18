# ICCAD 2026 Problem C Workspace

這個 repository 現在是拿來做 ICCAD 2026 Problem C 比賽準備、solver 開發、和本地驗證的工作樹，不再把 repo 根目錄當成原始 FloorSet 專案首頁。

## 你現在會用到的區塊

- `problem_c_work/`
  - 主要研究工作區
  - 已整理成 `active/`、`archive/`、`artifacts/`、`docs/`
- `iccad2026contest/`
  - 官方 contest framework 參考版本
- `.official-reference/`
  - 完整 upstream 參考工作樹
- `LiteTensorDataTest/`
  - 本地 validation dataset

如果你要看目前工作區的細節，請先讀 [problem_c_work/README.md](problem_c_work/README.md)。

## Repo 定位

這份 repo 的用途是：

1. 保留官方 contest 相關內容當基準
2. 提供你們自己的 active solver / evaluator / ML 工作區
3. 讓你們之後仍然可以抓 upstream 更新來比對

這份 repo **不是** 在根目錄直接延續官方 FloorSet README 的使用方式。

## 目前建議入口

本地驗證 packaged solver：

```bash
python problem_c_work/active/contest/iccad2026_evaluate.py --validate problem_c_work/active/solvers/final_solver.py --data-path .
```

本地評估 active M4.7 solver：

```bash
python problem_c_work/active/contest/iccad2026_evaluate.py --evaluate problem_c_work/active/solvers/m47/solver_m47.py --test-id 0 --data-path . --output solver_m47_check.json
```

查看工作區說明：

```bash
Get-Content problem_c_work/README.md
```

## 資料集假設

- 目前 repo 根目錄已有 `LiteTensorDataTest/`，所以 validation 類流程可跑
- 如果要跑 ML training，還需要在 repo 根目錄放 `LiteTensorData/`
- active scripts 預設把 `--data-path` 視為 repo 根目錄

## 官方參考內容在哪裡

- 目前使用中的官方 contest README： [iccad2026contest/README.md](iccad2026contest/README.md)
- 完整 upstream 參考樹： `.official-reference/`
- upstream/worktree 管理方式： [CONTEST_WORKFLOW.md](CONTEST_WORKFLOW.md)

## 備註

- 根目錄 README 現在是「你們這份 repo 怎麼用」的說明，不是原始 FloorSet 專案首頁
- 原始官方內容仍然保留在 upstream 參考樹與 contest 子目錄，不需要再放在根目錄重複展示
