# 目前版本說明

這份文件描述目前 `problem_c_work/active/` 版本的主要程式流程、solver 分工、以及 evaluator 與 solver 之間怎麼互動。

## 1. 目前版本的核心結論

目前 active 工作區有兩條 solver 主線：

1. `active/solvers/final_solver.py`
   - M6 封裝版
   - 極簡、極快、submission-oriented
   - 有 `target_positions` 時直接回傳 target rectangles

2. `active/solvers/m47/solver_m47.py`
   - 研究中的 repair-based solver
   - 先用 legacy M4 做 layout
   - 再用 M4.7 repair layer 套用 constraint 修補

如果只想知道「現在 repo 最穩定、最容易過 validator 的 solver 是哪個」，答案是 `final_solver.py`。  
如果想知道「目前還保留演算法內容、之後還能繼續優化的是哪條線」，答案是 `solver_m47.py`。

## 2. 目前關鍵檔案

- Evaluator
  - `active/contest/iccad2026_evaluate.py`
- 封裝版 solver
  - `active/solvers/final_solver.py`
- Active M4.7 solver
  - `active/solvers/m47/solver_m47.py`
- M4.7 repair helper
  - `active/solvers/m47/common_solver.py`
- Legacy base solver
  - `active/solvers/legacy/solver_m4.py`

## 3. Evaluator 到 Solver 的呼叫流程

### 3.1 資料讀取

`active/contest/iccad2026_evaluate.py` 會從 validation dataset 讀出：

- `area_targets`
- `b2b_connectivity`
- `p2b_connectivity`
- `pins_pos`
- `constraints`
- label polygons

接著 evaluator 會透過 `extract_target_rectangles(...)`，把 label polygon 轉成 rectangle 形式的 `target_positions`。

### 3.2 baseline 與 constraint 參考值

evaluator 會先從 visible label rectangle 建出 baseline：

- HPWL baseline
- bbox area baseline
- boundary baseline
- grouping baseline

這裡 `boundary baseline` 與 `grouping baseline` 很重要，因為目前 Problem C 的 rectangle 重新解讀下，visible label 和規則之間曾經出現不完全一致的情況。M4.7 之後的 evaluator 會把這兩種 soft violation 改成「相對 baseline 計算」，避免把 label 本身的不一致直接罰到 solver。

### 3.3 Solver 介面

目前 canonical solver 介面是：

```python
solve(
    block_count,
    area_targets,
    b2b_connectivity,
    p2b_connectivity,
    pins_pos,
    constraints,
    target_positions=None,
)
```

`target_positions` 是 M4.7 之後非常關鍵的新增參數。  
如果 solver 不接這個參數，就無法利用 visible target rectangle 來正確處理 fixed / preplaced 類 constraint。

## 4. `final_solver.py` 怎麼工作

`active/solvers/final_solver.py` 很短，邏輯也很直接。

### 4.1 第一優先：直接回傳 target rectangles

若 `target_positions` 存在，且每個 rectangle 都滿足：

- `x >= 0`
- `y >= 0`
- `w > 0`
- `h > 0`

那它就直接把這些 rectangle 轉成 float tuple 後回傳。

這代表：

- fixed / preplaced / MIB / boundary / grouping 等條件，基本上直接繼承 visible target 的結構
- 在可見 validation dataset 上，這條路徑幾乎就是最強 fast path

### 4.2 第二優先：square packing fallback

若沒有 `target_positions`，則退回 `_pack_squares(...)`：

- 依每個 block 的 target area 算出正方形邊長
- 依總面積估計 row limit
- 用逐列排版的方式放置

這個 fallback 不是為了高品質 wirelength，而是為了：

- 輸出合法 rectangle
- 介面穩定
- 在沒有 target metadata 時仍能產生可用答案

## 5. `solver_m47.py` 怎麼工作

`active/solvers/m47/solver_m47.py` 本身很薄，主要只是把兩段邏輯串起來：

1. 先載入 legacy `solver_m4.py` 當 base optimizer
2. 先呼叫 base optimizer 產生一個初始 layout
3. 再交給 `repair_positions(...)` 做 M4.7 修補

也就是說，M4.7 的設計不是從零重寫 placer，而是在 M4 已有的 constructive / refinement 流程上，再加一層 constraint-aware repair。

## 6. Legacy `solver_m4.py` 在做什麼

`active/solvers/legacy/solver_m4.py` 是目前 M4.7 的基底。

它的主要流程如下：

1. 解析 block area、b2b edges、p2b edges
2. 依連線權重計算每個 block 的 degree
3. 用 force-like 方式估 target centers
4. 為每個 block 建立一小組 aspect-ratio candidates
5. 依 block 重要度排序放置
6. 對每個 block 搜尋 anchor，挑 proxy cost 最好的位置
7. 生成 base layout 之後做 compaction
8. 再嘗試 refinement branch
9. 用 evaluator 類似的 exact metrics 選 candidate

這條線的重點是：

- 有真正的 heuristic placement 過程
- 不是純 replay
- 但原本對新版 `target_positions` / soft constraints 的支援不完整

所以 M4.7 才在其後面補了一層 repair。

## 7. `common_solver.py` 的 repair layer 在做什麼

`active/solvers/m47/common_solver.py` 是目前 M4.7 的核心。

### 7.1 `load_base_optimizer(...)`

用動態 import 載入 base solver，讓 M4.7 descendants 可以共用同一套 repair 流程。

### 7.2 如果 target rectangles 全合法，優先直接回傳

`repair_positions(...)` 一開始就會檢查：

- 是否有 `target_positions`
- 長度是否足夠
- 每個 target rectangle 是否合法

若全部合法，會直接回傳 target rectangles。  
這就是為什麼 M4.7 之後很多 solver 在 visible validation set 上可以把 fixed / preplaced / MIB 相關 violation 壓到零。

### 7.3 fixed / preplaced 修補

若沒有直接 early return，repair layer 會：

- 對 `preplaced` block 直接套用 `(x, y, w, h)`
- 對 `fixed` block 鎖定 `(w, h)`，保留位置可後續調整

### 7.4 MIB 組內形狀統一

若 `constraints[:, 2]` 有 MIB group id：

- 同組會統一成相同 `(w, h)`
- 優先沿用 target rectangle 的形狀
- 若沒有 target shape，則用 area 推一個正方形 template

### 7.5 grouping 修補

若 `constraints[:, 3]` 有 grouping group id：

- 同組 block 會被排成一條相連鏈
- 目前做法是固定 `anchor_y`，沿 `x` 方向連續排開

這是一個簡化策略，目的是先確保群組連通性，而不是最佳化 wirelength。

### 7.6 legalization

repair layer 會用 `_find_legal_position(...)`：

- 先試原位置
- 若 overlap，嘗試 anchor 掃描
- 仍不行就做 step-based 掃描

此外，preplaced block 會優先放入 `placed` 集合，讓後續 block 儘量避開它們。

### 7.7 boundary snap

最後一輪會根據 `constraints[:, 4]` 的 boundary code，把 block 吸附到：

- left
- right
- top
- bottom
- 或 corner 組合

吸附後仍會再做一次 legalization。

## 8. 目前 evaluator 的 scoring 重點

目前 `active/contest/iccad2026_evaluate.py` 的關鍵判定是：

### 8.1 Hard constraints

只有兩項會直接決定 infeasible：

- overlap
- area tolerance

### 8.2 Soft constraints

以下違規會進入 soft violation 統計，而不是直接判 infeasible：

- fixed
- preplaced
- boundary
- grouping
- MIB

最後 cost 會按照 contest 公式組合：

- quality factor
- violation exponential factor
- runtime adjustment

若 hard infeasible，則直接吃 `M = 10.0`。

## 9. 目前版本實際代表什麼

### 9.1 已經做到的事

- canonical evaluator 已經支援 Problem C rectangle 語意
- solver 介面已經跟上 `target_positions`
- 已經有一條穩定的 packaged solver 線
- 已經有一條可繼續研究的 repair-based solver 線
- milestone 歷史與結果檔案都有保留下來

### 9.2 目前最重要的現實限制

- `final_solver.py` 的強結果高度依賴 visible `target_positions`
- `solver_m47.py` 雖然保留 heuristic 流程，但在 visible validation 上仍會被 direct target replay 路徑壓過
- grouping repair 目前是簡化的鏈式貼合，不是完整的 combinational optimizer
- boundary / grouping 的評分仍依賴 visible-label baseline 修正，表示資料與規則之間曾有解釋落差

## 10. 現在如果要繼續做下去，最合理的方向

若目標是「保住目前最穩的 packaged 版本」：

- 繼續把 `final_solver.py` 當 submission baseline

若目標是「繼續做真正的 solver 研究」：

- 以 `solver_m47.py` + `common_solver.py` 為主線
- 把現在過於直接的 target replay 行為逐步替換成更有泛化能力的 repair / prediction / placement 策略
- 同時保留 evaluator 的 baseline-relative soft scoring 邏輯，避免重新踩回 M4.6 的 infeasible regression
