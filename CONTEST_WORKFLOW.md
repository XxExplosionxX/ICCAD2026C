# 比賽開發工作流程

這個 repository 目前已經整理成 ICCAD 2026 比賽開發用的結構：

- `main`：你們平常開發比賽程式用的 branch，且工作樹已套用 sparse checkout
- `official-reference`：完整官方內容的參考 branch，實際 checkout 在 `.official-reference/`

## `main` 會保留什麼

目前 `main` 的工作樹只保留比賽直接會用到的內容：

- `iccad2026contest/`
- `LiteTensorDataTest/`
- `cost.py`
- `liteLoader.py`
- `litetestLoader.py`
- `lite_dataset.py`
- `utils.py`
- 根目錄的一些基本檔案，例如 `README.md`、`requirements.txt`、`.gitignore`、`LICENSE`

`main` 刻意不追蹤任何 upstream branch。這樣可以避免你在 `main` 上不小心執行 `git pull` 時，又把整份官方 repository 的內容重新帶回目前這個比賽工作樹。

## 更新官方參考內容

先把 upstream 的最新 commit 抓進本地 repo：

```bash
git fetch upstream
```

再更新完整官方參考工作樹：

```bash
git -C .official-reference pull --ff-only
```

## 檢查官方有哪些 contest 相關更新

如果你只想看 contest 相關檔案的差異，可以用：

```bash
git diff main..official-reference -- iccad2026contest cost.py liteLoader.py litetestLoader.py lite_dataset.py utils.py README.md requirements.txt
```

如果你想看這些檔案涉及哪些 commit，可以用：

```bash
git log --oneline main..official-reference -- iccad2026contest cost.py liteLoader.py litetestLoader.py lite_dataset.py utils.py README.md requirements.txt
```

## 把需要的官方更新帶回 `main`

如果確認某些官方更新你想納入 `main`，可以只把需要的檔案從 `official-reference` 還原回來：

```bash
git restore --source official-reference -- iccad2026contest cost.py liteLoader.py litetestLoader.py lite_dataset.py utils.py README.md requirements.txt
```

接著再照一般流程檢查並提交：

```bash
git status
git add <paths>
git commit -m "Sync selected upstream ICCAD 2026 updates"
```

## 推送你們自己的比賽開發內容

因為 `main` 沒有設定 upstream tracking，所以推送時請明確指定：

```bash
git push origin main
```
