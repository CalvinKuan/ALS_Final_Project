# optimizer.py — 使用說明

## 環境需求

- Python 3.10+
- `student/abc` 執行檔（已附在此目錄）
- （選用）mockturtle_opt 執行檔，放在 `mockturtle/build/examples/mockturtle_opt`

## 基本執行

```bash
# 對所有 benchmarks 執行（從專案根目錄）
python student/optimizer.py

# 只跑單一 case
python student/optimizer.py --case ex215

# 指定自訂路徑
python student/optimizer.py \
  --benchmarks benchmarks/ \
  --output output/ \
  --abc student/abc
```

## 常用選項

| 參數 | 預設值 | 說明 |
|------|--------|------|
| `--case NAME` | 全部 | 只處理指定的 benchmark，例如 `ex215` |
| `--effort quick\|medium\|high` | `medium` | 搜尋力道；`quick` 最快，`high` 跑全部 flow |
| `--timeout N` | `30` | 每個 ABC / mockturtle flow 的秒數上限 |
| `--max-workers N` | 全部 CPU | 平行執行的 thread 數；多個 optimizer 同時跑時建議設 2–4 |
| `--no-abc` | — | 停用 ABC（純 Python 合成） |
| `--no-mockturtle` | — | 停用 mockturtle 後處理 |

## 效能調整

```bash
# 快速測試（只跑最少 flow）
python student/optimizer.py --effort quick --case ex215

# 最強搜尋（跑所有 BDD order、ANF phase、ABC flow）
python student/optimizer.py --effort high --timeout 60

# 平行跑多個 case，限制每個 optimizer 的 thread 數
python student/optimizer.py --case ex215 --max-workers 4 &
python student/optimizer.py --case ex217 --max-workers 4 &
```

## 合成策略說明

optimizer.py 依序產生多個 AIG 候選，取 ADP（area × delay）最小者輸出：

1. **Sparse SOP/POS** — 針對真值表稀疏的 output
2. **Recursive MUX** — Shannon 展開，依 variable influence 動態排序
3. **ANF/FPRM** — 代數標準型，支援多種 polarity phase
4. **ROBDD** — 支援 natural、reverse、interleave、byte_msb、influence_desc 等變數順序
5. **ABC flows** — 透過 `student/abc` 執行數十種優化序列（rewrite、refactor、dc2、compress2rs、dch、lcorr 等）
6. **ABC AIG refinement** — 對前 K 個候選再次用 ABC AIG-to-AIG flow 優化
7. **mockturtle** — 對前 K 個候選做後處理（需另行編譯）

## 進階參數

```bash
# 指定 BDD 變數順序（可用逗號分隔多個）
python student/optimizer.py --orders reverse,byte_msb,influence_desc

# 指定 ANF phase
python student/optimizer.py --anf-phases none,all,lower,upper

# 調整 ABC AIG refinement
python student/optimizer.py --abc-aig-top-k 5 --abc-aig-rounds 2

# mockturtle 相關
python student/optimizer.py \
  --mockturtle path/to/mockturtle_opt \
  --mockturtle-flows deep,rewrite_balance \
  --mockturtle-top-k 5 \
  --mockturtle-rounds 2

# 不保留舊的 output 作為 fallback
python student/optimizer.py --ignore-existing
```

## 驗證結果

```bash
# 從專案根目錄，用 evaluate.py 檢查正確性與 ADP
python evaluate.py
```

輸出格式：
```
[BEST] ex215: abc:compress2rs3   inputs=16 area=312    delay=15  adp=4680
Generated 100 AIG file(s) in output/
Improved cases this run: 12
Total local ADP estimate: 483201
```

## 編譯 mockturtle（選用）

```bash
cd mockturtle
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make mockturtle_opt -j$(nproc)
```

編譯完成後執行時加上 `--mockturtle mockturtle/build/examples/mockturtle_opt`。
