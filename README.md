# AI CUP 2026 春季賽 - ESG 永續承諾驗證競賽 TEAM_9939 - 2

本儲存庫為 TEAM_9939 參與 [AI CUP 2026 春季賽：ESG 永續承諾驗證競賽](https://www.aidea-web.tw/aicup_veripromiseesg) 的**任務專屬文本訓練策略**程式碼。

> **⚠️ 專案定位說明**  
> 本儲存庫採用「任務專屬文本訓練策略」，在原始文本前依各子任務拼接不同的 metadata 前綴，訓練七個預訓練模型並產出 Out-Of-Fold（OOF）與測試集的預測機率矩陣（`.pkl` 檔）。  
> 這些 pkl 檔案後續會與隊友的原始文本訓練策略（4 個 pkl）在集成腳本（`aiesg_0608.ipynb`）中合併，透過版本 C 集成策略得出最終提交成績（Private Leaderboard: 0.6375782, Rank 21）。

---

## 本階段模型特色與創新性

- **任務專屬文本輸入（Metadata 前綴）**：每筆資料依四個子任務各自建構不同的輸入文本，在原始文本前拼接透過關鍵字分析（`ESG_data_keyword.ipynb`）所設計的 metadata 前綴，無需修改模型架構即可注入領域知識。
- **多任務聯合學習架構**：四個子任務分類頭共享 backbone 權重，backbone 對各任務的 task-specific 文本各跑一次，同時接收四個任務的梯度訊號，在資料量有限的情況下提升泛化能力。
- **Attention-weighted Mean Pooling**：捨棄標準 `[CLS]` token，對所有有效 token 的隱藏向量依 attention mask 加權平均（`pooled = (last_hidden × mask).sum(1) / mask.sum(1)`），更均勻捕捉全句語意。
- **截斷式類別加權損失**：針對樣本極少的類別（如 Misleading、within_2_years），引入平衡類別權重並以 `CLASS_WEIGHT_MAX=5.0` 截斷，防止稀有類別梯度爆炸。
- **FGM 對抗訓練**：對共用 backbone 的 word embedding 層施加對抗擾動（eps=1.0），提升模型對語言表面變異的魯棒性。

---

## Metadata 前綴規則

各子任務的輸入文本在原始文本前拼接以下前綴：

| 子任務 | 欄位 | 前綴類型 | 說明 |
|---|---|---|---|
| 子任務一 | promise_status | confusion_aware_metadata | 含九項規則特徵，針對易混淆類型設計 |
| 子任務二 | evidence_status | esg_only | 僅含 ESG 類別標籤 |
| 子任務三 | evidence_quality | task_specific | 含五項證據相關特徵 |
| 子任務四 | verification_timeline | esg_only | 僅含 ESG 類別標籤 |

---

## 檔案說明

| 檔案 | 說明 |
|---|---|
| `M2_train_roberta_only.py` | 訓練 `hfl/chinese-roberta-wwm-ext`，產出 `M2_roberta.pkl` |
| `M2_train_lert_only.py` | 訓練 `hfl/chinese-lert-base`，產出 `M2_lert.pkl` |
| `M2_train_large_only.py` | 訓練 `hfl/chinese-roberta-wwm-ext-large`，產出 `M2_large.pkl` |
| `M2_train_ernie_only.py` | 訓練 `nghuyong/ernie-3.0-base-zh`，產出 `M2_ernie.pkl` |
| `M2_train_macbert_only.py` | 訓練 `hfl/chinese-macbert-base`，產出 `M2_macbert.pkl` |
| `M2_train_mengzi_only.py` | 訓練 `Langboat/mengzi-bert-base`，產出 `M2_mengzi.pkl` |
| `M2_train_electra_only.py` | 訓練 `hfl/chinese-electra-180g-base-discriminator`，產出 `M2_electra.pkl` |
| `aiesg_0608.ipynb` | 集成腳本，合併本策略 7 個 pkl 與隊友 4 個 pkl，輸出版本 C submission CSV |
| `ESG_data_keyword.ipynb` | 關鍵字分析工具，用於分析各 ESG 類別高頻關鍵字，輔助設計 metadata 前綴詞彙表（前處理參考用） |

---

## 執行環境與依賴套件

本專案設計於 Google Colab 上執行，建議啟用 GPU（T4）。

- **程式語言**：Python 3.10
- **核心套件**：
  - `transformers==4.44.2`（Hugging Face）
  - `torch`（支援 fp16 混合精度加速）
  - `numpy`, `pandas`, `sklearn`, `pickle`

---

## 執行步驟

### Step 1：準備資料

從競賽官網（[AIdea](https://www.aidea-web.tw/aicup_veripromiseesg)）下載資料集，上傳至 Google Drive：

```
MyDrive/ESGtest/
├── vpesg4k_train_1000.json
├── vpesg4k_val_1000.json
└── vpesg4k_test_2000.json
```

### Step 2：訓練七個模型

分別於 Google Colab 執行七支訓練腳本，每支腳本產出對應的 `.pkl` 檔至 `MyDrive/ESGtest/M2_pkl/`：

```
M2_train_roberta_only.py  →  M2_roberta.pkl
M2_train_lert_only.py     →  M2_lert.pkl
M2_train_large_only.py    →  M2_large.pkl
M2_train_ernie_only.py    →  M2_ernie.pkl
M2_train_macbert_only.py  →  M2_macbert.pkl
M2_train_mengzi_only.py   →  M2_mengzi.pkl
M2_train_electra_only.py  →  M2_electra.pkl
```

> **注意**：執行前請確認腳本頂部的路徑設定：
> ```python
> PKL_DIR = "/content/drive/MyDrive/ESGtest/M2_pkl"
> DATA_DIR = "/content/drive/MyDrive/ESGtest"
> ```

### Step 3：集成產出 submission

開啟 `aiesg_0608.ipynb`，修改以下路徑後依序執行所有 cell：

```python
PKL_DIR = "/content/drive/MyDrive/ESGtest/M2_pkl"          # 本策略 7 個 pkl 所在資料夾
JIE_PKL = "/content/drive/MyDrive/ESGtest/M2_pkl/0.62esg/probs_ensemble.pkl"  # 隊友原始文本策略的 pkl（參考連結一：https://github.com/yuxi20001010/AICUP_2026_ESG_Verification_Team9939/blob/main/README.md）
OUT_DIR = "/content/drive/MyDrive/ESGtest"                  # 輸出 CSV 的資料夾
```

腳本會輸出三個版本，最終提交使用 `submission_wavg_avg_t3.csv`（版本 C）。

---

## 訓練策略細節

- **資料切分**：官方訓練集（1,000 筆）與驗證集（1,000 筆）合併為 2,000 筆，以 `verification_timeline` 欄位作為分層依據進行 5 折 Stratified K-Fold 切分。
- **Seeds**：每個模型執行 3 組隨機種子（42、1024、2025），共 15 折結果取平均，降低隨機初始化帶來的不穩定性。
- **最佳化與排程**：AdamW（weight_decay=0.01），搭配 10% Linear Warmup Decay 排程。
- **模型評估**：每個 epoch 結束後在折內驗證子集計算加權 Macro-F1，保留最高分 epoch 的機率矩陣，跨 fold 與 seed 取平均後存為 pkl。
