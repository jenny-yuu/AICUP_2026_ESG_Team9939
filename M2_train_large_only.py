# ============================================================
# VeriPromise ESG M2 - MultiTask Single Model: large × 3 Seed × 5-fold
# ============================================================
# 整合學姊的 electra 程式碼架構：
#
# ✅ MultiTaskModel：1個backbone共享，4個task head同時訓練
#    （不是我們舊版的4個獨立模型）
# ✅ Attention-weighted Mean Pooling（比CLS更穩，特別適合ELECTRA）
#    pooled = (last_hidden * mask).sum(1) / mask.sum(1)
# ✅ AMP 混合精度：autocast + GradScaler（訓練速度大幅提升）
# ✅ 4任務 Loss 加總，一次 backward
# ✅ FGM：套用在全部任務的 embedding
# ✅ MAX_LEN=384（比舊版256更長，覆蓋更完整）
# ✅ PKL：1個模型存1個pkl（含所有seed×fold的平均probs）
#    → 7模型 × 3seeds = 21 pkl檔案
#    → 各pkl有 {oof, test} 兩份probs，涵蓋4個任務
# ✅ decode()：hard rule post-processing
# ✅ Stratify on verification_timeline（同學姊）
# ============================================================

import subprocess, sys
subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q',
    'transformers', 'scikit-learn', 'pandas', 'tqdm'])

import json, re, random, urllib.request, os, hashlib, pickle, warnings
from collections import Counter
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from sklearn.utils.class_weight import compute_class_weight

warnings.filterwarnings("ignore")

try:
    from google.colab import drive
    drive.mount('/content/drive')
    DRIVE_AVAILABLE = True
    print("✅ Google Drive 已掛載")
except:
    DRIVE_AVAILABLE = False

# ============================================================
# 1. 資料載入
# ============================================================

DATA_URL = "https://raw.githubusercontent.com/veripromiseesg/veripromiseesgdataset/ac91c1c8b5d116edf6fc44cccc1ee3b618f5a207/vpesg4ktrain1000v1.json"
urllib.request.urlretrieve(DATA_URL, "data.json")
with open("data.json", encoding="utf-8") as f:
    data = json.load(f)

def fix_labels(items):
    for item in items:
        if item.get("verification_timeline") == "longer_than_5_years":
            item["verification_timeline"] = "more_than_5_years"
    return items

data = fix_labels(data)

HAS_VAL = False; val_data = []
for vpath in ["/content/drive/MyDrive/ESGtest/aicup驗證集/vpesg4k_val_1000.json",
              "/content/drive/MyDrive/ESGtest/vpesg4k_val_1000.json", "vpesg4k_val_1000.json"]:
    try:
        with open(vpath, encoding="utf-8") as f: val_data = json.load(f)
        fix_labels(val_data); print(f"✅ 驗證集: {vpath} ({len(val_data)} 筆)")
        HAS_VAL = True; break
    except: continue

HAS_TEST = False; test_data = []
for tpath in ["/content/drive/MyDrive/ESGtest/測試集/vpesg4k_test_2000.json",
              "/content/drive/MyDrive/ESGtest/vpesg4k_test_2000.json", "vpesg4k_test_2000.json"]:
    try:
        with open(tpath, encoding="utf-8") as f: test_data = json.load(f)
        print(f"✅ 測試集: {tpath} ({len(test_data)} 筆)"); HAS_TEST = True; break
    except: continue

# train + val 合併作為訓練資料
train_data = list(data) + (list(val_data) if HAS_VAL else [])
print(f"✅ 訓練池: {len(train_data)} 筆")

# ============================================================
# 2. 設定（學姊版本整合）
# ============================================================

# ── 任務定義（學姊原版 EVAL_FIELDS）──────────────────────
EVAL_FIELDS = {
    "promise_status":        ["Yes", "No"],
    "verification_timeline": ["already", "within_2_years", "between_2_and_5_years", "more_than_5_years", "N/A"],
    "evidence_status":       ["Yes", "No", "N/A"],
    "evidence_quality":      ["Clear", "Not Clear", "Misleading", "N/A"],
}
FIELD_WEIGHTS = {
    "promise_status": 0.20, "verification_timeline": 0.15,
    "evidence_status": 0.30, "evidence_quality": 0.35,
}
LABEL2ID = {f: {l: i for i, l in enumerate(ls)} for f, ls in EVAL_FIELDS.items()}
NUM_LABELS = {f: len(ls) for f, ls in EVAL_FIELDS.items()}

# ── 訓練設定（學姊版本）─────────────────────────────────
MAX_LEN    = 384    # 學姊用 384（比舊版 256 更長）
EPOCHS     = 8      # 學姊用 8
LR         = 2e-5
USE_FGM    = True   # 全任務 FGM（學姊：所有任務）
USE_AMP    = True   # 混合精度（學姊：使用）
CLASS_WEIGHT_MAX = 5.0

# ── M2：7模型 × 3 Seed ──────────────────────────────────
N_SPLITS = 5

ENSEMBLE_SEEDS = [42, 1024, 2025]   # 白板三種 seed

SEVEN_MODELS = {
    "large": ("hfl/chinese-roberta-wwm-ext-large", 8),  # single model version
}

# 白板右側：Task2 class-conditional ensemble 權重
TASK2_YES_MODELS = {"roberta": 0.5, "lert": 0.3, "electra": 0.2}
TASK2_NO_MODELS  = {"large": 0.7}
TASK2_DEFAULT_W  = 0.1

# PKL 儲存目錄
PKL_DIR_LOCAL = "/content/M2_pkl"
PKL_DIR_DRIVE = "/content/drive/MyDrive/ESGtest/M2_pkl"
os.makedirs(PKL_DIR_LOCAL, exist_ok=True)
if DRIVE_AVAILABLE: os.makedirs(PKL_DIR_DRIVE, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 第二 public CSV（保底用）
SECOND_PUBLIC_CSV = [
    "/content/drive/MyDrive/ESGtest/submission_test_final_trainval_5fold_raw_checked.csv",
    "/content/drive/MyDrive/ESGtest/submission_test_final_trainval_5fold_raw.csv",
]

def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)

print(f"\nDevice: {DEVICE}")
print(f"7 Models: {list(SEVEN_MODELS.keys())}")
print(f"Seeds: {ENSEMBLE_SEEDS}")
print(f"MAX_LEN={MAX_LEN}, EPOCHS={EPOCHS}, AMP={USE_AMP}, FGM={USE_FGM}")

# ============================================================
# 3. 規則特徵（H版完整移植）
# ============================================================

PROMISE_WORDS=["目標","承諾","致力","致力於","推動","規劃","預計","將","將會","將持續","持續","並持續","逐步","提升","減少","降低","達成","實現","建立","強化","改善","2030","2050","淨零","減碳","碳中和","零排放","低碳","RE100","未來","努力","願景","希望","貢獻"]
FUTURE_WORDS=["將","將會","將持續","預計","規劃","未來","目標","承諾","致力","致力於","逐步","持續","並持續","達成","實現","2030","2050","淨零","RE100","年內","年底前"]
ACTION_WORDS=["執行","實施","導入","建立","推動","改善","降低","提升","減少","強化","完成","取得","通過","舉辦","簽署","演練","維修","建置","盤查","稽核","查證","訂定","提出","要求","管理","監控","考核"]
REPORT_DISCLOSURE_WORDS=["報告","揭露","發布報告","定期發布","說明","公司介紹","內容","董事會","資訊揭露","列出","介紹"]
PRODUCT_SERVICE_WORDS=["產品","服務","碳標籤","多國語言","ATM","功能性","列出服務","產品項目","服務項目"]
ROUTINE_COMMITMENT_WORDS=["每年","每月","定期","每季","持續執行","制度","員工旅遊","下午茶","社團","福利","持續提升服務"]
EVIDENCE_WORDS=["完成","取得","通過","導入","建置","執行","實施","查證","認證","投入","舉辦","已","達成","累計","ISO","GRI","SGS","BSI","TCFD","報告","稽核","數據","揭露","檢驗","盤查","審查","驗證","成果","制度","機制","措施","方案","計畫","KPI","完成率","達成率","第三方","簽署","每月","每年","事故"]
MEASURABLE_EVIDENCE_WORDS=["數字","數據","比例","完成率","達成率","每月","每年","事故","件","人次","項","噸","度","KPI","第三方","查證","稽核","認證","ISO","SGS","TCFD","GRI"]
ABSTRACT_EVIDENCE_WORDS=["挑戰","永續創新","責任","評估","願景","理念","重視","努力","持續改善","強化管理","提升韌性","積極推動"]
QUALITY_CLEAR_WORDS=["數據","比例","達成率","完成率","認證","查證","稽核","ISO","SGS","GRI","TCFD","第三方","具體","量化","揭露","報告","盤查","制度","文件","簽署","退休金","洗錢防制","維修","演練","每月","每年"]
QUALITY_NOT_CLEAR_WORDS=["持續","致力","強化","提升","落實","重視","推動","改善","努力","規劃","期望","逐步","積極","韌性","布局"]
MISLEADING_WORDS=["提升","強化","推動","持續","致力","布局","韌性","穩健經營","財務穩定","持續成長","高標準","透明度"]
LAW_WORDS=["遵循","法規","規範","依法","合規","法律","法令","稅務法規","財務與稅務法規"]
ALREADY_WORDS=["已完成","已導入","目前已","已執行","已建立","已取得","已通過","管理","執行","委員會","系統","現在","目前"]
NEAR_YEAR_WORDS=["2024","2025","2026","今年","明年","近期","短期","年內","年底前","兩年內","2年內"]
MID_LONG_WORDS=["2030","2050","淨零","RE100","減碳","中期","長期","氣候目標","國際標準"]
ESG_E_WORDS=["廢棄物","綠能","水資源","用水","減碳","碳排","碳中和","淨零","排放","能源","綠色","氣候","溫室氣體","RE100","再生能源","節能"]
ESG_S_WORDS=["勞工","弱勢","高齡","兒童","員工","同仁","福利","訓練","教育","人權","職安","安全","健康","社會責任","退休","照護"]
ESG_G_WORDS=["檢舉","不誠信","資恐","治理","董事","稽核","法規","合規","反貪腐","洗錢防制","風險管理","透明度","股東","委員會","誠信經營"]

QUANT_RE=re.compile(r"(\d+(\.\d+)?\s*%|\d+(\.\d+)?\s*％|\d+\s*年|\d+\s*月|\d+\s*日|\d+\s*人|\d+\s*件|\d+\s*項|\d+\s*噸|\d+\s*度|\d+\s*元|\d+\s*億|\d+\s*萬|20\d{2})")
TIME_PATTERNS=[r"20\d{2}\s*年",r"20\d{2}",r"\d+\s*年內",r"\d+\s*年",r"\d+\s*個?\s*月",r"\d+\s*天",r"\d+\s*[週周]",r"今年",r"明年",r"近期",r"短期",r"中期",r"長期",r"年底前",r"年內",r"淨零",r"已完成",r"已導入",r"目前已",r"每年",r"每月",r"定期"]

def normalize_chinese_number(text):
    text=str(text); cn_map={"零":"0","〇":"0","一":"1","二":"2","兩":"2","三":"3","四":"4","五":"5","六":"6","七":"7","八":"8","九":"9","十":"10"}
    for cn,num in cn_map.items():
        text=re.sub(rf"{cn}\s*年",f"{num}年",text); text=re.sub(rf"{cn}\s*個?\s*月",f"{num}個月",text)
        text=re.sub(rf"{cn}\s*天",f"{num}天",text); text=re.sub(rf"{cn}\s*[週周]",f"{num}週",text)
    return re.sub(r"(\d)\s+(年|月|天|週|周)",r"\1\2",text)
def has_any(text, words): return any(w in str(text) for w in words)
def get_esg(item):
    esg=item.get("esg_type","Unknown"); return str(esg) if esg and str(esg).strip() else "Unknown"
def normalize_esg_combo(esg):
    esg=str(esg).replace(",",";").replace("；",";")
    parts=sorted(list(set([x.strip() for x in esg.split(";") if x.strip()])))
    return ";".join(parts) if parts else "Unknown"
def extract_esg_keyword_domain(text):
    d=[]
    if has_any(text,ESG_E_WORDS): d.append("E")
    if has_any(text,ESG_S_WORDS): d.append("S")
    if has_any(text,ESG_G_WORDS): d.append("G")
    return ";".join(d) if d else "無明顯領域詞"
def extract_esg_prior(esg):
    n=normalize_esg_combo(esg)
    return {"esg_norm":n,
            "promise_prior":"高" if n in ["E;G","G;S"] else("低" if n=="Unknown" else "普通"),
            "quality_prior":"高機率Clear" if n=="E;G" else("偏Clear" if n=="G;S" else "普通"),
            "timeline_prior":{"E;G":"中期","E;G;S":"中期","G;S":"長期","S":"已實行"}.get(n,"未知")}
def extract_time_signal(text):
    text=normalize_chinese_number(text); found=[]
    for pattern in TIME_PATTERNS: found.extend(re.findall(pattern,text))
    if not found: return "無"
    joined=" ".join(found)
    if has_any(text,ALREADY_WORDS): return "已實行"
    if has_any(text,NEAR_YEAR_WORDS): return "短期"
    if "2050" in joined or "淨零" in joined: return "長期"
    if "2030" in joined: return "中期"
    if any(x in joined for x in ["短期","近期","今年","明年","年底前"]): return "短期"
    nums=[int(x) for x in re.findall(r"\d+",joined) if x.isdigit()]
    if nums:
        if any(x in [2024,2025,2026] for x in nums): return "短期"
        if any(2027<=x<=2030 for x in nums): return "中期"
        if any(x>2030 for x in nums): return "長期"
        if any(1<=x<=2 for x in nums): return "短期"
        if any(3<=x<=5 for x in nums): return "中期"
        if any(x>5 and x<100 for x in nums): return "長期"
    return "有"
def extract_rule_metadata(text, esg="Unknown"):
    text=normalize_chinese_number(str(text)); ep=extract_esg_prior(esg)
    hpr=has_any(text,PROMISE_WORDS); hfu=has_any(text,FUTURE_WORDS); hac=has_any(text,ACTION_WORDS); hqu=bool(QUANT_RE.search(text))
    hrd=has_any(text,REPORT_DISCLOSURE_WORDS); hps=has_any(text,PRODUCT_SERVICE_WORDS); hrc=has_any(text,ROUTINE_COMMITMENT_WORDS)
    hev=has_any(text,EVIDENCE_WORDS); hme=has_any(text,MEASURABLE_EVIDENCE_WORDS) or hqu; hae=has_any(text,ABSTRACT_EVIDENCE_WORDS)
    hqc=has_any(text,QUALITY_CLEAR_WORDS); hnc=has_any(text,QUALITY_NOT_CLEAR_WORDS); hml=has_any(text,MISLEADING_WORDS); hlw=has_any(text,LAW_WORDS)
    hal=has_any(text,ALREADY_WORDS); hny=has_any(text,NEAR_YEAR_WORDS); hmg=has_any(text,MID_LONG_WORDS); ts=extract_time_signal(text)
    ps=("低" if hrd and not hac and not hrc else "低" if hps and not hfu else "制度型承諾" if hrc else "高" if hfu and hac and hqu else "中" if hfu and hac else "低" if hpr else "無")
    es=("高" if hme and hev else "中" if hme else "低" if hae and not hme else "中" if hev else "低")
    cr=("可能誤導" if hlw and hml and not hqu else "清楚" if hqc and hme else "模糊" if hac and hnc and not hqu else "模糊" if hnc and not hqu else "偏清楚" if hqc else "普通")
    return {"esg_norm":ep["esg_norm"],"esg_keyword_domain":extract_esg_keyword_domain(text),
            "promise_prior":ep["promise_prior"],"quality_prior":ep["quality_prior"],"timeline_prior":ep["timeline_prior"],
            "has_promise":"是" if hpr else "否","has_future":"是" if hfu else "否","has_action":"是" if hac else "否","has_quant":"是" if hqu else "否",
            "has_report_disclosure":"是" if hrd else "否","has_product_service":"是" if hps else "否","has_routine_commitment":"是" if hrc else "否",
            "has_evidence":"是" if hev else "否","has_measurable_evidence":"是" if hme else "否","has_abstract_evidence":"是" if hae else "否",
            "time_signal":ts,"has_already":"是" if hal else "否","has_near_year":"是" if hny else "否","has_mid_long":"是" if hmg else "否",
            "has_quality_clear":"是" if hqc else "否","has_quality_not_clear":"是" if hnc else "否","has_misleading":"是" if hml else "否","has_law":"是" if hlw else "否",
            "promise_strength":ps,"evidence_strength":es,"clarity_risk":cr}

# ============================================================
# 4. 每個任務的 Metadata 格式（你的前處理 + 學姊的架構）
# ============================================================
# 學姊架構：MultiTaskModel + mean pooling + AMP + joint loss
# 你的貢獻：task-specific metadata prefix（H版驗證最有效）
#   T1: confusion_aware_metadata
#   T2: esg_only
#   T3: task_specific
#   T4: esg_only
# 每筆資料跑 4 次 backbone（各自用自己任務的 text），共享 backbone 權重

def build_esg_only(item):
    feats = extract_rule_metadata(str(item["data"]), get_esg(item))
    return f"[ESG類別={feats['esg_norm']}] {item['data']}"

def build_task_specific(item, task_name):
    text  = str(item["data"])
    feats = extract_rule_metadata(text, get_esg(item))
    if task_name == "promise_status":
        return f"[ESG類別={feats['esg_norm']}] [承諾訊號={feats['has_promise']}] [量化指標={feats['has_quant']}] {text}"
    if task_name == "evidence_status":
        return f"[ESG類別={feats['esg_norm']}] [證據訊號={feats['has_evidence']}] [量化指標={feats['has_quant']}] {text}"
    if task_name == "evidence_quality":
        return (f"[ESG類別={feats['esg_norm']}] [證據訊號={feats['has_evidence']}] "
                f"[量化指標={feats['has_quant']}] [清楚證據詞={feats['has_quality_clear']}] "
                f"[模糊語氣詞={feats['has_quality_not_clear']}] {text}")
    if task_name == "verification_timeline":
        return f"[ESG類別={feats['esg_norm']}] [時間訊號={feats['time_signal']}] [量化指標={feats['has_quant']}] {text}"
    return build_esg_only(item)

def build_confusion_aware(item, task_name):
    text  = str(item["data"])
    feats = extract_rule_metadata(text, get_esg(item))
    if task_name == "promise_status":
        return (f"[任務=承諾語句識別] [ESG類別={feats['esg_norm']}] [ESG承諾先驗={feats['promise_prior']}] "
                f"[未來導向={feats['has_future']}] [行動訊號={feats['has_action']}] "
                f"[制度型承諾={feats['has_routine_commitment']}] [報告揭露描述={feats['has_report_disclosure']}] "
                f"[產品服務描述={feats['has_product_service']}] [承諾強度={feats['promise_strength']}] {text}")
    if task_name == "evidence_quality":
        return (f"[任務=清晰度分類] [ESG類別={feats['esg_norm']}] [ESG品質先驗={feats['quality_prior']}] "
                f"[清楚證據詞={feats['has_quality_clear']}] [模糊語氣詞={feats['has_quality_not_clear']}] "
                f"[法規義務詞={feats['has_law']}] [抽象誤導詞={feats['has_misleading']}] "
                f"[清晰度風險={feats['clarity_risk']}] {text}")
    return build_esg_only(item)

# H版最佳模式：各任務用自己最有效的 metadata
TASK_TEXT_BUILDERS = {
    "promise_status":        lambda item: build_confusion_aware(item, "promise_status"),
    "evidence_status":       lambda item: build_esg_only(item),
    "evidence_quality":      lambda item: build_task_specific(item, "evidence_quality"),
    "verification_timeline": lambda item: build_esg_only(item),
}

# ============================================================
# 4. Dataset（學姊版本：一次餵入所有 label）
# ============================================================

_tokenizer_cache = {}
def get_tokenizer(model_path):
    if model_path not in _tokenizer_cache:
        try:
            _tokenizer_cache[model_path] = AutoTokenizer.from_pretrained(model_path)
            print(f"  ✅ Tokenizer: {model_path.split('/')[-1]}")
        except Exception as e:
            print(f"  ❌ {model_path}: {e}")
            _tokenizer_cache[model_path] = AutoTokenizer.from_pretrained("hfl/chinese-roberta-wwm-ext")
    return _tokenizer_cache[model_path]

class ESGDataset(Dataset):
    """
    每筆資料產生 4 份 task-specific 編碼（你的 metadata）
    backbone 跑 4 次，權重共享（學姊的架構）
    """
    def __init__(self, data, tokenizer, has_labels=True):
        self.data = data; self.tok = tokenizer; self.has_labels = has_labels
    def __len__(self): return len(self.data)
    def __getitem__(self, i):
        x = self.data[i]; item = {}
        for f, build_fn in TASK_TEXT_BUILDERS.items():
            enc = self.tok(build_fn(x), truncation=True, max_length=MAX_LEN,
                           padding="max_length", return_tensors="pt")
            item[f"ids_{f}"]  = enc["input_ids"].squeeze(0)
            item[f"mask_{f}"] = enc["attention_mask"].squeeze(0)
        if self.has_labels:
            item["labels"] = {
                f: torch.tensor(LABEL2ID[f][x[f]], dtype=torch.long)
                for f in EVAL_FIELDS
            }
        return item

# ============================================================
# 5. MultiTaskModel（學姊版本核心）
# ============================================================

class MultiTaskModel(nn.Module):
    """
    每個任務用自己的 task-specific text 跑 backbone：
    - T1 看 confusion_aware text
    - T2 看 esg_only text
    - T3 看 task_specific text
    - T4 看 esg_only text
    backbone 權重共享 → multi-task 正則化
    各任務看最適合自己的 text → 保留 H版 task-specific metadata 優勢
    """
    def __init__(self, model_path, num_labels, dropout=0.2):
        super().__init__()
        try:
            self.bb = AutoModel.from_pretrained(model_path, use_safetensors=False)
        except:
            self.bb = AutoModel.from_pretrained(model_path)
        h = self.bb.config.hidden_size
        self.dp    = nn.Dropout(dropout)
        self.heads = nn.ModuleDict({f: nn.Linear(h, n) for f, n in num_labels.items()})

    def _pool(self, ids, mask):
        """Attention-weighted mean pooling（學姊版本）"""
        last = self.bb(input_ids=ids, attention_mask=mask).last_hidden_state
        m    = mask.unsqueeze(-1).float()
        return (last * m).sum(1) / m.sum(1).clamp(min=1e-9)

    def forward(self, batch):
        """
        每個任務各自的 text 跑一次 backbone
        batch 含 {ids_promise_status, mask_promise_status, ids_evidence_status, ...}
        """
        out = {}
        for f in EVAL_FIELDS:
            ids  = batch[f"ids_{f}"]
            mask = batch[f"mask_{f}"]
            pooled = self._pool(ids, mask)
            out[f] = self.heads[f](self.dp(pooled))
        return out

# ============================================================
# 6. 訓練工具（學姊版本 + AMP）
# ============================================================

class FGM:
    def __init__(self, model, eps=1.0):
        self.m = model; self.eps = eps; self.bak = {}
    def attack(self, n="word_embeddings"):
        for nm, p in self.m.named_parameters():
            if p.requires_grad and n in nm and p.grad is not None:
                self.bak[nm] = p.data.clone()
                nrm = torch.norm(p.grad)
                if nrm != 0 and not torch.isnan(nrm):
                    p.data.add_(self.eps * p.grad / nrm)
    def restore(self, n="word_embeddings"):
        for nm, p in self.m.named_parameters():
            if p.requires_grad and n in nm and nm in self.bak:
                p.data = self.bak[nm]
        self.bak = {}

def build_class_weights(data):
    """Balanced class weight for all 4 tasks"""
    cw = {}
    for f, labs in EVAL_FIELDS.items():
        y  = [LABEL2ID[f][s[f]] for s in data]
        pr = np.unique(y)
        w  = np.ones(len(labs), dtype=np.float32)
        for c, v in zip(pr, compute_class_weight("balanced", classes=pr, y=y)):
            w[c] = v
        cw[f] = torch.tensor(np.clip(w, None, CLASS_WEIGHT_MAX),
                              dtype=torch.float32).to(DEVICE)
    return cw

def train_one_epoch(model, loader, optimizer, scheduler, criterions, scaler, fgm=None):
    """學姊版本訓練：AMP + FGM + 4任務 loss 加總
    每個任務各自的 task-specific text 跑 backbone（backbone 權重共享）
    """
    model.train(); total_loss = 0.0
    for batch in loader:
        # 把 batch 內所有 tensor 送到 GPU
        gpu_batch = {
            k: v.to(DEVICE) if isinstance(v, torch.Tensor) else
               {kk: vv.to(DEVICE) for kk, vv in v.items()}
            for k, v in batch.items()
        }
        lbls = gpu_batch["labels"]
        optimizer.zero_grad()

        with torch.cuda.amp.autocast(enabled=USE_AMP):
            logits = model(gpu_batch)
            loss   = sum(criterions[f](logits[f], lbls[f]) for f in EVAL_FIELDS)

        scaler.scale(loss).backward()

        if fgm:
            fgm.attack()
            with torch.cuda.amp.autocast(enabled=USE_AMP):
                lo2 = model(gpu_batch)
                l2  = sum(criterions[f](lo2[f], lbls[f]) for f in EVAL_FIELDS)
            scaler.scale(l2).backward()
            fgm.restore()

        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer); scaler.update(); scheduler.step()
        total_loss += loss.item()

    return total_loss / max(1, len(loader))

@torch.no_grad()
def predict_probs(model, loader):
    """推論：每個任務各自的 text → {task: ndarray [n, num_labels]}"""
    model.eval()
    out = {f: [] for f in EVAL_FIELDS}
    for batch in loader:
        gpu_batch = {
            k: v.to(DEVICE) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items() if k != "labels"
        }
        with torch.cuda.amp.autocast(enabled=USE_AMP):
            logits = model(gpu_batch)
        for f in EVAL_FIELDS:
            out[f].append(torch.softmax(logits[f].float(), dim=1).cpu().numpy())
    return {f: np.concatenate(out[f], axis=0) for f in EVAL_FIELDS}

def compute_wmf1(gt, pred_dicts):
    """加權 Macro-F1（學姊版本）"""
    s = 0.0
    for f, labs in EVAL_FIELDS.items():
        y_true = [g[f] for g in gt]
        y_pred = [p[f] for p in pred_dicts]
        s += f1_score(y_true, y_pred, labels=labs, average="macro",
                      zero_division=0) * FIELD_WEIGHTS[f]
    return s

def decode_probs(probs, n_items):
    """
    學姊版本 decode：hard rule post-processing
    - ps=No  → vt=N/A, es=N/A, eq=N/A
    - es≠Yes → eq=N/A
    - ps=Yes → vt 從 [already, within_2, between_2_5, more_5] 選
    - es=Yes → eq 從 [Clear, Not Clear, Misleading] 選
    """
    y_idx = LABEL2ID["promise_status"]["Yes"]
    n_idx = LABEL2ID["promise_status"]["No"]

    # verification_timeline: 只在 ps=Yes 時從這4個選
    vt_labels = ["already", "within_2_years", "between_2_and_5_years", "more_than_5_years"]
    vt_idx    = [LABEL2ID["verification_timeline"][l] for l in vt_labels]

    # evidence_status: 只在 ps=Yes 時從 Yes/No 選
    es_labels = ["Yes", "No"]
    es_idx    = [LABEL2ID["evidence_status"][l] for l in es_labels]

    # evidence_quality: 只在 es=Yes 時從這3個選（不含N/A）
    eq_labels = ["Clear", "Not Clear", "Misleading"]
    eq_idx    = [LABEL2ID["evidence_quality"][l] for l in eq_labels]

    out = []
    for i in range(n_items):
        ps_p  = probs["promise_status"][i]
        ps    = "Yes" if ps_p[y_idx] >= ps_p[n_idx] else "No"
        r     = {"promise_status": ps}

        if ps == "No":
            r.update(verification_timeline="N/A",
                     evidence_status="N/A",
                     evidence_quality="N/A")
        else:
            # verification_timeline（排除 N/A）
            vt_p = probs["verification_timeline"][i, vt_idx]
            r["verification_timeline"] = vt_labels[int(np.argmax(vt_p))]

            # evidence_status（排除 N/A）
            es_p = probs["evidence_status"][i, es_idx]
            es   = es_labels[int(np.argmax(es_p))]
            r["evidence_status"] = es

            # evidence_quality（只在 es=Yes 時判斷，排除 N/A）
            if es == "Yes":
                eq_p = probs["evidence_quality"][i, eq_idx]
                r["evidence_quality"] = eq_labels[int(np.argmax(eq_p))]
            else:
                r["evidence_quality"] = "N/A"
        out.append(r)
    return out

# ============================================================
# 7. 可用模型確認
# ============================================================

print("\n確認 7 模型可用性...")
AVAILABLE_MODELS = {}
for short, (path, batch) in SEVEN_MODELS.items():
    try:
        AutoTokenizer.from_pretrained(path)
        AVAILABLE_MODELS[short] = (path, batch)
        print(f"  ✅ {short}: {path} (batch={batch})")
    except Exception as e:
        print(f"  ❌ {short}: 無法載入 ({e})")

print(f"\n可用模型: {len(AVAILABLE_MODELS)}/{len(SEVEN_MODELS)} 個")

# ============================================================
# 8. 核心訓練：1 模型 × 3 Seed × 5-fold → 1 pkl
# ============================================================

def pkl_path(model_short, seed):
    fname = f"M2_{model_short}_s{seed}.pkl"
    local = os.path.join(PKL_DIR_LOCAL, fname)
    drive = os.path.join(PKL_DIR_DRIVE, fname) if DRIVE_AVAILABLE else None
    return local, drive

def train_one_model(model_short, model_path, batch_size):
    """
    7模型 × 3Seed × 5fold 策略的核心：
    對 1 個模型，跑所有 SEEDS × FOLDS，累積 OOF + test probs，
    最後存 1 個 pkl。

    pkl 格式：
    {
      "model": model_short,
      "oof":  {task: ndarray [n_train, num_labels]},  # OOF probs（平均over seeds）
      "test": {task: ndarray [n_test,  num_labels]},  # test probs（平均over seed×fold）
      "cv":   float,   # OOF weighted macro F1
    }
    """
    print(f"\n{'#'*100}")
    print(f"[M2] 訓練模型: {model_short} ({model_path})")
    print(f"      Seeds={ENSEMBLE_SEEDS}, Folds={N_SPLITS}")
    print(f"{'#'*100}")

    # 確認 pkl 是否已存在
    local, drive_p = pkl_path(model_short, "ALL")
    # 重新命名：model_short_ALL = 全部seed合併的結果
    local = os.path.join(PKL_DIR_LOCAL, f"M2_{model_short}.pkl")
    if drive_p: drive_p = os.path.join(PKL_DIR_DRIVE, f"M2_{model_short}.pkl")

    if os.path.exists(local):
        print(f"  ✅ pkl 已存在，跳過: {local}")
        return

    tok = get_tokenizer(model_path)
    test_loader = DataLoader(
        ESGDataset(test_data, tok, has_labels=False),
        batch_size=batch_size, shuffle=False, num_workers=0
    )

    # 累積器（學姊版本：跨 seed×fold 加總 test，跨 seed 加總 oof）
    test_sum = {f: np.zeros((len(test_data), NUM_LABELS[f])) for f in EVAL_FIELDS}
    oof_sum  = {f: np.zeros((len(train_data), NUM_LABELS[f])) for f in EVAL_FIELDS}
    total_runs = 0

    # 用 verification_timeline 做 stratify（學姊版本）
    strat = [d["verification_timeline"] for d in train_data]

    for seed in ENSEMBLE_SEEDS:
        skf   = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
        folds = list(skf.split(range(len(train_data)), strat))
        oof   = {f: np.zeros((len(train_data), NUM_LABELS[f])) for f in EVAL_FIELDS}

        for fold, (tr_idx, va_idx) in enumerate(folds, start=1):
            print(f"\n  ── {model_short} | seed={seed} | fold={fold}/{N_SPLITS} ──")
            set_seed(seed * 100 + fold)

            tr_data  = [train_data[i] for i in tr_idx]
            va_data  = [train_data[i] for i in va_idx]
            tr_loader = DataLoader(ESGDataset(tr_data, tok),
                                   batch_size=batch_size, shuffle=True, num_workers=0)
            va_loader = DataLoader(ESGDataset(va_data, tok, has_labels=False),
                                   batch_size=batch_size, shuffle=False, num_workers=0)

            # 建立模型
            model  = MultiTaskModel(model_path, NUM_LABELS).to(DEVICE)
            cws    = build_class_weights(tr_data)
            crits  = {f: nn.CrossEntropyLoss(weight=cws[f]) for f in EVAL_FIELDS}
            opt    = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
            ts     = len(tr_loader) * EPOCHS
            sch    = get_linear_schedule_with_warmup(opt, int(0.1 * ts), ts)
            scaler = torch.cuda.amp.GradScaler(enabled=USE_AMP)
            fgm    = FGM(model) if USE_FGM else None

            best_score = -1; best_vp = None; best_tp = None

            for ep in range(1, EPOCHS + 1):
                loss = train_one_epoch(model, tr_loader, opt, sch, crits, scaler, fgm)
                vp   = predict_probs(model, va_loader)
                va_preds = decode_probs(vp, len(va_data))
                score = compute_wmf1(va_data, va_preds)
                print(f"    E{ep}: loss={loss:.4f} val_wF1={score:.5f}"
                      f"{'  ★' if score > best_score else ''}")
                if score > best_score:
                    best_score = score
                    best_vp    = vp
                    best_tp    = predict_probs(model, test_loader)

            # 累積
            for f in EVAL_FIELDS:
                oof[f][va_idx] = best_vp[f]
                test_sum[f]   += best_tp[f]
            total_runs += 1

            del model; torch.cuda.empty_cache()
            print(f"    ✅ fold {fold} best_wF1={best_score:.5f}")

        # OOF 累積（across seeds）
        for f in EVAL_FIELDS:
            oof_sum[f] += oof[f]

        # Seed OOF CV
        oof_avg   = {f: oof[f] for f in EVAL_FIELDS}
        oof_preds = decode_probs(oof_avg, len(train_data))
        cv_score  = compute_wmf1(train_data, oof_preds)
        print(f"\n  ✅ {model_short} seed={seed} OOF CV = {cv_score:.5f}")

    # 最終平均
    test_prob = {f: test_sum[f] / total_runs for f in EVAL_FIELDS}
    oof_prob  = {f: oof_sum[f] / len(ENSEMBLE_SEEDS) for f in EVAL_FIELDS}

    oof_preds = decode_probs(oof_prob, len(train_data))
    final_cv  = compute_wmf1(train_data, oof_preds)
    print(f"\n{'='*60}")
    print(f"[M2] {model_short} 全部完成 | Final OOF CV = {final_cv:.5f}")

    # 儲存 pkl
    result = {
        "model": model_short,
        "oof":   oof_prob,
        "test":  test_prob,
        "cv":    final_cv,
        "n_runs": total_runs,
    }
    with open(local, "wb") as f: pickle.dump(result, f)
    print(f"  💾 pkl: {local}")
    if drive_p:
        with open(drive_p, "wb") as f: pickle.dump(result, f)
        print(f"  💾 Drive: {drive_p}")

# ============================================================
# 9. 載入 pkl + Ensemble
# ============================================================

def load_all_pkl():
    """載入所有 7 模型的 pkl"""
    results = {}
    for short in AVAILABLE_MODELS:
        path = os.path.join(PKL_DIR_LOCAL, f"M2_{short}.pkl")
        if not os.path.exists(path) and DRIVE_AVAILABLE:
            path = os.path.join(PKL_DIR_DRIVE, f"M2_{short}.pkl")
        if os.path.exists(path):
            with open(path, "rb") as f:
                data_pkl = pickle.load(f)
            results[short] = data_pkl
            print(f"  ✅ {short}: CV={data_pkl['cv']:.5f}")
        else:
            print(f"  ⚠️ {short}: pkl 找不到")
    return results

def ensemble_test_probs(all_results):
    """
    白板策略：每一筆資料都用 7 個模型的結果集成
    Task2 使用 class-conditional 加權（白板右側）
    其他任務使用等權平均
    """
    if not all_results:
        return None

    n_test = len(test_data)
    # 等權平均（Task1/3/4）
    avg_probs = {f: np.zeros((n_test, NUM_LABELS[f])) for f in EVAL_FIELDS}
    total_w   = 0.0
    for short, res in all_results.items():
        w = 1.5 if short == "large" else 1.0   # large 模型稍微加權
        for f in EVAL_FIELDS:
            avg_probs[f] += w * res["test"][f]
        total_w += w
    for f in EVAL_FIELDS:
        avg_probs[f] /= total_w

    # Task2 class-conditional ensemble（白板右側策略）
    cc_es = np.zeros((n_test, NUM_LABELS["evidence_status"]))
    for label_name, cc_w in [("Yes", TASK2_YES_MODELS), ("No", TASK2_NO_MODELS)]:
        l_idx = LABEL2ID["evidence_status"][label_name]
        tw    = 0.0
        for short, res in all_results.items():
            w = cc_w.get(short, TASK2_DEFAULT_W)
            cc_es[:, l_idx] += w * res["test"]["evidence_status"][:, l_idx]
            tw += w
        cc_es[:, l_idx] /= max(tw, 1e-9)
    # N/A column 從等權平均取
    na_idx = LABEL2ID["evidence_status"]["N/A"]
    cc_es[:, na_idx] = avg_probs["evidence_status"][:, na_idx]
    # Renormalize
    row_sums = cc_es.sum(axis=1, keepdims=True).clip(min=1e-9)
    cc_es /= row_sums
    avg_probs["evidence_status"] = cc_es

    return avg_probs

# ============================================================
# 10. 提交生成
# ============================================================

def load_second_public():
    for p in SECOND_PUBLIC_CSV:
        if os.path.exists(p):
            df = pd.read_csv(p, dtype=str).fillna("N/A")
            print(f"✅ 第二 public: {p}")
            return df
    return None

def sanity_check(df, title):
    print(f"\n{'='*70}\n{title}\n{'='*70}")
    for col in ["promise_status","evidence_status","evidence_quality","verification_timeline"]:
        v=df[col].value_counts(normalize=True)*100; print(f"\n{col}:"); print(v.round(2))
    no_pct = (df["evidence_status"]=="No").mean()*100
    cls_n  = (df["evidence_quality"]=="Clear").sum()
    nc_n   = (df["evidence_quality"]=="Not Clear").sum()
    s2 = "✅" if 13<=no_pct<=14 else "⚠️"
    s3 = "✅" if 1040<=cls_n<=1120 and 200<=nc_n<=240 else "⚠️"
    print(f"\nTask2 No%: {no_pct:.1f}%  {s2}（目標 13~14%）")
    print(f"Task3: Clear={cls_n}, NC={nc_n}  {s3}")

def save_sub(df, fname):
    col_order = ["id","promise_status","verification_timeline","evidence_status","evidence_quality"]
    df = df[col_order]
    df.to_csv(fname, index=False, encoding="utf-8-sig", na_rep="N/A")
    print(f"✅ {fname}")
    if DRIVE_AVAILABLE:
        dp = f"/content/drive/MyDrive/ESGtest/{fname}"
        df.to_csv(dp, index=False, encoding="utf-8-sig", na_rep="N/A")

def make_M2_submissions(all_results):
    if not HAS_TEST or not all_results:
        print("⚠️ test data 或 pkl 缺失"); return

    print(f"\n{'='*80}")
    print("M2 提交生成")
    print(f"{'='*80}")

    ensemble_probs = ensemble_test_probs(all_results)

    # OOF CV 報告
    print("\n各模型 OOF CV：")
    for short, res in sorted(all_results.items(), key=lambda x: -x[1]["cv"]):
        print(f"  {short:10s}: {res['cv']:.5f}")

    # Decode（hard rule post-process）
    preds = decode_probs(ensemble_probs, len(test_data))

    rows = []
    for i, item in enumerate(test_data):
        p = preds[i]
        rows.append({
            "id":                    str(item["id"]),
            "promise_status":        p["promise_status"],
            "verification_timeline": p["verification_timeline"],
            "evidence_status":       p["evidence_status"],
            "evidence_quality":      p["evidence_quality"],
        })
    df_main = pd.DataFrame(rows)
    fname = "submission_test_M2_ensemble_raw.csv"
    save_sub(df_main, fname)
    sanity_check(df_main, "M2 main ensemble")

    # SAFE 分支：Task3/4 用第二 public
    old = load_second_public()
    if old is not None:
        test_ids = [str(x["id"]) for x in test_data]
        old["id"] = old["id"].astype(str)
        old_map   = old.set_index("id")

        # Override Task3/4 with second public
        for i, item in enumerate(test_data):
            tid = str(item["id"])
            if tid in old_map.index:
                rows[i]["evidence_quality"]      = old_map.loc[tid, "evidence_quality"]
                rows[i]["verification_timeline"] = old_map.loc[tid, "verification_timeline"]

        df_safe = pd.DataFrame(rows)
        fname_safe = "submission_test_M2_SAFE_t34_2ndpub.csv"
        save_sub(df_safe, fname_safe)
        sanity_check(df_safe, "M2 SAFE (Task3/4=2nd pub)")
        print(f"\n⭐ 推薦先交：{fname_safe}")
    else:
        print("⚠️ 第二 public CSV 找不到，只輸出 main ensemble 版")

    # 儲存 ensemble pkl（1個最終結果）
    final_pkl = os.path.join(PKL_DIR_LOCAL, "M2_FINAL_ensemble.pkl")
    with open(final_pkl, "wb") as f:
        pickle.dump({"probs": ensemble_probs, "preds": rows}, f)
    print(f"\n✅ Final ensemble pkl: {final_pkl}")
    if DRIVE_AVAILABLE:
        with open(os.path.join(PKL_DIR_DRIVE, "M2_FINAL_ensemble.pkl"), "wb") as f:
            pickle.dump({"probs": ensemble_probs, "preds": rows}, f)

# ============================================================
# 11. 主流程
# ============================================================

print(f"\n{'#'*120}")
print("M2 單模型開始：large MultiTaskModel × 3Seed × 5fold")
print(f"總訓練次數：{len(AVAILABLE_MODELS)} × {len(ENSEMBLE_SEEDS)} × {N_SPLITS} = "
      f"{len(AVAILABLE_MODELS)*len(ENSEMBLE_SEEDS)*N_SPLITS} runs")
print("預計輸出：1 個 pkl 檔案")
print(f"{'#'*120}")

# ── 依序訓練 7 個模型（每個存 1 pkl）──────────────────────
for model_short, (model_path, batch_size) in AVAILABLE_MODELS.items():
    try:
        train_one_model(model_short, model_path, batch_size)
    except Exception as e:
        print(f"❌ {model_short} 訓練失敗: {e}")
        import traceback; traceback.print_exc()

# ── 單模型版本：不在這裡做最終 ensemble ─────────────────────────
print()
print("="*80)
print("✅ 單模型 large 訓練完成。")
print("📌 這份程式只負責產生 /content/M2_pkl/M2_large.pkl")
print("📌 等 7 個模型的 pkl 都完成後，請另外執行 M2_make_ensemble_only.py")
print("="*80)

print("""
╔══════════════════════════════════════════════════════════════════════════╗
║  🎉 M2 large 單模型完成                                                             ║
║                                                                          ║
║  學姊程式碼整合：                                                        ║
║  ✅ MultiTaskModel（1 backbone + 4 heads）                               ║
║  ✅ Attention mean pooling（ELECTRA / LERT 更穩）                        ║
║  ✅ AMP 混合精度（訓練更快）                                             ║
║  ✅ 4任務 loss 加總，互相正則化                                          ║
║  ✅ MAX_LEN=384（更長上下文）                                            ║
║  ✅ 單模型 large × 3 seeds × 5 folds → 1 pkl                           ║
║  ✅ Class-conditional Task2（Yes: roberta+lert+electra / No: large）     ║
║  ✅ Hard rule decode（ps=No → 全N/A, es≠Yes → eq=N/A）                  ║
║                                                                          ║
║  提交優先：                                                              ║
║  ⭐ M2_SAFE（Task3/4 用第二 public）                                     ║
║  🔬 M2_ensemble（Task3/4 用新訓練）                                      ║
╚══════════════════════════════════════════════════════════════════════════╝
""")