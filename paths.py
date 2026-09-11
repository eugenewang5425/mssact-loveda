"""
集中路径配置 — 仓库代码不含本地绝对路径
优先级: 环境变量 > 本地 .env 文件 > 仓库相对默认值

本地使用时在仓库根目录创建 .env（已被 .gitignore 排除）：
    LDA_DATA_ROOT=<本地 LoveDA 数据根目录>
    LDA_CKPT=<检查点目录>
    LDA_GF_DATA=<GF 影像目录（可选，制图用）>
"""
import os

REPO = os.path.dirname(os.path.abspath(__file__))

def _load_dotenv():
    """读取仓库根目录 .env（若存在）"""
    p = os.path.join(REPO, ".env")
    if not os.path.exists(p):
        return
    try:
        for line in open(p, encoding="utf-8"):
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
    except Exception:
        pass

_load_dotenv()

# ---- 数据根目录（包含 newsplit2/ newsplit/ loveda_official/ data_ladder/ train/ val/ test/）----
DATA_ROOT = os.environ.get("LDA_DATA_ROOT", os.path.join(REPO, "_data"))

# ---- 常用数据子目录 ----
DATA_NEWSPLIT2 = os.environ.get("LDA_DATA_NEWSPLIT2", os.path.join(DATA_ROOT, "newsplit2"))
DATA_NEWSPLIT = os.environ.get("LDA_DATA_NEWSPLIT", os.path.join(DATA_ROOT, "newsplit"))
DATA_OFFICIAL = os.environ.get("LDA_DATA_OFFICIAL", os.path.join(DATA_ROOT, "loveda_official"))
DATA_LADDER = os.environ.get("LDA_DATA_LADDER", os.path.join(DATA_ROOT, "data_ladder"))
DATA_LEGACY = os.environ.get("LDA_DATA_LEGACY", DATA_ROOT)   # 早期版本(train/val/test 直挂)

# ---- 训练产物 ----
CKPT = os.environ.get("LDA_CKPT", os.path.join(REPO, "checkpoints"))

# ---- GF 影像（可选，制图管线用）----
GF_DATA = os.environ.get("LDA_GF_DATA", os.path.join(REPO, "_gf_data"))
GF_TIF = os.environ.get("LDA_GF_TIF", os.path.join(GF_DATA, "gf_multiband_labeled.tif"))
GF_SHP = os.environ.get("LDA_GF_SHP", os.path.join(GF_DATA, "admin_boundary.shp"))

# ---- 结果目录（均位于仓库内，可复现生成）----
FIGURES = os.path.join(REPO, "figures")
FULLTILE = os.path.join(REPO, "fulltile_eval")
TTA = os.path.join(REPO, "tta_eval")
TTA2 = os.path.join(REPO, "tta_eval2")
HELDOUT = os.path.join(REPO, "heldout_test")
CONSENSUS = os.path.join(REPO, "consensus_analysis")
OVERFIT = os.path.join(REPO, "overfit_diag")
LADDER_EVAL = os.path.join(REPO, "ladder_eval")
PATCHES = os.path.join(REPO, "patches")
VIS = os.path.join(REPO, "vis")

if __name__ == "__main__":
    for k in ["REPO","DATA_ROOT","DATA_NEWSPLIT2","CKPT","GF_DATA"]:
        print(f"{k:18s} = {globals()[k]}")
