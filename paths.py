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

# ---- Windows 上 torch._inductor 的 getpass 陷阱（必须最先处理）----
# 现象: 训练报 "ModuleNotFoundError: No module named 'pwd'" 或
#       "AssertionError: Artifact of type=precompile already registered ..."
# 根因: torch.optim.AdamW.add_param_group -> import torch._dynamo ->
#       _dynamo/package.py 建 DiskDynamoCache -> _inductor cache_dir_utils.cache_dir()
#       -> default_cache_dir() -> getpass.getuser()。getpass 先查环境变量
#       LOGNAME/USER/LNAME/USERNAME，四个都缺失时才 `import pwd` —— 而 pwd 是
#       Unix 专有模块，Windows 上不存在。
#       由 nohup/计划任务等启动时环境被剥离，恰好会缺这几个变量，于是训练在
#       建优化器时崩溃（表现为 0.2 分钟即失败）。
# 修法: cache_dir() 的逻辑是"若 TORCHINDUCTOR_CACHE_DIR 已存在则不再调用
#       default_cache_dir()"，故显式设定即可完全绕过该失败路径。
os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", os.path.join(REPO, ".torch_inductor_cache"))

# ---- 同理: 防止 torch.hub 把预训练权重下进仓库 ----
# HOME/USERPROFILE 缺失时 os.path.expanduser("~") **原样返回 "~"**，torch.hub 便以
# 相对路径 ./~/ 建缓存（基准是 CWD = 仓库根），实测把 97MB 的 resnet50 权重下进
# 仓库并被 git add -A 误提交（.git 由 63MB 涨到 161MB）。
# 显式指定 TORCH_HOME 即可根治：优先复用真实用户目录，拿不到就退到仓库内
# .torch_cache/（已在 .gitignore 排除）。
_home = os.environ.get("USERPROFILE") or os.environ.get("HOME")
os.environ.setdefault("TORCH_HOME",
                      os.path.join(_home, ".cache", "torch") if _home
                      else os.path.join(REPO, ".torch_cache"))

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
