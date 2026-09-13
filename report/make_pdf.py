# -*- coding: utf-8 -*-
"""报告 HTML -> PDF（Chrome headless）+ 页数与关键内容自检

make_report_v3.py 只产出 HTML，PDF 需要 Chrome headless 转换。本脚本把它固定下来，
并做两项自检：页数、以及必须出现的勘误/回退关键词（防止再次输出错误结论）。

用法: python make_pdf.py [html路径]
"""

# --- 路径引导（本文件位于子目录 report/，仓库根为上一级）---
# 说明: paths.py / train_v3.py / experiment_matrix*.py 保留在仓库根目录，
#       故须把仓库根加入 sys.path；同目录模块（如 queue_guard）用 _HERE。
import os as _os, sys as _sys
import tempfile
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_BASE = _os.path.dirname(_HERE)          # 仓库根
for _p in (_BASE, _HERE):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
# --- 路径引导结束 ---
import os, sys, subprocess, glob, time, tempfile

sys.path.insert(0, _HERE)
import paths

BASE = paths.REPO
CANDIDATES = [
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
]
# 报告必须包含的关键内容 (防止回退为错误结论或丢失关键发现)
MUST_CONTAIN = [
    # ---- 关键勘误与更正（不得被静默改回）----
    "勘误",                       # 边界统计与类别占比的勘误段
    "2.36%",                      # 已更正的边界像素占比（原误报 93%）
    "37.15%",                     # 已更正的验证集背景占比（原误报 45.6%）
    "核验更正", "weights_backbone", "预训练主干",   # DeepLab 预训练混淆的取证
    "0.0039", "0.78σ",            # 从零对照的主协议结论
    # ---- 核心结论的存在性守卫 ----
    "死计算", "置换等变",          # 三处架构缺陷（D1/D3）
    "4.563",                      # D2 的过度碎裂实测（4.56×）
    "σ_seed = 0.0050", "0.87 倍", # 种子方差实测与消融散布之比
    "杠杆排序",                    # 全文最凝练的结论
    "3.8–4.5σ", "更优的工程取舍", "4.4×",   # 裁剪策略的效应量与取舍
    "4/4 一致",                    # ECSAM 移除在四种设置下一致更好
    "5.11",                       # 可分辨性总表
    # ---- 章节存在性 ----
    "5.6 架构有效性分析", "5.7 随机种子噪声底线", "5.8 训练策略对比",
    "5.9 数据量阶梯", "5.10 预算攻击",
]


def find_browser():
    for c in CANDIDATES:
        if os.path.exists(c):
            return c
    return None


def main():
    html = sys.argv[1] if len(sys.argv) > 1 else None
    if html is None:
        hs = sorted(glob.glob(os.path.join(paths.REPORTS, "项目报告_*.html")), key=os.path.getmtime)
        if not hs:
            print("未找到 项目报告_*.html，请先运行 make_report_v3.py"); raise SystemExit(1)
        html = hs[-1]
    pdf = os.path.splitext(html)[0] + ".pdf"
    br = find_browser()
    if br is None:
        print("未找到 Chrome/Edge，无法生成 PDF"); raise SystemExit(1)

    # Chrome 需要 file:// URI
    uri = "file:///" + os.path.abspath(html).replace("\\", "/")
    # 必须显式指定 user-data-dir: 用户自己的 Chrome 正占用默认 profile,
    # 否则 headless 实例会因等不到 profile 锁而挂起 (实测 600s 超时)
    # Chrome 需要独立的 user-data-dir（用户自身的 Chrome 占用默认 profile，
    # 否则 headless 会挂起）。用系统临时目录，避免写入仓库或写死本机路径。
    udd = os.path.join(tempfile.gettempdir(), "mssact_headless")
    os.makedirs(udd, exist_ok=True)
    print(f"浏览器 : {br}")
    print(f"HTML   : {os.path.basename(html)}")
    t0 = time.time()
    # --no-pdf-header-footer 必须保留：Chrome 默认会在页脚打印 file:// 源路径，
    # 而 PDF 是交付件，会把本机路径带给读者（此前已实际发生过）。
    cmd = [br, "--headless=new", "--disable-gpu", f"--user-data-dir={udd}",
           "--no-first-run", "--no-pdf-header-footer",
           f"--print-to-pdf={pdf}", uri]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    except subprocess.TimeoutExpired:
        print("Chrome 超时 (300s)。请确认没有其他 Chrome 实例占用 profile，"
              "或检查 user-data-dir 是否可写：")
        print(f"  {udd}")
        raise SystemExit(1)
    if not os.path.exists(pdf):
        print("Chrome 未产出 PDF，stderr:")
        print(r.stderr[-2000:])
        raise SystemExit(1)
    size = os.path.getsize(pdf)

    # 自检
    from pypdf import PdfReader
    n_pages = len(PdfReader(pdf).pages)
    txt = ""
    for p in PdfReader(pdf).pages:
        try:
            txt += p.extract_text() or ""
        except Exception:
            pass
    print(f"PDF    : {os.path.basename(pdf)}  {size:,} bytes  {n_pages} 页  ({time.time()-t0:.0f}s)")
    print("内容自检:")
    # 页脚泄露检查：PDF 不得含本机 file:// 路径
    if "file:///" in txt or "file://" in txt:
        print("   ✗ PDF 含 file:// 页脚（本机路径泄露）—— 请确认 --no-pdf-header-footer 生效")
        ok = False
    else:
        print("   ✓ 无 file:// 页脚（无本机路径泄露）")
    ok = True
    for k in MUST_CONTAIN:
        hit = k in txt
        ok &= hit
        print(f"   {'✓' if hit else '✗'} {k}")
    print("自检通过" if ok else "!! 自检未通过：报告可能仍含错误结论")
    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
