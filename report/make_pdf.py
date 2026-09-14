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
CHROME_TIMEOUT = 900      # 秒; 42 页 + 内嵌图片, 训练并行时会更慢

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
    # ---- 缺陷修复（2026-09-13）----
    "6,099,144",                  # 真实 nn.Parameter 计数（原误用含 BN buffer 的 6,101,784）
    "29.98%",                     # FPN 死参数占真实参数之比
    "ecsam[0]",                   # D5 死模块
    "0.000e+00",                  # 修复的向后兼容证据：默认路径逐位一致
    "G4-MEM-ROLL-120ep",          # 120 轮可比组（禁止与 60 轮混比）
    "tost_equivalence.json",      # TOST 一等产物
    "5.6.13",                     # 修复的性能效果是负面的（第一轮实测）
    "过拟合假设已被推翻",           # 欠拟合而非过拟合的训练 loss 证据
    "非加性",                      # 两项修复合起来比单项更差
    "5.6.14",                     # 修复后消融换底 + 数值发散
    "有害的成分是跳连本身",         # H1/H2 双证伪后的稳健结论
    "1.26e4",                     # 发散的确证证据（权重爆炸）
    "一条被推翻的归因",             # DeepLab +0.0720 曾驱动出实测有害的跳连改动
    "5.6.15",                     # 首次给出方向可解释的消融表
    "可解释性",                    # 位置编码修复的价值不在表头 Kappa 而在可解释性
    "条件性的训练稳定器",           # ECSAM 在修复后架构上的作用
    "5.6.16",                     # 消融"平坦"依协议而变
    "依协议而变",                  # 同上（正文用词）
    "符号相反",                    # 6/8 变体在两协议下方向相反
    "5.6.17",                     # 协议解剖
    "符号翻转由图像集驱动",         # 解剖的结论 (守卫短语不能含 markdown 标记, PDF 文本里没有)
    "protocol_dissection",        # 产物文件
    # ---- 章节存在性 ----
    "5.6 架构有效性分析", "5.6.12", "5.7 随机种子噪声底线", "5.8 训练策略对比",
    "5.9 数据量阶梯", "5.10 预算攻击",
]


def find_browser():
    for c in CANDIDATES:
        if os.path.exists(c):
            return c
    return None


def _kill_tree(pid):
    """只杀本脚本启动的那棵进程树（其 user-data-dir 唯一, 不会影响用户的 Chrome）"""
    try:
        if os.name == "nt":
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(pid)],
                           capture_output=True, timeout=30)
        else:
            os.kill(pid, 15)
    except Exception:
        pass


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
    # 每次调用用**独立**的 profile 目录: 共用同一个目录时, 上一次的 headless
    # 实例若未完全退出就会留下 profile 锁, 使下一次渲染直接挂死到超时。
    # 独立目录的代价只是冷启动略慢, 换来的是"绝不因残留锁而失败"。
    udd = os.path.join(tempfile.gettempdir(), "mssact_headless_%d" % os.getpid())
    os.makedirs(udd, exist_ok=True)
    print(f"浏览器 : {br}")
    print(f"HTML   : {os.path.basename(html)}")
    t0 = time.time()
    # --no-pdf-header-footer 必须保留：Chrome 默认会在页脚打印 file:// 源路径，
    # 而 PDF 是交付件，会把本机路径带给读者（此前已实际发生过）。
    # 先渲染到临时文件: 自检不通过时**不留**半成品 PDF（否则读者会引用到带错的版本）
    tmp_pdf = pdf + ".tmp_rendering"
    cmd = [br, "--headless=new", "--disable-gpu", f"--user-data-dir={udd}",
           "--no-first-run", "--no-pdf-header-footer",
           f"--print-to-pdf={tmp_pdf}", uri]
    # 不依赖 Chrome 退出：实测它常常**写完 PDF 却不退出**（headless 的已知行为），
    # 于是 subprocess.run 一直等到超时，而结果其实已经完整落盘 (3.1MB)。
    # 旧写法因此把好端端的 PDF 扔掉、报"超时"—— 之前几次成功只是碰巧它退出了。
    # 改为轮询输出文件：出现且大小连续两次探测不变即认为写完，随后只杀我们自己
    # 启动的那棵进程树（有唯一 user-data-dir，不会碰到用户的 Chrome）。
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    t0 = time.time()
    last_size, stable = -1, 0
    while time.time() - t0 < CHROME_TIMEOUT:
        if os.path.exists(tmp_pdf):
            cur = os.path.getsize(tmp_pdf)
            if cur > 0 and cur == last_size:
                stable += 1
                if stable >= 2:            # 连续两次(约 6s)不变 -> 写完
                    break
            else:
                stable = 0
            last_size = cur
        time.sleep(3)
    else:
        print("Chrome 在 %ds 内未写出 PDF。请确认 user-data-dir 可写：" % CHROME_TIMEOUT)
        print(f"  {udd}")
        _kill_tree(proc.pid)
        raise SystemExit(1)
    _kill_tree(proc.pid)
    if not os.path.exists(tmp_pdf) or os.path.getsize(tmp_pdf) == 0:
        print("Chrome 未产出 PDF")
        raise SystemExit(1)
    size = os.path.getsize(tmp_pdf)
    print(f"渲染   : {size:,} bytes ({time.time()-t0:.0f}s; 按文件写稳定判定，不依赖 Chrome 退出)")

    # 自检
    from pypdf import PdfReader
    n_pages = len(PdfReader(tmp_pdf).pages)
    txt = ""
    for p in PdfReader(tmp_pdf).pages:
        try:
            txt += p.extract_text() or ""
        except Exception:
            pass
    print(f"PDF    : {os.path.basename(pdf)}  {size:,} bytes  {n_pages} 页  ({time.time()-t0:.0f}s)")
    print("内容自检:")
    fails = []
    # 页脚泄露检查：PDF 不得含本机 file:// 路径
    # 注意: 这里必须只 append 失败项, 不能再用一个 `ok = True` 覆盖前面结果 ——
    # 此前正是那行 `ok = True` 把本检查的 ok=False 抹掉了, 使整条 file:// 防线失效。
    if "file://" in txt:
        print("   ✗ PDF 含 file:// 页脚（本机路径泄露）—— 请确认 --no-pdf-header-footer 生效")
        fails.append("file:// 页脚泄露")
    else:
        print("   ✓ 无 file:// 页脚（无本机路径泄露）")
    # 匹配前剥离**全部**空白: PDF 抽取会在任意位置断行/加空格, 例如
    # 「工程取舍」被抽成「工程取 [换行] 舍」, 「sigma_seed = 0.0050」被抽成
    # 「sigma_seed = [换行] 0.0050」。逐条改写短语是打地鼠; 统一去空白可一次性
    # 消除这一整类假阴性。
    squash = lambda t: "".join(t.split())
    flat = squash(txt)
    for k in MUST_CONTAIN:
        hit = squash(k) in flat
        if not hit:
            fails.append(k)
        print(f"   {'✓' if hit else '✗'} {k}")
    if fails:
        print(f"!! 自检未通过（{len(fails)} 项）：报告可能仍含错误结论")
        print(f"   未通过: {fails}")
        print(f"   临时 PDF 保留在 {tmp_pdf} 供排查；**未**覆盖 {os.path.basename(pdf)}")
        return 2
    # 自检通过后才落盘；覆盖前备份旧 PDF（命名纪律: 覆盖前必备份）
    if os.path.exists(pdf):
        # 备份放 backups/（该目录已被 .gitignore 忽略），不要堆在 reports/ 里
        bak_dir = os.path.join(os.path.dirname(pdf), "..", "backups")
        os.makedirs(bak_dir, exist_ok=True)
        bak = os.path.join(bak_dir, os.path.basename(pdf)[:-4]
                           + "_bak_%s.pdf" % time.strftime("%Y%m%d_%H%M"))
        os.replace(pdf, bak)
        print(f"   旧 PDF 已备份 -> backups/{os.path.basename(bak)}")
    os.replace(tmp_pdf, pdf)
    print("自检通过")
    print(f"输出   : {pdf}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
