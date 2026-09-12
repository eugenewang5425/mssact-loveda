# -*- coding: utf-8 -*-
"""报告 HTML -> PDF（Chrome headless）+ 页数与关键内容自检

make_report_v2.py 只产出 HTML，PDF 需要 Chrome headless 转换。本脚本把它固定下来，
并做两项自检：页数、以及必须出现的勘误/回退关键词（防止再次输出错误结论）。

用法: python make_pdf.py [html路径]
"""
import os, sys, subprocess, glob, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paths

BASE = paths.REPO
CANDIDATES = [
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
]
# 报告必须包含的关键内容 (防止回退为错误结论或丢失关键发现)
MUST_CONTAIN = ["回退验证", "边界像素占比", "勘误",
                "5.6 架构有效性分析", "死计算", "置换等变"]


def find_browser():
    for c in CANDIDATES:
        if os.path.exists(c):
            return c
    return None


def main():
    html = sys.argv[1] if len(sys.argv) > 1 else None
    if html is None:
        hs = sorted(glob.glob(os.path.join(BASE, "项目报告_*.html")), key=os.path.getmtime)
        if not hs:
            print("未找到 项目报告_*.html，请先运行 make_report_v2.py"); raise SystemExit(1)
        html = hs[-1]
    pdf = os.path.splitext(html)[0] + ".pdf"
    br = find_browser()
    if br is None:
        print("未找到 Chrome/Edge，无法生成 PDF"); raise SystemExit(1)

    # Chrome 需要 file:// URI
    uri = "file:///" + os.path.abspath(html).replace("\\", "/")
    # 必须显式指定 user-data-dir: 用户自己的 Chrome 正占用默认 profile,
    # 否则 headless 实例会因等不到 profile 锁而挂起 (实测 600s 超时)
    udd = os.path.join(paths.DATA_ROOT.replace(os.sep, "/").split(":/")[0] + ":/AI点子/_cr_headless")
    os.makedirs(udd, exist_ok=True)
    print(f"浏览器 : {br}")
    print(f"HTML   : {os.path.basename(html)}")
    t0 = time.time()
    cmd = [br, "--headless=new", "--disable-gpu", f"--user-data-dir={udd}",
           "--no-first-run", f"--print-to-pdf={pdf}", uri]
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
    ok = True
    for k in MUST_CONTAIN:
        hit = k in txt
        ok &= hit
        print(f"   {'✓' if hit else '✗'} {k}")
    print("自检通过" if ok else "!! 自检未通过：报告可能仍含错误结论")
    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
