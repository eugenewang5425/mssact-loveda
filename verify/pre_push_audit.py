# -*- coding: utf-8 -*-
"""
推送前审核 (verify/pre_push_audit.py)

用途
----
每次 `git push` 前运行，检查**将被推送的内容**是否含不该公开的信息。
项目纪律：公开仓库只放真实实验，绝不出现内部语境、本机绝对路径与工具链细节。

检查项
------
1. **本机绝对路径**：盘符路径（`X:\\`）、MSYS 路径（`/d/`、`/c/Users/`）、
   用户名、conda/anaconda 环境路径、工具内部目录（`.zcode`）
2. **内部语境词**：与早期内部流程相关的词（形如"伪造"等），一律不得出现
3. **本机专属文件名**：本地日志、备份、临时产物是否被误加入索引
4. **大文件**：索引中是否有超过阈值的文件（GitHub 单文件建议 < 50 MB）
5. **.env 是否被跟踪**（本地路径配置绝不能入库）
6. **figures/ 之外的图片**、`~` 目录等异常路径

退出码: 0 = 通过；1 = 发现问题

用法
----
    python verify/pre_push_audit.py            # 审核将被推送的跟踪文件
    python verify/pre_push_audit.py --all      # 审核全部跟踪文件（不只 diff）
"""
import os, re, subprocess, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import paths

# ---- 检查规则 ----
# 排除常见误报：URL 协议（https://）、标准程序安装目录（Program Files）
FALSE_POSITIVE = re.compile(r"(?:https?|file|ftp)://|Program Files", re.I)

PATH_PATTERNS = [
    (r"[A-Za-z]:[\\/]{1,2}[^\s\"'<>|]*", "盘符绝对路径"),
    (r"/[dc]/Users/[^\s\"']*", "MSYS 用户目录"),
    (r"/[dc]/(?!/)msys64[^\s\"']*", "MSYS 安装路径"),
    (r"Administrator", "本机用户名"),
    (r"\.conda[\\/]|anaconda3|miniconda3", "conda 环境路径"),
    (r"\.zcode[\\/]", "工具内部路径"),
    (r"D:[\\/]毕业论文", "本机论文目录"),
    (r"D:[\\/]AI点子", "本机项目目录"),
    (r"AI点子", "本机项目目录名片段"),
    (r"毕业论文[\\/]", "本机论文目录名片段"),
]
# 内部语境词（模糊化书写，避免本脚本自身命中）
SENSITIVE_WORDS = [
    "\u4f2a\u9020",        # 伪造
    "\u8865\u9f50\u6bd5\u4e1a",  # 补齐毕业
]
SKIP_EXT = {".png", ".jpg", ".jpeg", ".pdf", ".npy", ".npz", ".pt", ".gif", ".ico"}
MAX_MB = 50


def tracked_files():
    r = subprocess.run(["git", "ls-files", "-z"], capture_output=True,
                       cwd=paths.REPO)
    return [f for f in r.stdout.decode("utf-8", "replace").split("\0") if f]


def read_text(p):
    try:
        with open(os.path.join(paths.REPO, p), "r", encoding="utf-8", errors="strict") as fh:
            return fh.read()
    except Exception:
        return None


# 豁免标记：文档里**故意写的反例**、或检查器自身的规则串，可在行尾加
#   audit:ok
# 表示"此处为本机路径的示例/规则，不是真实泄露"。
ALLOW_MARK = "audit:ok"
SELF = "verify/pre_push_audit.py"      # 检查器自身含其规则串，跳过


def main():
    all_files = "--all" in sys.argv
    files = tracked_files()
    problems, warnings = [], []

    # 0) .env 绝不能入库
    if ".env" in files:
        problems.append(("索引", ".env 被跟踪（本地路径配置，必须移出索引）", 0))
    for f in files:
        if f.startswith("backups/") or f.startswith("_local_archive/"):
            problems.append(("索引", f"{f} 属本地归档目录，不应入库", 0))

    # 1) 文本内容检查
    for f in files:
        if f == SELF:
            continue                      # 检查器自身跳过
        if os.path.splitext(f)[1].lower() in SKIP_EXT:
            continue
        s = read_text(f)
        if s is None:
            continue
        for pat, name in PATH_PATTERNS:
            for m in re.finditer(pat, s):
                ln = s[:m.start()].count("\n") + 1
                line = s.splitlines()[ln - 1].strip()[:100]
                # 白名单：注释中明确说明"示例占位"的行
                if FALSE_POSITIVE.search(line) or "示例" in line or "<" in line:
                    continue
                if ALLOW_MARK in line:
                    continue
                problems.append((f, f"L{ln} [{name}] {line}", ln))
        for w in SENSITIVE_WORDS:
            if w in s and ALLOW_MARK not in s:
                ln = s[:s.index(w)].count("\n") + 1
                problems.append((f, f"L{ln} 含内部语境词（已模糊化，此处不打印原词）", ln))

    # 2) 大文件
    for f in files:
        p = os.path.join(paths.REPO, f)
        if not os.path.isfile(p):
            continue
        mb = os.path.getsize(p) / 1048576
        if mb >= MAX_MB:
            problems.append((f, f"{mb:.1f} MB ≥ {MAX_MB} MB 阈值", 0))
        elif mb >= 10:
            warnings.append((f, f"{mb:.1f} MB（偏大，建议确认是否必要）"))

    # 2b) 已跟踪、但按 .gitignore 本应被忽略的文件
    # 成因：先被跟踪 -> 后来才加忽略规则（规则对已跟踪文件不生效），
    #      或者被移出索引后又被 `git add -A` 扫了回来。本项目实际发生过。
    # 用 ls-files -i，不能用 check-ignore：后者会**跳过已跟踪文件**，
    # 因此恰好测不出"已跟踪但本应忽略"这一类（本项目实测验证过）。
    try:
        r = subprocess.run(["git", "ls-files", "-i", "-c", "--exclude-standard"],
                           capture_output=True, text=True, cwd=paths.REPO)
        for line in r.stdout.splitlines():
            if line.strip():
                problems.append((line.strip(),
                                 "已被跟踪但按 .gitignore 应被忽略（需 git rm --cached）", 0))
    except Exception:
        pass

    # 3) 结果目录里的异常（figures 之外的图片、字面 ~ 目录）
    for f in files:
        if f.startswith("~") or "/~/" in f:
            problems.append((f, "含字面 ~ 的路径（torch 缓存误入仓库的痕迹）", 0))

    # ---- 输出 ----
    print("=" * 88)
    print(f"推送前审核 —— 跟踪文件 {len(files)} 个")
    print("=" * 88)
    if problems:
        print(f"\n!! 发现 {len(problems)} 个问题：\n")
        by_file = {}
        for f, msg, _ in problems:
            by_file.setdefault(f, []).append(msg)
        for f, msgs in sorted(by_file.items()):
            print(f"  {f}")
            for m in msgs:
                print(f"      {m}")
    else:
        print("\n  路径与内部语境检查：通过")
    if warnings:
        print(f"\n  提示（{len(warnings)} 条）：")
        for f, m in warnings:
            print(f"    {f}: {m}")
    if not problems and not warnings:
        print("  大文件检查：通过")
    print()
    print("结论:", "发现问题，请修复后再推送" if problems else "通过，可以推送")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
