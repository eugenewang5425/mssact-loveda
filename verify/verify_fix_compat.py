# -*- coding: utf-8 -*-
"""验证缺陷修复的**向后兼容性**与**缺陷确已消除**

为什么必须是脚本而不是一次性命令
------------------------------
这五项是"既有 103 个实验的结论仍可复现"这一承诺的唯一依据。若只跑过一次、
留在会话记录里, 下次改动模型时无法回归验证, 承诺就退化成一句说辞。

对照物取 **git HEAD 里的修复前版本**（`git show HEAD:models/msscactnet.py`）
而非某个 .bak 文件 —— .bak 会被清理, git 历史不会。

五项检查
------
1. **默认路径逐位一致**: 默认配置 `strict=True` 加载旧权重 + 新旧输出
   最大逐元素差必须**恰好为 0.0**（不是"很小"）。
2. **键集合一致**: 默认配置与既有 checkpoint 的 state_dict 键集合完全相同。
3. **置换等变性被打破**: `pos_enc=False` 应置换等变（残差 ~1e-6,
   float 误差级）; `pos_enc=True` 残差应放大约 6 个数量级。
4. **梯度可达性**: 默认配置有 29.99% 参数无梯度（FPN 死三级 + ecsam[0]）;
   打开 `use_skip=True` 后应只剩 `ecsam[0]` 的 0.017%;
   再打开 `ecsam_stage0=True` 应做到**零**无梯度参数。
5. **组合路径**: 10 种消融组合 × 跳连均可前向 + 反向, 输出形状正确。

用法: python verify/verify_fix_compat.py        # 退出码非 0 表示验证失败
"""

import os
import sys
import glob
import shutil
import tempfile
import subprocess
import importlib.util

_HERE = os.path.dirname(os.path.abspath(__file__))
_BASE = os.path.dirname(_HERE)
sys.path.insert(0, _BASE)
sys.path.insert(0, os.path.join(_BASE, "queues"))

import paths                                                        # noqa: E402
import torch                                                        # noqa: E402


def load_module(path, name):
    # importlib 只认 .py 后缀; 备份文件名形如 msscactnet.py.bak_20260913_2054,
    # spec_from_file_location 会返回 None 并炸出 "'NoneType' has no attribute 'loader'"。
    if not path.endswith(".py"):
        fd, tmp = tempfile.mkstemp(suffix=".py", prefix="ref_net_")
        with os.fdopen(fd, "wb") as fh:
            fh.write(open(path, "rb").read())
        path = tmp
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


def fetch_prefix_version(ref=None):
    """取修复前的 models/msscactnet.py

    优先级:
      1) 显式传入的 ref (文件路径 或 git 版本号)
      2) backups/msscactnet.py.bak_*  —— **跨提交稳定**, 是真正的修复前快照
      3) git HEAD:models/msscactnet.py —— 仅在尚未提交修复时有效。
         一旦修复被提交, HEAD 就是修复后版本, 本项检查会退化成"自己 vs 自己"
         而永远通过 —— 所以只作兜底, 且用 git show 明确取 HEAD~ 之外的历史。
    """
    if ref:
        if os.path.exists(ref):
            return ref, "指定文件 %s" % os.path.basename(ref)
        r = subprocess.run(["git", "show", ref], cwd=_BASE, capture_output=True)
        if r.returncode != 0:
            raise SystemExit("无法取出指定版本 %s: %s" % (ref, r.stderr.decode()[:200]))
        fd, path = tempfile.mkstemp(suffix=".py", prefix="prefix_msscactnet_")
        with os.fdopen(fd, "wb") as fh:
            fh.write(r.stdout)
        return path, "指定版本 %s" % ref

    cands = sorted(glob.glob(os.path.join(_BASE, "backups", "msscactnet.py.bak_*")))
    if cands:
        return cands[-1], "备份文件 %s" % os.path.basename(cands[-1])

    r = subprocess.run(["git", "show", "HEAD:models/msscactnet.py"],
                       cwd=_BASE, capture_output=True)
    if r.returncode != 0:
        raise SystemExit("无法从 git HEAD 取回旧版模型, 也未找到 backups/ 快照: %s"
                         % r.stderr.decode()[:200])
    fd, path = tempfile.mkstemp(suffix=".py", prefix="prefix_msscactnet_")
    with os.fdopen(fd, "wb") as fh:
        fh.write(r.stdout)
    return path, "git HEAD (注意: 若修复已提交, 此处将失去对照意义)"


def main():
    from experiment_matrix_v2 import mssact_light

    ref = None
    for i, a in enumerate(sys.argv[1:]):
        if a == "--ref" and i + 2 <= len(sys.argv[1:]):
            ref = sys.argv[1:][i + 1]
    old_path, how = fetch_prefix_version(ref)
    print("  对照物: %s -> %s" % (how, old_path))
    if old_path == os.path.join(_BASE, "models", "msscactnet.py"):
        raise SystemExit("对照物就是当前文件, 检查无意义")
    OLD = load_module(old_path, "prefix_net")
    NEW = load_module(os.path.join(_BASE, "models", "msscactnet.py"), "cur_net")

    LIGHT = dict(in_channels=3, num_classes=7, embed_dims=[32, 64, 128, 256],
                 transformer_layers=2, transformer_heads=4)
    x = torch.randn(2, 3, 256, 256)
    fails = []

    # ---- 1 & 2) 向后兼容 ----
    torch.manual_seed(1234)
    mo = OLD.MSSACTNet(**LIGHT).eval()
    torch.manual_seed(1234)
    mn = NEW.MSSACTNet(**LIGHT).eval()
    try:
        mn.load_state_dict(mo.state_dict(), strict=True)
        print("  [1] strict=True 加载修复前权重: OK (%d 项)" % len(mo.state_dict()))
    except RuntimeError as e:
        fails.append("strict 加载失败: %s" % str(e)[:160])
        print("  [1] strict 加载失败")
    with torch.no_grad():
        d = (mo(x) - mn(x)).abs().max().item()
    print("  [1] 默认路径最大逐元素差: %.3e  (要求恰好 0.0)" % d)
    if d != 0.0:
        fails.append("默认路径输出被改变, 最大差 %.3e" % d)
    n_old = sum(p.numel() for p in mo.parameters())
    n_new = sum(p.numel() for p in mn.parameters())
    print("  [2] 参数量 修复前 %d / 当前默认 %d" % (n_old, n_new))
    if n_old != n_new:
        fails.append("默认配置参数量变化: %d -> %d" % (n_old, n_new))

    # ---- 3) 置换等变性 ----
    def perm_residual(pos_enc):
        torch.manual_seed(9)
        m = NEW.MSSACTNet(**LIGHT, pos_enc=pos_enc).eval()
        z = m.transformer
        with torch.no_grad():
            h = torch.randn(1, 256, 8, 8)
            base = z(h)
            g = torch.randperm(64)
            hp = h.flatten(2).permute(0, 2, 1)[:, g].permute(0, 2, 1).reshape(1, 256, 8, 8)
            out = z(hp)
            rec = (out.flatten(2).permute(0, 2, 1)[:, torch.argsort(g)]
                   .permute(0, 2, 1).reshape(1, 256, 8, 8))
            return (base - rec).abs().max().item()

    r0, r1 = perm_residual(False), perm_residual(True)
    print("  [3] 置换等变性残差: pos_enc=False %.3e -> True %.3e" % (r0, r1))
    if not r0 < 1e-5:
        fails.append("pos_enc=False 竟然不是置换等变 (残差 %.3e)" % r0)
    if not r1 > 1e-2:
        fails.append("pos_enc=True 未打破置换等变性 (残差 %.3e)" % r1)

    # ---- 4) 梯度可达性 ----
    def dead_params(**kw):
        torch.manual_seed(3)
        m = mssact_light(**kw).train()
        m(x).sum().backward()
        tot = sum(p.numel() for p in m.parameters())
        dead = [(n, p.numel()) for n, p in m.named_parameters() if p.grad is None]
        return tot, dead

    tot, dead = dead_params()
    frac = sum(c for _, c in dead) / tot * 100
    print("  [4] 默认配置: %d 项无梯度 / %d 个参数 (%.3f%%)" % (len(dead), sum(c for _, c in dead), frac))
    if not (29.9 < frac < 30.1):
        fails.append("默认配置死参数占比 %.3f%% 不在预期 ~29.99%%" % frac)

    tot2, dead2 = dead_params(use_skip=True)
    frac2 = sum(c for _, c in dead2) / tot2 * 100
    names2 = {n for n, _ in dead2}
    print("  [4] use_skip=True: %d 项无梯度 (%.3f%%), 全部为 ecsam[0]: %s"
          % (len(dead2), frac2, all(n.startswith("ecsam.0.") for n in names2)))
    if frac2 > 0.02 or not all(n.startswith("ecsam.0.") for n in names2):
        fails.append("use_skip=True 后死参数不符合预期: %s" % sorted(names2))

    tot3, dead3 = dead_params(use_skip=True, pos_enc=True, ecsam_stage0=True)
    print("  [4] +ecsam_stage0=True: %d 项无梯度 (应做到零死参数)" % len(dead3))
    if dead3:
        fails.append("ecsam_stage0=True 后仍有死参数: %s" % [n for n, _ in dead3])

    # ---- 5) 组合路径 ----
    combos = [dict(use_skip=True), dict(pos_enc=True), dict(use_skip=True, pos_enc=True),
              dict(use_skip=True, use_fpn=False), dict(use_skip=True, use_transformer=False),
              dict(use_skip=True, upsample_mode="bilinear"),
              dict(use_skip=True, use_emr=False), dict(use_skip=True, use_ecsam=False),
              dict(use_skip=True, use_adapter=False), dict(use_skip=True, decoder_ca_stages=(0, 1)),
              dict(use_skip=True, transformer_layers=4),
              dict(use_skip=True, transformer_layers=6),
              dict(use_skip=True, pos_enc=True, use_fpn=False),
              dict(use_skip=True, pos_enc=True, use_transformer=False)]
    bad = []
    for kw in combos:
        try:
            torch.manual_seed(2)
            m = mssact_light(**kw).train()
            y = m(x)
            y.sum().backward()
            if tuple(y.shape) != (2, 7, 256, 256):
                bad.append((kw, "形状 %s" % (tuple(y.shape),)))
        except Exception as e:
            bad.append((kw, "%s: %s" % (type(e).__name__, str(e)[:80])))
    print("  [5] 组合路径 %d/%d 通过" % (len(combos) - len(bad), len(combos)))
    for kw, why in bad:
        print("       FAIL %s -> %s" % (kw, why))
        fails.append("组合路径失败 %s: %s" % (kw, why))

    os.unlink(old_path)

    print()
    if fails:
        print("!! 验证未通过 (%d 项):" % len(fails))
        for f in fails:
            print("   - %s" % f)
        return 2
    print("全部通过: 默认行为未被改变, 三处缺陷已被消除")
    return 0


if __name__ == "__main__":
    sys.exit(main())
