# -*- coding: utf-8 -*-
"""筛选(no-data<=10%) + MD5去重 + 8:1:1 重划分 (无泄漏)
产出: newsplit2/{train,val,test,test_clean}
    - test_clean: 额外排除旧模型(957 chloechia)训练集, 用于新旧公平对比
校验: 三集两两无交集; test_clean ⊆ test; 与旧训练集无交集
"""
import os, hashlib, json, shutil, random
import paths  # 集中路径配置 (环境变量/.env)
import numpy as np
from PIL import Image

ROOT = paths.DATA_ROOT
OFFICIAL = f"{ROOT}/loveda_official/train"           # 官方 2522 (images_png / masks_png)
OLD_TRAIN = f"{ROOT}/train"                          # 旧: chloechia + off_官方
OUT = f"{ROOT}/newsplit2"
NODATA_MAX = 0.10                                    # 阈值: no-data <= 10%
SEED = 42

def md5(p, chunk=1<<20):
    h = hashlib.md5()
    with open(p, 'rb') as f:
        while True:
            b = f.read(chunk)
            if not b: break
            h.update(b)
    return h.hexdigest()

def nodata_frac(mask_path):
    m = np.array(Image.open(mask_path))
    return float((m == 0).mean())

def main():
    pool = []          # (md5, img_path, mask_path, src)
    seen = set()
    dropped_nodata = 0
    dup = 0

    # 1) 官方 2522
    for n in sorted(os.listdir(f"{OFFICIAL}/images_png")):
        if not n.endswith(".png"): continue
        ip, mp_ = f"{OFFICIAL}/images_png/{n}", f"{OFFICIAL}/masks_png/{n}"
        if not os.path.exists(mp_): continue
        if nodata_frac(mp_) > NODATA_MAX:
            dropped_nodata += 1; continue
        h = md5(ip)
        if h in seen: dup += 1; continue
        seen.add(h); pool.append((h, ip, mp_, "official"))
    print(f"官方: 保留 {len(pool)}, 因no-data剔除 {dropped_nodata}", flush=True)

    # 2) 旧 chloechia 非重复部分 (无 off_ 前缀)
    added = 0
    for n in sorted(os.listdir(f"{OLD_TRAIN}/images")):
        if not n.endswith(".png") or n.startswith("off_"): continue
        ip, mp_ = f"{OLD_TRAIN}/images/{n}", f"{OLD_TRAIN}/masks/{n}"
        if not os.path.exists(mp_): continue
        if nodata_frac(mp_) > NODATA_MAX: continue
        h = md5(ip)
        if h in seen: dup += 1; continue
        seen.add(h); pool.append((h, ip, mp_, "chloechia")); added += 1
    print(f"chloechia 新增非重复: {added}, 累计池 {len(pool)} (MD5重复跳过 {dup})", flush=True)

    # 3) 旧模型训练集 md5 (用于 test_clean 排除)
    old_train_md5 = set()
    for n in sorted(os.listdir(f"{OLD_TRAIN}/images")):
        if n.startswith("off_"): continue
        old_train_md5.add(md5(f"{OLD_TRAIN}/images/{n}"))
    print(f"旧模型训练集: {len(old_train_md5)} 张", flush=True)

    # 4) 8:1:1 划分
    random.seed(SEED); random.shuffle(pool)
    n = len(pool)
    n_test = n // 10; n_val = n // 10; n_train = n - n_test - n_val
    tr, va, te = pool[:n_train], pool[n_train:n_train+n_val], pool[n_train+n_val:]
    print(f"划分: train={len(tr)} val={len(va)} test={len(te)}", flush=True)

    if os.path.exists(OUT): shutil.rmtree(OUT)
    for sub in ["train","val","test","test_clean"]:
        os.makedirs(f"{OUT}/{sub}/images", exist_ok=True)
        os.makedirs(f"{OUT}/{sub}/masks", exist_ok=True)

    def dump(items, split):
        for i,(h,ip,mp_,src) in enumerate(items):
            nm = f"{i:05d}.png"
            shutil.copy(ip, f"{OUT}/{split}/images/{nm}")
            shutil.copy(mp_, f"{OUT}/{split}/masks/{nm}")
    dump(tr, "train"); dump(va, "val"); dump(te, "test")

    # 5) test_clean: test 中排除旧模型训练集
    clean = [x for x in te if x[0] not in old_train_md5]
    dump(clean, "test_clean")
    print(f"test_clean: {len(clean)} 张 (排除旧训练集重叠 {len(te)-len(clean)} 张)", flush=True)

    # 6) 校验
    def md5set(split):
        d = f"{OUT}/{split}/images"
        return {md5(f"{d}/{n}") for n in os.listdir(d)}
    mtr, mva, mte, mcl = md5set("train"), md5set("val"), md5set("test"), md5set("test_clean")
    assert not (mtr & mva), "train∩val 泄漏!"
    assert not (mtr & mte), "train∩test 泄漏!"
    assert not (mva & mte), "val∩test 泄漏!"
    assert mcl <= mte, "test_clean 必须 ⊆ test"
    assert not (mcl & old_train_md5), "test_clean ∩ 旧训练集 泄漏!"
    print("✓ 泄漏校验通过: train/val/test 两两无交集; test_clean ⊆ test 且与旧训练集无交集", flush=True)

    json.dump(dict(threshold=NODATA_MAX, seed=SEED,
                   train=len(tr), val=len(va), test=len(te), test_clean=len(clean),
                   dropped_nodata=dropped_nodata, dup_removed=dup,
                   note="no-data<=10% 筛选 + MD5去重 + 8:1:1; test_clean额外排除旧模型训练集"),
              open(f"{OUT}/split_meta.json","w"), indent=1)
    print(f"\n产出目录: {OUT}", flush=True)

if __name__ == "__main__":
    main()
