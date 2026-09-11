# -*- coding: utf-8 -*-
"""校验 newsplit2 无泄漏 + 补写 split_meta.json"""
import os, hashlib, json
import paths  # 集中路径配置 (环境变量/.env)
import numpy as np
from PIL import Image
ROOT = paths.DATA_ROOT
OUT = f"{ROOT}/newsplit2"

def md5(p, chunk=1<<20):
    h = hashlib.md5()
    with open(p,'rb') as f:
        while True:
            b = f.read(chunk)
            if not b: break
            h.update(b)
    return h.hexdigest()

def mset(split):
    d = f"{OUT}/{split}/images"
    return {md5(f"{d}/{n}") for n in os.listdir(d)}

tr, va, te, cl = mset("train"), mset("val"), mset("test"), mset("test_clean")
old_train = set()
for n in os.listdir(f"{ROOT}/train/images"):
    if not n.startswith("off_"):
        old_train.add(md5(f"{ROOT}/train/images/{n}"))

print(f"train={len(tr)} val={len(va)} test={len(te)} test_clean={len(cl)}")
assert not (tr & va), "train∩val 泄漏!"
assert not (tr & te), "train∩test 泄漏!"
assert not (va & te), "val∩test 泄漏!"
assert cl <= te, "test_clean 必须⊆test"
print(f"test_clean ∩ 旧训练集: {len(cl & old_train)} (应为0)")
assert not (cl & old_train), "test_clean与旧训练集泄漏!"
# no-data 占比复检
ns = []
for n in sorted(os.listdir(f"{OUT}/train/masks"))[:200]:
    m = np.array(Image.open(f"{OUT}/train/masks/{n}"))
    ns.append((m==0).mean())
print(f"train no-data 占比: 均值{np.mean(ns)*100:.2f}% 最大{np.max(ns)*100:.2f}% (阈值10%)")
assert np.max(ns) <= 0.101, "存在超阈值图!"
print("✓ 全部校验通过")
json.dump(dict(threshold=0.10, seed=42, train=len(tr), val=len(va), test=len(te),
               test_clean=len(cl), note="no-data<=10%筛选 + MD5去重 + 8:1:1; test_clean排除旧训练集"),
          open(f"{OUT}/split_meta.json","w"), indent=1)
print("split_meta.json 已写入")
