# -*- coding: utf-8 -*-
"""构建数据量阶梯子集: 250 / 500 / 1000 (从 newsplit2/train 分层抽样)
原则:
  - val(221) / test_clean(141) 完全不变 -> 各档位的验证/测试集一致, 结果可直接比较
  - 分层抽样: 按"主导类"(patch 中出现最多的类别)分层, 保证小档位也类别均衡
  - 子集内两两嵌套 (250 ⊂ 500 ⊂ 1000 ⊂ 1768) -> 数据量是唯一变量
输出: data_ladder/n{250,500,1000}/{images,masks}  (硬链接/复制)
"""
import os, shutil, json, random
import paths  # 集中路径配置 (环境变量/.env)
import numpy as np
from PIL import Image

ROOT = paths.DATA_NEWSPLIT2
OUT = paths.DATA_LADDER
SIZES = [250, 500, 1000]
SEED = 42

def dominant_class(mask_path):
    m = np.array(Image.open(mask_path))
    v = m[m != 255]
    if v.size == 0: return -1
    # 值1..7 -> 0..6
    return int(np.bincount(v, minlength=8)[1:8].argmax())

def main():
    tr_img = f"{ROOT}/train/images"
    tr_msk = f"{ROOT}/train/masks"
    names = sorted(set(os.listdir(tr_img)) & set(os.listdir(tr_msk)))
    print(f"train 池: {len(names)} 张", flush=True)

    # 1) 分层: 按主导类分组
    strata = {}
    for n in names:
        c = dominant_class(f"{tr_msk}/{n}")
        strata.setdefault(c, []).append(n)
    print("主导类分布:", {k: len(v) for k, v in sorted(strata.items())}, flush=True)

    # 2) 固定随机序 (嵌套保证: 先生成总顺序, 再按比例取前N)
    rng = random.Random(SEED)
    # 按层内随机排序, 再交错合并 -> 保证各档位都覆盖全部主导类
    for c in strata: rng.shuffle(strata[c])
    ordered = []
    # 按每层占比分配名额 (最大余数法简化: 轮转取)
    total = len(names)
    idx = {c: 0 for c in strata}
    while len(ordered) < total:
        for c in sorted(strata):
            if idx[c] < len(strata[c]):
                ordered.append(strata[c][idx[c]]); idx[c] += 1
    print(f"分层交错序列: {len(ordered)} 张", flush=True)

    # 3) 输出各档位
    for size in SIZES:
        subset = ordered[:size]
        dst = f"{OUT}/n{size}/train"      # 加 train 层, 与 LoveDADataset(split="train") 一致
        if os.path.exists(dst): shutil.rmtree(dst)
        os.makedirs(f"{dst}/images", exist_ok=True); os.makedirs(f"{dst}/masks", exist_ok=True)
        for n in subset:
            shutil.copy(f"{tr_img}/{n}", f"{dst}/images/{n}")
            shutil.copy(f"{tr_msk}/{n}", f"{dst}/masks/{n}")
        # 统计类别分布
        cnt = np.zeros(8, np.int64)
        for n in subset[:80]:   # 抽样统计前80张(速度)
            m = np.array(Image.open(f"{tr_msk}/{n}"))
            cnt += np.bincount(m.flatten(), minlength=8)[:8]
        tot = cnt[1:8].sum()
        dist = {i: round(cnt[i+1]/max(tot,1)*100,1) for i in range(7)}
        print(f"  n{size}: {len(subset)} 张  类别分布 {dist}", flush=True)

    json.dump(dict(sizes=SIZES, seed=SEED, nested=True,
                   note="分层抽样(按主导类), 嵌套子集; val/test_clean 不变"),
              open(f"{OUT}/ladder_meta.json","w"), indent=1)
    print(f"\n产出: {OUT}", flush=True)

if __name__ == "__main__":
    main()
