"""回退验证: 跑 D1 配置（无FPN+无Transformer），检查 Kappa 是否恢复至 ~0.628
配置: 回退版增强（全局RNG）+ num_workers=0 + memmap 加速
对照: 旧管线 D1 = 0.6277 / 新管线(确定性增强) D1 = 0.6031
"""
import os, sys, time, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paths
sys.path.insert(0, paths.REPO)
from experiment_matrix import train_one
from experiment_matrix_v2 import mssact_light

TAG = "lgV_D1_rollback"
if __name__ == "__main__":
    t0 = time.time()
    train_one(lambda: mssact_light(use_fpn=False, use_transformer=False), TAG,
              max_epochs=60, patience=20, batch=8, lr=2e-4, root=paths.DATA_NEWSPLIT2)
    h = json.load(open(os.path.join(paths.CKPT, f"{TAG}_history.json")))
    b = max(h, key=lambda r: r.get("kappa", -9))
    print(f"\n=== 回退验证结果 ===")
    print(f"  回退版 D1: {b['kappa']:.4f} ({len(h)}轮, 用时 {(time.time()-t0)/60:.1f} 分钟)")
    print(f"  旧管线 D1: 0.6277  (参照)")
    print(f"  确定增强版 D1: 0.6031")
    print(f"  与旧管线差: {b['kappa']-0.6277:+.4f}")
