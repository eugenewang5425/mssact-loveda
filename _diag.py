import sys, traceback
sys.path.insert(0, ".")
try:
    from experiment_matrix import train_one
    from experiment_matrix_v2 import mssact_light
    import paths
    train_one(lambda: mssact_light(), "diag_tmp", max_epochs=1, patience=1,
              batch=4, lr=2e-4, root=paths.DATA_NEWSPLIT2, seed=7)
    print("OK 训练成功")
except Exception:
    traceback.print_exc()
