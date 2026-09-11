"""实验矩阵 v2（另存补齐）：论文表4-2/4-3 缺失项 + 公平性验证实验
新增: FPN基线 / Swin-Unet基线 / Transformer-4L / Transformer-6L / Bilinear-Up / UNet-lr1e-4
复用 experiment_matrix.train_one 同一收敛协议
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch, torch.nn as nn
from experiment_matrix import train_one, NUM_CLASSES
from models.msscactnet import MSSACTNet

class FPNSeg(nn.Module):
    """FPN 分割基线: ResNet-50 主干 + torchvision FPN + 轻头"""
    def __init__(self, in_ch=3, nc=NUM_CLASSES):
        super().__init__()
        import torchvision.models as tvm
        from torchvision.ops import FeaturePyramidNetwork
        m = tvm.resnet50(weights=None)
        m.conv1 = nn.Conv2d(in_ch, 64, 7, 2, 3, bias=False)
        self.stem = nn.Sequential(m.conv1, m.bn1, m.relu, m.maxpool)
        self.layer1, self.layer2, self.layer3, self.layer4 = m.layer1, m.layer2, m.layer3, m.layer4
        self.fpn = FeaturePyramidNetwork([256,512,1024,2048], 256)
        self.head = nn.Sequential(nn.Conv2d(256,128,3,padding=1,bias=False), nn.GroupNorm(8,128), nn.ReLU(True),
                                  nn.Conv2d(128,nc,1))
    def forward(self, x):
        c1 = self.stem(x); c2 = self.layer1(c1); c3 = self.layer2(c2); c4 = self.layer3(c3); c5 = self.layer4(c4)
        p = self.fpn({"0":c2,"1":c3,"2":c4,"3":c5})   # p0 /4
        out = self.head(p["0"])
        return nn.functional.interpolate(out, scale_factor=4, mode="bilinear", align_corners=False)

class SwinUnetLite(nn.Module):
    """Swin-Unet 风格基线: torchvision swin_t 编码器 (/4,/8,/16,/32 跳连) + CNN 解码器"""
    def __init__(self, in_ch=3, nc=NUM_CLASSES):
        super().__init__()
        import torchvision.models as tvm
        m = tvm.swin_t(weights=None)
        self.features = m.features   # 0:patch/4 1:stage/4 2:merge/8 3:stage/8 4:merge/16 5:stage/16 6:merge/32 7:stage/32
        def up(i,o): return nn.Sequential(nn.ConvTranspose2d(i,o,2,2), nn.GroupNorm(8,o), nn.GELU())
        self.up3 = up(768,384); self.f3 = nn.Sequential(nn.Conv2d(768,384,3,padding=1,bias=False), nn.GroupNorm(8,384), nn.GELU())
        self.up2 = up(384,192); self.f2 = nn.Sequential(nn.Conv2d(384,192,3,padding=1,bias=False), nn.GroupNorm(8,192), nn.GELU())
        self.up1 = up(192,96);  self.f1 = nn.Sequential(nn.Conv2d(192,96,3,padding=1,bias=False),  nn.GroupNorm(8,96),  nn.GELU())
        self.head = nn.Sequential(nn.Conv2d(96,96,3,padding=1,bias=False), nn.GroupNorm(8,96), nn.GELU(),
                                  nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False), nn.Conv2d(96,nc,1))
    def forward(self, x):
        s = {}
        for i in range(8):
            x = self.features[i](x)
            if i in (1,3,5,7): s[i] = x.permute(0,3,1,2).contiguous()  # swin输出(B,H,W,C) -> (B,C,H,W)
        d = self.up3(s[7]); d = self.f3(torch.cat([d, s[5]], 1))
        d = self.up2(d);    d = self.f2(torch.cat([d, s[3]], 1))
        d = self.up1(d);    d = self.f1(torch.cat([d, s[1]], 1))
        return self.head(d)

def mssact_light(transformer_layers=2, **flags):
    return MSSACTNet(in_channels=3, num_classes=NUM_CLASSES,
                     embed_dims=[32,64,128,256], transformer_layers=transformer_layers,
                     transformer_heads=4, **flags)

MATRIX_V2 = [
    ("fpn_seg",          lambda: FPNSeg(),                                     8, 2e-4),
    ("swin_unet",        lambda: SwinUnetLite(),                               8, 2e-4),
    ("ablate_trans4l",   lambda: mssact_light(transformer_layers=4),           8, 2e-4),
    ("ablate_trans6l",   lambda: mssact_light(transformer_layers=6),           8, 2e-4),
    ("ablate_bilinear",  lambda: mssact_light(upsample_mode='bilinear'),       8, 2e-4),
    ("unet_lr1e4",       None,                                                 8, 1e-4),  # UNet lr敏感性验证
]

if __name__ == "__main__":
    import argparse
    from experiment_matrix import UNet
    ap = argparse.ArgumentParser()
    ap.add_argument("--single", type=str, default=None)
    ap.add_argument("--max-epochs", type=int, default=60)
    ap.add_argument("--patience", type=int, default=15)
    args = ap.parse_args()
    sel = MATRIX_V2 if args.single is None else [m for m in MATRIX_V2 if m[0] == args.single]
    if not sel:
        print(f"ERROR: {args.single} not in {[m[0] for m in MATRIX_V2]}"); raise SystemExit(1)
    print(f"matrix_v2 ({len(sel)}):", [m[0] for m in sel], flush=True)
    for name, fn, batch, lr in sel:
        if fn is None:  # unet_lr1e4
            fn = lambda: UNet()
        try:
            train_one(fn, name, max_epochs=args.max_epochs, patience=args.patience, batch=batch, lr=lr)
        except Exception as e:
            print(f"!!! {name} FAILED: {type(e).__name__}: {str(e)[:300]}", flush=True)
