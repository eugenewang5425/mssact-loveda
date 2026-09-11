"""
MSSACT-Net: Multi-Scale Self-Attention ConvTransformer Network
基于多尺度自注意力机制的森林类型高分遥感提取网络
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict


class EMR(nn.Module):
    """
    Encoder Residual Module - 多尺度残差编码器

    结构: 3×3卷积 + BN + ReLU + 3×3卷积 + BN + 残差连接

    原理:
    - 残差学习: H(x) = F(x) + x, 其中F(x)是残差函数
    - 梯度直接传递: 反向传播时梯度可绕过卷积层直接流向浅层
    - 缓解梯度消失: 适合深层网络训练
    """
    def __init__(self, in_channels, out_channels):
        super(EMR, self).__init__()

        # 第一个卷积块
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)

        # 第二个卷积块
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU(inplace=True)

        # 短路连接: 通道数不匹配时用1×1卷积调整
        self.shortcut = nn.Conv2d(in_channels, out_channels, 1, bias=False) \
            if in_channels != out_channels else nn.Identity()

        # 初始化
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        residual = self.shortcut(x)

        # 主路径: Conv -> BN -> ReLU -> Conv -> BN
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        # 残差连接
        out += residual
        out = self.relu(out)

        return out


class ResBlock(nn.Module):
    """
    标准ResNet残差块（用于消融实验）
    """
    def __init__(self, in_channels, out_channels):
        super(ResBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.shortcut = nn.Conv2d(in_channels, out_channels, 1, bias=False) \
            if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        residual = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return self.relu(out)


class ECSAM(nn.Module):
    """
    Efficient Convolution Spatial Attention Module - 高效卷积空间注意力模块
    实现基于坐标注意力机制 (Coordinate Attention)

    原理:
    1. 坐标分离池化 - 分别编码垂直和水平方向的长程依赖
    2. 自适应卷积 - 生成方向感知的注意力权重
    3. 特征增强 - 将注意力权重应用于原始特征
    """
    def __init__(self, channels, reduction=8):
        super(ECSAM, self).__init__()
        self.channels = channels
        self.reduction = reduction

        # 坐标分离池化
        # 水平方向: (B, C, H, 1) - 保留高度信息
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        # 垂直方向: (B, C, 1, W) - 保留宽度信息
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        # 共享的特征变换层
        mid_channels = max(channels // reduction, 8)
        self.conv1 = nn.Conv2d(channels, mid_channels, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.act = nn.ReLU(inplace=True)

        # 水平和垂直注意力分支
        self.conv_h = nn.Conv2d(mid_channels, channels, 1, bias=False)
        self.conv_w = nn.Conv2d(mid_channels, channels, 1, bias=False)

        # Squeeze and Excitation-like 通道注意力 (辅助)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False),
            nn.Sigmoid()
        )

        # 初始化权重
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        n, c, h, w = x.size()

        # ========== 坐标分离池化 ==========
        # 水平池化: (n, c, h, 1) - 每个通道沿宽度方向池化
        x_h = self.pool_h(x)  # (n, c, h, 1)
        # 垂直池化: (n, c, 1, w) - 每个通道沿高度方向池化
        x_w = self.pool_w(x)  # (n, c, 1, w)

        # ========== 拼接并变换 ==========
        # 将垂直特征转置后与水平特征拼接
        x_w_t = x_w.permute(0, 1, 3, 2)  # (n, c, w, 1)
        x_cat = torch.cat([x_h, x_w_t], dim=2)  # (n, c, h+w, 1)

        # 共享特征变换: (n, c, h+w, 1) -> (n, c//r, h+w, 1)
        y = self.act(self.bn1(self.conv1(x_cat)))

        # ========== 分离并生成注意力 ==========
        y_h, y_w = torch.split(y, [h, w], dim=2)  # 分割回h和w
        y_w_t = y_w.permute(0, 1, 3, 2)  # (n, c//r, 1, w)

        # 生成方向感知的注意力权重
        a_h = torch.sigmoid(self.conv_h(y_h))  # (n, c, h, 1) - 水平注意力
        a_w = torch.sigmoid(self.conv_w(y_w_t))  # (n, c, 1, w) - 垂直注意力

        # ========== 通道注意力 (SE) ==========
        channel_att = self.se(x)  # (n, c, 1, 1)

        # ========== 应用注意力 ==========
        # 坐标注意力: 水平 × 垂直注意力 (使特征同时具有方向感知性)
        # 通道注意力: 全局通道加权
        out = x * a_h * a_w * channel_att

        return out


class FPN(nn.Module):
    """
    Feature Pyramid Network - 特征金字塔网络
    采用自顶向下的特征融合和横向连接
    """
    def __init__(self, in_channels_list, out_channels):
        super(FPN, self).__init__()
        self.in_channels_list = in_channels_list
        self.out_channels = out_channels
        
        # 横向连接卷积
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(in_ch, out_channels, 1) 
            for in_ch in in_channels_list
        ])
        
        # 输出卷积
        self.fpn_convs = nn.ModuleList([
            nn.Conv2d(out_channels, out_channels, 3, padding=1)
            for _ in in_channels_list
        ])
    
    def forward(self, features):
        # features: [feat1, feat2, feat3, feat4, feat5] from encoder
        
        laterals = [conv(feat) for conv, feat in zip(self.lateral_convs, features)]
        
        # 自顶向下融合
        for i in range(len(laterals) - 1, 0, -1):
            laterals[i-1] = laterals[i-1] + F.interpolate(
                laterals[i], size=laterals[i-1].shape[-2:], mode='nearest'
            )
        
        # 输出
        outputs = [conv(feat) for conv, feat in zip(self.fpn_convs, laterals)]
        
        return outputs


class TransformerEncoder(nn.Module):
    """
    Scale-Adaptive Transformer Encoder - 尺度适应Transformer编码器
    利用多头自注意力机制捕捉全局上下文信息
    """
    def __init__(self, d_model, nhead, num_layers, dim_feedforward=2048, dropout=0.1, use_adapter=True):
        super(TransformerEncoder, self).__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.use_adapter = use_adapter

        # Adapter-Scale模块
        if use_adapter:
            self.adapter_down = nn.Linear(d_model, d_model // 4)
            self.adapter_up = nn.Linear(d_model // 4, d_model)
            self.scale_factor = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        # x: (B, C, H, W) -> (B, H*W, C)
        B, C, H, W = x.shape
        x_flat = x.flatten(2).permute(0, 2, 1)  # (B, H*W, C)

        # Transformer编码
        x_trans = self.transformer(x_flat)

        # Adapter-Scale
        if self.use_adapter:
            adapter = self.adapter_up(F.gelu(self.adapter_down(x_trans)))
            x_trans = x_trans + self.scale_factor * adapter

        # 恢复形状
        x_out = x_trans.permute(0, 2, 1).reshape(B, C, H, W)

        return x_out


class SegmentationDecoder(nn.Module):
    """
    语义分割解码器：将高层语义特征上采样回原始分辨率

    支持两种上采样模式:
    - 'deconv': 转置卷积上采样 (默认)
    - 'bilinear': 双线性插值上采样
    """
    def __init__(self, in_channels, num_classes, upsample_mode='deconv', ca_stages=()):
        """ca_stages: 在上采样路径的指定阶段后插入 ECSAM(坐标注意力)
        例: ca_stages=(0,) 表示在第一次上采样后(1/4分辨率)应用;
        默认 () = 不插入, 保持原行为"""
        super(SegmentationDecoder, self).__init__()
        self.upsample_mode = upsample_mode
        self.ca_stages = tuple(ca_stages)

        if upsample_mode == 'deconv':
            # 上采样路径: 32 -> 64 -> 128 -> 256 (3次转置卷积)
            self.upsample1 = nn.ConvTranspose2d(in_channels, in_channels // 2, 4, stride=2, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(in_channels // 2)

            self.upsample2 = nn.ConvTranspose2d(in_channels // 2, in_channels // 4, 4, stride=2, padding=1, bias=False)
            self.bn2 = nn.BatchNorm2d(in_channels // 4)

            self.upsample3 = nn.ConvTranspose2d(in_channels // 4, 64, 4, stride=2, padding=1, bias=False)
            self.bn3 = nn.BatchNorm2d(64)
        else:
            # 双线性插值模式：使用卷积调整通道
            self.conv1 = nn.Conv2d(in_channels, in_channels // 2, 1)
            self.bn1 = nn.BatchNorm2d(in_channels // 2)

            self.conv2 = nn.Conv2d(in_channels // 2, in_channels // 4, 1)
            self.bn2 = nn.BatchNorm2d(in_channels // 4)

            self.conv3 = nn.Conv2d(in_channels // 4, 64, 1)
            self.bn3 = nn.BatchNorm2d(64)

        # 最终分类头: 1x1卷积输出每个像素的类别概率
        self.classifier = nn.Conv2d(64, num_classes, 1)

        # 解码器侧的坐标注意力 (可选; 通道数随阶段变化)
        ch_by_stage = {0: in_channels // 2, 1: in_channels // 4, 2: 64}
        self.dec_ca = nn.ModuleDict()
        for st in self.ca_stages:
            if st in ch_by_stage:
                self.dec_ca[str(st)] = ECSAM(ch_by_stage[st])

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        if self.upsample_mode == 'deconv':
            # 转置卷积上采样
            x = F.relu(self.bn1(self.upsample1(x)))   # -> (B, 256, 64, 64)
            if "0" in self.dec_ca: x = self.dec_ca["0"](x)
            x = F.relu(self.bn2(self.upsample2(x)))   # -> (B, 128, 128, 128)
            if "1" in self.dec_ca: x = self.dec_ca["1"](x)
            x = F.relu(self.bn3(self.upsample3(x)))    # -> (B, 64, 256, 256)
            if "2" in self.dec_ca: x = self.dec_ca["2"](x)
        else:
            # 双线性插值上采样
            x = F.relu(self.bn1(self.conv1(x)))       # 调整通道
            x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
            x = F.relu(self.bn2(self.conv2(x)))
            x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
            x = F.relu(self.bn3(self.conv3(x)))
            x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)

        return self.classifier(x)  # -> (B, num_classes, 256, 256)


class MSSACTNet(nn.Module):
    """
    Multi-Scale Self-Attention ConvTransformer Network
    融合多尺度卷积和自注意力的端到端像素级语义分割网络

    输出: (B, num_classes, H, W) 每个像素的类别概率
    """
    def __init__(self, in_channels=4, num_classes=9, embed_dims=[64, 128, 256, 512],
                 use_emr=True, use_ecsam=True, use_fpn=True,
                 use_transformer=True, use_adapter=True,
                 transformer_layers=4, transformer_heads=8,
                 upsample_mode='deconv', decoder_ca_stages=()):
        super(MSSACTNet, self).__init__()

        self.in_channels = in_channels
        self.num_classes = num_classes
        self.embed_dims = embed_dims
        self.use_emr = use_emr
        self.use_ecsam = use_ecsam
        self.use_fpn = use_fpn
        self.use_transformer = use_transformer
        self.use_adapter = use_adapter
        self.upsample_mode = upsample_mode

        # Stem卷积
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, embed_dims[0], 3, padding=1),
            nn.BatchNorm2d(embed_dims[0]),
            nn.ReLU(inplace=True)
        )

        # EMR编码器 - 4个阶段
        # 若use_emr=False，则用标准ResBlock替代
        self.encoder = nn.ModuleList()
        for i in range(4):
            in_ch = embed_dims[max(0, i-1)]
            out_ch = embed_dims[i]
            if use_emr:
                self.encoder.append(EMR(in_ch, out_ch))
            else:
                # 标准残差块
                self.encoder.append(ResBlock(in_ch, out_ch))

        # ECSAM注意力模块
        self.ecsam = nn.ModuleList()
        if use_ecsam:
            for i in range(4):
                self.ecsam.append(ECSAM(embed_dims[i]))

        # FPN特征金字塔
        if use_fpn:
            self.fpn = FPN(embed_dims, embed_dims[-1])
        else:
            self.fpn = None

        # Transformer编码器
        if use_transformer:
            self.transformer = TransformerEncoder(
                d_model=embed_dims[-1],
                nhead=transformer_heads,
                num_layers=transformer_layers,
                dim_feedforward=embed_dims[-1] * 4,
                use_adapter=use_adapter
            )
        else:
            self.transformer = None

        # 语义分割解码器 (替代全局池化)
        self.decoder = SegmentationDecoder(
            embed_dims[-1], num_classes,
            upsample_mode=upsample_mode,
            ca_stages=decoder_ca_stages
        )

    def forward(self, x):
        # Stem
        x = self.stem(x)

        # EMR编码 + ECSAM
        encoder_features = []
        for i, emr in enumerate(self.encoder):
            x = emr(x)

            # ECSAM (仅当use_ecsam=True且非第一阶段)
            if self.use_ecsam and i > 0:
                x = self.ecsam[i](x)

            encoder_features.append(x)
            if i < 3:
                x = F.max_pool2d(x, 2)  # 下采样

        # FPN融合多尺度特征
        if self.use_fpn:
            fpn_features = self.fpn(encoder_features)
            x = fpn_features[-1]
        else:
            # 不用FPN，直接用最高层特征
            x = encoder_features[-1]

        # Transformer编码 - 捕捉全局上下文
        if self.use_transformer:
            x = self.transformer(x)

        # 解码器上采样 - 恢复空间分辨率
        x = self.decoder(x)

        return x  # (B, num_classes, 256, 256)


def msscactnet(num_classes=9, in_channels=4, embed_dims=None,
               use_emr=True, use_ecsam=True, use_fpn=True,
               use_transformer=True, use_adapter=True,
               transformer_layers=4, transformer_heads=8,
               upsample_mode='deconv'):
    """
    创建MSSACT-Net模型（支持消融实验配置）

    Args:
        num_classes: 类别数
        in_channels: 输入通道数
        embed_dims: 嵌入维度列表
        use_emr: 是否使用EMR（False则用标准ResBlock）
        use_ecsam: 是否使用ECSAM注意力
        use_fpn: 是否使用FPN
        use_transformer: 是否使用Transformer
        use_adapter: 是否使用Adapter-Scale
        transformer_layers: Transformer层数
        transformer_heads: Transformer头数
        upsample_mode: 'deconv'（转置卷积）或 'bilinear'（双线性插值）
    """
    if embed_dims is None:
        embed_dims = [64, 128, 256, 512]

    return MSSACTNet(
        in_channels=in_channels,
        num_classes=num_classes,
        embed_dims=embed_dims,
        use_emr=use_emr,
        use_ecsam=use_ecsam,
        use_fpn=use_fpn,
        use_transformer=use_transformer,
        use_adapter=use_adapter,
        transformer_layers=transformer_layers,
        transformer_heads=transformer_heads,
        upsample_mode=upsample_mode
    )


def create_model_from_config(config):
    """
    从配置文件字典创建模型

    Args:
        config: 包含model配置的字典
    """
    model_cfg = config.get('model', {})

    return msscactnet(
        num_classes=model_cfg.get('num_classes', 9),
        in_channels=model_cfg.get('in_channels', 4),
        embed_dims=model_cfg.get('embed_dims', [64, 128, 256, 512]),
        use_emr=model_cfg.get('use_emr', True),
        use_ecsam=model_cfg.get('use_ecsam', True),
        use_fpn=model_cfg.get('use_fpn', True),
        use_transformer=model_cfg.get('use_transformer', True),
        use_adapter=model_cfg.get('use_adapter', True),
        transformer_layers=model_cfg.get('transformer_layers', 4),
        transformer_heads=model_cfg.get('transformer_heads', 8),
        upsample_mode=model_cfg.get('upsample_mode', 'deconv')
    )


def count_parameters(model):
    """统计模型参数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    # 测试模型 - 像素级语义分割输出
    model = msscactnet(num_classes=9)

    # 打印模型结构
    print("=" * 60)
    print("MSSACT-Net 像素级语义分割模型")
    print("=" * 60)
    print(model)
    print("=" * 60)
    print(f"模型参数量: {count_parameters(model):,} (约 {count_parameters(model)/1e6:.2f}M)")
    print("=" * 60)

    # 测试前向传播 - 256x256输入
    x = torch.randn(2, 4, 256, 256)  # (batch, channels, height, width)
    y = model(x)
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {y.shape}  (batch, classes, height, width)")
    assert y.shape == (2, 9, 256, 256), f"期望输出(2, 9, 256, 256), 得到{y.shape}"
    print("输出形状验证通过!")
