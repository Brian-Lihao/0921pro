
import torch
import torch.nn as nn
import torch.nn.functional as F

# 轻量 CNN 干路 + FPN，输出单尺度 fmap（缺省 1/8 尺度）

class ConvBNAct(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, k, s, p, bias=False)
        self.bn   = nn.BatchNorm2d(out_ch)
        self.act  = nn.ReLU(inplace=True)
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class BasicBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = ConvBNAct(in_ch, out_ch, 3, stride, 1)
        self.conv2 = ConvBNAct(out_ch, out_ch, 3, 1, 1)
        self.proj  = None
        if in_ch != out_ch or stride != 1:
            self.proj = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride, 0, bias=False),
                nn.BatchNorm2d(out_ch)
            )
        self.act = nn.ReLU(inplace=True)
    def forward(self, x):
        idt = x if self.proj is None else self.proj(x)
        x = self.conv2(self.conv1(x))
        return self.act(x + idt)

class CNNFPNBackbone(nn.Module):
    """
    输入:  (B, C_in, H, W)  —— 例如 (B, 3, 256, 256)
    输出:  dict:
      - 'fmap': 单尺度特征图 (B, C_out, H/8, W/8)
      - 'pyramids': [P3,P4,P5]（可选，多尺度，先留接口）
    """
    def __init__(self, in_ch=3, out_ch=128):
        super().__init__()
        # stem: 1/2
        self.stem = nn.Sequential(
            ConvBNAct(in_ch, 32, 3, 2, 1),  # H/2
            ConvBNAct(32, 32, 3, 1, 1),
            ConvBNAct(32, 64, 3, 1, 1),
        )
        # stages
        self.layer2 = nn.Sequential(            # 1/4
            BasicBlock(64, 128, stride=2),
            BasicBlock(128,128, stride=1),
        )
        self.layer3 = nn.Sequential(            # 1/8
            BasicBlock(128,192, stride=2),
            BasicBlock(192,192, stride=1),
        )
        self.layer4 = nn.Sequential(            # 1/16
            BasicBlock(192,256, stride=2),
            BasicBlock(256,256, stride=1),
        )
        # FPN: 只保留到 1/8 的输出，减少显存/算力
        self.lateral3 = nn.Conv2d(192, out_ch, 1, 1, 0)
        self.lateral4 = nn.Conv2d(256, out_ch, 1, 1, 0)
        self.out3     = ConvBNAct(out_ch, out_ch, 3, 1, 1)
    def forward(self, x):
        # x: (B,C,H,W)
        x = self.stem(x)         # 1/2
        c2 = self.layer2(x)      # 1/4
        c3 = self.layer3(c2)     # 1/8
        c4 = self.layer4(c3)     # 1/16
        # 简化 FPN, 只回传 P3 (1/8)
        p4 = self.lateral4(c4)
        p3 = self.lateral3(c3) + F.interpolate(p4, size=c3.shape[-2:], mode='nearest')
        p3 = self.out3(p3)       # (B, out_ch, H/8, W/8)
        return {'fmap': p3, 'pyramids': [p3, p4]}