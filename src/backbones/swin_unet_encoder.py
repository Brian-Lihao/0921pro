import torch, torch.nn as nn
from timm import create_model

class SwinUNetEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = create_model(
            'swin_tiny_patch4_window7_224',
            pretrained=False,
            features_only=True,
            out_indices=(2,),        # stage-2 → H=W=32
            img_size=(256,256),
        )
        self.final_conv = nn.Conv2d(384, 192, kernel_size=1)

    def forward(self, x):
        feat = self.backbone(x)[0]              # (B,32,32,192)  *channel-last!*
        if feat.shape[1] == 32:                 # (B,H,W,C)
            feat = feat.permute(0, 3, 1, 2)     # → (B,C,H,W)
        fmap = feat.contiguous()                # (B,192,32,32)
        gvec = fmap.mean(dim=(2,3))             # (B,192)
        # new
        fmap = fmap.permute(0,3,1,2).contiguous() # 先调整到[B,C,H,W]
        fmap = self.final_conv(fmap)  # 调整通道到192 (需要你手动加上self.final_conv)
        # new
        return fmap, gvec
