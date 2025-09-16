import torch.nn as nn
from src.backbones.swin_unet_encoder import SwinUNetEncoder
from src.configs import New_DataName

# class Perception(nn.Module):
#     def __init__(self, cfgs):
#         super().__init__()
#         self.encoder = SwinUNetEncoder()    # ← 新 encoder

#     def forward(self, input_dict):
#         x = input_dict[New_DataName.rgb_map]             # (B,3,128,128)
#         fmap, gvec = self.encoder(x)
#         # print("fmap shape(from percetion):", fmap.shape)  # 应该是 [B,192,H,W]
#         return {"fmap": fmap, "gvec": gvec}

import torch.nn as nn
from src.backbones.cnn_fpn_encoder import CNNFPNBackbone
from src.configs import New_DataName

class Perception(nn.Module):
    """
    轻量 CNN+FPN 感知器：
      - 自动从 batch[New_DataName.rgb_map] 推断输入通道数
      - 输出 dict: {"fmap": (B,C,H/8,W/8)}
    cfg.perception 可为空；若提供可支持:
      - in_channels: int
      - out_channels: int  (默认用 cfg.model.cvae_core.fmap_channels)
      - fix_perception: bool (由 model.py 处理 freeze)
    """
    def __init__(self, pcfg=None):
        super().__init__()
        self.pcfg = pcfg if pcfg is not None else {}
        # 默认输出通道对齐 cvae_core.fmap_channels
        from src.configs import cfg as _cfg
        out_ch = int(getattr(_cfg.model.cvae_core, "fmap_channels", 192))
        in_ch  = int(getattr(self.pcfg, "in_channels", 0) or 0)
        # 先用占位，真正 in_ch 在第一步 forward 时确定（lazy）
        self.backbone = None
        self._lazy_in_ch = in_ch
        self._out_ch = out_ch

    def _lazy_build(self, x):
        if self.backbone is None:
            in_ch = self._lazy_in_ch if self._lazy_in_ch > 0 else x.shape[1]
            self.backbone = CNNFPNBackbone(in_ch=in_ch, out_ch=self._out_ch).to(x.device)

    def forward(self, batch):
        x = batch[New_DataName.rgb_map]  # 期望 (B,C,H,W) 或 (C,H,W)
        if x.dim() == 3:
            x = x.unsqueeze(0)
        # 若被错误 squeeze 成 (B,H,W)，恢复成 (B,1,H,W)
        if x.dim() == 3:  # 安全兜底
            x = x.unsqueeze(1)
        self._lazy_build(x)
        feat = self.backbone(x)
        return {"fmap": feat["fmap"]}