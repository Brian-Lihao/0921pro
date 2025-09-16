import torch
import torch.nn as nn
import torch.nn.functional as F

"""
Deformable Cross-Attention (2D)
一次性并行：Query 序列 [B,T,D] ；特征图 fmap [B,C,Hf,Wf]
给定每个 Query 的参考坐标 ref_xy ∈ [0,1]^2（例如起终点直线内插 + 学习偏移），
从 fmap 周围采样 H×P 个点并聚合，得到 [B,T,D] 的上下文。
"""

def _normalize_xy_to_grid(xy: torch.Tensor):
    # [0,1] → [-1,1]  (grid_sample 坐标系)
    return xy * 2.0 - 1.0

class DeformCrossAttention2D(nn.Module):
    def __init__(self, c_fmap: int, d_model: int = 128, n_heads: int = 4, n_points: int = 16):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model  = d_model
        self.n_heads  = n_heads
        self.n_points = n_points
        self.d_head   = d_model // n_heads
        # 将 fmap 映射为 value 平面（逐头分块）
        self.proj_v   = nn.Conv2d(c_fmap, d_model, kernel_size=1, bias=False)
        # 从 Query 预测每头每点的 偏移(2) 与 权重(1)
        self.offset_mlp = nn.Linear(d_model, n_heads * n_points * 2)
        self.weight_mlp = nn.Linear(d_model, n_heads * n_points)
        # 合并各头
        self.out = nn.Linear(d_model, d_model)

    def forward(self, q: torch.Tensor, fmap: torch.Tensor, ref_xy: torch.Tensor, radius: float = 0.08):
        """
        q:      (B,T,D)
        fmap:   (B,C,Hf,Wf)
        ref_xy: (B,T,2)  in [0,1]
        radius: 采样偏移的归一化半径（相对输入坐标）
        """
        B, T, D = q.shape
        B2, C, Hf, Wf = fmap.shape
        assert B == B2, "Batch mismatch"
        # 1) 准备 Value 平面
        vmap = self.proj_v(fmap)        # (B,D,Hf,Wf)
        vmap = vmap.view(B, self.n_heads, self.d_head, Hf, Wf)  # (B,H,d_head,Hf,Wf)
        # 2) 预测偏移与权重
        off = self.offset_mlp(q)        # (B,T,H*P*2)
        w   = self.weight_mlp(q)        # (B,T,H*P)
        off = off.view(B, T, self.n_heads, self.n_points, 2)
        w   = w.view(B, T, self.n_heads, self.n_points)
        w   = torch.softmax(w, dim=-1)
        # 3) 采样坐标（归一化到 grid_sample 空间）
        ref = ref_xy.unsqueeze(2).unsqueeze(3)                     # (B,T,1,1,2)
        samp = ref + radius * off                                  # (B,T,H,P,2) in [0,1]±
        samp = _normalize_xy_to_grid(samp)                         # [-1,1]
        # 4) grid_sample 聚合（循环头以节省显存、代码更直观）
        ctx_heads = []
        for h in range(self.n_heads):
            # (B,d_head,Hf,Wf) → 采样到 (B,T,P,d_head)
            v_h = vmap[:, h]                                       # (B,d_head,Hf,Wf)
            # grid_sample 需要 (B,C,H,W) 和 grid (B,Hout,Wout,2)，我们把 T×P 当作 Hout×Wout
            grid = samp[:, :, h].reshape(B, T*self.n_points, 1, 2) # (B,T*P,1,2)
            feat = F.grid_sample(v_h, grid, mode='bilinear', padding_mode='zeros', align_corners=True)  # (B,d_head,T*P,1)
            feat = feat.squeeze(-1).transpose(1,2)                 # (B,T*P,d_head)
            feat = feat.view(B, T, self.n_points, self.d_head)     # (B,T,P,d_head)
            # 按权重加权
            w_h  = w[:, :, h].unsqueeze(-1)                        # (B,T,P,1)
            ctx_h = (feat * w_h).sum(dim=2)                        # (B,T,d_head)
            ctx_heads.append(ctx_h)
        ctx = torch.cat(ctx_heads, dim=-1)                         # (B,T,D)
        return self.out(ctx)                                       # (B,T,D)