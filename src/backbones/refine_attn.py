
import torch
import torch.nn as nn
import torch.nn.functional as F

def _normalize_xy_to_grid(xy: torch.Tensor):
    return xy * 2.0 - 1.0

class PathRefiner(nn.Module):
    """
    对初步路径坐标做一次局部细化：
      - 在 fmap 上以每个点为中心采样 K 个邻域点（固定八方向 + 中心）
      - 聚合后通过 MLP 预测 Δp，最后用 gate 限制 |Δp| ≤ max_delta (像素/归一化)
    """
    def __init__(self, c_fmap: int, d_model: int = 128, k_points: int = 9, max_delta_norm: float = 0.5/512.0):
        super().__init__()
        self.k = k_points
        self.max_delta = max_delta_norm
        self.proj = nn.Conv2d(c_fmap, d_model, 1, bias=False)
        self.mlp  = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, 2),
        )
        # 固定 3×3 采样偏移（含中心），在 [-r,r] 归一化半径内
        base = torch.tensor([
            [ 0,  0],
            [-1, -1], [ 0, -1], [ 1, -1],
            [-1,  0],           [ 1,  0],
            [-1,  1], [ 0,  1], [ 1,  1],
        ], dtype=torch.float32)  # (9,2)+        self.register_buffer('base_offsets', base / 64.0)  # 小范围：约 1/64 图宽高

    def forward(self, fmap: torch.Tensor, coords: torch.Tensor):
        """
        fmap  : (B,C,Hf,Wf)
        coords: (B,T,2) in [0,1]
        """
        B, C, Hf, Wf = fmap.shape
        B2, T, _ = coords.shape
        assert B == B2
        vmap = self.proj(fmap)                         # (B,d_model,Hf,Wf)
        # 采样 3×3 邻域
        offsets = self.base_offsets.view(1,1,self.k,2).to(coords.device)  # (1,1,K,2)
        grid = coords.unsqueeze(2) + offsets                                # (B,T,K,2)
        grid = _normalize_xy_to_grid(grid)
        grid = grid.view(B, T*self.k, 1, 2)
        feat = F.grid_sample(vmap, grid, mode='bilinear', padding_mode='zeros', align_corners=True)  # (B,d_model,T*K,1)
        feat = feat.squeeze(-1).transpose(1,2).view(B, T, self.k, -1).mean(dim=2)                   # (B,T,d_model) 简单平均
        # 预测 Δp 并门控
        delta = torch.tanh(self.mlp(feat)) * self.max_delta  # (B,T,2)
        return coords + delta