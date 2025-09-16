
import math
import torch

def build_2d_sincos_pos_embed(H, W, dim, device=None):
    """
    返回 (H*W, dim) 的二维正余弦位置编码（不依赖网络参数）
    """
    device = device or torch.device("cpu")
    grid_y = torch.arange(H, device=device, dtype=torch.float32)
    grid_x = torch.arange(W, device=device, dtype=torch.float32)
    grid = torch.stack(torch.meshgrid(grid_y, grid_x, indexing='ij'), dim=0)  # (2,H,W)
    pos_embed = _get_2d_sincos_pos_embed_from_grid(dim, grid)
    return pos_embed  # (H*W, dim)

def _get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0
    emb_h = get_1d_sincos_pos_embed(embed_dim // 2, grid[0].reshape(-1))
    emb_w = get_1d_sincos_pos_embed(embed_dim // 2, grid[1].reshape(-1))
    return torch.cat([emb_h, emb_w], dim=1)

def get_1d_sincos_pos_embed(embed_dim, positions):
    omega = torch.arange(embed_dim // 2, dtype=torch.float32, device=positions.device)
    omega = 1. / (10000 ** (omega / (embed_dim / 2)))
    out = torch.einsum('n,d->nd', positions, omega)
    emb_sin = torch.sin(out)
    emb_cos = torch.cos(out)
    return torch.cat([emb_sin, emb_cos], dim=1)
