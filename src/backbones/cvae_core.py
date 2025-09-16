
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from src.configs import New_DataName, cfg as global_cfg
from src.backbones.deform_cross_attn import DeformCrossAttention2D
from src.backbones.refine_attn import PathRefiner

# ============ Encoder：四象限 GAP + 坐标嵌入 ============ #
class QuadGAPEncoder(nn.Module):
    """
    输入:
      fmap : (B, C, H, W)  —— 感知特征图
      start, goal : (B, 2) —— 归一化坐标 [0,1]
    输出:
      mu, logvar : (B, z_dim)
    """
    def __init__(self, c_in: int, z_dim: int):
        super().__init__()
        self.c_in  = c_in
        self.z_dim = z_dim
        self.fc1   = nn.Linear(c_in*4 + 4, 256)
        self.fc_mu = nn.Linear(256, z_dim)
        self.fc_lv = nn.Linear(256, z_dim)
        self.act   = nn.ReLU(inplace=True)

    @staticmethod
    def _quad_gap(fmap: torch.Tensor) -> torch.Tensor:
        B, C, H, W = fmap.shape
        h2, w2 = H//2, W//2
        q1 = fmap[..., :h2, :w2].mean(dim=(-2,-1))  # 左上
        q2 = fmap[..., :h2,  w2:].mean(dim=(-2,-1)) # 右上
        q3 = fmap[...,  h2:, :w2].mean(dim=(-2,-1)) # 左下
        q4 = fmap[...,  h2:,  w2:].mean(dim=(-2,-1))# 右下
        return torch.cat([q1,q2,q3,q4], dim=-1)     # (B, 4C)

    def forward(self, fmap: torch.Tensor,
                      start: torch.Tensor,
                      goal:  torch.Tensor):
        q = self._quad_gap(fmap)                    # (B, 4C)
        enc = torch.cat([q, start, goal], dim=-1)   # (B, 4C+4)
        h   = self.act(self.fc1(enc))
        mu  = self.fc_mu(h)
        lv  = self.fc_lv(h)
        return mu, lv

class CoordDecoder(nn.Module):
    """
    并行 38 点：
      - 生成 T 个 Query：q_t = MLP([z, start, goal, t_embed])
      - 参考坐标 ref_t = 直线内插(start→goal) + 小偏移
      - Deformable Cross-Attn 在 ref_t 周围采样 fmap
      - 预测 Δp_t 相对 ref_t 的残差，得到 p_t = ref_t + Δp_t
      - 一次性输出所有中间点；随后 PathRefiner 小修正
    """
    # def __init__(self, c_in: int, z_dim: int, out_steps: int = 38,
    #              d_model: int = 128, n_heads: int = 4, n_points: int = 16):
    def __init__(self, c_in: int, z_dim: int, out_steps: int = 38,
                 d_model: int = 128, n_heads: int = 4, n_points: int = 16,
                 radius_norm: float = 0.08):
        super().__init__()
        self.out_steps = out_steps
        self.d_model   = d_model
        self.n_heads   = n_heads
        self.n_points  = n_points
        self.radius    = float(radius_norm)

        # 生成 per-step Query
        self.q_mlp = nn.Sequential(
            nn.Linear(z_dim + 4 + 16, d_model),  # z(64)+start(2)+goal(2)+t_embed(16)
            nn.ReLU(inplace=True),
            nn.Linear(d_model, d_model),
            nn.ReLU(inplace=True),
        )
        # 参考偏移（把 ref_t 从直线稍微挪开，避免全在直线上采样）
        self.ref_off = nn.Linear(d_model, 2)
        # Deformable Cross-Attn
        self.dca = DeformCrossAttention2D(c_in, d_model, n_heads, n_points)
        # 输出 Δp
        self.out_mlp = nn.Sequential(
            nn.Linear(d_model*2, d_model),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, 2),
        )
        # 细化器
        self.refiner = PathRefiner(c_in, d_model=d_model, k_points=9, max_delta_norm=0.5/512.0)

    @staticmethod
    def _t_embed(T: int, device):
        # 简单 sin-cos 时间嵌入：每步一个 16 维
        t = torch.arange(1, T+1, device=device).float()  # 1..T
        t = t / (T + 1)
        dims = torch.arange(8, device=device).float()
        freq = torch.exp(-dims * 2.0)  # 粗略频率
        pe_sin = torch.sin(t[:, None] * freq[None, :] * 2*3.14159)
        pe_cos = torch.cos(t[:, None] * freq[None, :] * 2*3.14159)
        pe = torch.cat([pe_sin, pe_cos], dim=-1)  # (T,16)
        return pe  # (T,16)

    def forward(self, z: torch.Tensor, fmap: torch.Tensor, start: torch.Tensor, goal: torch.Tensor):
        """
        z     : (B, z_dim)
        fmap  : (B, C, Hf, Wf)
        start : (B, 2)  in [0,1]
        goal  : (B, 2)  in [0,1]
        return: (B, T, 2)  in [0,1]
        """
        B = z.size(0); T = self.out_steps; device = z.device
        # 1) 构造并行 Query
        t_emb = self._t_embed(T, device).unsqueeze(0).expand(B, T, -1)      # (B,T,16)
        z_rep = z.unsqueeze(1).expand(B, T, -1)                             # (B,T,z)
        s_rep = start.unsqueeze(1).expand(B, T, -1)                          # (B,T,2)
        g_rep = goal.unsqueeze(1).expand(B, T, -1)                           # (B,T,2)
        q_in  = torch.cat([z_rep, s_rep, g_rep, t_emb], dim=-1)              # (B,T,z+4+16)
        q     = self.q_mlp(q_in)                                             # (B,T,D)
        # 2) 参考坐标：start→goal 线性内插 + 小偏移
        alphas = (torch.arange(1, T+1, device=device).float() / (T+1)).view(1,T,1)  # (1,T,1)
        ref = (1.0 - alphas) * s_rep + alphas * g_rep                        # (B,T,2)
        ref = torch.clamp(ref + 0.02*torch.tanh(self.ref_off(q)), 0.0, 1.0)  # 轻微挪开直线
        # 3) 位置对齐 Deformable Cross-Attn
        # ctx = self.dca(q, fmap, ref_xy=ref, radius=0.08)                     # (B,T,D)
        ctx = self.dca(q, fmap, ref_xy=ref, radius=self.radius)               # (B,T,D)
        # 4) 预测 Δp（相对 ref）
        feat = torch.cat([q, ctx], dim=-1)                                   # (B,T,2D)
        delta = torch.tanh(self.out_mlp(feat)) * (8.0/512.0)                 # 每步最大偏移 ≈ 8 像素
        mid = torch.clamp(ref + delta, 0.0, 1.0)                              # (B,T,2)
        # 5) 细化
        mid_refined = self.refiner(fmap, mid)                                 # (B,T,2)
        return mid_refined
#  ============ CVAE 容器 ============ #
class CVAE(nn.Module):
    """
    将在 trainer 中被调用：
      - encode_from_fmap(fmap, start, goal)   → (mu, logvar)
      - decode(z, fmap, start, goal)          → y_hat(38×2)
      - forward(batch) 返回 output_dict（供 loss 使用）
    """
    def __init__(self, cfg=None):
        super().__init__()
        self.cfg = cfg #or global_cfg
        mcfg = self.cfg
        self.z_dim   = int(mcfg.z_dim)
        self.out_all = int(mcfg.out_steps)         # 40（含起终点）
        self.out_mid = self.out_all - 2            # 38 中间点
        self.c_fmap  = int(mcfg.fmap_channels)     # 与 backbone 对齐

        # 编码/解码
        self.encoder = QuadGAPEncoder(self.c_fmap, self.z_dim)
        # 默认超参（若 cfg 无相应字段，则采用以下默认）
        # dcfg = getattr(self.cfg.model, 'deform_attn', None)
        # d_model  = getattr(dcfg, 'd_model', 128) if dcfg is not None else 128
        # n_heads  = getattr(dcfg, 'n_heads', 4)   if dcfg is not None else 4
        # n_points = getattr(dcfg, 'n_points',16)  if dcfg is not None else 16
        # self.decoder = CoordDecoder(self.c_fmap, self.z_dim, out_steps=self.out_mid,
        #                             d_model=d_model, n_heads=n_heads, n_points=n_points)
        # 从 cvae_core 自身读取注意力与采样参数（与你的 config 片段匹配）
        d_model   = int(getattr(mcfg, 'd_model',   128))
        n_heads   = int(getattr(mcfg, 'n_heads',   4))
        n_points  = int(getattr(mcfg, 'k_points', 16))
        radius_px = float(getattr(mcfg, 'radius_px', 6.0))
        # 将像素半径转成归一化坐标系半径（默认按 512 尺寸归一化；若你后续改变输入尺寸，可改成从 batch 给）
        radius_norm = radius_px / 512.0
        self.decoder = CoordDecoder(self.c_fmap, self.z_dim, out_steps=self.out_mid,
                                    d_model=d_model, n_heads=n_heads, n_points=n_points,
                                    radius_norm=radius_norm)

    # ---------- reparam ----------
    @staticmethod
    def reparameterize(mu, logvar):
        std = (0.5 * logvar).exp()
        eps = torch.randn_like(std)
        return mu + eps * std

    # ---------- 兼容 old-trainer 的接口 ----------
    def encode_from_fmap(self, fmap, start, goal):
        return self.encoder(fmap, start, goal)

    # def decode(self, z, fmap, start, goal):
    #     return self.decoder(z, fmap, start, goal)
    def decode(self, z, fmap, start, goal, gt=None, teacher_ratio: float = 0.0,
               return_debug: bool = False):
        """
        兼容 trainer 的调用签名；当前版本未启用 teacher forcing，先忽略 gt/teacher_ratio。
        return_debug=True 时返回 (mid, dbg)
        """
        mid = self.decoder(z, fmap, start, goal)
        if return_debug:
            dbg = {"parp": None, "perp": None, "gate_t": None}
            return mid, dbg
        return mid

    # ---------- 训练/验证统一前向 ----------
    def forward(self, batch: dict, fmap: torch.Tensor):
        """
        参数:
          batch[Start], batch[Goal] : (B,2)  归一化坐标
          fmap : (B,C,H',W')         来自感知 backbone（CNN-FPN 或 TinyViT）
          可选: batch[split_path]    (B,40,2) 供 loss_split
        返回:
          output_dict 供 loss 使用
        """
        start = batch[New_DataName.Start]  # (B,2)
        goal  = batch[New_DataName.Goal]   # (B,2)

        mu, logvar = self.encoder(fmap, start, goal)
        z = self.reparameterize(mu, logvar)
        mid = self.decoder(z, fmap, start, goal)             # (B,38,2)

        # 拼接起终点 → 40 点
        y_hat = torch.cat([start.unsqueeze(1), mid, goal.unsqueeze(1)], dim=1)

        out = {
            New_DataName.y_hat:   y_hat,
            New_DataName.mu:      mu,
            New_DataName.logvar:  logvar,
            # 供 terrain 相关 loss 采样
            New_DataName.terrain_cost_map: batch[New_DataName.terrain_cost_map],
            # 供 split 对齐
            New_DataName.split_path: batch.get(New_DataName.split_path, None),
            New_DataName.Start:   start,
            New_DataName.Goal:    goal,
        }
        return out
