# src/alignment/dpo.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

# ---------- 工具：路径几何 ----------
def path_length(traj: torch.Tensor) -> torch.Tensor:
    # traj: [B,N,2] -> [B]
    diff = traj[:, 1:] - traj[:, :-1]
    return (diff.norm(dim=-1) + 1e-8).sum(dim=1)

def path_curvature(traj: torch.Tensor) -> torch.Tensor:
    # 简单二阶差分近似曲率范数和（可替换为更严格的离散曲率）
    v1 = traj[:, 1:-1] - traj[:, :-2]
    v2 = traj[:, 2:]   - traj[:, 1:-1]
    # 弯折度：相邻速度向量夹角的正弦近似（或二阶差分范数）
    # 这里用二阶差分范数，稳定可导
    a  = v2 - v1
    curv = a.norm(dim=-1).mean(dim=1)  # [B]
    return curv

def oob_fraction(traj: torch.Tensor) -> torch.Tensor:
    # [0,1] 框外比例
    x_ok = (traj[..., 0] >= 0.0) & (traj[..., 0] <= 1.0)
    y_ok = (traj[..., 1] >= 0.0) & (traj[..., 1] <= 1.0)
    ok = (x_ok & y_ok).float()
    return 1.0 - ok.mean(dim=1)

def sample_gray_along(gray: torch.Tensor, traj: torch.Tensor) -> torch.Tensor:
    """
    gray: [B,1,H,W], 值域建议[0,1]
    traj: [B,N,2] in [0,1]
    return: [B] avg gray along the path
    """
    B, _, H, W = gray.shape
    # grid_sample 需要 [-1,1]
    g = traj.clone()
    g[..., 0] = g[..., 0] * 2 - 1
    g[..., 1] = g[..., 1] * 2 - 1
    g = g.view(B, -1, 1, 2)  # [B,N,1,2]
    samp = F.grid_sample(gray, g, mode="bilinear", align_corners=True)  # [B,1,N,1]
    samp = samp.view(B, -1)  # [B,N]
    return samp.mean(dim=1)


@dataclass
class DPOConfig:
    use: bool = False
    beta: float = 0.1
    lambda_dpo: float = 0.1
    num_candidates: int = 4      # K
    ref_update_every: int = 1000 # step 间隔，用于更新参考策略
    # 打分权重（越小越好）
    w_gray: float = 0.5
    w_oob: float  = 5.0
    w_curv: float = 0.5
    w_len: float  = 0.2
    w_gt: float   = 1.0
    # 能量项用于近似 -log πθ
    w_mse_energy: float = 1.0


class DPOAligner(nn.Module):
    """
    无奖励偏好对齐（EB-DPO 近似）：
      • 对每样本采样 K 条候选，基于可计算指标选 τ+ 与 τ-
      • 参考策略 π_ref 为冻结的学生拷贝（或 EMA）
      • 近似 log πθ(τ|x) - log π_ref(τ|x) ≈ -Eθ(τ,x) + E_ref(τ,x)
      • E 包含：候选 vs 模型确定性输出的 MSE（主导可导项） + 可选几何/能量项
    """
    def __init__(self, cfg: DPOConfig):
        super().__init__()
        self.cfg = cfg

    # --------- 打分（用于选 τ+/τ-）---------
    def score_candidates(
        self,
        input_dict: Dict,
        candidates: torch.Tensor,   # [B,K,N,2], in [0,1]
        gt: Optional[torch.Tensor], # [B,N,2] or None
        gray: Optional[torch.Tensor] # [B,1,H,W] or None
    ) -> torch.Tensor:
        B, K, N, _ = candidates.shape
        flat = candidates.view(B*K, N, 2)

        # 组件
        s_len  = path_length(flat)        # [B*K]
        s_curv = path_curvature(flat)     # [B*K]
        s_oob  = oob_fraction(flat)       # [B*K]

        # 与 GT 的 L2（若无 GT，则置 0）
        if gt is not None:
            gt_rep = gt[:, None, :, :].expand(B, K, N, 2).contiguous().view(B*K, N, 2)
            s_gt   = F.mse_loss(flat, gt_rep, reduction='none').mean(dim=(1,2))
        else:
            s_gt   = torch.zeros_like(s_len)

        # 采样灰度（越低越好），若无灰度，置 0
        if gray is not None:
            # 灰度采样需要逐 batch
            g_list = []
            ofs = 0
            for b in range(B):
                gb = gray[b:b+1]                         # [1,1,H,W]
                xb = flat[ofs:ofs+K]                     # [K,N,2]
                g_list.append(sample_gray_along(gb.expand(K,-1,-1,-1), xb).detach())
                ofs += K
            s_gray = torch.cat(g_list, dim=0)            # [B*K]
        else:
            s_gray = torch.zeros_like(s_len)

        # 总分（越小越好）
        w = self.cfg
        score = (w.w_gray * s_gray +
                 w.w_oob  * s_oob  +
                 w.w_curv * s_curv +
                 w.w_len  * s_len  +
                 w.w_gt   * s_gt)
        return score.view(B, K)

    # --------- 能量（用于 DPO 的 log π 近似）---------
    def energy_model(
        self,
        model,                 # 当前学生：应为 model.generator
        ref_model,             # 冻结参考：generator 的拷贝
        observation: Dict,
        start: torch.Tensor, goal: torch.Tensor,
        traj_plus: torch.Tensor, traj_minus: torch.Tensor  # [B,N,2] in [0,1]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        计算 Eθ(τ±|x) 与 Eref(τ±|x)
        这里只用 MSE(τ, y_det(μ)) 作为可导主项（其余几何项已用于打分）
        """
        fmap = observation["fmap"]
        with torch.no_grad():
            mu_ref, _ = ref_model.encode_from_fmap(fmap, start, goal)
            y_ref     = ref_model.decode(mu_ref, fmap, start, goal)  # [B,N,2]

        mu, _  = model.encode_from_fmap(fmap, start, goal)
        y_det  = model.decode(mu, fmap, start, goal)                 # [B,N,2]

        w = self.cfg.w_mse_energy
        E_theta_plus = w * F.mse_loss(traj_plus,  y_det, reduction='none').mean(dim=(1,2))
        E_theta_minus= w * F.mse_loss(traj_minus, y_det, reduction='none').mean(dim=(1,2))

        with torch.no_grad():
            E_ref_plus  = w * F.mse_loss(traj_plus,  y_ref, reduction='none').mean(dim=(1,2))
            E_ref_minus = w * F.mse_loss(traj_minus, y_ref, reduction='none').mean(dim=(1,2))

        return E_theta_plus, E_theta_minus, E_ref_plus, E_ref_minus

    # --------- DPO 损失 ----------
    def dpo_loss(
        self,
        E_theta_plus: torch.Tensor, E_theta_minus: torch.Tensor,
        E_ref_plus:   torch.Tensor, E_ref_minus:  torch.Tensor
    ) -> torch.Tensor:
        # d = β * ( (E_ref - Eθ)_+ - (E_ref - Eθ)_- )
        beta = self.cfg.beta
        d = beta * ((E_ref_plus - E_theta_plus) - (E_ref_minus - E_theta_minus))
        return -F.logsigmoid(d).mean()

    # --------- 一站式前向：产生 K 候选 → 选偏好 → DPO loss ----------
    @torch.no_grad()
    def _maybe_get_gray(self, input_dict: Dict):
        # 尝试找原灰度图（根据你的工程习惯可改键名）
        for k in ["gray", "img", "image", "heightmap", "map"]:
            if k in input_dict:
                g = input_dict[k]
                if isinstance(g, torch.Tensor) and g.dim() == 4 and g.size(1) == 1:
                    return g
        return None

    def forward(
        self,
        model, ref_model,
        input_dict: Dict,
        observation: Dict,
        start: torch.Tensor, goal: torch.Tensor,
        gt: Optional[torch.Tensor],
        num_candidates: Optional[int] = None
    ) -> Dict[str, torch.Tensor]:
        """
        产生候选并计算 DPO：
          • 先用 encode_from_fmap 得到 μ,logσ，采样 K 组 z_g
          • decode 得到 K 条候选
          • 用 score 选 τ+/τ-；再计算 EB-DPO loss
        """
        K = num_candidates or self.cfg.num_candidates
        fmap = observation["fmap"]
        mu, logvar = model.encode_from_fmap(fmap, start, goal)
        B = mu.size(0)
        std = (0.5*logvar).exp()

        # 采样 K 条候选（含一条 μ 作为确定性候选）
        zs = [mu] + [mu + torch.randn_like(std) * std for _ in range(K-1)]  # list of [B,Z]
        cands = []
        for z in zs:
            y = model.decode(z, fmap, start, goal)  # [B,N,2]
            cands.append(y.unsqueeze(1))
        candidates = torch.cat(cands, dim=1)        # [B,K,N,2]

        gray = self._maybe_get_gray(input_dict)
        scores = self.score_candidates(input_dict, candidates, gt, gray)  # [B,K]
        idx_best  = torch.argmin(scores, dim=1)  # [B]
        idx_worst = torch.argmax(scores, dim=1)  # [B]

        # gather τ+/τ-
        b_idx = torch.arange(B, device=mu.device)
        traj_plus  = candidates[b_idx, idx_best]   # [B,N,2]
        traj_minus = candidates[b_idx, idx_worst]

        # 计算能量与 DPO
        E_theta_p, E_theta_m, E_ref_p, E_ref_m = self.energy_model(
            model, ref_model, observation, start, goal, traj_plus, traj_minus
        )
        loss_dpo = self.dpo_loss(E_theta_p, E_theta_m, E_ref_p, E_ref_m)

        # 记录 margin（仅日志）
        margin = ((E_ref_p - E_theta_p) - (E_ref_m - E_theta_m)).mean()

        return {
            "loss_dpo": loss_dpo,
            "dpo_margin": margin,
            "idx_best": idx_best.detach(),
            "idx_worst": idx_worst.detach(),
        }
