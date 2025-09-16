import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.configs import New_DataName, LossDictKeys
minimum_loss_ratio = 1e-6


def chamfer_1d(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """
    pred, gt : (B, N, 2)  坐标∈[0,1]
    计算双向最近邻距离均值 (Chamfer distance)；返回标量张量。
    """
    B, N, _ = pred.shape
    M       = gt.size(1)

    # (B,N,1,2) - (B,1,M,2) → (B,N,M)
    diff = pred.unsqueeze(2) - gt.unsqueeze(1)
    dist = (diff ** 2).sum(-1)          # L2²
    min_p2g = dist.min(dim=2)[0]        # (B,N)
    min_g2p = dist.min(dim=1)[0]        # (B,M)
    chamfer = (min_p2g.mean() + min_g2p.mean()) * 0.5
    return chamfer


class LossEvaluation(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.th_ep   = 3.0
        self.w_ep    = cfg.w_ep 
        self.w_terr  = cfg.w_terr 
        # self.w_smooth= cfg.w_smooth 
        self.alpha   = cfg.alpha     # 控制地形惩罚陡峭度 8.0
        self.m_plate = cfg.m_plate    # “平坦”窗口一半宽度
        self.w_split = cfg.w_split
        # self.w_cham  = cfg.w_cham 
        self.w_cos   = cfg.w_cos 
        self.w_oob   = cfg.w_oob
        self.w_dis   = cfg.w_dis
        self.w_uniform = cfg.w_uniform
        self.w_jump = cfg.w_jump  
        self.margin  = 0.1 #0.05 
        self.w_lossmap = cfg.w_lossmap
        # ---- VAE KL ----
        self.w_vae_kld = getattr(cfg, "vae_kld_ratio", 0.0)  # 權重系數
        self.kld_beta  = 0.0  # 由 Trainer 每個 epoch 動態設置（β-anneal）

        self.w_map = {
            "loss_ep": cfg.w_ep,
            "loss_terr": cfg.w_terr,
            # "loss_smooth": cfg.w_smooth,
            "loss_split": cfg.w_split,
            "loss_cos": cfg.w_cos,
            "loss_oob": cfg.w_oob,
            "loss_uniform": cfg.w_uniform,
            "loss_jump": cfg.w_jump,
            "loss_map": cfg.w_lossmap,
            "loss_dis": cfg.w_dis,                                                                  
        }

    # ---------- util ----------
    @staticmethod
    def _softplus(x):          # 避免 nan
        return F.softplus(x, beta=1, threshold=20)

    # just for test
    # def terrain_aware_loss(self, y_pred, fmap):
    #     B, T, _ = y_pred.shape
    #     y_pixel = (y_pred * fmap.shape[-1]).long().clamp(0, fmap.shape[-1]-1)
    #     fmap_value = fmap[torch.arange(B).unsqueeze(1), y_pixel[:, :, 1], y_pixel[:, :, 0]]  # 提取fmap对应位置值
    #     diff = fmap_value[:,1:] - fmap_value[:,:-1]
    #     loss = torch.mean(torch.abs(diff))
    #     return loss

    # ---------- 主入口 ----------
    def forward(self, output_dict):

        # ========= 必要张量 =========
        y_hat      = output_dict[New_DataName.y_hat]          # (B,N,2)  0‑1
        start      = output_dict[New_DataName.Start]          # (B,2)
        goal       = output_dict[New_DataName.Goal]           # (B,2)
        bmap       = output_dict[New_DataName.terrain_cost_map]      # (B,H,W) or (B,1,H,W)
        split_path = output_dict.get(New_DataName.split_path) # (B,N,2) or None

        extra_dict = {} 

        xy = y_hat    
        oob_raw = (F.relu(-xy) + F.relu(xy - 1.0)).pow(2)  

        dist_to_edge = torch.minimum(xy, 1 - xy).min(-1).values        # (B,N)
        margin_penalty = F.relu(self.margin - dist_to_edge).pow(2)     # (B,N)

        # ❶ 把每个点的两项相加
        oob_point = oob_raw.sum(-1) + margin_penalty                   # (B,N)
        # ❷ clip 到 ≥0，防止任何负贡献
        oob_point = oob_point.clamp_min(0.0)

        # ❸ 再对 Batch+Point 求均值
        loss_oob = F.relu(oob_point.mean())

        if bmap.dim() == 4:                                   # (B,1,H,W) → (B,H,W)
            bmap = bmap.squeeze(1)
        B, H, W = bmap.shape  

        # ========= 1) 端点 Huber =========
        loss_ep = (
            F.smooth_l1_loss(y_hat[:, 0], start, reduction='none').sum(-1) +
            F.smooth_l1_loss(y_hat[:, -1], goal,  reduction='none').sum(-1)
        ).mean()

        # ---------- 连续采样 ----------
        # ① 计算 cost_map（按新的梯度图逻辑：白亮=陡壁）
        grad_x = bmap - torch.roll(bmap, shifts=1, dims=2)
        grad_y = bmap - torch.roll(bmap, shifts=1, dims=1)
        slope2 = (grad_x ** 2 + grad_y ** 2)
        cost_h = torch.log1p(3.0 * slope2)          # log-scale, β=3
        cost_map = cost_h                           # (B,H,W)

        # # ② 对每段插 7 个点 (共 8 采样)
        B, N, _ = y_hat.shape

        T   = 8                                      # 8 采样点
        seg = torch.linspace(0, 1, steps=T, device=y_hat.device)   # (T,)
        y_l = y_hat[:, :-1]                          # (B,N-1,2)
        y_r = y_hat[:, 1:]                           # (B,N-1,2)

        # (T,B,N-1,2)  ← 先在 batch / 段 维度上 broadcast，再拉平成 (B, T*(N-1), 2)
        xy_mid = (1 - seg.view(T, 1, 1, 1)) * y_l.unsqueeze(0) + \
                 (    seg.view(T, 1, 1, 1)) * y_r.unsqueeze(0)
        xy_mid = xy_mid.permute(1,0,2,3).reshape(B, -1, 2)  # (B, T*(N-1), 2)

        # # grid_sample 需要 [-1,1]
        g = xy_mid.clone()
        g[..., 0] = g[..., 0] * 2 - 1
        g[..., 1] = g[..., 1] * 2 - 1
        cost_smp = F.grid_sample(cost_map.unsqueeze(1),   # (B,1,H,W)
                                 g.view(B, -1, 1, 2),
                                 align_corners=True,
                                 mode='bilinear').view(B, -1)   # (B,7*(N-1))

        # max_terr = cost_smp.view(B, -1).max(dim=1)[0]
        # loss_terr = torch.exp(4.0 * max_terr).mean()
        # loss_terr = torch.tensor(0.0, device=y_hat.device)
        # ---- cost_smp: (B, T*(N-1)) 已有 ----
        # 软极大：top-k + softmax 混合，既看均值也看“险点”
        alpha = 6.0                      # 越大越向最坏点倾斜
        cost_mean = cost_smp.mean(dim=1) # [B]
        cost_worst = torch.logsumexp(alpha * cost_smp, dim=1) / alpha
        # 也可以加一个 top-k，k 取 10% 段点
        k = max(1, cost_smp.size(1) // 10)
        topk = torch.topk(cost_smp, k, dim=1).values.mean(dim=1)
        loss_terr = (0.5 * cost_mean + 0.3 * cost_worst + 0.2 * topk).mean()


        # v1 = y_hat[:,1:-1] - y_hat[:,:-2] # 向量 1
        # v2 =y_hat[:,2:] - y_hat[:,1:-1] # 向量 2
        # cos_theta = F.cosine_similarity(v1, v2, dim=-1) # [-1,1]
        # k = torch.acos(cos_theta.clamp(-0.999, 0.999)) # 曲率角
        # k_max = np.deg2rad(45.0)
        # loss_smooth = F.relu(k - k_max).pow(2).mean()

        # ---------- 4) Split‑Path & Chamfer ----------
        loss_split = torch.tensor(0.0, device=y_hat.device)

        y_mid   = y_hat#[:,1:-1]                # (B,18,2)
        split_m = split_path#[:,1:-1]

        w_c, w_v, w_curve = 2.0, 4.0, 1.0

        # 1) 绝对坐标 L1
        loss_coord = F.smooth_l1_loss(y_mid, split_m, reduction='mean')

        # 2) 段向量 L1  (平移不变形状约束)
        v_pred = y_mid[:, 1:] - y_mid[:, :-1]      # (B,37,2)
        v_gt   = split_m[:, 1:] - split_m[:, :-1]
        loss_vec = F.smooth_l1_loss(v_pred, v_gt, reduction='mean')

        # 3) 曲率 (相邻向量夹角 MSE)
        cos_pred = F.cosine_similarity(v_pred[:, :-1], v_pred[:, 1:], dim=-1)
        cos_gt   = F.cosine_similarity(v_gt[:, :-1],   v_gt[:, 1:],   dim=-1)
        loss_curve = F.mse_loss(cos_pred, cos_gt)

        # 汇总 → 仍叫 loss_split 方便 trainer 记录
        loss_split = (w_c * loss_coord +
                      w_v * loss_vec   +
                      w_curve * loss_curve) * self.w_split

        # trycosine smooth loss
        # v1 = y_hat[:,1:] - y_hat[:,:-1]          # (B,19,2)
        # v2 = y_hat[:,2:] - y_hat[:,1:-1]         # (B,18,2)
        # cos = F.cosine_similarity(v1[:,:-1], v2, dim=-1)  # (B,18)
        loss_cos = torch.tensor(0.0, device=y_hat.device) #(1-cos).mean()

        if self.w_dis > 0:
            # y_hat : (B, N, 2)  这里 N = 40
            step = y_hat[:, 1:] - y_hat[:, :-1]          # (B, N-1, 2)
            dist = step.norm(dim=-1)                     # (B, N-1)
            var  = dist.var(dim=-1, unbiased=False)      # (B,)
            loss_dis = torch.tensor(0.0, device=y_hat.device) #var.mean() * self.w_dis
            # ---------- 新增：首尾段方差惩罚 ----------
            # 取出首尾两段位移
            first_step = y_hat[:, 1] - y_hat[:, 0] # (B, 2)
            last_step = y_hat[:, -1] - y_hat[:, -2] # (B, 2)

            # 计算首尾位移长度
            first_dist = first_step.norm(dim=-1) # (B,)
            last_dist = last_step.norm(dim=-1) # (B,)

            # 把首尾位移拼在一起，计算每组的方差
            end_pair = torch.stack([first_dist, last_dist], dim=-1) # (B, 2)
            var_end = end_pair.var(dim=-1, unbiased=False) # (B,)

            # 如果首尾方差显著大于主段方差，则额外惩罚
            # 这里用“var_end > k * var”作为阈值，k 可调（示例取 2.0）
            k = 2.0
            w_2nd = 2.0
            w_last2 = 2.0
            mask = var_end > k * var # (B,) bool
            extra_loss = (var_end * mask.float()).mean() * self.w_dis * 0.5 # 权重可再调
            loss_2nd = ((y_hat[:,1] - split_m[:,1])**2).sum(-1).mean()
            loss_last2 = ((y_hat[:,-2] - split_m[:,-2])**2).sum(-1).mean()
            loss_2point = w_2nd * loss_2nd + w_last2 * loss_last2

            loss_dis = loss_dis + extra_loss + loss_2point
        else:
            loss_dis = 0

        # ---------- 5) 均匀间距 loss_uniform ----------
        loss_uniform = torch.tensor(0.0, device=y_hat.device)
        # if self.w_map.get("loss_uniform", 0) > 0:
        #     step = y_hat[:, 1:] - y_hat[:, :-1]          # (B, N-1, 2)
        #     dist = step.norm(dim=-1)                     # (B, N-1)
        #     mean_dist = dist.mean(dim=-1, keepdim=True)  # (B,1)
        #     var_dist = ((dist - mean_dist) ** 2).mean()  # 标量

        #     # 计算所有段距离的最小值，如果小于阈值就额外惩罚
        #     min_dist = dist.min()
        #     min_dist_loss = F.relu(0.0003 - min_dist) * self.w_map["loss_uniform"] * 10
        #     loss_uniform = var_dist * self.w_map["loss_uniform"] + min_dist_loss

        # ---------- 2‑b) 地形跳变 loss_jump ----------
        loss_jump = torch.tensor(0.0, device=y_hat.device)
        tau = 0.5
        if self.w_map.get("loss_jump", 0) > 0:
            # 重组为 (B,N-1,T) 便于取最大跳变；T=8
            jump = cost_smp.view(B, N-1, -1).max(dim=-1)[0]   # (B,N-1) 选择段内最险点
            diff = jump[:,1:] - jump[:,:-1]                   # 连续段差
            # loss_jump = F.relu(diff.abs() - 0.15).pow(2).mean() * self.w_map["loss_jump"]
            loss_jump = F.relu(diff.abs() - 0.15).clamp_max(tau).mean() * self.w_map["loss_jump"]
     
        # just for test
        loss_map = torch.tensor(0.0, device=y_hat.device) #self.terrain_aware_loss(y_hat, cost_map)

        # ---- VAE KL （若模型有輸出 mu/logvar）----
        vae_kld_loss = torch.tensor(0.0, device=y_hat.device) #y_hat.new_zeros(())
        mu     = output_dict.get(New_DataName.mu, None)
        logvar = output_dict.get(New_DataName.logvar, None)
        if (mu is not None) and (logvar is not None):
            # KL(q||N(0,I)) = -0.5 * sum(1 + logvar - mu^2 - exp(logvar))
            # 這裡按 batch、latent 維度做 mean，得到標量
            vae_kld = -0.5 * (1.0 + logvar - mu.pow(2) - logvar.exp())
            vae_kld = vae_kld.mean()
            vae_kld_loss = self.w_vae_kld * self.kld_beta * vae_kld

        # ---------- 汇总 ----------
        total = (
            self.w_ep     * loss_ep     +
            self.w_terr   * loss_terr   +
            self.w_uniform * loss_uniform +
            # self.w_smooth * loss_smooth +
            self.w_split  * loss_split  + 
            # self.w_cham   * loss_cham   + 
            self.w_cos    * loss_cos    +
            self.w_oob    * loss_oob    +
            self.w_dis    * loss_dis    +
            self.w_jump   * loss_jump   +
            self.w_lossmap   * loss_map +
            vae_kld_loss
        )

        lossDICTtotal =  {
            LossDictKeys.loss:         total,
            LossDictKeys.loss_total:   total,
            LossDictKeys.loss_ep:      loss_ep,
            LossDictKeys.loss_terr:    loss_terr,
            # LossDictKeys.loss_smooth:  loss_smooth,
            LossDictKeys.loss_split:   loss_split,
            # LossDictKeys.loss_cham:    loss_cham,
            LossDictKeys.loss_cos:     loss_cos,
            LossDictKeys.loss_oob:     loss_oob,
            LossDictKeys.loss_dis:     loss_dis,
            LossDictKeys.loss_uniform: loss_uniform,
            LossDictKeys.loss_jump:     loss_jump,
            LossDictKeys.loss_map:   loss_map,
            LossDictKeys.loss_vae_kld:      vae_kld_loss,
        }

        return lossDICTtotal, extra_dict
