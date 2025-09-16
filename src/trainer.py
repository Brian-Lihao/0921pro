from torch.nn.parameter import UninitializedParameter 
import copy
import time
import os

import torch
from tqdm import tqdm
import numpy as np
import torch as th
import torch.nn.init as init  
from torch import nn as nn
import cv2
import pandas as pd
import matplotlib.pyplot as plt
import os.path as osp
from datetime import datetime
from contextlib import nullcontext
from pathlib import Path
import torch.nn.functional as F
from contextlib import nullcontext
import copy

from src.configs import cfg, LossDictKeys, New_DataName
from src.utils.functions import get_device, to_device, release_cuda
from src.utils.logger import TBLogger    
from src.utils.val_aug import jitter_batch  
from src.data_loader import train_eval_data_loader
from src.model.model import SwinPathBiCVAE
from src.loss import LossEvaluation
from src.backbones.dpo import DPOAligner, DPOConfig
from src.utils.teacher import get_teacher_traj

# from src.debug_viz import TBLogger, viz_fmap, paths_heatmap, plot_parp_perp, ensure_dir

class Trainer:
    def __init__(self, cfgs: cfg):
        """
        初始化函数，根据配置对象cfgs设置训练和评估所需的各项参数和组件。
        
        参数:
        - cfgs: 包含所有配置信息的配置对象。
        """
        # 设置设备（如GPU或CPU）
        self.device = get_device(device=cfgs.device)

        # 是否加载快照和设置模型名称
        self.snapshot = cfgs.load_snapshot
        self.name = cfgs.name

        # 初始化数据加载器和验证数据加载器
        self.data_loader, self.val_loader = train_eval_data_loader(cfg=cfgs.data)
        # 初始化模型并将其移动到指定设备
        self.model = SwinPathBiCVAE(cfgs=cfgs.model).to(self.device)
        # self._kaiming_init() 
        self.min_of_k = int(getattr(cfgs.training,"min_of_k",1))

        # 加载训练配置
        self.cfg = cfgs.training
        self.cfgs = cfgs
        self.w_eval = self.cfg.w_eval
        self.max_epoch = self.cfg.max_epoch
        self.max_iteration_per_epoch = self.cfg.max_iteration_per_epoch
        # 初始化优化器
        self.optimizer = th.optim.Adam(self.model.parameters(), lr=self.cfg.lr,weight_decay=self.cfg.weight_decay)
        # 初始化学习率调度器
        self.scheduler = th.optim.lr_scheduler.StepLR(self.optimizer, self.cfg.lr_decay_steps,
                                                    gamma=self.cfg.lr_decay)

        # 初始化迭代次数和纪元数
        self.iteration = 0
        self.epoch = 0
        self.training = True
        # 梯度累积步骤数
        self.grad_acc_steps = self.cfg.grad_acc_steps
        # 初始化最佳损失为无穷大
        self.best_loss = np.inf

        # 初始化损失函数和评估器
        self.loss_func = LossEvaluation(cfg=cfgs.loss_eval).to(self.device)
        self.evaluator = LossEvaluation(cfg=cfgs.loss_eval).to(self.device)

        # ---- KL β-anneal 參數 ----
        self.kld_beta_start = getattr(cfgs.loss_eval, "vae_kld_beta_start", 0.0)
        self.kld_beta_end   = getattr(cfgs.loss_eval, "vae_kld_beta_end", 1.0)
        self.kld_warmup_ep  = getattr(cfgs.loss_eval, "vae_kld_warmup_epochs", 10)

        log_dir = Path(cfgs.logger.log_name) / cfgs.name
        self.logger = TBLogger(log_dir)   
        for name, param in self.model.named_parameters():
            # ↓ 只给已初始化参数加 hook
            if isinstance(param, UninitializedParameter):
                continue
            param.register_hook(
                lambda grad, name=name: self._grad_hook(grad, name))
            
        # add csv
        self.vis_dir   = Path(cfgs.csv_output_dir) / "val_vis"
        self.vis_dir.mkdir(parents=True, exist_ok=True)

        # === 1.1  Early‑Stopping / Best‑Model 相关状态 ===
        self.best_sp            = float('inf')   # 当前最优 split 路径误差
        self.epochs_no_improve   = 0
        self.patience           = getattr(cfgs, "patience", 200)   # 无提升最多容忍多少 epoch
        self.ckpt_dir           = Path(cfgs.ckpt_dir) if hasattr(cfgs, "ckpt_dir") else Path("./checkpoints")
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

        ### ONLY DEBUG
        # ── put near other self.debug_* definitions ──
        self.debug_n_samples   = getattr(cfgs.training, "debug_n_samples", 2)     # 每 batch 取前 n
        self.debug_max_total   = getattr(cfgs.training, "debug_max_total", 50)    # 全程最多 50 条
        self._debug_saved      = 0                                                # 已保存计数
        self.debug_txt         = (self.vis_dir / "pred_vs_gt.txt").open("a", encoding="utf-8")

        # 调试输出设置（沿用已扩展的 TBLogger，可直接 plot_* / add_hist）
        self.debug_every_epochs = getattr(cfg, "debug_every_epochs", 1)
        self._best_epoch_cache = None  # 存放当轮最优样本的信息
        # --- DPO 对齐器（可选） ---
        self.use_dpo = True #if self.epoch > 10 else False
        if self.use_dpo:
            dpo_cfg = DPOConfig(
                use=True,
                beta=self.cfg.dpo.beta,
                lambda_dpo=self.cfg.dpo.lambda_dpo,
                num_candidates=self.cfg.dpo.num_candidates,
                ref_update_every=self.cfg.dpo.ref_update_every,
                w_gray=self.cfg.dpo.w_gray,
                w_oob=self.cfg.dpo.w_oob,
                w_curv=self.cfg.dpo.w_curv,
                w_len=self.cfg.dpo.w_len,
                w_gt=self.cfg.dpo.w_gt,
                w_mse_energy=self.cfg.dpo.w_mse_energy,
            )
            self.dpo = DPOAligner(dpo_cfg).to(self.device)
            # 冻结参考策略（学生 generator 的拷贝）
            self.ref_generator = copy.deepcopy(self.model.generator).to(self.device).eval()
            for p in self.ref_generator.parameters():
                p.requires_grad_(False)
            self._last_ref_update = 0

    def __del__(self):
        if hasattr(self, "debug_txt") and not self.debug_txt.closed:
            self.debug_txt.close()

    # --------------------------------------------------
    def _adjust_curriculum(self):
        """epoch‑wise 动态调整部分 loss 权重"""
        e = self.epoch
        # —— teacher forcing 比例：前20ep 线性 1.0→0.0，之后恒 0
        if e <= 20:
            self.teacher_ratio = float(max(0.0, 1.0 - e / 20.0))
        else:
            self.teacher_ratio = 0.0
        
        if e < 20:                 # warm‑up 形状为主
            # self.loss_func.w_map['loss_uniform'] = 10
            self.loss_func.w_map['loss_split']  = 350
            # self.loss_func.w_map['loss_cham']    = 2
            # self.loss_func.w_map['loss_smooth']  = 10
            self.loss_func.w_map['loss_terr']    = 40
        elif e < 50:              # 过渡期
            # self.loss_func.w_map['loss_uniform'] = 20
            # self.loss_func.w_map['loss_smooth']  = 30
            self.loss_func.w_map['loss_split']  = 300 
            # self.loss_func.w_map['loss_cham']    = 5
            self.loss_func.w_map['loss_terr']   = 10
        else:                     # 避障为主
            # self.loss_func.w_map['loss_uniform'] = 30
            # self.loss_func.w_map['loss_smooth']  = 40
            self.loss_func.w_map['loss_split']  = 200
            # self.loss_func.w_map['loss_cham']    = 10
            self.loss_func.w_map['loss_terr']    = 80

    # TODO: check later
    def save_snapshot(self, filename):
        model_state_dict = self.model.state_dict()

        # save model
        state_dict = {'model': model_state_dict}
        th.save(state_dict, filename)
        self.logger.info('Model saved to "{}"'.format(filename))

        # save snapshot
        state_dict['epoch'] = self.epoch
        state_dict['iteration'] = self.iteration
        snapshot_filename = osp.join(str(self.name) + 'snapshot.pth.tar')
        state_dict['optimizer'] = self.optimizer.state_dict()
        if self.scheduler is not None:
            state_dict['scheduler'] = self.scheduler.state_dict()
        th.save(state_dict, snapshot_filename)
        self.logger.info('Snapshot saved to "{}"'.format(snapshot_filename))

    # TODO: check later
    def load_snapshot(self, snapshot):
        self.logger.info('Loading from "{}".'.format(snapshot))
        state_dict = th.load(snapshot, map_location=th.device('cpu'))

        # Load model
        model_dict = state_dict['model']
        self.model.load_state_dict(model_dict, strict=False)

        # log missing keys and unexpected keys
        snapshot_keys = set(model_dict.keys())
        model_keys = set(self.model.state_dict().keys())
        missing_keys = model_keys - snapshot_keys
        unexpected_keys = snapshot_keys - model_keys
        if len(missing_keys) > 0:
            message = f'Missing keys: {missing_keys}'
            self.logger.error(message)
        if len(unexpected_keys) > 0:
            message = f'Unexpected keys: {unexpected_keys}'
            self.logger.error(message)
        self.logger.info('Model has been loaded.')

        # Load other attributes
        if 'epoch' in state_dict:
            self.epoch = state_dict['epoch']
            self.logger.info('Epoch has been loaded: {}.'.format(self.epoch))
        if 'iteration' in state_dict:
            self.iteration = state_dict['iteration']
            self.logger.info('Iteration has been loaded: {}.'.format(self.iteration))
        if 'optimizer' in state_dict and self.optimizer is not None:
            try:
                self.optimizer.load_state_dict(state_dict['optimizer'])
                self.logger.info('Optimizer has been loaded.')
            except:
                print("doesn't load optimizer")
        if 'scheduler' in state_dict and self.scheduler is not None:
            try:
                self.scheduler.load_state_dict(state_dict['scheduler'])
                self.logger.info('Scheduler has been loaded.')
            except:
                print("doesn't load scheduler")

    def set_train_mode(self):
        self.training = True
        self.model.train()
        th.set_grad_enabled(True)

    def set_eval_mode(self):
        self.training = False
        self.model.eval()
        th.set_grad_enabled(False)


    def optimizer_step(self, iteration):
        """梯度裁剪，防止梯度爆炸"""
        if iteration % self.grad_acc_steps == 0:
            scale = 1.0 / self.grad_acc_steps
            for p in self.model.parameters():
                if p.grad is not None:
                    p.grad.mul_(scale)

            # 加入梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)  # 1.0 是最大裁剪值

            self.optimizer.step()
            self.optimizer.zero_grad()
            
    @torch.no_grad()
    def validate_epoch(self, epoch_idx: int):
        if self.val_loader is None:
            return

        self.model.eval()
        sum_ep = sum_sp = sum_terr = sum_sm = 0.0; n_total = 0

        best_sp = float("inf"); best_pack = None   # 存储本轮最好样本
        for batch in tqdm(self.val_loader, desc=f"[VAL] {epoch_idx}", leave=False):
            if cfg.validation.enable:
                batch = jitter_batch(
                    batch,
                    max_shift = cfg.validation.pixel_shift_px,
                    n_variants= cfg.validation.n_variants,
                    n_split   = cfg.validation.split_point,
                )
            # ---------- 前向 ----------
            batch = {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in batch.items()}
            output = self.model(batch)
            loss_d,_ = self.loss_func(output)

            bs = output[New_DataName.y_hat].size(0)
            n_total   += bs
            sum_ep    += loss_d[LossDictKeys.loss_ep].item()     * bs
            sum_sp    += loss_d[LossDictKeys.loss_split].item()  * bs
            sum_terr  += loss_d[LossDictKeys.loss_terr].item()   * bs
            # sum_sm    += loss_d[LossDictKeys.loss_smooth].item() * bs

            # ---------- 拿本 batch 最小 sp 作为候选 ----------
            sp_sample = loss_d[LossDictKeys.loss_split].item()
            if sp_sample < best_sp:
                best_sp  = sp_sample
                best_pack = (batch, output)      # 保存 batch+预测

        # ---------- 均值 ----------
        avg_sp   = sum_sp   / n_total

        # ---------- 保存最佳可视化 ----------
        # if best_pack is not None:
        #     b, o = best_pack
        #     img   = b[New_DataName.rgb_map][0].cpu().numpy().transpose(1,2,0)  # (128,128,3)
        #     img   = (img * 255).astype(np.uint8)
        #     wp_n  = o[New_DataName.y_hat][0].cpu().numpy()                    # (N,2)
        #     gt    = o[New_DataName.split_path][0].cpu().numpy()
        #     # 绘制
        #     fig, ax = plt.subplots()
        #     ax.imshow(img); ax.plot(wp_n[:,0]*256, wp_n[:,1]*256, "r.-")
        #     ax.plot(gt[:,0]*256, gt[:,1]*256, "b.-")
        #     # plot green-start point and red-end point
        #     sx, sy = b[New_DataName.Start][0].cpu().numpy() * 256
        #     gx, gy = b[New_DataName.Goal ][0].cpu().numpy() * 256
        #     ax.plot(sx, sy, "gx"); ax.plot(gx, gy, "rx")
        #     self.total_best = loss_d[LossDictKeys.loss_total].item()
        #     # vis_path = self.vis_dir / f"epoch{epoch_idx:03d}_sp{sp_best:.3f}.png"
        #     vis_path = self.vis_dir / f"epoch{epoch_idx:03d}_total{self.total_best:.3f}.png"
        #     fig.savefig(vis_path, dpi=150); plt.close(fig)
        # 在 best_pack 确定后，重采 K 候选（无梯度），并缓存到 _best_epoch_cache
        if best_pack is not None:
            b, o = best_pack
            with torch.no_grad():
                obs = self.model.perception(b)
                fmap = obs["fmap"]
                start = b[New_DataName.Start]; goal = b[New_DataName.Goal]
                mu, logvar = self.model.generator.encode_from_fmap(fmap, start, goal)
                std = (0.5*logvar).exp()
                K = max(4, self.min_of_k)
                eps = torch.randn((K,)+mu.shape, device=mu.device)
                cands = []
                for k in range(K):
                    z = mu + eps[k] * std
                    y = self.model.generator.decode(z, fmap, start, goal)   # [B,N,2]
                    cands.append(y)
                C = torch.stack(cands, dim=0)  # [K,B,N,2]

                b0 = 0
                self._best_epoch_cache = dict(
                    fmap=fmap[b0:b0+1].cpu(),
                    raw=b[New_DataName.terrain_cost_map][b0:b0+1].cpu(),   # 原灰度图
                    gt=b[New_DataName.split_path][b0].cpu(),
                    pred=o[New_DataName.y_hat][b0].cpu(),
                    cands=C[:, b0].cpu(),
                    H=fmap.size(-2), W=fmap.size(-1),
                    parp=None, perp=None, gate_t=None,           # ← 占位，避免 KeyError
                    mu=mu.detach().cpu(), std=std.detach().cpu()
                )
                img   = b[New_DataName.rgb_map][0].cpu().numpy().transpose(1,2,0)  # (128,128,3)
                img   = (img * 255).astype(np.uint8)
                wp_n  = o[New_DataName.y_hat][0].cpu().numpy()                    # (N,2)
                gt    = o[New_DataName.split_path][0].cpu().numpy()
                fig, ax = plt.subplots()
                ax.imshow(img); ax.plot(wp_n[:,0]*256, wp_n[:,1]*256, "r.-")
                ax.plot(gt[:,0]*256, gt[:,1]*256, "b.-")
                # plot green-start point and red-end point
                sx, sy = b[New_DataName.Start][0].cpu().numpy() * 256
                gx, gy = b[New_DataName.Goal ][0].cpu().numpy() * 256
                ax.plot(sx, sy, "gx"); ax.plot(gx, gy, "rx")
                self.total_best = loss_d[LossDictKeys.loss_total].item()
                # vis_path = self.vis_dir / f"epoch{epoch_idx:03d}_sp{sp_best:.3f}.png"
                vis_path = self.vis_dir / f"epoch{epoch_idx:03d}_total{self.total_best:.3f}.png"
                fig.savefig(vis_path, dpi=150); plt.close(fig)


        if avg_sp < self.best_sp:
            # improved = True
            self.best_sp          = avg_sp
            self.epochs_no_improve = 0
            ckpt_path = self.ckpt_dir / f"best_ep{epoch_idx:03d}_sp{avg_sp:.4f}.pth"
            torch.save(self.model.state_dict(), ckpt_path)#"best_model.pth")
            print(f"[VAL] ✅  New best sp={avg_sp:.4f}, model saved to {ckpt_path}")
        else:
            self.epochs_no_improve += 1
            print(f"[VAL] 🔸  Now: {avg_sp:.4f} No improv. best_sp={self.best_sp:.4f} ▸ {self.epochs_no_improve}/{self.patience}")

        self.model.train()
        return best_pack

    def step(self, data_dict):
        """
        执行单次训练或推理步骤。

        参数:
        - data_dict (dict): 包含输入数据的字典，通常包括特征和标签等信息。

        返回:
        - output_dict (dict): 模型的输出结果。
        - loss_dict (dict): 计算得到的损失值字典。
        """
        # 将数据字典中的所有张量移动到指定设备（如GPU）
        data_dict = to_device(data_dict, device=self.device)
        # 确保 terrain_cost_map 的形状是 (B, W, H)
        # if New_DataName.rgb_map in data_dict and data_dict[New_DataName.rgb_map].dim() == 4:
        #     data_dict[New_DataName.rgb_map] = data_dict[New_DataName.rgb_map].squeeze(1)
        
        # 确保 last_poses 是张量
        if isinstance(data_dict.get(New_DataName.last_poses, None), list):
            data_dict[New_DataName.last_poses] = torch.stack([torch.tensor(p) for p in data_dict[New_DataName.last_poses]]).to(self.device)
        elif data_dict.get(New_DataName.last_poses, None) is not None:
            data_dict[New_DataName.last_poses] = data_dict[New_DataName.last_poses].clone().detach().to(self.device)
        
        # # 使用模型对输入数据进行前向传播，获取输出结果
        # output_dict = self.model(data_dict)

        # # 使用损失函数计算输出结果对应的损失值
        # loss_dict, extra_dict = self.loss_func(output_dict)
        # ========== Min-of-K（Best-of-K） ==========
        # 思路：前向取樣 K 次 → 用與 GT 的 MSE 作為便宜分數 → 對每個 batch item 選出最佳候選
        #  # 然後只在 "最佳候選" 上計算完整任務損失並反傳
        # ========= Memory-Efficient Min-of-K =========
        K = self.min_of_k if self.model.training else 1 # 驗證階段如需觀察覆蓋，可把這裡改成 self.min_of_k
        # print(f"value of k:{K}")
        if (not self.model.training) or (K <= 1):
            # 常規路徑
            output_dict = self.model(data_dict)
            loss_dict, extra_dict = self.loss_func(output_dict)
        else:
            # ---- 1) 無梯度的「候選打分」階段：只為選 z，不留圖 ----
            start = data_dict[New_DataName.Start]
            goal  = data_dict[New_DataName.Goal]
            split_gt = data_dict[New_DataName.split_path]        # (B,N,2)
            with torch.no_grad():
                # 只算一次感知編碼
                obs = self.model.perception(data_dict)           # {"fmap","gvec"}

                fmap = obs["fmap"]
                # 统一：改用 3×3 区域池化的编码入口，避免维度 196/1732 不匹配
                mu, logvar = self.model.generator.encode_from_fmap(fmap, start, goal)

                std = (0.5 * logvar).exp()
                # K 個 eps，先抽好以便稍後復現同一個最佳 z
                eps = torch.randn((K,)+mu.shape, device=mu.device)
                # 逐個 z 解碼 → 路徑 → MSE 打分（省顯存、不留圖）
                cand_paths = []
                for k in range(K):
                    z_k = mu + eps[k] * std
                    # y_k = self.model.generator.decode(z_k, fmap, start, goal)   # (B,N,2)
                    y_k = self.model.generator.decode(
                        z_k, fmap, start, goal,
                        gt=split_gt, teacher_ratio=self.teacher_ratio
                    )   # (B,N,2)
                    cand_paths.append(y_k)

                Y = torch.stack(cand_paths, dim=0)                              # (K,B,N,2)

                mse_coord = ((Y - split_gt.unsqueeze(0))**2).mean(dim=(2,3))
                Vp = Y[:, :, 1:, :] - Y[:, :, :-1, :]
                Vg = (split_gt[:, 1:, :] - split_gt[:, :-1, :]).unsqueeze(0)
                mse_vec = (Vp - Vg).pow(2).mean(dim=(2,3))
                Cp = Y[:, :, 2:, :] - 2*Y[:, :, 1:-1, :] + Y[:, :, :-2, :]
                Cg = (split_gt[:, 2:, :] - 2*split_gt[:, 1:-1, :] + split_gt[:, :-2, :]).unsqueeze(0)
                mse_curv = (Cp - Cg).pow(2).mean(dim=(2,3))
                # 最终打分（可微调：直线化时提高曲率权重）

                score = 0.5 * mse_coord + 0.3 * mse_vec + 0.2 * mse_curv        # (K,B)
                best_idx = torch.argmin(score, dim=0)                            # (B,)
                # 取每个样本对应的最佳 eps，后面带梯度重算一次
                # best_idx = torch.argmin(score, dim=0)  
                # y_best = self.model.generator.decode(
                #     z_best, fmap, start, goal,
                #     gt=split_gt, teacher_ratio=self.teacher_ratio
                # )
                # 取每個樣本對應的最佳 eps，後面帶梯度重算一次
                B = mu.size(0)
                ar = torch.arange(B, device=mu.device)
                best_eps = eps[best_idx, ar, ...].detach()                      # (B,zd)

            # ---- 2) 正式帶梯度的「單次前向」：重算同一個最佳 z ----
            # 重新算一遍感知與 encode（帶梯度）
            obs = self.model.perception(data_dict)

            fmap = obs["fmap"]
            # 同样这里也使用统一入口
            mu, logvar = self.model.generator.encode_from_fmap(fmap, start, goal)

            std = (0.5 * logvar).exp()
            # z_best = mu + best_eps * std                                        # 梯度僅流向 μ,logσ
            # y_best = self.model.generator.decode(z_best, fmap, start, goal)
            z_best = mu + best_eps * std                                        # 梯度仅流向 μ,logσ
            y_best = self.model.generator.decode(
                z_best, fmap, start, goal, gt=split_gt, teacher_ratio=self.teacher_ratio
            )
            # —— 记录当轮“最优样本”用于调试 —— 
            with torch.no_grad():
                b0 = 0
                y_dbg, dbg = self.model.generator.decode(z_best, fmap, start, goal, return_debug=True)
                H, W = fmap.shape[-2], fmap.shape[-1]
                self._best_epoch_cache = dict(
                    loss=None,
                    fmap=fmap[b0:b0+1].detach().cpu(),
                    gt=split_gt[b0].detach().cpu(),
                    pred=y_dbg[b0].detach().cpu(),
                    cands=Y[:, b0].detach().cpu(),
                    H=H, W=W,
                    parp=(dbg.get("parp")[b0] if dbg.get("parp") is not None else None),
                    perp=(dbg.get("perp")[b0] if dbg.get("perp") is not None else None),
                    gate_t=(dbg.get("gate_t")[b0] if dbg.get("gate_t") is not None else None),
                    mu=mu.detach().cpu(), std=std.detach().cpu(),
                    score_coord=mse_coord[:, b0].detach().cpu(),
                    score_vec=mse_vec[:, b0].detach().cpu(),
                    score_curv=mse_curv[:, b0].detach().cpu()
                )
               


            # 組裝 output_dict（沿用模型 forward 的鍵位，方便原 loss 使用）
            output_dict = {
                New_DataName.y_hat: y_best,
                New_DataName.mu: mu,
                New_DataName.logvar: logvar,
                # 以下從 batch 直接帶過去
                New_DataName.Start: data_dict[New_DataName.Start],
                New_DataName.Goal:  data_dict[New_DataName.Goal],
                New_DataName.terrain_cost_map: data_dict[New_DataName.terrain_cost_map],
                New_DataName.split_path: split_gt,
            }
            loss_dict, extra_dict = self.loss_func(output_dict)

        # === DPO 偏好微调（可选，加到 total_loss 上） ===
        if self.use_dpo:
            # 参考策略周期性更新（EMA/拷贝当前权重）
            self._last_ref_update += 1
            if self._last_ref_update >= self.cfg.dpo.ref_update_every:
                self.ref_generator.load_state_dict(self.model.generator.state_dict())
                self.ref_generator.eval()
                self._last_ref_update = 0

            # 准备输入
            observation = self.model.perception(data_dict)
            start = data_dict[New_DataName.Start]
            goal  = data_dict[New_DataName.Goal]
            gt_traj = get_teacher_traj(data_dict)  # T0: 直接用数据集 GT 作为老师

            dpo_out = self.dpo(
                model=self.model.generator,
                ref_model=self.ref_generator,
                input_dict=data_dict,
                observation=observation,
                start=start, goal=goal,
                gt=gt_traj,
                num_candidates=self.cfg.dpo.num_candidates
            )
            loss_dpo = dpo_out["loss_dpo"] * self.cfg.dpo.lambda_dpo
            # total_loss = loss_dict[] + loss_dpo

            loss_dict[LossDictKeys.loss_total] = loss_dict[LossDictKeys.loss_total] + loss_dpo

            
        return output_dict, loss_dict, extra_dict

    def update_logger(self, result_dict: dict):
        for key, value in result_dict.items():
            self.logger.record("train/{}".format(key), value=value)

 
    def run_epoch(self):
        """
        执行一个训练周期。
        
        本函数将完成以下任务：
        - 清零梯度
        - 重置数据加载器（已注释）
        - 记录周期开始时间
        - 遍历数据集进行训练
        - 根据最大迭代次数限制每个周期的迭代数
        - 执行前向传播和后向传播
        - 优化模型参数global_step
        - 更新学习率调度器
        - 保存模型快照
        """

        self._adjust_curriculum()
        self.optimizer.zero_grad()
        ctx = torch.autograd.detect_anomaly if self.cfg.debug_anomaly else nullcontext

        # 每轮初始化“最佳样本”缓存
        self._best_epoch_cache = dict(loss=float("inf"))

        dl_iter = tqdm(
            enumerate(self.data_loader),
            total=len(self.data_loader),
            desc=f"Epoch {self.epoch}"
        )

        for iteration, data_dict in dl_iter:

            if iteration and iteration % self.max_iteration_per_epoch == 0:
                break
            # ---- 設置當前 epoch 的 KL β ----
            if self.kld_warmup_ep > 0:
                t = min(1.0, float(self.epoch) / float(self.kld_warmup_ep))
            else:
                t = 1.0
            cur_beta = self.kld_beta_start + (self.kld_beta_end - self.kld_beta_start) * t
            # 傳給 loss 模組
            if hasattr(self.loss_func, "kld_beta"):
                self.loss_func.kld_beta = float(cur_beta)
            self.iteration += 1
            # output_dict, result_dict = self.step(data_dict)
            output_dict, result_dict, extra_dict = self.step(data_dict)

            loss = result_dict[LossDictKeys.loss]

            # ==== tqdm 实时信息 ====
            postfix = {
                "loss":      f"{result_dict[LossDictKeys.loss_total].item():.3f}",
                # "ep":        f"{result_dict[LossDictKeys.loss_ep].item():.3f}",
                "oob":       f"{result_dict[LossDictKeys.loss_oob].item():.3f}",
                # "unif":  f"{result_dict[LossDictKeys.loss_uniform].item():.3f}",
                # "terr":      f"{result_dict[LossDictKeys.loss_terr].item():.3f}",
                # "sm":        f"{result_dict[LossDictKeys.loss_smooth].item():.3f}",
                "sp":        f"{result_dict[LossDictKeys.loss_split].item():.3f}",
                # "jump":      f"{result_dict[LossDictKeys.loss_jump].item():.3f}",
                "kl":        f"{result_dict.get(LossDictKeys.vae_kld_loss, 0.0):.3f}",
                "βkl":       f"{float(cur_beta):.2f}",
                # "cos":       f"{result_dict[LossDictKeys.loss_cos].item():.3f}",
                "|ŷ|":       f"{output_dict[New_DataName.y_hat].norm(dim=-1).mean().item():.3f}",
                "lr":        f"{self.scheduler.get_last_lr()[0]:.3e}",  # 如果需要科学计数法的小数点后三位
            }
            dl_iter.set_postfix(postfix)

            # ==== NaN 检查 ====
            if torch.isnan(loss):
                tqdm.write("⚠ NaN detected — skip batch")
                continue

            # ==== backward ====
            with ctx():
                loss.backward()

            ### new logger
            global_step = self.epoch * len(self.data_loader) + iteration

            # ① 全部 raw loss
            raw_losses = {k: v.item() for k, v in result_dict.items() if "loss_" in k}

            # ② “raw × 权重”  (跳过 w_map 中没有的键，例如 loss_total)
            contrib = {f"{k}_c": raw_losses[k] * self.loss_func.w_map.get(k, 1.0)
                    for k in raw_losses if k in self.loss_func.w_map}

            # ③ 写 TensorBoard / CSV
            # self.logger.add_scalar("train_raw",     raw_losses, global_step)
            # self.logger.add_scalar("train_contrib", contrib,    global_step)
            # if extra_dict:
            #     self.logger.add_scalar("train_extra", extra_dict, global_step)
            # ③ 写 TensorBoard / CSV（字典必须用 add_scalars）
            self.logger.add_scalars("train_raw",     raw_losses, global_step)
            self.logger.add_scalars("train_contrib", contrib,    global_step)
            if extra_dict:
                # 将 extra_dict 里可能的 tensor/数组转为 float
                extras = {}
                for k, v in extra_dict.items():
                    if torch.is_tensor(v):
                        extras[k] = v.detach().float().mean().item()
                    else:
                        try: extras[k] = float(v)
                        except: continue
                if len(extras) > 0:
                    self.logger.add_scalars("train_extra", extras, global_step)

            # 梯度清理
            for p in self.model.parameters():
                if p.grad is not None:
                    torch.nan_to_num_(p.grad, nan=0.0, posinf=1e4, neginf=-1e4)

            self.optimizer_step(iteration + 1)
            
            # 将结果字典中的数据从GPU转移到CPU，以释放显存空间
            result_dict = release_cuda(result_dict)

            # 再次释放缓存，以节省显存空间
            th.cuda.empty_cache()
        
        # 如果学习率调度器不为空，则更新学习率
        if self.scheduler is not None:
            self.scheduler.step()
        
        # 创建保存模型的目录，如果不存在则创建
        os.makedirs("./models/{}".format(self.name), exist_ok=True)
        
        # 保存模型快照
        self.save_snapshot(f'models/{self.name}/last.pth.tar')
        
    def _grad_hook(self, grad, name):
        """梯度异常检测钩子"""
        if grad is None:
            return
        if torch.isnan(grad).any():
            print(f"[!] NaN gradient detected in {name}")
        if torch.isinf(grad).any():
            print(f"[!] Inf gradient detected in {name}")

    def _write_png(self, terrain_cost_map=None, center=None, targets=None, paths=None, path=None,
                   others=None, file="test_terrain_cost_map.png"):
        dis = 2
        if len(terrain_cost_map.shape) == 2:
            terrain_cost_map_fig = np.repeat(terrain_cost_map[:, :, np.newaxis], 3, axis=2) * 255
        else:
            terrain_cost_map_fig = copy.deepcopy(terrain_cost_map)

            for path in paths:
                if len(path) == 1 or np.any(path[0] == np.inf):
                    continue
                path = np.asarray(path, dtype=int)
                assert path.shape[1] == 2 and len(path.shape) == 2 and path.shape[0] >= 2, "path should be Nx2"
                all_pts = np.concatenate((path + np.array([0, -1], dtype=int), path + np.array([1, 0], dtype=int),
                                          path + np.array([-1, 0], dtype=int), path + np.array([0, 1], dtype=int),
                                          path), axis=0)
                all_pts = np.clip(all_pts, 0, terrain_cost_map_fig.shape[0] - 1)
                terrain_cost_map_fig[all_pts[:, 0], all_pts[:, 1], 0] = 255
                terrain_cost_map_fig[all_pts[:, 0], all_pts[:, 1], 1] = 0
                terrain_cost_map_fig[all_pts[:, 0], all_pts[:, 1], 2] = 255
        cv2.imwrite(file, terrain_cost_map_fig)
        return terrain_cost_map_fig

    def _display_output(self, output_dict, data_dict, iteration, terrain_cost_map_resolution=0.1, terrain_cost_map_threshold=300,
                        root_path="training"):
        if not os.path.exists(root_path):
            os.makedirs(root_path)
        y_hat = output_dict[New_DataName.y_hat]  # BxNx2
        for i in range(len(y_hat)):
            file_name = str(int(data_dict[New_DataName.terrain_cost_map][i].detach().cpu().numpy()[0])) + ".png"
            terrain_cost_map = cv2.imread(os.path.join(self.data_loader.dataset.root, self.data_loader.dataset.figures, file_name))
            terrain_cost_map = terrain_cost_map[int(terrain_cost_map.shape[0] / 2 - terrain_cost_map_threshold):
                                  int(terrain_cost_map.shape[0] / 2 + terrain_cost_map_threshold),
                        int(terrain_cost_map.shape[1] / 2 - terrain_cost_map_threshold):
                        int(terrain_cost_map.shape[1] / 2 + terrain_cost_map_threshold)]
            center = np.array([terrain_cost_map.shape[0] / 2.0, terrain_cost_map.shape[1] / 2.0], dtype=int)

            scan_pixels = np.clip(np.floor(data_dict[New_DataName.scan][i].detach().cpu().numpy()[:, :2] /
                                           terrain_cost_map_threshold).astype(int) + center, 0, terrain_cost_map.shape[0] - 1)
            if len(y_hat.shape) == 3:
                points = torch.cumsum(y_hat[i], dim=0).detach().cpu().numpy()
                pixels = np.clip(np.floor(points / terrain_cost_map_resolution).astype(int) + center, 0,
                                 terrain_cost_map.shape[0] - 1)
                self._write_png(terrain_cost_map=terrain_cost_map, center=center, path=pixels, others=scan_pixels,
                                file=os.path.join(root_path, "evaluation_{}_{}_{}.png".format(self.epoch, iteration, i)))
            else:
                points = torch.cumsum(y_hat[i], dim=1).detach().cpu().numpy()
                pixels = np.clip(np.floor(points / terrain_cost_map_resolution).astype(int) + center, 0,
                                 terrain_cost_map.shape[0] - 1)
                self._write_png(terrain_cost_map=terrain_cost_map, center=center, paths=pixels, others=scan_pixels,
                                file=os.path.join(root_path, "evaluation_{}_{}_{}.png".format(self.epoch, iteration, i)))

    def inference_epoch(self):
        """
        执行一个验证轮次的推理。
        该方法用于在当前epoch结束后，对验证数据集进行推理，记录推理结果，并最终恢复训练模式。
        """
        # 设置模型为评估模式，以便在推理过程中禁用dropout等训练时的特性
        self.set_eval_mode()
        
        # 遍历验证数据加载器提供的每个数据批次
        for iteration, data_dict in enumerate(self.val_loader):

            # 执行推理步骤，获取输出和结果字典
            output_dict, result_dict = self.step(data_dict)
            # 每20个批次可视化一次输出，以便检查推理结果
            if iteration % 20 == 0:
                self._display_output(output_dict=output_dict, data_dict=data_dict, iteration=iteration,
                                    root_path="training/"+self.name)
            # 确保所有CUDA操作完成
            th.cuda.synchronize()

            # 释放CUDA张量，避免内存泄漏
            result_dict = release_cuda(result_dict)

            # 清空CUDA缓存，释放不必要的内存
            th.cuda.empty_cache()

        self.set_train_mode()

    def run(self):
        """
        启动训练过程。
        
        本函数首先检查是否存在快照，如果存在，则加载快照。
        然后设置训练模式，并开始训练循环，直到达到最大训练周期数。
        在每个训练周期之后，如果设置了评估间隔，则进行一次推理周期。
        """
        # 检查是否存在快照，如果存在则加载
        if self.snapshot:
            self.load_snapshot(self.snapshot)

        # 设置训练模式
        self.set_train_mode()

        # print all loss weights setting
        for key, value in self.loss_func.w_map.items():
            print(f"loss setting {key}: {value}")
        
        # 训练循环，直到达到最大训练周期数
        while self.epoch < self.max_epoch:
            self.epoch += 1
            self.run_epoch()
            best_pack = self.validate_epoch(self.epoch)                # 立即跑验证
            if self.epoch % self.debug_every_epochs == 0 and self._best_epoch_cache:
                # 1) fmap（通道拼贴 + 均值）
                self.logger.plot_fmap(self._best_epoch_cache["fmap"][0], step=self.epoch, tag_prefix="debug/fmap")
                # 2) 候选路径热图（背景用 fmap 均值）
                bg = self._best_epoch_cache["fmap"][0].mean(0)
                # self.logger.plot_paths_heatmap(self._best_epoch_cache["cands"],
                #                                self._best_epoch_cache["pred"],
                #                                self._best_epoch_cache["gt"],
                #                                self._best_epoch_cache["H"], self._best_epoch_cache["W"],
                #                                step=self.epoch, tag_prefix="debug/paths", bg=bg)
                self.logger.plot_paths_heatmap(self._best_epoch_cache["cands"],
                               self._best_epoch_cache["pred"],
                               self._best_epoch_cache["gt"],
                               self._best_epoch_cache["H"], self._best_epoch_cache["W"],
                               step=self.epoch, tag_prefix="debug/paths", bg=bg)
                # # 直方
                # # coord = ((self._best_epoch_cache["cands"] - self._best_epoch_cache["gt"])**2).mean(dim=(2,3))
                # C   = self._best_epoch_cache["cands"]         # [K,N,2] 或 [K,B,N,2]
                # GT  = self._best_epoch_cache["gt"]            # [N,2]
                # # 统一到 [K,?,N,2] 的最后两维做均值
                # if C.dim() == 3:   # [K,N,2]
                #     coord = ((C - GT.unsqueeze(0))**2).mean(dim=(1,2))  # → [K]
                # else:               # [K,B,N,2]（更通用的兜底）
                #     # 你这里存的是单样本 b0，仍可做 (2,3) 平均，再在 B 维上取均值
                #     coord = ((C - GT.unsqueeze(0).unsqueeze(1))**2).mean(dim=(2,3)).mean(dim=1)  # → [K]
                # # self.logger.add_hist("score/coord", coord, self.epoch)
                # self.logger.add_hist("score/coord", coord, self.epoch)
                # # 3) parp/perp 曲线 + 直方图
                # if self._best_epoch_cache["parp"] is not None and self._best_epoch_cache["perp"] is not None:
                #     self.logger.plot_parp_perp(self._best_epoch_cache["parp"],
                #                                self._best_epoch_cache["perp"],
                #                                step=self.epoch, tag_prefix="debug/step")
                # # 4) 候选打分直方图
                # if self._best_epoch_cache["score_coord"] is not None:
                #     self.logger.add_hist("score/coord", self._best_epoch_cache["score_coord"], self.epoch)
                # if self._best_epoch_cache["score_vec"] is not None:
                #     self.logger.add_hist("score/vec",   self._best_epoch_cache["score_vec"],   self.epoch)
                # if self._best_epoch_cache["score_curv"] is not None:
                #     self.logger.add_hist("score/curv",  self._best_epoch_cache["score_curv"],  self.epoch)
                # === paths heatmap stays unchanged above ===

                C  = self._best_epoch_cache["cands"]   # [K,N,2] or [K,B,N,2]
                GT = self._best_epoch_cache["gt"]      # [N,2]

                # coord (pointwise MSE)
                if C.dim() == 3:  # [K,N,2]
                    score_coord = ((C - GT.unsqueeze(0))**2).mean(dim=(1,2))           # [K]
                else:             # [K,B,N,2]
                    score_coord = ((C - GT.unsqueeze(0).unsqueeze(1))**2).mean(dim=(2,3)).mean(dim=1)

                # vec (first-diff MSE)
                if C.dim() == 3:
                    Vp = C[:, 1:, :] - C[:, :-1, :]
                    Vg = GT[1:, :] - GT[:-1, :]
                    score_vec = ((Vp - Vg.unsqueeze(0))**2).mean(dim=(1,2))
                else:
                    Vp = C[:, :, 1:, :] - C[:, :, :-1, :]
                    Vg = (GT[1:, :] - GT[:-1, :]).unsqueeze(0).unsqueeze(1)
                    score_vec = ((Vp - Vg)**2).mean(dim=(2,3)).mean(dim=1)

                # curv (second-diff MSE)
                if C.dim() == 3:
                    Cp = C[:, 2:, :] - 2*C[:, 1:-1, :] + C[:, :-2, :]
                    Cg = GT[2:, :] - 2*GT[1:-1, :] + GT[:-2, :]
                    score_curv = ((Cp - Cg.unsqueeze(0))**2).mean(dim=(1,2))
                else:
                    Cp = C[:, :, 2:, :] - 2*C[:, :, 1:-1, :] + C[:, :, :-2, :]
                    Cg = (GT[2:, :] - 2*GT[1:-1, :] + GT[:-2, :]).unsqueeze(0).unsqueeze(1)
                    score_curv = ((Cp - Cg)**2).mean(dim=(2,3)).mean(dim=1)

                # write histograms
                self.logger.add_hist("score/coord", score_coord, self.epoch)
                self.logger.add_hist("score/vec",   score_vec,   self.epoch)
                self.logger.add_hist("score/curv",  score_curv,  self.epoch)
                parp = self._best_epoch_cache.get("parp", None)
                perp = self._best_epoch_cache.get("perp", None)
                if (parp is not None) and (perp is not None):
                    self.logger.plot_parp_perp(parp, perp, step=self.epoch, tag_prefix="debug/step")
                # 5) KL / z 统计
                kl = (-0.5 * (1 + 2*torch.log(self._best_epoch_cache["std"]) - self._best_epoch_cache["mu"]**2 - (self._best_epoch_cache["std"]**2))).sum(dim=1).mean()
                self.logger.add_scalars("latent", {"KL_mean": float(kl.detach().cpu().item())}, self.epoch)
                if self._best_epoch_cache.get("loss") is not None:
                    self.logger.add_scalars("train", {"best_loss": self._best_epoch_cache["loss"]}, self.epoch)
            
            # ---- Early‑Stopping ----
            if self.epochs_no_improve >= self.patience:
                print(f"Early Stopping: no sp improvement for {self.patience} epochs")
                break

if __name__ == "__main__":
    trainer = Trainer(cfgs=cfg)
    trainer.run_epoch()
