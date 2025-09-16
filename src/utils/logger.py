# logger.py  ──────────────────────────────────────────────────────────
from pathlib import Path
import csv, time, math, os
from typing import Dict, Any

import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import matplotlib.pyplot as plt


class TBLogger:
    """
    日志统一入口：
      • TensorBoard: add_scalar / add_scalars
      • CSV: 按行追加 (step, key, value)
    """
    def __init__(self, log_dir: str):
        # 统一根目录：事件与PNG都放在同一个 run 目录
        self.root = Path(log_dir)
        self.root.mkdir(parents=True, exist_ok=True)
        self.tb = SummaryWriter(str(self.root))
        # CSV 也放到同一个 run 目录
        self.csv_path = self.root / "metrics.csv"
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)

        
        self.csv_fh = self.csv_path.open("a", newline="")
        self.csv_writer = csv.writer(self.csv_fh)
        if self.csv_fh.tell() == 0:          # header
            self.csv_writer.writerow(
                ["wall_time", "step", "phase", "metric", "value"]
            )

    # ---------------- scalar ---------------- #
    def add_scalars(self, phase: str, scalars: Dict[str, float], step: int):
        """phase = 'train' / 'val'"""
        for k, v in scalars.items():
            tag = f"{phase}/{k}"
            self.tb.add_scalar(tag, v, step)
            self.csv_writer.writerow([time.time(), step, phase, k, v])

    # ---------------- 专用快捷 ---------------- #
    def add_lr(self, lr: float, step: int):
        self.tb.add_scalar("lr", lr, step)
        self.csv_writer.writerow([time.time(), step, "meta", "lr", lr])

    def close(self):
        self.tb.flush(); self.tb.close()
        self.csv_fh.flush(); self.csv_fh.close()

    def info(self, msg: str):
        print(f"[INFO] {msg}")
        # 也写入 CSV，方便留档（可选）
        self.csv_writer.writerow([time.time(), -1, "meta", "info", msg])

    # ---------- 新增：通用图像/直方图/图像落盘 ---------- #
    def add_hist(self, tag: str, tensor: torch.Tensor, step: int):
        arr = tensor.detach().cpu().numpy()
        self.tb.add_histogram(tag, arr, step)
        self.csv_writer.writerow([time.time(), step, "hist", tag, float(getattr(tensor, "mean", lambda:0)())])

    def add_figure(self, tag: str, fig, step: int, save_name: str = None):
        self.tb.add_figure(tag, fig, step)
        if save_name:
            out = (self.root / "debug"); out.mkdir(parents=True, exist_ok=True)
            fig.savefig(out / save_name, dpi=160)

    # ---------- 新增：便捷可视化（matplotlib） ---------- #
    def plot_fmap(self, fmap: torch.Tensor, step: int, tag_prefix: str = "debug/fmap", max_ch: int = 16):
        """
        fmap: [C,H,W] 或 [1,C,H,W] / [B,C,H,W]
        输出：通道拼贴 + 通道均值图
        """
        import math, matplotlib.pyplot as plt
        from torchvision.utils import make_grid
        if fmap.dim()==4: fmap = fmap[0]
        C,H,W = fmap.shape
        x = fmap[:min(max_ch,C)]
        # 逐通道 0-1 归一
        x = (x - x.amin(dim=(1,2), keepdim=True)) / (x.amax(dim=(1,2), keepdim=True) - x.amin(dim=(1,2), keepdim=True) + 1e-6)

        grid = make_grid(
            x.unsqueeze(1),  # [C,1,H,W] 视作 C 张灰度图
            nrow=int(math.sqrt(x.size(0))),
            pad_value=1.0
        ).cpu()  # [C_out,Hg,Wg]，C_out 可能为 1 或 3（取决于 torchvision 行为）

        # 统一转为 matplotlib 可接受形状：HxW 或 HxWx3
        if grid.dim() == 3:
            # [C,H,W] -> [H,W,C]
            grid_np = grid.permute(1, 2, 0).numpy()
            if grid_np.shape[-1] == 1:
                grid_np = grid_np[..., 0]  # 变成 [H,W] 灰度
        else:
            # 保险分支：异常情况直接 squeeze 成 2D
            grid_np = grid.squeeze().numpy()

        mean_map = fmap.mean(0).cpu().numpy()
        fig,axs = plt.subplots(1,2,figsize=(10,4))
        axs[0].imshow(grid_np, cmap="gray"); axs[0].set_title("fmap channels"); axs[0].axis("off")
        axs[1].imshow(mean_map, cmap="gray"); axs[1].set_title("fmap mean"); axs[1].axis("off")
        fig.tight_layout()
        self.add_figure(f"{tag_prefix}", fig, step, save_name=f"epoch{step:03d}_fmap.png")
        plt.close(fig)

    def plot_paths_heatmap(self, cands: torch.Tensor, pred: torch.Tensor, gt: torch.Tensor,
                           H: int, W: int, step: int, tag_prefix: str = "debug/paths", bg: torch.Tensor = None):
        """
        cands: [K,N,2] 单位坐标；pred/gt: [N,2]；bg: [H,W]（可传 fmap mean）
        """
        
        def to_pix(x):
            xs = (x[:,0]*(W-1)).clamp(0, W-1); ys = (x[:,1]*(H-1)).clamp(0, H-1)
            return xs.cpu().numpy(), ys.cpu().numpy()
        K,N,_ = cands.shape
        heat = torch.zeros(H, W, dtype=torch.float32)
        for k in range(K):
            xs,ys = to_pix(cands[k])
            for i in range(N):
                heat[int(round(ys[i])), int(round(xs[i]))] += 1.0
        heat = (heat / (heat.max()+1e-6)).cpu().numpy()
        fig,ax = plt.subplots(1,1,figsize=(6,6))
        if bg is not None:
            ax.imshow(bg.cpu().numpy(), cmap="gray", origin="upper")
            ax.imshow(heat, cmap="inferno", alpha=0.6, origin="upper")
        else:
            ax.imshow(heat, cmap="inferno", origin="upper")
        px,py = to_pix(pred); gx,gy = to_pix(gt)
        ax.plot(px,py,"r.-",lw=2,ms=3,label="pred")
        ax.plot(gx,gy,"b.-",lw=2,ms=3,label="gt")
        ax.set_xlim([0,W-1]); ax.set_ylim([H-1,0]); ax.legend(); fig.tight_layout()
        self.add_figure(f"{tag_prefix}", fig, step, save_name=f"epoch{step:03d}_paths.png")
        plt.close(fig)

    def plot_parp_perp(self, parp: torch.Tensor, perp: torch.Tensor, step: int, tag_prefix: str = "debug/step"):
        """
        parp: [N,1]；perp: [N,2] 或 [N,1]
        """
        
        p = parp.squeeze(-1).detach().cpu()
        q = (perp.norm(dim=-1) if perp.dim()==2 else perp.squeeze(-1)).detach().cpu()
        fig,ax = plt.subplots(1,1,figsize=(7,3))
        ax.plot(p.numpy(), label="parp(//)"); ax.plot(q.numpy(), label="|perp|")
        ax.set_title("progress vs lateral"); ax.set_xlabel("step"); ax.grid(True); ax.legend(); fig.tight_layout()
        self.add_figure(f"{tag_prefix}", fig, step, save_name=f"epoch{step:03d}_parp_perp.png")
        plt.close(fig)
        self.add_hist("debug/parp_hist", p, step)
        self.add_hist("debug/perp_hist", q, step)