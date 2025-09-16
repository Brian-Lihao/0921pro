import os, multiprocessing as mp
os.environ["MPLBACKEND"] = "Agg"      # B: 非交互后端
mp.set_start_method("spawn", force=True)  # A: 子进程重新启动


import torch
from src.trainer import Trainer
from src.configs import get_configs
     

if __name__ == "__main__":
    cfgs = get_configs()
    trainer = Trainer(cfgs=cfgs)
    torch.autograd.set_detect_anomaly(True)
    trainer.run()
    torch.autograd.set_detect_anomaly(False)
