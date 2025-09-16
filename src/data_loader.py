import os
import pickle
import random
from functools import partial
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

import cv2
import glob

from src.configs import New_DataName ,get_configs

def reset_seed_worker_init_fn(worker_id):
    """
    为数据加载器的每个工作线程（worker）重置随机种子。

    参数:
        worker_id (int): 当前线程的工作线程ID。

    功能:
        1. 使用 `torch.initial_seed()` 获取当前线程的初始种子值，并通过取模运算 `(2 ** 32)` 确保种子值在有效范围内。
        2. 将计算得到的种子值分别设置为 NumPy 和 Python 标准库 `random` 的随机种子。
        3. 这一操作确保了多线程环境下，每个 worker 的随机性是独立且可复现的。
    """
    # 计算当前线程的随机种子值
    seed = torch.initial_seed() % (2 ** 32)
    
    # 设置 NumPy 和 Python random 的随机种子
    np.random.seed(seed)
    random.seed(seed)


def registration_collate_fn_stack_mode(batch):
    collated = {}
    for item in batch:
        for k, v in item.items():
            if v is None:
                continue
            if k not in collated:
                collated[k] = []
            collated[k].append(v)

    # 把需要堆叠的键统一 stack
    stack_keys = {New_DataName.split_path,
                  New_DataName.rgb_map,
                  New_DataName.terrain_cost_map,
                  New_DataName.Start,
                  New_DataName.Goal,
                  New_DataName.last_poses}

    for k in stack_keys:
        if k in collated:
            collated[k] = torch.stack(collated[k], dim=0)

    return collated

class PathPlanningDataset(Dataset):
    def __init__(self, data_file: str, 
                 train: bool, 
                 data_percentage=0.8, 
                 w_eval=False,
                 terrain_cost_map_threshold=200
                 ):
        self.root = data_file
        self.train = train
        self.w_eval = w_eval
        self.terrain_cost_map_threshold = terrain_cost_map_threshold

        # 加载 images_without_path1 文件夹中的所有图片
        self.image_files = sorted(glob.glob(os.path.join(self.root, "labelled_imgs", "*.png")))
        # 加载 points_pkls 文件夹中的所有 pkl 文件
        self.pkl_files = sorted(glob.glob(os.path.join(self.root, "points_pkls", "*.pkl")))

        # 确保图片和 pkl 文件数量一致
        assert len(self.image_files) == len(self.pkl_files), "Number of images and pkl files must match"

        # 根据数据比例选择训练或测试样本
        select_split = int(len(self.image_files) * data_percentage)
        if self.train:
            self.data_indices = list(range(select_split))
        else:
            self.data_indices = list(range(select_split, len(self.image_files)))

    def __len__(self):
        return len(self.data_indices)

    def _process_terrain_cost_map(self, terrain_cost_map):
        # 裁剪局部地图
        cropped_terrain_cost_map = terrain_cost_map[int(terrain_cost_map.shape[0] / 2 - self.terrain_cost_map_threshold):
                                      int(terrain_cost_map.shape[0] / 2 + self.terrain_cost_map_threshold),
                            int(terrain_cost_map.shape[1] / 2 - self.terrain_cost_map_threshold):
                            int(terrain_cost_map.shape[1] / 2 + self.terrain_cost_map_threshold)]
        return cropped_terrain_cost_map


    def __getitem__(self, idx):
        # ---------- 路径 ----------
        img_path = self.image_files[self.data_indices[idx]]
        pkl_path = self.pkl_files [self.data_indices[idx]]

        # ---------- 1) 读图 (RGB) ----------
        img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)   # 512×512×3

        # ---------- 2) 中心裁 400×400 ----------
        H0, W0, _  = img.shape
        th         = self.terrain_cost_map_threshold         # 200 ⇒ 400×400
        row0, col0 = H0//2 - th, W0//2 - th

        # ---------- 3) resize → 128×128 (保持起/终点像素) ----------
        # H_rs = W_rs = 128
        H_rs = W_rs = 256
        img_rs = cv2.resize(img, (W_rs, H_rs), interpolation=cv2.INTER_NEAREST)

        # ---------- 4) Tensor & 归一化 ----------
        terrain_cost_map = torch.from_numpy(img_rs.transpose(2,0,1)).float() / 255.0  # (3,128,128)

        # ---------- 5) 读 pkl ----------
        with open(pkl_path, "rb") as f:
            meta = pickle.load(f)

        # 像素 → 0‑1 归一化   (沿用旧代码的三步映射)
        def _map(arr):
            arr = np.asarray(arr, np.float32)[..., [1, 0]]     # (row,col)→(x,y)
            arr /= 511.0                                       # 0‑1 归一
            return torch.from_numpy(arr)

        start       = _map(meta[New_DataName.Start])
        goal        = _map(meta[New_DataName.Goal])
        path        = _map(meta[New_DataName.path])        if meta.get(New_DataName.path) is not None else None
        split_path  = _map(meta[New_DataName.split_path])  if meta.get(New_DataName.split_path) is not None else None
        last_pose   = path[-1].clone() if path is not None else None

        return {
            New_DataName.rgb_map:     terrain_cost_map,       # (3,128,128)
            New_DataName.terrain_cost_map:  terrain_cost_map[0],    # B 通道
            New_DataName.path:       path,
            New_DataName.split_path: split_path,
            New_DataName.Start:      start,
            New_DataName.Goal:       goal,
            New_DataName.last_poses: last_pose,
        }

def train_eval_data_loader(cfg):
    # 创建训练数据集
    train_dataset = PathPlanningDataset(
        data_file=cfg.file,
        # cfg.name,
        train=True,
        data_percentage=cfg.training_data_percentage,
        # lidar_threshold=cfg.lidar_threshold,
        w_eval=cfg.w_eval,
        # vel_num=cfg.vel_num,
        terrain_cost_map_threshold=cfg.terrain_cost_map_threshold,
        # lidar_max_num=cfg.lidar_max_points,
        # lidar_vx_size=cfg.lidar_downsample_vx_size
    )
    # 创建训练数据集的加载器
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        shuffle=cfg.shuffle,
        sampler=None,
        collate_fn=partial(registration_collate_fn_stack_mode),
        worker_init_fn=reset_seed_worker_init_fn,
        pin_memory=False,
        drop_last=False,
    )

    # 创建评估数据集
    eval_dataset = PathPlanningDataset(
        data_file=cfg.file,
        # cfg.name,
        train=False,
        data_percentage=cfg.training_data_percentage,
        w_eval=cfg.w_eval,
        terrain_cost_map_threshold=cfg.terrain_cost_map_threshold,
    )
    # 创建评估数据集的加载器
    eval_loader = DataLoader(
        dataset=eval_dataset,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        shuffle=cfg.shuffle,
        sampler=None,
        collate_fn=partial(registration_collate_fn_stack_mode),
        worker_init_fn=reset_seed_worker_init_fn,
        pin_memory=False,
        drop_last=False,
    )
    print(f"✅ 训练数据集样本数：{len(train_dataset)}")
    print(f"✅ 评估数据集样本数：{len(eval_dataset)}")
    print(f"✅ 训练批次数 len(train_loader) : {len(train_loader)}")
    print(f"✅ 评估批次数 len(eval_loader) : {len(eval_loader)}")
    return train_loader, eval_loader

if __name__ == "__main__":

    cfgs = get_configs()
    cfg = cfgs.data
    # 直接创建数据加载器
    train_loader, eval_loader = train_eval_data_loader(cfg)

    # 测试训练集中的数据读取
    for i, batch in enumerate(train_loader):
        # 打印一些关键信息进行检查
        print(f"Batch {i+1}:")
        print(f"Image shape: {batch[New_DataName.rgb_map].shape}")
        print(f"Start point: {batch[New_DataName.Start]}")
        print(f"Goal point: {batch[New_DataName.Goal]}")
        print(f"Split Path: {batch[New_DataName.split_path]}")
        
        # 检查是否存在异常值
        if torch.isnan(batch[New_DataName.rgb_map]).any():
            print("Warning: NaN detected in the image.")
        if torch.isnan(batch[New_DataName.Start]).any() or torch.isnan(batch[New_DataName.Goal]).any():
            print("Warning: NaN detected in the coordinates.")

        # 在训练时，检查一个batch之后退出
        if i == 1:  # 可以修改数字，查看更多的batch
            break
    
    # 测试验证集中的数据读取
    for i, batch in enumerate(eval_loader):
        # 打印一些关键信息进行检查
        print(f"Eval Batch {i+1}:")
        print(f"Image shape: {batch[New_DataName.rgb_map].shape}")
        print(f"Start point: {batch[New_DataName.Start]}")
        print(f"Goal point: {batch[New_DataName.Goal]}")
        # 检查是否存在异常值
        if torch.isnan(batch[New_DataName.rgb_map]).any():
            print("Warning: NaN detected in the image.")
        if torch.isnan(batch[New_DataName.Start]).any() or torch.isnan(batch[New_DataName.Goal]).any():
            print("Warning: NaN detected in the coordinates.")

        # 在验证时，检查一个batch之后退出
        if i == 1:  # 可以修改数字，查看更多的batch
            break