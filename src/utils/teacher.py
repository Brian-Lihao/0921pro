# src/utils/teacher.py
import torch
from typing import Dict, Optional
from src.configs import New_DataName

LIKELY_GT_KEYS = [
    "Split", "SplitPath", "split", "split_path",
    "Waypoints", "waypoints", "GT", "gt"
]

def get_teacher_traj(input_dict: Dict) -> Optional[torch.Tensor]:
    """
    T0 教师接口：直接返回数据集自带的 A* 轨迹（N×2）。
    优先使用标准键 New_DataName.Split；若无，则从常见别名中寻找。
    返回: [B,N,2]（float, [0,1] 范围）或 None
    """
    # 标准键
    if New_DataName.Split in input_dict:
        return input_dict[New_DataName.Split]
    # 常见别名
    for k in LIKELY_GT_KEYS:
        if k in input_dict:
            val = input_dict[k]
            if isinstance(val, torch.Tensor) and val.dim() == 3 and val.size(-1) == 2:
                return val
    return None
