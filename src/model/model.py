from torch import nn
import torch

from src.model.perception import Perception
# from src.backbones.vae import CVAE
from src.backbones.cvae_core import CVAE
from src.configs import New_DataName, ModelType

class SwinPathBiCVAE(nn.Module):
    """
    此类用于处理多路径生成任务，包含不同的模型类型和前向传播方法。
    
    参数:
    - cfgs: 包含模型配置的字典。
    """
    
    def __init__(self, cfgs):
        super().__init__()
        self.cfg = cfgs

        # ① 正确实例化感知 & 生成器
        # self.perception = Perception(self.cfg.perception)
        # 允许 cfg 没有 perception 字段时使用默认参数
        self.perception = Perception(getattr(self.cfg, "perception", None))
        self.model_type = self.cfg.dlow.model_type
        if self.model_type == ModelType.cvae:
            # 采用新的层级 CVAE（可对齐跨注意力解码）
            # self.generator = CVAE(cfgs.model.cvae_core)
            # 采用 CVAE（注意 cfgs 已经是 cfgs.model，不能再写 cfgs.model.xxx）
            self.generator = CVAE(cfgs.cvae_core)

        else:
            raise ValueError("unsupported model type")

        # ② 是否冻结感知
        if self.cfg.perception.fix_perception:
            self.set_perception_fixed()

    def set_perception_fixed(self):
        """
        固定感知模块的参数，使其在训练中不被更新。
        """
        for param in self.perception.parameters():
            param.requires_grad = False


    def forward(self, input_dict):
        # print(f"Input keys: {input_dict.keys()}")
        last_poses = input_dict.get(New_DataName.last_poses, None)
        if last_poses is not None and not isinstance(last_poses, torch.Tensor):
            last_poses = torch.tensor(last_poses)
        output = {
            New_DataName.split_path: input_dict.get(New_DataName.split_path, None),
            New_DataName.path:       input_dict.get(New_DataName.path, None),
            New_DataName.rgb_map:     input_dict.get(New_DataName.rgb_map, None),
            New_DataName.last_poses: last_poses,
            New_DataName.Start:      input_dict.get(New_DataName.Start, None),
            New_DataName.Goal:       input_dict.get(New_DataName.Goal, None),
            New_DataName.terrain_cost_map:  input_dict.get(New_DataName.terrain_cost_map, None),
            }
        if output[New_DataName.last_poses] is None and output[New_DataName.path] is not None:
            # 如果 last_poses 未设置，则尝试从路径数据中获取最后一个点
            output[New_DataName.last_poses] = output[New_DataName.path][:, -1, :] if len(output[New_DataName.path].shape) > 2 else output[New_DataName.path][-1]


        # 确保输入图像的维度是 (batch_size, channels, height, width)
        # image = input_dict[New_DataName.rgb_map]

        # if image.dim() == 3:
        #     image = image.unsqueeze(0)  # 添加批量维度

        # observation = self.perception(input_dict) # {"fmap","gvec"} 与旧接口一致
        # if self.model_type == ModelType.cvae:
        #     # waypoints, mu, logvar = self.generator(observation)
        #     waypoints, mu, logvar = self.generator(
        #             observation,
        #             input_dict[New_DataName.Start],
        #             input_dict[New_DataName.Goal]
        #     )

        #     output.update({New_DataName.mu: mu, New_DataName.logvar: logvar})
        # else:
        #     raise Exception("model type is not defined")
        # output.update({New_DataName.y_hat: waypoints})
        # return output
        observation = self.perception(input_dict)  # {"fmap": ...}
        fmap  = observation["fmap"]
        start = input_dict[New_DataName.Start]
        goal  = input_dict[New_DataName.Goal]
        if self.model_type != ModelType.cvae:
            raise Exception("model type is not defined")
        # 编码→重参数化→解码（一次性 38 点）
        mu, logvar = self.generator.encode_from_fmap(fmap, start, goal)
        std = (0.5 * logvar).exp()
        eps = torch.randn_like(std)
        z   = mu + eps * std
        mid = self.generator.decode(z, fmap, start, goal)                 # (B,38,2)
        y_hat = torch.cat([start.unsqueeze(1), mid, goal.unsqueeze(1)], dim=1)  # (B,40,2)
        output.update({
            New_DataName.mu: mu, New_DataName.logvar: logvar,
            New_DataName.y_hat: y_hat
        })
        return output