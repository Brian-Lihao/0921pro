from easydict import EasyDict as edict
import argparse
import torch.cuda
import os
import numpy as np
import yaml
import time
from torch import nn

with open("./data_cfg.yaml", 'r') as file:
    configs = yaml.safe_load(file)

POINT_SPLIT_NUMBER = 40

LidarMode = edict()
LidarMode.image = 0
LidarMode.ptcnn = 1
LidarMode.kpconv = 2

RNNType = edict()
RNNType.gru = 0
RNNType.lstm = 1

ModelType = edict()
ModelType.cvae = 0
# ModelType.dlow = 1
# ModelType.dlowae = 2
ModelType.terrapn = 3

DistanceFunction = edict()
DistanceFunction.euclidean = 0
DistanceFunction.point_wise = 1

LossDisType = edict()
LossDisType.dtw = 0
LossDisType.hausdorff = 1

CollisionLossType = edict()
CollisionLossType.global_dis = 0
CollisionLossType.local_dis = 1

DiversityType = edict()
DiversityType.target_diversity = 0
DiversityType.self_diversity = 1

GTType = edict()
GTType.generated = 0
GTType.demonstration = 1

ActivateFunction = edict()
ActivateFunction.soft = 0
ActivateFunction.tanh = 1

DataName = edict()
DataName.camera = "camera"
DataName.lidar = "lidar"
DataName.lidar2d = "lidar_array"
DataName.vel = "vel"
DataName.imu = "imu"
DataName.path = "path"
DataName.last_poses = "last_poses"
DataName.mu = "mu"
DataName.logvar = "logvar"
DataName.A = "A"
DataName.b = "b"
DataName.y_hat = "y_hat"
DataName.scores = "scores"
DataName.png = "png"
DataName.scan = "scan"
DataName.terrain_cost_map = "terrain_cost_map"
DataName.all_paths = "all_paths"
DataName.pose = "pose"

New_DataName = edict()
New_DataName.Start = "start"
New_DataName.Goal = "goal"
New_DataName.terrain_cost_map = "terrain_cost_map"
New_DataName.rgb_map = "rgb_map"
New_DataName.path = "path"
New_DataName.split_path = "split_path"
New_DataName.mu = "mu"
New_DataName.logvar = "logvar"
New_DataName.A = "A"
New_DataName.b = "b"
New_DataName.y_hat = "y_hat"
New_DataName.scores = "scores"
New_DataName.last_poses = "last_poses"
New_DataName.passable_mask = "passable_mask"
New_DataName.mu_prior = "mu_prior"
New_DataName.logvar_prior = "logvar_prior"
New_DataName.mu_post = "mu_post"
New_DataName.logvar_post = "logvar_post"
New_DataName.Split = "split"

LossDictKeys = edict()
LossDictKeys.loss = "loss"
# LossDictKeys.dlow_kld_loss = "dlow_kld_loss"
LossDictKeys.vae_kld_loss = "vae_kld_loss"
LossDictKeys.last_point_loss = "last_point_loss"
LossDictKeys.distance_loss = "distance_loss"
LossDictKeys.diversity_loss = "diversity_loss"
LossDictKeys.collision_loss_max = "collision_loss_max"
LossDictKeys.collision_loss_mean = "collision_loss_mean"
LossDictKeys.coverage_distance = "coverage_distance"
LossDictKeys.coverage_last = "coverage_last"
LossDictKeys.coverage_diverse = "coverage_diverse"
LossDictKeys.asymmetric_loss = "asymmetric_loss"

LossDictKeys.loss_total = "loss_total"
LossDictKeys.loss_ep = "loss_ep"
LossDictKeys.loss_terr = "loss_terr"
# LossDictKeys.loss_smooth = "loss_smooth"
LossDictKeys.loss_split = "loss_split"
# LossDictKeys.loss_cham = "loss_cham"
LossDictKeys.loss_cos = "loss_cos"
LossDictKeys.loss_uniform = "loss_uniform"
LossDictKeys.loss_oob = "loss_oob"
LossDictKeys.loss_dis = "loss_dis"
LossDictKeys.loss_jump = "loss_jump"
LossDictKeys.loss_map = "loss_map"
# LossDictKeys.loss_combine = "loss_combine"
LossDictKeys.loss_vae_kld = "loss_vae_kld"

cfg = edict()
cfg.name = ""
cfg.device = "cuda:0"
cfg.eval = False
cfg.load_snapshot = "/home/rob/code/SwinPath_BiCVAE_copy_40_try_tf/checkpoints/best_ep134_sp0.00380717-last-1.pth"
cfg.csv_output_dir = "./csv_output_dir"

cfg.data = edict()
cfg.data.file = ""
cfg.data.name = ""
cfg.data.batch_size = 16
cfg.data.num_workers = 8
cfg.data.shuffle = True
cfg.data.training_data_percentage = 0.70
cfg.data.lidar_max_points = 5120
cfg.data.lidar_downsample_vx_size = 0.08
cfg.data.lidar_mode = LidarMode.image
cfg.data.lidar_threshold = configs["rosbag"]["lidar"]["threshold"]
cfg.data.vel_num = 10
cfg.data.use_terrain_cost_map = False
cfg.data.terrain_cost_map_threshold = int(configs["terrain_cost_map"]["target_distance"] * 2 / configs["terrain_cost_map"]["resolution"])
cfg.data.w_eval = False

cfg.training = edict()
cfg.training.no_eval = False
cfg.training.max_epoch = 500
cfg.training.max_iteration_per_epoch = 5000
cfg.training.lr = 1e-5
cfg.training.lr_decay = 0.5
cfg.training.lr_decay_steps = 6
cfg.training.weight_decay = 1e-6
cfg.training.grad_acc_steps = 5
cfg.training.debug_anomaly = True

### ONLY DEBUG
# cfg.training 部分
cfg.training.debug_n_samples = 2        # 每 batch 采 2 条；设 0 关闭
cfg.training.debug_max_total = 50       # 整个训练最多写 50 条

# === 新增 DPO 子配置（默认关闭）===
cfg.training.dpo = edict()
cfg.training.dpo.use_alignment = False
cfg.training.dpo.use = False
cfg.training.dpo.beta = 0.1
cfg.training.dpo.lambda_dpo = 0.1
cfg.training.dpo.num_candidates = 4
cfg.training.dpo.ref_update_every = 1000
cfg.training.dpo.w_gray = 0.5
cfg.training.dpo.w_oob  = 5.0
cfg.training.dpo.w_curv = 0.5
cfg.training.dpo.w_len  = 0.2
cfg.training.dpo.w_gt   = 1.0
cfg.training.dpo.w_mse_energy = 1.0


cfg.loss_eval = edict()
cfg.loss_eval.type = ModelType.cvae
# cfg.loss_eval.gt_type = GTType.generated
cfg.loss_eval.distance_type = LossDisType.hausdorff # 可能删除
# cfg.loss_eval.hausdorff_dis = Hausdorff.average
cfg.loss_eval.dtw_use_cuda = True
cfg.loss_eval.dtw_gamma = 0.1
cfg.loss_eval.dtw_normalize = True
cfg.loss_eval.dtw_dist_func = DistanceFunction.euclidean
cfg.loss_eval.scale_waypoints = 1.0
cfg.loss_eval.dlow_sigma = 100.0
cfg.loss_eval.terrain_cost_map_resolution = configs["terrain_cost_map"]["resolution"]
cfg.loss_eval.collision_threshold = 1.0 / configs["terrain_cost_map"]["resolution"]

cfg.loss_eval.collision_type = CollisionLossType.global_dis
cfg.loss_eval.collision_detection_dis = int(1.0 / configs["terrain_cost_map"]["resolution"])
cfg.loss_eval.terrain_cost_map_sample_dis = 2
cfg.loss_eval.terrain_cost_map_threshold = cfg.data.terrain_cost_map_threshold
cfg.loss_eval.diversity_type = DiversityType.self_diversity

cfg.loss_eval.w_ep          = 100.0   # end point loss
cfg.loss_eval.w_terr        = 180.0   # 地形代价
# cfg.loss_eval.w_smooth      = 0.0
cfg.loss_eval.alpha         = 8.0   # 控制地形惩罚陡峭度 8.0
cfg.loss_eval.m_plate       = 0.15  # “平坦”窗口一半宽度
cfg.loss_eval.w_split       = 250.0   # split_path 调整权重
# cfg.loss_eval.w_cham        = 0.0
cfg.loss_eval.w_cos         = 0.0   # cosine loss 夹角惩罚
cfg.loss_eval.w_uniform     = 1.0   # 等距 loss 权重
cfg.loss_eval.dist_tol_px   = 3.0   # 允许的最大像素误差
cfg.loss_eval.w_oob         = 200.0 # 越界 loss 权重
cfg.loss_eval.w_dis         = 50.0   # 距离 loss 权重
cfg.loss_eval.w_jump         = 180.0  # 跳转 loss 权重
cfg.loss_eval.w_lossmap       = 3.0
# cfg.loss_eval.w_combine = 1.0  # 组合 loss 权重 
cfg.loss_eval.w_gray = 5.0          # 等灰阶权重

# ---- VAE KL β-anneal（線性暖啟）----
# β 在前 warmup 週期內從 beta_start 線性升到 beta_end，之後保持不變
cfg.loss_eval.vae_kld_beta_start = 1.0
cfg.loss_eval.vae_kld_beta_end   = 1.0
cfg.loss_eval.vae_kld_warmup_epochs = 10

cfg.loss_eval.asymmetric_ratio = 0
cfg.loss_eval.last_ratio = 2.0
cfg.loss_eval.vae_kld_ratio = 1.0
# cfg.loss_eval.dlow_kld_ratio = 0.01
cfg.loss_eval.distance_ratio = 10.0 # 可能删除
cfg.loss_eval.diversity_ratio = 1000.0
cfg.loss_eval.collision_mean_ratio = 10.0
cfg.loss_eval.collision_max_ratio = 10.0

cfg.loss_eval.coverage_with_last = True
cfg.loss_eval.coverage_distance_ratio = 10
cfg.loss_eval.coverage_last_ratio = 1
cfg.loss_eval.coverage_diverse_ratio = 1

cfg.logger = edict()
cfg.logger.log_steps = 5
cfg.logger.verbose = False
cfg.logger.reset_num_timesteps = True
cfg.logger.log_name = "./loginfo/"

cfg.model = edict()
cfg.model.perception = edict()
cfg.model.perception.fix_perception = False
cfg.model.perception.vel_dim = 20
cfg.model.perception.vel_out = 256
cfg.model.perception.lidar_out = 512
cfg.model.perception.lidar_norm_layer = False
cfg.model.perception.lidar_num = configs["rosbag"]["lidar"]["number"]
cfg.model.perception.lidar_mode = cfg.data.lidar_mode
cfg.model.perception.kpconv = edict()

cfg.model.perception.use_terrain_cost_map = False
cfg.model.perception.terrain_cost_map_size = cfg.data.terrain_cost_map_threshold * 2
cfg.model.perception.terrain_cost_map_out = cfg.model.perception.lidar_out + cfg.model.perception.vel_out

cfg.model.dlow = edict()
cfg.model.dlow.w_others = False
cfg.model.dlow.transformer_heads = 4
cfg.model.dlow.activation_func = None
cfg.model.dlow.fix_cvae = False
cfg.model.dlow.model_type = ModelType.cvae
cfg.model.dlow.rnn_type = RNNType.gru
cfg.model.dlow.perception_in = cfg.model.perception.lidar_out #+ cfg.model.perception.vel_out
cfg.model.dlow.vae_zd = 512
cfg.model.dlow.vae_output_threshold = 1
cfg.model.dlow.paths_num = 5
cfg.model.dlow.waypoints_num = 40 #20
cfg.model.dlow.waypoint_dim = 2
cfg.model.dlow.fix_first = False
cfg.model.dlow.cvae_file = None

# === 新增（cvae_core 的维度与注意力参数）===
cfg.model.cvae_core = edict()
cfg.model.cvae_core.activation_func = "relu"
cfg.model.cvae_core.out_steps   = 40
cfg.model.cvae_core.fmap_channels = 192
cfg.model.cvae_core.z_dim       = 32
cfg.model.cvae_core.q_dim       = 128
cfg.model.cvae_core.dec_hidden  = 256
cfg.model.cvae_core.n_scales    = 3
cfg.model.cvae_core.k_points    = 4
cfg.model.cvae_core.radius_px   = 6.0

cfg.model.cvae_core.red_ch    = 32
cfg.model.cvae_core.patch     = 5
cfg.model.cvae_core.d_model   = 128
cfg.model.cvae_core.n_heads   = 4


cfg.experiment = edict()
cfg.experiment.data_file = ""
cfg.experiment.saving_root = ""
cfg.experiment.name = "experiment"
cfg.experiment.display = True
# cfg.experiment.cropping_row = int(Camera_cfg.realsense_d435i.image_size[1] / 2.0)
cfg.experiment.metrics = edict()
# cfg.experiment.metrics.hausdorff_dis = Hausdorff.max
cfg.experiment.metrics.terrain_cost_map_resolution = cfg.loss_eval.terrain_cost_map_resolution
cfg.experiment.metrics.root = "experiments/results"
cfg.experiment.metrics.camera_type = 0

cfg.experiment.data = edict()
cfg.experiment.data.root = "datasets/terrain_cost_map_files_120"
cfg.experiment.data.idx = 1034
cfg.experiment.data.terrain_cost_map_threshold = cfg.data.terrain_cost_map_threshold
cfg.experiment.data.vel_num = cfg.data.vel_num
cfg.experiment.data.batch_size = 8

cfg.validation = edict()
cfg.validation.enable = False
cfg.validation.pixel_shift_px = 6
cfg.validation.n_variants = 4
cfg.validation.split_point = POINT_SPLIT_NUMBER




def get_args():
    parser = argparse.ArgumentParser(description='mapping')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--eval', action='store_true', default=False)
    parser.add_argument('--name', type=str, default="new")

    # Data settings
    # parser.add_argument('--data_root', type=str, default="/media/rob/D7E22D1D0C9FEC7A/1-Dataset/mtg/local_map_files_120")
    parser.add_argument('--data_root', type=str, default="/media/rob/D7E22D1D0C9FEC7A/1-Dataset/mapf")#"/media/rob/D7E22D1D0C9FEC7A/1-Dataset/gazebo_terrain_dataset")
    # parser.add_argument('--data_name', type=str, default="/media/rob/D7E22D1D0C9FEC7A/1-Dataset/mtg/local_map_files_120/data.pkl")
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--batch_size', type=int, default=16, help="batch size")
    parser.add_argument('--not_shuffle', action='store_true', default=False)
    parser.add_argument('--training_percentage', type=float, default=0.95)
    parser.add_argument('--lidar_mode', type=int, default=LidarMode.image, help="0: lidar image;  1: lidar point cloud")

    # Training settings
    parser.add_argument('--max_epoch', type=int, default=500, help="max epochs")
    parser.add_argument('--lr_decay_steps', type=int, default=2, help="number of waypoints")
    parser.add_argument('--grad_step', type=int, default=1, help="number of waypoints")
    parser.add_argument('--gamma', type=float, default=0.95, help="number of waypoints")

    parser.add_argument('--snap_shot', type=str, default="")
    parser.add_argument('--dlow_type', type=int, default=ModelType.cvae, help="0: CVAE, 1: dlow, 2: dlowae")
    parser.add_argument('--rnn', type=int, default=RNNType.gru, help="0: gru;  1: lstm")
    parser.add_argument('--fix_cvae', action='store_true', default=False)
    parser.add_argument('--w_eval', action='store_true', default=False)

    # Models settings
    parser.add_argument('--norm_lidar', action='store_true', default=False) # will be removed
    # parser.add_argument('--fix_obs', action='store_true', default=False)
    parser.add_argument('--use_terrain_cost_map', action='store_true', default=False)
    parser.add_argument('--waypoints_num', type=int, default=16, help="number of waypoints")
    parser.add_argument('--paths_num', type=int, default=5, help="number of paths of dlow")
    parser.add_argument('--activation_func', type=int, default=None, help="0 softsign,  1 tanh")
    parser.add_argument('--w_others', action='store_true', default=False)

    # Loss settings
    parser.add_argument('--distance_type', type=int, default=LossDisType.hausdorff, help="0: dtw;  1: hausdorff")
    parser.add_argument('--collision_type', type=int, default=CollisionLossType.local_dis, help="number of waypoints")
    parser.add_argument('--last_ratio', type=float, default=0, help="number of waypoints")
    parser.add_argument('--distance_ratio', type=float, default=0, help="number of waypoints")
    parser.add_argument('--min_of_k',type=int,default=4,help="number of VAE samples per item")

    return parser.parse_args()


def get_configs():
    args = get_args()

    if args.device is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)
        torch.cuda.set_device(args.device)
        cfg.device = "cuda:" + str(args.device)
        cfg.loss_eval.dtw_use_cuda = True
    else:
        cfg.device = "cpu"
        cfg.loss_eval.dtw_use_cuda = False

    cfg.eval = args.eval
    cfg.load_snapshot = args.snap_shot

    cfg.training.lr_decay_steps = args.lr_decay_steps
    cfg.training.lr_decay = args.gamma
    cfg.training.weight_decay = 1e-6
    cfg.training.grad_acc_steps = args.grad_step
    cfg.training.max_epoch = args.max_epoch
    cfg.data.w_eval = cfg.training.w_eval = args.w_eval
    cfg.training.min_of_k = max(1,int(args.min_of_k))

    # 更新数据文件路径
    # cfg.evaluation.root = args.data_root
    cfg.data.file = args.data_root
    cfg.data.name = args.data_root  # 更新为数据文件夹路径
    cfg.data.num_workers = args.workers
    cfg.data.batch_size = args.batch_size
    cfg.data.shuffle = not args.not_shuffle
    cfg.data.training_data_percentage = args.training_percentage
    cfg.data.lidar_mode = cfg.model.perception.lidar_mode = args.lidar_mode

    # 其他配置保持不变
    cfg.model.perception.lidar_norm_layer = args.norm_lidar
    if not cfg.model.perception.lidar_norm_layer:
        cfg.data.lidar_threshold = None


    if args.name is not None:
        cfg.name = args.name

    # cfg.name += "_wn{}".format(args.waypoints_num)
    # cfg.name += "_pn{}".format(args.paths_num)
    # cfg.name += "_lm{}".format(args.lidar_mode)
    # cfg.name += "_T{}".format(args.dlow_type)
    # # cfg.name += "_gt{}".format(args.gt_type)
    # cfg.name += "_oth{}".format(args.w_others)
    # cfg.name += "_lds{}".format(args.lr_decay_steps)
    # cfg.name += "_gs{}".format(args.grad_step)
    # cfg.name += "_vkl{}".format(args.vae_kld_ratio)
    # # cfg.name += "_dkl{}".format(args.dlow_kld_ratio)
    # cfg.name += "_lr{}".format(args.last_ratio)
    # cfg.name += "_disr{}".format(args.distance_ratio)
    # cfg.name += "_divr{}".format(args.diversity_ratio)
    # cfg.name += "_car{}".format(args.collision_mean_ratio)
    # cfg.name += "_cmr{}".format(args.collision_max_ratio)
    # cfg.name += "_cvlr{}".format(args.coverage_last_ratio)
    # cfg.name += "_cvdr{}".format(args.coverage_distance_ratio)
    # cfg.name += "_asmr{}".format(args.asymmetric_ratio)
    # cfg.name += "_cvdr{}".format(args.coverage_diverse_ratio)
        
    cfg.name += "_b{}".format(args.batch_size)
    cfg.name += "_wpn{}".format(args.waypoints_num)
    # cfg.name += "_act{}".format(args.activation_func)
    cfg.name += "_time{}".format(time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime()))

    return cfg