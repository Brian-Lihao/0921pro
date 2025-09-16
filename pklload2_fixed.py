
import os
import pickle
import cv2
import numpy as np
from tqdm import tqdm

def _points_to_xy(points, coord_fmt):
    """把 (y,x) 或 (x,y) 统一成绘制用的 (x,y)"""
    points = np.array(points, dtype=float)
    if coord_fmt == "(y,x)":
        points = points[:, ::-1]  # swap -> (x,y)
    # 否则默认已经是 (x,y)
    return points.astype(int)

def process_single_pkl(pkl_path, image_dir, output_dir):
    base_name = os.path.splitext(os.path.basename(pkl_path))[0]
    image_path = os.path.join(image_dir, base_name + ".png")
    if not os.path.exists(image_path):
        image_path = os.path.join(image_dir, base_name + ".jpg")
        if not os.path.exists(image_path):
            print(f"[跳过] 找不到对应图像文件: {base_name}.png/.jpg")
            return

    # 读取图像
    lidar_image = cv2.imread(image_path)
    if lidar_image is None:
        print(f"[错误] 无法读取图像: {image_path}")
        return

    # 读取 pkl
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)

    if 'split_path' not in data:
        print(f"[警告] 跳过 {pkl_path}, 没有 'split_path' 键")
        return

    coord_fmt = data.get('coord_fmt', '(y,x)')
    path_pixels_xy = _points_to_xy(data['split_path'], coord_fmt)
    start_xy = _points_to_xy([data.get('start', path_pixels_xy[0])], coord_fmt)[0]
    goal_xy  = _points_to_xy([data.get('goal',  path_pixels_xy[-1])], coord_fmt)[0]

    height, width = lidar_image.shape[:2]
    path_pixels_xy[:, 0] = np.clip(path_pixels_xy[:, 0], 0, width - 1)
    path_pixels_xy[:, 1] = np.clip(path_pixels_xy[:, 1], 0, height - 1)

    if lidar_image.dtype != np.uint8:
        lidar_image = np.clip(lidar_image * 255 if lidar_image.dtype in [np.float32, np.float64] else lidar_image, 0, 255).astype(np.uint8)

    if len(lidar_image.shape) == 2:
        vis_map = cv2.cvtColor(lidar_image, cv2.COLOR_GRAY2BGR)
    else:
        vis_map = lidar_image.copy()

    # 起点、终点
    cv2.circle(vis_map, tuple(start_xy), 8, (0, 255, 0), -1)
    cv2.circle(vis_map, tuple(goal_xy),  8, (0, 0, 255), -1)

    # 路径线 + 路径点
    for i in range(len(path_pixels_xy) - 1):
        cv2.line(vis_map, tuple(path_pixels_xy[i]), tuple(path_pixels_xy[i + 1]), (255, 0, 0), 2)
    for pt in path_pixels_xy:
        cv2.circle(vis_map, tuple(pt), 2, (0, 255, 255), -1)

    # 保存图片
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, base_name + "_path.png")
    cv2.imwrite(save_path, vis_map)

def process_all_pkls(pkl_dir, image_dir, output_dir):
    pkl_files = sorted([f for f in os.listdir(pkl_dir) if f.endswith('.pkl')])
    if not pkl_files:
        print("[错误] 未找到任何 .pkl 文件")
        return

    for filename in tqdm(pkl_files, desc="批量绘图中"):
        pkl_path = os.path.join(pkl_dir, filename)
        process_single_pkl(pkl_path, image_dir, output_dir)

if __name__ == "__main__":
    dataset_test = "/media/rob/D7E22D1D0C9FEC7A/1-Dataset/mapf/points_pkls"
    image_path = "/media/rob/D7E22D1D0C9FEC7A/1-Dataset/mapf/labelled_imgs"
    output_dir = "/media/rob/D7E22D1D0C9FEC7A/1-Dataset/mapf/draw_dir"

    process_all_pkls(dataset_test, image_path, output_dir)
