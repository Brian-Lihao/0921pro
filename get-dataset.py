import cv2
import os
import numpy as np
from tqdm import tqdm

def sliding_window(image, window_size, step_size):
    """生成滑动窗口的坐标"""
    for y in range(0, image.shape[0] - window_size[1] + 1, step_size[1]):
        for x in range(0, image.shape[1] - window_size[0] + 1, step_size[0]):
            yield (x, y, image[y:y + window_size[1], x:x + window_size[0]])

def extract_patches_from_image(image_path, output_dir, window_sizes, step_ratio=0.5):
    image = cv2.imread(image_path)
    if image is None:
        print(f"无法读取图像: {image_path}")
        return

    base_name = os.path.splitext(os.path.basename(image_path))[0]

    for window_size in window_sizes:
        step_size = (int(window_size[0] * step_ratio), int(window_size[1] * step_ratio))
        winW, winH = window_size

        count = 0
        for (x, y, patch) in sliding_window(image, (winW, winH), step_size):
            patch_filename = f"{base_name}_w{winW}_h{winH}_x{x}_y{y}.jpg"
            patch_path = os.path.join(output_dir, patch_filename)
            cv2.imwrite(patch_path, patch)
            count += 1
        print(f"提取 {count} 个窗口图像，窗口大小: {window_size}")

def process_folder(input_folder, output_folder, window_sizes, step_ratio=0.5):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
    for img_file in tqdm(image_files, desc="处理图像"):
        img_path = os.path.join(input_folder, img_file)
        extract_patches_from_image(img_path, output_folder, window_sizes, step_ratio)

# === 参数设置 ===
input_folder = "/media/rob/D7E22D1D0C9FEC7A/1-Dataset/mapf"
output_folder = "/media/rob/D7E22D1D0C9FEC7A/1-Dataset/mapf/output"

# 窗口大小（模拟卷积核大小），你可以自定义
window_sizes = [(128, 128), (256, 256), (512, 512)]

# 步长比例（相对于窗口大小），0.5 表示 50% 重叠
step_ratio = 0.5

# === 运行 ===
process_folder(input_folder, output_folder, window_sizes, step_ratio)