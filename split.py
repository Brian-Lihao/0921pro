import os
import shutil
import random
from tqdm import tqdm

# ========= 1. 用户只需改这里 =========
patch_dir   = "/media/rob/D7E22D1D0C9FEC7A/1-Dataset/mapf/output"   # 切片所在目录
inf_dir     = "/media/rob/D7E22D1D0C9FEC7A/1-Dataset/mapf/inference_set"  # 推理集输出目录
seed        = 42       # 可复现随机种子
# =====================================

# 32 张原图前缀（已去掉 .png）
PREFIXES = {
    "Berlin_1_256", "Boston_0_256", "brc202d", "den312d", "den520d",
    "empty-16-16", "empty-32-32", "empty-48-48", "empty-8-8",
    "ht_chantry", "ht_mansion_n", "lak303d", "lt_gallowstemplar_n",
    "maze-128-128-10", "maze-128-128-1", "maze-128-128-2",
    "maze-32-32-2", "maze-32-32-4", "orz900d", "ost003d",
    "Paris_1_256", "random-32-32-10", "random-32-32-20",
    "random-64-64-10", "random-64-64-20", "room-32-32-4",
    "room-64-64-16", "room-64-64-8", "warehouse-10-20-10-2-1",
    "warehouse-10-20-10-2-2", "warehouse-20-40-10-2-1",
    "warehouse-20-40-10-2-2", "w_woundedcoast"
}

def belong_to_prefix(fname):
    """返回 fname 属于哪个前缀，找不到返回 None"""
    for p in PREFIXES:
        if fname.startswith(p):
            return p
    return None

def split_inference():
    random.seed(seed)
    os.makedirs(inf_dir, exist_ok=True)

    # 按前缀聚合切片
    prefix_files = {p: [] for p in PREFIXES}
    for f in os.listdir(patch_dir):
        if not f.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue
        p = belong_to_prefix(f)
        if p:
            prefix_files[p].append(f)

    # 对每个前缀抽 10 %
    for p, files in prefix_files.items():
        n_total = len(files)
        if n_total == 0:
            continue
        n_infer = max(1, int(n_total * 0.1)) if n_total >= 10 else 1
        infer_samples = random.sample(files, n_infer)

        print(f"{p:25s} | total: {n_total:4d} → infer: {n_infer:3d}")
        for f in tqdm(infer_samples, desc=p, leave=False):
            shutil.copy2(os.path.join(patch_dir, f),
                         os.path.join(inf_dir, f))

    print("\n✅ 推理集已生成至:", inf_dir)

if __name__ == "__main__":
    split_inference()