import numpy as np, torch
from pathlib import Path
import heapq

from src.configs import New_DataName
# from src.utils.astar import astar   # ← 直接把 annotator_3 的 astar 抽到 utils/astar.py，再引用
class Node:
    def __init__(self, x, y, cost, priority):
        self.x, self.y, self.cost, self.priority = x, y, cost, priority
    def __lt__(self, other): return self.priority < other.priority

def heuristic(x1, y1, x2, y2, z1, z2, w=10.5):
    return np.hypot(x2-x1, y2-y1) + w * abs(z2-z1)

def astar(map2d, start, goal, w=10.5):
    h, w_ = map2d.shape
    sx, sy = start; gx, gy = goal
    openq, closed = [], set()
    g_cost = np.full_like(map2d, np.inf, dtype=np.float32)
    g_cost[sx, sy] = 0
    parents = {(sx, sy): None}
    heapq.heappush(openq, (0, Node(sx, sy, 0, 0)))
    moves = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
    costs = [np.sqrt(2),1,np.sqrt(2),1,1,np.sqrt(2),1,np.sqrt(2)]
    while openq:
        _, cur = heapq.heappop(openq)
        if (cur.x, cur.y) in closed: continue
        closed.add((cur.x, cur.y))
        if (cur.x, cur.y) == (gx, gy):
            path, p = [], (gx, gy)
            while p:
                path.append(p); p = parents[p]
            return path[::-1]
        for (dx, dy), base in zip(moves, costs):
            nx, ny = cur.x+dx, cur.y+dy
            if 0 <= nx < h and 0 <= ny < w_:
                zc, zn = map2d[cur.x, cur.y], map2d[nx, ny]
                new_cost = g_cost[cur.x, cur.y] + base + abs(zc-zn)
                if new_cost < g_cost[nx, ny]:
                    g_cost[nx, ny] = new_cost
                    pr = new_cost + heuristic(nx, ny, gx, gy, zn, map2d[gx, gy])
                    parents[(nx, ny)] = (cur.x, cur.y)
                    heapq.heappush(openq, (pr, Node(nx, ny, new_cost, pr)))
    return []


# ────────────────────────────────────────────────────────────
def _resample_path(path_px: list[tuple[int,int]], n: int = 40) -> np.ndarray:
    """把任意长度像素路径等距抽 n 点 → (n,2)[x,y] 归一化到 0-1"""
    path = np.asarray(path_px, np.float32)                # (L,2)[row,col]
    idx  = np.linspace(0, len(path)-1, n).astype(int)
    sub  = path[idx]                                      # (n,2)
    # row=y, col=x  →  正常 [x,y]，再除以 wh
    return np.stack([sub[:,1], sub[:,0]], 1)              # (n,2)[x,y]

# ────────────────────────────────────────────────────────────
@torch.no_grad()
def jitter_batch(
        batch: dict,
        max_shift: int = 6,
        n_variants: int = 4,
        n_split: int = 40,
    ) -> dict:
    """
    • 对 batch 做“平移 Start / Goal”数据增强  
    • 返回一个 **新的 batch**（有可能更大），字段同原先一致
    """
    rgb  = batch[New_DataName.rgb_map]          # (B,3,H,W) in [0,1]
    st   = batch[New_DataName.Start]      # (B,2) in [0,1] (x,y)
    gl   = batch[New_DataName.Goal]       # (B,2)
    B,_,H,W = rgb.shape; device = rgb.device

    # 平移集合：上 / 下 / 左 / 右 / 原位  → 共 5
    offsets = [(0,0), ( 0,-max_shift), ( 0, max_shift),
                     (-max_shift, 0), ( max_shift, 0)]

    # 收集新样本
    rgb_lst, st_lst, gl_lst, sp_lst = [], [], [], []
    src_idx = []   
    for i in range(B):
        # --- 准备高度图 (blue channel) ---
        hmap = (rgb[i,2].cpu().numpy()*255).astype(np.float32)   # (H,W)
        # 像素坐标
        sx, sy = (st[i] * torch.tensor([W, H])).round().long().tolist()
        gx, gy = (gl[i] * torch.tensor([W, H])).round().long().tolist()


        # 16 组合并随机抽 n_variants
        combos = [(sx+dx1, sy+dy1, gx+dx2, gy+dy2)
                  for dx1,dy1 in offsets[1:]  # start 不取 (0,0)
                  for dx2,dy2 in offsets[1:]] # goal 不取 (0,0)
        np.random.shuffle(combos)
        combos = combos[:n_variants]

        for xs,ys,xg,yg in combos:
            if not (0<=xs<W and 0<=ys<H and 0<=xg<W and 0<=yg<H):   # 出界跳过
                continue
            path = astar(hmap, (ys,xs), (yg,xg))                    # A*
            if not path:                                            # 找不到路
                continue
            # ----- 填充字段 -----
            split = _resample_path(path, n_split)                   # (n,2)[x,y]0-1
            rgb_lst.append(rgb[i])                                  # 原图直接复用
            st_lst .append(torch.tensor([xs/W, ys/H]))
            gl_lst .append(torch.tensor([xg/W, yg/H]))
            sp_lst .append(torch.from_numpy(split))
            src_idx.append(i)

    if not sp_lst:          # 若全部失败，退回原 batch
        return batch

    # ===== 拼成新 batch =====
    new_batch = {}
    rep = len(rgb_lst)      # N_new
    keep_keys = {New_DataName.rgb_map,
                 New_DataName.Start,
                 New_DataName.Goal,
                 New_DataName.split_path}

    # ① 先塞改动过的 4 个字段
    new_batch[New_DataName.rgb_map]     = torch.stack(rgb_lst).to(device)
    new_batch[New_DataName.Start] = torch.stack(st_lst ).to(device)
    new_batch[New_DataName.Goal]  = torch.stack(gl_lst ).to(device)
    new_batch[New_DataName.split_path]  = torch.stack(sp_lst).to(device)

    idx_tensor = torch.tensor(src_idx, device=rgb.device)

    # ② 其他字段直接 repeat
    for k, v in batch.items():
        if k in keep_keys:
            continue
        if torch.is_tensor(v):                                 # Tensor → index_select
            new_batch[k] = v.index_select(0, idx_tensor)
        elif isinstance(v, list):                              # list → 列表推导
            new_batch[k] = [v[j] for j in src_idx]
        else:                                                  # 标量 / str → 直接复用
            new_batch[k] = v

    return new_batch
