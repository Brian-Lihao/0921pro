
import os
import cv2
import pickle
import heapq
import numpy as np
import random

# ==================== 全局设置 ====================
INPUT_IMG_PATH      = "/media/rob/D7E22D1D0C9FEC7A/1-Dataset/mapf/output"
OUT_PKL_DIR         = "/media/rob/D7E22D1D0C9FEC7A/1-Dataset/mapf/points_pkls"
OUT_IMG_DIR         = "/media/rob/D7E22D1D0C9FEC7A/1-Dataset/mapf/labelled_imgs"

USE_WEIGHTED_ASTAR  = False   # True=加权A*，False=标准A*
FLIP_MODE           = 1       # 0=不翻转 1=随机单次旋转(0/90/180/270) 2=四次都存
POINT_SPLIT_NUMBER  = 40

# 显示/统一尺寸（强制把所有图像 reshape 为 512x512，坐标也在 512x512 空间内记录）
STD_SIZE            = 512

# UI 尺寸
BTN_H, BTN_W, PAD   = 40, 120, 10
MIN_SHOW_H          = STD_SIZE  # 缩略显示的最小高度（保证按钮区可见）

# 八邻域 A*
MOVES   = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
COSTS   = [np.sqrt(2),1,np.sqrt(2),1,1,np.sqrt(2),1,np.sqrt(2)]

# ==================== A* ====================
class Node:
    def __init__(self, x, y, g, f):
        self.x, self.y, self.g, self.f = x, y, g, f
    def __lt__(self, other): return self.f < other.f

def heuristic(x1, y1, x2, y2, z1, z2, w=10.5):
    return np.hypot(x2-x1, y2-y1) + w*abs(z2-z1)

def astar(map2d, start, goal):
    h, w_ = map2d.shape
    sx, sy = start; gx, gy = goal
    open_list, closed = [], set()
    g_score = np.full_like(map2d, np.inf, dtype=np.float32)
    g_score[sx, sy] = 0
    parent = {(sx, sy): None}
    f0 = 0
    if USE_WEIGHTED_ASTAR:
        f0 = heuristic(sx, sy, gx, gy, map2d[sx, sy], map2d[gx, gy])
    heapq.heappush(open_list, Node(sx, sy, 0, f0))

    while open_list:
        cur = heapq.heappop(open_list)
        if (cur.x, cur.y) in closed: continue
        closed.add((cur.x, cur.y))
        if (cur.x, cur.y) == (gx, gy):
            path, p = [], (gx, gy)
            while p:
                path.append(p); p = parent[p]
            return path[::-1]
        for idx, (dx, dy) in enumerate(MOVES):
            nx, ny = cur.x+dx, cur.y+dy
            if 0<=nx<h and 0<=ny<w_:
                zc, zn = map2d[cur.x, cur.y], map2d[nx, ny]
                tentative = cur.g + COSTS[idx] + abs(zn-zc)
                if tentative < g_score[nx, ny]:
                    g_score[nx, ny] = tentative
                    parent[(nx, ny)] = (cur.x, cur.y)
                    f = tentative
                    if USE_WEIGHTED_ASTAR:
                        f += heuristic(nx, ny, gx, gy, zn, map2d[gx, gy])
                    heapq.heappush(open_list, Node(nx, ny, tentative, f))
    return []

# ==================== 坐标旋转（与 np.rot90(img, k) 完全一致，CCW） ====================
def rotate_point_ccw(y, x, H, W, k):
    """把 (y,x) 在 HxW 图里逆时针旋转 k 次（与 np.rot90 保持一致）"""
    k = int(k) % 4
    if k == 0:
        return y, x
    elif k == 1:
        # CCW 90: (y, x) -> (W-1 - x, y)
        return W - 1 - x, y
    elif k == 2:
        # CCW 180: (y, x) -> (H-1 - y, W-1 - x)
        return H - 1 - y, W - 1 - x
    else:  # k == 3
        # CCW 270: (y, x) -> (x, H-1 - y)
        return x, H - 1 - y

# ==================== 存储 ====================
def save_all(start, goal, path, base_name, k, img_rot):
    os.makedirs(OUT_PKL_DIR, exist_ok=True)
    os.makedirs(OUT_IMG_DIR, exist_ok=True)

    name = f"{base_name}_r{k*90}"
    pkl_path = os.path.join(OUT_PKL_DIR, f"{name}.pkl")
    img_path = os.path.join(OUT_IMG_DIR, f"{name}.png")

    arr = np.array(path, dtype=np.float32)
    idxs = np.linspace(0, len(arr)-1, POINT_SPLIT_NUMBER, dtype=int)
    data = {
        "start": np.array(start, dtype=np.float32),      # (y,x)
        "goal":  np.array(goal,  dtype=np.float32),      # (y,x)
        "path":  arr,                                    # (y,x)
        "split_path": arr[idxs],                         # (y,x)
        "img_size": (STD_SIZE, STD_SIZE),                # (H,W)
        "rot_k": int(k),                                 # 与 np.rot90(img, k) 一致（逆时针）
        "coord_fmt": "(y,x)",                            # 元组语义
        "note": "np.rot90 CCW by k; annotator_512_fixrot"
    }
    with open(pkl_path, 'wb') as f:
        pickle.dump(data, f)
    cv2.imwrite(img_path, img_rot)
    return name

# ==================== 标注器 ====================
class Annotator:
    def __init__(self):
        self.files = sorted([f for f in os.listdir(INPUT_IMG_PATH)
                             if f.lower().endswith(('.png','.jpg','.jpeg'))])
        if not self.files:
            print("No images found."); exit()

        self.img_idx = 0
        self.scale = 1.0  # 显示->内部(512x512) 的缩放系数： raw = disp * scale

        cv2.namedWindow('Annotator', cv2.WINDOW_NORMAL)
        cv2.setMouseCallback('Annotator', self.on_mouse)

        self._init_blank_canvas()   # 先画一次按钮，避免小图无按钮
        self.load_current()
        self.loop()

    # ---- 初始化一个空画布，强制包含按钮区 ----
    def _init_blank_canvas(self):
        btn_h = BTN_H + 2 * PAD
        self.show_h = max(MIN_SHOW_H, 512)   # 至少 512
        self.show_w = max(MIN_SHOW_H, 512)
        canvas = np.zeros((self.show_h + btn_h, self.show_w, 3), np.uint8)
        self.display = canvas
        self.btns = {}
        self.draw_ui()
        cv2.imshow('Annotator', self.display)
        cv2.resizeWindow('Annotator', self.show_w, self.show_h + btn_h)

    # ---- 载入图像并强制 reshape 为 512x512 ----
    def load_current(self):
        self.fname = self.files[self.img_idx]
        self.base  = os.path.splitext(self.fname)[0]

        img = cv2.imread(os.path.join(INPUT_IMG_PATH, self.fname))
        if img is None:
            print(f"[WARN] Failed to read {self.fname}, skip.")
            self.next_image()
            return

        # 灰度图转 BGR，统一三通道便于绘制
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        # 强制 reshape 为 512x512（上采样用线性插值，下采样用 AREA）
        ih, iw = img.shape[:2]
        interp = cv2.INTER_LINEAR if max(ih, iw) < STD_SIZE else cv2.INTER_AREA
        self.proc = cv2.resize(img, (STD_SIZE, STD_SIZE), interpolation=interp)

        # 内部逻辑在 512x512 坐标系下进行
        self.h, self.w = STD_SIZE, STD_SIZE

        # 计算用于显示的缩放比例（保证显示区 >= 512）
        btn_h = BTN_H + 2 * PAD
        self.show_h = max(MIN_SHOW_H, STD_SIZE)
        self.show_w = max(MIN_SHOW_H, STD_SIZE)
        self.scale  = self.h / self.show_h  # raw = disp * scale ；这里通常为 1.0

        # 状态复位
        self.reset()

    def reset(self):
        btn_h = BTN_H + 2 * PAD
        # 画布：图像显示区 + 按钮区
        self.display = np.zeros((self.show_h + btn_h, self.show_w, 3), dtype=np.uint8)
        # 把 512x512 图像缩放后贴到上半部（通常是 1:1 显示）
        img_disp = cv2.resize(self.proc, (self.show_w, self.show_h), interpolation=cv2.INTER_NEAREST)
        self.display[:self.show_h, :self.show_w] = img_disp
        self.points  = []
        self.path    = []
        self.draw_ui()
        cv2.imshow('Annotator', self.display)
        cv2.resizeWindow('Annotator', self.show_w, self.show_h + btn_h)

    # ---- 绘制按钮 ----
    def draw_ui(self):
        y0 = self.show_h + PAD
        self.btns = {}
        for i, (key, text, color) in enumerate([('del', 'Delete', (0, 0, 255)),
                                                ('save', 'Save', (0, 255, 0)),
                                                ('undo', 'Undo', (255, 0, 0))]):
            x1 = PAD + i * (BTN_W + PAD)
            x2, y1, y2 = x1 + BTN_W, y0, y0 + BTN_H
            self.btns[key] = (x1, y1, x2, y2)
            cv2.rectangle(self.display, (x1, y1), (x2, y2), color, -1)
            cv2.putText(self.display, text, (x1 + 10, y1 + 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # ---- 鼠标事件 ----
    def on_mouse(self, event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return

        # 按钮检测（显示坐标系）
        for key, (x1, y1, x2, y2) in self.btns.items():
            if x1 <= x <= x2 and y1 <= y <= y2:
                getattr(self, f'btn_{key}')()
                return

        # 在图像区域内标点（显示坐标 -> 512x512 原始内部坐标）
        if y < self.show_h and x < self.show_w and len(self.points) < 2:
            y_raw = int(y * self.scale)
            x_raw = int(x * self.scale)
            y_raw = max(0, min(self.h-1, y_raw))
            x_raw = max(0, min(self.w-1, x_raw))
            self.points.append((y_raw, x_raw))
            if len(self.points) == 2:
                gray = cv2.cvtColor(self.proc, cv2.COLOR_BGR2GRAY).astype(np.float32)
                self.path = astar(gray, self.points[0], self.points[1])
            self.redraw()

    def redraw(self):
        btn_h = BTN_H + 2 * PAD
        self.display = np.zeros((self.show_h + btn_h, self.show_w, 3), dtype=np.uint8)
        img_disp = cv2.resize(self.proc, (self.show_w, self.show_h), interpolation=cv2.INTER_NEAREST)
        self.display[:self.show_h, :self.show_w] = img_disp

        # 画路径（把 512x512 内部坐标 -> 显示坐标）
        if self.path:
            pts = np.array([(int(x / self.scale), int(y / self.scale)) for (y, x) in self.path], np.int32)
            cv2.polylines(self.display[:self.show_h, :self.show_w], [pts], False, (255, 0, 255), 2)

        # 画起点/终点（同样坐标映射）
        for idx, (y_raw, x_raw) in enumerate(self.points):
            x_show = int(x_raw / self.scale)
            y_show = int(y_raw / self.scale)
            color = (0, 255, 0) if idx == 0 else (0, 0, 255)
            cv2.circle(self.display[:self.show_h, :self.show_w], (x_show, y_show), 6, color, -1)

        self.draw_ui()
        cv2.imshow('Annotator', self.display)

    # ---- 按钮逻辑 ----
    def btn_del(self):
        if not self.confirm(f"Delete {self.fname}? "): return
        try:
            os.remove(os.path.join(INPUT_IMG_PATH, self.fname))
        except Exception as e:
            print(f"[WARN] Failed to delete {self.fname}: {e}")
        else:
            self.files.pop(self.img_idx)
            if self.img_idx >= len(self.files): self.img_idx = 0
        if self.files:
            self.load_current()
        else:
            print("No images left.")
            cv2.destroyAllWindows()

    def btn_save(self):
        if len(self.points)!=2 or not self.path:
            print("Need 2 points and a valid path."); return

        gray512 = cv2.cvtColor(self.proc, cv2.COLOR_BGR2GRAY).astype(np.float32)

        if FLIP_MODE == 0:
            save_all(self.points[0], self.points[1], self.path, self.base, 0, self.proc)
            print("Saved (no flip).")
            self.next_image()

        elif FLIP_MODE == 1:
            k = random.choice([0,1,2,3])
            if k == 0:
                save_all(self.points[0], self.points[1], self.path, self.base, 0, self.proc)
            else:
                # 与 np.rot90 的 CCW 一致
                gray_r = np.rot90(gray512, k)
                img_r  = np.rot90(self.proc, k)
                y1,x1 = rotate_point_ccw(self.points[0][0], self.points[0][1], STD_SIZE, STD_SIZE, k)
                y2,x2 = rotate_point_ccw(self.points[1][0], self.points[1][1], STD_SIZE, STD_SIZE, k)
                path_r = astar(gray_r, (y1,x1), (y2,x2))
                if not path_r:
                    print("Rotated path failed."); return
                save_all((y1,x1), (y2,x2), path_r, self.base, k, img_r)
            print("Saved (random single flip).")
            self.next_image()

        else:  # FLIP_MODE == 2
            for k in range(4):
                if k == 0:
                    save_all(self.points[0], self.points[1], self.path, self.base, 0, self.proc)
                    continue
                gray_r = np.rot90(gray512, k)
                img_r  = np.rot90(self.proc, k)
                y1,x1 = rotate_point_ccw(self.points[0][0], self.points[0][1], STD_SIZE, STD_SIZE, k)
                y2,x2 = rotate_point_ccw(self.points[1][0], self.points[1][1], STD_SIZE, STD_SIZE, k)
                path_r = astar(gray_r, (y1,x1), (y2,x2))
                if path_r:
                    save_all((y1,x1), (y2,x2), path_r, self.base, k, img_r)
            print("Saved (4 rotations).")
            self.next_image()

    def btn_undo(self):
        self.reset()

    # ---- 辅助 ----
    def confirm(self, msg):
        print(msg, '[y/n]:', end='', flush=True)
        while True:
            k = cv2.waitKey(0) & 0xFF
            if k in (ord('y'), ord('Y')): print('y'); return True
            if k in (ord('n'), ord('N')): print('n'); return False

    def next_image(self):
        self.img_idx = (self.img_idx + 1) % len(self.files)
        self.load_current()

    # ---- 主循环 ----
    def loop(self):
        while True:
            k = cv2.waitKey(50) & 0xFF
            if k == 27: break           # ESC 退出
            if k == ord('n'): self.next_image()
        cv2.destroyAllWindows()

# ==================== 入口 ====================
if __name__ == '__main__':
    random.seed(42)
    Annotator()
