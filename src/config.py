"""
config.py — 全局配置与个性化参数
《薄膜技术》AI4S 大作业：基于 MLP 的多层介质膜光谱预测与辅助设计

学号: 2023270171
  N             = 71          (学号后两位)
  lambda_target = 450 + 10*(N mod 31) = 450 + 10*9 = 540 nm
  seed         = 270171      (学号后 6 位整数)
  design_seed  = seed + 1 = 270172
"""

# ---------------------------------------------------------------------------
# 个性化参数（由学号 2023270171 唯一确定）
# ---------------------------------------------------------------------------
STUDENT_ID = "2023270171"
N_LAST_TWO = int(STUDENT_ID[-2:])            # 71
LAMBDA_TARGET = 450 + 10 * (N_LAST_TWO % 31) # 540 nm
SEED = int(STUDENT_ID[-6:])                  # 270171
DESIGN_SEED = SEED + 1                       # 270172

# ---------------------------------------------------------------------------
# 光学模型：Air / H / L / H / L / Glass，共 4 层，正入射、无吸收、忽略色散
# ---------------------------------------------------------------------------
N_H = 2.30          # 高折射率层
N_L = 1.45          # 低折射率层
N_S = 1.52          # 玻璃基底
N_INC = 1.00        # 空气（入射介质）

# 膜系：4 层，交替高/低，从空气侧数起
LAYER_INDICES = [N_H, N_L, N_H, N_L]
N_LAYERS = 4

# 膜厚范围 (nm)
D_MIN = 40.0
D_MAX = 180.0

# 波长范围 (nm)：400–800 nm，步长 10 nm，共 41 点
WAVELENGTH_MIN = 400.0
WAVELENGTH_MAX = 800.0
WAVELENGTH_STEP = 10.0

# ---------------------------------------------------------------------------
# 数据集
# ---------------------------------------------------------------------------
N_TOTAL = 5000
N_TRAIN = 4000
N_VAL = 500
N_TEST = 500

# ---------------------------------------------------------------------------
# MLP 与训练
# ---------------------------------------------------------------------------
HIDDEN_DIMS = [128, 128, 64]
ACTIVATION = "ReLU"          # 隐藏层激活
OUT_ACTIVATION = "Sigmoid"   # 输出层：R 落在 [0,1]
LR = 1e-3                    # Adam 初始学习率（余弦退火至 0）
LR_SCHEDULER = "CosineAnnealing"
BATCH_SIZE = 64
EPOCHS = 4000
INPUT_NORMALIZE = True       # 输入膜厚归一化 (d-40)/140 → [0,1]
LOSS = "MSE"

# 数据量实验
TRAIN_SIZES = [500, 1000, 2000, 4000]

# 辅助设计
N_CANDIDATES = 10000
TOP_K_PRED = 10
TOP_K_FINAL = 5

# ---------------------------------------------------------------------------
# 路径
# ---------------------------------------------------------------------------
import os
_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(_SRC_DIR)
DATA_DIR = os.path.join(PROJECT_DIR, "data")
FIG_DIR = os.path.join(PROJECT_DIR, "figures")
RESULT_DIR = os.path.join(PROJECT_DIR, "results")
MODEL_DIR = os.path.join(PROJECT_DIR, "results", "models")



def wavelength_grid():
    """返回 41 点波长网格 (nm)。"""
    import numpy as np
    n = int(round((WAVELENGTH_MAX - WAVELENGTH_MIN) / WAVELENGTH_STEP)) + 1
    return np.linspace(WAVELENGTH_MIN, WAVELENGTH_MAX, n)

def target_index():
    """λtarget 在波长网格中的下标。"""
    import numpy as np
    wl = wavelength_grid()
    return int(np.argmin(np.abs(wl - LAMBDA_TARGET)))


if __name__ == "__main__":
    wl = wavelength_grid()
    print(f"STUDENT_ID    = {STUDENT_ID}")
    print(f"N (后两位)     = {N_LAST_TWO}")
    print(f"lambda_target = {LAMBDA_TARGET} nm")
    print(f"seed          = {SEED}")
    print(f"design_seed   = {DESIGN_SEED}")
    print(f"波长网格点数   = {len(wl)}  ({wl[0]:.0f}–{wl[-1]:.0f} nm)")
    print(f"λtarget 下标   = {target_index()}")
