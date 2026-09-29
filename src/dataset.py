"""
dataset.py — 生成膜厚-光谱数据集并做固定划分

流程
----
1. 固定 random / numpy 随机种子为 seed = 270171
2. 在 [40, 180] nm 内均匀随机生成 5000 组 4 层膜厚
3. 用 TMM 计算每组在 400–800 nm (41 点) 的反射率谱
4. 一次固定随机排列后划分为：前 4000 训练 / 随后 500 验证 / 最后 500 测试
   该划分在后续所有实验中保持不变（写入磁盘复用）
"""

import os
import json
import numpy as np

import config as cfg
from tmm import reflectance_spectrum_batch


def generate_dataset(save=True, verbose=True):
    """生成数据集并固定划分。返回 dict。"""
    rng = np.random.RandomState(cfg.SEED)

    # 1) 5000 组膜厚，每层 U(40, 180) nm
    D = rng.uniform(cfg.D_MIN, cfg.D_MAX, size=(cfg.N_TOTAL, cfg.N_LAYERS))

    # 2) TMM 计算光谱
    wl = cfg.wavelength_grid()
    R = reflectance_spectrum_batch(D, cfg.LAYER_INDICES,
                                   cfg.N_INC, cfg.N_S, wl)

    # 3) 固定随机排列，确保划分可复现
    perm = np.random.RandomState(cfg.SEED).permutation(cfg.N_TOTAL)
    train_idx = perm[:cfg.N_TRAIN]
    val_idx = perm[cfg.N_TRAIN:cfg.N_TRAIN + cfg.N_VAL]
    test_idx = perm[cfg.N_TRAIN + cfg.N_VAL:]

    data = {
        "thickness": D.astype(np.float32),
        "spectra": R.astype(np.float32),
        "wavelengths": wl.astype(np.float32),
        "train_idx": train_idx.astype(np.int64),
        "val_idx": val_idx.astype(np.int64),
        "test_idx": test_idx.astype(np.int64),
    }

    if verbose:
        print(f"数据集: 膜厚 {D.shape}, 光谱 {R.shape}")
        print(f"划分: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")
        print(f"反射率范围: {R.min():.4f} – {R.max():.4f}")
        print(f"λtarget={cfg.LAMBDA_TARGET} nm 处 R 范围: "
              f"{R[:, cfg.target_index()].min():.4f} – {R[:, cfg.target_index()].max():.4f}")

    if save:
        os.makedirs(cfg.DATA_DIR, exist_ok=True)
        path = os.path.join(cfg.DATA_DIR, "dataset.npz")
        np.savez_compressed(path, **data)
        if verbose:
            print(f"已保存: {path}")

    return data


def load_dataset():
    """优先读磁盘缓存，否则重新生成。"""
    path = os.path.join(cfg.DATA_DIR, "dataset.npz")
    if os.path.exists(path):
        z = np.load(path)
        return {k: z[k] for k in z.files}
    return generate_dataset(save=True)


if __name__ == "__main__":
    generate_dataset()
