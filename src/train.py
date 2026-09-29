"""
train.py — 训练 MLP 代理模型 + 数据量影响实验

实验
----
A. 完整训练：用 4000 组训练样本训练基础模型，记录 train/val loss 曲线，
   在固定 500 组测试集上评估，并保存预测光谱、失败案例。
B. 数据量实验：从同一 4000 组训练池中取前 500 / 1000 / 2000 / 4000 组分别
   训练，验证集与测试集始终不变，比较测试误差。
"""

import os
import json
import time
import numpy as np
import torch
import torch.nn as nn

import config as cfg
from model import build_model
from dataset import load_dataset


def set_all_seeds(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def normalize_X(X):
    """输入膜厚归一化到 [0, 1]：(d - 40) / 140。

    原始膜厚数值范围 40–180 nm，直接输入会使第一层线性层处于
    较大的数值尺度，收敛慢且批间波动大；归一化后训练更稳定。
    训练与设计阶段的预测共用本函数，保证预处理一致。
    """
    return (np.asarray(X, dtype=np.float32) - cfg.D_MIN) / (cfg.D_MAX - cfg.D_MIN)


def train_one(train_X, train_Y, val_X, val_Y, seed=None,
              epochs=None, verbose=False, tag=""):
    """训练单个模型，返回 (model, history)。

    输入 X 为原始膜厚（nm），在函数内部统一做归一化。
    """
    seed = cfg.SEED if seed is None else seed
    epochs = epochs or cfg.EPOCHS

    set_all_seeds(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_model(seed=seed, device=device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.LR)
    # 余弦退火：学习率从 1e-3 平滑衰减到 ~0，后期小步长精修
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs)

    Xtr = torch.tensor(normalize_X(train_X), dtype=torch.float32, device=device)
    Ytr = torch.tensor(train_Y, dtype=torch.float32, device=device)
    Xva = torch.tensor(normalize_X(val_X), dtype=torch.float32, device=device)
    Yva = torch.tensor(val_Y, dtype=torch.float32, device=device)

    n = Xtr.shape[0]
    g = torch.Generator(device=device).manual_seed(seed)

    hist = {"train_loss": [], "val_loss": [], "epoch": []}

    for ep in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(n, generator=g, device=device)
        ep_loss = 0.0
        for i in range(0, n, cfg.BATCH_SIZE):
            idx = perm[i:i + cfg.BATCH_SIZE]
            xb, yb = Xtr[idx], Ytr[idx]
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            ep_loss += loss.item() * xb.shape[0]
        tr_loss = ep_loss / n
        scheduler.step()

        model.eval()
        with torch.no_grad():
            va_loss = criterion(model(Xva), Yva).item()

        hist["epoch"].append(ep)
        hist["train_loss"].append(tr_loss)
        hist["val_loss"].append(va_loss)

        if verbose and (ep % 400 == 0 or ep == 1):
            print(f"  [{tag}] epoch {ep:4d}/{epochs}  train={tr_loss:.3e}  "
                  f"val={va_loss:.3e}  lr={scheduler.get_last_lr()[0]:.2e}")

    return model, hist


@torch.no_grad()
def predict(model, X):
    """对原始膜厚（nm）做预测：内部先归一化再前向传播。"""
    device = next(model.parameters()).device
    model.eval()
    Xn = torch.tensor(normalize_X(X), dtype=torch.float32, device=device)
    return model(Xn).cpu().numpy()


def main():
    os.makedirs(cfg.MODEL_DIR, exist_ok=True)
    os.makedirs(cfg.RESULT_DIR, exist_ok=True)

    d = load_dataset()
    D, R = d["thickness"], d["spectra"]
    tr, va, te = d["train_idx"], d["val_idx"], d["test_idx"]

    Xtr, Ytr = D[tr], R[tr]
    Xva, Yva = D[va], R[va]
    Xte, Yte = D[te], R[te]

    print("=" * 70)
    print("实验 A：完整训练（4000 训练样本）")
    print("=" * 70)
    t0 = time.time()
    model, hist = train_one(Xtr, Ytr, Xva, Yva, seed=cfg.SEED, verbose=True)
    dt = time.time() - t0
    print(f"训练完成，用时 {dt:.1f} s")

    # ---- 测试集评估 ----
    Yte_pred = predict(model, Xte)
    mse = float(np.mean((Yte_pred - Yte) ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(Yte_pred - Yte)))
    print(f"测试集: MSE={mse:.3e}  RMSE={rmse:.4f}  MAE={mae:.4f}")

    # 逐样本误差（用于失败案例）
    per_sample_mse = np.mean((Yte_pred - Yte) ** 2, axis=1)
    worst = int(np.argmax(per_sample_mse))
    print(f"最差样本: idx={worst} (测试集内), MSE={per_sample_mse[worst]:.3e}")

    # 保存模型与预测
    torch.save(model.state_dict(), os.path.join(cfg.MODEL_DIR, "mlp_full.pt"))
    np.savez(os.path.join(cfg.RESULT_DIR, "full_model_results.npz"),
             hist_epoch=np.array(hist["epoch"]),
             hist_train=np.array(hist["train_loss"]),
             hist_val=np.array(hist["val_loss"]),
             Yte=Yte, Yte_pred=Yte_pred,
             Xte=Xte, per_sample_mse=per_sample_mse,
             test_idx=te)

    # ---- 实验 B：数据量影响 ----
    print()
    print("=" * 70)
    print("实验 B：训练数据量影响（500 / 1000 / 2000 / 4000）")
    print("=" * 70)
    size_results = []
    for n_size in cfg.TRAIN_SIZES:
        Xs, Ys = Xtr[:n_size], Ytr[:n_size]
        t1 = time.time()
        m, h = train_one(Xs, Ys, Xva, Yva, seed=cfg.SEED, verbose=False,
                         tag=f"n={n_size}")
        Yp = predict(m, Xte)
        m_mse = float(np.mean((Yp - Yte) ** 2))
        m_rmse = float(np.sqrt(m_mse))
        m_mae = float(np.mean(np.abs(Yp - Yte)))
        # λtarget 处的误差
        ti = cfg.target_index()
        tgt_mae = float(np.mean(np.abs(Yp[:, ti] - Yte[:, ti])))
        size_results.append({
            "n_train": n_size, "test_mse": m_mse, "test_rmse": m_rmse,
            "test_mae": m_mae, "target_mae": tgt_mae,
            "final_val_loss": h["val_loss"][-1],
            "time_s": time.time() - t1,
        })
        print(f"  n_train={n_size:5d}  test_MSE={m_mse:.3e}  RMSE={m_rmse:.4f}  "
              f"MAE={m_mae:.4f}  R(λt)_MAE={tgt_mae:.4f}")

    with open(os.path.join(cfg.RESULT_DIR, "training_size_results.json"),
              "w", encoding="utf-8") as f:
        json.dump(size_results, f, ensure_ascii=False, indent=2)

    summary = {
        "student_id": cfg.STUDENT_ID,
        "lambda_target": cfg.LAMBDA_TARGET,
        "seed": cfg.SEED,
        "design_seed": cfg.DESIGN_SEED,
        "epochs": cfg.EPOCHS,
        "batch_size": cfg.BATCH_SIZE,
        "lr": cfg.LR,
        "hidden_dims": cfg.HIDDEN_DIMS,
        "n_params": int(sum(p.numel() for p in model.parameters())),
        "test_mse": mse, "test_rmse": rmse, "test_mae": mae,
        "train_time_s": dt,
        "size_results": size_results,
    }
    with open(os.path.join(cfg.RESULT_DIR, "summary.json"), "w",
              encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print()
    print(f"结果已保存到 {cfg.RESULT_DIR}")


if __name__ == "__main__":
    main()
