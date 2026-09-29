"""
design.py — MLP 辅助薄膜筛选与 TMM 验证

流程
----
1. 用 design_seed = seed + 1 = 270172 随机生成 10000 组新候选膜厚
   (与训练数据相互独立，避免信息泄漏)
2. 用训练好的 MLP 预测各候选的反射光谱
3. 按个人 λtarget = 540 nm 处的预测反射率降序排序，取 Top 10
4. 对 Top 10 用 TMM 重新计算真实光谱，按真实 R(λtarget) 确定最终 Top 5
"""

import os
import json
import numpy as np
import torch

import config as cfg
from tmm import reflectance_spectrum, reflectance_spectrum_batch
from model import build_model
from train import predict


def main():
    # ---- 载入模型 ----
    model = build_model(seed=cfg.SEED)
    sd = torch.load(os.path.join(cfg.MODEL_DIR, "mlp_full.pt"), map_location="cpu")
    model.load_state_dict(sd)

    # ---- 生成新候选（design_seed）----
    rng = np.random.RandomState(cfg.DESIGN_SEED)
    D_cand = rng.uniform(cfg.D_MIN, cfg.D_MAX,
                         size=(cfg.N_CANDIDATES, cfg.N_LAYERS))
    print(f"生成候选: {D_cand.shape} (design_seed={cfg.DESIGN_SEED})")

    # ---- MLP 预测光谱 ----
    R_pred = predict(model, D_cand)
    ti = cfg.target_index()
    R_pred_target = R_pred[:, ti]

    # ---- 按 λtarget 预测反射率排序，取 Top 10 ----
    order = np.argsort(-R_pred_target)
    top_pred_idx = order[:cfg.TOP_K_PRED]
    print(f"MLP 预测 R(λtarget) 最大值 = {R_pred_target[top_pred_idx[0]]:.4f}")

    # ---- TMM 验证 Top 10 ----
    wl = cfg.wavelength_grid()
    rows = []
    for rank, ci in enumerate(top_pred_idx, start=1):
        d_i = D_cand[ci]
        R_true = reflectance_spectrum(d_i, cfg.LAYER_INDICES,
                                      cfg.N_INC, cfg.N_S, wl)
        rows.append({
            "pred_rank": rank,
            "cand_index": int(ci),
            "d": [float(x) for x in d_i],
            "mlp_Rtarget": float(R_pred_target[ci]),
            "tmm_Rtarget": float(R_true[ti]),
            "tmm_spectrum": R_true.tolist(),
            "mlp_spectrum": R_pred[ci].tolist(),
            "abs_err_target": float(abs(R_pred_target[ci] - R_true[ti])),
        })

    # ---- 按 TMM 真实 R(λtarget) 排序，确定最终 Top 5 ----
    rows_sorted = sorted(rows, key=lambda r: -r["tmm_Rtarget"])
    for final_rank, r in enumerate(rows_sorted, start=1):
        r["final_rank"] = final_rank

    print()
    print("=" * 92)
    print(f"{'Pred':>4} {'Final':>5} | {'d1':>7} {'d2':>7} {'d3':>7} {'d4':>7} "
          f"| {'MLP R(λt)':>10} {'TMM R(λt)':>10} {'|err|':>8}")
    print("=" * 92)
    for r in rows_sorted:
        d1, d2, d3, d4 = r["d"]
        print(f"{r['pred_rank']:>4} {r['final_rank']:>5} | {d1:7.2f} {d2:7.2f} "
              f"{d3:7.2f} {d4:7.2f} | {r['mlp_Rtarget']:10.4f} "
              f"{r['tmm_Rtarget']:10.4f} {r['abs_err_target']:8.4f}")

    top5 = [r for r in rows_sorted if r["final_rank"] <= cfg.TOP_K_FINAL]

    # ---- 保存 ----
    np.savez(os.path.join(cfg.RESULT_DIR, "design_results.npz"),
             D_cand=D_cand, R_pred=R_pred,
             R_pred_target=R_pred_target,
             top_pred_idx=top_pred_idx)
    with open(os.path.join(cfg.RESULT_DIR, "top_designs.json"), "w",
              encoding="utf-8") as f:
        json.dump({"top10": rows_sorted, "top5": top5}, f,
                  ensure_ascii=False, indent=2)

    print()
    print(f"最终 Top 5 已保存到 {os.path.join(cfg.RESULT_DIR, 'top_designs.json')}")

    # 保存用于绘图：Top1 的 MLP 预测谱 vs TMM 真实谱
    best = rows_sorted[0]
    np.savez(os.path.join(cfg.RESULT_DIR, "best_design.npz"),
             d=np.array(best["d"]),
             mlp_spectrum=np.array(best["mlp_spectrum"]),
             tmm_spectrum=np.array(best["tmm_spectrum"]),
             wavelengths=wl)
    print(f"Top1 设计: d = {[round(x,2) for x in best['d']]} nm, "
          f"TMM R(λtarget) = {best['tmm_Rtarget']:.4f}")


if __name__ == "__main__":
    main()
