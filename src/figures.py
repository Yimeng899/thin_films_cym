"""
figures.py — 生成论文所需的全部 8 张图（印刷尺寸版）

设计规范：
- 图幅按最终插入宽度（5.9–6.5 in）设计，图内文字最终有效字号约 7.5–10 pt，
  与正文五号/题注小五的层级一致
- 全图统一字体：微软雅黑（中文、英文、数字一致，无混搭）
- 图内文字统一中文表述；科学计数法用 mathtext（$a.bc\\times10^{-n}$）
- Figure 4 训练损失曲线：原始逐轮记录（细线）+ 50 轮滑动平均（粗线），
  毛刺为小批量 SGD 的批间波动，属正常现象且完整保留

Figure 1  总体工作流图
Figure 2  物理模型 + TMM 数据生成示意
Figure 3  MLP 结构图
Figure 4  训练/验证损失曲线
Figure 5  TMM 与 MLP 光谱对比（≥3 个测试样本）
Figure 6  训练数据量对测试误差的影响
Figure 7  选中设计的预测谱与验证谱（标出 λtarget）
Figure 8  一个误差较大的代表性失败案例
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

import config as cfg
from dataset import load_dataset
from tmm import reflectance_spectrum

# ---------------------------------------------------------------------------
# 字体：全图统一微软雅黑（含拉丁字符与数字）
# ---------------------------------------------------------------------------
import matplotlib.font_manager as fm
from matplotlib import font_manager

_CJK_CANDIDATES = [
    r"C:\Windows\Fonts\msyh.ttc",
    r"C:\Windows\Fonts\simhei.ttf",
]
_CJK_NAME = None
for _p in _CJK_CANDIDATES:
    if os.path.exists(_p):
        try:
            font_manager.fontManager.addfont(_p)
            _CJK_NAME = fm.FontProperties(fname=_p).get_name()
            break
        except Exception:
            continue

matplotlib.rcParams["font.family"] = _CJK_NAME or "sans-serif"
matplotlib.rcParams["font.sans-serif"] = [_CJK_NAME] if _CJK_NAME else []
matplotlib.rcParams["axes.unicode_minus"] = False
# 数学文本同样使用雅黑（上下标由排版位置实现，不依赖 Unicode 上标字形）
matplotlib.rcParams["mathtext.fontset"] = "custom"
matplotlib.rcParams["mathtext.default"] = "regular"

plt.rcParams.update({
    "font.size": 9,
    "axes.linewidth": 0.8,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.6,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

C_TMM = "#c0392b"     # TMM 真实值：红
C_MLP = "#2e6db4"     # MLP 预测：蓝
C_TRAIN = "#2e6db4"
C_VAL = "#e08a1e"
C_GREY = "#5a6478"


def sci(v, sig=3):
    """3 位有效数字科学计数法 mathtext，如 $6.90\\times 10^{-4}$。"""
    if v == 0:
        return "0"
    exp = int(np.floor(np.log10(abs(v))))
    mant = v / 10 ** exp
    return f"${mant:.{sig - 1}f}\\times 10^{{{exp}}}$"


def moving_average(y, w=50):
    """滑动平均（窗口 w），边缘收缩窗口。"""
    y = np.asarray(y, dtype=float)
    n = len(y)
    out = np.empty(n)
    half = w // 2
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        out[i] = y[lo:hi].mean()
    return out


def _save(fig, name):
    os.makedirs(cfg.FIG_DIR, exist_ok=True)
    p = os.path.join(cfg.FIG_DIR, name)
    fig.savefig(p)
    plt.close(fig)
    print(f"  已生成 {name}")


# ---------------------------------------------------------------------------
# Figure 1 — 总体工作流（插入宽 6.1 in）
# ---------------------------------------------------------------------------
def fig1_workflow():
    fig, ax = plt.subplots(figsize=(6.5, 2.55))
    ax.set_xlim(0, 100); ax.set_ylim(0, 34); ax.axis("off")

    boxes = [
        (2.0,  "物理模型\nAir/H/L/H/L/Glass", "#eef4fc", "#7ba3d4"),
        (20.5, "TMM 生成数据\n5000 组膜厚→光谱", "#eef4fc", "#7ba3d4"),
        (39.0, "固定划分\n4000/500/500", "#eef4fc", "#7ba3d4"),
        (57.5, "MLP 代理模型\n4-128-128-64-41", "#fdf1e3", "#e0a75e"),
        (76.0, "MLP 筛选候选\nTop10 → TMM 验证", "#eaf6ee", "#6fb98a"),
    ]
    w, h, y = 17.5, 11.5, 12
    for x, label, fc, ec in boxes:
        ax.add_patch(FancyBboxPatch((x, y), w, h,
                     boxstyle="round,pad=0.5,rounding_size=1.2",
                     fc=fc, ec=ec, lw=1.3))
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center",
                fontsize=7.8, color="#1a2233", linespacing=1.5)

    for i in range(len(boxes) - 1):
        x0 = boxes[i][0] + w
        ax.add_patch(FancyArrowPatch((x0 + 0.3, y + h / 2), (boxes[i + 1][0] - 0.3, y + h / 2),
                     arrowstyle="-|>", mutation_scale=11, lw=1.3, color=C_GREY))

    ax.text(50, 30.5, "AI4S 流程：物理模型 → 数据 → AI 代理模型 → 设计验证",
            ha="center", fontsize=10, fontweight="bold", color="#1a2233")
    ax.text(2, 6.5, "λtarget = 540 nm    seed = 270171    design_seed = 270172",
            ha="left", fontsize=8, color="#8a5a00")
    ax.text(2, 2.0, "注：最终设计结果一律以 TMM 复算为准。",
            ha="left", fontsize=7.5, color=C_GREY, style="italic")

    _save(fig, "Figure1_workflow.png")


# ---------------------------------------------------------------------------
# Figure 2 — 物理模型 + TMM 数据生成（插入宽 6.1 in）
# ---------------------------------------------------------------------------
def fig2_physical_model():
    d = load_dataset()
    D, R = d["thickness"], d["spectra"]
    wl = d["wavelengths"]

    fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.85),
                             gridspec_kw={"width_ratios": [1, 1.25]})

    # 左：层状结构示意
    ax = axes[0]
    layers = [("空气", 1.00, "#f2f5f9"), ("H", cfg.N_H, "#c9dbf2"),
              ("L", cfg.N_L, "#e8eef7"), ("H", cfg.N_H, "#c9dbf2"),
              ("L", cfg.N_L, "#e8eef7"), ("玻璃", cfg.N_S, "#dfe6ee")]
    th = 1.0
    for i, (name, n, color) in enumerate(layers):
        yy = len(layers) - 1 - i
        ax.add_patch(Rectangle((0, yy * th), 3, th, fc=color,
                               ec="#5a6478", lw=1.0))
        ax.text(1.5, yy * th + th / 2, f"{name}", ha="center", va="center",
                fontsize=8, color="#1a2233",
                fontweight="bold" if name in ("H", "L") else "normal")
        if name in ("H", "L"):
            ax.text(3.15, yy * th + th / 2, f"n={n}", ha="left", va="center",
                    fontsize=7, color=C_GREY)
        if name not in ("空气", "玻璃"):
            ax.text(-0.15, yy * th + th / 2, "40–180 nm", ha="right",
                    va="center", fontsize=6.8, color=C_GREY)
    ax.annotate("", xy=(0.35, len(layers) * th + 0.25), xytext=(0.35, len(layers) * th - 0.15),
                arrowprops=dict(arrowstyle="-|>", color="#c0392b", lw=1.4))
    ax.text(0.5, len(layers) * th + 0.45, "入射光（正入射）", fontsize=7.2, color="#c0392b")
    ax.annotate("", xy=(2.6, len(layers) * th - 0.15), xytext=(2.6, len(layers) * th + 0.25),
                arrowprops=dict(arrowstyle="-|>", color="#2e6db4", lw=1.4))
    ax.text(2.75, len(layers) * th + 0.45, "反射 R(λ)", fontsize=7.2, color="#2e6db4")
    ax.set_xlim(-1.4, 4.6); ax.set_ylim(-0.3, len(layers) + 0.9)
    ax.axis("off")
    ax.set_title("(a) 膜系结构", fontsize=9.5, color="#1a2233")

    # 右：TMM 生成的示例光谱
    ax = axes[1]
    rng = np.random.RandomState(cfg.SEED)
    sample_ids = rng.choice(len(D), 6, replace=False)
    for sid in sample_ids:
        ax.plot(wl, R[sid], lw=0.9, alpha=0.75)
    ax.axvline(cfg.LAMBDA_TARGET, color="#c0392b", ls="--", lw=1.1,
               label=f"λtarget = {cfg.LAMBDA_TARGET} nm")
    ax.set_xlabel("波长 (nm)")
    ax.set_ylabel("反射率 R")
    ax.set_ylim(-0.02, 0.72)
    ax.set_xticks([400, 500, 600, 700, 800])
    ax.legend(fontsize=7.2, loc="upper right")
    ax.set_title("(b) TMM 计算的示例反射光谱", fontsize=9.5, color="#1a2233")

    fig.tight_layout()
    _save(fig, "Figure2_physical_model.png")


# ---------------------------------------------------------------------------
# Figure 3 — MLP 结构图（插入宽 6.1 in）
# ---------------------------------------------------------------------------
def fig3_mlp_arch():
    fig, ax = plt.subplots(figsize=(6.5, 2.55))
    ax.set_xlim(0, 100); ax.set_ylim(0, 40); ax.axis("off")

    dims = [cfg.N_LAYERS] + cfg.HIDDEN_DIMS + [len(cfg.wavelength_grid())]
    labels = ["输入层\n4 ($d_1$–$d_4$)", "128", "128", "64", "输出层\n41 (R)"]
    xs = [6, 26, 46, 66, 88]
    colors = ["#dfe6ee", "#c9dbf2", "#c9dbf2", "#c9dbf2", "#f7dcc8"]
    edges = ["#7b8ba3", "#7ba3d4", "#7ba3d4", "#7ba3d4", "#e0a75e"]

    def n_circles(k):
        return min(9, max(3, k // 12 + 3))

    for xi, lab, c, e, dim in zip(xs, labels, colors, edges, dims):
        n = n_circles(dim)
        ys = np.linspace(6, 34, n)
        for yy in ys:
            ax.add_patch(plt.Circle((xi, yy), 1.15, fc=c, ec=e, lw=1.0, zorder=3))
        ax.text(xi, 38.0, lab, ha="center", fontsize=8, color="#1a2233",
                fontweight="bold", linespacing=1.4)

    for i in range(len(xs) - 1):
        n0 = n_circles(dims[i]); n1 = n_circles(dims[i + 1])
        for y0 in np.linspace(6, 34, n0):
            for y1 in np.linspace(6, 34, n1):
                ax.plot([xs[i] + 1.15, xs[i + 1] - 1.15], [y0, y1],
                        color="#b9c6d8", lw=0.3, alpha=0.55, zorder=1)

    ax.text(50, 2.0,
            "隐藏层激活 ReLU · 输出层 Sigmoid · 损失 MSE · 优化器 Adam（学习率 $1\\times 10^{-3}$）",
            ha="center", fontsize=7.8, color=C_GREY)
    ax.set_title("MLP 代理模型结构：4–128–128–64–41",
                 fontsize=10, fontweight="bold", color="#1a2233", pad=8)

    _save(fig, "Figure3_mlp_architecture.png")


# ---------------------------------------------------------------------------
# Figure 4 — 训练/验证损失曲线（插入宽 5.9 in）
# ---------------------------------------------------------------------------
def fig4_loss():
    z = np.load(os.path.join(cfg.RESULT_DIR, "full_model_results.npz"))
    ep, tr, va = z["hist_epoch"], z["hist_train"], z["hist_val"]

    W = 50
    tr_s = moving_average(tr, W)
    va_s = moving_average(va, W)

    fig, ax = plt.subplots(figsize=(5.9, 3.9))
    ax.semilogy(ep, tr, color=C_TRAIN, lw=0.6, alpha=0.22)
    ax.semilogy(ep, va, color=C_VAL, lw=0.6, alpha=0.22)
    ax.semilogy(ep, tr_s, color=C_TRAIN, lw=1.7, label="训练损失")
    ax.semilogy(ep, va_s, color=C_VAL, lw=1.7, label="验证损失")

    ax.set_xlabel("训练轮次（Epoch）")
    ax.set_ylabel("MSE 损失（对数坐标）")
    ax.legend(fontsize=8.5, loc="upper right")
    ax.set_title("MLP 代理模型训练与验证损失曲线", fontsize=10,
                 fontweight="bold", color="#1a2233")
    ax.text(0.97, 0.62,
            f"细线：逐轮原始记录\n粗线：{W} 轮滑动平均\n\n"
            f"最终训练损失 = {sci(tr[-1])}\n最终验证损失 = {sci(va[-1])}",
            transform=ax.transAxes, ha="right", va="top", fontsize=8,
            bbox=dict(fc="#f4f7fb", ec="#c9d4e4", boxstyle="round,pad=0.4"))
    fig.tight_layout()
    _save(fig, "Figure4_loss_curves.png")


# ---------------------------------------------------------------------------
# Figure 5 — TMM vs MLP 光谱对比（插入宽 5.9 in）
# ---------------------------------------------------------------------------
def fig5_spectra():
    z = np.load(os.path.join(cfg.RESULT_DIR, "full_model_results.npz"))
    Yte, Yp = z["Yte"], z["Yte_pred"]
    wl = cfg.wavelength_grid()

    mse = np.mean((Yp - Yte) ** 2, axis=1)
    order = np.argsort(mse)
    picks = [order[2], order[len(order) // 2], order[-30]]

    fig, axes = plt.subplots(1, 3, figsize=(6.5, 2.4))
    for ax, si in zip(axes, picks):
        ax.plot(wl, Yte[si], color=C_TMM, lw=1.6, label="TMM 真实值")
        ax.plot(wl, Yp[si], color=C_MLP, lw=1.4, ls="--", label="MLP 预测值")
        ax.axvline(cfg.LAMBDA_TARGET, color="#9aa6b8", ls=":", lw=0.9)
        ax.set_title(f"样本 #{si}\nMSE = {sci(mse[si])}",
                     fontsize=8, color="#1a2233")
        ax.set_xlabel("波长 (nm)")
        ax.set_ylim(-0.02, 0.72)
        ax.set_xticks([400, 600, 800])
        ax.tick_params(labelsize=8)
    axes[0].set_ylabel("反射率 R")
    axes[0].legend(fontsize=7, loc="upper right")
    fig.suptitle("测试集上 TMM 真实光谱与 MLP 预测光谱对比",
                 fontsize=10, fontweight="bold", color="#1a2233", y=1.04)
    fig.tight_layout()
    _save(fig, "Figure5_spectra_comparison.png")

    info = [{"index": int(si), "mse": float(mse[si])} for si in picks]
    with open(os.path.join(cfg.RESULT_DIR, "fig5_samples.json"), "w",
              encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)
    return info


# ---------------------------------------------------------------------------
# Figure 6 — 训练数据量影响（插入宽 5.9 in）
# ---------------------------------------------------------------------------
def fig6_training_size():
    with open(os.path.join(cfg.RESULT_DIR, "training_size_results.json"),
              encoding="utf-8") as f:
        res = json.load(f)
    n = [r["n_train"] for r in res]
    mse = [r["test_mse"] for r in res]
    mae = [r["test_mae"] for r in res]

    fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.7))

    ax = axes[0]
    ax.plot(n, mse, "o-", color=C_TRAIN, lw=1.6, ms=5)
    ax.set_xlabel("训练样本数")
    ax.set_ylabel("测试集 MSE")
    ax.set_title("(a) 测试集均方误差与训练数据量的关系", fontsize=9, color="#1a2233")
    ax.set_xticks(n)
    ax.tick_params(labelsize=8)
    for xi, yi in zip(n, mse):
        ax.annotate(sci(yi), (xi, yi), textcoords="offset points",
                    xytext=(0, 7), ha="center", fontsize=7)

    ax = axes[1]
    ax.plot(n, mae, "s-", color="#6fb98a", lw=1.6, ms=5)
    ax.set_xlabel("训练样本数")
    ax.set_ylabel("测试集 MAE")
    ax.set_title("(b) 测试集平均绝对误差与训练数据量的关系", fontsize=9, color="#1a2233")
    ax.set_xticks(n)
    ax.tick_params(labelsize=8)
    for xi, yi in zip(n, mae):
        ax.annotate(f"{yi:.4f}", (xi, yi), textcoords="offset points",
                    xytext=(0, 7), ha="center", fontsize=7)

    fig.suptitle("训练数据量对预测误差的影响",
                 fontsize=10, fontweight="bold", color="#1a2233", y=1.04)
    fig.tight_layout()
    _save(fig, "Figure6_training_size.png")


# ---------------------------------------------------------------------------
# Figure 7 — 选中设计：MLP 预测 vs TMM 验证（插入宽 5.9 in）
# ---------------------------------------------------------------------------
def fig7_design():
    z = np.load(os.path.join(cfg.RESULT_DIR, "best_design.npz"))
    wl, mlp, tmm = z["wavelengths"], z["mlp_spectrum"], z["tmm_spectrum"]
    d = z["d"]
    ti = cfg.target_index()

    fig, ax = plt.subplots(figsize=(6.1, 3.85))
    ax.plot(wl, tmm, color=C_TMM, lw=2.0, label="TMM 验证值")
    ax.plot(wl, mlp, color=C_MLP, lw=1.6, ls="--", label="MLP 预测值")
    ax.axvline(cfg.LAMBDA_TARGET, color="#9aa6b8", ls=":", lw=1.1)
    ax.scatter([cfg.LAMBDA_TARGET], [tmm[ti]], color=C_TMM, zorder=5, s=38,
               edgecolor="white", linewidth=1.0)
    ax.scatter([cfg.LAMBDA_TARGET], [mlp[ti]], color=C_MLP, zorder=5, s=38,
               marker="D", edgecolor="white", linewidth=1.0)
    ax.annotate(f"λtarget = {cfg.LAMBDA_TARGET} nm\nTMM R = {tmm[ti]:.4f}\nMLP R = {mlp[ti]:.4f}",
                xy=(cfg.LAMBDA_TARGET, tmm[ti]),
                xytext=(cfg.LAMBDA_TARGET + 55, max(0.10, tmm[ti] - 0.32)),
                fontsize=8, color="#1a2233",
                arrowprops=dict(arrowstyle="->", color=C_GREY, lw=1.0),
                bbox=dict(fc="#f4f7fb", ec="#c9d4e4", boxstyle="round,pad=0.4"))
    ax.set_xlabel("波长 (nm)")
    ax.set_ylabel("反射率 R")
    ax.set_ylim(-0.02, max(0.75, tmm.max() + 0.08))
    ax.set_xticks([400, 500, 600, 700, 800])
    ax.legend(fontsize=8.5, loc="upper left")
    ax.set_title(f"最优设计（d = {', '.join(f'{x:.2f}' for x in d)} nm）\n"
                 "MLP 预测谱与 TMM 验证谱",
                 fontsize=9.5, fontweight="bold", color="#1a2233")
    fig.tight_layout()
    _save(fig, "Figure7_design.png")


# ---------------------------------------------------------------------------
# Figure 8 — 失败案例（插入宽 5.9 in）
# ---------------------------------------------------------------------------
def fig8_failure():
    z = np.load(os.path.join(cfg.RESULT_DIR, "full_model_results.npz"))
    Yte, Yp = z["Yte"], z["Yte_pred"]
    Xte = z["Xte"]
    e = np.mean((Yp - Yte) ** 2, axis=1)
    wi = int(np.argmax(e))
    wl = cfg.wavelength_grid()

    fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.6))
    ax = axes[0]
    ax.plot(wl, Yte[wi], color=C_TMM, lw=1.7, label="TMM 真实值")
    ax.plot(wl, Yp[wi], color=C_MLP, lw=1.5, ls="--", label="MLP 预测值")
    ax.fill_between(wl, Yte[wi], Yp[wi], color="#c0392b", alpha=0.12)
    ax.axvline(cfg.LAMBDA_TARGET, color="#9aa6b8", ls=":", lw=1.0)
    ax.set_xlabel("波长 (nm)")
    ax.set_ylabel("反射率 R")
    ax.legend(fontsize=7.5, loc="upper right")
    ax.set_title(f"(a) 失败案例：样本 #{wi}\nMSE = {sci(e[wi])}",
                 fontsize=8.5, color="#1a2233", fontweight="bold")
    ax.set_xticks([400, 600, 800])
    ax.tick_params(labelsize=8)

    ax = axes[1]
    ax.hist(np.log10(e + 1e-12), bins=40, color="#c9dbf2",
            edgecolor="#7ba3d4", lw=0.7)
    ax.axvline(np.log10(e[wi] + 1e-12), color="#c0392b", lw=1.8,
               label="最差样本")
    ax.axvline(np.log10(np.median(e) + 1e-12), color="#e08a1e", lw=1.8,
               ls="--", label="中位数")
    ax.set_xlabel("单样本 MSE 的常用对数（lg）")
    ax.set_ylabel("样本数")
    ax.legend(fontsize=7.5)
    ax.set_title("(b) 测试集逐样本误差分布", fontsize=8.5,
                 color="#1a2233", fontweight="bold")
    ax.tick_params(labelsize=8)

    fig.suptitle("代表性失败案例及其在误差分布中的位置",
                 fontsize=10, fontweight="bold", color="#1a2233", y=1.05)
    fig.tight_layout()
    _save(fig, "Figure8_failure_case.png")

    info = {"worst_index": wi, "worst_mse": float(e[wi]),
            "worst_d": [float(x) for x in Xte[wi]],
            "median_mse": float(np.median(e))}
    with open(os.path.join(cfg.RESULT_DIR, "failure_case.json"), "w",
              encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)
    return info


def main():
    print("开始生成图表...")
    fig1_workflow()
    fig2_physical_model()
    fig3_mlp_arch()
    fig4_loss()
    info5 = fig5_spectra()
    fig6_training_size()
    fig7_design()
    info8 = fig8_failure()
    print("图5 代表样本: " + ", ".join(
        f"#{s['index']} (MSE={s['mse']:.3e})" for s in info5))
    print(f"失败案例: sample #{info8['worst_index']}, "
          f"d = {[round(x, 2) for x in info8['worst_d']]} nm, "
          f"MSE = {info8['worst_mse']:.3e}")
    print("全部图表生成完毕。")


if __name__ == "__main__":
    main()
