"""
model.py — MLP 代理模型

输入：4 个膜厚 [d1, d2, d3, d4]
输出：41 个波长采样点的反射率

结构：4 – 128 – 128 – 64 – 41
损失：MSE
"""

import torch
import torch.nn as nn

import config as cfg


class MLPSurrogate(nn.Module):
    """多层感知机代理模型：膜厚 → 反射光谱。"""

    def __init__(self, in_dim=4, out_dim=41, hidden_dims=None,
                 hidden_act="ReLU", out_act="Sigmoid"):
        super().__init__()
        hidden_dims = hidden_dims or cfg.HIDDEN_DIMS

        act_map = {
            "ReLU": nn.ReLU, "LeakyReLU": nn.LeakyReLU,
            "Tanh": nn.Tanh, "SiLU": nn.SiLU, "GELU": nn.GELU,
        }
        act_cls = act_map[hidden_act]

        layers = []
        prev = in_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), act_cls()]
            prev = h
        layers.append(nn.Linear(prev, out_dim))

        self.body = nn.Sequential(*layers)

        # 输出层：反射率落在 [0, 1]
        if out_act == "Sigmoid":
            self.out_act = nn.Sigmoid()
        elif out_act is None or out_act == "None":
            self.out_act = nn.Identity()
        else:
            raise ValueError(f"未知输出激活: {out_act}")

    def forward(self, x):
        return self.out_act(self.body(x))


def build_model(seed=None, device="cpu"):
    """按统一结构构建模型，并用 seed 初始化权重。"""
    seed = cfg.SEED if seed is None else seed
    torch.manual_seed(seed)
    model = MLPSurrogate(
        in_dim=cfg.N_LAYERS,
        out_dim=len(cfg.wavelength_grid()),
        hidden_dims=cfg.HIDDEN_DIMS,
        hidden_act=cfg.ACTIVATION,
        out_act=cfg.OUT_ACTIVATION,
    )
    return model.to(device)


if __name__ == "__main__":
    torch.manual_seed(cfg.SEED)
    m = build_model()
    n_params = sum(p.numel() for p in m.parameters())
    print(m)
    print(f"可训练参数量: {n_params:,}")
    x = torch.rand(8, 4)
    y = m(x)
    print(f"输入 {tuple(x.shape)} → 输出 {tuple(y.shape)}")
    print(f"输出范围: [{y.min():.4f}, {y.max():.4f}]")
