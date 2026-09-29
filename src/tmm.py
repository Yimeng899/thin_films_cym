"""
tmm.py — 传输矩阵法 (Transfer Matrix Method) 计算多层介质膜反射率

物理模型
--------
膜系：Air / H / L / H / L / Glass，共 4 层
条件：正入射 (θ = 0)、无吸收 (k = 0)、忽略色散 (n 与 λ 无关)

对第 i 层，相位厚度
    δ_i = 2π n_i d_i / λ

单层的特征矩阵 (正入射, TE/TM 退化)
    M_i = [ cos δ_i          (i/n_i) sin δ_i ]
          [ i n_i sin δ_i    cos δ_i         ]

整个膜系的特征矩阵为各层矩阵连乘
    M = M_1 · M_2 · M_3 · M_4 = [ m11  m12 ]
                                [ m21  m22 ]

入射介质 (空气, n0) 到基底 (玻璃, ns) 的反射系数
    r = ( m11·n0 + m12·ns·n0 − m21 − m22·ns )
        / ( m11·n0 + m12·ns·n0 + m21 + m22·ns )

反射率
    R = |r|^2

说明：上式采用光从 n0 入射、经膜系到达 ns 的通用形式。下文实现中直接
用 (η0, ηs) 写成标准 Abeles 形式，便于与教科书结果核对。
"""

import numpy as np


def layer_matrix(n, d, lam):
    """单层 2x2 特征矩阵 (正入射)。

    Parameters
    ----------
    n   : float 折射率
    d   : float 膜厚 (nm)
    lam : float 波长 (nm)
    """
    delta = 2.0 * np.pi * n * d / lam
    c, s = np.cos(delta), np.sin(delta)
    return np.array([[c, 1j * s / n],
                     [1j * n * s, c]], dtype=complex)


def reflectance_spectrum(thicknesses, indices, n_inc, n_sub, wavelengths):
    """计算多层膜在给定波长网格上的反射率谱。

    Parameters
    ----------
    thicknesses : array_like, shape (L,)
        各层膜厚 (nm)
    indices : array_like, shape (L,)
        各层折射率
    n_inc : float  入射介质折射率（空气 = 1.0）
    n_sub : float  基底折射率（玻璃 = 1.52）
    wavelengths : array_like, shape (M,)
        波长网格 (nm)

    Returns
    -------
    R : ndarray, shape (M,)
        反射率，取值 [0, 1]
    """
    thicknesses = np.asarray(thicknesses, dtype=float)
    indices = np.asarray(indices, dtype=float)
    wavelengths = np.asarray(wavelengths, dtype=float)

    R = np.empty(wavelengths.shape[0], dtype=float)

    for k, lam in enumerate(wavelengths):
        # 各层矩阵连乘（从前到后：空气侧 → 基底侧）
        M = np.eye(2, dtype=complex)
        for n_i, d_i in zip(indices, thicknesses):
            M = M @ layer_matrix(n_i, d_i, lam)

        m11, m12 = M[0, 0], M[0, 1]
        m21, m22 = M[1, 0], M[1, 1]

        # Abeles 形式：Y = (m21 + m22*ns) / (m11 + m12*ns)，r = (n0 - Y)/(n0 + Y)
        Y = (m21 + m22 * n_sub) / (m11 + m12 * n_sub)
        r = (n_inc - Y) / (n_inc + Y)
        R[k] = np.abs(r) ** 2

    return R


def reflectance_spectrum_batch(thicknesses_all, indices, n_inc, n_sub, wavelengths):
    """批量计算 (N, L) 膜厚组合 → (N, M) 反射率谱。"""
    thicknesses_all = np.asarray(thicknesses_all, dtype=float)
    out = np.empty((thicknesses_all.shape[0], len(wavelengths)), dtype=float)
    for i, t in enumerate(thicknesses_all):
        out[i] = reflectance_spectrum(t, indices, n_inc, n_sub, wavelengths)
    return out


def reflectance_at(thicknesses, indices, n_inc, n_sub, lam):
    """单波长反射率（标量）。"""
    return float(reflectance_spectrum(thicknesses, indices, n_inc, n_sub,
                                      np.array([float(lam)]))[0])


if __name__ == "__main__":
    # 自检：空气 / 单层 / 玻璃 的解析解对比
    import config as cfg

    # 测试 1：单层 SiO2 型膜 (n=1.45, d=100nm)
    # 测试 2：空气/玻璃 界面（无膜）应给出菲涅耳公式结果
    n0, ns = 1.0, 1.52
    R_fresnel = ((ns - n0) / (ns + n0)) ** 2
    R_tmm = reflectance_spectrum([], [], n0, ns, np.array([550.0]))[0]
    print(f"无膜菲涅耳 R = {R_fresnel:.6f}")
    print(f"TMM 计算结果  = {R_tmm:.6f}")
    assert abs(R_fresnel - R_tmm) < 1e-12, "无膜情形应与菲涅耳公式一致"

    # 测试 3：QW 增透膜 —— 单层 n=sqrt(n0*ns), d=λ/(4n) 时 R 应接近 0
    n_ar = np.sqrt(n0 * ns)
    lam = 550.0
    d_ar = lam / (4 * n_ar)
    R_ar = reflectance_spectrum([d_ar], [n_ar], n0, ns, np.array([lam]))[0]
    print(f"四分之一波长增透膜 R = {R_ar:.3e}")

    # 测试 4：4 层结构在 41 点网格上跑通
    wl = cfg.wavelength_grid()
    R = reflectance_spectrum([100, 100, 100, 100], cfg.LAYER_INDICES,
                             cfg.N_INC, cfg.N_S, wl)
    print(f"4层结构 R 范围: {R.min():.4f} – {R.max():.4f}, 点数={len(R)}")
    print("TMM 自检通过。")
