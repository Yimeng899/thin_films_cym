# AI4S Thin-Film Project
# 基于多层感知机（MLP）的多层介质薄膜光谱预测与辅助设计

《薄膜技术》AI4S 课程大作业 —— 个人项目

---

## 1. 个性化参数

| 参数 | 值 | 来源 |
|---|---|---|
| 学号 | 2023270171 | — |
| N（学号后两位） | 71 | — |
| **λtarget** | **540 nm** | 450 + 10 × (71 mod 31) = 450 + 90 |
| **seed** | **270171** | 学号后 6 位整数 |
| **design_seed** | **270172** | seed + 1 |

> 数据集生成、数据划分与 MLP 初始化全部由 `seed = 270171` 控制；
> 设计阶段候选膜厚由 `design_seed = 270172` 生成，以确保候选与训练数据不重复。

---

## 2. 统一物理设置

| 项目 | 设置 |
|---|---|
| 膜系结构 | Air / H / L / H / L / Glass（共 4 层） |
| 折射率 | nH = 2.30，nL = 1.45，ns = 1.52 |
| 入射条件 | 正入射、无吸收、忽略色散 |
| 膜厚范围 | 每层 40–180 nm |
| 波长范围 | 400–800 nm，步长 10 nm，共 41 点 |
| 数据集 | 5000 组：4000 训练 + 500 验证 + 500 测试 |
| MLP 结构 | 4 – 128 – 128 – 64 – 41 |

---

## 3. 环境配置

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux / macOS
source venv/bin/activate

pip install -r requirements.txt
```

依赖：`numpy`、`torch`、`matplotlib`。

---

## 4. 目录结构

```
AI4S_ThinFilm/
├── README.md
├── requirements.txt
├── src/
│   ├── config.py       # 全局配置与个性化参数（λtarget / seed / design_seed）
│   ├── tmm.py           # 传输矩阵法（TMM）反射率计算
│   ├── dataset.py       # 数据集生成与固定划分
│   ├── model.py         # MLP 代理模型定义
│   ├── train.py         # 模型训练 + 数据量影响实验
│   ├── design.py        # MLP 辅助筛选 + TMM 验证
│   └── figures.py       # 论文全部图表生成
├── data/                # 生成的 dataset.npz
├── results/             # 训练结果、设计结果（json / npz / 模型权重）
└── figures/             # 论文 Figure 1–8（PNG, 300 dpi）
```

---

## 5. 复现步骤

按顺序执行以下命令（在 `src/` 目录下运行）：

```bash
cd src

# Step 0（可选）自检 TMM 实现：菲涅耳公式与四分之一波长增透膜
python tmm.py

# Step 1  生成数据集：5000 组膜厚-光谱，固定划分 4000/500/500
python dataset.py

# Step 2  训练 MLP（4000 样本）+ 数据量实验（500/1000/2000/4000）
#         产出：results/summary.json, full_model_results.npz,
#               training_size_results.json, models/mlp_full.pt
python train.py

# Step 3  MLP 辅助设计：10000 组新候选 → Top10 → TMM 验证 → Top5
#         产出：results/top_designs.json, best_design.npz
python design.py

# Step 4  生成论文全部图表 Figure 1–8
python figures.py
```

运行完成后，`figures/` 目录下应出现 8 张 PNG，`results/` 下出现对应数值文件，
即为论文中全部图表的数据来源。

---

## 6. 主要结果对照

论文中的图与数值均由 `results/` 目录下的文件生成，对应关系如下：

| 论文图/表 | 数据来源 | 生成脚本 |
|---|---|---|
| Figure 1 总体工作流 | 示意图 | `figures.py` |
| Figure 2 物理模型 + TMM 数据 | `data/dataset.npz` | `figures.py` |
| Figure 3 MLP 结构 | 模型定义 | `figures.py` |
| Figure 4 训练/验证损失 | `results/full_model_results.npz` | `figures.py` |
| Figure 5 光谱预测对比 | `results/full_model_results.npz` | `figures.py` |
| Figure 6 数据量影响 | `results/training_size_results.json` | `figures.py` |
| Figure 7 设计方案光谱 | `results/best_design.npz` | `figures.py` |
| Figure 8 失败案例 | `results/full_model_results.npz` | `figures.py` |
| Table 1 Top5 设计 | `results/top_designs.json` | `design.py` → `figures.py` |

---

## 7. 方法要点

- **TMM**：正入射下每层特征矩阵 `M_i = [[cos δ_i, i sin δ_i / n_i], [i n_i sin δ_i, cos δ_i]]`，
  `δ_i = 2π n_i d_i / λ`；膜系矩阵为各层连乘，反射率 `R = |r|²`。
- **数据划分**：一次固定随机排列（`RandomState(seed).permutation`）后切分，
  后续所有实验共用同一划分，保证可比性。
- **MLP**：输入 4 维膜厚（线性归一化 (d−40)/140 → [0,1]），输出 41 点反射率；ReLU 隐藏激活、
  Sigmoid 输出（R ∈ [0,1]）、MSE 损失、Adam 优化器（初始学习率 1×10⁻³，余弦退火至 0）、
  batch size 64、训练 4000 epoch。
- **辅助设计**：MLP 仅用于快速筛选，最终设计一律由 TMM 复算确认。
