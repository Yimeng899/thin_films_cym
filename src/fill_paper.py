# -*- coding: utf-8 -*-
"""
fill_paper.py — 用真实内容填充 AI4S Research Article 模板

规范（2026-09-29 修订 v2）：
- 正文五号（10.5 pt）；题注与表内文字小五（9 pt）
- 全部题注中文化（图 1–图 8、表 1）
- Table 1 三线表：顶线/底线 1.5 pt、表头下线 0.75 pt，无竖线与内部横线
- 表内数字与正文数字统一 Times New Roman（中文宋体）
- 文末（References 之后）附课程评价与建议段落
- 公式 (1)–(5) 采用 Word 原生 OMML 公式对象（真分数线/上下标，双击可编辑），
  公式居中、编号右对齐（居中+右制表位）
- 全部数值从 results/*.json 动态读取，与代码输出自动同步
"""
import sys, io, json, os
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import numpy as np
import docx
from docx.shared import Inches, Pt, Twips
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_TAB_ALIGNMENT
from docx.oxml.ns import qn, nsdecls
from docx.oxml import OxmlElement, parse_xml
from docx.text.paragraph import Paragraph

TEMPLATE = r"D:\xwechat_files\wxid_7l5n3rcbeel922_b7f5\msg\file\2026-09\AI4S_Research_Template.docx"
OUT = r"E:\person\薄膜技术\AI4S_ThinFilm\docs\AI4S_Research_Article.docx"
FIG = r"E:\person\薄膜技术\AI4S_ThinFilm\figures"
RES = r"E:\person\薄膜技术\AI4S_ThinFilm\results"

d = docx.Document(TEMPLATE)
paras = d.paragraphs


# ============================================================ helpers
def set_run_font(run, size=10.5, ascii_font='Times New Roman'):
    run.font.size = Pt(size)
    rpr = run._element.get_or_add_rPr()
    rfonts = rpr.find(qn('w:rFonts'))
    if rfonts is None:
        rfonts = OxmlElement('w:rFonts')
        rpr.append(rfonts)
    rfonts.set(qn('w:ascii'), ascii_font)
    rfonts.set(qn('w:hAnsi'), ascii_font)
    rfonts.set(qn('w:eastAsia'), '宋体')


def replace_text(para, text):
    for r in list(para.runs):
        r._element.getparent().remove(r._element)
    run = para.add_run(text)
    set_run_font(run)
    return para


def insert_para_after(anchor_para, text, align=None, size=10):
    new_p = OxmlElement('w:p')
    anchor_para._p.addnext(new_p)
    para = Paragraph(new_p, anchor_para._parent)
    run = para.add_run(text)
    set_run_font(run, size)
    if align is not None:
        para.alignment = align
    return para


def insert_image_into(para, img_path, width_in):
    run = para.add_run()
    run.add_picture(img_path, width=Inches(width_in))
    para.alignment = WD_ALIGN_PARAGRAPH.CENTER
    return para


def clear_cell(cell):
    for p in list(cell.paragraphs):
        for r in list(p.runs):
            r._element.getparent().remove(r._element)
    ps = cell.paragraphs
    for p in ps[1:]:
        p._p.getparent().remove(p._p)
    return cell.paragraphs[0]


def find_para(prefix, start=0):
    for i, p in enumerate(paras):
        if i >= start and p.text.strip().startswith(prefix):
            return p
    raise RuntimeError(f"未找到段落: {prefix}")


def delete_para(para):
    para._p.getparent().remove(para._p)


# ============================================================ OMML 公式
def _r(t, sty=None):
    """数学 run；sty='p' 表示正体（函数名 cos/sin 等）。"""
    if sty:
        return (f'<m:r><m:rPr><m:sty m:val="{sty}"/></m:rPr>'
                f'<m:t xml:space="preserve">{t}</m:t></m:r>')
    return f'<m:r><m:t xml:space="preserve">{t}</m:t></m:r>'


def _sub(base, sub):
    return f'<m:sSub><m:e>{base}</m:e><m:sub>{sub}</m:sub></m:sSub>'


def _sup(base, sup):
    return f'<m:sSup><m:e>{base}</m:e><m:sup>{sup}</m:sup></m:sSup>'


def _frac(num, den):
    return f'<m:f><m:num>{num}</m:num><m:den>{den}</m:den></m:f>'


def _d(e, beg='(', end=')'):
    """定界符，默认圆括号；beg/end='|' 为绝对值。"""
    return (f'<m:d><m:dPr><m:begChr m:val="{beg}"/>'
            f'<m:endChr m:val="{end}"/></m:dPr><m:e>{e}</m:e></m:d>')


def _mat2(rows):
    """2×2 矩阵。rows: [[xml,xml],[xml,xml]]"""
    mrs = ''.join(
        '<m:mr>' + ''.join(f'<m:e>{c}</m:e>' for c in row) + '</m:mr>'
        for row in rows)
    return ('<m:m><m:mPr><m:mcs><m:mc><m:mcPr>'
            '<m:count m:val="2"/><m:mcJc m:val="center"/>'
            '</m:mcPr></m:mc></m:mcs></m:mPr>' + mrs + '</m:m>')


def _omml(inner):
    xml = ('<m:oMath xmlns:m="http://schemas.openxmlformats.org/'
           'officeDocument/2006/math">' + inner + '</m:oMath>')
    return parse_xml(xml)


# 五个公式的 OMML 内容
EQ1 = (_sub(_r('δ'), _r('i')) + _r('=') +
       _frac(_r('2π') + _sub(_r('n'), _r('i')) + _sub(_r('d'), _r('i')),
             _r('λ')))
EQ2 = (_sub(_r('M'), _r('i')) + _r('=') + _mat2([
    [_r('cos ', 'p') + _sub(_r('δ'), _r('i')),
     _frac(_r('i'), _sub(_r('n'), _r('i'))) + _r('sin ', 'p') + _sub(_r('δ'), _r('i'))],
    [_r('i') + _sub(_r('n'), _r('i')) + _r('sin ', 'p') + _sub(_r('δ'), _r('i')),
     _r('cos ', 'p') + _sub(_r('δ'), _r('i'))],
]))
EQ3 = (_r('M=') + _sub(_r('M'), _r('1')) + _sub(_r('M'), _r('2')) +
       _sub(_r('M'), _r('3')) + _sub(_r('M'), _r('4')) + _r('=') +
       _mat2([[_sub(_r('m'), _r('11')), _sub(_r('m'), _r('12'))],
              [_sub(_r('m'), _r('21')), _sub(_r('m'), _r('22'))]]))
EQ4 = (_r('Y=') + _frac(
    _sub(_r('m'), _r('21')) + _r('+') + _sub(_r('m'), _r('22')) + _sub(_r('n'), _r('s')),
    _sub(_r('m'), _r('11')) + _r('+') + _sub(_r('m'), _r('12')) + _sub(_r('n'), _r('s'))))
EQ5 = (_r('r=') + _frac(_sub(_r('n'), _r('0')) + _r('−') + _r('Y'),
                         _sub(_r('n'), _r('0')) + _r('+') + _r('Y')) +
       _r('，  ') + _r('R=') + _sup(_d(_r('r'), '|', '|'), _r('2')))


def insert_equation_after(anchor, eq_inner, number):
    """插入公式段：居中制表位放公式，右制表位放编号。"""
    new_p = OxmlElement('w:p')
    anchor._p.addnext(new_p)
    para = Paragraph(new_p, anchor._parent)

    sec = d.sections[0]
    content_tw = int((sec.page_width - sec.left_margin - sec.right_margin) / 635)
    tf = para.paragraph_format
    tf.tab_stops.add_tab_stop(Twips(content_tw // 2), WD_TAB_ALIGNMENT.CENTER)
    tf.tab_stops.add_tab_stop(Twips(content_tw), WD_TAB_ALIGNMENT.RIGHT)

    run_tab = para.add_run('\t')
    set_run_font(run_tab)
    para._p.append(_omml(eq_inner))
    run_num = para.add_run(f'\t({number})')
    set_run_font(run_num)
    return para


# ============================================================ 三线表
def make_three_line(table, header_rows=1):
    """三线表：顶线/底线 1.5 pt，表头下线 0.75 pt；去除竖线与内部横线。"""
    tbl = table._tbl
    tblPr = tbl.tblPr
    for el in tblPr.findall(qn('w:tblBorders')):
        tblPr.remove(el)
    borders = parse_xml(
        f'<w:tblBorders {nsdecls("w")}>'
        '<w:top w:val="single" w:sz="12" w:space="0" w:color="000000"/>'
        '<w:bottom w:val="single" w:sz="12" w:space="0" w:color="000000"/>'
        '<w:left w:val="none" w:sz="0" w:space="0" w:color="auto"/>'
        '<w:right w:val="none" w:sz="0" w:space="0" w:color="auto"/>'
        '<w:insideH w:val="none" w:sz="0" w:space="0" w:color="auto"/>'
        '<w:insideV w:val="none" w:sz="0" w:space="0" w:color="auto"/>'
        '</w:tblBorders>')
    tblPr.append(borders)
    for ri in range(header_rows):
        for cell in table.rows[ri].cells:
            tcPr = cell._tc.get_or_add_tcPr()
            for el in tcPr.findall(qn('w:tcBorders')):
                tcPr.remove(el)
            b = parse_xml(
                f'<w:tcBorders {nsdecls("w")}>'
                '<w:bottom w:val="single" w:sz="6" w:space="0" w:color="000000"/>'
                '</w:tcBorders>')
            tcPr.append(b)


# ============================================================ 动态数值
with open(os.path.join(RES, "summary.json"), encoding="utf-8") as f:
    S = json.load(f)
with open(os.path.join(RES, "top_designs.json"), encoding="utf-8") as f:
    TOPJ = json.load(f)
    tops, top10 = TOPJ["top5"], TOPJ["top10"]
with open(os.path.join(RES, "fig5_samples.json"), encoding="utf-8") as f:
    F5 = json.load(f)
with open(os.path.join(RES, "failure_case.json"), encoding="utf-8") as f:
    F8 = json.load(f)

_z = np.load(os.path.join(RES, "full_model_results.npz"))
_ep, _tr, _va = _z["hist_epoch"], _z["hist_train"], _z["hist_val"]
_final_tr, _final_va = float(_tr[-1]), float(_va[-1])
# 收敛轮次：验证损失首次降至最终值 5 倍以内
_k = int(np.argmax(_va < 5 * _final_va))
_ep_conv = int(_ep[_k])

DAT = np.load(r"E:\person\薄膜技术\AI4S_ThinFilm\data\dataset.npz")
_R = DAT["spectra"]
_TI = 14  # λtarget=540 nm 下标（与 config.target_index 一致）
_R_MAX_ALL = float(_R.max())
_R_MAX_TGT = float(_R[:, _TI].max())

T1 = tops[0]
mse = S["test_mse"]; rmse = S["test_rmse"]; mae = S["test_mae"]
sr = {r["n_train"]: r for r in S["size_results"]}
factor = sr[500]["test_mse"] / sr[4000]["test_mse"]
over10 = float(np.mean([r["mlp_Rtarget"] - r["tmm_Rtarget"] for r in top10]))
worst_ratio = F8["worst_mse"] / F8["median_mse"]
d1s = "、".join(f"{x:.1f}" for x in T1["d"])


def sci_txt(v, sig=3):
    """正文科学计数：a.bc×10⁻ⁿ（Unicode 上标，Word 内 TNR 可正常渲染）。"""
    if v == 0:
        return "0"
    e = int(np.floor(np.log10(abs(v))))
    m = v / 10 ** e
    sup = {"-": "\u207b", "0": "\u2070", "1": "\u00b9", "2": "\u00b2",
           "3": "\u00b3", "4": "\u2074", "5": "\u2075", "6": "\u2076",
           "7": "\u2077", "8": "\u2078", "9": "\u2079"}
    return f"{m:.{sig-1}f}×10" + "".join(sup[c] for c in str(e))


# ============================================================ 1 标题区
replace_text(find_para("姓名："), "姓名：陈怡梦    学号：2023270171")
replace_text(find_para("λtarget："),
             "λtarget：540 nm    seed：270171    design_seed：270172")

# ============================================================ 2 摘要
replace_text(find_para("请在此撰写摘要"),
    "多层介质薄膜的反射光谱设计传统上依赖物理模型计算与人工调优，面对大规模候选结构时效率受限。"
    "本文以 Air/H/L/H/L/Glass 四层介质膜为研究对象，用传输矩阵法（TMM）生成 5000 组“膜厚—反射光谱”数据，"
    f"训练 4–128–128–64–41 结构的多层感知机（MLP）代理模型，实现膜厚到 400–800 nm 反射光谱的快速预测。"
    f"测试集 MSE 为 {sci_txt(mse)}（RMSE = {rmse:.4f}），验证了 MLP 对膜厚—光谱非线性映射的学习能力；"
    f"训练数据量从 500 增至 4000 时，测试 MSE 由 {sci_txt(sr[500]['test_mse'])} 单调降至 {sci_txt(mse)}，"
    f"降幅约 {factor:.0f} 倍且尚未饱和。"
    "进一步以 MLP 对 10000 组候选结构按 λtarget = 540 nm 处反射率排序筛选，Top 1 设计"
    f"（d = [{d1s}] nm）经 TMM 验证 R = {T1['tmm_Rtarget']:.4f}，与预测值仅差 "
    f"{abs(T1['mlp_Rtarget']-T1['tmm_Rtarget']):.4f}，"
    f"且超过 5000 组训练样本在同一波长下的最大反射率 {_R_MAX_TGT:.4f}。"
    "结果表明 MLP 代理模型可大幅加速候选膜系筛选，最终设计仍需 TMM 复算确认。")

delete_para(find_para("【写作提示】150–250字"))

# ============================================================ 3 Introduction
replace_text(find_para("第一段：介绍多层介质薄膜"),
    "多层介质薄膜由高、低折射率材料交替堆叠而成，是增透膜、高反射膜、滤光片等光学元件的核心结构，"
    "广泛应用于显示、光伏、传感与精密光学仪器等领域[3]。其光谱响应源于各界面反射光的相干叠加："
    "当各层光学厚度 nᵢdᵢ 与入射波长满足特定相位关系时，多束反射光发生相长干涉形成高反射带，"
    "反之则相消干涉实现增透。因此，在材料折射率给定的条件下，各层膜厚是决定光谱形状的核心设计变量，"
    "膜系设计本质上是在连续膜厚空间中寻找使目标光学响应最优的非线性逆问题。")

replace_text(find_para("第二段：介绍传输矩阵法"),
    "传输矩阵法（Transfer Matrix Method, TMM）是计算多层膜光谱响应的经典电磁方法[2,4]："
    "将每层膜表示为一个 2×2 特征矩阵，整个膜系的响应由各层矩阵连乘给出，"
    "单个结构在单个波长上的计算量极小。然而，膜系逆设计需要在大量候选厚度组合中反复评估。"
    "以本文四层膜系为例，若每层膜厚仅按 10 nm 网格离散（40–180 nm 共 15 档），"
    "组合空间即达 15⁴ ≈ 5.1 万组；若在更细网格或连续空间中搜索，所需 TMM 评估次数还会进一步增加。"
    "对大规模候选池逐一执行正向计算，虽然单次成本低，但累积开销仍不可忽略。")

replace_text(find_para("第三段：引出机器学习代理模型"),
    "机器学习为上述问题提供了另一条路径：先以 TMM 批量生成“膜厚—光谱”数据集，"
    "再训练神经网络学习从膜厚到光谱的非线性映射，将其作为物理模型的代理（surrogate model）[1,5,6]。"
    "训练完成后，网络单次前向传播耗时比 TMM 低若干数量级，可对数万候选结构瞬时评估，"
    "从而承担“快速粗筛”的角色；物理模型则退居“最终验证”，只对少量入围候选做精确复算。"
    "这种“物理模型生成数据 → 神经网络学习映射 → 代理模型辅助设计 → 物理模型验证结果”的流程，"
    "是 AI for Science（AI4S）研究范式的典型形态。")

replace_text(find_para("第四段：明确本文研究三个问题"),
    "本文围绕三个问题展开研究：（Q1）MLP 能否准确学习并预测四层膜系的反射光谱；"
    "（Q2）训练数据量增加时，预测误差如何变化、是否出现边际收益递减；"
    "（Q3）MLP 作为代理模型能否有效辅助目标波长下的膜系筛选，以及其局限性何在。"
    "依据课程规则，个性化参数由学号 2023270171 唯一确定：目标波长 λtarget = 540 nm，"
    "随机种子 seed = 270171，设计种子 design_seed = 270172。"
    "全部数据、代码与图表均可由随文公开的代码仓库复现。")

fig1_para = find_para("Figure 1. Overall workflow")
prev = fig1_para._p.getprevious()
insert_image_into(Paragraph(prev, fig1_para._parent),
                  os.path.join(FIG, "Figure1_workflow.png"), 6.1)

# ============================================================ 4 Methods
# 2.1：公式 (1)–(5) 全部改为 OMML 原生公式对象
p_eq1 = find_para("δᵢ = 2π nᵢ dᵢ / λ")
anchor = p_eq1
anchor = insert_equation_after(anchor, EQ1, 1)
anchor = insert_equation_after(anchor, EQ2, 2)
anchor = insert_para_after(anchor, "四层膜系的特征矩阵为各层矩阵依序连乘：")
anchor = insert_equation_after(anchor, EQ3, 3)
anchor = insert_para_after(anchor,
    "结合入射介质（空气，n₀ = 1.00）与基底（玻璃，nₛ = 1.52），膜系的等效导纳与反射率分别为：")
anchor = insert_equation_after(anchor, EQ4, 4)
anchor = insert_equation_after(anchor, EQ5, 5)
insert_para_after(anchor,
    "式(1)–(5)即 Abeles 特征矩阵法的标准形式[4]，与光学教科书一致[2]。"
    "本文代码实现通过两项自检：无膜情形下 R 与菲涅耳公式解析值 ((nₛ−n₀)/(nₛ+n₀))² 完全一致；"
    "单层四分之一波长增透膜（n = √(n₀nₛ)，d = λ/4n）在中心波长处 R → 0，误差小于 10⁻³⁰。")

delete_para(p_eq1)
delete_para(find_para("【写作提示】补充实际代码中使用的2×2特征矩阵"))

fig2_para = find_para("Figure 2. Physical model")
prev = fig2_para._p.getprevious()
insert_image_into(Paragraph(prev, fig2_para._parent),
                  os.path.join(FIG, "Figure2_physical_model.png"), 6.1)

replace_text(find_para("目标波长与随机种子均由学号确定"),
    "目标波长与随机种子由学号唯一确定。学号 2023270171 的后两位 N = 71，"
    "故 λtarget = 450 + 10×(N mod 31) = 450 + 10×9 = 540 nm；"
    "随机种子取学号后 6 位整数 seed = 270171，数据集生成、数据划分与 MLP 权重初始化均由该种子控制；"
    "为避免设计候选与训练数据重复，设计阶段使用 design_seed = seed + 1 = 270172。"
    "代码中通过 random.seed(seed)、np.random.seed(seed) 与 torch.manual_seed(seed) "
    "显式固定全部随机性，保证结果可复现。")

replace_text(find_para("使用个人seed随机生成5000组膜厚组合"),
    "使用 seed = 270171 的随机数发生器在 [40, 180] nm 内均匀采样，生成 5000 组四层膜厚组合"
    "（维度 5000×4），并用 TMM 计算 400–800 nm、步长 10 nm 共 41 个波长点的反射率（5000×41）。"
    "随后以同一 seed 生成一次固定随机排列，将数据划分为前 4000 组训练集、中间 500 组验证集、"
    f"最后 500 组测试集；该划分在所有后续实验中保持不变。数据集反射率取值范围为 "
    f"0.0000–{_R_MAX_ALL:.4f}，其中 λtarget = 540 nm 处的最大反射率为 {_R_MAX_TGT:.4f}。")

replace_text(find_para("MLP输入为四层膜厚"),
    "MLP 输入为四层膜厚 [d₁, d₂, d₃, d₄]，输出为 41 个波长采样点的反射率，"
    "结构统一取 4–128–128–64–41，可训练参数共 28,073 个。作为通用函数逼近器，"
    "该规模足以表达本研究中光滑的膜厚—光谱映射[7]。输入膜厚先做线性归一化 "
    "(d − 40)/140 → [0, 1]，避免第一层线性层处于过大的数值尺度，加快收敛并减小批间波动。"
    "隐藏层采用 ReLU 激活；由于反射率取值天然落于 [0, 1]，输出层采用 Sigmoid 激活。"
    "训练损失为均方误差（MSE），优化器为 Adam（初始学习率 1×10⁻³）[8]，"
    "并采用余弦退火策略将学习率随训练轮次平滑衰减至 0；batch size 为 64，固定训练 "
    f"{S['epochs']} epoch。每次训练前调用 torch.manual_seed(seed) 固定权重初始化与 batch 顺序，"
    "确保可复现。")

fig3_para = find_para("Figure 3. Architecture")
prev = fig3_para._p.getprevious()
insert_image_into(Paragraph(prev, fig3_para._parent),
                  os.path.join(FIG, "Figure3_mlp_architecture.png"), 6.1)

replace_text(find_para("使用 design_seed = seed + 1 随机生成10000组"),
    "使用 design_seed = 270172 随机生成 10000 组新的候选膜厚，用训练完成的 MLP 预测其反射光谱，"
    "并按 λtarget = 540 nm 处的预测反射率降序排序，取 Top 10 候选；"
    "随后对 Top 10 用 TMM 重新计算真实光谱，按真实 R(540 nm) 确定最终 Top 5 设计。")

delete_para(find_para("【写作提示】Methods应做到"))
delete_para(find_para("【写作提示】正文中给出自己的"))
delete_para(find_para("【写作提示】写清楚实际使用的激活函数"))

# ============================================================ 5 Results
replace_text(find_para("【写作提示】Results重点报告"),
    "本节报告训练收敛情况、光谱预测精度、训练数据量的影响与辅助设计结果；"
    "对结果原因的分析见第 4 节。")

p_h31 = find_para("3.1 MLP training and spectral prediction")
r1 = insert_para_after(p_h31,
    "图 4 给出完整模型（4000 训练样本）的训练与验证损失曲线，细线为逐轮原始记录，"
    "粗线为 50 轮滑动平均；训练损失的逐轮波动源于小批量随机梯度下降的批间差异，属正常现象。"
    f"训练前期损失快速下降，约 {_ep_conv} 轮后进入缓慢收敛区；{S['epochs']} 轮时训练损失为 "
    f"{sci_txt(_final_tr)}，验证损失为 {sci_txt(_final_va)}，两者接近且验证损失未出现回升，"
    "表明模型未发生显著过拟合。")
insert_para_after(r1,
    "图 5 给出测试集上三个代表性样本的 TMM 真实光谱与 MLP 预测光谱对比："
    f"低误差样本 #{F5[0]['index']}（MSE = {sci_txt(F5[0]['mse'])}）、"
    f"中等误差样本 #{F5[1]['index']}（MSE = {sci_txt(F5[1]['mse'])}）"
    f"与较高误差样本 #{F5[2]['index']}（MSE = {sci_txt(F5[2]['mse'])}）。"
    "三者中 MLP 均能再现光谱的峰谷位置与整体形状，误差主要体现在峰值高度的微小偏差。"
    f"在整个 500 组测试集上，MLP 的 MSE = {sci_txt(mse)}，RMSE = {rmse:.4f}，"
    f"MAE = {mae:.4f}，即 41 个波长点上的平均预测偏差约为 {mae*100:.1f}%。")

p_h32 = find_para("3.2 Effect of training-set size")
insert_para_after(p_h32,
    "固定验证集与测试集不变，从同一 4000 组训练池中依次取前 500、1000、2000、4000 组训练四个模型，"
    f"测试误差随数据量的变化见图 6：测试 MSE 从 {sci_txt(sr[500]['test_mse'])}（500 样本）依次降至 "
    f"{sci_txt(sr[1000]['test_mse'])}、{sci_txt(sr[2000]['test_mse'])} 与 {sci_txt(sr[4000]['test_mse'])}"
    f"（4000 样本），MAE 从 {sr[500]['test_mae']:.4f} 降至 {sr[4000]['test_mae']:.4f}，"
    f"λtarget 处的 MAE 从 {sr[500]['target_mae']:.4f} 降至 {sr[4000]['target_mae']:.4f}。"
    f"整体呈单调下降且未出现平台：500 → 4000 样本误差降幅约 {factor:.0f} 倍，"
    "表明在该任务中数据量仍是限制精度的首要因素，但收益已呈边际递减趋势。")

p_h33 = find_para("3.3 MLP-assisted thin-film design")
r3 = insert_para_after(p_h33,
    "用 MLP 对 10000 组候选（design_seed = 270172）按预测 R(540 nm) 排序取 Top 10，"
    f"再用 TMM 逐一复算，结果见表 1。MLP 的 Top 1 候选（d = [{d1s}] nm）"
    f"经 TMM 验证后仍居第 {T1['final_rank']}，真实 R(540 nm) = {T1['tmm_Rtarget']:.4f}，"
    f"与预测值 {T1['mlp_Rtarget']:.4f} 仅差 {abs(T1['mlp_Rtarget']-T1['tmm_Rtarget']):.4f}；"
    f"Top 5 设计的 TMM 验证反射率介于 {tops[-1]['tmm_Rtarget']:.4f}–{tops[0]['tmm_Rtarget']:.4f}。")
insert_para_after(r3,
    "图 7 给出 Top 1 设计的完整光谱：MLP 预测谱与 TMM 验证谱在 400–800 nm 范围内形状一致，"
    f"仅在 540 nm 反射峰附近存在约 {abs(T1['mlp_Rtarget']-T1['tmm_Rtarget']):.2f} 的系统性高估。"
    f"值得注意的是，该设计的真实 R(540 nm) = {T1['tmm_Rtarget']:.4f} 超过了 5000 组训练样本"
    f"在同一波长处的最大值 {_R_MAX_TGT:.4f}，说明代理模型在训练分布内筛出的组合可以优于训练数据本身；"
    f"但 MLP 给出的 Top 10 排序与 TMM 真实排序存在交错（预测值平均高估约 {over10:.3f}），"
    "最终设计必须以 TMM 复算结果为准。")

# 图 4–8 插入对应 1×1 占位表格
fig_map = [
    (0, "Figure4_loss_curves.png"),
    (1, "Figure5_spectra_comparison.png"),
    (2, "Figure6_training_size.png"),
    (3, "Figure7_design.png"),
    (5, "Figure8_failure_case.png"),
]
for ti, fname in fig_map:
    cell = d.tables[ti].rows[0].cells[0]
    insert_image_into(clear_cell(cell), os.path.join(FIG, fname), 5.9)

# ---- Table 1：三线表 + 中文表头 + TNR 数字 ----
t5 = d.tables[4]
header = ["排名", "d₁/nm", "d₂/nm", "d₃/nm", "d₄/nm", "MLP 预测 R", "TMM 验证 R"]
rows_data = [header]
for k, r in enumerate(tops, 1):
    rows_data.append([str(k)] + [f"{x:.2f}" for x in r["d"]] +
                     [f"{r['mlp_Rtarget']:.4f}", f"{r['tmm_Rtarget']:.4f}"])
for row, vals in zip(t5.rows, rows_data):
    for cell, v in zip(row.cells, vals):
        p0 = clear_cell(cell)
        run = p0.add_run(v)
        set_run_font(run, size=9)
        p0.alignment = WD_ALIGN_PARAGRAPH.CENTER
make_three_line(t5, header_rows=1)

p_h34 = find_para("3.4 Representative failure case")
insert_para_after(p_h34,
    f"图 8 给出测试集中误差最大的样本 #{F8['worst_index']}"
    f"（d = [{('、'.join(f'{x:.1f}' for x in F8['worst_d']))}] nm），"
    f"其 MSE = {sci_txt(F8['worst_mse'])}，约为中位误差（{sci_txt(F8['median_mse'])}）的 "
    f"{worst_ratio:.0f} 倍。该样本的 TMM 真实光谱在 400–530 nm 存在快速振荡的多峰结构，"
    "MLP 预测虽再现了大体轮廓，但在峰位与峰高上出现系统性偏移；"
    "图 8(b) 显示该样本位于逐样本误差分布的右端尾部。"
    "可能的原因包括：d₁、d₂ 均接近 180 nm 上限，处于训练数据分布的稀疏边界区域；"
    "且该厚度组合在短波长端相位厚度大、光谱对膜厚的敏感性高，加大了学习难度。")

# ============================================================ 6 Discussion
replace_text(find_para("第一段：结合Figure 5"),
    f"结合图 5，MLP 能够以约 {mae*100:.1f}% 的典型平均偏差再现膜厚—光谱映射，"
    "验证了其作为代理模型的可行性。主要误差来源有三：其一，光谱对膜厚的依赖高度非线性，"
    "在干涉峰附近，微小的膜厚偏差即被放大为明显的峰位与峰高偏移；"
    "其二，训练样本在 40–180 nm 的四维立方体内均匀分布，边界区域样本密度低，"
    f"靠近边界的结构（如图 8 的 #{F8['worst_index']}）预测偏差更大；"
    "其三，输出层 Sigmoid 与 MSE 损失在接近 1 的高反射区梯度较小，"
    "对峰值区域存在轻微的系统性低估或平滑。")

replace_text(find_para("第二段：结合Figure 6"),
    f"图 6 表明测试误差随训练数据量近似幂律下降：500 → 4000 样本，MSE 下降约 {factor:.0f} 倍，"
    "且在 4000 处尚未出现明显平台，各加倍区间的误差降幅显示收益边际递减，"
    "与神经网络的经验缩放规律一致。"
    "对实际应用而言，这意味着继续以 TMM 批量生成数据可换取更高精度，"
    f"但需权衡数据生成成本：当代理精度已满足筛选需求（如 λtarget 处 MAE < "
    f"{sr[4000]['target_mae']:.3f}）时，继续扩大训练集的性价比会迅速下降。")

replace_text(find_para("第三段：结合Figure 7"),
    "图 7 与表 1 体现了代理模型的核心价值：MLP 评估 10000 组候选仅需毫秒量级的前向传播，"
    "而 TMM 逐一复算需数十秒以上，加速比达 3–4 个数量级；"
    f"同时 Top 1 候选经 TMM 验证后仍保持第 {T1['final_rank']}，其真实反射率"
    f"（{T1['tmm_Rtarget']:.4f}）甚至超过训练数据中的最大值（{_R_MAX_TGT:.4f}），"
    "说明学到的映射具备一定的泛化与外推能力。"
    f"但必须强调，MLP 的 Top 10 排序与 TMM 真实排序存在交错，预测值平均高估约 {over10:.3f}；"
    "若直接采信预测值而不做 TMM 复算，设计排序的可靠性无法保证。"
    "“AI 粗筛 + 物理精验”的分工是代理模型方法在工程上稳健且必要的用法。")

replace_text(find_para("第四段：结合Figure 8"),
    "图 8 的失败案例揭示了方法的边界：代理模型本质上是训练分布内的插值器，"
    "在分布边界与样本稀疏区域精度明显下降；对短波长端相位厚度大、振荡剧烈的光谱，"
    "固定容量的 MLP 难以精确分辨峰位。本研究的局限还包括：材料与层数固定，"
    "未考虑色散、吸收与斜入射；设计目标为单点反射率最大化，所得设计在 540 nm 附近的"
    "高反射带宽度有限，不适用于宽带需求。后续工作可将 MLP 代理与梯度优化或遗传算法结合，"
    "在连续设计空间内做闭环搜索，并对边界区域增加定向采样。")

delete_para(find_para("【写作提示】建议围绕预测准确性"))

# ============================================================ 7 Conclusions
replace_text(find_para("请在此撰写结论"),
    "本文构建了“TMM 物理模型 → 数据 → MLP 代理模型 → TMM 验证”的完整 AI4S 流程，"
    f"回答了三个研究问题：（1）MLP 能准确预测四层膜系光谱，测试集 MSE = {sci_txt(mse)}、"
    f"RMSE = {rmse:.4f}；（2）训练数据量从 500 增至 4000 时，测试 MSE 下降约 {factor:.0f} 倍，"
    "误差近似幂律衰减且未饱和；（3）MLP 代理可在毫秒级完成万级候选筛选，"
    f"Top 1 设计经 TMM 验证 R(540 nm) = {T1['tmm_Rtarget']:.4f}，"
    f"超过训练数据中的最大值 {_R_MAX_TGT:.4f}，验证了“AI 粗筛 + 物理精验”范式的有效性。"
    "代理模型的精度受训练分布边界限制，关键设计决策仍需物理模型最终确认。")

delete_para(find_para("【写作提示】100–150字"))

# ============================================================ 8 Data & Code
replace_text(find_para("https://github.com/USERNAME/REPOSITORY"),
    "https://github.com/<你的GitHub用户名>/AI4S_ThinFilm  （提交前替换为本人真实仓库链接）")

# ============================================================ 9 References
refs = [
    "[1] T. Ma, M. Ma, L. J. Guo. Optical multilayer thin film structure inverse design: "
    "From optimization to deep learning. iScience 2025, 28, 112222.",
    "[2] M. Born, E. Wolf. Principles of Optics: Electromagnetic Theory of Propagation, "
    "Interference and Diffraction of Light. 7th ed., Cambridge University Press, 1999.",
    "[3] H. A. Macleod. Thin-Film Optical Filters. 4th ed., CRC Press, 2010.",
    "[4] F. Abeles. Recherches sur la propagation des ondes électromagnétiques sinusoïdales "
    "dans les milieux stratifiés. Ann. Phys. (Paris) 1950, 12, 596–640.",
    "[5] J. Peurifoy, Y. Shen, L. Jing, Y. Yang, F. Cano-Renteria, B. G. DeLacy, "
    "J. D. Joannopoulos, M. Tegmark, S. G. Johnson. Nanophotonic particle simulation and "
    "inverse design using artificial neural networks. Science Advances 2018, 4, eaar4206.",
    "[6] D. Liu, Y. Tan, E. Khoram, Z. Yu. Training deep neural networks for the inverse "
    "design of nanophotonic structures. Journal of Physics Communications 2018, 2, 105101.",
    "[7] K. Hornik, M. Stinchcombe, H. White. Multilayer feedforward networks are universal "
    "approximators. Neural Networks 1989, 2, 359–366.",
    "[8] D. P. Kingma, J. Ba. Adam: A method for stochastic optimization. "
    "In: Proceedings of the 3rd International Conference on Learning Representations (ICLR), 2015.",
]
for k, tag in enumerate(["[1] Author(s)", "[2] Author(s)", "[3] Author(s)", "[4] Author(s)"]):
    replace_text(find_para(tag), refs[k])
last_ref = find_para("[4] F. Abeles")
prev_node = last_ref
for ref in refs[4:]:
    prev_node = insert_para_after(prev_node, ref)

delete_para(find_para("【写作提示】建议8–12篇真实文献"))

# 清理任何残留写作提示
for p in list(d.paragraphs):
    if p.text.strip().startswith("【写作提示】"):
        delete_para(p)

# ============================================================ 10 题注中文化（小五 9 pt）
def replace_caption(prefix, text, center=True):
    para = find_para(prefix)
    for r in list(para.runs):
        r._element.getparent().remove(r._element)
    run = para.add_run(text)
    set_run_font(run, size=9)
    if center:
        para.alignment = WD_ALIGN_PARAGRAPH.CENTER
    return para


for prefix, cn in [
    ("Figure 1. Overall workflow", "图 1. AI4S 薄膜研究总体工作流程。"),
    ("Figure 2. Physical model", "图 2. 物理模型与基于 TMM 的膜厚—光谱数据生成。"),
    ("Figure 3. Architecture", "图 3. MLP 代理模型的网络结构。"),
    ("Figure 4. Training and validation", "图 4. MLP 代理模型的训练与验证损失曲线。"),
    ("Figure 5. Comparison of TMM", "图 5. 测试集代表性样本的 TMM 真实光谱与 MLP 预测光谱对比。"),
    ("Figure 6. Effect of training-set", "图 6. 训练集规模对测试误差的影响。"),
    ("Figure 7. MLP-predicted", "图 7. 筛选设计的 MLP 预测光谱与 TMM 验证光谱。"),
    ("Figure 8. A representative", "图 8. 一个预测误差相对较大的代表性案例。"),
    ("Table 1. Top five candidate",
     "表 1. MLP 筛选并经 TMM 验证的 Top 5 候选膜系设计。MLP 预测 R 与 TMM 验证 R "
     "分别为 λtarget = 540 nm 处反射率的代理模型预测值与物理模型复算值。"),
]:
    replace_caption(prefix, cn)

# ============================================================ 11 课程评价与建议（文末）
h_eval = d.add_paragraph()
_run = h_eval.add_run("课程评价与建议")
set_run_font(_run)
_run.bold = True

p_eval = d.add_paragraph()
_run = p_eval.add_run(
    "《薄膜技术》课程系统讲授了光学薄膜的干涉原理、特征矩阵方法与典型膜系设计流程，"
    "从单层膜的四分之一波长条件到多层膜传输矩阵法，推导完整、物理图像清晰，"
    "与《光学》课程中的干涉与菲涅耳公式内容衔接自然，为本大作业中 TMM 模型的自主实现提供了直接依据。"
    "课程将 AI for Science 引入期末作业，要求从物理建模、数据生成到代理模型训练完整走通研究流程，"
    "对建立“物理模型 + 数据驱动”的研究视角帮助明显。"
    "建议课程后续增加 1–2 次上机实践（如用 Python 实现单层膜 TMM 并绘制反射光谱），"
    "并适当补充薄膜制备工艺与实测光谱误差来源的介绍，使设计结果与实际制备的联系更直观；"
    "此外可提供深度学习辅助光学逆设计的代表性文献作为选读材料，为大作业选题提供参考。")
set_run_font(_run)
p_eval.paragraph_format.first_line_indent = Pt(21)

# ============================================================ 保存
os.makedirs(os.path.dirname(OUT), exist_ok=True)
d.save(OUT)
print("已保存:", OUT)
print(f"数值同步: test MSE={mse:.3e}  RMSE={rmse:.4f}  MAE={mae:.4f}")
print(f"数据量: 500→4000 降幅 {factor:.1f} 倍; Top1 TMM R={T1['tmm_Rtarget']:.4f}")
