"""Draw the experiment pipeline diagram used in the report."""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

from train import ROOT  # noqa: E402

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]

BOXES = [
    # (x, y, w, h, text, color)
    (0.1, 2.2, 2.0, 1.4, "MedMNIST v2\nPneumoniaMNIST\nBloodMNIST\n128×128", "#dbe9f6"),
    (2.6, 2.2, 2.0, 1.4, "分层随机抽样\n训练集比例 0.5%~100%\n(Pneumonia 6 档 × 5 种子,\nBlood 4 档 × 3 种子)", "#fde8d0"),
    (5.1, 2.2, 2.0, 1.4, "GPU 数据增强\n旋转/缩放/平移\n翻转/亮度对比度", "#fde8d0"),
    (7.6, 2.2, 2.4, 1.4, "模型\nSimpleCNN\nResNet-18 (从头/预训练)\nViT-Tiny (从头/预训练)", "#e2f0d9"),
    (10.5, 2.2, 2.2, 1.4, "训练\nAdamW + 余弦退火\n类别加权交叉熵\nbf16 混合精度", "#e2f0d9"),
    (10.5, 0.1, 2.2, 1.4, "评估 (完整测试集)\nACC / AUC\nMacro-F1 / ECE", "#f3e0f3"),
    (7.6, 0.1, 2.4, 1.4, "分析\n数据量曲线 / 预训练增益\n消融实验\nGrad-CAM / t-SNE", "#f3e0f3"),
]


def main():
    fig, ax = plt.subplots(figsize=(11, 3.3))
    for x, y, w, h, t, c in BOXES:
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05", fc=c, ec="#444", lw=1))
        ax.text(x + w / 2, y + h / 2, t, ha="center", va="center", fontsize=8.5)
    arrows = [((2.1, 2.9), (2.6, 2.9)), ((4.6, 2.9), (5.1, 2.9)), ((7.1, 2.9), (7.6, 2.9)),
              ((10.0, 2.9), (10.5, 2.9)), ((11.6, 2.2), (11.6, 1.5)), ((10.5, 0.8), (10.0, 0.8))]
    for a, b in arrows:
        ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=12, color="#444"))
    ax.text(0.1, 0.8, "研究问题 (医学影像标注稀缺时)\nRQ1 预训练迁移在多少数据下最有用？\nRQ2 CNN 与 ViT 对数据量的依赖有何差异？\nRQ3 数据增强/类别加权/微调方式各贡献多少？\nRQ4 模型关注区域与特征是否可信？",
            fontsize=9, va="center", bbox=dict(boxstyle="round,pad=0.6", fc="#fff8dc", ec="#aa8"))
    ax.set_xlim(0, 12.9)
    ax.set_ylim(0, 3.8)
    ax.axis("off")
    plt.tight_layout()
    out = os.path.join(ROOT, "results", "figures")
    os.makedirs(out, exist_ok=True)
    plt.savefig(os.path.join(out, "fig_pipeline.png"), dpi=220)


if __name__ == "__main__":
    main()
