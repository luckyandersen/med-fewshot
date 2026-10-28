"""Aggregate results/runs/*.json into tables (CSV/Markdown) and figures."""
import glob
import json
import os

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from train import ROOT, RUN_DIR  # noqa: E402

OUT = os.path.join(ROOT, "results")
FIG = os.path.join(OUT, "figures")
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

MODEL_NAMES = {
    ("cnn", False): "SimpleCNN (从头)",
    ("resnet18", False): "ResNet-18 (从头)",
    ("resnet18", True): "ResNet-18 (预训练)",
    ("vit_tiny", False): "ViT-Tiny (从头)",
    ("vit_tiny", True): "ViT-Tiny (预训练)",
}
STYLE = {
    ("cnn", False): ("#7f7f7f", "o", "-"),
    ("resnet18", False): ("#1f77b4", "s", "--"),
    ("resnet18", True): ("#1f77b4", "s", "-"),
    ("vit_tiny", False): ("#d62728", "^", "--"),
    ("vit_tiny", True): ("#d62728", "^", "-"),
}


def load_runs():
    rows = []
    for p in glob.glob(os.path.join(RUN_DIR, "*.json")):
        r = json.load(open(p))
        c = r["cfg"]
        row = {k: c[k] for k in ("dataset", "model", "pretrained", "fraction", "seed", "aug", "class_weight", "freeze")}
        row.update(n_train=r["n_train"], steps=r["steps"], time=r["train_time_s"])
        for s in ("val", "test"):
            for m in ("acc", "auc", "f1", "ece"):
                row[f"{s}_{m}"] = r[s][m]
        rows.append(row)
    return pd.DataFrame(rows)


def fmt(mean, std, pct=True):
    k = 100 if pct else 1
    return f"{mean * k:.1f}±{std * k:.1f}" if pct else f"{mean:.3f}±{std:.3f}"


def default_mask(df):
    return df["aug"] & df["class_weight"] & ~df["freeze"]


def main_table(df, dataset):
    d = df[(df.dataset == dataset) & default_mask(df)]
    g = d.groupby(["model", "pretrained", "fraction"]).agg(
        n=("seed", "count"), n_train=("n_train", "first"),
        **{f"{m}_{a}": (f"test_{m}", a) for m in ("acc", "auc", "f1", "ece") for a in ("mean", "std")},
        val_acc_mean=("val_acc", "mean")).reset_index()
    g.to_csv(os.path.join(OUT, f"table_{dataset}_main.csv"), index=False)
    fracs = sorted(g.fraction.unique())
    lines = ["| 模型 | " + " | ".join(f"{f:g} (n={int(g[g.fraction == f].n_train.iloc[0])})" for f in fracs) + " |",
             "|---|" + "---|" * len(fracs)]
    for metric in ("auc", "acc", "f1"):
        lines.append(f"| **{metric.upper()} (%)** |" + " |" * len(fracs))
        for key, name in MODEL_NAMES.items():
            gg = g[(g.model == key[0]) & (g.pretrained == key[1])]
            if gg.empty:
                continue
            cells = []
            for f in fracs:
                r = gg[gg.fraction == f]
                cells.append(fmt(r[f"{metric}_mean"].iloc[0], r[f"{metric}_std"].iloc[0]) if len(r) else "-")
            lines.append(f"| {name} | " + " | ".join(cells) + " |")
    md = "\n".join(lines)
    open(os.path.join(OUT, f"table_{dataset}_main.md"), "w", encoding="utf-8").write(md)
    return g, md


def plot_curves(gp, gb):
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    panels = [(gp, "auc", "PneumoniaMNIST 测试集 AUC"), (gp, "acc", "PneumoniaMNIST 测试集 ACC"),
              (gb, "acc", "BloodMNIST 测试集 ACC")]
    for ax, (g, m, title) in zip(axes, panels):
        if g is None or g.empty:
            continue
        for key, name in MODEL_NAMES.items():
            gg = g[(g.model == key[0]) & (g.pretrained == key[1])].sort_values("fraction")
            if gg.empty:
                continue
            col, mk, ls = STYLE[key]
            ax.errorbar(gg.fraction * 100, gg[f"{m}_mean"] * 100, yerr=gg[f"{m}_std"] * 100, color=col,
                        marker=mk, ls=ls, ms=4, capsize=2, label=name)
        ax.set_xscale("log")
        ax.set_xlabel("训练集使用比例 (%)")
        ax.set_ylabel(f"{m.upper()} (%)")
        ax.set_title(title, fontsize=10)
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=7.5)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, "fig_main_curves.png"), dpi=200)
    plt.close()


def plot_gain(gp, gb):
    """Pre-training gain (pretrained minus scratch) as a function of data fraction."""
    fig, ax = plt.subplots(figsize=(4.6, 3.3))
    for g, ds, mk in ((gp, "Pneumonia", "o"), (gb, "Blood", "s")):
        if g is None or g.empty:
            continue
        for model, col in (("resnet18", "#1f77b4"), ("vit_tiny", "#d62728")):
            a = g[(g.model == model) & g.pretrained].set_index("fraction")["acc_mean"]
            b = g[(g.model == model) & ~g.pretrained].set_index("fraction")["acc_mean"]
            gain = (a - b).dropna().sort_index() * 100
            name = {"resnet18": "ResNet-18", "vit_tiny": "ViT-Tiny"}[model]
            ax.plot(gain.index * 100, gain.values, marker=mk, color=col,
                    ls="-" if ds == "Pneumonia" else ":", label=f"{name} / {ds}MNIST")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xscale("log")
    ax.set_xlabel("训练集使用比例 (%)")
    ax.set_ylabel("预训练带来的 ACC 提升 (百分点)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7.5)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, "fig_pretrain_gain.png"), dpi=200)
    plt.close()


def ablation(df):
    d = df[(df.dataset == "pneumonia") & (df.fraction.isin([0.05, 1.0]))]
    rows = []
    variants = [
        ("ResNet-18 预训练 · 完整设置", (d.model == "resnet18") & d.pretrained & default_mask(d)),
        ("  − 数据增强", (d.model == "resnet18") & d.pretrained & ~d.aug),
        ("  − 类别加权", (d.model == "resnet18") & d.pretrained & ~d.class_weight),
        ("  冻结骨干 (线性探测)", (d.model == "resnet18") & d.pretrained & d.freeze),
        ("ResNet-18 从头 · 完整设置", (d.model == "resnet18") & ~d.pretrained & default_mask(d)),
        ("  − 数据增强", (d.model == "resnet18") & ~d.pretrained & ~d.aug),
        ("  − 类别加权", (d.model == "resnet18") & ~d.pretrained & ~d.class_weight),
        ("ViT-Tiny 预训练 · 完整设置", (d.model == "vit_tiny") & d.pretrained & default_mask(d)),
        ("  冻结骨干 (线性探测)", (d.model == "vit_tiny") & d.pretrained & d.freeze),
    ]
    for vid, (name, m) in enumerate(variants):
        for f in (0.05, 1.0):
            x = d[m & (d.fraction == f)]
            if x.empty:
                continue
            rows.append({"vid": vid, "variant": name.strip(), "label": name, "fraction": f, "n": len(x),
                         **{f"{k}_{a}": getattr(x[f"test_{k}"], a)() for k in ("acc", "auc", "f1", "ece")
                            for a in ("mean", "std")}})
    t = pd.DataFrame(rows)
    t.to_csv(os.path.join(OUT, "table_ablation.csv"), index=False)
    lines = ["| 设置 | 5%: ACC | 5%: AUC | 5%: F1 | 100%: ACC | 100%: AUC | 100%: F1 |", "|---|---|---|---|---|---|---|"]
    for vid, (name, _) in enumerate(variants):
        cells = []
        for f in (0.05, 1.0):
            r = t[(t.vid == vid) & (t.fraction == f)]
            for k in ("acc", "auc", "f1"):
                cells.append(fmt(r[f"{k}_mean"].iloc[0], r[f"{k}_std"].iloc[0]) if len(r) else "-")
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    md = "\n".join(lines)
    open(os.path.join(OUT, "table_ablation.md"), "w", encoding="utf-8").write(md)
    return t, md


def plot_confusion():
    tags = [("pneumonia_resnet18_scratch_f0.05_s0", "ResNet-18 从头 / 5%"),
            ("pneumonia_resnet18_pt_f0.05_s0", "ResNet-18 预训练 / 5%"),
            ("pneumonia_resnet18_pt_f1.0_s0", "ResNet-18 预训练 / 100%")]
    fig, axes = plt.subplots(1, 3, figsize=(9.5, 3.0))
    for ax, (tag, title) in zip(axes, tags):
        p = os.path.join(RUN_DIR, tag + ".json")
        if not os.path.exists(p):
            continue
        cm = np.array(json.load(open(p))["test"]["cm"])
        ax.imshow(cm, cmap="Blues")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, cm[i, j], ha="center", va="center", color="white" if cm[i, j] > cm.max() / 2 else "black")
        ax.set_xticks([0, 1], ["正常", "肺炎"])
        ax.set_yticks([0, 1], ["正常", "肺炎"])
        ax.set_xlabel("预测")
        ax.set_ylabel("真实")
        ax.set_title(title, fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, "fig_confusion.png"), dpi=200)
    plt.close()


def main():
    os.makedirs(FIG, exist_ok=True)
    df = load_runs()
    df.to_csv(os.path.join(OUT, "all_runs.csv"), index=False)
    print(f"{len(df)} runs loaded")
    gp, mdp = main_table(df, "pneumonia")
    gb, mdb = main_table(df, "blood") if (df.dataset == "blood").any() else (None, "")
    print(mdp, "\n\n", mdb)
    plot_curves(gp, gb)
    plot_gain(gp, gb)
    t, mda = ablation(df)
    print("\n", mda)
    plot_confusion()
    # validation vs test gap on PneumoniaMNIST (distribution shift between splits)
    d = df[(df.dataset == "pneumonia") & default_mask(df) & (df.fraction == 1.0)]
    gap = d.groupby(["model", "pretrained"])[["val_acc", "test_acc", "test_ece"]].mean()
    gap.to_csv(os.path.join(OUT, "table_val_test_gap.csv"))
    print("\n", gap)


if __name__ == "__main__":
    main()
