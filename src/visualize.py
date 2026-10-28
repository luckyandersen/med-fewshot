"""Grad-CAM and t-SNE for the saved seed-0 ResNet-18 checkpoints (PneumoniaMNIST).

Grad-CAM: Selvaraju et al., ICCV 2017 (re-implemented here with forward/backward hooks).
"""
import json
import os

import matplotlib
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from data import load_split, normalize  # noqa: E402
from models import build_model, extract_features  # noqa: E402
from train import CKPT_DIR, ROOT  # noqa: E402

FIG_DIR = os.path.join(ROOT, "results", "figures")
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

CKPTS = [  # (tag, column title)
    ("pneumonia_resnet18_pt_f0.05_s0", "预训练 / 5%"),
    ("pneumonia_resnet18_scratch_f0.05_s0", "从头训练 / 5%"),
    ("pneumonia_resnet18_pt_f1.0_s0", "预训练 / 100%"),
    ("pneumonia_resnet18_scratch_f1.0_s0", "从头训练 / 100%"),
]


def load(tag, device):
    m = build_model("resnet18", 2, pretrained=False).to(device)
    m.load_state_dict(torch.load(os.path.join(CKPT_DIR, tag + ".pt"), map_location=device))
    return m.eval()


def gradcam(model, x, target):
    acts, grads = {}, {}
    h1 = model.layer4.register_forward_hook(lambda m, i, o: acts.__setitem__("a", o))
    h2 = model.layer4.register_full_backward_hook(lambda m, gi, go: grads.__setitem__("g", go[0]))
    x = x.clone().requires_grad_(True)
    logits = model(x)
    model.zero_grad()
    logits.gather(1, target[:, None]).sum().backward()
    h1.remove()
    h2.remove()
    w = grads["g"].mean(dim=(2, 3), keepdim=True)
    cam = F.relu((w * acts["a"]).sum(1, keepdim=True))
    cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[:, 0]
    cam = cam / (cam.flatten(1).max(1)[0].view(-1, 1, 1) + 1e-8)
    return cam.detach().cpu().numpy(), F.softmax(logits, 1).detach().cpu().numpy()


def main(device="cuda"):
    os.makedirs(FIG_DIR, exist_ok=True)
    x, y, names = load_split("pneumonia", "test", 128)
    xd, yd = x.to(device), y.to(device)
    models = {tag: load(tag, device) for tag, _ in CKPTS}

    # 1 pneumonia + 1 normal random test image, plus 2 normal images misclassified as
    # pneumonia by the pretrained/100% model (its dominant error type); target = true class
    rng = np.random.RandomState(3)
    with torch.no_grad():
        pred = torch.cat([models[CKPTS[2][0]](normalize(xd[i:i + 256])).argmax(1)
                          for i in range(0, len(xd), 256)]).cpu().numpy()
    fp = np.where((y.numpy() == 0) & (pred == 1))[0]
    idx = np.concatenate([rng.choice(np.where(y.numpy() == c)[0], 1, replace=False) for c in (1, 0)]
                         + [rng.choice(fp, 2, replace=False)])
    xs, ys = normalize(xd[idx]), yd[idx]
    cams = {tag: gradcam(m, xs, ys) for tag, m in models.items()}

    fig, axes = plt.subplots(len(idx), len(CKPTS) + 1, figsize=(1.9 * (len(CKPTS) + 1), 1.9 * len(idx)))
    label_cn = {0: "正常", 1: "肺炎"}
    for r, i in enumerate(idx):
        img = x[i, 0].numpy()
        axes[r, 0].imshow(img, cmap="gray")
        axes[r, 0].set_ylabel(f"真实: {label_cn[int(y[i])]}", fontsize=9)
        for c, (tag, title) in enumerate(CKPTS, start=1):
            cam, prob = cams[tag]
            axes[r, c].imshow(img, cmap="gray")
            axes[r, c].imshow(cam[r], cmap="jet", alpha=0.45, vmin=0, vmax=1)
            p = prob[r, int(y[i])]
            axes[r, c].set_xlabel(f"p(真实类)={p:.2f}", fontsize=8, color="green" if p > 0.5 else "red")
            if r == 0:
                axes[r, c].set_title(title, fontsize=9)
        axes[0, 0].set_title("原图", fontsize=9)
    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "gradcam.png"), dpi=200)
    plt.close()

    # t-SNE of penultimate features on the whole test set
    stats = {}
    fig, axes = plt.subplots(1, len(CKPTS), figsize=(3.2 * len(CKPTS), 3.2))
    for ax, (tag, title) in zip(axes, CKPTS):
        with torch.no_grad():
            feats = torch.cat([extract_features(models[tag], "resnet18", normalize(xd[i:i + 256])).float()
                               for i in range(0, len(xd), 256)]).cpu().numpy()
        emb = TSNE(n_components=2, init="pca", perplexity=30, random_state=0).fit_transform(feats)
        sil = float(silhouette_score(feats, y.numpy()))
        stats[tag] = {"silhouette_feature_space": sil}
        for c, col in ((0, "#2b8cbe"), (1, "#e34a33")):
            m = y.numpy() == c
            ax.scatter(emb[m, 0], emb[m, 1], s=4, c=col, label=label_cn[c], alpha=0.7)
        ax.set_title(f"{title}\nsilhouette={sil:.3f}", fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
    axes[0].legend(fontsize=8, markerscale=3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "tsne.png"), dpi=200)
    plt.close()
    with open(os.path.join(ROOT, "results", "tsne_silhouette.json"), "w") as f:
        json.dump(stats, f, indent=2)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
