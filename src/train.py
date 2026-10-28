"""Train / evaluate one configuration. Usable as a library (run()) or from the CLI."""
import argparse
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score

from data import IMAGENET_MEAN, IMAGENET_STD, augment, load_split, normalize, stratified_subset
from models import build_model, head_parameters

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUN_DIR = os.path.join(ROOT, "results", "runs")
CKPT_DIR = os.path.join(ROOT, "checkpoints")

# Learning rates fixed a priori (not tuned on the test set).
DEFAULT_LR = {
    ("cnn", False): 1e-3,
    ("resnet18", False): 1e-3,
    ("resnet18", True): 1e-4,
    ("vit_tiny", False): 5e-4,
    ("vit_tiny", True): 1e-4,
}

_CACHE = {}


def get_data(dataset, size, device):
    key = (dataset, size)
    if key not in _CACHE:
        splits = {}
        for s in ("train", "val", "test"):
            x, y, names = load_split(dataset, s, size)
            splits[s] = (x.to(device), y.to(device))
        _CACHE[key] = (splits, names)
    return _CACHE[key]


def run_tag(cfg):
    tag = f"{cfg['dataset']}_{cfg['model']}_{'pt' if cfg['pretrained'] else 'scratch'}_f{cfg['fraction']}"
    if not cfg["aug"]:
        tag += "_noaug"
    if not cfg["class_weight"]:
        tag += "_nocw"
    if cfg["freeze"]:
        tag += "_linear"
    return tag + f"_s{cfg['seed']}"


def expected_calibration_error(probs, labels, n_bins=15):
    conf = probs.max(1)
    pred = probs.argmax(1)
    correct = (pred == labels).astype(float)
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            ece += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(ece)


@torch.no_grad()
def predict(model, x_uint8, bs=256):
    model.eval()
    out = []
    for i in range(0, len(x_uint8), bs):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(normalize(x_uint8[i:i + bs]))
        out.append(F.softmax(logits.float(), 1))
    return torch.cat(out).cpu().numpy()


def compute_metrics(probs, labels):
    labels = labels.cpu().numpy()
    pred = probs.argmax(1)
    if probs.shape[1] == 2:
        auc = roc_auc_score(labels, probs[:, 1])
    else:
        auc = roc_auc_score(labels, probs, multi_class="ovr", average="macro")
    return {
        "acc": float(accuracy_score(labels, pred)),
        "auc": float(auc),
        "f1": float(f1_score(labels, pred, average="macro")),
        "ece": expected_calibration_error(probs, labels),
        "cm": confusion_matrix(labels, pred).tolist(),
    }


def run(cfg, device="cuda", save_ckpt=False, verbose=True):
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    splits, names = get_data(cfg["dataset"], cfg["size"], device)
    num_classes = len(names)
    xtr_all, ytr_all = splits["train"]
    idx = stratified_subset(ytr_all.cpu(), cfg["fraction"], cfg["seed"]).to(device)
    xtr, ytr = xtr_all[idx], ytr_all[idx]
    n = len(ytr)

    model = build_model(cfg["model"], num_classes, cfg["pretrained"], cfg["size"]).to(device)
    model = model.to(memory_format=torch.channels_last)
    if cfg["freeze"]:
        for p in model.parameters():
            p.requires_grad_(False)
        params = head_parameters(model, cfg["model"])
        for p in params:
            p.requires_grad_(True)
    else:
        params = [p for p in model.parameters()]
    lr = cfg["lr"] or (1e-3 if cfg["freeze"] else DEFAULT_LR[(cfg["model"], cfg["pretrained"])])
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=cfg["wd"])

    bs = min(cfg["batch_size"], n)
    steps = max(cfg["min_steps"], math.ceil(cfg["epochs"] * n / bs))
    warmup = max(1, int(0.05 * steps))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warmup if s < warmup else 0.5 * (1 + math.cos(math.pi * (s - warmup) / (steps - warmup))))

    if cfg["class_weight"]:
        counts = torch.bincount(ytr, minlength=num_classes).float()
        weight = (n / (num_classes * counts.clamp(min=1))).to(device)
    else:
        weight = None

    mean, std = IMAGENET_MEAN.to(device), IMAGENET_STD.to(device)
    log, t0, perm, ptr = [], time.time(), torch.randperm(n, device=device), 0
    for step in range(steps):
        if ptr + bs > n:
            perm, ptr = torch.randperm(n, device=device), 0
        b = perm[ptr:ptr + bs]
        ptr += bs
        x = xtr[b].float() / 255.0
        if cfg["aug"]:
            x = augment(x)
        x = ((x - mean) / std).contiguous(memory_format=torch.channels_last)
        if cfg["freeze"]:
            model.eval()  # keep BN statistics of the frozen backbone
        else:
            model.train()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(x)
        loss = F.cross_entropy(logits.float(), ytr[b], weight=weight)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        if step % 20 == 0 or step == steps - 1:
            log.append([step, loss.item()])

    train_time = time.time() - t0
    res = {"cfg": cfg, "tag": run_tag(cfg), "n_train": int(n), "steps": steps, "lr": lr,
           "train_time_s": train_time, "loss_log": log, "class_names": names}
    for s in ("val", "test"):
        xs, ys = splits[s]
        probs = predict(model, xs)
        res[s] = compute_metrics(probs, ys)
    if verbose:
        t = res["test"]
        print(f"[{res['tag']}] n={n} steps={steps} {train_time:.0f}s | "
              f"test acc={t['acc']:.4f} auc={t['auc']:.4f} f1={t['f1']:.4f} ece={t['ece']:.4f}", flush=True)
    os.makedirs(RUN_DIR, exist_ok=True)
    with open(os.path.join(RUN_DIR, res["tag"] + ".json"), "w") as f:
        json.dump(res, f)
    if save_ckpt:
        os.makedirs(CKPT_DIR, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(CKPT_DIR, res["tag"] + ".pt"))
    return res


def default_cfg(**kw):
    cfg = dict(dataset="pneumonia", model="resnet18", pretrained=True, fraction=1.0, seed=0, size=128,
               aug=True, class_weight=True, freeze=False, lr=None, wd=0.05, batch_size=64,
               epochs=30, min_steps=300)
    cfg.update(kw)
    return cfg


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="pneumonia")
    ap.add_argument("--model", default="resnet18")
    ap.add_argument("--pretrained", action="store_true")
    ap.add_argument("--fraction", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no_aug", action="store_true")
    ap.add_argument("--no_cw", action="store_true")
    ap.add_argument("--freeze", action="store_true")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--save_ckpt", action="store_true")
    a = ap.parse_args()
    run(default_cfg(dataset=a.dataset, model=a.model, pretrained=a.pretrained, fraction=a.fraction, seed=a.seed,
                    aug=not a.no_aug, class_weight=not a.no_cw, freeze=a.freeze, epochs=a.epochs),
        save_ckpt=a.save_ckpt)
