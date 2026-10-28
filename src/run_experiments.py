"""Run all experiment groups sequentially; finished runs (json exists) are skipped.

python src/run_experiments.py --group main|blood|ablation|all
"""
import argparse
import os
import traceback

from train import RUN_DIR, default_cfg, run, run_tag

SEEDS = [0, 1, 2, 3, 4]
BLOOD_SEEDS = [0, 1, 2]
MODELS = [  # (model, pretrained)
    ("cnn", False),
    ("resnet18", False),
    ("resnet18", True),
    ("vit_tiny", False),
    ("vit_tiny", True),
]


def group_main():
    """E1: model x training-set fraction on PneumoniaMNIST."""
    for frac in [0.005, 0.01, 0.05, 0.1, 0.25, 1.0]:
        for m, pt in MODELS:
            for s in SEEDS:
                yield default_cfg(model=m, pretrained=pt, fraction=frac, seed=s)


def group_blood():
    """E2: the same question on BloodMNIST (8 classes, RGB) to check generality."""
    for frac in [0.01, 0.05, 0.1, 1.0]:
        for m, pt in MODELS:
            for s in BLOOD_SEEDS:
                yield default_cfg(dataset="blood", model=m, pretrained=pt, fraction=frac, seed=s)


def group_ablation():
    """E3: augmentation / class weighting / linear probing on PneumoniaMNIST."""
    for frac in [0.05, 1.0]:
        for s in SEEDS:
            for m, pt in [("resnet18", True), ("resnet18", False)]:
                yield default_cfg(model=m, pretrained=pt, fraction=frac, seed=s, aug=False)
                yield default_cfg(model=m, pretrained=pt, fraction=frac, seed=s, class_weight=False)
            yield default_cfg(model="resnet18", pretrained=True, fraction=frac, seed=s, freeze=True)
            yield default_cfg(model="vit_tiny", pretrained=True, fraction=frac, seed=s, freeze=True)


# Seed-0 ResNet-18 weights are kept for Grad-CAM / t-SNE.
CKPT_TAGS = {run_tag(default_cfg(model="resnet18", pretrained=pt, fraction=f, seed=0))
             for pt in (True, False) for f in (0.05, 1.0)}

GROUPS = {"main": group_main, "blood": group_blood, "ablation": group_ablation}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", default="all")
    a = ap.parse_args()
    names = list(GROUPS) if a.group == "all" else [a.group]
    for g in names:
        cfgs = list(GROUPS[g]())
        print(f"== group {g}: {len(cfgs)} runs", flush=True)
        for cfg in cfgs:
            if os.path.exists(os.path.join(RUN_DIR, run_tag(cfg) + ".json")):
                continue
            try:
                run(cfg, save_ckpt=run_tag(cfg) in CKPT_TAGS)
            except Exception:
                traceback.print_exc()
