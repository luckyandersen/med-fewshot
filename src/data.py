"""MedMNIST v2 data loading, stratified few-shot subsampling and GPU augmentation.

Dataset: MedMNIST v2 (Yang et al., Scientific Data 2023), https://medmnist.com/
The whole split is held as a uint8 tensor on the GPU; augmentation is done
per-sample with affine_grid/grid_sample, so no DataLoader workers are needed.
"""
import math
import os

import medmnist
import numpy as np
import torch
import torch.nn.functional as F

DATA_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

DATASETS = {
    "pneumonia": "pneumoniamnist",
    "blood": "bloodmnist",
}


def load_split(name, split, size=128):
    """Return (images uint8 [N,3,H,W], labels int64 [N], class names)."""
    flag = DATASETS[name]
    info = medmnist.INFO[flag]
    cls = getattr(medmnist, info["python_class"])
    os.makedirs(DATA_ROOT, exist_ok=True)
    ds = cls(split=split, download=True, size=size, root=DATA_ROOT)
    imgs = ds.imgs
    if imgs.ndim == 3:  # grayscale -> 3 channels so ImageNet weights apply
        imgs = np.repeat(imgs[:, None], 3, axis=1)
    else:
        imgs = imgs.transpose(0, 3, 1, 2)
    labels = ds.labels.reshape(-1).astype(np.int64)
    names = [info["label"][str(i)] for i in range(len(info["label"]))]
    return torch.from_numpy(np.ascontiguousarray(imgs)), torch.from_numpy(labels), names


def stratified_subset(labels, fraction, seed):
    """Indices of a stratified random subset; at least 2 samples per class are kept."""
    if fraction >= 1.0:
        return torch.arange(len(labels))
    rng = np.random.RandomState(seed)
    labels = labels.numpy()
    idx = []
    for c in np.unique(labels):
        cidx = np.where(labels == c)[0]
        k = max(2, int(round(len(cidx) * fraction)))
        idx.append(rng.choice(cidx, k, replace=False))
    idx = np.sort(np.concatenate(idx))
    return torch.from_numpy(idx)


def normalize(x_uint8):
    x = x_uint8.float() / 255.0
    return (x - IMAGENET_MEAN.to(x.device)) / IMAGENET_STD.to(x.device)


def augment(x, hflip=True, max_rot=10.0, scale=(0.9, 1.1), max_shift=0.08, jitter=0.15):
    """Random affine (rotation/scale/shift/flip) + brightness/contrast jitter, per sample.

    x: float tensor in [0,1], shape [B,3,H,W] on GPU.
    """
    b = x.shape[0]
    dev = x.device
    ang = (torch.rand(b, device=dev) * 2 - 1) * math.radians(max_rot)
    s = torch.empty(b, device=dev).uniform_(*scale)
    flip = torch.where(torch.rand(b, device=dev) < 0.5, -1.0, 1.0) if hflip else torch.ones(b, device=dev)
    tx = (torch.rand(b, device=dev) * 2 - 1) * max_shift * 2
    ty = (torch.rand(b, device=dev) * 2 - 1) * max_shift * 2
    cos, sin = torch.cos(ang) / s, torch.sin(ang) / s
    theta = torch.stack([
        torch.stack([cos * flip, -sin, tx], 1),
        torch.stack([sin * flip, cos, ty], 1),
    ], 1)
    grid = F.affine_grid(theta, x.shape, align_corners=False)
    x = F.grid_sample(x, grid, mode="bilinear", padding_mode="border", align_corners=False)
    bright = 1 + (torch.rand(b, 1, 1, 1, device=dev) * 2 - 1) * jitter
    contrast = 1 + (torch.rand(b, 1, 1, 1, device=dev) * 2 - 1) * jitter
    mean = x.mean(dim=(1, 2, 3), keepdim=True)
    x = ((x - mean) * contrast + mean) * bright
    return x.clamp(0, 1)
