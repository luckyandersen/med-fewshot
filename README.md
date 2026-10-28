# 小样本医学影像分类中的迁移学习研究

深度学习课程设计的代码和实验结果。数据用的是 MedMNIST v2 里的 PneumoniaMNIST（胸片，肺炎/正常二分类）和
BloodMNIST（血细胞，8 类）。把训练集从 0.5% 逐级加到 100%，比较 SimpleCNN、ResNet-18、ViT-Tiny
在从头训练和 ImageNet 预训练两种情况下的表现，另外做了数据增强、类别加权、线性探测的消融，以及 Grad-CAM 和 t-SNE 可视化。

## 结果

PneumoniaMNIST 测试集 ACC（%，5 次重复取平均）：

| 模型 | 0.5% (23 张) | 1% | 5% | 10% | 25% | 100% (4708 张) |
|---|---|---|---|---|---|---|
| SimpleCNN (从头) | 81.7 | 84.8 | 91.5 | 91.9 | 94.3 | 96.3 |
| ResNet-18 (从头) | 81.6 | 85.3 | 90.7 | 92.8 | 94.5 | 96.2 |
| ResNet-18 (预训练) | 84.0 | 88.5 | 92.3 | 93.7 | 95.3 | 94.9 |
| ViT-Tiny (从头) | 77.0 | 81.9 | 85.3 | 84.6 | 86.2 | 88.9 |
| ViT-Tiny (预训练) | 81.0 | 87.1 | 93.8 | 94.5 | 96.2 | 96.2 |

几点观察：

- 数据越少，预训练的作用越大。ResNet-18 在 1% 数据时预训练提高 3.2 个百分点，用全部数据时反而低 1.2 个百分点。BloodMNIST 上也是这样。
- ViT 很依赖预训练，从头训练时用全部数据还不如只用 5% 数据的 SimpleCNN。
- 消融里数据增强影响最大，去掉后 5% 数据下 ACC 掉了 7～10 个百分点；全量微调比线性探测好很多；类别加权影响不大。
- PneumoniaMNIST 的验证集和测试集分布不太一样，测试集上把正常判成肺炎的错误明显变多。

其他表格在 `results/table_*.md`，每次运行的原始结果在 `results/runs/`。

## 文件说明

```
src/
  data.py             读数据、分层抽样、GPU 上做数据增强
  models.py           SimpleCNN / ResNet-18 / ViT-Tiny
  train.py            训练一个配置并在验证集、测试集上评估
  run_experiments.py  跑全部实验（main / blood / ablation），已跑过的会跳过
  analyze.py          汇总成表格和曲线
  visualize.py        Grad-CAM 和 t-SNE
  pipeline_figure.py  画流程图
results/
  runs/*.json         每次运行的配置、loss 和各项指标
  table_*.md, *.csv   汇总表
  figures/            图
```

## 运行

```bash
conda create -n dlcd python=3.11 -y
conda activate dlcd
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
cd src
python run_experiments.py --group all
python analyze.py
python visualize.py
```

第一次运行会自动下载数据到 `data/`。在 RTX 3070 Ti Laptop 上全部跑完大概一个半到两个小时。

## 参考

- MedMNIST v2: J. Yang et al., Scientific Data, 2023. https://medmnist.com/
- ResNet: K. He et al., CVPR 2016（预训练权重来自 torchvision）
- ViT: A. Dosovitskiy et al., ICLR 2021（预训练权重 `vit_tiny_patch16_224.augreg_in21k_ft_in1k` 来自 timm）
- Grad-CAM: R. R. Selvaraju et al., ICCV 2017
