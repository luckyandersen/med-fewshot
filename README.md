# 小样本医学影像分类中的迁移学习研究

在 MedMNIST v2 的 PneumoniaMNIST（胸片肺炎二分类）与 BloodMNIST（8 类血细胞）上，
系统比较训练数据从 0.5% 到 100% 变化时：

- **RQ1** ImageNet 预训练迁移相对从头训练的收益如何随数据量变化；
- **RQ2** CNN（SimpleCNN / ResNet-18）与 ViT-Tiny 对数据量的依赖差异；
- **RQ3** 数据增强、类别加权损失、全量微调 vs. 线性探测各自的贡献（消融）；
- **RQ4** 通过 Grad-CAM 与 t-SNE 检查模型的关注区域与特征可分性。

## 主要结论（270 次训练，PneumoniaMNIST 5 个种子 / BloodMNIST 3 个种子）

| PneumoniaMNIST 测试 ACC (%) | 0.5% (23 张) | 1% | 5% | 10% | 25% | 100% (4708 张) |
|---|---|---|---|---|---|---|
| SimpleCNN (从头) | 81.7 | 84.8 | 91.5 | 91.9 | 94.3 | 96.3 |
| ResNet-18 (从头) | 81.6 | 85.3 | 90.7 | 92.8 | 94.5 | 96.2 |
| ResNet-18 (预训练) | 84.0 | 88.5 | 92.3 | 93.7 | 95.3 | 94.9 |
| ViT-Tiny (从头) | 77.0 | 81.9 | 85.3 | 84.6 | 86.2 | 88.9 |
| ViT-Tiny (预训练) | 81.0 | 87.1 | 93.8 | 94.5 | 96.2 | 96.2 |

- 预训练的收益随数据量增加而递减（ResNet-18：1% 数据 +3.2 → 全量 −1.2 个百分点），BloodMNIST 上规律一致；
- ViT 比 CNN 更依赖预训练，从头训练的 ViT 用全部数据也不如用 5% 数据的 SimpleCNN；
- 消融：数据增强贡献最大（5% 数据下去掉后 ACC −7.6～−9.5），全量微调显著优于线性探测，类别加权影响较小；
- PneumoniaMNIST 验证集与测试集存在分布偏移，导致测试集上“正常误判为肺炎”的假阳性增多。

完整表格见 `results/table_*.md`。

## 目录结构

```
src/
  data.py             数据加载、分层抽样、GPU 端数据增强
  models.py           SimpleCNN / ResNet-18 (torchvision) / ViT-Tiny (timm)
  train.py            单次训练与评估 (ACC / AUC / Macro-F1 / ECE)
  run_experiments.py  全部实验组 (main / blood / ablation)，可断点续跑
  analyze.py          汇总表格与曲线
  visualize.py        Grad-CAM 与 t-SNE
  pipeline_figure.py  报告中的流程图
results/
  runs/*.json         每次运行的配置、损失曲线与指标（原始结果）
  table_*.md|csv      汇总表
  figures/*.png       报告插图
```

## 复现

```bash
conda create -n dlcd python=3.11 -y
conda activate dlcd
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
cd src
python run_experiments.py --group all   # 约 1.5 小时 (RTX 3070 Ti Laptop)
python analyze.py
python visualize.py
```

数据集首次运行时由 `medmnist` 自动下载到 `data/`。

## 参考

- 数据：MedMNIST v2 — J. Yang et al., *MedMNIST v2: A large-scale lightweight benchmark for 2D and 3D biomedical image classification*, Scientific Data, 2023. https://medmnist.com/
- ResNet-18：He et al., CVPR 2016；权重来自 torchvision。
- ViT-Tiny：Dosovitskiy et al., ICLR 2021；权重 `vit_tiny_patch16_224.augreg_in21k_ft_in1k` 来自 timm (Steiner et al., 2022)。
- Grad-CAM：Selvaraju et al., ICCV 2017（本仓库自行实现）。
