"""Model zoo: a small CNN baseline, ResNet-18 (torchvision) and ViT-Tiny (timm).

ResNet-18: He et al., CVPR 2016; weights from torchvision (IMAGENET1K_V1).
ViT-Tiny:  Dosovitskiy et al., ICLR 2021; weights from timm
           (vit_tiny_patch16_224.augreg_in21k_ft_in1k, Steiner et al., TMLR 2022).
"""
import timm
import torch.nn as nn
import torchvision


class SimpleCNN(nn.Module):
    """4 conv blocks (conv-BN-ReLU x2 + maxpool) + global average pooling."""

    def __init__(self, num_classes, width=32):
        super().__init__()
        layers, cin = [], 3
        for i in range(4):
            cout = width * 2 ** i
            layers += [
                nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
                nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            ]
            cin = cout
        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(cin, num_classes)

    def forward_features(self, x):
        return self.pool(self.features(x)).flatten(1)

    def forward(self, x):
        return self.fc(self.forward_features(x))


def build_model(name, num_classes, pretrained, img_size=128):
    if name == "cnn":
        return SimpleCNN(num_classes)
    if name == "resnet18":
        weights = torchvision.models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        m = torchvision.models.resnet18(weights=weights)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
        return m
    if name == "vit_tiny":
        return timm.create_model("vit_tiny_patch16_224.augreg_in21k_ft_in1k", pretrained=pretrained,
                                 num_classes=num_classes, img_size=img_size)
    raise ValueError(name)


def head_parameters(model, name):
    """Parameters of the classification head (used for linear probing)."""
    if name == "resnet18":
        return list(model.fc.parameters())
    if name == "vit_tiny":
        return list(model.head.parameters())
    return list(model.fc.parameters())


def extract_features(model, name, x):
    """Penultimate-layer features, used for t-SNE."""
    if name == "resnet18":
        m = model
        x = m.maxpool(m.relu(m.bn1(m.conv1(x))))
        x = m.layer4(m.layer3(m.layer2(m.layer1(x))))
        return m.avgpool(x).flatten(1)
    if name == "vit_tiny":
        return model.forward_head(model.forward_features(x), pre_logits=True)
    return model.forward_features(x)
