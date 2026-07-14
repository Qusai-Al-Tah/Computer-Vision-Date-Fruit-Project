"""
Model definitions: a shared pretrained backbone with one or two
classification heads.

Architecture inspiration: the course slides on Classification + Localization
(13DetectionSegmentation) — a shared CNN backbone feeding two parallel FC
heads with a summed loss. Here the second head predicts quality grade
instead of box coordinates.

Backbones (all from 12FamousCNN): resnet18, resnet34, resnet50, vgg16,
googlenet — loaded with ImageNet pretrained weights via torchvision.
"""

import torch
import torch.nn as nn
from torchvision import models


def _make_backbone(name: str, pretrained: bool = True):
    """Returns (feature_extractor_module, feature_dim)."""
    weights = "IMAGENET1K_V1" if pretrained else None
    if name == "resnet18":
        m = models.resnet18(weights=weights)
        dim = m.fc.in_features
        m.fc = nn.Identity()
    elif name == "resnet34":
        m = models.resnet34(weights=weights)
        dim = m.fc.in_features
        m.fc = nn.Identity()
    elif name == "resnet50":
        m = models.resnet50(weights=weights)
        dim = m.fc.in_features
        m.fc = nn.Identity()
    elif name == "vgg16":
        m = models.vgg16_bn(weights=weights)
        dim = m.classifier[0].in_features
        m.classifier = nn.Identity()
    elif name == "googlenet":
        m = models.googlenet(weights=weights, aux_logits=True)
        m.aux_logits = False  # disable aux heads for fine-tuning simplicity
        m.aux1 = None
        m.aux2 = None
        dim = m.fc.in_features
        m.fc = nn.Identity()
    else:
        raise ValueError(f"Unknown backbone: {name}")
    return m, dim


class DateNet(nn.Module):
    """task='variety' | 'grade' -> single head; task='multi' -> two heads."""

    def __init__(
        self,
        backbone: str = "resnet18",
        task: str = "multi",
        n_varieties: int = 4,
        n_grades: int = 3,
        dropout: float = 0.5,
        freeze_backbone: bool = False,
        pretrained: bool = True,
    ):
        super().__init__()
        self.task = task
        self.backbone, dim = _make_backbone(backbone, pretrained)
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        def head(n_out):
            layers = []
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            layers.append(nn.Linear(dim, n_out))
            return nn.Sequential(*layers)

        if task in ("variety", "multi"):
            self.variety_head = head(n_varieties)
        if task in ("grade", "multi"):
            self.grade_head = head(n_grades)

    def forward(self, x):
        feats = self.backbone(x)
        out = {}
        if self.task in ("variety", "multi"):
            out["variety"] = self.variety_head(feats)
        if self.task in ("grade", "multi"):
            out["grade"] = self.grade_head(feats)
        return out


class ScratchCNN(nn.Module):
    """Small CNN trained from scratch (baseline showing why transfer
    learning wins — conv/pool/FC pattern per 10CNN slides)."""

    def __init__(self, task: str, n_varieties: int = 4, n_grades: int = 3):
        super().__init__()
        self.task = task

        def block(cin, cout):
            return nn.Sequential(
                nn.Conv2d(cin, cout, 3, padding=1),
                nn.BatchNorm2d(cout),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            )

        self.features = nn.Sequential(
            block(3, 32), block(32, 64), block(64, 128), block(128, 256),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
        )
        if task in ("variety", "multi"):
            self.variety_head = nn.Linear(256, n_varieties)
        if task in ("grade", "multi"):
            self.grade_head = nn.Linear(256, n_grades)

    def forward(self, x):
        feats = self.features(x)
        out = {}
        if self.task in ("variety", "multi"):
            out["variety"] = self.variety_head(feats)
        if self.task in ("grade", "multi"):
            out["grade"] = self.grade_head(feats)
        return out


def build_model(backbone, task, n_varieties, n_grades, dropout, freeze):
    if backbone == "scratch":
        return ScratchCNN(task, n_varieties, n_grades)
    return DateNet(backbone, task, n_varieties, n_grades, dropout, freeze)
