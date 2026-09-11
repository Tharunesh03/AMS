"""Model zoo for the supervised candidates.

Deep candidates (all CPU-friendly, all with an explicit Grad-CAM target):

* ``lightcnn``  - tiny purpose-built CNN (native 32x32 input, ~0.1 M params)
* ``mobilenet_v3_small`` / ``efficientnet_b0`` / ``resnet18`` - torchvision architectures,
  used with ImageNet transfer learning when the weights are reachable
  (``--pretrained``, default) and from scratch when they are not.  This module never
  silently pretends a model was pretrained: ``pretrained`` is recorded in the metadata.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- lightcnn
class _ConvBlock(nn.Module):
    def __init__(self, cin: int, cout: int, stride: int = 1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(cin, cout, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class LightCNN(nn.Module):
    """4 blocks, 3x3 convs, global pooling head. Returns logits (1 output for binary).

    There is no ImageNet-pretrained version of this architecture, so ``pretrained_applied``
    is always False here - the comparison report shows that instead of implying transfer learning.
    """

    def __init__(self, width: int = 24, num_classes: int = 1, dropout: float = 0.1, in_ch: int = 3) -> None:
        super().__init__()
        c = [width, width * 2, width * 4, width * 6]
        self.b1 = _ConvBlock(in_ch, c[0])
        self.b2 = _ConvBlock(c[0], c[1])
        self.b3 = _ConvBlock(c[1], c[2])
        self.b4 = _ConvBlock(c[2], c[3])
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(c[3], num_classes)
        self.pretrained_applied = False
        self.pretrained_note = "architecture has no published ImageNet weights; trained from scratch by design"
        self.cam_module = self.b4.net[5]  # last activation before pooling: Grad-CAM target

    # Grad-CAM needs the last conv feature map
    def forward(self, x: torch.Tensor, return_features: bool = False):
        h = F.max_pool2d(self.b1(x), 2)
        h = F.max_pool2d(self.b2(h), 2)
        h = F.max_pool2d(self.b3(h), 2)
        h = self.b4(h)
        feat = h
        z = self.fc(self.drop(self.pool(h).flatten(1)))
        if return_features:
            return z, feat
        return z

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# --------------------------------------------------------------------------- wrappers
class TorchVisionNet(nn.Module):
    """torchvision backbone + adaptive-pool head, with a Grad-CAM feature path.

    ``body`` is everything up to the last spatial activation and ``head`` pools it, so the
    network accepts any input size (we use 64px because the sources are 32px crops - see the
    resolution ablation in reports/model_comparison.md).
    """

    def __init__(
        self,
        arch: str,
        num_classes: int = 1,
        pretrained: bool = True,
        dropout: float = 0.2,
        freeze_backbone: bool = False,
    ) -> None:
        super().__init__()
        from torchvision import models

        self.arch = arch
        weights = "IMAGENET1K_V1" if pretrained else None
        try:
            if arch == "mobilenet_v3_small":
                net = models.mobilenet_v3_small(weights=weights)
            elif arch == "mobilenet_v3_large":
                net = models.mobilenet_v3_large(weights=weights)
            elif arch == "efficientnet_b0":
                net = models.efficientnet_b0(weights=weights)
            elif arch == "resnet18":
                net = models.resnet18(weights=weights)
            else:
                raise KeyError(arch)
        except Exception as exc:  # noqa: BLE001 - weight download may be unavailable offline
            if not pretrained:
                raise
            print(f"[warn] could not load pretrained weights for {arch} ({type(exc).__name__}); from scratch")
            self._weight_error = f"{type(exc).__name__}: {exc}"
            return self.__init__(arch, num_classes=num_classes, pretrained=False, dropout=dropout,
                                 freeze_backbone=False)      # freezing a random backbone would be useless
        if arch.startswith(("mobilenet", "efficientnet")):
            body = net.features
            # first Linear in the stock classifier consumes the pooled feature map
            in_features = int(next(m.in_features for m in net.classifier.modules() if isinstance(m, nn.Linear)))
        else:
            in_features = int(net.fc.in_features)
            body = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool,
                                 net.layer1, net.layer2, net.layer3, net.layer4)
        self.body = body
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(dropout), nn.Linear(in_features, num_classes)
        )
        self.cam_module = self._last_spatial(body)
        self.backbone_pretrained = bool(pretrained)
        self.pretrained_applied = bool(pretrained)
        self.pretrained_note = (
            "ImageNet weights loaded" if pretrained
            else getattr(self, "_weight_error", "") and f"requested but unreachable ({self._weight_error}); from scratch"
            or "not requested"
        )
        if freeze_backbone and self.pretrained_applied:
            self.freeze_backbone()
        elif freeze_backbone:
            print("[warn] freeze_backbone ignored: no pretrained weights were loaded, so the backbone stays trainable")

    @staticmethod
    def _last_spatial(body: nn.Module) -> nn.Module:
        """Last Conv2d/BN inside the body: the map Grad-CAM differentiates against."""
        target = None
        for m in body.modules():
            if isinstance(m, (nn.Conv2d,)):
                target = m
        return target

    def freeze_backbone(self) -> None:
        for p in self.body.parameters():
            p.requires_grad = False

    def feature_map(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)

    def forward(self, x: torch.Tensor, return_features: bool = False):
        h = self.body(x)
        z = self.head(h)
        return (z, h) if return_features else z

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


# --------------------------------------------------------------------------- factory
@dataclass
class ModelSpec:
    """Everything needed to rebuild a model from its saved artifact."""

    name: str
    input_size: int = 64
    in_channels: int = 3
    num_classes: int = 1
    pretrained: bool = False
    dropout: float = 0.15
    width: int = 24
    freeze_backbone: bool = False

    def to_dict(self) -> Dict[str, object]:
        return dict(self.__dict__)

    @staticmethod
    def from_dict(d: Dict[str, object]) -> "ModelSpec":
        return ModelSpec(**{k: v for k, v in d.items() if k in ModelSpec.__dataclass_fields__})


def build_model(spec: ModelSpec) -> nn.Module:
    if spec.name == "lightcnn":
        return LightCNN(width=spec.width, num_classes=spec.num_classes, in_ch=spec.in_channels, dropout=spec.dropout)
    return TorchVisionNet(
        spec.name,
        num_classes=spec.num_classes,
        pretrained=spec.pretrained,
        dropout=spec.dropout,
        freeze_backbone=spec.freeze_backbone,
    )


CANDIDATE_PRESETS: List[Dict[str, object]] = [
    {
        "id": "lightcnn_32",
        "family": "deep",
        "spec": {"name": "lightcnn", "input_size": 32, "in_channels": 3, "dropout": 0.15, "width": 24},
        "epochs": 20,
        "batch_size": 128,
        "lr": 2.2e-3,
        "note": "native 32x32 input, no upsampling; cheapest deep candidate",
    },
    {
        "id": "mobilenet_v3_small_64",
        "family": "deep",
        "spec": {"name": "mobilenet_v3_small", "input_size": 64, "dropout": 0.2},
        "epochs": 12,
        "batch_size": 48,
        "lr": 1.6e-3,
        "note": "lightweight transfer candidate; 64px because sources are 32px crops (see resolution ablation)",
    },
    {
        "id": "efficientnet_b0_64",
        "family": "deep",
        "spec": {"name": "efficientnet_b0", "input_size": 64, "dropout": 0.3},
        "epochs": 8,
        "batch_size": 48,
        "lr": 1.4e-3,
        "note": "compound-scaled transfer candidate at reduced input",
    },
    {
        "id": "resnet18_64",
        "family": "deep",
        "spec": {"name": "resnet18", "input_size": 64, "dropout": 0.15},
        "epochs": 8,
        "batch_size": 48,
        "lr": 1.4e-3,
        "note": "highest-capacity candidate; slowest per epoch on 2 CPU cores",
    },
]


def classical_presets() -> List[Dict[str, object]]:
    """Sklearn candidates on the forensic descriptor (and one on raw pixels)."""
    return [
        {"id": "logreg_forensic", "model": "logreg", "feature": "forensic", "C": 1.0, "note": "linear baseline on forensic features"},
        {"id": "svm_forensic", "model": "svm", "feature": "forensic", "C": 2.0, "note": "RBF SVM, class-balanced"},
        {"id": "rf_forensic", "model": "rf", "feature": "forensic", "n_estimators": 250, "note": "random forest, 250 trees"},
        {"id": "gbdt_forensic", "model": "gbdt", "feature": "forensic", "n_estimators": 250, "note": "histogram gradient boosting"},
        {"id": "knn_forensic", "model": "knn", "feature": "forensic", "note": "k-NN (k=15) on PCA-100 forensic space"},
        {"id": "logreg_pixels", "model": "logreg", "feature": "pixels", "C": 1.0, "note": "linear baseline on raw downsampled pixels"},
        {"id": "rf_pixels", "model": "rf", "feature": "pixels", "note": "random forest on raw pixels"},
    ]
