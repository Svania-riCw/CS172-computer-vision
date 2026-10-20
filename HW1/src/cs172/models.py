"""Neural-network models for CAPTCHA recognition."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F
from torchvision import models


class TorchvisionResNet(nn.Module):
    """A ResNet backbone with one ten-class head for each digit position."""

    def __init__(self, num_digits: int = 5, num_classes: int = 10, **kwargs):
        super().__init__()
        del kwargs
        self.num_digits = num_digits
        self.num_classes = num_classes

        self.backbone = models.resnet18(weights=None)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, num_digits * num_classes)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        logits = self.backbone(images)
        return logits.reshape(images.shape[0], self.num_classes, self.num_digits)

    @staticmethod
    def compute_loss(logits: torch.Tensor, targets: torch.Tensor):
        """Compute cross-entropy over classes and all five positions."""
        return F.cross_entropy(logits, targets), None


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, channels, stride=1, projection=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, channels, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.projection = projection

    def forward(self, x):
        identity = x
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.projection is not None:
            identity = self.projection(x)
        return torch.relu(out + identity)


class CAPTCHAResNet(nn.Module):
    """ResNet-18 variant that preserves spatial features before classification."""

    def __init__(self, num_digits=5, num_classes=10, image_size=(80, 260), **kwargs):
        super().__init__()
        del kwargs
        self.num_digits = num_digits
        self.num_classes = num_classes
        self.in_channels = 64
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        self.layer1 = self._make_layer(64, 2)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.layer4 = self._make_layer(512, 2, stride=2)
        with torch.no_grad():
            features = self._features(torch.zeros(1, 3, *image_size))
        self.classifier = nn.Linear(features.flatten(1).shape[1], num_digits * num_classes)

    def _make_layer(self, channels, blocks, stride=1):
        projection = None
        if stride != 1 or self.in_channels != channels:
            projection = nn.Sequential(
                nn.Conv2d(self.in_channels, channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(channels),
            )
        layers = [BasicBlock(self.in_channels, channels, stride, projection)]
        self.in_channels = channels
        for _ in range(1, blocks):
            layers.append(BasicBlock(self.in_channels, channels))
        return nn.Sequential(*layers)

    def _features(self, images):
        x = self.stem(images)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return self.layer4(x)

    def forward(self, images):
        logits = self.classifier(self._features(images).flatten(1))
        return logits.reshape(images.shape[0], self.num_classes, self.num_digits)

    @staticmethod
    def compute_loss(logits, targets):
        return F.cross_entropy(logits, targets), None


class SpatialSequenceNet(nn.Module):
    """A width-preserving CNN with one classifier for each ordered position."""

    def __init__(self, num_digits=5, num_classes=10, **kwargs):
        super().__init__()
        del kwargs
        self.num_digits = num_digits
        self.num_classes = num_classes
        self.features = nn.Sequential(
            self._block(3, 32, stride=(2, 1)),
            self._block(32, 64, stride=(2, 1)),
            self._block(64, 128, stride=(2, 1)),
            self._block(128, 192, stride=(2, 1)),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, num_digits))
        self.head = nn.Sequential(
            nn.LayerNorm(192),
            nn.Linear(192, 128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, num_classes),
        )

    @staticmethod
    def _block(in_channels, out_channels, stride):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )

    def forward(self, images):
        features = self.features(images)
        pooled = self.pool(features).squeeze(2).transpose(1, 2)
        logits = self.head(pooled)
        return logits.transpose(1, 2)

    @staticmethod
    def compute_loss(logits, targets):
        return F.cross_entropy(logits, targets), None


MODELS = {
    "resnet18": TorchvisionResNet,
    "captcha_resnet18": CAPTCHAResNet,
    "spatial_sequence": SpatialSequenceNet,
}


def create_model(name: str = "resnet18", **kwargs) -> nn.Module:
    """Construct a model by its command-line name."""
    try:
        model_class = MODELS[name]
    except KeyError as error:
        choices = ", ".join(sorted(MODELS))
        raise ValueError(f"unknown model {name!r}; choose from: {choices}") from error
    return model_class(**kwargs)

