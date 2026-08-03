from __future__ import annotations

from typing import NamedTuple

import torch
from torch import nn
from torch.nn import functional


class MultiTaskOutput(NamedTuple):
    tumor_logits: torch.Tensor
    integrity_logits: torch.Tensor


class ConvBlock(nn.Sequential):
    def __init__(self, input_channels: int, output_channels: int, *, double: bool = True) -> None:
        layers: list[nn.Module] = [
            nn.Conv2d(input_channels, output_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
        ]
        if double:
            layers.extend(
                [
                    nn.Conv2d(
                        output_channels,
                        output_channels,
                        kernel_size=3,
                        padding=1,
                        bias=False,
                    ),
                    nn.BatchNorm2d(output_channels),
                    nn.ReLU(inplace=True),
                ]
            )
        layers.append(nn.MaxPool2d(2))
        super().__init__(*layers)


class BaselineCNN(nn.Module):
    """Small custom CNN used as the computational baseline."""

    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Linear(32, 2)

    @property
    def gradcam_layer(self) -> nn.Module:
        return self.encoder[6]

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.encoder(images).flatten(1)
        return self.classifier(features)


class SharedEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.stage1 = ConvBlock(1, 16)
        self.stage2 = ConvBlock(16, 32)
        self.stage3 = ConvBlock(32, 64)
        self.stage4 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

    @property
    def gradcam_layer(self) -> nn.Module:
        return self.stage4[0]

    def forward_features(self, images: torch.Tensor) -> torch.Tensor:
        features = self.stage1(images)
        features = self.stage2(features)
        features = self.stage3(features)
        return self.stage4(features)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.forward_features(images)
        return self.pool(features).flatten(1)


class ForensicEncoder(nn.Module):
    """Extract local residual patterns that global semantic pooling can suppress."""

    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

    @property
    def gradcam_layer(self) -> nn.Module:
        return self.features[6]

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        residual = images - functional.avg_pool2d(images, kernel_size=5, stride=1, padding=2)
        features = self.features(residual)
        average = functional.adaptive_avg_pool2d(features, 1).flatten(1)
        maximum = functional.adaptive_max_pool2d(features, 1).flatten(1)
        return torch.cat([average, maximum], dim=1)


def _head(dropout: float, input_features: int = 128) -> nn.Sequential:
    return nn.Sequential(
        nn.Dropout(dropout),
        nn.Linear(input_features, 64),
        nn.ReLU(inplace=True),
        nn.Dropout(dropout / 2),
        nn.Linear(64, 2),
    )


class ImprovedCNN(nn.Module):
    def __init__(self, dropout: float = 0.30) -> None:
        super().__init__()
        self.encoder = SharedEncoder()
        self.classifier = _head(dropout)

    @property
    def gradcam_layer(self) -> nn.Module:
        return self.encoder.gradcam_layer

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.encoder(images))


class MultiTaskCNN(nn.Module):
    def __init__(self, dropout: float = 0.30) -> None:
        super().__init__()
        self.encoder = SharedEncoder()
        self.forensic_encoder = ForensicEncoder()
        self.tumor_head = _head(dropout)
        self.integrity_head = _head(dropout, input_features=192)

    @property
    def gradcam_layer(self) -> nn.Module:
        return self.encoder.gradcam_layer

    @property
    def integrity_gradcam_layer(self) -> nn.Module:
        return self.forensic_encoder.gradcam_layer

    def forward(self, images: torch.Tensor) -> MultiTaskOutput:
        semantic_features = self.encoder(images)
        forensic_features = self.forensic_encoder(images)
        return MultiTaskOutput(
            tumor_logits=self.tumor_head(semantic_features),
            integrity_logits=self.integrity_head(
                torch.cat([semantic_features, forensic_features], dim=1)
            ),
        )


def build_model(model_name: str, dropout: float = 0.30) -> nn.Module:
    if model_name == "baseline":
        return BaselineCNN()
    if model_name == "improved":
        return ImprovedCNN(dropout=dropout)
    if model_name == "multitask":
        return MultiTaskCNN(dropout=dropout)
    raise ValueError(f"Unknown model_name: {model_name}")


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def trainable_parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
