"""
Baseline Models Implementation
包括: ResNet-50, Standard CBM, Post-hoc CBM, DisDiff-FNNC
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from typing import Dict, Optional


class ResNet50BlackBox(nn.Module):
    """
    Black-Box Baseline: 标准ResNet-50分类器
    直接从图像预测类别,没有可解释性
    """

    def __init__(self,
                 num_classes: int = 2,
                 pretrained: bool = True):
        super().__init__()

        # 加载预训练ResNet-50
        self.backbone = models.resnet50(pretrained=pretrained)

        # 替换最后的全连接层
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        直接分类
        Args:
            x: 输入图像 [B, 3, H, W]
        Returns:
            logits: 类别logits [B, num_classes]
        """
        return self.backbone(x)


class StandardCBM(nn.Module):
    """
    Standard Concept Bottleneck Model (Koh et al., ICML 2020)
    架构: Encoder -> Linear Concept Projection -> Classifier

    存在的问题 (Theorem 1): 线性投影无法捕获非欧几里得流形
    """

    def __init__(self,
                 num_concepts: int = 8,
                 num_classes: int = 2,
                 backbone: str = 'resnet50',
                 pretrained: bool = True):
        super().__init__()

        # 特征提取器
        if backbone == 'resnet50':
            resnet = models.resnet50(pretrained=pretrained)
            # 移除最后的全连接层
            self.encoder = nn.Sequential(*list(resnet.children())[:-1])
            feature_dim = 2048
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        self.feature_dim = feature_dim
        self.num_concepts = num_concepts
        self.num_classes = num_classes

        # 线性概念投影 (这是CBM的核心瓶颈)
        self.concept_projector = nn.Linear(feature_dim, num_concepts)

        # 从概念到类别的分类器
        self.classifier = nn.Sequential(
            nn.Linear(num_concepts, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )

    def forward(self,
                x: torch.Tensor,
                return_concepts: bool = False) -> Dict[str, torch.Tensor]:
        """
        前向传播: x -> features -> concepts -> predictions

        Args:
            x: 输入图像 [B, 3, H, W]
            return_concepts: 是否返回概念表示
        Returns:
            outputs: 包含预测和概念的字典
        """
        # 提取特征
        features = self.encoder(x)
        features = features.view(features.size(0), -1)  # [B, feature_dim]

        # 线性投影到概念空间
        concepts = torch.sigmoid(self.concept_projector(features))  # [B, num_concepts]

        # 基于概念的分类
        logits = self.classifier(concepts)

        outputs = {
            'predictions': F.softmax(logits, dim=1),
            'logits': logits,
            'concepts': concepts,
            'features': features
        }

        return outputs


class PostHocCBM(nn.Module):
    """
    Post-hoc Concept Bottleneck Model (Yuksekgonul et al., ICLR 2023)
    将预训练的黑盒模型转换为CBM

    关键思想: 使用线性探针从冻结的特征中提取概念
    """

    def __init__(self,
                 num_concepts: int = 8,
                 num_classes: int = 2,
                 backbone: str = 'resnet50',
                 freeze_backbone: bool = True):
        super().__init__()

        # 预训练的特征提取器 (冻结)
        if backbone == 'resnet50':
            resnet = models.resnet50(pretrained=True)
            self.encoder = nn.Sequential(*list(resnet.children())[:-1])
            feature_dim = 2048
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        if freeze_backbone:
            for param in self.encoder.parameters():
                param.requires_grad = False

        self.feature_dim = feature_dim
        self.num_concepts = num_concepts

        # Post-hoc概念探针 (线性层)
        self.concept_probes = nn.ModuleList([
            nn.Linear(feature_dim, 1) for _ in range(num_concepts)
        ])

        # 残差连接: 同时使用原始特征和概念
        self.residual_weight = nn.Parameter(torch.tensor(0.5))

        # 最终分类器
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim + num_concepts, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self,
                x: torch.Tensor,
                return_concepts: bool = False) -> Dict[str, torch.Tensor]:
        """
        Post-hoc CBM前向传播

        Args:
            x: 输入图像 [B, 3, H, W]
            return_concepts: 是否返回概念
        Returns:
            outputs: 预测和概念字典
        """
        # 提取冻结的特征
        with torch.no_grad() if not self.training else torch.enable_grad():
            features = self.encoder(x)
            features = features.view(features.size(0), -1)

        # 使用线性探针提取概念
        concepts = []
        for probe in self.concept_probes:
            concept_score = torch.sigmoid(probe(features))
            concepts.append(concept_score)
        concepts = torch.cat(concepts, dim=1)  # [B, num_concepts]

        # 组合原始特征和概念 (残差连接)
        alpha = torch.sigmoid(self.residual_weight)
        combined = torch.cat([
            features * (1 - alpha),
            concepts * alpha
        ], dim=1)

        # 分类
        logits = self.classifier(combined)

        outputs = {
            'predictions': F.softmax(logits, dim=1),
            'logits': logits,
            'concepts': concepts,
            'features': features
        }

        return outputs


class DisDiffFNNC(nn.Module):
    """
    Disentangled Diffusion with Fuzzy Neural Network Classifier
    这是NS-Diff的简化版本,作为生成式基线

    主要区别:
    - 使用冻结的扩散特征 (不进行end-to-end训练)
    - 没有SMA模块的Jacobian正交正则化
    - 使用标准模糊神经网络而非DNSL
    """

    def __init__(self,
                 latent_dim: int = 512,
                 num_concepts: int = 8,
                 num_classes: int = 2):
        super().__init__()

        # 简化的扩散编码器 (冻结)
        self.encoder = DiffusionEncoderSimple(latent_dim)
        for param in self.encoder.parameters():
            param.requires_grad = False

        # 线性概念投影 (无正交约束)
        self.concept_projector = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, num_concepts),
            nn.Sigmoid()
        )

        # 简单的模糊神经网络
        self.fuzzy_classifier = SimpleFuzzyNN(num_concepts, num_classes)

    def forward(self,
                x: torch.Tensor,
                return_concepts: bool = False) -> Dict[str, torch.Tensor]:
        """
        前向传播

        Args:
            x: 输入图像 [B, 3, H, W]
            return_concepts: 是否返回概念
        Returns:
            outputs: 预测字典
        """
        # 提取冻结的扩散特征
        with torch.no_grad():
            z = self.encoder(x)

        # 概念投影
        concepts = self.concept_projector(z)

        # 模糊分类
        logits = self.fuzzy_classifier(concepts)

        outputs = {
            'predictions': F.softmax(logits, dim=1),
            'logits': logits,
            'concepts': concepts
        }

        return outputs


class DiffusionEncoderSimple(nn.Module):
    """简化的扩散编码器用于DisDiff基线"""

    def __init__(self, latent_dim: int = 512):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(256, latent_dim, 4, 2, 1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )

    def forward(self, x):
        return self.encoder(x).squeeze(-1).squeeze(-1)


class SimpleFuzzyNN(nn.Module):
    """简化的模糊神经网络分类器"""

    def __init__(self, num_concepts: int, num_classes: int):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(num_concepts, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes)
        )

    def forward(self, concepts):
        return self.fc(concepts)


def build_model(model_name: str,
                num_concepts: int = 8,
                num_classes: int = 2,
                **kwargs) -> nn.Module:
    """
    模型工厂函数

    Args:
        model_name: 模型名称
        num_concepts: 概念数量
        num_classes: 类别数量
    Returns:
        model: 构建的模型
    """
    models_dict = {
        'resnet50': ResNet50BlackBox,
        'standard_cbm': StandardCBM,
        'posthoc_cbm': PostHocCBM,
        'disdiff_fnnc': DisDiffFNNC
    }

    if model_name not in models_dict:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(models_dict.keys())}")

    model_class = models_dict[model_name]

    # 根据模型类型传递参数
    if model_name == 'resnet50':
        return model_class(num_classes=num_classes, **kwargs)
    else:
        return model_class(num_concepts=num_concepts, num_classes=num_classes, **kwargs)