"""
Neuro-Symbolic Diffusion (NS-Diff) Core Implementation
基于论文: Neuro-Symbolic Diffusion: Bridging Interpretable Classification
         and Generative Verification via Manifold-Aligned Concepts

完整实现五个核心类:
1. DiffusionEncoder - 扩散编码器 E_φ
2. SemanticManifoldAlignment - 语义流形对齐 P_θ
3. DifferentiableNeuroSymbolicLogic - 可微神经符号逻辑
4. DiffusionDecoder - 扩散解码器 D_ψ
5. NSDiff - 完整框架

📝 使用示例
# 创建模型
model = NSDiff(num_concepts=8, num_classes=2)

# 前向传播
x = torch.randn(4, 3, 256, 256)
outputs = model(x)  # 包含predictions, concepts等

# 计算损失
y = torch.randint(0, 2, (4,))
losses = model.compute_total_loss(x, y, lambda_cls=1.0, lambda_ortho=0.1)

# 生成反事实
x_cf, info = model.generate_counterfactual(x[:1], target_concept_idx=0, target_value=1.0)

# 解释预测
explanation = model.explain_prediction(x[:1], concept_names=['Bangs', 'Beard', ...])
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import numpy as np


class DiffusionEncoder(nn.Module):
    """
    Diffusion Encoder E_φ

    功能: 从输入图像x提取潜在表示z
    架构: U-Net风格的编码器，bottleneck作为流形坐标
    输出: z ∈ R^D，编码数据流形的score function
    """

    def __init__(self,
                 in_channels: int = 3,
                 latent_dim: int = 512,
                 base_channels: int = 64,
                 pretrained_path: Optional[str] = None):
        super().__init__()
        self.latent_dim = latent_dim

        # 下采样块1: 256x256 -> 128x128
        self.down1 = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(8, base_channels),
            nn.SiLU()
        )

        # 下采样块2: 128x128 -> 64x64
        self.down2 = nn.Sequential(
            nn.Conv2d(base_channels, base_channels * 2, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(16, base_channels * 2),
            nn.SiLU()
        )

        # 下采样块3: 64x64 -> 32x32
        self.down3 = nn.Sequential(
            nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(32, base_channels * 4),
            nn.SiLU()
        )

        # 下采样块4: 32x32 -> 16x16
        self.down4 = nn.Sequential(
            nn.Conv2d(base_channels * 4, base_channels * 8, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(32, base_channels * 8),
            nn.SiLU()
        )

        # Bottleneck: 16x16 -> 8x8
        self.bottleneck = nn.Sequential(
            nn.Conv2d(base_channels * 8, latent_dim, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(32, latent_dim),
            nn.SiLU()
        )

        # 全局池化
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        if pretrained_path:
            self._load_pretrained(pretrained_path)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        Args:
            x: [B, C, H, W]
        Returns:
            z: [B, latent_dim]
        """
        if x.size(2) != 256 or x.size(3) != 256:
            x = F.interpolate(x, size=(256, 256), mode='bilinear', align_corners=False)

        h1 = self.down1(x)
        h2 = self.down2(h1)
        h3 = self.down3(h2)
        h4 = self.down4(h3)
        h5 = self.bottleneck(h4)

        z = self.global_pool(h5).squeeze(-1).squeeze(-1)
        return z

    def _load_pretrained(self, path: str):
        checkpoint = torch.load(path, map_location='cpu')
        self.load_state_dict(checkpoint.get('encoder', checkpoint), strict=False)


class SemanticManifoldAlignment(nn.Module):
    """
    Semantic Manifold Alignment (SMA) Module P_θ

    功能:
    1. 非线性投影 z -> c (解决Theorem 1的线性瓶颈)
    2. Jacobian正交正则化 (确保概念独立性)

    关键公式 (Eq. 1):
    L_ortho = E_z[∑_{i≠j} (j_i · j_j^T / ||j_i|| ||j_j||)^2]
    其中 j_k = ∇_z c_k
    """

    def __init__(self,
                 latent_dim: int = 512,
                 num_concepts: int = 8,
                 hidden_dims: List[int] = [256, 128],
                 dropout_rate: float = 0.1):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_concepts = num_concepts

        # 构建MLP投影器
        layers = []
        prev_dim = latent_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, num_concepts))
        layers.append(nn.Sigmoid())

        self.projector = nn.Sequential(*layers)
        self.last_jacobian = None

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        投影: z -> c
        Args:
            z: [B, latent_dim]
        Returns:
            c: [B, num_concepts], 每个元素 ∈ [0, 1]
        """
        return self.projector(z)

    def compute_jacobian_orthogonality_loss(self,
                                            z: torch.Tensor,
                                            c: torch.Tensor) -> torch.Tensor:
        """
        计算Jacobian正交正则化损失 (Eq. 1)

        实现步骤:
        1. 计算 j_k = ∇_z c_k for each k
        2. 归一化梯度向量
        3. 计算Gram矩阵 G = J @ J^T
        4. 最小化非对角线元素

        Args:
            z: [B, latent_dim], requires_grad=True
            c: [B, num_concepts]
        Returns:
            L_ortho: scalar
        """
        batch_size = z.size(0)
        jacobians = []

        for k in range(self.num_concepts):
            grad_outputs = torch.zeros_like(c)
            grad_outputs[:, k] = 1.0

            grads = torch.autograd.grad(
                outputs=c,
                inputs=z,
                grad_outputs=grad_outputs,
                create_graph=True,
                retain_graph=True,
                only_inputs=True
            )[0]

            jacobians.append(grads)

        J = torch.stack(jacobians, dim=1)  # [B, K, D]
        J_norm = F.normalize(J, p=2, dim=2)

        gram_matrix = torch.bmm(J_norm, J_norm.transpose(1, 2))  # [B, K, K]

        mask = ~torch.eye(self.num_concepts, dtype=torch.bool, device=z.device)
        off_diagonal = gram_matrix[:, mask].reshape(batch_size, -1)

        L_ortho = (off_diagonal ** 2).mean()

        self.last_jacobian = J_norm.detach()
        return L_ortho


class DifferentiableNeuroSymbolicLogic(nn.Module):
    """
    Differentiable Neuro-Symbolic Logic (DNSL) Module

    实现可微的模糊逻辑推理:
    1. 模糊化 (Eq. 2): c -> μ (高斯隶属函数)
    2. 规则推理 (Eq. 3): μ -> α (Product T-Norm)
    3. 去模糊化 (Eq. 5): α -> ŷ (加权聚合)

    关键优势:
    - Product T-Norm保证梯度流通
    - 端到端可学习的模糊规则
    """

    def __init__(self,
                 num_concepts: int = 8,
                 num_linguistic_terms: int = 3,
                 num_rules: int = 16,
                 num_classes: int = 2):
        super().__init__()
        self.num_concepts = num_concepts
        self.num_linguistic_terms = num_linguistic_terms
        self.num_rules = num_rules
        self.num_classes = num_classes

        # 高斯隶属函数参数 (Eq. 2)
        init_centers = torch.linspace(0.2, 0.8, num_linguistic_terms)
        self.membership_centers = nn.Parameter(
            init_centers.unsqueeze(0).repeat(num_concepts, 1)
        )

        self.membership_widths = nn.Parameter(
            torch.ones(num_concepts, num_linguistic_terms) * 0.15
        )

        # 规则参数
        self.rule_antecedents = nn.Parameter(
            torch.randn(num_rules, num_concepts * num_linguistic_terms) * 0.1
        )

        self.rule_weights = nn.Parameter(
            torch.randn(num_rules, num_classes) * 0.1
        )

    def fuzzify(self, c: torch.Tensor) -> torch.Tensor:
        """
        语义模糊化 (Eq. 2)
        μ_{k,j}(c_k) = exp(-(c_k - m_{k,j})^2 / (2σ_{k,j}^2))

        Args:
            c: [B, num_concepts]
        Returns:
            μ: [B, num_concepts, num_linguistic_terms]
        """
        c_expanded = c.unsqueeze(2)
        centers = self.membership_centers.unsqueeze(0)
        widths = torch.clamp(self.membership_widths.unsqueeze(0), min=0.01)

        squared_diff = (c_expanded - centers) ** 2
        variance = 2 * widths ** 2
        mu = torch.exp(-squared_diff / variance)

        return mu

    def product_tnorm(self, mu: torch.Tensor) -> torch.Tensor:
        """
        Product T-Norm规则推理 (Eq. 3)
        α_l = ∏_{k∈I_l} μ_{k,j_k}(c_k)

        使用对数空间计算以提高数值稳定性

        Args:
            mu: [B, K, num_terms]
        Returns:
            α: [B, num_rules]
        """
        batch_size = mu.size(0)
        mu_flat = mu.reshape(batch_size, -1)

        rule_weights = F.softplus(self.rule_antecedents)

        log_mu = torch.log(mu_flat + 1e-8)
        weighted_log = torch.matmul(log_mu, rule_weights.t())

        weight_sum = rule_weights.sum(dim=1, keepdim=True).t()
        alpha = torch.exp(weighted_log / (weight_sum + 1e-8))

        return alpha

    def forward(self, c: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        完整逻辑推理流程

        Args:
            c: [B, num_concepts]
        Returns:
            y_hat: [B, num_classes]
            aux: 辅助信息字典
        """
        mu = self.fuzzify(c)
        alpha = self.product_tnorm(mu)

        logits = torch.matmul(alpha, self.rule_weights)
        y_hat = F.softmax(logits, dim=1)

        aux = {
            'membership': mu,
            'rule_activation': alpha,
            'logits': logits
        }

        return y_hat, aux


class DiffusionDecoder(nn.Module):
    """
    Diffusion Decoder D_ψ

    功能: 从修改的概念c'生成反事实图像x'
    架构: 概念 -> 潜在空间 -> 图像 (转置卷积上采样)
    用途: 生成验证 (Generative Counterfactual Verification)
    """

    def __init__(self,
                 num_concepts: int = 8,
                 latent_dim: int = 512,
                 out_channels: int = 3,
                 base_channels: int = 64):
        super().__init__()
        self.num_concepts = num_concepts
        self.latent_dim = latent_dim

        # 概念到潜在空间的映射
        self.concept_to_latent = nn.Sequential(
            nn.Linear(num_concepts, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, latent_dim)
        )

        # 初始卷积: 向量 -> 4x4特征图
        self.init_conv = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, base_channels * 8,
                               kernel_size=4, stride=1, padding=0),
            nn.GroupNorm(32, base_channels * 8),
            nn.SiLU()
        )

        # 上采样块1: 4x4 -> 8x8
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(base_channels * 8, base_channels * 8,
                               kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(32, base_channels * 8),
            nn.SiLU()
        )

        # 上采样块2: 8x8 -> 16x16
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(base_channels * 8, base_channels * 4,
                               kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(32, base_channels * 4),
            nn.SiLU()
        )

        # 上采样块3: 16x16 -> 32x32
        self.up3 = nn.Sequential(
            nn.ConvTranspose2d(base_channels * 4, base_channels * 2,
                               kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(16, base_channels * 2),
            nn.SiLU()
        )

        # 上采样块4: 32x32 -> 64x64
        self.up4 = nn.Sequential(
            nn.ConvTranspose2d(base_channels * 2, base_channels,
                               kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(8, base_channels),
            nn.SiLU()
        )

        # 上采样块5: 64x64 -> 128x128
        self.up5 = nn.Sequential(
            nn.ConvTranspose2d(base_channels, base_channels // 2,
                               kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(4, base_channels // 2),
            nn.SiLU()
        )

        # 最终输出: 128x128 -> 256x256
        self.final_conv = nn.Sequential(
            nn.ConvTranspose2d(base_channels // 2, out_channels,
                               kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, c_prime: torch.Tensor) -> torch.Tensor:
        """
        反事实生成
        Args:
            c_prime: [B, num_concepts]
        Returns:
            x_cf: [B, out_channels, 256, 256]
        """
        z_prime = self.concept_to_latent(c_prime)
        z_prime = z_prime.unsqueeze(-1).unsqueeze(-1)

        h = self.init_conv(z_prime)
        h = self.up1(h)
        h = self.up2(h)
        h = self.up3(h)
        h = self.up4(h)
        h = self.up5(h)
        x_cf = self.final_conv(h)

        return x_cf


class NSDiff(nn.Module):
    """
    Neuro-Symbolic Diffusion - 完整框架

    整合所有模块实现端到端的可解释分类和生成验证

    训练流程 (Algorithm 1):
    1. x -> Encoder -> z (流形感知)
    2. z -> SMA -> c (语义对齐)
    3. c -> DNSL -> ŷ (逻辑推理)
    4. 计算损失: L = λ_cls*L_cls + λ_ortho*L_ortho
    5. 周期性生成反事实验证
    """

    def __init__(self,
                 in_channels: int = 3,
                 latent_dim: int = 512,
                 num_concepts: int = 8,
                 num_classes: int = 2,
                 num_linguistic_terms: int = 3,
                 num_rules: int = 16,
                 base_channels: int = 64):
        super().__init__()

        self.num_concepts = num_concepts
        self.num_classes = num_classes

        # 四大核心模块
        self.encoder = DiffusionEncoder(
            in_channels=in_channels,
            latent_dim=latent_dim,
            base_channels=base_channels
        )

        self.sma = SemanticManifoldAlignment(
            latent_dim=latent_dim,
            num_concepts=num_concepts
        )

        self.dnsl = DifferentiableNeuroSymbolicLogic(
            num_concepts=num_concepts,
            num_linguistic_terms=num_linguistic_terms,
            num_rules=num_rules,
            num_classes=num_classes
        )

        self.decoder = DiffusionDecoder(
            num_concepts=num_concepts,
            latent_dim=latent_dim,
            out_channels=in_channels,
            base_channels=base_channels
        )

    def forward(self,
                x: torch.Tensor,
                return_concepts: bool = False) -> Dict[str, torch.Tensor]:
        """
        完整前向传播 (Algorithm 1 Phase 1-3)

        Args:
            x: [B, C, H, W]
            return_concepts: 是否返回概念
        Returns:
            outputs: 包含predictions, latent, concepts等
        """
        z = self.encoder(x)
        c = self.sma(z)
        y_hat, aux = self.dnsl(c)

        outputs = {
            'predictions': y_hat,
            'latent': z,
            'concepts': c,
            'membership': aux['membership'],
            'rule_activation': aux['rule_activation'],
            'logits': aux['logits']
        }

        return outputs

    def generate_counterfactual(self,
                                x: torch.Tensor,
                                target_concept_idx: int,
                                target_value: float) -> Tuple[torch.Tensor, Dict]:
        """
        生成反事实验证 (Algorithm 1 Phase 5)

        流程:
        1. 编码: x -> z -> c
        2. 干预: c[idx] = target_value
        3. 生成: c' -> x_cf
        4. 验证: x_cf -> c_cf

        Args:
            x: [B, C, H, W]
            target_concept_idx: 干预的概念索引
            target_value: 目标值 [0, 1]
        Returns:
            x_cf: 反事实图像
            info: 干预信息
        """
        with torch.no_grad():
            z = self.encoder(x)
            c = self.sma(z)

        c_prime = c.clone()
        c_prime[:, target_concept_idx] = target_value

        x_cf = self.decoder(c_prime)

        with torch.no_grad():
            z_cf = self.encoder(x_cf)
            c_cf = self.sma(z_cf)

        intervention_error = torch.abs(c_cf[:, target_concept_idx] - target_value).mean()

        info = {
            'original_concepts': c,
            'intervened_concepts': c_prime,
            'generated_concepts': c_cf,
            'intervention_success': intervention_error
        }

        return x_cf, info

    def compute_total_loss(self,
                           x: torch.Tensor,
                           y: torch.Tensor,
                           lambda_cls: float = 1.0,
                           lambda_ortho: float = 0.1) -> Dict[str, torch.Tensor]:
        """
        计算总损失 (Algorithm 1 Phase 4)

        L_total = λ_cls * L_cls + λ_ortho * L_ortho

        Args:
            x: [B, C, H, W]
            y: [B]
            lambda_cls: 分类损失权重
            lambda_ortho: 正交损失权重
        Returns:
            losses: 损失字典
        """
        z = self.encoder(x)
        z.requires_grad_(True)

        c = self.sma(z)
        y_hat, _ = self.dnsl(c)

        L_cls = F.cross_entropy(y_hat, y)
        L_ortho = self.sma.compute_jacobian_orthogonality_loss(z, c)

        L_total = lambda_cls * L_cls + lambda_ortho * L_ortho

        losses = {
            'total': L_total,
            'classification': L_cls,
            'orthogonality': L_ortho
        }

        return losses

    def explain_prediction(self,
                           x: torch.Tensor,
                           concept_names: Optional[List[str]] = None) -> Dict:
        """
        生成预测解释

        Args:
            x: [1, C, H, W]
            concept_names: 概念名称列表
        Returns:
            explanation: 包含预测类别、概念值、规则等
        """
        if concept_names is None:
            concept_names = [f"Concept_{i}" for i in range(self.num_concepts)]

        with torch.no_grad():
            outputs = self.forward(x)

        pred_prob, pred_class = outputs['predictions'][0].max(0)
        concept_values = outputs['concepts'][0].cpu().numpy()

        concepts_dict = {
            name: float(val)
            for name, val in zip(concept_names, concept_values)
        }

        rule_activation = outputs['rule_activation'][0]
        top_rules = torch.topk(rule_activation, k=min(5, len(rule_activation)))

        explanation = {
            'predicted_class': int(pred_class),
            'confidence': float(pred_prob),
            'concepts': concepts_dict,
            'top_rules': [
                (int(idx), float(val))
                for idx, val in zip(top_rules.indices.tolist(), top_rules.values.tolist())
            ]
        }

        return explanation


# 测试代码
if __name__ == "__main__":
    print("=" * 70)
    print("NS-Diff Module Test")
    print("=" * 70)

    # 创建模型
    print("\n[1/5] Creating NS-Diff model...")
    model = NSDiff(
        in_channels=3,
        latent_dim=512,
        num_concepts=8,
        num_classes=2
    )
    num_params = sum(p.numel() for p in model.parameters())
    print(f"✓ Model created with {num_params / 1e6:.2f}M parameters")

    # 测试前向传播
    print("\n[2/5] Testing forward pass...")
    x = torch.randn(4, 3, 256, 256)
    outputs = model(x)
    print(f"✓ Forward pass successful")
    print(f"  Predictions: {outputs['predictions'].shape}")
    print(f"  Concepts: {outputs['concepts'].shape}")

    # 测试损失计算
    print("\n[3/5] Testing loss computation...")
    y = torch.randint(0, 2, (4,))
    losses = model.compute_total_loss(x, y)
    print(f"✓ Loss computation successful")
    print(f"  Total: {losses['total'].item():.4f}")
    print(f"  Classification: {losses['classification'].item():.4f}")
    print(f"  Orthogonality: {losses['orthogonality'].item():.6f}")

    # 测试反事实生成
    print("\n[4/5] Testing counterfactual generation...")
    x_cf, info = model.generate_counterfactual(x[:1], 0, 1.0)
    print(f"✓ Counterfactual generation successful")
    print(f"  Generated shape: {x_cf.shape}")
    print(f"  Intervention error: {info['intervention_success'].item():.4f}")

    # 测试解释
    print("\n[5/5] Testing explanation...")
    concept_names = ['Bangs', 'Beard', 'Smiling', 'Male', 'Young', 'Eyeglasses', 'Wavy_Hair', 'Hat']
    explanation = model.explain_prediction(x[:1], concept_names)
    print(f"✓ Explanation generated")
    print(f"  Predicted class: {explanation['predicted_class']}")
    print(f"  Confidence: {explanation['confidence']:.4f}")

    print("\n" + "=" * 70)
    print("All tests passed! ✓")
    print("=" * 70)