"""
Neuro-Symbolic Diffusion (NS-Diff) Core Implementation
基于论文: Neuro-Symbolic Diffusion: Bridging Interpretable Classification
         and Generative Verification via Manifold-Aligned Concepts

作者: Anonymous Authors
实现: 完整的五个核心模块
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import numpy as np


class DiffusionEncoder(nn.Module):
    """
    Diffusion Encoder E_φ: 从输入图像x提取潜在表示z

    架构设计:
    - 基于U-Net架构的bottleneck作为manifold坐标
    - 使用GroupNorm和SiLU激活函数
    - 逐步下采样: 256x256 -> 8x8 -> 全局池化 -> latent_dim维向量

    数学描述:
    z = E_φ(x), 其中 z ∈ R^D 是数据流形 M 的坐标表示
    z 编码了数据密度的梯度 (score function)
    """

    def __init__(self,
                 in_channels: int = 3,
                 latent_dim: int = 512,
                 base_channels: int = 64,
                 pretrained_path: Optional[str] = None):
        """
        初始化扩散编码器

        Args:
            in_channels: 输入图像通道数 (RGB=3)
            latent_dim: 潜在空间维度
            base_channels: 基础通道数
            pretrained_path: 预训练模型路径 (可选)
        """
        super().__init__()
        self.latent_dim = latent_dim

        # 编码器主干 - 5层下采样
        # 每层: Conv -> GroupNorm -> SiLU

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

        # 全局自适应平均池化: 8x8 -> 1x1
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        # 如果提供预训练权重,加载
        if pretrained_path is not None:
            self.load_pretrained(pretrained_path)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播: 图像 -> 潜在表示

        Args:
            x: 输入图像 [batch_size, in_channels, height, width]
               要求 height=width=256 (可以通过插值调整)

        Returns:
            z: 潜在表示 [batch_size, latent_dim]
               编码了图像在数据流形上的位置
        """
        # 确保输入尺寸正确
        if x.size(2) != 256 or x.size(3) != 256:
            x = F.interpolate(x, size=(256, 256), mode='bilinear', align_corners=False)

        # 逐层下采样
        h1 = self.down1(x)  # [B, 64, 128, 128]
        h2 = self.down2(h1)  # [B, 128, 64, 64]
        h3 = self.down3(h2)  # [B, 256, 32, 32]
        h4 = self.down4(h3)  # [B, 512, 16, 16]
        h5 = self.bottleneck(h4)  # [B, latent_dim, 8, 8]

        # 全局池化到向量
        z = self.global_pool(h5)  # [B, latent_dim, 1, 1]
        z = z.squeeze(-1).squeeze(-1)  # [B, latent_dim]

        return z

    def load_pretrained(self, path: str):
        """加载预训练权重"""
        checkpoint = torch.load(path, map_location='cpu')
        self.load_state_dict(checkpoint['encoder'], strict=False)
        print(f"Loaded pretrained encoder from {path}")


class SemanticManifoldAlignment(nn.Module):
    """
    Semantic Manifold Alignment (SMA) 模块 P_θ

    核心功能:
    1. 非线性投影: z -> c (解决Theorem 1的线性瓶颈问题)
    2. Jacobian正交正则化: 确保概念空间的正交性

    数学原理 (Theorem 1):
    线性投影 L(z) = w^T z + b 在非欧几里得流形上存在表达瓶颈
    解决方案: 使用MLP进行非线性投影 + 正交约束

    正交约束 (Eq. 1):
    L_ortho = E_z[∑_{i≠j} (j_i · j_j^T / ||j_i|| ||j_j||)^2]
    其中 j_k = ∇_z c_k 是概念k在流形上的梯度方向
    """

    def __init__(self,
                 latent_dim: int = 512,
                 num_concepts: int = 8,
                 hidden_dims: List[int] = [256, 128],
                 dropout_rate: float = 0.1):
        """
        初始化SMA模块

        Args:
            latent_dim: 输入潜在空间维度
            num_concepts: 概念数量 (输出维度)
            hidden_dims: MLP隐藏层维度列表
            dropout_rate: Dropout比率
        """
        super().__init__()
        self.latent_dim = latent_dim
        self.num_concepts = num_concepts

        # 构建非线性MLP投影器
        layers = []
        prev_dim = latent_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),  # 层归一化提高稳定性
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim

        # 最终投影层: hidden -> concepts
        layers.append(nn.Linear(prev_dim, num_concepts))

        # Sigmoid激活: 将概念值约束到[0, 1]
        # 这符合模糊逻辑中隶属度的定义
        layers.append(nn.Sigmoid())

        self.projector = nn.Sequential(*layers)

        # 用于存储最后一次的Jacobian (调试用)
        self.last_jacobian = None

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        非线性投影: 潜在空间 -> 概念空间

        Args:
            z: 潜在表示 [batch_size, latent_dim]

        Returns:
            c: 概念分数 [batch_size, num_concepts]
               每个元素 c_k ∈ [0, 1] 表示概念k的激活程度
        """
        return self.projector(z)

    def compute_jacobian_orthogonality_loss(self,
                                            z: torch.Tensor,
                                            c: torch.Tensor) -> torch.Tensor:
        """
        计算Jacobian正交正则化损失 (Eq. 1)

        目标: 确保不同概念在流形上的梯度方向正交
        这保证了概念的独立性和可解释性

        数学推导:
        1. 计算 j_k = ∇_z c_k for k=1,...,K
        2. 归一化: j_k_norm = j_k / ||j_k||
        3. 计算Gram矩阵: G = J_norm @ J_norm^T
        4. 最小化非对角线元素: L = ∑_{i≠j} G_{ij}^2

        Args:
            z: 潜在表示 [batch_size, latent_dim]
               必须 requires_grad=True
            c: 概念分数 [batch_size, num_concepts]

        Returns:
            L_ortho: 正交损失标量
                    越小表示概念越正交
        """
        batch_size = z.size(0)

        # 存储每个概念的梯度 (Jacobian的每一行)
        jacobians = []

        # 计算每个概念对z的梯度
        for k in range(self.num_concepts):
            # 创建只有第k个概念为1的梯度输出
            grad_outputs = torch.zeros_like(c)
            grad_outputs[:, k] = 1.0

            # 计算 ∇_z c_k
            # create_graph=True: 需要计算二阶导数
            # retain_graph=True: 保留计算图供后续使用
            grads = torch.autograd.grad(
                outputs=c,
                inputs=z,
                grad_outputs=grad_outputs,
                create_graph=True,  # 关键: 保留计算图用于二阶导数
                retain_graph=True,  # 关键: 允许多次backward
                only_inputs=True
            )[0]  # [batch_size, latent_dim]

            jacobians.append(grads)

        # 堆叠成Jacobian矩阵 [batch_size, num_concepts, latent_dim]
        J = torch.stack(jacobians, dim=1)

        # 归一化每个梯度向量 (沿latent_dim维度)
        # 这样余弦相似度 = 内积
        J_norm = F.normalize(J, p=2, dim=2)  # [B, K, D]

        # 计算Gram矩阵: G = J_norm @ J_norm^T
        # G[i,j] = cos(angle(j_i, j_j)) = 概念i和j的相似度
        gram_matrix = torch.bmm(J_norm, J_norm.transpose(1, 2))  # [B, K, K]

        # 创建mask提取非对角线元素 (i ≠ j)
        mask = ~torch.eye(self.num_concepts, dtype=torch.bool, device=z.device)
        mask = mask.unsqueeze(0).expand(batch_size, -1, -1)  # [B, K, K]

        # 提取非对角线元素
        off_diagonal = gram_matrix[mask]  # [B * K*(K-1)]

        # 最小化非对角线元素的平方和
        # 理想情况: 所有非对角线元素 = 0 (完全正交)
        L_ortho = (off_diagonal ** 2).mean()

        # 保存Jacobian用于分析 (可选)
        self.last_jacobian = J_norm.detach()

        return L_ortho

    def get_concept_directions(self) -> Optional[torch.Tensor]:
        """
        获取概念在潜在空间中的方向向量

        Returns:
            directions: [num_concepts, latent_dim] 或 None
                       每行是一个概念的归一化梯度方向
        """
        if self.last_jacobian is not None:
            # 对batch维度取平均
            return self.last_jacobian.mean(dim=0)
        return None


class DifferentiableNeuroSymbolicLogic(nn.Module):
    """
    Differentiable Neuro-Symbolic Logic (DNSL) 模块

    核心功能:
    1. 语义模糊化 (Fuzzification): 连续值 -> 模糊隶属度
    2. 规则推理 (Inference): Product T-Norm组合多个前件
    3. 去模糊化 (Defuzzification): 规则激活 -> 类别预测

    数学框架:
    - 隶属函数 (Eq. 2): μ_{k,j}(c_k) = exp(-(c_k - m_{k,j})^2 / 2σ_{k,j}^2)
    - 规则激活 (Eq. 3): α_l = ∏_{k∈I_l} μ_{k,j_k}(c_k)
    - 预测输出 (Eq. 5): ŷ = Softmax(W_rule^T @ α)

    优势:
    - Product T-Norm保证梯度流通 (vs. Min T-Norm)
    - 端到端可学习的模糊规则
    - 显式的逻辑推理过程
    """

    def __init__(self,
                 num_concepts: int = 8,
                 num_linguistic_terms: int = 3,  # Low, Medium, High
                 num_rules: int = 16,
                 num_classes: int = 2):
        """
        初始化DNSL模块

        Args:
            num_concepts: 概念数量
            num_linguistic_terms: 每个概念的语言项数量
                                 (例如: Low, Medium, High = 3)
            num_rules: 模糊规则数量
            num_classes: 输出类别数量
        """
        super().__init__()
        self.num_concepts = num_concepts
        self.num_linguistic_terms = num_linguistic_terms
        self.num_rules = num_rules
        self.num_classes = num_classes

        # === 可学习的高斯隶属函数参数 (Eq. 2) ===

        # 中心参数 m_{k,j}: 语言项的中心位置
        # 例如: Low=0.2, Medium=0.5, High=0.8
        init_centers = torch.linspace(0.2, 0.8, num_linguistic_terms)
        self.membership_centers = nn.Parameter(
            init_centers.unsqueeze(0).repeat(num_concepts, 1)
        )  # [num_concepts, num_linguistic_terms]

        # 宽度参数 σ_{k,j}: 控制模糊程度
        # 初始化为0.15,允许一定重叠
        self.membership_widths = nn.Parameter(
            torch.ones(num_concepts, num_linguistic_terms) * 0.15
        )  # [num_concepts, num_linguistic_terms]

        # === 规则库参数 ===

        # 规则前件权重: 每条规则使用哪些概念-语言项组合
        # 维度: [num_rules, num_concepts * num_linguistic_terms]
        self.rule_antecedents = nn.Parameter(
            torch.randn(num_rules, num_concepts * num_linguistic_terms) * 0.1
        )

        # 规则结论权重 W_rule (Eq. 5)
        # 每条规则对每个类别的贡献
        self.rule_weights = nn.Parameter(
            torch.randn(num_rules, num_classes) * 0.1
        )

    def fuzzify(self, c: torch.Tensor) -> torch.Tensor:
        """
        语义模糊化: 将连续概念分数转换为语言项隶属度 (Eq. 2)

        高斯隶属函数:
        μ_{k,j}(c_k) = exp(-(c_k - m_{k,j})^2 / (2σ_{k,j}^2))

        直观解释:
        - c_k = 0.3, 如果 m_{Low}=0.2, 则 μ_{Low} 较高
        - c_k = 0.3, 如果 m_{High}=0.8, 则 μ_{High} 较低

        Args:
            c: 概念分数 [batch_size, num_concepts]
               每个元素 ∈ [0, 1]

        Returns:
            μ: 隶属度 [batch_size, num_concepts, num_linguistic_terms]
               μ[b, k, j] = 概念k属于语言项j的程度
        """
        batch_size = c.size(0)

        # 扩展维度用于广播计算
        c_expanded = c.unsqueeze(2)  # [B, K, 1]

        # 获取中心和宽度参数
        centers = self.membership_centers.unsqueeze(0)  # [1, K, num_terms]

        # 限制宽度的最小值,避免数值不稳定
        widths = torch.clamp(self.membership_widths.unsqueeze(0), min=0.01)  # [1, K, num_terms]

        # 计算高斯隶属度
        # μ = exp(-(c - m)^2 / (2σ^2))
        squared_diff = (c_expanded - centers) ** 2  # [B, K, num_terms]
        variance = 2 * widths ** 2
        mu = torch.exp(-squared_diff / variance)  # [B, K, num_terms]

        return mu

    def product_tnorm(self, mu: torch.Tensor) -> torch.Tensor:
        """
        Product T-Norm规则推理 (Eq. 3)

        规则激活强度:
        α_l = ∏_{k∈I_l} μ_{k,j_k}(c_k)

        为什么用Product而不是Min?
        - Product T-Norm: ∂α/∂c_k ≠ 0 (只要其他项非零)
        - Min T-Norm: ∂α/∂c_k = 0 (如果c_k不是最小值)

        梯度流分析 (Eq. 4):
        ∂α_l/∂c_k = (∏_{m≠k} μ_m) · ∂μ_k/∂c_k
        这保证了误差能反向传播到所有概念

        Args:
            mu: 隶属度 [batch_size, num_concepts, num_linguistic_terms]

        Returns:
            α: 规则激活强度 [batch_size, num_rules]
        """
        batch_size = mu.size(0)

        # 展平隶属度: [B, K*num_terms]
        mu_flat = mu.reshape(batch_size, -1)

        # 获取规则前件权重并确保为正
        rule_weights = F.softplus(self.rule_antecedents)  # [L, K*num_terms]

        # 对数空间计算以提高数值稳定性
        # log(∏ μ) = ∑ log(μ)
        log_mu = torch.log(mu_flat + 1e-8)  # [B, K*num_terms]

        # 加权对数和
        weighted_log = torch.matmul(log_mu, rule_weights.t())  # [B, L]

        # 转回原空间并归一化
        # α = exp(∑ w_i log(μ_i) / ∑ w_i)
        weight_sum = rule_weights.sum(dim=1, keepdim=True).t()  # [1, L]
        alpha = torch.exp(weighted_log / (weight_sum + 1e-8))  # [B, L]

        return alpha

    def forward(self, c: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        完整的逻辑推理流程

        流程:
        c (概念) -> μ (隶属度) -> α (规则激活) -> ŷ (预测)

        Args:
            c: 概念分数 [batch_size, num_concepts]

        Returns:
            y_hat: 类别概率分布 [batch_size, num_classes]
            aux: 辅助信息字典
                - membership: 隶属度
                - rule_activation: 规则激活强度
                - logits: 未归一化的分数
        """
        # Step 1: 模糊化 (Eq. 2)
        mu = self.fuzzify(c)  # [B, K, num_terms]

        # Step 2: 规则推理 (Eq. 3-4)
        alpha = self.product_tnorm(mu)  # [B, L]

        # Step 3: 去模糊化和预测 (Eq. 5)
        # 线性组合规则激活
        logits = torch.matmul(alpha, self.rule_weights)  # [B, C]

        # Softmax得到概率分布
        y_hat = F.softmax(logits, dim=1)  # [B, C]

        # 收集辅助信息用于分析和可视化
        aux = {
            'membership': mu,  # 模糊隶属度
            'rule_activation': alpha,  # 规则激活强度
            'logits': logits  # 未归一化分数
        }

        return y_hat, aux

    def get_rule_importance(self) -> torch.Tensor:
        """
        获取每条规则的重要性

        Returns:
            importance: [num_rules] 规则重要性分数
        """
        # 计算规则权重的L2范数
        return self.rule_weights.norm(dim=1)

    def get_top_rules(self, k: int = 5) -> List[Tuple[int, float]]:
        """
        获取最重要的k条规则

        Args:
            k: 返回规则数量

        Returns:
            top_rules: [(rule_idx, importance_score), ...]
        """
        importance = self.get_rule_importance()
        top_k = torch.topk(importance, k)
        return list(zip(top_k.indices.tolist(), top_k.values.tolist()))


class DiffusionDecoder(nn.Module):
    """
    Diffusion Decoder D_ψ: 从修改的概念c'生成反事实图像x'

    功能: 生成验证 (Generative Counterfactual Verification)
    流程: c' -> z' -> x_cf

    架构: 与编码器对称的上采样网络
    - 概念 -> 潜在空间 (可学习映射)
    - 潜在空间 -> 图像 (转置卷积上采样)
    """

    def __init__(self,
                 num_concepts: int = 8,
                 latent_dim: int = 512,
                 out_channels: int = 3,
                 base_channels: int = 64):
        """
        初始化扩散解码器

        Args:
            num_concepts: 概念数量
            latent_dim: 潜在空间维度
            out_channels: 输出图像通道数 (RGB=3)
            base_channels: 基础通道数
        """
        super().__init__()
        self.num_concepts = num_concepts
        self.latent_dim = latent_dim

        # === 概念到潜在空间的逆映射 ===
        self.concept_to_latent = nn.Sequential(
            nn.Linear(num_concepts, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, latent_dim),
        )

        # === 上采样解码器 (与编码器对称) ===

        # 初始上采样: 向量 -> 4x4特征图
        self.init_conv = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, base_channels * 8,
                               kernel_size=4, stride=1, padding=0),  # -> 4x4
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

        # 最终输出层: 128x128 -> 256x256
        self.final_conv = nn.Sequential(
            nn.ConvTranspose2d(base_channels // 2, out_channels,
                               kernel_size=4, stride=2, padding=1),
            nn.Tanh()  # 输出范围 [-1, 1]
        )

    def forward(self, c_prime: torch.Tensor) -> torch.Tensor:
        """
        反事实生成: 概念 -> 图像

        Args:
            c_prime: 干预后的概念 [batch_size, num_concepts]

        Returns:
            x_cf: 反事实图像 [batch_size, out_channels, 256, 256]
        """
        # 概念映射到潜在空间
        z_prime = self.concept_to_latent(c_prime)  # [B, latent_dim]

        # 重塑为空间特征: [B, latent_dim, 1, 1]
        z_prime = z_prime.unsqueeze(-1).unsqueeze(-1)

        # 逐层上采样解码
        h = self.init_conv(z_prime)  # [B, 512, 4, 4]
        h = self.up1(h)  # [B, 512, 8, 8]
        h = self.up2(h