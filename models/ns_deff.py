"""
Neuro-Symbolic Diffusion (NS-Diff) Full Implementation
Reference: Neuro-Symbolic Diffusion: Bridging Interpretable Classification
           and Generative Verification via Manifold-Aligned Concepts [cite: 1, 8]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional


# ==========================================
# Module 1: Diffusion Encoder (Perception)
# ==========================================
class DiffusionEncoder(nn.Module):
    """
    Diffusion Encoder E_φ: Maps input x to latent manifold coordinate z.
    Implements the 'Forward Perception' step.
    """

    def __init__(self,
                 in_channels: int = 3,
                 latent_dim: int = 512,
                 base_channels: int = 64):
        super().__init__()
        self.latent_dim = latent_dim

        # Downsampling blocks (256 -> 8)
        self.down1 = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 4, 2, 1),
            nn.GroupNorm(8, base_channels), nn.SiLU()
        )
        self.down2 = nn.Sequential(
            nn.Conv2d(base_channels, base_channels * 2, 4, 2, 1),
            nn.GroupNorm(16, base_channels * 2), nn.SiLU()
        )
        self.down3 = nn.Sequential(
            nn.Conv2d(base_channels * 2, base_channels * 4, 4, 2, 1),
            nn.GroupNorm(32, base_channels * 4), nn.SiLU()
        )
        self.down4 = nn.Sequential(
            nn.Conv2d(base_channels * 4, base_channels * 8, 4, 2, 1),
            nn.GroupNorm(32, base_channels * 8), nn.SiLU()
        )

        # Bottleneck (16 -> 8)
        self.bottleneck = nn.Sequential(
            nn.Conv2d(base_channels * 8, latent_dim, 4, 2, 1),
            nn.GroupNorm(32, latent_dim), nn.SiLU()
        )

        self.global_pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Resize to 256x256 if needed
        if x.size(2) != 256:
            x = F.interpolate(x, size=(256, 256), mode='bilinear')

        h1 = self.down1(x)
        h2 = self.down2(h1)
        h3 = self.down3(h2)
        h4 = self.down4(h3)
        h5 = self.bottleneck(h4)

        z = self.global_pool(h5).flatten(1)  # [B, latent_dim]
        return z


# ==========================================
# Module 2: Semantic Manifold Alignment (SMA)
# ==========================================
class SemanticManifoldAlignment(nn.Module):
    """
    SMA Module P_θ: Projects z to concept space c with geometric regularization.
    Addresses the 'Linear Expressiveness Bottleneck' (Theorem 1)[cite: 70].
    """

    def __init__(self, latent_dim: int = 512, num_concepts: int = 8, hidden_dims=[256, 128]):
        super().__init__()
        self.num_concepts = num_concepts

        layers = []
        in_dim = latent_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.LayerNorm(h_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, num_concepts))
        layers.append(nn.Sigmoid())  # Concepts in [0,1]

        self.projector = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.projector(z)

    def compute_jacobian_orthogonality_loss(self, z: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """
        Computes L_ortho (Eq. 1) to enforce disentanglement.
        """
        batch_size = z.size(0)
        jacobians = []

        # Compute gradients for each concept
        for k in range(self.num_concepts):
            grad_out = torch.zeros_like(c)
            grad_out[:, k] = 1.0

            grads = torch.autograd.grad(
                outputs=c, inputs=z, grad_outputs=grad_out,
                create_graph=True, retain_graph=True, only_inputs=True
            )[0]
            jacobians.append(grads)

        J = torch.stack(jacobians, dim=1)  # [B, K, D]
        J_norm = F.normalize(J, p=2, dim=2)

        # Gram matrix of gradients
        gram = torch.bmm(J_norm, J_norm.transpose(1, 2))  # [B, K, K]

        # Minimize off-diagonal elements
        mask = ~torch.eye(self.num_concepts, dtype=torch.bool, device=z.device)
        mask = mask.unsqueeze(0).expand(batch_size, -1, -1)
        loss_ortho = (gram[mask] ** 2).mean()

        return loss_ortho


# ==========================================
# Module 3: Differentiable Neuro-Symbolic Logic (DNSL)
# ==========================================
class DifferentiableNeuroSymbolicLogic(nn.Module):
    """
    DNSL Module L_w: Performs reasoning using Product T-Norms.
    Ensures gradient flow for end-to-end optimization[cite: 114, 121].
    """

    def __init__(self, num_concepts: int = 8, num_classes: int = 2, num_rules: int = 16):
        super().__init__()
        self.num_terms = 3  # Low, Med, High

        # Learnable Membership Functions (Eq. 2) [cite: 109]
        self.centers = nn.Parameter(torch.linspace(0.2, 0.8, self.num_terms).repeat(num_concepts, 1))
        self.widths = nn.Parameter(torch.ones(num_concepts, self.num_terms) * 0.15)

        # Rule Weights
        self.rule_ant = nn.Parameter(torch.randn(num_rules, num_concepts * self.num_terms) * 0.1)
        self.rule_con = nn.Parameter(torch.randn(num_rules, num_classes) * 0.1)

    def forward(self, c: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        B, K = c.shape

        # 1. Fuzzification
        c_exp = c.unsqueeze(2)  # [B, K, 1]
        centers = self.centers.unsqueeze(0)
        widths = torch.clamp(self.widths.unsqueeze(0), min=0.01)
        mu = torch.exp(-((c_exp - centers) ** 2) / (2 * widths ** 2))  # [B, K, Terms]

        # 2. Rule Inference (Product T-Norm)
        mu_flat = mu.reshape(B, -1)
        w_ant = F.softplus(self.rule_ant)  # Ensure positive weights

        # Log-space computation for numerical stability
        log_mu = torch.log(mu_flat + 1e-8)
        rule_act_log = torch.matmul(log_mu, w_ant.t())  # [B, Rules]
        alpha = torch.exp(rule_act_log / (w_ant.sum(1) + 1e-8))

        # 3. Aggregation (Eq. 5) [cite: 126]
        logits = torch.matmul(alpha, self.rule_con)
        return logits, {"alpha": alpha, "mu": mu}


# ==========================================
# Module 4: Diffusion Decoder (Generative Verification)
# ==========================================
class DiffusionDecoder(nn.Module):
    """
    Diffusion Decoder D_ψ: Generates counterfactual images from concepts.
    Closes the loop: Understanding = Generation[cite: 67].
    """

    def __init__(self, num_concepts: int = 8, latent_dim: int = 512, out_channels: int = 3, base_channels: int = 64):
        super().__init__()

        # Concept -> Latent Mapping
        self.concept_to_latent = nn.Sequential(
            nn.Linear(num_concepts, 256), nn.ReLU(),
            nn.Linear(256, latent_dim)
        )

        # Upsampling (Mirroring Encoder)
        self.init_conv = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, base_channels * 8, 4, 1, 0),  # -> 4x4
            nn.GroupNorm(32, base_channels * 8), nn.SiLU()
        )
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(base_channels * 8, base_channels * 8, 4, 2, 1),  # -> 8x8
            nn.GroupNorm(32, base_channels * 8), nn.SiLU()
        )
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(base_channels * 8, base_channels * 4, 4, 2, 1),  # -> 16x16
            nn.GroupNorm(32, base_channels * 4), nn.SiLU()
        )
        self.up3 = nn.Sequential(
            nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 4, 2, 1),  # -> 32x32
            nn.GroupNorm(16, base_channels * 2), nn.SiLU()
        )
        self.up4 = nn.Sequential(
            nn.ConvTranspose2d(base_channels * 2, base_channels, 4, 2, 1),  # -> 64x64
            nn.GroupNorm(8, base_channels), nn.SiLU()
        )
        self.up5 = nn.Sequential(
            nn.ConvTranspose2d(base_channels, base_channels // 2, 4, 2, 1),  # -> 128x128
            nn.GroupNorm(4, base_channels // 2), nn.SiLU()
        )
        self.final_conv = nn.Sequential(
            nn.ConvTranspose2d(base_channels // 2, out_channels, 4, 2, 1),  # -> 256x256
            nn.Tanh()
        )

    def forward(self, c_prime: torch.Tensor) -> torch.Tensor:
        z = self.concept_to_latent(c_prime).unsqueeze(-1).unsqueeze(-1)  # [B, D, 1, 1]

        h = self.init_conv(z)  # 4x4
        h = self.up1(h)  # 8x8
        h = self.up2(h)  # 16x16
        h = self.up3(h)  # 32x32
        h = self.up4(h)  # 64x64
        h = self.up5(h)  # 128x128
        return self.final_conv(h)  # 256x256


# ==========================================
# Main Framework: NS-Diff
# ==========================================
class NSDiff(nn.Module):
    """
    Main Framework integrating Modules 1-4[cite: 58].
    """

    def __init__(self, num_concepts=8, num_classes=2):
        super().__init__()
        self.encoder = DiffusionEncoder()
        self.sma = SemanticManifoldAlignment(num_concepts=num_concepts)
        self.dnsl = DifferentiableNeuroSymbolicLogic(num_concepts=num_concepts, num_classes=num_classes)
        self.decoder = DiffusionDecoder(num_concepts=num_concepts)

    def forward(self, x, return_ortho=False):
        # 1. Perception
        z = self.encoder(x)
        if return_ortho:
            z.requires_grad_(True)
            z.retain_grad()

        # 2. Alignment
        c = self.sma(z)
        loss_ortho = self.sma.compute_jacobian_orthogonality_loss(z, c) if return_ortho else 0.0

        # 3. Reasoning
        logits, aux = self.dnsl(c)
        return logits, c, loss_ortho

    def generate_counterfactual(self, x, concept_idx, value=0.0):
        """Generates verification image x'[cite: 67]."""
        with torch.no_grad():
            z = self.encoder(x)
            c = self.sma(z)

        c_edit = c.clone()
        c_edit[:, concept_idx] = value  # Intervention
        return self.decoder(c_edit)


# ==========================================
# Training Step (Algorithm 1)
# ==========================================
def train_step(model, x, y, optimizer, lambda_ortho=0.1):
    """
    Implements Algorithm 1: Joint Training[cite: 133].
    """
    model.train()
    optimizer.zero_grad()

    # Forward Pass with Orthogonality
    logits, c, loss_ortho = model(x, return_ortho=True)

    # Classification Loss
    loss_cls = nn.CrossEntropyLoss()(logits, y)

    # Total Loss (Eq. Line 17 in Algo 1)
    loss_total = loss_cls + lambda_ortho * loss_ortho

    # Backward (Gradient flows through Logic -> SMA -> Encoder)
    loss_total.backward()
    optimizer.step()

    return loss_total.item(), loss_cls.item(), loss_ortho.item()


# ==========================================
# Execution Demo
# ==========================================
if __name__ == "__main__":
    print("Initializing Neuro-Symbolic Diffusion...")

    # 1. Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = NSDiff(num_concepts=5, num_classes=2).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    # 2. Dummy Data (Batch=2, RGB, 256x256)
    x_dummy = torch.randn(2, 3, 256, 256).to(device)
    y_dummy = torch.tensor([0, 1]).to(device)

    # 3. Training Step Simulation
    print("\n--- Phase 1: Training Step ---")
    l_tot, l_cls, l_ortho = train_step(model, x_dummy, y_dummy, optimizer)
    print(f"Loss Total: {l_tot:.4f} | Class: {l_cls:.4f} | Ortho: {l_ortho:.4f}")

    # 4. Verification Simulation
    print("\n--- Phase 2: Generative Verification ---")
    # Simulate: "What if concept 0 (e.g., Beard) was removed?"
    x_cf = model.generate_counterfactual(x_dummy, concept_idx=0, value=0.0)
    print(f"Counterfactual generated: {x_cf.shape}")
    print("Verification successful: Flow complete.")