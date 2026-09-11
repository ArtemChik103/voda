"""Track A: Deep Learning Multi-Modal Architecture and Hydrological Loss Functions.

Implements:
1. Multi-Modal Attention UNet (15-channel input, Gated Cross-Modal Attention, Modality Dropout).
2. Focal Loss for extreme class imbalance (alpha=0.75, gamma=2.0).
3. Lovasz-Softmax / Lovasz-Hinge Loss for direct Jaccard/IoU optimization.
4. Boundary Loss for waterline contour sharpness.
5. CombinedHydrologicalLoss (0.35 Focal + 0.45 Lovasz + 0.20 Boundary).
"""

from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# 1. Custom Hydrological Loss Functions
# ---------------------------------------------------------------------------

class FocalLoss(nn.Module):
    """Binary Focal Loss to address extreme ~1% class imbalance."""

    def __init__(self, alpha: float = 0.75, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        probs = torch.sigmoid(logits)
        p_t = probs * targets + (1.0 - probs) * (1.0 - targets)
        alpha_t = self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)
        focal_weight = alpha_t * torch.pow(1.0 - p_t, self.gamma)
        loss = focal_weight * bce

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class LovaszLoss(nn.Module):
    """Lovasz-Hinge surrogate loss for direct IoU maximization on binary masks."""

    def __init__(self):
        super().__init__()

    @staticmethod
    def _lovasz_grad(gt_sorted: torch.Tensor) -> torch.Tensor:
        """Computes gradient of the Lovasz extension for sorted errors."""
        p = len(gt_sorted)
        gts = gt_sorted.sum()
        intersection = gts - gt_sorted.float().cumsum(0)
        union = gts + (1 - gt_sorted).float().cumsum(0)
        jaccard = 1.0 - intersection / union
        if p > 1:
            jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
        return jaccard

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Args: logits (B, 1, H, W) or (B, H, W), targets (B, 1, H, W) or (B, H, W)."""
        logits = logits.reshape(-1)
        targets = targets.reshape(-1)

        signs = 2.0 * targets.float() - 1.0
        errors = 1.0 - logits * signs
        errors_sorted, perm = torch.sort(errors, dim=0, descending=True)
        perm = perm.data
        gt_sorted = targets[perm]
        grad = self._lovasz_grad(gt_sorted)
        loss = torch.dot(F.relu(errors_sorted), grad)
        return loss


class BoundaryLoss(nn.Module):
    """Edge/Boundary loss penalizing discrepancies along the waterline perimeter."""

    def __init__(self, kernel_size: int = 3):
        super().__init__()
        # Laplacian kernel for edge extraction
        laplacian = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        self.register_buffer("kernel", laplacian)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if logits.dim() == 3:
            logits = logits.unsqueeze(1)
        if targets.dim() == 3:
            targets = targets.unsqueeze(1)

        probs = torch.sigmoid(logits)
        # Compute boundary gradients
        pred_edges = torch.abs(F.conv2d(probs, self.kernel, padding=1))
        target_edges = torch.abs(F.conv2d(targets.float(), self.kernel, padding=1))
        return F.l1_loss(pred_edges, target_edges)


class CombinedHydrologicalLoss(nn.Module):
    """Combined loss function: 0.35 Focal + 0.45 Lovasz + 0.20 Boundary."""

    def __init__(self, w_focal: float = 0.35, w_lovasz: float = 0.45, w_boundary: float = 0.20):
        super().__init__()
        self.w_focal = w_focal
        self.w_lovasz = w_lovasz
        self.w_boundary = w_boundary

        self.focal = FocalLoss(alpha=0.75, gamma=2.0)
        self.lovasz = LovaszLoss()
        self.boundary = BoundaryLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> Dict[str, torch.Tensor]:
        l_focal = self.focal(logits, targets)
        l_lovasz = self.lovasz(logits, targets)
        l_boundary = self.boundary(logits, targets)

        total_loss = (
            self.w_focal * l_focal
            + self.w_lovasz * l_lovasz
            + self.w_boundary * l_boundary
        )

        return {
            "total_loss": total_loss,
            "focal_loss": l_focal,
            "lovasz_loss": l_lovasz,
            "boundary_loss": l_boundary,
        }


# ---------------------------------------------------------------------------
# 2. Multi-Modal Neural Network Architecture
# ---------------------------------------------------------------------------

class GatedCrossModalAttention(nn.Module):
    """Dynamically weights SAR and Optical feature streams based on cloud validity."""

    def __init__(self, sar_channels: int = 6, opt_channels: int = 5):
        super().__init__()
        # Squeeze-and-Excitation channel gating
        self.gate_fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(sar_channels + opt_channels, 16),
            nn.ReLU(inplace=True),
            nn.Linear(16, opt_channels),
            nn.Sigmoid(),
        )

    def forward(self, sar_feat: torch.Tensor, opt_feat: torch.Tensor) -> torch.Tensor:
        combined = torch.cat([sar_feat, opt_feat], dim=1)
        gate = self.gate_fc(combined).unsqueeze(-1).unsqueeze(-1)  # (B, opt_ch, 1, 1)
        gated_opt = opt_feat * gate
        return torch.cat([sar_feat, gated_opt], dim=1)


class ConvBlock(nn.Module):
    """Standard double convolution block with BatchNorm and GELU."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class MultiModalHydrologyNet(nn.Module):
    """Multi-Modal UNet with Gated Cross-Modal Attention and Modality Dropout.

    Takes a 15-channel multi-modal tensor:
      - Channels 0-5: SAR (Sentinel-1)
      - Channels 6-10: Optics (Sentinel-2)
      - Channels 11-14: Auxiliary (Slope, HAND, GSW, Builtup)
    Produces 2 heads:
      - Head 0: Water surface logits (water_peak)
      - Head 1: New flood inundation logits (flood)
    """

    def __init__(
        self,
        in_channels: int = 15,
        num_classes: int = 2,
        base_filters: int = 32,
        modality_dropout_prob: float = 0.40,
    ):
        super().__init__()
        self.modality_dropout_prob = modality_dropout_prob

        # Cross-modal attention module (SAR: 6 ch, Optics: 5 ch)
        self.attention = GatedCrossModalAttention(sar_channels=6, opt_channels=5)

        # Total fused input channels: 6 (SAR) + 5 (gated Optics) + 4 (Aux) = 15
        self.inc = ConvBlock(in_channels, base_filters)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), ConvBlock(base_filters, base_filters * 2))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), ConvBlock(base_filters * 2, base_filters * 4))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), ConvBlock(base_filters * 4, base_filters * 8))

        # Bottleneck with dilated convs for receptive field
        self.bottleneck = nn.Sequential(
            nn.Conv2d(base_filters * 8, base_filters * 8, kernel_size=3, padding=2, dilation=2, bias=False),
            nn.BatchNorm2d(base_filters * 8),
            nn.GELU(),
        )

        # Decoder
        self.up1 = nn.ConvTranspose2d(base_filters * 8, base_filters * 4, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(base_filters * 8, base_filters * 4)

        self.up2 = nn.ConvTranspose2d(base_filters * 4, base_filters * 2, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(base_filters * 4, base_filters * 2)

        self.up3 = nn.ConvTranspose2d(base_filters * 2, base_filters, kernel_size=2, stride=2)
        self.dec3 = ConvBlock(base_filters * 2, base_filters)

        # Multi-task Prediction Heads:
        # Head 1: Water surface probability
        self.head_water = nn.Conv2d(base_filters, 1, kernel_size=1)
        # Head 2: Flood inundation probability
        self.head_flood = nn.Conv2d(base_filters, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Args: x: Tensor of shape (B, 15, H, W)."""
        # Modality Dropout during training: simulate 100% cloudiness
        if self.training and self.modality_dropout_prob > 0.0:
            if torch.rand(1).item() < self.modality_dropout_prob:
                # Zero out optical indices (channels 6..9) and set cloud flag (channel 10) to 1.0
                x = x.clone()
                x[:, 6:10, :, :] = 0.0
                x[:, 10, :, :] = 1.0

        sar_stream = x[:, 0:6, :, :]
        opt_stream = x[:, 6:11, :, :]
        aux_stream = x[:, 11:15, :, :]

        # Cross-modal gating
        fused_so = self.attention(sar_stream, opt_stream)
        fused_input = torch.cat([fused_so, aux_stream], dim=1)

        # UNet Forward
        x1 = self.inc(fused_input)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)

        b = self.bottleneck(x4)

        d1 = self.up1(b)
        d1 = self.dec1(torch.cat([d1, x3], dim=1))

        d2 = self.up2(d1)
        d2 = self.dec2(torch.cat([d2, x2], dim=1))

        d3 = self.up3(d2)
        d3 = self.dec3(torch.cat([d3, x1], dim=1))

        logits_water = self.head_water(d3)
        logits_flood = self.head_flood(d3)

        return {
            "logits_water": logits_water,
            "logits_flood": logits_flood,
            "prob_water": torch.sigmoid(logits_water),
            "prob_flood": torch.sigmoid(logits_flood),
        }
