"""Siamese U-Net architecture for satellite imagery change detection.

The model processes two co-registered satellite images (pre-change T1 and
post-change T2) through a shared encoder, fuses their features, and decodes
a binary change mask.

Key architectural choices:
- Siamese (weight-sharing) encoder ensures both time steps are embedded
  in the same feature space, making feature differences meaningful.
- ResNet-34 backbone provides strong pretrained features that transfer
  well from natural images to satellite imagery.
- Multi-scale feature fusion (difference + concatenation) captures both
  "what changed" and "what was there before/after."
- Deep supervision provides gradient signal at every decoder level,
  helping the encoder learn useful features at all spatial scales.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class ConvBlock(nn.Module):
    """Double convolution block used in the U-Net decoder.

    Two 3x3 convolutions with BatchNorm and ReLU, following the
    original U-Net design. BatchNorm helps with training stability
    especially when using mixed precision.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through double convolution.

        Args:
            x: Input tensor of shape (B, C_in, H, W).

        Returns:
            Output tensor of shape (B, C_out, H, W).
        """
        return self.block(x)


class FeatureFusion(nn.Module):
    """Fuse features from two time steps at a given spatial scale.

    Computes both absolute difference |F1 - F2| (captures magnitude of
    change regardless of direction) and concatenation [F1, F2] (preserves
    full context of both time steps). The fused features are compressed
    through a 1x1 convolution.

    Args:
        in_channels: Number of channels per input feature map.
        fusion_mode: "diff" for difference only, "concat" for concat only,
            "both" for difference + concatenation (recommended).
    """

    def __init__(self, in_channels: int, fusion_mode: str = "both") -> None:
        super().__init__()
        self.fusion_mode = fusion_mode

        if fusion_mode == "diff":
            fused_channels = in_channels
        elif fusion_mode == "concat":
            fused_channels = in_channels * 2
        elif fusion_mode == "both":
            # Difference (C channels) + Concatenation (2C channels) = 3C
            fused_channels = in_channels * 3
        else:
            raise ValueError(f"Unknown fusion mode: {fusion_mode}")

        # 1x1 conv to compress fused features back to original channel count
        self.compress = nn.Sequential(
            nn.Conv2d(fused_channels, in_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, f1: torch.Tensor, f2: torch.Tensor) -> torch.Tensor:
        """Fuse feature maps from two time steps.

        Args:
            f1: Features from T1 image, shape (B, C, H, W).
            f2: Features from T2 image, shape (B, C, H, W).

        Returns:
            Fused features, shape (B, C, H, W).
        """
        if self.fusion_mode == "diff":
            fused = torch.abs(f1 - f2)
        elif self.fusion_mode == "concat":
            fused = torch.cat([f1, f2], dim=1)
        else:  # "both"
            diff = torch.abs(f1 - f2)
            fused = torch.cat([diff, f1, f2], dim=1)

        return self.compress(fused)


class ResNetEncoder(nn.Module):
    """ResNet-34 encoder adapted for satellite imagery.

    Extracts multi-scale features at 5 spatial resolutions (1/1, 1/2,
    1/4, 1/8, 1/16 of input). The first conv layer is modified to
    accept an arbitrary number of input channels (not just 3 for RGB).

    For pretrained models with non-RGB input, the first layer weights
    are replicated across extra channels and averaged, which provides
    a reasonable initialization.

    Args:
        in_channels: Number of input spectral bands.
        pretrained: Whether to load ImageNet-pretrained weights.
    """

    def __init__(self, in_channels: int = 3, pretrained: bool = True) -> None:
        super().__init__()

        weights = models.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
        resnet = models.resnet34(weights=weights)

        # Modify first conv layer for arbitrary number of input channels
        if in_channels != 3:
            original_conv = resnet.conv1
            new_conv = nn.Conv2d(
                in_channels,
                64,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=False,
            )
            if pretrained:
                # Replicate pretrained RGB weights across extra channels
                # and normalize so the output magnitude is preserved
                with torch.no_grad():
                    weight = original_conv.weight.data
                    if in_channels > 3:
                        repeats = (in_channels + 2) // 3
                        weight = weight.repeat(1, repeats, 1, 1)[:, :in_channels, :, :]
                        weight = weight * (3.0 / in_channels)
                    else:
                        weight = weight[:, :in_channels, :, :]
                    new_conv.weight.data = weight
            resnet.conv1 = new_conv

        # Extract encoder stages
        # Stage 0: conv1 + bn1 + relu (stride 2, output = H/2)
        self.stage0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu)
        # Stage 1: maxpool + layer1 (stride 2, output = H/4)
        self.stage1 = nn.Sequential(resnet.maxpool, resnet.layer1)
        # Stage 2: layer2 (stride 2, output = H/8)
        self.stage2 = resnet.layer2
        # Stage 3: layer3 (stride 2, output = H/16)
        self.stage3 = resnet.layer3
        # Stage 4: layer4 (stride 2, output = H/32)
        self.stage4 = resnet.layer4

        # Channel counts at each stage (for ResNet-34)
        self.channels = [64, 64, 128, 256, 512]

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Extract multi-scale features.

        Args:
            x: Input image tensor (B, C, H, W).

        Returns:
            List of 5 feature maps at decreasing spatial resolutions.
        """
        f0 = self.stage0(x)  # (B, 64,  H/2,  W/2)
        f1 = self.stage1(f0)  # (B, 64,  H/4,  W/4)
        f2 = self.stage2(f1)  # (B, 128, H/8,  W/8)
        f3 = self.stage3(f2)  # (B, 256, H/16, W/16)
        f4 = self.stage4(f3)  # (B, 512, H/32, W/32)
        return [f0, f1, f2, f3, f4]


class DecoderBlock(nn.Module):
    """Single decoder stage: upsample, concatenate skip, apply convolutions.

    Uses transposed convolution for learnable upsampling (better than
    bilinear interpolation for recovering fine spatial details in change
    masks).

    Args:
        in_channels: Channels from the previous decoder level.
        skip_channels: Channels from the corresponding encoder skip connection.
        out_channels: Output channels for this decoder level.
    """

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        self.upsample = nn.ConvTranspose2d(
            in_channels,
            in_channels,
            kernel_size=2,
            stride=2,
        )
        self.conv = ConvBlock(in_channels + skip_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        """Upsample, concatenate skip connection, and apply convolutions.

        Handles spatial dimension mismatches by padding if necessary
        (can happen when input dimensions aren't exact powers of 2).

        Args:
            x: Feature map from deeper decoder level (B, C_in, H, W).
            skip: Skip connection from encoder (B, C_skip, 2H, 2W).

        Returns:
            Decoded feature map (B, C_out, 2H, 2W).
        """
        x = self.upsample(x)

        # Handle spatial dimension mismatch from non-power-of-2 inputs
        if x.shape[2:] != skip.shape[2:]:
            diff_h = skip.shape[2] - x.shape[2]
            diff_w = skip.shape[3] - x.shape[3]
            x = F.pad(x, [0, diff_w, 0, diff_h])

        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class SiameseUNet(nn.Module):
    """Siamese U-Net for bi-temporal satellite image change detection.

    Architecture overview:
    1. Shared ResNet-34 encoder processes T1 and T2 independently
    2. Feature fusion combines T1 and T2 features at each scale
    3. U-Net decoder with skip connections produces change mask
    4. Optional deep supervision for multi-scale loss computation

    Args:
        in_channels: Number of spectral bands per input image.
        num_classes: Number of output classes (1 for binary change).
        pretrained: Use ImageNet-pretrained encoder weights.
        fusion_mode: Feature fusion strategy ("diff", "concat", "both").
        deep_supervision: Enable auxiliary outputs at each decoder scale.
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 1,
        pretrained: bool = True,
        fusion_mode: str = "both",
        deep_supervision: bool = True,
    ) -> None:
        super().__init__()
        self.deep_supervision = deep_supervision

        # Shared encoder (Siamese = same weights for both images)
        self.encoder = ResNetEncoder(in_channels, pretrained)
        channels = self.encoder.channels  # [64, 64, 128, 256, 512]

        # Feature fusion modules at each encoder scale
        self.fusions = nn.ModuleList([FeatureFusion(c, fusion_mode) for c in channels])

        # Decoder path
        self.decoder4 = DecoderBlock(channels[4], channels[3], channels[3])
        self.decoder3 = DecoderBlock(channels[3], channels[2], channels[2])
        self.decoder2 = DecoderBlock(channels[2], channels[1], channels[1])
        self.decoder1 = DecoderBlock(channels[1], channels[0], channels[0])

        # Final upsampling to match input resolution
        # (encoder's first stage has stride 2)
        self.final_upsample = nn.ConvTranspose2d(
            channels[0],
            channels[0],
            kernel_size=2,
            stride=2,
        )

        # Classification head
        self.classifier = nn.Conv2d(channels[0], num_classes, kernel_size=1)

        # Deep supervision heads (auxiliary classifiers at each scale)
        if deep_supervision:
            self.aux_classifiers = nn.ModuleList(
                [
                    nn.Conv2d(channels[0], num_classes, kernel_size=1),  # scale 1
                    nn.Conv2d(channels[1], num_classes, kernel_size=1),  # scale 2
                    nn.Conv2d(channels[2], num_classes, kernel_size=1),  # scale 3
                    nn.Conv2d(channels[3], num_classes, kernel_size=1),  # scale 4
                ]
            )

    def forward(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Forward pass for bi-temporal change detection.

        Args:
            x1: Pre-change image tensor (B, C, H, W).
            x2: Post-change image tensor (B, C, H, W).

        Returns:
            Dictionary containing:
            - "pred": Main prediction logits (B, 1, H, W).
            - "aux_preds": List of auxiliary predictions (only during
              training with deep_supervision=True).
        """
        # Encode both time steps through the shared encoder
        features1 = self.encoder(x1)
        features2 = self.encoder(x2)

        # Fuse features at each scale
        fused = [
            fusion(f1, f2)
            for fusion, f1, f2 in zip(self.fusions, features1, features2, strict=False)
        ]

        # Decode with skip connections from fused features
        d4 = self.decoder4(fused[4], fused[3])
        d3 = self.decoder3(d4, fused[2])
        d2 = self.decoder2(d3, fused[1])
        d1 = self.decoder1(d2, fused[0])

        # Upsample to original resolution
        d0 = self.final_upsample(d1)

        # Handle spatial mismatch with input
        if d0.shape[2:] != x1.shape[2:]:
            d0 = F.interpolate(
                d0, size=x1.shape[2:], mode="bilinear", align_corners=False
            )

        # Main prediction
        pred = self.classifier(d0)

        output = {"pred": pred}

        # Deep supervision: auxiliary predictions at each decoder scale
        if self.deep_supervision and self.training:
            target_size = x1.shape[2:]
            aux_preds = [
                F.interpolate(
                    clf(feat), size=target_size, mode="bilinear", align_corners=False
                )
                for clf, feat in zip(
                    self.aux_classifiers, [d1, d2, d3, d4], strict=False
                )
            ]
            output["aux_preds"] = aux_preds

        return output


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------


class DiceLoss(nn.Module):
    """Soft Dice Loss for binary segmentation.

    Dice coefficient measures the overlap between prediction and target.
    It is particularly useful for change detection because it handles
    class imbalance naturally -- it focuses on the overlap with the
    minority class (changed pixels) rather than overall accuracy.

    The "soft" version uses sigmoid probabilities instead of hard
    thresholding, making it differentiable for backpropagation.

    Args:
        smooth: Small constant to avoid division by zero.
    """

    def __init__(self, smooth: float = 1.0) -> None:
        super().__init__()
        self.smooth = smooth

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute soft Dice loss.

        Args:
            pred: Raw logits (before sigmoid), shape (B, 1, H, W).
            target: Binary ground truth, shape (B, 1, H, W).

        Returns:
            Scalar Dice loss (1 - Dice coefficient).
        """
        pred = torch.sigmoid(pred)
        pred_flat = pred.contiguous().view(-1)
        target_flat = target.contiguous().view(-1)

        intersection = (pred_flat * target_flat).sum()
        dice = (2.0 * intersection + self.smooth) / (
            pred_flat.sum() + target_flat.sum() + self.smooth
        )
        return 1.0 - dice


class BCEDiceLoss(nn.Module):
    """Combined Binary Cross-Entropy and Dice Loss.

    BCE provides stable per-pixel gradients (good for optimization),
    while Dice directly optimizes the F1-like metric we care about.
    The combination gives faster convergence and better final metrics
    than either loss alone.

    Args:
        bce_weight: Weight for the BCE component.
        dice_weight: Weight for the Dice component.
    """

    def __init__(self, bce_weight: float = 0.5, dice_weight: float = 0.5) -> None:
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute weighted sum of BCE and Dice losses.

        Args:
            pred: Raw logits, shape (B, 1, H, W).
            target: Binary ground truth, shape (B, 1, H, W).

        Returns:
            Scalar combined loss.
        """
        return self.bce_weight * self.bce(pred, target) + self.dice_weight * self.dice(
            pred, target
        )


def build_model(
    in_channels: int = 3,
    num_classes: int = 1,
    pretrained: bool = True,
    fusion_mode: str = "both",
    deep_supervision: bool = True,
) -> SiameseUNet:
    """Factory function to create a SiameseUNet model.

    Args:
        in_channels: Number of input spectral bands.
        num_classes: Number of output classes.
        pretrained: Use ImageNet-pretrained encoder.
        fusion_mode: Feature fusion strategy.
        deep_supervision: Enable multi-scale auxiliary losses.

    Returns:
        Initialized SiameseUNet model.
    """
    return SiameseUNet(
        in_channels=in_channels,
        num_classes=num_classes,
        pretrained=pretrained,
        fusion_mode=fusion_mode,
        deep_supervision=deep_supervision,
    )


def build_loss(
    name: str = "bce_dice",
    bce_weight: float = 0.5,
    dice_weight: float = 0.5,
) -> nn.Module:
    """Factory function to create a loss function.

    Args:
        name: Loss function name ("bce_dice", "bce", "dice").
        bce_weight: Weight for BCE component (if using bce_dice).
        dice_weight: Weight for Dice component (if using bce_dice).

    Returns:
        Loss module.
    """
    if name == "bce_dice":
        return BCEDiceLoss(bce_weight, dice_weight)
    elif name == "bce":
        return nn.BCEWithLogitsLoss()
    elif name == "dice":
        return DiceLoss()
    else:
        raise ValueError(f"Unknown loss function: {name}")


def count_parameters(model: nn.Module) -> int:
    """Count the number of trainable parameters in a model.

    Args:
        model: PyTorch module.

    Returns:
        Total number of trainable parameters.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
