"""Tests for model architecture and loss functions.

Verifies that the Siamese U-Net produces correct output shapes,
handles various input configurations, and that loss functions
produce valid gradients.
"""

import pytest
import torch

from src.models import (
    BCEDiceLoss,
    ConvBlock,
    DiceLoss,
    FeatureFusion,
    SiameseUNet,
    build_loss,
    build_model,
    count_parameters,
)


class TestConvBlock:
    """Test the double convolution building block."""

    def test_output_shape(self) -> None:
        """ConvBlock should preserve spatial dims and change channels."""
        block = ConvBlock(in_channels=64, out_channels=128)
        x = torch.randn(2, 64, 32, 32)
        out = block(x)
        assert out.shape == (2, 128, 32, 32)

    def test_different_sizes(self) -> None:
        """Should work with non-square and non-power-of-2 inputs."""
        block = ConvBlock(in_channels=32, out_channels=64)
        x = torch.randn(1, 32, 17, 23)
        out = block(x)
        assert out.shape == (1, 64, 17, 23)


class TestFeatureFusion:
    """Test feature fusion strategies."""

    @pytest.mark.parametrize("mode", ["diff", "concat", "both"])
    def test_fusion_modes(self, mode: str) -> None:
        """All fusion modes should produce (B, C, H, W) output."""
        fusion = FeatureFusion(in_channels=64, fusion_mode=mode)
        f1 = torch.randn(2, 64, 16, 16)
        f2 = torch.randn(2, 64, 16, 16)
        out = fusion(f1, f2)
        assert out.shape == (2, 64, 16, 16)

    def test_invalid_mode_raises(self) -> None:
        """Invalid fusion mode should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown fusion mode"):
            FeatureFusion(in_channels=64, fusion_mode="invalid")


class TestSiameseUNet:
    """Test the full Siamese U-Net architecture."""

    def test_output_shape_rgb(self) -> None:
        """Output should match input spatial dimensions for RGB input."""
        model = build_model(in_channels=3, pretrained=False)
        x1 = torch.randn(2, 3, 256, 256)
        x2 = torch.randn(2, 3, 256, 256)
        model.eval()
        output = model(x1, x2)
        assert output["pred"].shape == (2, 1, 256, 256)

    def test_output_shape_multispectral(self) -> None:
        """Should handle non-RGB input (e.g., 13-band Sentinel-2)."""
        model = build_model(in_channels=13, pretrained=False)
        x1 = torch.randn(1, 13, 256, 256)
        x2 = torch.randn(1, 13, 256, 256)
        model.eval()
        output = model(x1, x2)
        assert output["pred"].shape == (1, 1, 256, 256)

    def test_deep_supervision_training(self) -> None:
        """Deep supervision should produce auxiliary outputs in train mode."""
        model = build_model(pretrained=False, deep_supervision=True)
        model.train()
        x1 = torch.randn(2, 3, 256, 256)
        x2 = torch.randn(2, 3, 256, 256)
        output = model(x1, x2)

        assert "aux_preds" in output
        assert len(output["aux_preds"]) == 4
        # All auxiliary predictions should match input spatial dims
        for aux in output["aux_preds"]:
            assert aux.shape == (2, 1, 256, 256)

    def test_no_deep_supervision_eval(self) -> None:
        """No auxiliary outputs in eval mode."""
        model = build_model(pretrained=False, deep_supervision=True)
        model.eval()
        x1 = torch.randn(1, 3, 256, 256)
        x2 = torch.randn(1, 3, 256, 256)
        output = model(x1, x2)
        assert "aux_preds" not in output

    def test_non_square_input(self) -> None:
        """Should handle non-square inputs (common in satellite imagery)."""
        model = build_model(pretrained=False)
        model.eval()
        x1 = torch.randn(1, 3, 128, 192)
        x2 = torch.randn(1, 3, 128, 192)
        output = model(x1, x2)
        assert output["pred"].shape == (1, 1, 128, 192)

    def test_parameter_count(self) -> None:
        """Model should have a reasonable number of parameters."""
        model = build_model(pretrained=False)
        n_params = count_parameters(model)
        # ResNet-34 based model should be roughly 20-30M params
        assert 10_000_000 < n_params < 50_000_000


class TestLossFunctions:
    """Test loss functions for change detection."""

    def test_dice_loss_range(self) -> None:
        """Dice loss should be in [0, 1]."""
        loss_fn = DiceLoss()
        pred = torch.randn(2, 1, 32, 32)
        target = torch.randint(0, 2, (2, 1, 32, 32)).float()
        loss = loss_fn(pred, target)
        assert 0.0 <= loss.item() <= 1.0

    def test_dice_loss_perfect_prediction(self) -> None:
        """Dice loss should be near 0 for perfect predictions."""
        loss_fn = DiceLoss()
        target = torch.ones(1, 1, 32, 32)
        # Large positive logits -> sigmoid near 1.0
        pred = torch.ones(1, 1, 32, 32) * 10.0
        loss = loss_fn(pred, target)
        assert loss.item() < 0.05

    def test_bce_dice_combined(self) -> None:
        """Combined loss should be finite and produce valid gradients."""
        loss_fn = BCEDiceLoss(bce_weight=0.5, dice_weight=0.5)
        pred = torch.randn(2, 1, 32, 32, requires_grad=True)
        target = torch.randint(0, 2, (2, 1, 32, 32)).float()
        loss = loss_fn(pred, target)

        assert torch.isfinite(loss)
        loss.backward()
        assert pred.grad is not None
        assert torch.isfinite(pred.grad).all()

    def test_build_loss_factory(self) -> None:
        """Factory should produce valid loss modules."""
        for name in ["bce_dice", "bce", "dice"]:
            loss_fn = build_loss(name=name)
            assert isinstance(loss_fn, torch.nn.Module)

    def test_build_loss_invalid(self) -> None:
        """Invalid loss name should raise."""
        with pytest.raises(ValueError):
            build_loss(name="invalid_loss")
