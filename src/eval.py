"""Evaluation and inference pipeline for change detection models.

Supports:
- Full test set evaluation with metrics reporting
- Single image pair inference
- Sliding window inference for large satellite scenes
- Batch visualization of predictions

Usage:
    python -m src.eval \
        --config configs/default.yaml \
        --checkpoint models/checkpoints/best_model.pth
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.amp import autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.config import Config, load_config, parse_args
from src.data_loader import build_dataloaders
from src.models import build_model
from src.utils import (
    compute_confusion_matrix,
    compute_metrics,
    get_device,
    plot_confusion_matrix,
    set_seed,
    visualize_change_detection,
)


def load_model_from_checkpoint(
    checkpoint_path: str | Path,
    config: Config,
    device: torch.device,
) -> nn.Module:
    """Load a trained model from a checkpoint file.

    Args:
        checkpoint_path: Path to the .pth checkpoint.
        config: Model configuration (architecture must match checkpoint).
        device: Device to load the model onto.

    Returns:
        Model with loaded weights, set to eval mode.
    """
    model = build_model(
        in_channels=config.model.in_channels,
        num_classes=config.model.num_classes,
        pretrained=False,  # We'll load our own weights
        fusion_mode=config.model.fusion,
        deep_supervision=False,  # Not needed at inference
    ).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    model.eval()

    print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', '?')}")
    if "metrics" in checkpoint:
        print(f"Checkpoint metrics: {checkpoint['metrics']}")

    return model


@torch.no_grad()
def predict_single(
    model: nn.Module,
    image1: torch.Tensor,
    image2: torch.Tensor,
    device: torch.device,
    threshold: float = 0.5,
) -> np.ndarray:
    """Run inference on a single image pair.

    Args:
        model: Trained change detection model.
        image1: Pre-change image tensor (1, C, H, W) or (C, H, W).
        image2: Post-change image tensor (1, C, H, W) or (C, H, W).
        device: Compute device.
        threshold: Binarization threshold for sigmoid output.

    Returns:
        Binary change mask as numpy array (H, W).
    """
    model.eval()

    # Add batch dimension if needed
    if image1.ndim == 3:
        image1 = image1.unsqueeze(0)
        image2 = image2.unsqueeze(0)

    image1 = image1.to(device)
    image2 = image2.to(device)

    with autocast("cuda", enabled=torch.cuda.is_available()):
        output = model(image1, image2)

    # Sigmoid to get probabilities, then threshold to get binary mask
    pred_prob = torch.sigmoid(output["pred"]).squeeze().cpu().numpy()
    pred_binary = (pred_prob > threshold).astype(np.uint8)

    return pred_binary


@torch.no_grad()
def sliding_window_inference(
    model: nn.Module,
    image1: torch.Tensor,
    image2: torch.Tensor,
    device: torch.device,
    patch_size: int = 256,
    stride: int = 128,
    threshold: float = 0.5,
) -> np.ndarray:
    """Run inference on large images using a sliding window.

    Satellite scenes (e.g., 10980x10980 for Sentinel-2) don't fit in
    GPU memory at once. This function tiles the image into overlapping
    patches, runs inference on each, and stitches the results together.

    Overlapping regions are averaged, which smooths out boundary artifacts
    that can occur at patch edges.

    Args:
        model: Trained change detection model.
        image1: Full pre-change image (C, H, W).
        image2: Full post-change image (C, H, W).
        device: Compute device.
        patch_size: Size of each inference patch.
        stride: Step size between patches (< patch_size for overlap).
        threshold: Binarization threshold.

    Returns:
        Binary change mask for the full image (H, W).
    """
    model.eval()
    _, h, w = image1.shape

    # Accumulator for predictions and a count map for averaging overlaps
    pred_sum = np.zeros((h, w), dtype=np.float32)
    count_map = np.zeros((h, w), dtype=np.float32)

    # Slide over the image
    for y in range(0, h, stride):
        for x in range(0, w, stride):
            # Handle edge cases: if patch would go out of bounds,
            # shift it back so it ends at the image edge
            y_end = min(y + patch_size, h)
            x_end = min(x + patch_size, w)
            y_start = max(y_end - patch_size, 0)
            x_start = max(x_end - patch_size, 0)

            patch1 = image1[:, y_start:y_end, x_start:x_end].unsqueeze(0).to(device)
            patch2 = image2[:, y_start:y_end, x_start:x_end].unsqueeze(0).to(device)

            with autocast("cuda", enabled=torch.cuda.is_available()):
                output = model(patch1, patch2)

            pred_prob = torch.sigmoid(output["pred"]).squeeze().cpu().numpy()

            # Handle case where patch is smaller than expected
            actual_h = y_end - y_start
            actual_w = x_end - x_start
            pred_prob = pred_prob[:actual_h, :actual_w]

            pred_sum[y_start:y_end, x_start:x_end] += pred_prob
            count_map[y_start:y_end, x_start:x_end] += 1.0

    # Average overlapping predictions
    count_map = np.maximum(count_map, 1.0)
    pred_avg = pred_sum / count_map
    pred_binary = (pred_avg > threshold).astype(np.uint8)

    return pred_binary


@torch.no_grad()
def evaluate_test_set(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    config: Config,
    save_visualizations: bool = True,
    max_vis: int = 20,
) -> dict[str, float]:
    """Evaluate model on the full test set and generate reports.

    Computes aggregate metrics, saves per-sample visualizations,
    and writes a confusion matrix plot.

    Args:
        model: Trained model in eval mode.
        loader: Test DataLoader.
        device: Compute device.
        config: Configuration with paths and thresholds.
        save_visualizations: Whether to save prediction visualizations.
        max_vis: Maximum number of visualizations to save.

    Returns:
        Dictionary of aggregate test metrics.
    """
    model.eval()
    all_preds = []
    all_targets = []
    per_sample_metrics = []

    pbar = tqdm(loader, desc="Evaluating")

    for batch_idx, batch in enumerate(pbar):
        image1 = batch["image1"].to(device, non_blocking=True)
        image2 = batch["image2"].to(device, non_blocking=True)
        mask = batch["mask"]

        with autocast("cuda", enabled=torch.cuda.is_available()):
            output = model(image1, image2)

        pred_prob = torch.sigmoid(output["pred"]).cpu()
        pred_binary = (pred_prob > config.evaluation.threshold).numpy().astype(int)
        target_np = mask.numpy().astype(int)

        all_preds.append(pred_binary)
        all_targets.append(target_np)

        # Per-sample metrics and visualizations
        for i in range(image1.shape[0]):
            sample_pred = pred_binary[i, 0]
            sample_target = target_np[i, 0]
            sample_metrics = compute_metrics(sample_pred, sample_target)
            sample_metrics["filename"] = batch["filename"][i]
            per_sample_metrics.append(sample_metrics)

            # Save visualizations for the first max_vis samples
            vis_count = batch_idx * loader.batch_size + i
            if save_visualizations and vis_count < max_vis:
                vis_path = (
                    Path(config.paths.results_dir)
                    / "visualizations"
                    / f"test_{vis_count:04d}.png"
                )
                visualize_change_detection(
                    image_t1=batch["image1"][i].numpy(),
                    image_t2=batch["image2"][i].numpy(),
                    prediction=sample_pred,
                    ground_truth=sample_target,
                    save_path=vis_path,
                    title=f"Sample {vis_count} | F1: {sample_metrics['f1']:.3f}",
                )

    # Aggregate metrics
    preds = np.concatenate(all_preds, axis=0)
    targets = np.concatenate(all_targets, axis=0)
    aggregate_metrics = compute_metrics(preds, targets)

    # Confusion matrix
    cm = compute_confusion_matrix(preds, targets)
    plot_confusion_matrix(
        cm,
        save_path=Path(config.paths.results_dir) / "metrics" / "confusion_matrix.png",
    )

    # Save metrics to JSON
    metrics_path = Path(config.paths.results_dir) / "metrics" / "test_metrics.json"
    os.makedirs(metrics_path.parent, exist_ok=True)
    with open(metrics_path, "w") as f:
        json.dump(
            {
                "aggregate": aggregate_metrics,
                "per_sample": per_sample_metrics,
                "confusion_matrix": cm.tolist(),
            },
            f,
            indent=2,
        )

    # Print results
    print(f"\n{'=' * 50}")
    print("Test Set Results:")
    print(f"{'=' * 50}")
    for name, value in aggregate_metrics.items():
        print(f"  {name:>12s}: {value:.4f}")
    print(f"{'=' * 50}")

    return aggregate_metrics


def main() -> None:
    """CLI entry point for evaluation."""
    args = parse_args()
    config = load_config(args)

    set_seed(config.seed)
    device = get_device(getattr(args, "device", None))
    print(f"Using device: {device}")

    # Load model
    checkpoint_path = args.checkpoint
    if checkpoint_path is None:
        checkpoint_path = Path(config.paths.checkpoint_dir) / "best_model.pth"

    if not Path(checkpoint_path).exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}. "
            f"Train a model first with: python -m src.train"
        )

    model = load_model_from_checkpoint(checkpoint_path, config, device)

    # Build test loader
    loaders = build_dataloaders(
        data_root=config.paths.data_root,
        batch_size=config.training.batch_size,
        num_workers=config.training.num_workers,
    )

    if "test" not in loaders:
        raise RuntimeError("Test data not found. Check your data directory structure.")

    # Run evaluation
    evaluate_test_set(model, loaders["test"], device, config)

    print(f"\nResults saved to: {config.paths.results_dir}")


if __name__ == "__main__":
    main()
