"""Utility functions for metrics, visualization, and reproducibility.

Covers everything from seeding random number generators to plotting
side-by-side change detection results with overlays.
"""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility across all libraries.

    Ensures deterministic behavior for Python, NumPy, and PyTorch.
    Note: CUDA operations may still have minor non-determinism unless
    torch.backends.cudnn.deterministic is also set.

    Args:
        seed: Integer seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        # Trade speed for reproducibility
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device(override: Optional[str] = None) -> torch.device:
    """Select the best available compute device.

    Priority: user override > CUDA GPU > MPS (Apple Silicon) > CPU.

    Args:
        override: Force a specific device string (e.g. "cuda:1", "cpu").

    Returns:
        torch.device for model and tensor placement.
    """
    if override:
        return torch.device(override)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_iou(pred: np.ndarray, target: np.ndarray) -> float:
    """Compute Intersection over Union for binary masks.

    IoU (also called Jaccard Index) measures the overlap between predicted
    and ground truth change regions. It is the standard metric for
    segmentation tasks because it penalizes both false positives and
    false negatives equally.

    Args:
        pred: Binary prediction array (H, W) with values in {0, 1}.
        target: Binary ground truth array (H, W) with values in {0, 1}.

    Returns:
        IoU score in [0, 1]. Returns 0.0 if both pred and target are empty.
    """
    intersection = np.logical_and(pred, target).sum()
    union = np.logical_or(pred, target).sum()
    if union == 0:
        return 0.0
    return float(intersection / union)


def compute_metrics(
    pred: np.ndarray,
    target: np.ndarray,
) -> dict[str, float]:
    """Compute a full suite of binary classification metrics.

    Args:
        pred: Flat or 2D binary prediction array.
        target: Flat or 2D binary ground truth array.

    Returns:
        Dictionary with f1, iou, precision, recall, and accuracy.
    """
    pred_flat = pred.flatten().astype(int)
    target_flat = target.flatten().astype(int)

    return {
        "f1": float(f1_score(target_flat, pred_flat, zero_division=0)),
        "iou": compute_iou(pred, target),
        "precision": float(precision_score(target_flat, pred_flat, zero_division=0)),
        "recall": float(recall_score(target_flat, pred_flat, zero_division=0)),
        "accuracy": float((pred_flat == target_flat).mean()),
    }


def compute_confusion_matrix(
    pred: np.ndarray,
    target: np.ndarray,
) -> np.ndarray:
    """Compute the 2x2 confusion matrix for binary change detection.

    Layout:
        [[TN, FP],
         [FN, TP]]

    In change detection terms:
    - TN: correctly identified as "no change"
    - FP: false alarm (predicted change where there was none)
    - FN: missed change (failed to detect actual change)
    - TP: correctly detected change

    Args:
        pred: Binary prediction array.
        target: Binary ground truth array.

    Returns:
        2x2 numpy array confusion matrix.
    """
    return confusion_matrix(
        target.flatten().astype(int),
        pred.flatten().astype(int),
        labels=[0, 1],
    )


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

# Custom colormap: transparent for no-change, red for change
CHANGE_CMAP = LinearSegmentedColormap.from_list(
    "change", [(0, 0, 0, 0), (1, 0, 0, 0.7)], N=2
)


def normalize_for_display(image: np.ndarray) -> np.ndarray:
    """Normalize a satellite image array to [0, 1] for matplotlib display.

    Handles both single-band and multi-band images. For multi-band images,
    uses the first 3 bands as RGB. Clips outliers using 2nd/98th percentile
    stretch, which is standard practice in remote sensing to handle the
    wide dynamic range of satellite sensors.

    Args:
        image: Array of shape (C, H, W) or (H, W).

    Returns:
        Display-ready array of shape (H, W, 3) or (H, W) in [0, 1].
    """
    if image.ndim == 3:
        # Take first 3 bands as RGB (or fewer if available)
        img = image[:3].transpose(1, 2, 0).copy()
    else:
        img = image.copy()

    # Percentile stretch to handle satellite imagery's dynamic range
    p2, p98 = np.percentile(img, [2, 98])
    if p98 - p2 > 0:
        img = (img - p2) / (p98 - p2)
    img = np.clip(img, 0, 1)
    return img


def visualize_change_detection(
    image_t1: np.ndarray,
    image_t2: np.ndarray,
    prediction: np.ndarray,
    ground_truth: Optional[np.ndarray] = None,
    save_path: Optional[str | Path] = None,
    title: Optional[str] = None,
) -> plt.Figure:
    """Create a publication-quality visualization of change detection results.

    Generates a multi-panel figure showing:
    1. Pre-change image (T1)
    2. Post-change image (T2)
    3. Predicted change mask overlaid on T2
    4. Ground truth (if provided) with error overlay

    Colors in the overlay:
    - Red: predicted change
    - Green (in error map): correctly detected change (TP)
    - Red (in error map): false alarm (FP)
    - Blue (in error map): missed change (FN)

    Args:
        image_t1: Pre-change image, shape (C, H, W).
        image_t2: Post-change image, shape (C, H, W).
        prediction: Binary change prediction, shape (H, W).
        ground_truth: Optional binary ground truth, shape (H, W).
        save_path: If provided, saves the figure to this path.
        title: Optional super-title for the figure.

    Returns:
        matplotlib Figure object.
    """
    n_panels = 4 if ground_truth is not None else 3
    fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 5))

    # Display pre-change image
    t1_display = normalize_for_display(image_t1)
    axes[0].imshow(t1_display)
    axes[0].set_title("Pre-Change (T1)")
    axes[0].axis("off")

    # Display post-change image
    t2_display = normalize_for_display(image_t2)
    axes[1].imshow(t2_display)
    axes[1].set_title("Post-Change (T2)")
    axes[1].axis("off")

    # Prediction overlay on T2
    axes[2].imshow(t2_display)
    axes[2].imshow(prediction, cmap=CHANGE_CMAP, vmin=0, vmax=1)
    axes[2].set_title("Predicted Changes")
    axes[2].axis("off")

    # Error analysis panel (if ground truth is available)
    if ground_truth is not None:
        # Build an RGB error map:
        # Green = true positive, Red = false positive, Blue = false negative
        error_map = np.zeros((*prediction.shape, 3))
        tp = np.logical_and(prediction == 1, ground_truth == 1)
        fp = np.logical_and(prediction == 1, ground_truth == 0)
        fn = np.logical_and(prediction == 0, ground_truth == 1)

        error_map[tp] = [0, 1, 0]  # Green: correctly detected
        error_map[fp] = [1, 0, 0]  # Red: false alarm
        error_map[fn] = [0, 0, 1]  # Blue: missed

        axes[3].imshow(t2_display * 0.4)  # Dimmed background
        axes[3].imshow(error_map, alpha=0.7)
        axes[3].set_title("Error Map (G=TP, R=FP, B=FN)")
        axes[3].axis("off")

    if title:
        fig.suptitle(title, fontsize=14, fontweight="bold")

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_training_curves(
    train_losses: list[float],
    val_losses: list[float],
    val_f1_scores: list[float],
    save_path: Optional[str | Path] = None,
) -> plt.Figure:
    """Plot training and validation loss curves with F1 score.

    Args:
        train_losses: Training loss per epoch.
        val_losses: Validation loss per epoch.
        val_f1_scores: Validation F1 score per epoch.
        save_path: If provided, saves the figure.

    Returns:
        matplotlib Figure object.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    epochs = range(1, len(train_losses) + 1)

    # Loss curves
    ax1.plot(epochs, train_losses, label="Train Loss", color="#2196F3")
    ax1.plot(epochs, val_losses, label="Val Loss", color="#F44336")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training and Validation Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # F1 score curve
    ax2.plot(epochs, val_f1_scores, label="Val F1", color="#4CAF50")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("F1 Score")
    ax2.set_title("Validation F1 Score")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_confusion_matrix(
    cm: np.ndarray,
    save_path: Optional[str | Path] = None,
) -> plt.Figure:
    """Plot a labeled confusion matrix heatmap.

    Args:
        cm: 2x2 confusion matrix from compute_confusion_matrix().
        save_path: If provided, saves the figure.

    Returns:
        matplotlib Figure object.
    """
    fig, ax = plt.subplots(figsize=(6, 5))

    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["No Change", "Change"])
    ax.set_yticklabels(["No Change", "Change"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")

    # Annotate cells with counts
    for i in range(2):
        for j in range(2):
            color = "white" if cm[i, j] > cm.max() / 2 else "black"
            ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center", color=color, fontsize=14)

    fig.colorbar(im)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig
