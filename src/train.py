"""Training loop for satellite change detection model.

Implements a complete training pipeline with:
- Mixed precision (FP16) training for memory efficiency
- Learning rate warmup + cosine annealing schedule
- Deep supervision loss aggregation
- TensorBoard logging
- Checkpoint saving with best-model tracking
- Early stopping based on validation F1 score

Usage:
    python -m src.train --config configs/default.yaml
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.config import Config, load_config, parse_args
from src.data_loader import build_dataloaders
from src.models import build_loss, build_model, count_parameters
from src.utils import compute_metrics, get_device, plot_training_curves, set_seed


class EarlyStopping:
    """Stop training when a monitored metric stops improving.

    Tracks the best value of a given metric and triggers a stop if no
    improvement is seen for `patience` consecutive epochs. This prevents
    overfitting and saves compute.

    Args:
        patience: Number of epochs to wait for improvement.
        mode: "max" if higher is better (e.g., F1), "min" for losses.
        min_delta: Minimum change to qualify as an improvement.
    """

    def __init__(
        self,
        patience: int = 15,
        mode: str = "max",
        min_delta: float = 1e-4,
    ) -> None:
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.counter = 0
        self.best_value: float | None = None
        self.should_stop = False

    def step(self, value: float) -> bool:
        """Check if training should stop.

        Args:
            value: Current epoch's metric value.

        Returns:
            True if training should stop.
        """
        if self.best_value is None:
            self.best_value = value
            return False

        if self.mode == "max":
            improved = value > self.best_value + self.min_delta
        else:
            improved = value < self.best_value - self.min_delta

        if improved:
            self.best_value = value
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True

        return self.should_stop


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    device: torch.device,
    epoch: int,
    config: Config,
    writer: SummaryWriter | None = None,
) -> float:
    """Run one training epoch.

    Iterates over all batches, computes loss (including deep supervision),
    performs gradient scaling for mixed precision, and logs to TensorBoard.

    Args:
        model: The change detection model.
        loader: Training DataLoader.
        criterion: Loss function.
        optimizer: Parameter optimizer.
        scaler: Gradient scaler for mixed precision.
        device: Compute device.
        epoch: Current epoch number (for logging).
        config: Training configuration.
        writer: Optional TensorBoard writer.

    Returns:
        Average training loss for this epoch.
    """
    model.train()
    use_amp = config.training.mixed_precision and torch.cuda.is_available()
    running_loss = 0.0
    num_batches = 0

    pbar = tqdm(loader, desc=f"Epoch {epoch + 1} [Train]", leave=False)

    for batch_idx, batch in enumerate(pbar):
        image1 = batch["image1"].to(device, non_blocking=True)
        image2 = batch["image2"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)

        optimizer.zero_grad()

        # Mixed precision forward pass
        with autocast("cuda", enabled=use_amp):
            output = model(image1, image2)
            loss = criterion(output["pred"], mask)

            # Deep supervision: add weighted auxiliary losses
            # These provide gradient signal at multiple scales, helping
            # the encoder learn meaningful features at every resolution
            if "aux_preds" in output:
                aux_weights = [0.4, 0.2, 0.1, 0.05]
                for aux_pred, weight in zip(
                    output["aux_preds"], aux_weights, strict=False
                ):
                    loss = loss + weight * criterion(aux_pred, mask)

        # Backward pass with gradient scaling (prevents FP16 underflow)
        scaler.scale(loss).backward()
        # Gradient clipping to prevent exploding gradients
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()
        num_batches += 1
        pbar.set_postfix(loss=f"{loss.item():.4f}")

        # TensorBoard logging
        if writer and batch_idx % config.training.optimizer.lr == 0:
            global_step = epoch * len(loader) + batch_idx
            writer.add_scalar("train/batch_loss", loss.item(), global_step)

    avg_loss = running_loss / max(num_batches, 1)
    return avg_loss


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    config: Config,
) -> tuple[float, dict[str, float]]:
    """Run validation and compute metrics.

    Evaluates the model on the entire validation set without gradients.
    Computes both the loss (for scheduler decisions) and segmentation
    metrics (for early stopping and reporting).

    Args:
        model: The change detection model (set to eval mode internally).
        loader: Validation DataLoader.
        criterion: Loss function.
        device: Compute device.
        config: Configuration with evaluation threshold.

    Returns:
        Tuple of (average_loss, metrics_dict).
    """
    model.eval()
    use_amp = config.training.mixed_precision and torch.cuda.is_available()
    running_loss = 0.0
    all_preds = []
    all_targets = []
    num_batches = 0

    pbar = tqdm(loader, desc="Validating", leave=False)

    for batch in pbar:
        image1 = batch["image1"].to(device, non_blocking=True)
        image2 = batch["image2"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)

        with autocast("cuda", enabled=use_amp):
            output = model(image1, image2)
            loss = criterion(output["pred"], mask)

        running_loss += loss.item()
        num_batches += 1

        # Binarize predictions using sigmoid + threshold
        pred_binary = (
            (torch.sigmoid(output["pred"]) > config.evaluation.threshold)
            .cpu()
            .numpy()
            .astype(int)
        )
        target_np = mask.cpu().numpy().astype(int)

        all_preds.append(pred_binary)
        all_targets.append(target_np)

    avg_loss = running_loss / max(num_batches, 1)

    # Compute metrics over the entire validation set
    preds = np.concatenate(all_preds, axis=0)
    targets = np.concatenate(all_targets, axis=0)
    metrics = compute_metrics(preds, targets)

    return avg_loss, metrics


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    epoch: int,
    metrics: dict[str, float],
    config: Config,
    path: str | Path,
) -> None:
    """Save a training checkpoint to disk.

    Includes model weights, optimizer state, scheduler state, and
    metadata for resuming training or running inference.

    Args:
        model: Model to save.
        optimizer: Optimizer state for training resumption.
        scheduler: LR scheduler state.
        epoch: Current epoch number.
        metrics: Validation metrics at this checkpoint.
        config: Training configuration (for reproducibility).
        path: File path for the checkpoint.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "metrics": metrics,
            "config": config.to_dict(),
        },
        path,
    )


def train(config: Config) -> None:
    """Full training pipeline.

    Orchestrates data loading, model creation, training loop, validation,
    checkpointing, early stopping, and logging. This is the main function
    called by the CLI entry point.

    Args:
        config: Complete training configuration.
    """
    # Reproducibility
    set_seed(config.seed)
    device = get_device()
    print(f"Using device: {device}")

    # Data
    print("Building data loaders...")
    loaders = build_dataloaders(
        data_root=config.paths.data_root,
        batch_size=config.training.batch_size,
        num_workers=config.training.num_workers,
    )

    if "train" not in loaders:
        raise RuntimeError(
            "Training data not found. See DATA.md for setup instructions."
        )

    # Model
    print("Building model...")
    model = build_model(
        in_channels=config.model.in_channels,
        num_classes=config.model.num_classes,
        pretrained=config.model.pretrained,
        fusion_mode=config.model.fusion,
        deep_supervision=config.model.deep_supervision,
    ).to(device)

    param_count = count_parameters(model)
    print(f"Model parameters: {param_count:,} ({param_count / 1e6:.1f}M)")

    # Loss, optimizer, scheduler
    criterion = build_loss(
        name=config.training.loss.name,
        bce_weight=config.training.loss.bce_weight,
        dice_weight=config.training.loss.dice_weight,
    )

    # AdamW expects exactly two betas; tuple(...) alone widens to tuple[float, ...]
    beta1, beta2 = config.training.optimizer.betas
    optimizer = AdamW(
        model.parameters(),
        lr=config.training.optimizer.lr,
        weight_decay=config.training.optimizer.weight_decay,
        betas=(float(beta1), float(beta2)),
    )

    scheduler = CosineAnnealingWarmRestarts(
        optimizer,
        T_0=config.training.scheduler.T_0,
        T_mult=config.training.scheduler.T_mult,
        eta_min=config.training.scheduler.min_lr,
    )

    use_amp = config.training.mixed_precision and torch.cuda.is_available()
    scaler = GradScaler("cuda", enabled=use_amp)

    # Logging
    writer = SummaryWriter(log_dir=config.paths.log_dir)
    early_stopping = EarlyStopping(
        patience=config.training.early_stopping_patience,
        mode="max",
    )

    # Save config for reproducibility
    config.save_yaml(Path(config.paths.log_dir) / "config.yaml")

    # Training history
    train_losses = []
    val_losses = []
    val_f1_scores = []
    best_f1 = 0.0

    print(f"\nStarting training for {config.training.epochs} epochs...")
    print(f"{'=' * 60}")

    for epoch in range(config.training.epochs):
        start_time = time.time()

        # Train
        train_loss = train_one_epoch(
            model,
            loaders["train"],
            criterion,
            optimizer,
            scaler,
            device,
            epoch,
            config,
            writer,
        )
        train_losses.append(train_loss)

        # Validate
        val_loss, metrics = validate(
            model,
            loaders.get("val", loaders["train"]),
            criterion,
            device,
            config,
        )
        val_losses.append(val_loss)
        val_f1_scores.append(metrics["f1"])

        # Update learning rate
        scheduler.step()

        elapsed = time.time() - start_time

        # Log to console
        print(
            f"Epoch {epoch + 1:3d}/{config.training.epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"F1: {metrics['f1']:.4f} | "
            f"IoU: {metrics['iou']:.4f} | "
            f"LR: {optimizer.param_groups[0]['lr']:.2e} | "
            f"Time: {elapsed:.1f}s"
        )

        # Log to TensorBoard
        writer.add_scalar("train/epoch_loss", train_loss, epoch)
        writer.add_scalar("val/epoch_loss", val_loss, epoch)
        for name, value in metrics.items():
            writer.add_scalar(f"val/{name}", value, epoch)
        writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], epoch)

        # Save best model
        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            save_checkpoint(
                model,
                optimizer,
                scheduler,
                epoch,
                metrics,
                config,
                Path(config.paths.checkpoint_dir) / "best_model.pth",
            )
            print(f"  -> New best F1: {best_f1:.4f}, checkpoint saved.")

        # Early stopping check
        if early_stopping.step(metrics["f1"]):
            print(f"\nEarly stopping triggered after {epoch + 1} epochs.")
            break

    # Save final model
    save_checkpoint(
        model,
        optimizer,
        scheduler,
        epoch,
        metrics,
        config,
        Path(config.paths.checkpoint_dir) / "final_model.pth",
    )

    # Plot training curves
    plot_training_curves(
        train_losses,
        val_losses,
        val_f1_scores,
        save_path=Path(config.paths.results_dir) / "metrics" / "training_curves.png",
    )

    writer.close()
    print(f"\n{'=' * 60}")
    print(f"Training complete. Best validation F1: {best_f1:.4f}")
    print(f"Checkpoints saved to: {config.paths.checkpoint_dir}")


def main() -> None:
    """CLI entry point for training."""
    args = parse_args()
    config = load_config(args)
    train(config)


if __name__ == "__main__":
    main()
