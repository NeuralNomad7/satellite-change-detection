"""Data loading pipeline for satellite imagery change detection.

Handles bi-temporal image pairs (pre-change and post-change) with their
corresponding binary change masks. Supports both standard image formats
(PNG, TIFF) and geospatial rasters via rasterio.

Satellite imagery concepts:
- Bi-temporal: Two images of the same location at different times.
- Co-registered: Images are spatially aligned so pixel (i,j) refers to
  the same ground location in both T1 and T2.
- Multispectral: Each pixel has multiple spectral bands (not just RGB).
  Sentinel-2 has 13 bands spanning visible to shortwave infrared.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import albumentations as A
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler


def load_image(path: str | Path) -> np.ndarray:
    """Load an image from disk, supporting both standard and geospatial formats.

    For standard formats (PNG, JPG), uses PIL.
    For geospatial formats (GeoTIFF), uses rasterio to preserve band ordering
    and geospatial metadata.

    Args:
        path: Path to image file.

    Returns:
        Image array of shape (H, W, C) with float32 values in [0, 255].
    """
    path = str(path)
    if path.endswith((".tif", ".tiff")):
        try:
            import rasterio

            with rasterio.open(path) as src:
                # rasterio reads as (C, H, W), we need (H, W, C)
                img = src.read().astype(np.float32)
                return np.transpose(img, (1, 2, 0))
        except ImportError:
            raise ImportError(
                "rasterio is required for loading GeoTIFF files. "
                "Install it with: pip install rasterio"
            ) from None
    else:
        img = np.array(Image.open(path).convert("RGB"), dtype=np.float32)
        return img


def load_mask(path: str | Path) -> np.ndarray:
    """Load a binary change mask.

    Ensures the mask is binary (0 or 1) regardless of the source format.
    Some datasets encode change as 255 in 8-bit images.

    Args:
        path: Path to mask image file.

    Returns:
        Binary mask array of shape (H, W) with values in {0, 1}.
    """
    mask = np.array(Image.open(path).convert("L"), dtype=np.float32)
    # Datasets encode change either as 1 or as 255. A fixed 128 cutoff silently
    # zeroes out every pixel of a 0/1-encoded mask, so pick the threshold from
    # the data instead.
    threshold = 0.5 if mask.max() <= 1.0 else 128.0
    mask = (mask > threshold).astype(np.float32)
    return mask


def get_transforms(
    split: str,
    patch_size: int = 256,
    normalization: str = "imagenet",
) -> A.Compose:
    """Build augmentation and preprocessing transforms.

    Uses albumentations which applies geometric transforms identically
    to both images AND the mask (critical for change detection -- you
    can't flip one image without flipping the other).

    The additional_targets parameter tells albumentations that "image2"
    should receive the same spatial transforms as "image" (the T1 image).

    Args:
        split: One of "train", "val", "test".
        patch_size: Target spatial size for cropping.
        normalization: Strategy -- "imagenet" for pretrained RGB models,
            "minmax" for simple [0,1] scaling.

    Returns:
        albumentations Compose transform pipeline.
    """
    transforms_list = []

    if split == "train":
        transforms_list.extend(
            [
                A.RandomCrop(height=patch_size, width=patch_size),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                # Apply photometric augmentation to simulate different
                # atmospheric conditions between acquisition dates
                A.RandomBrightnessContrast(
                    brightness_limit=0.2, contrast_limit=0.2, p=0.3
                ),
                # albumentations 2.x replaced GaussNoise's `var_limit` (variance
                # in 0-255 units) with `std_range` (std as a fraction of 255).
                # Passing the old kwarg is silently ignored with only a
                # UserWarning, so keep this on the current API.
                # Equivalent to the previous var_limit=(10.0, 50.0):
                #   sqrt(10)/255 ~= 0.012, sqrt(50)/255 ~= 0.028
                A.GaussNoise(std_range=(0.012, 0.028), p=0.2),
            ]
        )
    else:
        # For validation/test: just center-crop to patch_size
        transforms_list.append(A.CenterCrop(height=patch_size, width=patch_size))

    # Normalize pixel values
    if normalization == "imagenet":
        transforms_list.append(
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                max_pixel_value=255.0,
            )
        )
    elif normalization == "minmax":
        transforms_list.append(
            A.Normalize(
                mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0], max_pixel_value=255.0
            )
        )

    transforms_list.append(ToTensorV2())

    return A.Compose(
        transforms_list,
        # This is the key: "image2" gets the same spatial transforms as "image"
        additional_targets={"image2": "image"},
    )


class ChangeDetectionDataset(Dataset):
    """Dataset for bi-temporal satellite image pairs with change labels.

    Directory structure expected:
        root/
        ├── A/          # Pre-change images (T1)
        ├── B/          # Post-change images (T2)
        └── label/      # Binary change masks

    File names must match across A/, B/, and label/ directories.

    Args:
        root: Path to split directory containing A/, B/, label/.
        split: Data split name ("train", "val", "test").
        transform: albumentations Compose pipeline.
        band_indices: If using multispectral data, which bands to select.
    """

    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        transform: A.Compose | None = None,
        band_indices: list[int] | None = None,
    ) -> None:
        self.root = Path(root)
        self.split = split
        self.transform = transform
        self.band_indices = band_indices

        # Locate image pairs
        self.image_dir_a = self.root / "A"
        self.image_dir_b = self.root / "B"
        self.label_dir = self.root / "label"

        # Collect sorted file names (must match across directories)
        self.filenames = sorted(
            [
                f
                for f in os.listdir(self.image_dir_a)
                if f.endswith((".png", ".jpg", ".tif", ".tiff"))
            ]
        )

        if len(self.filenames) == 0:
            raise FileNotFoundError(
                f"No images found in {self.image_dir_a}. "
                f"Expected .png, .jpg, or .tif files."
            )

    def __len__(self) -> int:
        return len(self.filenames)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Load and transform a single bi-temporal image pair with label.

        Returns:
            Dictionary with keys:
            - "image1": Pre-change image tensor (C, H, W)
            - "image2": Post-change image tensor (C, H, W)
            - "mask": Binary change mask tensor (1, H, W)
            - "filename": Original file name for tracking
        """
        filename = self.filenames[idx]

        # Load both time steps and the change mask. The mask starts as an
        # ndarray and becomes a tensor once transforms (ToTensorV2) run.
        img_a = load_image(self.image_dir_a / filename)
        img_b = load_image(self.image_dir_b / filename)
        mask: np.ndarray | torch.Tensor = load_mask(self.label_dir / filename)

        # Select specific spectral bands if requested
        # (e.g., pick RGB from a 13-band Sentinel-2 image)
        if self.band_indices is not None:
            img_a = img_a[:, :, self.band_indices]
            img_b = img_b[:, :, self.band_indices]

        # Apply transforms -- albumentations handles both images and mask
        if self.transform:
            transformed = self.transform(image=img_a, image2=img_b, mask=mask)
            img_a = transformed["image"]
            img_b = transformed["image2"]
            mask = transformed["mask"]

        # Ensure mask has a channel dimension: (H, W) -> (1, H, W)
        if isinstance(mask, torch.Tensor) and mask.ndim == 2:
            mask = mask.unsqueeze(0)
        elif isinstance(mask, np.ndarray) and mask.ndim == 2:
            mask = torch.from_numpy(mask).unsqueeze(0).float()

        return {
            "image1": img_a,
            "image2": img_b,
            "mask": mask,
            "filename": filename,
        }

    def get_change_fractions(self) -> list[float]:
        """Compute the fraction of changed pixels per sample.

        Used for creating a weighted sampler that oversamples patches
        containing actual changes, addressing the severe class imbalance
        common in change detection (most of the landscape doesn't change).

        Returns:
            List of change fractions, one per sample.
        """
        fractions = []
        for filename in self.filenames:
            mask = load_mask(self.label_dir / filename)
            fractions.append(float(mask.mean()))
        return fractions


def create_weighted_sampler(
    dataset: ChangeDetectionDataset,
    oversample_ratio: float = 2.0,
    min_change_fraction: float = 0.01,
) -> WeightedRandomSampler:
    """Create a sampler that oversamples patches containing changes.

    In a typical satellite scene, 90%+ of patches show no change at all.
    Without oversampling, the model rarely sees positive examples during
    training and tends to predict "no change" everywhere.

    Args:
        dataset: The training dataset to sample from.
        oversample_ratio: How much more likely to sample changed patches.
        min_change_fraction: Minimum fraction of changed pixels to consider
            a patch as "changed" (filters out nearly-empty patches).

    Returns:
        WeightedRandomSampler for use with DataLoader.
    """
    fractions = dataset.get_change_fractions()

    weights = []
    for frac in fractions:
        if frac >= min_change_fraction:
            weights.append(oversample_ratio)
        else:
            weights.append(1.0)

    return WeightedRandomSampler(
        weights=weights,
        num_samples=len(weights),
        replacement=True,
    )


def build_dataloaders(
    data_root: str | Path,
    batch_size: int = 16,
    num_workers: int = 4,
    patch_size: int = 256,
    normalization: str = "imagenet",
    oversample: bool = True,
    band_indices: list[int] | None = None,
) -> dict[str, DataLoader]:
    """Build train, validation, and test DataLoaders.

    This is the main entry point for the data pipeline. It sets up
    datasets with appropriate transforms and returns ready-to-use
    DataLoaders for each split.

    Args:
        data_root: Root directory containing train/, val/, test/ splits.
        batch_size: Number of image pairs per batch.
        num_workers: Parallel data loading workers.
        patch_size: Spatial size of training patches.
        normalization: Normalization strategy ("imagenet" or "minmax").
        oversample: Whether to oversample patches with changes.
        band_indices: Optional band selection for multispectral data.

    Returns:
        Dictionary with "train", "val", and "test" DataLoader instances.
    """
    data_root = Path(data_root)
    loaders = {}

    for split in ["train", "val", "test"]:
        split_dir = data_root / split
        if not split_dir.exists():
            print(f"Warning: {split_dir} not found, skipping {split} split.")
            continue

        transform = get_transforms(split, patch_size, normalization)
        dataset = ChangeDetectionDataset(
            root=split_dir,
            split=split,
            transform=transform,
            band_indices=band_indices,
        )

        sampler = None
        shuffle = split == "train"

        if split == "train" and oversample:
            sampler = create_weighted_sampler(dataset)
            shuffle = False  # Sampler handles randomization

        loaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=(split == "train"),
        )

    return loaders
