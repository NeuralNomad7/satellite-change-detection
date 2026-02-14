"""Tests for utility functions: metrics, visualization, and seeding."""

import numpy as np
import pytest

from src.utils import (
    compute_confusion_matrix,
    compute_iou,
    compute_metrics,
    normalize_for_display,
    set_seed,
)


class TestMetrics:
    """Test metric computation functions."""

    def test_iou_perfect(self) -> None:
        """IoU of identical masks should be 1.0."""
        mask = np.ones((32, 32), dtype=int)
        assert compute_iou(mask, mask) == 1.0

    def test_iou_no_overlap(self) -> None:
        """IoU of non-overlapping masks should be 0.0."""
        pred = np.zeros((32, 32), dtype=int)
        target = np.ones((32, 32), dtype=int)
        assert compute_iou(pred, target) == 0.0

    def test_iou_empty(self) -> None:
        """IoU of two empty masks should be 0.0."""
        empty = np.zeros((32, 32), dtype=int)
        assert compute_iou(empty, empty) == 0.0

    def test_iou_partial(self) -> None:
        """IoU should be between 0 and 1 for partial overlap."""
        pred = np.zeros((10, 10), dtype=int)
        target = np.zeros((10, 10), dtype=int)
        pred[:5, :5] = 1
        target[3:8, 3:8] = 1
        iou = compute_iou(pred, target)
        assert 0.0 < iou < 1.0

    def test_compute_metrics_keys(self) -> None:
        """compute_metrics should return all expected metric keys."""
        pred = np.random.randint(0, 2, (32, 32))
        target = np.random.randint(0, 2, (32, 32))
        metrics = compute_metrics(pred, target)
        expected_keys = {"f1", "iou", "precision", "recall", "accuracy"}
        assert set(metrics.keys()) == expected_keys

    def test_compute_metrics_perfect(self) -> None:
        """All metrics should be 1.0 for perfect predictions."""
        mask = np.ones((32, 32), dtype=int)
        metrics = compute_metrics(mask, mask)
        assert metrics["f1"] == 1.0
        assert metrics["precision"] == 1.0
        assert metrics["recall"] == 1.0
        assert metrics["accuracy"] == 1.0

    def test_confusion_matrix_shape(self) -> None:
        """Confusion matrix should be 2x2."""
        pred = np.random.randint(0, 2, (32, 32))
        target = np.random.randint(0, 2, (32, 32))
        cm = compute_confusion_matrix(pred, target)
        assert cm.shape == (2, 2)

    def test_confusion_matrix_sum(self) -> None:
        """Confusion matrix entries should sum to total pixels."""
        pred = np.random.randint(0, 2, (32, 32))
        target = np.random.randint(0, 2, (32, 32))
        cm = compute_confusion_matrix(pred, target)
        assert cm.sum() == 32 * 32


class TestVisualization:
    """Test visualization helper functions."""

    def test_normalize_multichannel(self) -> None:
        """normalize_for_display should produce (H, W, 3) from (C, H, W)."""
        img = np.random.rand(3, 64, 64).astype(np.float32) * 255
        result = normalize_for_display(img)
        assert result.shape == (64, 64, 3)
        assert result.min() >= 0.0
        assert result.max() <= 1.0

    def test_normalize_single_channel(self) -> None:
        """Should handle single-channel (grayscale) images."""
        img = np.random.rand(64, 64).astype(np.float32) * 255
        result = normalize_for_display(img)
        assert result.shape == (64, 64)
        assert result.min() >= 0.0
        assert result.max() <= 1.0

    def test_normalize_many_channels(self) -> None:
        """Should handle >3 channels by taking first 3."""
        img = np.random.rand(13, 64, 64).astype(np.float32) * 255
        result = normalize_for_display(img)
        assert result.shape == (64, 64, 3)


class TestSeeding:
    """Test reproducibility utilities."""

    def test_seed_numpy(self) -> None:
        """Same seed should produce same numpy random values."""
        set_seed(42)
        a = np.random.rand(10)
        set_seed(42)
        b = np.random.rand(10)
        np.testing.assert_array_equal(a, b)

    def test_different_seeds(self) -> None:
        """Different seeds should produce different values."""
        set_seed(42)
        a = np.random.rand(10)
        set_seed(99)
        b = np.random.rand(10)
        assert not np.array_equal(a, b)
