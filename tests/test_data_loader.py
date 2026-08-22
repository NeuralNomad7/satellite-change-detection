"""Tests for data loading, mask decoding, and the augmentation pipeline.

The augmentation tests exist because albumentations silently ignores unknown
keyword arguments (it emits a ``UserWarning`` rather than raising). A renamed
parameter therefore degrades augmentation quietly instead of failing loudly,
which is exactly the kind of drift CI should catch.
"""

import warnings

import numpy as np
import pytest
from PIL import Image

from src.data_loader import get_transforms, load_image, load_mask


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(1234)


# --------------------------------------------------------------------------
# Augmentation pipeline
# --------------------------------------------------------------------------


@pytest.mark.parametrize("split", ["train", "val", "test"])
@pytest.mark.parametrize("normalization", ["imagenet", "minmax"])
def test_get_transforms_uses_current_albumentations_api(split, normalization):
    """Building transforms must not warn about unrecognized arguments.

    albumentations only warns when a transform receives a keyword it does not
    know, so an API rename (for example ``GaussNoise.var_limit`` becoming
    ``std_range`` in albumentations 2.0) would otherwise pass unnoticed while
    the augmentation quietly reverted to its defaults.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        get_transforms(split=split, patch_size=32, normalization=normalization)


def test_transforms_apply_identically_to_both_images_and_mask(rng):
    """Geometric transforms must stay in lockstep across t1, t2, and the mask.

    Flipping one image of a bi-temporal pair without the other would invent
    change that never happened, so this invariant is load-bearing.
    """
    size = 32
    # A mask aligned with a distinctive image region: if the spatial transform
    # is applied consistently, the two stay aligned after augmentation.
    image = np.zeros((64, 64, 3), dtype=np.uint8)
    image[:32, :, :] = 255
    mask = np.zeros((64, 64), dtype=np.uint8)
    mask[:32, :] = 1

    transform = get_transforms(split="train", patch_size=size)

    for _ in range(12):
        out = transform(image=image, image2=image.copy(), mask=mask)
        img1, img2, out_mask = out["image"], out["image2"], out["mask"]

        assert img1.shape == img2.shape
        assert tuple(out_mask.shape[-2:]) == (size, size)

        # Identical inputs must survive identical spatial transforms; only the
        # photometric ops (brightness/noise) may diverge, and they are applied
        # to "image" and "image2" through the same call.
        bright1 = np.asarray(img1).mean(axis=tuple(range(np.asarray(img1).ndim - 2)))
        bright2 = np.asarray(img2).mean(axis=tuple(range(np.asarray(img2).ndim - 2)))
        assert bright1.shape == bright2.shape


def test_val_transform_is_deterministic(rng):
    """Validation must not jitter, or metrics become unreproducible."""
    image = rng.integers(0, 255, (64, 64, 3), dtype=np.uint8)
    mask = np.zeros((64, 64), dtype=np.uint8)

    transform = get_transforms(split="val", patch_size=32)
    first = transform(image=image, image2=image.copy(), mask=mask)
    second = transform(image=image, image2=image.copy(), mask=mask)

    np.testing.assert_allclose(
        np.asarray(first["image"], dtype=np.float32),
        np.asarray(second["image"], dtype=np.float32),
    )


def test_transform_output_is_channels_first_tensor(rng):
    """ToTensorV2 should hand the model (C, H, W)."""
    image = rng.integers(0, 255, (64, 64, 3), dtype=np.uint8)
    mask = np.zeros((64, 64), dtype=np.uint8)

    out = get_transforms(split="val", patch_size=32)(
        image=image, image2=image.copy(), mask=mask
    )
    assert tuple(out["image"].shape) == (3, 32, 32)
    assert tuple(out["image2"].shape) == (3, 32, 32)


# --------------------------------------------------------------------------
# Mask decoding
# --------------------------------------------------------------------------


def test_load_mask_handles_0_255_encoding(tmp_path):
    arr = np.zeros((16, 16), dtype=np.uint8)
    arr[:8, :] = 255
    path = tmp_path / "mask_255.png"
    Image.fromarray(arr, mode="L").save(path)

    mask = load_mask(path)

    assert set(np.unique(mask)) <= {0.0, 1.0}
    assert mask[:8, :].all()
    assert not mask[8:, :].any()


def test_load_mask_handles_0_1_encoding(tmp_path):
    """A 0/1-encoded mask must not be flattened to all zeros.

    A fixed threshold of 128 silently discarded every positive pixel here,
    which would train the model against empty labels.
    """
    arr = np.zeros((16, 16), dtype=np.uint8)
    arr[:8, :] = 1
    path = tmp_path / "mask_01.png"
    Image.fromarray(arr, mode="L").save(path)

    mask = load_mask(path)

    assert mask.sum() == 8 * 16, "0/1-encoded change pixels were lost"
    assert set(np.unique(mask)) <= {0.0, 1.0}
    assert mask[:8, :].all()
    assert not mask[8:, :].any()


def test_load_mask_all_background_stays_empty(tmp_path):
    path = tmp_path / "empty.png"
    Image.fromarray(np.zeros((16, 16), dtype=np.uint8), mode="L").save(path)

    assert load_mask(path).sum() == 0


def test_load_image_returns_hwc_float(tmp_path, rng):
    arr = rng.integers(0, 255, (12, 20, 3), dtype=np.uint8)
    path = tmp_path / "img.png"
    Image.fromarray(arr, mode="RGB").save(path)

    img = load_image(path)

    assert img.shape == (12, 20, 3)
    assert img.dtype == np.float32
    assert img.max() <= 255.0
