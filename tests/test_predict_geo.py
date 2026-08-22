"""Tests for the georeferenced prediction CLI helpers (``scripts.predict_geo``).

These cover the cloud-mask plumbing that connects ``sat-cd-ingest`` output to
inference, without needing a checkpoint or any network access.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
from rasterio.crs import CRS
from rasterio.transform import from_origin

from scripts.predict_geo import load_invalid_mask, resolve_cloud_mask_paths
from src.geo import write_geotiff

UTM_CRS = CRS.from_epsg(32631)
UTM_TRANSFORM = from_origin(500_000.0, 5_000_000.0, 10.0, 10.0)


def _write_manifest(tmp_path, outputs: dict) -> str:
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps({"outputs": outputs}))
    return str(path)


def _write_cloud(tmp_path, name: str, arr: np.ndarray) -> str:
    path = tmp_path / name
    write_geotiff(
        path, arr, transform=UTM_TRANSFORM, crs=UTM_CRS, nodata=0, dtype="uint8"
    )
    return str(path)


def test_resolve_reads_both_masks_from_manifest(tmp_path):
    manifest = _write_manifest(
        tmp_path, {"before_cloud": "/a/before_cloud.tif", "after_cloud": "/a/after.tif"}
    )

    assert resolve_cloud_mask_paths(manifest, None, None) == (
        "/a/before_cloud.tif",
        "/a/after.tif",
    )


def test_explicit_flags_override_the_manifest(tmp_path):
    manifest = _write_manifest(
        tmp_path, {"before_cloud": "/from/manifest.tif", "after_cloud": "/from/m2.tif"}
    )

    assert resolve_cloud_mask_paths(manifest, "/explicit.tif", None) == (
        "/explicit.tif",
        "/from/m2.tif",
    )


def test_resolve_without_manifest_passes_flags_through():
    assert resolve_cloud_mask_paths(None, "/a.tif", "/b.tif") == ("/a.tif", "/b.tif")
    assert resolve_cloud_mask_paths(None, None, None) == (None, None)


def test_manifest_without_cloud_masks_warns_and_continues(tmp_path, capsys):
    manifest = _write_manifest(tmp_path, {"before": "/a/before.tif"})

    assert resolve_cloud_mask_paths(manifest, None, None) == (None, None)
    assert "no cloud masks" in capsys.readouterr().out


def test_load_invalid_mask_unions_both_dates(tmp_path):
    t1 = np.zeros((16, 16), dtype=np.uint8)
    t1[:4, :] = 1
    t2 = np.zeros((16, 16), dtype=np.uint8)
    t2[-4:, :] = 1

    combined = load_invalid_mask(
        _write_cloud(tmp_path, "t1.tif", t1),
        _write_cloud(tmp_path, "t2.tif", t2),
        (16, 16),
    )

    assert combined.sum() == 128
    assert combined[:4].all() and combined[-4:].all()


def test_load_invalid_mask_returns_none_when_absent():
    assert load_invalid_mask(None, None, (16, 16)) is None


def test_load_invalid_mask_rejects_wrong_grid(tmp_path):
    path = _write_cloud(tmp_path, "t1.tif", np.zeros((8, 8), dtype=np.uint8))

    with pytest.raises(ValueError, match="expected"):
        load_invalid_mask(path, None, (16, 16))
