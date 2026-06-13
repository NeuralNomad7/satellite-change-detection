"""Tests for the geospatial I/O and vectorization module (``src.geo``).

These are intentionally torch-free: they exercise GeoTIFF read/write roundtrips,
polygonization, area math, and change statistics using only rasterio + shapely +
numpy, so they run without the deep-learning stack installed.
"""

from __future__ import annotations

import numpy as np
import pytest
from rasterio.crs import CRS
from rasterio.transform import from_origin

from src.geo import (
    change_statistics,
    mask_to_polygons,
    normalize_imagenet,
    read_geotiff,
    write_geotiff,
)

# A UTM zone 31N grid (metres) with 10 m pixels, centred near 45 deg N.
UTM_CRS = CRS.from_epsg(32631)
UTM_TRANSFORM = from_origin(500_000.0, 5_000_000.0, 10.0, 10.0)


def _rectangle_mask(rows: int = 20, cols: int = 10, size: int = 100) -> np.ndarray:
    """A ``size``x``size`` mask with a single ``rows``x``cols`` change block."""
    mask = np.zeros((size, size), dtype=np.uint8)
    mask[10 : 10 + rows, 10 : 10 + cols] = 1
    return mask


def test_write_read_roundtrip_preserves_geometadata(tmp_path):
    data = (np.random.rand(32, 48, 3) * 1000).astype(np.uint16)
    path = tmp_path / "img.tif"

    write_geotiff(path, data, transform=UTM_TRANSFORM, crs=UTM_CRS, dtype="uint16")
    raster = read_geotiff(path)

    assert raster.data.shape == (32, 48, 3)
    np.testing.assert_array_equal(raster.data.astype(np.uint16), data)
    assert raster.crs == UTM_CRS
    assert list(raster.transform)[:6] == pytest.approx(list(UTM_TRANSFORM)[:6])


def test_write_geotiff_accepts_2d_mask(tmp_path):
    mask = _rectangle_mask()
    path = tmp_path / "mask.tif"

    write_geotiff(path, mask, transform=UTM_TRANSFORM, crs=UTM_CRS, dtype="uint8")
    raster = read_geotiff(path)

    assert raster.count == 1
    assert raster.data.shape == (100, 100, 1)


def test_mask_to_polygons_area_in_hectares_utm():
    # 20 rows x 10 cols of 10 m pixels = 200 m x 100 m = 20000 m^2 = 2 ha.
    mask = _rectangle_mask(rows=20, cols=10)

    fc = mask_to_polygons(mask, transform=UTM_TRANSFORM, crs=UTM_CRS)

    assert fc["type"] == "FeatureCollection"
    assert len(fc["features"]) == 1
    feature = fc["features"][0]
    assert feature["properties"]["area_ha"] == pytest.approx(2.0, rel=1e-6)
    assert feature["properties"]["area_m2"] == pytest.approx(20_000.0, rel=1e-6)
    # Geometry is reprojected to WGS84 lon/lat (UTM 31N -> roughly 3 deg E, 45 N).
    lon, lat = feature["geometry"]["coordinates"][0][0]
    assert 0.0 < lon < 6.0
    assert 44.0 < lat < 46.0
    # Centroid is reported in lon/lat too.
    assert "centroid_lon" in feature["properties"]
    assert "centroid_lat" in feature["properties"]


def test_min_area_filter_drops_small_regions():
    mask = _rectangle_mask(rows=20, cols=10)  # 20000 m^2

    kept = mask_to_polygons(mask, transform=UTM_TRANSFORM, crs=UTM_CRS, min_area_m2=0.0)
    dropped = mask_to_polygons(
        mask, transform=UTM_TRANSFORM, crs=UTM_CRS, min_area_m2=50_000.0
    )

    assert len(kept["features"]) == 1
    assert len(dropped["features"]) == 0


def test_change_statistics_reports_area_and_fraction():
    mask = _rectangle_mask(rows=20, cols=10)

    stats = change_statistics(mask, transform=UTM_TRANSFORM, crs=UTM_CRS)

    assert stats["changed_pixels"] == 200
    assert stats["total_pixels"] == 10_000
    assert stats["change_fraction"] == pytest.approx(0.02)
    assert stats["num_change_regions"] == 1
    assert stats["changed_area_ha"] == pytest.approx(2.0, rel=1e-6)
    assert stats["pixel_size_m"] == [10.0, 10.0]
    assert "32631" in stats["crs"]


def test_geographic_crs_area_is_positive():
    mask = _rectangle_mask(rows=10, cols=10)
    transform = from_origin(10.0, 45.0, 0.001, 0.001)  # degrees
    crs = CRS.from_epsg(4326)

    fc = mask_to_polygons(mask, transform=transform, crs=crs)

    assert len(fc["features"]) == 1
    area_ha = fc["features"][0]["properties"]["area_ha"]
    assert area_ha is not None
    assert 0.0 < area_ha < 100.0
    # Already WGS84, so geometry stays in the source lon/lat range.
    lon, lat = fc["features"][0]["geometry"]["coordinates"][0][0]
    assert 9.0 < lon < 11.0
    assert 44.0 < lat < 46.0


def test_no_crs_yields_null_area():
    mask = _rectangle_mask()

    fc = mask_to_polygons(mask, transform=None, crs=None)
    stats = change_statistics(mask, transform=None, crs=None, polygons=fc)

    assert len(fc["features"]) == 1
    assert fc["features"][0]["properties"]["area_m2"] is None
    assert stats["changed_area_ha"] is None
    assert stats["crs"] is None
    assert stats["num_change_regions"] == 1


def test_empty_mask_has_no_regions():
    mask = np.zeros((50, 50), dtype=np.uint8)

    stats = change_statistics(mask, transform=UTM_TRANSFORM, crs=UTM_CRS)

    assert stats["changed_pixels"] == 0
    assert stats["change_fraction"] == 0.0
    assert stats["num_change_regions"] == 0
    assert stats["changed_area_ha"] == pytest.approx(0.0)


def test_normalize_imagenet_shape_and_values():
    data = np.full((8, 8, 3), 255.0, dtype=np.float32)

    out = normalize_imagenet(data, in_channels=3)

    assert out.shape == (1, 3, 8, 8)
    assert out.dtype == np.float32
    # Channel 0: (1.0 - 0.485) / 0.229
    expected = (1.0 - 0.485) / 0.229
    assert out[0, 0].mean() == pytest.approx(expected, rel=1e-4)


def test_normalize_imagenet_rejects_too_few_bands():
    data = np.zeros((8, 8, 1), dtype=np.float32)

    with pytest.raises(ValueError, match="expects"):
        normalize_imagenet(data, in_channels=3)
