"""Tests for the Sentinel-2 ingestion module (``src.ingest``).

These are intentionally network-free and torch-free. The pure helpers (grid
construction, SCL cloud classification, reprojection, scene selection,
reflectance scaling) are tested directly, and the full :func:`ingest_pair`
orchestration is exercised against a fake STAC catalog backed by local
synthetic GeoTIFFs -- so the whole pipeline is covered without ever touching
the Planetary Computer or the deep-learning stack.
"""

from __future__ import annotations

import json
from dataclasses import FrozenInstanceError
from types import SimpleNamespace

import numpy as np
import pytest
from rasterio.crs import CRS
from rasterio.transform import from_origin

from src.geo import write_geotiff
from src.ingest import (
    SCL_CLOUD_CLASSES,
    TargetGrid,
    build_target_grid,
    cloud_fraction,
    ingest_pair,
    reproject_to_grid,
    scale_reflectance,
    scl_to_masks,
    select_least_cloudy,
    utm_epsg_for_lonlat,
)

# A small AOI near Venice, Italy (UTM zone 33N).
BBOX = (12.40, 45.40, 12.42, 45.42)


# ---------------------------------------------------------------------------
# utm_epsg_for_lonlat
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("lon", "lat", "expected"),
    [
        (12.4, 45.4, 32633),  # Venice -> zone 33N
        (-122.3, 37.8, 32610),  # San Francisco -> zone 10N
        (151.2, -33.9, 32756),  # Sydney -> zone 56S
        (2.35, 48.85, 32631),  # Paris -> zone 31N
    ],
)
def test_utm_epsg_for_lonlat(lon, lat, expected):
    assert utm_epsg_for_lonlat(lon, lat) == expected


# ---------------------------------------------------------------------------
# build_target_grid
# ---------------------------------------------------------------------------


def test_build_target_grid_auto_utm():
    grid = build_target_grid(BBOX, resolution=10.0)
    assert grid.crs == CRS.from_epsg(32633)
    # ~0.02 deg of longitude near 45N is ~1.5 km -> ~150 px at 10 m.
    assert 100 < grid.width < 250
    assert 150 < grid.height < 300
    assert grid.shape == (grid.height, grid.width)
    # Pixel size is the requested resolution.
    assert grid.transform.a == pytest.approx(10.0)
    assert grid.transform.e == pytest.approx(-10.0)


def test_build_target_grid_explicit_crs_and_resolution():
    grid = build_target_grid(BBOX, resolution=20.0, dst_crs="EPSG:32633")
    assert grid.crs == CRS.from_epsg(32633)
    assert grid.transform.a == pytest.approx(20.0)
    # Half the resolution of the 10 m grid -> roughly half the pixel count.
    grid10 = build_target_grid(BBOX, resolution=10.0, dst_crs="EPSG:32633")
    assert grid.width == pytest.approx(grid10.width // 2, abs=2)


def test_build_target_grid_rejects_inverted_bbox():
    with pytest.raises(ValueError, match="Invalid bbox"):
        build_target_grid((12.42, 45.42, 12.40, 45.40), resolution=10.0)


# ---------------------------------------------------------------------------
# scl_to_masks / cloud_fraction
# ---------------------------------------------------------------------------


def test_scl_to_masks_classifies_clouds_and_nodata():
    scl = np.array(
        [
            [4, 5, 6],  # vegetation, bare, water -> clear
            [8, 9, 10],  # cloud med/high, cirrus -> cloud
            [0, 1, 3],  # nodata, defective, cloud shadow
        ],
        dtype=np.uint8,
    )
    cloud, nodata = scl_to_masks(scl)
    assert cloud.tolist() == [
        [False, False, False],
        [True, True, True],
        [False, False, True],
    ]
    assert nodata.tolist() == [
        [False, False, False],
        [False, False, False],
        [True, True, False],
    ]


def test_cloud_fraction_excludes_nodata():
    # 9 pixels: 2 nodata (class 0), 3 cloud (class 9), 4 clear (class 4).
    scl = np.array([0, 0, 9, 9, 9, 4, 4, 4, 4], dtype=np.uint8)
    # Valid pixels = 7; cloud = 3 -> 3/7.
    assert cloud_fraction(scl) == pytest.approx(3.0 / 7.0)


def test_cloud_fraction_all_nodata_is_one():
    assert cloud_fraction(np.zeros((4, 4), dtype=np.uint8)) == 1.0


def test_scl_cloud_classes_constant():
    assert SCL_CLOUD_CLASSES == {3, 8, 9, 10}


# ---------------------------------------------------------------------------
# select_least_cloudy
# ---------------------------------------------------------------------------


def _fake_item(item_id, cloud_cover):
    props = {"eo:cloud_cover": cloud_cover} if cloud_cover is not None else {}
    return SimpleNamespace(id=item_id, properties=props)


def test_select_least_cloudy_picks_minimum():
    items = [_fake_item("a", 40.0), _fake_item("b", 5.0), _fake_item("c", 22.0)]
    assert select_least_cloudy(items).id == "b"


def test_select_least_cloudy_handles_missing_property():
    items = [_fake_item("missing", None), _fake_item("present", 30.0)]
    assert select_least_cloudy(items).id == "present"


def test_select_least_cloudy_empty_raises():
    with pytest.raises(ValueError, match="No scenes"):
        select_least_cloudy([])


# ---------------------------------------------------------------------------
# scale_reflectance
# ---------------------------------------------------------------------------


def test_scale_reflectance_maps_to_uint8():
    raw = np.array([0, 1500, 3000, 6000], dtype=np.uint16)
    scaled = scale_reflectance(raw, ceiling=3000.0)
    assert scaled.dtype == np.uint8
    assert scaled[0] == 0
    assert scaled[1] == pytest.approx(128, abs=1)
    assert scaled[2] == 255
    assert scaled[3] == 255  # clipped at the ceiling


# ---------------------------------------------------------------------------
# reproject_to_grid (local synthetic raster, no network)
# ---------------------------------------------------------------------------


def _write_wgs84_constant(path, value, bands=1, pad=0.005):
    """Write a constant-valued EPSG:4326 GeoTIFF covering (a padded) BBOX."""
    min_lon, min_lat, max_lon, max_lat = BBOX
    res = 0.0005  # ~50 m, fine enough to resample from
    west, north = min_lon - pad, max_lat + pad
    width = int((max_lon + pad - west) / res)
    height = int((north - (min_lat - pad)) / res)
    transform = from_origin(west, north, res, res)
    shape = (height, width) if bands == 1 else (height, width, bands)
    array = np.full(shape, value, dtype=np.uint8)
    write_geotiff(path, array, transform=transform, crs="EPSG:4326", dtype="uint8")


def test_reproject_to_grid_warps_to_target(tmp_path):
    src = tmp_path / "src.tif"
    _write_wgs84_constant(src, value=100, bands=3)
    grid = build_target_grid(BBOX, resolution=10.0)

    out = reproject_to_grid(src, grid)

    assert out.shape == (grid.height, grid.width, 3)
    # The grid is the BBOX reprojected; the source fully covers it, so the
    # centre pixel must carry the constant source value.
    cy, cx = grid.height // 2, grid.width // 2
    assert out[cy, cx, 0] == 100


def test_reproject_to_grid_band_indexes_and_dtype(tmp_path):
    src = tmp_path / "rgb.tif"
    _write_wgs84_constant(src, value=42, bands=3)
    grid = build_target_grid(BBOX, resolution=10.0)

    out = reproject_to_grid(src, grid, band_indexes=[1])
    assert out.shape == (grid.height, grid.width, 1)


# ---------------------------------------------------------------------------
# ingest_pair end-to-end against a fake STAC catalog
# ---------------------------------------------------------------------------


class _FakeAsset:
    def __init__(self, href):
        self.href = href


class _FakeItem:
    def __init__(self, item_id, datetime, cloud_cover, visual_href, scl_href):
        self.id = item_id
        self.properties = {"datetime": datetime, "eo:cloud_cover": cloud_cover}
        self.assets = {
            "visual": _FakeAsset(visual_href),
            "SCL": _FakeAsset(scl_href),
        }


class _FakeSearch:
    def __init__(self, items):
        self._items = items

    def items(self):
        return list(self._items)


class _FakeCatalog:
    """Returns preset items keyed by the requested datetime string."""

    def __init__(self, by_datetime):
        self._by_datetime = by_datetime

    def search(self, *, collections, bbox, datetime, query, max_items):
        return _FakeSearch(self._by_datetime.get(datetime, []))


def _make_scene_files(tmp_path, tag, rgb_value, cloud_block):
    """Write a synthetic visual (3-band) + SCL raster pair, return their paths."""
    visual = tmp_path / f"{tag}_visual.tif"
    _write_wgs84_constant(visual, value=rgb_value, bands=3)

    # SCL: mostly vegetation (4), with a block of high-probability cloud (9).
    min_lon, min_lat, max_lon, max_lat = BBOX
    res = 0.0005
    pad = 0.005
    west, north = min_lon - pad, max_lat + pad
    width = int((max_lon + pad - west) / res)
    height = int((north - (min_lat - pad)) / res)
    scl_arr = np.full((height, width), 4, dtype=np.uint8)
    if cloud_block:
        scl_arr[: height // 4, : width // 4] = 9
    scl_path = tmp_path / f"{tag}_scl.tif"
    write_geotiff(
        scl_path,
        scl_arr,
        transform=from_origin(west, north, res, res),
        crs="EPSG:4326",
        dtype="uint8",
    )
    return str(visual), str(scl_path)


def test_ingest_pair_writes_coregistered_pair(tmp_path):
    before_v, before_scl = _make_scene_files(tmp_path, "before", 80, cloud_block=False)
    after_v, after_scl = _make_scene_files(tmp_path, "after", 160, cloud_block=True)

    catalog = _FakeCatalog(
        {
            "2023-06-01/2023-06-30": [
                _FakeItem(
                    "S2_BEFORE", "2023-06-15T10:00:00Z", 4.0, before_v, before_scl
                )
            ],
            "2024-06-01/2024-06-30": [
                _FakeItem("S2_AFTER", "2024-06-12T10:00:00Z", 8.0, after_v, after_scl)
            ],
        }
    )

    out_dir = tmp_path / "out"
    manifest = ingest_pair(
        bbox=BBOX,
        datetime_t1="2023-06-01/2023-06-30",
        datetime_t2="2024-06-01/2024-06-30",
        output_dir=out_dir,
        resolution=10.0,
        catalog=catalog,
    )

    # Both images plus cloud masks and a manifest were written.
    before_tif = out_dir / "before.tif"
    after_tif = out_dir / "after.tif"
    assert before_tif.exists() and after_tif.exists()
    assert (out_dir / "before_cloud.tif").exists()
    assert (out_dir / "after_cloud.tif").exists()
    assert (out_dir / "manifest.json").exists()

    # Manifest records both scenes and the grid.
    assert manifest["scenes"]["before"]["item_id"] == "S2_BEFORE"
    assert manifest["scenes"]["after"]["item_id"] == "S2_AFTER"
    assert manifest["grid"]["crs"] == CRS.from_epsg(32633).to_string()

    # The "after" scene had a cloud block; the "before" scene had none.
    assert manifest["scenes"]["after"]["aoi_cloud_fraction"] > 0.0
    assert manifest["scenes"]["before"]["aoi_cloud_fraction"] == 0.0

    # Outputs are genuinely co-registered: identical CRS, transform and size.
    import rasterio

    with rasterio.open(before_tif) as b, rasterio.open(after_tif) as a:
        assert b.crs == a.crs == CRS.from_epsg(32633)
        assert b.transform == a.transform
        assert (b.width, b.height) == (a.width, a.height)
        assert b.count == 3

    # Manifest round-trips as JSON.
    with open(out_dir / "manifest.json") as f:
        on_disk = json.load(f)
    assert on_disk["scenes"]["before"]["item_id"] == "S2_BEFORE"


def test_target_grid_is_frozen():
    grid = build_target_grid(BBOX, resolution=10.0)
    assert isinstance(grid, TargetGrid)
    with pytest.raises(FrozenInstanceError):
        grid.width = 5  # type: ignore[misc]
