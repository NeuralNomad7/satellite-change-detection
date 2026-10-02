"""Tests for the Sentinel-2 ingestion module (``src.ingest``).

These are intentionally network-free and torch-free. The pure helpers (grid
construction, SCL cloud classification, scene ranking, processing-baseline
offsets, reprojection, reflectance scaling) are tested directly, and the full
:func:`ingest_pair` orchestration is exercised against a fake STAC catalog
backed by local synthetic GeoTIFFs -- so the whole pipeline is covered without
ever touching the Planetary Computer or the deep-learning stack.
"""

from __future__ import annotations

import json
import sys
from dataclasses import FrozenInstanceError
from types import SimpleNamespace

import numpy as np
import pytest
import rasterio
from rasterio.crs import CRS
from rasterio.env import get_gdal_config
from rasterio.transform import from_origin

import src
from src.geo import write_geotiff
from src.ingest import (
    _GDAL_CLOUD_OPTIONS,
    BOA_ADD_OFFSET,
    DEFAULT_CANDIDATES,
    MASK_CLEAR,
    MASK_CLOUD,
    MASK_NODATA,
    RGB_BANDS,
    SCL_CLOUD_CLASSES,
    STAC_URL,
    TargetGrid,
    _gdal_cloud_env,
    aoi_coverage,
    boa_offset_for_baseline,
    boa_offset_for_item,
    build_target_grid,
    check_scene_pair,
    cloud_fraction,
    ingest_pair,
    invalid_fraction,
    open_catalog,
    rank_candidates,
    read_item_imagery,
    reproject_to_grid,
    scale_reflectance,
    scl_to_masks,
    search_items,
    select_least_cloudy,
    select_scene,
    utm_epsg_for_lonlat,
    validate_bbox,
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


@pytest.mark.parametrize("resolution", [0.0, -10.0, float("nan"), float("inf")])
def test_build_target_grid_rejects_bad_resolution(resolution):
    with pytest.raises(ValueError, match="resolution"):
        build_target_grid(BBOX, resolution=resolution)


# ---------------------------------------------------------------------------
# validate_bbox
# ---------------------------------------------------------------------------


def test_validate_bbox_returns_four_floats():
    assert validate_bbox(["12.3", 45, 12.45, 45.5]) == (12.3, 45.0, 12.45, 45.5)


@pytest.mark.parametrize(
    ("bbox", "match"),
    [
        ((12.3, 45.4, 12.45), "4 finite numbers"),
        ((12.3, 45.4, 12.45, float("nan")), "4 finite numbers"),
        (("east", 45.4, 12.45, 45.5), "expected 4 numbers"),
        (None, "expected 4 numbers"),
        # Tokyo with latitude first: 139.6 is no latitude.
        ((35.6, 139.6, 35.8, 139.8), "longitude comes first"),
        ((-200.0, 0.0, 10.0, 1.0), "longitude"),
        ((12.45, 45.4, 12.3, 45.5), "min < max"),
        ((12.3, 45.4, 12.3, 45.5), "min < max"),
    ],
)
def test_validate_bbox_rejects(bbox, match):
    with pytest.raises(ValueError, match=match):
        validate_bbox(bbox)


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


def test_invalid_fraction_counts_cloud_and_nodata():
    # 10 pixels: 2 nodata, 3 cloud, 5 clear -> half the AOI is unusable.
    scl = np.array([0, 0, 9, 9, 9, 4, 4, 4, 4, 4], dtype=np.uint8)
    assert invalid_fraction(scl) == pytest.approx(0.5)


def test_invalid_fraction_empty_is_one():
    assert invalid_fraction(np.zeros((0, 0), dtype=np.uint8)) == 1.0


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
# aoi_coverage / rank_candidates
# ---------------------------------------------------------------------------


def _footprint(min_lon, min_lat, max_lon, max_lat):
    """A GeoJSON polygon, as found in a STAC item's ``geometry``."""
    ring = [
        [min_lon, min_lat],
        [max_lon, min_lat],
        [max_lon, max_lat],
        [min_lon, max_lat],
        [min_lon, min_lat],
    ]
    return {"type": "Polygon", "coordinates": [ring]}


# Footprints relative to BBOX (12.40-12.42 E): all of it, the eastern half, and
# a sliver along the eastern edge (5%).
SLIVER_WEST = 12.419
FULL = _footprint(12.0, 45.0, 13.0, 46.0)
EAST_HALF = _footprint(12.41, 45.0, 13.0, 46.0)
EAST_SLIVER = _footprint(SLIVER_WEST, 45.0, 13.0, 46.0)


def _stac_item(item_id, cloud_cover=None, *, geometry=None, **properties):
    props = {key.replace("__", ":"): value for key, value in properties.items()}
    if cloud_cover is not None:
        props["eo:cloud_cover"] = cloud_cover
    return SimpleNamespace(id=item_id, properties=props, geometry=geometry)


def test_aoi_coverage_measures_the_share_of_the_aoi_with_data():
    assert aoi_coverage(_stac_item("a", geometry=FULL), BBOX) == pytest.approx(1.0)
    assert aoi_coverage(_stac_item("b", geometry=EAST_HALF), BBOX) == pytest.approx(0.5)
    far_away = _stac_item("c", geometry=_footprint(0.0, 0.0, 1.0, 1.0))
    assert aoi_coverage(far_away, BBOX) == 0.0


def test_aoi_coverage_is_unknown_without_a_usable_footprint():
    assert aoi_coverage(SimpleNamespace(id="x"), BBOX) is None
    assert aoi_coverage(_stac_item("y", geometry={"type": "Bogus"}), BBOX) is None


def test_rank_candidates_prefers_aoi_coverage_over_scene_cloud():
    # The README's Venice example: the least-cloudy tile barely reaches the AOI.
    sliver = _stac_item("sliver", 13.31, geometry=EAST_SLIVER)
    full = _stac_item("full", 13.94, geometry=FULL)
    assert [i.id for i in rank_candidates([sliver, full], BBOX)] == [
        "full",
        "sliver",
    ]


def test_rank_candidates_falls_back_to_scene_cloud_cover():
    items = [_stac_item("cloudy", 40.0), _stac_item("clear", 5.0), _stac_item("?")]
    assert [i.id for i in rank_candidates(items, BBOX)] == ["clear", "cloudy", "?"]


def test_rank_candidates_keeps_the_newest_processing_of_an_acquisition():
    when = "2023-06-17T10:05:59.024000Z"
    old, new, other = (
        _stac_item(
            f"S2A_MSIL2A_20230617T100559_R022_T{tile}_{processed}",
            4.0,
            datetime=when,
            s2__mgrs_tile=tile,
        )
        for tile, processed in (
            ("32TQR", "20230617T154323"),
            ("32TQR", "20240926T031136"),
            ("33TUL", "20230617T154323"),
        )
    )
    ranked = rank_candidates([new, old, other], BBOX)
    assert [i.id for i in ranked] == [new.id, other.id]


# ---------------------------------------------------------------------------
# Processing-baseline reflectance offset
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("baseline", "expected"),
    [
        ("02.12", 0),
        ("03.00", 0),
        ("04.00", BOA_ADD_OFFSET),
        ("05.10", BOA_ADD_OFFSET),
        (None, 0),
        ("", 0),
        ("unknown", 0),
    ],
)
def test_boa_offset_for_baseline(baseline, expected):
    assert boa_offset_for_baseline(baseline) == expected


def test_boa_offset_for_item_skips_already_harmonized_catalogs():
    item = _stac_item("a", s2__processing_baseline="05.10")
    assert boa_offset_for_item(item) == -1000
    item.properties["earthsearch:boa_offset_applied"] = True
    assert boa_offset_for_item(item) == 0


# ---------------------------------------------------------------------------
# check_scene_pair
# ---------------------------------------------------------------------------


def test_check_scene_pair_rejects_the_same_scene():
    item = _stac_item("S2_X", datetime="2024-06-12T10:00:00Z")
    with pytest.raises(ValueError, match="same scene"):
        check_scene_pair(item, item)


def test_check_scene_pair_rejects_two_tiles_of_one_acquisition():
    a = _stac_item("S2_T32TQR", datetime="2024-06-12T10:00:00Z")
    b = _stac_item("S2_T33TUL", datetime="2024-06-12T10:00:00.000000Z")
    with pytest.raises(ValueError, match="same acquisition"):
        check_scene_pair(a, b)


def test_check_scene_pair_warns_when_dates_are_reversed():
    later = _stac_item("S2_2024", datetime="2024-06-12T10:00:00Z")
    earlier = _stac_item("S2_2023", datetime="2023-06-12T10:00:00Z")
    with pytest.warns(UserWarning, match="older"):
        check_scene_pair(later, earlier)


def test_check_scene_pair_accepts_ordered_or_undated_pairs(recwarn):
    check_scene_pair(
        _stac_item("a", datetime="2023-06-12T10:00:00Z"),
        _stac_item("b", datetime="2024-06-12T10:00:00Z"),
    )
    check_scene_pair(_stac_item("a"), _stac_item("b"))
    assert not [w for w in recwarn if issubclass(w.category, UserWarning)]


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


@pytest.mark.parametrize("ceiling", [0.0, -1.0, float("nan"), float("inf")])
def test_scale_reflectance_rejects_bad_ceiling(ceiling):
    with pytest.raises(ValueError, match="ceiling"):
        scale_reflectance(np.zeros(3, dtype=np.uint16), ceiling=ceiling)


# ---------------------------------------------------------------------------
# reproject_to_grid (local synthetic raster, no network)
# ---------------------------------------------------------------------------

_SRC_RES = 0.0005  # ~50 m, fine enough to resample from
_SRC_PAD = 0.005


def _source_grid(pad=_SRC_PAD):
    """``(height, width)`` and transform of an EPSG:4326 raster over a padded BBOX."""
    min_lon, min_lat, max_lon, max_lat = BBOX
    west, north = min_lon - pad, max_lat + pad
    width = int((max_lon + pad - west) / _SRC_RES)
    height = int((north - (min_lat - pad)) / _SRC_RES)
    return (height, width), from_origin(west, north, _SRC_RES, _SRC_RES)


def _write_wgs84(path, array, dtype="uint8"):
    """Write ``array`` as an EPSG:4326 GeoTIFF covering the padded BBOX."""
    _, transform = _source_grid()
    write_geotiff(path, array, transform=transform, crs="EPSG:4326", dtype=dtype)
    return str(path)


def _write_wgs84_constant(path, value, bands=1, dtype="uint8"):
    """Write a constant-valued EPSG:4326 GeoTIFF covering (a padded) BBOX."""
    (height, width), _ = _source_grid()
    shape = (height, width) if bands == 1 else (height, width, bands)
    return _write_wgs84(path, np.full(shape, value, dtype=dtype), dtype=dtype)


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


def test_reproject_to_grid_keeps_nodata_out_of_interpolation(tmp_path):
    # Western half empty (0), eastern half a constant 200 -- a scene edge.
    (height, width), _ = _source_grid()
    edge = np.full((height, width), 200, dtype=np.uint8)
    edge[:, : width // 2] = 0
    src = _write_wgs84(tmp_path / "edge.tif", edge)
    grid = build_target_grid(BBOX, resolution=10.0)

    blended = reproject_to_grid(src, grid)
    clean = reproject_to_grid(src, grid, src_nodata=0)

    # Without a nodata value, bilinear resampling smears the edge into a fringe
    # of darkened pixels; with it, every pixel is either empty or real data.
    assert not set(np.unique(blended)) <= {0, 200}
    assert set(np.unique(clean)) == {0, 200}


# ---------------------------------------------------------------------------
# read_item_imagery (raw-bands path)
# ---------------------------------------------------------------------------


def _band_item(tmp_path, tag, raw, baseline):
    """A fake item whose B04/B03/B02 assets all hold the ``raw`` uint16 array."""
    hrefs = {
        band: _write_wgs84(tmp_path / f"{tag}_{band}.tif", raw, dtype="uint16")
        for band in RGB_BANDS
    }
    scl = _write_wgs84_constant(tmp_path / f"{tag}_scl.tif", 4)
    return _FakeItem(
        f"S2_{tag.upper()}",
        "2024-06-12T10:00:00Z",
        5.0,
        visual_href="unused",
        scl_href=scl,
        bands=hrefs,
        properties={"s2:processing_baseline": baseline},
    )


@pytest.mark.parametrize(("baseline", "expected"), [("05.10", 102), ("03.00", 187)])
def test_read_item_imagery_bands_removes_the_baseline_offset(
    tmp_path, baseline, expected
):
    (height, width), _ = _source_grid()
    raw = np.full((height, width), 2200, dtype=np.uint16)
    raw[:, : width // 2] = 0  # no data in the western half
    item = _band_item(tmp_path, "scene", raw, baseline)
    grid = build_target_grid(BBOX, resolution=10.0)

    out = read_item_imagery(item, grid, asset="bands", reflectance_ceiling=3000.0)

    # Baseline 05.10 stores 1200 reflectance as 2200 DN -> 102/255 at a 3000
    # ceiling; baseline 03.00 stores it unshifted -> 187/255.
    valid = out[..., 0] > 0
    assert valid.any() and (~valid).any()
    assert set(np.unique(out[valid])) == {expected}
    assert (out[~valid] == 0).all()


def test_read_item_imagery_bands_keeps_dark_pixels_distinct_from_nodata(tmp_path):
    (height, width), _ = _source_grid()
    # 1000 DN at baseline 05.10 is zero reflectance -- black, but real data.
    raw = np.full((height, width), 1000, dtype=np.uint16)
    item = _band_item(tmp_path, "dark", raw, "05.10")
    grid = build_target_grid(BBOX, resolution=10.0)

    out = read_item_imagery(item, grid, asset="bands")

    assert out.min() == 1 and out.max() == 1


# ---------------------------------------------------------------------------
# ingest_pair end-to-end against a fake STAC catalog
# ---------------------------------------------------------------------------


class _FakeAsset:
    def __init__(self, href):
        self.href = href


class _FakeItem:
    def __init__(
        self,
        item_id,
        datetime,
        cloud_cover,
        visual_href,
        scl_href,
        *,
        geometry=None,
        bands=None,
        properties=None,
    ):
        self.id = item_id
        self.properties = {
            "datetime": datetime,
            "eo:cloud_cover": cloud_cover,
            **(properties or {}),
        }
        self.geometry = geometry
        self.assets = {
            "visual": _FakeAsset(visual_href),
            "SCL": _FakeAsset(scl_href),
        }
        for key, href in (bands or {}).items():
            self.assets[key] = _FakeAsset(href)


class _FakeSearch:
    def __init__(self, items):
        self._items = items

    def items(self):
        return list(self._items)


class _FakeCatalog:
    """Returns preset items keyed by the requested datetime string."""

    def __init__(self, by_datetime, url="https://stac.example/v1"):
        self._by_datetime = by_datetime
        self._url = url
        self.searches = []

    def get_self_href(self):
        return self._url

    def search(self, *, collections, bbox, datetime, query, sortby, max_items):
        self.searches.append({"datetime": datetime, "query": query, "sortby": sortby})
        return _FakeSearch(self._by_datetime.get(datetime, []))


def _make_scene_files(tmp_path, tag, rgb_value, cloud_block, empty_west=False):
    """Write a synthetic visual (3-band) + SCL raster pair, return their paths.

    ``empty_west`` blanks the western half of both, like a scene that only
    partly covers the AOI.
    """
    (height, width), _ = _source_grid()
    rgb = np.full((height, width, 3), rgb_value, dtype=np.uint8)
    # SCL: mostly vegetation (4), with a block of high-probability cloud (9).
    scl = np.full((height, width), 4, dtype=np.uint8)
    if cloud_block:
        scl[: height // 4, -width // 4 :] = 9
    if empty_west:
        rgb[:, : width // 2] = 0
        scl[:, : width // 2] = 0
    visual = _write_wgs84(tmp_path / f"{tag}_visual.tif", rgb)
    return visual, _write_wgs84(tmp_path / f"{tag}_scl.tif", scl)


T1, T2 = "2023-06-01/2023-06-30", "2024-06-01/2024-06-30"


def _pair_catalog(tmp_path, *, after_empty_west=False):
    before_v, before_scl = _make_scene_files(tmp_path, "before", 80, cloud_block=False)
    after_v, after_scl = _make_scene_files(
        tmp_path, "after", 160, cloud_block=True, empty_west=after_empty_west
    )
    return _FakeCatalog(
        {
            T1: [
                _FakeItem(
                    "S2_BEFORE", "2023-06-15T10:00:00Z", 4.0, before_v, before_scl
                )
            ],
            T2: [
                _FakeItem("S2_AFTER", "2024-06-12T10:00:00Z", 8.0, after_v, after_scl)
            ],
        }
    )


def test_search_items_filters_inclusively_and_sorts_by_cloud():
    catalog = _FakeCatalog({})
    assert search_items(catalog, BBOX, T1, max_cloud=20.0) == []
    (call,) = catalog.searches
    # "lte" keeps a scene at exactly the limit; sorting keeps the least cloudy
    # scenes when a long window has more matches than the result limit.
    assert call["query"] == {"eo:cloud_cover": {"lte": 20.0}}
    assert call["sortby"] == "+properties.eo:cloud_cover"


# ---------------------------------------------------------------------------
# open_catalog / GDAL environment
# ---------------------------------------------------------------------------


def test_open_catalog_signs_planetary_computer_assets_and_sets_a_timeout(
    monkeypatch,
):
    pystac_client = pytest.importorskip("pystac_client")
    planetary_computer = pytest.importorskip("planetary_computer")
    calls = []

    def fake_open(url, **kwargs):
        calls.append({"url": url, **kwargs})
        return "catalog"

    monkeypatch.setattr(pystac_client.Client, "open", fake_open)

    assert open_catalog() == "catalog"
    assert open_catalog("https://stac.example/v1", timeout=5) == "catalog"

    pc_call, other_call = calls
    assert pc_call["url"] == STAC_URL
    assert pc_call["modifier"] is planetary_computer.sign_inplace
    assert pc_call["timeout"] == 60.0
    # Only Planetary Computer asset URLs are run through its signer.
    assert other_call["modifier"] is None
    assert other_call["timeout"] == 5


def test_open_catalog_explains_how_to_install_the_missing_extra(monkeypatch):
    # A None entry in sys.modules makes `import pystac_client` raise ImportError.
    monkeypatch.setitem(sys.modules, "pystac_client", None)
    with pytest.raises(ImportError, match=r'pip install -e "\.\[ingest\]"'):
        open_catalog()


def test_gdal_cloud_env_fills_in_defaults_without_overriding_the_user(monkeypatch):
    for key in _GDAL_CLOUD_OPTIONS:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("GDAL_HTTP_MAX_RETRY", "0")

    with _gdal_cloud_env():
        options = rasterio.env.getenv()
        assert get_gdal_config("GDAL_DISABLE_READDIR_ON_OPEN") == "EMPTY_DIR"

    # The user's own setting is left for GDAL to read from the environment.
    assert "GDAL_HTTP_MAX_RETRY" not in options
    for key, value in _GDAL_CLOUD_OPTIONS.items():
        if key != "GDAL_HTTP_MAX_RETRY":
            assert options[key] == value
    # The defaults are scoped to the block and don't leak into later GDAL use.
    assert get_gdal_config("GDAL_DISABLE_READDIR_ON_OPEN") is None


# ---------------------------------------------------------------------------
# select_scene
# ---------------------------------------------------------------------------


def _coverage_items(tmp_path, *, with_footprints):
    """A clearer scene that barely reaches the AOI and a cloudier full one."""
    (height, width), transform = _source_grid()
    # The sliver scene only holds data east of SLIVER_WEST, like its footprint.
    first_col = round((SLIVER_WEST - transform.c) / _SRC_RES)
    sliver_scl = np.zeros((height, width), dtype=np.uint8)
    sliver_scl[:, first_col:] = 4
    sliver = _FakeItem(
        "S2_SLIVER",
        "2024-06-28T10:00:00Z",
        13.31,
        visual_href="unused",
        scl_href=_write_wgs84(tmp_path / "sliver_scl.tif", sliver_scl),
        geometry=EAST_SLIVER if with_footprints else None,
    )
    full = _FakeItem(
        "S2_FULL",
        "2024-06-28T10:00:00Z",
        13.94,
        visual_href="unused",
        scl_href=_write_wgs84_constant(tmp_path / "full_scl.tif", 4),
        geometry=FULL if with_footprints else None,
    )
    return [sliver, full]


def test_select_scene_checks_the_aoi_not_just_scene_cloud(tmp_path):
    catalog = _FakeCatalog({T2: _coverage_items(tmp_path, with_footprints=False)})
    grid = build_target_grid(BBOX, resolution=10.0)

    selection = select_scene(catalog, BBOX, T2, grid)
    assert selection.item.id == "S2_FULL"
    assert (selection.candidates_found, selection.candidates_evaluated) == (2, 2)

    # Checking a single candidate is the old least-cloudy behaviour.
    assert select_scene(catalog, BBOX, T2, grid, candidates=1).item.id == "S2_SLIVER"


def test_select_scene_stops_at_a_fully_clear_candidate(tmp_path):
    catalog = _FakeCatalog({T2: _coverage_items(tmp_path, with_footprints=True)})
    grid = build_target_grid(BBOX, resolution=10.0)

    selection = select_scene(catalog, BBOX, T2, grid)

    assert selection.item.id == "S2_FULL"
    assert selection.candidates_evaluated == 1


def test_select_scene_explains_an_empty_search():
    grid = build_target_grid(BBOX, resolution=10.0)
    with pytest.raises(ValueError, match="Widen the date window"):
        select_scene(_FakeCatalog({}), BBOX, T2, grid, max_cloud=5)
    with pytest.raises(ValueError, match="candidates"):
        select_scene(_FakeCatalog({}), BBOX, T2, grid, candidates=0)


# ---------------------------------------------------------------------------
# ingest_pair end-to-end against a fake STAC catalog
# ---------------------------------------------------------------------------


def test_ingest_pair_writes_coregistered_pair(tmp_path):
    catalog = _pair_catalog(tmp_path)

    out_dir = tmp_path / "out"
    manifest = ingest_pair(
        bbox=BBOX,
        datetime_t1=T1,
        datetime_t2=T2,
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
    assert manifest["scenes"]["after"]["aoi_nodata_fraction"] == 0.0

    # Provenance: what was asked for, of whom, by which version.
    assert manifest["software_version"] == src.__version__
    assert manifest["stac_url"] == "https://stac.example/v1"
    assert manifest["request"] == {
        "datetime_t1": T1,
        "datetime_t2": T2,
        "max_cloud": 20.0,
        "candidates": DEFAULT_CANDIDATES,
    }
    assert manifest["scenes"]["after"]["candidates_found"] == 1
    assert "boa_offset" not in manifest["scenes"]["after"]  # visual asset
    assert manifest["cloud_mask_values"] == {"clear": 0, "cloud": 1, "nodata": 255}

    # Outputs are genuinely co-registered: identical CRS, transform and size.
    with rasterio.open(before_tif) as b, rasterio.open(after_tif) as a:
        assert b.crs == a.crs == CRS.from_epsg(32633)
        assert b.transform == a.transform
        assert (b.width, b.height) == (a.width, a.height)
        assert b.count == 3
        assert b.nodata == a.nodata == 0

    # Cloud masks keep 0 for clear sky; their nodata value is 255.
    with rasterio.open(out_dir / "after_cloud.tif") as m:
        assert m.nodata == MASK_NODATA
        assert set(np.unique(m.read(1))) == {MASK_CLEAR, MASK_CLOUD}

    # Manifest round-trips as JSON.
    with open(out_dir / "manifest.json") as f:
        on_disk = json.load(f)
    assert on_disk["scenes"]["before"]["item_id"] == "S2_BEFORE"
    assert on_disk["software_version"] == src.__version__


def test_ingest_pair_marks_pixels_without_data(tmp_path):
    catalog = _pair_catalog(tmp_path, after_empty_west=True)
    out_dir = tmp_path / "out"

    manifest = ingest_pair(BBOX, T1, T2, out_dir, catalog=catalog)

    assert 0.4 < manifest["scenes"]["after"]["aoi_nodata_fraction"] < 0.6
    with rasterio.open(out_dir / "after_cloud.tif") as m:
        mask = m.read(1)
    with rasterio.open(out_dir / "after.tif") as a:
        image = a.read()
    nodata = mask == MASK_NODATA
    assert nodata.any() and (mask == MASK_CLOUD).any() and (mask == MASK_CLEAR).any()
    # Missing pixels are 0 in every band, and the scene edge is crisp: real
    # pixels keep their value instead of fading towards black.
    assert (image[:, nodata] == 0).all()
    assert set(np.unique(image[:, ~nodata])) == {160}


def test_ingest_pair_bands_mode_harmonizes_processing_baselines(tmp_path):
    (height, width), _ = _source_grid()
    # The same 1200 reflectance stored before (03.00) and after (05.10) the
    # January 2022 baseline change, which added 1000 DN.
    before = _band_item(
        tmp_path, "before", np.full((height, width), 1200, np.uint16), "03.00"
    )
    after = _band_item(
        tmp_path, "after", np.full((height, width), 2200, np.uint16), "05.10"
    )
    before.properties["datetime"] = "2021-06-12T10:00:00Z"
    catalog = _FakeCatalog({T1: [before], T2: [after]})
    out_dir = tmp_path / "out"

    manifest = ingest_pair(BBOX, T1, T2, out_dir, asset="bands", catalog=catalog)

    assert manifest["scenes"]["before"]["boa_offset"] == 0
    assert manifest["scenes"]["after"]["boa_offset"] == -1000
    with (
        rasterio.open(out_dir / "before.tif") as b,
        rasterio.open(out_dir / "after.tif") as a,
    ):
        # Unchanged ground looks unchanged.
        np.testing.assert_array_equal(b.read(), a.read())


def test_ingest_pair_rejects_windows_resolving_to_one_scene(tmp_path):
    visual, scl = _make_scene_files(tmp_path, "same", 80, cloud_block=False)
    item = _FakeItem("S2_SAME", "2024-06-12T10:00:00Z", 4.0, visual, scl)
    catalog = _FakeCatalog({"2024-06": [item], T2: [item]})
    out_dir = tmp_path / "out"

    with pytest.raises(ValueError, match="same scene"):
        ingest_pair(BBOX, "2024-06", T2, out_dir, catalog=catalog)
    assert not out_dir.exists()  # nothing is written for a rejected pair


def test_ingest_pair_validates_arguments_before_searching(tmp_path):
    catalog = _FakeCatalog({})
    with pytest.raises(ValueError, match="candidates"):
        ingest_pair(BBOX, T1, T2, tmp_path, catalog=catalog, candidates=0)
    with pytest.raises(ValueError, match="ceiling"):
        ingest_pair(
            BBOX,
            T1,
            T2,
            tmp_path,
            catalog=catalog,
            asset="bands",
            reflectance_ceiling=0.0,
        )
    assert catalog.searches == []


def test_target_grid_is_frozen():
    grid = build_target_grid(BBOX, resolution=10.0)
    assert isinstance(grid, TargetGrid)
    with pytest.raises(FrozenInstanceError):
        grid.width = 5  # type: ignore[misc]
