"""Sentinel-2 ingestion: turn a bounding box + two dates into a co-registered pair.

This is Phase 2 of the geospatial work. Phase 1 (:mod:`src.geo`) made the model's
*outputs* georeferenced; this module makes the *inputs* real by fetching actual
Sentinel-2 L2A imagery from a STAC catalog (Microsoft Planetary Computer by
default), cloud-screening it with the Scene Classification Layer (SCL), and
writing two GeoTIFFs that share an identical grid -- exactly what ``sat-cd-geo``
expects.

Pipeline
--------
1. **Search** the STAC catalog for scenes intersecting the AOI in each date
   window, filtered and sorted by scene-level cloud cover.
2. **Rank** the candidates by how much of the AOI each should see clearly: the
   share of the AOI inside the scene's data footprint times its clear-sky
   fraction. Reprocessed copies of one acquisition collapse to the newest.
3. **Build a common target grid** (a projected CRS + fixed ground resolution)
   from the AOI so both dates land on the same pixels.
4. **Choose** a scene per date by reading the top candidates' SCL onto the grid
   and keeping the one with the fewest cloudy or empty AOI pixels.
5. **Reproject** the chosen scene's imagery onto the grid, treating the L2A
   no-data value as missing rather than as black.
6. **Write** ``before.tif`` / ``after.tif`` (+ optional cloud masks + a manifest).

Design notes
------------
- The module is **torch-free** -- it only needs ``rasterio``/``shapely``/``numpy``
  plus the optional ``pystac-client`` + ``planetary-computer`` clients (installed
  via the ``[ingest]`` extra). Those two are imported lazily so the pure helpers
  below can be unit-tested without them or any network access.
- By default it pulls the pre-rendered 8-bit true-colour ``visual`` asset, whose
  ``[0, 255]`` RGB range and band order match what the change-detection model was
  trained on. A raw multispectral path (``asset="bands"``) is available for
  experimentation: it removes the reflectance offset that processing baseline
  04.00 introduced, so scenes from before and after January 2022 stay
  comparable, then rescales reflectance to ``[0, 255]``.
- ``0`` stays reserved for no-data in the written imagery, as in the L2A
  product, so GIS tools and ``sat-cd-geo`` can tell missing pixels from dark ones.
"""

from __future__ import annotations

import datetime as dt
import json
import math
import os
import re
import warnings
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.transform import from_origin
from rasterio.warp import reproject, transform_bounds
from shapely import make_valid
from shapely.errors import ShapelyError
from shapely.geometry import box, shape

from src import __version__
from src.geo import write_geotiff

# Microsoft Planetary Computer STAC endpoint and the Sentinel-2 L2A collection.
STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
S2_L2A_COLLECTION = "sentinel-2-l2a"

# True-colour band assets, in the red, green, blue order the RGB model expects.
RGB_BANDS = ("B04", "B03", "B02")
VISUAL_ASSET = "visual"
SCL_ASSET = "SCL"

# Scene Classification Layer (SCL) class values.
# Reference: Sentinel-2 L2A algorithm theoretical basis.
SCL_CLOUD_CLASSES = frozenset({3, 8, 9, 10})  # shadow, cloud med/high, thin cirrus
SCL_NODATA_CLASSES = frozenset({0, 1})  # no-data, saturated/defective

# L2A reserves 0 for no-data in every reflectance band and in the true-colour
# image. Planetary Computer's COGs do not declare it, so it is passed to the
# warper explicitly -- otherwise resampling blends empty pixels into real ones.
L2A_NODATA = 0

# Processing baseline 04.00 (operational from 25 January 2022) shifted L2A
# reflectance digital numbers up by 1000 so dark surfaces can go negative.
# ESA's BOA_ADD_OFFSET undoes it.
BOA_ADD_OFFSET = -1000
BOA_OFFSET_SINCE_BASELINE = (4, 0)

# Pixel values of the per-date cloud-mask GeoTIFFs.
MASK_CLEAR = 0
MASK_CLOUD = 1
MASK_NODATA = 255

# Candidate scenes per date whose SCL is checked over the AOI before choosing.
DEFAULT_CANDIDATES = 3

# GDAL settings for reading cloud-hosted COGs: skip sidecar-file probing, retry
# transient HTTP failures, and merge adjacent byte-range requests. Values the
# user has already set in the environment win.
_GDAL_CLOUD_OPTIONS = {
    "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
    "GDAL_HTTP_MAX_RETRY": "5",
    "GDAL_HTTP_RETRY_DELAY": "1",
    "GDAL_HTTP_MERGE_CONSECUTIVE_RANGES": "YES",
}

_BASELINE_PATTERN = re.compile(r"(\d+)\.(\d+)")

WGS84 = "EPSG:4326"


@dataclass(frozen=True)
class TargetGrid:
    """A fixed raster grid that both dates are resampled onto.

    Attributes:
        crs: Projected CRS of the grid (metres).
        transform: Affine pixel-to-world transform.
        width: Grid width in pixels.
        height: Grid height in pixels.
    """

    crs: CRS
    transform: rasterio.Affine
    width: int
    height: int

    @property
    def shape(self) -> tuple[int, int]:
        """``(height, width)`` in pixels."""
        return (self.height, self.width)


# ---------------------------------------------------------------------------
# Pure helpers (no network, no optional deps) -- unit-tested directly
# ---------------------------------------------------------------------------


def utm_epsg_for_lonlat(lon: float, lat: float) -> int:
    """Return the EPSG code of the UTM zone containing ``(lon, lat)``.

    Northern-hemisphere zones are ``326xx``; southern are ``327xx``.
    """
    zone = int(math.floor((lon + 180.0) / 6.0)) % 60 + 1
    return (32600 if lat >= 0 else 32700) + zone


def validate_bbox(bbox: Sequence[float]) -> tuple[float, float, float, float]:
    """Check a WGS84 bounding box and return it as four floats.

    Args:
        bbox: ``(min_lon, min_lat, max_lon, max_lat)`` -- longitude first.

    Raises:
        ValueError: If the box is not four finite numbers, falls outside the
            lon/lat range, or is empty or inverted.
    """
    try:
        values = tuple(float(v) for v in bbox)
    except (TypeError, ValueError) as e:
        raise ValueError(f"Invalid bbox: expected 4 numbers, got {bbox!r}") from e
    if len(values) != 4 or not all(math.isfinite(v) for v in values):
        raise ValueError(f"Invalid bbox: expected 4 finite numbers, got {bbox!r}")

    min_lon, min_lat, max_lon, max_lat = values
    lon_ok = -180.0 <= min_lon <= 180.0 and -180.0 <= max_lon <= 180.0
    lat_ok = -90.0 <= min_lat <= 90.0 and -90.0 <= max_lat <= 90.0
    if not (lon_ok and lat_ok):
        raise ValueError(
            f"Invalid bbox {values}: expected WGS84 (min_lon, min_lat, max_lon, "
            "max_lat) with longitude in [-180, 180] and latitude in [-90, 90]. "
            "Check that longitude comes first."
        )
    if min_lon >= max_lon or min_lat >= max_lat:
        raise ValueError(f"Invalid bbox (need min < max): {values}")
    return min_lon, min_lat, max_lon, max_lat


def build_target_grid(
    bbox: Sequence[float],
    resolution: float,
    dst_crs: CRS | str | None = None,
) -> TargetGrid:
    """Derive a common projected grid from a lon/lat bounding box.

    Args:
        bbox: ``(min_lon, min_lat, max_lon, max_lat)`` in WGS84.
        resolution: Ground sample distance in metres (e.g. ``10``).
        dst_crs: Target CRS; defaults to the UTM zone of the AOI centroid.

    Returns:
        A :class:`TargetGrid` covering the AOI at the requested resolution.

    Raises:
        ValueError: If the bbox is invalid (see :func:`validate_bbox`) or the
            resolution is not a positive number.
    """
    min_lon, min_lat, max_lon, max_lat = validate_bbox(bbox)
    if not (math.isfinite(resolution) and resolution > 0):
        raise ValueError(
            f"resolution must be a positive number of metres, got {resolution}"
        )

    if dst_crs is None:
        center_lon = (min_lon + max_lon) / 2.0
        center_lat = (min_lat + max_lat) / 2.0
        dst_crs = CRS.from_epsg(utm_epsg_for_lonlat(center_lon, center_lat))
    else:
        dst_crs = CRS.from_user_input(dst_crs)

    left, bottom, right, top = transform_bounds(
        WGS84, dst_crs, min_lon, min_lat, max_lon, max_lat
    )
    width = max(1, int(math.ceil((right - left) / resolution)))
    height = max(1, int(math.ceil((top - bottom) / resolution)))
    transform = from_origin(left, top, resolution, resolution)
    return TargetGrid(crs=dst_crs, transform=transform, width=width, height=height)


def scl_to_masks(scl: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Split an SCL band into ``(cloud_mask, nodata_mask)`` boolean arrays.

    ``cloud_mask`` is ``True`` for cloud, cloud-shadow and thin-cirrus pixels;
    ``nodata_mask`` is ``True`` for no-data / defective pixels.
    """
    scl_int = np.asarray(scl).astype(np.int16)
    cloud = np.isin(scl_int, list(SCL_CLOUD_CLASSES))
    nodata = np.isin(scl_int, list(SCL_NODATA_CLASSES))
    return cloud, nodata


def cloud_fraction(scl: np.ndarray) -> float:
    """Fraction of valid (non-nodata) AOI pixels flagged as cloud, in ``[0, 1]``."""
    cloud, nodata = scl_to_masks(scl)
    valid = int((~nodata).sum())
    if valid == 0:
        return 1.0
    return float(cloud.sum()) / float(valid)


def invalid_fraction(scl: np.ndarray) -> float:
    """Fraction of AOI pixels that are cloudy *or* hold no data, in ``[0, 1]``.

    This is what scene selection minimizes. Unlike :func:`cloud_fraction` it
    counts pixels the scene does not cover at all, so a clear scene that only
    reaches one corner of the AOI scores badly.
    """
    cloud, nodata = scl_to_masks(scl)
    if cloud.size == 0:
        return 1.0
    return float(np.logical_or(cloud, nodata).mean())


def select_least_cloudy(items: Sequence[Any]) -> Any:
    """Return the STAC item with the lowest ``eo:cloud_cover`` property.

    Works on any object exposing a ``.properties`` mapping, so it can be tested
    with lightweight stand-ins instead of real STAC items. Scene selection in
    :func:`ingest_pair` uses the AOI-aware :func:`rank_candidates` and
    :func:`select_scene` instead.
    """
    if not items:
        raise ValueError("No scenes found for the given AOI and date range.")

    def _cover(item: Any) -> float:
        value = getattr(item, "properties", {}).get("eo:cloud_cover")
        return float(value) if value is not None else math.inf

    return min(items, key=_cover)


def _properties(item: Any) -> dict[str, Any]:
    return getattr(item, "properties", None) or {}


def _cloud_cover(item: Any) -> float | None:
    value = _properties(item).get("eo:cloud_cover")
    return float(value) if value is not None else None


def aoi_coverage(item: Any, bbox: Sequence[float]) -> float | None:
    """Share of the AOI inside ``item``'s data footprint, in ``[0, 1]``.

    Sentinel-2 tiles overlap and are often only partly filled where the edge of
    the orbit swath crosses them, so a scene that merely *intersects* the AOI
    can still leave most of it empty. A STAC item's ``geometry`` traces the
    valid data, so intersecting it with the AOI measures what the scene can
    see. The ratio is taken in lon/lat degrees, which is plenty for ranking.

    Returns:
        The covered fraction, or ``None`` if the item has no usable geometry.
    """
    geometry = getattr(item, "geometry", None)
    if not geometry:
        return None
    try:
        footprint = make_valid(shape(geometry))
        aoi = box(*(float(v) for v in bbox))
        return float(min(1.0, footprint.intersection(aoi).area / aoi.area))
    except (
        AttributeError,
        KeyError,
        ShapelyError,
        TypeError,
        ValueError,
        ZeroDivisionError,
    ):
        return None


def _acquisition_key(item: Any) -> tuple[str, str] | None:
    """``(datetime, tile)`` identifying one overpass of one tile, if known."""
    props = _properties(item)
    when = props.get("datetime")
    tile = props.get("s2:mgrs_tile") or props.get("grid:code")
    if not when or not tile:
        return None
    return str(when), str(tile)


def rank_candidates(items: Sequence[Any], bbox: Sequence[float]) -> list[Any]:
    """Order scenes by the share of the AOI each is expected to see clearly.

    The score is ``AOI coverage x (1 - scene cloud cover)``, an estimate of the
    AOI's clear fraction made before reading any pixels; ties go to the less
    cloudy scene. Catalogs can hold several processings of one acquisition (same
    overpass, same tile); only the newest is kept, i.e. the one whose id sorts
    last, since Sentinel-2 ids end with the processing time.

    Items without a footprint are assumed to cover the whole AOI, and items
    without ``eo:cloud_cover`` are ranked as fully cloudy.
    """
    newest: dict[tuple[str, str], Any] = {}
    unique: list[Any] = []
    for item in items:
        key = _acquisition_key(item)
        if key is None:
            unique.append(item)
        elif key not in newest or str(item.id) > str(newest[key].id):
            newest[key] = item
    unique.extend(newest.values())

    def _rank(item: Any) -> tuple[float, float, str]:
        coverage = aoi_coverage(item, bbox)
        cloud = _cloud_cover(item)
        clear = 0.0 if cloud is None else max(0.0, 1.0 - cloud / 100.0)
        expected = (1.0 if coverage is None else coverage) * clear
        return (
            -round(expected, 6),
            math.inf if cloud is None else cloud,
            str(item.id),
        )

    return sorted(unique, key=_rank)


def boa_offset_for_baseline(baseline: str | None) -> int:
    """Digital-number offset to add to L2A reflectance for a processing baseline.

    Baseline 04.00 and later store reflectance shifted up by 1000 DN, so their
    offset is :data:`BOA_ADD_OFFSET` (``-1000``). Earlier or unrecognised
    baselines get ``0``.
    """
    match = _BASELINE_PATTERN.search(str(baseline)) if baseline else None
    if match is None:
        return 0
    version = (int(match.group(1)), int(match.group(2)))
    return BOA_ADD_OFFSET if version >= BOA_OFFSET_SINCE_BASELINE else 0


def boa_offset_for_item(item: Any) -> int:
    """Digital-number offset to add to ``item``'s L2A reflectance bands.

    Based on the item's ``s2:processing_baseline`` (see
    :func:`boa_offset_for_baseline`). Catalogs that serve already-harmonized
    reflectance -- Element 84's Earth Search flags it with
    ``earthsearch:boa_offset_applied`` -- get ``0`` so the offset is never
    removed twice.
    """
    props = _properties(item)
    if props.get("earthsearch:boa_offset_applied"):
        return 0
    return boa_offset_for_baseline(props.get("s2:processing_baseline"))


def check_scene_pair(before: Any, after: Any) -> None:
    """Reject pairs that cannot show change and warn about reversed ones.

    Raises:
        ValueError: If both dates resolved to the same scene, or to two tiles of
            the same acquisition. Overlapping date windows usually cause this.
    """
    if str(before.id) == str(after.id):
        raise ValueError(
            f"Both dates resolved to the same scene ({before.id}). "
            "Use non-overlapping date windows."
        )
    t1 = _parse_datetime(_properties(before).get("datetime"))
    t2 = _parse_datetime(_properties(after).get("datetime"))
    if t1 is None or t2 is None:
        return
    if t1 == t2:
        raise ValueError(
            f"Both dates resolved to the same acquisition ({t1.isoformat()}: "
            f"{before.id} and {after.id}), so there is no change to detect. "
            "Use non-overlapping date windows."
        )
    if t2 < t1:
        warnings.warn(
            f"The 'after' scene ({t2.isoformat()}) is older than the 'before' "
            f"scene ({t1.isoformat()}); change will be detected backwards in time.",
            UserWarning,
            stacklevel=2,
        )


def _parse_datetime(value: Any) -> dt.datetime | None:
    if not value:
        return None
    try:
        parsed = dt.datetime.fromisoformat(str(value))
    except ValueError:
        return None
    return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=dt.UTC)


@contextmanager
def _opened(src: Any) -> Iterator[Any]:
    """Yield an open rasterio dataset from a path/href or an already-open one."""
    if hasattr(src, "read") and hasattr(src, "transform"):
        yield src
    else:
        with rasterio.open(str(src)) as dataset:
            yield dataset


def reproject_to_grid(
    src: Any,
    grid: TargetGrid,
    *,
    band_indexes: Sequence[int] | None = None,
    resampling: Resampling = Resampling.bilinear,
    dtype: str | None = None,
    src_nodata: float | None = None,
) -> np.ndarray:
    """Reproject a raster source onto ``grid``, returning an ``(H, W, C)`` array.

    Args:
        src: A file path/URL or an already-open rasterio dataset.
        grid: The :class:`TargetGrid` to resample onto.
        band_indexes: 1-based band indexes to read (default: all bands).
        resampling: Resampling method (use ``nearest`` for categorical bands).
        dtype: Output dtype; defaults to the source's dtype.
        src_nodata: Source value to treat as missing, overriding the dataset's
            own nodata (if any). Missing pixels are left out of interpolation and
            come back as this value, as do grid cells the source does not cover.

    Returns:
        Array of shape ``(grid.height, grid.width, n_bands)``.
    """
    with _opened(src) as dataset:
        indexes = (
            list(band_indexes)
            if band_indexes is not None
            else list(range(1, dataset.count + 1))
        )
        out_dtype = dtype or dataset.dtypes[indexes[0] - 1]
        dst = np.zeros((len(indexes), grid.height, grid.width), dtype=out_dtype)
        for i, bidx in enumerate(indexes):
            reproject(
                source=rasterio.band(dataset, bidx),
                destination=dst[i],
                dst_transform=grid.transform,
                dst_crs=grid.crs,
                resampling=resampling,
                src_nodata=src_nodata,
                dst_nodata=src_nodata,
            )
    return np.transpose(dst, (1, 2, 0))


def scale_reflectance(array: np.ndarray, ceiling: float = 3000.0) -> np.ndarray:
    """Scale raw L2A reflectance (``uint16``) to an 8-bit ``[0, 255]`` image.

    Values are clipped at ``ceiling`` (a typical bright-surface reflectance) and
    linearly mapped to ``[0, 255]`` so multispectral bands match the value range
    the model was trained on.

    Raises:
        ValueError: If ``ceiling`` is not a positive number.
    """
    if not (math.isfinite(ceiling) and ceiling > 0):
        raise ValueError(f"Reflectance ceiling must be positive, got {ceiling}")
    scaled = np.clip(np.asarray(array, dtype=np.float32) / ceiling, 0.0, 1.0)
    return (scaled * 255.0).round().astype(np.uint8)


# ---------------------------------------------------------------------------
# Network layer (lazy optional deps) -- thin wrappers around STAC
# ---------------------------------------------------------------------------


@contextmanager
def _gdal_cloud_env() -> Iterator[None]:
    """Apply :data:`_GDAL_CLOUD_OPTIONS` that the user has not set themselves."""
    options = {k: v for k, v in _GDAL_CLOUD_OPTIONS.items() if k not in os.environ}
    with rasterio.Env(**options):
        yield


def open_catalog(url: str = STAC_URL, *, timeout: float = 60.0) -> Any:
    """Open a STAC catalog, auto-signing Planetary Computer assets when available.

    Args:
        url: STAC API root.
        timeout: Seconds to wait on each request to the API before failing.

    Raises:
        ImportError: If the ``[ingest]`` optional dependencies are not installed.
    """
    try:
        import pystac_client
    except ImportError as e:
        raise ImportError(
            "Sentinel-2 ingestion needs the optional dependencies. Install with:\n"
            '    pip install -e ".[ingest]"'
        ) from e

    modifier = None
    try:
        import planetary_computer

        if url == STAC_URL:
            modifier = planetary_computer.sign_inplace
    except ImportError:  # pragma: no cover - PC signing is optional
        modifier = None

    return pystac_client.Client.open(url, modifier=modifier, timeout=timeout)


def search_items(
    catalog: Any,
    bbox: Sequence[float],
    datetime: str,
    *,
    collection: str = S2_L2A_COLLECTION,
    max_cloud: float = 100.0,
    limit: int = 50,
) -> list[Any]:
    """Search a STAC catalog for scenes over ``bbox`` within ``datetime``.

    Results come back least-cloudy first, so when a long window holds more than
    ``limit`` scenes the cut drops the cloudiest ones.

    Args:
        catalog: An open ``pystac_client.Client``.
        bbox: ``(min_lon, min_lat, max_lon, max_lat)`` in WGS84.
        datetime: A STAC datetime string, e.g. ``"2024-06-01/2024-06-30"``.
        collection: STAC collection id.
        max_cloud: Maximum scene-level ``eo:cloud_cover`` percent (inclusive).
        limit: Maximum number of items to return.

    Returns:
        A list of STAC items (possibly empty).
    """
    search = catalog.search(
        collections=[collection],
        bbox=list(bbox),
        datetime=datetime,
        query={"eo:cloud_cover": {"lte": max_cloud}},
        sortby="+properties.eo:cloud_cover",
        max_items=limit,
    )
    return list(search.items())


def _asset_href(item: Any, key: str) -> str:
    """Return the (signed) href for ``item``'s asset ``key``."""
    assets = item.assets
    if key not in assets:
        available = ", ".join(sorted(assets.keys()))
        raise KeyError(f"Asset '{key}' not on item {item.id}. Available: {available}")
    return str(assets[key].href)


def read_item_imagery(
    item: Any,
    grid: TargetGrid,
    *,
    asset: str = VISUAL_ASSET,
    bands: Sequence[str] = RGB_BANDS,
    reflectance_ceiling: float = 3000.0,
) -> np.ndarray:
    """Read an item's imagery onto ``grid`` as an ``(H, W, 3)`` uint8 array.

    With ``asset="visual"`` the pre-rendered 8-bit true-colour COG is used.
    Otherwise each band in ``bands`` is read separately, stacked in order,
    corrected for the processing-baseline offset (:func:`boa_offset_for_item`)
    and rescaled from reflectance to ``[0, 255]``.

    Either way the L2A no-data value is kept out of resampling, so scene edges
    stay crisp instead of fading towards black, and ``0`` marks missing pixels
    in the result.
    """
    if asset == VISUAL_ASSET:
        rgb = reproject_to_grid(
            _asset_href(item, VISUAL_ASSET),
            grid,
            resampling=Resampling.bilinear,
            src_nodata=L2A_NODATA,
        )
        return rgb[:, :, :3].astype(np.uint8)

    channels = [
        reproject_to_grid(
            _asset_href(item, band),
            grid,
            resampling=Resampling.bilinear,
            src_nodata=L2A_NODATA,
        )[:, :, 0]
        for band in bands
    ]
    raw = np.stack(channels, axis=-1)
    valid = np.all(raw != L2A_NODATA, axis=-1, keepdims=True)
    reflectance = np.clip(raw.astype(np.float32) + boa_offset_for_item(item), 0.0, None)
    scaled = scale_reflectance(reflectance, ceiling=reflectance_ceiling)
    # Keep 0 for no-data: the darkest valid pixels round up to 1 instead.
    return np.where(valid, np.maximum(scaled, 1), L2A_NODATA).astype(np.uint8)


def read_item_scl(item: Any, grid: TargetGrid) -> np.ndarray:
    """Read an item's SCL band onto ``grid`` as a 2-D uint8 array."""
    scl = reproject_to_grid(
        _asset_href(item, SCL_ASSET), grid, resampling=Resampling.nearest
    )
    return scl[:, :, 0].astype(np.uint8)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


@dataclass
class SceneSelection:
    """The scene chosen for one date window, with its SCL already on the grid."""

    item: Any
    scl: np.ndarray
    candidates_found: int
    candidates_evaluated: int


@dataclass
class SceneResult:
    """A single resolved date: the chosen scene plus its rasters and stats.

    ``imagery`` is ``0`` in every band wherever the scene holds no data.
    ``cloud_mask`` is :data:`MASK_CLEAR`, :data:`MASK_CLOUD` or
    :data:`MASK_NODATA` per pixel. ``boa_offset`` is the reflectance offset
    applied in raw-bands mode, or ``None`` for the visual asset.
    """

    item_id: str
    datetime: str
    scene_cloud_cover: float | None
    aoi_cloud_fraction: float
    imagery: np.ndarray
    cloud_mask: np.ndarray
    aoi_nodata_fraction: float = 0.0
    platform: str | None = None
    mgrs_tile: str | None = None
    processing_baseline: str | None = None
    boa_offset: int | None = None
    candidates_found: int | None = None
    candidates_evaluated: int | None = None


def select_scene(
    catalog: Any,
    bbox: Sequence[float],
    datetime: str,
    grid: TargetGrid,
    *,
    max_cloud: float = 20.0,
    candidates: int = DEFAULT_CANDIDATES,
) -> SceneSelection:
    """Pick the scene that sees the most of the AOI in one date window.

    Candidates are ordered with :func:`rank_candidates`, then the SCL of the top
    ``candidates`` is read onto ``grid`` and the one with the lowest
    :func:`invalid_fraction` wins, with earlier rank breaking ties. Scene-level
    cloud cover describes a whole 110 km tile, so checking the AOI itself avoids
    picking a mostly clear tile whose clouds sit right over the AOI.

    Raises:
        ValueError: If ``candidates`` is below 1 or no scene matches.
    """
    if candidates < 1:
        raise ValueError(f"candidates must be at least 1, got {candidates}")
    items = search_items(catalog, bbox, datetime, max_cloud=max_cloud)
    if not items:
        raise ValueError(
            f"No Sentinel-2 scenes found for {datetime} with scene cloud cover "
            f"<= {max_cloud:g}%. Widen the date window or raise the cloud limit."
        )

    scored: list[tuple[float, Any, np.ndarray]] = []
    for item in rank_candidates(items, bbox)[:candidates]:
        scl = read_item_scl(item, grid)
        scored.append((invalid_fraction(scl), item, scl))
        if scored[-1][0] == 0.0:
            break
    _, best, best_scl = min(scored, key=lambda entry: entry[0])
    return SceneSelection(
        item=best,
        scl=best_scl,
        candidates_found=len(items),
        candidates_evaluated=len(scored),
    )


def read_scene(
    selection: SceneSelection,
    grid: TargetGrid,
    *,
    asset: str = VISUAL_ASSET,
    bands: Sequence[str] = RGB_BANDS,
    reflectance_ceiling: float = 3000.0,
) -> SceneResult:
    """Read a selected scene's imagery onto ``grid`` and derive its masks.

    A pixel counts as no-data if any band is empty or the SCL flags it as
    no-data or defective; it is then zeroed in every band and marked
    :data:`MASK_NODATA` in the cloud mask.
    """
    item = selection.item
    props = _properties(item)
    imagery = read_item_imagery(
        item,
        grid,
        asset=asset,
        bands=bands,
        reflectance_ceiling=reflectance_ceiling,
    )
    cloud, scl_nodata = scl_to_masks(selection.scl)
    nodata = scl_nodata | np.any(imagery == L2A_NODATA, axis=-1)
    imagery[nodata] = L2A_NODATA

    cloud_mask = np.full(grid.shape, MASK_CLEAR, dtype=np.uint8)
    cloud_mask[cloud] = MASK_CLOUD
    cloud_mask[nodata] = MASK_NODATA

    return SceneResult(
        item_id=str(item.id),
        datetime=str(props.get("datetime", "")),
        scene_cloud_cover=props.get("eo:cloud_cover"),
        aoi_cloud_fraction=round(cloud_fraction(selection.scl), 4),
        imagery=imagery,
        cloud_mask=cloud_mask,
        aoi_nodata_fraction=round(float(nodata.mean()), 4),
        platform=props.get("platform"),
        mgrs_tile=props.get("s2:mgrs_tile"),
        processing_baseline=props.get("s2:processing_baseline"),
        boa_offset=None if asset == VISUAL_ASSET else boa_offset_for_item(item),
        candidates_found=selection.candidates_found,
        candidates_evaluated=selection.candidates_evaluated,
    )


def _catalog_url(catalog: Any) -> str | None:
    """Best-effort root URL of an open STAC client, for the manifest."""
    get_self_href = getattr(catalog, "get_self_href", None)
    href = get_self_href() if callable(get_self_href) else None
    return str(href) if href else None


def ingest_pair(
    bbox: Sequence[float],
    datetime_t1: str,
    datetime_t2: str,
    output_dir: str | Path,
    *,
    resolution: float = 10.0,
    asset: str = VISUAL_ASSET,
    bands: Sequence[str] = RGB_BANDS,
    max_cloud: float = 20.0,
    dst_crs: CRS | str | None = None,
    reflectance_ceiling: float = 3000.0,
    write_cloud_masks: bool = True,
    catalog: Any | None = None,
    candidates: int = DEFAULT_CANDIDATES,
) -> dict[str, Any]:
    """Fetch a co-registered Sentinel-2 pair for an AOI and write GeoTIFFs.

    Args:
        bbox: ``(min_lon, min_lat, max_lon, max_lat)`` in WGS84.
        datetime_t1: STAC datetime/range for the "before" scene.
        datetime_t2: STAC datetime/range for the "after" scene.
        output_dir: Directory to write outputs into (created if absent).
        resolution: Ground sample distance in metres.
        asset: ``"visual"`` (8-bit true colour) or ``"bands"`` for raw bands.
        bands: Band assets to stack when ``asset != "visual"``.
        max_cloud: Maximum scene-level cloud cover percent to consider.
        dst_crs: Output CRS (defaults to the AOI's UTM zone).
        reflectance_ceiling: Reflectance clip ceiling for the raw-bands path.
        write_cloud_masks: Also write per-date cloud-mask GeoTIFFs
            (:data:`MASK_CLEAR` / :data:`MASK_CLOUD` / :data:`MASK_NODATA`).
        catalog: An open STAC client (defaults to Planetary Computer).
        candidates: Scenes per date whose SCL is checked over the AOI before
            choosing (see :func:`select_scene`).

    Returns:
        A manifest dict (also written to ``manifest.json``) describing the
        request, the grid, both chosen scenes, and the output files.

    Raises:
        ValueError: On invalid arguments, when a date window has no matching
            scene, or when both windows resolve to the same acquisition.
    """
    band_keys = tuple(b.upper() for b in bands)
    grid = build_target_grid(bbox, resolution, dst_crs=dst_crs)
    if candidates < 1:
        raise ValueError(f"candidates must be at least 1, got {candidates}")
    if asset != VISUAL_ASSET and not (
        math.isfinite(reflectance_ceiling) and reflectance_ceiling > 0
    ):
        raise ValueError(
            f"Reflectance ceiling must be positive, got {reflectance_ceiling}"
        )

    stac_url = STAC_URL if catalog is None else _catalog_url(catalog)
    if catalog is None:
        catalog = open_catalog()

    windows = {"before": datetime_t1, "after": datetime_t2}
    with _gdal_cloud_env():
        selections = {
            label: select_scene(
                catalog,
                bbox,
                window,
                grid,
                max_cloud=max_cloud,
                candidates=candidates,
            )
            for label, window in windows.items()
        }
        check_scene_pair(selections["before"].item, selections["after"].item)
        scenes = {
            label: read_scene(
                selection,
                grid,
                asset=asset,
                bands=band_keys,
                reflectance_ceiling=reflectance_ceiling,
            )
            for label, selection in selections.items()
        }

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, str] = {}
    scene_manifest: dict[str, Any] = {}
    for label, scene in scenes.items():
        image_path = out_dir / f"{label}.tif"
        write_geotiff(
            image_path,
            scene.imagery,
            transform=grid.transform,
            crs=grid.crs,
            nodata=L2A_NODATA,
            dtype="uint8",
        )
        outputs[label] = str(image_path)

        if write_cloud_masks:
            mask_path = out_dir / f"{label}_cloud.tif"
            write_geotiff(
                mask_path,
                scene.cloud_mask,
                transform=grid.transform,
                crs=grid.crs,
                nodata=MASK_NODATA,
                dtype="uint8",
            )
            outputs[f"{label}_cloud"] = str(mask_path)

        scene_manifest[label] = {
            "item_id": scene.item_id,
            "datetime": scene.datetime,
            "platform": scene.platform,
            "mgrs_tile": scene.mgrs_tile,
            "processing_baseline": scene.processing_baseline,
            "scene_cloud_cover": scene.scene_cloud_cover,
            "aoi_cloud_fraction": scene.aoi_cloud_fraction,
            "aoi_nodata_fraction": scene.aoi_nodata_fraction,
            "candidates_found": scene.candidates_found,
            "candidates_evaluated": scene.candidates_evaluated,
        }
        if scene.boa_offset is not None:
            scene_manifest[label]["boa_offset"] = scene.boa_offset

    manifest: dict[str, Any] = {
        "created": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "software_version": __version__,
        "stac_url": stac_url,
        "collection": S2_L2A_COLLECTION,
        "request": {
            "datetime_t1": datetime_t1,
            "datetime_t2": datetime_t2,
            "max_cloud": max_cloud,
            "candidates": candidates,
        },
        "bbox": [float(v) for v in bbox],
        "resolution_m": resolution,
        "asset": asset,
        "bands": list(band_keys) if asset != VISUAL_ASSET else ["R", "G", "B"],
        "nodata": L2A_NODATA,
        "grid": {
            "crs": grid.crs.to_string(),
            "width": grid.width,
            "height": grid.height,
            "transform": list(grid.transform)[:6],
        },
        "scenes": scene_manifest,
        "outputs": outputs,
    }
    if write_cloud_masks:
        manifest["cloud_mask_values"] = {
            "clear": MASK_CLEAR,
            "cloud": MASK_CLOUD,
            "nodata": MASK_NODATA,
        }
    manifest_path = out_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    manifest["outputs"]["manifest"] = str(manifest_path)
    return manifest
