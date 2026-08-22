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
   window, filtered by scene-level cloud cover.
2. **Select** the least-cloudy scene per window.
3. **Build a common target grid** (a projected CRS + fixed ground resolution)
   from the AOI so both dates land on the same pixels.
4. **Reproject** each scene's imagery (and its SCL band) onto that grid.
5. **Write** ``before.tif`` / ``after.tif`` (+ optional cloud masks + a manifest).

Design notes
------------
- The module is **torch-free** -- it only needs ``rasterio``/``numpy`` plus the
  optional ``pystac-client`` + ``planetary-computer`` clients (installed via the
  ``[ingest]`` extra). Those two are imported lazily so the pure helpers below
  can be unit-tested without them or any network access.
- By default it pulls the pre-rendered 8-bit true-colour ``visual`` asset, whose
  ``[0, 255]`` RGB range and band order match what the change-detection model was
  trained on. A raw multispectral path (``--bands B04 B03 B02 ...``) is available
  for experimentation and rescales reflectance to ``[0, 255]``.
"""

from __future__ import annotations

import json
import math
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
    """
    min_lon, min_lat, max_lon, max_lat = (float(v) for v in bbox)
    if min_lon >= max_lon or min_lat >= max_lat:
        raise ValueError(f"Invalid bbox (need min < max): {tuple(bbox)}")

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


def select_least_cloudy(items: Sequence[Any]) -> Any:
    """Return the STAC item with the lowest ``eo:cloud_cover`` property.

    Works on any object exposing a ``.properties`` mapping, so it can be tested
    with lightweight stand-ins instead of real STAC items.
    """
    if not items:
        raise ValueError("No scenes found for the given AOI and date range.")

    def _cover(item: Any) -> float:
        value = getattr(item, "properties", {}).get("eo:cloud_cover")
        return float(value) if value is not None else math.inf

    return min(items, key=_cover)


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
) -> np.ndarray:
    """Reproject a raster source onto ``grid``, returning an ``(H, W, C)`` array.

    Args:
        src: A file path/URL or an already-open rasterio dataset.
        grid: The :class:`TargetGrid` to resample onto.
        band_indexes: 1-based band indexes to read (default: all bands).
        resampling: Resampling method (use ``nearest`` for categorical bands).
        dtype: Output dtype; defaults to the source's dtype.

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
            )
    return np.transpose(dst, (1, 2, 0))


def scale_reflectance(array: np.ndarray, ceiling: float = 3000.0) -> np.ndarray:
    """Scale raw L2A reflectance (``uint16``) to an 8-bit ``[0, 255]`` image.

    Values are clipped at ``ceiling`` (a typical bright-surface reflectance) and
    linearly mapped to ``[0, 255]`` so multispectral bands match the value range
    the model was trained on.
    """
    scaled = np.clip(np.asarray(array, dtype=np.float32) / ceiling, 0.0, 1.0)
    return (scaled * 255.0).round().astype(np.uint8)


# ---------------------------------------------------------------------------
# Network layer (lazy optional deps) -- thin wrappers around STAC
# ---------------------------------------------------------------------------


def open_catalog(url: str = STAC_URL) -> Any:
    """Open a STAC catalog, auto-signing Planetary Computer assets when available.

    Raises:
        ImportError: If the ``[ingest]`` optional dependencies are not installed.
    """
    try:
        import pystac_client
    except ImportError as e:  # pragma: no cover - exercised only without extras
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

    return pystac_client.Client.open(url, modifier=modifier)


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

    Args:
        catalog: An open ``pystac_client.Client``.
        bbox: ``(min_lon, min_lat, max_lon, max_lat)`` in WGS84.
        datetime: A STAC datetime string, e.g. ``"2024-06-01/2024-06-30"``.
        collection: STAC collection id.
        max_cloud: Maximum scene-level ``eo:cloud_cover`` percent.
        limit: Maximum number of items to return.

    Returns:
        A list of STAC items (possibly empty).
    """
    search = catalog.search(
        collections=[collection],
        bbox=list(bbox),
        datetime=datetime,
        query={"eo:cloud_cover": {"lt": max_cloud}},
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
    Otherwise each band in ``bands`` is read separately, stacked in order, and
    rescaled from reflectance to ``[0, 255]``.
    """
    if asset == VISUAL_ASSET:
        rgb = reproject_to_grid(
            _asset_href(item, VISUAL_ASSET), grid, resampling=Resampling.bilinear
        )
        return rgb[:, :, :3].astype(np.uint8)

    channels = [
        reproject_to_grid(
            _asset_href(item, band), grid, resampling=Resampling.bilinear
        )[:, :, 0]
        for band in bands
    ]
    stacked = np.stack(channels, axis=-1)
    return scale_reflectance(stacked, ceiling=reflectance_ceiling)


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
class SceneResult:
    """A single resolved date: the chosen scene plus its rasters and stats."""

    item_id: str
    datetime: str
    scene_cloud_cover: float | None
    aoi_cloud_fraction: float
    imagery: np.ndarray
    cloud_mask: np.ndarray


def _resolve_scene(
    catalog: Any,
    bbox: Sequence[float],
    datetime: str,
    grid: TargetGrid,
    *,
    asset: str,
    bands: Sequence[str],
    max_cloud: float,
    reflectance_ceiling: float,
) -> SceneResult:
    """Search, pick the least-cloudy scene, and read it onto ``grid``."""
    items = search_items(catalog, bbox, datetime, max_cloud=max_cloud)
    item = select_least_cloudy(items)
    imagery = read_item_imagery(
        item,
        grid,
        asset=asset,
        bands=bands,
        reflectance_ceiling=reflectance_ceiling,
    )
    scl = read_item_scl(item, grid)
    cloud, _ = scl_to_masks(scl)
    return SceneResult(
        item_id=str(item.id),
        datetime=str(item.properties.get("datetime", "")),
        scene_cloud_cover=item.properties.get("eo:cloud_cover"),
        aoi_cloud_fraction=round(cloud_fraction(scl), 4),
        imagery=imagery,
        cloud_mask=cloud.astype(np.uint8),
    )


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
        write_cloud_masks: Also write per-date SCL cloud-mask GeoTIFFs.
        catalog: An open STAC client (defaults to Planetary Computer).

    Returns:
        A manifest dict (also written to ``manifest.json``) describing the grid,
        both chosen scenes, and the output files.
    """
    band_keys = tuple(b.upper() for b in bands)
    grid = build_target_grid(bbox, resolution, dst_crs=dst_crs)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if catalog is None:
        catalog = open_catalog()

    scenes = {
        "before": _resolve_scene(
            catalog,
            bbox,
            datetime_t1,
            grid,
            asset=asset,
            bands=band_keys,
            max_cloud=max_cloud,
            reflectance_ceiling=reflectance_ceiling,
        ),
        "after": _resolve_scene(
            catalog,
            bbox,
            datetime_t2,
            grid,
            asset=asset,
            bands=band_keys,
            max_cloud=max_cloud,
            reflectance_ceiling=reflectance_ceiling,
        ),
    }

    outputs: dict[str, str] = {}
    scene_manifest: dict[str, Any] = {}
    for label, scene in scenes.items():
        image_path = out_dir / f"{label}.tif"
        write_geotiff(
            image_path,
            scene.imagery,
            transform=grid.transform,
            crs=grid.crs,
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
                nodata=0,
                dtype="uint8",
            )
            outputs[f"{label}_cloud"] = str(mask_path)

        scene_manifest[label] = {
            "item_id": scene.item_id,
            "datetime": scene.datetime,
            "scene_cloud_cover": scene.scene_cloud_cover,
            "aoi_cloud_fraction": scene.aoi_cloud_fraction,
        }

    manifest: dict[str, Any] = {
        "bbox": [float(v) for v in bbox],
        "resolution_m": resolution,
        "asset": asset,
        "bands": list(band_keys) if asset != VISUAL_ASSET else ["R", "G", "B"],
        "grid": {
            "crs": grid.crs.to_string(),
            "width": grid.width,
            "height": grid.height,
            "transform": list(grid.transform)[:6],
        },
        "scenes": scene_manifest,
        "outputs": outputs,
    }
    manifest_path = out_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    manifest["outputs"]["manifest"] = str(manifest_path)
    return manifest
