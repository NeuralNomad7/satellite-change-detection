"""Geospatial I/O and vectorization for change-detection outputs.

This module is what makes the pipeline *georeferenced*. It:

- reads GeoTIFFs while preserving their CRS and affine transform (the existing
  ``data_loader.load_image`` only kept the pixels and threw the geo-metadata
  away);
- writes change masks back out as GeoTIFFs that line up with the source imagery
  in any GIS;
- converts a raster change mask into GeoJSON polygons annotated with real-world
  **area in hectares** and **centroid lon/lat**, plus a compact change-statistics
  summary.

It is deliberately free of any ``torch`` dependency -- only ``rasterio``,
``shapely`` and ``numpy`` -- so it can be imported and unit-tested without the
deep-learning stack installed.

Coordinate conventions
----------------------
Output GeoJSON geometries are always emitted in WGS84 lon/lat (EPSG:4326) to
comply with RFC 7946 and drop straight into web maps (Leaflet, geojson.io).
Ground areas are computed in the source projected CRS when it is metric
(e.g. Sentinel-2 UTM tiles) and via an equal-area projection when the source is
geographic, so reported hectares stay accurate either way.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from rasterio import Affine
from rasterio.crs import CRS
from rasterio.features import shapes as rasterio_shapes
from rasterio.io import MemoryFile
from rasterio.warp import transform_geom
from shapely.geometry import shape

# WGS84 / NSIDC EASE-Grid 2.0 Global -- a global equal-area projection in
# metres, used to measure ground area when the source raster is geographic.
EQUAL_AREA_CRS = "EPSG:6933"
WGS84_CRS = "EPSG:4326"

# ImageNet normalization constants (must match the training/serving pipeline in
# ``data_loader.get_transforms`` and ``serving.app``).
_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


@dataclass
class GeoRaster:
    """A raster plus the geospatial metadata needed to write it back out.

    Attributes:
        data: Pixel array of shape ``(H, W, C)``, float32.
        transform: Affine pixel-to-world transform, or ``None`` if absent.
        crs: Coordinate reference system, or ``None`` if absent.
        nodata: Nodata sentinel value, or ``None``.
    """

    data: np.ndarray
    transform: Affine | None
    crs: CRS | None
    nodata: float | None = None

    @property
    def height(self) -> int:
        return int(self.data.shape[0])

    @property
    def width(self) -> int:
        return int(self.data.shape[1])

    @property
    def count(self) -> int:
        return int(self.data.shape[2]) if self.data.ndim == 3 else 1


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------


def _raster_from_dataset(src: Any) -> GeoRaster:
    """Build a :class:`GeoRaster` from an open rasterio dataset."""
    arr = src.read().astype(np.float32)  # (C, H, W)
    data = np.transpose(arr, (1, 2, 0))  # (H, W, C)
    return GeoRaster(
        data=data,
        transform=src.transform,
        crs=src.crs,
        nodata=src.nodata,
    )


def read_geotiff(path: str | Path) -> GeoRaster:
    """Read a GeoTIFF from disk, preserving CRS and affine transform.

    Args:
        path: Path to a ``.tif``/``.tiff`` file.

    Returns:
        A :class:`GeoRaster` with pixels in ``(H, W, C)`` order.
    """
    with rasterio.open(str(path)) as src:
        return _raster_from_dataset(src)


def read_geotiff_bytes(data: bytes) -> GeoRaster:
    """Read a GeoTIFF from an in-memory byte buffer (e.g. an upload).

    Args:
        data: Raw GeoTIFF file bytes.

    Returns:
        A :class:`GeoRaster` with pixels in ``(H, W, C)`` order.
    """
    with MemoryFile(data) as memfile, memfile.open() as src:
        return _raster_from_dataset(src)


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------


def write_geotiff(
    path: str | Path,
    array: np.ndarray,
    *,
    transform: Affine | None = None,
    crs: CRS | str | None = None,
    nodata: float | None = None,
    dtype: str | None = None,
) -> None:
    """Write an array to a GeoTIFF, attaching CRS and transform.

    Accepts a 2-D mask ``(H, W)`` or a 3-D image ``(H, W, C)``.

    Args:
        path: Destination ``.tif`` path (parent dirs are created).
        array: Raster data, ``(H, W)`` or ``(H, W, C)``.
        transform: Affine pixel-to-world transform.
        crs: Coordinate reference system (CRS or EPSG string).
        nodata: Nodata value to record.
        dtype: Output dtype; defaults to the array's dtype.
    """
    arr = np.asarray(array)
    if arr.ndim == 2:
        arr = arr[np.newaxis, ...]  # (1, H, W)
    elif arr.ndim == 3:
        arr = np.transpose(arr, (2, 0, 1))  # (H, W, C) -> (C, H, W)
    else:
        raise ValueError(f"Expected a 2-D or 3-D array, got shape {arr.shape}")

    out_dtype = dtype or str(arr.dtype)
    arr = arr.astype(out_dtype)
    count, height, width = arr.shape

    profile: dict[str, Any] = {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "count": count,
        "dtype": out_dtype,
        "compress": "deflate",
    }
    if transform is not None:
        profile["transform"] = transform
    if crs is not None:
        profile["crs"] = crs
    if nodata is not None:
        profile["nodata"] = nodata

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(str(path), "w", **profile) as dst:
        dst.write(arr)


# ---------------------------------------------------------------------------
# Preprocessing (torch-free)
# ---------------------------------------------------------------------------


def normalize_imagenet(data: np.ndarray, in_channels: int = 3) -> np.ndarray:
    """Normalize a raster for model input, matching the training pipeline.

    Selects the first ``in_channels`` bands. For 3-band (RGB) input this applies
    ImageNet mean/std normalization; for other band counts it falls back to a
    simple ``[0, 1]`` scaling.

    Args:
        data: Raster of shape ``(H, W, C)`` (or ``(H, W)``) with values in
            ``[0, 255]``.
        in_channels: Number of bands the model expects.

    Returns:
        Float32 array of shape ``(1, in_channels, H, W)``, ready to wrap in a
        ``torch.Tensor``.
    """
    arr = np.asarray(data, dtype=np.float32)
    if arr.ndim == 2:
        arr = arr[..., np.newaxis]
    _, _, channels = arr.shape
    if channels < in_channels:
        raise ValueError(
            f"Raster has {channels} band(s) but the model expects {in_channels}."
        )
    arr = arr[:, :, :in_channels]

    if in_channels == 3:
        arr = arr / 255.0
        arr = (arr - _IMAGENET_MEAN) / _IMAGENET_STD
    else:
        arr = arr / 255.0

    chw = np.transpose(arr, (2, 0, 1))  # (C, H, W)
    return chw[np.newaxis, ...].astype(np.float32)  # (1, C, H, W)


# ---------------------------------------------------------------------------
# Vectorization
# ---------------------------------------------------------------------------


def _as_crs(crs: CRS | str | None) -> CRS | None:
    if crs is None:
        return None
    if isinstance(crs, CRS):
        return crs
    return CRS.from_user_input(crs)


def _is_wgs84(crs: CRS) -> bool:
    try:
        return crs.to_epsg() == 4326
    except Exception:
        return False


def _geom_area_m2(geom: dict[str, Any], crs: CRS | None) -> float | None:
    """Ground area of a GeoJSON geometry in square metres."""
    if crs is None:
        return None
    if crs.is_geographic:
        geom = transform_geom(crs, EQUAL_AREA_CRS, geom)
    return float(shape(geom).area)


def _geom_centroid_lonlat(
    geom: dict[str, Any], crs: CRS | None
) -> tuple[float, float] | None:
    """Centroid of a geometry as ``(lon, lat)``, or ``None`` without a CRS."""
    centroid = shape(geom).centroid
    x, y = float(centroid.x), float(centroid.y)
    if crs is None:
        return None
    if crs.is_geographic:
        return (x, y)
    point = transform_geom(crs, WGS84_CRS, {"type": "Point", "coordinates": [x, y]})
    lon, lat = point["coordinates"]
    return (float(lon), float(lat))


def mask_to_polygons(
    mask: np.ndarray,
    transform: Affine | None = None,
    crs: CRS | str | None = None,
    min_area_m2: float = 0.0,
) -> dict[str, Any]:
    """Vectorize a binary change mask into a GeoJSON ``FeatureCollection``.

    Each connected changed region becomes a polygon feature. Geometries are
    emitted in WGS84 lon/lat (EPSG:4326) when a CRS is known; otherwise they
    stay in pixel coordinates. Per-feature properties include ground area
    (``area_m2`` / ``area_ha``) and centroid (``centroid_lon`` / ``centroid_lat``).

    Args:
        mask: Binary mask ``(H, W)``; any nonzero pixel is treated as change.
        transform: Affine transform of the mask raster.
        crs: Coordinate reference system of the mask raster.
        min_area_m2: Drop change regions smaller than this many square metres
            (only applied when a CRS is available).

    Returns:
        A GeoJSON ``FeatureCollection`` dict.
    """
    crs_obj = _as_crs(crs)
    mask_arr = (np.asarray(mask) > 0).astype(np.uint8)
    geom_transform = transform if transform is not None else Affine.identity()
    reproject = crs_obj is not None and not _is_wgs84(crs_obj)

    features: list[dict[str, Any]] = []
    for geom, value in rasterio_shapes(
        mask_arr, mask=mask_arr.astype(bool), transform=geom_transform
    ):
        if int(value) != 1:
            continue

        area_m2 = _geom_area_m2(geom, crs_obj)
        if area_m2 is not None and min_area_m2 > 0.0 and area_m2 < min_area_m2:
            continue

        centroid = _geom_centroid_lonlat(geom, crs_obj)
        out_geom = transform_geom(crs_obj, WGS84_CRS, geom) if reproject else geom

        properties: dict[str, Any] = {
            "area_m2": round(area_m2, 2) if area_m2 is not None else None,
            "area_ha": round(area_m2 / 10_000.0, 4) if area_m2 is not None else None,
        }
        if centroid is not None:
            properties["centroid_lon"] = round(centroid[0], 6)
            properties["centroid_lat"] = round(centroid[1], 6)

        features.append(
            {"type": "Feature", "geometry": out_geom, "properties": properties}
        )

    return {"type": "FeatureCollection", "features": features}


def change_statistics(
    mask: np.ndarray,
    transform: Affine | None = None,
    crs: CRS | str | None = None,
    polygons: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Summarize a change mask: changed area, fraction, and region count.

    Args:
        mask: Binary change mask ``(H, W)``.
        transform: Affine transform of the mask raster.
        crs: Coordinate reference system of the mask raster.
        polygons: Optional precomputed :func:`mask_to_polygons` output, to avoid
            recomputing it.

    Returns:
        Dictionary of summary statistics. Area fields are ``None`` when no CRS
        is available.
    """
    crs_obj = _as_crs(crs)
    mask_arr = (np.asarray(mask) > 0).astype(np.uint8)
    total_pixels = int(mask_arr.size)
    changed_pixels = int(mask_arr.sum())

    fc = (
        polygons
        if polygons is not None
        else mask_to_polygons(mask_arr, transform=transform, crs=crs_obj)
    )
    areas = [
        f["properties"]["area_m2"]
        for f in fc["features"]
        if f["properties"].get("area_m2") is not None
    ]
    changed_area_m2 = float(sum(areas)) if areas else 0.0

    stats: dict[str, Any] = {
        "total_pixels": total_pixels,
        "changed_pixels": changed_pixels,
        "change_fraction": (
            round(changed_pixels / total_pixels, 6) if total_pixels else 0.0
        ),
        "num_change_regions": len(fc["features"]),
        "changed_area_m2": round(changed_area_m2, 2) if crs_obj is not None else None,
        "changed_area_ha": (
            round(changed_area_m2 / 10_000.0, 4) if crs_obj is not None else None
        ),
        "crs": crs_obj.to_string() if crs_obj is not None else None,
    }
    if transform is not None and crs_obj is not None and crs_obj.is_projected:
        stats["pixel_size_m"] = [round(abs(transform.a), 4), round(abs(transform.e), 4)]
    return stats


def save_geojson(feature_collection: dict[str, Any], path: str | Path) -> None:
    """Write a GeoJSON ``FeatureCollection`` to disk."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(feature_collection, f, indent=2)
