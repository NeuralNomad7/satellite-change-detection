"""Run georeferenced change detection on a pair of GeoTIFFs.

Unlike ``src.eval`` (which works on plain image tensors and writes PNG
visualizations), this script preserves geospatial metadata end to end: it reads
two co-registered GeoTIFFs, runs the model, and writes outputs that line up with
the source imagery in any GIS:

- ``change_mask.tif``     -- the binary change mask as a georeferenced GeoTIFF
- ``change_polygons.geojson`` -- change regions as WGS84 polygons with area (ha)
- ``change_stats.json``   -- a summary (changed area, fraction, region count)

Usage:
    python scripts/predict_geo.py \
        --checkpoint models/checkpoints/best_model.pth \
        --image-t1 before.tif \
        --image-t2 after.tif \
        --output-dir results/geo

Note:
    This expects the two GeoTIFFs to already be co-registered (same grid) and in
    the same pixel value range the model was trained on (0-255 RGB by default).
    Fetching, cloud-masking and aligning raw Sentinel-2 scenes is handled by the
    forthcoming ingestion module (Phase 2).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from src.config import Config
from src.eval import (
    load_model_from_checkpoint,
    predict_single,
    sliding_window_inference,
)
from src.geo import (
    GeoRaster,
    change_statistics,
    mask_to_polygons,
    normalize_imagenet,
    read_geotiff,
    save_geojson,
    write_geotiff,
)
from src.utils import get_device, set_seed


def run_geo_inference(
    model: torch.nn.Module,
    raster1: GeoRaster,
    raster2: GeoRaster,
    device: torch.device,
    in_channels: int,
    patch_size: int,
    stride: int,
    threshold: float,
) -> np.ndarray:
    """Preprocess two rasters and run (tiled if large) inference.

    Returns:
        Binary change mask ``(H, W)`` as a uint8 numpy array.
    """
    x1 = torch.from_numpy(normalize_imagenet(raster1.data, in_channels)[0])
    x2 = torch.from_numpy(normalize_imagenet(raster2.data, in_channels)[0])

    _, height, width = x1.shape
    if height > patch_size or width > patch_size:
        return sliding_window_inference(
            model,
            x1,
            x2,
            device,
            patch_size=patch_size,
            stride=stride,
            threshold=threshold,
        )
    return predict_single(model, x1, x2, device, threshold=threshold)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Georeferenced change detection on a GeoTIFF pair"
    )
    parser.add_argument(
        "--config", type=str, default="configs/default.yaml", help="YAML config path"
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None, help="Path to .pth checkpoint"
    )
    parser.add_argument(
        "--image-t1", type=str, required=True, help="Pre-change GeoTIFF path"
    )
    parser.add_argument(
        "--image-t2", type=str, required=True, help="Post-change GeoTIFF path"
    )
    parser.add_argument(
        "--output-dir", type=str, default="results/geo", help="Directory for outputs"
    )
    parser.add_argument(
        "--patch-size", type=int, default=256, help="Sliding-window patch size"
    )
    parser.add_argument(
        "--stride", type=int, default=128, help="Sliding-window stride (overlap)"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Binarization threshold (defaults to config.evaluation.threshold)",
    )
    parser.add_argument(
        "--min-area-m2",
        type=float,
        default=0.0,
        help="Drop change regions smaller than this many square metres",
    )
    parser.add_argument(
        "--device", type=str, default=None, help="Device override (e.g. 'cpu')"
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    config = Config.from_yaml(args.config)
    set_seed(config.seed)
    device = get_device(args.device)
    print(f"Using device: {device}")

    checkpoint_path = args.checkpoint
    if checkpoint_path is None:
        checkpoint_path = str(Path(config.paths.checkpoint_dir) / "best_model.pth")
    if not Path(checkpoint_path).exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}. "
            f"Train a model first with: python -m src.train"
        )

    model = load_model_from_checkpoint(checkpoint_path, config, device)

    raster1 = read_geotiff(args.image_t1)
    raster2 = read_geotiff(args.image_t2)

    if raster1.data.shape[:2] != raster2.data.shape[:2]:
        raise ValueError(
            f"Image dimensions must match. Got T1 {raster1.data.shape[:2]} "
            f"vs T2 {raster2.data.shape[:2]}. Co-register the pair first."
        )
    if raster1.crs != raster2.crs or raster1.transform != raster2.transform:
        print(
            "Warning: T1 and T2 have different georeferencing; "
            "using T1's CRS/transform for the outputs."
        )
    if raster1.crs is None:
        print(
            "Warning: input has no CRS; outputs will lack real-world "
            "coordinates and area measurements."
        )

    threshold = (
        args.threshold if args.threshold is not None else config.evaluation.threshold
    )
    mask = run_geo_inference(
        model,
        raster1,
        raster2,
        device,
        in_channels=config.model.in_channels,
        patch_size=args.patch_size,
        stride=args.stride,
        threshold=threshold,
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    mask_path = out_dir / "change_mask.tif"
    write_geotiff(
        mask_path,
        mask.astype(np.uint8),
        transform=raster1.transform,
        crs=raster1.crs,
        nodata=0,
        dtype="uint8",
    )

    polygons = mask_to_polygons(
        mask,
        transform=raster1.transform,
        crs=raster1.crs,
        min_area_m2=args.min_area_m2,
    )
    geojson_path = out_dir / "change_polygons.geojson"
    save_geojson(polygons, geojson_path)

    stats = change_statistics(
        mask, transform=raster1.transform, crs=raster1.crs, polygons=polygons
    )
    stats_path = out_dir / "change_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    print(f"\n{'=' * 50}")
    print("Change Detection Summary")
    print(f"{'=' * 50}")
    print(f"  Changed pixels : {stats['changed_pixels']:,} / {stats['total_pixels']:,}")
    print(f"  Change fraction: {stats['change_fraction']:.4f}")
    print(f"  Change regions : {stats['num_change_regions']:,}")
    if stats["changed_area_ha"] is not None:
        print(f"  Changed area   : {stats['changed_area_ha']:.2f} ha")
    print(f"{'=' * 50}")
    print(f"  Mask    -> {mask_path}")
    print(f"  Polygons-> {geojson_path}")
    print(f"  Stats   -> {stats_path}")


if __name__ == "__main__":
    main()
