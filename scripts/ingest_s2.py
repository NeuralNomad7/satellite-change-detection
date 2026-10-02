"""Fetch a co-registered Sentinel-2 pair for an area of interest.

Given a bounding box and two dates (or date ranges), this picks the Sentinel-2
L2A scene that sees the most of the area clearly for each date on the Microsoft
Planetary Computer, reprojects both onto a shared grid, and writes
``before.tif`` / ``after.tif`` (plus cloud masks and a ``manifest.json``). The
outputs feed straight into ``sat-cd-geo`` for georeferenced change detection.

Requires the optional ingestion dependencies::

    pip install -e ".[ingest]"

Usage::

    sat-cd-ingest \\
        --bbox 12.30 45.40 12.45 45.50 \\
        --date-t1 2023-06-01/2023-06-30 \\
        --date-t2 2024-06-01/2024-06-30 \\
        --output-dir data/venice \\
        --resolution 10 \\
        --max-cloud 20
"""

from __future__ import annotations

import argparse
import math
import sys
from collections.abc import Sequence

from src.ingest import (
    DEFAULT_CANDIDATES,
    RGB_BANDS,
    VISUAL_ASSET,
    ingest_pair,
    validate_bbox,
)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch a co-registered Sentinel-2 L2A pair for change detection"
    )
    parser.add_argument(
        "--bbox",
        type=float,
        nargs=4,
        required=True,
        metavar=("MIN_LON", "MIN_LAT", "MAX_LON", "MAX_LAT"),
        help="Area of interest as WGS84 lon/lat bounds",
    )
    parser.add_argument(
        "--date-t1",
        type=str,
        required=True,
        help="Before date or range, e.g. 2023-06-01 or 2023-06-01/2023-06-30",
    )
    parser.add_argument(
        "--date-t2",
        type=str,
        required=True,
        help="After date or range",
    )
    parser.add_argument(
        "--output-dir", type=str, default="data/aoi", help="Directory for outputs"
    )
    parser.add_argument(
        "--resolution", type=float, default=10.0, help="Ground resolution in metres"
    )
    parser.add_argument(
        "--max-cloud",
        type=float,
        default=20.0,
        help="Maximum scene-level cloud cover percent to consider (0-100)",
    )
    parser.add_argument(
        "--candidates",
        type=int,
        default=DEFAULT_CANDIDATES,
        help=(
            "Scenes per date whose cloud mask is checked over the AOI before "
            "choosing (default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--asset",
        type=str,
        default=VISUAL_ASSET,
        choices=[VISUAL_ASSET, "bands"],
        help="'visual' (8-bit true colour) or 'bands' for raw reflectance bands",
    )
    parser.add_argument(
        "--bands",
        type=str,
        nargs="+",
        default=list(RGB_BANDS),
        help="Band assets to stack when --asset bands (red green blue order)",
    )
    parser.add_argument(
        "--dst-crs",
        type=str,
        default=None,
        help="Output CRS (defaults to the AOI's UTM zone)",
    )
    parser.add_argument(
        "--reflectance-ceiling",
        type=float,
        default=3000.0,
        help="Reflectance clip ceiling when rescaling raw bands to 0-255",
    )
    parser.add_argument(
        "--no-cloud-masks",
        action="store_true",
        help="Skip writing per-date cloud-mask GeoTIFFs",
    )
    args = parser.parse_args(argv)

    try:
        args.bbox = list(validate_bbox(args.bbox))
    except ValueError as e:
        parser.error(str(e))
    if not 0.0 <= args.max_cloud <= 100.0:
        parser.error(f"--max-cloud must be between 0 and 100, got {args.max_cloud}")
    if not (math.isfinite(args.resolution) and args.resolution > 0):
        parser.error(f"--resolution must be positive, got {args.resolution}")
    if not (math.isfinite(args.reflectance_ceiling) and args.reflectance_ceiling > 0):
        parser.error(
            f"--reflectance-ceiling must be positive, got {args.reflectance_ceiling}"
        )
    if args.candidates < 1:
        parser.error(f"--candidates must be at least 1, got {args.candidates}")
    return args


def _expected_errors() -> tuple[type[BaseException], ...]:
    """Errors worth a one-line message instead of a traceback.

    ``OSError`` covers rasterio read failures and ``requests`` network errors.
    """
    errors: tuple[type[BaseException], ...] = (
        ValueError,
        KeyError,
        ImportError,
        OSError,
    )
    try:
        from pystac_client.exceptions import APIError
    except ImportError:
        return errors
    return (*errors, APIError)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)

    print(f"Searching Sentinel-2 L2A over bbox {args.bbox} ...", flush=True)
    try:
        manifest = ingest_pair(
            bbox=args.bbox,
            datetime_t1=args.date_t1,
            datetime_t2=args.date_t2,
            output_dir=args.output_dir,
            resolution=args.resolution,
            asset=args.asset,
            bands=args.bands,
            max_cloud=args.max_cloud,
            dst_crs=args.dst_crs,
            reflectance_ceiling=args.reflectance_ceiling,
            write_cloud_masks=not args.no_cloud_masks,
            candidates=args.candidates,
        )
    except _expected_errors() as e:
        # KeyError's str() wraps the message in quotes.
        message = e.args[0] if isinstance(e, KeyError) and e.args else e
        print(f"error: {message}", file=sys.stderr)
        sys.exit(1)

    grid = manifest["grid"]
    print(f"\n{'=' * 56}")
    print("Sentinel-2 Ingestion Summary")
    print(f"{'=' * 56}")
    print(f"  Grid     : {grid['width']}x{grid['height']} @ {grid['crs']}")
    for label in ("before", "after"):
        scene = manifest["scenes"][label]
        cover = scene["scene_cloud_cover"]
        cover_str = f"{cover:.1f}%" if cover is not None else "n/a"
        print(
            f"  {label.capitalize():7}: {scene['item_id']}\n"
            f"           {scene['datetime']}  tile {scene['mgrs_tile'] or 'n/a'}, "
            f"baseline {scene['processing_baseline'] or 'n/a'}\n"
            f"           scene cloud {cover_str}, "
            f"AOI cloud {scene['aoi_cloud_fraction']:.1%}, "
            f"AOI no data {scene['aoi_nodata_fraction']:.1%}\n"
            f"           best of {scene['candidates_evaluated']} checked "
            f"({scene['candidates_found']} found)"
        )
    print(f"{'=' * 56}")
    for key in ("before", "after"):
        print(f"  {key:7} -> {manifest['outputs'][key]}")
    print(f"  manifest-> {manifest['outputs']['manifest']}")

    outputs = manifest["outputs"]
    masks_flag = (
        f" \\\n    --manifest {outputs['manifest']}"
        if "before_cloud" in outputs
        else ""
    )
    print("\nNext, run change detection on the pair:")
    print(
        f"  sat-cd-geo --checkpoint models/checkpoints/best_model.pth \\\n"
        f"    --image-t1 {outputs['before']} --image-t2 {outputs['after']}"
        f"{masks_flag} \\\n"
        f"    --output-dir results/geo"
    )


if __name__ == "__main__":
    main()
