"""Fetch a co-registered Sentinel-2 pair for an area of interest.

Given a bounding box and two dates (or date ranges), this downloads the
least-cloudy Sentinel-2 L2A scene for each from the Microsoft Planetary Computer,
reprojects both onto a shared grid, and writes ``before.tif`` / ``after.tif``
(plus SCL cloud masks and a ``manifest.json``). The outputs feed straight into
``sat-cd-geo`` for georeferenced change detection.

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

from src.ingest import RGB_BANDS, VISUAL_ASSET, ingest_pair


def _parse_args() -> argparse.Namespace:
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
        help="Maximum scene-level cloud cover percent to consider",
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
        help="Skip writing per-date SCL cloud-mask GeoTIFFs",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    print(f"Searching Sentinel-2 L2A over bbox {args.bbox} ...")
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
    )

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
            f"           {scene['datetime']}  "
            f"scene cloud {cover_str}, AOI cloud {scene['aoi_cloud_fraction']:.1%}"
        )
    print(f"{'=' * 56}")
    for key in ("before", "after"):
        print(f"  {key:7} -> {manifest['outputs'][key]}")
    print(f"  manifest-> {manifest['outputs']['manifest']}")

    before = manifest["outputs"]["before"]
    after = manifest["outputs"]["after"]
    print("\nNext, run change detection on the pair:")
    print(
        f"  sat-cd-geo --checkpoint models/checkpoints/best_model.pth \\\n"
        f"    --image-t1 {before} --image-t2 {after} --output-dir results/geo"
    )


if __name__ == "__main__":
    main()
