"""Tests for the ``sat-cd-ingest`` command line (``scripts.ingest_s2``).

Argument validation and error reporting are checked without any network
access: :func:`ingest_pair` is replaced with a stand-in where needed.
"""

from __future__ import annotations

import pytest

import scripts.ingest_s2 as cli
from src.ingest import DEFAULT_CANDIDATES

REQUIRED = [
    "--bbox",
    "12.30",
    "45.40",
    "12.45",
    "45.50",
    "--date-t1",
    "2023-06-01/2023-06-30",
    "--date-t2",
    "2024-06-01/2024-06-30",
]


def _fake_manifest(out_dir, *, with_masks=True):
    outputs = {
        "before": f"{out_dir}/before.tif",
        "after": f"{out_dir}/after.tif",
        "manifest": f"{out_dir}/manifest.json",
    }
    if with_masks:
        outputs["before_cloud"] = f"{out_dir}/before_cloud.tif"
        outputs["after_cloud"] = f"{out_dir}/after_cloud.tif"
    scene = {
        "datetime": "2024-06-28T10:05:59Z",
        "mgrs_tile": "32TQR",
        "processing_baseline": "05.10",
        "scene_cloud_cover": 13.94,
        "aoi_cloud_fraction": 0.012,
        "aoi_nodata_fraction": 0.0,
        "candidates_found": 7,
        "candidates_evaluated": 3,
    }
    return {
        "grid": {"width": 1176, "height": 1113, "crs": "EPSG:32633"},
        "scenes": {
            "before": {**scene, "item_id": "S2_BEFORE"},
            "after": {**scene, "item_id": "S2_AFTER"},
        },
        "outputs": outputs,
    }


def test_parse_args_defaults():
    args = cli._parse_args(REQUIRED)

    assert args.bbox == [12.30, 45.40, 12.45, 45.50]
    assert args.candidates == DEFAULT_CANDIDATES
    assert args.max_cloud == 20.0
    assert args.asset == "visual"
    assert not args.no_cloud_masks


@pytest.mark.parametrize(
    ("extra", "flag"),
    [
        (["--max-cloud", "120"], "--max-cloud"),
        (["--max-cloud", "-1"], "--max-cloud"),
        (["--resolution", "0"], "--resolution"),
        (["--resolution", "nan"], "--resolution"),
        (["--reflectance-ceiling", "0"], "--reflectance-ceiling"),
        (["--candidates", "0"], "--candidates"),
    ],
)
def test_parse_args_rejects_bad_values(extra, flag, capsys):
    with pytest.raises(SystemExit) as exc:
        cli._parse_args(REQUIRED + extra)

    assert exc.value.code == 2
    assert flag in capsys.readouterr().err


def test_parse_args_catches_a_lat_lon_bbox(capsys):
    swapped = ["--bbox", "35.6", "139.6", "35.8", "139.8", *REQUIRED[5:]]

    with pytest.raises(SystemExit) as exc:
        cli._parse_args(swapped)

    assert exc.value.code == 2
    assert "longitude comes first" in capsys.readouterr().err


def test_main_forwards_arguments_and_suggests_the_next_step(
    monkeypatch, capsys, tmp_path
):
    calls = []

    def fake_ingest_pair(**kwargs):
        calls.append(kwargs)
        return _fake_manifest(tmp_path)

    monkeypatch.setattr(cli, "ingest_pair", fake_ingest_pair)

    cli.main([*REQUIRED, "--candidates", "5", "--output-dir", str(tmp_path)])

    (kwargs,) = calls
    assert kwargs["candidates"] == 5
    assert kwargs["bbox"] == [12.30, 45.40, 12.45, 45.50]
    assert kwargs["write_cloud_masks"] is True
    out = capsys.readouterr().out
    assert "tile 32TQR, baseline 05.10" in out
    assert "best of 3 checked (7 found)" in out
    # The suggested command closes the loop to cloud-aware inference.
    assert f"--manifest {tmp_path}/manifest.json" in out


def test_main_omits_the_manifest_flag_without_cloud_masks(
    monkeypatch, capsys, tmp_path
):
    monkeypatch.setattr(
        cli,
        "ingest_pair",
        lambda **kwargs: _fake_manifest(tmp_path, with_masks=False),
    )

    cli.main([*REQUIRED, "--no-cloud-masks"])

    assert "--manifest" not in capsys.readouterr().out


@pytest.mark.parametrize(
    ("error", "message"),
    [
        (
            ValueError("No Sentinel-2 scenes found for 2023-06. Widen the window."),
            "error: No Sentinel-2 scenes found for 2023-06. Widen the window.\n",
        ),
        # str(KeyError) would wrap the message in quotes.
        (
            KeyError("Asset 'B08' not on item S2_X. Available: SCL, visual"),
            "error: Asset 'B08' not on item S2_X. Available: SCL, visual\n",
        ),
        (OSError("HTTP 503 reading SCL"), "error: HTTP 503 reading SCL\n"),
    ],
)
def test_main_reports_expected_failures_in_one_line(
    monkeypatch, capsys, error, message
):
    def failing_ingest_pair(**kwargs):
        raise error

    monkeypatch.setattr(cli, "ingest_pair", failing_ingest_pair)

    with pytest.raises(SystemExit) as exc:
        cli.main(REQUIRED)

    assert exc.value.code == 1
    assert capsys.readouterr().err == message


def test_main_reports_stac_api_errors(monkeypatch, capsys):
    exceptions = pytest.importorskip("pystac_client.exceptions")

    def failing_ingest_pair(**kwargs):
        raise exceptions.APIError("upstream timeout")

    monkeypatch.setattr(cli, "ingest_pair", failing_ingest_pair)

    with pytest.raises(SystemExit) as exc:
        cli.main(REQUIRED)

    assert exc.value.code == 1
    assert "error: upstream timeout" in capsys.readouterr().err


def test_main_lets_unexpected_errors_surface(monkeypatch):
    def broken_ingest_pair(**kwargs):
        raise RuntimeError("bug")

    monkeypatch.setattr(cli, "ingest_pair", broken_ingest_pair)

    with pytest.raises(RuntimeError, match="bug"):
        cli.main(REQUIRED)
