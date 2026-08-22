# Security Policy

## Supported versions

This project is under active development. Security fixes are applied to the
latest released version on `main`.

| Version | Supported |
| ------- | --------- |
| 1.3.x   | Yes       |
| < 1.3   | No        |

## Reporting a vulnerability

**Please do not report security vulnerabilities through public GitHub issues.**

Report privately using GitHub's
[private vulnerability reporting](https://github.com/neuralnomad7/satellite-change-detection/security/advisories/new)
on this repository. Include:

- A description of the issue and its impact
- Steps to reproduce, or a proof of concept
- Affected version or commit
- Any suggested mitigation

You can expect an initial response within 7 days. If the report is accepted,
you will be credited in the advisory unless you prefer to remain anonymous.

## Scope and known risk areas

This is a machine learning pipeline, and the highest-risk surfaces are:

- **Model checkpoint loading.** `torch.load` is used with the PyTorch 2.6+
  `weights_only=True` default, which blocks arbitrary code execution during
  unpickling. Never remove that safeguard to load an untrusted checkpoint.
  Only load checkpoints from sources you trust.
- **The inference API** (`serving/app.py`). It accepts uploaded imagery and is
  intended to run behind your own authentication and network controls. It ships
  with no auth, no rate limiting, and no upload size cap — treat it as an
  internal service, not a public endpoint, unless you add those yourself.
- **Raster parsing.** GeoTIFF decoding is handled by `rasterio`/GDAL. Keep it
  patched, and be cautious with rasters from untrusted sources.
- **Remote imagery ingestion.** `sat-cd-ingest` fetches from the Microsoft
  Planetary Computer STAC API over the network.

Model accuracy issues, false positives in change masks, and general
misclassification are **not** security vulnerabilities — please open a normal
issue for those.
