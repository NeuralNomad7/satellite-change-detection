# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- `tests/test_data_loader.py`, covering the augmentation pipeline and mask
  decoding. Includes a guard that promotes albumentations' "unrecognized
  argument" `UserWarning` to an error, so a renamed parameter fails CI instead
  of silently degrading augmentation.
- Python 3.14 to the CI matrix and the package classifiers. The full dependency
  set resolves on 3.14, and PyTorch has supported it since 2.12.
- Dependabot configuration for weekly `pip` and `github-actions` updates, with
  minor/patch bumps grouped into a single pull request.
- Community health files: `CONTRIBUTING.md`, `SECURITY.md`, issue forms for bug
  reports and feature requests, and a pull request template.
- This changelog.

### Fixed
- **CI now runs on every pull request.** The `pull_request` trigger was filtered
  to `branches: [main]`, and because that filter matches the *base* branch, any
  pull request targeting a feature branch received no checks at all. Stacked
  pull requests were merging without lint, type, or test coverage.
- **Gaussian noise augmentation was silently disabled.** albumentations 2.0
  renamed `GaussNoise`'s `var_limit` to `std_range`, and unknown keyword
  arguments only raise a `UserWarning`, so the transform quietly fell back to
  its defaults. Migrated to `std_range=(0.012, 0.028)`, which preserves the
  previous strength (`sqrt(10)/255` to `sqrt(50)/255`).
- **`load_mask` discarded every pixel of a 0/1-encoded mask.** It thresholded at
  a fixed 128 despite documenting support for both 0/1 and 0/255 encodings, so
  0/1 labels became entirely empty and training silently had nothing to learn
  from. The threshold is now derived from the data.
- **mypy was not actually type checking anything.** The lint job installed only
  `ruff` and `mypy`, so with `--ignore-missing-imports` every third-party type
  resolved to `Any`. The job now installs the project (using the CPU torch wheel
  to keep the download small), which surfaced and fixed four real errors:
  a `Tensor` assigned to an `ndarray`-typed variable in `data_loader.py`, a
  widened `betas` tuple passed to `AdamW`, and two `save_checkpoint` calls typed
  against the private, deprecated `_LRScheduler` instead of `LRScheduler`.
- Removed the hardcoded `python_version = "3.10"` from the mypy configuration.
  numpy 2.5's stubs use PEP 695 syntax that mypy cannot parse when targeting
  3.10, which broke local type checking outright.

### Changed
- Raised the `albumentations` floor to `>=2.0.0` so the transform API is
  unambiguous.
- CI can be triggered manually via `workflow_dispatch`.

### Removed
- `pandas`, which was unused across `src/`, `scripts/`, `serving/`,
  `streamlit_app.py`, and the notebooks.

## [1.3.0] - 2026-06-13

### Added
- **Sentinel-2 ingestion** (`src/ingest.py`): turn a bounding box and two dates
  into a co-registered GeoTIFF pair ready for inference.
  - `build_target_grid` with automatic UTM zone selection, `reproject_to_grid`
  - SCL-based cloud screening (`scl_to_masks`, `cloud_fraction`, `select_least_cloudy`)
  - `ingest_pair` writes `before.tif` / `after.tif`, cloud masks, and a `manifest.json`
- `sat-cd-ingest` console script, which prints a ready-to-run `sat-cd-geo` command.
- `[ingest]` optional extra (`pystac-client`, `planetary-computer`), lazily
  imported so the core package stays lightweight.
- 19 network-free tests covering ingestion against a fake STAC catalog.

## [1.2.0] - 2026-06-13

### Added
- **Geospatial I/O** (`src/geo.py`, torch-free): `GeoRaster`, GeoTIFF read/write,
  `mask_to_polygons` (GeoJSON in WGS84 with area in hectares and centroids),
  `change_statistics`, and `normalize_imagenet`.
- `sat-cd-geo` console script: a GeoTIFF pair in, `mask.tif` + `polygons.geojson`
  + `stats.json` out, reusing the sliding-window inference path.
- `POST /predict/geotiff` endpoint returning statistics and GeoJSON.
- 10 torch-free geospatial tests.
- `shapely` as a core dependency.

## [1.1.0] - 2026-06-13

### Added
- PEP 621 `pyproject.toml` with console scripts, optional dev dependencies, and
  Ruff / mypy / pytest configuration.
- `.pre-commit-config.yaml` (Ruff, ruff-format, mypy, standard hooks).
- Python 3.12 and 3.13 to the CI test matrix.

### Changed
- Ruff replaces Black, Flake8, and isort.
- Docker base images moved to `python:3.12-slim`.
- Minimum supported Python is now 3.10 (3.9 dropped).
- Dependency floors raised to current stable releases.

### Fixed
- **ONNX export under torch 2.9+**, which flipped `torch.onnx.export` to
  `dynamo=True` by default and broke on the model's data-dependent control flow.
  Pinned `dynamo=False`.
- **Deprecated AMP APIs**: migrated `torch.cuda.amp.{autocast, GradScaler}` to
  `torch.amp.{autocast, GradScaler}("cuda", ...)`, gated on CUDA availability so
  CPU runs no longer emit warnings.

### Removed
- `setup.py`, superseded by `pyproject.toml`.

## [1.0.0] - 2026-03-31

### Added
- Production deployment stack: FastAPI serving, Docker and Docker Compose, a
  Streamlit demo, and ONNX export with benchmarking.
- Animated pipeline visualization.

## [0.1.0] - 2026-02-14

### Added
- Initial Siamese U-Net change detection pipeline: model, data loading,
  training, and evaluation.

[Unreleased]: https://github.com/neuralnomad7/satellite-change-detection/compare/v1.3.0...HEAD
[1.3.0]: https://github.com/neuralnomad7/satellite-change-detection/compare/v1.2.0...v1.3.0
[1.2.0]: https://github.com/neuralnomad7/satellite-change-detection/compare/v1.1.0...v1.2.0
[1.1.0]: https://github.com/neuralnomad7/satellite-change-detection/compare/v1.0.0...v1.1.0
[1.0.0]: https://github.com/neuralnomad7/satellite-change-detection/compare/v0.1.0...v1.0.0
[0.1.0]: https://github.com/neuralnomad7/satellite-change-detection/releases/tag/v0.1.0
