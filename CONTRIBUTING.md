# Contributing

Thanks for your interest in improving this project. Change detection is most
useful when the people who need it — ecologists, city planners, disaster
responders, researchers — can actually run and extend it, so contributions of
all sizes are welcome.

## Getting set up

```bash
git clone https://github.com/neuralnomad7/satellite-change-detection.git
cd satellite-change-detection

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# Install the package plus dev tooling (Ruff, mypy, pytest, pre-commit)
pip install -e ".[dev]"

# Optional: Sentinel-2 STAC ingestion (only needed for sat-cd-ingest)
pip install -e ".[dev,ingest]"

pre-commit install
```

Python 3.10+ is required.

## The checks CI runs

Run these before opening a pull request — they are exactly what CI enforces:

```bash
ruff check src/ tests/ scripts/
ruff format --check src/ tests/ scripts/
mypy src/ --ignore-missing-imports --no-strict-optional
pytest tests/ -v
```

`ruff format src/ tests/ scripts/` (without `--check`) fixes formatting in place.
If you installed the pre-commit hooks, most of this runs automatically on commit.

## Testing expectations

- **Tests must not require network access.** The Sentinel-2 ingestion tests run
  against a fake STAC catalog backed by local rasters; please follow that
  pattern rather than hitting a live API.
- **Tests must not require a GPU or a trained checkpoint.** Build small tensors
  or temporary GeoTIFFs in the test itself.
- Keep geospatial logic torch-free where possible. `src/geo.py` and
  `src/ingest.py` deliberately avoid importing torch so they stay fast to test
  and usable without a deep learning stack.

## Pull requests

- Keep each PR focused on one concern; small PRs get reviewed faster.
- Update the README and `CHANGELOG.md` when you change behavior, add a CLI, or
  change dependencies.
- Note any new dependency in the PR description and explain why it is needed.
  Heavy or optional dependencies belong in an extra (see `[project.optional-dependencies]`),
  not in `requirements.txt`.
- CI runs on every pull request regardless of the base branch, so stacked PRs
  are covered too.

## Reporting bugs and requesting features

Please use the issue templates. For bugs, the single most helpful thing you can
include is the exact command you ran plus the full traceback, along with your OS,
Python version, and `torch` / `rasterio` versions.

For anything security-related, do **not** open a public issue — see
[SECURITY.md](SECURITY.md).

## Scope notes

Good areas to contribute:

- Additional change-detection backbones or loss functions in `src/models.py`
- More imagery providers alongside the Planetary Computer STAC source
- Benchmarks and reproducible evaluation on public change-detection datasets
- Documentation, tutorials, and worked examples

## License

By contributing, you agree that your contributions are licensed under the
project's [MIT License](LICENSE).
