## What does this change?

<!-- A short description of the change and why it is needed. -->

Closes #

## Type of change

- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation
- [ ] Tooling, CI, or dependencies

## How was this tested?

<!--
Commands you ran and what you observed. If this changes model behavior or
geospatial output, please describe the data you validated against.
-->

## Checklist

- [ ] `ruff check src/ tests/ scripts/` passes
- [ ] `ruff format --check src/ tests/ scripts/` passes
- [ ] `mypy src/ --ignore-missing-imports --no-strict-optional` passes
- [ ] `pytest tests/ -v` passes
- [ ] New tests run without network access, a GPU, or a trained checkpoint
- [ ] README updated if behavior, a CLI, or dependencies changed
- [ ] `CHANGELOG.md` updated under `## [Unreleased]`
- [ ] Any new dependency is justified above, and optional or heavy ones are in an extra
