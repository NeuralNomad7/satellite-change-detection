"""Packaging metadata consistency checks.

The release version and the supported Python versions are each declared in
more than one place. These tests fail when those declarations drift apart.
"""

import re
import tomllib
from pathlib import Path
from typing import Any

import yaml

import src

ROOT = Path(__file__).resolve().parents[1]

Version = tuple[int, int]


def _pyproject() -> dict[str, Any]:
    with (ROOT / "pyproject.toml").open("rb") as f:
        return tomllib.load(f)


def _parse_version(text: str) -> Version:
    major, minor = text.split(".")
    return int(major), int(minor)


def _classifier_versions(project: dict[str, Any]) -> list[Version]:
    pattern = r"Programming Language :: Python :: (\d+\.\d+)"
    return sorted(
        _parse_version(match[1])
        for classifier in project["classifiers"]
        if (match := re.fullmatch(pattern, classifier))
    )


def _ci_test_matrix_versions() -> list[Version]:
    workflow_path = ROOT / ".github" / "workflows" / "ci.yml"
    workflow = yaml.safe_load(workflow_path.read_text(encoding="utf-8"))
    matrix = workflow["jobs"]["test"]["strategy"]["matrix"]["python-version"]
    # str() so an unquoted 3.10 (which YAML reads as the float 3.1) fails loudly
    return sorted(_parse_version(str(version)) for version in matrix)


def test_package_version_matches_pyproject():
    assert src.__version__ == _pyproject()["project"]["version"]


def test_supported_python_versions_are_declared_consistently():
    pyproject = _pyproject()
    floor_match = re.fullmatch(
        r">=\s*(\d+\.\d+)", pyproject["project"]["requires-python"]
    )
    assert floor_match, "requires-python should be a plain '>=X.Y' floor"
    floor = _parse_version(floor_match[1])

    classifiers = _classifier_versions(pyproject["project"])
    assert classifiers, "pyproject should list Python version classifiers"
    assert classifiers[0] == floor
    assert pyproject["tool"]["ruff"]["target-version"] == f"py{floor[0]}{floor[1]}"
    # Every advertised version is tested in CI, and every tested one is advertised.
    assert _ci_test_matrix_versions() == classifiers
