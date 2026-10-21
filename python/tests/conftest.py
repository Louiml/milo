"""Shared setup for Milo's Python tests.

The extension module is built by CMake and lands in the build tree, so the
tests look for it there. Set MILO_EXTENSION_DIR to point somewhere else.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]


def _candidate_directories() -> list[Path]:
    """The places the built extension may be."""
    candidates: list[Path] = []
    override = os.environ.get("MILO_EXTENSION_DIR")
    if override:
        candidates.append(Path(override))
    for configuration in ("debug", "release"):
        candidates.append(REPO_ROOT / "build" / configuration / "lib")
    return candidates


def _find_extension() -> Path | None:
    """The built extension module, or None when it has not been built."""
    for directory in _candidate_directories():
        if not directory.is_dir():
            continue
        for path in sorted(directory.glob("milo*.pyd")):
            return path
        for path in sorted(directory.glob("milo*.so")):
            return path
    return None


_EXTENSION = _find_extension()

if _EXTENSION is None:
    pytest.skip(
        "the Milo extension has not been built; run `python build.py` first",
        allow_module_level=True,
    )

sys.path.insert(0, str(_EXTENSION.parent))

import milo  # noqa: E402  (the path has to be set up first)


@pytest.fixture(scope="session")
def extension() -> object:
    """The Milo extension module."""
    return milo


@pytest.fixture(scope="session")
def numpy():
    """NumPy, or a skip when it is not installed."""
    return pytest.importorskip("numpy")


@pytest.fixture(scope="session")
def torch():
    """PyTorch, or a skip when it is not installed."""
    return pytest.importorskip("torch")


@pytest.fixture(scope="session")
def fixture_model() -> Path:
    """A small safetensors file, from the test data."""
    path = REPO_ROOT / "tests" / "scripts" / "fixture.safetensors"
    if not path.is_file():
        pytest.skip("the test fixture model is missing")
    return path
