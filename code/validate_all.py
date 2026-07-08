#!/usr/bin/env python3
"""Validate the active paper-code test suite."""

import os
import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PROJECT_ROOT.parent


def setup_python_path() -> None:
    """Make the code package importable from any launch directory."""
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))


def main():
    """Run all validation checks."""
    setup_python_path()
    os.chdir(PROJECT_ROOT)
    return pytest.main([
        "tests",
    ])


def validate_gravitational_waves():
    """Validate gravitational wave predictions."""
    setup_python_path()
    os.chdir(PROJECT_ROOT)

    from generate_data import generate_gw_spectrum_data

    print("Validating gravitational wave predictions...")

    generate_gw_spectrum_data(REPO_ROOT / "data")

    integration_status = pytest.main([
        "tests/test_integration.py::test_gravitational_wave_coherence",
        "-v"
    ])
    prediction_status = pytest.main([
        "tests/test_predictions.py::test_gravitational_wave_spectrum",
        "-v"
    ])
    return integration_status or prediction_status


if __name__ == "__main__":
    sys.exit(main())
