"""Framework constants and compatibility exports.

This module keeps older imports such as ``core.constants`` working while
the canonical physical constants live in :mod:`core.physics_constants`.
"""

from dataclasses import dataclass

from .physics_constants import (
    X,
    E,
    T,
    P,
    Z_MASS,
    ALPHA_VAL,
    ALPHA_REF,
    g1_REF,
    g2_REF,
    g3_REF,
    GAMMA_1,
    GAMMA_2,
    GAMMA_3,
)

VERSION = "1.0.0"
DEBUG = False
MAX_WORKERS = 4
CACHE_SIZE = 1000
LOG_LEVEL = "INFO"

GUT_SCALE = 2.1e16  # GeV


@dataclass(frozen=True)
class Constants:
    """Common SI and electroweak reference constants."""

    G: float = 6.67430e-11
    hbar: float = 1.054571817e-34
    M_PLANCK: float = 2.176434e-8
    c: float = 299792458.0
    M_Z: float = 91.1876
