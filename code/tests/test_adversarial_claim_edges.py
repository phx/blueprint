"""Adversarial edge-case pressure for paper-to-code claims.

This file complements the paper validation matrix tests.  Those tests confirm
that the paper, registry, implementations, and assertions agree on the normal
path.  These tests ask where the same claims can fail at their boundaries.
"""

import csv
from pathlib import Path

import numpy as np
import pytest

from core import formulas
from core.errors import ValidationError
from core.field import UnifiedField
from core.physics_constants import ALPHA_VAL, G, GUT_SCALE, M_P
from core.self_replication import (
    generative_operator,
    replication_metrics,
    replication_sequence,
    self_similarity,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
VALIDATION_MATRIX_PATH = PROJECT_ROOT / "data" / "validation_matrix.csv"


CLAIM_EDGE_PRESSURE = {
    "fractal_scaling": (
        "level-zero identity",
        "monotone contraction for positive recursion levels",
    ),
    "holographic_entropy": (
        "zero-radius boundary",
        "large finite radius without overflow",
    ),
    "gut_predictions": (
        "prediction-table uncertainty envelope",
        "nonzero positive reference values",
    ),
    "coupling_unification": (
        "GUT-scale spread minimum",
        "away-from-GUT spread remains finite",
    ),
    "rg_flow": (
        "sub-reference energy clamping",
        "large finite energy remains positive",
    ),
    "fermion_hierarchy": (
        "strict quark ordering",
        "strict lepton ordering",
    ),
    "cp_violation": (
        "positive finite invariant",
        "small-magnitude reference proxy",
    ),
    "baryon_asymmetry": (
        "positive finite asymmetry",
        "small-magnitude reference proxy",
    ),
    "higgs_neutrino": (
        "ordered neutrino masses",
        "bounded oscillation probability",
    ),
    "dark_matter": (
        "zero-radius finite clamp",
        "radial monotonic decrease",
    ),
    "dark_energy": (
        "positive density",
        "finite positive cosmological constant at zero energy",
    ),
    "measurement": (
        "zero-amplitude rejection",
        "non-finite amplitude rejection",
    ),
    "gravity_uv": (
        "level-zero identity",
        "recursive suppression for higher levels",
    ),
    "information_dimension": (
        "zero-level boundary",
        "positive-level growth above four dimensions",
    ),
    "proton_decay": (
        "positive finite rate",
        "positive finite uncertainty",
    ),
    "sakharov_conditions": (
        "all required boolean keys present",
        "no truthy non-boolean placeholders",
    ),
    "low_energy": (
        "positive finite observables",
        "positive finite uncertainty values",
    ),
    "gw_spectrum": (
        "zero-frequency finite clamp",
        "power-law monotonicity",
    ),
    "generative_operator": (
        "zero-field rejection",
        "zero-step sequence identity",
    ),
    "uncertainty": (
        "zero-uncertainty boundary",
        "correlated extension stays finite",
    ),
    "documentation_traceability": (
        "matrix row completeness",
        "test-function existence",
    ),
}


def _validation_rows():
    with VALIDATION_MATRIX_PATH.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_every_validation_matrix_claim_has_named_edge_case_pressure():
    """Forces future paper claims to name their adversarial edge targets."""
    claim_ids = {row["claim_id"] for row in _validation_rows()}

    assert claim_ids == set(CLAIM_EDGE_PRESSURE)
    for claim_id, pressure_points in CLAIM_EDGE_PRESSURE.items():
        assert len(pressure_points) >= 2, claim_id
        assert all(point.strip() for point in pressure_points), claim_id


def test_measurement_probability_claim_rejects_invalid_normalization_edges():
    """Protects sum_i p_i = 1 from zero or non-finite normalization inputs."""
    field = UnifiedField()

    valid = field.compute_measurement_probabilities([1.0, 1.0j, 0.0])
    assert np.sum(valid) == pytest.approx(1.0)
    assert np.all(np.isfinite(valid))

    formula_valid = formulas.normalized_probabilities([1.0, 1.0j, 0.0])
    assert np.sum(formula_valid) == pytest.approx(1.0)
    assert np.all(np.isfinite(formula_valid))

    for invalid in ([], [0.0, 0.0, 0.0], [1.0, np.nan]):
        with pytest.raises(ValidationError):
            field.compute_measurement_probabilities(invalid)
        with pytest.raises(ValidationError):
            formulas.normalized_probabilities(invalid)


def test_scale_and_flow_claims_remain_finite_at_valid_numeric_edges():
    """Exercises scale, flow, dark-sector, and wave-spectrum edge behavior."""
    field = UnifiedField(alpha=ALPHA_VAL)

    assert field.compute_entropy(0.0).value == pytest.approx(0.0)
    assert np.isfinite(field.compute_entropy(1e10).value)
    assert field.compute_entropy(-2.0).value == pytest.approx(field.compute_entropy(2.0).value)

    energies = np.array([0.0, 91.1876, GUT_SCALE, 1e30])
    running = field.compute_running_coupling(energies)
    assert np.all(np.isfinite(running))
    assert np.all(running > 0.0)
    assert running[0] == pytest.approx(running[1])

    gut_spread = formulas.coupling_spread(
        [value.value for value in field.compute_couplings(GUT_SCALE).values()]
    )
    low_spread = formulas.coupling_spread(
        [value.value for value in field.compute_couplings(91.1876).values()]
    )
    assert gut_spread < low_spread

    assert np.isfinite(field.compute_dark_matter_density(0.0).value)
    assert field.compute_dark_matter_density(-2.0).value == pytest.approx(
        field.compute_dark_matter_density(2.0).value
    )
    assert field.compute_dark_matter_density(1.0).value > field.compute_dark_matter_density(2.0).value

    spectrum = field.compute_gravitational_wave_spectrum(np.array([0.0, 1e-4, 1e-2]))
    assert np.all(np.isfinite(spectrum))
    assert np.all(spectrum >= 0.0)
    assert spectrum[2] > spectrum[1] > spectrum[0]


def test_recursive_suppression_and_information_edges_are_bounded():
    """Checks level-zero and positive-level boundaries for recursive claims."""
    field = UnifiedField(alpha=ALPHA_VAL)

    assert field.compute_newton_constant(0).value == pytest.approx(G)
    assert field.compute_uv_cutoff(0).value == pytest.approx(M_P)
    assert field.compute_newton_constant(3).value < field.compute_newton_constant(0).value
    assert field.compute_uv_cutoff(3).value < field.compute_uv_cutoff(0).value

    assert field.compute_information_content(0).value == pytest.approx(0.0)
    assert field.compute_information_content(10).value > 0.0
    assert field.compute_framework_dimension(0).value == pytest.approx(4.0)
    assert field.compute_framework_dimension(10).value > 4.0

    sakharov = field.check_sakharov_conditions()
    assert set(sakharov) == {
        "baryon_number_violation",
        "c_and_cp_violation",
        "out_of_equilibrium",
    }
    assert all(isinstance(value, bool) for value in sakharov.values())


def test_generative_operator_rejects_degenerate_edge_inputs():
    """Attacks W[F] -> F' with invalid fields and boundary sequence length."""
    seed = np.array([1.0, 0.25, -0.5, 0.125], dtype=complex)

    sequence = replication_sequence(seed, steps=0, alpha=ALPHA_VAL)
    assert len(sequence) == 1
    assert np.array_equal(sequence[0], seed)

    for invalid in (
        [1.0, 2.0],
        [[1.0, 2.0, 3.0]],
        [1.0, np.nan, 2.0],
        [0.0, 0.0, 0.0],
    ):
        with pytest.raises(ValidationError):
            generative_operator(invalid, alpha=ALPHA_VAL)

    with pytest.raises(ValidationError):
        self_similarity([1.0, 2.0, 3.0], [1.0, 2.0, 3.0, 4.0])
    with pytest.raises(ValidationError):
        replication_metrics([0.0, 0.0, 0.0], [1.0, 0.0, 0.0])
