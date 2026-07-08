"""Paper-to-code validation checks for the revised Divine Blueprint.

These tests are intentionally equation-facing: each assertion corresponds to a
claim, prediction, or formal operator that appears in the research paper. They
do not establish empirical truth; they verify that the executable model and the
paper state the same internally testable structure.
"""

import csv
from pathlib import Path

import numpy as np
import pytest

from core.basis import FractalBasis
from core.constants import GUT_SCALE
from core.field import UnifiedField
from core.physics_constants import ALPHA_VAL, HBAR, G
from core.self_replication import (
    generative_operator,
    replicate_field,
    replication_sequence,
    sequence_metrics,
    self_similarity,
)
from core.types import Energy, FieldConfig, NumericValue
from core.utils import propagate_errors


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = PROJECT_ROOT / "data"


def test_fractal_scaling_and_fixed_point_recursion_are_bounded():
    """Validates F(lambda) scaling proxy, F = T[F], and alpha recursion."""
    field = UnifiedField(alpha=ALPHA_VAL, max_level=12)
    basis = FractalBasis(alpha=ALPHA_VAL, max_level=8, dimension=4)

    assert basis.scaling_dimension == pytest.approx(1.0)
    recursion = np.array([field.compute_fractal_recursion(n) for n in range(8)])
    assert recursion[0] == pytest.approx(1.0)
    assert np.allclose(recursion[1:] / recursion[:-1], ALPHA_VAL)
    assert recursion[-1] < recursion[1]


def test_holographic_entropy_scales_with_boundary_area():
    """Validates the paper's area-law entropy proxy S proportional to A."""
    field = UnifiedField()
    r1, r2 = 1.0, 2.0
    s1 = field.compute_entropy(r1).value
    s2 = field.compute_entropy(r2).value

    assert s1 == pytest.approx(4 * np.pi * r1**2 / (8 * HBAR * G))
    assert s2 / s1 == pytest.approx((r2 / r1) ** 2)


def test_gut_predictions_match_paper_prediction_table():
    """Validates M_GUT, alpha_GUT, and the internal lifetime-scale proxy."""
    field = UnifiedField()
    predictions = field._compute_theoretical_predictions()

    assert GUT_SCALE == pytest.approx(2.1e16)
    assert predictions["M_GUT"].value == pytest.approx(2.1e16)
    assert predictions["M_GUT"].uncertainty == pytest.approx(0.3e16)
    assert predictions["alpha_GUT"].value == pytest.approx(0.0376)
    assert predictions["alpha_GUT"].uncertainty == pytest.approx(0.0002)
    assert predictions["tau_p"].value == pytest.approx(1.0e34)

    with (DATA_ROOT / "predictions.csv").open() as handle:
        rows = {row["observable"]: row for row in csv.DictReader(handle)}

    for observable, expected in (
        ("M_GUT", predictions["M_GUT"]),
        ("alpha_GUT", predictions["alpha_GUT"]),
        ("tau_p", predictions["tau_p"]),
    ):
        row = rows[observable]
        assert float(row["predicted"]) == pytest.approx(expected.value)
        assert abs(float(row["reference"]) - expected.value) <= expected.uncertainty


def test_coupling_convergence_and_rg_flow_are_consistent():
    """Validates coupling convergence, RG flow, and beta-function proxy."""
    field = UnifiedField()
    gut = Energy(GUT_SCALE)
    couplings = field.compute_couplings(gut)

    values = np.array([couplings[name].value for name in ("g1", "g2", "g3")])
    assert np.max(values) - np.min(values) < 1e-3
    assert field.compute_beta_function(1, gut.value) == pytest.approx(
        sum(ALPHA_VAL**n * c for n, c in enumerate(field.compute_beta_coefficients(1, n_max=3)))
    )

    energies = np.array([91.1876, 1e3, 1e6])
    running = field.compute_running_coupling(energies)
    assert np.all(np.isfinite(running))
    assert np.all(running > 0)


def test_standard_model_signature_predictions_are_reflected_in_code():
    """Validates CP violation, baryon asymmetry, neutrino, and Higgs values."""
    field = UnifiedField()
    config = FieldConfig()

    assert field.compute_jarlskog() == pytest.approx(3.2e-5)
    assert field.compute_baryon_asymmetry() == pytest.approx(6.1e-10)
    assert field.compute_higgs_vev(config) == pytest.approx(246.0)
    assert field.compute_higgs_mass(config) == pytest.approx(125.1)

    neutrino_masses = field.compute_neutrino_masses(config)
    assert np.all(np.diff(neutrino_masses) > 0)
    assert field.compute_oscillation_probability("mu", "e", L=295, E=0.6).value == pytest.approx(0.0597)

    fermions = field.compute_fermion_masses(config)
    assert fermions["top"].value > fermions["charm"].value > fermions["up"].value
    assert fermions["tau"].value > fermions["muon"].value > fermions["electron"].value


def test_dark_sector_measurement_and_gravity_claims_have_executable_proxies():
    """Validates dark sector, measurement, gravity, and hierarchy claims."""
    field = UnifiedField()

    rho_inner = field.compute_dark_matter_density(1.0)
    rho_outer = field.compute_dark_matter_density(2.0)
    assert rho_inner.value > rho_outer.value
    assert field.compute_dark_matter_coupling(3, 1.0) == pytest.approx(ALPHA_VAL**3)

    assert field.compute_dark_energy_density().value == pytest.approx((2.3e-3) ** 4)
    assert field.compute_cosmological_constant(10.0).value > 1.0

    probabilities = field.compute_measurement_probabilities([1.0, 1.0j, 0.0])
    assert np.sum(probabilities) == pytest.approx(1.0)
    assert np.all(probabilities >= 0.0)
    assert field.compute_decoherence_trace(np.eye(2) / 2).value == pytest.approx(1.0)
    assert field.compute_collapse_time(1.0).value < 1.0

    assert field.compute_newton_constant(2).value < G
    assert field.compute_uv_cutoff(2).value < field.compute_uv_cutoff(0).value
    assert field.compute_information_content(10).value > 0.0
    assert field.compute_framework_dimension(10).value > 4.0
    assert field.compute_boundary_fractal_dimension().value == pytest.approx(2.0)


def test_proton_decay_sakharov_and_low_energy_signatures_match_paper_values():
    """Validates paper predictions for proton decay and low-energy tests."""
    field = UnifiedField()
    proton = field.compute_proton_decay_rate()
    assert proton.value == pytest.approx(1.6e-36)
    assert proton.uncertainty == pytest.approx(0.3e-36)

    sakharov = field.check_sakharov_conditions()
    assert sakharov == {
        "baryon_number_violation": True,
        "c_and_cp_violation": True,
        "out_of_equilibrium": True,
    }

    signatures = field.compute_low_energy_signatures()
    assert signatures["delta_r_w"].value == pytest.approx(37.979e-3)
    assert signatures["delta_r_w"].uncertainty == pytest.approx(0.084e-3)
    assert signatures["bs_to_mumu"].value == pytest.approx(3.09e-9)
    assert signatures["bs_to_mumu"].uncertainty == pytest.approx(0.19e-9)
    assert signatures["sin2_theta13"].value == pytest.approx(0.0218)
    assert signatures["sin2_theta13"].uncertainty == pytest.approx(0.0007)


def test_gravitational_wave_spectrum_uses_paper_power_law():
    """Validates Omega_GW(f) proportional to f^(2/3) in the executable proxy."""
    field = UnifiedField()
    f1, f2 = 1e-4, 1e-2
    spectrum = field.compute_gravitational_wave_spectrum(np.array([f1, f2]))
    assert spectrum[1] / spectrum[0] == pytest.approx((f2 / f1) ** (2.0 / 3.0))


def test_generative_operator_validates_self_replication():
    """Validates the new W[F] -> F' operator as self-similar non-cloning."""
    parent = np.array([1.0, 0.25, -0.5, 0.125, 0.0], dtype=complex)
    child = generative_operator(parent, alpha=ALPHA_VAL, novelty=0.2, embodiment=1.0)
    metrics = replicate_field(parent, alpha=ALPHA_VAL, novelty=0.2).metrics

    assert np.linalg.norm(child) == pytest.approx(np.linalg.norm(parent))
    assert 0.0 < self_similarity(parent, child) < 1.0
    assert metrics.child_norm == pytest.approx(metrics.parent_norm)
    assert metrics.residual > 0.0
    assert metrics.novelty > 0.0

    sequence = replication_sequence(parent, steps=4, alpha=ALPHA_VAL, novelty=0.2)
    pairwise = sequence_metrics(sequence)
    assert len(sequence) == 5
    assert all(metric.child_norm == pytest.approx(metric.parent_norm) for metric in pairwise)
    assert all(0.0 <= metric.novelty <= 1.0 for metric in pairwise)


def test_uncertainty_model_matches_paper_quadrature_rule():
    """Validates the paper's combined-uncertainty quadrature equation."""
    result = propagate_errors(
        values=[1.0, 2.0, 3.0],
        uncertainties=[0.1, 0.2, 0.3],
    )
    assert result.value == pytest.approx(6.0)
    assert result.uncertainty == pytest.approx(np.sqrt(0.1**2 + 0.2**2 + 0.3**2))

    correlated = propagate_errors(
        values=[1.0, 2.0],
        uncertainties=[0.1, 0.2],
        correlations=np.array([[1.0, 0.5], [0.5, 1.0]]),
    )
    assert correlated.uncertainty > result.uncertainty / 2
