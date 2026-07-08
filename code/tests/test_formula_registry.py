"""Formula-level traceability between the paper and executable code."""

import ast
import csv
import importlib
from pathlib import Path

import numpy as np
import pytest

from core import formulas
from core.field import UnifiedField
from core.physics_constants import ALPHA_VAL, G, HBAR
from core.self_replication import generative_operator, replication_sequence
from core.types import Energy, FieldConfig


PROJECT_ROOT = Path(__file__).resolve().parents[2]
FORMULA_REGISTRY_PATH = PROJECT_ROOT / "data" / "formula_registry.csv"
PAPER_PATH = PROJECT_ROOT / "paper" / "THE_DIVINE_BLUEPRINT.tex"

REQUIRED_COLUMNS = {
    "formula_id",
    "paper_anchor",
    "formula_plaintext",
    "implementation",
    "test_file",
    "test_function",
    "validation_scope",
    "status",
}


def _formula_rows():
    with FORMULA_REGISTRY_PATH.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _test_functions(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return {
        node.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name.startswith("test_")
    }


def _resolve_implementation(path: str):
    parts = path.split(".")
    for index in range(len(parts), 0, -1):
        module_name = ".".join(parts[:index])
        try:
            target = importlib.import_module(module_name)
        except ModuleNotFoundError:
            continue
        for attribute in parts[index:]:
            target = getattr(target, attribute)
        return target
    raise ModuleNotFoundError(path)


def test_formula_registry_entries_have_code_and_tests():
    rows = _formula_rows()
    paper = PAPER_PATH.read_text(encoding="utf-8")

    assert len(rows) >= 45
    assert set(rows[0]) == REQUIRED_COLUMNS
    assert len({row["formula_id"] for row in rows}) == len(rows)

    function_cache: dict[Path, set[str]] = {}
    for row in rows:
        for column in REQUIRED_COLUMNS:
            assert row[column].strip(), f"{row['formula_id']} missing {column}"
        assert row["paper_anchor"] in paper
        assert row["status"] == "internally_validated"

        for implementation in row["implementation"].split("; "):
            assert callable(_resolve_implementation(implementation)), implementation

        test_path = PROJECT_ROOT / row["test_file"]
        assert test_path.exists(), row["test_file"]
        if test_path not in function_cache:
            function_cache[test_path] = _test_functions(test_path)
        assert row["test_function"] in function_cache[test_path]


def test_formula_module_matches_paper_mathematics():
    field = UnifiedField()

    assert formulas.architecture_terms() == (
        "scale symmetry",
        "holographic bound",
        "recursive coupling flow",
        "fixed-point recursion",
        "generative succession",
    )

    identity = np.eye(2)
    contraction = 0.5 * np.eye(2)
    hermitian = np.array([[1.0, 1.0j], [-1.0j, 2.0]])
    assert np.allclose(formulas.commutator(identity, identity), np.zeros((2, 2)))
    assert formulas.contraction_norm(contraction) <= 1.0
    assert formulas.is_hermitian(hermitian)

    assert formulas.fractal_self_similarity(3.0, 2.0, 4.0) == pytest.approx(48.0)
    assert formulas.holographic_entropy_bound(16.0, 2.0) == pytest.approx(1.0)
    assert formulas.recursive_coupling(1.0, 0.1, [2.0, 3.0]) == pytest.approx(1.23)
    assert formulas.recursive_inverse_coupling(10.0, 2.0, 200.0, 100.0, 0.1, [1.0]) == pytest.approx(
        10.0 + np.log(2.0) / np.pi + 0.1
    )

    parent = np.array([1.0, 0.25, -0.5, 0.125, 0.0], dtype=complex)
    child = generative_operator(parent, alpha=ALPHA_VAL, novelty=0.2)
    constraints = formulas.generative_constraints(parent, child)
    assert constraints["child_norm"] == pytest.approx(constraints["parent_norm"])
    assert 0.0 < constraints["similarity"] < 1.0
    assert constraints["residual"] > 0.0
    assert len(replication_sequence(parent, steps=3, alpha=ALPHA_VAL)) == 4

    psi_levels = np.array([[1.0, 1.0], [2.0, 2.0]], dtype=complex)
    lagrangian = np.array([0.0, 0.0])
    assert formulas.unified_field_expansion(psi_levels, lagrangian, 0.1, dx=0.5) == pytest.approx(1.2 + 0.0j)
    assert np.allclose(formulas.basis_closure(0.1, np.array([2.0, 3.0])), np.array([0.2, 0.3]))
    assert np.allclose(formulas.resolution_identity(np.eye(2)), np.eye(2))
    assert formulas.convergence_tail_bound(2.0, 0.1, 3) == pytest.approx(2.0e-3 / 0.9)

    assert formulas.degrees_of_freedom(10.0, 1.0, 0.1, 2) == pytest.approx(100.0 / (1.1 * 1.01))
    assert formulas.boundary_dimension() == pytest.approx(2.0)
    assert formulas.area_entropy(2.0) / formulas.area_entropy(1.0) == pytest.approx(4.0)
    assert formulas.coupling_spread([0.7, 0.7002, 0.7005]) < 1e-3

    gut = formulas.gut_prediction_values()
    assert gut["M_GUT"] == pytest.approx(2.1e16)
    assert gut["Gamma_p_to_e_pi"] == pytest.approx(field.compute_proton_decay_rate().value)

    assert formulas.mass_product(10.0, 0.1, [1.0, 2.0]) == pytest.approx(10.0 * 1.1 * 1.02)
    fermions = field.compute_fermion_masses(FieldConfig())
    assert fermions["top"].value > fermions["charm"].value > fermions["up"].value
    assert abs(formulas.cp_phase(0.5, 1, 4, 0.0)) == pytest.approx(0.5)

    cp_values = formulas.jarlskog_baryon_values()
    assert cp_values["J"] == pytest.approx(field.compute_jarlskog())
    assert cp_values["eta_B"] == pytest.approx(field.compute_baryon_asymmetry())
    assert all(formulas.sakharov_conditions().values())

    assert formulas.dark_matter_density(2.0, 0.1, [1.0]) < formulas.dark_matter_density(1.0, 0.1, [1.0])
    assert formulas.dark_matter_coupling(3, 1.0, 0.1) == pytest.approx(field.compute_dark_matter_coupling(3, 1.0))
    assert formulas.dark_energy_density() == pytest.approx(field.compute_dark_energy_density().value)

    rho0 = np.eye(2, dtype=complex) / 2.0
    updated = formulas.measurement_update(rho0, 0.1, [rho0])
    assert np.trace(updated).real == pytest.approx(1.1)
    probabilities = formulas.normalized_probabilities([1.0, 1.0j, 0.0])
    assert np.sum(probabilities) == pytest.approx(1.0)
    assert field.compute_decoherence_trace(rho0).value == pytest.approx(1.0)
    assert field.compute_collapse_time(1.0).value < 1.0

    assert formulas.gravitational_action_proxy(10.0, 0.1, [2.0, 3.0]) == pytest.approx(10.23)
    assert formulas.recursive_product(100.0, 0.1, 2) == pytest.approx(100.0 / (1.1 * 1.01))
    assert field.compute_newton_constant(2).value < G
    assert field.compute_uv_cutoff(2).value < field.compute_uv_cutoff(0).value

    assert formulas.information_content(0.1, 10) == pytest.approx(field.compute_information_content(10).value)
    assert formulas.framework_dimension(0.1, 10) == pytest.approx(field.compute_framework_dimension(10).value)

    assert formulas.gw_spectrum(1.0, 2.0, 0.5, 2.0, 0.1, [1.0]) == pytest.approx(8.8)
    assert formulas.gw_power_law_ratio(1e-2, 1e-4) == pytest.approx((1e-2 / 1e-4) ** (2.0 / 3.0))
    spectrum = field.compute_gravitational_wave_spectrum(np.array([1e-4, 1e-2]))
    assert spectrum[1] / spectrum[0] == pytest.approx(formulas.gw_power_law_ratio(1e-2, 1e-4))

    low_energy = formulas.low_energy_signature_values()
    signatures = field.compute_low_energy_signatures()
    assert low_energy["Delta_r_W"] == pytest.approx(signatures["delta_r_w"].value)
    assert formulas.uncertainty_quadrature([0.1, 0.2, 0.3]) == pytest.approx(np.sqrt(0.14))

    entropy = field.compute_entropy(1.0).value
    assert entropy == pytest.approx(4 * np.pi / (8 * HBAR * G))
    couplings = field.compute_couplings(Energy(2.1e16))
    assert formulas.coupling_spread([item.value for item in couplings.values()]) < 1e-3
