"""Tests for empirical-anchor boundaries and dependency compatibility."""

import ast
import csv
import importlib
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ANCHOR_PATH = PROJECT_ROOT / "data" / "empirical_anchors.csv"
FORMULA_REGISTRY_PATH = PROJECT_ROOT / "data" / "formula_registry.csv"
README_PATH = PROJECT_ROOT / "README.md"
PAPER_PATH = PROJECT_ROOT / "paper" / "THE_DIVINE_BLUEPRINT.tex"
CORE_PATH = PROJECT_ROOT / "code" / "core"

REQUIRED_COLUMNS = {
    "anchor_id",
    "public_value",
    "implementation",
    "formula_registry_id",
    "role",
    "current_status",
    "not_a_claim_of",
    "upgrade_path",
}

REQUIRED_ANCHORS = {
    "gut_scale",
    "gut_coupling",
    "proton_lifetime_scale",
    "proton_decay_rate",
    "fermion_mass_ordering",
    "higgs_vev",
    "higgs_mass",
    "neutrino_oscillation_probability",
    "jarlskog_invariant",
    "baryon_asymmetry",
    "dark_energy_density_scale",
    "low_energy_signatures",
}

EXPLICITLY_ANCHORED_FORMULAS = {
    "gut_predictions",
    "fermion_ordering",
    "higgs_neutrino_values",
    "jarlskog_baryon",
    "dark_energy_density",
    "low_energy_values",
}


def _csv_rows(path: Path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


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


def test_empirical_anchor_ledger_marks_reference_values_as_not_derived_predictions():
    rows = _csv_rows(ANCHOR_PATH)
    formula_ids = {row["formula_id"] for row in _csv_rows(FORMULA_REGISTRY_PATH)}

    assert set(rows[0]) == REQUIRED_COLUMNS
    assert {row["anchor_id"] for row in rows} == REQUIRED_ANCHORS

    for row in rows:
        for column in REQUIRED_COLUMNS:
            assert row[column].strip(), f"{row['anchor_id']} missing {column}"
        assert row["formula_registry_id"] in formula_ids
        not_a_claim = row["not_a_claim_of"].lower()
        assert "deriv" in not_a_claim or "empirical" in not_a_claim
        assert any(word in row["upgrade_path"].lower() for word in ("derive", "replace", "compare"))
        for implementation in row["implementation"].split("; "):
            assert callable(_resolve_implementation(implementation)), implementation


def test_reference_value_proxy_formulas_are_covered_by_empirical_anchors():
    formula_rows = _csv_rows(FORMULA_REGISTRY_PATH)
    anchor_rows = _csv_rows(ANCHOR_PATH)

    reference_formula_ids = {
        row["formula_id"]
        for row in formula_rows
        if row["validation_scope"] == "reference_value_proxy"
    }
    anchored_formula_ids = {row["formula_registry_id"] for row in anchor_rows}

    assert reference_formula_ids <= anchored_formula_ids
    assert EXPLICITLY_ANCHORED_FORMULAS <= anchored_formula_ids


def test_public_docs_distinguish_reference_anchors_from_discoveries():
    readme = README_PATH.read_text(encoding="utf-8").lower()
    paper = PAPER_PATH.read_text(encoding="utf-8").lower()

    for document in (readme, paper):
        normalized = " ".join(document.split())
        assert "data/empirical_anchors.csv" in normalized
        assert "reference anchors are not derived predictions" in normalized
        assert "not a physical discovery" in normalized

    assert "replacing a reference anchor with a derivation" in readme
    assert "a future derivation" in readme


def test_core_docstrings_label_public_reference_anchors():
    field_tree = ast.parse((CORE_PATH / "field.py").read_text(encoding="utf-8"))
    formula_tree = ast.parse((CORE_PATH / "formulas.py").read_text(encoding="utf-8"))

    required_field_docstrings = {
        "_compute_theoretical_predictions_stable",
        "_proton_decay_rate",
        "_oscillation_probability",
        "_jarlskog",
        "_baryon_asymmetry",
        "_higgs_vev",
        "_higgs_mass",
        "_fermion_masses",
        "_dark_energy_density",
        "_low_energy_signatures",
    }
    required_formula_docstrings = {
        "gut_prediction_values",
        "jarlskog_baryon_values",
        "dark_energy_density",
        "low_energy_signature_values",
    }

    def function_docstrings(tree):
        return {
            node.name: ast.get_docstring(node) or ""
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef)
        }

    field_docstrings = function_docstrings(field_tree)
    formula_docstrings = function_docstrings(formula_tree)

    for name in required_field_docstrings:
        assert "anchor" in field_docstrings[name].lower() or "proxy" in field_docstrings[name].lower()
    for name in required_formula_docstrings:
        assert "anchor" in formula_docstrings[name].lower()


def test_numpy_trapezoid_selection_is_numpy_two_compatible():
    eager_fallback = 'getattr(np, "trapezoid", np.trapz)'
    offenders = []
    for path in sorted(CORE_PATH.glob("*.py")):
        if eager_fallback in path.read_text(encoding="utf-8"):
            offenders.append(path.relative_to(PROJECT_ROOT).as_posix())

    assert offenders == []
