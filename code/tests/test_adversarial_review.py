"""Tests for the public adversarial-review and term-definition ledgers."""

import csv
from pathlib import Path

import numpy as np
import pytest

from core.errors import ValidationError
from core.self_replication import generative_operator


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ADVERSARIAL_REVIEW_PATH = PROJECT_ROOT / "data" / "adversarial_review.csv"
TERM_REGISTRY_PATH = PROJECT_ROOT / "data" / "term_registry.csv"
VALIDATION_MATRIX_PATH = PROJECT_ROOT / "data" / "validation_matrix.csv"
FORMULA_REGISTRY_PATH = PROJECT_ROOT / "data" / "formula_registry.csv"
README_PATH = PROJECT_ROOT / "README.md"

REQUIRED_ATTACKS = {
    "undefined_terms",
    "unconstraining_formula",
    "assumption_only_test",
    "nonconvertible_claim",
    "w_notation_theater",
    "semantic_drift",
}

REQUIRED_TERMS = {
    "field_state",
    "recursive_operator",
    "generative_operator",
    "successor_state",
    "alpha",
    "eta",
    "mu",
    "phi",
    "similarity",
    "residual",
    "bounded_proxy",
    "internal_consistency",
    "external_validation",
    "validation_matrix",
    "formula_registry",
    "claim_triage",
    "adversarial_review",
    "semantic_drift",
}

VALID_ATTACK_STATUSES = {
    "mitigated",
    "mitigated_with_open_external_work",
}

VALID_FORMULA_SCOPES = {
    "internal_formula_proxy",
    "bounded_proxy",
    "finite_basis_proxy",
    "reference_value_proxy",
    "documentation_integrity",
}


def _csv_rows(path: Path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _assert_reference_path_exists(reference: str) -> None:
    path_text = reference.split(":", maxsplit=1)[0].strip()
    assert (PROJECT_ROOT / path_text).exists(), reference


def test_recursive_intelligence_attack_questions_are_registered():
    rows = _csv_rows(ADVERSARIAL_REVIEW_PATH)

    assert {row["attack_id"] for row in rows} == REQUIRED_ATTACKS
    for row in rows:
        assert row["attack_question"].strip()
        assert row["risk_if_true"].strip()
        assert row["current_mitigation"].strip()
        assert row["status"] in VALID_ATTACK_STATUSES
        assert row["next_failure_target"].strip()
        for reference in row["evidence_path"].split("; "):
            _assert_reference_path_exists(reference)


def test_core_terms_are_defined_before_they_can_drift():
    rows = _csv_rows(TERM_REGISTRY_PATH)

    assert {row["term_id"] for row in rows} == REQUIRED_TERMS
    for row in rows:
        assert row["term"].strip()
        assert len(row["definition"].split()) >= 8, row["term_id"]
        assert row["scope"].strip()
        _assert_reference_path_exists(row["primary_public_anchor"])

    readme = README_PATH.read_text(encoding="utf-8")
    assert "data/term_registry.csv" in readme
    assert "data/adversarial_review.csv" in readme


def test_formula_registry_rows_name_constraint_scopes_and_tests():
    rows = _csv_rows(FORMULA_REGISTRY_PATH)

    for row in rows:
        assert row["validation_scope"] in VALID_FORMULA_SCOPES
        assert row["formula_plaintext"].strip()
        assert row["implementation"].strip()
        assert row["test_file"].strip()
        assert row["test_function"].startswith("test_")
        _assert_reference_path_exists(row["test_file"])


def test_bounded_proxies_name_their_external_failure_boundary():
    rows = _csv_rows(VALIDATION_MATRIX_PATH)
    boundary_words = {
        "derivation",
        "external",
        "measurement",
        "comparison",
        "constraint",
        "target",
        "fit",
        "detection",
        "dynamical",
    }

    for row in rows:
        note = row["external_validation_note"].lower()
        if row["evidence_type"] == "bounded_proxy":
            assert any(word in note for word in boundary_words), row["claim_id"]


def test_w_operator_rejects_notation_theater_clone_cases():
    structured = np.array([1.0, 0.25, -0.5, 0.125], dtype=complex)
    successor = generative_operator(structured, alpha=0.1, novelty=0.2)

    assert np.linalg.norm(successor) == pytest.approx(np.linalg.norm(structured))
    assert 0.0 < abs(np.vdot(structured, successor)) / (
        np.linalg.norm(structured) * np.linalg.norm(successor)
    ) < 1.0

    with pytest.raises(ValidationError):
        generative_operator(np.ones(8, dtype=complex), alpha=0.1, novelty=0.0, phase=0.0)
    with pytest.raises(ValidationError):
        generative_operator(np.ones(8, dtype=complex), alpha=0.1, novelty=0.2)
