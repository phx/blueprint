"""Tests for the grade-level public explanation folder."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ELI5_DIR = PROJECT_ROOT / "ELI5_grade_school"

GRADE_FILES = ["GRADE_0-K.md"] + [
    f"GRADE_{grade:02d}.md" for grade in range(1, 13)
]


def test_grade_school_eli5_folder_has_expected_files():
    assert ELI5_DIR.exists()
    assert (ELI5_DIR / "README.md").exists()
    for filename in GRADE_FILES:
        assert (ELI5_DIR / filename).exists(), filename


def test_each_grade_file_preserves_the_validation_boundary():
    for filename in GRADE_FILES:
        text = (ELI5_DIR / filename).read_text(encoding="utf-8").lower()
        assert "not proven yet" in text, filename
        assert "universe" in text or "reality" in text, filename
        assert "pattern" in text, filename
        assert any(term in text for term in ("check", "test", "code")), filename


def test_grade_school_explainers_avoid_repository_shorthand():
    for filename in GRADE_FILES + ["README.md"]:
        text = (ELI5_DIR / filename).read_text(encoding="utf-8").lower()
        assert "repo" not in text, filename
        assert "repository" not in text, filename


def test_lower_grade_files_avoid_unintroduced_audit_jargon():
    lower_grade_files = ["GRADE_0-K.md"] + [
        f"GRADE_{grade:02d}.md" for grade in range(1, 7)
    ]
    banned_terms = (
        "validation matrix",
        "formula registry",
        "empirical-anchor",
        "ledger",
        "pytest",
        "falsification",
        "proxy",
    )
    for filename in lower_grade_files:
        text = (ELI5_DIR / filename).read_text(encoding="utf-8").lower()
        for term in banned_terms:
            assert term not in text, f"{filename}: {term}"


def test_later_grade_files_name_the_core_equation_spine():
    for filename in [f"GRADE_{grade:02d}.md" for grade in range(7, 13)]:
        text = (ELI5_DIR / filename).read_text(encoding="utf-8")
        assert "F = T[F]" in text, filename
        assert "W[F] -> F'" in text, filename


def test_public_eli5_index_uses_what_if_framing():
    text = (ELI5_DIR / "README.md").read_text(encoding="utf-8").lower()
    assert "if the framework survives outside testing" in text
    assert "it is not proven yet" in text
    assert "bounded, self-checking, self-updating information pattern" in text
