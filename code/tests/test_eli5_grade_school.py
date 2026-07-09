"""Tests for the grade-level public explanation folder."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ELI5_DIR = PROJECT_ROOT / "ELI5_grade_school"

GRADE_FILES = ["GRADE_K.md"] + [f"GRADE_{grade:02d}.md" for grade in range(1, 13)]


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
        assert "test" in text or "code" in text, filename


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
