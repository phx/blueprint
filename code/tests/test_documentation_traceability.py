"""Traceability tests linking the paper, README, and validation matrix."""

import ast
import csv
import re
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MATRIX_PATH = PROJECT_ROOT / "data" / "validation_matrix.csv"
PAPER_PATH = PROJECT_ROOT / "paper" / "THE_DIVINE_BLUEPRINT.tex"
README_PATH = PROJECT_ROOT / "README.md"
CODE_PATH = PROJECT_ROOT / "code"

REQUIRED_COLUMNS = {
    "claim_id",
    "paper_section",
    "paper_anchor",
    "claim",
    "equation_or_value",
    "implementation",
    "test_file",
    "test_function",
    "evidence_type",
    "status",
    "external_validation_note",
}

VALID_EVIDENCE_TYPES = {
    "internal_consistency",
    "bounded_proxy",
    "documentation_integrity",
}

VALID_STATUSES = {
    "internally_validated",
    "external_target",
}


def _validation_rows():
    with MATRIX_PATH.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return rows


def _test_functions(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return {
        node.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name.startswith("test_")
    }


def test_validation_matrix_rows_are_complete_and_reference_existing_tests():
    rows = _validation_rows()
    assert len(rows) >= 20
    assert set(rows[0]) == REQUIRED_COLUMNS
    assert len({row["claim_id"] for row in rows}) == len(rows)

    function_cache: dict[Path, set[str]] = {}
    for row in rows:
        for column in REQUIRED_COLUMNS:
            assert row[column].strip(), f"{row['claim_id']} missing {column}"

        test_path = PROJECT_ROOT / row["test_file"]
        assert test_path.exists(), row["test_file"]
        if test_path not in function_cache:
            function_cache[test_path] = _test_functions(test_path)
        assert row["test_function"] in function_cache[test_path]
        assert row["evidence_type"] in VALID_EVIDENCE_TYPES
        assert row["status"] in VALID_STATUSES


def test_validation_matrix_anchors_are_present_in_research_paper():
    paper = PAPER_PATH.read_text(encoding="utf-8")
    missing = [
        row["paper_anchor"]
        for row in _validation_rows()
        if row["paper_anchor"] not in paper
    ]
    assert missing == []


def test_core_framework_equations_are_documented_in_paper_and_readme():
    paper = PAPER_PATH.read_text(encoding="utf-8")
    readme = README_PATH.read_text(encoding="utf-8")
    equations = [
        "F(x, lambda t, lambda E) = lambda^D F(x, t, E)",
        "S <= A / (4 l_P^2)",
        "g_i(lambda E) = g_i(E) + sum alpha^n F_n^i(lambda)",
        "F = T[F]",
        "W[F] -> F'",
    ]

    for equation in equations:
        assert equation in paper
        assert equation in readme


def test_paper_and_readme_state_validation_boundary_and_source_of_truth():
    paper = PAPER_PATH.read_text(encoding="utf-8").lower()
    readme = README_PATH.read_text(encoding="utf-8").lower()

    for document in (paper, readme):
        assert "data/validation_matrix.csv" in document
        assert "not empirical proof" in document or "do not prove" in document
        assert "100%" in document
        assert "external" in document


def test_public_docs_do_not_reference_private_source_material():
    prohibited_terms = [
        "sup" + "plementary",
        "phenomen" + "ological",
        "exper" + "iential",
        "channel" + "ed",
        "the " + "word",
        "i " + "am",
        "ma" + "ce",
        "inter" + "dimensional",
        "hack" + "athon",
        "be" + "ings",
        "unusual " + "communication",
    ]
    public_docs = {
        "paper": PAPER_PATH.read_text(encoding="utf-8").lower(),
        "readme": README_PATH.read_text(encoding="utf-8").lower(),
    }

    for name, text in public_docs.items():
        for term in prohibited_terms:
            assert term not in text, f"{term!r} found in {name}"


def test_python_docstrings_and_comments_use_current_public_traceability():
    prohibited_patterns = [
        r"appendix_[A-Za-z0-9_]+(?:\.tex)?",
        r"\b[Ff]ractal [Ff]ield [Tt]heory\b",
        r"fractal-field-theory",
        r"framework framework",
        r"validation framework validation framework",
        r"\bthe The Divine Blueprint\b",
        r"current public paper Section [A-Z]",
        r"current public paper validation proxy",
        r"paper Sec\. [0-9]",
        r"\bEq [A-Z]\.",
    ]

    offenders: list[str] = []
    for path in sorted(CODE_PATH.rglob("*.py")):
        if path == Path(__file__).resolve():
            continue
        text = path.read_text(encoding="utf-8")
        for pattern in prohibited_patterns:
            if re.search(pattern, text):
                offenders.append(f"{path.relative_to(PROJECT_ROOT)}: {pattern}")

    assert offenders == []
