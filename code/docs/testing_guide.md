# Testing Guide

## Main Commands

```bash
cd code
python validate_all.py
./verify.sh --quick
./verify.sh --paper
./verify.sh --self-replication
```

## Current Gate

The active suite is expected to report:

```text
238 passed, 0 skipped
TOTAL 1818 statements, 0 missed, 100%
```

The coverage gate is configured in `pytest.ini` with `--cov-fail-under=100`.

## Test Layers

- Formula registry: `tests/test_formula_registry.py`
- Paper validation matrix: `tests/test_paper_validation_matrix.py`
- Documentation traceability: `tests/test_documentation_traceability.py`
- Self-replication: `tests/test_self_replication.py`
- Core behavior and edge coverage: the remaining `tests/test_*.py` files

## Adding A Paper Claim

1. Add or update the formula in `paper/THE_DIVINE_BLUEPRINT.tex`.
2. Add a row to `data/formula_registry.csv` when the claim is formula-shaped.
3. Add a row to `data/validation_matrix.csv` when the claim is paper-facing.
4. Implement the smallest clear code path in `core/`.
5. Add a pytest assertion that fails when the paper/code relationship breaks.
6. Run `python validate_all.py`.

## Interpretation

The tests make claims inspectable and falsifiable at the repository level. They
validate internal agreement. Empirical confirmation remains external work.
