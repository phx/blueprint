# Validation Framework

The validation framework checks whether The Divine Blueprint is internally
coherent as a paper-code artifact.

## What Is Validated

- Registered formulas have code references and pytest coverage.
- Paper-facing claims in `data/validation_matrix.csv` have executable checks.
- The README, paper, registry, and matrix preserve the internal/external proof
  boundary.
- `W[F] -> F'` self-replication preserves norm, produces a non-identical
  successor, remains bounded across finite sequences, and records parent-child
  metrics.
- Core numerical helpers, value types, physics proxies, and edge cases maintain
  100% active-core coverage.

## What Is Not Validated

Passing tests do not prove that the external universe is described by this
framework. Several paper-facing claims are currently represented as bounded
proxies or reference-value proxies. Replacing those proxies with independent
derivations and empirical comparisons is the main scientific work ahead.

## Commands

```bash
cd code
python validate_all.py
./verify.sh --paper
./verify.sh --self-replication
```

Expected active result:

```text
238 passed, 0 skipped
TOTAL 1818 statements, 0 missed, 100%
```

## Adversarial Review Targets

- `data/formula_registry.csv`
- `data/validation_matrix.csv`
- `paper/THE_DIVINE_BLUEPRINT.tex`
- `core/formulas.py`
- `core/field.py`
- `core/self_replication.py`
- `tests/test_formula_registry.py`
- `tests/test_paper_validation_matrix.py`
- `tests/test_self_replication.py`
