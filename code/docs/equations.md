# Equation Traceability

The primary equation source is `paper/THE_DIVINE_BLUEPRINT.tex`. The executable
equation ledger is `data/formula_registry.csv`.

## Core Relations

```text
F(x, lambda t, lambda E) = lambda^D F(x, t, E)
S <= A / (4 l_P^2)
g_i(lambda E) = g_i(E) + sum alpha^n F_n^i(lambda)
F = T[F]
W[F] -> F'
F_{n+1} = W[F_n]
```

## Test Coverage

Every registered formula row must provide:

- a stable formula identifier,
- a paper anchor,
- executable code references,
- at least one pytest reference,
- a validation type, and
- a status.

Run:

```bash
cd code
pytest --no-cov tests/test_formula_registry.py
```

This confirms that the registry, formula helpers, paper anchors, and tests stay
aligned. It validates internal consistency only; external empirical confirmation
requires independent derivations and measurements.
