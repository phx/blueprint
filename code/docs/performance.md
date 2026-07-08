# Performance Notes

Performance is secondary to traceability in the current repository. The active
goal is to keep paper claims, formula ledgers, Python implementations, and tests
in agreement.

## Current Benchmarks

The test suite includes benchmark-backed checks for:

- field evolution,
- phase-space integration, and
- memory usage.

Run:

```bash
cd code
pytest tests/test_performance.py
```

The full validation command also emits benchmark summaries:

```bash
cd code
python validate_all.py
```

## Practical Guidance

- Prefer clear executable checks over premature optimization.
- Keep new formula implementations deterministic and bounded.
- Add benchmark coverage only when a new operation is expensive enough to
  affect normal validation.
- Treat numerical warnings as review targets even when tests pass.

The current full suite passes with 100% active-core coverage. One integration
test can emit a SciPy subdivision warning; that warning is not a failure, but it
is a useful numerical hardening target.
