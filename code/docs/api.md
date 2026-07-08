# API Notes

The code in this repository is a validation harness for The Divine Blueprint,
not a general-purpose physics package. The stable public surface is the set of
modules exercised by the test suite, formula registry, and validation matrix.

## Core Modules

- `core.field.UnifiedField`: facade for executable paper-facing quantities.
- `core.formulas`: direct formula helpers used by registry tests.
- `core.self_replication`: implementation of the generative operator
  `W[F] -> F'`.
- `core.types`: numerical and physics value containers with validation.
- `core.compute`, `core.numeric`, `core.validation`: supporting numerical
  checks and integration helpers.

## Minimal Example

```python
import numpy as np

from core.field import UnifiedField
from core.self_replication import replicate_field
from core.types import Energy

field = UnifiedField()

couplings = field.compute_couplings(Energy(2.1e16))
entropy = field.compute_entropy(2.0)

seed = np.array([1.0, 0.5, -0.25, 0.125], dtype=complex)
replication = replicate_field(seed, novelty=0.1, embodiment=1.0)

print(couplings)
print(entropy)
print(replication.metrics)
```

## Source Of Truth

- `data/formula_registry.csv` maps formulas to code and tests.
- `data/validation_matrix.csv` maps paper claims to executable checks.
- `tests/test_formula_registry.py` verifies formula-code-test traceability.
- `tests/test_paper_validation_matrix.py` verifies paper-facing claims.
- `tests/test_self_replication.py` verifies `W[F] -> F'`.

Use those files before relying on any individual helper as a standalone API.
