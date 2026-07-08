# User Guide

## Quick Start

```bash
cd /path/to/blueprint
source .venv/bin/activate
cd code
python validate_all.py
```

## Running Focused Validation

```bash
./verify.sh --paper
./verify.sh --self-replication
pytest --no-cov tests/test_formula_registry.py
pytest --no-cov tests/test_paper_validation_matrix.py
```

## Using The Core API

```python
import numpy as np

from core.field import UnifiedField
from core.self_replication import replicate_field, replication_sequence
from core.types import Energy

field = UnifiedField()

gut = Energy(2.1e16)
couplings = field.compute_couplings(gut)
entropy = field.compute_entropy(2.0)

seed = np.array([1.0, 0.5, -0.25, 0.125], dtype=complex)
result = replicate_field(seed, novelty=0.1, embodiment=1.0)
sequence = replication_sequence(seed, steps=3)

print(couplings)
print(entropy)
print(result.metrics)
print(len(sequence))
```

## Interpretation

The code is a validation harness for the paper's executable claims. It is not a
production physics library and does not claim external empirical proof. Use it
to inspect formulas, run tests, add missing checks, and propose sharper
falsification targets.
