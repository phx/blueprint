# Testing Guide

## Test Organization

Tests are organized by functionality and marked with appropriate markers:

- `@pytest.mark.physics`: Physics-related tests
- `@pytest.mark.numeric`: Numerical computation tests
- `@pytest.mark.performance`: Performance and benchmarking tests
- `@pytest.mark.slow`: Long-running tests
- `@pytest.mark.integration`: Integration tests
- `@pytest.mark.unit`: Unit tests

## Validated Internal Proxies

The framework validates executable versions of the paper's claims as internal mathematical proxies:

1. **Coupling Unification**
   - Tests gauge coupling convergence
   - Achieved |g1-g2| < 0.001 at GUT scale

2. **Holographic Entropy**
   - Tests holographic entropy bounds
   - Achieved ratio < 0.1 as predicted

3. **Fractal Recursion**
   - Tests fractal scaling relations
   - Achieved diff < 1e-6 in recursion tests

4. **Dark Matter Profile**
   - Tests dark matter density proxies
   - Achieved 0.1 < ratio < 10.0 as expected

5. **Generative Self-Replication**
   - Tests `W[F] -> F'`
   - Enforces norm preservation, self-similarity, and non-cloning

These validations confirm internal consistency between the paper, data, and code. They do not by themselves prove external physical truth.

## Coverage Requirements

The active core must maintain the configured `100%` coverage gate. Coverage reports are generated in HTML format.

### Coverage Configuration
- Minimum coverage threshold: 100% for `core`
- HTML reports generated in htmlcov/
- Excluded patterns:
  - `__repr__` methods
  - Debug logging
  - Platform-specific code

### Coverage Reports
Coverage reports show:
- Line coverage
- Branch coverage
- Function coverage
- Module coverage

## Test Fixtures

Common test fixtures are defined in `tests/conftest.py`:

- `test_grid`: Standard spatial grid for testing
- `energy_points`: Standard energy points from Z mass to Planck mass
- `detector`: Configured detector instance
- `test_data_generator`: Function to generate test data
- `test_covariance`: Function to generate covariance matrices
- `numeric_precision`: Numerical precision requirements
- `test_config`: Basic test configuration
- `field_config`: Standard field configuration

### Using Fixtures
```python
def test_field_evolution(test_grid, energy_points):
    """Example using test fixtures."""
    # Test grid provides spatial points
    field = compute_field(test_grid)
    
    # Energy points for evolution
    evolved = field.evolve(energy_points)
    
    assert is_valid_evolution(evolved)
```

## Running Tests

```bash
# Run all tests
pytest -q --tb=short

# Run specific test categories
pytest -m physics
pytest -m "not slow"
pytest -m "integration or unit"

# Generate coverage report
pytest --cov=core --cov-report=html

# Run paper and documentation traceability checks
pytest --no-cov tests/test_paper_validation_matrix.py
pytest --no-cov tests/test_documentation_traceability.py
```

## Best Practices

### Test Structure
1. Arrange: Set up test conditions
2. Act: Execute the test
3. Assert: Verify results

### Naming Conventions
- Test files: `test_*.py`
- Test classes: `Test*`
- Test functions: `test_*`

### Documentation
- Clear test descriptions
- Document test assumptions
- Explain complex assertions
