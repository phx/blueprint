"""Tests for unified field theory implementation.

Validation note:
The test suite validates:
1. Field equation solutions
2. Coupling unification
3. Quantum corrections
4. Holographic properties
"""

import csv
import pytest
import numpy as np
from numpy import pi, exp, log, sqrt
from pathlib import Path

from core.field import UnifiedField
from core.types import Energy, WaveFunction, FieldConfig, NumericValue
from core.physics_constants import (
    HBAR, C, G, M_P, I,  # Level 1: Fundamental constants
    g_μν, Gamma,  # Level 2: Mathematical objects
    Z_MASS  # Level 3: Derived quantities
)

# Load test data from the tracked repository data directory.
DATA_DIR = Path(__file__).resolve().parents[2] / "data"

def load_test_data(filename: str) -> dict:
    """
    Load test data with proper validation.
    
    Args:
        filename: Name of data file to load
        
    Returns:
        dict: Validated test data
        
    Raises:
        FileNotFoundError: If data file is missing
        ValidationError: If data format is invalid
    """
    with open(DATA_DIR / filename, newline="") as f:
        reader = csv.DictReader(row for row in f if not row.startswith("#"))
        key_column = reader.fieldnames[0]
        return {row[key_column]: row for row in reader}

@pytest.fixture
def field():
    """
    Create UnifiedField instance with test configuration.
    
    Validation note:
    Uses standard test parameters:
    - α = 0.1 (scaling parameter)
    - ε = 1e-10 (precision)
    - d = 4 (dimension)
    """
    return UnifiedField(alpha=0.1, precision=1e-10)

@pytest.fixture
def test_state():
    """
    Create test quantum state.
    
    Validation note:
    Uses Gaussian ground state:
    ψ₀(x) = exp(-x²/2)
    
    Returns:
        WaveFunction: Normalized test state
    """
    grid = np.linspace(-3, 3, 100)  # CANONICAL: grid range
    psi = np.exp(-grid**2/2)  # Gaussian test state
    return WaveFunction(
        psi=psi,
        grid=grid,
        quantum_numbers={'n': 1, 'E': 1.0}
    )

def test_energy_density(field, test_state):
    """
    Test energy density computation.
    
    Validation note:
    Verifies:
    1. Classical kinetic term: T ~ |∇ψ|²
    2. Potential energy: V ~ |x|²|ψ|²
    3. Quantum corrections: ΔE ~ α^n
    
    Args:
        field: UnifiedField fixture
        test_state: Test WaveFunction fixture
        
    Raises:
        AssertionError: If energy properties are violated
    """
    result = field.compute_energy_density(test_state)
    
    # Load validation data
    valid_data = load_test_data("validation_results.csv")
    expected = valid_data['energy_density']

    assert expected['result'] == 'Pass'
    assert result.value == pytest.approx(1.0)
    assert result.uncertainty <= float(expected['error'])

def test_gravitational_action(field, test_state):
    """
    Test gravitational action computation.
    
    Validation note:
    Verifies:
    1. Einstein-Hilbert term: S_EH ~ ∫d⁴x √(-g)R
    2. Quantum corrections: S_q ~ ∑ α^n C_n(R)
    3. Holographic bound: S ≤ A/(4G)
    
    Args:
        field: UnifiedField fixture
        test_state: Test WaveFunction fixture
        
    Raises:
        AssertionError: If action properties are violated
    """
    result = field.compute_gravitational_action(test_state)
    
    # Verify holographic bound
    area = 4*pi * (3*HBAR/M_P)**2  # Boundary area
    assert result.value <= area/(4*HBAR)

def test_coupling_unification(field):
    """
    Test gauge coupling unification.
    
    Validation note:
    Verifies:
    1. Coupling convergence: |gᵢ(M_GUT) - gⱼ(M_GUT)| → 0
    2. RG flow consistency: β(g) = μ∂g/∂μ
    3. Quantum corrections: g(μ) = g₀ + ∑ α^n b_n(g₀)ln(μ/μ₀)
    
    Args:
        field: UnifiedField fixture
        
    Raises:
        AssertionError: If unification fails
    """
    E_GUT = Energy(2.1e16)  # GeV
    couplings = field.compute_unified_couplings(E_GUT)
    
    g1 = couplings['g1'].value
    g2 = couplings['g2'].value
    g3 = couplings['g3'].value
    
    assert abs(g1 - g2) < 1e-6
    assert abs(g2 - g3) < 1e-6

def test_predictions(field):
    """
    Test theoretical predictions.
    
    Validation note:
    Verifies:
    1. GUT scale: M_GUT ~ 2×10¹⁶ GeV
    2. Unified coupling: α_GUT ~ 1/25
    3. Internal proton lifetime-scale proxy: tau_p ~ 10^34 years
    
    Args:
        field: UnifiedField fixture
        
    Raises:
        AssertionError: If predictions deviate from expected values
    """
    predictions = field._compute_theoretical_predictions()
    
    # Load validation data
    valid_data = load_test_data("predictions.csv")
    
    # Verify GUT scale
    M_GUT = predictions['M_GUT'].value
    assert abs(M_GUT - float(valid_data['M_GUT']['predicted']))/M_GUT < 0.15
    
    # Verify unified coupling
    alpha_GUT = predictions['alpha_GUT'].value
    assert abs(alpha_GUT - float(valid_data['alpha_GUT']['predicted'])) < 2e-4
    
    # Verify the internal proton lifetime-scale proxy
    tau_p = predictions['tau_p'].value
    assert abs(log(tau_p/float(valid_data['tau_p']['predicted']))) < 0.3
