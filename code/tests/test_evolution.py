"""Tests for field evolution and dynamics."""

import pytest
from hypothesis import given, strategies as st
import numpy as np
from core.field import UnifiedField
from core.types import Energy, FieldConfig, WaveFunction

@pytest.fixture
def field():
    """Create UnifiedField instance for testing."""
    return UnifiedField(alpha=0.1)

@pytest.fixture
def test_state(field):
    """Create a normalized finite-grid field state."""
    return field.compute_basis_state(energy=Energy(1.0), mass=1.0)

class TestTimeEvolution:
    """Test time evolution properties."""
    
    def test_energy_conservation(self, field, test_state):
        """Test that stable evolution preserves the energy-density proxy."""
        initial_energy = field.compute_energy_density(test_state).value
        
        field.state = test_state
        evolved = field.evolve(Energy(1.0))
        final_energy = field.compute_energy_density(evolved).value
        
        assert isinstance(evolved, WaveFunction)
        assert final_energy == pytest.approx(initial_energy)
    
    def test_unitarity(self, field, test_state):
        """Test that stable evolution preserves finite-grid norm."""
        field.state = test_state
        evolved = field.evolve(Energy(1.0))
        
        assert evolved.norm == pytest.approx(test_state.norm)
        assert np.vdot(evolved.psi, evolved.psi).real == pytest.approx(
            np.vdot(test_state.psi, test_state.psi).real
        )

class TestDynamics:
    """Test field dynamics."""
    
    @given(st.floats(min_value=0.1, max_value=10.0))
    def test_equations_of_motion(self, mass):
        """Test that the finite-grid dynamics proxy stays bounded."""
        field = UnifiedField(alpha=0.1)
        config = FieldConfig(mass=mass, coupling=0.1, dimension=1)
        initial = field.compute_field(config)
        field.state = initial
        evolved = field.evolve(Energy(max(mass, 1.0)))
        
        assert isinstance(evolved, WaveFunction)
        assert evolved.quantum_numbers["field_config"] is True
        assert np.all(np.isfinite(evolved.psi))
        assert evolved.norm == pytest.approx(initial.norm)
        assert field.compute_energy_density(evolved).value >= 0.0
    
    def test_causality_preservation(self, field, test_state):
        """Test that stable evolution preserves the causality proxy."""
        field.state = test_state
        evolved = field.evolve(Energy(1.0))
        
        assert field.check_causality(evolved)

class TestScattering:
    """Test scattering properties."""
    
    def test_s_matrix_unitarity(self, field):
        """Test unitarity of the stable finite-state S-matrix proxy."""
        states = [
            field.compute_basis_state(energy=Energy(1.0), n=0),
            field.compute_basis_state(energy=Energy(2.0), n=1),
        ]
        
        s_matrix = field.compute_s_matrix(states)
        identity = s_matrix @ s_matrix.conj().T
        
        assert np.allclose(identity, np.eye(len(states)))
    
    def test_crossing_symmetry(self, field):
        """Test symmetry and boundedness of the stable scattering proxy."""
        psi_particle = field.compute_basis_state(energy=Energy(1.0), n=0)
        psi_antiparticle = field.compute_basis_state(energy=Energy(1.0), n=1)
        
        A_forward = field.compute_scattering_amplitude(psi_particle, psi_antiparticle)
        A_crossed = field.compute_scattering_amplitude(psi_antiparticle, psi_particle)
        
        assert abs(A_forward) <= 1.0
        assert A_forward == pytest.approx(A_crossed)
