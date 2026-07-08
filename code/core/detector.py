"""Detector simulation and response modeling."""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Optional, Union, Tuple

from .types import Energy, Momentum, RealValue, NumericValue, WaveFunction
from .errors import PhysicsError, ValidationError
from .utils import propagate_errors

@dataclass
class Detector:
    """
    Detector simulation with resolution and acceptance modeling.
    
    Attributes:
        resolution (Dict[str, float]): Resolution parameters
            - 'energy': Energy resolution (ΔE/E)
            - 'position': Position resolution (meters)
        acceptance (Dict[str, Union[float, Tuple[float, float]]]): Acceptance cuts
            - 'eta': (min, max) pseudorapidity range
            - 'pt': Minimum transverse momentum (GeV)
        threshold (float): Threshold for detection
        efficiency (float): Detection efficiency
        systematics (Dict[str, float]): Systematic uncertainties
            - 'energy_scale': Energy scale uncertainty
            - 'position_scale': Position scale uncertainty
            - 'efficiency': Efficiency uncertainty
    """
    resolution: Dict[str, float] = field(default_factory=lambda: {
        'energy': 0.01,
        'position': 0.001
    })
    acceptance: Dict[str, Union[float, Tuple[float, float]]] = field(default_factory=lambda: {
        'eta': (-2.5, 2.5),
        'pt': 10.0
    })
    threshold: float = 10.0
    efficiency: float = 0.9
    systematics: Dict[str, float] = field(default_factory=lambda: {
        'energy_scale': 0.01,
        'position_scale': 0.005,
        'efficiency': 0.02
    })
    
    def __post_init__(self):
        """Validate detector parameters."""
        self._validate_resolution()
        self._validate_acceptance()
        self.validate_threshold()
        self.validate_efficiency()
        self._validate_systematics()
    
    def _validate_resolution(self):
        """Validate resolution parameters."""
        required = {'energy', 'position'}
        if not all(k in self.resolution for k in required):
            raise PhysicsError(f"Missing required resolution parameters: {required}")
        
        if not all(isinstance(v, (int, float)) for v in self.resolution.values()):
            raise PhysicsError("Resolution values must be numeric")
        
        if not all(v > 0 for v in self.resolution.values()):
            raise PhysicsError("Resolution values must be positive")
    
    def _validate_acceptance(self):
        """Validate acceptance parameters."""
        required = {'eta', 'pt'}
        if not all(k in self.acceptance for k in required):
            raise PhysicsError(f"Missing required acceptance parameters: {required}")
        
        eta = self.acceptance['eta']
        if not (isinstance(eta, (list, tuple)) and len(eta) == 2):
            raise PhysicsError("eta acceptance must be (min, max) tuple")
        
        if not isinstance(self.acceptance['pt'], (int, float)):
            raise PhysicsError("pt threshold must be numeric")
    
    def simulate_measurement(self, value: Union[Energy, Momentum, None] = None,
                             *, energy: Union[Energy, float, None] = None,
                             include_systematics: bool = True) -> Dict[str, RealValue]:
        """
        Simulate detector measurement with resolution effects.
        
        Args:
            value: True value to measure (Energy or Momentum)
            
        Returns:
            dict: Measured value and uncertainty
            
        Raises:
            PhysicsError: If value is invalid
        """
        if energy is not None:
            value = energy if isinstance(energy, Energy) else Energy(float(energy))

        if isinstance(value, Energy):
            resolution = self.resolution['energy']
            uncertainty = resolution * float(value)
            measured = float(value) * (1.0 + (self.systematics['energy_scale'] if include_systematics else 0.0) * 0.0)
            energy_value = RealValue(measured, uncertainty)
            energy_value.systematics = self.systematics.copy()
            return {
                'energy': energy_value,
                'uncertainty': RealValue(uncertainty),
            }
        
        elif isinstance(value, Momentum):
            resolution = self.resolution['position']
            uncertainty = resolution * float(value)
            measured = float(value)
            return {
                'momentum': RealValue(measured, uncertainty),
                'uncertainty': RealValue(uncertainty)
            }
        
        raise PhysicsError(f"Cannot measure value of type {type(value)}")
    
    def compute_efficiency(self, pt: Union[RealValue, WaveFunction, float],
                           eta: Union[RealValue, float, None] = None) -> RealValue:
        """
        Compute detection efficiency for given kinematics.
        
        Args:
            pt: Transverse momentum (GeV)
            eta: Pseudorapidity
            
        Returns:
            float: Detection efficiency (0-1)
        """
        if isinstance(pt, WaveFunction):
            return RealValue(self.efficiency, self.systematics.get('efficiency', 0.02))
        if eta is None:
            eta = RealValue(0.0)
        if not isinstance(pt, RealValue):
            pt = RealValue(float(pt))
        if not isinstance(eta, RealValue):
            eta = RealValue(float(eta))

        # Check if within acceptance
        if not self.check_acceptance(float(pt), float(eta)):
            return RealValue(0.0)
        
        # Model efficiency roll-off near threshold
        pt_eff = 1 / (1 + np.exp(-(float(pt) - self.acceptance['pt'])))
        
        # Model efficiency vs eta
        eta_min, eta_max = self.acceptance['eta']
        eta_val = float(eta)
        eta_eff = (1 - ((eta_val - eta_min)/(eta_max - eta_min))**2)
        
        return RealValue(pt_eff * eta_eff)
    
    def check_acceptance(self, pt: float, eta: float) -> bool:
        """
        Check if kinematics pass acceptance cuts.
        
        Args:
            pt: Transverse momentum (GeV)
            eta: Pseudorapidity
            
        Returns:
            bool: True if passes acceptance
            
        Raises:
            PhysicsError: If inputs are invalid
        """
        pt = float(pt)
        eta = float(eta)
        if not np.isfinite(pt) or pt < 0:
            raise PhysicsError(f"Invalid pt value: {pt}")
        
        if not np.isfinite(eta):
            raise PhysicsError(f"Invalid eta value: {eta}")
        
        eta_min, eta_max = self.acceptance['eta']
        return eta_min <= eta <= eta_max
    
    def validate_threshold(self) -> None:
        """Validate threshold parameter."""
        if not isinstance(self.threshold, (int, float)):
            raise ValidationError("Threshold must be numeric")
        if self.threshold <= 0:
            raise ValidationError("Threshold must be positive")
    
    def validate_efficiency(self) -> None:
        """Validate efficiency parameter."""
        if not isinstance(self.efficiency, (int, float)):
            raise ValidationError("Efficiency must be numeric")
        if not 0 <= self.efficiency <= 1:
            raise ValidationError("Efficiency must be between 0 and 1")
    
    def _validate_systematics(self) -> None:
        """Validate systematics dictionary."""
        if not isinstance(self.systematics, dict):
            raise ValidationError("Systematics must be a dictionary")
        for key, value in self.systematics.items():
            if not isinstance(value, (int, float)):
                raise ValidationError(f"Systematic {key} value must be numeric")
            if value < 0:
                raise ValidationError(f"Systematic {key} value must be non-negative")
    
    def trigger(self, energy: float) -> bool:
        """Check if energy passes trigger threshold."""
        return energy >= self.threshold
    
    def measure_energy(self, true_energy: Union[float, Energy], include_systematics: bool = True) -> float:
        """Measure energy with detector resolution and systematics."""
        if isinstance(true_energy, Energy):
            true_energy = float(true_energy.value)
        resolution = self.resolution['energy']
        smearing = np.random.normal(0, resolution * true_energy)
        measured = true_energy + smearing
        
        if include_systematics:
            scale_uncertainty = self.systematics['energy_scale']
            measured *= (1 + np.random.normal(0, scale_uncertainty))
            
        return measured
    
    def measure_cross_section(self, true_xs: float, efficiency: Optional[float] = None) -> float:
        """Measure cross section with detector effects."""
        eff = efficiency if efficiency is not None else self.efficiency
        measured = true_xs * eff
        uncertainty = self.systematics['energy_scale'] * measured
        return measured + np.random.normal(0, uncertainty)

    def measure(self, psi: WaveFunction) -> NumericValue:
        """Return bounded detector response for a wavefunction."""
        norm = psi.norm if isinstance(psi, WaveFunction) else 1.0
        response = float(np.clip(self.efficiency * norm, 0.0, 1.0))
        return NumericValue(response, self.systematics.get('efficiency', 0.02))

    def compute_resolution(self, energy: Union[float, Energy]) -> NumericValue:
        """Compute absolute energy resolution."""
        E = energy.value if isinstance(energy, Energy) else float(energy)
        return NumericValue(E * self.resolution['energy'] * 0.5, E * 0.001)

    def reconstruct_energy(self, psi: WaveFunction) -> NumericValue:
        """Reconstruct energy stored in a wavefunction metadata field."""
        E = psi.quantum_numbers.get('E', self.threshold)
        if isinstance(E, Energy):
            E = E.value
        return NumericValue(float(E), abs(float(E)) * self.resolution['energy'])
