"""Error definitions for The Divine Blueprint validation framework."""


class FractalTheoryError(Exception):
    """Base class for framework errors."""


class PhysicsError(FractalTheoryError):
    """Physics constraint violation."""


class ValidationError(PhysicsError):
    """Input validation error."""


class ComputationError(PhysicsError):
    """Numerical or symbolic computation error."""


class StabilityError(ComputationError):
    """Numerical stability violation."""


class PrecisionError(ComputationError):
    """Precision requirement failure."""


class BoundsError(ComputationError):
    """Value outside acceptable bounds."""


class ConvergenceError(ComputationError):
    """Iteration or series convergence failure."""


class EnergyConditionError(PhysicsError):
    """Violation of an energy condition."""


class CausalityError(PhysicsError):
    """Violation of causal support."""


class GaugeError(PhysicsError):
    """Gauge transformation or covariance error."""


class ConfigurationError(FractalTheoryError):
    """Framework configuration error."""


class VersionError(FractalTheoryError):
    """Version metadata or compatibility error."""


class BasisError(ComputationError):
    """Fractal basis computation error."""


class CoherenceError(PhysicsError):
    """Quantum coherence preservation error."""


class CrossSectionError(ComputationError):
    """Cross-section computation error."""


class EnergyScaleError(PhysicsError):
    """Invalid energy scale."""


class NumericalStabilityError(StabilityError):
    """Numerical stability violation."""


class QuantumNumberError(ValidationError):
    """Invalid quantum-number configuration."""


class UnitarityError(PhysicsError):
    """Unitarity violation."""


class HolographicError(PhysicsError):
    """Holographic bound violation."""
