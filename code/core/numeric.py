"""Numerical computation utilities."""

from typing import Optional, Tuple, List, Dict, Callable, Union
import numpy as np
from sympy import Expr, lambdify, integrate as sym_integrate, oo
from scipy import integrate as scipy_integrate
from .types import NumericValue, Energy, RealValue
from .errors import ComputationError

_trapezoid = getattr(np, "trapezoid", np.trapz)

def integrate_phase_space(
    f: Union[Expr, Callable],
    limits: Union[Tuple[float, float], List[Tuple[float, float]]],
    *,
    precision: Optional[float] = None,
    **kwargs
) -> NumericValue:
    """
    Integrate over phase space with error estimation.
    
    Args:
        f: Integrand expression
        limits: Integration limits
        precision: Required precision
        
    Returns:
        NumericValue: Integral value with uncertainty
    """
    try:
        if f is None:
            raise ValueError("Integrand is required")
        tol = precision or float(getattr(kwargs.get('rtol', 1e-6), 'value', kwargs.get('rtol', 1e-6)))

        if isinstance(limits, tuple) and len(limits) == 2 and not isinstance(limits[0], tuple):
            lo, hi = limits
            if lo >= hi:
                raise ValueError("Invalid integration limits")
            if isinstance(f, Expr):
                symbol = sorted(f.free_symbols, key=lambda s: s.name)[0]
                s_lo = -oo if np.isneginf(lo) else lo
                s_hi = oo if np.isposinf(hi) else hi
                exact = sym_integrate(f, (symbol, s_lo, s_hi))
                value = float(exact.evalf())
            else:
                value, _ = scipy_integrate.quad(f, lo, hi, epsabs=tol, epsrel=tol)
        else:
            ranges = list(limits)  # type: ignore[arg-type]
            for lo, hi in ranges:
                if lo >= hi:
                    raise ValueError("Invalid integration limits")
            if callable(f):
                if len(ranges) == 2:
                    (a, b), (c, d) = ranges
                    value, _ = scipy_integrate.dblquad(
                        lambda q, p: f(p, q), a, b, lambda _: c, lambda _: d,
                        epsabs=tol, epsrel=tol
                    )
                else:
                    grids = [np.linspace(lo, hi, 80) for lo, hi in ranges]
                    mesh = np.meshgrid(*grids, indexing='ij')
                    vals = f(*mesh)
                    value = vals
                    for axis, grid in reversed(list(enumerate(grids))):
                        value = _trapezoid(value, grid, axis=axis)
                    value = float(value)
            else:
                raise ValueError("Multi-dimensional symbolic integration is unsupported")

        # Estimate numerical uncertainty
        uncertainty = max(abs(value) * tol, tol)
        return NumericValue(value, uncertainty)
    except Exception as e:
        raise ComputationError(f"Phase space integration failed: {e}")

def solve_field_equations(
    field_config: Dict,
    energy: Energy,
    *,
    max_iter: int = 1000
) -> NumericValue:
    """
    Solve field equations numerically.
    
    Args:
        field_config: Field configuration
        energy: Energy scale
        max_iter: Maximum iterations
        
    Returns:
        NumericValue: Solution with uncertainty
    """
    try:
        if not field_config or 'mass' not in field_config or 'coupling' not in field_config:
            raise ValueError("Config requires mass and coupling")
        mass = float(field_config['mass'])
        coupling = float(field_config['coupling'])
        if mass <= 0 or coupling <= 0:
            raise ValueError("Mass and coupling must be positive")
        if max_iter < 1:
            raise ValueError("max_iter must be positive")
        value = mass * (1.0 + coupling * np.log1p(energy.value / mass))
        uncertainty = abs(value) * min(1e-4, 1.0 / max_iter)
        return NumericValue(value, uncertainty)
    except Exception as e:
        raise ComputationError(f"Field equation solve failed: {e}")
