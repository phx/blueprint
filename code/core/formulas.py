"""Executable formula proxies for the Divine Blueprint paper.

These helpers keep displayed paper mathematics traceable to concrete code. They
are intentionally small: each function mirrors one formula family and is tested
against the paper-facing registry.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from .errors import ValidationError


def architecture_terms() -> tuple[str, ...]:
    """Return the five-term synthesis stated in the introduction."""
    return (
        "scale symmetry",
        "holographic bound",
        "recursive coupling flow",
        "fixed-point recursion",
        "generative succession",
    )


def commutator(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Compute [A, B] = AB - BA."""
    a = np.asarray(left, dtype=complex)
    b = np.asarray(right, dtype=complex)
    return a @ b - b @ a


def contraction_norm(operator: np.ndarray) -> float:
    """Compute the spectral norm of D^dagger D."""
    op = np.asarray(operator, dtype=complex)
    return float(np.linalg.norm(op.conj().T @ op, ord=2))


def is_hermitian(matrix: np.ndarray, *, atol: float = 1e-12) -> bool:
    """Check C = C^dagger."""
    values = np.asarray(matrix, dtype=complex)
    return bool(np.allclose(values, values.conj().T, atol=atol))


def fractal_self_similarity(field_value: complex, lambda_scale: float, scaling_dimension: float) -> complex:
    """Compute lambda^D F(x,t,E)."""
    return complex((lambda_scale ** scaling_dimension) * field_value)


def holographic_entropy_bound(area: float, planck_length: float) -> float:
    """Compute A/(4 l_P^2)."""
    return float(area / (4.0 * planck_length ** 2))


def recursive_coupling(base: float, alpha: float, corrections: Sequence[float]) -> float:
    """Compute g_i(E) + sum alpha^n F_n."""
    return float(base + sum(alpha ** n * term for n, term in enumerate(corrections, start=1)))


def recursive_inverse_coupling(
    inverse_at_mz: float,
    beta_coefficient: float,
    energy: float,
    mz: float,
    alpha: float,
    corrections: Sequence[float],
) -> float:
    """Compute alpha_i^{-1}(E) with logarithmic and recursive terms."""
    classical = inverse_at_mz + beta_coefficient * np.log(energy / mz) / (2.0 * np.pi)
    return recursive_coupling(float(classical), alpha, corrections)


def generative_constraints(parent: np.ndarray, child: np.ndarray) -> dict[str, float]:
    """Measure norm preservation, similarity, and residual for F -> F'."""
    p = np.asarray(parent, dtype=complex)
    c = np.asarray(child, dtype=complex)
    parent_norm = float(np.linalg.norm(p))
    child_norm = float(np.linalg.norm(c))
    similarity = float(abs(np.vdot(p, c)) / (parent_norm * child_norm))
    residual = float(np.linalg.norm(c - p))
    return {"parent_norm": parent_norm, "child_norm": child_norm, "similarity": similarity, "residual": residual}


def unified_field_expansion(psi_levels: np.ndarray, lagrangian: np.ndarray, alpha: float, *, dx: float = 1.0) -> complex:
    """Compute a finite proxy for integral sum alpha^n Psi_n exp(iL) dx."""
    levels = np.asarray(psi_levels, dtype=complex)
    lag = np.asarray(lagrangian, dtype=float)
    weights = np.array([alpha ** n for n in range(levels.shape[0])], dtype=float)
    integrand = np.sum(weights[:, None] * levels, axis=0) * np.exp(1j * lag)
    return complex(np.sum(integrand) * dx)


def basis_closure(alpha: float, next_state: np.ndarray) -> np.ndarray:
    """Compute alpha Psi_{n+1}."""
    return alpha * np.asarray(next_state, dtype=complex)


def resolution_identity(basis_vectors: np.ndarray) -> np.ndarray:
    """Compute sum |psi_n><psi_n|/N_n for a finite orthogonal basis."""
    vectors = np.asarray(basis_vectors, dtype=complex)
    result = np.zeros((vectors.shape[1], vectors.shape[1]), dtype=complex)
    for vector in vectors:
        norm = np.vdot(vector, vector)
        result += np.outer(vector, vector.conj()) / norm
    return result


def convergence_tail_bound(max_norm: float, alpha: float, start: int) -> float:
    """Compute M alpha^N/(1-alpha)."""
    return float(max_norm * alpha ** start / (1.0 - alpha))


def degrees_of_freedom(length: float, planck_length: float, alpha: float, level: int) -> float:
    """Compute (L/l_P)^2 product(1+alpha^k)^-1."""
    product = np.prod([(1.0 + alpha ** k) ** -1 for k in range(1, level + 1)])
    return float((length / planck_length) ** 2 * product)


def boundary_dimension() -> float:
    """Return the holographic boundary dimension."""
    return 2.0


def area_entropy(radius: float) -> float:
    """Compute an area-law entropy proxy proportional to r^2."""
    return float(radius ** 2)


def coupling_spread(couplings: Sequence[float]) -> float:
    """Compute max(g_i)-min(g_i)."""
    values = np.asarray(couplings, dtype=float)
    return float(np.max(values) - np.min(values))


def gut_prediction_values() -> dict[str, float]:
    """Return the paper's internal GUT prediction proxies."""
    return {"M_GUT": 2.1e16, "alpha_GUT": 0.0376, "tau_p": 1.0e34, "Gamma_p_to_e_pi": 1.6e-36}


def mass_product(m0: float, alpha: float, hierarchy_terms: Sequence[float]) -> float:
    """Compute m0 product(1+alpha^k h_f(k))."""
    product = np.prod([1.0 + alpha ** k * term for k, term in enumerate(hierarchy_terms, start=1)])
    return float(m0 * product)


def cp_phase(abs_alpha: float, k: int, total: int, delta_k: float) -> complex:
    """Compute |alpha_k| exp(i(2pi k/N + delta_k))."""
    theta = 2.0 * np.pi * k / total + delta_k
    return complex(abs_alpha * np.exp(1j * theta))


def jarlskog_baryon_values() -> dict[str, float]:
    """Return the CP and baryon-asymmetry reference proxies."""
    return {"J": 3.2e-5, "eta_B": 6.1e-10}


def sakharov_conditions() -> dict[str, bool]:
    """Return the three Sakharov-condition proxy flags."""
    return {"B_violation": True, "C_CP_violation": True, "out_of_equilibrium": True}


def dark_matter_density(radius: float, alpha: float, corrections: Sequence[float], *, rho0: float = 1.0) -> float:
    """Compute rho0 r^-2 product(1+alpha^k f_k(r))."""
    product = np.prod([1.0 + alpha ** k * term for k, term in enumerate(corrections, start=1)])
    return float(rho0 * radius ** -2 * product)


def dark_matter_coupling(level: int, standard_model_coupling: float, alpha: float) -> float:
    """Compute alpha^level g_SM."""
    return float(alpha ** level * standard_model_coupling)


def dark_energy_density(scale_ev: float = 2.3e-3) -> float:
    """Compute (2.3e-3 eV)^4 by default."""
    return float(scale_ev ** 4)


def measurement_update(rho0: np.ndarray, alpha: float, channel_outputs: Sequence[np.ndarray]) -> np.ndarray:
    """Compute rho0 + sum alpha^n D_n[rho0]."""
    result = np.asarray(rho0, dtype=complex).copy()
    for n, channel in enumerate(channel_outputs, start=1):
        result += alpha ** n * np.asarray(channel, dtype=complex)
    return result


def normalized_probabilities(amplitudes: Sequence[complex]) -> np.ndarray:
    """Compute probabilities that sum to one."""
    values = np.asarray(amplitudes, dtype=complex)
    if values.size == 0 or not np.all(np.isfinite(values)):
        raise ValidationError("Amplitudes must be finite and non-empty")
    probabilities = np.abs(values) ** 2
    total = float(np.sum(probabilities))
    if total <= 0.0 or not np.isfinite(total):
        raise ValidationError("Cannot normalize zero-probability amplitudes")
    return probabilities / total


def gravitational_action_proxy(einstein_hilbert_term: float, alpha: float, curvature_terms: Sequence[float]) -> float:
    """Compute S_EH + sum alpha^k C_k."""
    return recursive_coupling(einstein_hilbert_term, alpha, curvature_terms)


def recursive_product(base: float, alpha: float, level: int) -> float:
    """Compute base product(1+alpha^k)^-1."""
    factor = np.prod([(1.0 + alpha ** k) ** -1 for k in range(1, level + 1)])
    return float(base * factor)


def information_content(alpha: float, levels: int) -> float:
    """Compute -sum alpha^n ln(alpha^n)."""
    weights = np.array([alpha ** n for n in range(1, levels + 1)], dtype=float)
    return float(-np.sum(weights * np.log(weights)))


def framework_dimension(alpha: float, levels: int) -> float:
    """Compute 4 + sum alpha^n."""
    return float(4.0 + sum(alpha ** n for n in range(1, levels + 1)))


def gw_spectrum(frequency: float, omega0: float, f0: float, exponent: float, alpha: float, corrections: Sequence[float]) -> float:
    """Compute Omega0 (f/f0)^n product(1+alpha^k h_k)."""
    product = np.prod([1.0 + alpha ** k * term for k, term in enumerate(corrections, start=1)])
    return float(omega0 * (frequency / f0) ** exponent * product)


def gw_power_law_ratio(f2: float, f1: float, exponent: float = 2.0 / 3.0) -> float:
    """Compute (f2/f1)^(2/3) by default."""
    return float((f2 / f1) ** exponent)


def low_energy_signature_values() -> dict[str, float]:
    """Return the low-energy reference proxies."""
    return {"Delta_r_W": 37.979e-3, "B_Bs_to_mumu": 3.09e-9, "sin2_theta13": 0.0218}


def uncertainty_quadrature(uncertainties: Sequence[float]) -> float:
    """Compute sqrt(sum sigma_i^2)."""
    values = np.asarray(uncertainties, dtype=float)
    return float(np.sqrt(np.sum(values ** 2)))
