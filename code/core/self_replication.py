"""Generative self-replication operator for recursive field states.

The fixed-point form, ``F = T[F]``, captures recursive self-reference. The
generative step, ``W[F] -> F'``, lets a parent field produce a distinct
successor that preserves the recognizable pattern while introducing controlled
novelty.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Union

import numpy as np

from .errors import ValidationError
from .physics_constants import ALPHA_VAL
from .types import WaveFunction

FieldLike = Union[Sequence[complex], np.ndarray, WaveFunction]


@dataclass(frozen=True)
class ReplicationMetrics:
    """Measurements comparing a parent field to its generated successor."""

    parent_norm: float
    child_norm: float
    overlap: float
    residual: float
    novelty: float


@dataclass(frozen=True)
class ReplicationResult:
    """Container for ``W[F] -> F'`` outputs."""

    parent: np.ndarray
    child: np.ndarray
    metrics: ReplicationMetrics


def _as_complex_array(field: FieldLike) -> np.ndarray:
    """Convert supported field containers to a one-dimensional complex array."""
    if isinstance(field, WaveFunction):
        values = field.psi
    else:
        values = field

    array = np.asarray(values, dtype=complex)
    if array.ndim != 1:
        raise ValidationError("Field must be one-dimensional")
    if array.size < 3:
        raise ValidationError("Field must contain at least three samples")
    if not np.all(np.isfinite(array)):
        raise ValidationError("Field values must be finite")
    return array


def _validate_parameters(alpha: float, novelty: float, embodiment: float) -> None:
    """Validate operator parameters."""
    if not 0.0 < alpha < 1.0:
        raise ValidationError("alpha must satisfy 0 < alpha < 1")
    if not 0.0 <= novelty <= 1.0:
        raise ValidationError("novelty must satisfy 0 <= novelty <= 1")
    if embodiment < 0.0:
        raise ValidationError("embodiment must be non-negative")


def field_norm(field: FieldLike) -> float:
    """Return the Euclidean norm of a field state."""
    values = _as_complex_array(field)
    return float(np.linalg.norm(values))


def normalize_like(field: np.ndarray, target_norm: float) -> np.ndarray:
    """Normalize ``field`` to match ``target_norm``."""
    norm = np.linalg.norm(field)
    if norm == 0 or not np.isfinite(norm):
        raise ValidationError("Cannot normalize a zero or non-finite field")
    return field * (target_norm / norm)


def self_similarity(parent: FieldLike, child: FieldLike) -> float:
    """Compute normalized pattern overlap in ``[0, 1]``."""
    parent_values = _as_complex_array(parent)
    child_values = _as_complex_array(child)
    if parent_values.shape != child_values.shape:
        raise ValidationError("Fields must have matching shapes")

    denom = np.linalg.norm(parent_values) * np.linalg.norm(child_values)
    if denom == 0:
        raise ValidationError("Cannot compare zero-norm fields")
    return float(abs(np.vdot(parent_values, child_values)) / denom)


def generative_operator(
    field: FieldLike,
    *,
    alpha: float = ALPHA_VAL,
    novelty: float = 0.1,
    embodiment: float = 1.0,
    phase: float = np.pi / 7,
) -> np.ndarray:
    """Apply the generative ``W`` operator to produce ``F'``.

    The operator combines three pieces:

    - a recursive reflection of the parent field,
    - a translated/phase-shifted copy representing propagation, and
    - a bounded gradient term representing generative novelty.

    The result is normalized to the parent norm so replication is measurable
    without changing total amplitude.
    """
    _validate_parameters(alpha, novelty, embodiment)
    parent = _as_complex_array(field)
    parent_norm = np.linalg.norm(parent)
    if parent_norm == 0:
        raise ValidationError("Cannot replicate a zero-norm field")

    shifted = np.roll(parent, 1) * np.exp(1j * phase)
    gradient = np.gradient(parent)
    gradient_norm = np.linalg.norm(gradient)
    if gradient_norm > 0:
        gradient = gradient * (parent_norm / gradient_norm)

    reflective_part = (1.0 - alpha) * parent + alpha * shifted
    generative_part = novelty * alpha * embodiment * gradient
    child = reflective_part + generative_part
    return normalize_like(child, float(parent_norm))


def replication_metrics(parent: FieldLike, child: FieldLike) -> ReplicationMetrics:
    """Measure norm preservation, overlap, residual, and novelty."""
    parent_values = _as_complex_array(parent)
    child_values = _as_complex_array(child)
    if parent_values.shape != child_values.shape:
        raise ValidationError("Fields must have matching shapes")

    parent_norm = float(np.linalg.norm(parent_values))
    child_norm = float(np.linalg.norm(child_values))
    if parent_norm == 0:
        raise ValidationError("Cannot measure zero-norm parent")

    residual = float(np.linalg.norm(child_values - parent_values) / parent_norm)
    overlap = self_similarity(parent_values, child_values)
    novelty = max(0.0, min(1.0, 1.0 - overlap))
    return ReplicationMetrics(
        parent_norm=parent_norm,
        child_norm=child_norm,
        overlap=overlap,
        residual=residual,
        novelty=novelty,
    )


def replicate_field(
    field: FieldLike,
    *,
    alpha: float = ALPHA_VAL,
    novelty: float = 0.1,
    embodiment: float = 1.0,
    phase: float = np.pi / 7,
) -> ReplicationResult:
    """Generate ``F'`` from ``F`` and return validation metrics."""
    parent = _as_complex_array(field)
    child = generative_operator(
        parent,
        alpha=alpha,
        novelty=novelty,
        embodiment=embodiment,
        phase=phase,
    )
    return ReplicationResult(
        parent=parent.copy(),
        child=child,
        metrics=replication_metrics(parent, child),
    )


def replicate_wavefunction(
    wavefunction: WaveFunction,
    *,
    alpha: float = ALPHA_VAL,
    novelty: float = 0.1,
    embodiment: float = 1.0,
    phase: float = np.pi / 7,
) -> WaveFunction:
    """Apply ``W`` to a :class:`WaveFunction` while preserving metadata."""
    if not isinstance(wavefunction, WaveFunction):
        raise ValidationError("replicate_wavefunction requires a WaveFunction")

    result = replicate_field(
        wavefunction,
        alpha=alpha,
        novelty=novelty,
        embodiment=embodiment,
        phase=phase,
    )
    quantum_numbers = dict(wavefunction.quantum_numbers)
    quantum_numbers["generation"] = int(quantum_numbers.get("generation", 0)) + 1
    quantum_numbers["generative_overlap"] = result.metrics.overlap
    quantum_numbers["generative_residual"] = result.metrics.residual

    return WaveFunction(
        psi=result.child,
        grid=wavefunction.grid.copy(),
        mass=wavefunction.mass,
        quantum_numbers=quantum_numbers,
    )


def replication_sequence(
    seed: FieldLike,
    *,
    steps: int,
    alpha: float = ALPHA_VAL,
    novelty: float = 0.1,
    embodiment: float = 1.0,
    phase: float = np.pi / 7,
) -> List[np.ndarray]:
    """Generate a finite sequence ``F, F', F'', ...``."""
    if steps < 0:
        raise ValidationError("steps must be non-negative")

    current = _as_complex_array(seed).copy()
    sequence = [current]
    for _ in range(steps):
        current = generative_operator(
            current,
            alpha=alpha,
            novelty=novelty,
            embodiment=embodiment,
            phase=phase,
        )
        sequence.append(current)
    return sequence


def sequence_metrics(sequence: Iterable[FieldLike]) -> List[ReplicationMetrics]:
    """Return pairwise metrics for a replication sequence."""
    values = [_as_complex_array(item) for item in sequence]
    return [
        replication_metrics(values[index], values[index + 1])
        for index in range(len(values) - 1)
    ]


__all__ = [
    "ReplicationMetrics",
    "ReplicationResult",
    "field_norm",
    "normalize_like",
    "replicate_field",
    "replicate_wavefunction",
    "replication_metrics",
    "replication_sequence",
    "self_similarity",
    "sequence_metrics",
    "generative_operator",
]
