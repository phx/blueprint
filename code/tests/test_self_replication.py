"""Tests for the generative W[F] -> F' self-replication layer."""

import numpy as np
import pytest

from core.errors import ValidationError
from core.self_replication import (
    _validate_nontrivial_successor,
    generative_operator,
    replicate_field,
    replicate_wavefunction,
    replication_metrics,
    replication_sequence,
    self_similarity,
    sequence_metrics,
)
from core.types import WaveFunction


def normalized_seed(size=128):
    grid = np.linspace(-3, 3, size)
    psi = np.exp(-grid**2 / 2).astype(complex)
    psi = psi / np.linalg.norm(psi)
    return grid, psi


def test_generative_operator_preserves_norm_and_shape():
    _, seed = normalized_seed()

    child = generative_operator(seed, alpha=0.1, novelty=0.15, phase=np.pi / 9)

    assert child.shape == seed.shape
    assert np.all(np.isfinite(child))
    assert np.linalg.norm(child) == pytest.approx(np.linalg.norm(seed))


def test_generative_operator_creates_self_similar_non_clone():
    _, seed = normalized_seed()

    child = generative_operator(seed, alpha=0.12, novelty=0.2, phase=np.pi / 6)
    similarity = self_similarity(seed, child)
    metrics = replication_metrics(seed, child)

    assert 0.85 < similarity < 0.999999
    assert metrics.overlap == pytest.approx(similarity)
    assert metrics.residual > 0
    assert metrics.novelty == pytest.approx(1.0 - similarity)


def test_generative_operator_rejects_colinear_successors():
    constant_seed = np.ones(8, dtype=complex)

    with pytest.raises(ValidationError):
        generative_operator(constant_seed, alpha=0.1, novelty=0.0, phase=0.0)
    with pytest.raises(ValidationError):
        generative_operator(constant_seed, alpha=0.1, novelty=0.2, phase=np.pi / 7)
    with pytest.raises(ValidationError):
        _validate_nontrivial_successor(np.ones(3, dtype=complex), np.zeros(3, dtype=complex))


def test_replication_sequence_remains_bounded():
    _, seed = normalized_seed()

    sequence = replication_sequence(seed, steps=5, alpha=0.08, novelty=0.1)
    metrics = sequence_metrics(sequence)

    assert len(sequence) == 6
    assert len(metrics) == 5
    for state in sequence:
        assert np.linalg.norm(state) == pytest.approx(np.linalg.norm(seed))
    for item in metrics:
        assert item.overlap > 0.8
        assert 0.0 <= item.novelty < 0.2


def test_replicate_wavefunction_preserves_metadata_and_marks_generation():
    grid, seed = normalized_seed()
    parent = WaveFunction(
        psi=seed,
        grid=grid,
        mass=125.0,
        quantum_numbers={"n": 0, "generation": 2},
    )

    child = replicate_wavefunction(parent, alpha=0.1, novelty=0.12)

    assert isinstance(child, WaveFunction)
    assert child.mass == parent.mass
    assert np.array_equal(child.grid, parent.grid)
    assert child.quantum_numbers["n"] == 0
    assert child.quantum_numbers["generation"] == 3
    assert 0.0 <= child.quantum_numbers["generative_residual"]
    assert 0.85 < child.quantum_numbers["generative_overlap"] <= 1.0


def test_replicate_field_reports_parent_child_metrics():
    _, seed = normalized_seed()

    result = replicate_field(seed, alpha=0.1, novelty=0.1)

    assert result.parent.shape == result.child.shape
    assert result.metrics.parent_norm == pytest.approx(result.metrics.child_norm)
    assert result.metrics.overlap == pytest.approx(
        self_similarity(result.parent, result.child)
    )


def test_invalid_replication_parameters_raise_validation_error():
    _, seed = normalized_seed()

    with pytest.raises(ValidationError):
        generative_operator(seed, alpha=1.0)
    with pytest.raises(ValidationError):
        generative_operator(seed, novelty=-0.1)
    with pytest.raises(ValidationError):
        replication_sequence(seed, steps=-1)
