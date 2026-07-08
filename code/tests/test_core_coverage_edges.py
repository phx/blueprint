import builtins

import numpy as np
import pytest
from sympy import Symbol, oo

from core import self_replication as sr
from core.basis import FractalBasis
from core.compute import (
    ALLOWED_PROCESSES,
    compute_amplitude,
    compute_branching_ratio,
    compute_correlation_function,
    compute_cross_section,
    compute_matrix_element_squared,
    compute_phase_space,
    compute_phase_space_integral,
    compute_total_width,
)
from core.contexts import lorentz_boost, quantum_state
from core.detector import Detector
from core.errors import ComputationError, PhysicsError, StabilityError, ValidationError
from core.field import UnifiedField, validate_wavefunction as validate_field_wavefunction
from core.numeric import integrate_phase_space, solve_field_equations
from core.precision import validate_precision
from core.stability import analyze_perturbation, check_convergence, verify_error_bounds
from core.transforms import gauge_transform, lorentz_boost as matrix_lorentz_boost
from core.types import (
    BasisConfig,
    BranchingRatio,
    ComplexValue,
    CrossSection,
    Energy,
    ErrorEstimate,
    FieldConfig,
    FractalMode,
    Momentum,
    NumericValue,
    RealValue,
    WaveFunction,
    ensure_numeric_value,
)
from core.utils import (
    batch_process,
    cached_evaluation,
    check_numerical_stability,
    evaluate_expr,
    get_memory_usage,
    profile_computation,
    propagate_errors,
    stability_check,
)
from core.validation import validate_config, validate_numeric_range, validate_wavefunction
from core.version import Version, VersionManager


def normalized_wavefunction(length=64, start=-10.0, stop=10.0, mass=1.0):
    grid = np.linspace(start, stop, length)
    psi = np.exp(-grid ** 2 / 2).astype(complex)
    wf = WaveFunction(psi=psi, grid=grid, mass=mass, quantum_numbers={"n": 0, "E": 10.0})
    wf.normalize()
    return wf


def test_type_edges_cover_units_uncertainty_and_validation():
    assert (RealValue(1.0) + RealValue(2.0)).uncertainty is None
    assert (RealValue(2.0) * RealValue(3.0)).uncertainty is None
    assert float(RealValue(3.5)) == 3.5

    with pytest.raises(ValueError):
        RealValue("bad")
    with pytest.raises(ValueError):
        RealValue(1.0, NumericValue(float("inf")))
    assert RealValue(NumericValue(1.0), NumericValue(0.1)).uncertainty == 0.1
    with pytest.raises(ValueError):
        RealValue(1.0, "bad")
    with pytest.raises(ValueError):
        RealValue(1.0, -1.0)
    with pytest.raises(ValueError):
        RealValue(1.0, float("inf"))

    z = ComplexValue(1 + 1j, 0.2)
    assert np.isclose(z.magnitude, np.sqrt(2))
    assert np.isclose(z.phase, np.pi / 4)
    assert abs(z).uncertainty == 0.2
    assert abs(ComplexValue(3 + 4j)).value == 5.0
    assert z.conjugate().value == 1 - 1j
    with pytest.raises(ValueError):
        ComplexValue(object())
    with pytest.raises(ValueError):
        ComplexValue(float("inf"))
    with pytest.raises(ValueError):
        ComplexValue(1 + 0j, -1)
    with pytest.raises(ValueError):
        ComplexValue(1 + 0j, "wide")
    with pytest.raises(ValueError):
        ComplexValue(1 + 0j, float("inf"))

    e = Energy(10.0, 1.0)
    assert (e / 2).value == 5.0
    assert (e / Energy(2.0)) == 5.0
    assert (20 / e).value == 2.0
    assert e == 10.0
    assert e == Energy(10.0)
    assert e < 20.0 and e < Energy(20.0)
    assert e > 5.0 and e > Energy(5.0)
    assert e.to_units("GeV").value == 10.0
    assert e.__rtruediv__("bad") is NotImplemented
    assert e.__eq__("bad") is NotImplemented
    assert e.__lt__("bad") is NotImplemented
    assert e.__gt__("bad") is NotImplemented
    with pytest.raises(TypeError):
        Energy("hot")
    with pytest.raises(ValueError):
        e / Energy(1.0, units="TeV")
    with pytest.raises(TypeError):
        e / "bad"
    with pytest.raises(ValueError):
        e == Energy(10.0, units="TeV")
    with pytest.raises(ValueError):
        e < Energy(20.0, units="TeV")
    with pytest.raises(ValueError):
        e > Energy(5.0, units="TeV")
    with pytest.raises(ValueError):
        e.to_units("joule")

    p = Momentum(3.0, 0.3)
    assert p.to_units("MeV/c").value == 3000.0
    assert (p + Momentum(2.0, 0.4)).value == 5.0
    assert (p * 2).value == 6.0
    assert (p * Momentum(2.0, 0.1)).value == 6.0
    assert (Momentum(2.0) * Momentum(3.0)).uncertainty is None
    with pytest.raises(ValueError):
        Momentum(1.0, units="kg m/s")
    with pytest.raises(ValueError):
        p.to_units("kg m/s")
    with pytest.raises(TypeError):
        p + 1
    with pytest.raises(ValueError):
        p + Momentum(1.0, units="TeV/c")
    with pytest.raises(ValueError):
        p * Momentum(1.0, units="TeV/c")
    with pytest.raises(TypeError):
        p * "bad"

    xs = CrossSection(2.0, 0.2)
    assert xs.to_units("fb").value == 2000.0
    assert CrossSection(2.0, units="fb").to_units("pb").value == 0.002
    assert xs.to_units("nb").value == 0.002
    assert CrossSection(2.0, units="nb").to_units("pb").value == 2000.0
    with pytest.raises(ValueError):
        CrossSection(-1.0)
    with pytest.raises(ValueError):
        xs.to_units("barn")

    assert BranchingRatio(0.5, process="Z->ee").process == "Z->ee"
    with pytest.raises(ValueError):
        BranchingRatio(1.5)

    estimate = ErrorEstimate(10.0, 3.0)
    assert estimate.systematic == {}
    rendered_error = str(ErrorEstimate(10.0, 3.0, {"cal": 4.0}))
    assert "10" in rendered_error and "5" in rendered_error


def test_numeric_and_wavefunction_compatibility_edges():
    original = NumericValue(2.0, 0.2)
    copied = NumericValue(original)
    assert copied.value == 2.0
    assert NumericValue(np.array([3.0])).value == 3.0
    assert NumericValue(np.float64(4.0)).value == 4.0
    assert NumericValue(3 + 4j).real == 3.0
    assert NumericValue(3 + 4j).imag == 4.0
    assert NumericValue(3 + 4j).conjugate().value == 3 - 4j
    assert NumericValue(4.0).conjugate() is not None
    assert NumericValue(0.0, 0.1).relative_uncertainty is None
    assert NumericValue(1.0, None).relative_uncertainty is None
    assert (NumericValue(1.0) + NumericValue(2.0)).uncertainty is None
    assert (NumericValue(1.0) - NumericValue(2.0)).uncertainty is None
    assert (NumericValue(2.0) * NumericValue(3.0)).uncertainty is None
    assert (NumericValue(6.0) / NumericValue(3.0)).uncertainty is None
    assert (12 / NumericValue(3.0)).value == 4.0
    assert (5 - NumericValue(2.0)).value == 3.0
    assert np.array(NumericValue(2.0)).item() == 2.0
    assert NumericValue(2.0) ** NumericValue(3.0) == NumericValue(8.0)
    assert ensure_numeric_value(NumericValue(5.0)).value == 5.0
    with pytest.raises(TypeError):
        NumericValue(np.array([1.0, 2.0]))
    with pytest.raises(ValueError):
        NumericValue(1 + complex(0, float("inf")))
    with pytest.raises(ValueError):
        NumericValue(1.0, float("inf"))
    with pytest.raises(ValueError):
        NumericValue(5.0, valid_range=(0.0, 1.0))
    with pytest.raises(ZeroDivisionError):
        NumericValue(1.0) / 0

    config = FieldConfig(parameters={"mass": 200.0}, alpha=0.2, coupling=0.3)
    assert config.to_dict()["mass"] == 200.0
    for kwargs in (
        {"alpha": 0.0},
        {"mass": 0.0},
        {"coupling": 0.0},
        {"dimension": 0},
        {"max_level": 0},
        {"precision": 0.0},
    ):
        with pytest.raises(ValueError):
            FieldConfig(**kwargs)
    scalar = WaveFunction(1.0, grid=np.array([0.0]), mass=1.0)
    assert scalar.norm == 1.0
    wf = normalized_wavefunction()
    assert wf.evaluate_at(0.0).real > 0
    product = wf * wf
    assert product.quantum_numbers["n"] == 0
    scaled = wf * 2
    assert np.allclose(scaled.psi, wf.psi * 2)
    with pytest.raises(ValidationError):
        wf * WaveFunction(wf.psi[:-1], wf.grid[:-1], mass=1.0)
    with pytest.raises(ZeroDivisionError):
        wf / 0
    assert (wf / 2).norm > 0
    with pytest.raises(ValueError):
        WaveFunction([1, 2], grid=[0])
    with pytest.raises(ValueError):
        WaveFunction([np.inf], grid=[0])

    x = Symbol("x")
    expr_wf = WaveFunction.from_expression(x - x, grid=np.linspace(-1, 1, 5))
    assert expr_wf.norm > 0


def test_fractal_mode_and_basis_config_edges():
    grid = np.linspace(-3, 3, 5)
    mode = FractalMode(psi=np.ones(5, dtype=complex), grid=grid, n=0, alpha=0.1)
    assert mode.n == 0
    with pytest.raises(TypeError):
        FractalMode(psi=[1, 2], grid=grid, n=0, alpha=0.1)
    with pytest.raises(TypeError):
        FractalMode(psi=np.ones(5), grid=[-3, 3], n=0, alpha=0.1)
    with pytest.raises(ValueError):
        FractalMode(psi=np.ones(4), grid=grid, n=0, alpha=0.1)
    with pytest.raises(ValueError):
        FractalMode(psi=np.ones(5), grid=np.linspace(-2, 2, 5), n=0, alpha=0.1)
    with pytest.raises(ValueError):
        FractalMode(psi=np.ones(5), grid=grid, n=0, alpha=1.0)

    assert BasisConfig(dimension=4).scaling_dimension == 1.0
    with pytest.raises(ValueError):
        BasisConfig(dimension=0)
    with pytest.raises(ValueError):
        BasisConfig(precision=0.0)
    with pytest.raises(ValueError):
        BasisConfig(max_level=0)

    with pytest.raises(ValueError):
        FractalBasis().project_state("not a wavefunction")


def test_compute_module_edges():
    assert compute_phase_space(1.0, {"threshold": 1.0}) == 0.0
    assert compute_matrix_element_squared(3.0, {"spin_avg": 3}) == 3.0
    widths = compute_total_width(100.0, {name: 0.1 for name in ALLOWED_PROCESSES})
    assert widths > 0

    wf = normalized_wavefunction()
    phase_integral = compute_phase_space_integral(wf, Energy(10.0))
    assert np.isclose(phase_integral.value, 1.0)
    assert compute_amplitude(wf, Energy(10.0)) != 0
    assert compute_cross_section(10.0, 1.0, wf).value > 0
    assert compute_branching_ratio("Z->ee", Energy(1.0), {"Z->ee": 0.1, "Z->mumu": 0.1}) > 0
    with pytest.raises(ComputationError):
        compute_cross_section(-1.0, 1.0, wf)
    with pytest.raises(ComputationError):
        compute_correlation_function(0.0, "bad", Energy(1.0), wf)
    with pytest.raises(PhysicsError):
        compute_branching_ratio("missing", Energy(100.0), {})
    with pytest.raises(PhysicsError):
        compute_branching_ratio("Z->tautau", Energy(1.0), {"Z->tautau": 0.1})
    with pytest.raises(PhysicsError):
        compute_branching_ratio("Z->ee", Energy(100.0), {})


def test_detector_edges(monkeypatch):
    with pytest.raises(PhysicsError):
        Detector(resolution={"energy": 0.1})
    with pytest.raises(PhysicsError):
        Detector(resolution={"energy": "sharp", "position": 0.1})
    with pytest.raises(PhysicsError):
        Detector(resolution={"energy": 0.0, "position": 0.1})
    with pytest.raises(PhysicsError):
        Detector(acceptance={"eta": 2.5, "pt": 1.0})
    with pytest.raises(PhysicsError):
        Detector(acceptance={"eta": (-1, 1)})
    with pytest.raises(PhysicsError):
        Detector(acceptance={"eta": (-1, 1), "pt": "fast"})
    with pytest.raises(ValidationError):
        Detector(threshold="high")
    with pytest.raises(ValidationError):
        Detector(threshold=0)
    with pytest.raises(ValidationError):
        Detector(efficiency="full")
    with pytest.raises(ValidationError):
        Detector(efficiency=2)
    with pytest.raises(ValidationError):
        Detector(systematics=[])
    with pytest.raises(ValidationError):
        Detector(systematics={"energy_scale": "wide"})
    with pytest.raises(ValidationError):
        Detector(systematics={"energy_scale": -0.1})

    detector = Detector()
    momentum_measurement = detector.simulate_measurement(Momentum(20.0))
    assert momentum_measurement["momentum"].value == 20.0
    with pytest.raises(PhysicsError):
        detector.simulate_measurement("bad")

    wf = normalized_wavefunction()
    assert detector.compute_efficiency(wf).value == detector.efficiency
    assert detector.compute_efficiency(20.0, 3.0).value == 0.0
    assert detector.compute_efficiency(5.0, 0.0).value > 0.0
    assert detector.compute_efficiency(20.0).value > 0.0
    with pytest.raises(PhysicsError):
        detector.check_acceptance(float("nan"), 0.0)
    with pytest.raises(PhysicsError):
        detector.check_acceptance(1.0, float("nan"))
    assert detector.trigger(detector.threshold)

    monkeypatch.setattr(np.random, "normal", lambda *args, **kwargs: 0.0)
    assert detector.measure_energy(Energy(100.0)) == 100.0
    assert detector.measure_cross_section(10.0, efficiency=0.5) == 5.0
    wf.quantum_numbers["E"] = Energy(42.0)
    assert detector.reconstruct_energy(wf).value == 42.0


def test_field_facade_edges():
    field = UnifiedField()
    with pytest.raises(ValueError):
        UnifiedField(alpha=1.0)

    assert field.compute_rg_flow(Energy(10.0))["g1"].uncertainty == 0.005
    assert field.compute_couplings(Energy(1e16))["g1"].uncertainty == 0.05
    assert np.asarray(field.compute_coupling(1, np.array([1.0, 2.0]))).shape == (2,)
    assert field.compute_coupling(1, NumericValue(1.0)) > 0

    assert field.compute_energy_density(1).value == 1.0
    transformed_symbolic = field.apply_lorentz_transform(1, 0.5)
    assert transformed_symbolic == 1
    assert field.compute_energy_density(transformed_symbolic).value > 1.0

    wf = field.compute_basis_state()
    field.state = wf
    assert isinstance(field.evolve(Energy(1.0)), WaveFunction)
    field.state = "state"
    assert field.evolve(Energy(1.0)) == "state"
    assert field.evolve_field(2.0) == 2.0
    assert field.compute_correlator(wf, [(0, 0)]) == 0
    assert field.compute_correlator("not wf", [(0, 0), (1, 0)]).is_real
    assert field.compute_correlator(wf, [("bad", 0), (1, 0)]).is_real
    assert field.compute_correlator(wf, [(0, 0), (0.5, 0)]).real > 0
    field_config_state = field.compute_field(FieldConfig())
    assert field.compute_correlator(field_config_state, [(0, 0), (1, 0)]) == 0

    assert field.reconstruct_from_expansion({}, []).norm > 0
    assert field.validate_predictions()["M_GUT"].value == 0.0
    assert field.compute_spectral_function(wf) == 1.0
    assert field.compute_spectral_function(wf, 0.1) == 0.0
    field_config_state.quantum_numbers["field_config"] = True
    assert field.compute_basis_coefficient(field_config_state, 0) == 0.0
    assert field.compute_basis_coefficient(wf, 1) == 0.0

    grid = np.linspace(-3, 3, 5)
    dx = grid[1] - grid[0]
    good = WaveFunction(np.full(5, np.sqrt(1.0 / (5 * dx)), dtype=complex), grid, mass=1.0)
    validate_field_wavefunction(good)
    with pytest.raises(ValidationError):
        validate_field_wavefunction("bad")
    with pytest.raises(ValidationError):
        validate_field_wavefunction(normalized_wavefunction())
    bad_norm = WaveFunction(np.ones(5, dtype=complex), np.linspace(-3, 3, 5), mass=1.0)
    with pytest.raises(ValidationError):
        validate_field_wavefunction(bad_norm)


def test_numeric_precision_stability_and_validation_edges(monkeypatch):
    x = Symbol("x")
    assert np.isclose(integrate_phase_space(lambda y: y, (0.0, 1.0)).value, 0.5)
    assert integrate_phase_space(lambda a, b, c: a + b + c, [(0, 1), (0, 1), (0, 1)]).value > 0
    with pytest.raises(ComputationError):
        integrate_phase_space(x, [(0, 1), (0, 1)])
    with pytest.raises(ComputationError):
        integrate_phase_space(lambda a, b: a + b, [(0, 1), (2, 1)])
    with pytest.raises(ComputationError):
        solve_field_equations({"mass": -1.0, "coupling": 0.1}, Energy(1.0))
    with pytest.raises(ComputationError):
        solve_field_equations({"mass": 1.0, "coupling": 0.1}, Energy(1.0), max_iter=0)

    assert validate_precision(NumericValue(1.0, 1e-9), 1.0)
    assert validate_precision(NumericValue(1.0 + 0j, 1e-9), 1.0 + 0j)

    def unstable_after_nominal(x):
        if x != 1.0:
            raise RuntimeError("perturbed")
        return NumericValue(x)

    with pytest.raises(StabilityError):
        analyze_perturbation(unstable_after_nominal, {"x": 1.0}, n_samples=2)
    assert check_convergence([NumericValue(1.0)]) is False
    assert verify_error_bounds(1.0, 1.0, [0.9, 1.0, 1.1])

    with pytest.raises(ComputationError):
        evaluate_expr(1 / x, {"x": 0}, check_finite=True)
    with pytest.raises(ComputationError):
        evaluate_expr(x, {"x": float("inf")}, check_finite=True)

    @cached_evaluation
    def fails(value):
        raise RuntimeError(value)

    with pytest.raises(ComputationError):
        fails(1)

    with pytest.raises(StabilityError):
        check_numerical_stability(lambda: np.inf)

    class BadFinite:
        uncertainty = 0.0

        def __array__(self, dtype=None):
            raise TypeError("not arrayable")

    with pytest.raises(StabilityError):
        check_numerical_stability(lambda: BadFinite())
    assert check_numerical_stability(BadFinite()) is False
    assert check_numerical_stability(np.array([1.0, np.inf])) is False
    assert check_numerical_stability(np.array([1e20]), threshold=1e-10) is False
    assert check_numerical_stability(np.array([1e-12]), threshold=1e-10) is False

    @stability_check()
    def stable_value():
        return 1.0

    assert stable_value() == 1.0

    @stability_check()
    def unstable_value():
        return np.inf

    with pytest.raises(StabilityError):
        unstable_value()

    assert propagate_errors([1.0, 2.0], [0.1, 0.2]).uncertainty > 0
    corr = np.eye(2)
    assert propagate_errors([1.0, 2.0], [0.1, 0.2], corr).value == 3.0
    with pytest.raises(PhysicsError):
        propagate_errors([1.0], [0.1, 0.2])
    with pytest.raises(PhysicsError):
        propagate_errors([1.0], [0.1], np.eye(2))

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "psutil":
            raise ImportError("no psutil")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    assert get_memory_usage() == -1.0
    assert batch_process([1, 2, 3], 2, lambda batch: [sum(batch)]) == [3, 3]

    @profile_computation
    def identity(value):
        return value

    assert identity(1) == 1
    assert identity(2) == 2
    assert identity.get_profile()["call_count"] == 2

    assert validate_numeric_range(1, 0, 2) == 1.0
    with pytest.raises(ValidationError):
        validate_numeric_range("one", 0, 2)
    with pytest.raises(ValidationError):
        validate_numeric_range(3, 0, 2)
    with pytest.raises(ValidationError):
        validate_config([], ["x"])
    with pytest.raises(ValidationError):
        validate_config({}, ["x"])
    assert isinstance(validate_wavefunction(np.ones(3)), WaveFunction)


def test_self_replication_error_edges():
    assert sr.field_norm(np.array([1.0, 0.0, -1.0])) > 0
    with pytest.raises(ValidationError):
        sr.field_norm(np.ones((2, 2)))
    with pytest.raises(ValidationError):
        sr.field_norm(np.ones(2))
    with pytest.raises(ValidationError):
        sr.field_norm(np.array([1.0, np.inf, 2.0]))
    with pytest.raises(ValidationError):
        sr.generative_operator(np.ones(3), embodiment=-1.0)
    with pytest.raises(ValidationError):
        sr.normalize_like(np.zeros(3), 1.0)
    with pytest.raises(ValidationError):
        sr.self_similarity(np.ones(3), np.ones(4))
    with pytest.raises(ValidationError):
        sr.self_similarity(np.zeros(3), np.ones(3))
    with pytest.raises(ValidationError):
        sr.generative_operator(np.zeros(3))
    with pytest.raises(ValidationError):
        sr.replication_metrics(np.ones(3), np.ones(4))
    with pytest.raises(ValidationError):
        sr.replication_metrics(np.zeros(3), np.ones(3))
    with pytest.raises(ValidationError):
        sr.replicate_wavefunction(np.ones(3))

    sequence = sr.replication_sequence(np.array([1.0, 0.5, -0.25]), steps=1)
    assert len(sr.sequence_metrics(sequence)) == 1


def test_context_transform_and_version_edges(tmp_path):
    with lorentz_boost(0.0) as boost:
        beta, gamma = boost
        assert beta == 0.0 and gamma == 1.0
        assert boost.transform((1.0, 2.0))[0] == 1.0
    with pytest.raises(PhysicsError):
        with lorentz_boost(1.0, 1.0):
            pass
    with lorentz_boost(-0.9) as edge_boost:
        assert edge_boost.beta == -0.9
    with quantum_state(energy=2.0) as (state, norm):
        assert state.quantum_numbers["E"].value == 2.0
        assert norm == 10.0

    assert matrix_lorentz_boost(0).shape == (2, 2)
    assert gauge_transform(0) == 1

    with pytest.raises(Exception):
        VersionManager(tmp_path)
    assert str(Version(1, 2, 3)) == "1.2.3"
    assert str(Version(1, 2, 3, "alpha", "001")) == "1.2.3-alpha+001"
    assert Version.from_string("1.2.3-alpha").pre_release == "alpha"
    assert Version.from_string("1.2.3+001").build == "001"

    (tmp_path / "version.json").write_text('{"core": "1.2.3", "dep": "1.2.0"}')
    with pytest.raises(Exception):
        VersionManager(tmp_path)

    (tmp_path / "compatibility.json").write_text("{bad json")
    with pytest.raises(Exception):
        VersionManager(tmp_path)

    (tmp_path / "compatibility.json").write_text('{"core": ["dep"]}')
    manager = VersionManager(tmp_path)
    with pytest.raises(Exception):
        manager.validate_component("missing", "1.0.0")
    manager.validate_component = lambda component, version: False
    assert manager.check_compatibility("core") is False
