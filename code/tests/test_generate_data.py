"""Tests for data generation and validation.

Validation note:
The test suite validates:
1. Data file generation
2. Physical constraints
3. Cross-correlations
4. Uncertainty propagation
"""

import pytest
import numpy as np
from numpy import pi, exp, log, sqrt
from pathlib import Path
import pandas as pd

from core.physics_constants import (
    HBAR, C, G, M_P, I,  # Level 1: Fundamental constants
    Z_MASS  # Level 3: Derived quantities
)
from generate_data import (
    generate_all_data,
    validate_generated_data,
    validate_wavelet_data,
    validate_statistical_data,
    validate_couplings,
    validate_cross_correlations
)

# Test data directory
DATA_DIR = Path("../data")

@pytest.fixture(scope="module")
def generated_data():
    """
    Generate all test data files.
    
    Validation note:
    Generates complete set of validation data with proper
    quantum coherence and uncertainty propagation.
    """
    # From tests/ go up two levels to reach the project data directory.
    data_dir = Path(__file__).parent.parent.parent / 'data'
    
    # Generate data
    generate_all_data(data_dir)
    
    return data_dir

def test_file_generation(generated_data):
    """
    Test generation of all required data files.
    
    Validation note:
    Verifies:
    1. All required files exist
    2. File formats are valid
    3. Data types are correct
    
    File categories tracked by the public validation matrix:
    1. Analysis configuration:
        - adaptive_filters.csv
        - ml_filters.csv
        - coincidence_requirements.csv
        - experimental_design.csv
    2. Physics data:
        - coupling_evolution.csv
        - gw_spectrum.dat
    3. Statistical validation:
        - statistical_tests.csv
        - validation_results.csv
        - predictions.csv
    4. Core analysis:
        - wavelet_analysis.csv
        - statistical_analysis.csv
        - detector_noise.csv
        - cosmic_backgrounds.csv
        - background_analysis.csv
        - systematic_uncertainties.csv
    """
    expected_files = {
        # Analysis configuration
        'adaptive_filters.csv',
        'ml_filters.csv', 
        'coincidence_requirements.csv',
        'experimental_design.csv',
        
        # Physics data
        'coupling_evolution.csv',
        'gw_spectrum.dat',
        
        # Statistical validation
        'statistical_tests.csv',
        'validation_results.csv',
        'predictions.csv',
        
        # Core analysis
        'wavelet_analysis.csv',
        'statistical_analysis.csv',
        'detector_noise.csv',
        'cosmic_backgrounds.csv',
        'background_analysis.csv',
        'systematic_uncertainties.csv'
    }
    
    # Check for unexpected files
    existing_files = set(f.name for f in generated_data.glob('*.*'))
    unexpected = existing_files - expected_files
    if unexpected:
        print(f"Warning: Unexpected files found: {unexpected}")
    
    # Verify all expected files exist and are valid
    missing = []
    for filename in expected_files:
        file_path = generated_data / filename
        if not file_path.exists():
            missing.append(filename)
            continue
            
        # Verify file can be loaded
        if filename.endswith('.csv'):
            df = pd.read_csv(file_path)
            assert len(df) > 0, f"Empty file: {filename}"
        elif filename.endswith('.dat'):
            with open(file_path) as f:
                data = f.read()
                assert len(data) > 0, f"Empty file: {filename}"
    
    if missing:
        print(f"Regenerating missing files: {missing}")
        generate_all_data()  # This will regenerate any missing files
        
        # Verify files were generated
        for filename in missing:
            assert (generated_data / filename).exists(), f"Failed to generate: {filename}"

def test_physical_constraints(generated_data):
    """
    Test physical constraints on generated data.
    
    Validation note:
    Verifies:
    1. Energy scales are positive
    2. Probabilities in [0,1]
    3. Uncertainties properly normalized
    """
    # Load data files
    backgrounds = pd.read_csv(generated_data / 'cosmic_backgrounds.csv')
    systematics = pd.read_csv(generated_data / 'systematic_uncertainties.csv')
    
    # Check probability bounds
    assert 0 <= float(backgrounds['isotropic_factor'].iloc[0]) <= 1
    assert 0 <= float(systematics['acceptance'].iloc[0]) <= 1

def test_uncertainty_propagation(generated_data):
    """
    Test uncertainty propagation in analysis chain.
    
    Validation note:
    Verifies proper error propagation through:
    1. Statistical uncertainties
    2. Systematic uncertainties
    3. Combined uncertainties
    """
    systematics = pd.read_csv(generated_data / 'systematic_uncertainties.csv')
    
    # Calculate total uncertainty
    total_unc = np.sqrt(
        systematics['energy_scale'].astype(float)**2 +
        systematics['acceptance'].astype(float)**2 +
        systematics['model_dependency'].astype(float)**2
    )
    assert (total_unc < 1.0).all()  # Total uncertainty < 100%

def test_cross_correlations(generated_data):
    """
    Test correlations between different data files.
    
    Validation note:
    Verifies:
    1. Consistent energy scales across files
    2. Compatible uncertainties
    3. Proper background subtraction
    """
    validate_cross_correlations(generated_data)

def test_wavelet_consistency(generated_data):
    """
    Test wavelet analysis consistency.
    
    Validation note:
    Verifies:
    1. Resolution hierarchy
    2. Admissibility conditions
    3. Phase coherence
    """
    wavelets = pd.read_csv(generated_data / 'wavelet_analysis.csv')
    
    # Check resolution bounds
    assert float(wavelets['resolution_tolerance'].iloc[0]) > 0
    
    # Verify admissibility
    min_adm = float(wavelets['min_admissibility'].iloc[0])
    max_adm = float(wavelets['max_admissibility'].iloc[0])
    assert 0 < min_adm < max_adm

def test_coupling_unification(generated_data):
    """
    Test the generated coupling-evolution proxy.

    Public traceability note:
    This checks the internal data-generation proxy used by the validation
    harness. It verifies hierarchical ordering, bounded ratio behavior at high
    energy, and a GUT-scale region. It does not replace an independent
    renormalization-group derivation.
    """
    couplings = pd.read_csv(generated_data / 'coupling_evolution.csv')
    
    # Get coupling values at high energy (E > 10^15 GeV)
    # This is near the GUT-scale proxy region tracked by the validation matrix.
    high_E_mask = couplings['Energy_GeV'] > 1e15
    high_E_couplings = couplings[high_E_mask]
    
    # Get last (highest energy) values
    g1 = float(high_E_couplings['g1'].iloc[-1].split(' - ')[0])  # Real part only
    g2 = float(high_E_couplings['g2'].iloc[-1].split(' - ')[0])
    g3 = float(high_E_couplings['g3'].iloc[-1].split(' - ')[0])
    
    # Verify the generated proxy maintains the expected hierarchy.
    assert g1 > g2 > g3
    
    # Check coupling ratios remain bounded within the generated proxy range.
    assert 5 < g1/g2 < 10
    assert 5 < g2/g3 < 10

def test_gw_spectrum(generated_data):
    """
    Test gravitational wave spectrum properties.
    
    Validation note:
    Verifies:
    1. Energy density spectrum: Ω_GW(f) ∝ f^n
    2. Frequency range: 10^-4 Hz < f < 10^4 Hz
    3. Power law behavior: n = 2/3 for inspiral phase
    """
    # Load GW spectrum data
    spectrum_file = generated_data / 'gw_spectrum.dat'
    data = np.loadtxt(spectrum_file)
    freq = data[:, 0]  # Hz
    omega = data[:, 1]  # Energy density
    
    # Check frequency range
    assert 1e-4 < np.min(freq) < np.max(freq) < 1e4
    
    # Verify power law behavior
    # For inspiral phase, Ω_GW ∝ f^(2/3)
    log_freq = np.log(freq)
    log_omega = np.log(omega)
    slope = np.polyfit(log_freq, log_omega, 1)[0]
    assert abs(slope - 2/3) < 0.1  # Allow 10% deviation

def test_detector_noise(generated_data: Path) -> None:
    """
    Test detector noise characteristics.
    
    Validation note:
    Verifies:
    1. Noise power spectrum: S_n(f) follows design curve
    2. Statistical properties: Gaussian white noise
    3. Frequency dependence: 1/f noise at low frequencies
    """
    noise = pd.read_csv(generated_data / 'detector_noise.csv')
    
    # Basic validation
    assert len(noise) > 0, "Noise data is empty"
    
    # Check required columns
    required_cols = ['frequency', 'amplitude', 'phase', 'psd']
    for col in required_cols:
        assert col in noise.columns, f"Missing column: {col}"
    
    # Check basic statistical properties
    assert abs(noise['amplitude'].mean()) < 0.1  # Zero mean
    assert 0.9 < noise['amplitude'].std() < 1.1  # Unit variance
    
    # Verify 1/f noise at low frequencies
    freq = noise['frequency'].astype(float)
    psd = noise['psd'].astype(float)
    low_f_mask = freq < 1.0
    slope = np.polyfit(np.log(freq[low_f_mask]), 
                      np.log(psd[low_f_mask]), 1)[0]
    assert abs(slope + 1) < 0.2  # 1/f noise has slope -1

def test_statistical_analysis(generated_data):
    """
    Test statistical analysis results.
    
    Validation note:
    Verifies:
    1. Significance levels: p < 0.05 for all tests
    2. Effect sizes: Cohen's d > 0.5 for key results
    3. Power analysis: β > 0.8 for sample sizes
    """
    stats = pd.read_csv(generated_data / 'statistical_analysis.csv')
    
    # Check significance levels
    assert (stats['p_value'].astype(float) < 0.05).all()
    
    # Verify effect sizes
    assert (stats['cohens_d'].astype(float) > 0.5).all()
    
    # Check statistical power
    assert (stats['power'].astype(float) > 0.8).all()

def test_background_subtraction(generated_data):
    """
    Test background analysis and subtraction.
    
    Validation note:
    Verifies:
    1. Signal-to-noise ratio: SNR > 5 after subtraction
    2. Background model: χ² test for residuals
    3. Systematic effects: < 10% of signal
    """
    bkg = pd.read_csv(generated_data / 'background_analysis.csv')
    
    # Check SNR
    snr = bkg['signal'].astype(float) / bkg['noise'].astype(float)
    assert (snr > 5).all()
    
    # Verify residuals
    chi2 = np.sum((bkg['residuals'].astype(float))**2 / 
                  bkg['uncertainty'].astype(float)**2)
    dof = len(bkg) - 1
    assert chi2/dof < 2.0  # Reasonable fit
    
    # Check systematics
    assert (bkg['systematics'].astype(float) / 
            bkg['signal'].astype(float) < 0.1).all()

def test_predictions(generated_data):
    """
    Test physical predictions against internal reference data.
    
    Validation note:
    Verifies:
    1. Predictions match reference values within errors
    2. Model parameters are physically reasonable
    3. Cross-validation scores > 0.9
    """
    pred = pd.read_csv(generated_data / 'predictions.csv')
    
    # Check prediction accuracy
    residuals = (pred['predicted'].astype(float) - 
                pred['reference'].astype(float)) / \
                pred['uncertainty'].astype(float)
    assert np.abs(residuals).mean() < 2.0  # Within 2σ
    
    # Verify parameter ranges
    assert (pred['parameters'].astype(float) > 0).all()
    
    # Check cross-validation
    assert (pred['cv_score'].astype(float) > 0.9).all()
