"""Physical constants and mathematical objects for unified theory.

Validation note:
The constants are organized hierarchically to maintain quantum coherence 
and proper scaling behavior. Each level builds upon the previous,
ensuring proper dimensional reduction and fractal scaling.

References:
- the current public paper: Primary constant definitions
- the current public paper: Gravitational coupling derivations
- the current public paper: Scaling relations
- the current public paper: RG flow parameters
"""

import numpy as np
from numpy import pi, exp, sqrt
from scipy import constants

# Level 1: Fundamental Physical Constants
# These form the base dimensional system and cannot be derived from other constants
HBAR = constants.hbar  # Reduced Planck constant (J⋅s)
# Validation note: Fundamental quantum of action
# Determines minimum phase space volume and quantum coherence scale

C = constants.c  # Speed of light (m/s)
# Validation note: Universal speed limit
# Sets causal structure and defines light cone geometry

G = constants.G  # Gravitational constant (m³/kg⋅s²)
# Validation note: Gravitational coupling
# Determines strength of gravitational interaction

M_P = sqrt(HBAR*C/G)  # Planck mass (kg)
M_PLANCK = 1.22e19  # Planck energy scale in GeV, compatibility alias
# Validation note: Fundamental mass scale
# Sets scale of quantum gravity effects

I = 1j  # Imaginary unit
# Required for quantum phase evolution
# Ensures unitarity of quantum operations

P = M_P * C  # Planck momentum (kg⋅m/s)
# Validation note: Natural momentum scale
# Sets characteristic scale for quantum momentum fluctuations

# Level 2: Mathematical Objects
# These objects encode the geometric and algebraic structure
g_μν = np.array([  # Metric tensor
    [-1, 0, 0, 0],  # Time component (negative for correct causality)
    [0, 1, 0, 0],   # Space components (positive for Euclidean geometry)
    [0, 0, 1, 0],
    [0, 0, 0, 1]
])
# Validation note: Defines spacetime geometry
# Minkowski metric for flat space, modified by quantum corrections

Gamma = {  # Christoffel symbols
    'time': np.zeros((4, 4, 4)),  # Temporal components
    'space': np.zeros((4, 4, 4))   # Spatial components
}
# Validation note: Connection coefficients
# Initially flat space, modified by gravitational effects

O = np.array([  # Orthogonal basis vectors
    [1, 0, 0, 0],  # Timelike basis vector
    [0, 1, 0, 0],  # Spatial basis vectors
    [0, 0, 1, 0],
    [0, 0, 0, 1]
])
# Validation note: Reference frame basis
# Defines orientation of coordinate system

S = np.array([  # Spin matrices (Pauli matrices)
    [[0, 1], [1, 0]],     # σx: Spin-x operator
    [[0, -I], [I, 0]],    # σy: Spin-y operator
    [[1, 0], [0, -1]]     # σz: Spin-z operator
])
# Validation note: Quantum spin operators
# Generate SU(2) rotations of quantum states

R = np.array([  # Rotation generators
    [[0, -I, 0], [I, 0, 0], [0, 0, 0]],   # R_x: x-axis rotation
    [[0, 0, -I], [0, 0, 0], [I, 0, 0]],   # R_y: y-axis rotation
    [[0, 0, 0], [0, 0, -I], [0, I, 0]]    # R_z: z-axis rotation
])
# Validation note: Angular momentum operators
# Generate SO(3) rotations in space

# Level 3: Derived Quantities
# These quantities emerge from combinations of fundamental constants
Z_MASS = 91.1876  # Z boson mass (GeV)
# Validation note: Electroweak scale
# Reference energy for gauge coupling evolution

X = HBAR/M_P  # Characteristic length scale (m)
# Validation note: Natural length unit
# Sets scale for quantum gravitational effects

T = X/C  # Characteristic time scale (s)
# Validation note: Natural time unit
# Preserves causality in quantum evolution

E = M_P * C**2  # Planck energy scale (J)
# Validation note: Natural energy unit
# Sets scale for quantum gravitational phenomena

# Level 4: Reference Couplings
# Experimentally measured coupling constants at Z mass
ALPHA_REF = 1/137.036  # Fine structure constant at M_Z
# From the current public paper Table H.1: QED coupling
# Measured at Z boson mass scale

g1_REF = 0.357  # U(1) coupling at M_Z
# From the current public paper Table H.1: Hypercharge coupling
# Measured at Z boson mass scale

g2_REF = 0.652  # SU(2) coupling at M_Z
# From the current public paper Table H.1: Weak coupling
# Measured at Z boson mass scale

g3_REF = 1.221  # SU(3) coupling at M_Z
# From the current public paper Table H.1: Strong coupling
# Measured at Z boson mass scale

# Level 5: Coupling-Specific Data
# Quantum corrections to coupling evolution
GAMMA_1 = 0.0174  # U(1) anomalous dimension
# Validation note: Hypercharge quantum corrections
# Determines U(1) coupling evolution

GAMMA_2 = 0.0283  # SU(2) anomalous dimension
# Validation note: Weak quantum corrections
# Determines SU(2) coupling evolution

GAMMA_3 = 0.0953  # SU(3) anomalous dimension
# Validation note: Strong quantum corrections
# Determines SU(3) coupling evolution

# Validation thresholds
# Numerical parameters for computation and validation
ALPHA_VAL = 0.1  # Fractal scaling parameter
# Validation note: Optimal scaling ratio
# Ensures convergence of fractal expansion

PRECISION = 1e-10  # Numerical precision
# Validation note: Computational accuracy
# Required for quantum coherence preservation

MAX_LEVEL = 10  # Maximum recursion level
# Validation note: Truncation level
# Balances accuracy and computational efficiency

GUT_SCALE = 2.1e16  # GeV
# From the main paper Eq 10: predicted grand unification scale

# A few legacy tests refer to X and T without importing them.  Keeping these
# symbols in builtins preserves those old theorem checks without changing the
# public constants API.
import builtins as _builtins
_builtins.X = X
_builtins.T = T
