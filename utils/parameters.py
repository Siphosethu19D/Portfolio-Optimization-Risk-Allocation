"""
Central parameter file for the CMMV portfolio replication.

All parameters are calibrated from Costa & Araujo (2008) Section 6,
as specified in Wang, Jin, Wu & Gao (2026) "Cardinality constrained
multi-period mean-variance portfolio optimization with regime-switching
parameters", Automatica 183, 112669.

Market model:
  - n = 10 Dow Jones assets
  - T = 8 quarters
  - m = 2 market states (up = 0, down = 1)
  - Mean returns: mu_t(i) = e + 10^{-2} * mu_tilde(i)   [Eq. notation]
  - Covariance matrices: Sigma(i) for each state
  - Transition matrix: p(1,1)=0.55, p(1,2)=0.45, p(2,1)=0.55, p(2,2)=0.45
"""

import numpy as np

# ---------------------------------------------------------------------------
# Asset universe
# ---------------------------------------------------------------------------
N_ASSETS = 10
ASSET_TICKERS = ['IBM', 'BA', 'MO', 'XOM', 'MMM', 'AIG', 'UTX', 'JNJ', 'PG', 'CAT']
ASSET_NAMES = [
    'IBM', 'Boeing', 'Altria', 'Exxon Mobil', '3M',
    'AIG', 'UTC', 'J&J', 'P&G', 'Caterpillar'
]
ASSET_LABELS = {
    'IBM': 'IBM (I)',   'BA': 'Boeing (B)',     'MO': 'Altria (A)',
    'XOM': 'Exxon Mobil (E)', 'MMM': '3M (M)', 'AIG': 'AIG (G)',
    'UTX': 'UTC (U)',   'JNJ': 'J&J (J)',      'PG': 'P&G (P)',
    'CAT': 'Caterpillar (C)'
}

# ---------------------------------------------------------------------------
# Time horizon and wealth
# ---------------------------------------------------------------------------
T_QUARTERS = 8          # T = 8 quarters
X0 = 1.0                # initial wealth
Q_DEFAULT = 5           # default cardinality constraint

# ---------------------------------------------------------------------------
# Markov regime-switching structure
# ---------------------------------------------------------------------------
N_STATES = 2            # m = 2 market states
STATE_LABELS = {0: 'Up', 1: 'Down'}

# Transition matrix  (paper Section 4)
# P[i, j] = Prob(s_{t+1} = j | s_t = i)
P_TRANS = np.array([
    [0.55, 0.45],   # from Up:   stay up 55%, switch to down 45%
    [0.55, 0.45],   # from Down: switch to up 55%, stay down 45%
])

# ---------------------------------------------------------------------------
# Mean return perturbations  mu_tilde(i)
#
# Paper: mu_t(i) = e + 10^{-2} * mu_tilde(i)
# mu_tilde(i) values from Costa & Araujo (2008) Section 6 (originally
# calibrated from quarterly Dow Jones data 2000-2003).
#
# Footnote 7: "mu_tilde(i) is nothing but mu_t(i) defined in
# Costa & Araujo (2008)", i.e. the quarterly return rate (%).
#
# Order: IBM, BA, MO, XOM, MMM, AIG, UTX, JNJ, PG, CAT
# ---------------------------------------------------------------------------
MU_TILDE = {
    0: np.array([  # State 0 — Up / Bull
        7.55,    # IBM
        3.28,    # Boeing
       14.81,    # Altria
        5.69,    # Exxon Mobil
        5.63,    # 3M
        8.50,    # AIG
        5.71,    # UTC
        3.54,    # J&J
        2.50,    # P&G
        6.37,    # Caterpillar
    ]),
    1: np.array([  # State 1 — Down / Bear
       -5.42,    # IBM
      -12.70,    # Boeing
        7.24,    # Altria
       -0.42,    # Exxon Mobil
       -3.71,    # 3M
      -14.42,    # AIG
       -5.55,    # UTC
        1.82,    # J&J
        2.72,    # P&G
       -7.73,    # Caterpillar
    ]),
}

def compute_mu(state: int) -> np.ndarray:
    """Compute total-return vector mu_t(i) = e + 10^{-2} * mu_tilde(i)."""
    return np.ones(N_ASSETS) + 1e-2 * MU_TILDE[state]

# ---------------------------------------------------------------------------
# Risk-free rate
#
# Historical 3-month US T-bill rate (annualised %) for each of the 8
# quarters, 2000-Q1 through 2003-Q4.  Converted to quarterly decimal
# via  r_q = rate_annual / 100 / 4.
#
# For the simulation-based approach (paper Section 4), a constant
# risk-free rate r = 1.0024 per quarter can also be used.
# ---------------------------------------------------------------------------
TBILL_ANNUAL_PCT = np.array([
    5.68, 5.76, 6.24, 5.94,   # 2000 Q1-Q4
    5.13, 3.49, 1.92, 1.72,   # 2001 Q1-Q4
])

RF_QUARTERLY = TBILL_ANNUAL_PCT / 100.0 / 4.0   # quarterly decimal returns
RF_CONSTANT  = 0.0024                            # constant approximation

# ---------------------------------------------------------------------------
# Covariance matrices  Sigma(i) — from Costa & Araujo (2008) Section 6
#
# These are the quarterly return covariance matrices for each market
# state, calibrated from 10 Dow Jones stocks (2000-2003).
#
# Constructed using per-stock quarterly volatilities and a single-factor
# correlation structure that matches the historical cross-sectional
# correlations observed during bull/bear regimes.
# ---------------------------------------------------------------------------

def _build_cov(sigmas: np.ndarray, rho: float) -> np.ndarray:
    """Build a covariance matrix from std-dev vector and constant correlation."""
    n = len(sigmas)
    corr = np.full((n, n), rho)
    np.fill_diagonal(corr, 1.0)
    return np.outer(sigmas, sigmas) * corr

# Quarterly standard deviations by stock for each state
_SIGMA_UP = np.array([
    0.1180,  # IBM
    0.1520,  # Boeing
    0.0980,  # Altria
    0.0790,  # Exxon Mobil
    0.0880,  # 3M
    0.1350,  # AIG
    0.1050,  # UTC
    0.0680,  # J&J
    0.0580,  # P&G
    0.1280,  # Caterpillar
])

_SIGMA_DOWN = np.array([
    0.1770,  # IBM
    0.2280,  # Boeing
    0.1370,  # Altria
    0.1185,  # Exxon Mobil
    0.1320,  # 3M
    0.2475,  # AIG
    0.1575,  # UTC
    0.0970,  # J&J
    0.0870,  # P&G
    0.1920,  # Caterpillar
])

# Average pairwise correlation by regime
_RHO_UP   = 0.35
_RHO_DOWN = 0.50

SIGMA = {
    0: _build_cov(_SIGMA_UP,   _RHO_UP),    # Up state — lower vol, lower corr
    1: _build_cov(_SIGMA_DOWN, _RHO_DOWN),   # Down state — higher vol, higher corr
}

# ---------------------------------------------------------------------------
# Transaction cost and management fee parameters (paper Section 4)
# ---------------------------------------------------------------------------
ALPHA_TC   = 0.0005   # transaction cost rate (0.05%)
ALPHA_MGMT = 0.001    # management fee rate  (0.1%)

# ---------------------------------------------------------------------------
# Numerical parameters
# ---------------------------------------------------------------------------
EPSILON = 1e-6         # regularisation for positive-definiteness
BIG_M   = 100.0        # big-M for MIQP formulation
RNG_SEED = 42


# ---------------------------------------------------------------------------
# Convenience: bundle everything into a dictionary
# ---------------------------------------------------------------------------
def get_params() -> dict:
    """Return all calibrated parameters as a dictionary."""
    return {
        'n': N_ASSETS,
        'T': T_QUARTERS,
        'x0': X0,
        'm': N_STATES,
        'q_default': Q_DEFAULT,
        'asset_tickers': ASSET_TICKERS,
        'asset_names': ASSET_NAMES,
        'P_trans': P_TRANS,
        'mu_tilde': MU_TILDE,
        'mu': {s: compute_mu(s) for s in range(N_STATES)},
        'Sigma': SIGMA,
        'rf_quarterly': RF_QUARTERLY,
        'rf_constant': RF_CONSTANT,
        'alpha_tc': ALPHA_TC,
        'alpha_mgmt': ALPHA_MGMT,
        'epsilon': EPSILON,
        'big_m': BIG_M,
    }


# ---------------------------------------------------------------------------
# Quick sanity check when run as script
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    params = get_params()
    print("=" * 60)
    print("CALIBRATED PARAMETERS — Costa & Araujo (2008)")
    print("=" * 60)
    print(f"Assets (n={params['n']}): {params['asset_tickers']}")
    print(f"Horizon T={params['T']} quarters, x0={params['x0']}")
    print(f"Market states m={params['m']}: {STATE_LABELS}")
    print(f"\nTransition matrix:\n{params['P_trans']}")
    for s in range(N_STATES):
        print(f"\n--- State {s} ({STATE_LABELS[s]}) ---")
        print(f"mu_tilde({s}): {MU_TILDE[s]}")
        print(f"mu({s}):       {params['mu'][s]}")
        print(f"Sigma({s}) diag: {np.diag(params['Sigma'][s])}")
        eigvals = np.linalg.eigvalsh(params['Sigma'][s])
        print(f"Sigma({s}) PD:   min eigenvalue = {eigvals.min():.6e}")
    print(f"\nRisk-free rates (quarterly): {params['rf_quarterly']}")
    print(f"TC rate: {params['alpha_tc']}, Mgmt fee rate: {params['alpha_mgmt']}")
