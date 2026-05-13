"""
CCQO solver and backward theta recursion for the CMMV portfolio.

Implements the cardinality-constrained quadratic optimisation (CCQO)
from Section 3.1 (Lemma 2) and the recursive theta computation
from Eq. (10) of Wang et al. (2026).
"""

import numpy as np
from itertools import combinations

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from utils.parameters import (
    N_ASSETS, T_QUARTERS, N_STATES, P_TRANS,
    SIGMA, compute_mu, RF_CONSTANT, EPSILON
)
from utils.matrix_utils import compute_d_vector, compute_D_matrix, regularise_matrix


# ===================================================================
# 1. CCQO Solver
# ===================================================================

def solve_CCQO(D_i, d_i, q, n=10):
    """
    Solve the cardinality-constrained quadratic optimisation problem.

    # Solving P_ccqo(D_t(i), d_t(i), q) — paper Section 3.1, Lemma 2

    min_k  2 * d_i^T * k + k^T * D_i * k
    s.t.   ||k||_0 <= q

    Uses Gurobi MIQP if available, otherwise falls back to brute-force
    enumeration over all C(n, q) subsets.

    Parameters
    ----------
    D_i : array (n, n)
        Second-moment matrix of excess returns for state i.
    d_i : array (n,)
        Mean excess return vector for state i.
    q : int
        Cardinality constraint (max number of nonzero positions).
    n : int
        Number of assets.

    Returns
    -------
    k_star : array (n,)
        Optimal portfolio tilt vector.
    z_star : array (n,)
        Binary indicator vector I(k* != 0).
    """
    try:
        return _solve_CCQO_gurobi(D_i, d_i, q, n)
    except (ImportError, Exception):
        return _solve_CCQO_bruteforce(D_i, d_i, q, n)


def _solve_CCQO_gurobi(D_i, d_i, q, n):
    """Solve CCQO via Gurobi MIQP formulation with big-M constraints."""
    import gurobipy as gp
    from gurobipy import GRB

    M = 1e4  # large constant for big-M constraints

    model = gp.Model("CCQO")
    model.Params.OutputFlag = 0  # suppress output

    # Decision variables
    k = model.addMVar(n, lb=-M, ub=M, name="k")
    z = model.addMVar(n, vtype=GRB.BINARY, name="z")

    # Big-M constraints: -z_l * M <= k_l <= z_l * M
    for l in range(n):
        model.addConstr(k[l] >= -z[l] * M)
        model.addConstr(k[l] <= z[l] * M)

    # Cardinality constraint: sum(z) <= q
    model.addConstr(z.sum() <= q)

    # Objective: min 2 * d_i^T * k + k^T * D_i * k
    obj = 2.0 * d_i @ k + k @ D_i @ k
    model.setObjective(obj, GRB.MINIMIZE)

    model.optimize()

    if model.Status == GRB.OPTIMAL:
        k_star = k.X
        z_star = z.X
        return k_star, np.round(z_star).astype(float)
    else:
        raise RuntimeError(f"Gurobi did not find optimal solution (status {model.Status})")


def _solve_CCQO_bruteforce(D_i, d_i, q, n):
    """
    Brute-force enumeration over all C(n, q) subsets.

    For each subset S of size q, solve the unconstrained QP restricted
    to variables in S:  k_S = -D_S^{-1} d_S  (first-order condition).
    """
    best_obj = np.inf
    best_k = np.zeros(n)
    best_z = np.zeros(n)

    for subset in combinations(range(n), q):
        idx = list(subset)

        # Extract sub-matrices
        D_sub = D_i[np.ix_(idx, idx)]
        d_sub = d_i[idx]

        # Solve unconstrained QP on subset: k_S = -D_S^{-1} d_S
        try:
            k_sub = np.linalg.solve(D_sub, -d_sub)
        except np.linalg.LinAlgError:
            k_sub = np.linalg.lstsq(D_sub, -d_sub, rcond=None)[0]

        # Full vector
        k_full = np.zeros(n)
        k_full[idx] = k_sub

        # Objective value: 2 * d^T k + k^T D k
        obj = 2.0 * d_i @ k_full + k_full @ D_i @ k_full
        if obj < best_obj:
            best_obj = obj
            best_k = k_full.copy()
            best_z = np.zeros(n)
            best_z[idx] = 1.0

    return best_k, best_z


# ===================================================================
# 2. Theta Recursion
# ===================================================================

def compute_theta_recursive(T, m, q, D_states, d_states, transition_matrix):
    """
    Backward recursion for theta values and optimal k vectors.

    # Eq. (10): recursive theta computation via DP

    theta_T(j) = 1  for all j
    For t = T-1 down to 0, for each state i:
        Solve CCQO -> z_star
        theta_t(i) = [sum_j p(i,j) * theta_{t+1}(j)]
                    * [1 - (diag(z*) d)^T (diag(z*) D diag(z*))^dagger (diag(z*) d)]

    Parameters
    ----------
    T : int
        Number of periods.
    m : int
        Number of market states.
    q : int
        Cardinality constraint.
    D_states : dict {int: array (n, n)}
        Second-moment matrices D(i) for each state i.
    d_states : dict {int: array (n,)}
        Mean excess return vectors d(i) for each state i.
    transition_matrix : array (m, m)
        Markov chain transition probabilities.

    Returns
    -------
    theta : array (T+1, m)
        Theta values for each period and state.
    k_stars : dict
        Optimal k vectors keyed by (t, i).
    """
    n = len(d_states[0])
    theta = np.zeros((T + 1, m))
    k_stars = {}

    # Terminal condition: theta_T(j) = 1 for all j
    theta[T, :] = 1.0

    # Backward recursion
    for t in range(T - 1, -1, -1):
        for i in range(m):
            # Regularise D(i) before solving
            D_reg = regularise_matrix(D_states[i], epsilon=EPSILON)
            d_i = d_states[i]

            # Solve CCQO to get z_star and k_star
            k_star, z_star = solve_CCQO(D_reg, d_i, q, n)
            k_stars[(t, i)] = k_star

            # Compute expected theta from next period
            # sum_j p(i,j) * theta_{t+1}(j)
            expected_theta = transition_matrix[i, :] @ theta[t + 1, :]

            # Compute the quadratic ratio term:
            # (diag(z*) d)^T (diag(z*) D diag(z*))^dagger (diag(z*) d)
            Z = np.diag(z_star)
            Zd = Z @ d_i             # diag(z*) d
            ZDZ = Z @ D_reg @ Z      # diag(z*) D diag(z*)

            # Pseudoinverse for the restricted subspace
            ZDZ_pinv = np.linalg.pinv(ZDZ)

            quad_ratio = Zd @ ZDZ_pinv @ Zd

            # theta_t(i) = expected_theta * (1 - quad_ratio)
            theta[t, i] = expected_theta * (1.0 - quad_ratio)

    return theta, k_stars


# ===================================================================
# 3. Portfolio Strategy: CMMV Optimal Policy
# ===================================================================

def apply_cmmv_policy(sample_paths, sample_states, k_stars, theta_0_s0,
                      gamma, omega, x0, T, r):
    """
    Apply the CMMV optimal portfolio policy to simulated paths.

    # Eq. (28): CMMV optimal policy — Theorem 5
    # u*_t(i) = r_{t+1} * (x_t - gamma_0*x0/(theta_0(s0)) + omega/(theta_0(s0)*gamma_t)) * k*_t(s_t)

    Parameters
    ----------
    sample_paths : array (n_paths, T, n)
        Simulated asset return vectors (total returns R_{t+1}).
    sample_states : array (n_paths, T)
        Simulated market states at each period.
    k_stars : dict {(t, i): array (n,)}
        Optimal k vectors from theta recursion.
    theta_0_s0 : float
        theta_0(s_0), initial theta value for the starting state.
    gamma : float
        gamma_0 = product of r_{k+1} for k=0..T-1 = r^T (cumulative risk-free return).
    omega : float
        Lagrange multiplier lambda controlling risk-return tradeoff.
    x0 : float
        Initial wealth.
    T : int
        Number of periods.
    r : float
        Risk-free total return per period (constant).

    Returns
    -------
    terminal_wealth : array (n_paths,)
        Terminal wealth for each sample path.
    weights_history : array (n_paths, T, n)
        Portfolio allocations u_t at each step.
    """
    n_paths, _, n = sample_paths.shape
    wealth = np.full(n_paths, x0)
    weights_history = np.zeros((n_paths, T, n))

    for t in range(T):
        # gamma_t = r^(T-t) — cumulative risk-free return from t to T
        gamma_t = r ** (T - t)

        # Compute the scaling factor for each path
        # scalar_t = r * (x_t - gamma_0*x0/theta_0(s0) + omega/(theta_0(s0)*gamma_t))
        scaling = r * (wealth - gamma * x0 / theta_0_s0
                       + omega / (theta_0_s0 * gamma_t))

        # Get k_star for each path based on its current state
        states_t = sample_states[:, t]
        for i in np.unique(states_t):
            mask = states_t == i
            k_i = k_stars[(t, int(i))]
            # u_t = scaling * k*_t(s_t)
            weights_history[mask, t, :] = scaling[mask, np.newaxis] * k_i[np.newaxis, :]

        # Wealth evolution: x_{t+1} = r * x_t + P_{t+1}^T * u_t
        # where P_{t+1} = R_{t+1} - r*e is the excess return
        R_t1 = sample_paths[:, t, :]         # (n_paths, n)
        P_t1 = R_t1 - r * np.ones(n)         # excess return
        u_t = weights_history[:, t, :]        # (n_paths, n)
        wealth = r * wealth + np.sum(P_t1 * u_t, axis=1)

    return wealth, weights_history


# ===================================================================
# 4. Portfolio Strategy: CMV-Static Benchmark
# ===================================================================

def apply_cmv_static_policy(sample_paths, mu_pooled, Sigma_pooled,
                            q, x0, T, r):
    """
    Apply the CMV-static benchmark portfolio policy.

    # CMV-static benchmark — single-period, hold to maturity

    Solves a single static CCQO using pooled (time-averaged) parameters
    and holds the same portfolio for all T periods without rebalancing.

    Parameters
    ----------
    sample_paths : array (n_paths, T, n)
        Simulated asset return vectors.
    mu_pooled : array (n,)
        Pooled mean return vector (stationary-distribution weighted average).
    Sigma_pooled : array (n, n)
        Pooled covariance matrix.
    q : int
        Cardinality constraint.
    x0 : float
        Initial wealth.
    T : int
        Number of periods.
    r : float
        Risk-free total return per period.

    Returns
    -------
    terminal_wealth : array (n_paths,)
        Terminal wealth for each sample path.
    weights_history : array (n_paths, T, n)
        Portfolio allocations u_t at each step (constant across time).
    """
    n_paths, _, n = sample_paths.shape

    # Compute pooled d and D for the T-period static problem
    # Scale mean and covariance for T periods: mu_T = T*mu_per_period excess
    d_pooled = mu_pooled - r * np.ones(n)
    D_pooled = Sigma_pooled + np.outer(d_pooled, d_pooled)
    D_pooled = regularise_matrix(D_pooled, epsilon=EPSILON)

    # Solve single CCQO for static allocation direction
    k_static, z_static = solve_CCQO(D_pooled, d_pooled, q, n)

    # For CMV-static, invest a fixed fraction of initial wealth
    # The static portfolio holds: u = x0 * k_static (constant)
    u_static = x0 * k_static

    wealth = np.full(n_paths, x0)
    weights_history = np.zeros((n_paths, T, n))

    for t in range(T):
        weights_history[:, t, :] = u_static[np.newaxis, :]

        # Wealth evolution: x_{t+1} = r * x_t + P_{t+1}^T * u_t
        R_t1 = sample_paths[:, t, :]
        P_t1 = R_t1 - r * np.ones(n)
        wealth = r * wealth + np.sum(P_t1 * u_static[np.newaxis, :], axis=1)

    return wealth, weights_history


# ===================================================================
# 5. Portfolio Strategy: CMMV-Independent Benchmark
# ===================================================================

def apply_cmmv_independent_policy(sample_paths, k_stars_indep,
                                  theta_0_indep, gamma, omega,
                                  x0, T, r):
    """
    Apply the CMMV-independent benchmark policy.

    # CMMV-independent benchmark — Theorem 11, Assumption 10

    Same structure as CMMV but uses theta/k_stars computed assuming
    independent returns (no regime awareness). The k_stars are keyed
    by time t only (no state index), since there is only one set of
    pooled parameters.

    Parameters
    ----------
    sample_paths : array (n_paths, T, n)
        Simulated asset return vectors.
    k_stars_indep : dict {t: array (n,)}
        Optimal k vectors from independent theta recursion (keyed by t).
    theta_0_indep : float
        theta_0 under independent assumption.
    gamma : float
        gamma_0 = r^T.
    omega : float
        Lagrange multiplier lambda.
    x0 : float
        Initial wealth.
    T : int
        Number of periods.
    r : float
        Risk-free total return per period.

    Returns
    -------
    terminal_wealth : array (n_paths,)
        Terminal wealth for each sample path.
    weights_history : array (n_paths, T, n)
        Portfolio allocations u_t at each step.
    """
    n_paths, _, n = sample_paths.shape
    wealth = np.full(n_paths, x0)
    weights_history = np.zeros((n_paths, T, n))

    for t in range(T):
        gamma_t = r ** (T - t)

        # Eq. (44): same form as CMMV but no state-dependence
        scaling = r * (wealth - gamma * x0 / theta_0_indep
                       + omega / (theta_0_indep * gamma_t))

        k_t = k_stars_indep[t]
        weights_history[:, t, :] = scaling[:, np.newaxis] * k_t[np.newaxis, :]

        # Wealth evolution
        R_t1 = sample_paths[:, t, :]
        P_t1 = R_t1 - r * np.ones(n)
        u_t = weights_history[:, t, :]
        wealth = r * wealth + np.sum(P_t1 * u_t, axis=1)

    return wealth, weights_history


# ===================================================================
# Helper: compute pooled parameters under independent assumption
# ===================================================================

def compute_pooled_parameters(mu_states, Sigma_states, stationary_dist):
    """
    Compute pooled mean and covariance using stationary distribution.

    mu_pooled = sum_i pi_i * mu(i)
    Sigma_pooled = sum_i pi_i * Sigma(i)

    Parameters
    ----------
    mu_states : dict {int: array (n,)}
        Mean return vectors per state.
    Sigma_states : dict {int: array (n, n)}
        Covariance matrices per state.
    stationary_dist : array (m,)
        Stationary distribution of the Markov chain.

    Returns
    -------
    mu_pooled : array (n,)
    Sigma_pooled : array (n, n)
    """
    m = len(stationary_dist)
    n = len(mu_states[0])
    mu_pooled = np.zeros(n)
    Sigma_pooled = np.zeros((n, n))
    for i in range(m):
        mu_pooled += stationary_dist[i] * mu_states[i]
        Sigma_pooled += stationary_dist[i] * Sigma_states[i]
    return mu_pooled, Sigma_pooled


def compute_theta_independent(T, q, D_pooled, d_pooled):
    """
    Compute theta recursion under independent returns (Eq. 43).

    No regime-switching: single set of D, d for all periods.

    Returns
    -------
    theta : array (T+1,)
        Theta values for each period.
    k_stars : dict {t: array (n,)}
        Optimal k vectors keyed by time t.
    """
    n = len(d_pooled)
    theta = np.zeros(T + 1)
    k_stars = {}

    theta[T] = 1.0

    D_reg = regularise_matrix(D_pooled, epsilon=EPSILON)

    for t in range(T - 1, -1, -1):
        k_star, z_star = solve_CCQO(D_reg, d_pooled, q, n)
        k_stars[t] = k_star

        # theta_t = theta_{t+1} * (1 - quadratic ratio)
        Z = np.diag(z_star)
        Zd = Z @ d_pooled
        ZDZ = Z @ D_reg @ Z
        ZDZ_pinv = np.linalg.pinv(ZDZ)
        quad_ratio = Zd @ ZDZ_pinv @ Zd

        theta[t] = theta[t + 1] * (1.0 - quad_ratio)

    return theta, k_stars


# ===================================================================
# Quick test
# ===================================================================

if __name__ == '__main__':
    # Build D and d for each state
    r_f = 1.0 + RF_CONSTANT  # risk-free total return per quarter
    D_states = {}
    d_states = {}
    for s in range(N_STATES):
        mu_s = compute_mu(s)
        d_s = compute_d_vector(mu_s, r_f)
        D_s = compute_D_matrix(SIGMA[s], d_s)
        D_states[s] = D_s
        d_states[s] = d_s

    q = 5
    print(f"Computing theta recursion for q={q}, T={T_QUARTERS}, m={N_STATES}...")
    theta, k_stars = compute_theta_recursive(
        T=T_QUARTERS, m=N_STATES, q=q,
        D_states=D_states, d_states=d_states,
        transition_matrix=P_TRANS
    )

    print(f"\ntheta_0(0) [Up state]   = {theta[0, 0]:.6f}")
    print(f"theta_0(1) [Down state] = {theta[0, 1]:.6f}")
    print(f"\nAll theta values (T+1 x m):")
    for t in range(T_QUARTERS + 1):
        print(f"  t={t}: theta = [{theta[t, 0]:.6f}, {theta[t, 1]:.6f}]")

    # Sanity check: theta should be between 0 and 1
    assert np.all(theta >= 0) and np.all(theta <= 1), \
        f"ERROR: theta values outside [0, 1]! min={theta.min():.4f}, max={theta.max():.4f}"
    print("\n[OK] All theta values are in [0, 1].")

    # ------------------------------------------------------------------
    # Test CMMV policy on 1000 sample paths
    # ------------------------------------------------------------------
    from utils.simulation import simulate_market_paths

    n_test = 1000
    mu_states = {s: compute_mu(s) for s in range(N_STATES)}
    print(f"\n{'='*60}")
    print(f"Testing CMMV policy: q={q}, {n_test} sample paths")
    print(f"{'='*60}")

    returns, states = simulate_market_paths(
        n_paths=n_test, T=T_QUARTERS,
        transition_matrix=P_TRANS,
        mu_states=mu_states, Sigma_states=SIGMA,
        s0=0, seed=42
    )

    gamma_0 = r_f ** T_QUARTERS
    omega = 1.0
    s0 = 0
    theta_0_s0 = theta[0, s0]

    tw_cmmv, wh_cmmv = apply_cmmv_policy(
        returns, states, k_stars, theta_0_s0,
        gamma=gamma_0, omega=omega, x0=1.0, T=T_QUARTERS, r=r_f
    )
    print(f"CMMV mean terminal wealth:   {np.mean(tw_cmmv):.6f}")
    print(f"CMMV std terminal wealth:    {np.std(tw_cmmv):.6f}")
    print(f"CMMV median terminal wealth: {np.median(tw_cmmv):.6f}")

    # Stationary distribution: pi = [0.55, 0.45]
    pi_stat = np.array([0.55, 0.45])
    mu_pooled, Sigma_pooled = compute_pooled_parameters(mu_states, SIGMA, pi_stat)

    # CMV-static test
    tw_static, wh_static = apply_cmv_static_policy(
        returns, mu_pooled, Sigma_pooled, q, x0=1.0, T=T_QUARTERS, r=r_f
    )
    print(f"\nCMV-static mean terminal wealth: {np.mean(tw_static):.6f}")
    print(f"CMV-static std terminal wealth:  {np.std(tw_static):.6f}")

    # CMMV-independent test
    d_pooled = compute_d_vector(mu_pooled, r_f)
    D_pooled = compute_D_matrix(Sigma_pooled, d_pooled)
    theta_indep, k_stars_indep = compute_theta_independent(T_QUARTERS, q, D_pooled, d_pooled)
    print(f"\ntheta_0 (independent): {theta_indep[0]:.6f}")

    tw_indep, wh_indep = apply_cmmv_independent_policy(
        returns, k_stars_indep, theta_indep[0],
        gamma=gamma_0, omega=omega, x0=1.0, T=T_QUARTERS, r=r_f
    )
    print(f"CMMV-independent mean terminal wealth: {np.mean(tw_indep):.6f}")
    print(f"CMMV-independent std terminal wealth:  {np.std(tw_indep):.6f}")

    print(f"\n[OK] All three strategies executed successfully.")
