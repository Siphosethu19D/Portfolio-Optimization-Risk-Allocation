"""
Main replication runner for the CMMV portfolio paper.

Produces the cost-free Sharpe ratio table (Table 1 of Wang et al., 2026)
for q = 1..10 across three strategies: CMV-static, CMMV-independent, CMMV.
"""

import numpy as np
import pandas as pd
import pickle
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from utils.parameters import (
    N_ASSETS, T_QUARTERS, N_STATES, P_TRANS, X0,
    SIGMA, compute_mu, RF_CONSTANT, EPSILON
)
from utils.matrix_utils import compute_d_vector, compute_D_matrix, regularise_matrix
from utils.simulation import simulate_market_paths

from replication.solver import (
    solve_CCQO,
    compute_theta_recursive,
    compute_theta_independent,
    compute_pooled_parameters,
    apply_cmmv_policy,
    apply_cmv_static_policy,
    apply_cmmv_independent_policy,
)


# ===================================================================
# Configuration
# ===================================================================
N_PATHS = 100_000       # Paper uses 10^7; scale up if compute allows
SEED = 42
S0 = 0                  # initial market state (Up)
OMEGA = 1.0             # Lagrange multiplier (controls risk-return tradeoff)

# Risk-free total return per quarter
R_F = 1.0 + RF_CONSTANT  # 1.0024

# Cumulative risk-free return over full horizon
GAMMA_0 = R_F ** T_QUARTERS

# Stationary distribution of the Markov chain
# P has identical rows [0.55, 0.45], so pi = [0.55, 0.45]
PI_STAT = np.array([0.55, 0.45])

# Paper Table 1 reference values (SR column, cost-free)
PAPER_TABLE1 = {
    1:  {'CMV-static': 0.152, 'CMMV-independent': 0.157, 'CMMV': 0.157},
    2:  {'CMV-static': 0.189, 'CMMV-independent': 0.190, 'CMMV': 0.190},
    3:  {'CMV-static': 0.205, 'CMMV-independent': 0.192, 'CMMV': 0.214},
    4:  {'CMV-static': 0.209, 'CMMV-independent': 0.207, 'CMMV': 0.228},
    5:  {'CMV-static': 0.221, 'CMMV-independent': 0.217, 'CMMV': 0.231},
    6:  {'CMV-static': 0.228, 'CMMV-independent': 0.219, 'CMMV': 0.234},
    7:  {'CMV-static': 0.228, 'CMMV-independent': 0.216, 'CMMV': 0.231},
    8:  {'CMV-static': 0.228, 'CMMV-independent': 0.212, 'CMMV': 0.228},
    9:  {'CMV-static': 0.228, 'CMMV-independent': 0.209, 'CMMV': 0.223},
    10: {'CMV-static': 0.228, 'CMMV-independent': 0.205, 'CMMV': 0.220},
}


def compute_analytical_sr(theta_0):
    """
    Analytical Sharpe ratio from Corollary 7, Eq. (39):
    SR = sqrt(1/theta_0 - 1)
    """
    if theta_0 <= 0 or theta_0 >= 1:
        return 0.0
    return np.sqrt(1.0 / theta_0 - 1.0)


def main():
    print("=" * 72)
    print("CMMV PORTFOLIO REPLICATION — Cost-Free Sharpe Ratio Table")
    print("Wang, Jin, Wu & Gao (2026), Automatica 183, 112669")
    print("=" * 72)

    # ------------------------------------------------------------------
    # 1. Pre-compute state-dependent parameters
    # ------------------------------------------------------------------
    mu_states = {s: compute_mu(s) for s in range(N_STATES)}
    D_states = {}
    d_states = {}
    for s in range(N_STATES):
        d_s = compute_d_vector(mu_states[s], R_F)
        D_s = compute_D_matrix(SIGMA[s], d_s)
        D_states[s] = D_s
        d_states[s] = d_s

    # Pooled parameters for benchmarks
    mu_pooled, Sigma_pooled = compute_pooled_parameters(mu_states, SIGMA, PI_STAT)
    d_pooled = compute_d_vector(mu_pooled, R_F)
    D_pooled = compute_D_matrix(Sigma_pooled, d_pooled)

    # ------------------------------------------------------------------
    # 2. Simulate sample paths (shared across all strategies)
    # ------------------------------------------------------------------
    print(f"\nSimulating {N_PATHS:,} sample paths (T={T_QUARTERS}, n={N_ASSETS}, seed={SEED})...")
    t0 = time.time()
    returns, states = simulate_market_paths(
        n_paths=N_PATHS, T=T_QUARTERS,
        transition_matrix=P_TRANS,
        mu_states=mu_states, Sigma_states=SIGMA,
        s0=S0, seed=SEED
    )
    print(f"  Simulation done in {time.time()-t0:.1f}s")

    # ------------------------------------------------------------------
    # 3. Main loop over q = 1..10
    # ------------------------------------------------------------------
    results = []
    sample_path_data = {}  # for saving one representative path

    print(f"\nRunning strategies for q = 1..{N_ASSETS}...\n")

    for q in range(1, N_ASSETS + 1):
        tq = time.time()

        # --- CMMV (regime-switching) ---
        theta_cmmv, k_stars_cmmv = compute_theta_recursive(
            T=T_QUARTERS, m=N_STATES, q=q,
            D_states=D_states, d_states=d_states,
            transition_matrix=P_TRANS
        )
        theta_0_cmmv = theta_cmmv[0, S0]
        sr_cmmv_analytical = compute_analytical_sr(theta_0_cmmv)

        tw_cmmv, wh_cmmv = apply_cmmv_policy(
            returns, states, k_stars_cmmv, theta_0_cmmv,
            gamma=GAMMA_0, omega=OMEGA, x0=X0, T=T_QUARTERS, r=R_F
        )
        mean_cmmv = np.mean(tw_cmmv)
        var_cmmv = np.var(tw_cmmv)
        sr_cmmv_sim = (mean_cmmv - GAMMA_0 * X0) / np.std(tw_cmmv)

        # --- CMMV-independent ---
        theta_indep, k_stars_indep = compute_theta_independent(
            T=T_QUARTERS, q=q, D_pooled=D_pooled, d_pooled=d_pooled
        )
        theta_0_indep = theta_indep[0]
        sr_indep_analytical = compute_analytical_sr(theta_0_indep)

        tw_indep, wh_indep = apply_cmmv_independent_policy(
            returns, k_stars_indep, theta_0_indep,
            gamma=GAMMA_0, omega=OMEGA, x0=X0, T=T_QUARTERS, r=R_F
        )
        mean_indep = np.mean(tw_indep)
        var_indep = np.var(tw_indep)
        sr_indep_sim = (mean_indep - GAMMA_0 * X0) / np.std(tw_indep)

        # --- CMV-static ---
        # For CMV-static, the analytical SR uses the static theta
        # computed from pooled D and d over a single period
        D_pooled_reg = regularise_matrix(D_pooled, epsilon=EPSILON)
        k_static, z_static = solve_CCQO(D_pooled_reg, d_pooled, q, N_ASSETS)
        Z_s = np.diag(z_static)
        Zd_s = Z_s @ d_pooled
        ZDZ_s = Z_s @ D_pooled_reg @ Z_s
        quad_ratio_static = Zd_s @ np.linalg.pinv(ZDZ_s) @ Zd_s
        # Static SR over T periods: SR_static = sqrt(T * quad / (1 - quad))
        # where quad = d[z]^T D[z]^{-1} d[z]
        sr_static_analytical = np.sqrt(T_QUARTERS * quad_ratio_static
                                       / (1.0 - quad_ratio_static))

        tw_static, wh_static = apply_cmv_static_policy(
            returns, mu_pooled, Sigma_pooled, q, x0=X0, T=T_QUARTERS, r=R_F
        )
        mean_static = np.mean(tw_static)
        var_static = np.var(tw_static)
        sr_static_sim = (mean_static - GAMMA_0 * X0) / np.std(tw_static)

        elapsed = time.time() - tq

        # Store results (using analytical SR to match paper)
        for strategy, sr_a, sr_s, mean_w, var_w in [
            ('CMV-static',        sr_static_analytical, sr_static_sim, mean_static, var_static),
            ('CMMV-independent',  sr_indep_analytical,  sr_indep_sim,  mean_indep,  var_indep),
            ('CMMV',              sr_cmmv_analytical,   sr_cmmv_sim,   mean_cmmv,   var_cmmv),
        ]:
            results.append({
                'q': q,
                'Strategy': strategy,
                'Mean_Wealth': mean_w,
                'Var_Wealth': var_w,
                'Sharpe_Ratio_Analytical': sr_a,
                'Sharpe_Ratio_Simulated': sr_s,
            })

        # Save data for q=5 for sample path plot (Figure 1)
        if q == 5:
            # Pick path index 0 as the representative sample path
            sample_path_data = {
                'q': q,
                'states': states[0, :],
                'returns': returns[0, :, :],
                'wealth_cmmv': _recompute_single_path_wealth(
                    returns[0], states[0], k_stars_cmmv, theta_0_cmmv,
                    GAMMA_0, OMEGA, X0, T_QUARTERS, R_F, 'cmmv'),
                'wealth_indep': _recompute_single_path_wealth(
                    returns[0], states[0], k_stars_indep, theta_0_indep,
                    GAMMA_0, OMEGA, X0, T_QUARTERS, R_F, 'indep'),
                'wealth_static': _recompute_single_path_wealth(
                    returns[0], states[0], None, None,
                    GAMMA_0, OMEGA, X0, T_QUARTERS, R_F, 'static',
                    mu_pooled=mu_pooled, Sigma_pooled=Sigma_pooled, q=q),
                'weights_cmmv': wh_cmmv[0, :, :],
                'weights_indep': wh_indep[0, :, :],
                'weights_static': wh_static[0, :, :],
                'k_stars_cmmv': k_stars_cmmv,
                'k_stars_indep': k_stars_indep,
                'theta_cmmv': theta_cmmv,
                'theta_indep': theta_indep,
            }

        print(f"  q={q:2d}  CMV-static SR={sr_static_analytical:.3f}  "
              f"CMMV-indep SR={sr_indep_analytical:.3f}  "
              f"CMMV SR={sr_cmmv_analytical:.3f}  ({elapsed:.1f}s)")

    # ------------------------------------------------------------------
    # 4. Print formatted results table
    # ------------------------------------------------------------------
    print("\n" + "=" * 72)
    print("COST-FREE SHARPE RATIO TABLE (Analytical — Corollary 7)")
    print("=" * 72)
    print(f"{'q':>3s} | {'CMV-static':>12s} | {'CMMV-indep':>12s} | {'CMMV':>12s}")
    print("-" * 50)

    df = pd.DataFrame(results)
    for q in range(1, N_ASSETS + 1):
        row_s = df[(df['q'] == q) & (df['Strategy'] == 'CMV-static')].iloc[0]
        row_i = df[(df['q'] == q) & (df['Strategy'] == 'CMMV-independent')].iloc[0]
        row_c = df[(df['q'] == q) & (df['Strategy'] == 'CMMV')].iloc[0]
        print(f"{q:3d} | {row_s['Sharpe_Ratio_Analytical']:12.3f} | "
              f"{row_i['Sharpe_Ratio_Analytical']:12.3f} | "
              f"{row_c['Sharpe_Ratio_Analytical']:12.3f}")

    # ------------------------------------------------------------------
    # 5. Compare to paper's Table 1
    # ------------------------------------------------------------------
    print("\n" + "=" * 72)
    print("REPLICATION vs PAPER — Table 1 Comparison")
    print("=" * 72)
    print(f"{'q':>3s} | {'Strategy':<20s} | {'Paper':>7s} | {'Ours':>7s} | {'Diff':>7s}")
    print("-" * 60)

    for q_check in [3, 6]:
        for strat in ['CMV-static', 'CMMV-independent', 'CMMV']:
            paper_val = PAPER_TABLE1[q_check][strat]
            row = df[(df['q'] == q_check) & (df['Strategy'] == strat)].iloc[0]
            our_val = row['Sharpe_Ratio_Analytical']
            diff = our_val - paper_val
            print(f"{q_check:3d} | {strat:<20s} | {paper_val:7.3f} | {our_val:7.3f} | {diff:+7.3f}")

    # Also print full comparison for all q
    print(f"\n{'q':>3s} | {'CMV-s (P)':>9s} {'(Ours)':>7s} | "
          f"{'Indep (P)':>9s} {'(Ours)':>7s} | "
          f"{'CMMV (P)':>9s} {'(Ours)':>7s}")
    print("-" * 72)
    for q in range(1, N_ASSETS + 1):
        p = PAPER_TABLE1[q]
        row_s = df[(df['q'] == q) & (df['Strategy'] == 'CMV-static')].iloc[0]
        row_i = df[(df['q'] == q) & (df['Strategy'] == 'CMMV-independent')].iloc[0]
        row_c = df[(df['q'] == q) & (df['Strategy'] == 'CMMV')].iloc[0]
        print(f"{q:3d} | {p['CMV-static']:9.3f} {row_s['Sharpe_Ratio_Analytical']:7.3f} | "
              f"{p['CMMV-independent']:9.3f} {row_i['Sharpe_Ratio_Analytical']:7.3f} | "
              f"{p['CMMV']:9.3f} {row_c['Sharpe_Ratio_Analytical']:7.3f}")

    # ------------------------------------------------------------------
    # 6. Save results
    # ------------------------------------------------------------------
    results_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
    os.makedirs(results_dir, exist_ok=True)

    csv_path = os.path.join(results_dir, 'replication_results.csv')
    df.to_csv(csv_path, index=False)
    print(f"\n[SAVED] Results CSV: {os.path.abspath(csv_path)}")

    pkl_path = os.path.join(results_dir, 'sample_path_data.pkl')
    with open(pkl_path, 'wb') as f:
        pickle.dump(sample_path_data, f)
    print(f"[SAVED] Sample path data: {os.path.abspath(pkl_path)}")

    print("\n[DONE] Replication complete.")


def _recompute_single_path_wealth(returns_1, states_1, k_stars, theta_0,
                                  gamma, omega, x0, T, r, mode,
                                  mu_pooled=None, Sigma_pooled=None, q=None):
    """
    Recompute wealth trajectory for a single sample path.

    Returns array of length T+1 (including initial wealth x0).
    """
    n = returns_1.shape[1]
    wealth = np.zeros(T + 1)
    wealth[0] = x0

    if mode == 'static':
        d_p = mu_pooled - r * np.ones(n)
        D_p = Sigma_pooled + np.outer(d_p, d_p)
        D_p = regularise_matrix(D_p, epsilon=EPSILON)
        k_stat, _ = solve_CCQO(D_p, d_p, q, n)
        u_static = x0 * k_stat
        for t in range(T):
            R_t1 = returns_1[t, :]
            P_t1 = R_t1 - r * np.ones(n)
            wealth[t + 1] = r * wealth[t] + P_t1 @ u_static
    elif mode == 'cmmv':
        for t in range(T):
            gamma_t = r ** (T - t)
            scaling = r * (wealth[t] - gamma * x0 / theta_0
                           + omega / (theta_0 * gamma_t))
            s_t = int(states_1[t])
            k_t = k_stars[(t, s_t)]
            u_t = scaling * k_t
            R_t1 = returns_1[t, :]
            P_t1 = R_t1 - r * np.ones(n)
            wealth[t + 1] = r * wealth[t] + P_t1 @ u_t
    elif mode == 'indep':
        for t in range(T):
            gamma_t = r ** (T - t)
            scaling = r * (wealth[t] - gamma * x0 / theta_0
                           + omega / (theta_0 * gamma_t))
            k_t = k_stars[t]
            u_t = scaling * k_t
            R_t1 = returns_1[t, :]
            P_t1 = R_t1 - r * np.ones(n)
            wealth[t + 1] = r * wealth[t] + P_t1 @ u_t

    return wealth


if __name__ == '__main__':
    main()
