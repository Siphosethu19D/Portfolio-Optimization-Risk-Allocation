"""
Extension runner: transaction costs, sensitivity analysis, and full risk metrics.

Applies post-optimisation transaction costs and management fees to the
CMMV, CMMV-independent, and CMV-static portfolio strategies, then
computes comprehensive risk metrics across all cardinality levels q=1..10.
"""

import numpy as np
import pandas as pd
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from utils.parameters import (
    N_ASSETS, T_QUARTERS, N_STATES, P_TRANS, X0,
    SIGMA, compute_mu, RF_CONSTANT, EPSILON,
    ALPHA_TC, ALPHA_MGMT
)
from utils.matrix_utils import compute_d_vector, compute_D_matrix, regularise_matrix
from utils.simulation import simulate_market_paths
from utils.risk_metrics import (
    compute_sharpe_ratio, compute_max_drawdown,
    compute_var, compute_cvar, compute_turnover, kupiec_backtest
)

from replication.solver import (
    compute_theta_recursive, compute_theta_independent,
    compute_pooled_parameters,
    apply_cmmv_policy, apply_cmv_static_policy,
    apply_cmmv_independent_policy,
)

from extension.cost_model import (
    compute_transaction_costs,
    compute_management_fee,
    compute_net_terminal_wealth,
)


# ===================================================================
# Configuration
# ===================================================================
N_PATHS = 100_000       # Paper uses 10^7; scale up if compute allows
SEED = 42
S0 = 0
OMEGA = 1.0
R_F = 1.0 + RF_CONSTANT
GAMMA_0 = R_F ** T_QUARTERS
PI_STAT = np.array([0.55, 0.45])

ROOT = os.path.join(os.path.dirname(__file__), '..')
RES_DIR = os.path.join(ROOT, 'results')
os.makedirs(RES_DIR, exist_ok=True)

# Sensitivity analysis: transaction cost rates to test
ALPHA1_LEVELS = [0.0005, 0.001, 0.003, 0.005]  # 5bp, 10bp, 30bp, 50bp


def compute_all_metrics(tw_gross, total_tc, mgmt_fee, wh, sample_paths, r_f_wealth):
    """
    Compute full suite of risk metrics for one strategy/q combination.

    Parameters
    ----------
    tw_gross : array (n_paths,) — gross terminal wealth
    total_tc : array (n_paths,) — total transaction costs per path
    mgmt_fee : float — management fee
    wh : array (n_paths, T, n) — weights history
    sample_paths : array (n_paths, T, n) — return paths
    r_f_wealth : float — risk-free terminal wealth

    Returns
    -------
    dict of metric name -> value
    """
    tw_net = compute_net_terminal_wealth(tw_gross, total_tc, mgmt_fee)
    n_paths, T, n = wh.shape

    # Sharpe ratio (net of costs)
    sr = compute_sharpe_ratio(tw_net, r_f_wealth)

    # Mean & std
    mean_w = float(np.mean(tw_net))
    std_w = float(np.std(tw_net))

    # Wealth paths for drawdown (reconstruct from returns and weights)
    # Build period-by-period wealth for each path
    wealth_paths = np.zeros((n_paths, T + 1))
    wealth_paths[:, 0] = X0
    w = np.full(n_paths, X0)
    for t in range(T):
        R_t1 = sample_paths[:, t, :]
        P_t1 = R_t1 - R_F * np.ones(n)
        u_t = wh[:, t, :]
        w = R_F * w + np.sum(P_t1 * u_t, axis=1)
        wealth_paths[:, t + 1] = w

    avg_mdd, worst_mdd = compute_max_drawdown(wealth_paths)

    # VaR and CVaR
    var95 = compute_var(tw_net, 0.95)
    var99 = compute_var(tw_net, 0.99)
    cvar95 = compute_cvar(tw_net, 0.95)
    cvar99 = compute_cvar(tw_net, 0.99)

    # Average turnover (use mean path weights)
    # Compute average turnover across all paths
    turnovers = np.zeros(n_paths)
    for p in range(min(n_paths, 10000)):  # sample for speed
        path_weights = [wh[p, t, :] for t in range(T)]
        turnovers[p] = compute_turnover(path_weights)
    avg_turnover = float(np.mean(turnovers[:min(n_paths, 10000)]))

    # Kupiec backtest on period returns (use path 0 as representative)
    # Period returns: r_t = (x_{t+1} - x_t) / x_t
    repr_wealth = wealth_paths[0, :]
    period_returns = np.diff(repr_wealth) / np.where(
        np.abs(repr_wealth[:-1]) > 1e-10, repr_wealth[:-1], 1.0)
    var_level = compute_var(tw_net, 0.95) / np.mean(tw_net) if np.mean(tw_net) != 0 else 0.05
    kupiec_stat, kupiec_pval = kupiec_backtest(period_returns, var_level, 0.95)

    # Average TC
    avg_tc = float(np.mean(total_tc))

    return {
        'SR': sr,
        'Mean_Wealth': mean_w,
        'Std_Wealth': std_w,
        'MDD': avg_mdd,
        'VaR95': var95,
        'VaR99': var99,
        'CVaR95': cvar95,
        'CVaR99': cvar99,
        'Turnover': avg_turnover,
        'Kupiec_pval': kupiec_pval,
        'Avg_TC': avg_tc,
        'Mgmt_Fee': mgmt_fee,
    }


def run_strategies(returns, states, q, mu_states, D_states, d_states,
                   mu_pooled, Sigma_pooled, D_pooled, d_pooled):
    """
    Run all three strategies for a given q. Returns gross terminal wealth
    and weights history for each strategy.
    """
    # --- CMMV ---
    theta_cmmv, k_stars_cmmv = compute_theta_recursive(
        T_QUARTERS, N_STATES, q, D_states, d_states, P_TRANS)
    theta_0_cmmv = theta_cmmv[0, S0]
    tw_cmmv, wh_cmmv = apply_cmmv_policy(
        returns, states, k_stars_cmmv, theta_0_cmmv,
        GAMMA_0, OMEGA, X0, T_QUARTERS, R_F)

    # --- CMMV-independent ---
    theta_indep, k_stars_indep = compute_theta_independent(
        T_QUARTERS, q, D_pooled, d_pooled)
    theta_0_indep = theta_indep[0]
    tw_indep, wh_indep = apply_cmmv_independent_policy(
        returns, k_stars_indep, theta_0_indep,
        GAMMA_0, OMEGA, X0, T_QUARTERS, R_F)

    # --- CMV-static ---
    tw_static, wh_static = apply_cmv_static_policy(
        returns, mu_pooled, Sigma_pooled, q, X0, T_QUARTERS, R_F)

    return {
        'CMMV': (tw_cmmv, wh_cmmv),
        'CMMV-independent': (tw_indep, wh_indep),
        'CMV-static': (tw_static, wh_static),
    }


def main():
    print("=" * 72)
    print("EXTENSION: Transaction Costs, Sensitivity Analysis & Risk Metrics")
    print("=" * 72)

    # ------------------------------------------------------------------
    # Pre-compute parameters
    # ------------------------------------------------------------------
    mu_states = {s: compute_mu(s) for s in range(N_STATES)}
    D_states, d_states = {}, {}
    for s in range(N_STATES):
        d_s = compute_d_vector(mu_states[s], R_F)
        D_states[s] = compute_D_matrix(SIGMA[s], d_s)
        d_states[s] = d_s

    mu_pooled, Sigma_pooled = compute_pooled_parameters(mu_states, SIGMA, PI_STAT)
    d_pooled = compute_d_vector(mu_pooled, R_F)
    D_pooled = compute_D_matrix(Sigma_pooled, d_pooled)

    r_f_wealth = GAMMA_0 * X0  # risk-free terminal wealth

    # ------------------------------------------------------------------
    # Simulate paths (shared across all runs)
    # ------------------------------------------------------------------
    print(f"\nSimulating {N_PATHS:,} paths (T={T_QUARTERS}, n={N_ASSETS}, seed={SEED})...")
    returns, states = simulate_market_paths(
        n_paths=N_PATHS, T=T_QUARTERS,
        transition_matrix=P_TRANS,
        mu_states=mu_states, Sigma_states=SIGMA,
        s0=S0, seed=SEED
    )

    # ==================================================================
    # A) BASELINE RUN (alpha0=0.1%, alpha1=0.05%)
    # ==================================================================
    print(f"\n{'='*72}")
    print(f"A) BASELINE: alpha0={ALPHA_MGMT:.4f}, alpha1={ALPHA_TC:.4f}")
    print(f"{'='*72}")

    baseline_rows = []
    strat_order = ['CMV-static', 'CMMV-independent', 'CMMV']

    for q in range(1, N_ASSETS + 1):
        tq = time.time()
        results = run_strategies(returns, states, q, mu_states,
                                 D_states, d_states, mu_pooled,
                                 Sigma_pooled, D_pooled, d_pooled)

        for strat in strat_order:
            tw_gross, wh = results[strat]
            mgmt_fee = compute_management_fee(X0, q, ALPHA_MGMT)
            _, total_tc = compute_transaction_costs(wh, returns, ALPHA_TC)

            metrics = compute_all_metrics(
                tw_gross, total_tc, mgmt_fee, wh, returns, r_f_wealth)
            metrics['q'] = q
            metrics['Strategy'] = strat
            baseline_rows.append(metrics)

        elapsed = time.time() - tq
        # Print inline progress
        cmmv_sr = [r for r in baseline_rows if r['q'] == q and r['Strategy'] == 'CMMV'][0]['SR']
        print(f"  q={q:2d}  CMMV SR(net)={cmmv_sr:+.4f}  ({elapsed:.1f}s)")

    df_baseline = pd.DataFrame(baseline_rows)

    # Reorder columns
    col_order = ['q', 'Strategy', 'SR', 'Mean_Wealth', 'Std_Wealth', 'MDD',
                 'VaR95', 'VaR99', 'CVaR95', 'CVaR99', 'Turnover',
                 'Kupiec_pval', 'Avg_TC', 'Mgmt_Fee']
    df_baseline = df_baseline[col_order]

    baseline_path = os.path.join(RES_DIR, 'extension_baseline_results.csv')
    df_baseline.to_csv(baseline_path, index=False)
    print(f"\n[SAVED] {os.path.abspath(baseline_path)}")

    # ------------------------------------------------------------------
    # Print baseline performance table for q=5
    # ------------------------------------------------------------------
    print(f"\n{'='*72}")
    print("BASELINE PERFORMANCE TABLE (q=5, alpha0=0.1%, alpha1=0.05%)")
    print(f"{'='*72}")
    q5 = df_baseline[df_baseline['q'] == 5]
    print(q5.to_string(index=False, float_format='%.4f'))

    # ------------------------------------------------------------------
    # Full performance table
    # ------------------------------------------------------------------
    perf_path = os.path.join(RES_DIR, 'performance_table_full.csv')
    df_baseline.to_csv(perf_path, index=False)
    print(f"\n[SAVED] {os.path.abspath(perf_path)}")

    # ==================================================================
    # B) SENSITIVITY ANALYSIS over alpha1
    # ==================================================================
    print(f"\n{'='*72}")
    print("B) SENSITIVITY ANALYSIS: varying alpha1 (TC rate)")
    print(f"   alpha1 levels: {ALPHA1_LEVELS}")
    print(f"{'='*72}")

    sensitivity_rows = []

    for alpha1 in ALPHA1_LEVELS:
        print(f"\n  alpha1 = {alpha1:.4f} ({alpha1*10000:.0f}bp)...")
        for q in range(1, N_ASSETS + 1):
            results = run_strategies(returns, states, q, mu_states,
                                     D_states, d_states, mu_pooled,
                                     Sigma_pooled, D_pooled, d_pooled)
            for strat in strat_order:
                tw_gross, wh = results[strat]
                mgmt_fee = compute_management_fee(X0, q, ALPHA_MGMT)
                _, total_tc = compute_transaction_costs(wh, returns, alpha1)

                metrics = compute_all_metrics(
                    tw_gross, total_tc, mgmt_fee, wh, returns, r_f_wealth)
                metrics['q'] = q
                metrics['Strategy'] = strat
                metrics['alpha1'] = alpha1
                sensitivity_rows.append(metrics)

        # Print q=5 summary for this alpha1
        q5_rows = [r for r in sensitivity_rows
                   if r['alpha1'] == alpha1 and r['q'] == 5]
        sr_line = "    q=5: " + " | ".join(
            f"{r['Strategy']}={r['SR']:+.4f}" for r in q5_rows)
        print(sr_line)

    df_sens = pd.DataFrame(sensitivity_rows)
    sens_path = os.path.join(RES_DIR, 'extension_sensitivity_results.csv')
    df_sens.to_csv(sens_path, index=False)
    print(f"\n[SAVED] {os.path.abspath(sens_path)}")

    # ------------------------------------------------------------------
    # Sensitivity summary table: SR for q=5 across cost levels
    # ------------------------------------------------------------------
    sens_summary_rows = []
    for alpha1 in ALPHA1_LEVELS:
        row = {'alpha1': alpha1}
        for strat in strat_order:
            match = df_sens[(df_sens['alpha1'] == alpha1) &
                            (df_sens['q'] == 5) &
                            (df_sens['Strategy'] == strat)]
            col_name = strat.replace('-', '_') + '_SR'
            row[col_name] = float(match['SR'].iloc[0])
        sens_summary_rows.append(row)

    df_sens_summary = pd.DataFrame(sens_summary_rows)
    sens_summary_path = os.path.join(RES_DIR, 'sensitivity_table.csv')
    df_sens_summary.to_csv(sens_summary_path, index=False)
    print(f"[SAVED] {os.path.abspath(sens_summary_path)}")

    print(f"\n{'='*72}")
    print("SENSITIVITY TABLE: Sharpe Ratio for q=5 across TC levels")
    print(f"{'='*72}")
    print(df_sens_summary.to_string(index=False, float_format='%.4f'))

    # ==================================================================
    # C) OPTIMAL q SUMMARY
    # ==================================================================
    cmmv_baseline = df_baseline[df_baseline['Strategy'] == 'CMMV']
    best_row = cmmv_baseline.loc[cmmv_baseline['SR'].idxmax()]
    print(f"\n{'='*72}")
    print(f"Optimal q for CMMV at baseline costs: q = {int(best_row['q'])} "
          f"(SR = {best_row['SR']:.4f})")
    print(f"{'='*72}")

    print("\n[DONE] Extension complete.")


if __name__ == '__main__':
    main()
