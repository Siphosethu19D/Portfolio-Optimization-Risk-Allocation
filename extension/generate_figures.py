"""
Generate all extension figures (with transaction costs).

Figures 4-8: cost-adjusted Sharpe ratios, sensitivity analysis,
turnover, terminal wealth distributions, and tail risk profiles.

Styling convention (project-wide):
  CMV-static       → dotted blue,   marker='s'
  CMMV-independent → dashed orange, marker='^'
  CMMV (ours)      → solid green,   marker='o'
"""

import numpy as np
import pandas as pd
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from utils.parameters import N_ASSETS, T_QUARTERS, N_STATES, P_TRANS, X0, SIGMA, compute_mu, RF_CONSTANT, EPSILON
from utils.matrix_utils import compute_d_vector, compute_D_matrix
from utils.simulation import simulate_market_paths
from replication.solver import (
    compute_theta_recursive, compute_theta_independent,
    compute_pooled_parameters,
    apply_cmmv_policy, apply_cmv_static_policy,
    apply_cmmv_independent_policy,
)
from extension.cost_model import compute_transaction_costs, compute_management_fee, compute_net_terminal_wealth

# ===================================================================
# Paths
# ===================================================================
ROOT = os.path.join(os.path.dirname(__file__), '..')
FIG_DIR = os.path.join(ROOT, 'figures')
RES_DIR = os.path.join(ROOT, 'results')
os.makedirs(FIG_DIR, exist_ok=True)

# ===================================================================
# Styling
# ===================================================================
sns.set_style("whitegrid")

STYLE = {
    'CMV-static':        dict(color='tab:blue',   linestyle=':', marker='s', label='CMV-static'),
    'CMMV-independent':  dict(color='tab:orange', linestyle='--', marker='^', label='CMMV-independent'),
    'CMMV':              dict(color='tab:green',  linestyle='-', marker='o', label='CMMV'),
}

# ===================================================================
# Load data
# ===================================================================
df_base = pd.read_csv(os.path.join(RES_DIR, 'extension_baseline_results.csv'))
df_sens = pd.read_csv(os.path.join(RES_DIR, 'extension_sensitivity_results.csv'))


# ===================================================================
# FIGURE 4 — Sharpe ratio vs cardinality with costs (baseline)
# ===================================================================
def figure4():
    print("Generating Figure 4: Sharpe vs cardinality (baseline costs)...")
    fig, ax = plt.subplots(figsize=(8, 5))

    for strat in ['CMV-static', 'CMMV-independent', 'CMMV']:
        sub = df_base[df_base['Strategy'] == strat].sort_values('q')
        ax.plot(sub['q'], sub['SR'], **STYLE[strat], markersize=6, linewidth=1.5)

    # Annotate CMMV peak
    cmmv = df_base[df_base['Strategy'] == 'CMMV'].sort_values('q')
    best_idx = cmmv['SR'].idxmax()
    best_q = int(cmmv.loc[best_idx, 'q'])
    best_sr = cmmv.loc[best_idx, 'SR']
    ax.annotate(f'Peak: q={best_q}\nSR={best_sr:.2f}',
                xy=(best_q, best_sr),
                xytext=(best_q - 2.5, best_sr * 0.85),
                arrowprops=dict(arrowstyle='->', color='tab:green'),
                fontsize=9, color='tab:green')

    ax.set_xlabel('Cardinality q')
    ax.set_ylabel('Sharpe Ratio (net of costs)')
    ax.set_title(r'Sharpe Ratio vs Cardinality ($\alpha_0$=0.1%, $\alpha_1$=5bp)')
    ax.set_xticks(range(1, N_ASSETS + 1))
    ax.legend(loc='best', fontsize=10)

    plt.tight_layout()
    path = os.path.join(FIG_DIR, 'fig4_sharpe_vs_cardinality_with_costs.png')
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  [SAVED] {path}")


# ===================================================================
# FIGURE 5 — Sensitivity: CMMV SR vs q across TC levels
# ===================================================================
def figure5():
    print("Generating Figure 5: CMMV sensitivity to TC levels...")
    fig, ax = plt.subplots(figsize=(8, 5))

    alpha_levels = sorted(df_sens['alpha1'].unique())
    # Colour gradient: light to dark green
    greens = plt.cm.Greens(np.linspace(0.35, 0.95, len(alpha_levels)))
    line_styles = ['-', '--', '-.', ':']

    for i, alpha1 in enumerate(alpha_levels):
        sub = df_sens[(df_sens['Strategy'] == 'CMMV') &
                      (df_sens['alpha1'] == alpha1)].sort_values('q')
        bp = int(alpha1 * 10000)
        ax.plot(sub['q'], sub['SR'],
                color=greens[i], linestyle=line_styles[i % len(line_styles)],
                marker='o', markersize=5, linewidth=1.5,
                label=f'{bp}bp')

    ax.set_xlabel('Cardinality q')
    ax.set_ylabel('Sharpe Ratio (net of costs)')
    ax.set_title('Sensitivity of CMMV Sharpe Ratio to Transaction Cost Level')
    ax.set_xticks(range(1, N_ASSETS + 1))
    ax.legend(title=r'$\alpha_1$ (TC rate)', loc='best', fontsize=9)

    plt.tight_layout()
    path = os.path.join(FIG_DIR, 'fig5_sensitivity_sharpe_vs_q.png')
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  [SAVED] {path}")


# ===================================================================
# FIGURE 6 — Turnover vs cardinality
# ===================================================================
def figure6():
    print("Generating Figure 6: Turnover vs cardinality...")
    fig, ax = plt.subplots(figsize=(8, 5))

    for strat in ['CMV-static', 'CMMV-independent', 'CMMV']:
        sub = df_base[df_base['Strategy'] == strat].sort_values('q')
        ax.plot(sub['q'], sub['Turnover'], **STYLE[strat], markersize=6, linewidth=1.5)

    ax.set_xlabel('Cardinality q')
    ax.set_ylabel('Average Turnover per Period')
    ax.set_title('Portfolio Turnover vs Cardinality (baseline costs)')
    ax.set_xticks(range(1, N_ASSETS + 1))
    ax.legend(loc='best', fontsize=10)
    ax.set_yscale('log')

    plt.tight_layout()
    path = os.path.join(FIG_DIR, 'fig6_turnover_vs_q.png')
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  [SAVED] {path}")


# ===================================================================
# FIGURE 7 — Terminal wealth distribution (q=5)
# ===================================================================
def figure7():
    print("Generating Figure 7: Terminal wealth distribution (q=5)...")

    # Need to re-simulate to get the actual terminal wealth arrays
    R_F = 1.0 + RF_CONSTANT
    GAMMA_0 = R_F ** T_QUARTERS
    PI_STAT = np.array([0.55, 0.45])
    OMEGA = 1.0
    S0 = 0
    q = 5
    ALPHA_TC = 0.0005
    ALPHA_MGMT = 0.001
    N_PATHS = 100_000

    mu_states = {s: compute_mu(s) for s in range(N_STATES)}
    D_states, d_states = {}, {}
    for s in range(N_STATES):
        d_s = compute_d_vector(mu_states[s], R_F)
        D_states[s] = compute_D_matrix(SIGMA[s], d_s)
        d_states[s] = d_s
    mu_pooled, Sigma_pooled = compute_pooled_parameters(mu_states, SIGMA, PI_STAT)
    d_pooled = compute_d_vector(mu_pooled, R_F)
    D_pooled = compute_D_matrix(Sigma_pooled, d_pooled)

    returns, states = simulate_market_paths(
        n_paths=N_PATHS, T=T_QUARTERS,
        transition_matrix=P_TRANS,
        mu_states=mu_states, Sigma_states=SIGMA,
        s0=S0, seed=42
    )

    # CMMV
    theta_cmmv, k_cmmv = compute_theta_recursive(
        T_QUARTERS, N_STATES, q, D_states, d_states, P_TRANS)
    tw_cmmv, wh_cmmv = apply_cmmv_policy(
        returns, states, k_cmmv, theta_cmmv[0, S0],
        GAMMA_0, OMEGA, X0, T_QUARTERS, R_F)
    _, tc_cmmv = compute_transaction_costs(wh_cmmv, returns, ALPHA_TC)
    mgmt = compute_management_fee(X0, q, ALPHA_MGMT)
    net_cmmv = compute_net_terminal_wealth(tw_cmmv, tc_cmmv, mgmt)

    # CMMV-independent
    theta_ind, k_ind = compute_theta_independent(T_QUARTERS, q, D_pooled, d_pooled)
    tw_ind, wh_ind = apply_cmmv_independent_policy(
        returns, k_ind, theta_ind[0],
        GAMMA_0, OMEGA, X0, T_QUARTERS, R_F)
    _, tc_ind = compute_transaction_costs(wh_ind, returns, ALPHA_TC)
    net_ind = compute_net_terminal_wealth(tw_ind, tc_ind, mgmt)

    # CMV-static
    tw_stat, wh_stat = apply_cmv_static_policy(
        returns, mu_pooled, Sigma_pooled, q, X0, T_QUARTERS, R_F)
    _, tc_stat = compute_transaction_costs(wh_stat, returns, ALPHA_TC)
    net_stat = compute_net_terminal_wealth(tw_stat, tc_stat, mgmt)

    # VaR95 for each
    from utils.risk_metrics import compute_var
    var95_cmmv = compute_var(net_cmmv, 0.95)
    var95_ind = compute_var(net_ind, 0.95)
    var95_stat = compute_var(net_stat, 0.95)

    fig, ax = plt.subplots(figsize=(10, 5))

    # KDE plots
    for net, sty, var95, name in [
        (net_stat, STYLE['CMV-static'], var95_stat, 'CMV-static'),
        (net_ind, STYLE['CMMV-independent'], var95_ind, 'CMMV-independent'),
        (net_cmmv, STYLE['CMMV'], var95_cmmv, 'CMMV'),
    ]:
        ax.hist(net, bins=100, density=True, alpha=0.3,
                color=sty['color'], label=sty['label'])
        # VaR line
        var_x = np.mean(net) - var95
        ax.axvline(var_x, color=sty['color'], linestyle='--', linewidth=1.2,
                   alpha=0.8)
        ax.text(var_x, ax.get_ylim()[1] * 0.01, f'VaR95',
                color=sty['color'], fontsize=7, rotation=90,
                va='bottom', ha='right')

    ax.set_xlabel('Net Terminal Wealth')
    ax.set_ylabel('Density')
    ax.set_title('Distribution of Net Terminal Wealth (q=5, baseline costs)')
    ax.legend(loc='upper right', fontsize=9)

    plt.tight_layout()
    path = os.path.join(FIG_DIR, 'fig7_terminal_wealth_distribution.png')
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  [SAVED] {path}")


# ===================================================================
# FIGURE 8 — VaR and CVaR vs cardinality
# ===================================================================
def figure8():
    print("Generating Figure 8: VaR and CVaR vs cardinality...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    for strat in ['CMV-static', 'CMMV-independent', 'CMMV']:
        sub = df_base[df_base['Strategy'] == strat].sort_values('q')
        sty = STYLE[strat]

        ax1.plot(sub['q'], sub['VaR95'], color=sty['color'],
                 linestyle=sty['linestyle'], marker=sty['marker'],
                 markersize=5, linewidth=1.5, label=sty['label'])

        ax2.plot(sub['q'], sub['CVaR95'], color=sty['color'],
                 linestyle=sty['linestyle'], marker=sty['marker'],
                 markersize=5, linewidth=1.5, label=sty['label'])

    ax1.set_xlabel('Cardinality q')
    ax1.set_ylabel('VaR (95%)')
    ax1.set_title('Value-at-Risk (95%) vs Cardinality')
    ax1.set_xticks(range(1, N_ASSETS + 1))
    ax1.legend(loc='best', fontsize=9)

    ax2.set_xlabel('Cardinality q')
    ax2.set_ylabel('CVaR (95%)')
    ax2.set_title('Conditional VaR (95%) vs Cardinality')
    ax2.set_xticks(range(1, N_ASSETS + 1))
    ax2.legend(loc='best', fontsize=9)

    plt.tight_layout()
    path = os.path.join(FIG_DIR, 'fig8_var_cvar_vs_q.png')
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  [SAVED] {path}")


# ===================================================================
# Main
# ===================================================================
if __name__ == '__main__':
    import time
    t0 = time.time()
    figure4()
    figure5()
    figure6()
    figure7()
    figure8()
    print(f"\n[DONE] All extension figures generated in {time.time()-t0:.1f}s")
