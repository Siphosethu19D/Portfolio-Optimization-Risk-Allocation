"""
Transaction cost and management fee model for the CMMV extension.

NOTE ON MYOPIC COST TREATMENT:
Transaction costs are applied POST-OPTIMISATION — we solve the CCQO without
costs, then subtract costs from terminal wealth. This is a myopic/additive
approach that does NOT integrate costs into the DP objective function.
It does not capture the forward-looking trade-off between rebalancing
benefits and costs. A fully dynamic cost-adjusted DP would require modifying
the Bellman equation (significantly harder). Our results represent a
POST-OPTIMISATION COST ASSESSMENT, not an optimal dynamic strategy with
transaction costs. (See Prof. Mba feedback point 7; Cajas Ch. 9)
"""

import numpy as np


def compute_transaction_costs(weights_history, sample_paths, alpha1):
    """
    Compute transaction costs for all paths over all periods.

    # Proportional transaction cost based on change in holding shares
    # TC_t = alpha1 * sum_i |u_t(i)/S_t(i) - u_{t-1}(i)/S_{t-1}(i)|

    Holding shares h_t(i) = u_t(i) / S_t(i), where S_t is the
    cumulative price level. With S_0 = 1 and S_t = prod_{k=1}^{t} R_k,
    the holding share change captures the actual number of shares traded.

    Parameters
    ----------
    weights_history : array (n_paths, T, n)
        Portfolio dollar allocations u_t at each step.
    sample_paths : array (n_paths, T, n)
        Simulated asset total returns R_{t+1} for each period.
    alpha1 : float
        Proportional transaction cost rate (e.g. 0.0005 for 5bp).

    Returns
    -------
    tc_per_period : array (n_paths, T)
        Transaction cost at each period for each path.
    total_tc : array (n_paths,)
        Total transaction cost over all periods for each path.
    """
    n_paths, T, n = weights_history.shape

    # Build cumulative price levels: S_0 = 1, S_t = S_{t-1} * R_t
    # S[t] is the price level at the START of period t (before investing)
    # S[0] = 1 (initial), S[1] = R_1, S[2] = R_1*R_2, etc.
    S = np.ones((n_paths, T + 1, n))
    for t in range(T):
        S[:, t + 1, :] = S[:, t, :] * sample_paths[:, t, :]

    # Compute holding shares: h_t(i) = u_t(i) / S_t(i)
    # u_t is the dollar allocation at time t, S_t is the price at time t
    tc_per_period = np.zeros((n_paths, T))

    for t in range(T):
        # Current holdings
        h_t = weights_history[:, t, :] / S[:, t, :]  # (n_paths, n)

        # Previous holdings (h_{-1} = 0, no prior position)
        if t == 0:
            h_prev = np.zeros((n_paths, n))
        else:
            h_prev = weights_history[:, t - 1, :] / S[:, t - 1, :]

        # TC_t = alpha1 * sum_i |h_t(i) - h_prev(i)|
        tc_per_period[:, t] = alpha1 * np.sum(np.abs(h_t - h_prev), axis=1)

    total_tc = np.sum(tc_per_period, axis=1)
    return tc_per_period, total_tc


def compute_management_fee(x0, q, alpha0):
    """
    Compute linear management fee proportional to cardinality.

    # Linear management fee proportional to cardinality
    # M = alpha0 * q * x0

    Parameters
    ----------
    x0 : float
        Initial wealth.
    q : int
        Cardinality constraint (number of active assets).
    alpha0 : float
        Unit management fee rate (e.g. 0.001 for 0.1%).

    Returns
    -------
    fee : float
        Total management fee.
    """
    return alpha0 * q * x0


def compute_net_terminal_wealth(x_T, total_tc, management_fee):
    """
    Compute net terminal wealth after deducting costs.

    # Costs deducted from terminal wealth (post-optimisation, see myopic note)
    # Net = x_T - total_TC - management_fee

    Parameters
    ----------
    x_T : array (n_paths,)
        Gross terminal wealth.
    total_tc : array (n_paths,)
        Total transaction costs per path.
    management_fee : float
        Management fee (scalar, same for all paths).

    Returns
    -------
    net_wealth : array (n_paths,)
        Net terminal wealth.
    """
    return x_T - total_tc - management_fee
