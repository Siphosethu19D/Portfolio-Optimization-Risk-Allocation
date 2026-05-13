"""
Risk measurement functions for the CMMV portfolio replication.

Provides Sharpe ratio, drawdown, VaR, CVaR, turnover, and
Kupiec back-test statistics.
"""

import numpy as np
from scipy import stats


def compute_sharpe_ratio(terminal_wealth: np.ndarray,
                         risk_free_wealth: float) -> float:
    """
    Compute the Sharpe ratio from terminal wealth outcomes.

    SR = (E[x_T] - x_rf) / std(x_T)

    Parameters
    ----------
    terminal_wealth : array (n_paths,)
        Terminal wealth from each sample path.
    risk_free_wealth : float
        Terminal wealth from investing entirely in the risk-free asset.

    Returns
    -------
    sr : float
        Sharpe ratio.
    """
    excess = terminal_wealth - risk_free_wealth
    std = np.std(terminal_wealth)
    if std < 1e-15:
        return 0.0
    return np.mean(excess) / std


def compute_max_drawdown(wealth_paths: np.ndarray) -> tuple[float, float]:
    """
    Compute maximum drawdown statistics over wealth paths.

    Parameters
    ----------
    wealth_paths : array (n_paths, T) or (T,)
        Cumulative wealth at each period.  If 2-D, each row is a path.

    Returns
    -------
    avg_mdd : float
        Average maximum drawdown across paths.
    worst_mdd : float
        Worst-case (largest) maximum drawdown.
    """
    if wealth_paths.ndim == 1:
        wealth_paths = wealth_paths.reshape(1, -1)

    n_paths = wealth_paths.shape[0]
    mdds = np.empty(n_paths)

    for i in range(n_paths):
        running_max = np.maximum.accumulate(wealth_paths[i])
        drawdowns = (running_max - wealth_paths[i]) / np.where(
            running_max > 0, running_max, 1.0
        )
        mdds[i] = np.max(drawdowns)

    return float(np.mean(mdds)), float(np.max(mdds))


def compute_var(terminal_wealth: np.ndarray,
                confidence: float = 0.95) -> float:
    """
    Compute Value-at-Risk (VaR) of terminal wealth.

    VaR is the loss threshold such that losses exceed it with
    probability (1 - confidence).

    Parameters
    ----------
    terminal_wealth : array (n_paths,)
        Terminal wealth from each sample path.
    confidence : float
        Confidence level (e.g. 0.95 for 95% VaR).

    Returns
    -------
    var : float
        VaR as a positive loss amount (wealth shortfall from mean).
    """
    quantile = np.percentile(terminal_wealth, (1 - confidence) * 100)
    return float(np.mean(terminal_wealth) - quantile)


def compute_cvar(terminal_wealth: np.ndarray,
                 confidence: float = 0.95) -> float:
    """
    Compute Conditional Value-at-Risk (CVaR / Expected Shortfall).

    CVaR is the expected loss given that the loss exceeds VaR.

    Parameters
    ----------
    terminal_wealth : array (n_paths,)
        Terminal wealth from each sample path.
    confidence : float
        Confidence level (e.g. 0.95 for 95% CVaR).

    Returns
    -------
    cvar : float
        CVaR as a positive loss amount.
    """
    threshold = np.percentile(terminal_wealth, (1 - confidence) * 100)
    tail = terminal_wealth[terminal_wealth <= threshold]
    if len(tail) == 0:
        return 0.0
    return float(np.mean(terminal_wealth) - np.mean(tail))


def compute_turnover(weights_over_time: list[np.ndarray]) -> float:
    """
    Compute average portfolio turnover per period.

    Turnover_t = sum_i |w_t(i) - w_{t-1}(i)|

    Parameters
    ----------
    weights_over_time : list of arrays (n,)
        Portfolio weight vector at each rebalancing period.

    Returns
    -------
    avg_turnover : float
        Average turnover per period.
    """
    if len(weights_over_time) < 2:
        return 0.0

    turnovers = []
    prev_w = np.zeros_like(weights_over_time[0])
    for w in weights_over_time:
        turnovers.append(np.sum(np.abs(w - prev_w)))
        prev_w = w.copy()

    return float(np.mean(turnovers))


def kupiec_backtest(returns: np.ndarray, var_level: float,
                    confidence: float = 0.95) -> tuple[float, float]:
    """
    Kupiec Proportion of Failures (POF) back-test for VaR.

    Compares the actual number of VaR exceedances against the
    expected number under the null hypothesis.

    Parameters
    ----------
    returns : array (T,)
        Portfolio returns (or P&L) series.
    var_level : float
        VaR threshold (as a positive loss, e.g. 0.05 for 5% loss).
    confidence : float
        VaR confidence level (e.g. 0.95).

    Returns
    -------
    test_stat : float
        Kupiec likelihood ratio test statistic.
    p_value : float
        p-value from chi-squared distribution with 1 degree of freedom.
    """
    T_obs = len(returns)
    # Count exceedances (returns worse than -VaR)
    n_exceed = np.sum(returns < -var_level)
    p_expected = 1 - confidence
    p_actual = n_exceed / T_obs if T_obs > 0 else 0

    # Avoid log(0)
    eps = 1e-15
    p_actual = np.clip(p_actual, eps, 1 - eps)

    # Likelihood ratio statistic
    # LR = -2 * ln[ p_e^x * (1-p_e)^(T-x) / p_a^x * (1-p_a)^(T-x) ]
    lr = -2 * (
        n_exceed * np.log(p_expected / p_actual)
        + (T_obs - n_exceed) * np.log((1 - p_expected) / (1 - p_actual))
    )

    # Under H0, LR ~ chi2(1)
    p_value = 1 - stats.chi2.cdf(lr, df=1)

    return float(lr), float(p_value)
