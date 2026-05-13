"""
Simulation engine for the CMMV portfolio replication.

Generates sample paths of asset returns following the Markov
regime-switching model (Assumption 1 in the paper).

For each path:
  1. The market state s_t evolves as a Markov chain.
  2. Conditional on s_t = i, returns R_{t+1} ~ N(mu(i), Sigma(i)).
"""

import numpy as np


def simulate_market_paths(
    n_paths: int,
    T: int,
    transition_matrix: np.ndarray,
    mu_states: dict,
    Sigma_states: dict,
    s0: int = 0,
    seed: int | None = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate sample paths of market states and asset returns.

    # Paper uses 10^7 paths; adjust n_paths based on computational resources

    Parameters
    ----------
    n_paths : int
        Number of sample paths to generate.
        Default for quick runs: 100_000.  Paper uses 10^7.
    T : int
        Number of time periods (quarters).
    transition_matrix : array (m, m)
        Markov chain transition probabilities.
        P[i, j] = Prob(s_{t+1} = j | s_t = i).
    mu_states : dict {int: array (n,)}
        Expected total-return vector for each market state.
    Sigma_states : dict {int: array (n, n)}
        Covariance matrix of returns for each market state.
    s0 : int
        Initial market state at t=0.
    seed : int or None
        Random seed for reproducibility.

    Returns
    -------
    returns : array (n_paths, T, n)
        Simulated asset return vectors for each path and period.
    states : array (n_paths, T)
        Simulated market state at each period for each path.
    """
    rng = np.random.default_rng(seed)
    m = transition_matrix.shape[0]
    n = len(mu_states[0])

    # Pre-compute Cholesky decompositions for each state
    chol = {}
    for s in range(m):
        chol[s] = np.linalg.cholesky(Sigma_states[s])

    states = np.empty((n_paths, T), dtype=np.int32)
    returns = np.empty((n_paths, T, n))

    # Initial state
    current_states = np.full(n_paths, s0, dtype=np.int32)

    for t in range(T):
        # Transition: sample next state for each path
        if t > 0:
            u = rng.random(n_paths)
            new_states = np.empty(n_paths, dtype=np.int32)
            for s in range(m):
                mask = current_states == s
                if not np.any(mask):
                    continue
                cum_prob = np.cumsum(transition_matrix[s])
                new_states[mask] = np.searchsorted(cum_prob, u[mask])
            current_states = new_states

        states[:, t] = current_states

        # Generate returns conditional on state
        for s in range(m):
            mask = current_states == s
            count = np.sum(mask)
            if count == 0:
                continue
            # R_{t+1} ~ N(mu(s), Sigma(s))
            z = rng.standard_normal((count, n))
            returns[mask, t, :] = mu_states[s] + z @ chol[s].T

    return returns, states
