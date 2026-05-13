"""
Matrix utility functions for the CMMV portfolio replication.

Key equations from Wang et al. (2026):
  - Excess return mean:   d_t(i) = mu_t(i) - r_{t+1} * e       [Eq. 8]
  - Second moment matrix: D_t(i) = Sigma_t(i) + d_t(i)*d_t(i)^T [Eq. 9]
"""

import numpy as np
import warnings


def compute_d_vector(mu_i: np.ndarray, r_t: float) -> np.ndarray:
    """
    Compute the mean excess return vector.

    d_t(i) = mu_t(i) - r_{t+1} * e      (Paper Eq. 8)

    Parameters
    ----------
    mu_i : array (n,)
        Expected total-return vector for market state i.
    r_t : float
        Risk-free return for the period (total return, e.g. 1.0024).
        If passed as a decimal rate (e.g. 0.0024), the function still
        works — just be consistent across calls.

    Returns
    -------
    d : array (n,)
        Mean excess return vector.
    """
    e = np.ones(len(mu_i))
    return mu_i - r_t * e


def compute_D_matrix(Sigma_i: np.ndarray, d_i: np.ndarray) -> np.ndarray:
    """
    Compute the second-moment matrix of excess returns.

    D_t(i) = Sigma_t(i) + d_t(i) * d_t(i)^T      (Paper Eq. 9)

    Parameters
    ----------
    Sigma_i : array (n, n)
        Covariance matrix of returns for market state i.
    d_i : array (n,)
        Mean excess return vector (from compute_d_vector).

    Returns
    -------
    D : array (n, n)
        Second moment matrix of excess returns.
    """
    return Sigma_i + np.outer(d_i, d_i)


def regularise_matrix(D: np.ndarray, method: str = 'ridge',
                      epsilon: float = 1e-6) -> np.ndarray:
    """
    Ensure a matrix is positive definite via regularisation.

    Checks positive definiteness and applies ridge regularisation
    if needed: D_reg = D + epsilon * I.

    # Regularisation to ensure PD — cf. Cajas Ch. 3 on shrinkage estimation

    Parameters
    ----------
    D : array (n, n)
        Symmetric matrix to check / regularise.
    method : str
        Regularisation method. Currently only 'ridge' is implemented.
    epsilon : float
        Ridge parameter added to the diagonal.

    Returns
    -------
    D_reg : array (n, n)
        Regularised positive-definite matrix.
    """
    if method != 'ridge':
        raise ValueError(f"Unknown regularisation method: {method}")

    # Check positive definiteness via eigenvalues
    eigvals = np.linalg.eigvalsh(D)
    if np.all(eigvals > 0):
        return D.copy()

    # Apply ridge regularisation
    warnings.warn(
        f"Matrix is not positive definite (min eigenvalue = {eigvals.min():.2e}). "
        f"Applying ridge regularisation with epsilon = {epsilon}."
    )
    n = D.shape[0]
    D_reg = D + epsilon * np.eye(n)

    # Verify fix
    eigvals_reg = np.linalg.eigvalsh(D_reg)
    if not np.all(eigvals_reg > 0):
        warnings.warn(
            f"Regularisation insufficient (min eigenvalue = {eigvals_reg.min():.2e}). "
            f"Increasing epsilon."
        )
        D_reg = D + max(epsilon, abs(eigvals.min()) + epsilon) * np.eye(n)

    return D_reg
