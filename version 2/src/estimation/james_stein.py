"""
James-Stein shrinkage estimator for covariance matrices.

Implements the Ledoit-Wolf single-factor (market-model) shrinkage as used
in the original project, which is the JSE-style estimator where the target
is a single-factor structure derived from the leading eigenvector.

Reference: Ledoit & Wolf (2003), "Improved estimation of the covariance matrix
of stock returns with an application to portfolio selection"
"""

import numpy as np
import pandas as pd


def james_stein(returns: pd.DataFrame) -> tuple[np.ndarray, float]:
    """
    James-Stein shrinkage of the sample covariance toward a single-factor target.

    The target F is constructed from the market (first principal component):
        F = beta @ beta.T * sigma_m^2 + diag(residual variances)

    Optimal shrinkage intensity alpha is computed analytically.

    Parameters
    ----------
    returns : T × N DataFrame of excess returns

    Returns
    -------
    shrunk_cov : N × N shrunk covariance matrix
    alpha      : shrinkage intensity in [0, 1] (0 = pure sample, 1 = pure target)
    """
    X = returns.values
    T, N = X.shape

    S = np.cov(X, rowvar=False)

    # --- Build single-factor target ---
    eigenvalues, eigenvectors = np.linalg.eigh(S)
    # Leading eigenvector = market factor proxy
    market = eigenvectors[:, -1]
    lambda_1 = eigenvalues[-1]

    # Factor loadings (betas) and residual variances
    beta = market * np.sqrt(lambda_1)          # N-vector
    residual_var = np.diag(S) - beta**2        # must stay non-negative
    residual_var = np.maximum(residual_var, 0)

    F = np.outer(beta, beta) + np.diag(residual_var)

    # --- Optimal shrinkage intensity (Ledoit-Wolf oracle) ---
    X_centered = X - X.mean(axis=0)
    sum_sq = 0.0
    for t in range(T):
        x = X_centered[t, :].reshape(-1, 1)
        sum_sq += np.linalg.norm(x @ x.T - S, "fro") ** 2
    delta_sq = sum_sq / T**2

    denominator = np.linalg.norm(S - F, "fro") ** 2
    alpha = min(1.0, delta_sq / denominator) if denominator > 0 else 0.0

    shrunk = alpha * F + (1 - alpha) * S
    return shrunk, alpha
