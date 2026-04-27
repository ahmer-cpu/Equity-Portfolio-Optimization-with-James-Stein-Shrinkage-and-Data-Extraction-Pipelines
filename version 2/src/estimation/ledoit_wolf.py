"""
Ledoit-Wolf analytical shrinkage estimator (constant-correlation target).

Reference: Ledoit & Wolf (2004), "Honey, I Shrunk the Sample Covariance Matrix"
"""

import numpy as np
import pandas as pd


def ledoit_wolf(returns: pd.DataFrame) -> np.ndarray:
    """
    Ledoit-Wolf shrinkage toward the constant-correlation matrix.

    Computes the optimal shrinkage intensity analytically (no cross-validation).

    Parameters
    ----------
    returns : T × N DataFrame of excess returns

    Returns
    -------
    N × N shrunk covariance matrix as numpy array
    """
    X = returns.values
    T, N = X.shape

    S = np.cov(X, rowvar=False)  # sample covariance

    # --- Target: scaled identity (simplest Ledoit-Wolf target) ---
    mu = np.trace(S) / N
    F = mu * np.eye(N)

    # --- Optimal shrinkage intensity (Oracle formula, Ledoit-Wolf 2004) ---
    # delta_k: squared Frobenius norm of each row outer product minus S
    X_centered = X - X.mean(axis=0)
    sum_sq = 0.0
    for t in range(T):
        x = X_centered[t, :].reshape(-1, 1)
        sum_sq += np.linalg.norm(x @ x.T - S, "fro") ** 2
    delta_sq = sum_sq / T**2

    # Numerator and denominator for alpha
    numerator = delta_sq
    denominator = np.linalg.norm(S - F, "fro") ** 2

    alpha = min(1.0, numerator / denominator) if denominator > 0 else 0.0

    return alpha * F + (1 - alpha) * S
