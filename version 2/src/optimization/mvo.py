"""
Portfolio optimization solvers.

All solvers return a weight vector w (N,) that sums to 1.
Short selling is allowed by default.
"""

import numpy as np
from scipy.optimize import minimize


def global_minimum_variance(cov: np.ndarray) -> np.ndarray:
    """
    Solve the Global Minimum Variance (GMV) portfolio analytically.

    w* = Sigma^{-1} 1 / (1^T Sigma^{-1} 1)

    Parameters
    ----------
    cov : N × N covariance matrix

    Returns
    -------
    w : N-vector of portfolio weights summing to 1
    """
    N = cov.shape[0]
    ones = np.ones(N)
    try:
        cov_inv = np.linalg.inv(cov)
    except np.linalg.LinAlgError:
        cov_inv = np.linalg.pinv(cov)

    raw = cov_inv @ ones
    return raw / raw.sum()


def maximum_sharpe(
    cov: np.ndarray,
    mu: np.ndarray,
    allow_short: bool = True,
) -> np.ndarray:
    """
    Solve for the Maximum Sharpe Ratio (tangency) portfolio.

    Uses the closed-form solution when short selling is allowed:
        w* = Sigma^{-1} mu / (1^T Sigma^{-1} mu)

    Falls back to numerical optimization when short selling is disallowed.

    Parameters
    ----------
    cov         : N × N covariance matrix
    mu          : N-vector of expected excess returns
    allow_short : if False, enforces w >= 0 constraints

    Returns
    -------
    w : N-vector of portfolio weights summing to 1
    """
    N = cov.shape[0]

    if allow_short:
        try:
            cov_inv = np.linalg.inv(cov)
        except np.linalg.LinAlgError:
            cov_inv = np.linalg.pinv(cov)
        raw = cov_inv @ mu
        total = raw.sum()
        if abs(total) < 1e-12:
            return global_minimum_variance(cov)
        return raw / total

    # Numerical fallback for long-only
    def neg_sharpe(w):
        port_ret = w @ mu
        port_var = w @ cov @ w
        if port_var <= 0:
            return 0.0
        return -port_ret / np.sqrt(port_var)

    constraints = [{"type": "eq", "fun": lambda w: w.sum() - 1}]
    bounds = [(0, None)] * N
    w0 = np.ones(N) / N
    result = minimize(neg_sharpe, w0, method="SLSQP", bounds=bounds, constraints=constraints)
    return result.x
