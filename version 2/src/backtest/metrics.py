"""
Performance metrics for evaluating backtest results.
"""

import numpy as np
import pandas as pd
from scipy import stats


def annualized_return(returns: pd.Series, periods_per_year: int = 52) -> float:
    return returns.mean() * periods_per_year


def annualized_volatility(returns: pd.Series, periods_per_year: int = 52) -> float:
    return returns.std() * np.sqrt(periods_per_year)


def sharpe_ratio(returns: pd.Series, periods_per_year: int = 52) -> float:
    """Annualized Sharpe ratio (returns are already excess)."""
    vol = annualized_volatility(returns, periods_per_year)
    if vol == 0:
        return np.nan
    return annualized_return(returns, periods_per_year) / vol


def max_drawdown(returns: pd.Series) -> float:
    """Maximum drawdown from cumulative return series."""
    cum = (1 + returns).cumprod()
    rolling_max = cum.cummax()
    drawdown = (cum - rolling_max) / rolling_max
    return drawdown.min()


def summary_table(returns: pd.DataFrame, periods_per_year: int = 52) -> pd.DataFrame:
    """
    Compute a summary statistics table for each strategy column.

    Returns a DataFrame with columns: Ann. Return, Ann. Volatility, Sharpe, Max Drawdown.
    """
    rows = {}
    for col in returns.columns:
        r = returns[col].dropna()
        rows[col] = {
            "Ann. Return (%)": annualized_return(r, periods_per_year) * 100,
            "Ann. Volatility (%)": annualized_volatility(r, periods_per_year) * 100,
            "Sharpe Ratio": sharpe_ratio(r, periods_per_year),
            "Max Drawdown (%)": max_drawdown(r) * 100,
        }
    return pd.DataFrame(rows).T.round(4)


def diebold_mariano_test(
    returns_a: pd.Series,
    returns_b: pd.Series,
) -> dict:
    """
    Diebold-Mariano test for equal out-of-sample portfolio variance.

    Tests H0: E[r_a^2] == E[r_b^2]  (equal squared realized returns as loss proxy)
    A negative test statistic means strategy A has lower realized variance.

    Returns dict with keys: statistic, p_value, interpretation.
    """
    loss_a = returns_a.values ** 2
    loss_b = returns_b.values ** 2
    d = loss_a - loss_b

    T = len(d)
    d_bar = d.mean()
    # Newey-West variance with lag 1
    gamma_0 = np.var(d, ddof=1)
    gamma_1 = np.cov(d[:-1], d[1:])[0, 1] if T > 1 else 0.0
    nw_var = (gamma_0 + 2 * gamma_1) / T

    if nw_var <= 0:
        return {"statistic": np.nan, "p_value": np.nan, "interpretation": "undefined"}

    dm_stat = d_bar / np.sqrt(nw_var)
    p_value = 2 * (1 - stats.norm.cdf(abs(dm_stat)))

    interp = (
        f"Strategy A has {'lower' if dm_stat < 0 else 'higher'} realized variance "
        f"({'significant' if p_value < 0.05 else 'not significant'} at 5%)"
    )

    return {"statistic": round(dm_stat, 4), "p_value": round(p_value, 4), "interpretation": interp}
