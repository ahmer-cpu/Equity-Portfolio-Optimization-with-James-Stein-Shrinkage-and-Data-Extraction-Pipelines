"""
Walk-forward rolling backtest engine with configurable rebalancing frequency.

At each rebalancing date:
  1. Slice the past `window` days of excess returns
  2. Call each estimator to get a covariance matrix
  3. Solve for portfolio weights
  4. Hold those weights until the next rebalancing date
  5. Record the cumulative return over the holding period
"""

import numpy as np
import pandas as pd
from typing import Callable


Estimator = Callable[[pd.DataFrame], np.ndarray]


def _rebal_dates(index: pd.DatetimeIndex, freq: str) -> list:
    """Return first trading day of each period after grouping by freq."""
    s = pd.Series(index, index=index)
    groups = s.groupby(s.index.to_period(freq))
    return [g.iloc[0] for _, g in groups]


def run_backtest(
    excess_returns: pd.DataFrame,
    estimators: dict[str, Estimator],
    window: int = 252,
    rebal_freq: str = "M",
    optimizer: Callable[[np.ndarray], np.ndarray] = None,
) -> pd.DataFrame:
    """
    Walk-forward backtest comparing multiple covariance estimators.

    Parameters
    ----------
    excess_returns : T x N DataFrame of daily excess returns
    estimators     : dict mapping strategy name -> estimator function.
                     Each estimator takes a DataFrame, returns an N x N array.
    window         : number of past trading days used for estimation (e.g. 252)
    rebal_freq     : pandas period alias for rebalancing frequency.
                     "M" = monthly, "W" = weekly, "D" = daily.
    optimizer      : function mapping covariance matrix -> weight vector.
                     Defaults to global_minimum_variance.

    Returns
    -------
    DataFrame of per-period portfolio returns (one column per strategy),
    indexed by the end date of each holding period.
    """
    from src.optimization.mvo import global_minimum_variance

    if optimizer is None:
        optimizer = global_minimum_variance

    dates = excess_returns.index
    burn_in_end = dates[window]

    # All candidate rebalancing dates
    all_rebal = _rebal_dates(dates, rebal_freq)

    # Keep only those after the burn-in period
    rebal = [d for d in all_rebal if d >= burn_in_end]

    if len(rebal) < 2:
        raise ValueError(
            f"Not enough data for even one holding period. "
            f"Need window={window} days before first rebalancing date."
        )

    results = {name: [] for name in estimators}
    result_dates = []

    for i in range(len(rebal) - 1):
        rebal_date = rebal[i]
        next_rebal_date = rebal[i + 1]

        t = dates.get_loc(rebal_date)
        in_sample = excess_returns.iloc[t - window : t]

        # Holding-period returns: day after rebal_date up to and including next_rebal_date
        hold = excess_returns.loc[
            (dates > rebal_date) & (dates <= next_rebal_date)
        ]

        if hold.empty:
            continue

        for name, estimator in estimators.items():
            cov = estimator(in_sample)
            w = optimizer(cov)
            daily_port = hold.values @ w          # daily portfolio return series
            period_ret = (1 + daily_port).prod() - 1  # compound over holding period
            results[name].append(period_ret)

        result_dates.append(next_rebal_date)

    return pd.DataFrame(results, index=result_dates)
