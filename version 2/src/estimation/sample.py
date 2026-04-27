"""
Sample covariance estimator.
"""

import numpy as np
import pandas as pd


def sample_covariance(returns: pd.DataFrame) -> np.ndarray:
    """
    Compute the sample covariance matrix from a returns DataFrame.

    Parameters
    ----------
    returns : T × N DataFrame of excess returns

    Returns
    -------
    N × N numpy array
    """
    return np.cov(returns.values, rowvar=False)
