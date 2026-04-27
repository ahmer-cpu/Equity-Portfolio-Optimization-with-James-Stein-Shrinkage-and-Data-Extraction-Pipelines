"""
End-to-end pipeline runner.
Executes: data download -> estimation comparison -> rolling backtest -> results.

Run from the version 2/ directory:
    python run_pipeline.py
"""

import sys
sys.stdout.reconfigure(encoding="utf-8")
import os
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from src.data.download import get_sp100_tickers, download_prices
from src.data.preprocess import to_log_returns, get_risk_free_rate, to_excess_returns
from src.estimation.sample import sample_covariance
from src.estimation.ledoit_wolf import ledoit_wolf
from src.estimation.james_stein import james_stein
from src.optimization.mvo import global_minimum_variance
from src.backtest.rolling import run_backtest
from src.backtest.metrics import summary_table, diebold_mariano_test

os.makedirs("data", exist_ok=True)
os.makedirs("results", exist_ok=True)

# ── CONFIG ────────────────────────────────────────────────────────────────────
START       = "2020-01-01"
END         = "2025-04-01"
RF_ANNUAL   = 0.0422
N_STOCKS    = 100      # top N S&P 500 stocks by market cap
WINDOW      = 252      # trading days used per estimation (1 year)
REBAL_FREQ  = "M"      # "M"=monthly, "W"=weekly, "D"=daily
PERIODS_PER_YEAR = 12  # for annualizing: 12=monthly, 52=weekly, 252=daily

PRICES_PATH  = "data/prices_daily.csv"
EXCESS_PATH  = "data/excess_returns_daily.csv"

# ── STEP 1: Download data ─────────────────────────────────────────────────────
print("=" * 60)
print("STEP 1: Downloading data")
print("=" * 60)

if os.path.exists(PRICES_PATH):
    prices = pd.read_csv(PRICES_PATH, index_col=0, parse_dates=True)
    if prices.shape[1] >= N_STOCKS:
        print(f"  Loaded cached prices: {prices.shape[0]} days x {prices.shape[1]} stocks")
    else:
        prices = None
else:
    prices = None

if prices is None:
    tickers = get_sp100_tickers()
    prices = download_prices(tickers, start=START, end=END, interval="1d")
    print(f"  Price matrix: {prices.shape[0]} days x {prices.shape[1]} stocks")
    prices.to_csv(PRICES_PATH)
    print(f"  Saved -> {PRICES_PATH}")

# ── STEP 2: Excess returns ────────────────────────────────────────────────────
print("\nSTEP 2: Computing excess log returns")
log_ret = to_log_returns(prices)
rf      = get_risk_free_rate(annual_rate=RF_ANNUAL, freq="D")
excess  = to_excess_returns(log_ret, rf)
excess.to_csv(EXCESS_PATH)
print(f"  Excess return matrix: {excess.shape[0]} days x {excess.shape[1]} stocks")
print(f"  T/N ratio: {excess.shape[0] / excess.shape[1]:.2f}  (window/N = {WINDOW}/{excess.shape[1]} = {WINDOW/excess.shape[1]:.2f})")

# ── STEP 3: Estimator inspection ─────────────────────────────────────────────
print("\nSTEP 3: Estimator inspection (most recent 252-day window)")
win = excess.iloc[-WINDOW:]
N   = win.shape[1]

S         = sample_covariance(win)
LW        = ledoit_wolf(win)
JS, alpha = james_stein(win)

print(f"  N stocks in window: {N},  T/N = {WINDOW/N:.2f}")
print(f"  JS shrinkage intensity alpha = {alpha:.4f}")
print(f"  Condition numbers:")
print(f"    Sample:       {np.linalg.cond(S):.1f}")
print(f"    Ledoit-Wolf:  {np.linalg.cond(LW):.1f}")
print(f"    James-Stein:  {np.linalg.cond(JS):.1f}")

# Eigenvalue spectra
fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
for ax, (name, M) in zip(axes, [("Sample", S), ("Ledoit-Wolf", LW), ("James-Stein", JS)]):
    eigs = np.sort(np.linalg.eigvalsh(M))[::-1]
    ax.plot(eigs, linewidth=1.2)
    ax.set_title(name)
    ax.set_xlabel("Eigenvalue rank")
    ax.set_yscale("log")
axes[0].set_ylabel("Eigenvalue (log scale)")
plt.suptitle(f"Eigenvalue Spectra  (T={WINDOW}, N={N}, T/N={WINDOW/N:.2f})")
plt.tight_layout()
plt.savefig("results/eigenvalue_spectra.png", dpi=150)
plt.close()
print("  Saved -> results/eigenvalue_spectra.png")

# Covariance heatmaps
n_heat = min(40, N)
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for ax, (name, M) in zip(axes, [("Sample", S), ("Ledoit-Wolf", LW), ("James-Stein", JS)]):
    sns.heatmap(M[:n_heat, :n_heat], ax=ax, cmap="RdBu_r", center=0,
                xticklabels=False, yticklabels=False)
    ax.set_title(name)
plt.suptitle(f"Covariance Heatmaps (top {n_heat} stocks)")
plt.tight_layout()
plt.savefig("results/covariance_heatmaps.png", dpi=150)
plt.close()
print("  Saved -> results/covariance_heatmaps.png")

# ── STEP 4: Rolling backtest ──────────────────────────────────────────────────
print(f"\nSTEP 4: Rolling backtest  (window={WINDOW}d, rebal={REBAL_FREQ})")

def js_estimator(returns):
    cov, _ = james_stein(returns)
    return cov

estimators = {
    "Sample":      sample_covariance,
    "Ledoit-Wolf": ledoit_wolf,
    "James-Stein": js_estimator,
}

port_returns = run_backtest(
    excess, estimators, window=WINDOW, rebal_freq=REBAL_FREQ
)

# 1/N benchmark: equal-weight, same holding periods
# Align to the same rebalancing structure by summing daily equal-weight returns per period
dates = excess.index
all_rebal = port_returns.index  # end-of-period dates from backtest

from src.backtest.rolling import _rebal_dates
start_dates = _rebal_dates(dates, REBAL_FREQ)
start_dates = [d for d in start_dates if d >= dates[WINDOW]]

eq_rets = []
for i in range(len(start_dates) - 1):
    hold = excess.loc[(dates > start_dates[i]) & (dates <= start_dates[i + 1])]
    if hold.empty:
        continue
    daily_eq = hold.mean(axis=1)
    eq_rets.append((1 + daily_eq).prod() - 1)

port_returns["1/N"] = eq_rets[:len(port_returns)]

print(f"  Backtest: {port_returns.index[0].date()} -> {port_returns.index[-1].date()}")
print(f"  Periods out-of-sample: {len(port_returns)}")
port_returns.to_csv("results/portfolio_returns.csv")
print("  Saved -> results/portfolio_returns.csv")

# ── STEP 5: Performance summary ───────────────────────────────────────────────
print("\nSTEP 5: Performance summary")
summary = summary_table(port_returns, periods_per_year=PERIODS_PER_YEAR)
print(summary.to_string())
summary.to_csv("results/performance_summary.csv")
print("  Saved -> results/performance_summary.csv")

# ── STEP 6: Diebold-Mariano tests ─────────────────────────────────────────────
print("\nSTEP 6: Diebold-Mariano tests (vs Sample baseline)")
dm_rows = []
for name in ["Ledoit-Wolf", "James-Stein", "1/N"]:
    result = diebold_mariano_test(port_returns["Sample"], port_returns[name])
    print(f"  Sample vs {name}:  DM={result['statistic']},  p={result['p_value']}")
    print(f"    {result['interpretation']}")
    dm_rows.append({"vs": name, **result})
pd.DataFrame(dm_rows).to_csv("results/dm_tests.csv", index=False)
print("  Saved -> results/dm_tests.csv")

# ── STEP 7: Plots ─────────────────────────────────────────────────────────────
print("\nSTEP 7: Plots")

cum = (1 + port_returns).cumprod()
fig, ax = plt.subplots(figsize=(12, 5))
for col in cum.columns:
    ax.plot(cum.index, cum[col], label=col, linestyle="--" if col == "1/N" else "-")
ax.set_title(
    f"Cumulative Out-of-Sample Returns -- GMV Portfolio\n"
    f"Top {N} S&P 500 stocks, {WINDOW}-day estimation window, monthly rebalancing"
)
ax.set_ylabel("Cumulative gross return")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("results/cumulative_returns.png", dpi=150)
plt.close()
print("  Saved -> results/cumulative_returns.png")

roll_var = port_returns.rolling(12).var() * PERIODS_PER_YEAR
fig, ax = plt.subplots(figsize=(12, 4))
for col in roll_var.columns:
    ax.plot(roll_var.index, roll_var[col], label=col, linestyle="--" if col == "1/N" else "-")
ax.set_title("Rolling 12-month Realized Annualized Variance")
ax.set_ylabel("Annualized variance")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("results/rolling_variance.png", dpi=150)
plt.close()
print("  Saved -> results/rolling_variance.png")

print("\n" + "=" * 60)
print("DONE. All outputs in data/ and results/")
print("=" * 60)
