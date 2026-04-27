# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Academic research project implementing equity portfolio optimization using James–Stein shrinkage estimation. The project has two phases: a data extraction pipeline (Python scripts) and portfolio optimization analysis (Jupyter notebooks). Project website: https://ahmernadeem.com/research/portfolio-optimization.html

## Commands

### Data Extraction Pipeline (run in order)

```bash
# Fetch S&P 500 tickers from Slickcharts
python "data extraction/scrape_S&P_ordered.py"

# Download adjusted close prices for all S&P 500 stocks (~30-60 min)
python "data extraction/savedata_allStocks.py"

# Re-run for any failed tickers, then merge datasets
python "data extraction/combine_CSVs.py"
```

### Portfolio Optimization Notebooks

```bash
jupyter notebook
# Open: portfolio optimization/MVO.ipynb   (Mean-Variance Optimization)
# Open: portfolio optimization/JSE.ipynb   (James-Stein Estimator)
```

### Dependencies

No `requirements.txt` — install manually:

```bash
# Data extraction
pip install pandas beautifulsoup4 playwright requests
playwright install  # required for browser automation

# Portfolio optimization
pip install pandas numpy matplotlib seaborn jupyter
```

## Architecture

### Data Flow

```
scrape_S&P_ordered.py  →  sp500_tickers.csv
        ↓
savedata_allStocks.py  →  sp500_adjusted_close_2y.csv + failed_tickers.csv
        ↓
   combine_CSVs.py     →  sp500_adjusted_close_2y_full.csv (final dataset)
```

The notebooks then consume:
- `JSE.ipynb` ← `sp500_adjusted_close_2y_full.csv` + `daily-treasury-rates-*.csv`
- `MVO.ipynb` ← Google Sheets (9-stock manually curated data, independent of pipeline)

### Key Implementation Details

**Scraping (`savedata_allStocks.py`)**: Uses Playwright (headless browser) to handle Yahoo Finance's dynamic JS content. Implements async/await with `asyncio.gather()` for concurrency. Has explicit Windows event loop fix (`asyncio.WindowsProactorEventLoopPolicy()`). MAX_RETRIES=3 per ticker with 2-second delays.

**MVO.ipynb**: Loads a 9-stock portfolio, computes sample covariance, and solves for the minimum-variance portfolio (Portfolio C). Allows short selling (negative weights). Risk-free rate: 4.8% annual.

**JSE.ipynb**: Processes the full S&P 500 dataset (~400 tickers after coverage filtering). Resamples to weekly (Friday close), uses 26-week rolling windows, removes risk-free rate (4.22% from U.S. Treasury CSVs). Implements Ledoit-Wolf single-factor shrinkage leading into James-Stein estimation. Core computation involves 400×400 covariance matrices.

### Module Dependencies

- `MVO.ipynb` is standalone (no dependency on data extraction outputs)
- `JSE.ipynb` requires the full data extraction pipeline to have been run first
- `combine_CSVs.py` requires both `sp500_adjusted_close_2y.csv` and retry data to exist
