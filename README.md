# Equity Portfolio Optimization with James–Stein Shrinkage and Data Extraction Pipelines

A comprehensive framework for equity portfolio optimization using James–Stein shrinkage estimation techniques, combined with robust data extraction pipelines for financial market data.

## Project Website

- Website: https://ahmernadeem.com/research/numerical-monte-carlo-library.html
- This page contains a project overview and a detailed paper-style survey of the methods and results in the literature, as well as the results from my implementation.

## Overview

This project implements advanced portfolio optimization techniques using James–Stein shrinkage estimators to improve covariance matrix estimation. The framework includes:

- Data extraction pipelines for retrieving and processing financial market data
- James–Stein shrinkage estimation for covariance matrices
- Portfolio optimization algorithms
- Performance analysis and backtesting tools

## Repository Structure

### data extraction/
Python scripts for scraping and processing financial market data:
- `scrape_S&P_ordered.py` - Scrape S&P 500 data in ordered sequence
- `scrape_S&P_unordered.py` - Scrape S&P 500 data (unordered)
- `savedata_allStocks.py` - Save data for all stocks in portfolio
- `savedata_oneStock.py` - Save data for individual stock
- `combine_CSVs.py` - Combine multiple CSV data files

### portfolio optimization/
Jupyter notebooks implementing portfolio optimization techniques:
- `MVO.ipynb` - Mean-Variance Optimization implementation
- `JSE.ipynb` - James–Stein Estimator implementation for shrinkage

### data/
Extracted financial market data for portfolio analysis:

#### Stock Data
- `sp500_adjusted_close_2y_full.csv` - Complete 2-year adjusted close prices for all S&P 500 constituents
- `sp500_adjusted_close_2y.csv` - Cleaned 2-year adjusted close prices for S&P 500 constituents
- `retry_adjusted_close_2y.csv` - Reprocessed data for tickers that initially failed
- `aapl_twoyearly_adjclose.csv` - Sample individual stock data (Apple Inc.)

#### Reference Data
- `sp500_tickers.csv` - List of S&P 500 ticker symbols used for data extraction
- `constituents.csv` - S&P 500 constituent information (sector, industry, etc.)
- `failed_tickers.csv` - Log of tickers that failed during data extraction

#### Market Data
- `daily-treasury-rates-2024.csv` - Daily U.S. Treasury rates for 2024 (risk-free rate proxy)
- `daily-treasury-rates-2025.csv` - Daily U.S. Treasury rates for 2025 (risk-free rate proxy)
- `Stocks Weekly.xlsx` - Weekly stock data compilation in Excel format

## Features

- Automated data extraction from financial sources
- James–Stein shrinkage estimators for improved covariance estimation
- Mean-variance portfolio optimization
- Risk management and portfolio rebalancing
- Performance metrics and visualization

## Languages

**Primary:** Python (Jupyter Notebook)
