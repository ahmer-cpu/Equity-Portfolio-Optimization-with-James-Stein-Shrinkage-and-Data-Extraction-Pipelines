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
Storage for extracted financial data (CSV files, Excel files) - not tracked in git

## Features

- Automated data extraction from financial sources
- James–Stein shrinkage estimators for improved covariance estimation
- Mean-variance portfolio optimization
- Risk management and portfolio rebalancing
- Performance metrics and visualization

## Languages

**Primary:** Python (Jupyter Notebook)
