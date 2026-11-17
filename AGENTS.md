# K League Match Prediction & Betting Backtest Project

## Overview
An end-to-end ML pipeline to scrape K League match data and Betman odds, engineer pre-match features, predict match outcomes (Win/Draw/Loss), and run betting backtests (ROI, Kelly criterion, flat betting).

## Agent Behavior & Communication Rules
- **Language**: Always respond and explain in **Korean**, while keeping all code, comments, variable names, and git commits in **English**.
- **Read Before Edit**: Always read target files before modifying them. Do not hallucinate file paths or column names.
- **Data Leakage Prevention (CRITICAL)**: Only **pre-match data** (rolling averages, prior round stats, historical H2H) may be used as features (`X`). Never include post-match stats (e.g., match shots, possession, goals from the target match itself) in training or prediction features.

## Data Pipeline & Execution
Run these scripts in order to execute the full pipeline:

1. **Scrape K League Data**: `python Kleague.py`
   - Fetches raw team/match stats from `portal.kleague.com`.
   - Saves to `kleague_csv/{year}/`.
2. **Scrape Betman Odds**: `python betman.py` *(planned/in-progress)*
   - Fetches historical and upcoming Win/Draw/Loss (승/무/패) dividend odds from Betman.
   - Saves to `betman_csv/{year}/` (ensure team names match K League naming conventions).
3. **Process & Merge Data**: `python pruning.py`
   - Cleans raw data, decodes HTML entities, normalizes team names across K League and Betman datasets, and builds pre-match rolling features.
   - Saves to `kleague_csv/processed/{year}/`.
4. **Train & Predict**: `python feature.py` (or `train.py`)
   - Trains classification models (e.g., Random Forest, XGBoost/LightGBM) using time-series/round-based splits (no random shuffling).
   - Outputs predicted probabilities (`P_win`, `P_draw`, `P_loss`) and saves feature importance to `feature_importance/`.
5. **Betting Backtest**: `python backtest.py` *(planned/in-progress)*
   - Compares model predicted probabilities against Betman implied probabilities (`1 / odds`) to calculate Expected Value (EV).
   - Simulates betting strategies (EV threshold, flat stake, fractional Kelly) and outputs ROI, MDD, and win rate.

## Key Directories
- `kleague_csv/`: Raw (`{year}/`) and cleaned/merged (`processed/{year}/`) K League match data.
- `betman_csv/`: Scraped Betman odds data.
- `feature_importance/`: Feature importance plots/CSVs and model evaluation metrics.
- `backtest_results/`: Betting simulation logs and ROI reports.

## Technical Quirks & Constraints
- **Encoding**: Always read/write CSV files with `utf-8-sig` (fallback to `cp949` when reading legacy files).
- **Time-Series Split**: `feature.py` MUST split train/test sets chronologically by match date or round (`70/30` split). Never use random `train_test_split` to avoid future leakage.
- **Team Name Mapping**: K League portal team names (e.g., `울산 HD`, `전북 현대`) and Betman team names (e.g., `울산HD`, `전북현대`) often differ in spacing or naming. Always use a centralized mapping dictionary in `pruning.py`.
- **Scraping**: `Kleague.py` and Betman scrapers require custom `User-Agent`/`Referer` headers and rate-limiting (`time.sleep`) to avoid IP blocks.
- **Dependencies**: `requests`, `pandas`, `numpy`, `beautifulsoup4`, `scikit-learn` (add `lightgbm` or `xgboost` if needed).
