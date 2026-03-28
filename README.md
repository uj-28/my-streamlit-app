## Stock Scanner + Backtester (AI-Enhanced)

This Streamlit app scans Indian market universes (NSE/BSE) with technical filters and an AI scoring layer. It also includes a rule-based backtester.

### Features
- Universe scanner with EMA/RSI/Supertrend filters
- Optional HalfTrend signals
- AI scoring with confidence threshold
- Backtesting with custom entry/exit rules
- Local disk caching for OHLCV + indicators

### Project Structure
```
scanner_app/
├─ scanner_project.py   # Streamlit entry point
├─ scanner_core.py      # Scan + backtest engine
├─ ai_model.py          # Features + XGBoost training/scoring
├─ indicators.py        # RSI/ATR/Supertrend/HalfTrend/ADX
├─ data_fetch.py        # yfinance + NSE list loading + cache
├─ ui.py                # Banner + styling + UI helpers
├─ utils.py             # Validation + logging helpers
├─ config.py            # Central settings/constants
```

### Setup
1. Create a virtual environment
2. Install requirements
3. Run Streamlit

```bash
pip install -r requirements.txt
streamlit run scanner_project.py
```

### Notes
- Cache and model artifacts are written to `data_cache/`
- Logging writes to `data_cache/scanner.log`
- Make sure `.gitignore` keeps caches, models, and local config out of Git
