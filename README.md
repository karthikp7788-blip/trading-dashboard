# SPY Trading Prediction System - Phase 0.1

## Overview

This is the foundation of an intelligent trading prediction system that combines:
- **Technical Analysis**: 30+ indicators across multiple timeframes
- **Options Intelligence**: Skew, premium imbalance, OI concentration, rich/cheap analysis
- **Machine Learning**: Hybrid LSTM + XGBoost ensemble (coming in Phase 0.3)
- **Explainability**: Every prediction comes with reasons

**Current Status**: Phase 0.1 Complete - Data Pipeline & Feature Engineering Foundations

## What's Been Built (Phase 0.1)

### ✅ Data Pipeline Module (`data_pipeline.py`)
- Fetches historical price data (OHLCV) from Yahoo Finance
- Fetches real-time options chains (calls, puts, IV, OI)
- Validates data quality (missing bars, negative prices, bid-ask sanity)
- Supports multiple tickers: SPY, TSLA, GOOGL, NVDA, AMD, META (easily extensible)

### ✅ Technical Indicators Module (`technical_indicators.py`)
Calculates 30+ technical indicators:
- **Moving Averages**: SMA (20, 50, 200), EMA (12, 26, 50)
- **Oscillators**: RSI, MACD, MACD Signal, MACD Histogram
- **Bands & Ranges**: Bollinger Bands, ATR
- **Volume**: Relative volume, OBV
- **Momentum**: Multi-period returns
- **Volatility**: Rolling volatility (annualized)
- **Trend Detection**: MA alignment, price relative to MAs

### ✅ Options Analytics Module (`options_analytics.py`)
Sophisticated options analysis:

**Skew Metrics**:
- ATM call/put IV levels
- Put-call IV skew
- OTM skew (risk reversal proxy)
- Skew slope (smile steepness)

**Premium Analysis**:
- Call/put premium ratio
- Premium imbalance (directional bias indicator)
- Volume-weighted premium flows

**Open Interest**:
- Max pain calculation
- OI concentration by strike
- Gamma pin zones identification
- Top strikes by OI

**Fair Value Analysis**:
- Black-Scholes theoretical pricing
- Rich vs cheap classification
- Options trading above/below fair value

## Installation

### Prerequisites
- Python 3.8+
- pip package manager

### Quick Start

1. **Clone/Download** this directory

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Run the demo** (uses synthetic data):
```bash
python demo_synthetic.py
```

4. **Run with real data** (requires network access):
```bash
python phase_01_demo.py
```

## File Structure

```
trading_system/
├── data_pipeline.py           # Data fetching and validation
├── technical_indicators.py    # Technical analysis features
├── options_analytics.py       # Options-derived signals
├── demo_synthetic.py          # Demonstration with synthetic data
├── phase_01_demo.py          # Demonstration with real market data
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

## Usage Examples

### Fetch Data for SPY

```python
from data_pipeline import DataPipeline

# Initialize pipeline
pipeline = DataPipeline(['SPY'])

# Fetch all data
results = pipeline.fetch_all_data()

# Access price data
spy_prices = results['price_data']['SPY']
print(f"Latest close: ${spy_prices['Close'].iloc[-1]:.2f}")

# Access options
spy_options = results['options_data']['SPY']
calls = spy_options['calls']
puts = spy_options['puts']
```

### Calculate Technical Indicators

```python
from technical_indicators import TechnicalIndicators

# Add all indicators to price data
spy_with_indicators = TechnicalIndicators.calculate_all_indicators(spy_prices)

# Access specific indicators
latest = spy_with_indicators.iloc[-1]
print(f"RSI: {latest['RSI']:.2f}")
print(f"MACD: {latest['MACD']:.4f}")
print(f"ATR: ${latest['ATR']:.2f}")
```

### Analyze Options

```python
from options_analytics import OptionsAnalytics

current_price = spy_prices['Close'].iloc[-1]

# Calculate all options metrics
metrics = OptionsAnalytics.calculate_all_metrics(spy_options, current_price)

# Check skew
if 'skew_pc_iv_skew' in metrics:
    skew = metrics['skew_pc_iv_skew'] * 100
    print(f"Put-Call IV Skew: {skew:+.2f}%")
    
    if skew > 0:
        print("→ Defensive positioning (fear premium)")
    else:
        print("→ Complacent positioning")

# Check premium imbalance
if 'premium_premium_imbalance' in metrics:
    imbalance = metrics['premium_premium_imbalance']
    print(f"Premium Imbalance: {imbalance:+.2f}")
    
    if imbalance > 0.2:
        print("→ Bullish (heavy call buying)")
    elif imbalance < -0.2:
        print("→ Bearish (heavy put buying)")
```

## Key Features

### Data Quality Validation
- Automatic detection of missing bars
- Negative price checks
- Bid-ask spread sanity checks
- IV anomaly detection
- Volume validation

### Options Intelligence
All options metrics are **first-class features** - they'll feed directly into the ML models in Phase 0.3.

**Why this matters**:
Most prediction systems ignore options data. But options reflect:
- Forward-looking volatility expectations
- Institutional positioning
- Risk sentiment
- Supply/demand imbalances

By integrating options skew, premium flows, and OI concentration, we're capturing information that price alone doesn't reveal.

### Designed for ML
All features are calculated in a way that:
- Avoids look-ahead bias
- Handles missing data gracefully
- Scales properly for model training
- Maintains temporal alignment

## Next Phases

### Phase 0.2: Feature Engineering (Next)
- Combine price + options features into ML-ready dataset
- Multi-timeframe alignment (5m, 15m, 60m, daily)
- Feature selection and correlation analysis
- Target variable creation (next-day direction, expected range)

### Phase 0.3: Prediction Models
- XGBoost for structured features (indicators, options metrics)
- LSTM for sequential price behavior
- Ensemble meta-layer
- Confidence calibration
- Regime detection (trend, range, high-vol, squeeze risk)

### Phase 0.4: Backtesting Framework
- Walk-forward validation (no look-ahead bias)
- Performance metrics: accuracy, win rate, profit factor, drawdown
- **Critical test**: Do options features improve predictions?
- Version control and model comparison

### Phase 0.5: Streamlit Dashboard
- Private hosted web app
- Daily forecast (direction + range)
- Intraday alerts (with confluence requirements)
- Options intelligence visualizations
- Backtesting results
- Explanations for every signal

## Data Sources

### Current (Free Tier)
- **yfinance**: Price data + basic options chains
- **Limitations**: Daily data only, rate limits, less reliable

### Planned (When Validated)
- **Polygon.io**: Professional-grade market data
  - Real-time and historical
  - Tick-level granularity
  - Comprehensive options chains
  - Cost: ~$99/month for starter plan

**Strategy**: Prove the concept with free data first, upgrade to Polygon only when we've validated that options features actually improve predictions.

## Design Philosophy

1. **Build foundations properly**: Data quality and feature engineering matter more than fancy models
2. **Test the hypothesis**: Do options features improve predictions? Measure this explicitly
3. **Explainability first**: Every prediction comes with reasons
4. **Know when not to trade**: "Do nothing" is a valid and frequent output
5. **Scale intelligently**: Start simple (SPY daily), add complexity only when validated

## Performance Expectations

This is **NOT** a holy grail system that will print money. Realistic expectations:

- Direction accuracy: 55-65% (vs 50% random)
- Win rate after confluence filters: 40-50%
- Profit factor: 1.5-2.5 (if edge exists)
- Drawdowns: Expect them
- Signal frequency: Low (maybe 1-3 good setups per week)

The goal is **consistent edge**, not perfection.

## Deployment (Phase 0.5)

When ready to deploy:

1. **Streamlit Community Cloud** (free tier)
   - Simple deployment
   - No infrastructure management
   - Perfect for personal use

2. **OR: AWS/Cloud hosting** (if you need more control)
   - EC2 or Lightsail for compute
   - Scheduled jobs for backtesting
   - Your Qualizeal contacts can handle deployment

## Contributing to Development

Phases are built sequentially. Each phase must be **fully validated** before moving to the next.

**Current priorities**:
1. ✅ Phase 0.1: Data foundations (COMPLETE)
2. 🔄 Phase 0.2: Feature engineering (NEXT)
3. ⏳ Phase 0.3: Prediction models
4. ⏳ Phase 0.4: Backtesting
5. ⏳ Phase 0.5: Dashboard

## Questions or Issues?

This is a work in progress. Phase 0.1 is complete and functional. The remaining phases build on this foundation.

**What works right now**:
- ✅ Data fetching and validation
- ✅ Technical indicator calculations
- ✅ Options analytics (skew, premium, OI)
- ✅ Multi-ticker support

**What's coming next**:
- Feature engineering pipeline
- ML models
- Backtesting
- Web dashboard

---

**Built with**: Python, pandas, numpy, scipy, yfinance, scikit-learn, xgboost, tensorflow, streamlit

**License**: Private use only

**Version**: 0.1.0 (Phase 0.1 Complete)

**Last Updated**: January 2026
