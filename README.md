# Macro-Technical Sentiment Forex Classifier

A sophisticated ML-based forex directional classifier implementing a hybrid XGBoost-Random Forest-MLP stacking ensemble architecture for 4-24 hour prediction horizons. This system integrates macroeconomic surprise factors, technical indicators, and NLP-based sentiment analysis.

## 🎯 Overview

This system predicts directional movements (Buy/Sell/Hold) in major currency pairs using:

- **Technical Features**: EMA, RSI, ATR, MACD, Bollinger Bands, Stochastic oscillators
- **Macroeconomic Features**: Surprise Z-scores, temporal proximity encoding, PCA-reduced indicators
- **Sentiment Features**: FinBERT-based differential sentiment for currency pairs, thematic analysis (LDA)
- **COT Data**: Institutional positioning from CFTC Commitment of Traders reports

### Architecture

**Level-0 Base Learners:**
- XGBoost Classifier (event-driven signals)
- Random Forest Classifier (robust variance-reduced estimates)

**Level-1 Meta-Learner:**
- Multi-Layer Perceptron (PyTorch) for optimal blending

**Validation:**
- Walk-Forward Optimization with nested cross-validation
- Optuna hyperparameter tuning
- Time-series aware splits (no data leakage)

## 📋 Requirements

### System Requirements
- Python 3.8+
- 16GB+ RAM recommended
- GPU optional (speeds up PyTorch MLP training)

### Dependencies
```bash
pip install -r requirements.txt
```

Key packages:
- PyTorch 2.0+
- XGBoost 2.0+
- TA-Lib (requires system-level installation)
- Transformers (HuggingFace)
- Optuna for hyperparameter optimization

### TA-Lib Installation

**Windows:**
```powershell
# Download wheel from: https://www.lfd.uci.edu/~gohlke/pythonlibs/#ta-lib
pip install TA_Lib‑0.4.28‑cp39‑cp39‑win_amd64.whl
```

**Linux/Mac:**
```bash
# Install system library first
wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz
tar -xzf ta-lib-0.4.0-src.tar.gz
cd ta-lib/
./configure --prefix=/usr
make
sudo make install

# Then install Python wrapper
pip install TA-Lib
```

## 🚀 Quick Start

### 1. Configuration

Create a `.env` file with your API keys:

```env
OANDA_API_KEY=your_oanda_key_here
OANDA_ACCOUNT_ID=your_account_id_here
FINNHUB_API_KEY=your_finnhub_key_here
TRADING_ECONOMICS_API_KEY=your_te_key_here
```

### 2. Data Acquisition

```python
from src.data_acquisition.fx_data import FXDataAcquisition
from src.data_acquisition.macro_data import MacroDataAcquisition
from datetime import datetime, timedelta

# Fetch FX price data
fx_data = FXDataAcquisition()
end_date = datetime.now()
start_date = end_date - timedelta(days=365*5)  # 5 years

df_eur_usd = fx_data.fetch_oanda_candles(
    instrument="EUR_USD",
    granularity="M5",
    start_date=start_date,
    end_date=end_date,
)

# Resample to 4H
df_4h = fx_data.resample_to_timeframe(df_eur_usd, "4H")

# Fetch macro events
macro_data = MacroDataAcquisition()
events_df = macro_data.get_events_for_currency_pair(
    pair="EUR_USD",
    start_date=start_date,
    end_date=end_date,
)
```

### 3. Feature Engineering

```python
from src.feature_engineering.technical_features import TechnicalFeatureEngineer
from src.feature_engineering.sentiment_features import SentimentAnalyzer

# Technical features
tech_engineer = TechnicalFeatureEngineer()
df_features = tech_engineer.calculate_all_features(df_4h)
df_features = tech_engineer.calculate_feature_crosses(df_features)

# Sentiment features (requires news corpus)
sentiment_analyzer = SentimentAnalyzer()
# ... process news data
```

### 4. Train Model

```python
from src.models.hybrid_ensemble import HybridStackingEnsemble
from src.validation.walk_forward import WalkForwardOptimizer

# Initialize ensemble
ensemble = HybridStackingEnsemble()

# Setup WFO
optimizer = WalkForwardOptimizer(model_class=HybridStackingEnsemble)

# Run walk-forward optimization
results = optimizer.run_walk_forward_optimization(
    df=df_features,
    feature_columns=feature_cols,
    target_column="target",
    optimize_each_window=True,
)

# Aggregate results
summary = optimizer.aggregate_results()
```

### 5. Generate Predictions

```python
# Get latest predictions
y_pred_proba = ensemble.predict_proba(X_latest)

# Extract confidence and signals
confidence_buy = y_pred_proba[:, 0]  # Buy class
confidence_sell = y_pred_proba[:, 1]  # Sell class
confidence_hold = y_pred_proba[:, 2]  # Hold class

# Apply confidence threshold
confidence_threshold = 0.70
signal = "BUY" if confidence_buy > confidence_threshold else \
         "SELL" if confidence_sell > confidence_threshold else "HOLD"
```

## 📊 Project Structure

```
Macro-Technical Sentiment Classifier/
│
├── src/
│   ├── config.py                      # Configuration settings
│   │
│   ├── data_acquisition/
│   │   ├── fx_data.py                 # FX price data (OANDA)
│   │   ├── macro_data.py              # Economic calendar (Finnhub)
│   │   └── cot_data.py                # COT reports (CFTC)
│   │
│   ├── feature_engineering/
│   │   ├── technical_features.py      # TA-Lib indicators
│   │   ├── macro_features.py          # Surprise Z-scores, proximity
│   │   └── sentiment_features.py      # FinBERT sentiment analysis
│   │
│   ├── models/
│   │   ├── hybrid_ensemble.py         # Stacking ensemble
│   │   └── lstm_model.py              # Alternative LSTM (optional)
│   │
│   ├── validation/
│   │   ├── walk_forward.py            # WFO framework
│   │   └── metrics.py                 # Evaluation metrics
│   │
│   ├── risk_management/
│   │   ├── position_sizing.py         # ATR-based risk sizing
│   │   └── execution.py               # Entry/exit logic
│   │
│   └── backtesting/
│       ├── backtest_engine.py         # Simulation engine
│       └── performance.py             # Performance analytics
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_feature_engineering.ipynb
│   └── 03_model_training.ipynb
│
├── data/                              # Data storage (gitignored)
├── models/                            # Saved models (gitignored)
├── logs/                              # Log files
├── results/                           # Backtest results
│
├── requirements.txt
├── .env.example
└── README.md
```

## 🔬 Key Features

### 1. Technical Indicator Suite
- **Trend**: EMA (50, 100, 200), ADX, CCI
- **Momentum**: RSI, Stochastic, MACD
- **Volatility**: ATR, Bollinger Bands
- **Normalization**: Z-scores, ATR-normalized distances

### 2. Macroeconomic Integration
- **Surprise Factor**: `(Actual - Consensus) / σ(Historical Errors)`
- **Temporal Proximity**: 
  - Pre-event anticipation (inverse proximity)
  - Post-event decay (exponential)
- **PCA Reduction**: Multicollinearity handling

### 3. NLP Sentiment Pipeline
- **FinBERT**: Pre-trained on financial corpus
- **Differential Scoring**: `Sentiment(Base) - Sentiment(Quote)`
- **Time Weighting**: EMA (3, 7, 14 days)
- **Thematic Analysis**: LDA topic modeling

### 4. Risk Management
- **ATR-Based Stops**: `SL = ATR × Multiplier`
- **Dynamic R/R**: Optimized via WFO
- **Confidence Threshold**: `C_base + C_sensitivity × ATR_normalized`
- **Time-Based Exits**: Maximum position duration

### 5. Validation Framework
- **Walk-Forward Optimization**: 2-year train, 6-month test
- **Nested CV**: Hyperparameter tuning within IS window
- **No Data Leakage**: Strict temporal ordering
- **Optuna**: Efficient hyperparameter search

## 📈 Performance Metrics

The system tracks comprehensive metrics:

- **Classification**: Balanced Accuracy, F-beta scores
- **Financial**: Profit Factor, Sharpe Ratio, Calmar Ratio
- **Risk**: Maximum Drawdown, Win Rate, Avg P&L
- **Feature Importance**: SHAP values for interpretability

## ⚙️ Configuration

Key parameters in `src/config.py`:

```python
# Risk Management
RISK_MANAGEMENT = {
    "base_confidence_threshold": 0.70,
    "confidence_sensitivity": 0.10,
    "stop_loss_multiplier": 1.5,
    "risk_reward_ratio": 2.0,
    "max_time_bars": 6,  # 24 hours
}

# Walk-Forward Optimization
WFO_CONFIG = {
    "train_window_years": 2,
    "test_window_months": 6,
    "step_months": 6,
    "cv_folds": 5,
}

# Model Architecture
ENSEMBLE_CONFIG = {
    "base_learners": {
        "xgboost": {...},
        "random_forest": {...},
    },
    "meta_learner": {
        "hidden_layers": [128, 64, 32],
        "dropout_rate": 0.3,
        "learning_rate": 0.001,
    },
}
```

## 🧪 Testing

Run unit tests:
```bash
pytest tests/ -v --cov=src
```

## 📝 Citation

If you use this system in your research, please cite:

```bibtex
@software{forex_macro_technical_classifier,
  title={Macro-Technical Sentiment Forex Classifier},
  author={Your Name},
  year={2025},
  url={https://github.com/yourusername/forex-classifier}
}
```

## 📜 License

MIT License - See LICENSE file for details

## ⚠️ Disclaimer

**This software is for educational and research purposes only.** 

Trading forex involves substantial risk of loss and is not suitable for all investors. Past performance is not indicative of future results. The authors assume no responsibility for your trading results. Always conduct your own due diligence and consult with a licensed financial advisor before trading.

## 🤝 Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request with tests

## 📧 Contact

For questions or issues, please open a GitHub issue or contact [your-email@example.com]

## 🙏 Acknowledgments

- **TA-Lib**: Technical analysis library
- **HuggingFace**: FinBERT sentiment model
- **OANDA**: High-fidelity FX data API
- **Optuna**: Hyperparameter optimization framework

---

Built with ❤️ for quantitative trading research
