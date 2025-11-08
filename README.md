# Macro-Technical Sentiment Forex Classifier

A production-ready ML-based forex trading system with hybrid ensemble architecture, live inference server, and MetaTrader 5 integration. Predicts directional movements (Buy/Sell/Hold) in major currency pairs using technical indicators and macroeconomic calendar events.

## 🎯 Overview

### System Architecture

```
┌──────────────┐       HTTP/JSON        ┌──────────────────┐
│   MT5 EA     │ ───────────────────▶   │ Inference Server │
│              │                         │    (Flask)       │
│ • M5 OHLCV   │                         │                  │
│ • Calendar   │ ◀───────────────────   │ • Feature Eng.   │
│   Events     │      Predictions        │ • Model Serving  │
│              │                         │ • Validation     │
└──────────────┘                         └──────────────────┘
      │                                           │
      │                                           │
      ▼                                           ▼
┌──────────────┐                         ┌──────────────────┐
│  Live Market │                         │  Trained Models  │
│   Trading    │                         │                  │
│              │                         │ • XGBoost        │
│ • Execution  │                         │ • LSTM           │
│ • Risk Mgmt  │                         │ • Hybrid Ens.    │
└──────────────┘                         └──────────────────┘
```

### Features

**Technical Analysis (55 features):**
- Momentum: RSI (14/21/28), MACD, Stochastic, CCI, Williams %R
- Trend: EMA (20/50/100/200), ADX, Parabolic SAR
- Volatility: ATR, Bollinger Bands, Donchian Channels
- Volume: OBV, Chaikin Money Flow, MFI
- Feature Crosses: RSI×ATR, MACD×BB_width, Volume×ATR, etc.

**Macroeconomic Integration (3 features):**
- `tau_pre`: Anticipation before high-impact events (NFP, CPI, Fed decisions)
- `tau_post`: Post-event influence decay
- `weighted_surprise`: Actual vs forecast surprise factor

**Model Architecture:**
- XGBoost Classifier (gradient boosting)
- LSTM Sequence Classifier (time-series patterns)
- Hybrid Ensemble (weighted combination)

**Validation:**
- Walk-Forward Optimization
- Time-series aware splits (no data leakage)
- Feature importance tracking

## 📋 Requirements

### System Requirements
- Python 3.11+
- 8GB+ RAM
- GPU optional (speeds up LSTM training on Kaggle)
- MetaTrader 5 (for live trading)

### Installation

1. **Clone Repository:**
```bash
git clone https://github.com/Morriase/Macro-Technical-Sentiment-Classifier.git
cd Macro-Technical-Sentiment-Classifier
```

2. **Create Virtual Environment:**
```bash
python -m venv venv
.\venv\Scripts\Activate.ps1  # Windows
source venv/bin/activate      # Linux/Mac
```

3. **Install Dependencies:**
```bash
pip install -r requirements.txt
```

Key packages:
- PyTorch 2.6+
- XGBoost 2.1+
- TA-Lib (technical analysis)
- Flask (inference server)
- pandas, numpy, scikit-learn
- loguru (logging)

### TA-Lib Installation

**Windows:**
```powershell
# Download wheel from: https://www.lfd.uci.edu/~gohlke/pythonlibs/#ta-lib
pip install TA_Lib‑0.4.29‑cp311‑cp311‑win_amd64.whl
```

**Linux/Mac:**
```bash
# Install system library
brew install ta-lib  # Mac
sudo apt-get install ta-lib  # Ubuntu

# Then install Python wrapper
pip install TA-Lib
```

See `TALIB_INSTALL_WINDOWS.md` for detailed Windows instructions.

## 🚀 Quick Start

### 1. Train Models (Kaggle)

The project is designed to train on Kaggle with pre-downloaded data:

```bash
# On Kaggle or local
python main.py
```

This trains models for all 4 currency pairs:
- EUR_USD
- GBP_USD  
- USD_JPY
- AUD_USD

**Outputs per pair:**
- `{PAIR}_model.pth_scaler.pkl` - StandardScaler
- `{PAIR}_model.pth_xgb_base.pkl` - XGBoost model
- `{PAIR}_model.pth_lstm_base.pth` - LSTM model
- `{PAIR}_model.pth_meta.pkl` - Ensemble metadata
- `{PAIR}_feature_schema.json` - Feature validation schema

Training time: ~4 hours on Kaggle GPU

### 2. Download Models

After training, download all model files to local `models/` directory:
```
models/
├── EUR_USD_model.pth_*
├── EUR_USD_feature_schema.json
├── GBP_USD_model.pth_*
├── GBP_USD_feature_schema.json
├── USD_JPY_model.pth_*
├── USD_JPY_feature_schema.json
├── AUD_USD_model.pth_*
└── AUD_USD_feature_schema.json
```

### 3. Start Inference Server

```bash
python inference_server.py
```

Server runs on `http://127.0.0.1:5000`

**API Endpoints:**
- `GET /health` - Health check
- `POST /predict` - Single prediction
- `POST /batch_predict` - Multiple pairs
- `GET /model_info/<pair>` - Model metadata

### 4. Test Inference Server

```bash
python test_inference_server.py
```

Expected: 7/7 tests pass ✅

### 5. Setup MT5 Expert Advisor

1. Copy `MQL5/Auron AI.mq5` and `MQL5/core_functions.mqh` to MT5 directory:
   ```
   C:\Users\{User}\AppData\Roaming\MetaTrader 5\MQL5\Experts\
   C:\Users\{User}\AppData\Roaming\MetaTrader 5\MQL5\Include\
   ```

2. Compile in MetaEditor

3. Configure EA settings:
   - `ServerURL`: `http://127.0.0.1:5000/predict`
   - `NewsFilterOn`: `true` (to send calendar events)
   - Enable trading and set risk parameters

4. Attach EA to chart (M5 timeframe recommended)

### 6. Live Trading

EA workflow:
1. Collects 500 M5 OHLCV candles
2. Queries MT5 Calendar API for high-impact events (±48h)
3. Sends JSON to inference server
4. Server engineers 58 features (55 technical + 3 macro)
5. Returns prediction: BUY/SELL/HOLD with confidence
6. EA executes trades based on signals and risk management

See `MQL5/TESTING_GUIDE.md` for detailed testing procedures.

## 📊 Project Structure

```
Macro-Technical-Sentiment-Classifier/
│
├── main.py                            # Training pipeline (Kaggle)
├── inference_server.py                # Flask production server
├── test_inference_server.py           # Server test suite
├── requirements.txt                   # Python dependencies
│
├── MQL5/                              # MetaTrader 5 Integration
│   ├── Auron AI.mq5                   # Expert Advisor
│   ├── core_functions.mqh             # Calendar & risk functions
│   ├── EA_UPDATE_NOTES.md             # EA changelog
│   ├── INTEGRATION_SUMMARY.md         # Integration overview
│   └── TESTING_GUIDE.md               # Testing procedures
│
├── src/
│   ├── config.py                      # System configuration
│   │
│   ├── data_acquisition/
│   │   ├── fx_data.py                 # FX OHLCV data (Kaggle/OANDA)
│   │   ├── macro_data.py              # Economic calendar events
│   │   └── mt5_data.py                # MetaTrader 5 data fetching
│   │
│   ├── feature_engineering/
│   │   ├── technical_features.py      # 55 TA-Lib indicators
│   │   └── sentiment_features.py      # Macro feature engineering
│   │
│   ├── models/
│   │   ├── hybrid_ensemble.py         # Hybrid XGB+LSTM ensemble
│   │   └── lstm_model.py              # LSTM sequence classifier
│   │
│   └── validation/
│       └── walk_forward.py            # Walk-forward validation
│
├── data/                              # Training data (gitignored)
│   └── kaggle_dataset/
│       ├── fx_data/                   # M5 OHLCV parquet files
│       └── macro_events/              # Calendar events parquet
│
├── models/                            # Trained models (gitignored)
│   ├── EUR_USD_model.pth_*            # EUR/USD models (4 files)
│   ├── EUR_USD_feature_schema.json    # Feature validation
│   └── ...                            # GBP_USD, USD_JPY, AUD_USD
│
├── logs/                              # Application logs
│   ├── inference_server.log
│   └── training.log
│
├── results/                           # Validation results
│
└── docs/                              # Documentation
    ├── ARCHITECTURE.md                # System architecture
    ├── IMPLEMENTATION_SUMMARY.md      # Implementation notes
    ├── SETUP.md                       # Setup guide
    └── KAGGLE_SETUP.md                # Kaggle instructions
```

## 🔬 Technical Details

### Feature Engineering (58 Total Features)

**Technical Indicators (55 features):**
- **Momentum (12):** RSI (14/21/28), MACD (line/signal/histogram), Stochastic (K/D/slowD), CCI, Williams %R, ROC
- **Trend (10):** EMA (20/50/100/200), ADX, +DI, -DI, Parabolic SAR (up/down), Aroon (up/down)
- **Volatility (7):** ATR, ATR Z-score, BB (upper/middle/lower/width/position)
- **Volume (3):** OBV, CMF, MFI
- **Channels (2):** Donchian (upper/lower)
- **Feature Crosses (21):** RSI×ATR, MACD×BB_width, Volume×ATR, EMA_crosses, etc.

**Macroeconomic Features (3 features):**
- `tau_pre`: Anticipation proximity to upcoming events (48h window)
  - Formula: `1.0 / (1.0 + hours_to_event / 48)`
- `tau_post`: Post-event decay influence (48h window)
  - Formula: `exp(-0.1 × hours_since_event)`
- `weighted_surprise`: Surprise factor weighted by decay
  - Formula: `(actual - forecast) / std(previous) × tau_post`

**Macro Event Sources (MT5 Calendar API):**
- NFP, Non-Farm Employment
- Interest Rate Decisions (Fed, ECB, BoE, BoJ)
- CPI, GDP, Unemployment Rate
- Retail Sales, PMI, Consumer Confidence
- Central Bank Statements

### Model Architecture

**1. XGBoost Classifier:**
- Gradient boosting with 200 estimators
- Max depth: 6
- Learning rate: 0.1
- Handles non-linear feature interactions
- Fast inference

**2. LSTM Sequence Classifier:**
- Input: 50-step sequences of 58 features
- Architecture: LSTM(128) → Dropout(0.3) → LSTM(64) → Dropout(0.3) → FC(32) → Output(3)
- Captures temporal dependencies
- Min-max scaled features

**3. Hybrid Ensemble:**
- Weighted combination: 60% XGBoost + 40% LSTM
- Feature importance tracking
- Saves complete training metadata
- Production-ready serialization

### Inference Pipeline

**Request Format:**
```json
{
  "pair": "EUR_USD",
  "ohlcv": [
    {"timestamp": "2025-11-08 12:00:00", "open": 1.1000, "high": 1.1010, "low": 1.0990, "close": 1.1005, "volume": 1000},
    ... (500 M5 candles)
  ],
  "events": [
    {"timestamp": "2025-11-08 14:30:00", "event_name": "NFP", "country": "US", "actual": 150000, "forecast": 180000, "previous": 200000, "impact": "high"}
  ]
}
```

**Response Format:**
```json
{
  "pair": "EUR_USD",
  "prediction": "BUY",
  "confidence": 0.847,
  "probabilities": {"BUY": 0.847, "SELL": 0.089, "HOLD": 0.064},
  "timestamp": "2025-11-08 12:05:00",
  "feature_count": 58,
  "status": "success"
}
```

### Validation Framework

**Walk-Forward Optimization:**
- Training window: Last 80% of available data
- Testing window: Final 20%
- Time-series aware splits (no data leakage)
- Anchored or rolling window options

**Evaluation Metrics:**
- Classification: Accuracy, Precision, Recall, F1-Score
- Per-class performance tracking
- Feature importance analysis
- Prediction confidence distributions

## ⚙️ Configuration

Key settings in `src/config.py`:

```python
# Supported currency pairs
CURRENCY_PAIRS = ["EUR_USD", "GBP_USD", "USD_JPY", "AUD_USD"]

# Model paths
MODELS_DIR = Path("models/")
DATA_DIR = Path("data/")

# High-impact macro events
HIGH_IMPACT_EVENTS = [
    "NFP", "Non-Farm", "Interest Rate", "FOMC", "CPI", "GDP",
    "Unemployment", "Retail Sales", "PMI", "Central Bank"
]

# Feature engineering
TARGET_HOURS_AHEAD = 24  # Prediction horizon
MIN_PIPS_MOVEMENT = 10.0  # Minimum movement for directional signal

# Walk-forward validation
TRAIN_RATIO = 0.8  # 80% training, 20% testing
```

### MT5 EA Configuration

Key EA parameters:
```mql5
input string   ServerURL = "http://127.0.0.1:5000/predict";  // Inference server
input bool     NewsFilterOn = true;                           // Send calendar events
input int      MaxLotSize = 1;                                // Maximum position size
input double   RiskPercent = 2.0;                             // Risk per trade %
input int      StopLossPips = 50;                             // Fixed stop loss
input int      TakeProfitPips = 100;                          // Fixed take profit
```

## 🧪 Testing

### Test Inference Server
```bash
python test_inference_server.py
```

**Test Suite:**
1. ✅ Health Check
2. ✅ Single Prediction (with events)
3. ✅ Insufficient Data Handling
4. ✅ Invalid Pair Handling
5. ✅ Model Info Endpoint
6. ✅ Batch Prediction
7. ✅ All Supported Pairs

### Test MT5 EA

See `MQL5/TESTING_GUIDE.md` for:
- Demo account setup
- Strategy tester usage
- Live market testing checklist
- Common issues and solutions

## � Documentation

- **ARCHITECTURE.md** - System architecture and design decisions
- **IMPLEMENTATION_SUMMARY.md** - Development timeline and key changes
- **SETUP.md** - Detailed setup and configuration guide
- **KAGGLE_SETUP.md** - Training on Kaggle instructions
- **TALIB_INSTALL_WINDOWS.md** - Windows TA-Lib installation
- **MQL5/EA_UPDATE_NOTES.md** - EA changelog and modifications
- **MQL5/INTEGRATION_SUMMARY.md** - EA-Server integration overview
- **MQL5/TESTING_GUIDE.md** - MT5 testing procedures

## 🔄 Development Workflow

### 1. Feature Development
```bash
# Make changes
git checkout -b feature/new-feature
# ... edit code ...
git commit -m "Add new feature"
git push origin feature/new-feature
```

### 2. Model Retraining (when features change)
```bash
# On Kaggle
git pull origin main
python main.py
# Download new models and schemas
```

### 3. Testing
```bash
# Test server
python test_inference_server.py

# Test EA (MT5 Strategy Tester)
# See MQL5/TESTING_GUIDE.md
```

### 4. Deployment
```bash
# Update production server
git pull origin main
# Restart inference server
python inference_server.py
```

## 🐛 Known Issues

1. **PyTorch 2.6 Compatibility**: Use `weights_only=False` when loading old models
2. **sklearn Version Warnings**: Non-blocking, will resolve after retraining
3. **Mock Feature Schemas**: Current models expect 53 features, need retraining for 58
4. **Pandas FutureWarnings**: Chained assignment warnings in technical_features.py (non-critical)

## 🎯 Roadmap

- [x] Core training pipeline
- [x] Inference server with Flask
- [x] MT5 EA integration
- [x] Macro feature engineering
- [ ] Retrain models with 58 features
- [ ] Production deployment with gunicorn
- [ ] Real-time monitoring dashboard
- [ ] Backtesting framework
- [ ] Paper trading mode
- [ ] Performance analytics

## 📜 License

MIT License - See LICENSE file for details

## ⚠️ Disclaimer

**This software is for educational and research purposes only.** 

Trading forex involves substantial risk of loss and is not suitable for all investors. Past performance is not indicative of future results. The authors assume no responsibility for your trading results. Always conduct your own due diligence and consult with a licensed financial advisor before trading.

**Key Risks:**
- Leverage can amplify losses
- Market volatility can cause rapid losses
- Technical failures can occur
- Models may underperform in changing market conditions

## 🙏 Acknowledgments

- **TA-Lib** - Technical analysis library
- **PyTorch** - Deep learning framework
- **XGBoost** - Gradient boosting library
- **MetaTrader 5** - Trading platform and Calendar API
- **Kaggle** - Training environment and datasets
- **Flask** - Lightweight web framework
- **loguru** - Python logging library

---

**Repository**: [Macro-Technical-Sentiment-Classifier](https://github.com/Morriase/Macro-Technical-Sentiment-Classifier)

Built for quantitative forex trading research 📈
