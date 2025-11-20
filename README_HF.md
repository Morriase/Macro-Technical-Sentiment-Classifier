---
title: Forex Inference Server
emoji: 📈
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
license: mit
---

# Forex AI Trading Inference Server

Real-time forex trading predictions using hybrid ML models (XGBoost + LSTM).

## Features

- **4 Currency Pairs**: EUR/USD, GBP/USD, USD/JPY, AUD/USD
- **81 Features**: Technical indicators + Multi-timeframe analysis + Macro events
- **Hybrid Models**: XGBoost + LSTM ensemble with meta-classifier
- **Quality Scoring**: Fuzzy logic-based signal quality assessment
- **REST API**: Simple JSON endpoints for predictions

## API Endpoints

### Health Check
```bash
GET /health
```

### Make Prediction
```bash
POST /predict
Content-Type: application/json

{
  "pair": "EUR_USD",
  "ohlcv_m5": [...],  // 500 M5 candles
  "ohlcv_h1": [...],  // 300 H1 candles
  "ohlcv_h4": [...],  // 250 H4 candles
  "events": []        // Optional calendar events
}
```

### Get Model Info
```bash
GET /model_info/EUR_USD
```

## Response Format

```json
{
  "pair": "EUR_USD",
  "prediction": "BUY",
  "confidence": 0.75,
  "quality_score": 65.0,
  "position_size_pct": 0.50,
  "should_trade": true,
  "probabilities": {
    "BUY": 0.75,
    "SELL": 0.15,
    "HOLD": 0.10
  }
}
```

## Model Architecture

- **Base Learners**: XGBoost (tabular) + LSTM (sequential)
- **Meta-Classifier**: XGBoost stacking ensemble
- **Training**: Walk-forward optimization with 5-fold CV
- **Features**: 81 engineered features per prediction

## Technical Stack

- **Framework**: Flask + Gunicorn
- **ML**: XGBoost, PyTorch (LSTM)
- **Features**: pandas, ta-lib, numpy
- **Deployment**: Docker on Hugging Face Spaces

## Usage

This server is designed to work with MetaTrader 5 Expert Advisors (EAs) for automated forex trading.

## License

MIT License - See LICENSE file for details
