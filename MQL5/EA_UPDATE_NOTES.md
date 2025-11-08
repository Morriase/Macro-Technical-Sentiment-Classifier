# MQL5 EA Update Notes - Hybrid Ensemble Integration

## Overview
Updated MQL5 EA (`Auron AI.mq5`) to integrate with the Python Hybrid Ensemble inference server. The EA now sends raw OHLCV data instead of computing features, ensuring perfect feature alignment with the training pipeline.

## Key Changes

### 1. **Timeframe Changed: M15/H1/H4 → M5**
- **Before**: Multi-timeframe SMC analysis (M15 + H1 + H4, 250 bars each)
- **After**: Single M5 timeframe (250 bars)
- **Reason**: Training data uses M5 5-minute candles
- **Impact**: All bar checks now use `PERIOD_M5` instead of `PERIOD_M15`

### 2. **JSON Request Format**
**Before (Multi-timeframe SMC)**:
```json
{
  "symbol": "EURUSD",
  "data": {
    "M15": [{time, open, high, low, close, volume}, ...],
    "H1": [{time, open, high, low, close, volume}, ...],
    "H4": [{time, open, high, low, close, volume}, ...]
  }
}
```

**After (Single timeframe OHLCV)**:
```json
{
  "pair": "EUR_USD",
  "ohlcv": [
    {
      "timestamp": "2024.01.15 10:00",
      "open": 1.08456,
      "high": 1.08478,
      "low": 1.08432,
      "close": 1.08465,
      "volume": 1234
    },
    ...250 more candles
  ]
}
```

### 3. **Symbol Format Conversion**
- **Function**: `ConvertSymbolFormat()`
- **Conversion**: `EURUSD` → `EUR_USD`, `GBPUSD` → `GBP_USD`, etc.
- **Handles Suffixes**: `EURUSD.ecn` → `EUR_USD` (ignores broker suffixes)
- **Reason**: Training data uses underscore format

### 4. **Response Parsing Updated**
**Before (SMC Server)**:
```json
{
  "prediction": -1,
  "signal": "SELL",
  "confidence": 0.85,
  "consensus": true,
  "models": {"RandomForest": 1, "XGBoost": 0, "NeuralNetwork": 1},
  "smc_context": {...},
  "order_blocks": {...}
}
```

**After (Inference Server)**:
```json
{
  "prediction": "SELL",
  "confidence": 0.85,
  "probabilities": {
    "SELL": 0.85,
    "HOLD": 0.10,
    "BUY": 0.05
  }
}
```

### 5. **Removed SMC-Specific Code**
- No longer extracts: `smc_context`, `order_blocks`, `fair_value_gaps`, `structure`, `regime`
- No longer draws: Order block zones on chart
- No longer displays: Model predictions, explanations, narratives

### 6. **Simplified Chart Display**
- **New Function**: `DisplayPredictionInfo()` (replaces complex SMC display)
- **Shows**:
  - Prediction (BUY/SELL/HOLD)
  - Confidence percentage
  - Class probabilities (SELL/HOLD/BUY)
  - Min confidence threshold
  - Trading mode (ENABLED/DEMO)
  - Request count
  - Server URL

### 7. **Signal Conversion**
- **Before**: Numeric predictions (-1=SELL, 0=HOLD, 1=BUY)
- **After**: String signals ("SELL", "HOLD", "BUY")
- **Added**: Conversion logic to maintain compatibility with existing `ExecuteTrade()` function

## Modified Functions

| Function | Changes |
|----------|---------|
| `PrepareOHLCVData()` | Now sends M5 data only with "pair" and "ohlcv" keys |
| `CollectTimeframeData()` | Renamed to `CollectM5TimeframeData()`, uses "timestamp" instead of "time" |
| `ConvertSymbolFormat()` | **NEW** - Converts EURUSD → EUR_USD |
| `ParseAndExecute()` | Updated JSON parsing for new response format |
| `DisplayPredictionInfo()` | **NEW** - Simplified chart display without SMC |
| `OnInit()` | Updated print statements to reflect M5/250 bars/58 features |
| `OnTick()` | Changed from PERIOD_M15 to PERIOD_M5 |

## Configuration

### Unchanged Input Parameters
All input parameters remain the same:
- `RestServerURL` (default: `http://127.0.0.1:5000/predict`)
- `MinConfidence` (recommended: 0.55 - 0.65)
- `EnableTrading` (DEMO/LIVE toggle)
- `RiskManagement` settings
- `StopLoss` modes
- `TrailingStop` options
- `AutoClose` settings
- `UpdateInterval` modes

### Recommended Settings
```mql5
input string RestServerURL = "http://127.0.0.1:5000/predict";  // Inference server
input double MinConfidence = 0.60;                              // 60% minimum confidence
input bool   EnableTrading = false;                             // Start in DEMO mode
input bool   ShowDebugInfo = true;                              // Enable for testing
```

## Testing Checklist

### Before Live Trading
- [ ] Start inference server: `python inference_server.py`
- [ ] Verify models loaded: Check server logs for 4 pairs
- [ ] Test health endpoint: `http://127.0.0.1:5000/health`
- [ ] Run test suite: `python test_inference_server.py`
- [ ] Attach EA to chart in DEMO mode (`EnableTrading = false`)
- [ ] Verify symbol conversion: Check Experts log for "EUR_USD" format
- [ ] Verify 250 M5 bars collected: Check "Sending request" log
- [ ] Verify response received: Check prediction/confidence output
- [ ] Test with all 4 supported pairs: EUR_USD, GBP_USD, USD_JPY, AUD_USD
- [ ] Verify chart display: Should show prediction, confidence, probabilities
- [ ] Enable trading: Set `EnableTrading = true` in DEMO account
- [ ] Verify trades execute: Check position opens when confidence ≥ MinConfidence

### Error Handling Validation
- [ ] Server offline: EA should print "WebRequest failed. Error: 4060"
- [ ] Insufficient data: EA should print "Insufficient M5 bars. Need 250"
- [ ] Unsupported pair: Server returns 400 error (e.g., "XYZ_ABC")
- [ ] Feature validation failure: Server returns 400 error with details
- [ ] Model not found: Server returns 404 error

## Known Limitations

1. **Symbol Support**: Only EUR_USD, GBP_USD, USD_JPY, AUD_USD
2. **Timeframe**: Must use M5 (5-minute) candles - other timeframes not supported
3. **Data Requirement**: Needs 250+ M5 bars (approximately 21 hours of history)
4. **Server Dependency**: EA requires inference server running locally or on network
5. **Feature Engineering**: Server must have sufficient data for 58 features (will fail with < 250 candles)

## Troubleshooting

### "WebRequest failed. Error: 4060"
- **Cause**: URL not whitelisted in MetaTrader
- **Fix**: Tools → Options → Expert Advisors → Allow WebRequest for: `http://127.0.0.1:5000`

### "Insufficient M5 bars"
- **Cause**: Not enough historical data loaded
- **Fix**: 
  1. Right-click chart → Properties → Max bars in chart: 10000
  2. Wait for data download (MetaTrader loads from broker)
  3. Restart EA

### "SERVER ERROR: Unsupported pair"
- **Cause**: Symbol not in trained models
- **Fix**: Use only EUR_USD, GBP_USD, USD_JPY, AUD_USD

### "SERVER ERROR: Feature count mismatch"
- **Cause**: Models expect 58 features, engineered features != 58
- **Fix**: Retrain models with updated code (includes feature metadata)

### Prediction always "HOLD"
- **Cause**: Confidence below MinConfidence threshold
- **Fix**: Lower MinConfidence input (try 0.50 - 0.55)
- **Note**: HOLD is valid - model may be uncertain about direction

## Next Steps After Retraining

1. **Download New Models**: After Kaggle retraining completes (~4 hours)
   - Download 4 model sets (16 .pkl/.pth files)
   - Download 4 `feature_schema.json` files
   - Place in `models/` directory

2. **Test Inference Server**:
   ```bash
   python inference_server.py
   python test_inference_server.py
   ```

3. **Update EA** (if needed):
   - Already updated to match inference server
   - Just verify `RestServerURL` points to server

4. **Integration Test**:
   - Start inference server
   - Attach EA to chart (DEMO mode)
   - Verify predictions received
   - Enable trading in demo account
   - Monitor for 24 hours before live

## Architecture Flow

```
MQL5 EA (MetaTrader)
    ↓
Collects 250 M5 OHLCV candles
    ↓
Converts EURUSD → EUR_USD
    ↓
Sends JSON to http://127.0.0.1:5000/predict
    ↓
Inference Server (Python Flask)
    ↓
Engineers 58 features (TechnicalFeatureEngineer + 3 macro)
    ↓
Validates features match training schema
    ↓
Loads HybridEnsemble model from cache
    ↓
Predicts: {prediction: "BUY", confidence: 0.85, probabilities: {...}}
    ↓
Returns JSON response
    ↓
EA parses prediction
    ↓
Converts "BUY" → 1, "SELL" → -1, "HOLD" → 0
    ↓
Executes trade if confidence ≥ MinConfidence
```

## Critical Success Factors

✅ **Feature Alignment**: Inference server uses EXACT same Python code as training
✅ **Symbol Format**: EA converts to underscore format (EUR_USD)
✅ **Timeframe Match**: EA sends M5 data, models trained on M5
✅ **Bar Count**: EA sends 250+ bars, sufficient for all features
✅ **Validation**: Server validates feature names/order before prediction
✅ **Error Handling**: EA handles server errors gracefully

## Files Modified

- `MQL5/Auron AI.mq5` (updated 7 functions, added 2 new functions)

## Files Created

- `inference_server.py` (326 lines, production Flask server)
- `test_inference_server.py` (343 lines, comprehensive test suite)
- `MQL5/EA_UPDATE_NOTES.md` (this document)

## Commit History

- **Commit 0faf85d**: Added feature metadata tracking to models
- **Commit 36f9eb8**: Created validation script and identified retraining need
- **Current**: Updated MQL5 EA + created inference server + test suite

---

**Author**: AI Assistant  
**Date**: 2024  
**Project**: Macro-Technical Sentiment Classifier  
**Phase**: Integration - Ready for Post-Retraining Testing
