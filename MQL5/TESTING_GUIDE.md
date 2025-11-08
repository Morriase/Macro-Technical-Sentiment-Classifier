# MQL5 EA Testing Guide - Quick Reference

## Pre-Flight Checklist

### 1. Inference Server Setup
```bash
# Terminal 1: Start inference server
cd "c:\Users\Morris\Desktop\Macro-Technical Sentiment Classifier"
python inference_server.py

# Expected output:
# [INFO] Loading model for EUR_USD...
# [INFO] Loading model for GBP_USD...
# [INFO] Loading model for USD_JPY...
# [INFO] Loading model for AUD_USD...
# [INFO] All models loaded successfully
# * Running on http://127.0.0.1:5000
```

### 2. Test Suite Validation
```bash
# Terminal 2: Run comprehensive tests
python test_inference_server.py

# Expected output:
# ========================================
# RUNNING ALL TESTS
# ========================================
# 
# [1/7] Health Check...
# ✅ Health check PASSED
# 
# [2/7] Single Prediction (EUR_USD)...
# ✅ Single prediction PASSED
# 
# [3/7] Insufficient Data Error...
# ✅ Insufficient data error handling PASSED
# 
# [4/7] Invalid Pair Error...
# ✅ Invalid pair error handling PASSED
# 
# [5/7] Model Info Retrieval...
# ✅ Model info PASSED
# 
# [6/7] Batch Predictions...
# ✅ Batch prediction PASSED
# 
# [7/7] All Supported Pairs...
# ✅ All pairs PASSED
# 
# ========================================
# TEST SUMMARY
# ========================================
# Total: 7 | Passed: 7 | Failed: 0
```

### 3. MetaTrader Setup

#### A. Allow WebRequest
1. **Tools** → **Options** → **Expert Advisors**
2. **Allow WebRequest for listed URLs**: ✅
3. Add URL: `http://127.0.0.1:5000`
4. Click **OK**

#### B. EA Configuration
Open EA settings on chart:

**Critical Settings**:
```
RestServerURL = http://127.0.0.1:5000/predict
EnableTrading = false                    // Start in DEMO
MinConfidence = 0.60                     // 60% minimum
ShowDebugInfo = true                     // For testing
```

**Risk Management** (adjust to your account):
```
FixedRiskPercent = 1.0                   // 1% risk per trade
MinRiskReward = 1.5                      // 1:1.5 RR ratio
```

**Update Mode**:
```
UpdateIntervalSeconds = 0                // 0 = New M5 bar only
                                         // 1 = Timer mode
updateSeconds = 300                      // If timer mode (5 min)
```

#### C. Supported Symbols
Attach EA to **one of these charts**:
- **EURUSD** (or EURUSD.ecn, EURUSD.raw, etc.)
- **GBPUSD** (or GBPUSD.ecn, GBPUSD.raw, etc.)
- **USDJPY** (or USDJPY.ecn, USDJPY.raw, etc.)
- **AUDUSD** (or AUDUSD.ecn, AUDUSD.raw, etc.)

**Timeframe**: Any (EA auto-fetches M5 data)

---

## Testing Sequence

### Phase 1: Server Validation (5 minutes)

#### Test 1: Health Check
```bash
# In browser or curl:
http://127.0.0.1:5000/health

# Expected response:
{
  "status": "healthy",
  "loaded_models": 4,
  "supported_pairs": ["EUR_USD", "GBP_USD", "USD_JPY", "AUD_USD"]
}
```

#### Test 2: Model Info
```bash
http://127.0.0.1:5000/model_info/EUR_USD

# Expected response:
{
  "pair": "EUR_USD",
  "n_features": 58,
  "feature_names": ["rsi_14", "macd", "macd_signal", ...],
  "model_version": "HybridEnsemble_v1.0",
  "trained_date": "2024-..."
}
```

---

### Phase 2: EA Startup (2 minutes)

#### Step 1: Attach EA
1. Drag `Auron AI.ex5` to EURUSD chart
2. Configure settings (see above)
3. Click **OK**

#### Step 2: Check Experts Log
Press **Ctrl+T** → **Experts** tab

**Expected output**:
```
========================================
HYBRID ENSEMBLE EA - MACRO-TECHNICAL SENTIMENT
========================================
Server: http://127.0.0.1:5000/predict
Timeframe: M5 (5-minute candles)
Bars: 250 (for feature engineering)
Features: 58 (55 Technical + 3 Macro)
Chart Timeframe: PERIOD_CURRENT
Update Mode: New M5 Bar Only
Min Confidence: 0.6
Trading: DEMO MODE
========================================
Making first prediction...
```

#### Step 3: Verify Symbol Conversion
Look for this in log:
```
Symbol: EURUSD → Formatted: EUR_USD
```

#### Step 4: Verify Request Sent
```
Sending request to: http://127.0.0.1:5000/predict
Response received: {"prediction":"BUY","confidence":0.85...
```

#### Step 5: Verify Prediction Displayed
```
========================================
HYBRID ENSEMBLE PREDICTION: BUY
Confidence: 85.0%
Probabilities:
  SELL: 5.0%
  HOLD: 10.0%
  BUY: 85.0%
========================================
```

#### Step 6: Check Chart Display
Chart should show:
```
HYBRID ENSEMBLE - MACRO-TECHNICAL SENTIMENT
═══════════════════════════════════════

Timeframe: M5 (250 bars)
Features: 58 (55 Technical + 3 Macro)

PREDICTION: BUY
Confidence: 85.0%

───────────────────────────────────────
Probabilities:
  SELL: 5.0%
  HOLD: 10.0%
  BUY: 85.0%

───────────────────────────────────────
Min Confidence: 60%
Trading: DEMO
Requests: 1 | Success: 1

Server: http://127.0.0.1:5000/predict
```

---

### Phase 3: Trade Execution Test (30 minutes)

#### Prerequisites
- EA attached and receiving predictions ✅
- `EnableTrading = false` (DEMO mode) ✅
- Switch to `EnableTrading = true` ⚠️

#### Step 1: Enable Trading
1. Right-click EA on chart
2. **Expert Advisors** → **Properties**
3. **Inputs** → `EnableTrading = true`
4. Click **OK**

#### Step 2: Wait for Signal
- **New M5 bar triggers prediction** (every 5 minutes)
- If confidence ≥ MinConfidence → Trade executes
- If confidence < MinConfidence → "SKIPPED" in log

#### Step 3: Verify Trade Execution
**In Experts Log**:
```
✅ EXECUTING: Server-approved high-quality setup
EXECUTING BUY: 0.01 lots at 1.08456 (SL:1.08356 [100 pts] TP:1.08606)
SUCCESS: Order placed. Ticket: 123456789
```

**In Terminal → Trade Tab**:
- Position opened
- Symbol: EURUSD
- Type: Buy/Sell
- Volume: (based on risk)
- SL/TP: Set

#### Step 4: Monitor for 30 Minutes
- EA should update every M5 bar (5 minutes)
- Should receive 6 predictions total
- Trades execute when confidence ≥ MinConfidence

---

### Phase 4: All Pairs Test (1 hour)

#### Test Each Supported Pair
1. **EURUSD** → Attach EA → Wait for prediction ✅
2. **GBPUSD** → Attach EA → Wait for prediction ✅
3. **USDJPY** → Attach EA → Wait for prediction ✅
4. **AUDUSD** → Attach EA → Wait for prediction ✅

**Expected**: All 4 pairs should receive predictions without errors

---

## Error Scenarios

### Error 1: "WebRequest failed. Error: 4060"
**Symptoms**:
```
ERROR: WebRequest failed. Error: 4060
  URL not allowed. Add to Tools > Options > Expert Advisors
```

**Fix**:
1. Tools → Options → Expert Advisors
2. Allow WebRequest for: `http://127.0.0.1:5000`
3. Restart EA

---

### Error 2: "Insufficient M5 bars"
**Symptoms**:
```
ERROR: Insufficient M5 bars. Need 250, have 120
ERROR: Failed to prepare OHLCV data
```

**Fix**:
1. Right-click chart → **Properties**
2. **Max bars in chart**: 10000
3. Wait 5-10 minutes for data download
4. Restart EA

**Note**: Some brokers limit historical data. If problem persists, contact broker.

---

### Error 3: "SERVER ERROR: Unsupported pair"
**Symptoms**:
```
SERVER ERROR: Unsupported pair: XYZ_ABC
```

**Fix**: Use only EUR_USD, GBP_USD, USD_JPY, AUD_USD

---

### Error 4: Feature Validation Failed
**Symptoms**:
```
SERVER ERROR: Feature validation failed
Expected features: ['rsi_14', 'macd', ...]
Received features: ['rsi_14', 'ema_20', ...]
```

**Cause**: Models were trained with old feature code, new code generates different features

**Fix**: Retrain models with updated code (includes feature metadata)

---

### Error 5: "Cannot connect to server"
**Symptoms**:
```
ERROR: WebRequest failed. Error: 4014
```

**Checks**:
1. Is inference server running? Check Terminal 1
2. Is URL correct? `http://127.0.0.1:5000/predict`
3. Is firewall blocking? Temporarily disable and test

---

### Error 6: Prediction Always "HOLD"
**Symptoms**:
```
HYBRID ENSEMBLE PREDICTION: HOLD
Confidence: 45.0%
...
SKIPPED: Confidence 45.0% below minimum 60%
```

**Cause**: Model confidence below threshold

**Options**:
1. **Lower MinConfidence**: Try 0.50 - 0.55 (less selective)
2. **Wait for better setup**: HOLD is valid - market may be uncertain
3. **Check probabilities**: If HOLD probability > 80%, model is genuinely uncertain

---

## Performance Monitoring

### Key Metrics to Track

#### 1. Request Success Rate
```
Requests: 100 | Success: 98
```
**Target**: > 95% success rate  
**If < 90%**: Check server logs for errors

#### 2. Confidence Distribution
Track confidence levels over 24 hours:
- **High Confidence (> 70%)**: Should generate trades
- **Medium Confidence (55-70%)**: Depends on MinConfidence
- **Low Confidence (< 55%)**: Should be filtered (HOLD)

#### 3. Signal Distribution
Over 24 hours:
- **BUY**: ~30-40% of signals
- **SELL**: ~30-40% of signals
- **HOLD**: ~20-30% of signals

**If HOLD > 80%**: Model may be too conservative (lower MinConfidence)  
**If HOLD < 10%**: Model may be too aggressive (raise MinConfidence)

#### 4. Trade Execution Rate
```
Predictions: 100 | Trades: 30
```
**Execution Rate**: 30% (depends on MinConfidence)

**Optimal**:
- MinConfidence = 0.55 → ~50% execution
- MinConfidence = 0.60 → ~30% execution
- MinConfidence = 0.70 → ~10% execution

---

## Log Files

### 1. Inference Server Log
**Location**: `logs/inference_server.log`

**Check For**:
- Feature engineering errors
- Model loading errors
- Validation failures
- Request processing times

### 2. MetaTrader Experts Log
**Location**: MetaTrader → Terminal → Experts tab

**Check For**:
- WebRequest errors
- Prediction results
- Trade execution confirmations
- Confidence levels

### 3. Trade Log CSV
**Location**: `MQL5/Files/BlackIce_Trades.csv`

**Contains**:
- Timestamp
- Symbol
- Signal (BUY/SELL)
- Confidence
- Entry price
- SL/TP levels

---

## Next Steps After 24-Hour Test

### If All Tests Pass ✅
1. **Review Performance**:
   - Win rate (target: > 55%)
   - Average confidence (target: > 0.60)
   - Risk/Reward ratio (target: > 1.5)

2. **Consider Live Trading**:
   - Start with minimum lot size
   - Use broker demo account first
   - Monitor for 1 week before real account

### If Tests Fail ❌
1. **Collect Diagnostics**:
   - MetaTrader Experts log (full)
   - `logs/inference_server.log`
   - `BlackIce_Trades.csv`

2. **Check Common Issues**:
   - Server not running
   - URL not whitelisted
   - Insufficient bars
   - Unsupported symbol
   - Models not trained/loaded

3. **Debug Steps**:
   - Enable `ShowDebugInfo = true`
   - Run `test_inference_server.py`
   - Check feature count (should be 58)
   - Verify model files exist

---

## Support Checklist

If you need help, provide:
- [ ] MetaTrader version
- [ ] Broker name
- [ ] Symbol tested
- [ ] EA settings (screenshot)
- [ ] Experts log (last 100 lines)
- [ ] `inference_server.log` (last 100 lines)
- [ ] Test suite results (`test_inference_server.py`)

---

**Document Version**: 1.0  
**Last Updated**: 2024  
**Status**: Ready for Post-Retraining Testing
