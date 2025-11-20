# Feature Mismatch Fix - November 20, 2025

## Problem

EA was receiving this error from the inference server:
```
"error": "Validation error: X has 81 features, but StandardScaler is expecting 58 features as input."
```

## Root Cause

**Model Path Naming Mismatch:**

1. **Training script** (main.py) saves models as:
   - `EUR_USD_model_config.pkl`
   - `EUR_USD_model_xgb_base.pkl`
   - `EUR_USD_model_lstm_base.pth`
   - `EUR_USD_model_meta.pkl`

2. **Inference server** (inference_server.py) was loading:
   - `EUR_USD_model.pth_config.pkl` ← **WRONG!**
   - This loaded OLD models from Nov 8 with 58 features

3. **Result:** Server loaded old scaler (58 features) but received new data (81 features)

## Training Progress Analysis

From `resources/training_progress.txt`:

```
2025-11-16 12:35:31 | INFO | main:engineer_features:278 | ✓ 86 total features created, 79800 samples ready.
2025-11-16 12:35:31 | INFO | main:train_model:417 | Using 81 features for training
```

**Feature Breakdown (81 total):**
- 67 base technical features (M5 timeframe)
- 14 multi-timeframe features (H1 + H4)
- 3 macro features (tau_pre, tau_post, weighted_surprise)
- 0 sentiment features (news dataset not attached in Kaggle)

**Note:** 86 features are created initially, but 5 are excluded during training:
- `open`, `high`, `low`, `close`, `volume` (OHLCV columns)

## Solution

**Fixed inference_server.py line 109:**

```python
# BEFORE (WRONG):
model_path = MODELS_DIR / f"{pair}_model.pth"

# AFTER (CORRECT):
model_path = MODELS_DIR / f"{pair}_model"
```

This ensures the server loads the correct config files:
- ✓ `EUR_USD_model_config.pkl` (Nov 20, 81 features)
- ✗ `EUR_USD_model.pth_config.pkl` (Nov 8, 58 features)

## Model Files

### Current Models (Nov 20, 2025 - 81 features):
```
AUD_USD_model_config.pkl     20/11/2025 12:35:36
EUR_USD_model_config.pkl     20/11/2025 12:35:48
GBP_USD_model_config.pkl     20/11/2025 12:36:34
USD_JPY_model_config.pkl     20/11/2025 12:36:36
```

### Old Models (Nov 8, 2025 - 58 features):
```
AUD_USD_model.pth_config.pkl    08/11/2025 11:42:28
EUR_USD_model.pth_config.pkl    08/11/2025 11:42:32
GBP_USD_model.pth_config.pkl    08/11/2025 11:42:36
USD_JPY_model.pth_config.pkl    08/11/2025 11:42:38
```

**Recommendation:** Delete old `.pth_*` files to avoid confusion.

## Verification

After deploying the fix, the EA should receive successful predictions:

```json
{
  "pair": "USDJPY",
  "prediction": "BUY",
  "confidence": 0.85,
  "feature_count": 81,
  "status": "success"
}
```

## Next Steps

1. ✅ Fix applied to `inference_server.py`
2. 🔄 Restart inference server (or redeploy to Render)
3. 🧪 Test with EA to confirm predictions work
4. 🗑️ Optional: Clean up old model files from Nov 8
