# CRITICAL INFERENCE BUGS - MUST FIX BEFORE DEPLOYMENT

## Summary
The inference server has **feature misalignment** with training. This will cause incorrect predictions or validation failures.

---

## Bug #1: Sentiment Features Added But Not Trained ❌

**Training:** 0 sentiment features (news dataset not attached in Kaggle)
**Inference:** 9 sentiment features added if `ENABLE_LIVE_SENTIMENT = True`

**Impact:** Feature count mismatch (81 vs 90 features)

**Fix:**
```python
# In src/config.py
ENABLE_LIVE_SENTIMENT = False  # Keep disabled until models retrained with sentiment
```

---

## Bug #2: Multi-Timeframe Features Missing ❌

**Training:** 14 MTF features from H1 + H4 data
**Inference:** 0 MTF features (only M5 data from EA)

**Impact:** Missing 14 features → only 70 features instead of 81

**Root Cause:** EA only sends M5 OHLCV data, but training uses H1 + H4 data

**Fix Options:**

### Option A: Update EA to send H1 + H4 data (RECOMMENDED)
```json
{
    "pair": "EUR_USD",
    "ohlcv_m5": [...],   // 250 M5 candles
    "ohlcv_h1": [...],   // 250 H1 candles  
    "ohlcv_h4": [...]    // 250 H4 candles
}
```

Then update inference server:
```python
def engineer_features_from_ohlcv(df_m5, df_h1, df_h4, pair):
    # Base features on M5
    df_features = TECH_ENGINEER.calculate_all_features(df_m5.copy())
    df_features = TECH_ENGINEER.calculate_feature_crosses(df_features)
    
    # Add MTF features (CRITICAL - was missing!)
    higher_timeframes = {'H1': df_h1, 'H4': df_h4}
    df_features = TECH_ENGINEER.add_multi_timeframe_features(
        df_primary=df_features,
        higher_timeframes=higher_timeframes
    )
    
    return df_features
```

### Option B: Resample M5 to H1/H4 in inference (FALLBACK)
```python
def resample_to_higher_tf(df_m5, timeframe):
    """Resample M5 data to H1 or H4"""
    resample_rule = {'H1': '1H', 'H4': '4H'}[timeframe]
    return df_m5.resample(resample_rule).agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    })
```

**Pros:** No EA changes needed
**Cons:** Less accurate than real H1/H4 data (gaps, missing bars)

---

## Bug #3: Feature Count Documentation Wrong

**Documentation says:** 58 features
**Actual training:** 81 features
**Actual inference:** 70 features (without MTF) or 79 features (with sentiment)

**Fix:** Update all documentation to reflect 81 features

---

## Actual Feature Breakdown

### Training (Kaggle - CORRECT):
```
67 base technical (M5)
  ├─ 55 from calculate_all_features()
  └─ 12 from calculate_feature_crosses()
  
14 multi-timeframe (H1 + H4)
  ├─ 7 from H1 (RSI, MACD, ATR, etc.)
  └─ 7 from H4 (RSI, MACD, ATR, etc.)
  
3 macro
  ├─ tau_pre
  ├─ tau_post
  └─ weighted_surprise
  
0 sentiment (news not attached)

TOTAL: 81 features ✅
```

### Inference (Current - WRONG):
```
67 base technical (M5)
0 multi-timeframe ❌ MISSING!
3 macro
0-9 sentiment (if enabled, but shouldn't be)

TOTAL: 70 or 79 features ❌
```

---

## Action Items (Priority Order)

### 🔴 CRITICAL (Before ANY deployment):

1. **Disable sentiment in inference**
   ```python
   # src/config.py
   ENABLE_LIVE_SENTIMENT = False
   ```

2. **Add MTF features to inference** (choose Option A or B above)

3. **Test feature validation**
   ```bash
   python test_inference_server.py
   # Should see: "Feature validation passed ✓"
   # Should NOT see: "Feature count mismatch"
   ```

4. **Verify feature count**
   ```bash
   curl http://localhost:5000/model_info/EUR_USD
   # Should show: "n_features": 81
   ```

### 🟡 HIGH (Before live trading):

5. **Update all documentation** (58 → 81 features)
6. **Add integration test** to verify feature alignment
7. **Log feature names** on first prediction for debugging

### 🟢 MEDIUM (Nice to have):

8. **Add feature drift detection**
9. **Version feature schemas** with models
10. **Add automated feature validation tests**

---

## Testing Checklist

Before deploying inference server:

- [ ] `ENABLE_LIVE_SENTIMENT = False` in config
- [ ] MTF features added (Option A or B)
- [ ] Feature count = 81 (not 70 or 79)
- [ ] Feature validation passes
- [ ] Test prediction returns valid signal
- [ ] Feature names match training schema exactly
- [ ] No "Feature count mismatch" errors in logs

---

## Why This Happened

1. **Sentiment:** Code was written to support sentiment, but Kaggle training didn't have news dataset attached
2. **MTF:** Inference server was simplified to only use M5 data, but training uses H1+H4
3. **Documentation:** Written before final feature engineering was complete

---

## Recommended Fix (Quickest)

**Option B (Resample) + Disable Sentiment:**

1. Set `ENABLE_LIVE_SENTIMENT = False`
2. Add resampling function to inference server
3. Call `add_multi_timeframe_features()` with resampled data
4. Test and deploy

**Time:** ~30 minutes
**Risk:** Low (resampling is deterministic)

---

## Long-term Fix (Best)

**Option A (EA sends H1/H4) + Retrain with Sentiment:**

1. Update EA to send M5, H1, H4 data
2. Retrain models with news dataset attached
3. Update inference to use all timeframes + sentiment
4. Full integration test

**Time:** ~5 hours (4hr training + 1hr testing)
**Risk:** Medium (requires EA changes + retraining)

---

**Status:** 🔴 BLOCKING - Must fix before deployment
**Priority:** P0 - Critical
**Owner:** You
**ETA:** 30 minutes (Option B) or 5 hours (Option A)
