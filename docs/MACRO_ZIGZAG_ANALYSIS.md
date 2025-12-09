# Macro Features Analysis with ZigZag Target

**Date:** December 10, 2025  
**Status:** ✅ COMPLETE  
**Script:** `test_macros_with_zigzag.py`

---

## Executive Summary

Tested whether macro features (VIX, yield curve, rate differential, etc.) improve predictions when using the ZigZag-based target definition. 

**VERDICT:** ✅ **ADD 2 MACRO FEATURES**

---

## Key Findings

### Base Features (Engineer's 3)
| Feature | Direction Corr | Magnitude Corr |
|---------|---------------|----------------|
| RSI (normalized) | **33.2%** | **17.5%** |
| Candle Body | 9.9% | 5.5% |
| MACD Diff | 3.5% | 1.3% |

**Average:** 15.6% (direction), 8.1% (magnitude)

### Macro Features Performance
| Feature | Direction Corr | Magnitude Corr |
|---------|---------------|----------------|
| Yield Curve | **4.4%** | 1.7% |
| DXY Index | **4.4%** | 3.3% |
| VIX | 2.5% | **6.0%** |
| Rate Differential | 3.6% | 0.8% |
| Oil Price | 0.8% | 0.4% |

**Average:** 3.1% (direction), 2.5% (magnitude)

---

## Critical Comparison

### Direction Prediction
- **Best macro:** Yield Curve (4.4%)
- **Worst base:** MACD Diff (3.5%)
- **Result:** ✅ Best macro beats worst base feature

### Magnitude Prediction
- **Best macro:** VIX (6.0%)
- **Worst base:** MACD Diff (1.3%)
- **Result:** ✅ Best macro significantly beats worst base feature

---

## Recommendation

### Final Feature Set (5 features)

**Base Features (3):**
1. RSI (normalized to [-1, 1])
2. Candle Body (normalized)
3. MACD Diff (normalized)

**Macro Features (2):**
4. **Yield Curve** (T10Y2Y) - Best for direction (4.4%)
5. **DXY Index** (Dollar strength) - Consistent across both targets (4.4% direction, 3.3% magnitude)

**Alternative:** Could use VIX instead of DXY for magnitude-focused predictions (6.0% correlation)

---

## Why Add Macros?

1. **Beat baseline:** Best macros outperform worst base feature (MACD)
2. **Above threshold:** Yield Curve and DXY both >4% correlation
3. **Complementary:** Macros capture different market dynamics than technical indicators
4. **Minimal complexity:** Only adding 2 features (5 total vs engineer's 3)

---

## Why NOT Add More Macros?

1. **Diminishing returns:** Rate differential (3.6%), VIX (2.5%), Oil (0.8%) add little value
2. **Simplicity:** Engineer achieved 65% with 3 features - we're already at 5
3. **Overfitting risk:** More features = more noise in LSTM training

---

## Implementation Notes

### Feature Normalization
All features normalized to [-1, 1] range:
```python
feature_norm = (feature - mean) / (std * 3)
feature_norm = feature_norm.clip(-1, 1)
```

### Data Merge Strategy
- Macro data is daily frequency
- M5 price data is 5-minute frequency
- Solution: Forward-fill macro values throughout the day
- This is realistic (traders know yesterday's VIX all day today)

---

## Comparison to Original Approach

### Before ZigZag (Binary Up/Down Target)
- RSI correlation: **1.3%** (essentially random)
- VIX correlation: **3.2%** (best macro)
- All features weak (<7%)

### After ZigZag (Direction + Magnitude to Extremum)
- RSI correlation: **33.2%** (25x improvement!)
- VIX correlation: **6.0%** (2x improvement)
- Yield Curve: **4.4%** (new signal discovered)

**Insight:** ZigZag target reveals true predictive power by filtering noise

---

## Next Steps

1. ✅ Macro analysis complete
2. ⏭️ Integrate ZigZag approach into `main.py`
3. ⏭️ Update feature engineering to use 5 features (RSI, Candle, MACD, Yield, DXY)
4. ⏭️ Modify LSTM for dual output (direction + magnitude)
5. ⏭️ Retrain and validate

---

## Files Generated

- `macros_vs_base_zigzag.png` - Visualization of all feature correlations
- `test_macros_with_zigzag.py` - Analysis script
- This document

---

## Conclusion

**Macros are worth adding, but only the best 2:**
- Yield Curve (macro regime indicator)
- DXY Index (currency strength baseline)

This gives us a **5-feature model** that balances:
- ✅ Simplicity (close to engineer's 3)
- ✅ Performance (macros beat worst base feature)
- ✅ Interpretability (each feature has clear market meaning)

Expected accuracy: **60-65%** (matching engineer's performance with slight macro boost)
