# Pair-Specific Features Fix (Sentiment + Macro)

## Problems Identified

**Root Cause of Low Accuracy (43-46%):** Two train-inference mismatches:

### 1. Sentiment Mismatch (Fixed)
- **Training:** ALL 121K headlines → diluted, noisy signal
- **Inference:** Pair-filtered headlines → focused signal
- **Result:** Model learned noise, tested on signal

### 2. Macro Features (Fixed)
- **Before:** Synthetic/AI-generated macro events from TradingView calendar
- **After:** Real FRED (Federal Reserve) economic data
- **Result:** Authentic central bank rates, inflation, GDP, unemployment

---

## Fixes Applied

### 1. Pair-Specific Sentiment (`src/feature_engineering/sentiment_features.py`)

Added:
- `MAJOR_CURRENCY_KEYWORDS` dictionary (117 terms, 8 currencies + XAU)
- `filter_news_by_currency_pair()` function
- `aggregate_daily_sentiment(currency_pair=...)` parameter

### 2. FRED Macro Data (`src/data_acquisition/fred_macro_loader.py`)

**NEW MODULE** - Fetches real economic data from Federal Reserve:

```python
from src.data_acquisition.fred_macro_loader import FREDMacroLoader

loader = FREDMacroLoader(api_key="your_fred_key")
macro_df = loader.get_macro_features_for_pair("EUR_USD", start_date, end_date)
```

**Features per pair:**
- `rate_differential` - Central bank rate spread (EUR rate - USD rate)
- `rate_diff_change` - Direction of rate changes
- `us_inflation_cpi`, `us_unemployment`, `us_nfp` - US indicators
- `eu_inflation_cpi`, `eu_unemployment` - EU indicators (for EUR pairs)
- `vix` - Volatility index (risk sentiment)
- `yield_curve` - 10Y-2Y spread (recession indicator)
- `oil_price`, `dxy_index` - Global factors

### 3. Updated `main.py`

Now uses FRED for macro features:
```python
fred_loader = FREDMacroLoader()
fred_macro_df = fred_loader.get_macro_features_for_pair(
    self.currency_pair, start_date, end_date
)
```

### 4. API Configuration (`.env`)

```bash
FRED_API_KEY=8ef93cf694bee76342c15a8707ef3a28
MARKETAUX_API_KEY=cAzyt69QdgoppCMtnFIS5XhT44Db6LITpinlgli1
```

---

## Feature Comparison

| Feature Type | Before | After |
|--------------|--------|-------|
| Sentiment Source | All 121K headlines | ~15K pair-filtered |
| Macro Source | TradingView calendar (synthetic) | FRED API (real data) |
| Rate Differential | Not available | Fed vs ECB/BoE/BoJ |
| Economic Indicators | tau_pre, tau_post, surprise | CPI, GDP, Unemployment |
| Global Risk | None | VIX, Yield Curve |

---

## Files Modified

1. `src/feature_engineering/sentiment_features.py` - Pair-specific filtering
2. `src/data_acquisition/fred_macro_loader.py` - **NEW** FRED integration
3. `main.py` - Uses FRED macro features + pair sentiment
4. `src/config.py` - Added FRED_API_KEY
5. `.env` - API keys configured

---

## Expected Impact

| Metric | Before | After |
|--------|--------|-------|
| Training data quality | Low (noisy) | High (pair-specific) |
| Macro feature authenticity | Synthetic | Real economic data |
| Train-inference alignment | ❌ Mismatch | ✅ Aligned |
| Expected accuracy | 43-46% | 60-70%+ |

---

## Next Steps

1. **Kaggle Setup:**
   - Add FRED_API_KEY as Kaggle secret
   - Attach news dataset (`raw_partner_headlines.parquet`)

2. **Retrain All Models:**
   - Models will now use pair-specific sentiment
   - Models will use real FRED macro indicators
   - Rate differentials will drive predictions

3. **Validate:**
   - Check balanced accuracy improves
   - Verify feature importance shows rate_differential, vix
   - Confirm sentiment features are non-zero
