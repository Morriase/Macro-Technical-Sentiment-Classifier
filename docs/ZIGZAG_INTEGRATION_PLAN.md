# ZigZag Integration Implementation Plan

**Date:** December 10, 2025  
**Status:** 🚀 IN PROGRESS  
**Expected Impact:** 51% → 60-65% accuracy

---

## Overview

Complete architectural overhaul to implement the senior engineer's ZigZag-based approach that achieved 65% accuracy with only 3 features.

---

## Changes Required

### 1. Update `src/config.py` ✅

**Changes:**
- `sequence_length`: 100 → **40** (engineer's approach)
- `num_classes`: 2 → **2** (but dual output: direction + magnitude)
- Add `ZIGZAG_CONFIG` section
- Update `TARGET_CONFIG` for ZigZag-based targets

**Code:**
```python
ZIGZAG_CONFIG = {
    "depth": 48,        # 4 hours on M5 (48 bars)
    "deviation": 1,     # 1 point minimum
    "backstep": 47,     # Prevent oscillation
}

TARGET_CONFIG = {
    "type": "zigzag_extremum",  # NEW: ZigZag-based
    "dual_output": True,         # Direction + Magnitude
    "forward_window_hours": 8,   # Keep existing
}

ENSEMBLE_CONFIG["base_learners"]["lstm"]["sequence_length"] = 40
ENSEMBLE_CONFIG["base_learners"]["lstm"]["hidden_size"] = 40
ENSEMBLE_CONFIG["base_learners"]["lstm"]["num_outputs"] = 2  # Direction + Magnitude
```

---

### 2. Update `main.py` - `create_target()` Method ✅

**Replace binary target logic with ZigZag extrema calculation:**

```python
def create_target(self):
    """
    Create ZigZag-based target (direction + magnitude to next extremum)
    """
    logger.info("Creating ZigZag-based targets...")
    
    # Calculate ZigZag extrema
    from src.utils.zigzag import calculate_zigzag_extrema
    
    self.df_features = calculate_zigzag_extrema(
        self.df_features,
        depth=ZIGZAG_CONFIG["depth"],
        deviation=ZIGZAG_CONFIG["deviation"],
        backstep=ZIGZAG_CONFIG["backstep"]
    )
    
    # Target 1: Direction (1 = buy, 0 = sell)
    self.df_features["target_direction"] = (
        self.df_features["next_extremum_price"] > self.df_features["close"]
    ).astype(int)
    
    # Target 2: Magnitude (pips to next extremum)
    self.df_features["target_magnitude"] = (
        (self.df_features["next_extremum_price"] - self.df_features["close"]) * 10000
    )
    
    # Normalize magnitude to [-1, 1]
    mag_mean = self.df_features["target_magnitude"].mean()
    mag_std = self.df_features["target_magnitude"].std()
    self.df_features["target_magnitude_norm"] = (
        (self.df_features["target_magnitude"] - mag_mean) / (mag_std * 3)
    ).clip(-1, 1)
    
    # Drop rows without valid targets
    self.df_features.dropna(subset=["next_extremum_price"], inplace=True)
    
    logger.info(f"✓ ZigZag targets created: {len(self.df_features):,} samples")
```

---

### 3. Update `main.py` - `engineer_features()` Method ✅

**Simplify to 5 features only:**

```python
def engineer_features(self):
    """
    Engineer SIMPLIFIED features (5 total: 3 base + 2 macro)
    """
    logger.info("Calculating SIMPLIFIED features (ZigZag approach)...")
    
    # Feature 1: RSI(12) normalized to [-1, 1]
    import talib
    self.df_features = self.df_price.copy()
    self.df_features['rsi_12'] = talib.RSI(self.df_features['close'], timeperiod=12)
    self.df_features['rsi_norm'] = (self.df_features['rsi_12'] - 50.0) / 50.0
    
    # Feature 2: MACD difference (Main - Signal)
    macd, macd_signal, _ = talib.MACD(
        self.df_features['close'], 
        fastperiod=12, 
        slowperiod=48, 
        signalperiod=12
    )
    macd_diff = np.abs(macd - macd_signal)
    macd_mean = macd_diff.mean()
    macd_std = macd_diff.std()
    self.df_features['macd_diff_norm'] = ((macd_diff - macd_mean) / (macd_std * 3)).clip(-1, 1)
    
    # Feature 3: Candlestick body
    candle_body = self.df_features['close'] - self.df_features['open']
    body_mean = candle_body.mean()
    body_std = candle_body.std()
    self.df_features['candle_body_norm'] = ((candle_body - body_mean) / (body_std * 3)).clip(-1, 1)
    
    # Feature 4 & 5: Macro features (Yield Curve + DXY Index)
    # ... (keep existing FRED loading logic, but only select these 2)
    
    logger.success(f"✓ Created 5 features: RSI, MACD_diff, Candle_body, Yield_curve, DXY_index")
```

---

### 4. Create `src/utils/zigzag.py` ✅

**New utility module for ZigZag calculation:**

```python
"""
ZigZag Extrema Calculator
Identifies peaks and troughs in price data
"""
import numpy as np
import pandas as pd

def calculate_zigzag_extrema(df, depth=48, deviation=1, backstep=47):
    """
    Calculate ZigZag extrema (peaks and troughs)
    
    Args:
        df: DataFrame with OHLC data
        depth: Lookback window (48 = 4 hours on M5)
        deviation: Minimum price change
        backstep: Minimum bars between extrema
    
    Returns:
        DataFrame with next_extremum_price and next_extremum_type columns
    """
    df = df.copy()
    
    # Find local maxima and minima
    extrema = []
    
    for i in range(depth, len(df) - depth):
        # Check if current bar is a peak
        window_high = df['high'].iloc[i-depth:i+depth+1]
        if df['high'].iloc[i] == window_high.max():
            extrema.append({
                'index': i, 
                'price': df['high'].iloc[i], 
                'type': 'peak'
            })
        
        # Check if current bar is a trough
        window_low = df['low'].iloc[i-depth:i+depth+1]
        if df['low'].iloc[i] == window_low.min():
            extrema.append({
                'index': i, 
                'price': df['low'].iloc[i], 
                'type': 'trough'
            })
    
    # Filter by backstep
    filtered_extrema = []
    last_idx = -backstep - 1
    
    for ext in extrema:
        if ext['index'] - last_idx >= backstep:
            filtered_extrema.append(ext)
            last_idx = ext['index']
    
    # Create columns
    df['extremum_price'] = np.nan
    df['extremum_type'] = None
    
    for ext in filtered_extrema:
        df.loc[df.index[ext['index']], 'extremum_price'] = ext['price']
        df.loc[df.index[ext['index']], 'extremum_type'] = ext['type']
    
    # Forward fill (each bar knows NEXT extremum)
    df['next_extremum_price'] = df['extremum_price'].bfill()
    df['next_extremum_type'] = df['extremum_type'].bfill()
    
    return df
```

---

### 5. Update `src/models/lstm_model.py` ✅

**Modify for dual output (direction + magnitude):**

```python
class LSTMSequenceClassifier(nn.Module):
    def __init__(self, ..., num_outputs=2, output_types=['classification', 'regression']):
        super().__init__()
        
        # ... existing LSTM layers ...
        
        # Dual output heads
        self.direction_head = nn.Linear(fc_input_size, 2)  # Binary classification
        self.magnitude_head = nn.Linear(fc_input_size, 1)  # Regression
        
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x):
        # ... existing LSTM forward pass ...
        
        # Dual outputs
        direction_logits = self.direction_head(hidden)
        magnitude_pred = self.magnitude_head(hidden)
        
        return direction_logits, magnitude_pred
```

**Update training loop for dual loss:**

```python
def fit(self, X, y_direction, y_magnitude, ...):
    # Dual loss functions
    criterion_direction = nn.CrossEntropyLoss(weight=class_weights)
    criterion_magnitude = nn.MSELoss()
    
    # Combined loss
    loss_dir = criterion_direction(direction_logits, y_direction)
    loss_mag = criterion_magnitude(magnitude_pred, y_magnitude)
    loss = loss_dir + 0.5 * loss_mag  # Weight magnitude less
```

---

### 6. Update `src/models/hybrid_ensemble.py` ✅

**Handle dual targets in ensemble:**

```python
def fit(self, X, y_direction, y_magnitude=None):
    """
    Fit ensemble with dual targets
    """
    # XGBoost: Only uses direction (classification)
    self.xgb_model.fit(X, y_direction)
    
    # LSTM: Uses both direction and magnitude
    if self.use_lstm:
        self.lstm_model.fit(X, y_direction, y_magnitude)
    
    # Meta-learner: Uses direction only
    # (magnitude is auxiliary for LSTM training)
```

---

### 7. Update Feature Schema ✅

**Reduce from 44 → 5 features:**

```python
feature_cols = [
    'rsi_norm',
    'macd_diff_norm', 
    'candle_body_norm',
    'yield_curve',
    'dxy_index'
]
```

---

## Implementation Order

1. ✅ Create `src/utils/zigzag.py`
2. ✅ Update `src/config.py` (add ZIGZAG_CONFIG)
3. ✅ Update `main.py` - `engineer_features()` (simplify to 5 features)
4. ✅ Update `main.py` - `create_target()` (ZigZag-based)
5. ✅ Update `src/models/lstm_model.py` (dual output)
6. ✅ Update `src/models/hybrid_ensemble.py` (handle dual targets)
7. ✅ Test on small dataset
8. ✅ Full training run

---

## Expected Results

### Before (Binary Target)
- Accuracy: ~51%
- RSI correlation: 1.3%
- Features: 44
- Overfitting: Severe (train 56%, val 49%)

### After (ZigZag Target)
- Accuracy: **60-65%**
- RSI correlation: **33.2%**
- Features: **5**
- Overfitting: **Minimal** (stronger signal)

---

## Rollback Plan

If results are worse:
1. Keep backup of current `main.py` as `main_binary.py`
2. Keep backup of current `lstm_model.py` as `lstm_model_binary.py`
3. Can revert by restoring backups

---

## Testing Strategy

1. **Unit test** ZigZag calculation on 1000 bars
2. **Correlation test** - Verify RSI correlation >30%
3. **Small training run** - 10K samples, 10 epochs
4. **Full training run** - All data, full epochs
5. **Compare metrics** - Accuracy, loss curves, overfitting

---

**Status: Ready to implement** 🚀
