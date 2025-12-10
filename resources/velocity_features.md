You can add "velocity" (rate of change) features directly in your `engineer_features` function using simple pandas difference operations.

This works because an LSTM often struggles to "calculate" the slope of a line from raw data points quickly. By explicitly calculating the slope (velocity) for the model, you give it a direct signal about **momentum** rather than just **position**.

Here is the step-by-step implementation.

### 1\. The Code Implementation

Locate your `engineer_features` function (likely in `main.py` or `src/features.py`). You need to add these lines **after** you calculate the base `rsi_norm` and `macd_diff_norm` but **before** you drop NaNs.

```python
def engineer_features(self, df):
    # ... [Existing code that calculates rsi_norm, macd_diff_norm, etc.] ...

    # --- NEW CODE BLOCK: VELOCITY FEATURES ---
    
    # 1. RSI Velocity (How fast is sentiment changing?)
    # We use a 1-period difference to see immediate momentum changes
    df['rsi_velocity'] = df['rsi_norm'].diff()

    # 2. MACD Velocity (Is the trend accelerating or decelerating?)
    df['macd_velocity'] = df['macd_diff_norm'].diff()
    
    # 3. (Optional) Acceleration - The "change of the change"
    # useful for detecting when a trend is losing steam before it actually reverses
    df['rsi_acceleration'] = df['rsi_velocity'].diff()

    # Fill the NaNs created by diff() (the first row will be NaN)
    df.fillna(0, inplace=True)
    
    # --- END NEW CODE BLOCK ---

    # ... [Rest of your function] ...
    return df
```

### 2\. Update the Feature Selection List

Just creating the columns isn't enough; you must tell the training pipeline to **include** them in the input vector.

Look for the line where you define the features for training. In your logs, it showed:
`Features: rsi_norm, macd_diff_norm, candle_body_norm, yield_curve, dxy_index`

You need to update that list (likely in `main.py` around line 520) to:

```python
# Update your feature columns list to include the new velocity features
feature_cols = [
    'rsi_norm', 
    'macd_diff_norm', 
    'candle_body_norm', 
    'yield_curve', 
    'dxy_index',
    'rsi_velocity',   # <--- Added
    'macd_velocity'   # <--- Added
]
```

### Why This Helps Your LSTM

  * **Context:** `RSI = 70` just means "Overbought."
  * **Context + Velocity:** `RSI = 70` AND `Velocity = -5` means "Overbought, but **crashing down**."

This explicit signal helps the LSTM distinguish between a strong trend continuing (positive velocity) and a reversal starting (negative velocity), which is critical for the ZigZag targets you are trying to predict.