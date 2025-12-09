"""
Test ZigZag Integration
Verify ZigZag calculation and target creation before full integration
"""
import pandas as pd
import numpy as np
from src.data_acquisition.kaggle_loader import KaggleFXDataLoader
from src.utils.zigzag import calculate_zigzag_extrema, create_zigzag_targets, validate_zigzag_quality
from src.config import ZIGZAG_CONFIG
import talib
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("TESTING ZIGZAG INTEGRATION")
print("="*80)

# Load small sample of data
print("\n[1/5] Loading M5 data...")
loader = KaggleFXDataLoader()
df = loader.load_symbol_data("EURUSD", timeframe="M5")
print(f"  ✓ Loaded {len(df):,} M5 candles")

# Take first 10K bars for quick test
df = df.head(10000).copy()
print(f"  Using first {len(df):,} bars for testing")

# Calculate ZigZag extrema
print("\n[2/5] Calculating ZigZag extrema...")
df = calculate_zigzag_extrema(
    df,
    depth=ZIGZAG_CONFIG["depth"],
    deviation=ZIGZAG_CONFIG["deviation"],
    backstep=ZIGZAG_CONFIG["backstep"]
)

# Create targets
print("\n[3/5] Creating ZigZag targets...")
df = create_zigzag_targets(df, pip_multiplier=10000)

# Calculate simplified features
print("\n[4/5] Calculating simplified features...")

# Feature 1: RSI(12) normalized
df['rsi_12'] = talib.RSI(df['close'], timeperiod=12)
df['rsi_norm'] = (df['rsi_12'] - 50.0) / 50.0

# Feature 2: MACD difference
macd, macd_signal, _ = talib.MACD(df['close'], fastperiod=12, slowperiod=48, signalperiod=12)
macd_diff = np.abs(macd - macd_signal)
macd_mean = macd_diff.mean()
macd_std = macd_diff.std()
df['macd_diff_norm'] = ((macd_diff - macd_mean) / (macd_std * 3)).clip(-1, 1)

# Feature 3: Candlestick body
candle_body = df['close'] - df['open']
body_mean = candle_body.mean()
body_std = candle_body.std()
df['candle_body_norm'] = ((candle_body - body_mean) / (body_std * 3)).clip(-1, 1)

# Drop NaNs
df.dropna(inplace=True)
print(f"  ✓ Calculated 3 features, {len(df):,} samples ready")

# Validate quality
print("\n[5/5] Validating ZigZag quality...")
is_valid = validate_zigzag_quality(df, min_correlation=0.20)

# Detailed correlation analysis
print("\n" + "="*80)
print("FEATURE-TARGET CORRELATIONS")
print("="*80)

features = ['rsi_norm', 'macd_diff_norm', 'candle_body_norm']
targets = ['target_direction', 'target_magnitude']

for feature in features:
    for target in targets:
        valid_mask = ~df[target].isna()
        corr = np.corrcoef(df.loc[valid_mask, feature].values, df.loc[valid_mask, target].values)[0, 1]
        print(f"  {feature:20s} → {target:25s}: {corr:7.4f}")

# Summary
print("\n" + "="*80)
print("TEST SUMMARY")
print("="*80)

if is_valid:
    print("\n✓ ZIGZAG INTEGRATION TEST PASSED")
    print("  Ready to integrate into main.py")
else:
    print("\n⚠ ZIGZAG QUALITY BELOW THRESHOLD")
    print("  May need to adjust parameters")

print(f"\nSample data shape: {df.shape}")
print(f"Features: {features}")
print(f"Targets: ['target_direction', 'target_magnitude']")

# Save sample for inspection
df_sample = df[['close', 'rsi_norm', 'macd_diff_norm', 'candle_body_norm',
                'target_direction', 'target_magnitude', 'bars_to_extremum',
                'zigzag', 'extremum_type']].head(100)
df_sample.to_csv('zigzag_integration_test_sample.csv')
print(f"\n✓ Saved sample to: zigzag_integration_test_sample.csv")

print("\n" + "="*80)
print("✓ TEST COMPLETE")
print("="*80)
