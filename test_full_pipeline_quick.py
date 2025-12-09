"""
Quick test of the full pipeline with ZigZag integration.
Tests: Data loading → Feature engineering → Target creation → Model training (1 fold)
"""
import sys
import warnings
warnings.filterwarnings('ignore')

from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from loguru import logger
import pandas as pd
import numpy as np

# Configure minimal logging
logger.remove()
logger.add(sys.stderr, level="INFO", format="{time:HH:mm:ss} | {level} | {message}")

print("="*80)
print("QUICK PIPELINE TEST - ZigZag Integration")
print("="*80)

# Step 1: Load data
print("\n[1/5] Loading data...")
from src.data_acquisition.kaggle_loader import KaggleFXDataLoader
loader = KaggleFXDataLoader()
df = loader.load_symbol_data("EURUSD", timeframe="M5")
df = df.head(20000).copy()  # Use 20K bars for quick test
print(f"  ✓ Loaded {len(df):,} bars")

# Step 2: Engineer features (simplified - 5 features)
print("\n[2/5] Engineering features...")
import talib

# RSI normalized
df['rsi_12'] = talib.RSI(df['close'], timeperiod=12)
df['rsi_norm'] = (df['rsi_12'] - 50.0) / 50.0

# MACD diff normalized
macd, macd_signal, _ = talib.MACD(df['close'], fastperiod=12, slowperiod=48, signalperiod=12)
macd_diff = np.abs(macd - macd_signal)
df['macd_diff_norm'] = ((macd_diff - macd_diff.mean()) / (macd_diff.std() * 3)).clip(-1, 1)

# Candle body normalized
candle_body = df['close'] - df['open']
df['candle_body_norm'] = ((candle_body - candle_body.mean()) / (candle_body.std() * 3)).clip(-1, 1)

# Placeholder macro features
df['yield_curve'] = 0.0
df['dxy_index'] = 100.0

print(f"  ✓ Calculated 5 features")

# Step 3: Create ZigZag targets
print("\n[3/5] Creating ZigZag targets...")
from src.utils.zigzag import calculate_zigzag_extrema, create_zigzag_targets, validate_zigzag_quality
from src.config import ZIGZAG_CONFIG

df = calculate_zigzag_extrema(df, **ZIGZAG_CONFIG)
df = create_zigzag_targets(df, pip_multiplier=10000)

# Map to target_class for compatibility
df['target_class'] = df['target_direction'].astype(float)
df.dropna(inplace=True)

print(f"  ✓ Created targets, {len(df):,} samples ready")

# Step 4: Validate quality
print("\n[4/5] Validating ZigZag quality...")
is_valid = validate_zigzag_quality(df, min_correlation=0.20)

if not is_valid:
    print("  ⚠ Quality validation failed - but continuing for test")

# Step 5: Quick model test (just XGBoost, no LSTM for speed)
print("\n[5/5] Testing model training (XGBoost only)...")

feature_cols = ['rsi_norm', 'macd_diff_norm', 'candle_body_norm', 'yield_curve', 'dxy_index']

# Simple train/test split
train_size = int(len(df) * 0.8)
X_train = df[feature_cols].values[:train_size]
y_train = df['target_class'].values[:train_size].astype(int)
X_test = df[feature_cols].values[train_size:]
y_test = df['target_class'].values[train_size:].astype(int)

from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report

model = XGBClassifier(
    n_estimators=100,
    max_depth=4,
    learning_rate=0.1,
    random_state=42,
    use_label_encoder=False,
    eval_metric='logloss'
)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"  ✓ XGBoost trained, Test Accuracy: {accuracy:.2%}")

# Classification report
print("\n" + "="*80)
print("CLASSIFICATION REPORT")
print("="*80)
print(classification_report(y_test, y_pred, target_names=['Sell (0)', 'Buy (1)']))

# Feature importance
print("\n" + "="*80)
print("FEATURE IMPORTANCE")
print("="*80)
for name, importance in sorted(zip(feature_cols, model.feature_importances_), key=lambda x: -x[1]):
    print(f"  {name:20s}: {importance:.4f}")

# Summary
print("\n" + "="*80)
print("TEST SUMMARY")
print("="*80)
print(f"  Data samples: {len(df):,}")
print(f"  Features: {len(feature_cols)}")
print(f"  Train/Test split: {train_size:,} / {len(df)-train_size:,}")
print(f"  Test Accuracy: {accuracy:.2%}")
print(f"  ZigZag Quality: {'✓ PASSED' if is_valid else '⚠ NEEDS REVIEW'}")

if accuracy > 0.55:
    print("\n✓ PIPELINE TEST PASSED - Ready for full training")
else:
    print("\n⚠ Accuracy below 55% - May need parameter tuning")

print("="*80)
