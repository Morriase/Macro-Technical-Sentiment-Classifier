"""
Simple training script that bypasses walk-forward complexity.
Uses a simple train/val/test split with the ZigZag approach.
"""
import sys
import warnings
warnings.filterwarnings('ignore')

from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from loguru import logger
import pandas as pd
import numpy as np
import talib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import xgboost as xgb

# Configure logging
logger.remove()
logger.add(sys.stderr, level="INFO", format="{time:HH:mm:ss} | {level} | {message}")

print("="*80)
print("SIMPLE TRAINING - ZigZag Approach")
print("="*80)

# Step 1: Load data
print("\n[1/6] Loading data...")
from src.data_acquisition.kaggle_loader import KaggleFXDataLoader
from src.config import DATA_DIR, ZIGZAG_CONFIG

loader = KaggleFXDataLoader()
df = loader.load_symbol_data("EURUSD", timeframe="M5")
print(f"  ✓ Loaded {len(df):,} bars")

# Step 2: Engineer features (simplified - 5 features)
print("\n[2/6] Engineering features...")

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

# Placeholder macro features (will be 0 without FRED API)
df['yield_curve'] = 0.0
df['dxy_index'] = 100.0

print(f"  ✓ Calculated 5 features")

# Step 3: Create ZigZag targets
print("\n[3/6] Creating ZigZag targets...")
from src.utils.zigzag import calculate_zigzag_extrema, create_zigzag_targets, validate_zigzag_quality

df = calculate_zigzag_extrema(df, **ZIGZAG_CONFIG)
df = create_zigzag_targets(df, pip_multiplier=10000)

# Drop NaN targets
df = df.dropna(subset=['target_direction'])
df['target'] = df['target_direction'].astype(int)

print(f"  ✓ Created targets, {len(df):,} samples ready")

# Step 4: Validate quality
print("\n[4/6] Validating ZigZag quality...")
is_valid = validate_zigzag_quality(df, min_correlation=0.20)

# Step 5: Prepare data for training
print("\n[5/6] Preparing train/val/test splits...")

feature_cols = ['rsi_norm', 'macd_diff_norm', 'candle_body_norm', 'yield_curve', 'dxy_index']
X = df[feature_cols].values
y = df['target'].values

# Time-series split: 70% train, 15% val, 15% test
train_size = int(len(X) * 0.70)
val_size = int(len(X) * 0.15)

X_train = X[:train_size]
y_train = y[:train_size]
X_val = X[train_size:train_size+val_size]
y_val = y[train_size:train_size+val_size]
X_test = X[train_size+val_size:]
y_test = y[train_size+val_size:]

print(f"  Train: {len(X_train):,} samples")
print(f"  Val: {len(X_val):,} samples")
print(f"  Test: {len(X_test):,} samples")

# Step 6: Train XGBoost
print("\n[6/6] Training XGBoost...")

model = xgb.XGBClassifier(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=3,
    gamma=0.1,
    reg_alpha=0.1,
    reg_lambda=1.0,
    random_state=42,
    early_stopping_rounds=50,
    eval_metric='logloss',
    tree_method='hist',
    device='cuda' if __import__('torch').cuda.is_available() else 'cpu',
)

model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    verbose=50
)

# Evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("\n" + "="*80)
print("RESULTS")
print("="*80)
print(f"\nTest Accuracy: {accuracy:.2%}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Sell (0)', 'Buy (1)']))

print("\nFeature Importance:")
for name, importance in sorted(zip(feature_cols, model.feature_importances_), key=lambda x: -x[1]):
    print(f"  {name:20s}: {importance:.4f}")

# Save model
from src.config import MODELS_DIR
model.save_model(str(MODELS_DIR / "EUR_USD_xgb_simple.json"))
print(f"\n✓ Model saved to {MODELS_DIR / 'EUR_USD_xgb_simple.json'}")

print("\n" + "="*80)
if accuracy > 0.60:
    print("✓ SUCCESS - Model achieves >60% accuracy!")
else:
    print("⚠ Model accuracy below 60% - may need tuning")
print("="*80)
