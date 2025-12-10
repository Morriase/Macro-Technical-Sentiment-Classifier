"""Debug script to find the WFO data duplication bug"""
import pandas as pd
import numpy as np
from src.data_acquisition.kaggle_loader import KaggleFXDataLoader
from src.config import ZIGZAG_CONFIG
from src.utils.zigzag import calculate_zigzag_extrema, create_zigzag_targets
import talib

print("="*80)
print("DEBUGGING WFO DATA DUPLICATION BUG")
print("="*80)

# Step 1: Load and prepare data exactly like main.py
print("\n[1] Loading data...")
loader = KaggleFXDataLoader()
df = loader.load_symbol_data("EURUSD", timeframe="M5")
print(f"  Original df shape: {df.shape}")
print(f"  Index type: {type(df.index)}")

# Step 2: Engineer features
print("\n[2] Engineering features...")
df['rsi_12'] = talib.RSI(df['close'], timeperiod=12)
df['rsi_norm'] = (df['rsi_12'] - 50.0) / 50.0

macd, macd_signal, _ = talib.MACD(df['close'], fastperiod=12, slowperiod=48, signalperiod=12)
macd_diff = np.abs(macd - macd_signal)
df['macd_diff_norm'] = ((macd_diff - macd_diff.mean()) / (macd_diff.std() * 3)).clip(-1, 1)

candle_body = df['close'] - df['open']
df['candle_body_norm'] = ((candle_body - candle_body.mean()) / (candle_body.std() * 3)).clip(-1, 1)

df['yield_curve'] = 0.0
df['dxy_index'] = 100.0

# Step 3: Create ZigZag targets
print("\n[3] Creating ZigZag targets...")
df = calculate_zigzag_extrema(df, **ZIGZAG_CONFIG)
df = create_zigzag_targets(df, pip_multiplier=10000)
df = df.dropna(subset=['target_direction'])
df['target_class'] = df['target_direction'].astype(int)

print(f"  After feature engineering: {df.shape}")

# Step 4: Simulate walk-forward split
print("\n[4] Simulating walk-forward split...")

dates = pd.to_datetime(df.index)
min_date = dates.min()
max_date = dates.max()
train_end = min_date + pd.DateOffset(months=4)

print(f"  Date range: {min_date} to {max_date}")
print(f"  Train end: {train_end}")

# Create train mask
train_mask = (dates >= min_date) & (dates < train_end)
train_indices = df.index[train_mask]

print(f"\n  train_mask sum: {train_mask.sum()}")
print(f"  train_indices length: {len(train_indices)}")
print(f"  train_indices type: {type(train_indices)}")

# Step 5: Extract data using .loc - THIS IS WHERE THE BUG MIGHT BE
print("\n[5] Extracting data with df.loc...")

feature_columns = ['rsi_norm', 'macd_diff_norm', 'candle_body_norm', 'yield_curve', 'dxy_index']

# Method 1: Using .loc with index
X_train_loc = df.loc[train_indices, feature_columns]
print(f"  df.loc[train_indices, feature_columns].shape: {X_train_loc.shape}")

# Method 2: Using .loc with index and .values
X_train_values = df.loc[train_indices, feature_columns].values
print(f"  df.loc[train_indices, feature_columns].values.shape: {X_train_values.shape}")

# Method 3: Using boolean mask directly
X_train_mask = df.loc[train_mask, feature_columns].values
print(f"  df.loc[train_mask, feature_columns].values.shape: {X_train_mask.shape}")

# Method 4: Using iloc with integer positions
train_positions = np.where(train_mask)[0]
X_train_iloc = df.iloc[train_positions][feature_columns].values
print(f"  df.iloc[positions][feature_columns].values.shape: {X_train_iloc.shape}")

# Check if index has duplicates
print(f"\n[6] Checking for index issues...")
print(f"  Index is unique: {df.index.is_unique}")
print(f"  Index duplicates: {df.index.duplicated().sum()}")

# Check if train_indices has duplicates
print(f"  train_indices is unique: {train_indices.is_unique}")
print(f"  train_indices duplicates: {train_indices.duplicated().sum()}")

# Check the actual values
print(f"\n[7] Checking actual extraction...")
print(f"  Expected rows: {len(train_indices)}")
print(f"  Got rows: {len(X_train_values)}")

if len(X_train_values) != len(train_indices):
    print(f"\n  ⚠ BUG FOUND! Mismatch: {len(X_train_values)} vs {len(train_indices)}")
    print(f"  Ratio: {len(X_train_values) / len(train_indices):.2f}x")
else:
    print(f"\n  ✓ No bug found locally - issue might be Kaggle-specific")
