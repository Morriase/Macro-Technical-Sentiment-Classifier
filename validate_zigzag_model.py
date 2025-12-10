"""
Proper validation of ZigZag model with convergence analysis.
Shows train/val/test accuracy and loss curves to detect overfitting.
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
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report, log_loss

# Configure logging
logger.remove()
logger.add(sys.stderr, level="INFO", format="{time:HH:mm:ss} | {level} | {message}")

print("="*80)
print("ZIGZAG MODEL VALIDATION - Convergence Analysis")
print("="*80)

# Step 1: Load data
print("\n[1/7] Loading data...")
from src.data_acquisition.kaggle_loader import KaggleFXDataLoader
from src.config import ZIGZAG_CONFIG

loader = KaggleFXDataLoader()
df = loader.load_symbol_data("EURUSD", timeframe="M5")
print(f"  ✓ Loaded {len(df):,} bars")
print(f"  Date range: {df.index.min()} to {df.index.max()}")

# Step 2: Engineer features
print("\n[2/7] Engineering features...")

df['rsi_12'] = talib.RSI(df['close'], timeperiod=12)
df['rsi_norm'] = (df['rsi_12'] - 50.0) / 50.0

macd, macd_signal, _ = talib.MACD(df['close'], fastperiod=12, slowperiod=48, signalperiod=12)
macd_diff = np.abs(macd - macd_signal)
df['macd_diff_norm'] = ((macd_diff - macd_diff.mean()) / (macd_diff.std() * 3)).clip(-1, 1)

candle_body = df['close'] - df['open']
df['candle_body_norm'] = ((candle_body - candle_body.mean()) / (candle_body.std() * 3)).clip(-1, 1)

df['yield_curve'] = 0.0
df['dxy_index'] = 100.0

print(f"  ✓ Calculated 5 features")

# Step 3: Create ZigZag targets
print("\n[3/7] Creating ZigZag targets...")
from src.utils.zigzag import calculate_zigzag_extrema, create_zigzag_targets

df = calculate_zigzag_extrema(df, **ZIGZAG_CONFIG)
df = create_zigzag_targets(df, pip_multiplier=10000)
df = df.dropna(subset=['target_direction'])
df['target'] = df['target_direction'].astype(int)

print(f"  ✓ Created targets, {len(df):,} samples ready")

# Step 4: Time-series split (CRITICAL: no shuffling, no leakage)
print("\n[4/7] Creating time-series splits...")

feature_cols = ['rsi_norm', 'macd_diff_norm', 'candle_body_norm', 'yield_curve', 'dxy_index']
X = df[feature_cols].values
y = df['target'].values

# 70% train, 15% val, 15% test (time-ordered)
n = len(X)
train_end = int(n * 0.70)
val_end = int(n * 0.85)

X_train, y_train = X[:train_end], y[:train_end]
X_val, y_val = X[train_end:val_end], y[train_end:val_end]
X_test, y_test = X[val_end:], y[val_end:]

print(f"  Train: {len(X_train):,} samples ({df.index[0]} to {df.index[train_end-1]})")
print(f"  Val:   {len(X_val):,} samples ({df.index[train_end]} to {df.index[val_end-1]})")
print(f"  Test:  {len(X_test):,} samples ({df.index[val_end]} to {df.index[-1]})")

# Check class balance in each split
print(f"\n  Class balance:")
print(f"    Train: {y_train.mean()*100:.1f}% Buy")
print(f"    Val:   {y_val.mean()*100:.1f}% Buy")
print(f"    Test:  {y_test.mean()*100:.1f}% Buy")

# Step 5: Train with evaluation logging
print("\n[5/7] Training XGBoost with convergence tracking...")

# Store evaluation results
evals_result = {}

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
)

model.fit(
    X_train, y_train,
    eval_set=[(X_train, y_train), (X_val, y_val)],
    verbose=False
)

# Get evaluation results
train_logloss = model.evals_result()['validation_0']['logloss']
val_logloss = model.evals_result()['validation_1']['logloss']

print(f"  ✓ Training complete after {len(train_logloss)} rounds")
print(f"  Best iteration: {model.best_iteration}")

# Step 6: Evaluate on all splits
print("\n[6/7] Evaluating model...")

# Predictions
y_train_pred = model.predict(X_train)
y_val_pred = model.predict(X_val)
y_test_pred = model.predict(X_test)

y_train_proba = model.predict_proba(X_train)
y_val_proba = model.predict_proba(X_val)
y_test_proba = model.predict_proba(X_test)

# Accuracy
train_acc = accuracy_score(y_train, y_train_pred)
val_acc = accuracy_score(y_val, y_val_pred)
test_acc = accuracy_score(y_test, y_test_pred)

# Log loss
train_loss = log_loss(y_train, y_train_proba)
val_loss = log_loss(y_val, y_val_proba)
test_loss = log_loss(y_test, y_test_proba)

print(f"\n  ACCURACY:")
print(f"    Train: {train_acc:.2%}")
print(f"    Val:   {val_acc:.2%}")
print(f"    Test:  {test_acc:.2%}")

print(f"\n  LOG LOSS:")
print(f"    Train: {train_loss:.4f}")
print(f"    Val:   {val_loss:.4f}")
print(f"    Test:  {test_loss:.4f}")

# Overfitting check
overfit_gap = train_acc - val_acc
print(f"\n  OVERFITTING CHECK:")
print(f"    Train-Val gap: {overfit_gap:.2%}")
if overfit_gap > 0.05:
    print(f"    ⚠ WARNING: Possible overfitting (gap > 5%)")
elif overfit_gap < 0.02:
    print(f"    ✓ Good generalization (gap < 2%)")
else:
    print(f"    ~ Acceptable (gap 2-5%)")

# Step 7: Plot convergence
print("\n[7/7] Generating convergence plot...")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Loss curve
ax1 = axes[0]
ax1.plot(train_logloss, label='Train Loss', color='blue', alpha=0.7)
ax1.plot(val_logloss, label='Val Loss', color='orange', alpha=0.7)
ax1.axvline(x=model.best_iteration, color='red', linestyle='--', label=f'Best iter ({model.best_iteration})')
ax1.set_xlabel('Boosting Round')
ax1.set_ylabel('Log Loss')
ax1.set_title('Training Convergence')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Final metrics bar chart
ax2 = axes[1]
metrics = ['Train', 'Val', 'Test']
accuracies = [train_acc, val_acc, test_acc]
colors = ['blue', 'orange', 'green']
bars = ax2.bar(metrics, accuracies, color=colors, alpha=0.7)
ax2.set_ylabel('Accuracy')
ax2.set_title('Accuracy by Split')
ax2.set_ylim(0.5, 0.8)
ax2.axhline(y=0.65, color='red', linestyle='--', label='Engineer baseline (65%)')
ax2.legend()

# Add value labels on bars
for bar, acc in zip(bars, accuracies):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
             f'{acc:.1%}', ha='center', va='bottom', fontsize=12)

plt.tight_layout()
plt.savefig('zigzag_convergence_analysis.png', dpi=150)
print(f"  ✓ Saved plot to zigzag_convergence_analysis.png")

# Classification report for test set
print("\n" + "="*80)
print("TEST SET CLASSIFICATION REPORT")
print("="*80)
print(classification_report(y_test, y_test_pred, target_names=['Sell (0)', 'Buy (1)']))

# Feature importance
print("\n" + "="*80)
print("FEATURE IMPORTANCE")
print("="*80)
for name, importance in sorted(zip(feature_cols, model.feature_importances_), key=lambda x: -x[1]):
    print(f"  {name:20s}: {importance:.4f}")

# Summary
print("\n" + "="*80)
print("VALIDATION SUMMARY")
print("="*80)
print(f"""
  Data:
    Total samples: {len(df):,}
    Train/Val/Test: {len(X_train):,} / {len(X_val):,} / {len(X_test):,}
    
  Results:
    Train Accuracy: {train_acc:.2%}
    Val Accuracy:   {val_acc:.2%}
    Test Accuracy:  {test_acc:.2%}
    
  Convergence:
    Best iteration: {model.best_iteration} / 500
    Final train loss: {train_logloss[-1]:.4f}
    Final val loss:   {val_logloss[-1]:.4f}
    
  Overfitting:
    Train-Val gap: {overfit_gap:.2%}
    Train-Test gap: {train_acc - test_acc:.2%}
""")

if test_acc >= 0.65 and overfit_gap < 0.05:
    print("  ✓ MODEL VALIDATED - Good accuracy with acceptable generalization")
elif test_acc >= 0.60:
    print("  ~ MODEL ACCEPTABLE - Meets minimum threshold")
else:
    print("  ⚠ MODEL NEEDS IMPROVEMENT")

print("="*80)
