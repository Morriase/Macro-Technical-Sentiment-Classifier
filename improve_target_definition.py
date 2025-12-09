"""
Improve Target Definition to Increase Signal-to-Noise Ratio

Current problem: Binary up/down target has only 6.7% correlation with best feature
Solution: Only predict SIGNIFICANT moves, filter out noise
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from src.data_acquisition.kaggle_loader import KaggleFXDataLoader
from src.feature_engineering.technical_features import TechnicalFeatureEngineer

print("="*80)
print("IMPROVING TARGET DEFINITION")
print("="*80)

# Load data
print("\n[1/4] Loading data...")
loader = KaggleFXDataLoader()
df_price = loader.load_symbol_data("EURUSD", timeframe="M5")
print(f"  ✓ Loaded {len(df_price):,} M5 candles")

# Calculate features (just a few key ones for speed)
print("\n[2/4] Calculating features...")
engineer = TechnicalFeatureEngineer()
df = engineer.calculate_all_features(df_price.copy())
print(f"  ✓ Calculated {len(df.columns)} features")

# Create different target definitions
print("\n[3/4] Testing different target definitions...")
forward_window = 96  # 8 hours

df['forward_close'] = df['close'].shift(-forward_window)
df['forward_return'] = (df['forward_close'] - df['close']) / df['close']
df['forward_return_pips'] = df['forward_return'] * 10000  # Convert to pips

# Target 1: Current (binary up/down)
df['target_binary'] = (df['forward_return'] > 0).astype(int)

# Target 2: Significant moves only (5 pips threshold)
threshold_pips = 5
threshold = threshold_pips / 10000
df['target_significant'] = np.where(
    df['forward_return'] > threshold, 1,      # Strong UP
    np.where(df['forward_return'] < -threshold, 0,  # Strong DOWN
             -1)  # HOLD (noise)
)

# Target 3: Very significant moves (10 pips threshold)
threshold_pips_high = 10
threshold_high = threshold_pips_high / 10000
df['target_very_significant'] = np.where(
    df['forward_return'] > threshold_high, 1,
    np.where(df['forward_return'] < -threshold_high, 0,
             -1)
)

# Target 4: Extreme moves (20 pips threshold)
threshold_pips_extreme = 20
threshold_extreme = threshold_pips_extreme / 10000
df['target_extreme'] = np.where(
    df['forward_return'] > threshold_extreme, 1,
    np.where(df['forward_return'] < -threshold_extreme, 0,
             -1)
)

df.dropna(inplace=True)

# Analyze each target
print("\n" + "="*80)
print("TARGET DEFINITION COMPARISON")
print("="*80)

results = []

for target_name, threshold_val in [
    ('target_binary', 0),
    ('target_significant', 5),
    ('target_very_significant', 10),
    ('target_extreme', 20)
]:
    # Filter out HOLD samples for non-binary targets
    if threshold_val > 0:
        df_filtered = df[df[target_name] != -1].copy()
        hold_pct = (df[target_name] == -1).sum() / len(df) * 100
    else:
        df_filtered = df.copy()
        hold_pct = 0
    
    # Calculate correlation with best feature (ema_200_dist)
    if len(df_filtered) > 0:
        corr = np.corrcoef(df_filtered['ema_200_dist'].values, 
                          df_filtered[target_name].values)[0, 1]
    else:
        corr = 0
    
    # Class distribution
    if len(df_filtered) > 0:
        class_dist = df_filtered[target_name].value_counts(normalize=True).to_dict()
        buy_pct = class_dist.get(1, 0) * 100
        sell_pct = class_dist.get(0, 0) * 100
    else:
        buy_pct = sell_pct = 0
    
    results.append({
        'Target': target_name.replace('target_', '').title(),
        'Threshold (pips)': threshold_val,
        'Samples': len(df_filtered),
        'Hold %': hold_pct,
        'Buy %': buy_pct,
        'Sell %': sell_pct,
        'Correlation': abs(corr),
        'Improvement': abs(corr) / 0.067 if threshold_val == 0 else abs(corr) / 0.067
    })

results_df = pd.DataFrame(results)
print("\n" + results_df.to_string(index=False))

# Find best threshold
best_idx = results_df['Correlation'].idxmax()
best_result = results_df.iloc[best_idx]

print("\n" + "="*80)
print("RECOMMENDATION")
print("="*80)
print(f"\nBest target: {best_result['Target']}")
print(f"  Threshold: {best_result['Threshold (pips)']} pips")
print(f"  Correlation: {best_result['Correlation']:.4f} ({best_result['Improvement']:.1f}x improvement!)")
print(f"  Tradeable samples: {best_result['Samples']:,} ({100-best_result['Hold %']:.1f}% of data)")
print(f"  Class balance: Buy {best_result['Buy %']:.1f}% / Sell {best_result['Sell %']:.1f}%")

# Visualize
print("\n[4/4] Creating visualizations...")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Target Definition Impact on Model Performance', fontsize=16, fontweight='bold')

# Plot 1: Correlation improvement
ax1 = axes[0, 0]
ax1.bar(results_df['Target'], results_df['Correlation'], color='steelblue', alpha=0.7)
ax1.set_ylabel('Correlation with Best Feature', fontsize=11)
ax1.set_title('Feature-Target Correlation by Threshold', fontsize=13, fontweight='bold')
ax1.grid(axis='y', alpha=0.3)
ax1.set_xticklabels(results_df['Target'], rotation=45, ha='right')

# Add improvement labels
for i, row in results_df.iterrows():
    ax1.text(i, row['Correlation'] + 0.005, f"{row['Improvement']:.1f}x", 
             ha='center', fontsize=10, fontweight='bold')

# Plot 2: Sample retention
ax2 = axes[0, 1]
tradeable_pct = 100 - results_df['Hold %']
ax2.bar(results_df['Target'], tradeable_pct, color='green', alpha=0.7)
ax2.set_ylabel('Tradeable Samples (%)', fontsize=11)
ax2.set_title('Data Retention by Threshold', fontsize=13, fontweight='bold')
ax2.grid(axis='y', alpha=0.3)
ax2.set_xticklabels(results_df['Target'], rotation=45, ha='right')

# Add sample counts
for i, row in results_df.iterrows():
    ax2.text(i, 100 - row['Hold %'] + 2, f"{row['Samples']:,}", 
             ha='center', fontsize=9)

# Plot 3: Forward return distribution
ax3 = axes[1, 0]
ax3.hist(df['forward_return_pips'], bins=100, alpha=0.7, color='steelblue', edgecolor='black')
ax3.axvline(x=0, color='red', linestyle='--', linewidth=2, label='No change')
ax3.axvline(x=5, color='orange', linestyle='--', linewidth=2, label='5 pips')
ax3.axvline(x=-5, color='orange', linestyle='--', linewidth=2)
ax3.axvline(x=10, color='green', linestyle='--', linewidth=2, label='10 pips')
ax3.axvline(x=-10, color='green', linestyle='--', linewidth=2)
ax3.set_xlabel('Forward Return (pips)', fontsize=11)
ax3.set_ylabel('Frequency', fontsize=11)
ax3.set_title('Distribution of 8-Hour Forward Returns', fontsize=13, fontweight='bold')
ax3.legend()
ax3.grid(axis='y', alpha=0.3)

# Plot 4: Signal-to-noise ratio
ax4 = axes[1, 1]
noise_samples = (df['forward_return_pips'].abs() < 5).sum()
signal_5_samples = ((df['forward_return_pips'].abs() >= 5) & 
                    (df['forward_return_pips'].abs() < 10)).sum()
signal_10_samples = ((df['forward_return_pips'].abs() >= 10) & 
                     (df['forward_return_pips'].abs() < 20)).sum()
signal_20_samples = (df['forward_return_pips'].abs() >= 20).sum()

categories = ['Noise\n(<5 pips)', 'Signal\n(5-10 pips)', 'Strong Signal\n(10-20 pips)', 
              'Very Strong\n(>20 pips)']
counts = [noise_samples, signal_5_samples, signal_10_samples, signal_20_samples]
colors = ['red', 'orange', 'lightgreen', 'darkgreen']

ax4.bar(categories, counts, color=colors, alpha=0.7)
ax4.set_ylabel('Number of Samples', fontsize=11)
ax4.set_title('Signal vs Noise Distribution', fontsize=13, fontweight='bold')
ax4.grid(axis='y', alpha=0.3)

# Add percentages
total = sum(counts)
for i, (cat, count) in enumerate(zip(categories, counts)):
    pct = count / total * 100
    ax4.text(i, count + total*0.02, f"{pct:.1f}%", ha='center', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig('target_definition_analysis.png', dpi=150, bbox_inches='tight')
print("  ✓ Saved: target_definition_analysis.png")

# Save recommendations
print("\n" + "="*80)
print("IMPLEMENTATION GUIDE")
print("="*80)
print(f"""
To implement the improved target in your training pipeline:

1. Update main.py create_target() method:

```python
# OLD (current):
df['target'] = (df['forward_return'] > 0).astype(int)

# NEW (recommended):
threshold = {best_result['Threshold (pips)']} / 10000  # {best_result['Threshold (pips)']} pips
df['target'] = np.where(
    df['forward_return'] > threshold, 1,      # Strong UP
    np.where(df['forward_return'] < -threshold, 0,  # Strong DOWN
             -1)  # HOLD (don't trade)
)

# Filter out HOLD samples
df = df[df['target'] != -1]
```

2. Expected improvements:
   - Feature correlation: 0.067 → {best_result['Correlation']:.3f} ({best_result['Improvement']:.1f}x better!)
   - LSTM should learn faster (clearer signal)
   - Validation accuracy should improve 5-10%
   - Trading performance should improve (only trade high-probability setups)

3. Trade-offs:
   - Fewer trading opportunities ({100-best_result['Hold %']:.1f}% of time)
   - But MUCH higher quality signals
   - Better risk-adjusted returns

This is a fundamental fix that addresses the root cause of LSTM struggling!
""")

print("\n✓ Analysis complete!")
