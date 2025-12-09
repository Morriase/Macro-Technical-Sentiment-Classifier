"""
Implement the Senior Engineer's ZigZag-Based Approach

Key changes:
1. ZigZag-based target (direction + magnitude to next extremum)
2. Simplified features (RSI, MACD difference, candlestick body)
3. Proper normalization to [-1, 1]

Expected improvement: 6.7% → 40% feature correlation
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from src.data_acquisition.kaggle_loader import KaggleFXDataLoader
import talib
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("IMPLEMENTING ZIGZAG-BASED APPROACH")
print("="*80)

# Load M5 data
print("\n[1/5] Loading M5 data...")
loader = KaggleFXDataLoader()
df = loader.load_symbol_data("EURUSD", timeframe="M5")
print(f"  ✓ Loaded {len(df):,} M5 candles")

# Calculate ZigZag extrema
print("\n[2/5] Calculating ZigZag extrema...")

def calculate_zigzag_extrema(df, depth=48, deviation=1, backstep=47):
    """
    Calculate ZigZag extrema (peaks and troughs)
    
    Parameters from the engineer:
    - depth: 48 (4 hours on M5 = 48 bars)
    - deviation: 1 point
    - backstep: 47
    
    Returns DataFrame with extremum column
    """
    df = df.copy()
    
    # Simple ZigZag implementation
    # Find local maxima and minima
    extrema = []
    
    for i in range(depth, len(df) - depth):
        # Check if current bar is a peak
        window_high = df['high'].iloc[i-depth:i+depth+1]
        if df['high'].iloc[i] == window_high.max():
            extrema.append({'index': i, 'price': df['high'].iloc[i], 'type': 'peak'})
        
        # Check if current bar is a trough
        window_low = df['low'].iloc[i-depth:i+depth+1]
        if df['low'].iloc[i] == window_low.min():
            extrema.append({'index': i, 'price': df['low'].iloc[i], 'type': 'trough'})
    
    # Filter extrema by backstep
    filtered_extrema = []
    last_idx = -backstep - 1
    
    for ext in extrema:
        if ext['index'] - last_idx >= backstep:
            filtered_extrema.append(ext)
            last_idx = ext['index']
    
    print(f"  ✓ Found {len(filtered_extrema)} ZigZag extrema")
    
    # Create extremum column (forward-looking for target creation)
    df['extremum_price'] = np.nan
    df['extremum_type'] = None
    
    for ext in filtered_extrema:
        df.loc[df.index[ext['index']], 'extremum_price'] = ext['price']
        df.loc[df.index[ext['index']], 'extremum_type'] = ext['type']
    
    # Forward fill extremum (each bar knows the NEXT extremum)
    df['next_extremum_price'] = df['extremum_price'].bfill()
    df['next_extremum_type'] = df['extremum_type'].bfill()
    
    return df, filtered_extrema

df, extrema = calculate_zigzag_extrema(df)

# Calculate simplified features (engineer's approach)
print("\n[3/5] Calculating simplified features...")

# Feature 1: RSI(12) normalized to [-1, 1]
df['rsi_12'] = talib.RSI(df['close'], timeperiod=12)
df['rsi_norm'] = (df['rsi_12'] - 50.0) / 50.0

# Feature 2: MACD difference (Main - Signal)
macd, macd_signal, macd_hist = talib.MACD(df['close'], 
                                           fastperiod=12, 
                                           slowperiod=48, 
                                           signalperiod=12)
df['macd_diff'] = np.abs(macd - macd_signal)

# Normalize MACD difference to [-1, 1]
macd_diff_mean = df['macd_diff'].mean()
macd_diff_std = df['macd_diff'].std()
df['macd_diff_norm'] = (df['macd_diff'] - macd_diff_mean) / (macd_diff_std * 3)  # 3-sigma range
df['macd_diff_norm'] = df['macd_diff_norm'].clip(-1, 1)

# Feature 3: Candlestick body (Close - Open)
df['candle_body'] = df['close'] - df['open']

# Normalize candlestick body
body_mean = df['candle_body'].mean()
body_std = df['candle_body'].std()
df['candle_body_norm'] = (df['candle_body'] - body_mean) / (body_std * 3)
df['candle_body_norm'] = df['candle_body_norm'].clip(-1, 1)

print(f"  ✓ Calculated 3 normalized features")

# Create ZigZag-based targets
print("\n[4/5] Creating ZigZag-based targets...")

# Target 1: Direction (1 = buy, 0 = sell)
# Positive magnitude = buy, negative = sell
df['target_direction'] = (df['next_extremum_price'] > df['close']).astype(int)

# Target 2: Magnitude (distance to next extremum in pips)
df['target_magnitude'] = (df['next_extremum_price'] - df['close']) * 10000  # Convert to pips

# Normalize magnitude to [-1, 1] for neural network
mag_mean = df['target_magnitude'].mean()
mag_std = df['target_magnitude'].std()
df['target_magnitude_norm'] = (df['target_magnitude'] - mag_mean) / (mag_std * 3)
df['target_magnitude_norm'] = df['target_magnitude_norm'].clip(-1, 1)

# Drop rows without valid targets
df.dropna(subset=['next_extremum_price', 'rsi_norm', 'macd_diff_norm'], inplace=True)

print(f"  ✓ Created targets, {len(df):,} samples ready")
print(f"  Direction distribution: Buy={df['target_direction'].sum():,}, Sell={(1-df['target_direction']).sum():,}")
print(f"  Magnitude range: {df['target_magnitude'].min():.1f} to {df['target_magnitude'].max():.1f} pips")

# Analyze correlations
print("\n[5/5] Analyzing feature-target correlations...")

features = ['rsi_norm', 'macd_diff_norm', 'candle_body_norm']
targets = ['target_direction', 'target_magnitude_norm']

print("\n" + "="*80)
print("FEATURE-TARGET CORRELATIONS (ZigZag-Based)")
print("="*80)

correlation_results = []

for feature in features:
    for target in targets:
        corr = np.corrcoef(df[feature].values, df[target].values)[0, 1]
        correlation_results.append({
            'Feature': feature,
            'Target': target.replace('target_', '').replace('_norm', ''),
            'Correlation': corr,
            'Abs_Correlation': abs(corr)
        })

corr_df = pd.DataFrame(correlation_results)
corr_pivot = corr_df.pivot(index='Feature', columns='Target', values='Correlation')

print("\n" + corr_pivot.to_string())

# Compare with engineer's results
print("\n" + "="*80)
print("COMPARISON WITH ENGINEER'S RESULTS")
print("="*80)

engineer_rsi_dir = 0.40
engineer_rsi_mag = 0.22

your_rsi_dir = corr_pivot.loc['rsi_norm', 'direction']
your_rsi_mag = corr_pivot.loc['rsi_norm', 'magnitude']

print(f"\nRSI → Direction:")
print(f"  Engineer: {engineer_rsi_dir:.3f}")
print(f"  Your:     {your_rsi_dir:.3f}")
print(f"  Ratio:    {abs(your_rsi_dir)/engineer_rsi_dir:.2f}x")

print(f"\nRSI → Magnitude:")
print(f"  Engineer: {engineer_rsi_mag:.3f}")
print(f"  Your:     {your_rsi_mag:.3f}")
print(f"  Ratio:    {abs(your_rsi_mag)/engineer_rsi_mag:.2f}x")

# Visualizations
print("\n" + "="*80)
print("GENERATING VISUALIZATIONS")
print("="*80)

fig, axes = plt.subplots(2, 2, figsize=(18, 14))
fig.suptitle('ZigZag-Based Approach: Feature Analysis', fontsize=16, fontweight='bold')

# Plot 1: ZigZag extrema on price chart
ax1 = axes[0, 0]
sample_range = slice(0, 2000)  # First 2000 bars for visibility
ax1.plot(df.index[sample_range], df['close'].iloc[sample_range], 
         'b-', linewidth=1, alpha=0.5, label='Close Price')

# Mark extrema
extrema_indices = df[df['extremum_price'].notna()].index
extrema_in_range = [idx for idx in extrema_indices if idx in df.index[sample_range]]

for idx in extrema_in_range:
    ext_type = df.loc[idx, 'extremum_type']
    ext_price = df.loc[idx, 'extremum_price']
    color = 'green' if ext_type == 'peak' else 'red'
    marker = 'v' if ext_type == 'peak' else '^'
    ax1.scatter(idx, ext_price, color=color, marker=marker, s=100, zorder=5)

ax1.set_xlabel('Time', fontsize=11)
ax1.set_ylabel('Price', fontsize=11)
ax1.set_title('ZigZag Extrema (First 2000 bars)', fontsize=13, fontweight='bold')
ax1.legend()
ax1.grid(alpha=0.3)

# Plot 2: Feature-Target Correlation Heatmap
ax2 = axes[0, 1]
sns.heatmap(corr_pivot, annot=True, fmt='.3f', cmap='RdYlGn', center=0,
            vmin=-0.5, vmax=0.5, ax=ax2, cbar_kws={'label': 'Correlation'})
ax2.set_title('Feature-Target Correlations', fontsize=13, fontweight='bold')

# Plot 3: RSI vs Target Direction
ax3 = axes[1, 0]
buy_mask = df['target_direction'] == 1
ax3.scatter(df.loc[buy_mask, 'rsi_norm'].iloc[:5000], 
           df.loc[buy_mask, 'target_magnitude'].iloc[:5000],
           alpha=0.3, s=10, c='green', label='Buy')
ax3.scatter(df.loc[~buy_mask, 'rsi_norm'].iloc[:5000], 
           df.loc[~buy_mask, 'target_magnitude'].iloc[:5000],
           alpha=0.3, s=10, c='red', label='Sell')
ax3.set_xlabel('RSI (normalized)', fontsize=11)
ax3.set_ylabel('Magnitude to Next Extremum (pips)', fontsize=11)
ax3.set_title('RSI vs Target (First 5000 samples)', fontsize=13, fontweight='bold')
ax3.legend()
ax3.grid(alpha=0.3)
ax3.axhline(y=0, color='black', linestyle='--', linewidth=1)
ax3.axvline(x=0, color='black', linestyle='--', linewidth=1)

# Plot 4: Target magnitude distribution
ax4 = axes[1, 1]
ax4.hist(df['target_magnitude'], bins=100, alpha=0.7, color='steelblue', edgecolor='black')
ax4.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero')
ax4.set_xlabel('Target Magnitude (pips)', fontsize=11)
ax4.set_ylabel('Frequency', fontsize=11)
ax4.set_title('Distribution of Target Magnitudes', fontsize=13, fontweight='bold')
ax4.legend()
ax4.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('zigzag_approach_analysis.png', dpi=150, bbox_inches='tight')
print("  ✓ Saved: zigzag_approach_analysis.png")

# Save processed data
df_export = df[['close', 'rsi_norm', 'macd_diff_norm', 'candle_body_norm',
                'target_direction', 'target_magnitude', 'target_magnitude_norm',
                'next_extremum_price', 'next_extremum_type']]
df_export.to_csv('zigzag_training_data.csv')
print("  ✓ Saved: zigzag_training_data.csv")

# Final recommendations
print("\n" + "="*80)
print("IMPLEMENTATION RECOMMENDATIONS")
print("="*80)

if abs(your_rsi_dir) >= 0.30:
    print("\n✓ EXCELLENT: RSI correlation is strong (>0.30)")
    print("  Ready to implement in main pipeline")
elif abs(your_rsi_dir) >= 0.20:
    print("\n✓ GOOD: RSI correlation is decent (>0.20)")
    print("  Should improve LSTM performance significantly")
else:
    print("\n⚠️  MODERATE: RSI correlation is lower than expected")
    print("  May need to tune ZigZag parameters")

print(f"\nNext steps:")
print(f"  1. Update main.py to use ZigZag-based targets")
print(f"  2. Simplify features to: RSI(12), MACD_diff, Candle_body")
print(f"  3. Use 40-bar sequences (not 100)")
print(f"  4. Train with dual output: direction + magnitude")
print(f"  5. Expected accuracy: 60-65% (vs current 51%)")

print("\n" + "="*80)
print("✓ ANALYSIS COMPLETE")
print("="*80)
