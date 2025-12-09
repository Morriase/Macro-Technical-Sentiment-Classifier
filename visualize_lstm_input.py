"""
Visualize what the LSTM actually receives as input
Shows how regime features flow through time in sequences
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from src.feature_engineering.technical_features import TechnicalFeatureEngineer

# Set style
sns.set_style("darkgrid")
plt.rcParams['figure.figsize'] = (16, 12)

# Generate sample data with regime change
np.random.seed(42)
n = 500  # More data to survive dropna()

dates = pd.date_range(start='2024-01-01', periods=n, freq='4h')

# Simulate regime change at t=300
# First half: Low volatility, choppy
# Second half: High volatility, trending
close = np.zeros(n)
close[0] = 100

for i in range(1, n):
    if i < 300:
        # Low vol regime: small random moves
        close[i] = close[i-1] + np.random.randn() * 0.2
    else:
        # High vol regime: trending with larger moves
        close[i] = close[i-1] + 0.3 + np.random.randn() * 0.8

high = close + np.abs(np.random.randn(n) * 0.5)
low = close - np.abs(np.random.randn(n) * 0.5)
open_ = close + np.random.randn(n) * 0.3
volume = np.random.randint(1000, 10000, n)

df = pd.DataFrame({
    'open': open_,
    'high': high,
    'low': low,
    'close': close,
    'volume': volume
}, index=dates)

# Calculate features
engineer = TechnicalFeatureEngineer()
df_features = engineer.calculate_all_features(df)

print("="*80)
print("VISUALIZING LSTM INPUT: What the LSTM Actually Sees")
print("="*80)
print(f"\nData shape: {df_features.shape}")
print(f"Sequence length: 30 timesteps")
print(f"Features per timestep: {df_features.shape[1]}")

# Select a sequence from available data
sequence_length = min(30, len(df_features))  # Use available data
if len(df_features) < sequence_length:
    print(f"\nWARNING: Only {len(df_features)} rows after dropna(), need at least {sequence_length}")
    print("Using all available data...")
    sequence_data = df_features
else:
    # Take the last 30 timesteps
    sequence_data = df_features.iloc[-sequence_length:]

print(f"\nShowing sequence: {len(sequence_data)} timesteps")
print(f"From {sequence_data.index[0]} to {sequence_data.index[-1]}")

# Create visualization
fig, axes = plt.subplots(5, 1, figsize=(16, 14))
fig.suptitle('LSTM Input Sequence: How Regime Features Flow Through Time', 
             fontsize=16, fontweight='bold')

# Plot 1: Price and RSI
ax1 = axes[0]
ax1_twin = ax1.twinx()
ax1.plot(sequence_data.index, sequence_data['close'], 'b-', linewidth=2, label='Close Price')
ax1_twin.plot(sequence_data.index, sequence_data['rsi'], 'r-', linewidth=2, label='RSI')
ax1_twin.axhline(y=30, color='r', linestyle='--', alpha=0.5, label='RSI Oversold')
ax1_twin.axhline(y=70, color='r', linestyle='--', alpha=0.5, label='RSI Overbought')
ax1.set_ylabel('Close Price', color='b', fontsize=12)
ax1_twin.set_ylabel('RSI', color='r', fontsize=12)
ax1.set_title('Price & RSI Over Time', fontsize=14, fontweight='bold')
ax1.legend(loc='upper left')
ax1_twin.legend(loc='upper right')
ax1.grid(True, alpha=0.3)

# Plot 2: Volatility Regime (categorical)
ax2 = axes[1]
regime_colors = {-1: 'green', 0: 'yellow', 1: 'red'}
regime_labels = {-1: 'Low Vol', 0: 'Medium Vol', 1: 'High Vol'}
for regime_val, color in regime_colors.items():
    mask = sequence_data['vol_regime'] == regime_val
    if mask.any():
        ax2.scatter(sequence_data.index[mask], 
                   sequence_data['vol_regime'][mask],
                   c=color, s=100, label=regime_labels[regime_val], alpha=0.7)
ax2.set_ylabel('Volatility Regime', fontsize=12)
ax2.set_yticks([-1, 0, 1])
ax2.set_yticklabels(['Low (-1)', 'Medium (0)', 'High (1)'])
ax2.set_title('Volatility Regime Classification (LSTM sees this at every timestep!)', 
              fontsize=14, fontweight='bold')
ax2.legend(loc='upper left')
ax2.grid(True, alpha=0.3)
# Mark regime changes if any exist
if len(sequence_data) > 1:
    regime_changes = sequence_data['vol_regime'].diff() != 0
    if regime_changes.any():
        change_idx = sequence_data.index[regime_changes][0] if len(sequence_data.index[regime_changes]) > 0 else None
        if change_idx:
            ax2.axvline(x=change_idx, color='purple', linestyle='--', 
                       linewidth=2, alpha=0.7, label='Regime Change')

# Plot 3: Volatility Percentile (continuous)
ax3 = axes[2]
ax3.plot(sequence_data.index, sequence_data['vol_percentile'], 
         'purple', linewidth=2, label='Vol Percentile')
ax3.axhline(y=0.33, color='green', linestyle='--', alpha=0.5, label='Low Vol Threshold')
ax3.axhline(y=0.67, color='red', linestyle='--', alpha=0.5, label='High Vol Threshold')
ax3.fill_between(sequence_data.index, 0, 0.33, alpha=0.2, color='green')
ax3.fill_between(sequence_data.index, 0.33, 0.67, alpha=0.2, color='yellow')
ax3.fill_between(sequence_data.index, 0.67, 1, alpha=0.2, color='red')
ax3.set_ylabel('Vol Percentile', fontsize=12)
ax3.set_ylim(0, 1)
ax3.set_title('Volatility Percentile (LSTM sees vol level changing over time)', 
              fontsize=14, fontweight='bold')
ax3.legend(loc='upper left')
ax3.grid(True, alpha=0.3)

# Plot 4: Efficiency Ratio (trending vs choppy)
ax4 = axes[3]
ax4.plot(sequence_data.index, sequence_data['efficiency_ratio'], 
         'orange', linewidth=2, label='Efficiency Ratio')
ax4.axhline(y=0.3, color='red', linestyle='--', alpha=0.5, label='Choppy Threshold')
ax4.axhline(y=0.7, color='green', linestyle='--', alpha=0.5, label='Trending Threshold')
ax4.fill_between(sequence_data.index, 0, 0.3, alpha=0.2, color='red', label='Choppy')
ax4.fill_between(sequence_data.index, 0.3, 0.7, alpha=0.2, color='yellow', label='Mixed')
ax4.fill_between(sequence_data.index, 0.7, 1, alpha=0.2, color='green', label='Trending')
ax4.set_ylabel('Efficiency Ratio', fontsize=12)
ax4.set_ylim(0, 1)
ax4.set_title('Price Efficiency Ratio (LSTM sees market transitioning from choppy to trending)', 
              fontsize=14, fontweight='bold')
ax4.legend(loc='upper left')
ax4.grid(True, alpha=0.3)

# Plot 5: Vol-Adjusted Momentum
ax5 = axes[4]
ax5.plot(sequence_data.index, sequence_data['vol_adj_momentum'], 
         'cyan', linewidth=2, label='Vol-Adjusted Momentum')
ax5.axhline(y=0, color='black', linestyle='-', alpha=0.5)
ax5.fill_between(sequence_data.index, 0, sequence_data['vol_adj_momentum'], 
                 where=sequence_data['vol_adj_momentum']>=0, alpha=0.3, color='green', label='Positive')
ax5.fill_between(sequence_data.index, 0, sequence_data['vol_adj_momentum'], 
                 where=sequence_data['vol_adj_momentum']<0, alpha=0.3, color='red', label='Negative')
ax5.set_ylabel('Vol-Adj Momentum', fontsize=12)
ax5.set_xlabel('Time', fontsize=12)
ax5.set_title('Volatility-Adjusted Momentum (LSTM sees signal quality changing)', 
              fontsize=14, fontweight='bold')
ax5.legend(loc='upper left')
ax5.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('lstm_input_visualization.png', dpi=150, bbox_inches='tight')
print(f"\n✓ Saved visualization to: lstm_input_visualization.png")

# Now show what LSTM actually processes
print("\n" + "="*80)
print("LSTM SEQUENCE PROCESSING")
print("="*80)

# Show 3 example timesteps from the sequence
seq_len = len(sequence_data)
example_timesteps = [0, seq_len//2, seq_len-1]  # Start, middle, end
feature_subset = ['rsi', 'macd_diff', 'vol_regime', 'vol_percentile', 
                  'efficiency_ratio', 'vol_adj_momentum']

print("\nLSTM receives a 3D tensor: [batch_size, sequence_length, num_features]")
print(f"For one sample: [1, 30, {df_features.shape[1]}]")
print("\nHere are 3 timesteps from the sequence:\n")

for i, ts in enumerate(example_timesteps):
    row = sequence_data.iloc[ts]
    print(f"Timestep t-{seq_len-1-ts} (position {ts} in sequence):")
    print(f"  Date: {row.name}")
    for feat in feature_subset:
        value = row[feat]
        print(f"    {feat:20s}: {value:8.4f}")
    
    # Interpretation
    if ts == 0:
        print(f"  → Interpretation: Start of sequence, establishing baseline")
    elif ts == seq_len//2:
        print(f"  → Interpretation: Middle of sequence, checking for regime changes")
    else:
        print(f"  → Interpretation: Current timestep (most recent)")
    print()

# Show the full sequence shape
print("="*80)
print("WHAT LSTM ACTUALLY PROCESSES")
print("="*80)

# Build the sequence display dynamically
seq_len = len(sequence_data)
print(f"""
The LSTM receives this 3D structure:

Shape: [batch_size=1, sequence_length={seq_len}, num_features={df_features.shape[1]}]

Timestep t-{seq_len-1}: [{', '.join([f'{v:.2f}' for v in sequence_data.iloc[0][feature_subset].values])}]
Timestep t-{seq_len-2}: [{', '.join([f'{v:.2f}' for v in sequence_data.iloc[1][feature_subset].values])}]
...
Timestep t-{seq_len//2}: [{', '.join([f'{v:.2f}' for v in sequence_data.iloc[seq_len//2][feature_subset].values])}]
...
Timestep t-1:  [{', '.join([f'{v:.2f}' for v in sequence_data.iloc[-2][feature_subset].values])}]
Timestep t:    [{', '.join([f'{v:.2f}' for v in sequence_data.iloc[-1][feature_subset].values])}]  ← Current

The LSTM hidden state "remembers":
1. Historical regime states across all {seq_len} timesteps
2. Regime transitions (when vol_regime changes)
3. Feature evolution patterns (efficiency, vol percentile, etc.)
4. Pattern: "Regime dynamics + price action = trading signal"

Output: Probability of UP/DOWN at t+1
""")

print("="*80)
print("KEY INSIGHT")
print("="*80)
print("""
The volatility regime features are NOT static labels!

They are TIME-VARYING features that:
✓ Change over time (regime transitions)
✓ Flow through the LSTM sequence
✓ Provide temporal context for price movements
✓ Help LSTM learn regime-conditional patterns

Example pattern LSTM learns:
"When I see vol_regime transition from 0→1 AND efficiency_ratio rising from 0.2→0.7
 over 15 timesteps, this is a BREAKOUT pattern → Follow momentum"

This is EXACTLY what you want for adaptive trading!
""")

# Create a heatmap showing feature evolution
print("\nCreating feature evolution heatmap...")
fig, ax = plt.subplots(figsize=(16, 8))

# Select key features for heatmap
heatmap_features = ['rsi', 'macd_diff', 'vol_regime', 'vol_percentile', 
                    'efficiency_ratio', 'vol_adj_momentum', 'vol_trend', 'vol_breakout']
heatmap_data = sequence_data[heatmap_features].T

# Normalize each feature to 0-1 for visualization
heatmap_normalized = (heatmap_data - heatmap_data.min(axis=1).values.reshape(-1, 1)) / \
                     (heatmap_data.max(axis=1).values.reshape(-1, 1) - 
                      heatmap_data.min(axis=1).values.reshape(-1, 1) + 1e-8)

seq_len = len(sequence_data)
sns.heatmap(heatmap_normalized, cmap='RdYlGn', center=0.5, 
            xticklabels=[f't-{seq_len-1-i}' for i in range(seq_len)],
            yticklabels=heatmap_features,
            cbar_kws={'label': 'Normalized Value (0=min, 1=max)'},
            ax=ax)

ax.set_title(f'Feature Evolution Over LSTM Sequence ({seq_len} timesteps)\nLSTM sees ALL features changing over time', 
             fontsize=14, fontweight='bold')
ax.set_xlabel('Timestep in Sequence', fontsize=12)
ax.set_ylabel('Features', fontsize=12)

# Mark regime changes if any
if len(sequence_data) > 1:
    regime_changes = sequence_data['vol_regime'].diff() != 0
    if regime_changes.any():
        for idx, is_change in enumerate(regime_changes):
            if is_change:
                ax.axvline(x=idx, color='purple', linewidth=2, linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig('lstm_feature_heatmap.png', dpi=150, bbox_inches='tight')
print(f"✓ Saved heatmap to: lstm_feature_heatmap.png")

print("\n" + "="*80)
print("✓ VISUALIZATION COMPLETE")
print("="*80)
print("\nGenerated files:")
print("  1. lstm_input_visualization.png - Time series plots showing regime evolution")
print("  2. lstm_feature_heatmap.png - Heatmap showing all features over sequence")
print("\nThese show exactly what the LSTM sees: features flowing through time!")
