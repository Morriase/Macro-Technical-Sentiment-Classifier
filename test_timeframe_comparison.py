"""
Timeframe Comparison Test
Compare M5 vs H1 vs H4 to see which gives best feature-target correlations
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from src.data_acquisition.kaggle_loader import KaggleFXDataLoader
from src.feature_engineering.technical_features import TechnicalFeatureEngineer
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("TIMEFRAME COMPARISON TEST")
print("="*80)

loader = KaggleFXDataLoader()
engineer = TechnicalFeatureEngineer()

results_all = []

# Test each timeframe
for timeframe, forward_bars in [
    ("M5", 96),   # 8 hours = 96 × 5min
    ("H1", 8),    # 8 hours = 8 × 1hour
    ("H4", 2),    # 8 hours = 2 × 4hour
]:
    print(f"\n{'='*80}")
    print(f"TESTING {timeframe} TIMEFRAME")
    print(f"{'='*80}")
    
    # Load data
    print(f"\n[1/3] Loading {timeframe} data...")
    df_price = loader.load_symbol_data("EURUSD", timeframe=timeframe)
    print(f"  ✓ Loaded {len(df_price):,} candles")
    print(f"  Date range: {df_price.index[0]} to {df_price.index[-1]}")
    
    # Calculate features
    print(f"\n[2/3] Calculating features...")
    df = engineer.calculate_all_features(df_price.copy())
    print(f"  ✓ Calculated {len(df.columns)} features")
    
    # Create target
    print(f"\n[3/3] Creating target (forward window = {forward_bars} bars = 8 hours)...")
    df['forward_close'] = df['close'].shift(-forward_bars)
    df['forward_return'] = (df['forward_close'] - df['close']) / df['close']
    df['forward_return_pips'] = df['forward_return'] * 10000
    
    # Binary target
    df['target'] = (df['forward_return'] > 0).astype(int)
    
    # Drop NaN
    df.dropna(inplace=True)
    print(f"  ✓ {len(df):,} samples ready")
    
    # Calculate correlations for key features
    print(f"\n  Analyzing correlations...")
    
    exclude_cols = ['open', 'high', 'low', 'close', 'volume', 
                    'forward_close', 'forward_return', 'forward_return_pips', 'target']
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    y = df['target'].values
    
    feature_corrs = []
    for col in feature_cols:
        corr = np.corrcoef(df[col].values, y)[0, 1]
        feature_corrs.append({
            'feature': col,
            'correlation': abs(corr),
            'raw_correlation': corr
        })
    
    corrs_df = pd.DataFrame(feature_corrs).sort_values('correlation', ascending=False)
    
    # Get top 10
    top_10 = corrs_df.head(10)
    
    print(f"\n  Top 10 Features for {timeframe}:")
    for idx, row in top_10.iterrows():
        print(f"    {row['feature']:25s}: {row['raw_correlation']:7.4f}")
    
    # Store results
    results_all.append({
        'timeframe': timeframe,
        'samples': len(df),
        'features': len(feature_cols),
        'best_corr': corrs_df.iloc[0]['correlation'],
        'best_feature': corrs_df.iloc[0]['feature'],
        'mean_corr': corrs_df['correlation'].mean(),
        'top_10_mean': top_10['correlation'].mean(),
        'top_10_features': top_10,
        'all_corrs': corrs_df,
        'forward_return_std': df['forward_return_pips'].std(),
        'forward_return_mean': df['forward_return_pips'].mean(),
    })

# Summary comparison
print("\n" + "="*80)
print("TIMEFRAME COMPARISON SUMMARY")
print("="*80)

summary_df = pd.DataFrame([{
    'Timeframe': r['timeframe'],
    'Samples': r['samples'],
    'Best Correlation': r['best_corr'],
    'Best Feature': r['best_feature'],
    'Top 10 Mean': r['top_10_mean'],
    'All Features Mean': r['mean_corr'],
    'Return Std (pips)': r['forward_return_std'],
} for r in results_all])

print("\n" + summary_df.to_string(index=False))

# Find best timeframe
best_idx = summary_df['Best Correlation'].idxmax()
best_tf = summary_df.iloc[best_idx]['Timeframe']
improvement = summary_df.iloc[best_idx]['Best Correlation'] / summary_df.iloc[0]['Best Correlation']

print("\n" + "="*80)
print("RECOMMENDATION")
print("="*80)
print(f"\nBest timeframe: {best_tf}")
print(f"  Best correlation: {summary_df.iloc[best_idx]['Best Correlation']:.4f}")
print(f"  Improvement over M5: {improvement:.2f}x")
print(f"  Top 10 features mean: {summary_df.iloc[best_idx]['Top 10 Mean']:.4f}")

# Visualizations
print("\n" + "="*80)
print("GENERATING VISUALIZATIONS")
print("="*80)

fig, axes = plt.subplots(2, 2, figsize=(18, 14))
fig.suptitle('Timeframe Comparison: M5 vs H1 vs H4', fontsize=16, fontweight='bold')

# Plot 1: Best correlation by timeframe
ax1 = axes[0, 0]
colors = ['red', 'orange', 'green']
bars = ax1.bar(summary_df['Timeframe'], summary_df['Best Correlation'], 
               color=colors, alpha=0.7)
ax1.set_ylabel('Best Feature Correlation', fontsize=12)
ax1.set_title('Best Feature-Target Correlation by Timeframe', fontsize=13, fontweight='bold')
ax1.grid(axis='y', alpha=0.3)

# Add improvement labels
for i, (idx, row) in enumerate(summary_df.iterrows()):
    if i == 0:
        label = "Baseline"
    else:
        improvement_pct = (row['Best Correlation'] / summary_df.iloc[0]['Best Correlation'] - 1) * 100
        label = f"+{improvement_pct:.0f}%"
    ax1.text(i, row['Best Correlation'] + 0.005, label, 
             ha='center', fontsize=11, fontweight='bold')

# Plot 2: Top 10 mean correlation
ax2 = axes[0, 1]
ax2.bar(summary_df['Timeframe'], summary_df['Top 10 Mean'], 
        color=colors, alpha=0.7)
ax2.set_ylabel('Mean Correlation (Top 10 Features)', fontsize=12)
ax2.set_title('Average Predictive Power of Top Features', fontsize=13, fontweight='bold')
ax2.grid(axis='y', alpha=0.3)

# Add values
for i, (idx, row) in enumerate(summary_df.iterrows()):
    ax2.text(i, row['Top 10 Mean'] + 0.002, f"{row['Top 10 Mean']:.4f}", 
             ha='center', fontsize=10)

# Plot 3: Top 10 features comparison (side by side)
ax3 = axes[1, 0]
x_pos = np.arange(10)
width = 0.25

for i, result in enumerate(results_all):
    top_10 = result['top_10_features']
    offset = (i - 1) * width
    ax3.barh(x_pos + offset, top_10['correlation'].values, 
             width, label=result['timeframe'], alpha=0.7)

ax3.set_yticks(x_pos)
ax3.set_yticklabels(results_all[0]['top_10_features']['feature'].values, fontsize=9)
ax3.set_xlabel('Correlation', fontsize=11)
ax3.set_title('Top 10 Features Comparison', fontsize=13, fontweight='bold')
ax3.legend()
ax3.invert_yaxis()
ax3.grid(axis='x', alpha=0.3)

# Plot 4: Return volatility by timeframe
ax4 = axes[1, 1]
ax4.bar(summary_df['Timeframe'], summary_df['Return Std (pips)'], 
        color=colors, alpha=0.7)
ax4.set_ylabel('8-Hour Return Std Dev (pips)', fontsize=12)
ax4.set_title('Signal Volatility by Timeframe', fontsize=13, fontweight='bold')
ax4.grid(axis='y', alpha=0.3)

# Add values
for i, (idx, row) in enumerate(summary_df.iterrows()):
    ax4.text(i, row['Return Std (pips)'] + 1, f"{row['Return Std (pips)']:.1f}", 
             ha='center', fontsize=10)

plt.tight_layout()
plt.savefig('timeframe_comparison.png', dpi=150, bbox_inches='tight')
print("  ✓ Saved: timeframe_comparison.png")

# Detailed feature comparison
fig2, axes2 = plt.subplots(1, 3, figsize=(20, 8))
fig2.suptitle('Feature Correlation Distribution by Timeframe', fontsize=16, fontweight='bold')

for i, (result, ax) in enumerate(zip(results_all, axes2)):
    all_corrs = result['all_corrs']
    
    ax.hist(all_corrs['raw_correlation'], bins=30, alpha=0.7, 
            color=colors[i], edgecolor='black')
    ax.axvline(x=0, color='red', linestyle='--', linewidth=2)
    ax.axvline(x=all_corrs['raw_correlation'].mean(), color='green', 
               linestyle='--', linewidth=2, 
               label=f"Mean: {all_corrs['raw_correlation'].mean():.4f}")
    ax.set_xlabel('Correlation with Target', fontsize=11)
    ax.set_ylabel('Number of Features', fontsize=11)
    ax.set_title(f'{result["timeframe"]} Timeframe', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('timeframe_correlation_distributions.png', dpi=150, bbox_inches='tight')
print("  ✓ Saved: timeframe_correlation_distributions.png")

# Save detailed results
print("\n" + "="*80)
print("SAVING RESULTS")
print("="*80)

for result in results_all:
    filename = f"correlations_{result['timeframe']}.csv"
    result['all_corrs'].to_csv(filename, index=False)
    print(f"  ✓ Saved: {filename}")

summary_df.to_csv('timeframe_comparison_summary.csv', index=False)
print("  ✓ Saved: timeframe_comparison_summary.csv")

# Final recommendation
print("\n" + "="*80)
print("FINAL RECOMMENDATION")
print("="*80)

if improvement > 1.5:
    print(f"\n✓ STRONG RECOMMENDATION: Switch to {best_tf}")
    print(f"  - {improvement:.1f}x better correlation")
    print(f"  - LSTM should learn much better")
    print(f"  - Expected accuracy improvement: +{(improvement-1)*10:.0f}%")
elif improvement > 1.2:
    print(f"\n✓ MODERATE RECOMMENDATION: Consider switching to {best_tf}")
    print(f"  - {improvement:.1f}x better correlation")
    print(f"  - Modest improvement expected")
    print(f"  - Expected accuracy improvement: +{(improvement-1)*10:.0f}%")
else:
    print(f"\n⚠️  MINIMAL IMPROVEMENT: Stay with M5 for now")
    print(f"  - Only {improvement:.1f}x better correlation")
    print(f"  - Not worth the switch")
    print(f"  - Focus on other improvements (features, target definition)")

print("\n" + "="*80)
print("✓ ANALYSIS COMPLETE")
print("="*80)
print("\nFiles generated:")
print("  - timeframe_comparison.png")
print("  - timeframe_correlation_distributions.png")
print("  - timeframe_comparison_summary.csv")
print("  - correlations_M5.csv")
print("  - correlations_H1.csv")
print("  - correlations_H4.csv")
