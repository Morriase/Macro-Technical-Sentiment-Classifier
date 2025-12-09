"""
Feature Correlation Analysis
Analyzes all features (technical, macro, volatility regime) to identify:
1. Features highly correlated with target (predictive power)
2. Features highly correlated with each other (redundancy)
3. Recommendations for feature selection
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from loguru import logger
import warnings
warnings.filterwarnings('ignore')

# Import project modules
from src.data_acquisition.kaggle_loader import KaggleFXDataLoader
from src.feature_engineering.technical_features import TechnicalFeatureEngineer
from src.data_acquisition.fred_macro_loader import FREDMacroLoader
from src.config import DATA_DIR, IS_KAGGLE

print("="*80)
print("FEATURE CORRELATION ANALYSIS")
print("="*80)

# Load data
print("\n[1/5] Loading EUR/USD data...")
loader = KaggleFXDataLoader()
symbol = "EURUSD"

# Load M5 data
df_price = loader.load_symbol_data(symbol, timeframe="M5")
print(f"  ✓ Loaded {len(df_price):,} M5 candles")

# Load higher timeframes for MTF features
higher_timeframes = {}
for tf in ["H1", "H4"]:
    try:
        df_ht = loader.load_symbol_data(symbol, timeframe=tf)
        higher_timeframes[tf] = df_ht
        print(f"  ✓ Loaded {len(df_ht):,} {tf} candles")
    except:
        print(f"  ✗ Could not load {tf} data")

# Calculate features
print("\n[2/5] Calculating technical features...")
engineer = TechnicalFeatureEngineer()
df_features = engineer.calculate_all_features(df_price.copy())
print(f"  ✓ Calculated {len(df_features.columns)} base technical features")

# Add MTF features
if higher_timeframes:
    df_features = engineer.add_multi_timeframe_features(
        df_features, higher_timeframes
    )
    print(f"  ✓ Added MTF features, total: {len(df_features.columns)} features")

# Add FRED macro features
print("\n[3/5] Adding FRED macro features...")
try:
    fred_loader = FREDMacroLoader()
    start_date = df_features.index.min()
    end_date = df_features.index.max()
    
    if isinstance(start_date, pd.Timestamp):
        start_date = start_date.to_pydatetime()
        end_date = end_date.to_pydatetime()
    
    fred_macro_df = fred_loader.get_macro_features_for_pair(
        "EUR_USD", start_date, end_date
    )
    
    if not fred_macro_df.empty:
        # Select key FRED features
        key_fred_features = ['rate_differential', 'vix', 'yield_curve', 'dxy_index', 'oil_price']
        selected_cols = ['date'] + [col for col in key_fred_features if col in fred_macro_df.columns]
        fred_macro_df = fred_macro_df[selected_cols]
        
        # Merge
        original_index_name = df_features.index.name or 'date'
        df_features = df_features.reset_index()
        
        if original_index_name != 'date' and original_index_name in df_features.columns:
            df_features = df_features.rename(columns={original_index_name: 'date'})
        
        df_features['date'] = pd.to_datetime(df_features['date']).dt.date
        fred_macro_df['date'] = pd.to_datetime(fred_macro_df['date']).dt.date
        
        df_features = pd.merge(df_features, fred_macro_df, on='date', how='left')
        
        # Forward-fill macro features
        macro_cols = [col for col in fred_macro_df.columns if col != 'date']
        df_features[macro_cols] = df_features[macro_cols].ffill()
        
        # Set index back
        df_features.set_index('date', inplace=True)
        
        print(f"  ✓ Added {len(macro_cols)} FRED macro features")
    else:
        print("  ✗ No FRED data available")
except Exception as e:
    print(f"  ✗ FRED loading failed: {e}")

# Create target
print("\n[4/5] Creating target variable...")
forward_window = 96  # 8 hours for M5 data
df_features['forward_close'] = df_features['close'].shift(-forward_window)
df_features['forward_return'] = (
    (df_features['forward_close'] - df_features['close']) / df_features['close']
)

# Binary target: Buy (1) if return > 0, Sell (0) if return < 0
df_features['target'] = (df_features['forward_return'] > 0).astype(int)

# Drop NaN rows
df_features.dropna(inplace=True)
print(f"  ✓ Created target, {len(df_features):,} samples ready")

# Define feature groups
print("\n[5/5] Analyzing correlations...")

# Exclude non-feature columns
exclude_cols = ['open', 'high', 'low', 'close', 'volume', 
                'forward_close', 'forward_return', 'target']
feature_cols = [col for col in df_features.columns if col not in exclude_cols]

print(f"  Analyzing {len(feature_cols)} features")

# Categorize features
technical_features = [col for col in feature_cols if any(x in col for x in 
    ['rsi', 'macd', 'ema', 'adx', 'di', 'return'])]
volatility_features = [col for col in feature_cols if any(x in col for x in 
    ['vol_', 'parkinson', 'garman', 'efficiency'])]
macro_features = [col for col in feature_cols if any(x in col for x in 
    ['rate_', 'vix', 'yield', 'dxy', 'oil'])]
mtf_features = [col for col in feature_cols if any(x in col for x in ['_H1', '_H4'])]

print(f"\n  Feature breakdown:")
print(f"    Technical: {len(technical_features)}")
print(f"    Volatility Regime: {len(volatility_features)}")
print(f"    Macro (FRED): {len(macro_features)}")
print(f"    Multi-Timeframe: {len(mtf_features)}")

# Calculate correlations
y = df_features['target'].values

# 1. Feature-Target Correlations
print("\n" + "="*80)
print("FEATURE-TARGET CORRELATIONS")
print("="*80)

target_corrs = []
for col in feature_cols:
    corr = np.corrcoef(df_features[col].values, y)[0, 1]
    target_corrs.append({
        'feature': col,
        'correlation': abs(corr),
        'raw_correlation': corr
    })

target_corrs_df = pd.DataFrame(target_corrs).sort_values('correlation', ascending=False)

print("\nTop 20 Most Predictive Features:")
print(target_corrs_df.head(20).to_string(index=False))

print("\nBottom 10 Least Predictive Features:")
print(target_corrs_df.tail(10).to_string(index=False))

# 2. Feature-Feature Correlations (Redundancy)
print("\n" + "="*80)
print("FEATURE-FEATURE CORRELATIONS (Redundancy Analysis)")
print("="*80)

feature_corr_matrix = df_features[feature_cols].corr()

# Find highly correlated pairs (redundant features)
redundant_pairs = []
for i in range(len(feature_cols)):
    for j in range(i+1, len(feature_cols)):
        corr = feature_corr_matrix.iloc[i, j]
        if abs(corr) > 0.8:  # High correlation threshold
            redundant_pairs.append({
                'feature_1': feature_cols[i],
                'feature_2': feature_cols[j],
                'correlation': corr
            })

if redundant_pairs:
    redundant_df = pd.DataFrame(redundant_pairs).sort_values('correlation', 
                                                              key=abs, ascending=False)
    print(f"\nFound {len(redundant_df)} highly correlated feature pairs (|r| > 0.8):")
    print(redundant_df.head(20).to_string(index=False))
else:
    print("\n✓ No highly redundant features found (all |r| < 0.8)")

# 3. Create Visualizations
print("\n" + "="*80)
print("GENERATING VISUALIZATIONS")
print("="*80)

# Figure 1: Feature-Target Correlations
fig, axes = plt.subplots(2, 2, figsize=(20, 16))
fig.suptitle('Feature Correlation Analysis for EUR/USD Trading Model', 
             fontsize=16, fontweight='bold')

# Plot 1: Top 20 Features
ax1 = axes[0, 0]
top_20 = target_corrs_df.head(20)
colors = ['green' if x > 0 else 'red' for x in top_20['raw_correlation']]
ax1.barh(range(len(top_20)), top_20['correlation'], color=colors, alpha=0.7)
ax1.set_yticks(range(len(top_20)))
ax1.set_yticklabels(top_20['feature'], fontsize=9)
ax1.set_xlabel('Absolute Correlation with Target', fontsize=11)
ax1.set_title('Top 20 Most Predictive Features', fontsize=13, fontweight='bold')
ax1.invert_yaxis()
ax1.grid(axis='x', alpha=0.3)

for i, (idx, row) in enumerate(top_20.iterrows()):
    ax1.text(row['correlation'] + 0.001, i, f"{row['raw_correlation']:.3f}", 
             va='center', fontsize=8)

# Plot 2: Feature Group Comparison
ax2 = axes[0, 1]
group_stats = []
for group_name, group_features in [
    ('Technical', technical_features),
    ('Volatility', volatility_features),
    ('Macro', macro_features),
    ('MTF', mtf_features)
]:
    if group_features:
        group_corrs = target_corrs_df[target_corrs_df['feature'].isin(group_features)]
        group_stats.append({
            'group': group_name,
            'mean_corr': group_corrs['correlation'].mean(),
            'max_corr': group_corrs['correlation'].max(),
            'count': len(group_features)
        })

group_stats_df = pd.DataFrame(group_stats)
x_pos = np.arange(len(group_stats_df))
ax2.bar(x_pos, group_stats_df['mean_corr'], alpha=0.7, color='steelblue', label='Mean')
ax2.bar(x_pos, group_stats_df['max_corr'], alpha=0.4, color='orange', label='Max')
ax2.set_xticks(x_pos)
ax2.set_xticklabels(group_stats_df['group'], fontsize=11)
ax2.set_ylabel('Correlation with Target', fontsize=11)
ax2.set_title('Feature Group Performance', fontsize=13, fontweight='bold')
ax2.legend()
ax2.grid(axis='y', alpha=0.3)

for i, row in group_stats_df.iterrows():
    ax2.text(i, row['max_corr'] + 0.01, f"n={row['count']}", 
             ha='center', fontsize=9)

# Plot 3: Correlation Heatmap (Top 25 Features)
ax3 = axes[1, 0]
top_25_features = target_corrs_df.head(25)['feature'].tolist()
corr_subset = feature_corr_matrix.loc[top_25_features, top_25_features]
sns.heatmap(corr_subset, cmap='RdYlGn', center=0, vmin=-1, vmax=1,
            square=True, linewidths=0.5, cbar_kws={"shrink": 0.8},
            xticklabels=True, yticklabels=True, ax=ax3, annot=False)
ax3.set_title('Feature-Feature Correlation Heatmap (Top 25)', 
              fontsize=13, fontweight='bold')
ax3.set_xticklabels(ax3.get_xticklabels(), rotation=45, ha='right', fontsize=7)
ax3.set_yticklabels(ax3.get_yticklabels(), rotation=0, fontsize=7)

# Plot 4: Distribution of Correlations
ax4 = axes[1, 1]
ax4.hist(target_corrs_df['raw_correlation'], bins=50, alpha=0.7, 
         color='steelblue', edgecolor='black')
ax4.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero')
ax4.axvline(x=target_corrs_df['raw_correlation'].mean(), color='green', 
            linestyle='--', linewidth=2, label=f"Mean: {target_corrs_df['raw_correlation'].mean():.3f}")
ax4.set_xlabel('Correlation with Target', fontsize=11)
ax4.set_ylabel('Number of Features', fontsize=11)
ax4.set_title('Distribution of Feature-Target Correlations', 
              fontsize=13, fontweight='bold')
ax4.legend()
ax4.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('feature_correlation_analysis.png', dpi=150, bbox_inches='tight')
print("  ✓ Saved: feature_correlation_analysis.png")

# Save results
target_corrs_df.to_csv('feature_target_correlations.csv', index=False)
print("  ✓ Saved: feature_target_correlations.csv")

if redundant_pairs:
    redundant_df.to_csv('redundant_features.csv', index=False)
    print("  ✓ Saved: redundant_features.csv")

print("\n" + "="*80)
print("✓ ANALYSIS COMPLETE")
print("="*80)
print("\nRecommendations:")
print("  1. Focus on top 20 features (highest correlation)")
print("  2. Remove redundant features (|r| > 0.8 with each other)")
print("  3. Check if macro features are working (should have >0 correlation)")
print("\nFiles generated:")
print("  - feature_correlation_analysis.png")
print("  - feature_target_correlations.csv")
if redundant_pairs:
    print("  - redundant_features.csv")
