"""
Test if macro features improve ZigZag-based predictions
Compare: Engineer's 3 features vs Engineer's 3 + Macros
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from src.data_acquisition.kaggle_loader import KaggleFXDataLoader
from src.data_acquisition.fred_macro_loader import FREDMacroLoader
import talib
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("TESTING MACROS WITH ZIGZAG APPROACH")
print("="*80)

# Load the ZigZag training data we just created
print("\n[1/3] Loading ZigZag training data...")
df = pd.read_csv('zigzag_training_data.csv', index_col=0, parse_dates=True)
print(f"  ✓ Loaded {len(df):,} samples")

# Add macro features
print("\n[2/3] Adding macro features...")
try:
    fred_loader = FREDMacroLoader()
    start_date = df.index.min()
    end_date = df.index.max()
    
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
        df_with_date = df.reset_index()
        # The index is already named 'time' from the CSV
        if 'time' in df_with_date.columns:
            df_with_date.rename(columns={'time': 'timestamp'}, inplace=True)
        elif 'index' in df_with_date.columns:
            df_with_date.rename(columns={'index': 'timestamp'}, inplace=True)
        df_with_date['date'] = pd.to_datetime(df_with_date['timestamp']).dt.date
        fred_macro_df['date'] = pd.to_datetime(fred_macro_df['date']).dt.date
        
        df_merged = pd.merge(df_with_date, fred_macro_df, on='date', how='left')
        
        # Forward-fill macro features
        macro_cols = [col for col in fred_macro_df.columns if col != 'date']
        df_merged[macro_cols] = df_merged[macro_cols].ffill()
        
        # Normalize macro features to [-1, 1]
        for col in macro_cols:
            if col in df_merged.columns:
                mean_val = df_merged[col].mean()
                std_val = df_merged[col].std()
                df_merged[f'{col}_norm'] = (df_merged[col] - mean_val) / (std_val * 3)
                df_merged[f'{col}_norm'] = df_merged[f'{col}_norm'].clip(-1, 1)
        
        df_merged.set_index('timestamp', inplace=True)
        df = df_merged
        
        print(f"  ✓ Added {len(macro_cols)} macro features")
        macro_features_available = [f'{col}_norm' for col in macro_cols if f'{col}_norm' in df.columns]
    else:
        print("  ✗ No FRED data available")
        macro_features_available = []
except Exception as e:
    print(f"  ✗ FRED loading failed: {e}")
    macro_features_available = []

# Analyze correlations
print("\n[3/3] Analyzing correlations...")

# Base features (engineer's approach)
base_features = ['rsi_norm', 'macd_diff_norm', 'candle_body_norm']

# All features (base + macros)
all_features = base_features + macro_features_available

targets = ['target_direction', 'target_magnitude']

print("\n" + "="*80)
print("CORRELATION COMPARISON: BASE vs BASE+MACROS")
print("="*80)

results = []

for feature in all_features:
    if feature in df.columns:
        for target in targets:
            corr = np.corrcoef(df[feature].dropna().values, 
                             df.loc[df[feature].notna(), target].values)[0, 1]
            
            feature_type = 'Base' if feature in base_features else 'Macro'
            
            results.append({
                'Feature': feature.replace('_norm', ''),
                'Type': feature_type,
                'Target': target.replace('target_', ''),
                'Correlation': corr,
                'Abs_Correlation': abs(corr)
            })

results_df = pd.DataFrame(results)

# Pivot for better display
print("\n--- DIRECTION TARGET ---")
dir_results = results_df[results_df['Target'] == 'direction'].sort_values('Abs_Correlation', ascending=False)
print(dir_results[['Feature', 'Type', 'Correlation']].to_string(index=False))

print("\n--- MAGNITUDE TARGET ---")
mag_results = results_df[results_df['Target'] == 'magnitude'].sort_values('Abs_Correlation', ascending=False)
print(mag_results[['Feature', 'Type', 'Correlation']].to_string(index=False))

# Summary statistics
print("\n" + "="*80)
print("SUMMARY STATISTICS")
print("="*80)

base_stats = results_df[results_df['Type'] == 'Base'].groupby('Target')['Abs_Correlation'].agg(['mean', 'max'])
macro_stats = results_df[results_df['Type'] == 'Macro'].groupby('Target')['Abs_Correlation'].agg(['mean', 'max'])

print("\nBase Features (Engineer's 3):")
print(base_stats.to_string())

if len(macro_features_available) > 0:
    print("\nMacro Features:")
    print(macro_stats.to_string())
    
    print("\n" + "="*80)
    print("VERDICT")
    print("="*80)
    
    # Compare best macro vs worst base feature
    best_macro_dir = macro_stats.loc['direction', 'max']
    worst_base_dir = dir_results[dir_results['Type'] == 'Base']['Abs_Correlation'].min()
    
    best_macro_mag = macro_stats.loc['magnitude', 'max']
    worst_base_mag = mag_results[mag_results['Type'] == 'Base']['Abs_Correlation'].min()
    
    print(f"\nDirection prediction:")
    print(f"  Best macro:  {best_macro_dir:.4f}")
    print(f"  Worst base:  {worst_base_dir:.4f}")
    
    if best_macro_dir > worst_base_dir:
        print(f"  → ✓ Best macro beats worst base feature")
        print(f"  → RECOMMENDATION: Add best macro features")
    else:
        print(f"  → ✗ Macros don't beat base features")
        print(f"  → RECOMMENDATION: Skip macros, use engineer's 3 only")
    
    print(f"\nMagnitude prediction:")
    print(f"  Best macro:  {best_macro_mag:.4f}")
    print(f"  Worst base:  {worst_base_mag:.4f}")
    
    # Overall recommendation
    print("\n" + "="*80)
    print("FINAL RECOMMENDATION")
    print("="*80)
    
    if best_macro_dir > 0.05 or best_macro_mag > 0.05:
        print("\n✓ ADD MACROS: Some macro features show decent correlation (>5%)")
        
        # Find best macros
        best_macros_dir = dir_results[dir_results['Type'] == 'Macro'].head(3)
        best_macros_mag = mag_results[mag_results['Type'] == 'Macro'].head(3)
        
        print("\nBest macros for direction:")
        for _, row in best_macros_dir.iterrows():
            print(f"  - {row['Feature']:20s}: {row['Correlation']:7.4f}")
        
        print("\nBest macros for magnitude:")
        for _, row in best_macros_mag.iterrows():
            print(f"  - {row['Feature']:20s}: {row['Correlation']:7.4f}")
        
        print("\nRecommended feature set:")
        print("  Base: RSI, MACD_diff, Candle_body")
        print(f"  Macro: {', '.join(best_macros_dir['Feature'].head(2).tolist())}")
        print(f"  Total: 5 features (vs engineer's 3)")
    else:
        print("\n⚠️  SKIP MACROS: Correlation too low (<5%)")
        print("\nRecommended feature set:")
        print("  Use engineer's 3 features only: RSI, MACD_diff, Candle_body")
        print("  Macros add complexity without improving signal")
else:
    print("\n⚠️  No macro features available for comparison")

# Visualization
if len(macro_features_available) > 0:
    print("\n" + "="*80)
    print("GENERATING VISUALIZATION")
    print("="*80)
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Feature Correlation: Base vs Macros (ZigZag Target)', 
                 fontsize=16, fontweight='bold')
    
    # Plot 1: Direction
    ax1 = axes[0]
    dir_sorted = dir_results.sort_values('Correlation', ascending=True)
    colors = ['steelblue' if t == 'Base' else 'orange' for t in dir_sorted['Type']]
    ax1.barh(range(len(dir_sorted)), dir_sorted['Correlation'], color=colors, alpha=0.7)
    ax1.set_yticks(range(len(dir_sorted)))
    ax1.set_yticklabels(dir_sorted['Feature'], fontsize=9)
    ax1.set_xlabel('Correlation', fontsize=11)
    ax1.set_title('Direction Target', fontsize=13, fontweight='bold')
    ax1.axvline(x=0, color='black', linestyle='-', linewidth=1)
    ax1.grid(axis='x', alpha=0.3)
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='steelblue', alpha=0.7, label='Base'),
                      Patch(facecolor='orange', alpha=0.7, label='Macro')]
    ax1.legend(handles=legend_elements, loc='lower right')
    
    # Plot 2: Magnitude
    ax2 = axes[1]
    mag_sorted = mag_results.sort_values('Correlation', ascending=True)
    colors = ['steelblue' if t == 'Base' else 'orange' for t in mag_sorted['Type']]
    ax2.barh(range(len(mag_sorted)), mag_sorted['Correlation'], color=colors, alpha=0.7)
    ax2.set_yticks(range(len(mag_sorted)))
    ax2.set_yticklabels(mag_sorted['Feature'], fontsize=9)
    ax2.set_xlabel('Correlation', fontsize=11)
    ax2.set_title('Magnitude Target', fontsize=13, fontweight='bold')
    ax2.axvline(x=0, color='black', linestyle='-', linewidth=1)
    ax2.grid(axis='x', alpha=0.3)
    ax2.legend(handles=legend_elements, loc='lower right')
    
    plt.tight_layout()
    plt.savefig('macros_vs_base_zigzag.png', dpi=150, bbox_inches='tight')
    print("  ✓ Saved: macros_vs_base_zigzag.png")

print("\n" + "="*80)
print("✓ ANALYSIS COMPLETE")
print("="*80)
