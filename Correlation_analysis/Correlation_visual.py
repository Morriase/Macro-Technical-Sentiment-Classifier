"""
Correlation_visual.py

Loads `correlation.csv` (expected in the same folder) and visualizes the correlation
between indicators and price. The script is robust to a few common CSV formats:

- A DataFrame with a numeric `price` column: computes Pearson correlations of all
  other numeric columns with `price` and draws a barplot.
- A two-column CSV with an indicator column and a correlation/value column: draws
  a barplot of the second column indexed by the first.
- A square correlation matrix: draws a heatmap.

Outputs saved as PNG files in the current directory.

Usage (from the folder containing `correlation.csv`):
	python Correlation_visual.py
	python Correlation_visual.py --input correlation.csv --show

Requirements: `pandas`, `seaborn`, `matplotlib`.
"""

from pathlib import Path
import argparse
import sys
import textwrap

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def load_data(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, index_col=None)


def is_square_matrix(df: pd.DataFrame) -> bool:
    return df.shape[0] == df.shape[1] and all(df.columns == list(df.index))


def plot_bar_correlations(series, out_path: Path, title: str = None, show: bool = False):
    # scale height with number of items so labels have space
    s = series.sort_values(ascending=False)
    n = max(1, len(s))
    height = max(6, n * 0.25)
    plt.figure(figsize=(10, height))

    # build a palette sized to the data to avoid seaborn future warning
    palette = sns.color_palette("vlag", n_colors=min(256, n))
    ax = sns.barplot(x=s.values, y=s.index, palette=palette)

    # wrap long y-labels and set font size depending on count
    max_width = 30
    wrapped = [textwrap.fill(str(lbl), max_width) for lbl in s.index]
    if n <= 40:
        y_font = 9
    elif n <= 120:
        y_font = 7
    else:
        y_font = 5
    # ensure ticks align with labels and avoid set_ticklabels warning
    ax.set_yticks(range(n))
    ax.set_yticklabels(wrapped, fontsize=y_font)

    # leave extra left margin for long labels
    plt.subplots_adjust(left=0.45)

    ax.set_xlabel('Correlation with price' if title is None else '')
    if title:
        ax.set_title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    if show:
        plt.show()
    plt.close()


def plot_heatmap(df: pd.DataFrame, out_path: Path, title: str = None, show: bool = False):
    plt.figure(figsize=(10, max(6, df.shape[0] * 0.4)))
    ax = sns.heatmap(df, annot=True, fmt='.2f', cmap='vlag', center=0)
    if title:
        ax.set_title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    if show:
        plt.show()
    plt.close()


def main():
    p = argparse.ArgumentParser(
        description="Visualize correlations from a CSV file.")
    p.add_argument('--input', '-i', default='correlation.csv',
                   help='Path to the CSV file')
    p.add_argument('--show', action='store_true',
                   help='Show plots interactively')
    args = p.parse_args()

    path = Path(args.input)
    if not path.exists():
        print(f"File not found: {path}")
        sys.exit(2)

    df = load_data(path)

    # Case E: Pairwise correlations (Indicator1, Indicator2, Correlation)
    if 'Indicator1' in df.columns and 'Indicator2' in df.columns and 'Correlation' in df.columns:
        # Pivot to matrix
        matrix = df.pivot(index='Indicator1',
                          columns='Indicator2', values='Correlation')
        # Fill NaN with 0 or leave as is
        matrix = matrix.fillna(0)
        out = Path('indicator_correlations_heatmap.png')
        plot_heatmap(
            matrix, out, title='Indicator Correlations Heatmap', show=args.show)
        print(f"Saved heatmap: {out.resolve()}")
        return

    # Case D: Rows with an `Indicator` column and one or more numeric correlation columns
    if 'Indicator' in df.columns:
        # Determine candidate numeric columns that are correlation values
        ignore = {'Indicator', 'Shift'}
        corr_cols = [
            c for c in df.columns if c not in ignore and pd.api.types.is_numeric_dtype(df[c])]
        if corr_cols:
            # Create a label column combining Indicator and Shift (if present)
            if 'Shift' in df.columns:
                labels = df['Indicator'].astype(
                    str) + ' (shift=' + df['Shift'].astype(str) + ')'
            else:
                labels = df['Indicator'].astype(str)

            # Plot each correlation column separately
            for col in corr_cols:
                series = pd.Series(df[col].values, index=labels)
                # Clean filename-friendly column name
                safe = ''.join(ch if ch.isalnum() or ch in (
                    ' ', '_', '-') else '_' for ch in col).strip()
                out = Path(f'correlation_{safe}.png')
                plot_bar_correlations(series, out, title=col, show=args.show)
                print(f"Saved barplot: {out.resolve()}")
            return

    # Case A: DataFrame contains a numeric 'price' column -> correlate others with price
    if 'price' in df.columns and pd.api.types.is_numeric_dtype(df['price']):
        corr = df.corr(method='pearson')
        if 'price' in corr.columns:
            series = corr['price'].drop('price', errors='ignore')
            out = Path('correlation_with_price_bar.png')
            plot_bar_correlations(
                series, out, title='Correlation of indicators with price', show=args.show)
            print(f"Saved barplot: {out.resolve()}")
            return

    # Case B: Two-column CSV: assume first col labels, second numeric values (correlations)
    if df.shape[1] == 2:
        col0, col1 = df.columns[0], df.columns[1]
        if pd.api.types.is_numeric_dtype(df[col1]):
            series = pd.Series(df[col1].values, index=df[col0].astype(str))
            out = Path('correlation_bar_from_two_column.csv.png')
            plot_bar_correlations(
                series, out, title=f'{col1} by {col0}', show=args.show)
            print(f"Saved barplot: {out.resolve()}")
            return

    # Case C: Square matrix (correlation matrix) -> heatmap
    # If index is not set equal to columns, try to set first column as index if that helps
    try:
        # If first column looks like labels repeated in columns, set index
        if df.shape[1] > 1 and df.columns[0] != df.index.name:
            # try interpreting first column as index if it contains non-numeric labels
            if not pd.api.types.is_numeric_dtype(df.iloc[:, 0]):
                df_indexed = df.set_index(df.columns[0])
                numeric = df_indexed.apply(pd.to_numeric, errors='coerce')
                if numeric.shape[0] == numeric.shape[1]:
                    df = numeric
        numeric_df = df.apply(pd.to_numeric, errors='coerce')
        if numeric_df.shape[0] == numeric_df.shape[1]:
            # align columns/index names
            numeric_df.index = numeric_df.index.astype(str)
            numeric_df.columns = numeric_df.columns.astype(str)
            out = Path('correlation_heatmap.png')
            plot_heatmap(numeric_df, out,
                         title='Correlation matrix heatmap', show=args.show)
            print(f"Saved heatmap: {out.resolve()}")
            return
    except Exception:
        pass

    # Fallback: attempt to compute correlations with any numeric column named similar to 'price'
    possible_price_cols = [c for c in df.columns if 'price' in c.lower()]
    if possible_price_cols:
        col = possible_price_cols[0]
        if pd.api.types.is_numeric_dtype(df[col]):
            corr = df.corr().get(col)
            if corr is not None:
                series = corr.drop(col, errors='ignore')
                out = Path('correlation_with_price_bar_fallback.png')
                plot_bar_correlations(
                    series, out, title=f'Correlation with {col}', show=args.show)
                print(f"Saved barplot: {out.resolve()}")
                return

    print("Could not detect a known correlation format in the CSV.\n"
          "Please provide a CSV with either a 'price' column, a two-column file (label,value),\n"
          "or a square correlation matrix.")


if __name__ == '__main__':
    main()
