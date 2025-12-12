"""
Technical indicator feature engineering module - OPTIMIZED VERSION
Based on correlation analysis by senior ML engineer:
- RSI: highest correlation (0.40 direction, 0.22 magnitude)
- MACD difference: strong correlation with target

REMOVED (redundant/irrelevant per author's analysis):
- Stochastic: >0.70 correlation with RSI (redundant)
- Bollinger Bands: -0.76 correlation with RSI (redundant)
- CCI: >0.70 correlation with RSI (redundant)
- ATR: near-zero correlation with target (irrelevant)
- Lagged features: LSTM handles sequences natively
- Multiple volatility windows: keep only vol_10
"""
import pandas as pd
import numpy as np
import talib
from typing import Dict, List, Optional
from loguru import logger

from src.config import TECHNICAL_INDICATORS


class TechnicalFeatureEngineer:
    """
    OPTIMIZED Technical indicator calculation
    Reduced from 81 to ~27 base features + 9 volatility regime features = ~36 total

    Core indicators (author's selection):
    - RSI: Highest predictive power
    - MACD: Strong directional signal
    - EMA 50/200: Trend context
    - ADX/DI: Trend strength (not redundant with oscillators)
    - Returns: Autoregressive signal
    - Single volatility measure
    
    NEW: Volatility Regime Features (9 features):
    - Parkinson & Garman-Klass volatility (efficient estimators)
    - Volatility percentile & regime classification
    - Volatility trend & vol-of-vol (regime stability)
    - Price efficiency ratio (trending vs choppy)
    - Vol-adjusted momentum (signal quality)
    - Regime change detector (vol breakout)
    """

    def __init__(self, config: Dict = TECHNICAL_INDICATORS):
        """
        Initialize technical feature engineer

        Args:
            config: Dictionary with indicator parameters
        """
        self.config = config

    def calculate_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate OPTIMIZED technical indicators + volatility regime features

        Args:
            df: OHLCV DataFrame

        Returns:
            DataFrame with ~27 base technical features + 9 volatility regime features = ~36 total
            
        Feature breakdown:
        - Moving averages: 5 features (EMA 50/200, distances, cross)
        - Momentum: 6 features (RSI, MACD components)
        - Trend: 4 features (ADX, DI+, DI-, DI diff)
        - Returns: 4 features (1/5/10 period returns, realized vol)
        - Volatility regime: 9 features (regime detection, vol dynamics, efficiency)
        """
        df = df.copy()

        # Ensure OHLCV columns exist
        required_cols = ["open", "high", "low", "close", "volume"]
        if not all(col in df.columns for col in required_cols):
            raise ValueError(
                f"DataFrame must contain columns: {required_cols}")

        # Calculate LEAN indicator set
        df = self._calculate_moving_averages(df)
        df = self._calculate_momentum_indicators(df)
        df = self._calculate_trend_indicators(df)
        df = self._calculate_returns(df)
        df = self._calculate_volatility_regime(df)  # NEW: Volatility regime features

        # Drop NaN rows created by indicators
        df.dropna(inplace=True)

        return df

    def _calculate_moving_averages(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate EMAs - ONLY 50 and 200 (author's selection)
        EMA 100 removed as redundant
        """
        # Only calculate EMA 50 and 200 for trend context
        df["ema_50"] = talib.EMA(df["close"], timeperiod=50)
        df["ema_200"] = talib.EMA(df["close"], timeperiod=200)

        # Normalized distances (use close range instead of ATR - ATR is irrelevant per author)
        price_range = df["high"].rolling(
            20).max() - df["low"].rolling(20).min()
        price_range = price_range.replace(0, np.nan).ffill()

        df["ema_50_dist"] = (df["close"] - df["ema_50"]) / price_range
        df["ema_200_dist"] = (df["close"] - df["ema_200"]) / price_range

        # Golden/Death cross signal (50 vs 200)
        df["ema_cross"] = np.where(df["ema_50"] > df["ema_200"], 1, -1)

        return df

    def _calculate_momentum_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate momentum indicators - RSI and MACD ONLY

        REMOVED per author's correlation analysis:
        - Stochastic: >0.70 correlation with RSI (redundant)
        """
        # RSI - Author's #1 indicator (0.40 direction correlation)
        rsi_period = self.config.get("RSI_PERIOD", 14)
        df["rsi"] = talib.RSI(df["close"], timeperiod=rsi_period)
        df["rsi_norm"] = df["rsi"] / 100.0  # Normalized 0-1

        # MACD - Author's #2 indicator
        macd_params = self.config.get("MACD_PERIODS", {
            "fastperiod": 12, "slowperiod": 26, "signalperiod": 9
        })
        df["macd"], df["macd_signal"], df["macd_hist"] = talib.MACD(
            df["close"],
            fastperiod=macd_params["fastperiod"],
            slowperiod=macd_params["slowperiod"],
            signalperiod=macd_params["signalperiod"],
        )

        # MACD difference (Main - Signal) - author's specific recommendation
        df["macd_diff"] = df["macd"] - df["macd_signal"]

        # MACD crossover signal
        df["macd_cross"] = np.where(df["macd"] > df["macd_signal"], 1, -1)

        return df

    def _calculate_trend_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate trend indicators - ADX and DI only

        REMOVED per author's analysis:
        - CCI: >0.70 correlation with RSI (redundant)
        """
        # ADX - Trend strength (not redundant with momentum oscillators)
        df["adx"] = talib.ADX(df["high"], df["low"],
                              df["close"], timeperiod=14)

        # Plus/Minus Directional Indicators
        df["plus_di"] = talib.PLUS_DI(
            df["high"], df["low"], df["close"], timeperiod=14)
        df["minus_di"] = talib.MINUS_DI(
            df["high"], df["low"], df["close"], timeperiod=14)

        # DI difference (directional signal)
        df["di_diff"] = df["plus_di"] - df["minus_di"]

        return df

    def _calculate_returns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate returns and SINGLE volatility measure

        REMOVED:
        - return_22: too long horizon
        - log_return: redundant with simple return
        - Multiple volatility windows: keep only vol_10
        - Annualized volatility: unnecessary complexity
        """
        # Simple returns (autoregressive signals)
        df["return_1"] = df["close"].pct_change(1)
        df["return_5"] = df["close"].pct_change(5)
        df["return_10"] = df["close"].pct_change(10)

        # Single volatility measure (10-period rolling std)
        df["realized_vol"] = df["return_1"].rolling(window=10).std()

        return df

    def _calculate_volatility_regime(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate volatility regime features - CRITICAL for adaptive trading
        
        Markets behave differently in different volatility regimes:
        - Low vol: Mean reversion works, trends are weak
        - High vol: Momentum works, breakouts are real
        - Vol expansion: Regime change, be cautious
        
        These features help the model adapt its strategy to market conditions.
        """
        # 1. Parkinson volatility (uses high-low range, more efficient than close-to-close)
        df["parkinson_vol"] = np.sqrt(
            (1 / (4 * np.log(2))) * 
            np.log(df["high"] / df["low"]) ** 2
        ).rolling(window=20).mean()
        
        # 2. Garman-Klass volatility (even more efficient, uses OHLC)
        hl = np.log(df["high"] / df["low"]) ** 2
        co = np.log(df["close"] / df["open"]) ** 2
        df["garman_klass_vol"] = np.sqrt(
            0.5 * hl - (2 * np.log(2) - 1) * co
        ).rolling(window=20).mean()
        
        # 3. Volatility percentile (where are we in the vol distribution?)
        # 0 = lowest vol in 100 periods, 1 = highest vol
        df["vol_percentile"] = df["realized_vol"].rolling(window=100).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) > 0 else 0.5
        )
        
        # 4. Volatility regime classification (categorical)
        # Low: bottom 33%, Medium: middle 33%, High: top 33%
        vol_33 = df["realized_vol"].rolling(window=100).quantile(0.33)
        vol_67 = df["realized_vol"].rolling(window=100).quantile(0.67)
        
        df["vol_regime"] = 0  # Medium by default
        df.loc[df["realized_vol"] < vol_33, "vol_regime"] = -1  # Low vol
        df.loc[df["realized_vol"] > vol_67, "vol_regime"] = 1   # High vol
        
        # 5. Volatility trend (is vol expanding or contracting?)
        df["vol_ma_short"] = df["realized_vol"].rolling(window=5).mean()
        df["vol_ma_long"] = df["realized_vol"].rolling(window=20).mean()
        df["vol_trend"] = (df["vol_ma_short"] - df["vol_ma_long"]) / df["vol_ma_long"]
        
        # 6. Volatility-of-volatility (vol stability)
        # High vol-of-vol = unstable regime, be cautious
        df["vol_of_vol"] = df["realized_vol"].rolling(window=20).std()
        
        # 7. Price efficiency ratio (trending vs choppy)
        # 1 = perfect trend, 0 = random walk
        price_change = abs(df["close"] - df["close"].shift(10))
        path_length = abs(df["close"].diff()).rolling(window=10).sum()
        df["efficiency_ratio"] = price_change / path_length.replace(0, np.nan)
        
        # 8. Volatility-adjusted momentum (momentum normalized by vol)
        # High momentum in low vol = strong signal
        # High momentum in high vol = noise
        df["vol_adj_momentum"] = df["return_5"] / (df["realized_vol"] + 1e-8)
        
        # 9. Regime change detector (vol breakout)
        # 1 = vol breaking out (regime change likely)
        vol_upper_band = df["realized_vol"].rolling(window=20).mean() + \
                        2 * df["realized_vol"].rolling(window=20).std()
        df["vol_breakout"] = (df["realized_vol"] > vol_upper_band).astype(int)
        
        return df

    def add_multi_timeframe_features(
        self,
        df_primary: pd.DataFrame,
        higher_timeframes: Dict[str, pd.DataFrame]
    ) -> pd.DataFrame:
        """
        Adds SIMPLIFIED multi-timeframe features
        Only RSI, ADX, and regime classification (reduced from full indicator set)

        Args:
            df_primary: The main dataframe (e.g., M5) to add features to.
            higher_timeframes: Dict mapping timeframe names (e.g., "H1", "H4")
                               to their OHLCV dataframes.

        Returns:
            Primary dataframe with MTF features merged in.
        """
        logger.info("Adding simplified multi-timeframe features...")
        df_merged = df_primary.copy()

        for tf, df_ht in higher_timeframes.items():
            logger.info(f"Calculating lean features for {tf} timeframe...")

            df_ht_feats = pd.DataFrame(index=df_ht.index)

            # Only RSI and ADX from higher timeframes (per author's selection)
            df_ht_feats[f'rsi_{tf}'] = talib.RSI(df_ht['close'], timeperiod=14)
            df_ht_feats[f'adx_{tf}'] = talib.ADX(
                df_ht['high'], df_ht['low'], df_ht['close'], timeperiod=14
            )

            # Simple regime classification
            ema_50 = talib.EMA(df_ht['close'], timeperiod=50)
            ema_200 = talib.EMA(df_ht['close'], timeperiod=200)

            df_ht_feats[f'regime_trending_{tf}'] = (
                df_ht_feats[f'adx_{tf}'] > 25).astype(int)
            df_ht_feats[f'regime_bullish_{tf}'] = (
                ema_50 > ema_200).astype(int)

            # Ensure datetime index for merging
            if not isinstance(df_merged.index, pd.DatetimeIndex):
                df_merged.index = pd.to_datetime(df_merged.index)
            if not isinstance(df_ht_feats.index, pd.DatetimeIndex):
                df_ht_feats.index = pd.to_datetime(df_ht_feats.index)

            # Merge using asof (forward fill from higher timeframe)
            df_merged = pd.merge_asof(
                left=df_merged.sort_index(),
                right=df_ht_feats.sort_index(),
                left_index=True,
                right_index=True,
                direction='backward'
            )
            logger.success(f"✓ Merged {tf} features (4 columns).")

        return df_merged

    def normalize_features_for_lstm(
        self, df: pd.DataFrame, feature_cols: List[str], scaler=None
    ) -> tuple:
        """
        Normalize features for LSTM input

        Args:
            df: DataFrame with features
            feature_cols: List of feature column names
            scaler: Pre-fitted scaler (if None, will fit new one)

        Returns:
            Tuple of (normalized_df, scaler)
        """
        from sklearn.preprocessing import MinMaxScaler

        if scaler is None:
            scaler = MinMaxScaler(feature_range=(0, 1))
            df[feature_cols] = scaler.fit_transform(df[feature_cols])
        else:
            df[feature_cols] = scaler.transform(df[feature_cols])

        return df, scaler

    def get_feature_list(self) -> List[str]:
        """
        Returns the expected feature column names for this optimized version.
        Useful for validation and documentation.

        Returns:
            List of feature column names (excluding OHLCV and target columns)
        """
        base_features = [
            # EMAs (3)
            "ema_50_dist", "ema_200_dist", "ema_cross",
            # RSI (2) - Author's #1
            "rsi", "rsi_norm",
            # MACD (5) - Author's #2
            "macd", "macd_signal", "macd_hist", "macd_diff", "macd_cross",
            # Trend (4)
            "adx", "plus_di", "minus_di", "di_diff",
            # Returns (3)
            "return_1", "return_5", "return_10",
            # Volatility (1)
            "realized_vol",
        ]

        mtf_features = [
            # H1 (4)
            "rsi_H1", "adx_H1", "regime_trending_H1", "regime_bullish_H1",
            # H4 (4)
            "rsi_H4", "adx_H4", "regime_trending_H4", "regime_bullish_H4",
        ]

        # Macro features are added separately by main.py
        # Can be either old (3): tau_pre, tau_post, weighted_surprise
        # Or new FRED (5+): rate_differential, vix, gdp_growth, unemployment_rate, inflation_rate

        return base_features + mtf_features


def engineer_simplified_zigzag_features(df: pd.DataFrame, pair: str, fred_loader=None, skip_fred_warning: bool = False) -> tuple[pd.DataFrame, list]:
    """
    Engineer a simplified set of 7 features using the ZigZag approach,
    matching the training pipeline exactly.

    This is the single source of truth for feature engineering for the 7-feature model.

    Features:
    - 3 base technical features (rsi_norm, macd_diff_norm, candle_body_norm)
    - 2 velocity features (rsi_velocity, macd_velocity)
    - 2 FRED macro features (yield_curve, dxy_index)

    Args:
        df: OHLCV DataFrame (expects M5).
        pair: Currency pair (e.g., 'EUR_USD').
        fred_loader: An initialized FREDMacroLoader object. If None, macro features
                     will be filled with placeholders.
        skip_fred_warning: If True, suppress warning when fred_loader is None
                          (useful when macro features are added separately).

    Returns:
        A tuple containing:
        - df_features: DataFrame with the engineered features.
        - feature_cols: List of engineered feature column names.
    """
    logger.info(
        f"Engineering simplified features for {pair} from {len(df)} candles")

    df_features = df.copy()

    # --- 1. Base technical features ---
    logger.debug("Calculating 3 base technical features...")
    df_features['rsi_12'] = talib.RSI(df_features['close'], timeperiod=12)
    df_features['rsi_norm'] = (df_features['rsi_12'] - 50.0) / 50.0

    macd, macd_signal, _ = talib.MACD(
        df_features['close'],
        fastperiod=12,
        slowperiod=48,
        signalperiod=12
    )
    macd_diff = np.abs(macd - macd_signal)
    macd_mean = macd_diff.mean() if not macd_diff.empty else 0.0
    macd_std = macd_diff.std() if not macd_diff.empty and macd_diff.std() != 0 else 1.0
    df_features['macd_diff_norm'] = ((macd_diff - macd_mean) / (macd_std * 3)).clip(-1, 1)

    candle_body = df_features['close'] - df_features['open']
    body_mean = candle_body.mean() if not candle_body.empty else 0.0
    body_std = candle_body.std() if not candle_body.empty and candle_body.std() != 0 else 1.0
    df_features['candle_body_norm'] = ((candle_body - body_mean) / (body_std * 3)).clip(-1, 1)

    logger.debug("✓ Calculated 3 base features")

    # --- 2. Velocity features ---
    df_features['rsi_velocity'] = df_features['rsi_norm'].diff().fillna(0)
    df_features['macd_velocity'] = df_features['macd_diff_norm'].diff().fillna(0)
    logger.debug("✓ Added 2 velocity features")

    # --- 3. Macro features (FRED ONLY: yield_curve, dxy_index) ---
    if fred_loader is not None:
        logger.debug(f"Fetching FRED macro features for {pair}...")
        try:
            start_date = df_features.index.min().to_pydatetime()
            end_date = df_features.index.max().to_pydatetime()

            fred_macro_df = fred_loader.get_macro_features_for_pair(
                pair, start_date, end_date
            )

            if not fred_macro_df.empty:
                key_fred_features = ['yield_curve', 'dxy_index']
                selected_cols = ['date'] + [col for col in key_fred_features if col in fred_macro_df.columns]
                fred_macro_df = fred_macro_df[selected_cols]

                original_index_name = df_features.index.name or 'timestamp'
                df_features['_merge_date'] = pd.to_datetime(df_features.index).normalize()
                fred_macro_df['_merge_date'] = pd.to_datetime(fred_macro_df['date']).dt.normalize()
                
                if fred_macro_df['_merge_date'].duplicated().any():
                    fred_macro_df = fred_macro_df.drop_duplicates(subset=['_merge_date'], keep='last')

                original_rows = len(df_features)
                df_features = df_features.reset_index(names=[original_index_name])

                fred_cols_to_merge = [col for col in fred_macro_df.columns if col not in ['date', '_merge_date']]
                fred_for_merge = fred_macro_df[['_merge_date'] + fred_cols_to_merge]
                df_features = pd.merge(df_features, fred_for_merge, on='_merge_date', how='left')

                if len(df_features) != original_rows:
                    logger.error(f"MERGE BUG after FRED: Row count changed from {original_rows} to {len(df_features)}")
                    raise ValueError(f"Merge created duplicate rows: {original_rows} -> {len(df_features)}")

                macro_cols = fred_cols_to_merge
                for col in macro_cols:
                    if col in df_features.columns:
                        df_features[col] = df_features[col].ffill().fillna(0)

                df_features = df_features.set_index(original_index_name)
                df_features = df_features.drop(columns=['_merge_date'], errors='ignore')

                logger.debug(f"✓ Added {len(macro_cols)} FRED macro features: {macro_cols}")
                
                # Ensure yield_curve and dxy_index exist (may not be in merged cols)
                if 'yield_curve' not in df_features.columns:
                    df_features['yield_curve'] = 0.0
                    logger.warning("yield_curve not in FRED data, using placeholder")
                if 'dxy_index' not in df_features.columns:
                    df_features['dxy_index'] = 100.0
                    logger.warning("dxy_index not in FRED data, using placeholder")
            else:
                logger.warning("No FRED data available - adding placeholders for yield_curve, dxy_index")
                df_features['yield_curve'] = 0.0
                df_features['dxy_index'] = 100.0
        except Exception as e:
            logger.warning(f"⚠ Failed to fetch FRED macro features: {e}. Using placeholders.")
            df_features['yield_curve'] = 0.0
            df_features['dxy_index'] = 100.0
    else:
        if not skip_fred_warning:
            logger.warning("No FRED loader available - adding placeholders for yield_curve, dxy_index")
        df_features['yield_curve'] = 0.0
        df_features['dxy_index'] = 100.0
    
    # Final safety check - ensure both macro features exist
    if 'yield_curve' not in df_features.columns:
        df_features['yield_curve'] = 0.0
    if 'dxy_index' not in df_features.columns:
        df_features['dxy_index'] = 100.0

    # Drop NaN from initial feature calcs
    initial_len = len(df_features)
    df_features.dropna(inplace=True)
    logger.debug(f"Dropped {initial_len - len(df_features)} rows with NaNs")

    feature_cols = [
        'rsi_norm',
        'macd_diff_norm',
        'candle_body_norm',
        'rsi_velocity',
        'macd_velocity',
        'yield_curve',
        'dxy_index'
    ]
    
    missing_cols = [col for col in feature_cols if col not in df_features.columns]
    if missing_cols:
        raise ValueError(f"Feature engineering failed to create required columns: {missing_cols}")
        
    df_features = df_features[feature_cols + [c for c in df_features.columns if c not in feature_cols and c not in ['rsi_12']]]

    logger.info(f"Engineered {len(feature_cols)} features, {len(df_features)} samples after dropna")

    return df_features, feature_cols


if __name__ == "__main__":
    # Example usage
    from src.data_acquisition.fx_data import FXDataAcquisition
    from datetime import datetime, timedelta

    # Fetch sample data
    fx_data = FXDataAcquisition()
    end_date = datetime.now()
    start_date = end_date - timedelta(days=90)

    df = fx_data.fetch_oanda_candles(
        instrument="EUR_USD",
        granularity="H4",
        start_date=start_date,
        end_date=end_date,
    )

    # Calculate technical features
    tech_engineer = TechnicalFeatureEngineer()
    df_features = tech_engineer.calculate_all_features(df)

    print(f"\n=== OPTIMIZED Feature Set ===")
    print(f"Total features: {len(df_features.columns)} columns")
    print(f"\nExpected features: {tech_engineer.get_feature_list()}")
    print(f"\nActual columns: {df_features.columns.tolist()}")
    print(f"\n{df_features.tail()}")
