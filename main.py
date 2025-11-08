"""
Main Pipeline for Macro-Technical Sentiment Forex Classifier
End-to-end training and prediction workflow
Optimized for both local and Kaggle (GPU/CUDA) environments
"""
import sys
from loguru import logger
from pathlib import Path
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from src.config import (
    CURRENCY_PAIRS,
    PRIMARY_PAIR,
    DATA_DIR,
    MODELS_DIR,
    LOGS_DIR,
    RESULTS_DIR,
    LOG_LEVEL,
    LOG_FORMAT,
    IS_KAGGLE,
    USE_CUDA,
    DEVICE,
    SENTIMENT_EMA_PERIODS,
)
from src.data_acquisition.kaggle_loader import KaggleFXDataLoader
from src.data_acquisition.news_loader import KaggleNewsLoader
from src.feature_engineering.technical_features import TechnicalFeatureEngineer
from src.feature_engineering.sentiment_features import SentimentAnalyzer
from src.models.hybrid_ensemble import HybridEnsemble
from src.validation.walk_forward import WalkForwardOptimizer
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)


# Optional imports for local environment (not needed on Kaggle)
try:
    from src.data_acquisition.macro_data import MacroDataAcquisition
    from src.data_acquisition.fx_data import FXDataAcquisition
    HAS_API_SOURCES = True
except ImportError:
    HAS_API_SOURCES = False
    MacroDataAcquisition = None
    FXDataAcquisition = None

# Add src to path
sys.path.append(str(Path(__file__).parent))


# Configure logging
logger.remove()
logger.add(sys.stderr, level=LOG_LEVEL, format=LOG_FORMAT)
logger.add(
    LOGS_DIR / f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
    level=LOG_LEVEL,
    format=LOG_FORMAT,
    rotation="100 MB",
)

# Log environment info
if IS_KAGGLE:
    logger.info("Running on Kaggle")
    if USE_CUDA:
        import torch
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"CUDA Version: {torch.version.cuda}")
else:
    logger.info("Running locally")
logger.info(f"Device: {DEVICE}")


class ForexClassifierPipeline:
    """
    Main pipeline orchestrating data acquisition, feature engineering,
    model training, and evaluation
    """

    def __init__(self, currency_pair: str = PRIMARY_PAIR, use_kaggle_data: bool = IS_KAGGLE):
        """
        Initialize pipeline

        Args:
            currency_pair: Currency pair to trade (e.g., 'EUR_USD')
            use_kaggle_data: Whether to use Kaggle dataset (auto-detected)
        """
        self.currency_pair = currency_pair
        self.use_kaggle_data = use_kaggle_data

        # Initialize data sources
        if use_kaggle_data:
            self.kaggle_loader = KaggleFXDataLoader()
            self.kaggle_news_loader = KaggleNewsLoader()
            self.fx_data = None
            # Create MacroDataAcquisition for feature calculations (doesn't need API)
            self.macro_data = MacroDataAcquisition() if (
                HAS_API_SOURCES and MacroDataAcquisition) else None
            logger.info("Using Kaggle dataset")
        else:
            if not HAS_API_SOURCES:
                raise ImportError(
                    "API data sources (OANDA, Finnhub) not available. "
                    "Install required packages or use Kaggle dataset."
                )
            self.kaggle_loader = None
            self.kaggle_news_loader = None
            self.fx_data = FXDataAcquisition()
            self.macro_data = MacroDataAcquisition()
            logger.info("Using OANDA/API data sources")

        # Initialize feature engineers (common to both modes)
        self.tech_engineer = TechnicalFeatureEngineer()
        self.sentiment_analyzer = SentimentAnalyzer()

        # Initialize data storage
        self.df_price = None
        self.df_events = None
        self.df_news = None
        self.df_features = None
        self.model = None

        logger.info(f"Pipeline initialized for {currency_pair}")

    def fetch_data(
        self,
        start_date: datetime = None,
        end_date: datetime = None,
        save: bool = True,
    ):
        """
        Step 1: Fetch all required data

        Args:
            start_date: Start date for data (ignored on Kaggle)
            end_date: End date for data (ignored on Kaggle)
            save: Whether to save data to disk
        """
        logger.info("="*60)
        logger.info("STEP 1: DATA ACQUISITION")
        logger.info("="*60)

        if self.use_kaggle_data:
            # Load from Kaggle dataset
            logger.info(f"Loading {self.currency_pair} from Kaggle dataset")
            # Convert symbol format: EUR_USD -> EURUSD
            symbol = self.currency_pair.replace("_", "")
            df_m5 = self.kaggle_loader.load_symbol_data(symbol, timeframe="M5")

            logger.info(f"Loaded {len(df_m5):,} M5 candles")
            logger.info(
                f"Date range: {df_m5.index.min()} to {df_m5.index.max()}")

            # For Kaggle, use M5 data directly or resample as needed
            self.df_price = df_m5  # Can resample if needed

            # Load pre-downloaded macro events
            logger.info(f"Loading macro events for {symbol}")
            self.df_events = self.kaggle_loader.load_macro_events(symbol)
            if self.df_events is not None and not self.df_events.empty:
                logger.success(f"✓ Loaded {len(self.df_events)} macro events")
            else:
                logger.warning(
                    "⚠ No macro events found - training without macro features")

            # Load historical news for sentiment analysis (REQUIRED for 67 features)
            logger.info("Loading historical news for sentiment analysis")
            self.df_news = self.kaggle_news_loader.load_historical_news(
                start_date=start_date, end_date=end_date
            )
            if self.df_news is not None and not self.df_news.empty:
                logger.success(f"✓ Loaded {len(self.df_news)} news articles")
            else:
                raise ValueError(
                    "Failed to load historical news - REQUIRED for 67-feature training")

            logger.info("✓ Data loaded successfully from Kaggle dataset")

        else:
            # Fetch FX price data from OANDA
            logger.info(f"Fetching FX data for {self.currency_pair}")
            df_m5 = self.fx_data.fetch_oanda_candles(
                instrument=self.currency_pair,
                granularity="M5",
                start_date=start_date,
                end_date=end_date,
            )

            # Resample to 4H
            self.df_price = self.fx_data.resample_to_timeframe(df_m5, "4H")

            # Validate quality
            quality = self.fx_data.validate_data_quality(self.df_price)
            logger.info(
                f"Price data quality score: {quality['quality_score']:.2%}")

            # Fetch macroeconomic events
            logger.info("Fetching macroeconomic events")
            self.df_events = self.macro_data.get_events_for_currency_pair(
                pair=self.currency_pair,
                start_date=start_date,
                end_date=end_date,
            )

        # Save data (skip on Kaggle - data is read-only)
        if save and not self.use_kaggle_data:
            if self.fx_data:
                self.fx_data.save_data(self.df_price, self.currency_pair, "4H")
            if self.df_events is not None and not self.df_events.empty and self.macro_data:
                self.macro_data.save_events(self.df_events, self.currency_pair)

        logger.info(f"Fetched {len(self.df_price)} price bars")
        if self.df_events is not None:
            logger.info(f"Fetched {len(self.df_events)} macro events")
        else:
            logger.info("No macro events (Kaggle mode)")

    def engineer_features(self):
        """
        Step 2: Engineer all features (technical, macro, sentiment)
        """
        logger.info("="*60)
        logger.info("STEP 2: FEATURE ENGINEERING")
        logger.info("="*60)

        if self.df_price is None:
            raise ValueError("Price data not loaded. Run fetch_data() first.")

        # Technical features
        self.df_features = self.tech_engineer.calculate_all_features(
            self.df_price.copy())
        self.df_features = self.tech_engineer.calculate_feature_crosses(
            self.df_features)

        # Macro features (temporal proximity)
        if self.df_events is not None and not self.df_events.empty and self.macro_data:
            self.df_features = self.macro_data.calculate_temporal_proximity(
                events_df=self.df_events,
                price_df=self.df_features,
            )
        else:
            self.df_features["tau_pre"] = 0.0
            self.df_features["tau_post"] = 0.0
            self.df_features["weighted_surprise"] = 0.0

        # Sentiment features (REQUIRED - must have 67 features total)
        if self.sentiment_analyzer.sentiment_pipeline is None:
            raise RuntimeError(
                "Sentiment model failed to load - REQUIRED for 67-feature training")

        if self.df_news is None or self.df_news.empty:
            raise ValueError(
                "No historical news data - REQUIRED for 67-feature training")

        logger.info("Calculating sentiment features...")
        daily_sentiment = self.sentiment_analyzer.aggregate_daily_sentiment(
            self.df_news
        )
        time_weighted_sentiment = self.sentiment_analyzer.calculate_time_weighted_sentiment(
            daily_sentiment
        )
        # Merge sentiment features into main df_features
        # Reset index to merge on 'date' column, then restore index
        self.df_features = self.df_features.reset_index()
        self.df_features = pd.merge(
            self.df_features,
            time_weighted_sentiment,
            left_on="date",
            right_on="date",
            how="left"
        )
        # Fill NaN values that result from merging (e.g., days with no news)
        sentiment_cols = [
            col for col in time_weighted_sentiment.columns if col != "date"]
        for col in sentiment_cols:
            self.df_features[col] = self.df_features[col].fillna(0.0)
        # Restore DateTimeIndex
        self.df_features = self.df_features.set_index("date")
        logger.success(
            "✓ Sentiment features calculated and merged (67 features total)")

        # Drop NaN
        initial_len = len(self.df_features)
        self.df_features.dropna(inplace=True)

        logger.info(
            f"✓ {len(self.df_features.columns)} features created, {len(self.df_features)} samples ready")

    def create_target(
        self,
        forward_window_hours: int = 24,
        min_move_pips: float = 10.0,
    ):
        """
        Step 3: Create target variable

        Args:
            forward_window_hours: Hours ahead to look for movement
            min_move_pips: Minimum pips move to classify as directional
        """
        logger.info("="*60)
        logger.info("STEP 3: TARGET CREATION")
        logger.info("="*60)

        if self.df_features is None:
            raise ValueError(
                "Features not engineered. Run engineer_features() first.")

        # Calculate forward return
        bars_ahead = forward_window_hours // 4  # 4H bars
        self.df_features["forward_close"] = self.df_features["close"].shift(
            -bars_ahead)
        self.df_features["forward_return"] = (
            self.df_features["forward_close"] - self.df_features["close"]
        )

        # Convert to pips (assuming 4-decimal pairs like EUR/USD)
        pip_multiplier = 10000
        self.df_features["forward_return_pips"] = (
            self.df_features["forward_return"] * pip_multiplier
        )

        # Create ternary target: Buy (1), Hold (0), Sell (-1)
        conditions = [
            self.df_features["forward_return_pips"] > min_move_pips,   # Buy
            self.df_features["forward_return_pips"] < -min_move_pips,  # Sell
        ]
        choices = [1, -1]
        self.df_features["target"] = np.select(conditions, choices, default=0)

        # Map to 0, 1, 2 for sklearn
        target_map = {-1: 1, 0: 2, 1: 0}  # Sell=1, Hold=2, Buy=0
        self.df_features["target_class"] = self.df_features["target"].map(
            target_map)

        # Drop future data
        self.df_features.dropna(subset=["forward_close"], inplace=True)

        # Class distribution
        class_counts = self.df_features["target_class"].value_counts(
        ).sort_index()
        logger.info("Target class distribution:")
        for cls, count in class_counts.items():
            class_name = ["Buy", "Sell", "Hold"][cls]
            pct = count / len(self.df_features) * 100
            logger.info(f"  {class_name}: {count} ({pct:.1f}%)")

    def train_model(
        self,
        use_walk_forward: bool = True,
        optimize_hyperparams: bool = True,
    ):
        """
        Step 4: Train model using walk-forward optimization

        Args:
            use_walk_forward: Whether to use WFO (recommended)
            optimize_hyperparams: Whether to optimize hyperparameters
        """
        logger.info("="*60)
        logger.info("STEP 4: MODEL TRAINING")
        logger.info("="*60)

        if self.df_features is None or "target_class" not in self.df_features.columns:
            raise ValueError("Target not created. Run create_target() first.")

        # Define feature columns (exclude target and derived columns)
        exclude_cols = [
            "open", "high", "low", "close", "volume",
            "forward_close", "forward_return", "forward_return_pips",
            "target", "target_class", "date"  # Exclude 'date' column from sentiment merge
        ]
        feature_cols = [
            col for col in self.df_features.columns
            if col not in exclude_cols
        ]

        logger.info(f"Using {len(feature_cols)} features for training")

        if use_walk_forward:
            # Walk-Forward Optimization
            optimizer = WalkForwardOptimizer(
                model_class=HybridEnsemble,
            )

            results = optimizer.run_walk_forward_optimization(
                df=self.df_features,
                feature_columns=feature_cols,
                target_column="target_class",
                optimize_each_window=optimize_hyperparams,
            )

            # Use the most recent model
            self.model = results[-1]["model"]

            # Save results
            summary_df = optimizer.aggregate_results()
            summary_path = RESULTS_DIR / \
                f"{self.currency_pair}_wfo_summary.csv"
            summary_df.to_csv(summary_path, index=False)
            logger.info(f"WFO summary saved to {summary_path}")

        else:
            # Simple train/test split (not recommended for time series)
            logger.warning(
                "Using simple split - not recommended for time series!")

            train_size = int(len(self.df_features) * 0.8)
            X_train = self.df_features[feature_cols].values[:train_size]
            y_train = self.df_features["target_class"].values[:train_size]

            self.model = HybridEnsemble()
            self.model.fit(X_train, y_train)

        # Save model
        model_path = MODELS_DIR / f"{self.currency_pair}_model"
        self.model.save_model(str(model_path))
        logger.info(f"Model saved to {model_path}_*")

        # Save feature schema separately for inference server (JSON format for easy loading)
        import json
        feature_schema = {
            "currency_pair": self.currency_pair,
            "n_features": len(feature_cols),
            "feature_names": feature_cols,
            "feature_order": {name: idx for idx, name in enumerate(feature_cols)},
            "model_version": "1.0",
            "trained_date": datetime.now().isoformat(),
        }
        schema_path = MODELS_DIR / f"{self.currency_pair}_feature_schema.json"
        with open(schema_path, "w") as f:
            json.dump(feature_schema, f, indent=2)
        logger.info(f"Feature schema saved to {schema_path}")

    def generate_predictions(
        self,
        confidence_threshold: float = 0.70,
    ) -> pd.DataFrame:
        """
        Step 5: Generate predictions for most recent data

        Args:
            confidence_threshold: Minimum confidence for signal

        Returns:
            DataFrame with predictions and signals
        """
        logger.info("="*60)
        logger.info("STEP 5: PREDICTION GENERATION")
        logger.info("="*60)

        if self.model is None:
            raise ValueError("Model not trained. Run train_model() first.")

        # Get feature columns
        exclude_cols = [
            "open", "high", "low", "close", "volume",
            "forward_close", "forward_return", "forward_return_pips",
            "target", "target_class"
        ]
        feature_cols = [
            col for col in self.df_features.columns
            if col not in exclude_cols
        ]

        # Predict on most recent data (last 100 bars)
        X_recent = self.df_features[feature_cols].tail(100).values

        # Generate predictions
        y_pred_proba = self.model.predict_proba(X_recent)
        y_pred = self.model.predict(X_recent)

        # Create results DataFrame
        results = pd.DataFrame({
            "timestamp": self.df_features.tail(100).index,
            "close": self.df_features["close"].tail(100).values,
            "pred_buy_prob": y_pred_proba[:, 0],
            "pred_sell_prob": y_pred_proba[:, 1],
            "pred_hold_prob": y_pred_proba[:, 2],
            "predicted_class": y_pred,
        })

        # Generate signals based on confidence
        def generate_signal(row):
            if row["pred_buy_prob"] > confidence_threshold:
                return "BUY"
            elif row["pred_sell_prob"] > confidence_threshold:
                return "SELL"
            else:
                return "HOLD"

        results["signal"] = results.apply(generate_signal, axis=1)

        # Latest signal
        latest = results.iloc[-1]
        logger.info(f"\nLatest Signal ({latest['timestamp']}):")
        logger.info(f"  Close: {latest['close']:.5f}")
        logger.info(f"  Signal: {latest['signal']}")
        logger.info(f"  Buy Confidence: {latest['pred_buy_prob']:.2%}")
        logger.info(f"  Sell Confidence: {latest['pred_sell_prob']:.2%}")
        logger.info(f"  Hold Confidence: {latest['pred_hold_prob']:.2%}")

        # Save predictions
        pred_path = RESULTS_DIR / f"{self.currency_pair}_predictions.csv"
        results.to_csv(pred_path, index=False)
        logger.info(f"Predictions saved to {pred_path}")

        return results

    def run_full_pipeline(
        self,
        years_history: int = 5,
        use_walk_forward: bool = True,
    ):
        """
        Run complete end-to-end pipeline

        Args:
            years_history: Years of historical data to fetch
            use_walk_forward: Whether to use walk-forward optimization
        """
        logger.info("="*60)
        logger.info("FOREX CLASSIFIER PIPELINE - FULL EXECUTION")
        logger.info(f"Currency Pair: {self.currency_pair}")
        logger.info("="*60)

        start_time = datetime.now()

        # Define date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=years_history * 365)

        try:
            # Step 1: Fetch data
            self.fetch_data(start_date, end_date)

            # Step 2: Engineer features
            self.engineer_features()

            # Step 3: Create target
            self.create_target()

            # Step 4: Train model
            self.train_model(use_walk_forward=use_walk_forward)

            # Step 5: Generate predictions
            predictions = self.generate_predictions()

            elapsed = datetime.now() - start_time
            logger.info("="*60)
            logger.info(f"PIPELINE COMPLETED SUCCESSFULLY in {elapsed}")
            logger.info("="*60)

            return predictions

        except Exception as e:
            logger.error(f"Pipeline failed: {e}", exc_info=True)
            raise


def main():
    """Main entry point - Train models for all currency pairs and aggregate signals"""
    from src.config import CURRENCY_PAIRS, RESULTS_DIR
    from src.models.portfolio_ensemble import PortfolioEnsemble

    all_results = {}

    for pair in CURRENCY_PAIRS:
        logger.info(f"\n{'='*80}")
        logger.info(f"STARTING TRAINING FOR {pair}")
        logger.info(f"{'='*80}\n")

        try:
            pipeline = ForexClassifierPipeline(currency_pair=pair)

            # Run full pipeline
            predictions = pipeline.run_full_pipeline(
                years_history=5,
                use_walk_forward=True,
            )

            all_results[pair] = predictions

            logger.success(f"✓ {pair} training completed successfully")
            logger.info(f"\nLatest {pair} Predictions:")
            logger.info(predictions.tail(5))

        except Exception as e:
            logger.error(f"✗ {pair} training failed: {e}")
            all_results[pair] = None

    # Summary
    logger.info(f"\n{'='*80}")
    logger.info("TRAINING SUMMARY")
    logger.info(f"{'='*80}")

    successful = [pair for pair, result in all_results.items()
                  if result is not None]
    failed = [pair for pair, result in all_results.items() if result is None]

    logger.info(f"Successful: {len(successful)}/{len(CURRENCY_PAIRS)} pairs")
    if successful:
        logger.info(f"  - {', '.join(successful)}")

    if failed:
        logger.warning(f"Failed: {len(failed)} pairs")
        logger.warning(f"  - {', '.join(failed)}")

    # Generate portfolio-level signals if we have successful predictions
    if successful:
        logger.info(f"\n{'='*80}")
        logger.info("GENERATING PORTFOLIO-LEVEL SIGNALS")
        logger.info(f"{'='*80}\n")

        try:
            # Create portfolio ensemble
            portfolio = PortfolioEnsemble()

            # Add predictions from each successful pair
            for pair in successful:
                predictions = all_results[pair]
                portfolio.add_pair_prediction(pair, predictions)

            # Get aggregate signals (minimum 3 pairs must agree, 60% confidence)
            portfolio_signals = portfolio.get_aggregate_signals(
                min_confidence=0.60,
                min_agreement=3
            )

            # Save portfolio signals
            portfolio_output = RESULTS_DIR / "portfolio_signals.csv"
            portfolio.save_portfolio_signals(
                portfolio_signals,
                str(portfolio_output)
            )

            # Display statistics
            logger.info("\nPortfolio Statistics:")
            stats = portfolio.get_portfolio_statistics(portfolio_signals)
            logger.info(f"  Total Signals: {stats['total_signals']}")
            logger.info(f"  Buy: {stats['buy_pct']:.1f}%")
            logger.info(f"  Sell: {stats['sell_pct']:.1f}%")
            logger.info(f"  Hold: {stats['hold_pct']:.1f}%")
            logger.info(f"  Avg Confidence: {stats['avg_confidence']:.2f}")

            # Display correlation matrix
            logger.info("\nCurrency Pair Correlation Matrix:")
            corr_matrix = portfolio.get_correlation_matrix()
            logger.info(f"\n{corr_matrix.to_string()}")

            logger.success(
                f"\n✓ Portfolio signals saved to: {portfolio_output}")
            logger.info(f"\nLatest Portfolio Signals:")
            logger.info(portfolio_signals.tail(10))

        except Exception as e:
            logger.error(f"✗ Portfolio signal generation failed: {e}")

    return all_results


if __name__ == "__main__":
    main()
