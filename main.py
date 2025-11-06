"""
Main Pipeline for Macro-Technical Sentiment Forex Classifier
End-to-end training and prediction workflow
Optimized for both local and Kaggle (GPU/CUDA) environments
"""
from src.validation.walk_forward import WalkForwardOptimizer
from src.models.hybrid_ensemble import HybridEnsemble
from src.feature_engineering.sentiment_features import SentimentAnalyzer
from src.feature_engineering.technical_features import TechnicalFeatureEngineer
from src.data_acquisition.kaggle_loader import KaggleFXDataLoader

# Optional imports for local environment (not needed on Kaggle)
try:
    from src.data_acquisition.macro_data import MacroDataAcquisition
    from src.data_acquisition.fx_data import FXDataAcquisition
    HAS_API_SOURCES = True
except ImportError:
    HAS_API_SOURCES = False
    MacroDataAcquisition = None
    FXDataAcquisition = None
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
)
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from loguru import logger
import sys

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
            self.fx_data = FXDataAcquisition()
            self.macro_data = MacroDataAcquisition()
            logger.info("Using OANDA/API data sources")

        self.tech_engineer = TechnicalFeatureEngineer()
        self.sentiment_analyzer = SentimentAnalyzer()

        self.df_price = None
        self.df_events = None
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
        logger.info("Calculating technical features")
        self.df_features = self.tech_engineer.calculate_all_features(
            self.df_price.copy())
        self.df_features = self.tech_engineer.calculate_feature_crosses(
            self.df_features)

        # Macro features (temporal proximity)
        if self.df_events is not None and not self.df_events.empty and self.macro_data:
            logger.info("Calculating macro proximity features")
            self.df_features = self.macro_data.calculate_temporal_proximity(
                events_df=self.df_events,
                price_df=self.df_features,
            )
        else:
            logger.warning(
                "No macro events available or macro_data not initialized")
            self.df_features["tau_pre"] = 0.0
            self.df_features["tau_post"] = 0.0
            self.df_features["weighted_surprise"] = 0.0

        # Drop NaN
        initial_len = len(self.df_features)
        self.df_features.dropna(inplace=True)
        logger.info(
            f"Features engineered. Dropped {initial_len - len(self.df_features)} rows with NaN")

        logger.info(f"Total features: {len(self.df_features.columns)}")

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
            "target", "target_class"
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
        model_path = MODELS_DIR / f"{self.currency_pair}_model.pth"
        self.model.save_model(str(model_path))
        logger.info(f"Model saved to {model_path}")

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
    """Main entry point"""
    pipeline = ForexClassifierPipeline(currency_pair="EUR_USD")

    # Run full pipeline
    predictions = pipeline.run_full_pipeline(
        years_history=5,
        use_walk_forward=True,
    )

    print("\n" + "="*60)
    print("Latest Predictions:")
    print("="*60)
    print(predictions.tail(10))


if __name__ == "__main__":
    main()
