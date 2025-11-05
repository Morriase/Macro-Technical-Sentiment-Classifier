"""
Data Preparation Script for Kaggle
Fetches and processes all required data for the forex classifier
Run this locally to generate the dataset, then upload to Kaggle
"""
from src.config import CURRENCY_PAIRS, DATA_DIR
from src.feature_engineering.sentiment_features import SentimentAnalyzer
from src.feature_engineering.technical_features import TechnicalFeatureEngineer
from src.data_acquisition.macro_data import MacroDataAcquisition
from src.data_acquisition.fx_data import FXDataAcquisition
import joblib
from loguru import logger
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


# Configure logger
logger.add("logs/data_preparation.log", rotation="1 day", retention="7 days")


class DataPreparationPipeline:
    """
    Prepares all data needed for Kaggle training
    """

    def __init__(
        self,
        currency_pairs: list = None,
        start_date: str = "2022-01-01",
        end_date: str = "2024-12-31",
        output_dir: str = "data/kaggle_dataset",
    ):
        """
        Initialize data preparation pipeline

        Args:
            currency_pairs: List of currency pairs (default: from config)
            start_date: Start date for data collection
            end_date: End date for data collection
            output_dir: Directory to save processed data
        """
        self.currency_pairs = currency_pairs or CURRENCY_PAIRS
        self.start_date = datetime.strptime(start_date, "%Y-%m-%d")
        self.end_date = datetime.strptime(end_date, "%Y-%m-%d")
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            f"Initialized data preparation for {len(self.currency_pairs)} pairs")
        logger.info(f"Date range: {start_date} to {end_date}")
        logger.info(f"Output directory: {self.output_dir}")

    def fetch_fx_data(self):
        """
        Fetch FX price data for all currency pairs
        """
        logger.info("=" * 80)
        logger.info("STEP 1: Fetching FX Price Data")
        logger.info("=" * 80)

        fx_client = FXDataAcquisition()
        fx_data_dir = self.output_dir / "fx_data"
        fx_data_dir.mkdir(exist_ok=True)

        for pair in self.currency_pairs:
            try:
                logger.info(f"Fetching data for {pair}...")

                # Fetch M5 candles
                df = fx_client.fetch_oanda_candles(
                    instrument=pair,
                    start_date=self.start_date,
                    end_date=self.end_date,
                    granularity="M5",
                )

                if df.empty:
                    logger.warning(f"No data fetched for {pair}")
                    continue

                # Validate data quality
                is_valid, accuracy = fx_client.validate_data_quality(df)
                logger.info(
                    f"{pair}: {len(df)} candles, accuracy: {accuracy:.2%}")

                # Save to parquet
                filename = fx_data_dir / f"{pair}_M5.parquet"
                df.to_parquet(filename, compression="gzip")
                logger.success(f"Saved {pair} data to {filename}")

            except Exception as e:
                logger.error(f"Error fetching {pair}: {e}")
                continue

        logger.success(f"FX data collection complete!")

    def fetch_macro_events(self):
        """
        Fetch economic calendar events for all relevant currencies
        """
        logger.info("=" * 80)
        logger.info("STEP 2: Fetching Macroeconomic Events")
        logger.info("=" * 80)

        macro_client = MacroDataAcquisition()
        macro_data_dir = self.output_dir / "macro_events"
        macro_data_dir.mkdir(exist_ok=True)

        for pair in self.currency_pairs:
            try:
                logger.info(f"Fetching events for {pair}...")

                # Get events for both currencies
                events_df = macro_client.get_events_for_currency_pair(
                    pair=pair,
                    start_date=self.start_date,
                    end_date=self.end_date,
                )

                if events_df.empty:
                    logger.warning(f"No events found for {pair}")
                    continue

                logger.info(f"{pair}: {len(events_df)} events")

                # Save to parquet
                filename = macro_data_dir / f"{pair}_events.parquet"
                events_df.to_parquet(filename, compression="gzip")
                logger.success(f"Saved {pair} events to {filename}")

            except Exception as e:
                logger.error(f"Error fetching events for {pair}: {e}")
                continue

        logger.success(f"Macro events collection complete!")

    def generate_technical_features(self):
        """
        Generate technical features for all pairs
        (Pre-compute to save time on Kaggle)
        """
        logger.info("=" * 80)
        logger.info("STEP 3: Generating Technical Features")
        logger.info("=" * 80)

        fx_data_dir = self.output_dir / "fx_data"
        features_dir = self.output_dir / "technical_features"
        features_dir.mkdir(exist_ok=True)

        tech_engineer = TechnicalFeatureEngineer()

        for pair in self.currency_pairs:
            try:
                # Load FX data
                fx_file = fx_data_dir / f"{pair}_M5.parquet"
                if not fx_file.exists():
                    logger.warning(f"No FX data found for {pair}")
                    continue

                logger.info(f"Processing technical features for {pair}...")
                df = pd.read_parquet(fx_file)

                # Generate features
                features_df = tech_engineer.calculate_all_features(df)

                logger.info(
                    f"{pair}: {len(features_df.columns)} features generated")

                # Save features
                filename = features_dir / f"{pair}_technical_features.parquet"
                features_df.to_parquet(filename, compression="gzip")
                logger.success(f"Saved {pair} features to {filename}")

            except Exception as e:
                logger.error(f"Error generating features for {pair}: {e}")
                continue

        logger.success(f"Technical features generation complete!")

    def prepare_sentiment_corpus(self):
        """
        Prepare sample news corpus for sentiment analysis
        (On Kaggle, you can use real-time news or pre-collected corpus)
        """
        logger.info("=" * 80)
        logger.info("STEP 4: Preparing Sentiment Corpus")
        logger.info("=" * 80)

        sentiment_dir = self.output_dir / "sentiment_data"
        sentiment_dir.mkdir(exist_ok=True)

        # Create sample news corpus structure
        # In production, replace with actual news data from:
        # - Reuters API
        # - Bloomberg API
        # - Finnhub news endpoint
        # - Web scraping

        sample_corpus = pd.DataFrame({
            "date": pd.date_range(self.start_date, self.end_date, freq="D"),
            "currency": ["EUR"] * ((self.end_date - self.start_date).days + 1),
            "headline": ["Sample news headline"] * ((self.end_date - self.start_date).days + 1),
            "content": ["Sample news content"] * ((self.end_date - self.start_date).days + 1),
        })

        filename = sentiment_dir / "news_corpus_template.parquet"
        sample_corpus.to_parquet(filename, compression="gzip")

        logger.warning(
            "Created template news corpus. Replace with actual news data for production."
        )
        logger.info(f"Saved template to {filename}")

    def create_dataset_metadata(self):
        """
        Create metadata file describing the dataset
        """
        logger.info("=" * 80)
        logger.info("STEP 5: Creating Dataset Metadata")
        logger.info("=" * 80)

        metadata = {
            "dataset_name": "Forex ML Training Data",
            "version": "1.0",
            "created_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "date_range": {
                "start": self.start_date.strftime("%Y-%m-%d"),
                "end": self.end_date.strftime("%Y-%m-%d"),
            },
            "currency_pairs": self.currency_pairs,
            "data_sources": {
                "fx_data": "OANDA v20 API",
                "macro_events": "Finnhub Economic Calendar",
                "sentiment": "Template (replace with actual data)",
            },
            "file_structure": {
                "fx_data/": "Raw FX price data (M5 candles)",
                "macro_events/": "Economic calendar events with surprise scores",
                "technical_features/": "Pre-computed technical indicators",
                "sentiment_data/": "News corpus template",
            },
            "usage": "Load with pandas.read_parquet() in Kaggle notebook",
        }

        # Save as JSON
        import json

        filename = self.output_dir / "dataset_metadata.json"
        with open(filename, "w") as f:
            json.dump(metadata, f, indent=2)

        logger.success(f"Metadata saved to {filename}")

        # Create README
        readme_content = f"""# Forex ML Training Dataset

## Overview
Prepared dataset for the Macro-Technical Sentiment Classifier project.

## Date Range
- Start: {self.start_date.strftime('%Y-%m-%d')}
- End: {self.end_date.strftime('%Y-%m-%d')}

## Currency Pairs
{', '.join(self.currency_pairs)}

## File Structure

### fx_data/
Raw FX price data (M5 candles) from OANDA
- Format: Parquet (gzip compressed)
- Columns: timestamp, open, high, low, close, volume

### macro_events/
Economic calendar events from Finnhub
- Format: Parquet (gzip compressed)
- Columns: timestamp, event_name, actual_value, consensus_forecast, surprise_zscore

### technical_features/
Pre-computed technical indicators (TA-Lib)
- Format: Parquet (gzip compressed)
- Includes: EMA, RSI, ATR, MACD, Bollinger Bands, Stochastic, etc.

### sentiment_data/
News corpus template (replace with actual data)
- Format: Parquet (gzip compressed)
- Template structure provided

## Usage in Kaggle

```python
import pandas as pd

# Load FX data
eur_usd = pd.read_parquet('/kaggle/input/forex-training-data/fx_data/EUR_USD_M5.parquet')

# Load events
events = pd.read_parquet('/kaggle/input/forex-training-data/macro_events/EUR_USD_events.parquet')

# Load technical features
features = pd.read_parquet('/kaggle/input/forex-training-data/technical_features/EUR_USD_technical_features.parquet')
```

## Data Sources
- **FX Prices**: OANDA v20 API (https://www.oanda.com/)
- **Economic Events**: Finnhub API (https://finnhub.io/)
- **Technical Indicators**: TA-Lib (https://ta-lib.org/)

## Size Information
Run `du -sh *` to check directory sizes before uploading to Kaggle.

## License
Data for educational/research purposes only. Check individual data source licenses.

## Generated By
Macro-Technical Sentiment Classifier Data Preparation Pipeline
Repository: https://github.com/Morriase/Macro-Technical-Sentiment-Classifier
"""

        readme_file = self.output_dir / "README.md"
        with open(readme_file, "w") as f:
            f.write(readme_content)

        logger.success(f"README saved to {readme_file}")

    def generate_summary_report(self):
        """
        Generate summary report of collected data
        """
        logger.info("=" * 80)
        logger.info("FINAL: Generating Summary Report")
        logger.info("=" * 80)

        report = []
        report.append("=" * 80)
        report.append("DATA PREPARATION SUMMARY")
        report.append("=" * 80)
        report.append(
            f"Date Range: {self.start_date.date()} to {self.end_date.date()}")
        report.append(f"Currency Pairs: {len(self.currency_pairs)}")
        report.append("")

        # Check each data directory
        for subdir in ["fx_data", "macro_events", "technical_features", "sentiment_data"]:
            dir_path = self.output_dir / subdir
            if dir_path.exists():
                files = list(dir_path.glob("*.parquet"))
                total_size = sum(
                    f.stat().st_size for f in files) / (1024 * 1024)  # MB
                report.append(f"{subdir}:")
                report.append(f"  Files: {len(files)}")
                report.append(f"  Size: {total_size:.2f} MB")
            else:
                report.append(f"{subdir}: NOT CREATED")
            report.append("")

        report.append("=" * 80)
        report.append("NEXT STEPS:")
        report.append("1. Review the data in: " + str(self.output_dir))
        report.append(
            "2. Zip the directory: zip -r forex_training_data.zip " + str(self.output_dir))
        report.append("3. Upload to Kaggle: https://www.kaggle.com/datasets")
        report.append(
            "4. Update paths in src/data_acquisition/kaggle_loader.py")
        report.append("=" * 80)

        report_text = "\n".join(report)
        print("\n" + report_text)

        # Save report
        report_file = self.output_dir / "PREPARATION_REPORT.txt"
        with open(report_file, "w") as f:
            f.write(report_text)

        logger.success(f"Summary report saved to {report_file}")

    def run(self):
        """
        Run complete data preparation pipeline
        """
        logger.info("Starting data preparation pipeline...")
        start_time = datetime.now()

        try:
            # Step 1: Fetch FX data
            self.fetch_fx_data()

            # Step 2: Fetch macro events
            self.fetch_macro_events()

            # Step 3: Generate technical features
            self.generate_technical_features()

            # Step 4: Prepare sentiment corpus
            self.prepare_sentiment_corpus()

            # Step 5: Create metadata
            self.create_dataset_metadata()

            # Final: Generate summary
            self.generate_summary_report()

            elapsed = datetime.now() - start_time
            logger.success(
                f"Data preparation complete! Elapsed time: {elapsed}")

        except Exception as e:
            logger.error(f"Data preparation failed: {e}")
            raise


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Prepare data for Kaggle")
    parser.add_argument(
        "--start-date",
        type=str,
        default="2022-01-01",
        help="Start date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default="2024-12-31",
        help="End date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/kaggle_dataset",
        help="Output directory",
    )
    parser.add_argument(
        "--pairs",
        type=str,
        nargs="+",
        default=None,
        help="Currency pairs (default: from config)",
    )

    args = parser.parse_args()

    # Initialize and run pipeline
    pipeline = DataPreparationPipeline(
        currency_pairs=args.pairs,
        start_date=args.start_date,
        end_date=args.end_date,
        output_dir=args.output_dir,
    )

    pipeline.run()

    print("\n✅ Data preparation complete!")
    print(f"📁 Dataset ready in: {args.output_dir}")
    print("📤 Next: Upload to Kaggle Datasets")
