"""
Kaggle News Data Loader for Historical Sentiment Analysis
Uses kagglehub to automatically download the dataset.
"""
import pandas as pd
from pathlib import Path
from loguru import logger
from src.config import DATA_DIR, IS_KAGGLE
from datetime import datetime
import kagglehub
import zipfile
import os


class KaggleNewsLoader:
    """
    Loads historical news data from the Kaggle 'Daily Financial News for 6000+ Stocks' dataset.
    Handles automatic download and extraction using kagglehub.
    """

    def __init__(self, data_dir: Path = DATA_DIR):
        """
        Initialize the news loader.

        Args:
            data_dir: Path to the data directory.
        """
        self.dataset_name = "miguelaenlle/massive-stock-news-analysis-db-for-nlpbacktests"

        # On Kaggle, use the attached input dataset (already available)
        if IS_KAGGLE:
            self.download_dir = Path(
                "/kaggle/input/massive-stock-news-analysis-db-for-nlpbacktests")
        else:
            self.download_dir = data_dir / "kaggle_dataset" / "massive-stock-news"

        self.news_filepath = self.download_dir / "analyst_ratings_processed.csv"
        logger.info(f"Kaggle News Dataset: {self.dataset_name}")
        logger.info(f"Kaggle News Data Directory: {self.download_dir}")

    def _download_and_unzip_data(self):
        """
        Downloads and unzips the Kaggle dataset if it doesn't already exist.
        On Kaggle, the dataset is attached as an input, so no download is needed.
        Returns False if dataset is not available (optional feature).
        """
        # On Kaggle, dataset is already attached - just verify it exists
        if IS_KAGGLE:
            if self.news_filepath.exists():
                logger.info("Using attached Kaggle news dataset.")
                return True
            else:
                logger.warning(f"News dataset not found at {self.news_filepath}")
                logger.warning(
                    "News sentiment features will be disabled. To enable, attach 'massive-stock-news-analysis-db-for-nlpbacktests' as input.")
                return False

        # Local environment: download if needed
        if self.news_filepath.exists():
            logger.info(
                "Kaggle news dataset already downloaded and extracted.")
            return True

        logger.info(
            f"Downloading dataset '{self.dataset_name}' to '{self.download_dir}'...")
        try:
            # Ensure the download directory exists
            self.download_dir.mkdir(parents=True, exist_ok=True)

            # Download the dataset
            path = kagglehub.dataset_download(
                self.dataset_name, path=self.download_dir)

            # Unzip the file
            with zipfile.ZipFile(path, 'r') as zip_ref:
                zip_ref.extractall(self.download_dir)

            # Remove the zip file after extraction
            os.remove(path)

            logger.success("Dataset downloaded and extracted successfully.")
            return True

        except Exception as e:
            logger.warning(f"Failed to download or extract Kaggle dataset: {e}")
            logger.warning(
                "News sentiment features will be disabled. To enable, set up Kaggle API credentials.")
            logger.warning("See: https://www.kaggle.com/docs/api")
            return False

    def load_historical_news(self, start_date: datetime = None, end_date: datetime = None) -> pd.DataFrame:
        """
        Loads historical news headlines from the Kaggle dataset.

        Args:
            start_date: Optional start date to filter news.
            end_date: Optional end date to filter news.

        Returns:
            A pandas DataFrame containing historical news articles,
            with 'date' and 'text' columns. Returns empty DataFrame if dataset unavailable.
        """
        # Ensure data is downloaded (returns False if unavailable)
        if not self._download_and_unzip_data():
            logger.info("News dataset unavailable - returning empty DataFrame")
            return pd.DataFrame()

        if not self.news_filepath.exists():
            logger.warning(
                f"Kaggle news file not found after download attempt: {self.news_filepath}")
            return pd.DataFrame()

        logger.info(f"Loading historical news from {self.news_filepath}")
        try:
            # Read only necessary columns to save memory
            df_news = pd.read_csv(
                self.news_filepath,
                usecols=['date', 'title']
            )
            df_news.rename(columns={'title': 'text'}, inplace=True)

            # Convert date column to datetime (handle string format)
            df_news['date'] = pd.to_datetime(df_news['date'], errors='coerce')

            # Drop rows with invalid dates
            df_news = df_news.dropna(subset=['date'])

            df_news.set_index('date', inplace=True)
            df_news.sort_index(inplace=True)

            initial_count = len(df_news)

            # Convert filter dates to timezone-naive if they have timezone info
            if start_date:
                if hasattr(start_date, 'tz') and start_date.tz is not None:
                    start_date = start_date.tz_localize(None)
                df_news = df_news[df_news.index >= start_date]
            if end_date:
                if hasattr(end_date, 'tz') and end_date.tz is not None:
                    end_date = end_date.tz_localize(None)
                df_news = df_news[df_news.index <= end_date]

            logger.info(
                f"Loaded {len(df_news)} news articles (filtered from {initial_count} total).")
            logger.info(
                f"Date range: {df_news.index.min()} to {df_news.index.max()}")

            return df_news

        except Exception as e:
            logger.error(f"Error loading Kaggle news data: {e}")
            return pd.DataFrame()


if __name__ == "__main__":
    # Example usage
    # Note: This will download the dataset which is ~400MB
    # It requires Kaggle API credentials to be set up.
    logger.info("Running KaggleNewsLoader example...")
    try:
        news_loader = KaggleNewsLoader()
        all_news_df = news_loader.load_historical_news()
        if not all_news_df.empty:
            print("\nAll News Head:")
            print(all_news_df.head())
            print(f"Total News Articles: {len(all_news_df)}")
    except Exception as e:
        logger.error(f"Example failed: {e}")
