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

        # On Kaggle, use /kaggle/working (writable) instead of /kaggle/input (read-only)
        if IS_KAGGLE:
            self.download_dir = Path("/kaggle/working") / "massive-stock-news"
        else:
            self.download_dir = data_dir / "kaggle_dataset" / "massive-stock-news"

        self.news_filepath = self.download_dir / "analyst_ratings_processed.csv"
        logger.info(f"Kaggle News Dataset: {self.dataset_name}")
        logger.info(f"Kaggle News Download Directory: {self.download_dir}")

    def _download_and_unzip_data(self):
        """
        Downloads and unzips the Kaggle dataset if it doesn't already exist.
        """
        if self.news_filepath.exists():
            logger.info(
                "Kaggle news dataset already downloaded and extracted.")
            return

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

        except Exception as e:
            logger.error(f"Failed to download or extract Kaggle dataset: {e}")
            logger.error(
                "Please ensure you have set up your Kaggle API credentials.")
            logger.error("See: https://www.kaggle.com/docs/api")
            raise

    def load_historical_news(self, start_date: datetime = None, end_date: datetime = None) -> pd.DataFrame:
        """
        Loads historical news headlines from the Kaggle dataset.

        Args:
            start_date: Optional start date to filter news.
            end_date: Optional end date to filter news.

        Returns:
            A pandas DataFrame containing historical news articles,
            with 'date' and 'text' columns.
        """
        # Ensure data is downloaded
        self._download_and_unzip_data()

        if not self.news_filepath.exists():
            logger.error(
                f"Kaggle news file not found after download attempt: {self.news_filepath}")
            raise FileNotFoundError(
                f"Kaggle news file not found: {self.news_filepath}")

        logger.info(f"Loading historical news from {self.news_filepath}")
        try:
            # Read only necessary columns to save memory
            df_news = pd.read_csv(
                self.news_filepath,
                usecols=['date', 'title'],
                parse_dates=['date'],
                dayfirst=False
            )
            df_news.rename(columns={'title': 'text'}, inplace=True)
            df_news.set_index('date', inplace=True)
            df_news.sort_index(inplace=True)

            initial_count = len(df_news)

            if start_date:
                df_news = df_news[df_news.index >= start_date]
            if end_date:
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
