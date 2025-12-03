"""
Kaggle News Data Loader for Historical Sentiment Analysis
Training happens exclusively on Kaggle - uses attached dataset directly.

Dataset: massive-stock-news-analysis-db-for-nlpbacktests
Files available (Parquet preferred, CSV fallback):
  - raw_partner_headlines.parquet (partner news headlines - best for sentiment)
  - analyst_ratings_processed.parquet (analyst ratings with headlines)
  - raw_analyst_ratings.parquet (raw analyst data)
"""
import pandas as pd
from pathlib import Path
from loguru import logger
from src.config import DATA_DIR, IS_KAGGLE
from datetime import datetime


class KaggleNewsLoader:
    """
    Loads historical news data from the Kaggle 'massive-stock-news-analysis-db' dataset.
    Training runs exclusively on Kaggle - dataset is pre-attached as input.
    Includes forex-specific filtering for currency pair analysis.

    Supports both Parquet (faster) and CSV formats.

    Dataset path on Kaggle: /kaggle/input/massive-stock-news-analysis-db-for-nlpbacktests
    """

    # Forex/currency-related keywords for filtering news
    FOREX_KEYWORDS = [
        # Currency names
        'forex', 'currency', 'exchange rate', 'fx market', 'foreign exchange',
        'dollar', 'euro', 'pound', 'yen', 'franc', 'yuan', 'renminbi',
        # Currency codes
        'usd', 'eur', 'gbp', 'jpy', 'chf', 'cad', 'aud', 'nzd', 'cny',
        # Central banks
        'federal reserve', 'fed', 'ecb', 'european central bank',
        'bank of england', 'boe', 'bank of japan', 'boj',
        'swiss national bank', 'snb', 'reserve bank', 'rba', 'rbnz',
        # Monetary policy
        'interest rate', 'rate hike', 'rate cut', 'monetary policy',
        'quantitative easing', 'qe', 'tightening', 'dovish', 'hawkish',
        'tapering', 'stimulus', 'bond buying', 'yield curve',
        # Economic indicators
        'inflation', 'cpi', 'gdp', 'employment', 'unemployment', 'jobs report',
        'nfp', 'non-farm payroll', 'retail sales', 'trade balance',
        'current account', 'pmi', 'manufacturing', 'services', 'consumer confidence',
        'housing', 'durable goods', 'industrial production',
        # Forex-specific events
        'devaluation', 'revaluation', 'peg', 'float', 'intervention',
        'capital flow', 'safe haven', 'risk appetite', 'risk-off', 'risk-on',
        # Geopolitical
        'tariff', 'trade war', 'sanctions', 'geopolitical', 'treasury',
    ]

    # Available files in the dataset (priority order) - Parquet preferred
    NEWS_FILES = [
        # Best for news sentiment
        ('raw_partner_headlines', 'headline', 'date'),
        ('analyst_ratings_processed', 'title', 'date'),     # Analyst ratings
        ('raw_analyst_ratings', 'headline', 'date'),        # Raw analyst data
    ]

    def __init__(self, data_dir: Path = DATA_DIR, enable_forex_filter: bool = True):
        """
        Initialize the news loader for Kaggle environment.

        Args:
            data_dir: Path to the data directory (unused on Kaggle)
            enable_forex_filter: If True, filter news to forex-related articles only
        """
        self.enable_forex_filter = enable_forex_filter

        # Kaggle dataset path (training happens exclusively on Kaggle)
        # News data is now part of the main macros-and-ohlc dataset
        self.kaggle_path = Path(
            "/kaggle/input/macros-and-ohlc/data/kaggle_dataset/massive-stock-news")

        # For local testing only (not used in production)
        self.local_path = data_dir / "kaggle_dataset" / "massive-stock-news"

        # Determine active path
        if IS_KAGGLE:
            self.download_dir = self.kaggle_path
            logger.info(f"📰 Kaggle News Dataset: {self.kaggle_path}")
        else:
            self.download_dir = self.local_path
            logger.info(
                f"📰 Local News Dataset: {self.local_path} (for testing only)")

        if self.enable_forex_filter:
            logger.info("🔍 Forex-specific news filtering: ENABLED")

    def _find_available_news_file(self) -> tuple:
        """
        Find the first available news file from priority list.
        Prefers Parquet format (faster loading) over CSV.

        Returns:
            Tuple of (filepath, text_column, date_column, is_parquet) or (None, None, None, False)
        """
        for basename, text_col, date_col in self.NEWS_FILES:
            # Try Parquet first (faster)
            parquet_path = self.download_dir / f"{basename}.parquet"
            if parquet_path.exists():
                logger.info(
                    f"✓ Found news file: {basename}.parquet (fast loading)")
                return parquet_path, text_col, date_col, True

            # Fall back to CSV
            csv_path = self.download_dir / f"{basename}.csv"
            if csv_path.exists():
                logger.info(f"✓ Found news file: {basename}.csv")
                return csv_path, text_col, date_col, False

        return None, None, None, False

    def _check_dataset_available(self) -> bool:
        """
        Check if the news dataset is available.
        On Kaggle, it must be attached as an input dataset.

        Returns:
            True if dataset is available, False otherwise
        """
        if not self.download_dir.exists():
            if IS_KAGGLE:
                logger.warning(
                    "❌ News dataset not found! Expected at:")
                logger.warning(
                    "   /kaggle/input/macros-and-ohlc/data/kaggle_dataset/massive-stock-news")
                logger.warning(
                    "   Ensure 'macros-and-ohlc' dataset is attached with news parquet files")
            else:
                logger.warning(
                    f"❌ News dataset not found at: {self.download_dir}")
            return False

        # Check if any news file exists
        filepath, _, _, _ = self._find_available_news_file()
        if filepath is None:
            logger.warning("❌ No news CSV files found in dataset directory")
            logger.warning(
                f"   Expected files: {[f[0] for f in self.NEWS_FILES]}")
            return False

        return True

    def _filter_forex_news(self, df: pd.DataFrame, text_column: str = 'text') -> pd.DataFrame:
        """
        Filter news articles to keep only forex/currency-related content.

        Args:
            df: DataFrame with news articles
            text_column: Column name containing the text to filter

        Returns:
            Filtered DataFrame with only forex-related news
        """
        if df.empty:
            return df

        initial_count = len(df)

        # Create case-insensitive pattern matching for any forex keyword
        pattern = '|'.join(self.FOREX_KEYWORDS)

        # Filter rows where text contains any forex keyword
        mask = df[text_column].str.contains(
            pattern,
            case=False,
            na=False,
            regex=True
        )

        df_filtered = df[mask].copy()

        filtered_count = len(df_filtered)
        logger.info(
            f"Forex filter: {filtered_count}/{initial_count} articles retained "
            f"({filtered_count/initial_count*100:.1f}% match rate)"
        )

        return df_filtered

    def load_historical_news(self, start_date: datetime = None, end_date: datetime = None) -> pd.DataFrame:
        """
        Loads historical news headlines from the Kaggle dataset.
        Automatically detects Parquet (preferred) or CSV files.

        Args:
            start_date: Optional start date to filter news.
            end_date: Optional end date to filter news.

        Returns:
            A pandas DataFrame containing historical news articles,
            with 'date' and 'text' columns. Returns empty DataFrame if dataset unavailable.
        """
        # Check if dataset is available
        if not self._check_dataset_available():
            logger.warning(
                "⚠ News sentiment features will be disabled for this training run")
            return pd.DataFrame()

        # Find the best available news file
        filepath, text_col, date_col, is_parquet = self._find_available_news_file()
        if filepath is None:
            return pd.DataFrame()

        logger.info(f"📰 Loading news from: {filepath.name}")
        try:
            # Load data based on format
            if is_parquet:
                # Parquet - fast loading
                df_news = pd.read_parquet(
                    filepath, columns=[date_col, text_col])
            else:
                # CSV - slower but fallback
                try:
                    df_news = pd.read_csv(
                        filepath,
                        usecols=[date_col, text_col],
                        low_memory=False
                    )
                except ValueError:
                    # If columns don't exist, load all and find alternatives
                    logger.warning(
                        f"Expected columns [{date_col}, {text_col}] not found, scanning file...")
                    df_news = pd.read_csv(filepath, nrows=5)
                    logger.info(f"Available columns: {list(df_news.columns)}")

                    # Try to find date and text columns
                    date_candidates = ['date', 'Date', 'DATE',
                                       'timestamp', 'time', 'published']
                    text_candidates = ['headline', 'title', 'text',
                                       'content', 'news', 'Headline', 'Title']

                    date_col = next(
                        (c for c in date_candidates if c in df_news.columns), None)
                    text_col = next(
                        (c for c in text_candidates if c in df_news.columns), None)

                    if date_col is None or text_col is None:
                        logger.error(
                            f"Could not find date/text columns in {filepath.name}")
                        return pd.DataFrame()

                    df_news = pd.read_csv(
                        filepath, usecols=[date_col, text_col], low_memory=False)

            # Standardize column names
            df_news = df_news.rename(
                columns={text_col: 'text', date_col: 'date'})

            # Convert date column to datetime (handle various formats)
            df_news['date'] = pd.to_datetime(
                df_news['date'], errors='coerce', utc=True)

            # Convert to timezone-naive for consistency
            if df_news['date'].dt.tz is not None:
                df_news['date'] = df_news['date'].dt.tz_localize(None)

            # Drop rows with invalid dates or empty text
            df_news = df_news.dropna(subset=['date', 'text'])
            # Filter very short texts
            df_news = df_news[df_news['text'].str.len() > 10]

            df_news.set_index('date', inplace=True)
            df_news.sort_index(inplace=True)

            initial_count = len(df_news)
            logger.info(f"📊 Total articles in dataset: {initial_count:,}")

            # Apply date range filter
            if start_date:
                if hasattr(start_date, 'tzinfo') and start_date.tzinfo is not None:
                    start_date = start_date.replace(tzinfo=None)
                df_news = df_news[df_news.index >= start_date]
            if end_date:
                if hasattr(end_date, 'tzinfo') and end_date.tzinfo is not None:
                    end_date = end_date.replace(tzinfo=None)
                df_news = df_news[df_news.index <= end_date]

            if not df_news.empty:
                logger.info(
                    f"📅 Date range: {df_news.index.min().date()} to {df_news.index.max().date()}")
                logger.info(
                    f"📰 Articles in range: {len(df_news):,} (from {initial_count:,} total)")

            # Apply forex-specific filtering if enabled
            if self.enable_forex_filter and not df_news.empty:
                df_news_reset = df_news.reset_index()
                df_news_reset = self._filter_forex_news(
                    df_news_reset, text_column='text')
                df_news = df_news_reset.set_index('date')

                if df_news.empty:
                    logger.warning(
                        "⚠ No forex-related news found after filtering. "
                        "This may reduce model accuracy for FX prediction."
                    )
                else:
                    logger.success(
                        f"✓ {len(df_news):,} forex-related articles ready for sentiment analysis"
                    )

            return df_news

        except Exception as e:
            logger.error(f"❌ Error loading news data: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return pd.DataFrame()

    def load_all_news_sources(self, start_date: datetime = None, end_date: datetime = None) -> pd.DataFrame:
        """
        Load and combine news from all available CSV files in the dataset.
        Useful for maximizing news coverage.

        Args:
            start_date: Optional start date filter
            end_date: Optional end date filter

        Returns:
            Combined DataFrame with news from all sources
        """
        all_news = []

        for filename, text_col, date_col in self.NEWS_FILES:
            filepath = self.download_dir / filename
            if not filepath.exists():
                continue

            logger.info(f"Loading from {filename}...")
            try:
                df = pd.read_csv(filepath, usecols=[
                                 date_col, text_col], low_memory=False)
                df = df.rename(columns={text_col: 'text', date_col: 'date'})
                df['source'] = filename
                all_news.append(df)
                logger.info(f"  → Loaded {len(df):,} articles")
            except Exception as e:
                logger.warning(f"  → Failed to load {filename}: {e}")

        if not all_news:
            return pd.DataFrame()

        # Combine all sources
        df_combined = pd.concat(all_news, ignore_index=True)

        # Process dates
        df_combined['date'] = pd.to_datetime(
            df_combined['date'], errors='coerce')
        df_combined = df_combined.dropna(subset=['date', 'text'])

        # Remove duplicates based on text content
        df_combined = df_combined.drop_duplicates(
            subset=['text'], keep='first')

        logger.success(
            f"✓ Combined {len(df_combined):,} unique articles from {len(all_news)} sources")

        # Apply date and forex filters
        if start_date:
            df_combined = df_combined[df_combined['date'] >= start_date]
        if end_date:
            df_combined = df_combined[df_combined['date'] <= end_date]

        if self.enable_forex_filter:
            df_combined = self._filter_forex_news(
                df_combined, text_column='text')

        df_combined.set_index('date', inplace=True)
        df_combined.sort_index(inplace=True)

        return df_combined


if __name__ == "__main__":
    # Example usage (for testing on Kaggle)
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
