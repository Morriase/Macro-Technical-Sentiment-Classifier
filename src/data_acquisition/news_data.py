"""
News Data Acquisition from Finnhub
"""
import finnhub
import pandas as pd
from datetime import datetime, timedelta
from loguru import logger
from src.config import FINNHUB_API_KEY, CURRENCY_PAIRS


class NewsDataAcquisition:
    """
    Acquires financial news data from Finnhub.
    """

    def __init__(self):
        """
        Initializes the Finnhub client.
        """
        if not FINNHUB_API_KEY:
            logger.error("FINNHUB_API_KEY not found in environment variables. News data acquisition will not work.")
            self.client = None
        else:
            self.client = finnhub.Client(api_key=FINNHUB_API_KEY)
        logger.info("NewsDataAcquisition initialized.")

    def fetch_market_news(self, start_date: datetime, end_date: datetime, category: str = "general") -> pd.DataFrame:
        """
        Fetches general market news from Finnhub for a given date range.

        Args:
            start_date: The start date for news articles.
            end_date: The end date for news articles.
            category: News category (e.g., "general", "forex", "crypto").

        Returns:
            A pandas DataFrame containing the fetched news articles.
        """
        if not self.client:
            logger.warning("Finnhub client not initialized due to missing API key. Cannot fetch market news.")
            return pd.DataFrame()

        logger.info(f"Fetching '{category}' market news from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
        
        try:
            news_data = self.client.market_news(category, _from=start_date.strftime('%Y-%m-%d'), to=end_date.strftime('%Y-%m-%d'))
            if not news_data:
                logger.info(f"No '{category}' market news found for the specified date range.")
                return pd.DataFrame()

            df = pd.DataFrame(news_data)
            df['datetime'] = pd.to_datetime(df['datetime'], unit='s')
            df.rename(columns={'datetime': 'date', 'headline': 'text'}, inplace=True)
            df.set_index('date', inplace=True)
            logger.info(f"Fetched {len(df)} '{category}' market news articles.")
            return df
        except finnhub.exceptions.FinnhubAPIException as e:
            logger.error(f"Finnhub API error fetching market news: {e}")
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"An unexpected error occurred while fetching market news: {e}")
            return pd.DataFrame()

    def fetch_company_news(self, symbol: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """
        Fetches company news for a given stock symbol from Finnhub.
        (Note: Finnhub's free tier primarily supports US stocks for company news.
        Forex pairs are not directly supported as 'companies'.)

        Args:
            symbol: The stock symbol (e.g., "AAPL").
            start_date: The start date for news articles.
            end_date: The end date for news articles.

        Returns:
            A pandas DataFrame containing the fetched company news articles.
        """
        if not self.client:
            logger.warning("Finnhub client not initialized due to missing API key. Cannot fetch company news.")
            return pd.DataFrame()

        logger.info(f"Fetching company news for '{symbol}' from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")

        try:
            news_data = self.client.company_news(symbol, _from=start_date.strftime('%Y-%m-%d'), to=end_date.strftime('%Y-%m-%d'))
            if not news_data:
                logger.info(f"No company news found for '{symbol}' for the specified date range.")
                return pd.DataFrame()

            df = pd.DataFrame(news_data)
            df['datetime'] = pd.to_datetime(df['datetime'], unit='s')
            df.rename(columns={'datetime': 'date', 'headline': 'text'}, inplace=True)
            df.set_index('date', inplace=True)
            logger.info(f"Fetched {len(df)} company news articles for '{symbol}'.")
            return df
        except finnhub.exceptions.FinnhubAPIException as e:
            logger.error(f"Finnhub API error fetching company news for '{symbol}': {e}")
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"An unexpected error occurred while fetching company news for '{symbol}': {e}")
            return pd.DataFrame()

    def fetch_forex_news(self, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """
        Fetches forex-specific news from Finnhub.

        Args:
            start_date: The start date for news articles.
            end_date: The end date for news articles.

        Returns:
            A pandas DataFrame containing the fetched forex news articles.
        """
        return self.fetch_market_news(start_date, end_date, category="forex")


if __name__ == "__main__":
    # Example Usage
    news_acq = NewsDataAcquisition()
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=7) # Last 7 days

    # Fetch general market news
    market_news_df = news_acq.fetch_market_news(start_date, end_date)
    if not market_news_df.empty:
        print("\nMarket News Head:")
        print(market_news_df.head())
        print(f"Total Market News: {len(market_news_df)}")

    # Fetch forex news
    forex_news_df = news_acq.fetch_forex_news(start_date, end_date)
    if not forex_news_df.empty:
        print("\nForex News Head:")
        print(forex_news_df.head())
        print(f"Total Forex News: {len(forex_news_df)}")

    # Example of fetching company news (might not be relevant for forex pairs directly)
    # company_news_df = news_acq.fetch_company_news("AAPL", start_date, end_date)
    # if not company_news_df.empty:
    #     print("\nAAPL News Head:")
    #     print(company_news_df.head())
    #     print(f"Total AAPL News: {len(company_news_df)}")
