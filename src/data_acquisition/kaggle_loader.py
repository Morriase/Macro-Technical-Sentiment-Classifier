"""
Data Loading Utilities for Kaggle Environment
Handles loading FX data and macro events from Kaggle dataset with GPU optimization
"""
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional
from loguru import logger
from src.config import DATA_DIR, IS_KAGGLE, GPU_CONFIG
import torch
from torch.utils.data import Dataset, DataLoader


class KaggleFXDataLoader:
    """Load FX data and macro events from Kaggle dataset"""

    def __init__(self, data_dir: Path = DATA_DIR):
        """
        Initialize data loader

        Args:
            data_dir: Path to data directory
        """
        self.data_dir = Path(data_dir)

        if IS_KAGGLE:
            # On Kaggle, new dataset structure: /kaggle/input/macros-and-ohlc/
            self.fx_data_dir = self.data_dir / "fx_data"
            self.macro_events_dir = self.data_dir / "macro_events"
        else:
            # Local structure
            self.fx_data_dir = self.data_dir / "kaggle_dataset" / "fx_data"
            self.macro_events_dir = self.data_dir / "kaggle_dataset" / "macro_events"

        logger.info(f"FX Data Directory: {self.fx_data_dir}")
        logger.info(f"Macro Events Directory: {self.macro_events_dir}")

    def load_symbol_data(self, symbol: str, timeframe: str = "M5") -> pd.DataFrame:
        """
        Load OHLCV data for a single symbol

        Args:
            symbol: Currency pair (e.g., 'EURUSD')
            timeframe: Timeframe (default: 'M5')

        Returns:
            DataFrame with OHLCV data
        """
        filename = f"{symbol}_{timeframe}.parquet"
        filepath = self.fx_data_dir / filename

        if not filepath.exists():
            raise FileNotFoundError(f"Data file not found: {filepath}")

        logger.info(f"Loading {symbol} data from {filepath}")
        df = pd.read_parquet(filepath)

        logger.info(f"Loaded {len(df):,} candles for {symbol}")
        logger.info(f"Date range: {df.index.min()} to {df.index.max()}")

        return df

    def load_all_symbols(self, timeframe: str = "M5") -> Dict[str, pd.DataFrame]:
        """
        Load data for all available symbols

        Args:
            timeframe: Timeframe (default: 'M5')

        Returns:
            Dictionary mapping symbol names to DataFrames
        """
        data = {}

        # Find all parquet files
        pattern = f"*_{timeframe}.parquet"
        files = list(self.fx_data_dir.glob(pattern))

        logger.info(f"Found {len(files)} data files")

        for filepath in files:
            # Extract symbol name (e.g., 'EURUSD_M5.parquet' -> 'EURUSD')
            symbol = filepath.stem.replace(f"_{timeframe}", "")

            df = pd.read_parquet(filepath)
            data[symbol] = df

            logger.info(f"  {symbol}: {len(df):,} candles")

        return data

    def get_available_symbols(self, timeframe: str = "M5") -> List[str]:
        """
        Get list of available symbols

        Args:
            timeframe: Timeframe (default: 'M5')

        Returns:
            List of symbol names
        """
        pattern = f"*_{timeframe}.parquet"
        files = list(self.fx_data_dir.glob(pattern))

        symbols = [f.stem.replace(f"_{timeframe}", "") for f in files]
        return sorted(symbols)

    def load_macro_events(self, symbol: Optional[str] = None) -> pd.DataFrame:
        """
        Load pre-downloaded macro events from Kaggle dataset

        Args:
            symbol: Specific symbol to load events for (e.g., 'EURUSD')
                   If None, loads all available events

        Returns:
            DataFrame with macro events
        """
        if not self.macro_events_dir.exists():
            logger.warning(
                f"Macro events directory not found: {self.macro_events_dir}")
            return pd.DataFrame()

        if symbol:
            # Load events for specific symbol
            filename = f"{symbol}_events.parquet"
            filepath = self.macro_events_dir / filename

            if not filepath.exists():
                logger.warning(
                    f"Events file not found for {symbol}: {filepath}")
                return pd.DataFrame()

            logger.info(f"Loading macro events for {symbol} from {filepath}")
            df = pd.read_parquet(filepath)

            logger.info(f"Loaded {len(df)} events for {symbol}")
            logger.info(
                f"Date range: {df['date'].min()} to {df['date'].max()}")

            return df
        else:
            # Load all available events
            pattern = "*_events.parquet"
            files = list(self.macro_events_dir.glob(pattern))

            if not files:
                logger.warning(
                    f"No macro event files found in {self.macro_events_dir}")
                return pd.DataFrame()

            logger.info(f"Found {len(files)} macro event files")

            all_events = []
            for filepath in files:
                symbol_name = filepath.stem.replace("_events", "")
                df = pd.read_parquet(filepath)
                df['symbol'] = symbol_name  # Add symbol column
                all_events.append(df)
                logger.info(f"  {symbol_name}: {len(df)} events")

            combined = pd.concat(all_events, ignore_index=True)
            logger.info(f"Total events loaded: {len(combined)}")

            return combined


class FXDataset(Dataset):
    """PyTorch Dataset for FX data with GPU optimization"""

    def __init__(self, X: torch.Tensor, y: torch.Tensor):
        """
        Initialize dataset

        Args:
            X: Features tensor
            y: Labels tensor
        """
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def create_dataloader(
    X: torch.Tensor,
    y: torch.Tensor,
    batch_size: int = 64,
    shuffle: bool = True,
    drop_last: bool = False
) -> DataLoader:
    """
    Create optimized DataLoader for GPU training

    Args:
        X: Features tensor
        y: Labels tensor
        batch_size: Batch size
        shuffle: Whether to shuffle data
        drop_last: Whether to drop last incomplete batch

    Returns:
        DataLoader instance
    """
    dataset = FXDataset(X, y)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=GPU_CONFIG['num_workers'],
        pin_memory=GPU_CONFIG['pin_memory'],
        drop_last=drop_last,
        persistent_workers=True if GPU_CONFIG['num_workers'] > 0 else False,
    )

    return dataloader


# Example usage
if __name__ == "__main__":
    # Initialize loader
    loader = KaggleFXDataLoader()

    # Get available symbols
    symbols = loader.get_available_symbols()
    print(f"Available symbols: {symbols}")

    # Load single symbol
    if symbols:
        df = loader.load_symbol_data(symbols[0])
        print(f"\n{symbols[0]} data:")
        print(df.head())
        print(f"\nShape: {df.shape}")

    # Load all symbols
    all_data = loader.load_all_symbols()
    print(f"\nLoaded {len(all_data)} symbols")
