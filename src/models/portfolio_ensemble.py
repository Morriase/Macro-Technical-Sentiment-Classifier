"""
Portfolio-Level Ensemble for Multi-Currency Trading
Aggregates predictions from individual currency pair models
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from loguru import logger
from pathlib import Path


class PortfolioEnsemble:
    """
    Combines predictions from multiple currency pair models
    into portfolio-level trading decisions
    """

    def __init__(self, pair_weights: Optional[Dict[str, float]] = None):
        """
        Initialize portfolio ensemble

        Args:
            pair_weights: Optional weights for each currency pair
                         If None, equal weights are used
        """
        self.pair_weights = pair_weights
        self.pair_predictions = {}

    def add_pair_prediction(
        self,
        pair: str,
        predictions: pd.DataFrame
    ):
        """
        Add predictions from a single currency pair model

        Args:
            pair: Currency pair name (e.g., "EUR_USD")
            predictions: DataFrame with columns [timestamp, signal, confidence, proba_buy, proba_sell, proba_hold]
        """
        self.pair_predictions[pair] = predictions
        logger.info(
            f"Added predictions for {pair} ({len(predictions)} samples)")

    def get_aggregate_signals(
        self,
        min_confidence: float = 0.60,
        min_agreement: int = 3
    ) -> pd.DataFrame:
        """
        Generate aggregate portfolio signals

        Args:
            min_confidence: Minimum confidence threshold for individual signals
            min_agreement: Minimum number of pairs that must agree

        Returns:
            DataFrame with portfolio-level signals
        """
        if not self.pair_predictions:
            raise ValueError(
                "No pair predictions added. Call add_pair_prediction() first.")

        # Get common timestamps across all pairs
        timestamps = self._get_common_timestamps()

        results = []
        for ts in timestamps:
            signals = self._aggregate_at_timestamp(
                ts, min_confidence, min_agreement)
            results.append(signals)

        df = pd.DataFrame(results)
        logger.info(f"Generated {len(df)} portfolio signals")

        return df

    def _get_common_timestamps(self) -> List[pd.Timestamp]:
        """Get timestamps that exist across all currency pairs"""
        all_timestamps = [
            set(pred['timestamp'].values)
            for pred in self.pair_predictions.values()
        ]

        # Intersection of all timestamps
        common = set.intersection(*all_timestamps)
        return sorted(list(common))

    def _aggregate_at_timestamp(
        self,
        timestamp: pd.Timestamp,
        min_confidence: float,
        min_agreement: int
    ) -> Dict:
        """
        Aggregate signals from all pairs at a specific timestamp

        Args:
            timestamp: Timestamp to aggregate
            min_confidence: Minimum confidence for a valid signal
            min_agreement: Minimum pairs that must agree

        Returns:
            Dictionary with aggregated signal data
        """
        buy_votes = 0
        sell_votes = 0
        hold_votes = 0

        buy_confidences = []
        sell_confidences = []

        total_weight = 0
        weighted_buy_prob = 0
        weighted_sell_prob = 0
        weighted_hold_prob = 0

        active_pairs = []

        for pair, predictions in self.pair_predictions.items():
            # Get prediction for this timestamp
            pred = predictions[predictions['timestamp'] == timestamp]

            if pred.empty:
                continue

            pred = pred.iloc[0]

            # Get weight for this pair
            weight = self.pair_weights.get(
                pair, 1.0) if self.pair_weights else 1.0

            # Extract probabilities and determine confidence
            buy_prob = pred.get(
                'pred_buy_prob', 0) if 'pred_buy_prob' in pred else pred.get('proba_buy', 0)
            sell_prob = pred.get(
                'pred_sell_prob', 0) if 'pred_sell_prob' in pred else pred.get('proba_sell', 0)
            hold_prob = pred.get(
                'pred_hold_prob', 0) if 'pred_hold_prob' in pred else pred.get('proba_hold', 0)

            # Confidence is the max probability
            confidence = float(max(buy_prob, sell_prob, hold_prob))

            # Convert signal to numeric if it's a string
            signal_str = pred.get(
                'signal', pred.get('predicted_class', 'HOLD'))
            if isinstance(signal_str, str):
                signal = {'BUY': 1, 'SELL': -1,
                          'HOLD': 0}.get(signal_str.upper(), 0)
            else:
                signal = signal_str

            # Only count if confidence exceeds threshold
            if confidence >= min_confidence:
                if signal == 1 or (isinstance(signal, str) and signal.upper() == 'BUY'):  # Buy
                    buy_votes += 1
                    buy_confidences.append(confidence)
                    active_pairs.append(f"{pair}:BUY")
                elif signal == -1 or (isinstance(signal, str) and signal.upper() == 'SELL'):  # Sell
                    sell_votes += 1
                    sell_confidences.append(confidence)
                    active_pairs.append(f"{pair}:SELL")
                else:  # Hold
                    hold_votes += 1
                    active_pairs.append(f"{pair}:HOLD")

            # Weighted probability aggregation
            weighted_buy_prob += buy_prob * weight
            weighted_sell_prob += sell_prob * weight
            weighted_hold_prob += hold_prob * weight
            total_weight += weight

        # Normalize weighted probabilities
        if total_weight > 0:
            weighted_buy_prob /= total_weight
            weighted_sell_prob /= total_weight
            weighted_hold_prob /= total_weight

        # Determine portfolio signal based on voting
        portfolio_signal = 0  # Default: Hold
        portfolio_confidence = 0.0

        if buy_votes >= min_agreement:
            portfolio_signal = 1  # Buy
            portfolio_confidence = np.mean(
                buy_confidences) if buy_confidences else 0.0
        elif sell_votes >= min_agreement:
            portfolio_signal = -1  # Sell
            portfolio_confidence = np.mean(
                sell_confidences) if sell_confidences else 0.0

        return {
            'timestamp': timestamp,
            'portfolio_signal': portfolio_signal,
            'portfolio_confidence': portfolio_confidence,
            'buy_votes': buy_votes,
            'sell_votes': sell_votes,
            'hold_votes': hold_votes,
            'weighted_buy_prob': weighted_buy_prob,
            'weighted_sell_prob': weighted_sell_prob,
            'weighted_hold_prob': weighted_hold_prob,
            'active_pairs': ', '.join(active_pairs),
            'num_active_pairs': len(active_pairs)
        }

    def get_correlation_matrix(self) -> pd.DataFrame:
        """
        Calculate correlation matrix of pair signals

        Returns:
            Correlation matrix DataFrame
        """
        if not self.pair_predictions:
            raise ValueError("No pair predictions added")

        # Get common timestamps
        timestamps = self._get_common_timestamps()

        # Build signal matrix
        signal_data = {}
        for pair, predictions in self.pair_predictions.items():
            signals = []
            for ts in timestamps:
                pred = predictions[predictions['timestamp'] == ts]
                if not pred.empty:
                    signals.append(pred.iloc[0]['signal'])
                else:
                    signals.append(0)
            signal_data[pair] = signals

        df = pd.DataFrame(signal_data, index=timestamps)
        corr_matrix = df.corr()

        logger.info("Calculated pair correlation matrix")
        return corr_matrix

    def get_portfolio_statistics(self, signals: pd.DataFrame) -> Dict:
        """
        Calculate portfolio-level statistics

        Args:
            signals: DataFrame from get_aggregate_signals()

        Returns:
            Dictionary with portfolio statistics
        """
        total_signals = len(signals)
        buy_signals = (signals['portfolio_signal'] == 1).sum()
        sell_signals = (signals['portfolio_signal'] == -1).sum()
        hold_signals = (signals['portfolio_signal'] == 0).sum()

        avg_confidence = signals[signals['portfolio_signal']
                                 != 0]['portfolio_confidence'].mean()
        avg_active_pairs = signals['num_active_pairs'].mean()

        stats = {
            'total_signals': total_signals,
            'buy_signals': buy_signals,
            'sell_signals': sell_signals,
            'hold_signals': hold_signals,
            'buy_pct': (buy_signals / total_signals * 100) if total_signals > 0 else 0,
            'sell_pct': (sell_signals / total_signals * 100) if total_signals > 0 else 0,
            'hold_pct': (hold_signals / total_signals * 100) if total_signals > 0 else 0,
            'avg_confidence': avg_confidence,
            'avg_active_pairs': avg_active_pairs
        }

        logger.info(
            f"Portfolio Stats - Buy: {stats['buy_pct']:.1f}%, Sell: {stats['sell_pct']:.1f}%, Hold: {stats['hold_pct']:.1f}%")
        logger.info(
            f"Avg Confidence: {avg_confidence:.3f}, Avg Active Pairs: {avg_active_pairs:.1f}")

        return stats

    def save_portfolio_signals(
        self,
        signals: pd.DataFrame,
        output_path: Path
    ):
        """
        Save portfolio signals to CSV

        Args:
            signals: DataFrame from get_aggregate_signals()
            output_path: Path to save CSV file
        """
        signals.to_csv(output_path, index=False)
        logger.info(f"Portfolio signals saved to {output_path}")
