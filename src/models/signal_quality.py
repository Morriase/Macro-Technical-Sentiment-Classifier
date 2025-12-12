"""
Signal Quality Scoring with Fuzzy Logic
Evaluates prediction confidence and market conditions

Updated for 7-feature ZigZag model:
- rsi_norm, macd_diff_norm, candle_body_norm
- rsi_velocity, macd_velocity
- yield_curve, dxy_index
"""
import numpy as np
import pandas as pd
from typing import Dict, Tuple


class SignalQualityScorer:
    """
    Fuzzy logic-based signal quality assessment
    Combines prediction confidence with feature-based indicators
    
    Adapted for the simplified 7-feature model.
    """

    def __init__(self):
        self.min_quality_threshold = 40  # Minimum quality to trade

    def calculate_quality(
        self,
        prediction_proba: np.ndarray,
        features: pd.Series,
        predicted_class: int
    ) -> Tuple[float, Dict[str, float]]:
        """
        Calculate signal quality score (0-100)

        Args:
            prediction_proba: [P(Sell), P(Buy)] - binary classification
            features: Feature values for this prediction
            predicted_class: 0=Sell, 1=Buy

        Returns:
            (quality_score, component_scores)
        """
        components = {}

        # 1. Confidence Score (0-50 points) - increased weight for binary model
        max_prob = np.max(prediction_proba)
        components['confidence'] = self._fuzzy_confidence(max_prob) * 50

        # 2. RSI Alignment (0-20 points)
        components['rsi'] = self._fuzzy_rsi_alignment(
            features, predicted_class
        ) * 20

        # 3. MACD Alignment (0-15 points)
        components['macd'] = self._fuzzy_macd_alignment(
            features, predicted_class
        ) * 15

        # 4. Momentum Confirmation (0-15 points)
        components['momentum'] = self._fuzzy_momentum_confirmation(
            features, predicted_class
        ) * 15

        # Total quality score
        quality = sum(components.values())

        return quality, components

    def _fuzzy_confidence(self, prob: float) -> float:
        """
        Fuzzy membership for prediction confidence
        Adjusted for binary classification (typically higher confidence)
        
        Returns: 0.0 to 1.0
        """
        if prob > 0.80:
            return 1.0  # Very high confidence
        elif prob > 0.70:
            return 0.85  # High confidence
        elif prob > 0.60:
            return 0.7  # Medium-high confidence
        elif prob > 0.55:
            return 0.5  # Medium confidence
        elif prob > 0.52:
            return 0.35  # Low confidence
        else:
            return 0.2  # Very low confidence (near 50/50)

    def _fuzzy_rsi_alignment(
        self, features: pd.Series, predicted_class: int
    ) -> float:
        """
        Check if RSI aligns with prediction direction
        Uses rsi_norm which is (RSI - 50) / 50, so range is [-1, 1]
        """
        if 'rsi_norm' not in features:
            return 0.5  # Neutral if no RSI data

        rsi_norm = features['rsi_norm']
        
        # rsi_norm > 0 means RSI > 50 (bullish)
        # rsi_norm < 0 means RSI < 50 (bearish)
        
        if predicted_class == 1:  # Buy
            if rsi_norm > 0.2:  # RSI > 60, strong bullish
                return 0.9
            elif rsi_norm > 0:  # RSI > 50, mild bullish
                return 0.7
            elif rsi_norm > -0.4:  # RSI 30-50, potential reversal
                return 0.5
            else:  # RSI < 30, oversold - could be good for buy
                return 0.6
                
        else:  # Sell (predicted_class == 0)
            if rsi_norm < -0.2:  # RSI < 40, strong bearish
                return 0.9
            elif rsi_norm < 0:  # RSI < 50, mild bearish
                return 0.7
            elif rsi_norm < 0.4:  # RSI 50-70, potential reversal
                return 0.5
            else:  # RSI > 70, overbought - could be good for sell
                return 0.6

        return 0.5

    def _fuzzy_macd_alignment(
        self, features: pd.Series, predicted_class: int
    ) -> float:
        """
        Check if MACD aligns with prediction direction
        Uses macd_diff_norm which is normalized MACD histogram
        """
        if 'macd_diff_norm' not in features:
            return 0.5  # Neutral if no MACD data

        macd_diff = features['macd_diff_norm']
        
        # macd_diff_norm > 0 means MACD > Signal (bullish)
        # macd_diff_norm < 0 means MACD < Signal (bearish)
        
        if predicted_class == 1:  # Buy
            if macd_diff > 0.3:
                return 1.0  # Strong bullish MACD
            elif macd_diff > 0:
                return 0.8  # Mild bullish MACD
            elif macd_diff > -0.3:
                return 0.4  # Mild bearish (conflicting)
            else:
                return 0.2  # Strong bearish (conflicting)
                
        else:  # Sell
            if macd_diff < -0.3:
                return 1.0  # Strong bearish MACD
            elif macd_diff < 0:
                return 0.8  # Mild bearish MACD
            elif macd_diff < 0.3:
                return 0.4  # Mild bullish (conflicting)
            else:
                return 0.2  # Strong bullish (conflicting)

        return 0.5

    def _fuzzy_momentum_confirmation(
        self, features: pd.Series, predicted_class: int
    ) -> float:
        """
        Check if velocity features confirm the prediction direction
        Uses rsi_velocity and macd_velocity
        """
        score = 0.5  # Start neutral
        
        # RSI velocity (rate of change of RSI)
        if 'rsi_velocity' in features:
            rsi_vel = features['rsi_velocity']
            
            if predicted_class == 1:  # Buy
                if rsi_vel > 0.02:  # RSI accelerating up
                    score += 0.25
                elif rsi_vel > 0:
                    score += 0.1
                elif rsi_vel < -0.02:  # RSI accelerating down (conflicting)
                    score -= 0.15
                    
            else:  # Sell
                if rsi_vel < -0.02:  # RSI accelerating down
                    score += 0.25
                elif rsi_vel < 0:
                    score += 0.1
                elif rsi_vel > 0.02:  # RSI accelerating up (conflicting)
                    score -= 0.15

        # MACD velocity (rate of change of MACD diff)
        if 'macd_velocity' in features:
            macd_vel = features['macd_velocity']
            
            if predicted_class == 1:  # Buy
                if macd_vel > 0.01:  # MACD histogram expanding bullish
                    score += 0.25
                elif macd_vel > 0:
                    score += 0.1
                    
            else:  # Sell
                if macd_vel < -0.01:  # MACD histogram expanding bearish
                    score += 0.25
                elif macd_vel < 0:
                    score += 0.1

        return max(0.0, min(1.0, score))

    def get_position_size_multiplier(self, quality_score: float) -> float:
        """
        Fuzzy position sizing based on quality

        Returns: 0.0 to 1.0 (multiplier for base position size)
        """
        if quality_score >= 80:
            return 1.0  # Full size
        elif quality_score >= 65:
            return 0.75  # 75% size
        elif quality_score >= 50:
            return 0.5  # 50% size
        elif quality_score >= 40:
            return 0.25  # 25% size
        else:
            return 0.0  # Skip trade

    def should_trade(self, quality_score: float) -> bool:
        """
        Fuzzy decision: should we take this trade?
        """
        return quality_score >= self.min_quality_threshold


if __name__ == "__main__":
    # Example usage with 7-feature model
    scorer = SignalQualityScorer()

    # Mock prediction and features (7-feature model)
    prediction_proba = np.array([0.35, 0.65])  # 65% Buy
    predicted_class = 1  # Buy

    features = pd.Series({
        'rsi_norm': 0.15,        # RSI ~57.5
        'macd_diff_norm': 0.2,   # Mild bullish MACD
        'candle_body_norm': 0.1,
        'rsi_velocity': 0.01,
        'macd_velocity': 0.005,
        'yield_curve': 0.5,
        'dxy_index': 105.0
    })

    quality, components = scorer.calculate_quality(
        prediction_proba, features, predicted_class
    )

    print(f"Signal Quality: {quality:.1f}/100")
    print(f"Components: {components}")
    print(f"Position Size: {scorer.get_position_size_multiplier(quality)*100:.0f}%")
    print(f"Should Trade: {scorer.should_trade(quality)}")
