"""
Signal Quality Scoring with Fuzzy Logic
Evaluates prediction confidence and market conditions
"""
import numpy as np
import pandas as pd
from typing import Dict, Tuple


class SignalQualityScorer:
    """
    Fuzzy logic-based signal quality assessment
    Combines prediction confidence with market regime indicators
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
            prediction_proba: [P(Buy), P(Sell), P(Hold)]
            features: Feature values for this prediction
            predicted_class: 0=Buy, 1=Sell, 2=Hold
            
        Returns:
            (quality_score, component_scores)
        """
        components = {}
        
        # 1. Confidence Score (0-40 points)
        max_prob = np.max(prediction_proba)
        components['confidence'] = self._fuzzy_confidence(max_prob) * 40
        
        # 2. Trend Alignment (0-25 points)
        components['trend'] = self._fuzzy_trend_alignment(
            features, predicted_class
        ) * 25
        
        # 3. Volatility Regime (0-20 points)
        components['volatility'] = self._fuzzy_volatility_regime(
            features
        ) * 20
        
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
        Returns: 0.0 to 1.0
        """
        if prob > 0.85:
            return 1.0  # Very high confidence
        elif prob > 0.75:
            return 0.8  # High confidence
        elif prob > 0.65:
            return 0.6  # Medium confidence
        elif prob > 0.55:
            return 0.4  # Low confidence
        else:
            return 0.2  # Very low confidence
    
    def _fuzzy_trend_alignment(
        self, features: pd.Series, predicted_class: int
    ) -> float:
        """
        Check if prediction aligns with trend indicators
        """
        score = 0.0
        
        # EMA alignment
        if 'ema_50' in features and 'ema_200' in features and 'close' in features:
            price = features['close']
            ema_50 = features['ema_50']
            ema_200 = features['ema_200']
            
            # Bullish alignment
            if predicted_class == 0:  # Buy
                if price > ema_50 > ema_200:
                    score += 0.5  # Strong bullish trend
                elif price > ema_50:
                    score += 0.3  # Weak bullish trend
            
            # Bearish alignment
            elif predicted_class == 1:  # Sell
                if price < ema_50 < ema_200:
                    score += 0.5  # Strong bearish trend
                elif price < ema_50:
                    score += 0.3  # Weak bearish trend
        
        # MACD confirmation
        if 'macd' in features and 'macd_signal' in features:
            macd = features['macd']
            signal = features['macd_signal']
            
            if predicted_class == 0 and macd > signal:  # Buy + bullish MACD
                score += 0.3
            elif predicted_class == 1 and macd < signal:  # Sell + bearish MACD
                score += 0.3
        
        return min(1.0, score)
    
    def _fuzzy_volatility_regime(self, features: pd.Series) -> float:
        """
        Assess volatility regime (lower volatility = higher quality)
        """
        if 'atr_zscore' not in features:
            return 0.5  # Neutral if no ATR data
        
        atr_z = features['atr_zscore']
        
        if atr_z < 0.5:
            return 1.0  # Very low volatility (best)
        elif atr_z < 1.0:
            return 0.8  # Low volatility (good)
        elif atr_z < 1.5:
            return 0.6  # Normal volatility (ok)
        elif atr_z < 2.0:
            return 0.3  # High volatility (risky)
        else:
            return 0.1  # Very high volatility (very risky)
    
    def _fuzzy_momentum_confirmation(
        self, features: pd.Series, predicted_class: int
    ) -> float:
        """
        Check if momentum indicators confirm the prediction
        """
        score = 0.0
        
        # RSI confirmation
        if 'rsi_14' in features:
            rsi = features['rsi_14']
            
            if predicted_class == 0:  # Buy
                if 30 < rsi < 70:  # Not overbought
                    score += 0.5
                elif rsi < 30:  # Oversold (good for buy)
                    score += 0.7
            
            elif predicted_class == 1:  # Sell
                if 30 < rsi < 70:  # Not oversold
                    score += 0.5
                elif rsi > 70:  # Overbought (good for sell)
                    score += 0.7
        
        # Stochastic confirmation
        if 'stoch_k' in features:
            stoch = features['stoch_k']
            
            if predicted_class == 0 and stoch < 80:  # Buy + not overbought
                score += 0.3
            elif predicted_class == 1 and stoch > 20:  # Sell + not oversold
                score += 0.3
        
        return min(1.0, score)
    
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
    # Example usage
    scorer = SignalQualityScorer()
    
    # Mock prediction and features
    prediction_proba = np.array([0.75, 0.15, 0.10])  # 75% Buy
    predicted_class = 0  # Buy
    
    features = pd.Series({
        'close': 1.0500,
        'ema_50': 1.0480,
        'ema_200': 1.0450,
        'atr_zscore': 0.8,
        'rsi_14': 55,
        'macd': 0.0005,
        'macd_signal': 0.0003,
        'stoch_k': 60
    })
    
    quality, components = scorer.calculate_quality(
        prediction_proba, features, predicted_class
    )
    
    print(f"Signal Quality: {quality:.1f}/100")
    print(f"Components: {components}")
    print(f"Position Size: {scorer.get_position_size_multiplier(quality)*100:.0f}%")
    print(f"Should Trade: {scorer.should_trade(quality)}")
