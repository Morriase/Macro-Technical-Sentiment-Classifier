"""
Local Inference Server for Testing with Live MT5 EA
Receives OHLCV data from EA, engineers features, and returns predictions.

Usage:
    python test_inference_local.py
    
Then point your EA to: http://localhost:5000/predict
"""
import sys
import gc
import json
import numpy as np
import pandas as pd
import threading
from pathlib import Path
from flask import Flask, request, jsonify
from loguru import logger
from datetime import datetime

# Add project root
sys.path.insert(0, str(Path(__file__).parent))

from src.config import MODELS_DIR, CURRENCY_PAIRS
from src.models.hybrid_ensemble import HybridEnsemble
from src.feature_engineering.technical_features import engineer_simplified_zigzag_features

# Configure logger
logger.add("logs/local_inference.log", rotation="1 day", retention="3 days", level="DEBUG")

app = Flask(__name__)

# Global model cache
MODELS = {}
FEATURE_SCHEMAS = {}

# FRED loader for macro features (with caching)
FRED_LOADER = None
try:
    from src.data_acquisition.fred_macro_loader import FREDMacroLoader
    FRED_LOADER = FREDMacroLoader()
    if FRED_LOADER.api_key:
        logger.info(f"✓ FRED loader initialized (API key configured)")
    else:
        logger.warning("⚠ FRED loader: No API key - will use cached data or defaults")
except Exception as e:
    logger.warning(f"⚠ FRED loader not available: {e}")
    FRED_LOADER = None

# Sentiment ensemble (Option C: Tech + FinBERT voting)
# Initialized lazily after server starts to avoid blocking startup
SENTIMENT_ENSEMBLE = None
SENTIMENT_PROVIDER = None
ENABLE_SENTIMENT = True  # Toggle sentiment integration
FINBERT_READY = False  # Flag to track if FinBERT is loaded


def init_sentiment_ensemble():
    """Initialize sentiment ensemble (called at module load)"""
    global SENTIMENT_ENSEMBLE, SENTIMENT_PROVIDER, ENABLE_SENTIMENT
    try:
        from src.models.sentiment_ensemble import SentimentEnsemble, CachedSentimentProvider
        SENTIMENT_ENSEMBLE = SentimentEnsemble(
            tech_weight=0.7,           # 70% weight to technical model
            sentiment_weight=0.3,      # 30% weight to sentiment
            require_agreement=False,   # Don't require agreement (less conservative)
            sentiment_threshold=0.1,   # Min polarity to count as directional
            boost_on_agreement=10.0,   # Quality boost when signals agree
            penalty_on_disagreement=15.0  # Quality penalty when they disagree
        )
        SENTIMENT_PROVIDER = CachedSentimentProvider(cache_minutes=60)  # 1 hour cache
        logger.info("✓ Sentiment ensemble initialized (tech=70%, sentiment=30%)")
        return True
    except Exception as e:
        logger.warning(f"⚠ Sentiment ensemble not available: {e}")
        ENABLE_SENTIMENT = False
        return False


def preload_finbert_background():
    """Pre-load FinBERT model in background thread after server starts"""
    global FINBERT_READY
    try:
        logger.info("🔄 Pre-loading FinBERT model in background...")
        from src.data_acquisition.live_news_loader import load_finbert_model
        load_finbert_model()  # This loads and caches the model
        FINBERT_READY = True
        logger.success("✓ FinBERT pre-loaded and ready for inference")
    except Exception as e:
        logger.warning(f"⚠ FinBERT pre-load failed: {e} (will load on first request)")


# Initialize sentiment ensemble at module load (lightweight)
init_sentiment_ensemble()

# Pair aliases (for testing with different symbols)
PAIR_ALIASES = {
    'BTCUSD': 'EUR_USD',
    'BTC_USD': 'EUR_USD',
    'XBTUSD': 'EUR_USD',
}


def load_model_and_schema(pair: str):
    """Load trained model and feature schema for a currency pair"""
    if pair in MODELS:
        return MODELS[pair], FEATURE_SCHEMAS[pair]

    logger.info(f"Loading model for {pair}...")

    model_path = MODELS_DIR / f"{pair}_model"
    if not Path(f"{model_path}_config.pkl").exists():
        raise FileNotFoundError(f"Model not found for {pair}")

    model = HybridEnsemble()
    model.load_model(str(model_path))

    schema_path = MODELS_DIR / f"{pair}_feature_schema.json"
    if not schema_path.exists():
        raise FileNotFoundError(f"Feature schema not found for {pair}")

    with open(schema_path, 'r') as f:
        schema = json.load(f)

    MODELS[pair] = model
    FEATURE_SCHEMAS[pair] = schema

    logger.success(f"Loaded {pair} model ({schema['n_features']} features)")
    return model, schema


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    response = {
        'status': 'healthy',
        'models_loaded': list(MODELS.keys()),
        'supported_pairs': CURRENCY_PAIRS,
        'fred_available': FRED_LOADER is not None,
        'sentiment_enabled': ENABLE_SENTIMENT,
        'finbert_ready': FINBERT_READY
    }
    
    # Add sentiment provider status
    if SENTIMENT_PROVIDER:
        try:
            response['sentiment_status'] = SENTIMENT_PROVIDER.get_health_status()
        except:
            response['sentiment_status'] = {'available': False}
    
    return jsonify(response)


@app.route('/predict', methods=['POST'])
def predict():
    """
    Main prediction endpoint - matches inference_server.py format
    
    Expected JSON:
    {
        "pair": "EUR_USD",
        "ohlcv_m5": [...],
        "ohlcv_h1": [...],   // optional, not used currently
        "ohlcv_h4": [...],   // optional, not used currently
        "macro_data": {"yield_curve": 0.25, "dxy_index": 104.5}  // optional
    }
    """
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400

        pair = data.get('pair')
        ohlcv_m5_data = data.get('ohlcv_m5')
        macro_data_from_ea = data.get('macro_data')
        
        if not pair or not ohlcv_m5_data:
            return jsonify({'error': 'Missing required fields: pair, ohlcv_m5'}), 400

        # Resolve pair alias
        original_pair = pair
        if pair in PAIR_ALIASES:
            pair = PAIR_ALIASES[pair]
            logger.info(f"Mapping {original_pair} → {pair}")

        if pair not in CURRENCY_PAIRS:
            return jsonify({'error': f'Unsupported pair: {original_pair}'}), 400

        logger.info(f"📥 Prediction request: {original_pair} ({pair}) | M5={len(ohlcv_m5_data)} candles")

        # Convert OHLCV to DataFrame
        df_m5 = pd.DataFrame(ohlcv_m5_data)
        df_m5['timestamp'] = pd.to_datetime(df_m5['timestamp'])
        df_m5.set_index('timestamp', inplace=True)

        required_cols = ['open', 'high', 'low', 'close', 'volume']
        if any(col not in df_m5.columns for col in required_cols):
            return jsonify({'error': 'Missing M5 OHLCV columns'}), 400

        if len(df_m5) < 100:
            return jsonify({'error': f'Insufficient M5 data: need 100+, got {len(df_m5)}'}), 400

        # Load model
        model, schema = load_model_and_schema(pair)

        # Feature engineering - pass fred_loader=None to skip slow FRED merge
        # We'll add macro features directly after using cached FRED data
        df_features, feature_cols = engineer_simplified_zigzag_features(
            df=df_m5,
            pair=pair,
            fred_loader=None,  # Skip FRED merge, add features directly below
            skip_fred_warning=True  # Suppress warning since we add macros manually
        )

        # Add macro features directly using cached FRED data
        if FRED_LOADER:
            try:
                yield_curve, dxy_index = FRED_LOADER.get_inference_macro_features()
                df_features['yield_curve'] = yield_curve
                df_features['dxy_index'] = dxy_index
                logger.info(f"  Macro features: yield_curve={yield_curve:.2f}, dxy_index={dxy_index:.2f}")
            except Exception as e:
                logger.warning(f"  FRED macro fetch failed: {e}, using defaults")
                df_features['yield_curve'] = 0.0
                df_features['dxy_index'] = 100.0
        else:
            # No FRED loader - use defaults
            df_features['yield_curve'] = 0.0
            df_features['dxy_index'] = 100.0

        # Validate features
        missing = [f for f in schema['feature_names'] if f not in df_features.columns]
        if missing:
            return jsonify({'error': f'Missing features: {missing}'}), 500

        feature_array = df_features[schema['feature_names']].values

        # Predict
        prediction_proba = model.predict_proba(feature_array)
        prediction_class = model.predict(feature_array)

        last_pred_class = prediction_class[-1]
        last_pred_proba = prediction_proba[-1]

        class_map = {0: "SELL", 1: "BUY"}
        raw_signal = class_map.get(last_pred_class, "UNKNOWN")
        confidence = float(last_pred_proba[last_pred_class])

        # === SENTIMENT ENSEMBLE (Option C) ===
        sentiment_data = None
        ensemble_result = None
        
        if ENABLE_SENTIMENT and SENTIMENT_ENSEMBLE and SENTIMENT_PROVIDER:
            if not FINBERT_READY:
                logger.debug("  FinBERT still loading in background, skipping sentiment for this request")
            else:
                try:
                    # Get cached sentiment for this pair
                    sentiment_data = SENTIMENT_PROVIDER.get_sentiment(pair, hours_back=24)
                    
                    # Combine tech model + sentiment
                    ensemble_result = SENTIMENT_ENSEMBLE.combine(
                        tech_proba=last_pred_proba,
                        sentiment=sentiment_data,
                        tech_predicted_class=last_pred_class
                    )
                    
                    logger.info(
                        f"  Sentiment: {ensemble_result.sentiment_signal} "
                        f"(polarity={ensemble_result.sentiment_score:.2f}) | "
                        f"Agreement: {ensemble_result.agreement}"
                    )
                except Exception as e:
                    logger.warning(f"  Sentiment ensemble failed: {e}")
                    ensemble_result = None

        # Quality scoring
        try:
            from src.models.signal_quality import SignalQualityScorer
            quality_scorer = SignalQualityScorer()
            last_features = df_features.iloc[-1]
            quality_score, quality_components = quality_scorer.calculate_quality(
                prediction_proba=last_pred_proba,
                features=last_features,
                predicted_class=last_pred_class
            )
            
            # Apply sentiment quality adjustment
            if ensemble_result:
                quality_score += ensemble_result.quality_boost
                quality_score = max(0, min(100, quality_score))  # Clamp to 0-100
                quality_components['sentiment_boost'] = ensemble_result.quality_boost
            
            position_size_pct = quality_scorer.get_position_size_multiplier(quality_score)

            if quality_score < quality_scorer.min_quality_threshold:
                signal = "HOLD"
                quality_filtered = True
            else:
                # Use ensemble signal if available, otherwise raw signal
                if ensemble_result:
                    signal = ensemble_result.signal
                    confidence = ensemble_result.confidence
                else:
                    signal = raw_signal
                quality_filtered = False
        except ImportError:
            logger.warning("SignalQualityScorer not found, using defaults")
            quality_score = 50.0
            quality_components = {}
            position_size_pct = 1.0
            signal = raw_signal
            quality_filtered = False

        # Build response
        response = {
            'pair': original_pair,
            'model_pair': pair,
            'prediction': signal,
            'raw_prediction': raw_signal,
            'confidence': round(confidence, 4),
            'probabilities': {
                'BUY': round(float(last_pred_proba[1]), 4),
                'SELL': round(float(last_pred_proba[0]), 4),
            },
            'quality_score': round(quality_score, 2),
            'quality_components': {k: round(v, 2) for k, v in quality_components.items()},
            'position_size_pct': round(position_size_pct, 2),
            'quality_filtered': quality_filtered,
            'timestamp': df_m5.index[-1].isoformat(),
            'feature_count': len(feature_cols),
            'candles_used': len(feature_array),
            'status': 'success'
        }
        
        # Add sentiment info to response if available
        if ensemble_result:
            response['sentiment'] = {
                'signal': ensemble_result.sentiment_signal,
                'polarity': round(ensemble_result.sentiment_score, 3),
                'agreement': ensemble_result.agreement,
                'tech_signal': ensemble_result.tech_signal,
                'tech_confidence': round(ensemble_result.tech_confidence, 4)
            }

        logger.success(
            f"📤 {original_pair}: {signal} | conf={confidence:.1%} | quality={quality_score:.0f}/100"
            + (f" | sentiment={ensemble_result.sentiment_signal}" if ensemble_result else "")
        )

        del df_m5, df_features, feature_array
        gc.collect()

        return jsonify(response)

    except Exception as e:
        logger.error(f"Prediction error: {e}", exc_info=True)
        gc.collect()
        return jsonify({'error': str(e)}), 500


@app.route('/model_info/<pair>', methods=['GET'])
def model_info(pair):
    """Get model information"""
    try:
        if pair not in CURRENCY_PAIRS:
            return jsonify({'error': f'Unsupported pair: {pair}'}), 400

        model, schema = load_model_and_schema(pair)

        return jsonify({
            'pair': pair,
            'n_features': schema['n_features'],
            'feature_names': schema['feature_names'],
            'model_version': schema.get('model_version', 'unknown'),
            'trained_date': schema.get('trained_date', 'unknown'),
            'is_loaded': pair in MODELS
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 404


def run_offline_tests():
    """Run offline tests (original functionality)"""
    from src.data_acquisition.kaggle_loader import KaggleFXDataLoader
    
    logger.info("=" * 60)
    logger.info("OFFLINE MODEL TESTS")
    logger.info("=" * 60)
    
    loader = KaggleFXDataLoader()
    pairs = ['EUR_USD', 'USD_JPY', 'GBP_USD']
    
    for pair in pairs:
        try:
            symbol = pair.replace('_', '')
            df = loader.load_symbol_data(symbol, 'M5')
            df = df.tail(500)
            
            df_features, feature_cols = engineer_simplified_zigzag_features(df, pair=pair, fred_loader=None)
            X = df_features[feature_cols].values
            
            model = HybridEnsemble()
            model.load_model(str(MODELS_DIR / f"{pair}_model"))
            
            proba = model.predict_proba(X)
            pred = model.predict(X)
            
            signal = "BUY" if pred[-1] == 1 else "SELL"
            conf = proba[-1][pred[-1]]
            
            logger.success(f"  ✓ {pair}: {signal} ({conf:.1%})")
        except Exception as e:
            logger.error(f"  ✗ {pair}: {e}")
    
    logger.info("=" * 60)


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Local Inference Server')
    parser.add_argument('--test', action='store_true', help='Run offline tests only')
    parser.add_argument('--port', type=int, default=5000, help='Server port (default: 5000)')
    parser.add_argument('--no-finbert', action='store_true', help='Disable FinBERT preloading')
    args = parser.parse_args()
    
    if args.test:
        run_offline_tests()
    else:
        logger.info("=" * 60)
        logger.info("LOCAL INFERENCE SERVER")
        logger.info("=" * 60)
        logger.info(f"Endpoint: http://localhost:{args.port}/predict")
        logger.info(f"Health:   http://localhost:{args.port}/health")
        logger.info(f"Supported pairs: {', '.join(CURRENCY_PAIRS)}")
        logger.info("=" * 60)
        
        # Pre-load FinBERT in background thread (avoids 40s delay on first request)
        if ENABLE_SENTIMENT and not args.no_finbert:
            finbert_thread = threading.Thread(target=preload_finbert_background, daemon=True)
            finbert_thread.start()
            logger.info("🚀 FinBERT loading in background (server ready immediately)")
        
        app.run(host='0.0.0.0', port=args.port, debug=True, threaded=True)
