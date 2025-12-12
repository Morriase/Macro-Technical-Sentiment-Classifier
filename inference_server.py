"""
Production Inference Server for Forex Trading Models
Handles feature engineering, model loading, and predictions

Architecture:
- EA sends OHLCV data (M5, H1, H4)
- Server engineers features using EXACT training pipeline
- Server validates features match model schema
- Server returns predictions with confidence scores
"""
import os
import gc
import sys
from pathlib import Path

# Add project root FIRST
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.config import MODELS_DIR, CURRENCY_PAIRS
from src.feature_engineering.technical_features import engineer_simplified_zigzag_features
from src.models.hybrid_ensemble import HybridEnsemble
from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
import json
from loguru import logger
from datetime import datetime

app = Flask(__name__)

# Configure logger (ensure logs directory exists)
os.makedirs("logs", exist_ok=True)
logger.add("logs/inference_server.log", rotation="1 day",
           retention="7 days", level="INFO")

logger.info("=" * 80)
logger.info("INITIALIZING FOREX INFERENCE SERVER")
logger.info("=" * 80)
logger.info(f"Flask app created")
logger.info(f"PORT environment variable: {os.environ.get('PORT', 'not set')}")
logger.info(f"Models directory: {MODELS_DIR}")
logger.info(f"Supported pairs: {CURRENCY_PAIRS}")

# Pair mapping for testing (allows BTC to use EUR_USD model)
PAIR_ALIASES = {
    'BTCUSD': 'EUR_USD',
    'BTC_USD': 'EUR_USD',
    'XBTUSD': 'EUR_USD',  # BitMEX notation
}

# Global model cache
MODELS = {}
FEATURE_SCHEMAS = {}

# Initialize FRED Macro Loader for real economic data (with caching)
FRED_LOADER = None
try:
    from src.data_acquisition.fred_macro_loader import FREDMacroLoader
    logger.info("Initializing FREDMacroLoader...")
    FRED_LOADER = FREDMacroLoader()
    if FRED_LOADER.api_key:
        logger.info(f"✓ FREDMacroLoader initialized (API key configured)")
        # Pre-fetch and cache macro features on startup
        try:
            yield_curve, dxy_index = FRED_LOADER.get_inference_macro_features()
            logger.info(f"✓ Cached macro features: yield_curve={yield_curve:.2f}, dxy_index={dxy_index:.2f}")
        except Exception as e:
            logger.warning(f"⚠ Failed to pre-cache macro features: {e}")
    else:
        logger.warning("⚠ FREDMacroLoader: No API key - will use cached data or defaults")
except ImportError:
    logger.warning("⚠ FREDMacroLoader not available")
    FRED_LOADER = None
except Exception as e:
    logger.warning(f"⚠ FREDMacroLoader error: {e}")
    FRED_LOADER = None

logger.info("=" * 80)
logger.info("SERVER INITIALIZATION COMPLETE")
logger.info("=" * 80)


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
    return jsonify({
        'status': 'healthy',
        'models_loaded': list(MODELS.keys()),
        'supported_pairs': CURRENCY_PAIRS,
        'fred_available': FRED_LOADER is not None
    })


@app.route('/predict', methods=['POST'])
def predict():
    """
    Main prediction endpoint

    Expected JSON format:
    {
        "pair": "EUR_USD",
        "ohlcv_m5": [ ... ],
        "ohlcv_h1": [ ... ],   // optional, not used currently
        "ohlcv_h4": [ ... ],   // optional, not used currently
        "calendar_events": [ ... ]  // optional, not used currently
    }
    """
    try:
        data = request.get_json()

        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400

        pair = data.get('pair')
        ohlcv_m5_data = data.get('ohlcv_m5')
        
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
        # We add macro features directly after using cached values
        df_features, feature_cols = engineer_simplified_zigzag_features(
            df=df_m5,
            pair=pair,
            fred_loader=None  # Skip FRED merge, add features directly below
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
            position_size_pct = quality_scorer.get_position_size_multiplier(quality_score)

            if quality_score < quality_scorer.min_quality_threshold:
                signal = "HOLD"
                quality_filtered = True
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

        logger.success(f"📤 {original_pair}: {signal} | conf={confidence:.1%} | quality={quality_score:.0f}/100")

        del df_m5, df_features, feature_array
        gc.collect()

        return jsonify(response)

    except Exception as e:
        logger.error(f"Prediction error: {e}", exc_info=True)
        gc.collect()
        return jsonify({'error': str(e)}), 500


@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    """
    Batch prediction for multiple pairs

    Expected JSON format:
    {
        "requests": [
            {"pair": "EUR_USD", "ohlcv": [...], "events": [...]},  // events optional
            {"pair": "GBP_USD", "ohlcv": [...], "events": [...]}
        ]
    }
    """
    try:
        data = request.get_json()
        requests = data.get('requests', [])

        if not requests:
            return jsonify({'error': 'No requests provided'}), 400

        results = []
        for req in requests:
            # Reuse single prediction logic
            app_context = app.test_request_context(
                json=req,
                method='POST',
                path='/predict'
            )
            with app_context:
                response = predict()
                # Handle both Response objects and tuples (response, status_code)
                if isinstance(response, tuple):
                    response_obj, status_code = response
                else:
                    response_obj = response
                    status_code = response.status_code

                if status_code == 200:  # Success
                    results.append(response_obj.get_json())
                else:
                    results.append(
                        {'pair': req.get('pair'), 'error': response_obj.get_json()})

        return jsonify({'results': results, 'count': len(results)})

    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        return jsonify({'error': f'Batch prediction error: {str(e)}'}), 500


@app.route('/model_info/<pair>', methods=['GET'])
def model_info(pair):
    """
    Get information about a loaded model
    """
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


if __name__ == '__main__':
    import os

    logger.info("=" * 80)
    logger.info("STARTING FOREX INFERENCE SERVER")
    logger.info("=" * 80)
    logger.info(f"Supported pairs: {', '.join(CURRENCY_PAIRS)}")
    logger.info(f"Models directory: {MODELS_DIR}")

    # Get port from environment variable (for Render/Docker) or use default
    port = int(os.environ.get('PORT', 5000))
    logger.info(f"Server will run on port: {port}")

    # Run server
    app.run(
        host='0.0.0.0',  # Allow external connections
        port=port,
        debug=False,  # Set to False in production
        threaded=True
    )

