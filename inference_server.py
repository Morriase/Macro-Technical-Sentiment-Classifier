"""
Production Inference Server for Forex Trading Models
Handles feature engineering, model loading, and predictions

Architecture:
- EA sends OHLCV data
- Server engineers features using EXACT training pipeline
- Server validates features match model schema
- Server returns predictions with confidence scores
"""
from src.config import MODELS_DIR, CURRENCY_PAIRS
from src.feature_engineering.technical_features import TechnicalFeatureEngineer
from src.models.hybrid_ensemble import HybridEnsemble
from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
import json
import joblib
from pathlib import Path
from loguru import logger
from datetime import datetime
import sys

# Add project root
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


app = Flask(__name__)

# Configure logger
logger.add("logs/inference_server.log", rotation="1 day",
           retention="7 days", level="INFO")

# Global model cache
MODELS = {}
FEATURE_SCHEMAS = {}
TECH_ENGINEER = TechnicalFeatureEngineer()


def load_model_and_schema(pair: str):
    """
    Load trained model and feature schema for a currency pair
    """
    if pair in MODELS:
        return MODELS[pair], FEATURE_SCHEMAS[pair]

    logger.info(f"Loading model for {pair}...")

    # Load model
    model_path = MODELS_DIR / f"{pair}_model.pth"
    if not Path(f"{model_path}_config.pkl").exists():
        raise FileNotFoundError(f"Model not found for {pair}")

    model = HybridEnsemble()
    model.load_model(str(model_path))

    # Load feature schema
    schema_path = MODELS_DIR / f"{pair}_feature_schema.json"
    if not schema_path.exists():
        raise FileNotFoundError(f"Feature schema not found for {pair}")

    with open(schema_path, 'r') as f:
        schema = json.load(f)

    # Cache
    MODELS[pair] = model
    FEATURE_SCHEMAS[pair] = schema

    logger.success(f"Loaded {pair} model ({schema['n_features']} features)")
    return model, schema


def engineer_features_from_ohlcv(df_ohlcv: pd.DataFrame, pair: str) -> tuple:
    """
    Engineer features from OHLCV data using EXACT training pipeline

    Returns:
        feature_array: numpy array of features
        feature_names: list of feature names in order
    """
    logger.info(
        f"Engineering features for {pair} from {len(df_ohlcv)} candles")

    # Technical features
    df_features = TECH_ENGINEER.calculate_all_features(df_ohlcv.copy())
    df_features = TECH_ENGINEER.calculate_feature_crosses(df_features)

    # Add macro features (set to 0 if no events - same as training)
    df_features["tau_pre"] = 0.0
    df_features["tau_post"] = 0.0
    df_features["weighted_surprise"] = 0.0

    # Drop NaN
    df_features.dropna(inplace=True)

    # Extract feature columns (SAME LOGIC AS TRAINING)
    exclude_cols = [
        "open", "high", "low", "close", "volume",
        "forward_close", "forward_return", "forward_return_pips",
        "target", "target_class", "timestamp"
    ]

    feature_cols = [
        col for col in df_features.columns
        if col not in exclude_cols
    ]

    logger.success(
        f"Engineered {len(feature_cols)} features, {len(df_features)} samples after dropna")

    return df_features[feature_cols].values, feature_cols


def validate_features(feature_names: list, schema: dict, pair: str):
    """
    Validate that engineered features match model training schema
    """
    expected_features = schema['feature_names']
    expected_count = schema['n_features']

    if len(feature_names) != expected_count:
        raise ValueError(
            f"{pair}: Feature count mismatch! "
            f"Expected {expected_count}, got {len(feature_names)}"
        )

    # Check feature names and order
    mismatches = []
    for i, (expected, actual) in enumerate(zip(expected_features, feature_names)):
        if expected != actual:
            mismatches.append(
                f"Position {i}: expected '{expected}', got '{actual}'")

    if mismatches:
        raise ValueError(
            f"{pair}: Feature order mismatch!\n" + "\n".join(mismatches[:10])
        )

    logger.success(f"{pair}: Feature validation passed ✓")


@app.route('/health', methods=['GET'])
def health_check():
    """
    Health check endpoint
    """
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'models_loaded': list(MODELS.keys()),
        'supported_pairs': CURRENCY_PAIRS
    })


@app.route('/predict', methods=['POST'])
def predict():
    """
    Main prediction endpoint

    Expected JSON format:
    {
        "pair": "EUR_USD",
        "ohlcv": [
            {"timestamp": "2025-01-01 00:00:00", "open": 1.1000, "high": 1.1010, "low": 1.0990, "close": 1.1005, "volume": 1000},
            ...
        ]
    }

    Returns:
    {
        "pair": "EUR_USD",
        "prediction": "BUY" | "SELL" | "HOLD",
        "confidence": 0.85,
        "probabilities": {"BUY": 0.85, "SELL": 0.10, "HOLD": 0.05},
        "timestamp": "2025-01-01 00:05:00",
        "feature_count": 58,
        "status": "success"
    }
    """
    try:
        # Parse request
        data = request.get_json()

        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400

        pair = data.get('pair')
        ohlcv_data = data.get('ohlcv')

        if not pair or not ohlcv_data:
            return jsonify({'error': 'Missing required fields: pair, ohlcv'}), 400

        if pair not in CURRENCY_PAIRS:
            return jsonify({'error': f'Unsupported pair: {pair}'}), 400

        logger.info(
            f"Prediction request for {pair} with {len(ohlcv_data)} candles")

        # Convert OHLCV to DataFrame
        df_ohlcv = pd.DataFrame(ohlcv_data)
        df_ohlcv['timestamp'] = pd.to_datetime(df_ohlcv['timestamp'])

        # Validate OHLCV data
        required_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        missing_cols = [
            col for col in required_cols if col not in df_ohlcv.columns]
        if missing_cols:
            return jsonify({'error': f'Missing OHLCV columns: {missing_cols}'}), 400

        # Check minimum data
        if len(df_ohlcv) < 250:  # Need enough for longest indicators
            return jsonify({
                'error': f'Insufficient data: need at least 250 candles, got {len(df_ohlcv)}'
            }), 400

        # Load model and schema
        model, schema = load_model_and_schema(pair)

        # Engineer features
        feature_array, feature_names = engineer_features_from_ohlcv(
            df_ohlcv, pair)

        # Validate features match training schema
        validate_features(feature_names, schema, pair)

        # Get most recent sample for prediction
        X_latest = feature_array[-1:, :]  # Last row

        # Predict
        prediction_class = model.predict(X_latest)[0]
        prediction_proba = model.predict_proba(X_latest)[0]

        # Map class to signal
        class_map = {0: "SELL", 1: "HOLD", 2: "BUY"}
        signal = class_map[prediction_class]
        confidence = float(prediction_proba[prediction_class])

        # Prepare response
        response = {
            'pair': pair,
            'prediction': signal,
            'confidence': round(confidence, 4),
            'probabilities': {
                'BUY': round(float(prediction_proba[2]), 4),
                'SELL': round(float(prediction_proba[0]), 4),
                'HOLD': round(float(prediction_proba[1]), 4)
            },
            'timestamp': df_ohlcv['timestamp'].iloc[-1].isoformat(),
            'feature_count': len(feature_names),
            'candles_processed': len(df_ohlcv),
            'candles_used': len(feature_array),
            'status': 'success'
        }

        logger.success(f"{pair}: {signal} (confidence: {confidence:.2%})")
        return jsonify(response)

    except FileNotFoundError as e:
        logger.error(f"Model not found: {e}")
        return jsonify({'error': f'Model not found: {str(e)}'}), 404

    except ValueError as e:
        logger.error(f"Validation error: {e}")
        return jsonify({'error': f'Validation error: {str(e)}'}), 400

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return jsonify({'error': f'Internal error: {str(e)}'}), 500


@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    """
    Batch prediction for multiple pairs

    Expected JSON format:
    {
        "requests": [
            {"pair": "EUR_USD", "ohlcv": [...]},
            {"pair": "GBP_USD", "ohlcv": [...]}
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
                if response[1] == 200:  # Success
                    results.append(response[0].get_json())
                else:
                    results.append(
                        {'pair': req.get('pair'), 'error': response[0].get_json()})

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
    logger.info("=" * 80)
    logger.info("STARTING FOREX INFERENCE SERVER")
    logger.info("=" * 80)
    logger.info(f"Supported pairs: {', '.join(CURRENCY_PAIRS)}")
    logger.info(f"Models directory: {MODELS_DIR}")

    # Run server
    app.run(
        host='0.0.0.0',  # Allow external connections
        port=5000,
        debug=False,  # Set to False in production
        threaded=True
    )
