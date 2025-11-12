"""
Production Inference Server for Forex Trading Models
Handles feature engineering, model loading, and predictions

Architecture:
- EA sends OHLCV data + raw calendar events
- Server engineers features using EXACT training pipeline
- Server validates features match model schema
- Server returns predictions with confidence scores
"""
import os
from src.config import MODELS_DIR, CURRENCY_PAIRS, SENTIMENT_EMA_PERIODS, ENABLE_LIVE_SENTIMENT, SENTIMENT_CACHE_MINUTES
from src.feature_engineering.technical_features import TechnicalFeatureEngineer
from src.data_acquisition.macro_data import MacroDataAcquisition
from src.data_acquisition.news_data import NewsDataAcquisition
from src.feature_engineering.sentiment_features import SentimentAnalyzer
from src.models.hybrid_ensemble import HybridEnsemble
from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
import json
import joblib
from pathlib import Path
from loguru import logger
from datetime import datetime, timedelta
import sys

# Add project root
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


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

# Initialize feature engineers
try:
    logger.info("Initializing TechnicalFeatureEngineer...")
    TECH_ENGINEER = TechnicalFeatureEngineer()
    logger.info("✓ TechnicalFeatureEngineer initialized")
except Exception as e:
    logger.error(f"✗ Failed to initialize TechnicalFeatureEngineer: {e}")
    raise

try:
    logger.info("Initializing MacroDataAcquisition...")
    MACRO_ENGINEER = MacroDataAcquisition()
    logger.info("✓ MacroDataAcquisition initialized")
except Exception as e:
    logger.error(f"✗ Failed to initialize MacroDataAcquisition: {e}")
    raise

# Only load sentiment components if live sentiment is enabled
if ENABLE_LIVE_SENTIMENT:
    try:
        logger.info("Initializing NewsDataAcquisition...")
        NEWS_ACQUIRER = NewsDataAcquisition()
        logger.info("✓ NewsDataAcquisition initialized")

        logger.info("Initializing SentimentAnalyzer...")
        SENTIMENT_ANALYZER = SentimentAnalyzer()
        logger.info("✓ SentimentAnalyzer initialized")
    except Exception as e:
        logger.error(f"✗ Failed to initialize sentiment components: {e}")
        raise
else:
    NEWS_ACQUIRER = None
    SENTIMENT_ANALYZER = None
    logger.info("Sentiment analysis disabled")

logger.info("=" * 80)
logger.info("SERVER INITIALIZATION COMPLETE")
logger.info("=" * 80)


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


def engineer_features_from_ohlcv(df_m5: pd.DataFrame, df_h1: pd.DataFrame, df_h4: pd.DataFrame, pair: str) -> tuple:
    """
    Engineer features from OHLCV data using EXACT training pipeline
    
    CRITICAL: Must match training exactly (81 features)
    - 67 base technical features (M5)
    - 14 multi-timeframe features (H1 + H4)
    - 3 macro features (tau_pre, tau_post, weighted_surprise)
    - 0 sentiment features (not trained)

    Args:
        df_m5: M5 OHLCV DataFrame
        df_h1: H1 OHLCV DataFrame
        df_h4: H4 OHLCV DataFrame
        pair: Currency pair

    Returns:
        df_features: DataFrame with all features
        feature_cols: List of feature column names
    """
    logger.info(
        f"Engineering features for {pair} from M5={len(df_m5)}, H1={len(df_h1)}, H4={len(df_h4)} candles")

    # Step 1: Base technical features on M5 (67 features)
    df_features = TECH_ENGINEER.calculate_all_features(df_m5.copy())
    df_features = TECH_ENGINEER.calculate_feature_crosses(df_features)
    logger.info(f"✓ Calculated {len(df_features.columns)} base technical features")

    # Step 2: Multi-timeframe features (14 features) using REAL H1/H4 data
    try:
        higher_timeframes = {
            'H1': df_h1,
            'H4': df_h4
        }
        
        df_features = TECH_ENGINEER.add_multi_timeframe_features(
            df_primary=df_features,
            higher_timeframes=higher_timeframes
        )
        logger.info(f"✓ Added multi-timeframe features from real H1 + H4 data")
    except Exception as e:
        logger.error(f"Failed to add MTF features: {e}")
        raise ValueError(f"MTF feature engineering failed: {e}")

    # Step 3: Macro features (3 features) - will be populated by engineer_macro_features()
    df_features["tau_pre"] = 0.0
    df_features["tau_post"] = 0.0
    df_features["weighted_surprise"] = 0.0

    # Drop NaN
    initial_len = len(df_features)
    df_features.dropna(inplace=True)
    logger.info(f"Dropped {initial_len - len(df_features)} rows with NaNs")

    # Extract feature columns (SAME LOGIC AS TRAINING)
    exclude_cols = [
        "open", "high", "low", "close", "volume",
        "forward_close", "forward_return", "forward_return_pips",
        "target", "target_class", "timestamp", "date"
    ]

    feature_cols = [
        col for col in df_features.columns
        if col not in exclude_cols
    ]

    logger.success(
        f"Engineered {len(feature_cols)} features, {len(df_features)} samples after dropna")

    return df_features, feature_cols  # Return full DataFrame for macro engineering


def engineer_macro_features(events_data: list, df_features: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer macro features from raw calendar events using EXACT training pipeline

    Args:
        events_data: List of raw calendar events from EA
            [
                {
                    "timestamp": "2025-11-08T14:30:00",
                    "event_name": "NFP",
                    "country": "US",
                    "actual": 150000,
                    "forecast": 180000,
                    "previous": 200000,
                    "impact": "high"
                }
            ]
        df_features: DataFrame with technical features and index (timestamps)

    Returns:
        DataFrame with macro features added (tau_pre, tau_post, weighted_surprise)
    """
    if not events_data or len(events_data) == 0:
        logger.info("No calendar events provided - using zero macro features")
        return df_features

    logger.info(
        f"Engineering macro features from {len(events_data)} calendar events")

    try:
        # Convert events to DataFrame with required structure
        df_events = pd.DataFrame(events_data)

        # Ensure timestamp column is datetime
        df_events['timestamp'] = pd.to_datetime(df_events['timestamp'])

        # Calculate surprise factor: (actual - forecast) / std(previous values)
        # This matches the training data calculation
        if 'actual' in df_events.columns and 'forecast' in df_events.columns:
            # Simple normalized surprise
            df_events['surprise_zscore'] = (
                (df_events['actual'] - df_events['forecast']) /
                (df_events['previous'].std() + 1e-8)  # Avoid division by zero
            )
        else:
            logger.warning(
                "Events missing actual/forecast - using zero surprise")
            df_events['surprise_zscore'] = 0.0

        # Use EXACT same function as training
        df_features = MACRO_ENGINEER.calculate_temporal_proximity(
            events_df=df_events,
            price_df=df_features,
            pre_event_hours=48,   # Same as training
            post_event_hours=48,  # Same as training
            decay_lambda=0.1      # Same as training
        )

        logger.success(
            f"Macro features engineered: tau_pre, tau_post, weighted_surprise")

    except Exception as e:
        logger.error(f"Failed to engineer macro features: {e}")
        logger.info("Falling back to zero macro features")
        # Keep the zeros already set

    return df_features


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
        "ohlcv_m5": [  // M5 candles (250+)
            {"timestamp": "2025-01-01 00:00:00", "open": 1.1000, "high": 1.1010, "low": 1.0990, "close": 1.1005, "volume": 1000},
            ...
        ],
        "ohlcv_h1": [  // H1 candles (250+)
            {"timestamp": "2025-01-01 00:00:00", "open": 1.1000, "high": 1.1010, "low": 1.0990, "close": 1.1005, "volume": 1000},
            ...
        ],
        "ohlcv_h4": [  // H4 candles (250+)
            {"timestamp": "2025-01-01 00:00:00", "open": 1.1000, "high": 1.1010, "low": 1.0990, "close": 1.1005, "volume": 1000},
            ...
        ],
        "events": [  // OPTIONAL: Raw calendar events for macro features
            {
                "timestamp": "2025-01-01 14:30:00",
                "event_name": "NFP",
                "country": "US",
                "actual": 150000,
                "forecast": 180000,
                "previous": 200000,
                "impact": "high"
            }
        ]
    }

    Returns:
    {
        "pair": "EUR_USD",
        "prediction": "BUY" | "SELL" | "HOLD",
        "confidence": 0.85,
        "probabilities": {"BUY": 0.85, "SELL": 0.10, "HOLD": 0.05},
        "timestamp": "2025-01-01 00:05:00",
        "feature_count": 81,  // 67 base + 14 MTF + 3 macro + 0 sentiment
        "status": "success"
    }
    """
    try:
        # Parse request
        data = request.get_json()

        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400

        pair = data.get('pair')
        ohlcv_m5_data = data.get('ohlcv_m5')
        ohlcv_h1_data = data.get('ohlcv_h1')
        ohlcv_h4_data = data.get('ohlcv_h4')
        events_data = data.get('events', [])  # Optional calendar events

        if not pair or not ohlcv_m5_data or not ohlcv_h1_data or not ohlcv_h4_data:
            return jsonify({'error': 'Missing required fields: pair, ohlcv_m5, ohlcv_h1, ohlcv_h4'}), 400

        # Resolve pair alias (e.g., BTCUSD → EUR_USD for testing)
        original_pair = pair
        if pair in PAIR_ALIASES:
            pair = PAIR_ALIASES[pair]
            logger.info(f"Mapping {original_pair} → {pair} (using alias)")

        if pair not in CURRENCY_PAIRS:
            return jsonify({'error': f'Unsupported pair: {original_pair}'}), 400

        logger.info(
            f"Prediction request for {original_pair} ({pair}) with M5={len(ohlcv_m5_data)}, H1={len(ohlcv_h1_data)}, H4={len(ohlcv_h4_data)} candles, {len(events_data)} events")

        # Convert OHLCV to DataFrames
        df_m5 = pd.DataFrame(ohlcv_m5_data)
        df_m5['timestamp'] = pd.to_datetime(df_m5['timestamp'])
        df_m5.set_index('timestamp', inplace=True)
        
        df_h1 = pd.DataFrame(ohlcv_h1_data)
        df_h1['timestamp'] = pd.to_datetime(df_h1['timestamp'])
        df_h1.set_index('timestamp', inplace=True)
        
        df_h4 = pd.DataFrame(ohlcv_h4_data)
        df_h4['timestamp'] = pd.to_datetime(df_h4['timestamp'])
        df_h4.set_index('timestamp', inplace=True)

        # Validate OHLCV data for all timeframes
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        
        for tf_name, df_tf in [('M5', df_m5), ('H1', df_h1), ('H4', df_h4)]:
            missing_cols = [col for col in required_cols if col not in df_tf.columns]
            if missing_cols:
                return jsonify({'error': f'Missing {tf_name} OHLCV columns: {missing_cols}'}), 400
            
            # Check minimum data
            if len(df_tf) < 250:
                return jsonify({
                    'error': f'Insufficient {tf_name} data: need at least 250 candles, got {len(df_tf)}'
                }), 400

        # Load model and schema
        model, schema = load_model_and_schema(pair)

        # Engineer features using real M5, H1, and H4 data (81 features total)
        df_features, feature_names = engineer_features_from_ohlcv(
            df_m5, df_h1, df_h4, pair)

        # Engineer macro features (3 features: tau_pre, tau_post, weighted_surprise)
        df_features = engineer_macro_features(events_data, df_features)

        # SENTIMENT FEATURES DISABLED
        # Models were trained WITHOUT sentiment features (news dataset not attached in Kaggle)
        # DO NOT add sentiment features here - it will cause feature count mismatch!
        # If you want sentiment, retrain models with news data first.
        if ENABLE_LIVE_SENTIMENT:
            logger.warning(
                "⚠ ENABLE_LIVE_SENTIMENT=True but models not trained with sentiment! "
                "Ignoring sentiment to match training (81 features)."
            )

        # Extract final feature array
        exclude_cols = [
            "open", "high", "low", "close", "volume",
            "forward_close", "forward_return", "forward_return_pips",
            "target", "target_class", "date"  # Exclude 'date' column from sentiment merge
        ]
        feature_cols = [
            col for col in df_features.columns if col not in exclude_cols]
        feature_array = df_features[feature_cols].values

        # Validate features match training schema
        validate_features(feature_cols, schema, pair)

        # For LSTM sequences, we need to pass enough samples for sequence creation
        # The model internally creates sequences from the data
        # Pass all available samples and the model will predict on the last sequence
        X_for_prediction = feature_array  # Pass all samples

        # Predict - model returns predictions for all valid sequences
        # We want the last prediction (most recent)
        prediction_proba = model.predict_proba(X_for_prediction)
        prediction_class = model.predict(X_for_prediction)

        # Get the last prediction (most recent sequence)
        prediction_class = prediction_class[-1]
        prediction_proba = prediction_proba[-1]

        # Map class to signal
        class_map = {0: "SELL", 1: "HOLD", 2: "BUY"}
        raw_signal = class_map[prediction_class]
        confidence = float(prediction_proba[prediction_class])

        # Apply fuzzy logic quality scoring (same as training pipeline)
        from src.models.signal_quality import SignalQualityScorer
        quality_scorer = SignalQualityScorer()
        
        # Get the last feature row for quality calculation
        last_features = df_features.iloc[-1]
        
        # Calculate quality score
        quality_score, quality_components = quality_scorer.calculate_quality(
            prediction_proba=prediction_proba,
            features=last_features,
            predicted_class=prediction_class
        )
        
        # Get position size multiplier
        position_size_pct = quality_scorer.get_position_size_multiplier(quality_score)
        
        # Apply quality filter (override signal if quality too low)
        if quality_score < quality_scorer.min_quality_threshold:
            signal = "HOLD"  # Override to HOLD if quality too low
            quality_filtered = True
            logger.warning(
                f"{original_pair}: Signal {raw_signal} filtered to HOLD due to low quality ({quality_score:.1f}/100)"
            )
        else:
            signal = raw_signal
            quality_filtered = False

        # Prepare response with fuzzy quality metrics
        response = {
            'pair': original_pair,  # Return original pair name (e.g., BTCUSD)
            'model_pair': pair,  # Show which model was used (e.g., EUR_USD)
            'prediction': signal,
            'raw_prediction': raw_signal,  # Original model prediction before quality filter
            'confidence': round(confidence, 4),
            'probabilities': {
                'BUY': round(float(prediction_proba[2]), 4),
                'SELL': round(float(prediction_proba[0]), 4),
                'HOLD': round(float(prediction_proba[1]), 4)
            },
            # Fuzzy quality metrics
            'quality_score': round(quality_score, 2),
            'quality_components': {
                'confidence': round(quality_components['confidence'], 2),
                'trend': round(quality_components['trend'], 2),
                'volatility': round(quality_components['volatility'], 2),
                'momentum': round(quality_components['momentum'], 2)
            },
            'position_size_pct': round(position_size_pct, 2),
            'quality_filtered': quality_filtered,
            'should_trade': quality_scorer.should_trade(quality_score),
            # Metadata
            'timestamp': df_m5.index[-1].isoformat(),
            'feature_count': len(feature_names),
            'candles_m5': len(df_m5),
            'candles_h1': len(df_h1),
            'candles_h4': len(df_h4),
            'candles_used': len(feature_array),
            'status': 'success'
        }

        # Enhanced logging with quality metrics
        logger.success(
            f"{original_pair} ({pair} model): {signal} (confidence: {confidence:.2%}, "
            f"quality: {quality_score:.1f}/100, position: {position_size_pct*100:.0f}%)"
        )
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
