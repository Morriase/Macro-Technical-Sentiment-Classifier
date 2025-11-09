"""
Production Inference Server for Forex Trading Models
Handles feature engineering, model loading, and predictions

Architecture:
- EA sends OHLCV data + raw calendar events
- Server engineers features using EXACT training pipeline
- Server validates features match model schema
- Server returns predictions with confidence scores
"""
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

# Configure logger
logger.add("logs/inference_server.log", rotation="1 day",
           retention="7 days", level="INFO")

# Pair mapping for testing (allows BTC to use EUR_USD model)
PAIR_ALIASES = {
    'BTCUSD': 'EUR_USD',
    'BTC_USD': 'EUR_USD',
    'XBTUSD': 'EUR_USD',  # BitMEX notation
}

# Global model cache
MODELS = {}
FEATURE_SCHEMAS = {}
TECH_ENGINEER = TechnicalFeatureEngineer()
MACRO_ENGINEER = MacroDataAcquisition()  # For macro feature engineering

# Only load sentiment components if live sentiment is enabled
if ENABLE_LIVE_SENTIMENT:
    NEWS_ACQUIRER = NewsDataAcquisition()  # For live news acquisition
    SENTIMENT_ANALYZER = SentimentAnalyzer()  # For live sentiment analysis
else:
    NEWS_ACQUIRER = None
    SENTIMENT_ANALYZER = None


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

    # Technical features (55 features)
    df_features = TECH_ENGINEER.calculate_all_features(df_ohlcv.copy())
    df_features = TECH_ENGINEER.calculate_feature_crosses(df_features)

    # Macro features (3 features) - will be added after technical features
    # Placeholder: Will be populated by engineer_macro_features()
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
        "ohlcv": [
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
        events_data = data.get('events', [])  # Optional calendar events

        if not pair or not ohlcv_data:
            return jsonify({'error': 'Missing required fields: pair, ohlcv'}), 400

        # Resolve pair alias (e.g., BTCUSD → EUR_USD for testing)
        original_pair = pair
        if pair in PAIR_ALIASES:
            pair = PAIR_ALIASES[pair]
            logger.info(f"Mapping {original_pair} → {pair} (using alias)")

        if pair not in CURRENCY_PAIRS:
            return jsonify({'error': f'Unsupported pair: {original_pair}'}), 400

        logger.info(
            f"Prediction request for {original_pair} ({pair}) with {len(ohlcv_data)} candles, {len(events_data)} events")

        # Convert OHLCV to DataFrame
        df_ohlcv = pd.DataFrame(ohlcv_data)
        df_ohlcv['timestamp'] = pd.to_datetime(df_ohlcv['timestamp'])
        df_ohlcv.set_index('timestamp', inplace=True)

        # Validate OHLCV data
        required_cols = ['open', 'high', 'low', 'close', 'volume']
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

        # Engineer technical features (55 features)
        df_features, feature_names = engineer_features_from_ohlcv(
            df_ohlcv, pair)

        # Engineer macro features (3 features: tau_pre, tau_post, weighted_surprise)
        df_features = engineer_macro_features(events_data, df_features)

        # Engineer sentiment features (live with caching)
        if ENABLE_LIVE_SENTIMENT:
            logger.info("Live sentiment enabled - checking cache...")

            # Check cache first (cache stored on predict function object)
            if not hasattr(predict, '_sentiment_cache') or \
               not hasattr(predict, '_sentiment_cache_time') or \
               (datetime.now() - predict._sentiment_cache_time).total_seconds() > SENTIMENT_CACHE_MINUTES * 60:

                logger.info("Cache miss or expired - fetching live news...")
                end_date = datetime.now()
                start_date = end_date - timedelta(hours=2)
                live_news_df = NEWS_ACQUIRER.fetch_forex_news(
                    start_date, end_date)

                if not live_news_df.empty:
                    logger.info(
                        f"Processing {len(live_news_df)} live news articles for sentiment.")
                    daily_sentiment = SENTIMENT_ANALYZER.aggregate_daily_sentiment(
                        live_news_df)
                    time_weighted_sentiment = SENTIMENT_ANALYZER.calculate_time_weighted_sentiment(
                        daily_sentiment)

                    # Cache the sentiment data
                    predict._sentiment_cache = time_weighted_sentiment
                    predict._sentiment_cache_time = datetime.now()
                    logger.success("✓ Sentiment cached for future requests")
                else:
                    logger.warning("⚠ No live news found - caching zeros")
                    # Cache zeros to avoid repeated API calls
                    time_weighted_sentiment = pd.DataFrame()
                    predict._sentiment_cache = time_weighted_sentiment
                    predict._sentiment_cache_time = datetime.now()
            else:
                logger.info(
                    f"Using cached sentiment (age: {(datetime.now() - predict._sentiment_cache_time).seconds}s)")
                time_weighted_sentiment = predict._sentiment_cache

            # Merge sentiment if we have data
            if not time_weighted_sentiment.empty:
                # Reset index to merge on 'date' column, then restore index
                df_features = df_features.reset_index()
                df_features = pd.merge(
                    df_features,
                    time_weighted_sentiment,
                    left_on="date",
                    right_on="date",
                    how="left"
                )
                sentiment_cols = [
                    col for col in time_weighted_sentiment.columns if col != "date"]
                for col in sentiment_cols:
                    df_features[col] = df_features[col].fillna(0.0)
                df_features = df_features.set_index("date")
                logger.success("✓ Live sentiment features merged")
            else:
                # No cached data - add zeros
                for period in SENTIMENT_EMA_PERIODS:
                    df_features[f"polarity_ema_{period}"] = 0.0
                    df_features[f"positive_ema_{period}"] = 0.0
                    df_features[f"negative_ema_{period}"] = 0.0
        else:
            logger.info(
                "Live sentiment disabled - skipping sentiment features entirely")

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
        signal = class_map[prediction_class]
        confidence = float(prediction_proba[prediction_class])

        # Prepare response
        response = {
            'pair': original_pair,  # Return original pair name (e.g., BTCUSD)
            'model_pair': pair,  # Show which model was used (e.g., EUR_USD)
            'prediction': signal,
            'confidence': round(confidence, 4),
            'probabilities': {
                'BUY': round(float(prediction_proba[2]), 4),
                'SELL': round(float(prediction_proba[0]), 4),
                'HOLD': round(float(prediction_proba[1]), 4)
            },
            'timestamp': df_ohlcv.index[-1].isoformat(),
            'feature_count': len(feature_names),
            'candles_processed': len(df_ohlcv),
            'candles_used': len(feature_array),
            'status': 'success'
        }

        logger.success(
            f"{original_pair} ({pair} model): {signal} (confidence: {confidence:.2%})")
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
