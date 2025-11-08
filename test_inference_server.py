"""
Inference Server Test Suite
Tests all endpoints and validates feature engineering
"""
import requests
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from loguru import logger

# Server configuration
BASE_URL = "http://127.0.0.1:5000"

logger.info("=" * 80)
logger.info("INFERENCE SERVER TEST SUITE")
logger.info("=" * 80)


def generate_test_ohlcv(n_candles=500, pair="EUR_USD"):
    """
    Generate realistic test OHLCV data
    """
    logger.info(f"Generating {n_candles} test candles for {pair}")

    np.random.seed(42)
    base_price = 1.1000 if "EUR" in pair else 1.3000

    returns = np.random.randn(n_candles) * 0.0005
    close_prices = base_price * np.cumprod(1 + returns)

    high_prices = close_prices * \
        (1 + np.abs(np.random.randn(n_candles) * 0.0002))
    low_prices = close_prices * \
        (1 - np.abs(np.random.randn(n_candles) * 0.0002))
    open_prices = np.roll(close_prices, 1)
    open_prices[0] = close_prices[0]

    high_prices = np.maximum.reduce([open_prices, close_prices, high_prices])
    low_prices = np.minimum.reduce([open_prices, close_prices, low_prices])

    volume = np.random.randint(100, 10000, n_candles)

    timestamps = pd.date_range(
        end=datetime.now(),
        periods=n_candles,
        freq='5min'
    )

    ohlcv = []
    for i in range(n_candles):
        ohlcv.append({
            'timestamp': timestamps[i].strftime('%Y-%m-%d %H:%M:%S'),
            'open': float(open_prices[i]),
            'high': float(high_prices[i]),
            'low': float(low_prices[i]),
            'close': float(close_prices[i]),
            'volume': int(volume[i])
        })

    return ohlcv


def test_health_check():
    """
    Test /health endpoint
    """
    logger.info("\n" + "=" * 60)
    logger.info("TEST 1: Health Check")
    logger.info("=" * 60)

    try:
        response = requests.get(f"{BASE_URL}/health", timeout=5)

        if response.status_code == 200:
            data = response.json()
            logger.success(f"✓ Health check passed")
            logger.info(f"  Status: {data['status']}")
            logger.info(f"  Models loaded: {data['models_loaded']}")
            logger.info(f"  Supported pairs: {data['supported_pairs']}")
            return True
        else:
            logger.error(f"✗ Health check failed: {response.status_code}")
            return False

    except requests.exceptions.ConnectionError:
        logger.error("✗ Cannot connect to server. Is it running?")
        return False
    except Exception as e:
        logger.error(f"✗ Health check error: {e}")
        return False


def test_single_prediction(pair="EUR_USD"):
    """
    Test /predict endpoint with single pair
    """
    logger.info("\n" + "=" * 60)
    logger.info(f"TEST 2: Single Prediction ({pair})")
    logger.info("=" * 60)

    try:
        # Generate test data
        ohlcv = generate_test_ohlcv(n_candles=500, pair=pair)

        payload = {
            'pair': pair,
            'ohlcv': ohlcv
        }

        logger.info(f"Sending prediction request for {pair}...")
        response = requests.post(
            f"{BASE_URL}/predict",
            json=payload,
            timeout=30
        )

        if response.status_code == 200:
            data = response.json()
            logger.success(f"✓ Prediction successful")
            logger.info(f"  Pair: {data['pair']}")
            logger.info(f"  Prediction: {data['prediction']}")
            logger.info(f"  Confidence: {data['confidence']:.2%}")
            logger.info(f"  Probabilities:")
            for signal, prob in data['probabilities'].items():
                logger.info(f"    {signal}: {prob:.2%}")
            logger.info(f"  Features used: {data['feature_count']}")
            logger.info(f"  Candles processed: {data['candles_processed']}")
            logger.info(f"  Candles used: {data['candles_used']}")

            # Validate response structure
            required_fields = ['pair', 'prediction',
                               'confidence', 'probabilities', 'status']
            missing = [f for f in required_fields if f not in data]
            if missing:
                logger.warning(f"  Missing fields: {missing}")

            # Validate probabilities sum to 1
            prob_sum = sum(data['probabilities'].values())
            if abs(prob_sum - 1.0) > 0.01:
                logger.warning(f"  Probabilities don't sum to 1.0: {prob_sum}")

            return True
        else:
            logger.error(f"✗ Prediction failed: {response.status_code}")
            logger.error(f"  Response: {response.json()}")
            return False

    except Exception as e:
        logger.error(f"✗ Prediction test error: {e}")
        return False


def test_insufficient_data():
    """
    Test error handling with insufficient data
    """
    logger.info("\n" + "=" * 60)
    logger.info("TEST 3: Insufficient Data Error Handling")
    logger.info("=" * 60)

    try:
        # Only 50 candles (should fail - need 250+)
        ohlcv = generate_test_ohlcv(n_candles=50)

        payload = {
            'pair': 'EUR_USD',
            'ohlcv': ohlcv
        }

        response = requests.post(
            f"{BASE_URL}/predict", json=payload, timeout=10)

        if response.status_code == 400:
            logger.success("✓ Correctly rejected insufficient data")
            logger.info(f"  Error message: {response.json().get('error')}")
            return True
        else:
            logger.error(
                f"✗ Should have rejected insufficient data: {response.status_code}")
            return False

    except Exception as e:
        logger.error(f"✗ Test error: {e}")
        return False


def test_invalid_pair():
    """
    Test error handling with unsupported pair
    """
    logger.info("\n" + "=" * 60)
    logger.info("TEST 4: Invalid Pair Error Handling")
    logger.info("=" * 60)

    try:
        ohlcv = generate_test_ohlcv(n_candles=500)

        payload = {
            'pair': 'XYZ_ABC',  # Invalid pair
            'ohlcv': ohlcv
        }

        response = requests.post(
            f"{BASE_URL}/predict", json=payload, timeout=10)

        if response.status_code == 400:
            logger.success("✓ Correctly rejected invalid pair")
            logger.info(f"  Error message: {response.json().get('error')}")
            return True
        else:
            logger.error(
                f"✗ Should have rejected invalid pair: {response.status_code}")
            return False

    except Exception as e:
        logger.error(f"✗ Test error: {e}")
        return False


def test_model_info(pair="EUR_USD"):
    """
    Test /model_info endpoint
    """
    logger.info("\n" + "=" * 60)
    logger.info(f"TEST 5: Model Info ({pair})")
    logger.info("=" * 60)

    try:
        response = requests.get(f"{BASE_URL}/model_info/{pair}", timeout=10)

        if response.status_code == 200:
            data = response.json()
            logger.success(f"✓ Model info retrieved")
            logger.info(f"  Pair: {data['pair']}")
            logger.info(f"  Features: {data['n_features']}")
            logger.info(
                f"  Model version: {data.get('model_version', 'unknown')}")
            logger.info(
                f"  Trained date: {data.get('trained_date', 'unknown')}")
            logger.info(f"  First 10 features: {data['feature_names'][:10]}")
            return True
        else:
            logger.error(f"✗ Model info failed: {response.status_code}")
            return False

    except Exception as e:
        logger.error(f"✗ Model info test error: {e}")
        return False


def test_batch_prediction():
    """
    Test /batch_predict endpoint
    """
    logger.info("\n" + "=" * 60)
    logger.info("TEST 6: Batch Prediction")
    logger.info("=" * 60)

    try:
        pairs = ['EUR_USD', 'GBP_USD']
        requests_list = []

        for pair in pairs:
            ohlcv = generate_test_ohlcv(n_candles=500, pair=pair)
            requests_list.append({
                'pair': pair,
                'ohlcv': ohlcv
            })

        payload = {'requests': requests_list}

        logger.info(f"Sending batch request for {len(pairs)} pairs...")
        response = requests.post(
            f"{BASE_URL}/batch_predict",
            json=payload,
            timeout=60
        )

        if response.status_code == 200:
            data = response.json()
            logger.success(f"✓ Batch prediction successful")
            logger.info(f"  Results count: {data['count']}")

            for result in data['results']:
                if 'error' in result:
                    logger.warning(
                        f"  {result.get('pair', 'unknown')}: ERROR - {result['error']}")
                else:
                    logger.info(
                        f"  {result['pair']}: {result['prediction']} ({result['confidence']:.2%})")

            return True
        else:
            logger.error(f"✗ Batch prediction failed: {response.status_code}")
            return False

    except Exception as e:
        logger.error(f"✗ Batch prediction test error: {e}")
        return False


def test_all_supported_pairs():
    """
    Test predictions for all supported pairs
    """
    logger.info("\n" + "=" * 60)
    logger.info("TEST 7: All Supported Pairs")
    logger.info("=" * 60)

    # Get supported pairs from health check
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=5)
        supported_pairs = response.json()['supported_pairs']
    except:
        supported_pairs = ['EUR_USD', 'GBP_USD', 'USD_JPY', 'AUD_USD']

    results = {}
    for pair in supported_pairs:
        logger.info(f"\nTesting {pair}...")
        success = test_single_prediction(pair)
        results[pair] = success

    # Summary
    passed = sum(results.values())
    total = len(results)

    logger.info("\n" + "=" * 60)
    logger.info(f"Results: {passed}/{total} pairs passed")
    for pair, success in results.items():
        status = "✓" if success else "✗"
        logger.info(f"  {status} {pair}")

    return passed == total


def run_all_tests():
    """
    Run complete test suite
    """
    logger.info("\n" + "=" * 80)
    logger.info("STARTING FULL TEST SUITE")
    logger.info("=" * 80)

    tests = [
        ("Health Check", test_health_check),
        ("Single Prediction", lambda: test_single_prediction("EUR_USD")),
        ("Insufficient Data Handling", test_insufficient_data),
        ("Invalid Pair Handling", test_invalid_pair),
        ("Model Info", lambda: test_model_info("EUR_USD")),
        ("Batch Prediction", test_batch_prediction),
        ("All Supported Pairs", test_all_supported_pairs),
    ]

    results = []
    for test_name, test_func in tests:
        try:
            passed = test_func()
            results.append((test_name, passed))
        except Exception as e:
            logger.error(f"Test '{test_name}' crashed: {e}")
            results.append((test_name, False))

    # Final summary
    logger.info("\n" + "=" * 80)
    logger.info("TEST SUITE SUMMARY")
    logger.info("=" * 80)

    passed = sum(1 for _, success in results if success)
    total = len(results)

    for test_name, success in results:
        status = "✓ PASS" if success else "✗ FAIL"
        logger.info(f"{status}: {test_name}")

    logger.info(f"\n{passed}/{total} tests passed")

    if passed == total:
        logger.success("\n🎉 ALL TESTS PASSED!")
        return True
    else:
        logger.warning(f"\n⚠️  {total - passed} tests failed")
        return False


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test inference server")
    parser.add_argument(
        '--url', default="http://127.0.0.1:5000", help="Server URL")
    parser.add_argument('--test', choices=['all', 'health', 'predict', 'batch'],
                        default='all', help="Which test to run")

    args = parser.parse_args()
    BASE_URL = args.url

    if args.test == 'all':
        run_all_tests()
    elif args.test == 'health':
        test_health_check()
    elif args.test == 'predict':
        test_single_prediction()
    elif args.test == 'batch':
        test_batch_prediction()
