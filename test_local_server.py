"""
Test the local inference server
Usage: python test_local_server.py
"""
import requests
import json
from datetime import datetime, timedelta

SERVER_URL = "http://localhost:5000"


def test_health():
    """Test health endpoint"""
    print("\n" + "=" * 80)
    print("TEST 1: Health Check")
    print("=" * 80)

    response = requests.get(f"{SERVER_URL}/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    return response.status_code == 200


def test_prediction():
    """Test prediction endpoint with sample data"""
    print("\n" + "=" * 80)
    print("TEST 2: Prediction Request")
    print("=" * 80)

    # Generate sample OHLCV data
    def generate_sample_data(bars, base_price=1.1000):
        data = []
        current_time = datetime.now()

        for i in range(bars):
            timestamp = (current_time - timedelta(minutes=5*i)
                         ).strftime("%Y-%m-%d %H:%M:%S")
            # Simple random walk
            price = base_price + (i % 100) * 0.0001
            data.append({
                "timestamp": timestamp,
                "open": price,
                "high": price + 0.0005,
                "low": price - 0.0005,
                "close": price + 0.0002,
                "volume": 1000 + (i % 500)
            })

        return list(reversed(data))  # Oldest first

    # Create request payload
    payload = {
        "pair": "EUR_USD",
        "ohlcv_m5": generate_sample_data(500, 1.1000),
        "ohlcv_h1": generate_sample_data(300, 1.1000),
        "ohlcv_h4": generate_sample_data(250, 1.1000),
        "events": []  # No calendar events for this test
    }

    print(f"Sending request with:")
    print(f"  - M5 bars: {len(payload['ohlcv_m5'])}")
    print(f"  - H1 bars: {len(payload['ohlcv_h1'])}")
    print(f"  - H4 bars: {len(payload['ohlcv_h4'])}")
    print(f"  - Events: {len(payload['events'])}")
    print("\nWaiting for response...")

    try:
        response = requests.post(
            f"{SERVER_URL}/predict",
            json=payload,
            timeout=60
        )

        print(f"\nStatus: {response.status_code}")

        if response.status_code == 200:
            result = response.json()
            print(f"\n✅ SUCCESS!")
            print(f"\nPrediction Results:")
            print(f"  Pair: {result.get('pair')}")
            print(f"  Signal: {result.get('prediction')}")
            print(f"  Confidence: {result.get('confidence', 0)*100:.2f}%")
            print(f"  Quality Score: {result.get('quality_score')}/100")
            print(
                f"  Position Size: {result.get('position_size_pct', 0)*100:.0f}%")
            print(f"  Should Trade: {result.get('should_trade')}")
            print(f"\nProbabilities:")
            probs = result.get('probabilities', {})
            print(f"  BUY:  {probs.get('BUY', 0)*100:.2f}%")
            print(f"  SELL: {probs.get('SELL', 0)*100:.2f}%")
            print(f"  HOLD: {probs.get('HOLD', 0)*100:.2f}%")
            print(f"\nMetadata:")
            print(f"  Features: {result.get('feature_count')}")
            print(f"  Candles used: {result.get('candles_used')}")
            return True
        else:
            print(f"\n❌ ERROR!")
            print(f"Response: {response.text}")
            return False

    except requests.exceptions.Timeout:
        print("\n❌ Request timed out (>60s)")
        return False
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return False


def test_model_info():
    """Test model info endpoint"""
    print("\n" + "=" * 80)
    print("TEST 3: Model Info")
    print("=" * 80)

    response = requests.get(f"{SERVER_URL}/model_info/EUR_USD")
    print(f"Status: {response.status_code}")

    if response.status_code == 200:
        info = response.json()
        print(f"\nModel Information:")
        print(f"  Pair: {info.get('pair')}")
        print(f"  Features: {info.get('n_features')}")
        print(f"  Trained: {info.get('trained_date')}")
        print(f"  Loaded: {info.get('is_loaded')}")
        return True
    else:
        print(f"Response: {response.text}")
        return False


def main():
    print("=" * 80)
    print("FOREX INFERENCE SERVER - LOCAL TEST SUITE")
    print("=" * 80)
    print(f"Server URL: {SERVER_URL}")
    print("\nMake sure the server is running:")
    print("  python run_local_server.py")
    print("=" * 80)

    input("\nPress Enter to start tests...")

    results = []

    # Run tests
    results.append(("Health Check", test_health()))
    results.append(("Prediction", test_prediction()))
    results.append(("Model Info", test_model_info()))

    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)

    for test_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status} - {test_name}")

    total = len(results)
    passed = sum(1 for _, p in results if p)

    print(f"\nTotal: {passed}/{total} tests passed")
    print("=" * 80)


if __name__ == "__main__":
    main()
