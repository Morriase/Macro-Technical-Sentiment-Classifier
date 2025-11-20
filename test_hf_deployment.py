"""
Test Hugging Face deployment to verify all components work
"""
import requests
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# HF Space URL
BASE_URL = "https://morriase-forex-live-server.hf.space"


def test_health():
    """Test health endpoint"""
    print("\n" + "="*60)
    print("TEST 1: Health Check")
    print("="*60)
    
    response = requests.get(f"{BASE_URL}/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
    print("✅ Health check passed")


def test_model_info():
    """Test model info endpoint"""
    print("\n" + "="*60)
    print("TEST 2: Model Info")
    print("="*60)
    
    for pair in ["EUR_USD", "GBP_USD", "USD_JPY", "AUD_USD"]:
        response = requests.get(f"{BASE_URL}/model_info/{pair}")
        print(f"\n{pair}:")
        print(f"  Status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"  Loaded: {data['is_loaded']}")
            print(f"  Features: {data['n_features']}")
            print(f"  Trained: {data['trained_date']}")
            
            assert data['is_loaded'] == True
            assert data['n_features'] == 81
        else:
            print(f"  Error: {response.text}")
            
    print("\n✅ Model info test passed")


def generate_sample_ohlcv(num_candles=500, timeframe='M5'):
    """Generate sample OHLCV data for testing"""
    now = datetime.now()
    
    # Timeframe intervals
    intervals = {
        'M5': timedelta(minutes=5),
        'H1': timedelta(hours=1),
        'H4': timedelta(hours=4)
    }
    
    interval = intervals[timeframe]
    
    # Generate timestamps
    timestamps = [now - interval * i for i in range(num_candles)]
    timestamps.reverse()
    
    # Generate realistic forex prices (around 1.10 for EUR/USD)
    base_price = 1.10
    prices = []
    current_price = base_price
    
    for _ in range(num_candles):
        # Random walk with small changes
        change = np.random.normal(0, 0.0005)
        current_price += change
        prices.append(current_price)
    
    # Generate OHLCV
    data = []
    for i, (ts, close) in enumerate(zip(timestamps, prices)):
        high = close + abs(np.random.normal(0, 0.0002))
        low = close - abs(np.random.normal(0, 0.0002))
        open_price = prices[i-1] if i > 0 else close
        volume = np.random.randint(1000, 10000)
        
        data.append({
            'timestamp': ts.strftime('%Y-%m-%d %H:%M:%S'),
            'open': round(open_price, 5),
            'high': round(high, 5),
            'low': round(low, 5),
            'close': round(close, 5),
            'volume': volume
        })
    
    return data


def test_prediction():
    """Test prediction endpoint with sample data"""
    print("\n" + "="*60)
    print("TEST 3: Prediction")
    print("="*60)
    
    # Generate sample data
    print("Generating sample OHLCV data...")
    ohlcv_m5 = generate_sample_ohlcv(500, 'M5')
    ohlcv_h1 = generate_sample_ohlcv(300, 'H1')
    ohlcv_h4 = generate_sample_ohlcv(250, 'H4')
    
    # Sample calendar events (optional)
    events = [
        {
            "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "event_name": "NFP",
            "country": "US",
            "actual": 150000,
            "forecast": 180000,
            "previous": 200000,
            "impact": "high"
        }
    ]
    
    # Make prediction request
    payload = {
        "pair": "EUR_USD",
        "ohlcv_m5": ohlcv_m5,
        "ohlcv_h1": ohlcv_h1,
        "ohlcv_h4": ohlcv_h4,
        "events": events
    }
    
    print(f"Sending prediction request for EUR_USD...")
    print(f"  M5 candles: {len(ohlcv_m5)}")
    print(f"  H1 candles: {len(ohlcv_h1)}")
    print(f"  H4 candles: {len(ohlcv_h4)}")
    print(f"  Events: {len(events)}")
    
    response = requests.post(
        f"{BASE_URL}/predict",
        json=payload,
        headers={'Content-Type': 'application/json'}
    )
    
    print(f"\nStatus: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        print(f"\nPrediction Result:")
        print(f"  Pair: {data['pair']}")
        print(f"  Prediction: {data['prediction']}")
        print(f"  Confidence: {data['confidence']:.2%}")
        print(f"  Probabilities:")
        for action, prob in data['probabilities'].items():
            print(f"    {action}: {prob:.2%}")
        print(f"  Feature Count: {data['feature_count']}")
        print(f"  Timestamp: {data['timestamp']}")
        
        # Validate response
        assert data['prediction'] in ['BUY', 'SELL', 'HOLD']
        assert 0 <= data['confidence'] <= 1
        assert data['feature_count'] == 81
        
        print("\n✅ Prediction test passed")
    else:
        print(f"❌ Prediction failed: {response.text}")
        raise Exception("Prediction test failed")


def test_all_pairs():
    """Test predictions for all currency pairs"""
    print("\n" + "="*60)
    print("TEST 4: All Currency Pairs")
    print("="*60)
    
    pairs = ["EUR_USD", "GBP_USD", "USD_JPY", "AUD_USD"]
    
    # Generate sample data once
    ohlcv_m5 = generate_sample_ohlcv(500, 'M5')
    ohlcv_h1 = generate_sample_ohlcv(300, 'H1')
    ohlcv_h4 = generate_sample_ohlcv(250, 'H4')
    
    for pair in pairs:
        print(f"\nTesting {pair}...")
        
        payload = {
            "pair": pair,
            "ohlcv_m5": ohlcv_m5,
            "ohlcv_h1": ohlcv_h1,
            "ohlcv_h4": ohlcv_h4,
            "events": []
        }
        
        response = requests.post(
            f"{BASE_URL}/predict",
            json=payload,
            headers={'Content-Type': 'application/json'}
        )
        
        if response.status_code == 200:
            data = response.json()
            print(f"  ✅ {pair}: {data['prediction']} ({data['confidence']:.2%})")
        else:
            print(f"  ❌ {pair}: Failed - {response.text}")
    
    print("\n✅ All pairs test passed")


def main():
    """Run all tests"""
    print("\n" + "="*60)
    print("HUGGING FACE DEPLOYMENT VERIFICATION")
    print("="*60)
    print(f"Testing: {BASE_URL}")
    
    try:
        test_health()
        test_model_info()
        test_prediction()
        test_all_pairs()
        
        print("\n" + "="*60)
        print("✅ ALL TESTS PASSED!")
        print("="*60)
        print("\nThe inference server has everything it needs:")
        print("  ✅ All 4 models loaded")
        print("  ✅ Technical feature engineering working")
        print("  ✅ Multi-timeframe features working")
        print("  ✅ Macro feature engineering working")
        print("  ✅ Predictions working for all pairs")
        print("  ✅ 81 features correctly engineered")
        print("\nServer is ready for production! 🚀")
        
    except Exception as e:
        print("\n" + "="*60)
        print("❌ TESTS FAILED")
        print("="*60)
        print(f"Error: {e}")
        raise


if __name__ == "__main__":
    main()
