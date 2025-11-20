# Local Server Testing Guide

## Quick Start

### 1. Start the Server

```bash
python run_local_server.py
```

You should see:
```
================================================================================
STARTING LOCAL FOREX INFERENCE SERVER
================================================================================
URL: http://localhost:5000
Models directory: D:\Macro-Technical Sentiment Classifier\models
Supported pairs: EUR_USD, GBP_USD, USD_JPY, AUD_USD
================================================================================

Endpoints:
  GET  /health          - Health check
  POST /predict         - Make prediction
  GET  /model_info/<pair> - Get model info

Press Ctrl+C to stop the server
================================================================================
```

### 2. Test the Server (in another terminal)

```bash
python test_local_server.py
```

This will run 3 tests:
1. Health check
2. Prediction with sample data
3. Model info retrieval

## Manual Testing

### Test Health Endpoint

```bash
curl http://localhost:5000/health
```

Expected response:
```json
{
  "status": "healthy",
  "timestamp": "2025-11-20T15:00:00",
  "models_loaded": [],
  "supported_pairs": ["EUR_USD", "GBP_USD", "USD_JPY", "AUD_USD"]
}
```

### Test Prediction Endpoint

Create a file `test_request.json`:
```json
{
  "pair": "EUR_USD",
  "ohlcv_m5": [
    {"timestamp": "2025-11-20 10:00:00", "open": 1.1000, "high": 1.1005, "low": 1.0995, "close": 1.1002, "volume": 1000}
    // ... add 499 more bars
  ],
  "ohlcv_h1": [
    // ... 300 bars
  ],
  "ohlcv_h4": [
    // ... 250 bars
  ],
  "events": []
}
```

Send request:
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d @test_request.json
```

## Test with Your EA

### Update EA to use local server:

In `Auron AI.mq5`, change:
```mql5
input string RestServerURL = "http://localhost:5000/predict";  // Local server
```

Then:
1. Recompile EA
2. Attach to chart
3. Watch the server logs in real-time

## What to Look For

### ✅ Good Signs:
```
2025-11-20 15:00:00.123 | INFO | Prediction request for EUR_USD with M5=500, H1=300, H4=250 candles
2025-11-20 15:00:00.456 | SUCCESS | Loaded EUR_USD model (81 features)
2025-11-20 15:00:01.789 | SUCCESS | EUR_USD: Feature validation passed ✓
2025-11-20 15:00:05.123 | SUCCESS | EUR_USD (EUR_USD model): BUY (confidence: 75.00%, quality: 65.0/100)
```

### ❌ Bad Signs:
```
ERROR: Insufficient M5 bars
ERROR: Feature count mismatch
ERROR: Validation error
```

## Performance Monitoring

Watch for:
- **Response time**: Should be < 5 seconds
- **Memory usage**: Should stay stable (not growing)
- **Feature count**: Always 81 features
- **Bar counts**: M5=500, H1=300, H4=250

## Debugging

### Enable verbose logging:

In `inference_server.py`, change:
```python
logger.add("logs/inference_server.log", rotation="1 day",
           retention="7 days", level="DEBUG")  # Change to DEBUG
```

### Check logs:
```bash
tail -f logs/inference_server.log
```

## Common Issues

### Issue: "Model not found"
**Solution:** Make sure you're in the project directory with the `models/` folder

### Issue: "Port already in use"
**Solution:** 
```bash
# Windows
netstat -ano | findstr :5000
taskkill /PID <PID> /F

# Or change port in run_local_server.py
os.environ['PORT'] = '5001'
```

### Issue: "Import errors"
**Solution:** Install dependencies:
```bash
pip install -r requirements.txt
```

## Stopping the Server

Press `Ctrl+C` in the terminal running the server.

## Next Steps

Once local testing passes:
1. Update EA back to Render URL
2. Deploy to Render
3. Monitor production logs

---

**Local URL:** http://localhost:5000  
**Production URL:** https://forex-inference-server.onrender.com
