# Render Deployment Guide

## Prerequisites
- GitHub repository: https://github.com/Morriase/Macro-Technical-Sentiment-Classifier
- Render account: https://render.com
- All trained models committed to repository in `models/` directory

## Deployment Steps

### 1. Commit and Push Changes

```bash
git add .
git commit -m "Add Render deployment configuration"
git push origin main
```

### 2. Create Render Service

#### Option A: Using Blueprint (Recommended)
1. Go to https://dashboard.render.com
2. Click "New" → "Blueprint"
3. Connect your GitHub repository
4. Select `Macro-Technical-Sentiment-Classifier`
5. Render will automatically detect `render.yaml`
6. Click "Apply"

#### Option B: Manual Setup
1. Go to https://dashboard.render.com
2. Click "New" → "Web Service"
3. Connect your GitHub repository
4. Configure:
   - **Name**: `forex-inference-server`
   - **Region**: Choose closest to your MT5 broker
   - **Branch**: `main`
   - **Runtime**: `Docker`
   - **Dockerfile Path**: `./Dockerfile`
   - **Plan**: Free (can upgrade later)
5. Add Environment Variables:
   ```
   PORT=5000
   FLASK_ENV=production
   PYTHONUNBUFFERED=1
   ```
6. Click "Create Web Service"

### 3. Monitor Deployment

The deployment will take 10-15 minutes:
- Docker image build: ~8 minutes (TA-Lib compilation)
- Model loading: ~2 minutes
- Server startup: ~1 minute

Watch the logs in Render dashboard for:
```
[INFO] Loading model for EUR_USD...
[SUCCESS] Loaded EUR_USD model (58 features)
[INFO] * Running on all addresses (0.0.0.0)
[INFO] * Running on http://0.0.0.0:5000
```

### 4. Test Your Deployed Server

Once deployed, you'll get a URL like:
```
https://forex-inference-server-xxxx.onrender.com
```

Test the health endpoint:
```bash
curl https://forex-inference-server-xxxx.onrender.com/health
```

Expected response:
```json
{
  "status": "healthy",
  "timestamp": "2025-11-09T14:40:54.332Z",
  "models_loaded": [],
  "supported_pairs": ["EUR_USD", "GBP_USD", "USD_JPY", "AUD_USD"]
}
```

### 5. Update MT5 EA Configuration

Update your EA's server URL:

```cpp
input string RestServerURL = "https://forex-inference-server-xxxx.onrender.com/predict";
```

Recompile and run the EA.

## Important Notes

### Free Tier Limitations
- **Sleep after 15 min inactivity**: Server will spin down
- **First request after sleep**: Takes 30-60 seconds to wake up
- **Cold start**: LSTM model loading takes ~2 minutes
- **Solution**: Upgrade to Starter plan ($7/month) for always-on

### Model Size Warning
Your `models/` directory contains ~500MB of trained models:
- EUR_USD_model.pth_*
- GBP_USD_model.pth_*
- USD_JPY_model.pth_*
- AUD_USD_model.pth_*

This may slow down:
- Git push/pull operations
- Docker image builds on Render

**Recommendation**: Use Git LFS for model files (optional):
```bash
git lfs install
git lfs track "models/*.pkl"
git lfs track "models/*.pth"
git add .gitattributes
git commit -m "Track model files with Git LFS"
```

### Auto-Deploy
The `render.yaml` has `autoDeploy: true`, meaning:
- Every push to `main` branch triggers a new deployment
- Takes 10-15 minutes per deployment
- Useful for updates, but be careful with frequent pushes

### Performance Optimization
1. **Workers**: Currently 2 workers, 4 threads
   - Handles ~8 concurrent requests
   - Increase if you have multiple EAs

2. **Timeout**: 120 seconds
   - Allows for model loading on first request
   - Calendar event fetching can be slow

3. **Caching**: Models are cached in memory after first load
   - Subsequent requests are fast (~100-200ms)

## Monitoring

### Health Check
Render automatically checks `/health` every 30 seconds:
- If unhealthy, Render restarts the container
- If restart fails 3 times, marks service as failed

### Logs
View logs in Render dashboard:
- Click on your service
- Go to "Logs" tab
- Real-time stream of server activity

### Metrics
Render provides:
- CPU usage
- Memory usage
- Request rate
- Response times

## Troubleshooting

### Build Fails
**Error**: "TA-Lib compilation failed"
- Solution: Multi-stage build should handle this
- Check Dockerfile logs in Render

**Error**: "Out of memory during build"
- Solution: Upgrade to Starter plan (2GB RAM)

### Server Crashes
**Error**: "Model not found for EUR_USD"
- Solution: Ensure all model files are committed to Git
- Check `models/` directory structure

**Error**: "Memory limit exceeded"
- Solution: Reduce workers or upgrade plan
- Current: 2 workers × 4 threads = ~1.5GB RAM

### Slow Responses
**Symptom**: First request takes 30+ seconds
- Cause: Cold start (model loading)
- Solution: Upgrade to Starter plan (no sleep)

**Symptom**: All requests slow
- Cause: Too many concurrent requests
- Solution: Increase workers or scale horizontally

## Cost Estimate

| Plan | Cost | RAM | Sleep | Build Time |
|------|------|-----|-------|------------|
| Free | $0 | 512MB | Yes (15min) | Slow |
| Starter | $7/mo | 512MB | No | Fast |
| Standard | $25/mo | 2GB | No | Fast |
| Pro | $85/mo | 4GB | No | Very Fast |

**Recommendation**: Start with Free for testing, upgrade to Starter ($7/mo) for live trading.

## Next Steps

1. ✅ Commit and push to GitHub
2. ✅ Create Render service
3. ✅ Wait for deployment (~15 min)
4. ✅ Test health endpoint
5. ✅ Update EA with new URL
6. ✅ Test live predictions
7. 📊 Monitor performance for 24 hours
8. 💰 Consider upgrading if cold starts are an issue

## Support

- Render Docs: https://render.com/docs
- Flask + Gunicorn: https://docs.gunicorn.org/
- GitHub Issues: https://github.com/Morriase/Macro-Technical-Sentiment-Classifier/issues
