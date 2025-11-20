# Complete Automated Retraining Flow

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    AUTOMATED RETRAINING SYSTEM                  │
└─────────────────────────────────────────────────────────────────┘

┌──────────────┐         ┌──────────────┐         ┌──────────────┐
│   TRIGGER    │         │   TRAINING   │         │  DEPLOYMENT  │
│              │         │              │         │              │
│  • Manual    │────────▶│   Kaggle     │────────▶│   GitHub     │
│  • Scheduled │         │   Notebook   │         │   Repository │
│  • GitHub    │         │              │         │              │
│    Actions   │         │  30GB RAM    │         │  Git LFS     │
└──────────────┘         │  T4 GPU      │         └──────────────┘
                         │  12h session │                │
                         └──────────────┘                │
                                                         │
                                                         ▼
                                                ┌──────────────┐
                                                │  Hugging Face│
                                                │    Spaces    │
                                                │              │
                                                │  Auto-Deploy │
                                                │  16GB RAM    │
                                                └──────────────┘
                                                         │
                                                         ▼
                                                ┌──────────────┐
                                                │ MetaTrader 5 │
                                                │      EA      │
                                                │              │
                                                │  Live Trading│
                                                └──────────────┘
```

## Detailed Flow

### Step 1: Trigger (Weekly)

**Option A: Manual (Simplest)**
```
You → Kaggle Notebook → Click "Run All"
```

**Option B: GitHub Actions (Automated)**
```
Sunday 2 AM UTC → GitHub Actions → Triggers Kaggle API → Notebook Runs
```

**Option C: External Scheduler (Zapier/IFTTT)**
```
Zapier Schedule → Webhook → Kaggle API → Notebook Runs
```

### Step 2: Training on Kaggle (2-3 hours)

```
┌─────────────────────────────────────────────────────────────┐
│  Kaggle Notebook Execution                                  │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  1. Clone GitHub Repo                                       │
│     git clone https://github.com/Morriase/...              │
│                                                             │
│  2. Install Dependencies                                    │
│     pip install -r requirements.txt                         │
│     Install TA-Lib                                          │
│                                                             │
│  3. Download Latest Data                                    │
│     from src.data_acquisition.kaggle_loader import ...      │
│     download_all_datasets()                                 │
│                                                             │
│  4. Backup Current Models                                   │
│     cp -r models/ models_backup/                            │
│                                                             │
│  5. Train All 4 Pairs                                       │
│     python train_pipeline.py --pairs EUR_USD GBP_USD ...    │
│                                                             │
│     EUR_USD: ~30 min (or ~10 min with GPU)                 │
│     GBP_USD: ~30 min (or ~10 min with GPU)                 │
│     USD_JPY: ~30 min (or ~10 min with GPU)                 │
│     AUD_USD: ~30 min (or ~10 min with GPU)                 │
│                                                             │
│  6. Compare Models                                          │
│     python scripts/compare_models.py                        │
│                                                             │
│     Metrics:                                                │
│     - Accuracy (20% weight)                                 │
│     - F1 Score (30% weight)                                 │
│     - Sharpe Ratio (30% weight)                             │
│     - Total Return (20% weight)                             │
│                                                             │
│     Decision Logic:                                         │
│     - Require 2%+ improvement                               │
│     - Need 2/4 pairs improved minimum                       │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Step 3: Deployment Decision

```
┌─────────────────────────────────────────────────────────────┐
│  Model Comparison Results                                   │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  EUR_USD: +3.5% improvement ✅                              │
│  GBP_USD: +1.8% improvement ❌ (below 2% threshold)         │
│  USD_JPY: +4.2% improvement ✅                              │
│  AUD_USD: -0.5% improvement ❌                              │
│                                                             │
│  Result: 2/4 improved → DEPLOY ✅                           │
│                                                             │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
                    ┌───────┴───────┐
                    │               │
            Should Deploy?      Should Skip?
                    │               │
                    ▼               ▼
        ┌───────────────────┐   ┌──────────────┐
        │ Push to GitHub    │   │ Keep Current │
        │                   │   │   Models     │
        │ git add models/   │   │              │
        │ git commit -m ... │   │ Log reason   │
        │ git push origin   │   │ Exit         │
        └───────────────────┘   └──────────────┘
```

### Step 4: GitHub Update (if deploying)

```
┌─────────────────────────────────────────────────────────────┐
│  Kaggle → GitHub Push                                       │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  1. Configure Git                                           │
│     git config user.name "Kaggle Bot"                       │
│     git config user.email "kaggle@bot.com"                  │
│                                                             │
│  2. Stage New Models                                        │
│     git add models/                                         │
│                                                             │
│  3. Commit Changes                                          │
│     git commit -m "Auto-retrain: Deploy improved models"    │
│                                                             │
│  4. Push to GitHub                                          │
│     git push origin main                                    │
│                                                             │
│  Files Updated:                                             │
│  - models/EUR_USD_model_*.pkl                               │
│  - models/GBP_USD_model_*.pkl                               │
│  - models/USD_JPY_model_*.pkl                               │
│  - models/AUD_USD_model_*.pkl                               │
│  - models/results/*_wfo_summary.csv                         │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Step 5: HF Space Auto-Deploy (5-10 min)

```
┌─────────────────────────────────────────────────────────────┐
│  Hugging Face Spaces Deployment                            │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  1. Detect GitHub Push                                      │
│     HF monitors linked GitHub repo                          │
│     Detects new commit on main branch                       │
│                                                             │
│  2. Pull Latest Code                                        │
│     git pull origin main                                    │
│     git lfs pull (download model files)                     │
│                                                             │
│  3. Rebuild Docker Image                                    │
│     docker build -f Dockerfile .                            │
│     - Install TA-Lib                                        │
│     - Install Python dependencies                           │
│     - Copy new models                                       │
│                                                             │
│  4. Start New Container                                     │
│     docker run -p 7860:7860 ...                             │
│     - Load all 4 models                                     │
│     - Start Flask server                                    │
│     - Health check passes                                   │
│                                                             │
│  5. Switch Traffic                                          │
│     Old container → New container                           │
│     Zero downtime deployment                                │
│                                                             │
│  6. Server Live                                             │
│     https://morriase-forex-live-server.hf.space             │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Step 6: MetaTrader EA Uses New Models

```
┌─────────────────────────────────────────────────────────────┐
│  MetaTrader 5 EA                                            │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Every M5 bar (or X seconds):                               │
│                                                             │
│  1. Collect OHLCV Data                                      │
│     - 500 M5 candles                                        │
│     - 300 H1 candles                                        │
│     - 250 H4 candles                                        │
│                                                             │
│  2. Send to API                                             │
│     POST https://morriase-forex-live-server.hf.space/predict│
│     {                                                       │
│       "pair": "EUR_USD",                                    │
│       "ohlcv_m5": [...],                                    │
│       "ohlcv_h1": [...],                                    │
│       "ohlcv_h4": [...]                                     │
│     }                                                       │
│                                                             │
│  3. Receive Prediction (NEW MODELS!)                        │
│     {                                                       │
│       "prediction": "BUY",                                  │
│       "confidence": 0.75,                                   │
│       "quality_score": 65.0,                                │
│       "should_trade": true                                  │
│     }                                                       │
│                                                             │
│  4. Execute Trade                                           │
│     If should_trade == true:                                │
│       - Calculate position size                             │
│       - Set stop loss & take profit                         │
│       - Place order                                         │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## Timeline Example

```
Sunday 2:00 AM UTC
├─ GitHub Actions triggers
│
Sunday 2:01 AM
├─ Kaggle notebook starts
│  └─ Clone repo, install deps (15 min)
│
Sunday 2:16 AM
├─ Download latest data (15 min)
│
Sunday 2:31 AM
├─ Train EUR_USD (30 min with CPU, 10 min with GPU)
│
Sunday 3:01 AM
├─ Train GBP_USD (30 min with CPU, 10 min with GPU)
│
Sunday 3:31 AM
├─ Train USD_JPY (30 min with CPU, 10 min with GPU)
│
Sunday 4:01 AM
├─ Train AUD_USD (30 min with CPU, 10 min with GPU)
│
Sunday 4:31 AM
├─ Compare models (5 min)
│  └─ Decision: DEPLOY (2/4 improved)
│
Sunday 4:36 AM
├─ Push to GitHub (2 min)
│
Sunday 4:38 AM
├─ HF detects push
│  └─ Start rebuild
│
Sunday 4:48 AM
├─ HF deployment complete
│  └─ New models live!
│
Sunday 4:48 AM onwards
└─ MetaTrader EA uses improved models ✅
```

**Total Time:** ~2.5-3 hours (CPU) or ~1-1.5 hours (GPU)

## Resource Usage

### Kaggle
- **Session:** 1 per week
- **Duration:** 2-3 hours
- **GPU Quota:** 2-3 hours (out of 30h/week)
- **Cost:** FREE

### GitHub
- **Storage:** ~500MB (models with LFS)
- **Actions:** 0 minutes (just triggers Kaggle)
- **Cost:** FREE

### Hugging Face
- **Compute:** 16GB RAM, 2 CPU cores
- **Storage:** ~500MB
- **Bandwidth:** Unlimited
- **Cost:** FREE

**Total Cost:** $0/month 🎉

## Monitoring & Alerts

### Check Training Progress
```
Kaggle Notebook → Output cells → Real-time logs
```

### Check Deployment Status
```
GitHub → Commits → Look for "Kaggle Bot" commits
HF Space → Logs → Watch rebuild
```

### Set Up Notifications
```
Kaggle → Notebook Settings → Email on completion
GitHub → Watch repository → Get push notifications
HF → Space Settings → Email on deployment
```

## Summary

✅ **Kaggle trains** (powerful, fast, free)  
✅ **GitHub stores** (version control, LFS)  
✅ **HF serves** (production API, auto-deploy)  
✅ **EA trades** (live, improved models)  

**Fully automated, zero cost, maximum performance!** 🚀

---

**Next Steps:**
1. Read `KAGGLE_AUTOMATED_RETRAINING.md` for setup
2. Create Kaggle notebook
3. Add secrets (GitHub token, HF token)
4. Run first training manually
5. Set up weekly schedule (optional)
