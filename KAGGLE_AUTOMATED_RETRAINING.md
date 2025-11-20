# Automated Retraining with Kaggle (Recommended)

## Why Kaggle?

| Feature | Kaggle | GitHub Actions |
|---------|--------|----------------|
| **CPU** | 2 cores | 2 cores |
| **RAM** | 30GB | 7GB |
| **GPU** | T4 GPU (optional) | None |
| **Session Time** | 12 hours | 6 hours |
| **Weekly Quota** | 30h GPU + unlimited CPU | 2,000 minutes |
| **Cost** | FREE | FREE |
| **Training Speed** | ⚡ Fast | 🐌 Slow |

**Verdict:** Kaggle is WAY better for training! 🚀

## How It Works

```
┌─────────────────────────────────────────────────────────────┐
│  1. Trigger: Manual or External Scheduler                   │
│     (Zapier, IFTTT, or manual weekly run)                   │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  2. Kaggle Notebook Runs                                    │
│     - Clones your GitHub repo                               │
│     - Downloads latest forex data                           │
│     - Trains all 4 models (EUR, GBP, USD, AUD)             │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  3. Compare Models                                          │
│     - Old vs New performance                                │
│     - Accuracy, F1, Sharpe, Returns                         │
│     - Require 2%+ improvement                               │
└─────────────────────────────────────────────────────────────┘
                            ↓
                    ┌───────┴───────┐
                    │               │
            Improved?               Not Improved?
                    │               │
                    ↓               ↓
        ┌───────────────────┐   ┌──────────────┐
        │ 4. Push to GitHub │   │ 4. Skip      │
        │    - Commit models│   │    - Keep old│
        │    - Push to repo │   │              │
        └───────────────────┘   └──────────────┘
                    ↓
        ┌───────────────────┐
        │ 5. HF Auto-Deploy │
        │    - Detects push │
        │    - Rebuilds     │
        │    - Goes live    │
        └───────────────────┘
```

## Setup (One-Time)

### 1. Create Kaggle Notebook

1. Go to: https://www.kaggle.com/code
2. Click **"New Notebook"**
3. Name it: `forex-model-retraining`
4. Settings:
   - **Accelerator:** GPU T4 x2 (optional, for faster training)
   - **Language:** Python
   - **Environment:** Latest

### 2. Upload Notebook

Upload `kaggle_notebooks/automated_retraining.ipynb` to your Kaggle notebook

Or copy-paste the cells manually.

### 3. Add Kaggle Secrets

Go to: **Kaggle Account → Settings → Add-ons → Secrets**

Add these secrets:

#### GitHub Token
```
Label: GITHUB_TOKEN
Value: ghp_your_github_personal_access_token
```

**Get GitHub token:**
1. Go to: https://github.com/settings/tokens
2. Click "Generate new token (classic)"
3. Scopes: Select `repo` (full control)
4. Generate and copy token

#### Hugging Face Token (Optional)
```
Label: HF_TOKEN
Value: your_hf_token_here
```

Only needed if you want to manually trigger HF rebuild (not required - HF auto-deploys from GitHub).

### 4. Link HF Space to GitHub

1. Go to: https://huggingface.co/spaces/morriase/forex-live_server/settings
2. Scroll to **"Repository"** section
3. Click **"Link to GitHub"**
4. Select: `Morriase/Macro-Technical-Sentiment-Classifier`
5. Branch: `main`
6. Click **"Link repository"**

Now HF will auto-deploy whenever you push to GitHub!

## Running the Pipeline

### Option 1: Manual (Recommended to Start)

1. Open your Kaggle notebook
2. Click **"Run All"**
3. Wait ~2-3 hours for training
4. Check output for deployment decision
5. If deployed, HF Space rebuilds automatically

### Option 2: Scheduled (External Trigger)

Kaggle doesn't have native scheduling, but you can use:

#### A. Zapier (Free tier: 100 tasks/month)

1. Create Zapier account: https://zapier.com
2. Create Zap:
   - **Trigger:** Schedule (Weekly, Sunday 2 AM)
   - **Action:** Webhooks → POST request
   - **URL:** `https://www.kaggle.com/api/v1/kernels/push`
   - **Headers:** `Authorization: Bearer YOUR_KAGGLE_API_KEY`
   - **Body:** 
   ```json
   {
     "id": "your-notebook-id",
     "action": "run"
   }
   ```

#### B. GitHub Actions (Trigger Kaggle)

Keep the GitHub Actions workflow but make it trigger Kaggle instead:

```yaml
name: Trigger Kaggle Retraining

on:
  schedule:
    - cron: '0 2 * * 0'  # Sunday 2 AM
  workflow_dispatch:

jobs:
  trigger:
    runs-on: ubuntu-latest
    steps:
    - name: Trigger Kaggle Notebook
      run: |
        curl -X POST \
          -H "Authorization: Bearer ${{ secrets.KAGGLE_API_KEY }}" \
          https://www.kaggle.com/api/v1/kernels/push \
          -d '{"id": "your-notebook-id", "action": "run"}'
```

#### C. IFTTT (Free)

1. Create IFTTT account: https://ifttt.com
2. Create Applet:
   - **If:** Date & Time → Every week
   - **Then:** Webhooks → Make a web request
   - **URL:** Kaggle API endpoint
   - **Method:** POST

#### D. Manual Weekly (Simplest)

Just run the notebook manually every Sunday. Takes 2 clicks!

## Monitoring

### View Notebook Output

1. Go to your Kaggle notebook
2. Check the output cells
3. Look for:
   - Training progress
   - Model comparison results
   - Deployment decision

### Check GitHub

1. Go to: https://github.com/Morriase/Macro-Technical-Sentiment-Classifier/commits/main
2. Look for commits from "Kaggle Bot"
3. Verify models were updated

### Check HF Space

1. Go to: https://huggingface.co/spaces/morriase/forex-live_server/logs
2. Watch for rebuild after GitHub push
3. Verify new models load correctly

## Training Time Estimates

| Pair | Training Time | With GPU |
|------|---------------|----------|
| EUR_USD | ~30 min | ~10 min |
| GBP_USD | ~30 min | ~10 min |
| USD_JPY | ~30 min | ~10 min |
| AUD_USD | ~30 min | ~10 min |
| **Total** | **~2 hours** | **~40 min** |

Add ~30 min for data download and setup.

**Total pipeline time:** 2.5-3 hours (CPU) or 1-1.5 hours (GPU)

## Cost Analysis

### Kaggle (Recommended)
- **Training:** FREE (30h GPU/week)
- **Scheduling:** FREE (manual) or $0-20/month (Zapier)
- **Total:** **$0-20/month**

### GitHub Actions
- **Training:** FREE (2,000 min/month)
- **Scheduling:** FREE (built-in)
- **Total:** **$0/month**
- **Problem:** Slow, limited resources

### HF Persistent Compute
- **Training:** $0.60/hour × 24/7 = $432/month
- **Total:** **$432/month**
- **Problem:** Expensive!

**Winner:** Kaggle! 🏆

## Deployment Criteria

Models deploy ONLY if:

1. **At least 2 out of 4 pairs improved**
2. **Improvement score > 2%**

### Metrics Weighted:
- Accuracy: 20%
- F1 Score: 30%
- Sharpe Ratio: 30%
- Total Return: 20%

## Troubleshooting

### Notebook Times Out

**Solution:** Enable GPU accelerator
- Notebook Settings → Accelerator → GPU T4 x2

### Git Push Fails

**Check:**
1. GitHub token is valid
2. Token has `repo` scope
3. Token is added to Kaggle secrets

### Models Not Improving

**This is normal!** The system will skip deployment and keep current models.

### HF Space Not Updating

**Verify:**
1. GitHub link is active (HF Space Settings)
2. Auto-deploy is enabled
3. Check HF Space logs for errors

## Best Practices

1. **Start with manual runs** - Verify everything works
2. **Monitor first few runs** - Check for errors
3. **Review comparisons** - Understand why models deploy/skip
4. **Keep GPU quota** - Don't waste on unnecessary runs
5. **Set up notifications** - Get alerted on deployments

## Comparison: Kaggle vs GitHub Actions

### Kaggle Approach ✅
**Pros:**
- Much faster training (GPU available)
- More RAM (30GB vs 7GB)
- Longer sessions (12h vs 6h)
- Better for ML workloads

**Cons:**
- No native scheduling (need external trigger)
- Manual or paid scheduling

### GitHub Actions Approach
**Pros:**
- Native scheduling (free)
- Fully automated
- No external dependencies

**Cons:**
- Slow training (CPU only)
- Limited RAM (7GB)
- Shorter timeout (6h)

## Recommended Setup

**For most users:**
1. Use **Kaggle for training** (this guide)
2. Run **manually every Sunday** (2 clicks)
3. Let **HF auto-deploy** from GitHub

**For full automation:**
1. Use **Kaggle for training**
2. Use **GitHub Actions to trigger Kaggle** (free)
3. Let **HF auto-deploy** from GitHub

**Best of both worlds!** 🎉

## Summary

✅ **Kaggle trains models** (fast, powerful, free)  
✅ **Smart deployment** (only if improved)  
✅ **GitHub stores models** (version control)  
✅ **HF serves models** (production API)  
✅ **Fully automated** (with external scheduler)  

Your models will stay fresh and improve over time! 🚀

---

**Setup Time:** ~30 minutes  
**Training Time:** 1-3 hours/week  
**Cost:** Free (or $0-20/month for scheduling)  
**Maintenance:** Minimal  
**Value:** Priceless 💎
