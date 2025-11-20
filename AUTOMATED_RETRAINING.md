# Automated Model Retraining & Deployment

## Overview

This system automatically retrains your forex models weekly and deploys improved versions to Hugging Face Spaces.

## How It Works

```
┌─────────────────────────────────────────────────────────────┐
│  1. GitHub Actions triggers weekly (Sunday 2 AM UTC)        │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  2. Download latest data from Kaggle                        │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  3. Train new models for all 4 pairs                        │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  4. Compare new vs old models                               │
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
        │ 5. Deploy to HF   │   │ 5. Skip      │
        │    - Commit models│   │    - Keep old│
        │    - Push to Git  │   │    - Notify  │
        │    - HF rebuilds  │   │              │
        └───────────────────┘   └──────────────┘
```

## Setup Instructions

### 1. Add GitHub Secrets

Go to: **Settings → Secrets and variables → Actions → New repository secret**

Add these secrets:

#### Kaggle API Credentials
```
Name: KAGGLE_USERNAME
Value: your_kaggle_username

Name: KAGGLE_KEY
Value: your_kaggle_api_key
```

Get your Kaggle API key:
1. Go to https://www.kaggle.com/settings
2. Scroll to "API" section
3. Click "Create New Token"
4. Copy username and key from downloaded `kaggle.json`

#### Hugging Face Token
```
Name: HF_TOKEN
Value: your_huggingface_token_here
```

Get HF token:
1. Go to https://huggingface.co/settings/tokens
2. Create new token with "write" access
3. Copy the token

### 2. Link HF Space to GitHub (Auto-Deploy)

1. Go to: https://huggingface.co/spaces/morriase/forex-live_server/settings
2. Scroll to "Repository" section
3. Click "Link to GitHub"
4. Select: `Morriase/Macro-Technical-Sentiment-Classifier`
5. Branch: `main`
6. Click "Link repository"

Now every push to GitHub will auto-deploy to HF!

### 3. Enable GitHub Actions

The workflow is already in `.github/workflows/retrain_models.yml`

To enable:
1. Go to: https://github.com/Morriase/Macro-Technical-Sentiment-Classifier/actions
2. Click "I understand my workflows, go ahead and enable them"

### 4. Test Manual Trigger

1. Go to: https://github.com/Morriase/Macro-Technical-Sentiment-Classifier/actions
2. Click "Retrain and Deploy Models"
3. Click "Run workflow"
4. Select branch: `main`
5. Click "Run workflow"

## Deployment Criteria

New models are deployed ONLY if:

1. **At least 2 out of 4 pairs improved**
2. **Improvement score > 2%** (weighted average of metrics)

### Metrics Weights:
- **Accuracy:** 20%
- **F1 Score:** 30%
- **Sharpe Ratio:** 30%
- **Total Return:** 20%

### Example Decision:

```
EUR_USD: +3.5% improvement ✅
GBP_USD: +1.8% improvement ❌ (below 2% threshold)
USD_JPY: +4.2% improvement ✅
AUD_USD: -0.5% improvement ❌

Result: 2/4 improved → DEPLOY ✅
```

## Schedule

**Default:** Every Sunday at 2:00 AM UTC

To change schedule, edit `.github/workflows/retrain_models.yml`:

```yaml
on:
  schedule:
    # Daily at midnight
    - cron: '0 0 * * *'
    
    # Every Monday and Thursday at 3 AM
    - cron: '0 3 * * 1,4'
    
    # First day of every month
    - cron: '0 0 1 * *'
```

## Monitoring

### View Workflow Runs
https://github.com/Morriase/Macro-Technical-Sentiment-Classifier/actions

### Check Logs
- Click on any workflow run
- View step-by-step logs
- See model comparison results

### HF Space Logs
https://huggingface.co/spaces/morriase/forex-live_server/logs

## Manual Retraining

### Option 1: GitHub Actions UI
1. Go to Actions tab
2. Select "Retrain and Deploy Models"
3. Click "Run workflow"

### Option 2: Local Training + Push
```bash
# Train locally
python train_pipeline.py --pairs EUR_USD GBP_USD USD_JPY AUD_USD

# Compare models
python scripts/compare_models.py --old models_backup --new models

# If improved, commit and push
git add models/
git commit -m "Manual retrain: Improved models"
git push origin main

# HF Space will auto-deploy
```

## Alternative: Scheduled Retraining on HF Spaces

If you want retraining to happen ON the HF Space itself:

### Pros:
- No GitHub Actions needed
- Runs on HF infrastructure

### Cons:
- Requires persistent compute (paid: ~$0.60/hour)
- More complex setup
- Less control over deployment

### Setup:
1. Upgrade Space to persistent compute
2. Add cron job in Dockerfile:
```dockerfile
RUN apt-get install -y cron
COPY retrain_cron /etc/cron.d/retrain_cron
RUN chmod 0644 /etc/cron.d/retrain_cron
RUN crontab /etc/cron.d/retrain_cron
```

3. Create `retrain_cron`:
```
0 2 * * 0 cd /app && python train_pipeline.py >> /app/logs/retrain.log 2>&1
```

**Not recommended** - GitHub Actions is free and more flexible.

## Cost Analysis

### GitHub Actions (Recommended)
- **Cost:** FREE (2,000 minutes/month)
- **Training time:** ~2-3 hours/week
- **Annual cost:** $0

### HF Persistent Compute
- **Cost:** $0.60/hour
- **Monthly:** ~$432 (24/7)
- **Annual cost:** ~$5,184

**Verdict:** Use GitHub Actions! 🎉

## Notifications

### Add Slack/Discord Notifications

Edit `.github/workflows/retrain_models.yml`:

```yaml
- name: Send Slack notification
  if: steps.compare.outputs.should_deploy == 'true'
  uses: slackapi/slack-github-action@v1
  with:
    webhook-url: ${{ secrets.SLACK_WEBHOOK }}
    payload: |
      {
        "text": "✅ New forex models deployed!",
        "blocks": [
          {
            "type": "section",
            "text": {
              "type": "mrkdwn",
              "text": "*Improved models deployed*\n${{ steps.compare.outputs.summary }}"
            }
          }
        ]
      }
```

## Troubleshooting

### Workflow Fails

**Check logs:**
1. Go to Actions tab
2. Click failed run
3. Expand failed step

**Common issues:**
- Kaggle credentials invalid
- Insufficient data
- Model training timeout (increase `timeout-minutes`)

### Models Not Deploying

**Check comparison results:**
```bash
# View comparison.json in workflow artifacts
# Or run locally:
python scripts/compare_models.py --old models_backup --new models
```

### HF Space Not Updating

**Verify GitHub link:**
1. HF Space Settings → Repository
2. Ensure linked to correct repo
3. Check "Auto-deploy" is enabled

## Best Practices

1. **Monitor first few runs** - Ensure training completes successfully
2. **Review model comparisons** - Verify improvement logic works
3. **Set up notifications** - Get alerted on deployments
4. **Keep backup models** - Workflow automatically backs up
5. **Test manually first** - Run workflow manually before relying on schedule

## Summary

✅ **Automated weekly retraining**  
✅ **Smart deployment** (only if improved)  
✅ **Zero cost** (GitHub Actions free tier)  
✅ **Auto-deploy to HF** (via GitHub link)  
✅ **Full control** (manual trigger available)  

Your models will stay fresh and improve over time automatically! 🚀

---

**Setup Time:** ~15 minutes  
**Maintenance:** Zero  
**Cost:** Free  
**Value:** Priceless 💎
