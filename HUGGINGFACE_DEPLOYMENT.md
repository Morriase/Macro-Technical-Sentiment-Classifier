# Deploy to Hugging Face Spaces via GitHub

## Why Hugging Face Spaces?

✅ **Better Free Tier**: 16GB RAM (vs Render's 512MB)  
✅ **No Crashes**: Enough memory for all 4 models  
✅ **GitHub Integration**: Auto-deploy on push  
✅ **Public URL**: Free HTTPS endpoint  
✅ **Docker Support**: Use existing Dockerfile  

## Step-by-Step Deployment

### 1. Create Hugging Face Account

1. Go to https://huggingface.co/join
2. Sign up (free account)
3. Verify your email

### 2. Create a New Space

1. Go to https://huggingface.co/spaces
2. Click **"Create new Space"**
3. Fill in details:
   - **Space name**: `forex-inference-server` (or your choice)
   - **License**: MIT
   - **SDK**: Select **Docker**
   - **Space hardware**: CPU basic (free)
   - **Visibility**: Public or Private

4. Click **"Create Space"**

### 3. Connect GitHub Repository

#### Option A: Direct GitHub Sync (Recommended)

1. In your new Space, click **"Settings"** tab
2. Scroll to **"Repository"** section
3. Click **"Link to GitHub"**
4. Authorize Hugging Face to access your GitHub
5. Select your repository: `Morriase/Macro-Technical-Sentiment-Classifier`
6. Choose branch: `main`
7. Click **"Link repository"**

Now every push to GitHub will auto-deploy to Hugging Face!

#### Option B: Manual Git Push

If you prefer manual control:

```bash
# Add Hugging Face as a remote
git remote add hf https://huggingface.co/spaces/YOUR_USERNAME/forex-inference-server

# Push to Hugging Face
git push hf main
```

### 4. Prepare Repository Files

Make sure these files exist in your repo:

```
✅ Dockerfile              # Already exists
✅ requirements_render.txt # Already exists (will be used)
✅ inference_server.py     # Already exists
✅ start.sh               # Already exists
✅ app.py                 # Just created
✅ README_HF.md           # Just created (rename to README.md)
✅ src/                   # Already exists
✅ models/                # Already exists
```

### 5. Update README for Hugging Face

Rename the Hugging Face README:

```bash
# Backup current README
mv README.md README_PROJECT.md

# Use Hugging Face README
mv README_HF.md README.md
```

The README.md with YAML frontmatter is required for Hugging Face Spaces.

### 6. Commit and Push

```bash
# Add new files
git add app.py README.md HUGGINGFACE_DEPLOYMENT.md

# Commit
git commit -m "Add Hugging Face Spaces deployment files"

# Push to GitHub (will auto-deploy to HF if linked)
git push origin main
```

### 7. Monitor Deployment

1. Go to your Space: `https://huggingface.co/spaces/YOUR_USERNAME/forex-inference-server`
2. Click **"Logs"** tab
3. Watch the build process:
   - Building Docker image
   - Installing dependencies
   - Starting server
   - Should see: "SERVER INITIALIZATION COMPLETE"

Build takes ~5-10 minutes first time.

### 8. Test Your Deployment

Once deployed, your server will be at:
```
https://YOUR_USERNAME-forex-inference-server.hf.space
```

Test endpoints:
```bash
# Health check
curl https://YOUR_USERNAME-forex-inference-server.hf.space/health

# Model info
curl https://YOUR_USERNAME-forex-inference-server.hf.space/model_info/EUR_USD
```

### 9. Update EA with New URL

In `Auron AI.mq5`:
```mql5
input string RestServerURL = "https://YOUR_USERNAME-forex-inference-server.hf.space/predict";
```

Recompile and test!

## Troubleshooting

### Build Fails

**Check logs** in the Logs tab. Common issues:

1. **Missing files**: Make sure all files are committed
2. **TA-Lib build error**: Dockerfile handles this, should work
3. **Memory during build**: Upgrade to paid tier if needed

### Server Starts but Crashes

Check logs for:
- Model loading errors
- Missing model files
- Import errors

### Can't Access URL

- Wait 1-2 minutes after "Running" status
- Check if Space is set to "Running" (not "Sleeping")
- Try incognito/private browsing

## Hugging Face vs Render

| Feature | Hugging Face | Render |
|---------|-------------|--------|
| **RAM** | 16GB | 512MB |
| **Build Time** | 5-10 min | 2-5 min |
| **Auto-deploy** | ✅ GitHub sync | ✅ GitHub sync |
| **Custom Domain** | ❌ | ✅ |
| **Sleep Policy** | After 48h inactive | After 15min inactive |
| **Pricing** | Free forever | Free tier limited |

## Advanced: Keep Space Awake

Hugging Face Spaces sleep after 48 hours of inactivity. To keep it awake:

### Option 1: Upgrade to Paid Tier
- $0.60/hour for persistent CPU
- No sleep, always available

### Option 2: Ping Service
Use a service like UptimeRobot to ping your health endpoint every 30 minutes.

## Files Checklist

Before deploying, verify:

- [ ] `app.py` exists (entry point for HF)
- [ ] `README.md` has YAML frontmatter
- [ ] `Dockerfile` exists
- [ ] `requirements_render.txt` exists
- [ ] `start.sh` is executable
- [ ] `models/` folder with all 4 models
- [ ] `src/` folder with all code
- [ ] All files committed to Git

## Next Steps

1. ✅ Create Hugging Face account
2. ✅ Create new Space
3. ✅ Link GitHub repository
4. ✅ Push code
5. ✅ Monitor build
6. ✅ Test endpoints
7. ✅ Update EA URL
8. ✅ Trade!

---

**Your Space URL**: `https://YOUR_USERNAME-forex-inference-server.hf.space`  
**Hugging Face Docs**: https://huggingface.co/docs/hub/spaces-overview  
**Support**: https://discuss.huggingface.co/
