#!/bin/bash
# Deploy Forex Inference Server to Hugging Face Spaces
# Following official HF Docker Space deployment guide

set -e

echo "=========================================="
echo "Hugging Face Spaces Deployment"
echo "=========================================="
echo ""

# Step 1: Install HF CLI if needed
if ! command -v hf &> /dev/null; then
    echo "📦 Installing Hugging Face CLI..."
    curl -LsSf https://hf.co/cli/install.sh | bash
    echo "✅ HF CLI installed"
    echo ""
    echo "⚠️  Please restart your terminal and run this script again"
    exit 0
fi

echo "✅ HF CLI is installed"
echo ""

# Step 2: Check if we're in the HF Space repo
if [ -d ".git" ] && git remote get-url hf &> /dev/null; then
    echo "✅ Already in HF Space repository"
    SPACE_DIR="."
else
    echo "📥 Cloning HF Space repository..."
    echo ""
    echo "You'll be prompted for credentials:"
    echo "  Username: Your HF username"
    echo "  Password: Use an access token from https://huggingface.co/settings/tokens"
    echo ""
    
    # Clone to a temporary directory
    SPACE_DIR="../forex-live_server-hf"
    
    if [ -d "$SPACE_DIR" ]; then
        echo "⚠️  Directory $SPACE_DIR already exists"
        read -p "Remove and re-clone? (y/n): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            rm -rf "$SPACE_DIR"
        else
            echo "Using existing directory"
        fi
    fi
    
    if [ ! -d "$SPACE_DIR" ]; then
        git clone https://huggingface.co/spaces/morriase/forex-live_server "$SPACE_DIR"
    fi
    
    cd "$SPACE_DIR"
fi

echo ""
echo "=========================================="
echo "📋 Preparing Files for Deployment"
echo "=========================================="

# Step 3: Copy/verify required files
SOURCE_DIR="/media/morris/LINUX MINT/Macro-Technical Sentiment Classifier"

echo "✓ Checking Dockerfile..."
if [ ! -f "Dockerfile" ]; then
    if [ -f "$SOURCE_DIR/Dockerfile" ]; then
        cp "$SOURCE_DIR/Dockerfile" .
        echo "  Copied from source"
    else
        echo "❌ Dockerfile not found!"
        exit 1
    fi
fi

echo "✓ Checking app.py..."
if [ ! -f "app.py" ]; then
    if [ -f "$SOURCE_DIR/app.py" ]; then
        cp "$SOURCE_DIR/app.py" .
        echo "  Copied from source"
    else
        echo "❌ app.py not found!"
        exit 1
    fi
fi

echo "✓ Checking requirements..."
if [ ! -f "requirements.txt" ] && [ ! -f "requirements_render.txt" ]; then
    if [ -f "$SOURCE_DIR/requirements_render.txt" ]; then
        cp "$SOURCE_DIR/requirements_render.txt" requirements.txt
        echo "  Copied from source"
    else
        echo "❌ requirements file not found!"
        exit 1
    fi
elif [ -f "requirements_render.txt" ] && [ ! -f "requirements.txt" ]; then
    cp requirements_render.txt requirements.txt
fi

echo "✓ Checking inference_server.py..."
if [ ! -f "inference_server.py" ]; then
    if [ -f "$SOURCE_DIR/inference_server.py" ]; then
        cp "$SOURCE_DIR/inference_server.py" .
        echo "  Copied from source"
    fi
fi

echo "✓ Checking start.sh..."
if [ ! -f "start.sh" ]; then
    if [ -f "$SOURCE_DIR/start.sh" ]; then
        cp "$SOURCE_DIR/start.sh" .
        chmod +x start.sh
        echo "  Copied from source"
    fi
fi

echo "✓ Checking README.md..."
if [ ! -f "README.md" ]; then
    if [ -f "$SOURCE_DIR/README_HF.md" ]; then
        cp "$SOURCE_DIR/README_HF.md" README.md
        echo "  Copied from source (README_HF.md)"
    else
        echo "⚠️  Creating basic README.md"
        cat > README.md << 'EOF'
---
title: Forex Live Server
emoji: 📈
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
license: mit
---

# Forex AI Trading Inference Server

Real-time forex trading predictions using hybrid ML models.

See the full documentation in the repository.
EOF
    fi
fi

echo "✓ Checking src/ directory..."
if [ ! -d "src" ]; then
    if [ -d "$SOURCE_DIR/src" ]; then
        cp -r "$SOURCE_DIR/src" .
        echo "  Copied from source"
    else
        echo "⚠️  src/ directory not found - may cause issues"
    fi
fi

echo "✓ Checking models/ directory..."
if [ ! -d "models" ]; then
    if [ -d "$SOURCE_DIR/models" ]; then
        echo "  Copying models (this may take a moment)..."
        cp -r "$SOURCE_DIR/models" .
        echo "  ✅ Models copied"
    else
        echo "⚠️  models/ directory not found - server will fail without models!"
    fi
fi

echo ""
echo "=========================================="
echo "📤 Committing and Pushing to HF"
echo "=========================================="

# Step 4: Commit and push
git add .

if git diff --staged --quiet; then
    echo "✅ No changes to commit (already up to date)"
else
    echo "✓ Committing changes..."
    git commit -m "Deploy forex inference server - $(date '+%Y-%m-%d %H:%M:%S')"
    echo "✅ Changes committed"
fi

echo ""
echo "✓ Pushing to Hugging Face..."
git push

echo ""
echo "=========================================="
echo "✅ Deployment Complete!"
echo "=========================================="
echo ""
echo "🌐 Your Space: https://huggingface.co/spaces/morriase/forex-live_server"
echo "📊 Build Logs: https://huggingface.co/spaces/morriase/forex-live_server/logs"
echo ""
echo "⏱️  Build will take 5-10 minutes"
echo ""
echo "Once running, test with:"
echo "  curl https://morriase-forex-live-server.hf.space/health"
echo ""
echo "Update your EA with:"
echo "  RestServerURL = \"https://morriase-forex-live-server.hf.space/predict\""
echo ""
