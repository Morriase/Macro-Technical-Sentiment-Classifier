#!/bin/bash
# Deploy to Hugging Face Spaces
# Space: https://huggingface.co/spaces/morriase/forex-live_server

set -e

echo "=========================================="
echo "Deploying to Hugging Face Spaces"
echo "=========================================="

# Check required files
echo "✓ Checking required files..."
required_files=("app.py" "Dockerfile" "inference_server.py" "start.sh" "requirements_render.txt")
for file in "${required_files[@]}"; do
    if [ ! -f "$file" ]; then
        echo "❌ Missing: $file"
        exit 1
    fi
done

# Prepare README for HF (with YAML frontmatter)
echo "✓ Preparing README..."
if [ -f README_HF.md ]; then
    cp README_HF.md README.md
    echo "  Using README_HF.md as README.md"
fi

# Stage and commit changes
echo "✓ Staging changes..."
git add -A

if ! git diff --staged --quiet; then
    echo "✓ Committing changes..."
    git commit -m "Deploy to Hugging Face Spaces - $(date '+%Y-%m-%d %H:%M:%S')"
else
    echo "  No changes to commit"
fi

# Push to Hugging Face
echo "✓ Pushing to Hugging Face..."
git push hf main

echo ""
echo "=========================================="
echo "✅ Deployment Complete!"
echo "=========================================="
echo ""
echo "Your Space: https://huggingface.co/spaces/morriase/forex-live_server"
echo ""
echo "Monitor build logs at:"
echo "https://huggingface.co/spaces/morriase/forex-live_server/logs"
echo ""
echo "Once deployed, test with:"
echo "curl https://morriase-forex-live-server.hf.space/health"
echo ""
