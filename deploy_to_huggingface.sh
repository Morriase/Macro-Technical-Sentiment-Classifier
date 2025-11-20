#!/bin/bash
# Quick deployment script for Hugging Face Spaces

echo "=========================================="
echo "Hugging Face Spaces Deployment Script"
echo "=========================================="

# Check if git is initialized
if [ ! -d .git ]; then
    echo "❌ Error: Not a git repository"
    echo "Run: git init"
    exit 1
fi

# Check if files exist
echo "Checking required files..."
required_files=("app.py" "Dockerfile" "inference_server.py" "start.sh" "requirements_render.txt")
missing_files=()

for file in "${required_files[@]}"; do
    if [ ! -f "$file" ]; then
        missing_files+=("$file")
    fi
done

if [ ${#missing_files[@]} -ne 0 ]; then
    echo "❌ Missing required files:"
    printf '   - %s\n' "${missing_files[@]}"
    exit 1
fi

echo "✅ All required files present"

# Backup and rename README
echo ""
echo "Preparing README for Hugging Face..."
if [ -f README.md ]; then
    if [ ! -f README_PROJECT.md ]; then
        echo "   Backing up README.md → README_PROJECT.md"
        cp README.md README_PROJECT.md
    fi
fi

if [ -f README_HF.md ]; then
    echo "   Using README_HF.md as README.md"
    cp README_HF.md README.md
else
    echo "⚠️  Warning: README_HF.md not found"
fi

# Stage files
echo ""
echo "Staging files for commit..."
git add app.py README.md HUGGINGFACE_DEPLOYMENT.md DEPLOYMENT_COMPARISON.md

# Check if there are changes to commit
if git diff --staged --quiet; then
    echo "✅ No changes to commit (already up to date)"
else
    # Commit
    echo ""
    echo "Committing changes..."
    git commit -m "Add Hugging Face Spaces deployment files"
    echo "✅ Changes committed"
fi

# Push to GitHub
echo ""
echo "Pushing to GitHub..."
git push origin main

if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "✅ Deployment Successful!"
    echo "=========================================="
    echo ""
    echo "Next steps:"
    echo "1. Go to https://huggingface.co/spaces"
    echo "2. Create a new Space (Docker SDK)"
    echo "3. Link your GitHub repository"
    echo "4. Wait for build to complete (~5-10 min)"
    echo "5. Test your endpoint"
    echo ""
    echo "Your Space URL will be:"
    echo "https://YOUR_USERNAME-forex-inference-server.hf.space"
    echo ""
    echo "See HUGGINGFACE_DEPLOYMENT.md for detailed instructions"
    echo "=========================================="
else
    echo ""
    echo "❌ Push failed. Check your git configuration."
    exit 1
fi
