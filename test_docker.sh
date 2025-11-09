#!/bin/bash
# Quick Docker validation - tests locally before pushing to Render
# This catches 90% of deployment errors in 2-3 minutes instead of 15+ minutes

echo "=========================================="
echo "QUICK DOCKER VALIDATION"
echo "=========================================="
echo ""
echo "This will:"
echo "  1. Build Docker image locally"
echo "  2. Test that it starts"
echo "  3. Test health endpoint"
echo ""
echo "Time: ~3 minutes (vs 15+ on Render)"
echo ""

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker is not running!"
    echo "   Start Docker Desktop and try again."
    exit 1
fi

echo "✓ Docker is running"
echo ""

# Build image
echo "Step 1: Building Docker image..."
echo "---------------------------------------"
docker build -t forex-test:latest . || {
    echo ""
    echo "❌ Docker build failed!"
    echo "   Fix the errors above before pushing to Render."
    exit 1
}

echo ""
echo "✓ Docker build successful"
echo ""

# Test run
echo "Step 2: Starting container..."
echo "---------------------------------------"
CONTAINER_ID=$(docker run -d -p 5555:10000 -e PORT=10000 forex-test:latest)

if [ -z "$CONTAINER_ID" ]; then
    echo "❌ Failed to start container"
    exit 1
fi

echo "✓ Container started: $CONTAINER_ID"
echo ""

# Wait for startup
echo "Step 3: Waiting for server to start (30 seconds)..."
echo "---------------------------------------"
sleep 30

# Check logs
echo ""
echo "Container logs:"
echo "---------------------------------------"
docker logs $CONTAINER_ID 2>&1 | tail -20
echo "---------------------------------------"
echo ""

# Test health endpoint
echo "Step 4: Testing health endpoint..."
echo "---------------------------------------"
RESPONSE=$(curl -s http://localhost:5555/health)

if [ -z "$RESPONSE" ]; then
    echo "❌ No response from health endpoint"
    echo ""
    echo "Full logs:"
    docker logs $CONTAINER_ID
    docker stop $CONTAINER_ID > /dev/null
    docker rm $CONTAINER_ID > /dev/null
    exit 1
fi

echo "Response:"
echo "$RESPONSE" | python -m json.tool 2>/dev/null || echo "$RESPONSE"
echo ""

# Cleanup
echo "Step 5: Cleaning up..."
echo "---------------------------------------"
docker stop $CONTAINER_ID > /dev/null
docker rm $CONTAINER_ID > /dev/null
echo "✓ Container stopped and removed"
echo ""

# Success!
echo "=========================================="
echo "✅ VALIDATION SUCCESSFUL!"
echo "=========================================="
echo ""
echo "Your Docker image works locally."
echo "It should work on Render too."
echo ""
echo "Next steps:"
echo "  git add ."
echo "  git commit -m 'Your message'"
echo "  git push origin main"
echo ""
