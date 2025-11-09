#!/bin/sh
# Startup script for Render deployment

# Get PORT from environment or use default
PORT=${PORT:-10000}

echo "=========================================="
echo "Starting Forex Inference Server"
echo "=========================================="
echo "PORT: $PORT"
echo "Workers: 1 (reduced for debugging)"
echo "Threads: 2"
echo "Timeout: 300s"
echo "Python: $(python --version)"
echo "Working directory: $(pwd)"
echo "=========================================="

# Test if app can import
echo "Testing app import..."
python -c "from inference_server import app; print('✓ App imported successfully')" || {
    echo "✗ Failed to import app"
    exit 1
}

echo "Starting Gunicorn..."

# Start gunicorn with reduced workers for stability
exec gunicorn \
    --workers 1 \
    --threads 2 \
    --timeout 300 \
    --bind 0.0.0.0:$PORT \
    --access-logfile - \
    --error-logfile - \
    --log-level debug \
    --preload \
    inference_server:app
