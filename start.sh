#!/bin/sh
# Startup script for Hugging Face deployment

echo "========================================="
echo "Starting Forex Inference Server for HF"
echo "========================================="
echo "PORT: 7860 (hardcoded for Hugging Face)"
echo "Workers: 1"
echo "Threads: 2"
echo "========================================="

# Test if the app can be imported before starting the server.
# This helps catch basic import errors early and provides startup logging.
echo "Testing app import..."
python -c "from inference_server import app; print('✓ App imported successfully')" || {
    echo "✗ FATAL: Failed to import Flask app from inference_server.py"
    exit 1
}

echo "Starting Gunicorn..."

# Start gunicorn
# - Port is hardcoded to 7860 for Hugging Face Spaces.
# - --preload is REMOVED as a likely cause of worker hangs with ML libraries.
exec gunicorn \
    --workers 1 \
    --threads 2 \
    --worker-class sync \
    --timeout 300 \
    --bind 0.0.0.0:7860 \
    --access-logfile - \
    --error-logfile - \
    --log-level info \
    inference_server:app
