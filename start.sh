#!/bin/sh
# Startup script for Hugging Face deployment

echo "=========================================="
echo "Starting Forex Inference Server for HF"
echo "=========================================="
echo "PORT: 7860 (hardcoded for Hugging Face)"
echo "Workers: 1"
echo "Threads: 2"
echo "Timeout: 300s"
echo "=========================================="

# Start gunicorn
# - Port is hardcoded to 7860 for Hugging Face Spaces.
# - --preload is REMOVED to prevent worker hangs with ML libraries.
# - The app and models will be loaded by the worker itself.
exec gunicorn \
    --workers 1 \
    --threads 2 \
    --worker-class sync \
    --timeout 300 \
    --graceful-timeout 30 \
    --max-requests 100 \
    --max-requests-jitter 10 \
    --bind 0.0.0.0:7860 \
    --access-logfile - \
    --error-logfile - \
    --log-level info \
    inference_server:app