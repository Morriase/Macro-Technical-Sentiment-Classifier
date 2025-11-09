#!/bin/sh
# Startup script for Render deployment

# Get PORT from environment or use default
PORT=${PORT:-10000}

echo "Starting Gunicorn on port $PORT"
echo "Workers: 2, Threads: 4, Timeout: 120s"

# Start gunicorn
exec gunicorn \
    --workers 2 \
    --threads 4 \
    --timeout 120 \
    --bind 0.0.0.0:$PORT \
    --access-logfile - \
    --error-logfile - \
    --log-level info \
    inference_server:app
