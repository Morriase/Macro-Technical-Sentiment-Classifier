"""
Run the inference server locally for testing
Usage: python run_local_server.py
"""
from inference_server import app
import os
import sys
from pathlib import Path

# Set environment variables
os.environ['PORT'] = '5000'
os.environ['ENABLE_LIVE_SENTIMENT'] = 'False'

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

print("=" * 80)
print("STARTING LOCAL FOREX INFERENCE SERVER")
print("=" * 80)
print(f"URL: http://localhost:5000")
print(f"Models directory: {project_root / 'models'}")
print(f"Supported pairs: EUR_USD, GBP_USD, USD_JPY, AUD_USD")
print("=" * 80)
print("\nEndpoints:")
print("  GET  /health          - Health check")
print("  POST /predict         - Make prediction")
print("  GET  /model_info/<pair> - Get model info")
print("\nPress Ctrl+C to stop the server")
print("=" * 80)

# Import and run the Flask app

if __name__ == '__main__':
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=True,  # Enable debug mode for local testing
        threaded=True
    )
