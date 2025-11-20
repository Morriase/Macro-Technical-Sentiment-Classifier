"""
Hugging Face Spaces Entry Point
This file is required by Hugging Face Spaces to run the application
"""
from inference_server import app

# Hugging Face Spaces will automatically use this app
if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 7860))  # HF Spaces uses port 7860
    app.run(host="0.0.0.0", port=port)
