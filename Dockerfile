# Stage 1: Build TA-Lib from source
# This keeps the final image clean and small
FROM python:3.9-slim as talib_builder

# Install build dependencies
RUN apt-get update && \
    apt-get install -y \
    build-essential \
    wget && \
    rm -rf /var/lib/apt/lists/*

# Download and compile TA-Lib
WORKDIR /tmp
RUN wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz && \
    tar -xzf ta-lib-0.4.0-src.tar.gz && \
    cd ta-lib/ && \
    ./configure --prefix=/usr && \
    make && \
    make install

# ---------------------------------------------------------------------

# Stage 2: Final application image
FROM python:3.9-slim

LABEL maintainer="Your Name"
LABEL description="Inference server for the Macro-Technical Sentiment Classifier"

# Set working directory
WORKDIR /app

# --- TA-Lib Installation ---
# Copy compiled TA-Lib libraries from the builder stage
COPY --from=talib_builder /usr/lib/libta_lib.so.0 /usr/lib/libta_lib.so.0
COPY --from=talib_builder /usr/lib/libta_lib.so.0.0.0 /usr/lib/libta_lib.so.0.0.0
COPY --from=talib_builder /usr/include/ta-lib/ta_defs.h /usr/include/ta-lib/ta_defs.h
COPY --from=talib_builder /usr/include/ta-lib/ta_common.h /usr/include/ta-lib/ta_common.h
COPY --from=talib_builder /usr/include/ta-lib/ta_func.h /usr/include/ta-lib/ta_func.h

# Update library cache
RUN ldconfig

# --- Python Dependencies ---
# Copy requirements first to leverage Docker cache
COPY requirements_render.txt ./requirements.txt

# Install Python packages
# We specify the TA-Lib version to ensure compatibility
RUN pip install --no-cache-dir -r requirements.txt

# --- Application Code ---
# Copy the rest of your application code
COPY src/ ./src
COPY models/ ./models
COPY inference_server.py .
COPY start.sh .

# Create logs directory for the application
RUN mkdir logs

# --- Server Configuration ---
# Expose the port the server runs on (Render uses dynamic PORT)
EXPOSE 10000

# Set the command to run the application
# Using gunicorn for a production-ready WSGI server
RUN pip install gunicorn

# Make startup script executable
RUN chmod +x start.sh

# Use startup script to handle PORT environment variable
CMD ["./start.sh"]
