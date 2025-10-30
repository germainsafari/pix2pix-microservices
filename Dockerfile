# Use Python 3.10 slim image for smaller size
FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY inference_app.py .
COPY models/ ./models/
COPY checkpoints/ ./checkpoints/

# Expose is optional on Render; harmless either way
EXPOSE 8000

# IMPORTANT: bind to the port Render provides
CMD ["sh", "-c", "uvicorn inference_app:app --host 0.0.0.0 --port ${PORT}"]
