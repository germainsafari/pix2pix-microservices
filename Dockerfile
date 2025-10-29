# Use Python 3.10 slim image for smaller size
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies for PyTorch and image processing
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY inference_app.py .
COPY models/ ./models/
COPY checkpoints/ ./checkpoints/

# Expose the port FastAPI will run on
EXPOSE 8000

# Run the application
CMD ["uvicorn", "inference_app:app", "--host", "0.0.0.0", "--port", "8000"]

