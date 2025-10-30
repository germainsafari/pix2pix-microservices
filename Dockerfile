FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libglib2.0-0 libsm6 libxext6 libxrender-dev libgomp1 \
    && rm -rf /var/lib/apt/lists/*

ENV OMP_NUM_THREADS=1
ENV MKL_NUM_THREADS=1
ENV TORCH_NUM_THREADS=1

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY inference_app.py .
COPY models/ ./models/
COPY checkpoints/ ./checkpoints/

EXPOSE 8000
CMD ["sh", "-c", "uvicorn inference_app:app --host 0.0.0.0 --port ${PORT} --workers 1"]
