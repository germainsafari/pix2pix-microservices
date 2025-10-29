# Pix2Pix Inference Microservice

A FastAPI-based microservice for running Pix2Pix image-to-image translation inference.

## Project Structure

```
.
├── inference_app.py          # Main FastAPI application
├── requirements.txt          # Python dependencies
├── Dockerfile               # Docker configuration
├── models/                  # Model architecture definitions
│   ├── __init__.py
│   └── networks.py          # U-Net generator network definition
└── checkpoints/             # Trained model weights
    └── latest_net_G.pth     # Generator checkpoint
```

## Features

- **Tiled Inference**: Handles large images by splitting them into tiles to prevent out-of-memory errors
- **FastAPI**: Modern async web framework for easy deployment
- **Docker Support**: Ready for containerized deployment
- **Health Check**: Built-in health endpoint for monitoring

## Local Setup

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the server:**
   ```bash
   uvicorn inference_app:app --host 0.0.0.0 --port 8000
   ```

3. **Test the health endpoint:**
   ```bash
   curl http://localhost:8000/health
   ```

## API Endpoints

### POST `/process`

Process an image through the Pix2Pix model.

**Request:**
```json
{
  "image_url": "https://example.com/image.jpg"
}
```

**Response:**
```json
{
  "editedImageBase64": "data:image/jpeg;base64,/9j/4AAQSkZJRg..."
}
```

### GET `/health`

Check service health and model status.

**Response:**
```json
{
  "status": "ok",
  "model_loaded": true,
  "device": "cuda"
}
```

## Docker Deployment

1. **Build the image:**
   ```bash
   docker build -t pix2pix-service .
   ```

2. **Run the container:**
   ```bash
   docker run -p 8000:8000 pix2pix-service
   ```

## Configuration

The following constants in `inference_app.py` can be adjusted:

- `MODEL_ARCH`: Model architecture (`unet_256`, `unet_128`, `resnet_9blocks`, `resnet_6blocks`)
- `BLOCK_SIZE`: Tile size for inference (default: 512)
- `INPUT_NC`: Number of input channels (default: 3 for RGB)
- `OUTPUT_NC`: Number of output channels (default: 3 for RGB)
- `NGF`: Number of generator filters (default: 64)

## Deployment Platforms

### Render

#### ⚠️ Important: Model File Size Limitation
The model file (`checkpoints/latest_net_G.pth`) is **~207 MB**, which **exceeds GitHub's 100 MB file size limit**. GitHub will reject pushes containing files over 100 MB.

#### Option 1: Using Git LFS (Recommended)
Use Git Large File Storage (LFS) to handle the large model file:

1. **Install Git LFS** (if not already installed):
   ```bash
   git lfs install
   ```

2. **Track the model file with Git LFS:**
   ```bash
   git lfs track "checkpoints/*.pth"
   git add .gitattributes
   git add checkpoints/latest_net_G.pth
   git commit -m "Add model checkpoint with Git LFS"
   git push
   ```

3. **Deploy on Render:**
   - Go to [Render Dashboard](https://dashboard.render.com)
   - Click "New +" → "Web Service"
   - Connect your repository
   - Render automatically detects `render.yaml` and configures the service
   - Git LFS files are automatically downloaded during build

#### Option 2: Download Model at Build Time
Store your model in cloud storage (S3, Google Drive, Dropbox, etc.) and download it during Docker build:

1. **Upload model to cloud storage** (get a direct download link)
2. **Update Dockerfile** to download the model:
   ```dockerfile
   # Add after COPY models/ ./models/
   RUN mkdir -p checkpoints && \
       curl -L <YOUR_MODEL_URL> -o checkpoints/latest_net_G.pth
   ```
3. **Don't commit the model file** - add it to `.gitignore`
4. **Use an environment variable** for the model URL in `render.yaml`

#### Option 3: Native Python Deployment (without Docker)
1. Connect your repository (without the model file)
2. **Build Command:** `pip install -r requirements.txt && python download_model.py`
3. **Start Command:** `uvicorn inference_app:app --host 0.0.0.0 --port $PORT`
4. **Environment:** Python 3.10
5. **Plan:** Starter plan ($7/month) recommended

#### Deployment Configuration:
- **Environment:** Docker (Option 1) or Python 3.10 (Option 3)
- **Health Check Path:** `/health`
- **Build time:** 5-10 minutes (due to PyTorch installation)

### Hugging Face Spaces
1. Push code to a repository
2. Create a new Space with Docker SDK
3. The Dockerfile will automatically be used

## Notes

- The model file (`latest_net_G.pth`) must be in the `checkpoints/` directory
- GPU support is automatic if CUDA is available
- Large images are automatically tiled to prevent memory issues

