from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from starlette.responses import JSONResponse
import torch
from torchvision import transforms
from PIL import Image
import io
import requests
import math
import os
import sys
import warnings
import base64
from typing import Dict, Tuple

# Suppress PIL decompression bomb warning for large images
warnings.filterwarnings("ignore", category=Image.DecompressionBombWarning)

# --- CONFIGURATION ---
MODEL_PATH: str = "checkpoints/latest_net_G.pth"
MODEL_ARCH: str = 'unet_256'
NORM_LAYER: str = 'instance'
INPUT_NC: int = 3
OUTPUT_NC: int = 3
NGF: int = 64
TARGET_SIZE: int = 512  # Size to resize images to before inference (matches your Colab setup)
OUTPUT_JPEG_QUALITY: int = 95
PAD_FILL_COLOR: Tuple[int, int, int] = (127, 127, 127)  # Grey padding

# --- Setup Device ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Import PyTorch Network Definitions ---
try:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(current_dir)
    if models_dir not in sys.path:
        sys.path.append(models_dir)
    from models.networks import define_G
except ImportError as e:
    print(f"❌ Failed to import 'define_G'. Searched paths: {sys.path}")
    raise ImportError(f"Could not find 'models.networks' or 'define_G'. Ensure 'models/networks.py' exists relative to the script. Original error: {e}")

app = FastAPI(title="Pix2pix Inference Service", version="2.0")
netG = None  # Global variable to hold the loaded model

class ImageRequest(BaseModel):
    image_url: str

# --- 1. Model Loading ---

def load_model() -> torch.nn.Module:
    """Loads the PyTorch Generator model into memory on startup."""
    global netG
    if netG is not None:
        print("Model already loaded.")
        return netG

    model_full_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), MODEL_PATH)
    if not os.path.exists(model_full_path):
        print(f"❌ ERROR: Model checkpoint not found at calculated path: {model_full_path}")
        raise FileNotFoundError(f"Model checkpoint not found at: {model_full_path}")

    try:
        print(f"Loading model from: {model_full_path}")
        netG = define_G(
            input_nc=INPUT_NC, output_nc=OUTPUT_NC, ngf=NGF, netG=MODEL_ARCH,
            norm=NORM_LAYER, use_dropout=False, init_type='normal',
            init_gain=0.02
        )
        print("Initializing model weights...")
        map_location = DEVICE
        netG.load_state_dict(torch.load(model_full_path, map_location=map_location, weights_only=True))
        netG.to(DEVICE).eval()
        print(f"✅ Pix2pix Model loaded successfully onto {DEVICE}.")
        return netG
    except Exception as e:
        print(f"❌ CRITICAL ERROR LOADING MODEL: {e}")
        import traceback
        traceback.print_exc()
        raise RuntimeError(f"Failed to load model: {e}")

# --- 2. Image Pre/Post-Processing (Matching Colab Logic) ---

def pad_to_multiple(img: Image.Image, target_size: int = TARGET_SIZE, fill: Tuple[int, int, int] = PAD_FILL_COLOR) -> Tuple[Image.Image, Dict[str, int]]:
    """
    Pads image to be at least target_size x target_size, with padding to next multiple of target_size.
    This matches the logic used in your Colab preprocessing.
    """
    w, h = img.size
    
    # Calculate new dimensions (pad to at least target_size, then to next multiple)
    new_w = max(target_size, math.ceil(w / target_size) * target_size)
    new_h = max(target_size, math.ceil(h / target_size) * target_size)

    # If already the right size, no padding needed
    if new_w == w and new_h == h:
        meta = {"orig_w": w, "orig_h": h, "pad_left": 0, "pad_top": 0, "new_w": w, "new_h": h}
        return img, meta

    # Create padded canvas
    canvas = Image.new("RGB", (new_w, new_h), fill)
    pad_left = (new_w - w) // 2
    pad_top = (new_h - h) // 2
    canvas.paste(img, (pad_left, pad_top))

    meta = {
        "orig_w": w, "orig_h": h,
        "pad_left": pad_left, "pad_top": pad_top,
        "new_w": new_w, "new_h": new_h
    }
    return canvas, meta

def restore_to_original(fake_img: Image.Image, meta: Dict[str, int]) -> Image.Image:
    """Crops padding from the generated image using metadata."""
    # If no padding was applied, return the image directly
    if meta["pad_left"] == 0 and meta["pad_top"] == 0:
        if meta["new_w"] == meta["orig_w"] and meta["new_h"] == meta["orig_h"]:
            return fake_img

    L, T = meta["pad_left"], meta["pad_top"]
    w, h = meta["orig_w"], meta["orig_h"]
    
    # Crop to original dimensions
    crop = fake_img.crop((L, T, L + w, T + h))
    return crop

# --- 3. Full Image Inference (Matching Colab test.py approach) ---

def full_image_inference(model: torch.nn.Module, img: Image.Image) -> Image.Image:
    """
    Performs inference on the full image at once (no tiling).
    This matches your Colab test.py behavior with --preprocess none.
    """
    # Define transforms (matching pytorch-CycleGAN-and-pix2pix preprocessing)
    transform = transforms.Compose([
        transforms.ToTensor(),  # [0, 255] -> [0.0, 1.0]
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # [0.0, 1.0] -> [-1.0, 1.0]
    ])

    model.eval()
    with torch.no_grad():
        # Convert to tensor and add batch dimension
        input_tensor = transform(img).unsqueeze(0).to(DEVICE)
        
        # Run inference
        output_tensor = model(input_tensor)
        
        # Post-process: denormalize from [-1, 1] to [0, 1]
        output_tensor = (output_tensor + 1.0) / 2.0
        output_tensor = torch.clamp(output_tensor, 0.0, 1.0)
        
        # Convert to PIL Image
        output_tensor = output_tensor[0].cpu()  # Remove batch dimension
        output_np = output_tensor.permute(1, 2, 0).numpy()  # CHW -> HWC
        output_np = (output_np * 255).astype('uint8')
        output_img = Image.fromarray(output_np, mode='RGB')
    
    return output_img

# --- FastAPI Hooks ---

@app.on_event("startup")
async def startup_event():
    """Load the model during application startup."""
    try:
        load_model()
        print("Application startup complete.")
    except Exception as e:
        print(f"❌ Application startup failed due to model loading error: {e}")

@app.get("/health", summary="Check service health and model status")
def health_check():
    """Provides health status, including whether the model is loaded."""
    model_status = "loaded" if netG is not None else "not loaded"
    return {"status": "ok", "model_status": model_status, "device": str(DEVICE)}

# --- 4. Inference Endpoint ---

@app.post("/process", summary="Process an image using the Pix2pix model")
async def inference_endpoint(request: ImageRequest):
    """Receives an image URL, processes it with the model, returns Base64 result."""
    if netG is None:
        print("❌ Inference request failed: Model is not loaded.")
        raise HTTPException(status_code=503, detail="Model not loaded or failed to initialize.")

    print(f"Processing image URL: {request.image_url[:100]}...")

    try:
        # --- A. Fetch Image ---
        print("Fetching image...")
        response = requests.get(request.image_url, timeout=60)
        response.raise_for_status()
        img_bytes = response.content
        if not img_bytes:
            raise ValueError("Fetched image content is empty.")
        print(f"Image fetched ({len(img_bytes)} bytes).")

        # --- B. Open and Convert Image ---
        print("Opening image with PIL...")
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        print(f"Image opened: Size=({img.width}, {img.height}), Mode={img.mode}")

        # --- C. Pad Image to Target Size ---
        print(f"Padding image to multiple of {TARGET_SIZE}...")
        padded_img, meta = pad_to_multiple(img, TARGET_SIZE)
        print(f"Padding complete: New Size=({padded_img.width}, {padded_img.height})")

        # --- D. Run Full Image Inference (NO TILING) ---
        print("Running full-image inference...")
        fake_B_padded_img = full_image_inference(netG, padded_img)
        print("Inference complete.")

        # --- E. Restore Original Size ---
        print("Restoring image to original size...")
        restored_img = restore_to_original(fake_B_padded_img, meta)
        print(f"Restoration complete: Final Size=({restored_img.width}, {restored_img.height})")

        # --- F. Convert to Base64 ---
        print(f"Encoding result as JPEG (Quality={OUTPUT_JPEG_QUALITY})...")
        byte_arr = io.BytesIO()
        restored_img.save(byte_arr, format='JPEG', quality=OUTPUT_JPEG_QUALITY, subsampling=0)
        encoded_img_bytes = byte_arr.getvalue()

        base64_encoded_data = base64.b64encode(encoded_img_bytes).decode('utf-8')
        data_url = f"data:image/jpeg;base64,{base64_encoded_data}"
        print("Encoding complete. Sending response.")

        return JSONResponse(content={"editedImageBase64": data_url})

    except requests.exceptions.Timeout:
        print(f"❌ Timeout error fetching image: {request.image_url}")
        raise HTTPException(status_code=408, detail="Timeout fetching image URL.")
    except requests.exceptions.RequestException as e:
        print(f"❌ Error fetching image: {e}")
        raise HTTPException(status_code=400, detail=f"Failed to fetch or invalid image URL: {e}")
    except torch.cuda.OutOfMemoryError:
        print(f"❌ CUDA Out of Memory error during inference")
        raise HTTPException(status_code=500, detail="GPU out of memory. Image may be too large.")
    except Exception as e:
        print(f"❌ Unexpected error during inference: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Internal server error during image processing: {type(e).__name__}")