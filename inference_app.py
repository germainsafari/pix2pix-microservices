from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from starlette.responses import JSONResponse
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
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
BLOCK_SIZE: int = 512 # Tile size for inference
OUTPUT_JPEG_QUALITY: int = 95 # Set output quality (1-100)
PAD_FILL_COLOR: Tuple[int, int, int] = (127, 127, 127) # Grey padding

# --- Setup Device ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Import PyTorch Network Definitions ---
try:
    # Adjust path to reliably find 'models' directory relative to this script
    current_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(current_dir) # Assuming models is in the same dir
    if models_dir not in sys.path:
        sys.path.append(models_dir)
    from models.networks import define_G
except ImportError as e:
     print(f"❌ Failed to import 'define_G'. Searched paths: {sys.path}")
     raise ImportError(f"Could not find 'models.networks' or 'define_G'. Ensure 'models/networks.py' exists relative to the script. Original error: {e}")


app = FastAPI(title="Pix2pix Inference Service", version="1.3") # Incremented version
netG = None # Global variable to hold the loaded model

class ImageRequest(BaseModel):
    image_url: str

# --- 1. Model Loading ---

def load_model() -> torch.nn.Module:
    """Loads the PyTorch Generator model into memory on startup."""
    global netG
    if netG is not None:
        print("Model already loaded.")
        return netG # Already loaded

    model_full_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), MODEL_PATH)
    if not os.path.exists(model_full_path):
        print(f"❌ ERROR: Model checkpoint not found at calculated path: {model_full_path}")
        raise FileNotFoundError(f"Model checkpoint not found at: {model_full_path}")

    try:
        print(f"Loading model from: {model_full_path}")
        netG = define_G(
            input_nc=INPUT_NC, output_nc=OUTPUT_NC, ngf=NGF, netG=MODEL_ARCH,
            norm=NORM_LAYER, use_dropout=False, init_type='normal',
            init_gain=0.02, gpu_ids=[] # Manual device assignment
        )
        print("Initializing model weights...")
        # Ensure map_location handles CPU/GPU correctly
        map_location = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        netG.load_state_dict(torch.load(model_full_path, map_location=map_location))
        netG.to(DEVICE).eval() # Set to evaluation mode
        print(f"✅ Pix2pix Model loaded successfully onto {DEVICE}.")
        return netG
    except Exception as e:
        print(f"❌ CRITICAL ERROR LOADING MODEL: {e}")
        import traceback
        traceback.print_exc()
        raise RuntimeError(f"Failed to load model: {e}")

# --- 2. Image Pre/Post-Processing Logic ---

def pad_to_multiple(img: Image.Image, block: int = BLOCK_SIZE, fill: Tuple[int, int, int] = PAD_FILL_COLOR) -> Tuple[Image.Image, Dict[str, int]]:
    """Pads image dimensions up to the nearest multiple of `block` size."""
    w, h = img.size
    new_w = math.ceil(w / block) * block
    new_h = math.ceil(h / block) * block

    # If already a multiple, no padding needed
    if new_w == w and new_h == h:
        meta = { "orig_w": w, "orig_h": h, "pad_left": 0, "pad_top": 0, "new_w": w, "new_h": h }
        return img, meta

    # Ensure canvas uses RGB for consistency
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
    if meta["pad_left"] == 0 and meta["pad_top"] == 0 and meta["new_w"] == meta["orig_w"] and meta["new_h"] == meta["orig_h"]:
        return fake_img

    L, T = meta["pad_left"], meta["pad_top"]
    w, h = meta["orig_w"], meta["orig_h"]
    # Define crop box: (left, top, right, bottom)
    right = min(L + w, fake_img.width)
    bottom = min(T + h, fake_img.height)
    crop_box = (L, T, right, bottom)

    # Validate crop box dimensions before attempting crop
    if crop_box[2] <= crop_box[0] or crop_box[3] <= crop_box[1]:
         print(f"⚠️ Warning: Invalid crop dimensions calculated {crop_box}. Returning UNPADDED but potentially incorrect image.")
         # Attempt to return the central part based on original dimensions if possible
         center_x, center_y = fake_img.width // 2, fake_img.height // 2
         half_w, half_h = w // 2, h // 2
         safe_crop_box = (max(0, center_x - half_w), max(0, center_y - half_h),
                          min(fake_img.width, center_x + (w - half_w)), min(fake_img.height, center_y + (h - half_h)))
         if safe_crop_box[2] > safe_crop_box[0] and safe_crop_box[3] > safe_crop_box[1]:
            return fake_img.crop(safe_crop_box)
         else:
            return fake_img # Return uncropped as last resort

    print(f"Cropping padded image from size {fake_img.size} using box {crop_box}")
    crop = fake_img.crop(crop_box)
    return crop

# --- 3. Tiled Inference Implementation (Using ToPILImage) ---

def tiled_inference(model: torch.nn.Module, padded_img: Image.Image, block: int = BLOCK_SIZE) -> Image.Image:
    """Performs inference by processing the image in tiles to save memory."""
    width, height = padded_img.size
    output_img = Image.new("RGB", (width, height))

    # Input transform (PIL -> Tensor -> Normalize)
    transform_to_tensor = transforms.Compose([
        transforms.ToTensor(), # PIL (HWC)[0,255] -> Tensor (CHW)[0.0, 1.0]
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # [0.0, 1.0] -> [-1.0, 1.0]
    ])

    # Output transform (Tensor -> PIL)
    transform_to_pil = transforms.ToPILImage()

    model.eval()
    with torch.no_grad():
        processed_tiles = 0
        for y in range(0, height, block):
            for x in range(0, width, block):
                # Define tile boundaries
                box = (x, y, min(x + block, width), min(y + block, height))
                tile = padded_img.crop(box)
                tile_w, tile_h = tile.size

                # Pad edge tiles before processing
                input_tile_for_model = tile
                if tile_w < block or tile_h < block:
                    padded_tile_canvas = Image.new("RGB", (block, block), PAD_FILL_COLOR)
                    padded_tile_canvas.paste(tile, (0, 0))
                    input_tile_for_model = padded_tile_canvas

                # Convert tile to tensor and normalize
                input_tensor = transform_to_tensor(input_tile_for_model).unsqueeze(0).to(DEVICE)

                # Perform inference
                fake_B_tensor = model(input_tensor) # Output: [-1.0, 1.0]

                # --- REFINED Post-process using ToPILImage ---
                # 1. Squeeze batch dimension, move to CPU
                output_data = fake_B_tensor[0].cpu() # (CHW)[-1.0, 1.0]

                # 2. De-normalize from [-1.0, 1.0] back to [0.0, 1.0]
                output_data = (output_data + 1.0) / 2.0 # (CHW)[0.0, 1.0]

                # 3. Clamp values strictly between 0 and 1 before converting to PIL
                output_data = torch.clamp(output_data, 0.0, 1.0)

                # 4. Convert tensor back to PIL Image (handles scaling to 0-255 internally)
                output_tile_pil = transform_to_pil(output_data) # PIL (HWC)[0, 255]

                # If the input tile was padded, crop the output tile back
                if tile_w < block or tile_h < block:
                    output_tile_pil = output_tile_pil.crop((0, 0, tile_w, tile_h))

                # Paste the processed tile into the output image
                output_img.paste(output_tile_pil, (x, y))
                processed_tiles += 1
        print(f"Processed {processed_tiles} tiles.")
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
        # Consider exiting if model load fails critically
        # sys.exit(1)

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

        # --- C. Pad Image ---
        print(f"Padding image to multiple of {BLOCK_SIZE}...")
        padded_img, meta = pad_to_multiple(img)
        print(f"Padding complete: New Size=({padded_img.width}, {padded_img.height})")

        # --- D. Run Tiled Inference ---
        print("Running tiled inference...")
        fake_B_padded_img = tiled_inference(netG, padded_img)
        print("Tiled inference complete.")

        # --- E. Restore Original Size ---
        print("Restoring image to original size...")
        restored_img = restore_to_original(fake_B_padded_img, meta)
        print(f"Restoration complete: Final Size=({restored_img.width}, {restored_img.height})")

        # --- F. Convert to Base64 ---
        print(f"Encoding result as JPEG (Quality={OUTPUT_JPEG_QUALITY})...")
        byte_arr = io.BytesIO()
        restored_img.save(byte_arr, format='JPEG', quality=OUTPUT_JPEG_QUALITY, subsampling=0)
        encoded_img_bytes = byte_arr.getvalue()

        # Correct Base64 encoding
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
    except FileNotFoundError as e:
         print(f"❌ Model file error during processing: {e}")
         raise HTTPException(status_code=500, detail="Model file missing or inaccessible.")
    except Exception as e:
        print(f"❌ Unexpected error during inference: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Internal server error during image processing: {type(e).__name__}")

