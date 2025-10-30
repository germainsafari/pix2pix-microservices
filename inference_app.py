from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from starlette.responses import JSONResponse

import io
import os
import sys
import math
import base64
import hashlib
import warnings
import logging
from typing import Dict, Tuple, Optional

import requests
import numpy as np
from PIL import Image

import torch
from torchvision import transforms

# Silence PIL DecompressionBomb warnings on large images
warnings.filterwarnings("ignore", category=Image.DecompressionBombWarning)

# -------- Configuration --------
MODEL_PATH: str = os.environ.get("MODEL_PATH", "checkpoints/latest_net_G.pth")
MODEL_ARCH: str = os.environ.get("MODEL_ARCH", "unet_256")
NORM_LAYER: str = os.environ.get("NORM_LAYER", "instance")
INPUT_NC: int = int(os.environ.get("INPUT_NC", "3"))
OUTPUT_NC: int = int(os.environ.get("OUTPUT_NC", "3"))
NGF: int = int(os.environ.get("NGF", "64"))

TILE_SIZE: int = int(os.environ.get("TILE_SIZE", "512"))
TILE_OVERLAP: int = int(os.environ.get("TILE_OVERLAP", "64"))
DEFAULT_FORMAT: str = os.environ.get("DEFAULT_FORMAT", "PNG")
JPEG_QUALITY: int = int(os.environ.get("JPEG_QUALITY", "95"))

# Memory safety
MAX_PIXELS_DIRECT: int = int(os.environ.get("MAX_PIXELS_DIRECT", str(4096 * 4096)))

# Network settings
FETCH_TIMEOUT_SECS: int = int(os.environ.get("FETCH_TIMEOUT_SECS", "20"))
HEAD_TIMEOUT_SECS: int = int(os.environ.get("HEAD_TIMEOUT_SECS", "5"))
MAX_BODY_BYTES: int = int(os.environ.get("MAX_BODY_BYTES", str(25 * 1024 * 1024)))  # 25MB

# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Logger
logger = logging.getLogger("pix2pix")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO)
logger.setLevel(logging.INFO)

# -------- Import model factory from your repo --------
try:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if current_dir not in sys.path:
        sys.path.append(current_dir)
    from models.networks import define_G
except Exception as e:
    print(f"Failed to import define_G from models.networks: {e}")
    raise

# -------- FastAPI app --------
app = FastAPI(title="Pix2Pix Inference Service", version="2.1")

class ImageRequest(BaseModel):
    image_url: str
    output_format: Optional[str] = None   # "PNG" or "JPEG"
    jpeg_quality: Optional[int] = None

class ImageBytesRequest(BaseModel):
    image_base64: str
    output_format: Optional[str] = None
    jpeg_quality: Optional[int] = None

netG = None  # global model handle

# -------- Helpers --------
def _weight_checksum(state_dict: Dict[str, torch.Tensor]) -> str:
    """Small checksum to confirm the exact weights loaded."""
    m = hashlib.md5()
    for k in sorted(state_dict.keys()):
        t = state_dict[k]
        m.update(k.encode("utf-8"))
        m.update(t.cpu().numpy().tobytes())
    return m.hexdigest()[:12]

def _to_tensor_transform():
    return transforms.Compose([
        transforms.ToTensor(),                          # HWC [0,255] -> CHW [0,1]
        transforms.Normalize((0.5, 0.5, 0.5),
                             (0.5, 0.5, 0.5))          # [0,1] -> [-1,1]
    ])

def _hann_window_2d(h: int, w: int) -> np.ndarray:
    if h <= 1 or w <= 1:
        return np.ones((h, w), dtype=np.float32)
    y = np.linspace(0.0, np.pi, h, dtype=np.float32)
    x = np.linspace(0.0, np.pi, w, dtype=np.float32)
    wy = np.sin(y) ** 2
    wx = np.sin(x) ** 2
    return np.sqrt(np.outer(wy, wx)).astype(np.float32)

def pad_to_multiple_reflect(img: Image.Image, block: int) -> Tuple[Image.Image, Dict[str, int]]:
    """Pad to multiple of block with reflection to avoid border contamination."""
    w, h = img.size
    new_w = math.ceil(w / block) * block
    new_h = math.ceil(h / block) * block
    if new_w == w and new_h == h:
        return img, {"orig_w": w, "orig_h": h, "pad_left": 0, "pad_top": 0, "new_w": w, "new_h": h}

    arr = np.array(img)
    pad_left = (new_w - w) // 2
    pad_right = new_w - w - pad_left
    pad_top = (new_h - h) // 2
    pad_bot = new_h - h - pad_top
    arr_pad = np.pad(arr, ((pad_top, pad_bot), (pad_left, pad_right), (0, 0)), mode="reflect")
    padded = Image.fromarray(arr_pad, mode="RGB")
    meta = {"orig_w": w, "orig_h": h, "pad_left": pad_left, "pad_top": pad_top, "new_w": new_w, "new_h": new_h}
    return padded, meta

def restore_to_original(fake_img: Image.Image, meta: Dict[str, int]) -> Image.Image:
    L, T, w, h = meta["pad_left"], meta["pad_top"], meta["orig_w"], meta["orig_h"]
    if L == 0 and T == 0 and meta["new_w"] == w and meta["new_h"] == h:
        return fake_img
    right = min(L + w, fake_img.width)
    bottom = min(T + h, fake_img.height)
    if right <= L or bottom <= T:
        cx, cy = fake_img.width // 2, fake_img.height // 2
        half_w, half_h = w // 2, h // 2
        return fake_img.crop((max(0, cx - half_w), max(0, cy - half_h),
                              min(fake_img.width, cx + half_w), min(fake_img.height, cy + half_h)))
    return fake_img.crop((L, T, right, bottom))

def run_model_on_tensor(model: torch.nn.Module, tensor_bchw: torch.Tensor) -> torch.Tensor:
    """Forward pass that returns de-normalized [0,1] CHW on CPU."""
    with torch.no_grad():
        out = model(tensor_bchw.to(DEVICE))[0].cpu()     # CHW in [-1,1]
        out = (out + 1.0) / 2.0                          # CHW in [0,1]
        out = torch.clamp(out, 0.0, 1.0)
    return out

def inference_no_tiling(model: torch.nn.Module, img: Image.Image) -> Image.Image:
    to_tensor = _to_tensor_transform()
    t = to_tensor(img).unsqueeze(0)
    out = run_model_on_tensor(model, t).permute(1, 2, 0).numpy()  # HWC
    return Image.fromarray((out * 255.0 + 0.5).astype(np.uint8), mode="RGB")

def inference_tiled_blend(model: torch.nn.Module, img: Image.Image, tile: int, overlap: int) -> Image.Image:
    W, H = img.size
    acc = np.zeros((H, W, 3), dtype=np.float32)
    wacc = np.zeros((H, W, 1), dtype=np.float32)
    win_full = _hann_window_2d(tile, tile)[..., None]  # HxW×1
    to_tensor = _to_tensor_transform()

    y = 0
    while True:
        x = 0
        tile_h = min(tile, H - y)
        y0, y1 = y, y + tile_h
        while True:
            tile_w = min(tile, W - x)
            x0, x1 = x, x + tile_w

            patch = img.crop((x0, y0, x1, y1))
            if tile_w != tile or tile_h != tile:
                pnp = np.array(patch)
                pr = tile - tile_w
                pb = tile - tile_h
                pnp = np.pad(pnp, ((0, pb), (0, pr), (0, 0)), mode="reflect")
                patch = Image.fromarray(pnp, "RGB")

            t = to_tensor(patch).unsqueeze(0)
            out = run_model_on_tensor(model, t).permute(1, 2, 0).numpy()
            out = out[:tile_h, :tile_w, :]

            wtile = win_full[:tile_h, :tile_w, :].astype(np.float32)
            acc[y0:y1, x0:x1, :] += out * wtile
            wacc[y0:y1, x0:x1, :] += wtile

            if x1 >= W:
                break
            x = x + tile - overlap
            if x + 1 >= W:
                x = W - tile
        if y1 >= H:
            break
        y = y + tile - overlap
        if y + 1 >= H:
            y = H - tile

    out = (acc / np.maximum(wacc, 1e-8)).clip(0.0, 1.0)
    return Image.fromarray((out * 255.0 + 0.5).astype(np.uint8), mode="RGB")

def _verify_public_url(url: str) -> None:
    """Fast check to fail early on dead or private URLs."""
    try:
        r = requests.head(url, timeout=HEAD_TIMEOUT_SECS, allow_redirects=True)
        if r.status_code >= 400:
            rg = requests.get(url, timeout=HEAD_TIMEOUT_SECS, headers={"Range": "bytes=0-0"})
            if rg.status_code >= 400:
                raise HTTPException(status_code=400, detail=f"Image URL not fetchable. Status {r.status_code}/{rg.status_code}")
    except requests.exceptions.Timeout:
        raise HTTPException(status_code=408, detail="Timeout verifying image URL")
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=400, detail=f"Bad image URL or blocked from server. {e}")

# -------- Model loading --------
def load_model() -> torch.nn.Module:
    global netG
    if netG is not None:
        return netG

    model_full_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), MODEL_PATH)
    if not os.path.exists(model_full_path):
        raise FileNotFoundError(f"Model checkpoint not found at {model_full_path}")

    net = define_G(
        input_nc=INPUT_NC,
        output_nc=OUTPUT_NC,
        ngf=NGF,
        netG=MODEL_ARCH,
        norm=NORM_LAYER,
        use_dropout=False,
        init_type="normal",
        init_gain=0.02,
    )

    map_location = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    state = torch.load(model_full_path, map_location=map_location)
    checksum = _weight_checksum(state)
    print(f"Loaded checkpoint: {model_full_path}  md5={checksum}")

    missing, unexpected = net.load_state_dict(state, strict=False)
    if missing:
        print(f"Warning. Missing keys: {missing}")
    if unexpected:
        print(f"Warning. Unexpected keys: {unexpected}")

    net.to(DEVICE).eval()
    netG = net
    print(f"Model ready on {DEVICE}: arch={MODEL_ARCH} norm={NORM_LAYER}")
    return netG

# -------- FastAPI lifecycle --------
@app.on_event("startup")
async def _startup():
    try:
        print("Starting up...")
        load_model()
        print("Startup complete. Model loaded.")
    except Exception as e:
        print(f"Startup failed: {e}")

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": netG is not None, "device": str(DEVICE)}

# -------- Inference endpoints --------
@app.post("/process")
async def process(req: ImageRequest):
    if netG is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # A. Fetch image by URL
    try:
        logger.info(f"Fetching URL: {req.image_url[:140]}")
        _verify_public_url(req.image_url)
        r = requests.get(req.image_url, timeout=FETCH_TIMEOUT_SECS)
        r.raise_for_status()
        content = r.content
        if not content:
            raise HTTPException(status_code=400, detail="Fetched image content is empty.")
        img = Image.open(io.BytesIO(content)).convert("RGB")
    except HTTPException:
        raise
    except requests.exceptions.Timeout:
        raise HTTPException(status_code=408, detail="Timeout fetching image URL")
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=400, detail=f"Failed to fetch image. {e}")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image data")

    # B. Pad
    padded_img, meta = pad_to_multiple_reflect(img, TILE_SIZE)

    # C. Choose tiling or full pass
    total_pixels = padded_img.width * padded_img.height
    if total_pixels <= MAX_PIXELS_DIRECT:
        out_padded = inference_no_tiling(netG, padded_img)
    else:
        out_padded = inference_tiled_blend(netG, padded_img, TILE_SIZE, TILE_OVERLAP)

    # D. Restore original dimensions
    restored = restore_to_original(out_padded, meta)

    # E. Encode
    fmt = (req.output_format or DEFAULT_FORMAT).upper()
    buf = io.BytesIO()
    if fmt == "JPEG":
        q = int(req.jpeg_quality or JPEG_QUALITY)
        q = min(max(q, 1), 100)
        restored.save(buf, format="JPEG", quality=q, subsampling=0, optimize=True)
        mime = "image/jpeg"
    else:
        restored.save(buf, format="PNG", compress_level=3)
        mime = "image/png"

    data_url = f"data:{mime};base64,{base64.b64encode(buf.getvalue()).decode('utf-8')}"
    return JSONResponse(content={"editedImageBase64": data_url})

@app.post("/process_bytes")
async def process_bytes(req: ImageBytesRequest):
    """Pass image bytes as base64 to avoid remote fetch and 502s from cross cloud access."""
    if netG is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        b64 = req.image_base64
        if b64.startswith("data:"):
            b64 = b64.split(",", 1)[1]
        raw = base64.b64decode(b64, validate=True)
        if len(raw) > MAX_BODY_BYTES:
            raise HTTPException(status_code=413, detail="Image too large")
        img = Image.open(io.BytesIO(raw)).convert("RGB")
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid base64 image")

    padded_img, meta = pad_to_multiple_reflect(img, TILE_SIZE)

    total_pixels = padded_img.width * padded_img.height
    if total_pixels <= MAX_PIXELS_DIRECT:
        out_padded = inference_no_tiling(netG, padded_img)
    else:
        out_padded = inference_tiled_blend(netG, padded_img, TILE_SIZE, TILE_OVERLAP)

    restored = restore_to_original(out_padded, meta)

    fmt = (req.output_format or DEFAULT_FORMAT).upper()
    buf = io.BytesIO()
    if fmt == "JPEG":
        q = int(req.jpeg_quality or JPEG_QUALITY)
        q = min(max(q, 1), 100)
        restored.save(buf, format="JPEG", quality=q, subsampling=0, optimize=True)
        mime = "image/jpeg"
    else:
        restored.save(buf, format="PNG", compress_level=3)
        mime = "image/png"

    data_url = f"data:{mime};base64,{base64.b64encode(buf.getvalue()).decode('utf-8')}"
    return JSONResponse(content={"editedImageBase64": data_url})
