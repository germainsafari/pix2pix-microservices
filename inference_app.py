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
from typing import Dict, Tuple, Optional, List

import requests
import numpy as np
from PIL import Image

# Keep CPU stable on small instances
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import torch

warnings.filterwarnings("ignore", category=Image.DecompressionBombWarning)

# =========================
# Configuration
# =========================
MODEL_PATH: str = os.environ.get("MODEL_PATH", "checkpoints/latest_net_G.pth")
MODEL_ARCH: str = os.environ.get("MODEL_ARCH", "unet_256")
NORM_LAYER: str = os.environ.get("NORM_LAYER", "instance")
INPUT_NC: int = int(os.environ.get("INPUT_NC", "3"))
OUTPUT_NC: int = int(os.environ.get("OUTPUT_NC", "3"))
NGF: int = int(os.environ.get("NGF", "64"))

# Quality tiling with batching
TILE_SIZE: int = int(os.environ.get("TILE_SIZE", "512"))
TILE_OVERLAP: int = int(os.environ.get("TILE_OVERLAP", "128"))
TILE_HALO: int = int(os.environ.get("TILE_HALO", "128"))
TILE_BATCH: int = max(1, int(os.environ.get("TILE_BATCH", "8")))  # process this many tiles at once

DEFAULT_FORMAT: str = os.environ.get("DEFAULT_FORMAT", "PNG")
JPEG_QUALITY: int = int(os.environ.get("JPEG_QUALITY", "95"))

# Optional full image path for small inputs
MAX_PIXELS_DIRECT: int = int(os.environ.get("MAX_PIXELS_DIRECT", str(2048 * 2048)))

# Networking
FETCH_TIMEOUT_SECS: int = int(os.environ.get("FETCH_TIMEOUT_SECS", "25"))
HEAD_TIMEOUT_SECS: int = int(os.environ.get("HEAD_TIMEOUT_SECS", "6"))
MAX_BODY_BYTES: int = int(os.environ.get("MAX_BODY_BYTES", str(25 * 1024 * 1024)))  # 25 MB

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_num_threads(int(os.environ.get("TORCH_NUM_THREADS", "1")))

# Model input stride so encoder and decoder align
if MODEL_ARCH.endswith("256"):
    MODEL_STRIDE = 256
elif MODEL_ARCH.endswith("128"):
    MODEL_STRIDE = 128
elif MODEL_ARCH.endswith("64"):
    MODEL_STRIDE = 64
else:
    MODEL_STRIDE = 256

# Logger
logger = logging.getLogger("pix2pix")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO)
logger.setLevel(logging.INFO)

# =========================
# Import model factory
# =========================
try:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if current_dir not in sys.path:
        sys.path.append(current_dir)
    from models.networks import define_G
except Exception as e:
    print(f"Failed to import define_G from models.networks: {e}")
    raise

# =========================
# FastAPI app
# =========================
app = FastAPI(title="Pix2Pix Inference Service", version="3.2-batched")

class ImageRequest(BaseModel):
    image_url: str
    output_format: Optional[str] = None
    jpeg_quality: Optional[int] = None

class ImageBytesRequest(BaseModel):
    image_base64: str
    output_format: Optional[str] = None
    jpeg_quality: Optional[int] = None

netG = None

# =========================
# Helpers
# =========================
def _weight_checksum(state_dict: Dict[str, torch.Tensor]) -> str:
    m = hashlib.md5()
    for k in sorted(state_dict.keys()):
        t = state_dict[k]
        m.update(k.encode("utf-8"))
        m.update(t.cpu().numpy().tobytes())
    return m.hexdigest()[:12]

def _hann_window_2d(h: int, w: int) -> np.ndarray:
    if h <= 1 or w <= 1:
        return np.ones((h, w), dtype=np.float32)
    y = np.linspace(0.0, np.pi, h, dtype=np.float32)
    x = np.linspace(0.0, np.pi, w, dtype=np.float32)
    wy, wx = np.sin(y) ** 2, np.sin(x) ** 2
    return np.sqrt(np.outer(wy, wx)).astype(np.float32)

def pad_to_multiple_reflect(img: Image.Image, block: int) -> Tuple[Image.Image, Dict[str, int]]:
    """Pad PIL image so width and height are multiples of block using reflect mode."""
    w, h = img.size
    new_w = math.ceil(w / block) * block
    new_h = math.ceil(h / block) * block
    if new_w == w and new_h == h:
        return img, {"orig_w": w, "orig_h": h, "pad_left": 0, "pad_top": 0, "new_w": w, "new_h": h}
    arr = np.array(img)
    pad_left = (new_w - w) // 2
    pad_right = new_w - w - pad_left
    pad_top = (new_h - h) // 2
    pad_bottom = new_h - h - pad_top
    arr_pad = np.pad(arr, ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)), mode="reflect")
    padded = Image.fromarray(arr_pad, mode="RGB")
    meta = {"orig_w": w, "orig_h": h, "pad_left": pad_left, "pad_top": pad_top, "new_w": new_w, "new_h": new_h}
    return padded, meta

def pad_np_to_multiple_reflect(arr: np.ndarray, mult: int) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    """Reflect-pad HxWxC array so H and W are multiples of mult. Returns padded array and (top, bottom, left, right)."""
    H, W, C = arr.shape
    new_h = ((H + mult - 1) // mult) * mult
    new_w = ((W + mult - 1) // mult) * mult
    pad_top = (new_h - H) // 2
    pad_bottom = new_h - H - pad_top
    pad_left = (new_w - W) // 2
    pad_right = new_w - W - pad_left
    if pad_top or pad_bottom or pad_left or pad_right:
        arr = np.pad(arr, ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)), mode="reflect")
    return arr, (pad_top, pad_bottom, pad_left, pad_right)

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

# Fast NumPy -> Torch normalize to [-1, 1]
def _np_to_input_tensor(arr_rgb_uint8: np.ndarray) -> torch.Tensor:
    # arr: HxWx3 uint8
    t = torch.from_numpy(arr_rgb_uint8).permute(2, 0, 1).to(torch.float32)  # CHW
    t = t / 255.0
    t = t * 2.0 - 1.0
    return t

def _tensor_to_uint8_image(t_chw_01: torch.Tensor) -> np.ndarray:
    # clamp [0,1], convert to uint8 HxWx3
    t = torch.clamp(t_chw_01, 0.0, 1.0)
    t = (t * 255.0 + 0.5).to(torch.uint8)
    return t.permute(1, 2, 0).cpu().numpy()

def run_model_on_tensor(model: torch.nn.Module, tensor_bchw: torch.Tensor) -> torch.Tensor:
    """Forward pass. Returns BxCxHxW in [0,1] on CPU."""
    with torch.no_grad():
        out = model(tensor_bchw.to(DEVICE))
        # model may return tensor or list; handle both
        if isinstance(out, (list, tuple)):
            out = out[0]
        out = out.detach().cpu()
        out = (out + 1.0) / 2.0
        out = torch.clamp(out, 0.0, 1.0)
    return out  # BxCxHxW

def inference_no_tiling_stride_safe(model: torch.nn.Module, img: Image.Image) -> Image.Image:
    arr = np.array(img)
    arr_pad, (pt, pb, pl, pr) = pad_np_to_multiple_reflect(arr, MODEL_STRIDE)
    inp = _np_to_input_tensor(arr_pad).unsqueeze(0)
    out_full = run_model_on_tensor(model, inp)[0]  # CxHxW
    out_unpad = out_full[:, pt:pt+arr.shape[0], pl:pl+arr.shape[1]]
    out_np = _tensor_to_uint8_image(out_unpad)
    return Image.fromarray(out_np, mode="RGB")

# =========================
# Batched context aware tiling
# Requires input padded so each tile = TILE_SIZE and same halo crop size everywhere
# =========================
def inference_tiled_halo_batched(model: torch.nn.Module, img: Image.Image,
                                 tile: int, overlap: int, halo: int, batch_sz: int) -> Image.Image:
    W, H = img.size
    base = np.array(img)  # HxWx3

    # Precompute grid since image is padded to multiple of TILE_SIZE
    xs = list(range(0, W, tile - overlap))
    ys = list(range(0, H, tile - overlap))
    if xs[-1] + tile > W:
        xs[-1] = W - tile
    if ys[-1] + tile > H:
        ys[-1] = H - tile

    # Precompute feather window once
    win = _hann_window_2d(tile, tile)[..., None].astype(np.float32)

    acc = np.zeros((H, W, 3), dtype=np.float32)
    wacc = np.zeros((H, W, 1), dtype=np.float32)

    # Helper to reflect-crop from base
    def crop_with_reflect(arr, x0, y0, x1, y1):
        pad_left = max(0, -x0)
        pad_top = max(0, -y0)
        pad_right = max(0, x1 - W)
        pad_bottom = max(0, y1 - H)
        x0c, y0c = max(0, x0), max(0, y0)
        x1c, y1c = min(W, x1), min(H, y1)
        crop = arr[y0c:y1c, x0c:x1c, :]
        if pad_left or pad_right or pad_top or pad_bottom:
            crop = np.pad(crop, ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)), mode="reflect")
        return crop

    # Collect tiles into batches
    batch_coords: List[Tuple[int, int, int, int]] = []
    batch_inputs: List[torch.Tensor] = []
    ptpbplpr_list: List[Tuple[int, int, int, int]] = []

    for y0 in ys:
        for x0 in xs:
            y1 = y0 + tile
            x1 = x0 + tile

            # Halo crop bounds in source coordinates
            hx0, hy0 = x0 - halo, y0 - halo
            hx1, hy1 = x1 + halo, y1 + halo
            halo_arr = crop_with_reflect(base, hx0, hy0, hx1, hy1)  # shape ~ (tile+2*halo, tile+2*halo, 3)

            # Pad halo to MODEL_STRIDE multiples to keep UNet happy
            halo_arr_padded, (pt, pb, pl, pr) = pad_np_to_multiple_reflect(halo_arr, MODEL_STRIDE)

            # Convert to input tensor and queue
            batch_inputs.append(_np_to_input_tensor(halo_arr_padded))
            ptpbplpr_list.append((pt, pb, pl, pr))
            batch_coords.append((x0, y0, x1, y1))

            # Flush batch
            if len(batch_inputs) == batch_sz:
                _flush_batch(model, batch_inputs, ptpbplpr_list, batch_coords, acc, wacc, win, tile, halo)
                batch_inputs.clear()
                ptpbplpr_list.clear()
                batch_coords.clear()

    # Flush remainder
    if batch_inputs:
        _flush_batch(model, batch_inputs, ptpbplpr_list, batch_coords, acc, wacc, win, tile, halo)

    out = (acc / np.maximum(wacc, 1e-8)).clip(0.0, 1.0)
    return Image.fromarray((out * 255.0 + 0.5).astype(np.uint8), mode="RGB")

def _flush_batch(model: torch.nn.Module,
                 batch_inputs: List[torch.Tensor],
                 pad_info: List[Tuple[int, int, int, int]],
                 coords: List[Tuple[int, int, int, int]],
                 acc: np.ndarray, wacc: np.ndarray, win: np.ndarray,
                 tile: int, halo: int) -> None:
    # Stack to BxCxHxW
    inp = torch.stack(batch_inputs, dim=0)  # float, [-1,1] already
    out_full = run_model_on_tensor(model, inp)  # BxCxHxW in [0,1]

    for i in range(out_full.shape[0]):
        x0, y0, x1, y1 = coords[i]
        pt, pb, pl, pr = pad_info[i]
        Hh = (out_full.shape[2] - pt - pb)
        Wh = (out_full.shape[3] - pl - pr)
        out_unpad = out_full[i, :, pt:pt+Hh, pl:pl+Wh]  # CxHhWx
        out_tile = out_unpad[:, halo:halo+tile, halo:halo+tile]    # CxTxT

        out_np = _tensor_to_uint8_image(out_tile)  # HxWx3 uint8
        out_f = out_np.astype(np.float32) / 255.0

        wtile = win
        acc[y0:y1, x0:x1, :] += out_f * wtile
        wacc[y0:y1, x0:x1, :] += wtile

# =========================
# Model loading
# =========================
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

# =========================
# Startup and health
# =========================
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

# =========================
# Networking helper
# =========================
def _verify_public_url(url: str) -> None:
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

# =========================
# Utility: decide output format to match input when possible
# =========================
def _pick_output_format(pil_img: Image.Image, requested: Optional[str]) -> Tuple[str, str]:
    """
    Returns (format_name, mime). Priority:
    1) requested if valid
    2) same as input if input format is one of PNG or JPEG
    3) PNG fallback
    """
    valid = {"PNG": "image/png", "JPEG": "image/jpeg"}
    if requested:
        fmt = requested.upper()
        if fmt in valid:
            return fmt, valid[fmt]
    in_fmt = (pil_img.format or "").upper()
    if in_fmt in valid:
        return in_fmt, valid[in_fmt]
    return "PNG", "image/png"

# =========================
# Endpoints
# =========================
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
        img = Image.open(io.BytesIO(content))
        img = img.convert("RGB")  # unify
        logger.info(f"Image opened size=({img.width},{img.height}) format={img.format}")
    except HTTPException:
        raise
    except requests.exceptions.Timeout:
        raise HTTPException(status_code=408, detail="Timeout fetching image URL")
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=400, detail=f"Failed to fetch image. {e}")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image data")

    # B. Pad to TILE_SIZE multiple so tiles are uniform
    padded_img, meta = pad_to_multiple_reflect(img, TILE_SIZE)
    logger.info(f"Padded to {padded_img.size} | tile={TILE_SIZE} overlap={TILE_OVERLAP} halo={TILE_HALO} batch={TILE_BATCH}")

    # C. Full pass for tiny images or batched halo tiling for everything else
    total_pixels = padded_img.width * padded_img.height
    if total_pixels <= MAX_PIXELS_DIRECT:
        out_padded = inference_no_tiling_stride_safe(netG, padded_img)
    else:
        out_padded = inference_tiled_halo_batched(netG, padded_img, TILE_SIZE, TILE_OVERLAP, TILE_HALO, TILE_BATCH)

    # D. Restore original size
    restored = restore_to_original(out_padded, meta)

    # E. Encode in same format if possible or requested override
    fmt, mime = _pick_output_format(img, req.output_format)
    buf = io.BytesIO()
    if fmt == "JPEG":
        q = int(req.jpeg_quality or JPEG_QUALITY)
        q = min(max(q, 1), 100)
        restored.save(buf, format="JPEG", quality=q, subsampling=0, optimize=True)
    else:
        restored.save(buf, format="PNG", compress_level=3)

    data_url = f"data:{mime};base64,{base64.b64encode(buf.getvalue()).decode('utf-8')}"
    return JSONResponse(content={"editedImageBase64": data_url})

@app.post("/process_bytes")
async def process_bytes(req: ImageBytesRequest):
    if netG is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        b64 = req.image_base64
        if b64.startswith("data:"):
            b64 = b64.split(",", 1)[1]
        raw = base64.b64decode(b64, validate=True)
        if len(raw) > MAX_BODY_BYTES:
            raise HTTPException(status_code=413, detail="Image too large")
        src = Image.open(io.BytesIO(raw))
        img = src.convert("RGB")
        logger.info(f"Received bytes image size=({img.width},{img.height}) format={src.format}")
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid base64 image")

    padded_img, meta = pad_to_multiple_reflect(img, TILE_SIZE)
    out_padded = inference_tiled_halo_batched(netG, padded_img, TILE_SIZE, TILE_OVERLAP, TILE_HALO, TILE_BATCH)
    restored = restore_to_original(out_padded, meta)

    fmt, mime = _pick_output_format(src, req.output_format)
    buf = io.BytesIO()
    if fmt == "JPEG":
        q = int(req.jpeg_quality or JPEG_QUALITY)
        q = min(max(q, 1), 100)
        restored.save(buf, format="JPEG", quality=q, subsampling=0, optimize=True)
    else:
        restored.save(buf, format="PNG", compress_level=3)

    data_url = f"data:{mime};base64,{base64.b64encode(buf.getvalue()).decode('utf-8')}"
    return JSONResponse(content={"editedImageBase64": data_url})
