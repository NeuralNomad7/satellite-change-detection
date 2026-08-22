"""FastAPI serving endpoint for satellite change detection inference.

Provides a production-ready REST API for running change detection on
bi-temporal satellite image pairs. Supports both PyTorch and ONNX backends.

Usage:
    uvicorn serving.app:app --host 0.0.0.0 --port 8000

Endpoints:
    POST /predict          - Upload two images, get a binary change mask
    POST /predict/geotiff  - Upload two GeoTIFFs, get GeoJSON change polygons
    GET  /health           - Health check
    GET  /model/info       - Model metadata and configuration
"""

from __future__ import annotations

import io
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

import numpy as np
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse
from PIL import Image

from src.config import Config
from src.geo import (
    change_statistics,
    mask_to_polygons,
    normalize_imagenet,
    read_geotiff_bytes,
)
from src.models import build_model, count_parameters

# ---------------------------------------------------------------------------
# Global model state
# ---------------------------------------------------------------------------

MODEL_STATE: dict = {
    "model": None,
    "device": None,
    "config": None,
    "backend": "pytorch",
    "onnx_session": None,
}

CHECKPOINT_PATH = Path("models/checkpoints/best_model.pth")
ONNX_PATH = Path("models/exports/model.onnx")


def _load_pytorch_model(config: Config, device: torch.device) -> torch.nn.Module:
    """Load PyTorch model from checkpoint or create with random weights."""
    model = build_model(
        in_channels=config.model.in_channels,
        num_classes=config.model.num_classes,
        pretrained=False,
        fusion_mode=config.model.fusion,
        deep_supervision=False,
    ).to(device)

    if CHECKPOINT_PATH.exists():
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        print(f"Loaded checkpoint from {CHECKPOINT_PATH}")
    else:
        print("No checkpoint found, using randomly initialized weights (demo mode)")

    model.eval()
    return model


def _load_onnx_model():
    """Load ONNX model if available."""
    if ONNX_PATH.exists():
        try:
            import onnxruntime as ort

            session = ort.InferenceSession(
                str(ONNX_PATH),
                providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
            )
            print(f"Loaded ONNX model from {ONNX_PATH}")
            return session
        except ImportError:
            print("onnxruntime not installed, falling back to PyTorch")
    return None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize model on startup, clean up on shutdown."""
    config = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    MODEL_STATE["config"] = config
    MODEL_STATE["device"] = device

    # Try ONNX first (faster inference), fall back to PyTorch
    onnx_session = _load_onnx_model()
    if onnx_session:
        MODEL_STATE["onnx_session"] = onnx_session
        MODEL_STATE["backend"] = "onnx"
    else:
        MODEL_STATE["model"] = _load_pytorch_model(config, device)
        MODEL_STATE["backend"] = "pytorch"

    print(f"Serving with backend: {MODEL_STATE['backend']} on {device}")
    yield
    # Cleanup
    MODEL_STATE["model"] = None
    MODEL_STATE["onnx_session"] = None


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Satellite Change Detection API",
    description=(
        "Deep learning API for detecting land-use changes from bi-temporal "
        "satellite imagery. Upload a pre-change and post-change image to "
        "receive a binary change mask."
    ),
    version="1.2.0",
    lifespan=lifespan,
)


def _preprocess_image(image_bytes: bytes) -> np.ndarray:
    """Load and preprocess an uploaded image to model-ready tensor format.

    Applies ImageNet normalization to match training preprocessing.

    Returns:
        Numpy array of shape (1, 3, H, W), float32.
    """
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img_np = np.array(img, dtype=np.float32)

    # ImageNet normalization (matches training pipeline)
    mean = np.array([0.485, 0.456, 0.406]) * 255.0
    std = np.array([0.229, 0.224, 0.225]) * 255.0
    img_np = (img_np - mean) / std

    # HWC -> NCHW
    img_np = np.transpose(img_np, (2, 0, 1))[np.newaxis, ...]
    return img_np.astype(np.float32)


def _run_inference(img1: np.ndarray, img2: np.ndarray) -> np.ndarray:
    """Run change detection inference using the active backend.

    Returns:
        Binary change mask of shape (H, W) with values in {0, 1}.
    """
    if MODEL_STATE["backend"] == "onnx" and MODEL_STATE["onnx_session"]:
        session = MODEL_STATE["onnx_session"]
        outputs = session.run(None, {"image_t1": img1, "image_t2": img2})
        logits = outputs[0]
    else:
        device = MODEL_STATE["device"]
        model = MODEL_STATE["model"]
        t1 = torch.from_numpy(img1).to(device)
        t2 = torch.from_numpy(img2).to(device)
        with torch.no_grad():
            output = model(t1, t2)
        logits = output["pred"].cpu().numpy()

    # Sigmoid + threshold
    prob = 1.0 / (1.0 + np.exp(-logits))
    mask = (prob > 0.5).astype(np.uint8).squeeze()
    return mask


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "backend": MODEL_STATE["backend"],
        "device": str(MODEL_STATE["device"]),
    }


@app.get("/model/info")
async def model_info():
    """Return model metadata."""
    config = MODEL_STATE["config"]
    info = {
        "architecture": "SiameseUNet",
        "encoder": config.model.encoder,
        "in_channels": config.model.in_channels,
        "fusion_mode": config.model.fusion,
        "backend": MODEL_STATE["backend"],
        "device": str(MODEL_STATE["device"]),
        "checkpoint_loaded": CHECKPOINT_PATH.exists(),
    }
    if MODEL_STATE["model"] is not None:
        info["parameters"] = count_parameters(MODEL_STATE["model"])
    return info


@app.post("/predict")
async def predict(
    image_t1: UploadFile = File(..., description="Pre-change satellite image"),
    image_t2: UploadFile = File(..., description="Post-change satellite image"),
    return_format: str = "png",
):
    """Run change detection on a bi-temporal image pair.

    Upload two satellite images (pre-change and post-change) and receive
    a binary change mask. Supported formats: PNG, JPEG, TIFF.

    Args:
        image_t1: Pre-change image file.
        image_t2: Post-change image file.
        return_format: Response format - "png" for image, "json" for raw mask.

    Returns:
        Binary change mask as PNG image or JSON array.
    """
    try:
        bytes_t1 = await image_t1.read()
        bytes_t2 = await image_t2.read()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error reading uploaded files: {e}")

    try:
        img1 = _preprocess_image(bytes_t1)
        img2 = _preprocess_image(bytes_t2)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error preprocessing images: {e}")

    # Validate dimensions match
    if img1.shape != img2.shape:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Image dimensions must match. "
                f"Got T1: {img1.shape[2:]} vs T2: {img2.shape[2:]}"
            ),
        )

    start = time.time()
    mask = _run_inference(img1, img2)
    inference_time = time.time() - start

    if return_format == "json":
        return JSONResponse({
            "change_mask": mask.tolist(),
            "shape": list(mask.shape),
            "inference_time_ms": round(inference_time * 1000, 2),
            "changed_pixels": int(mask.sum()),
            "total_pixels": int(mask.size),
            "change_fraction": round(float(mask.mean()), 4),
        })

    # Return as PNG image
    mask_img = Image.fromarray(mask * 255)
    buf = io.BytesIO()
    mask_img.save(buf, format="PNG")
    buf.seek(0)

    return StreamingResponse(
        buf,
        media_type="image/png",
        headers={
            "X-Inference-Time-Ms": str(round(inference_time * 1000, 2)),
            "X-Changed-Pixels": str(int(mask.sum())),
            "X-Change-Fraction": str(round(float(mask.mean()), 4)),
        },
    )


@app.post("/predict/geotiff")
async def predict_geotiff(
    image_t1: Annotated[UploadFile, File(description="Pre-change GeoTIFF")],
    image_t2: Annotated[UploadFile, File(description="Post-change GeoTIFF")],
    min_area_m2: float = 0.0,
):
    """Run change detection on a co-registered GeoTIFF pair.

    Unlike ``/predict`` (which returns a plain PNG mask), this endpoint preserves
    geospatial metadata and returns georeferenced results: a GeoJSON
    FeatureCollection of change polygons in WGS84 lon/lat -- each annotated with
    its ground area in hectares and centroid -- plus a change-statistics summary
    (changed area, fraction, and region count).

    Args:
        image_t1: Pre-change GeoTIFF (with CRS + transform).
        image_t2: Post-change GeoTIFF, co-registered to T1.
        min_area_m2: Drop change regions smaller than this many square metres.

    Returns:
        JSON with ``statistics`` and a ``geojson`` FeatureCollection.
    """
    try:
        bytes_t1 = await image_t1.read()
        bytes_t2 = await image_t2.read()
    except Exception as e:
        raise HTTPException(
            status_code=400, detail=f"Error reading uploaded files: {e}"
        ) from e

    try:
        raster1 = read_geotiff_bytes(bytes_t1)
        raster2 = read_geotiff_bytes(bytes_t2)
    except Exception as e:
        raise HTTPException(
            status_code=400, detail=f"Error reading GeoTIFFs: {e}"
        ) from e

    if raster1.data.shape[:2] != raster2.data.shape[:2]:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Image dimensions must match. "
                f"Got T1: {raster1.data.shape[:2]} vs T2: {raster2.data.shape[:2]}"
            ),
        )

    config = MODEL_STATE["config"]
    in_channels = config.model.in_channels
    try:
        img1 = normalize_imagenet(raster1.data, in_channels)
        img2 = normalize_imagenet(raster2.data, in_channels)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    start = time.time()
    mask = _run_inference(img1, img2)
    inference_time = time.time() - start

    polygons = mask_to_polygons(
        mask, raster1.transform, raster1.crs, min_area_m2=min_area_m2
    )
    stats = change_statistics(mask, raster1.transform, raster1.crs, polygons=polygons)

    return JSONResponse(
        {
            "statistics": stats,
            "geojson": polygons,
            "inference_time_ms": round(inference_time * 1000, 2),
        }
    )
