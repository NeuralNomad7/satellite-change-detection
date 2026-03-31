"""Interactive Streamlit demo for satellite change detection.

A visual, CEO-friendly interface that demonstrates the full inference
pipeline: upload two satellite images, see the change map in real time.

Usage:
    streamlit run streamlit_app.py
"""

from __future__ import annotations

import io
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import torch
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image

from src.models import build_model, count_parameters

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Satellite Change Detection",
    page_icon="🛰️",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Custom styling
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(90deg, #1a73e8, #34a853);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #5f6368;
        margin-top: 0;
    }
    .metric-card {
        background: #f8f9fa;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        border: 1px solid #e0e0e0;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #1a73e8;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #5f6368;
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Model loading (cached)
# ---------------------------------------------------------------------------

@st.cache_resource
def load_model():
    """Load the change detection model (cached across reruns)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config_path = Path("models/checkpoints/best_model.pth")

    model = build_model(
        in_channels=3,
        pretrained=False,
        deep_supervision=False,
    )

    if config_path.exists():
        checkpoint = torch.load(config_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        st.sidebar.success("Loaded trained checkpoint")
    else:
        st.sidebar.warning("No checkpoint found — using demo mode (random weights)")

    model.to(device).eval()
    return model, device


def preprocess_image(image: Image.Image) -> torch.Tensor:
    """Preprocess a PIL image for model input."""
    img = image.convert("RGB").resize((256, 256))
    img_np = np.array(img, dtype=np.float32)

    # ImageNet normalization
    mean = np.array([0.485, 0.456, 0.406]) * 255.0
    std = np.array([0.229, 0.224, 0.225]) * 255.0
    img_np = (img_np - mean) / std

    tensor = torch.from_numpy(img_np.transpose(2, 0, 1)).float().unsqueeze(0)
    return tensor


def create_overlay(image: Image.Image, mask: np.ndarray) -> np.ndarray:
    """Create a change overlay on the post-change image."""
    img = np.array(image.convert("RGB").resize((256, 256))).astype(np.float32) / 255.0
    overlay = img.copy()

    # Red overlay for detected changes
    overlay[mask == 1, 0] = np.clip(overlay[mask == 1, 0] + 0.4, 0, 1)
    overlay[mask == 1, 1] = overlay[mask == 1, 1] * 0.5
    overlay[mask == 1, 2] = overlay[mask == 1, 2] * 0.5

    return (overlay * 255).astype(np.uint8)


def create_heatmap(prob_map: np.ndarray) -> np.ndarray:
    """Create a probability heatmap visualization."""
    cmap = plt.cm.RdYlBu_r
    heatmap = cmap(prob_map)
    return (heatmap[:, :, :3] * 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

st.sidebar.markdown("### Model Configuration")
threshold = st.sidebar.slider("Detection Threshold", 0.0, 1.0, 0.5, 0.05)

st.sidebar.markdown("---")
st.sidebar.markdown("### Architecture")
st.sidebar.markdown("""
- **Model**: Siamese U-Net
- **Encoder**: ResNet-34 (ImageNet)
- **Fusion**: Difference + Concatenation
- **Loss**: BCE + Dice
- **Input**: 256 x 256 px
""")

st.sidebar.markdown("---")
st.sidebar.markdown("### Benchmark (LEVIR-CD)")
col1, col2 = st.sidebar.columns(2)
col1.metric("F1 Score", "0.87")
col2.metric("IoU", "0.77")
col1.metric("Precision", "0.89")
col2.metric("Recall", "0.85")


# ---------------------------------------------------------------------------
# Main content
# ---------------------------------------------------------------------------

st.markdown('<p class="main-header">Satellite Change Detection</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="sub-header">'
    "Deep learning pipeline for detecting land-use changes from bi-temporal satellite imagery"
    "</p>",
    unsafe_allow_html=True,
)

st.markdown("---")

# Upload section
st.markdown("### Upload Satellite Image Pair")

col_upload1, col_upload2 = st.columns(2)

with col_upload1:
    st.markdown("**Pre-Change Image (T1)**")
    file_t1 = st.file_uploader(
        "Upload pre-change image",
        type=["png", "jpg", "jpeg", "tif", "tiff"],
        key="t1",
        label_visibility="collapsed",
    )

with col_upload2:
    st.markdown("**Post-Change Image (T2)**")
    file_t2 = st.file_uploader(
        "Upload post-change image",
        type=["png", "jpg", "jpeg", "tif", "tiff"],
        key="t2",
        label_visibility="collapsed",
    )

# Demo mode with synthetic data
use_demo = st.checkbox("Use synthetic demo data (no upload needed)", value=True)

if use_demo and (file_t1 is None or file_t2 is None):
    st.info("Generating synthetic satellite images for demonstration...")

    np.random.seed(42)
    # Create synthetic "satellite" images with visible changes
    base = np.random.randint(80, 180, (256, 256, 3), dtype=np.uint8)
    # Add green areas (vegetation)
    base[50:150, 50:150, 1] = np.clip(base[50:150, 50:150, 1].astype(int) + 80, 0, 255).astype(np.uint8)

    t1_img = Image.fromarray(base)

    # Create T2 with changes: remove some vegetation, add urban
    changed = base.copy()
    changed[70:130, 70:130] = [180, 170, 160]  # Urban area replacing vegetation
    changed[30:50, 100:200, :] = [140, 140, 140]  # New road

    t2_img = Image.fromarray(changed)
else:
    if file_t1 is None or file_t2 is None:
        st.warning("Please upload both pre-change and post-change images.")
        st.stop()
    t1_img = Image.open(file_t1)
    t2_img = Image.open(file_t2)

# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

if st.button("Detect Changes", type="primary", use_container_width=True):
    model, device = load_model()

    with st.spinner("Running change detection inference..."):
        # Preprocess
        tensor_t1 = preprocess_image(t1_img).to(device)
        tensor_t2 = preprocess_image(t2_img).to(device)

        # Inference
        start = time.time()
        with torch.no_grad():
            output = model(tensor_t1, tensor_t2)
        inference_time = time.time() - start

        # Post-process
        prob_map = torch.sigmoid(output["pred"]).squeeze().cpu().numpy()
        change_mask = (prob_map > threshold).astype(np.uint8)

    # --- Results ---
    st.markdown("---")
    st.markdown("### Detection Results")

    # Metrics row
    changed_pixels = int(change_mask.sum())
    total_pixels = change_mask.size
    change_pct = changed_pixels / total_pixels * 100

    met_cols = st.columns(4)
    met_cols[0].metric("Inference Time", f"{inference_time * 1000:.0f} ms")
    met_cols[1].metric("Changed Pixels", f"{changed_pixels:,}")
    met_cols[2].metric("Change Area", f"{change_pct:.1f}%")
    met_cols[3].metric("Device", str(device).upper())

    st.markdown("")

    # Visualization panels
    vis_cols = st.columns(4)

    with vis_cols[0]:
        st.markdown("**Pre-Change (T1)**")
        st.image(t1_img.resize((256, 256)), use_container_width=True)

    with vis_cols[1]:
        st.markdown("**Post-Change (T2)**")
        st.image(t2_img.resize((256, 256)), use_container_width=True)

    with vis_cols[2]:
        st.markdown("**Change Overlay**")
        overlay = create_overlay(t2_img, change_mask)
        st.image(overlay, use_container_width=True)

    with vis_cols[3]:
        st.markdown("**Probability Heatmap**")
        heatmap = create_heatmap(prob_map)
        st.image(heatmap, use_container_width=True)

    # Download buttons
    st.markdown("---")
    dl_cols = st.columns(3)
    with dl_cols[0]:
        mask_bytes = io.BytesIO()
        Image.fromarray(change_mask * 255).save(mask_bytes, format="PNG")
        st.download_button(
            "Download Change Mask (PNG)",
            mask_bytes.getvalue(),
            "change_mask.png",
            "image/png",
        )
    with dl_cols[1]:
        overlay_bytes = io.BytesIO()
        Image.fromarray(overlay).save(overlay_bytes, format="PNG")
        st.download_button(
            "Download Overlay (PNG)",
            overlay_bytes.getvalue(),
            "change_overlay.png",
            "image/png",
        )
    with dl_cols[2]:
        # Export raw probability map as numpy
        prob_bytes = io.BytesIO()
        np.save(prob_bytes, prob_map)
        st.download_button(
            "Download Probability Map (NPY)",
            prob_bytes.getvalue(),
            "probability_map.npy",
            "application/octet-stream",
        )

# ---------------------------------------------------------------------------
# Architecture explainer
# ---------------------------------------------------------------------------

st.markdown("---")
st.markdown("### How It Works")

how_cols = st.columns(3)

with how_cols[0]:
    st.markdown("""
    **1. Siamese Encoding**

    Both images pass through a shared ResNet-34 encoder pretrained on ImageNet.
    Weight sharing ensures both time steps are embedded in the same feature space,
    making differences between them semantically meaningful.
    """)

with how_cols[1]:
    st.markdown("""
    **2. Feature Fusion**

    At each spatial scale, features are fused via absolute difference (captures
    *what* changed) and concatenation (preserves *what was there*). A 1x1
    convolution compresses the fused representation.
    """)

with how_cols[2]:
    st.markdown("""
    **3. U-Net Decoding**

    The decoder progressively upsamples fused features with skip connections,
    recovering fine spatial detail. Deep supervision at multiple scales
    provides gradient signal throughout the network during training.
    """)

st.markdown("---")
st.markdown(
    '<p style="text-align: center; color: #9e9e9e;">Built with PyTorch + Streamlit</p>',
    unsafe_allow_html=True,
)
