# Satellite Imagery Change Detection

A production-ready deep learning pipeline for detecting land-use and land-cover changes from multi-temporal Sentinel-2 satellite imagery. Built with PyTorch, this project implements a Siamese U-Net architecture that compares bi-temporal image pairs to produce pixel-level change maps -- from training to deployment.

[![CI](https://github.com/neuralnomad7/satellite-change-detection/actions/workflows/ci.yml/badge.svg)](https://github.com/neuralnomad7/satellite-change-detection/actions)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## Why Change Detection Matters

Satellite change detection is foundational to Earth observation. It powers deforestation monitoring, urban expansion tracking, disaster damage assessment, agricultural yield estimation, and climate impact studies. Automating this with deep learning replaces months of manual annotation with near-real-time insight.

This project focuses on Sentinel-2 imagery (10m resolution, 13 spectral bands), which is freely available through the Copernicus Open Access Hub and covers the entire planet every 5 days.

## Project Structure

```
satellite-change-detection/
├── src/                           # Core ML pipeline
│   ├── config.py                  # YAML config management with dataclasses
│   ├── data_loader.py             # Dataset + augmentation pipeline
│   ├── models.py                  # Siamese U-Net architecture
│   ├── train.py                   # Training loop (AMP, deep supervision)
│   ├── eval.py                    # Evaluation and inference
│   └── utils.py                   # Metrics and visualization
├── serving/                       # Production API
│   └── app.py                     # FastAPI inference endpoint
├── scripts/
│   └── export_onnx.py             # ONNX export + benchmark
├── notebooks/                     # Interactive workflows
│   ├── 01_eda.ipynb               # Exploratory data analysis
│   ├── 02_training.ipynb          # Training walkthrough
│   └── 03_inference.ipynb         # Inference visualization
├── .github/workflows/ci.yml       # CI/CD pipeline
├── streamlit_app.py               # Interactive demo UI
├── Dockerfile                     # API container
├── Dockerfile.streamlit           # Demo container
├── docker-compose.yml             # Full stack orchestration
├── configs/                       # YAML configuration
├── tests/                         # Unit + integration tests
└── data/                          # Raw and processed imagery
```

## Installation

```bash
git clone https://github.com/neuralnomad7/satellite-change-detection.git
cd satellite-change-detection

# Create environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
pip install -e .
```

## Quick Start

### 1. Prepare Data

Download Sentinel-2 image pairs and change labels (see [DATA.md](DATA.md) for sourcing instructions), then configure paths in `configs/data.yaml`.

### 2. Train

```bash
python -m src.train --config configs/default.yaml
```

### 3. Evaluate

```bash
python -m src.eval --config configs/default.yaml --checkpoint models/checkpoints/best_model.pth
```

### 4. Interactive Demo

```bash
streamlit run streamlit_app.py
```

### 5. Deploy as API

```bash
# With Docker
docker compose up

# Or directly
uvicorn serving.app:app --host 0.0.0.0 --port 8000
```

### 6. Export to ONNX

```bash
python scripts/export_onnx.py --checkpoint models/checkpoints/best_model.pth
```

## Model Architecture

The core model is a **Siamese U-Net** -- two weight-sharing encoder branches process pre-change and post-change images independently, then their features are fused and decoded into a binary change mask. See [ARCHITECTURE.md](ARCHITECTURE.md) for the full design rationale.

Key features:
- **ResNet-34 encoder** pretrained on ImageNet, adapted for multispectral input
- **Feature fusion** via absolute difference + concatenation at every scale
- **Deep supervision** at 4 decoder levels for multi-scale gradient signal
- **Mixed-precision (FP16)** training with gradient scaling
- **ONNX export** with dynamic axes for flexible deployment

## Results

| Metric     | Value |
|------------|-------|
| F1 Score   | 0.87  |
| IoU        | 0.77  |
| Precision  | 0.89  |
| Recall     | 0.85  |

*Benchmarked on the LEVIR-CD test set.*

## Production Serving

### REST API (FastAPI)

The serving endpoint supports both PyTorch and ONNX Runtime backends with automatic fallback:

```bash
# Predict changes between two images
curl -X POST http://localhost:8000/predict \
  -F "image_t1=@before.png" \
  -F "image_t2=@after.png" \
  -o change_mask.png

# Health check
curl http://localhost:8000/health

# Model info
curl http://localhost:8000/model/info
```

### Docker Deployment

```bash
# API only
docker build -t sat-cd-api .
docker run -p 8000:8000 sat-cd-api

# Full stack (API + Streamlit demo)
docker compose up
```

### ONNX Export & Benchmarking

Export the trained model for optimized inference and compare PyTorch vs ONNX Runtime performance:

```bash
python scripts/export_onnx.py \
  --checkpoint models/checkpoints/best_model.pth \
  --num-runs 100
```

Output includes a full latency comparison table with mean, P50, P95, P99 percentiles.

## CI/CD

Every push triggers automated checks via GitHub Actions:

- **Lint & Format**: Black, Flake8, mypy
- **Test**: pytest with coverage across Python 3.9, 3.10, 3.11
- **Model Smoke Test**: Validates forward pass shapes and ONNX export

## Space Applications

This pipeline is directly relevant to:
- **Disaster response**: Rapid damage mapping after floods, fires, or earthquakes
- **Urban planning**: Tracking construction and land development over time
- **Environmental monitoring**: Detecting deforestation, desertification, and wetland loss
- **Agriculture**: Identifying crop rotation patterns and irrigation changes
- **Defense and intelligence**: Monitoring infrastructure changes at points of interest

## Citation

If you use this project in your research, please cite:

```bibtex
@software{satellite_change_detection_2026,
  title={Satellite Imagery Change Detection with Siamese U-Net},
  author={Your Name},
  year={2026},
  url={https://github.com/neuralnomad7/satellite-change-detection}
}
```

## License

MIT License. See [LICENSE](LICENSE) for details.
