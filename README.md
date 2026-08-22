# Satellite Imagery Change Detection

A production-ready deep learning pipeline for detecting land-use and land-cover changes from multi-temporal Sentinel-2 satellite imagery. Built with PyTorch, this project implements a Siamese U-Net architecture that compares bi-temporal image pairs to produce pixel-level change maps -- from training to deployment.

[![CI](https://github.com/neuralnomad7/satellite-change-detection/actions/workflows/ci.yml/badge.svg)](https://github.com/neuralnomad7/satellite-change-detection/actions)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/pipeline-animation.svg">
    <source media="(prefers-color-scheme: light)" srcset="assets/pipeline-animation.svg">
    <img src="assets/pipeline-animation.svg" alt="End-to-end pipeline: satellite input → Siamese encoder → feature fusion → decoder → deployment" width="100%">
  </picture>
</p>

## System Overview

```mermaid
graph LR
    subgraph DATA ["Data Pipeline"]
        S2["Sentinel-2\nImagery"] --> PP["Preprocessing\n& Augmentation"]
        PP --> PAIRS["Bi-Temporal\nImage Pairs"]
    end

    subgraph TRAIN ["Training"]
        PAIRS --> TR["Training Loop\nAMP · Deep Supervision\nEarly Stopping"]
        TR --> CKPT["Model\nCheckpoint"]
    end

    subgraph DEPLOY ["Deployment"]
        CKPT --> ONNX["ONNX Export\n+ Benchmark"]
        CKPT --> API["FastAPI\nServing"]
        ONNX --> API
        API --> DOCKER["Docker\nContainer"]
    end

    subgraph OUTPUT ["Output"]
        API --> MASK["Change\nMask"]
        API --> DEMO["Streamlit\nDemo"]
    end

    style DATA fill:#e3f2fd,stroke:#1565c0,color:#000
    style TRAIN fill:#e8f5e9,stroke:#2e7d32,color:#000
    style DEPLOY fill:#fff3e0,stroke:#e65100,color:#000
    style OUTPUT fill:#f3e5f5,stroke:#6a1b9a,color:#000
```

## Model Architecture

```mermaid
graph TB
    subgraph INPUT ["Input"]
        T1["Pre-Change Image\n(T1)"]
        T2["Post-Change Image\n(T2)"]
    end

    subgraph ENCODER ["Shared ResNet-34 Encoder (Siamese)"]
        direction TB
        T1 --> E1_0["Stage 0\n64ch · H/2"]
        T2 --> E2_0["Stage 0\n64ch · H/2"]
        E1_0 --> E1_1["Stage 1\n64ch · H/4"]
        E2_0 --> E2_1["Stage 1\n64ch · H/4"]
        E1_1 --> E1_2["Stage 2\n128ch · H/8"]
        E2_1 --> E2_2["Stage 2\n128ch · H/8"]
        E1_2 --> E1_3["Stage 3\n256ch · H/16"]
        E2_2 --> E2_3["Stage 3\n256ch · H/16"]
        E1_3 --> E1_4["Stage 4\n512ch · H/32"]
        E2_3 --> E2_4["Stage 4\n512ch · H/32"]
    end

    subgraph FUSION ["Feature Fusion (|F1-F2| + [F1,F2])"]
        E1_4 & E2_4 --> F4["Fused\n512ch"]
        E1_3 & E2_3 --> F3["Fused\n256ch"]
        E1_2 & E2_2 --> F2["Fused\n128ch"]
        E1_1 & E2_1 --> F1["Fused\n64ch"]
        E1_0 & E2_0 --> F0["Fused\n64ch"]
    end

    subgraph DECODER ["U-Net Decoder + Deep Supervision"]
        F4 --> D4["Decoder 4\n256ch"]
        D4 --> D3["Decoder 3\n128ch"]
        D3 --> D2["Decoder 2\n64ch"]
        D2 --> D1["Decoder 1\n64ch"]
        D1 --> UP["Upsample\nto H×W"]
        F3 -.->|skip| D4
        F2 -.->|skip| D3
        F1 -.->|skip| D2
        F0 -.->|skip| D1
    end

    UP --> CLS["1×1 Conv\nSigmoid"]
    CLS --> OUT["Binary Change\nMask (H×W)"]

    style INPUT fill:#e3f2fd,stroke:#1565c0,color:#000
    style ENCODER fill:#e8f5e9,stroke:#2e7d32,color:#000
    style FUSION fill:#fff3e0,stroke:#e65100,color:#000
    style DECODER fill:#f3e5f5,stroke:#6a1b9a,color:#000
    style OUT fill:#ffebee,stroke:#c62828,color:#000
```

## Training & Optimization Pipeline

```mermaid
graph LR
    subgraph LOOP ["Training Loop"]
        direction TB
        FW["Forward Pass\n(FP16 Mixed Precision)"] --> LOSS["BCEDice Loss\n+ Deep Supervision\n(4 aux heads)"]
        LOSS --> BW["Backward Pass\nGradient Scaling"]
        BW --> CLIP["Gradient Clipping\n(max_norm=1.0)"]
        CLIP --> OPT["AdamW Optimizer\nlr=1e-4 · wd=1e-4"]
        OPT --> SCHED["Cosine Annealing\nWarm Restarts"]
    end

    subgraph MONITOR ["Monitoring"]
        SCHED --> TB["TensorBoard\nLoss · F1 · LR"]
        SCHED --> ES["Early Stopping\n(patience=15)"]
        ES --> |best F1| SAVE["Checkpoint\nSave"]
    end

    subgraph EXPORT ["Export & Serve"]
        SAVE --> ONNX["ONNX Export\nOpset 17 · Dynamic Axes"]
        SAVE --> PYTORCH["PyTorch\nServing"]
        ONNX --> BENCH["Benchmark\nP50 · P95 · P99"]
        PYTORCH --> FAPI["FastAPI\n:8000"]
        ONNX --> FAPI
        FAPI --> DOCK["Docker\nCompose"]
    end

    style LOOP fill:#e8f5e9,stroke:#2e7d32,color:#000
    style MONITOR fill:#e3f2fd,stroke:#1565c0,color:#000
    style EXPORT fill:#fff3e0,stroke:#e65100,color:#000
```

## CI/CD Pipeline

```mermaid
graph LR
    PUSH["git push"] --> CI["GitHub Actions"]

    CI --> LINT["Lint & Format\nRuff · mypy"]
    CI --> TEST["Test Matrix\nPython 3.10 → 3.13\npytest + coverage"]
    CI --> SMOKE["Model Smoke Test\nForward pass shape\nONNX export validation"]

    LINT --> PASS["All Checks Pass"]
    TEST --> PASS
    SMOKE --> PASS

    style PUSH fill:#e3f2fd,stroke:#1565c0,color:#000
    style CI fill:#fff3e0,stroke:#e65100,color:#000
    style PASS fill:#e8f5e9,stroke:#2e7d32,color:#000
```

---

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
source venv/bin/activate        # Windows: venv\Scripts\activate

# Install the package with its runtime dependencies
pip install -e .

# ...or include the dev tooling (Ruff, mypy, pytest, pre-commit)
pip install -e ".[dev]"
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

## Key Design Decisions

| Decision | Choice | Why |
|---|---|---|
| **Architecture** | Siamese U-Net | Weight-sharing ensures both time steps live in the same feature space |
| **Encoder** | ResNet-34 (ImageNet) | Strong transfer learning, adapted for arbitrary spectral bands |
| **Fusion** | \|F1-F2\| + \[F1,F2\] | Captures both *magnitude of change* and *temporal context* |
| **Loss** | BCE + Dice | BCE for stable gradients, Dice for class-imbalanced F1 optimization |
| **Deep Supervision** | 4 auxiliary heads | Gradient signal at every decoder scale during training |
| **Precision** | FP16 Mixed | 2x memory reduction with gradient scaling for stability |
| **Serving** | PyTorch + ONNX dual | Automatic ONNX-RT fallback for 2-5x faster production inference |
| **Sampling** | Weighted oversampling | 2x oversample changed patches to combat 90%+ no-change imbalance |

See [ARCHITECTURE.md](ARCHITECTURE.md) for the full design rationale.

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
  author={NeuralNomad7},
  year={2026},
  url={https://github.com/neuralnomad7/satellite-change-detection}
}
```

## License

MIT License. See [LICENSE](LICENSE) for details.
