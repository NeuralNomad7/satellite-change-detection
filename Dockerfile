FROM python:3.12-slim

WORKDIR /app

# System dependencies for image processing. libgl1-mesa-glx was dropped from
# Debian in bookworm; libgl1 replaces it. GDAL ships inside the rasterio wheels,
# so no system GDAL is needed.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies (serving and ONNX packages are in requirements.txt)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ src/
COPY serving/ serving/
COPY configs/ configs/

# Copy model checkpoint if available
COPY models/ models/

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "serving.app:app", "--host", "0.0.0.0", "--port", "8000"]
