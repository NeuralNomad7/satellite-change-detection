"""Export the trained model to ONNX format and benchmark inference speed.

ONNX (Open Neural Network Exchange) enables deployment on specialized
inference runtimes (TensorRT, OpenVINO, ONNX Runtime) that are typically
2-5x faster than native PyTorch for production serving.

Usage:
    # Export with default settings
    python scripts/export_onnx.py

    # Export a specific checkpoint
    python scripts/export_onnx.py --checkpoint models/checkpoints/best_model.pth

    # Benchmark only (skip export)
    python scripts/export_onnx.py --benchmark-only --onnx-path models/exports/model.onnx
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import numpy as np
import torch

from src.models import build_model, count_parameters


def export_to_onnx(
    checkpoint_path: str | None = None,
    output_path: str = "models/exports/model.onnx",
    input_size: int = 256,
    in_channels: int = 3,
    opset_version: int = 17,
) -> Path:
    """Export a SiameseUNet model to ONNX format.

    Args:
        checkpoint_path: Path to .pth checkpoint (None for random weights).
        output_path: Destination for the .onnx file.
        input_size: Spatial resolution for the exported model.
        in_channels: Number of input channels.
        opset_version: ONNX opset version.

    Returns:
        Path to the exported ONNX file.
    """
    print(f"Building model (in_channels={in_channels})...")
    model = build_model(
        in_channels=in_channels,
        pretrained=False,
        deep_supervision=False,  # Not needed at inference
    )

    if checkpoint_path and Path(checkpoint_path).exists():
        print(f"Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    else:
        print("No checkpoint specified, exporting with current weights")

    model.eval()
    param_count = count_parameters(model)
    print(f"Model parameters: {param_count:,} ({param_count / 1e6:.1f}M)")

    # Create dummy inputs
    x1 = torch.randn(1, in_channels, input_size, input_size)
    x2 = torch.randn(1, in_channels, input_size, input_size)

    # Export
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    print(f"Exporting to ONNX (opset {opset_version})...")
    torch.onnx.export(
        model,
        (x1, x2),
        output_path,
        input_names=["image_t1", "image_t2"],
        output_names=["change_mask"],
        dynamic_axes={
            "image_t1": {0: "batch", 2: "height", 3: "width"},
            "image_t2": {0: "batch", 2: "height", 3: "width"},
            "change_mask": {0: "batch", 2: "height", 3: "width"},
        },
        opset_version=opset_version,
        do_constant_folding=True,
    )

    # Validate
    import onnx

    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)

    file_size = os.path.getsize(output_path) / (1024 * 1024)
    print(f"ONNX model saved: {output_path} ({file_size:.1f} MB)")
    print("ONNX model validation passed")

    return Path(output_path)


def benchmark_pytorch(
    model: torch.nn.Module,
    device: torch.device,
    input_size: int = 256,
    in_channels: int = 3,
    num_warmup: int = 5,
    num_runs: int = 50,
) -> dict[str, float]:
    """Benchmark PyTorch inference latency.

    Returns:
        Dictionary with mean, std, min, max latency in milliseconds.
    """
    model.eval().to(device)
    x1 = torch.randn(1, in_channels, input_size, input_size, device=device)
    x2 = torch.randn(1, in_channels, input_size, input_size, device=device)

    # Warmup
    for _ in range(num_warmup):
        with torch.no_grad():
            _ = model(x1, x2)
    if device.type == "cuda":
        torch.cuda.synchronize()

    # Benchmark
    latencies = []
    for _ in range(num_runs):
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.no_grad():
            _ = model(x1, x2)
        if device.type == "cuda":
            torch.cuda.synchronize()
        latencies.append((time.perf_counter() - start) * 1000)

    return {
        "mean_ms": float(np.mean(latencies)),
        "std_ms": float(np.std(latencies)),
        "min_ms": float(np.min(latencies)),
        "max_ms": float(np.max(latencies)),
        "p50_ms": float(np.percentile(latencies, 50)),
        "p95_ms": float(np.percentile(latencies, 95)),
        "p99_ms": float(np.percentile(latencies, 99)),
    }


def benchmark_onnx(
    onnx_path: str,
    input_size: int = 256,
    in_channels: int = 3,
    num_warmup: int = 5,
    num_runs: int = 50,
) -> dict[str, float]:
    """Benchmark ONNX Runtime inference latency.

    Returns:
        Dictionary with mean, std, min, max latency in milliseconds.
    """
    import onnxruntime as ort

    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    session = ort.InferenceSession(onnx_path, providers=providers)

    active_provider = session.get_providers()[0]
    print(f"ONNX Runtime provider: {active_provider}")

    x1 = np.random.randn(1, in_channels, input_size, input_size).astype(np.float32)
    x2 = np.random.randn(1, in_channels, input_size, input_size).astype(np.float32)

    # Warmup
    for _ in range(num_warmup):
        _ = session.run(None, {"image_t1": x1, "image_t2": x2})

    # Benchmark
    latencies = []
    for _ in range(num_runs):
        start = time.perf_counter()
        _ = session.run(None, {"image_t1": x1, "image_t2": x2})
        latencies.append((time.perf_counter() - start) * 1000)

    return {
        "mean_ms": float(np.mean(latencies)),
        "std_ms": float(np.std(latencies)),
        "min_ms": float(np.min(latencies)),
        "max_ms": float(np.max(latencies)),
        "p50_ms": float(np.percentile(latencies, 50)),
        "p95_ms": float(np.percentile(latencies, 95)),
        "p99_ms": float(np.percentile(latencies, 99)),
    }


def print_benchmark_table(results: dict[str, dict[str, float]]) -> None:
    """Pretty-print benchmark results as a comparison table."""
    print(f"\n{'='*65}")
    print(f"{'Inference Benchmark Results':^65}")
    print(f"{'='*65}")
    print(f"{'Metric':<15}", end="")
    for name in results:
        print(f"{name:>20}", end="")
    print()
    print("-" * 65)

    metrics = ["mean_ms", "std_ms", "min_ms", "max_ms", "p50_ms", "p95_ms", "p99_ms"]
    labels = ["Mean (ms)", "Std (ms)", "Min (ms)", "Max (ms)", "P50 (ms)", "P95 (ms)", "P99 (ms)"]

    for label, metric in zip(labels, metrics):
        print(f"{label:<15}", end="")
        for name in results:
            val = results[name].get(metric, 0)
            print(f"{val:>20.2f}", end="")
        print()

    # Speedup
    names = list(results.keys())
    if len(names) == 2:
        speedup = results[names[0]]["mean_ms"] / max(results[names[1]]["mean_ms"], 1e-6)
        print("-" * 65)
        print(f"{'Speedup':<15}{speedup:>40.2f}x ({names[1]} vs {names[0]})")

    print(f"{'='*65}")


def main():
    parser = argparse.ArgumentParser(description="Export model to ONNX and benchmark")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to .pth checkpoint")
    parser.add_argument("--output", type=str, default="models/exports/model.onnx", help="ONNX output path")
    parser.add_argument("--input-size", type=int, default=256, help="Input spatial resolution")
    parser.add_argument("--in-channels", type=int, default=3, help="Number of input channels")
    parser.add_argument("--num-runs", type=int, default=50, help="Number of benchmark iterations")
    parser.add_argument("--benchmark-only", action="store_true", help="Skip export, only benchmark")
    parser.add_argument("--onnx-path", type=str, default=None, help="ONNX model path for benchmark")
    args = parser.parse_args()

    onnx_path = args.onnx_path or args.output

    # Export
    if not args.benchmark_only:
        export_to_onnx(
            checkpoint_path=args.checkpoint,
            output_path=args.output,
            input_size=args.input_size,
            in_channels=args.in_channels,
        )

    # Benchmark
    print(f"\nRunning benchmark ({args.num_runs} iterations, {args.input_size}x{args.input_size})...")

    results = {}

    # PyTorch benchmark
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nPyTorch ({device})...")
    model = build_model(in_channels=args.in_channels, pretrained=False, deep_supervision=False)
    results["PyTorch"] = benchmark_pytorch(
        model, device, args.input_size, args.in_channels, num_runs=args.num_runs,
    )

    # ONNX benchmark
    if Path(onnx_path).exists():
        try:
            print(f"\nONNX Runtime...")
            results["ONNX-RT"] = benchmark_onnx(
                onnx_path, args.input_size, args.in_channels, num_runs=args.num_runs,
            )
        except ImportError:
            print("onnxruntime not installed, skipping ONNX benchmark")
    else:
        print(f"ONNX model not found at {onnx_path}, skipping ONNX benchmark")

    print_benchmark_table(results)


if __name__ == "__main__":
    main()
