"""
GeoOSAM inference benchmark — generates performance table for JOSS paper.

Tests all SAM2.1 model sizes on GPU, SAM2.1_B (Ultralytics) on CPU,
and SAM3 on GPU, using point and bbox prompts on a synthetic 1024x1024
RGB image (representative satellite crop size).

Run from the plugin root directory:
    python benchmark_joss.py
"""

import os
import sys
import time
import statistics

import numpy as np
import torch

plugin_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, plugin_dir)

# Suppress hydra/SAM2 startup noise
os.environ.setdefault("HYDRA_FULL_ERROR", "0")

N_WARMUP = 2
N_RUNS = 8
IMAGE_SIZE = 1024  # pixels, representative satellite crop

# SAM2.1 (Meta) model configs — paths relative to plugin root
SAM2_MODELS = [
    ("SAM2.1 Tiny",   "sam2/checkpoints/sam2.1_hiera_tiny.pt",      "sam2.1/sam2.1_hiera_t"),
    ("SAM2.1 Small",  "sam2/checkpoints/sam2.1_hiera_small.pt",     "sam2.1/sam2.1_hiera_s"),
    ("SAM2.1 Base+",  "sam2/checkpoints/sam2.1_hiera_base_plus.pt", "sam2.1/sam2.1_hiera_b+"),
    ("SAM2.1 Large",  "sam2/checkpoints/sam2.1_hiera_large.pt",     "sam2.1/sam2.1_hiera_l"),
]

SAM21_ULTRALYTICS_WEIGHTS = "/home/rius/sam2.1_b.pt"
SAM3_WEIGHTS = os.path.join(plugin_dir, "sam3.pt")


def make_image():
    """Synthetic 1024x1024 RGB uint8 image."""
    rng = np.random.default_rng(42)
    return rng.integers(0, 256, (IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)


def make_point():
    return np.array([[IMAGE_SIZE // 2, IMAGE_SIZE // 2]], dtype=np.float32)


def make_bbox():
    c = IMAGE_SIZE // 2
    r = IMAGE_SIZE // 8
    return np.array([c - r, c - r, c + r, c + r], dtype=np.float32)


def median_ms(times):
    return round(statistics.median(times) * 1000, 1)


def stdev_ms(times):
    return round(statistics.stdev(times) * 1000, 1) if len(times) > 1 else 0.0


# ── SAM2.1 (Meta) ──────────────────────────────────────────────────────────────

def benchmark_sam2(label, checkpoint, config, device_str):
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    from hydra.core.global_hydra import GlobalHydra
    from hydra import initialize_config_module

    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()

    device = torch.device(device_str)
    image = make_image()
    point_coords = make_point()
    point_labels = np.array([1], dtype=np.int32)
    box = make_bbox()

    checkpoint_path = os.path.join(plugin_dir, checkpoint)
    if not os.path.exists(checkpoint_path):
        return None, None, f"SKIP (checkpoint not found: {checkpoint_path})"

    try:
        with initialize_config_module(config_module="sam2.configs"):
            sam2_model = build_sam2(config, checkpoint_path, device=device)
        sam2_model.eval()
        predictor = SAM2ImagePredictor(sam2_model)

        # Warmup
        for _ in range(N_WARMUP):
            with torch.inference_mode():
                predictor.set_image(image)
                predictor.predict(point_coords=point_coords, point_labels=point_labels,
                                  multimask_output=False)

        # Point prompt timing
        point_times = []
        for _ in range(N_RUNS):
            with torch.inference_mode():
                t0 = time.perf_counter()
                predictor.set_image(image)
                predictor.predict(point_coords=point_coords, point_labels=point_labels,
                                  multimask_output=False)
                if device_str == "cuda":
                    torch.cuda.synchronize()
                point_times.append(time.perf_counter() - t0)

        # Bbox prompt timing
        bbox_times = []
        for _ in range(N_RUNS):
            with torch.inference_mode():
                t0 = time.perf_counter()
                predictor.set_image(image)
                predictor.predict(box=box, multimask_output=False)
                if device_str == "cuda":
                    torch.cuda.synchronize()
                bbox_times.append(time.perf_counter() - t0)

        del predictor, sam2_model
        if device_str == "cuda":
            torch.cuda.empty_cache()
        if GlobalHydra.instance().is_initialized():
            GlobalHydra.instance().clear()

        return point_times, bbox_times, None

    except Exception as e:
        if GlobalHydra.instance().is_initialized():
            GlobalHydra.instance().clear()
        return None, None, str(e)


# ── SAM2.1_B (Ultralytics, CPU) ────────────────────────────────────────────────

def benchmark_ultralytics_cpu():
    try:
        from ultralytics import SAM
        image = make_image()
        point = [[IMAGE_SIZE // 2, IMAGE_SIZE // 2]]
        box = [IMAGE_SIZE // 2 - 128, IMAGE_SIZE // 2 - 128,
               IMAGE_SIZE // 2 + 128, IMAGE_SIZE // 2 + 128]

        model = SAM(SAM21_ULTRALYTICS_WEIGHTS)
        model.to("cpu")

        # Warmup
        for _ in range(N_WARMUP):
            model.predict(image, points=[point], labels=[[1]], verbose=False)

        point_times = []
        for _ in range(N_RUNS):
            t0 = time.perf_counter()
            model.predict(image, points=[point], labels=[[1]], verbose=False)
            point_times.append(time.perf_counter() - t0)

        bbox_times = []
        for _ in range(N_RUNS):
            t0 = time.perf_counter()
            model.predict(image, bboxes=[box], verbose=False)
            bbox_times.append(time.perf_counter() - t0)

        del model
        return point_times, bbox_times, None

    except Exception as e:
        return None, None, str(e)


# ── SAM3 (Ultralytics, GPU) ────────────────────────────────────────────────────

def benchmark_sam3_gpu():
    try:
        from ultralytics import SAM
        image = make_image()
        point = [[IMAGE_SIZE // 2, IMAGE_SIZE // 2]]
        box = [IMAGE_SIZE // 2 - 128, IMAGE_SIZE // 2 - 128,
               IMAGE_SIZE // 2 + 128, IMAGE_SIZE // 2 + 128]

        if not os.path.exists(SAM3_WEIGHTS):
            return None, None, f"SKIP (not found: {SAM3_WEIGHTS})"

        model = SAM(SAM3_WEIGHTS)
        model.to("cuda")

        # Warmup
        for _ in range(N_WARMUP):
            model.predict(image, points=[point], labels=[[1]], verbose=False)

        point_times = []
        for _ in range(N_RUNS):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            model.predict(image, points=[point], labels=[[1]], verbose=False)
            torch.cuda.synchronize()
            point_times.append(time.perf_counter() - t0)

        bbox_times = []
        for _ in range(N_RUNS):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            model.predict(image, bboxes=[box], verbose=False)
            torch.cuda.synchronize()
            bbox_times.append(time.perf_counter() - t0)

        del model
        torch.cuda.empty_cache()
        return point_times, bbox_times, None

    except Exception as e:
        return None, None, str(e)


# ── Main ───────────────────────────────────────────────────────────────────────

def fmt_row(label, device, point_t, bbox_t, err):
    if err:
        return f"  {label:<22} {device:<6}  {'ERROR/SKIP':<28}  {err[:60]}"
    p = f"{median_ms(point_t)}ms ±{stdev_ms(point_t)}"
    b = f"{median_ms(bbox_t)}ms ±{stdev_ms(bbox_t)}"
    return f"  {label:<22} {device:<6}  point: {p:<20}  bbox: {b}"


def main():
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A"
    cpu_count = os.cpu_count()
    print(f"\nGeoOSAM Inference Benchmark")
    print(f"GPU : {gpu_name}")
    print(f"CPU : {cpu_count} logical cores")
    print(f"PyTorch : {torch.__version__}")
    print(f"Image size : {IMAGE_SIZE}×{IMAGE_SIZE} px  |  Runs: {N_RUNS} (after {N_WARMUP} warmup)\n")
    print("-" * 80)

    results = []

    # SAM2.1 variants on GPU
    for label, ckpt, cfg in SAM2_MODELS:
        print(f"  Benchmarking {label} (GPU)...", flush=True)
        pt, bt, err = benchmark_sam2(label, ckpt, cfg, "cuda")
        results.append((label, "GPU", pt, bt, err))

    # SAM2.1 Tiny on CPU for comparison
    label_cpu = "SAM2.1 Tiny"
    ckpt_cpu = SAM2_MODELS[0][1]
    cfg_cpu = SAM2_MODELS[0][2]
    print(f"  Benchmarking SAM2.1 Tiny (CPU)...", flush=True)
    pt, bt, err = benchmark_sam2(label_cpu, ckpt_cpu, cfg_cpu, "cpu")
    results.append((label_cpu + " (Meta)", "CPU", pt, bt, err))

    # Ultralytics SAM2.1_B on CPU
    print(f"  Benchmarking SAM2.1_B Ultralytics (CPU)...", flush=True)
    pt, bt, err = benchmark_ultralytics_cpu()
    results.append(("SAM2.1_B Ultralytics", "CPU", pt, bt, err))

    # SAM3 on GPU
    print(f"  Benchmarking SAM3 (GPU)...", flush=True)
    pt, bt, err = benchmark_sam3_gpu()
    results.append(("SAM3", "GPU", pt, bt, err))

    print("-" * 80)
    print(f"\n{'Model':<24} {'Device':<6}  {'Point prompt':<28}  {'Bbox prompt'}")
    print("-" * 80)
    for label, device, pt, bt, err in results:
        print(fmt_row(label, device, pt, bt, err))
    print("-" * 80)
    print("\nAll times = median over 8 runs (set_image + predict, end-to-end).\n")


if __name__ == "__main__":
    main()
