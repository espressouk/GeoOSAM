# GeoOSAM Troubleshooting Guide

## 🚨 Quick Fixes

### Plugin Won't Load

```python
# Run in QGIS Python Console to check:
import sys
print("Python version:", sys.version)

# QGIS version check (Windows compatible):
try:
    import qgis.utils
    print("QGIS version:", qgis.utils.Qgis.QGIS_VERSION)
except:
    # Alternative for Windows:
    from qgis.core import Qgis
    print("QGIS version:", Qgis.QGIS_VERSION)

# Check if plugin directory exists:
import os
plugin_path = os.path.expanduser("~/.local/share/QGIS/QGIS3/profiles/default/python/plugins/geo_osam")
print("Plugin exists:", os.path.exists(plugin_path))
```

### Model Issues

```python
# Check model selection and availability:
try:
    from ultralytics import SAM
    print("✅ Ultralytics available - SAM2.1_B ready")

    # Test model loading
    test_model = SAM('sam2.1_b.pt')  # mobile_sam.pt
    print("✅ SAM2.1_B loaded successfully")
except ImportError:
    print("❌ Ultralytics not available - falling back to SAM 2.1")
except Exception as e:
    print(f"⚠️ SAM2.1_B error: {e}")

# Check SAM 2.1 model:
sam_path = os.path.expanduser("~/.local/share/QGIS/QGIS3/profiles/default/python/plugins/geo_osam/sam2/checkpoints/sam2.1_hiera_tiny.pt")
print(f"SAM 2.1 model exists: {os.path.exists(sam_path)}")
if os.path.exists(sam_path):
    print(f"SAM 2.1 size: {os.path.getsize(sam_path)/1024/1024:.1f}MB")
```

### Device Detection Issues

```python
# Check what device/model combination is selected:
import torch

print("=== Device Detection ===")
if torch.cuda.is_available():
    print(f"🎮 CUDA GPU: {torch.cuda.get_device_name(0)}")
    print("→ Will use SAM 2.1 for maximum accuracy")
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    print("🍎 Apple Silicon GPU detected")
    print("→ Will use SAM 2.1 for optimal performance")
else:
    print("💻 CPU-only system detected")
    print("→ Will use SAM2.1_B for optimal CPU performance")

# Check threading setup:
print(f"PyTorch threads: {torch.get_num_threads()}")
print(f"CPU cores: {os.cpu_count()}")
```

### Segmentation Not Working

1. **Check device/model selection** - Look at control panel header
2. **Check raster layer is selected** (not vector)
3. **Zoom to appropriate level** (not too far out)
4. **Try different click position** (center of object)
5. **Check image quality** (contrast, resolution)
6. **For multi-spectral images** - Verify bands contain data (not all zeros)

---

## 🛰️ Multi-spectral Image Issues

### **"Found 0 vegetation candidates" with UAV imagery**

**Cause:** Multi-spectral data not processed correctly  
**Solution:**

```python
# Check band count and data ranges:
import rasterio
import numpy as np

# Open your multi-spectral image
with rasterio.open("your_uav_image.tif") as src:
    print(f"Bands: {src.count}")
    print(f"Data type: {src.dtypes[0]}")
    
    # Check band 3 (Red) and band 4 (NIR) for NDVI
    if src.count >= 4:
        red = src.read(3, window=rasterio.windows.Window(0, 0, 100, 100))
        nir = src.read(4, window=rasterio.windows.Window(0, 0, 100, 100))
        
        print(f"Red band range: {red.min():.4f} to {red.max():.4f}")
        print(f"NIR band range: {nir.min():.4f} to {nir.max():.4f}")
        
        if red.max() == 0 and nir.max() == 0:
            print("❌ Bands contain no data (all zeros)")
        elif red.max() <= 1.0:
            print("✅ Reflectance values detected (0-1 range)")
        else:
            print("✅ Digital number values detected")
```

### **"Tensor size mismatch" with 5+ band images**

**Cause:** SAM2 receiving multi-spectral instead of RGB  
**Solution:** This should be automatically handled. If persistent:

```python
# Check if dual processing is working:
print("Plugin should show:")
print("📡 Multi-spectral mode: reading all 5 bands")
print("🔍 Detecting Vegetation candidates in (height, width, 5) region (multi-spectral)")

# If you see (height, width, 3) instead, there's an issue with the band processing
```

### **"All pixel values are 0" in vegetation detection**

**Cause:** Data type truncation during image reading  
**Solution:**

```python
# This was fixed in v1.2.0 - ensure you have the latest version
# Check plugin version in QGIS Plugin Manager

# Expected behavior:
# - Reflectance values (0.012-0.536) preserved as float32
# - Automatic scaling to 0-255 for processing
# - Should see: "🔧 NORM: Scaled reflectance to 0-255, result range: 3 to 137"
```

### **Batch mode returns huge masks (entire image)**

**Cause:** SAM segmenting large homogeneous areas instead of objects  
**Solution:**

```python
# Fixed in v1.2.0 with size validation
# Plugin now rejects masks > 10% of image area

# Expected output:
# "❌ REJECTED: Object 5 too large (1089459 > 145216, 75.0% of image)"

# If still occurring:
# 1. Use smaller bounding boxes
# 2. Try point mode instead
# 3. Check if vegetation is actually present in the area
```

### **Roads/tracks detected as vegetation**

**Cause:** Linear features passing shape validation  
**Solution:**

```python
# Enhanced filtering in v1.2.0:
# - Aspect ratio ≤ 2.0 (rejects elongated features)
# - Solidity ≥ 0.5 (rejects sparse linear features)
# - Thin feature detection (width/height < 8px with high aspect ratio)

# Expected output:
# "❌ CONTOUR 15: area=245.0px, too elongated (AR=4.2>2.0)"
# "❌ CONTOUR 23: area=156.0px, too thin (45x4px, track-like)"

# If roads still detected:
# 1. Use smaller bounding boxes
# 2. Adjust to areas with denser vegetation
# 3. Check if detected features are actually vegetation edges
```

### **RuntimeWarning: invalid value encountered in sqrt**

**Cause:** Numerical precision in texture calculation  
**Solution:**

```python
# Fixed in v1.2.0 with variance clamping
# Should no longer see this warning

# If warning persists, it's harmless but indicates:
# 1. Very low texture variation in the image area
# 2. Possible issues with data normalization
# 3. Try different area with more texture variation
```

---

## 📋 Common Error Messages

### Installation Errors

#### Error: "No module named 'ultralytics'"

**Cause:** Ultralytics not installed (needed for MobileSAM)  
**Solution:**

```python
# Install in QGIS Python Console:
import subprocess, sys
subprocess.check_call([sys.executable, "-m", "pip", "install", "ultralytics"])
print("✅ Ultralytics installed - MobileSAM now available")
```

#### Error: "No module named 'torch'"

**Cause:** PyTorch not installed
**Solution:**

**🎯 Windows (Recommended): Use OSGeo4W Shell**

```bash
# Open OSGeo4W Shell (Start Menu → OSGeo4W → OSGeo4W Shell)
pip install torch torchvision ultralytics opencv-python rasterio shapely hydra-core iopath
```

**🔧 Alternative: QGIS Python Console**

```python
# In QGIS Python Console:
import subprocess, sys
packages = ["torch", "torchvision", "ultralytics", "opencv-python", "rasterio", "shapely", "hydra-core", "iopath"]
for pkg in packages: subprocess.check_call([sys.executable, "-m", "pip", "install", pkg]); print(f"✅ Installed {pkg}")
```

#### Error: "No module named 'iopath'"

**Cause:** iopath dependency missing (required for SAM2)  
**Solution:**

```python
# Install iopath in QGIS Python Console:
import subprocess, sys
subprocess.check_call([sys.executable, "-m", "pip", "install", "iopath"])
print("✅ iopath installed")
```

#### Error: "Plugin could not be loaded"

**Cause:** Missing dependencies or corrupted files  
**Solution:**

```python
# Check all dependencies (including new ones):
deps = {
    "torch": "torch",
    "torchvision": "torchvision",
    "ultralytics": "ultralytics",
    "cv2": "opencv-python",
    "rasterio": "rasterio",
    "shapely": "shapely",
    "hydra": "hydra-core",
    "iopath": "iopath"
}

missing = []
for dep, pkg_name in deps.items():
    try:
        __import__(dep)
        print(f"✅ {dep}")
    except ImportError:
        missing.append(pkg_name)
        print(f"❌ {dep} MISSING")

if missing:
    import subprocess, sys
    for pkg in missing:
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])
```

### Model Selection Errors

#### Error: "MobileSAM prediction error"

**Cause:** Ultralytics model issue  
**Solution:**

```python
# Force re-download of MobileSAM:
import os
ultralytics_cache = os.path.expanduser("~/.ultralytics")
print(f"Clearing Ultralytics cache: {ultralytics_cache}")
# Remove cache and restart QGIS

# Or force CPU fallback:
os.environ["GEOOSAM_FORCE_CPU"] = "1"  # Forces SAM 2.1 on CPU
```

#### Error: "Device detection failed"

**Cause:** CUDA/MPS detection issues  
**Solution:**

```python
# Debug device detection:
import torch
import os

print("=== Device Debug ===")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    try:
        props = torch.cuda.get_device_properties(0)
        print(f"GPU memory: {props.total_memory / 1024**3:.1f}GB")
        print(f"GPU name: {torch.cuda.get_device_name(0)}")
    except Exception as e:
        print(f"CUDA error: {e}")

print(f"MPS available: {torch.backends.mps.is_available() if hasattr(torch.backends, 'mps') else 'Not supported'}")

# Force specific device:
# os.environ["GEOOSAM_FORCE_CPU"] = "1"  # Force CPU mode
# os.environ["GEOOSAM_FORCE_GPU"] = "1"  # Force GPU mode
```

### Runtime Errors

#### Error: "Ultralytics not available - falling back to SAM2"

**Cause:** Normal fallback behavior, but you might want MobileSAM  
**Solution:**

```bash
# Install Ultralytics for better CPU performance:
pip install ultralytics

# Restart QGIS to detect new installation
```

#### Error: "CUDA out of memory"

**Cause:** GPU memory insufficient  
**Solution:**

- Plugin automatically falls back to MobileSAM on CPU
- Close other GPU-intensive applications
- Restart QGIS
- Use smaller image crop sizes

#### Error: "SAM model failed to load"

**Cause:** Issues with either SAM 2.1 or MobileSAM  
**Solution:**

```bash
# For SAM 2.1 (GPU users):
cd ~/.local/share/QGIS/QGIS3/profiles/default/python/plugins/geo_osam/sam2/checkpoints/
rm -f sam2.1_hiera_tiny.pt
wget https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2.1_hiera_tiny.pt

# For MobileSAM (CPU users):
pip uninstall ultralytics
pip install ultralytics
# MobileSAM will auto-download on next use
```

#### Error: "UltralyticsPredictor failed"

**Cause:** Issues with MobileSAM model or Ultralytics  
**Solution:**

```python
# Debug Ultralytics installation:
try:
    from ultralytics import SAM
    model = SAM('mobile_sam.pt')
    print("✅ Ultralytics working")

    # Test prediction
    import numpy as np
    test_img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    result = model.predict(source=test_img, verbose=False)
    print("✅ MobileSAM prediction working")

except Exception as e:
    print(f"❌ Ultralytics error: {e}")
    print("→ Plugin will fallback to SAM 2.1")
```

### Performance Issues

#### Error: "CPU threading setup failed"

**Cause:** Multi-threading optimization issues  
**Solution:**

```python
# Check threading setup:
import torch
import multiprocessing
import os

print(f"CPU cores detected: {multiprocessing.cpu_count()}")
print(f"PyTorch threads: {torch.get_num_threads()}")
print(f"PyTorch interop threads: {torch.get_num_interop_threads()}")

# Environment variables:
thread_vars = ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"]
for var in thread_vars:
    print(f"{var}: {os.environ.get(var, 'Not set')}")

# Manual override:
torch.set_num_threads(8)  # Adjust based on your CPU
```

---

## 🔧 Platform-Specific Issues

### Windows Issues

#### Issue: "DLL load failed" (Ultralytics)

**Cause:** Missing Visual C++ redistributables  
**Solution:**

- Install Visual C++ Redistributable for Visual Studio 2019+
- Or use CPU fallback: `os.environ["GEOOSAM_FORCE_CPU"] = "1"`

#### Issue: Ultralytics slow on Windows

**Cause:** Windows Defender or antivirus scanning  
**Solution:**

- Add QGIS and Python to antivirus exceptions
- Add `~/.ultralytics` cache folder to exceptions

### macOS Issues

#### Issue: "MPS fallback warnings" (Apple Silicon)

**Cause:** Some operations not supported on MPS  
**Solution:**

```python
# This is normal - plugin handles fallback automatically
# Operations fall back to CPU when needed
# Performance is still excellent overall
```

#### Issue: Ultralytics download fails on macOS

**Cause:** Certificate or firewall issues  
**Solution:**

```bash
# Manual MobileSAM download:
pip uninstall ultralytics
pip install ultralytics --trusted-host pypi.org --trusted-host pypi.python.org
```

### Linux Issues

#### Issue: "libGL.so.1: cannot open shared file" (Ultralytics)

**Cause:** Missing OpenGL libraries for OpenCV  
**Solution:**

```bash
# Ubuntu/Debian:
sudo apt install libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender-dev

# CentOS/RHEL:
sudo yum install mesa-libGL libXext libSM
```

#### Issue: Permission denied for Ultralytics cache

**Cause:** Cache directory permissions  
**Solution:**

```bash
# Fix permissions:
mkdir -p ~/.ultralytics
chmod 755 ~/.ultralytics

# Or use system-wide installation:
sudo pip install ultralytics
```

---

## 🐛 Performance Issues

### Slow Segmentation

#### Diagnosis

```python
# Check current model and device:
print("=== Performance Diagnosis ===")

# Device check:
import torch
device = "unknown"
if torch.cuda.is_available():
    device = f"CUDA ({torch.cuda.get_device_name(0)})"
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    device = "Apple Silicon (MPS)"
else:
    device = f"CPU ({torch.get_num_threads()} threads)"

print(f"Device: {device}")

# Model check:
try:
    from ultralytics import SAM
    print("Model: MobileSAM (CPU optimized)")
except ImportError:
    print("Model: SAM 2.1 (Fallback)")

# Expected performance:
import os
cores = os.cpu_count()
if "CUDA" in device:
    print("Expected speed: 0.2-0.5 seconds")
elif "MPS" in device:
    print("Expected speed: 1-2 seconds")
elif cores >= 24:
    print("Expected speed: <1 second (high-core CPU)")
elif cores >= 16:
    print("Expected speed: 1-2 seconds")
elif cores >= 8:
    print("Expected speed: 2-4 seconds")
else:
    print("Expected speed: 3-6 seconds")
```

#### Solutions by Hardware

**GPU Users (slow performance):**

- Verify GPU is actually being used (check control panel header)
- Close other GPU applications
- Check VRAM usage (`nvidia-smi`)
- Try smaller crop sizes (zoom in more)

**CPU Users (slower than expected):**

- Verify MobileSAM is being used (shows in control panel)
- Check CPU usage during processing
- Close other CPU-intensive applications
- For 16+ cores: Should be using 75% of cores

**Apple Silicon Users:**

- Verify MPS is detected and used
- Some operations may fallback to CPU (normal)
- Performance should still be excellent

### Threading Issues

#### High CPU Usage

**Diagnosis:**

```python
import psutil
import os

print(f"System CPU cores: {os.cpu_count()}")
print(f"PyTorch threads: {torch.get_num_threads()}")
print(f"Current CPU usage: {psutil.cpu_percent()}%")

# During processing, CPU usage should be high but not 100%
# for extended periods
```

**Solution:**

```python
# Reduce thread count if system becomes unresponsive:
import torch
torch.set_num_threads(max(1, os.cpu_count() // 2))
```

---

## 🔍 Advanced Debugging

### Model Selection Debug

```python
# Complete model selection diagnostic:
print("=== GeoOSAM Model Selection Debug ===")

import os
import torch

# 1. Check force flags
force_cpu = os.getenv("GEOOSAM_FORCE_CPU")
force_gpu = os.getenv("GEOOSAM_FORCE_GPU")
print(f"Force CPU: {force_cpu}")
print(f"Force GPU: {force_gpu}")

# 2. Hardware detection
print("\n--- Hardware Detection ---")
cuda_available = torch.cuda.is_available()
mps_available = hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()

print(f"CUDA available: {cuda_available}")
if cuda_available:
    props = torch.cuda.get_device_properties(0)
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  Memory: {props.total_memory / 1024**3:.1f}GB")
    print(f"  Sufficient memory (≥3GB): {props.total_memory / 1024**3 >= 3}")

print(f"MPS available: {mps_available}")

# 3. Model availability
print("\n--- Model Availability ---")
try:
    from ultralytics import SAM
    test_model = SAM('mobile_sam.pt')
    print("✅ MobileSAM available")
    mobilesam_available = True
except Exception as e:
    print(f"❌ MobileSAM failed: {e}")
    mobilesam_available = False

sam2_path = os.path.expanduser("~/.local/share/QGIS/QGIS3/profiles/default/python/plugins/geo_osam/sam2/checkpoints/sam2.1_hiera_tiny.pt")
sam2_available = os.path.exists(sam2_path)
print(f"SAM 2.1 available: {sam2_available}")

# 4. Final selection logic
print("\n--- Final Selection ---")
if force_cpu:
    device = "cpu"
    model = "MobileSAM" if mobilesam_available else "SAM2"
    print(f"FORCED → {device}, {model}")
elif cuda_available and not force_cpu:
    gpu_props = torch.cuda.get_device_properties(0)
    if gpu_props.total_memory / 1024**3 >= 3:  # Changed from 4GB to 3GB
        device = "cuda"
        model = "SAM2"
        print(f"AUTO → {device}, {model} (sufficient GPU memory)")
    else:
        device = "cpu"
        model = "MobileSAM" if mobilesam_available else "SAM2"
        print(f"AUTO → {device}, {model} (insufficient GPU memory)")
elif mps_available:
    device = "mps"
    model = "SAM2"
    print(f"AUTO → {device}, {model}")
else:
    device = "cpu"
    model = "MobileSAM" if mobilesam_available else "SAM2"
    print(f"AUTO → {device}, {model}")

print(f"\nFinal: {model} on {device.upper()}")
```

### Performance Profiling

```python
# Profile segmentation performance:
import time

def profile_segmentation():
    """Profile a complete segmentation cycle."""

    print("=== Performance Profile ===")

    # Setup timing
    start_total = time.time()

    # Mock preparation (replace with actual)
    start_prep = time.time()
    # ... preparation code ...
    prep_time = time.time() - start_prep

    # Mock inference (replace with actual)
    start_inference = time.time()
    # ... inference code ...
    inference_time = time.time() - start_inference

    # Mock post-processing (replace with actual)
    start_post = time.time()
    # ... post-processing code ...
    post_time = time.time() - start_post

    total_time = time.time() - start_total

    print(f"Preparation: {prep_time:.3f}s ({prep_time/total_time*100:.1f}%)")
    print(f"Inference: {inference_time:.3f}s ({inference_time/total_time*100:.1f}%)")
    print(f"Post-process: {post_time:.3f}s ({post_time/total_time*100:.1f}%)")
    print(f"Total: {total_time:.3f}s")

    # Performance expectations
    import os
    cores = os.cpu_count()

    if torch.cuda.is_available():
        expected = "0.2-0.5s"
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        expected = "1-2s"
    elif cores >= 24:
        expected = "<1s"
    elif cores >= 16:
        expected = "1-2s"
    else:
        expected = "2-4s"

    print(f"Expected: {expected}")
    print(f"Performance: {'✅ Good' if total_time <= 5 else '⚠️ Slow' if total_time <= 10 else '❌ Very Slow'}")

# Run during actual segmentation:
# profile_segmentation()
```

### Memory Usage Analysis

```python
# Analyze memory usage patterns:
import psutil
import torch

def analyze_memory():
    """Analyze current memory usage."""

    print("=== Memory Analysis ===")

    # System memory
    memory = psutil.virtual_memory()
    print(f"System RAM: {memory.total / 1024**3:.1f}GB")
    print(f"Available: {memory.available / 1024**3:.1f}GB")
    print(f"Used: {memory.percent}%")

    # GPU memory (if available)
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(0).total_memory
        allocated = torch.cuda.memory_allocated(0)
        cached = torch.cuda.memory_reserved(0)

        print(f"\nGPU Memory: {gpu_memory / 1024**3:.1f}GB")
        print(f"Allocated: {allocated / 1024**3:.1f}GB")
        print(f"Cached: {cached / 1024**3:.1f}GB")
        print(f"Free: {(gpu_memory - cached) / 1024**3:.1f}GB")

    # Process memory
    process = psutil.Process()
    process_memory = process.memory_info()
    print(f"\nProcess RSS: {process_memory.rss / 1024**3:.1f}GB")
    print(f"Process VMS: {process_memory.vms / 1024**3:.1f}GB")

# Call during processing to monitor:
# analyze_memory()
```

### Dependency Conflict Detection

```python
# Detect dependency conflicts:
def check_dependency_conflicts():
    """Check for common dependency conflicts."""

    print("=== Dependency Conflict Analysis ===")

    import pkg_resources

    # Check for multiple PyTorch versions
    try:
        torch_dist = pkg_resources.get_distribution("torch")
        print(f"PyTorch version: {torch_dist.version}")
        print(f"PyTorch location: {torch_dist.location}")
    except:
        print("❌ PyTorch not found via pkg_resources")

    # Check for conflicting OpenCV versions
    cv_packages = ["opencv-python", "opencv-contrib-python", "opencv-python-headless"]
    cv_found = []
    for pkg in cv_packages:
        try:
            dist = pkg_resources.get_distribution(pkg)
            cv_found.append(f"{pkg}: {dist.version}")
        except:
            pass

    if len(cv_found) > 1:
        print("⚠️ Multiple OpenCV packages found:")
        for cv in cv_found:
            print(f"  {cv}")
        print("Consider keeping only one OpenCV package")
    elif cv_found:
        print(f"✅ OpenCV: {cv_found[0]}")
    else:
        print("❌ No OpenCV package found")

    # Check for path conflicts
    import sys
    python_paths = sys.path
    qgis_paths = [p for p in python_paths if 'qgis' in p.lower()]

    if qgis_paths:
        print(f"\n✅ QGIS Python paths found:")
        for path in qgis_paths[:3]:  # Show first 3
            print(f"  {path}")
    else:
        print("\n⚠️ No QGIS-specific paths found in sys.path")

# Run if experiencing import issues:
# check_dependency_conflicts()
```

---

## 📞 Getting Help

### Enhanced Bug Reports

When reporting issues, include this enhanced diagnostic:

```python
# Enhanced diagnostic script:
print("=== GeoOSAM Enhanced Diagnostic ===")

import os, sys, torch
from qgis.core import QgsApplication

# Basic info
print(f"Python: {sys.version}")
print(f"QGIS: {QgsApplication.version()}")
print(f"OS: {os.name} {os.uname().sysname if hasattr(os, 'uname') else 'Windows'}")

# Dependencies
deps = {
    "torch": torch.__version__,
    "ultralytics": None,
    "cv2": None,
    "rasterio": None,
    "shapely": None,
    "hydra": None,
    "iopath": None
}

for name in deps:
    try:
        if name == "torch":
            continue  # Already have version
        module = __import__(name)
        deps[name] = getattr(module, '__version__', 'unknown')
    except ImportError:
        deps[name] = "MISSING"

for name, version in deps.items():
    print(f"{name}: {version}")

# Hardware
print(f"\nCPU cores: {os.cpu_count()}")
print(f"PyTorch threads: {torch.get_num_threads()}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    print("GPU: Apple Silicon (MPS)")
else:
    print("GPU: None (CPU only)")

# Model selection
print(f"\nExpected device: ", end="")
if torch.cuda.is_available() and torch.cuda.get_device_properties(0).total_memory / 1024**3 >= 3:
    print("CUDA → SAM 2.1")
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    print("MPS → SAM 2.1")
else:
    mobilesam = "MobileSAM" if deps["ultralytics"] != "MISSING" else "SAM 2.1 (fallback)"
    print(f"CPU → {mobilesam}")

# Plugin status
plugin_path = os.path.expanduser("~/.local/share/QGIS/QGIS3/profiles/default/python/plugins/geo_osam")
print(f"\nPlugin installed: {os.path.exists(plugin_path)}")

# Models
sam2_path = os.path.join(plugin_path, "sam2", "checkpoints", "sam2.1_hiera_tiny.pt")
if os.path.exists(sam2_path):
    print(f"SAM 2.1 model: {os.path.getsize(sam2_path)/1024/1024:.1f}MB")
else:
    print("SAM 2.1 model: Missing")

if deps["ultralytics"] != "MISSING":
    print("MobileSAM: Available via Ultralytics")
else:
    print("MobileSAM: Not available")

# Check iopath specifically
if deps["iopath"] != "MISSING":
    print("iopath: Available")
else:
    print("iopath: MISSING (install with: pip install iopath)")

print("=== End Enhanced Diagnostic ===")
```

### Support Channels

- **GitHub Issues:** https://github.com/espressouk/GeoOSAM/issues

  - Include enhanced diagnostic output
  - Specify which model was being used
  - Note performance expectations vs. reality

- **Email:** bkst.dev@gmail.com
  - For complex performance issues
  - Hardware-specific problems

### Reporting New Issues

When reporting a new issue, please include:

1. **Enhanced diagnostic output** (from script above)
2. **Steps to reproduce** the problem
3. **Expected vs. actual behavior**
4. **Error messages** (full traceback if available)
5. **Hardware specs** (GPU model, CPU cores, RAM)
6. **Data characteristics** (image size, format, resolution)

### Common Issue Categories

#### **Installation Problems**

- Missing dependencies (torch, ultralytics, iopath, etc.)
- Plugin not loading
- Permission issues

#### **Model Issues**

- Wrong model selected
- Download failures
- Model loading errors

#### **Performance Problems**

- Slower than expected
- High CPU/GPU usage
- Memory issues

#### **Segmentation Quality**

- Poor results
- No segments found
- Incorrect boundaries

#### **Export Issues**

- Shapefile problems
- Attribute errors
- Projection issues

### Quick Fixes Summary

| Issue                  | Quick Fix                                    |
| ---------------------- | -------------------------------------------- |
| Plugin won't load      | Check dependencies with diagnostic script    |
| MobileSAM missing      | `pip install ultralytics`                    |
| SAM 2.1 won't download | Check internet, clear cache, manual download |
| Wrong model selected   | Check force environment variables            |
| Slow performance       | Check device detection, close other apps     |
| iopath missing         | `pip install iopath`                         |
| Threading issues       | Reduce thread count manually                 |
| GPU out of memory      | Plugin auto-fallback to CPU                  |

### Getting the Best Help

1. **Run diagnostic first** - Most issues are revealed by the diagnostic script
2. **Check existing issues** - Your problem might already be solved
3. **Provide complete information** - More details = faster resolution
4. **Test with minimal data** - Use small test images when possible
5. **Note your workflow** - What steps led to the problem?

### Environment Variable Overrides

For testing and debugging, you can force specific behaviors:

```python
import os

# Force specific model selection
os.environ["GEOOSAM_FORCE_CPU"] = "1"          # Force CPU mode
os.environ["GEOOSAM_FORCE_GPU"] = "1"          # Force GPU mode
os.environ["GEOOSAM_FORCE_SAM2"] = "1"         # Force SAM2 (any device)
os.environ["GEOOSAM_FORCE_MOBILESAM"] = "1"    # Force MobileSAM

# Debug mode
os.environ["GEOOSAM_DEBUG"] = "1"              # Enable debug output
os.environ["GEOOSAM_VERBOSE"] = "1"            # Verbose logging

# Performance tuning
os.environ["OMP_NUM_THREADS"] = "8"            # Limit OpenMP threads
os.environ["MKL_NUM_THREADS"] = "8"            # Limit MKL threads

# Restart QGIS after setting environment variables
```

### Log File Locations

Check these locations for additional debugging information:

**Windows:**

```
C:\Users\%USERNAME%\AppData\Local\Temp\qgis_logs\
C:\Users\%USERNAME%\AppData\Roaming\QGIS\QGIS3\profiles\default\python\plugins\geo_osam\logs\
```

**macOS:**

```
~/Library/Logs/QGIS/
~/Library/Application Support/QGIS/QGIS3/profiles/default/python/plugins/geo_osam/logs/
```

**Linux:**

```
~/.local/share/QGIS/QGIS3/profiles/default/python/plugins/geo_osam/logs/
/tmp/qgis_logs/
```

**Still having issues? The enhanced diagnostic output above will help us solve your problem quickly!** 📧
