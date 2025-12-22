# GeoOSAM Installation Guide

## 🎯 Quick Installation (Recommended)

### Step 1: Install from QGIS Plugin Repository

1. **Open QGIS** (version 3.16 or later)
2. **Menu:** Plugins → Manage and Install Plugins
3. **Search:** Type "GeoOSAM"
4. **Install:** Click "Install Plugin"
5. **Enable:** Ensure plugin is checked

### Step 2: Install Dependencies

**🎯 Windows Users: IMPORTANT - Choose CPU or CUDA Version**

First, check if you have an NVIDIA GPU and want to use it:

```bash
# Open OSGeo4W Shell (Start Menu → OSGeo4W → OSGeo4W Shell)
# Check if NVIDIA GPU is available:
nvidia-smi
```

**If you have NVIDIA GPU (nvidia-smi shows your GPU):**

```bash
# Install PyTorch with CUDA su
pport (CUDA 11.8 - most compatible)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install "ultralytics>=8.3.237" opencv-python rasterio shapely hydra-core iopath pillow numpy

# OR for newer GPUs with CUDA 12.1+:
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
# pip install "ultralytics>=8.3.237" opencv-python rasterio shapely hydra-core iopath pillow numpy
```

**If you DON'T have NVIDIA GPU or want CPU-only:**

```bash
# Install CPU-only version
pip install torch torchvision "ultralytics>=8.3.237" opencv-python rasterio shapely hydra-core iopath pillow numpy
```

**🍎 macOS Users: Use Terminal**

```bash
# macOS (CPU or Apple Silicon MPS)
pip3 install torch torchvision "ultralytics>=8.3.237" opencv-python rasterio shapely hydra-core iopath pillow numpy
```

**🐧 Linux Users: Use Terminal**

```bash
# For NVIDIA GPU with CUDA support:
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip3 install "ultralytics>=8.3.237" opencv-python rasterio shapely hydra-core iopath pillow numpy

# For CPU-only:
pip3 install torch torchvision "ultralytics>=8.3.237" opencv-python rasterio shapely hydra-core iopath pillow numpy
```

**🔧 Alternative: QGIS Python Console (All Platforms)**

```python
# Open QGIS → Plugins → Python Console
# Copy and paste this code:
import subprocess, sys
packages = ["torch", "torchvision", "ultralytics", "opencv-python", "rasterio", "shapely", "hydra-core", "iopath"]
for pkg in packages: subprocess.check_call([sys.executable, "-m", "pip", "install", pkg]); print(f"✅ Installed {pkg}")

```

**Optional: SAM3 Text/Similar (PCS) Dependencies**

SAM3 text and exemplar prompts require CLIP and are still evolving in Ultralytics:

```bash
pip install git+https://github.com/"ultralytics>=8.3.237"/CLIP.git ftfy wcwidth pillow numpy
```

**✅ Verify CUDA Installation (Windows/Linux with NVIDIA GPU)**

After installing PyTorch with CUDA, verify it's working:

```python
# Open QGIS → Plugins → Python Console
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU detected: {torch.cuda.get_device_name(0)}")
else:
    print("⚠️ CUDA not detected - see troubleshooting below")
```

**Expected output for GPU users:**

- `CUDA available: True`
- `GPU detected: NVIDIA GeForce RTX ...` (your GPU name)

**If CUDA available shows False:**
See troubleshooting section: [CUDA Not Detected on Windows](#issue-cuda-not-detected-on-windows-despite-nvidia-smi-working)

### Step 3: First Use

1. **Click GeoOSAM icon** 🛰️ in QGIS toolbar
2. **Automatic model selection** happens instantly:
   - **🎮 GPU detected**: Downloads SAM 2.1 (~160MB, one-time)
   - **💻 CPU detected**: Downloads Ultralytics SAM2.1_B (~160MB via Ultralytics)
   - **⚡ High-core CPU**: Optimized for sub-second performance
3. **Control panel opens** on the right side showing your hardware
4. **Start segmenting!** 🚀

### SAM3 Weights Download (Optional)

If you select SAM3 and the weights are missing, GeoOSAM will prompt you to download:

1. Request access at https://huggingface.co/facebook/sam3
2. Click **Download Now** in the dialog
3. Create a Hugging Face **Access Token** with **Token type: Read**
4. Paste the token (used once, not stored)

Weights are saved to `~/.ultralytics/weights/sam3.pt`.

---

## 📋 Detailed Installation Instructions

### System Requirements

#### Minimum Requirements

- **Operating System:** Windows 10, macOS 10.14, Ubuntu 18.04
- **QGIS Version:** 3.16 or later
- **Python:** 3.7 or later
- **RAM:** 8GB minimum
- **Storage:** 2GB free space
- **Internet:** For automatic model downloads

#### Recommended Requirements

- **Operating System:** Windows 11, macOS 12+, Ubuntu 20.04+
- **QGIS Version:** 3.28 or later
- **Python:** 3.9 or later
- **RAM:** 16GB or more
- **GPU:** NVIDIA GPU with CUDA or Apple Silicon (auto-detected)
- **CPU:** 16+ cores for optimal CPU performance (<1s segmentation)
- **Storage:** SSD with 4GB free space

### Installation Method 1: QGIS Plugin Repository

#### Windows Installation

**🎯 Recommended: Use OSGeo4W Shell (Most Reliable)**

```bash
# 1. Install plugin through QGIS interface
# 2. Open OSGeo4W Shell (comes with QGIS installation)
#    Start Menu → OSGeo4W → OSGeo4W Shell
# 3. Install dependencies in the correct Python environment:
pip install torch torchvision "ultralytics>=8.3.237" opencv-python rasterio shapely hydra-core iopath pillow numpy
```

**Alternative Methods:**

```powershell
# Method A: Command Prompt (may use different Python than QGIS)
pip install torch torchvision "ultralytics>=8.3.237" opencv-python rasterio shapely hydra-core iopath pillow numpy

# Method B: QGIS Python Console (always works but slower)
# Open QGIS → Plugins → Python Console
import subprocess, sys
packages = ["torch", "torchvision", "ultralytics", "opencv-python", "rasterio", "shapely", "hydra-core", "iopath"]
for pkg in packages: subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])
```

#### macOS Installation

```bash
# 1. Install plugin through QGIS interface
# 2. Install dependencies via Terminal:
pip3 install torch torchvision "ultralytics>=8.3.237" opencv-python rasterio shapely hydra-core iopath pillow numpy

# For Apple Silicon Macs (automatic optimization):
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip3 install "ultralytics>=8.3.237" opencv-python rasterio shapely hydra-core iopath pillow numpy

# Alternative: Use QGIS Python Console (recommended)
```

#### Linux Installation

```bash
# 1. Install plugin through QGIS interface
# 2. Install dependencies:
pip3 install torch torchvision "ultralytics>=8.3.237" opencv-python rasterio shapely hydra-core iopath pillow numpy

# Ubuntu/Debian additional dependencies:
sudo apt update
sudo apt install python3-pip python3-dev

# NVIDIA GPU support (auto-detected):
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip3 install "ultralytics>=8.3.237" opencv-python rasterio shapely hydra-core iopath pillow numpy
```

### Installation Method 2: Manual GitHub Installation

#### Download and Extract

```bash
# 1. Download plugin from GitHub
wget https://github.com/espressouk/geoOSAM/archive/main.zip
unzip main.zip
mv GeoOSAM-main geoOSAM  # Remove -main suffix
cd geoOSAM

# Or clone with git:
git clone https://github.com/espressouk/geoOSAM.git
cd geoOSAM
```

#### Copy to QGIS Plugins Directory

**Windows:**

```powershell
# Copy plugin to QGIS plugins folder:
xcopy . "C:\Users\%USERNAME%\AppData\Roaming\QGIS\QGIS3\profiles\default\python\plugins\geo_osam" /E /I
```

**macOS:**

```bash
# Copy plugin to QGIS plugins folder:
cp -r . ~/Library/Application\ Support/QGIS/QGIS3/profiles/default/python/plugins/geo_osam
```

**Linux:**

```bash
# Copy plugin to QGIS plugins folder:
cp -r . ~/.local/share/QGIS/QGIS3/profiles/default/python/plugins/geo_osam
```

**Note:** The `.` copies the current directory contents (all the plugin files) into a new folder named `geo_osam` in the QGIS plugins directory.

#### Install Dependencies

```bash
# Install required Python packages:
pip3 install torch torchvision "ultralytics>=8.3.237" opencv-python rasterio shapely hydra-core iopath pillow numpy
```

#### Enable Plugin

1. **Open QGIS**
2. **Go to:** Plugins → Manage and Install Plugins
3. **Click:** Installed tab
4. **Find:** GeoOSAM
5. **Check:** Enable checkbox

---

## 🧠 Intelligent Model Selection

### Automatic Hardware Detection

GeoOSAM automatically detects your hardware and selects the optimal model:

| Hardware Detected        | Model Selected       | Download Size | Performance |
| ------------------------ | -------------------- | ------------- | ----------- |
| NVIDIA GPU (CUDA)        | SAM 2.1              | ~160MB        | 0.2-0.5s    |
| Apple Silicon (M1/M2/M3) | SAM 2.1              | ~160MB        | 1-2s        |
| 24+ Core CPU             | Ultralytics SAM2.1_B | ~160MB        | <1s         |
| 16+ Core CPU             | Ultralytics SAM2.1_B | ~160MB        | 1-2s        |
| 8-16 Core CPU            | Ultralytics SAM2.1_B | ~160MB        | 2-3s        |
| 4-8 Core CPU             | Ultralytics SAM2.1_B | ~160MB        | 3-5s        |

### Download Process

**🔄 What Happens Automatically:**

1. **Device Detection**: Plugin detects GPU/CPU capabilities
2. **Model Selection**: Chooses SAM 2.1 (GPU) or Ultralytics SAM2.1_B (CPU)
3. **Smart Download**: Only downloads the model you need
4. **Ultralytics Magic**: SAM2.1_B handled seamlessly by Ultralytics
5. **One-time Setup**: Subsequent uses are instant

**📥 Download Details:**

- **GPU Systems**: Downloads SAM 2.1 checkpoint directly
- **CPU Systems**: Ultralytics automatically downloads SAM2.1_B
- **Total Time**: 1-3 minutes depending on connection
- **Storage**: Only uses space for your hardware's model

---

## 🔧 Advanced Installation Options

### GPU Acceleration Setup

#### NVIDIA GPU (CUDA) - Auto-Detected

```bash
# Check CUDA availability:
nvidia-smi

# Install PyTorch with CUDA support (auto-detected):
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip3 install "ultralytics>=8.3.237" iopath pillow numpy

# Verify CUDA in QGIS Python Console:
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")
```

#### Apple Silicon (M1/M2/M3) - Auto-Detected

```bash
# Install optimized PyTorch for Apple Silicon:
pip3 install torch torchvision "ultralytics>=8.3.237" iopath pillow numpy

# Verify MPS support in QGIS Python Console:
import torch
print(f"MPS available: {torch.backends.mps.is_available()}")
```

#### High-Performance CPU Systems

```bash
# For 16+ core systems (auto-optimized):
pip3 install torch torchvision "ultralytics>=8.3.237" iopath pillow numpy

# Verify threading in QGIS Python Console:
import torch
print(f"CPU threads: {torch.get_num_threads()}")
print(f"CPU cores: {torch.get_num_interop_threads()}")
```

### Development Installation

#### For Plugin Development

```bash
# Clone repository:
git clone https://github.com/espressouk/geoOSAM.git
cd geoOSAM

# Create development environment:
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Install all dependencies:
pip install torch torchvision "ultralytics>=8.3.237" opencv-python rasterio shapely hydra-core iopath pillow numpy

# Link to QGIS plugins directory:
ln -s $(pwd)/geo_osam ~/.local/share/QGIS/QGIS3/profiles/default/python/plugins/geo_osam
```

### Docker Installation (Advanced)

```dockerfile
# Dockerfile for containerized QGIS with GeoOSAM
FROM qgis/qgis:release-3_28

# Install dependencies with Ultralytics
RUN pip3 install torch torchvision "ultralytics>=8.3.237" opencv-python rasterio shapely hydra-core iopath pillow numpy

# Copy plugin
COPY geo_osam /root/.local/share/QGIS/QGIS3/profiles/default/python/plugins/geo_osam

# Models will auto-download on first use
# No manual download needed!
```

---

## ✅ Installation Verification

### Quick Test

1. **Open QGIS**
2. **Look for:** GeoOSAM icon 🛰️ in toolbar
3. **Click icon:** Control panel should open
4. **Check status:** Should show device type (🎮 GPU / 💻 CPU) and model

### Detailed Verification

```python
# Run in QGIS Python Console:

# Test 1: Plugin loads
try:
    from geo_osam import SegSam
    print("✅ Plugin import successful")
except Exception as e:
    print(f"❌ Plugin import failed: {e}")

# Test 2: All dependencies available
deps = ["torch", "torchvision", "cv2", "rasterio", "shapely", "hydra", "iopath"]
ultralytics_deps = ["ultralytics"]
for dep in deps + ultralytics_deps:
    try:
        __import__(dep)
        print(f"✅ {dep} available")
    except ImportError:
        print(f"❌ {dep} missing")

# Test 3: Device detection
from geo_osam_dialog import detect_best_device
device, model_choice, cores = detect_best_device()
print(f"🔍 Detected: {device.upper()} → {model_choice}")
if cores:
    print(f"💻 CPU cores configured: {cores}")

# Test 4: Model availability
if model_choice == "Ultralytics SAM2.1_B":
    try:
        from ultralytics import SAM
        test_model = SAM('sam2.1_b.pt')
        print("✅ Ultralytics SAM2.1_B ready (Ultralytics)")
    except Exception as e:
        print(f"⏳ Ultralytics SAM2.1_B will download on first use: {e}")
else:
    import os
    plugin_dir = os.path.dirname(__file__)
    model_path = os.path.join(plugin_dir, "plugins", "geo_osam", "sam2", "checkpoints", "sam2.1_hiera_tiny.pt")
    if os.path.exists(model_path):
        print(f"✅ SAM 2.1 model found: {os.path.getsize(model_path)/1024/1024:.1f}MB")
    else:
        print("⏳ SAM 2.1 model will download on first use")

# Test 5: Performance estimate
if model_choice == "Ultralytics SAM2.1_B" and cores and cores >= 24:
    print("🚀 Expected performance: <1 second per segment")
elif device == "cuda":
    print("🚀 Expected performance: 0.2-0.5 seconds per segment")
elif device == "mps":
    print("🚀 Expected performance: 1-2 seconds per segment")
else:
    print("🚀 Expected performance: 2-5 seconds per segment")
```

---

## 🚨 Troubleshooting Installation

### Common Issues

#### Issue: "Plugin not found in repository"

**Solution:**

- Update QGIS to latest version
- Check plugin repository settings
- Try manual installation from GitHub

#### Issue: "Import error: ultralytics"

**Solution:**

```bash
# Install Ultralytics separately:
pip install "ultralytics>=8.3.237" pillow numpy

# Or reinstall all dependencies:
pip install torch torchvision "ultralytics>=8.3.237" opencv-python rasterio shapely hydra-core iopath pillow numpy
```

#### Issue: "Import error: torch"

**Solution:**

```bash
# Reinstall PyTorch:
pip uninstall torch torchvision
pip install torch torchvision "ultralytics>=8.3.237" iopath pillow numpy
```

#### Issue: "Import error: iopath"

**Solution:**

```bash
# Install iopath separately:
pip install iopath

# Or reinstall all dependencies:
pip install torch torchvision "ultralytics>=8.3.237" opencv-python rasterio shapely hydra-core iopath pillow numpy
```

#### Issue: "Permission denied" (Windows)

**Solution:**

- Run Command Prompt as Administrator
- Or use QGIS Python Console (recommended)

#### Issue: "Ultralytics SAM2.1_B download fails"

**Solution:**

```python
# Test Ultralytics directly in QGIS Python Console:
from ultralytics import SAM
model = SAM('sam2.1_b.pt')  # Should auto-download
```

#### Issue: "SAM 2.1 model download fails"

**Solution:**

```bash
# Manual download for GPU systems:
cd ~/.local/share/QGIS/QGIS3/profiles/default/python/plugins/geo_osam/sam2/checkpoints
wget https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2.1_hiera_tiny.pt
```

#### Issue: "CUDA errors"

**Solution:**

- Check NVIDIA driver version
- Reinstall PyTorch with correct CUDA version
- Plugin will automatically fallback to Ultralytics SAM2.1_B on CPU

#### Issue: "CUDA Not Detected on Windows" (despite nvidia-smi working)

**Symptoms:**

- `nvidia-smi` command works and shows your GPU
- Plugin runs in CPU mode
- `torch.cuda.is_available()` returns `False`
- Plugin shows "💻 CPU" instead of "🎮 CUDA"

**Root Cause:**
PyTorch was installed without CUDA support (CPU-only version). The default `pip install torch` command installs the CPU version, even if you have NVIDIA drivers.

**Solution:**

**Step 1: Verify the issue**

```python
# In QGIS Python Console:
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA built: {torch.version.cuda}")
```

If it shows `CUDA available: False` and `CUDA built: None`, follow these steps:

**Step 2: Uninstall CPU-only PyTorch**

```bash
# In OSGeo4W Shell:
pip uninstall torch torchvision
```

**Step 3: Install PyTorch with CUDA support**

```bash
# For CUDA 11.8 (most compatible with current drivers):
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# OR for CUDA 12.1+ (newer GPUs/drivers):
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

**Step 4: Verify CUDA is now working**

```python
# Restart QGIS, then in Python Console:
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
```

You should now see:

- `CUDA available: True`
- Your GPU name displayed

**Step 5: Test plugin**

- Click GeoOSAM icon
- Control panel should now show "🎮 CUDA | SAM2"
- GPU acceleration active!

#### Issue: "Wrong model selected"

**Solution:**

```python
# Force specific model in QGIS Python Console:
import os
os.environ["GEOOSAM_FORCE_CPU"] = "1"  # Force CPU/Ultralytics SAM2.1_B
# Restart QGIS
```

### Device-Specific Troubleshooting

#### High-Core CPU Not Optimized

```python
# Check threading configuration:
import torch
import os
print(f"PyTorch threads: {torch.get_num_threads()}")
print(f"OMP threads: {os.environ.get('OMP_NUM_THREADS', 'not set')}")

# Should show 75% of your CPU cores for 16+ core systems
```

#### Apple Silicon Issues

```bash
# Ensure native ARM packages:
pip uninstall torch torchvision ultralytics
pip install torch torchvision "ultralytics>=8.3.237" iopath pillow numpy
```

### Getting Help

#### Before Asking for Help

1. **Run verification tests** above
2. **Check device detection** results
3. **Test with fresh QGIS installation**
4. **Verify internet connection** for downloads

#### Support Channels

- **GitHub Issues:** https://github.com/espressouk/GeoOSAM/issues
- **Email:** bkst.dev@gmail.com
- **QGIS Community:** https://qgis.org/en/site/forusers/support.html

#### Bug Reports

Include this information:

- Operating System and version
- Hardware specs (GPU, CPU cores)
- QGIS version
- Python version
- Device detection results (from verification script)
- Full error messages
- Steps to reproduce

---

## 🔄 Updates and Maintenance

### Updating GeoOSAM

```bash
# From QGIS Plugin Repository:
# Plugins → Manage and Install Plugins → Upgradeable → Upgrade GeoOSAM

# Manual update from GitHub:
cd geoOSAM
git pull origin main
# Or download new release
```

### Keeping Dependencies Updated

```bash
# Update Python packages:
pip install --upgrade torch torchvision "ultralytics>=8.3.237" opencv-python rasterio shapely hydra-core iopath pillow numpy
```

### Model Updates

- **Ultralytics SAM2.1_B**: Automatically updated via Ultralytics
- **SAM 2.1**: Plugin checks for newer checkpoints
- **Automatic**: Models update seamlessly in background

### Uninstallation

```bash
# Remove plugin:
# Plugins → Manage and Install Plugins → Installed → GeoOSAM → Uninstall

# Remove dependencies (optional):
pip uninstall torch torchvision ultralytics opencv-python rasterio shapely hydra-core iopath

# Remove data (optional):
rm -rf ~/GeoOSAM_shapefiles ~/GeoOSAM_masks
```

---

**Installation complete! Your system will automatically use the optimal AI model for your hardware.** 🚀

- **GPU Users**: Enjoy SAM 2.1's cutting-edge accuracy
- **CPU Users**: Experience Ultralytics SAM2.1_B's remarkable efficiency
- **High-End CPU**: Get sub-second performance rivaling GPUs

See [User Guide](user_guide.md) for next steps.
