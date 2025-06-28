# GeoOSAM Installation Guide

## 🎯 Quick Installation (Recommended)

### Step 1: Install from QGIS Plugin Repository

1. **Open QGIS** (version 3.16 or later)
2. **Menu:** Plugins → Manage and Install Plugins
3. **Search:** Type "GeoOSAM"
4. **Install:** Click "Install Plugin"
5. **Enable:** Ensure plugin is checked

### Step 2: Install Dependencies

```python
# Open QGIS → Plugins → Python Console
# Copy and paste this code:
import subprocess, sys
packages = ["torch", "torchvision", "ultralytics", "opencv-python", "rasterio", "shapely", "hydra-core"]
for pkg in packages:
    subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])
    print(f"✅ Installed {pkg}")
```

### Step 3: First Use

1. **Click GeoOSAM icon** 🛰️ in QGIS toolbar
2. **Models auto-download** based on your hardware:
   - **GPU Systems**: SAM 2.1 (~160MB, one-time)
   - **CPU Systems**: MobileSAM (~40MB via Ultralytics, automatic)
3. **Start segmenting!** 🚀

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

```powershell
# 1. Install plugin through QGIS interface
# 2. Install dependencies via Command Prompt (as Administrator):
pip install torch torchvision ultralytics opencv-python rasterio shapely hydra-core

# Alternative: Use QGIS Python Console (recommended)
# Open QGIS → Plugins → Python Console
import subprocess, sys
packages = ["torch", "torchvision", "ultralytics", "opencv-python", "rasterio", "shapely", "hydra-core"]
for pkg in packages: subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])
```

#### macOS Installation

```bash
# 1. Install plugin through QGIS interface
# 2. Install dependencies via Terminal:
pip3 install torch torchvision ultralytics opencv-python rasterio shapely hydra-core

# For Apple Silicon Macs (automatic optimization):
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip3 install ultralytics opencv-python rasterio shapely hydra-core

# Alternative: Use QGIS Python Console (recommended)
```

#### Linux Installation

```bash
# 1. Install plugin through QGIS interface
# 2. Install dependencies:
pip3 install torch torchvision ultralytics opencv-python rasterio shapely hydra-core

# Ubuntu/Debian additional dependencies:
sudo apt update
sudo apt install python3-pip python3-dev

# NVIDIA GPU support (auto-detected):
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip3 install ultralytics opencv-python rasterio shapely hydra-core
```

### Installation Method 2: Manual GitHub Installation

#### Download and Extract

```bash
# 1. Download plugin from GitHub
wget https://github.com/espressouk/geoOSAM/archive/main.zip
unzip main.zip
cd geo-osam-main

# Or clone with git:
git clone https://github.com/espressouk/geoOSAM.git
cd geo-osam
```

#### Copy to QGIS Plugins Directory

**Windows:**

```powershell
# Copy plugin to QGIS plugins folder:
xcopy geo_osam "C:\Users\%USERNAME%\AppData\Roaming\QGIS\QGIS3\profiles\default\python\plugins\geo_osam" /E /I
```

**macOS:**

```bash
# Copy plugin to QGIS plugins folder:
cp -r geo_osam ~/Library/Application\ Support/QGIS/QGIS3/profiles/default/python/plugins/
```

**Linux:**

```bash
# Copy plugin to QGIS plugins folder:
cp -r geo_osam ~/.local/share/QGIS/QGIS3/profiles/default/python/plugins/
```

#### Install Dependencies

```bash
# Install required Python packages:
pip3 install torch torchvision ultralytics opencv-python rasterio shapely hydra-core
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

| Hardware Detected        | Model Selected | Download Size | Performance |
| ------------------------ | -------------- | ------------- | ----------- |
| NVIDIA GPU (CUDA)        | SAM 2.1        | ~160MB        | 0.2-0.5s    |
| Apple Silicon (M1/M2/M3) | SAM 2.1        | ~160MB        | 1-2s        |
| 24+ Core CPU             | MobileSAM      | ~40MB         | <1s         |
| 16+ Core CPU             | MobileSAM      | ~40MB         | 1-2s        |
| 8-16 Core CPU            | MobileSAM      | ~40MB         | 2-3s        |
| 4-8 Core CPU             | MobileSAM      | ~40MB         | 3-5s        |

### Download Process

**🔄 What Happens Automatically:**

1. **Device Detection**: Plugin detects GPU/CPU capabilities
2. **Model Selection**: Chooses SAM 2.1 (GPU) or MobileSAM (CPU)
3. **Smart Download**: Only downloads the model you need
4. **Ultralytics Magic**: MobileSAM handled seamlessly by Ultralytics
5. **One-time Setup**: Subsequent uses are instant

**📥 Download Details:**

- **GPU Systems**: Downloads SAM 2.1 checkpoint directly
- **CPU Systems**: Ultralytics automatically downloads MobileSAM
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
pip3 install ultralytics

# Verify CUDA in QGIS Python Console:
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")
```

#### Apple Silicon (M1/M2/M3) - Auto-Detected

```bash
# Install optimized PyTorch for Apple Silicon:
pip3 install torch torchvision ultralytics

# Verify MPS support in QGIS Python Console:
import torch
print(f"MPS available: {torch.backends.mps.is_available()}")
```

#### High-Performance CPU Systems

```bash
# For 16+ core systems (auto-optimized):
pip3 install torch torchvision ultralytics

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
cd geo-osam

# Create development environment:
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Install all dependencies:
pip install torch torchvision ultralytics opencv-python rasterio shapely hydra-core

# Link to QGIS plugins directory:
ln -s $(pwd)/geo_osam ~/.local/share/QGIS/QGIS3/profiles/default/python/plugins/geo_osam
```

### Docker Installation (Advanced)

```dockerfile
# Dockerfile for containerized QGIS with GeoOSAM
FROM qgis/qgis:release-3_28

# Install dependencies with Ultralytics
RUN pip3 install torch torchvision ultralytics opencv-python rasterio shapely hydra-core

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
deps = ["torch", "torchvision", "cv2", "rasterio", "shapely", "hydra"]
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
if model_choice == "MobileSAM":
    try:
        from ultralytics import SAM
        test_model = SAM('mobile_sam.pt')
        print("✅ MobileSAM ready (Ultralytics)")
    except Exception as e:
        print(f"⏳ MobileSAM will download on first use: {e}")
else:
    import os
    plugin_dir = os.path.dirname(__file__)
    model_path = os.path.join(plugin_dir, "plugins", "geo_osam", "sam2", "checkpoints", "sam2_hiera_tiny.pt")
    if os.path.exists(model_path):
        print(f"✅ SAM 2.1 model found: {os.path.getsize(model_path)/1024/1024:.1f}MB")
    else:
        print("⏳ SAM 2.1 model will download on first use")

# Test 5: Performance estimate
if model_choice == "MobileSAM" and cores and cores >= 24:
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
pip install ultralytics

# Or reinstall all dependencies:
pip install torch torchvision ultralytics opencv-python rasterio shapely hydra-core
```

#### Issue: "Import error: torch"

**Solution:**

```bash
# Reinstall PyTorch:
pip uninstall torch torchvision
pip install torch torchvision ultralytics
```

#### Issue: "Permission denied" (Windows)

**Solution:**

- Run Command Prompt as Administrator
- Or use QGIS Python Console (recommended)

#### Issue: "MobileSAM download fails"

**Solution:**

```python
# Test Ultralytics directly in QGIS Python Console:
from ultralytics import SAM
model = SAM('mobile_sam.pt')  # Should auto-download
```

#### Issue: "SAM 2.1 model download fails"

**Solution:**

```bash
# Manual download for GPU systems:
cd ~/.local/share/QGIS/QGIS3/profiles/default/python/plugins/geo_osam/sam2/checkpoints
wget https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_tiny.pt
```

#### Issue: "CUDA errors"

**Solution:**

- Check NVIDIA driver version
- Reinstall PyTorch with correct CUDA version
- Plugin will automatically fallback to MobileSAM on CPU

#### Issue: "Wrong model selected"

**Solution:**

```python
# Force specific model in QGIS Python Console:
import os
os.environ["GEOOSAM_FORCE_CPU"] = "1"  # Force CPU/MobileSAM
# Restart QGIS
```

### Device-Specific Troubleshooting

#### High-Core CPU Not Optimized

```python
# Check threading configuration:
import torch
print(f"PyTorch threads: {torch.get_num_threads()}")
print(f"OMP threads: {os.environ.get('OMP_NUM_THREADS', 'not set')}")

# Should show 75% of your CPU cores for 16+ core systems
```

#### Apple Silicon Issues

```bash
# Ensure native ARM packages:
pip uninstall torch torchvision ultralytics
pip install torch torchvision ultralytics
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
cd geo_osam
git pull origin main
# Or download new release
```

### Keeping Dependencies Updated

```bash
# Update Python packages:
pip install --upgrade torch torchvision ultralytics opencv-python rasterio shapely hydra-core
```

### Model Updates

- **MobileSAM**: Automatically updated via Ultralytics
- **SAM 2.1**: Plugin checks for newer checkpoints
- **Automatic**: Models update seamlessly in background

### Uninstallation

```bash
# Remove plugin:
# Plugins → Manage and Install Plugins → Installed → GeoOSAM → Uninstall

# Remove dependencies (optional):
pip uninstall torch torchvision ultralytics opencv-python rasterio shapely hydra-core

# Remove data (optional):
rm -rf ~/GeoOSAM_shapefiles ~/GeoOSAM_masks
```

---

**Installation complete! Your system will automatically use the optimal AI model for your hardware.** 🚀

- **GPU Users**: Enjoy SAM 2.1's cutting-edge accuracy
- **CPU Users**: Experience MobileSAM's remarkable efficiency
- **High-End CPU**: Get sub-second performance rivaling GPUs

See [User Guide](user_guide.md) for next steps.
