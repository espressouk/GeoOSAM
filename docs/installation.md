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
packages = ["torch", "torchvision", "opencv-python", "rasterio", "shapely", "hydra-core"]
for pkg in packages:
    subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])
    print(f"✅ Installed {pkg}")
```

### Step 3: First Use

1. **Click GeoOSAM icon** 🛰️ in QGIS toolbar
2. **SAM2 model auto-downloads** (~160MB, one-time)
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
- **Internet:** For model download

#### Recommended Requirements

- **Operating System:** Windows 11, macOS 12+, Ubuntu 20.04+
- **QGIS Version:** 3.28 or later
- **Python:** 3.9 or later
- **RAM:** 16GB or more
- **GPU:** NVIDIA GPU with CUDA or Apple Silicon
- **Storage:** SSD with 4GB free space

### Installation Method 1: QGIS Plugin Repository

#### Windows Installation

```powershell
# 1. Install plugin through QGIS interface
# 2. Install dependencies via Command Prompt (as Administrator):
pip install torch torchvision opencv-python rasterio shapely hydra-core

# Alternative: Use QGIS Python Console (recommended)
# Open QGIS → Plugins → Python Console
import subprocess, sys
packages = ["torch", "torchvision", "opencv-python", "rasterio", "shapely", "hydra-core"]
for pkg in packages: subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])
```

#### macOS Installation

```bash
# 1. Install plugin through QGIS interface
# 2. Install dependencies via Terminal:
pip3 install torch torchvision opencv-python rasterio shapely hydra-core

# For Apple Silicon Macs:
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip3 install opencv-python rasterio shapely hydra-core

# Alternative: Use QGIS Python Console (recommended)
```

#### Linux Installation

```bash
# 1. Install plugin through QGIS interface
# 2. Install dependencies:
pip3 install torch torchvision opencv-python rasterio shapely hydra-core

# Ubuntu/Debian additional dependencies:
sudo apt update
sudo apt install python3-pip python3-dev

# NVIDIA GPU support:
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### Installation Method 2: Manual GitHub Installation

#### Download and Extract

```bash
# 1. Download plugin from GitHub
wget https://github.com/espressouk/geo-osam/archive/main.zip
unzip main.zip
cd geo-osam-main

# Or clone with git:
git clone https://github.com/espressouk/geo-osam.git
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
pip3 install torch torchvision opencv-python rasterio shapely hydra-core
```

#### Enable Plugin

1. **Open QGIS**
2. **Go to:** Plugins → Manage and Install Plugins
3. **Click:** Installed tab
4. **Find:** GeoOSAM
5. **Check:** Enable checkbox

---

## 🔧 Advanced Installation Options

### GPU Acceleration Setup

#### NVIDIA GPU (CUDA)

```bash
# Check CUDA availability:
nvidia-smi

# Install PyTorch with CUDA support (11.8):
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Verify CUDA in QGIS Python Console:
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")
```

#### Apple Silicon (M1/M2)

```bash
# Install optimized PyTorch for Apple Silicon:
pip3 install torch torchvision

# Verify MPS support in QGIS Python Console:
import torch
print(f"MPS available: {torch.backends.mps.is_available()}")
```

### Development Installation

#### For Plugin Development

```bash
# Clone repository:
git clone https://github.com/espressouk/geo-osam.git
cd geo-osam

# Create development environment:
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Install development dependencies:
pip install -r requirements-dev.txt

# Link to QGIS plugins directory:
ln -s $(pwd)/geo_osam ~/.local/share/QGIS/QGIS3/profiles/default/python/plugins/geo_osam
```

### Docker Installation (Advanced)

```dockerfile
# Dockerfile for containerized QGIS with GeoOSAM
FROM qgis/qgis:release-3_28

# Install dependencies
RUN pip3 install torch torchvision opencv-python rasterio shapely hydra-core

# Copy plugin
COPY geo_osam /root/.local/share/QGIS/QGIS3/profiles/default/python/plugins/geo_osam

# Download SAM2 model
RUN cd /root/.local/share/QGIS/QGIS3/profiles/default/python/plugins/geo_osam/sam2/checkpoints && \
    wget https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_tiny.pt
```

---

## ✅ Installation Verification

### Quick Test

1. **Open QGIS**
2. **Look for:** GeoOSAM icon 🛰️ in toolbar
3. **Click icon:** Control panel should open
4. **Check status:** Should show "Ready to segment"

### Detailed Verification

```python
# Run in QGIS Python Console:

# Test 1: Plugin loads
try:
    from geo_osam import SegSam
    print("✅ Plugin import successful")
except Exception as e:
    print(f"❌ Plugin import failed: {e}")

# Test 2: Dependencies available
deps = ["torch", "torchvision", "cv2", "rasterio", "shapely", "hydra"]
for dep in deps:
    try:
        __import__(dep)
        print(f"✅ {dep} available")
    except ImportError:
        print(f"❌ {dep} missing")

# Test 3: SAM2 model
import os
plugin_dir = os.path.dirname(__file__)
model_path = os.path.join(plugin_dir, "plugins", "geo_osam", "sam2", "checkpoints", "sam2_hiera_tiny.pt")
if os.path.exists(model_path):
    print(f"✅ SAM2 model found: {os.path.getsize(model_path)/1024/1024:.1f}MB")
else:
    print("⏳ SAM2 model will download on first use")

# Test 4: Device detection
import torch
if torch.cuda.is_available():
    print(f"🎮 GPU: {torch.cuda.get_device_name(0)}")
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    print("🍎 Apple Silicon GPU available")
else:
    print("💻 CPU mode")
```

---

## 🚨 Troubleshooting Installation

### Common Issues

#### Issue: "Plugin not found in repository"

**Solution:**

- Update QGIS to latest version
- Check plugin repository settings
- Try manual installation from GitHub

#### Issue: "Import error: torch"

**Solution:**

```bash
# Reinstall PyTorch:
pip uninstall torch torchvision
pip install torch torchvision
```

#### Issue: "Permission denied" (Windows)

**Solution:**

- Run Command Prompt as Administrator
- Or use QGIS Python Console (recommended)

#### Issue: "SAM2 model download fails"

**Solution:**

```bash
# Manual download:
cd ~/.local/share/QGIS/QGIS3/profiles/default/python/plugins/geo_osam/sam2/checkpoints
wget https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_tiny.pt
```

#### Issue: "CUDA errors"

**Solution:**

- Check NVIDIA driver version
- Reinstall PyTorch with correct CUDA version
- Plugin will fallback to CPU if needed

### Getting Help

#### Before Asking for Help

1. **Run verification tests** above
2. **Check QGIS version** (must be 3.16+)
3. **Try fresh QGIS installation**
4. **Read error messages** carefully

#### Support Channels

- **GitHub Issues:** https://github.com/espressouk/GeoOSAM/issues
- **Email:** ofer@butbega.com
- **QGIS Community:** https://qgis.org/en/site/forusers/support.html

#### Bug Reports

Include this information:

- Operating System and version
- QGIS version
- Python version
- Error messages (full text)
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
pip install --upgrade torch torchvision opencv-python rasterio shapely hydra-core
```

### Uninstallation

```bash
# Remove plugin:
# Plugins → Manage and Install Plugins → Installed → GeoOSAM → Uninstall

# Remove dependencies (optional):
pip uninstall torch torchvision opencv-python rasterio shapely hydra-core

# Remove data (optional):
rm -rf ~/GeoOSAM_shapefiles ~/GeoOSAM_masks
```

---

**Installation complete! Ready to start segmenting!** 🚀

See [User Guide](user_guide.md) for next steps.
