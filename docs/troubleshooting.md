# GeoOSAM Troubleshooting Guide

## 🚨 Quick Fixes

### Plugin Won't Load

```python
# Run in QGIS Python Console to check:
import sys
print("Python version:", sys.version)
print("QGIS version:", qgis.core.Qgis.QGIS_VERSION)

# Check if plugin directory exists:
import os
plugin_path = os.path.expanduser("~/.local/share/QGIS/QGIS3/profiles/default/python/plugins/geo_osam")
print("Plugin exists:", os.path.exists(plugin_path))
```

### SAM2 Model Issues

```bash
# Check model file:
ls -la ~/.local/share/QGIS/QGIS3/profiles/default/python/plugins/geo_osam/sam2/checkpoints/

# Re-download if corrupted:
cd ~/.local/share/QGIS/QGIS3/profiles/default/python/plugins/geo_osam/sam2/checkpoints/
rm sam2_hiera_tiny.pt
bash download_sam2_checkpoints.sh
```

### Segmentation Not Working

1. **Check raster layer is selected** (not vector)
2. **Zoom to appropriate level** (not too far out)
3. **Try different click position** (center of object)
4. **Check image quality** (contrast, resolution)

---

## 📋 Common Error Messages

### Installation Errors

#### Error: "No module named 'torch'"

**Cause:** PyTorch not installed  
**Solution:**

```python
# In QGIS Python Console:
import subprocess, sys
subprocess.check_call([sys.executable, "-m", "pip", "install", "torch", "torchvision"])
```

#### Error: "Permission denied"

**Cause:** Insufficient permissions (Windows)  
**Solution:**

- Use QGIS Python Console instead of Command Prompt
- Or run Command Prompt as Administrator

#### Error: "Plugin could not be loaded"

**Cause:** Missing dependencies or corrupted files  
**Solution:**

```python
# Check all dependencies:
deps = ["torch", "torchvision", "cv2", "rasterio", "shapely", "hydra"]
missing = []
for dep in deps:
    try:
        __import__(dep)
        print(f"✅ {dep}")
    except ImportError:
        missing.append(dep)
        print(f"❌ {dep} MISSING")

if missing:
    import subprocess, sys
    for pkg in missing:
        pkg_name = "opencv-python" if pkg == "cv2" else "hydra-core" if pkg == "hydra" else pkg
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg_name])
```

### Runtime Errors

#### Error: "CUDA out of memory"

**Cause:** GPU memory insufficient  
**Solution:**

- Plugin automatically falls back to CPU
- Close other GPU-intensive applications
- Restart QGIS

#### Error: "No raster layer selected"

**Cause:** Vector layer active or no layer loaded  
**Solution:**

```python
# Check active layer type:
layer = iface.activeLayer()
print(f"Active layer: {layer.name() if layer else 'None'}")
print(f"Layer type: {type(layer).__name__ if layer else 'None'}")

# Select a raster layer:
from qgis.core import QgsRasterLayer
for layer in QgsProject.instance().mapLayers().values():
    if isinstance(layer, QgsRasterLayer):
        iface.setActiveLayer(layer)
        print(f"Selected raster: {layer.name()}")
        break
```

#### Error: "SAM model failed to load"

**Cause:** Corrupted model file or wrong path  
**Solution:**

```bash
# Re-download SAM2 model:
cd ~/.local/share/QGIS/QGIS3/profiles/default/python/plugins/geo_osam/sam2/checkpoints/
rm -f sam2_hiera_tiny.pt
wget https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_tiny.pt

# Or use the download script:
bash download_sam2_checkpoints.sh
```

### Segmentation Errors

#### Error: "Empty crop area"

**Cause:** Clicked outside raster bounds or invalid coordinates  
**Solution:**

- Ensure click is within the raster image
- Check raster CRS matches project CRS
- Zoom to raster extent

#### Error: "No segments found"

**Cause:** Poor image quality or inappropriate settings  
**Solution:**

- Try different click position (center vs. edge)
- Switch between Point and BBox modes
- Zoom in for better resolution
- Check image contrast

#### Error: "Segmentation timeout"

**Cause:** Processing taking too long (CPU mode)  
**Solution:**

- Reduce crop size by zooming in
- Use GPU if available
- Close other applications

---

## 🔧 Platform-Specific Issues

### Windows Issues

#### Issue: "DLL load failed"

**Cause:** Missing Visual C++ redistributables  
**Solution:**

- Install Visual C++ Redistributable for Visual Studio 2019
- Download from Microsoft website

#### Issue: "pip is not recognized"

**Cause:** Python/pip not in PATH  
**Solution:**

```python
# Use QGIS Python Console instead:
import subprocess, sys
subprocess.check_call([sys.executable, "-m", "pip", "install", "package_name"])
```

#### Issue: Long path names on Windows

**Cause:** Windows path length limitations  
**Solution:**

- Move QGIS installation to shorter path (e.g., C:\QGIS)
- Enable long path support in Windows

### macOS Issues

#### Issue: "Killed: 9" error

**Cause:** macOS Gatekeeper blocking unsigned code  
**Solution:**

```bash
# Allow QGIS to run downloaded plugins:
sudo xattr -rd com.apple.quarantine /Applications/QGIS.app
```

#### Issue: "SSL certificate verify failed"

**Cause:** Corporate firewall or proxy  
**Solution:**

```bash
# Download model manually:
curl -k https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_tiny.pt -o sam2_hiera_tiny.pt
```

#### Issue: Apple Silicon compatibility

**Cause:** x86 PyTorch on ARM processor  
**Solution:**

```bash
# Install native ARM PyTorch:
pip uninstall torch torchvision
pip install torch torchvision
```

### Linux Issues

#### Issue: "libGL.so.1: cannot open shared file"

**Cause:** Missing OpenGL libraries  
**Solution:**

```bash
# Ubuntu/Debian:
sudo apt install libgl1-mesa-glx libglib2.0-0

# CentOS/RHEL:
sudo yum install mesa-libGL
```

#### Issue: "Permission denied" for pip

**Cause:** System Python permissions  
**Solution:**

```bash
# Use user installation:
pip install --user torch torchvision opencv-python rasterio shapely hydra-core
```

#### Issue: NVIDIA driver conflicts

**Cause:** Multiple CUDA versions  
**Solution:**

```bash
# Check CUDA version:
nvidia-smi

# Install matching PyTorch:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

---

## 🐛 Performance Issues

### Slow Segmentation

#### Diagnosis

```python
# Check device being used:
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"MPS available: {torch.backends.mps.is_available() if hasattr(torch.backends, 'mps') else False}")

# Check processing time:
import time
start = time.time()
# ... perform segmentation ...
print(f"Processing time: {time.time() - start:.2f} seconds")
```

#### Solutions

- **Enable GPU acceleration** (CUDA/MPS)
- **Reduce image resolution** (zoom in)
- **Close other applications**
- **Use faster storage** (SSD)
- **Increase RAM** if possible

### Memory Issues

#### High RAM Usage

**Solution:**

- Process smaller areas at a time
- Close other applications
- Restart QGIS periodically
- Use smaller crop sizes

#### GPU Memory Errors

**Solution:**

```python
# Plugin should auto-fallback to CPU, but if not:
import torch
torch.cuda.empty_cache()  # Clear GPU memory
```

### UI Freezing

#### Cause

Long processing operations blocking main thread

#### Solution

- Plugin uses threading to prevent this
- If UI freezes, restart QGIS
- Reduce processing complexity

---

## 🔍 Debugging Tools

### Enable Debug Mode

```python
# In QGIS Python Console:
import logging
logging.basicConfig(level=logging.DEBUG)

# Enable debug mask saving:
# Check "💾 Save debug masks" in control panel
```

### Diagnostic Script

```python
# Comprehensive system check:
import os, sys, torch, cv2, rasterio, shapely, hydra
from qgis.core import QgsApplication, QgsRasterLayer, QgsProject

print("=== GeoOSAM Diagnostic Report ===")
print(f"Python: {sys.version}")
print(f"QGIS: {QgsApplication.version()}")
print(f"PyTorch: {torch.__version__}")
print(f"OpenCV: {cv2.__version__}")
print(f"Rasterio: {rasterio.__version__}")
print(f"Shapely: {shapely.__version__}")
print(f"Hydra: {hydra.__version__}")

# Hardware info:
print(f"CPU cores: {os.cpu_count()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    print("GPU: Apple Silicon (MPS)")
else:
    print("GPU: None (CPU only)")

# Plugin status:
plugin_path = os.path.expanduser("~/.local/share/QGIS/QGIS3/profiles/default/python/plugins/geo_osam")
print(f"Plugin installed: {os.path.exists(plugin_path)}")

model_path = os.path.join(plugin_path, "sam2", "checkpoints", "sam2_hiera_tiny.pt")
if os.path.exists(model_path):
    print(f"SAM2 model: {os.path.getsize(model_path)/1024/1024:.1f}MB")
else:
    print("SAM2 model: Missing")

# Active layers:
layers = QgsProject.instance().mapLayers()
print(f"Loaded layers: {len(layers)}")
for name, layer in layers.items():
    print(f"  {layer.name()}: {type(layer).__name__}")

active = iface.activeLayer()
print(f"Active layer: {active.name() if active else 'None'}")
print("=== End Report ===")
```

### Log File Locations

#### Windows

```
C:\Users\[USERNAME]\AppData\Roaming\QGIS\QGIS3\profiles\default\python\plugins\geo_osam\logs\
```

#### macOS

```
~/Library/Application Support/QGIS/QGIS3/profiles/default/python/plugins/geo_osam/logs/
```

#### Linux

```
~/.local/share/QGIS/QGIS3/profiles/default/python/plugins/geo_osam/logs/
```

---

## 📞 Getting Help

### Before Reporting Issues

#### Collect Information

1. **Run diagnostic script** (above)
2. **Note exact error messages**
3. **Document steps to reproduce**
4. **Check QGIS and plugin versions**
5. **Test with different imagery**

#### Try These First

- **Restart QGIS**
- **Reload the plugin**
- **Test with sample data**
- **Check internet connection** (for model download)

### Reporting Bugs

#### GitHub Issues

Create issue at: https://github.com/espressouk/geo-osam/issues

#### Include This Information

```
**Environment:**
- OS: [Windows 11 / macOS 12 / Ubuntu 20.04]
- QGIS Version: [3.28.12]
- Plugin Version: [1.0.0]
- Python Version: [3.9.16]

**Hardware:**
- RAM: [16GB]
- GPU: [NVIDIA RTX 4090 / Apple M2 / None]
- Storage: [SSD / HDD]

**Error:**
[Paste full error message]

**Steps to Reproduce:**
1. Load raster layer
2. Click GeoOSAM icon
3. Select Buildings class
4. Click Point mode
5. Click on building
6. Error occurs

**Expected Behavior:**
Should segment the building

**Actual Behavior:**
Shows error: "No segments found"

**Additional Context:**
- Works with some images but not others
- Imagery: Sentinel-2, 10m resolution
- File format: GeoTIFF
```

### Emergency Solutions

#### Plugin Completely Broken

```python
# Disable plugin:
# Plugins → Manage and Install Plugins → Installed → GeoOSAM → Uncheck

# Reset plugin settings:
import os
settings_path = os.path.expanduser("~/.local/share/QGIS/QGIS3/profiles/default/QGIS/QGIS3.ini")
# Edit file and remove [geo_osam] section

# Fresh installation:
# Uninstall and reinstall plugin
```

#### QGIS Won't Start

```bash
# Reset QGIS configuration:
# Backup and remove QGIS3 folder, then restart QGIS
mv ~/.local/share/QGIS/QGIS3 ~/.local/share/QGIS/QGIS3_backup
```

---

## 🔄 Updates and Patches

### Staying Updated

- **Check for updates** in QGIS Plugin Manager
- **Watch GitHub releases** for announcements
- **Subscribe to issues** for bug fixes

### Beta Testing

- **Join beta program** for early access to fixes
- **Test with your specific workflows**
- **Report feedback** to improve stability

**Still having issues? Don't hesitate to reach out!** 📧

Email: ofer@butbega.com  
GitHub: https://github.com/espressouk/geo-osam/issues
