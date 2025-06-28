# GeoOSAM - Advanced Segmentation for QGIS

🛰️ **State-of-the-art image segmentation using Meta's SAM 2.1 and MobileSAM with intelligent hardware optimization**

[![QGIS Plugin](https://img.shields.io/badge/QGIS-Plugin-green)](https://plugins.qgis.org)
[![Python](https://img.shields.io/badge/Python-3.7+-blue)](https://python.org)
[![License](https://img.shields.io/badge/License-GPL%20v2-blue.svg)](LICENSE)

## 🌟 Features

- **🧠 Intelligent Model Selection**: Automatically chooses the best AI model for your hardware
- **🚀 Optimized Performance**: SAM 2.1 for GPU, MobileSAM for CPU
- **🎯 Dual Modes**: Point-click and bounding box segmentation
- **📋 12 Pre-defined Classes**: Buildings, Roads, Vegetation, Water, Vehicle, Ship, and more
- **↶ Undo Support**: Mistake correction with polygon-level undo
- **📁 Custom Output**: User-selectable output folders
- **🎨 Class Management**: Custom classes with color coding
- **📡 Smart Workflow**: Auto-raster selection, progress tracking
- **💾 Professional Export**: Shapefile export with detailed attributes
- **🔧 Adaptive Processing**: Optimized based on zoom level and hardware

## 📊 Performance & Model Selection

| Hardware       | Model Used | Typical Speed | Improvement       |
| -------------- | ---------- | ------------- | ----------------- |
| NVIDIA RTX GPU | SAM 2.1    | 0.2-0.5s      | **10-50x faster** |
| Apple M1/M2    | SAM 2.1    | 1-2s          | **5-15x faster**  |
| Multi-core CPU | MobileSAM  | 2-4s          | **5-10x faster**  |

**🎯 Smart Model Selection:**

- **GPU Available** (CUDA/Apple Silicon) → **SAM 2.1** (latest accuracy)
- **CPU Only** → **MobileSAM** (5x smaller, 7x faster than SAM)
- **Automatic Fallback** → SAM 2 if MobileSAM unavailable

## 🚀 Quick Start

### 1. Install Plugin

Enable in QGIS Plugin Manager or download from GitHub

### 2. Load Imagery & Select Class

![Main Interface](screenshots/main_interface.png)

### 3. Point & Click to Segment

Select a class (Buildings, Water, etc.) and click on objects in your imagery

### 4. View Results

![Segmentation Results](screenshots/results_view.png)

### 5. Export Professional Shapefiles

![Export Functionality](screenshots/export_shape.png)
_Export segmented polygons as shapefiles with detailed attributes_

## 📋 System Requirements

### Minimum

- QGIS 3.16+
- Python 3.7+
- 8GB RAM
- 2GB disk space

### Recommended

- QGIS 3.28+
- NVIDIA GPU with CUDA or Apple Silicon
- 16GB+ RAM
- SSD storage

## 📦 Installation

**⚠️ Important: Both installation methods require manual dependency installation**

### Option 1: From QGIS Plugin Repository (Recommended)

1. Open QGIS
2. Go to **Plugins > Manage and Install Plugins**
3. Search "GeoOSAM"
4. Click **Install Plugin**
5. **Install dependencies** (see below)

### Option 2: Download from GitHub (Manual)

1. Download ZIP from: https://github.com/espressouk/GeoOSAM
2. Extract the plugin
3. Copy to QGIS plugins directory:

   ```bash
   # Linux/Mac
   cp -r geo_osam ~/.local/share/QGIS/QGIS3/profiles/default/python/plugins/

   # Windows
   xcopy geo_osam "C:\Users\%USERNAME%\AppData\Roaming\QGIS\QGIS3\profiles\default\python\plugins\geo_osam" /E /I
   ```

4. **Install dependencies** (see below)

### Required Dependencies (Both Options)

**Choose one method:**

**Method A: QGIS Python Console (Recommended)**

```python
# Open QGIS > Plugins > Python Console, paste and run:
import subprocess
import sys
packages = ["torch", "torchvision", "ultralytics", "opencv-python", "rasterio", "shapely", "hydra-core"]
for pkg in packages:
    subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])
    print(f"✅ Installed {pkg}")
```

**Method B: System Terminal**

```bash
pip install torch torchvision ultralytics opencv-python rasterio shapely hydra-core
```

### Model Download (Automatic)

**Both SAM 2.1 and MobileSAM models (~200MB total) will be automatically downloaded when you first use the plugin.**

The plugin intelligently downloads only the models needed for your hardware:

- **GPU systems**: Downloads SAM 2.1 checkpoint
- **CPU systems**: Downloads MobileSAM via Ultralytics
- **Automatic fallback**: SAM 2 if other models fail

If auto-download fails, manually run:

```bash
# For SAM 2.1 (GPU users)
cd ~/.local/share/QGIS/QGIS3/profiles/default/python/plugins/geo_osam/sam2/checkpoints/
bash download_sam2_checkpoints.sh

# For MobileSAM (CPU users) - automatic via Ultralytics
# No manual download needed
```

## 🎯 Use Cases

- **🏙️ Urban Planning**: Building and infrastructure mapping
- **🌱 Environmental Monitoring**: Vegetation and land cover analysis
- **🚗 Transportation**: Vehicle and traffic analysis
- **🌊 Coastal Studies**: Ship detection and water body mapping
- **🏗️ Construction**: Site monitoring and progress tracking
- **📡 Remote Sensing**: Large-scale imagery analysis

## ⚙️ Technical Details

### Model Architecture

- **SAM 2.1**: Latest from Meta AI with improved accuracy for small objects
- **MobileSAM**: Lightweight version with Tiny-ViT encoder (5M vs 632M parameters)
- **Automatic Selection**: Based on available GPU memory and compute capability

### Performance Optimization

- **Multi-threaded Processing**: Responsive UI during inference
- **Adaptive Crop Sizes**: Zoom-level aware processing
- **Memory Management**: Efficient handling of large imagery
- **Device Detection**: Automatic CUDA/MPS/CPU optimization

## 📚 Documentation

- [User Guide](docs/user_guide.md)
- [Installation Guide](docs/installation.md)
- [Troubleshooting](docs/troubleshooting.md)
- [API Reference](docs/api.md)

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md).

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/espressouk/GeoOSAM/issues)
- **Email**: bkst.dev@gmail.com
- **Documentation**: [Wiki](https://github.com/espressouk/GeoOSAM/wiki)

## 🙏 Acknowledgments

- **Meta AI**: For the Segment Anything Model (SAM 2.1)
- **Ultralytics**: For MobileSAM integration and optimization
- **QGIS Community**: For the excellent GIS platform
- **PyTorch Team**: For the deep learning framework

## 📄 License

This project is licensed under the GPL v2 License - see the [LICENSE](LICENSE) file for details.

## 🏆 Citation

If you use GeoOSAM in your research, please cite:

```bibtex
@software{geosam2025,
  title={GeoOSAM: Advanced Segmentation for QGIS with Intelligent Model Selection},
  author={Butbega, Ofer},
  year={2025},
  url={https://github.com/espressouk/GeoOSAM}
}
```

## 🔄 Changelog

### v1.0.0 - Latest

- **Intelligent Model Selection**: Automatic SAM 2.1 vs MobileSAM selection
- **Enhanced CPU Performance**: MobileSAM integration for 5-10x CPU speedup
- **Ultralytics Integration**: Professional computer vision library support
- **Improved Device Detection**: Better GPU/CPU/Apple Silicon handling
- **Updated Dependencies**: Modern ML stack with automatic model downloads
