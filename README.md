# GeoOSAM - Advanced Segmentation for QGIS

🛰️ **State-of-the-art image segmentation using Meta's SAM2 with professional QGIS integration**

[![QGIS Plugin](https://img.shields.io/badge/QGIS-Plugin-green)](https://plugins.qgis.org)
[![Python](https://img.shields.io/badge/Python-3.7+-blue)](https://python.org)
[![License](https://img.shields.io/badge/License-GPL%20v2-blue.svg)](LICENSE)

## 🌟 Features

- **🚀 High Performance**: GPU acceleration (CUDA/Apple Silicon) + multi-threading
- **🎯 Dual Modes**: Point-click and bounding box segmentation
- **📋 12 Pre-defined Classes**: Buildings, Roads, Vegetation, Water, Vehicle, Ship, and more
- **↶ Undo Support**: Mistake correction with polygon-level undo
- **📁 Custom Output**: User-selectable output folders
- **🎨 Class Management**: Custom classes with color coding
- **📡 Smart Workflow**: Auto-raster selection, progress tracking
- **💾 Professional Export**: Shapefile export with detailed attributes
- **🔧 Optimized**: Adaptive processing based on zoom level and hardware

## 📊 Performance

| Hardware       | Typical Speed | Improvement       |
| -------------- | ------------- | ----------------- |
| NVIDIA RTX GPU | 0.2-0.5s      | **10-50x faster** |
| Apple M1/M2    | 1-2s          | **5-15x faster**  |
| Multi-core CPU | 3-5s          | **2-8x faster**   |

## 🚀 Quick Start

1. **Install Plugin**: Enable in QGIS Plugin Manager
2. **Load Raster**: Open satellite/aerial imagery
3. **Select Class**: Choose from 12 pre-defined classes
4. **Choose Mode**: Point or BBox segmentation
5. **Click & Segment**: Point-and-click to create polygons
6. **Export Results**: Save as professional shapefiles

## 📋 System Requirements

### Minimum

- QGIS 3.16+
- Python 3.7+
- 8GB RAM
- 2GB disk space

### Recommended

- QGIS 3.28+
- NVIDIA GPU with CUDA
- 16GB+ RAM
- SSD storage

## 📦 Installation

### From QGIS Plugin Repository

1. Open QGIS
2. Go to **Plugins > Manage and Install Plugins**
3. Search "GeoOSAM"
4. Click **Install Plugin**
5. Dependencies will be installed automatically

### Manual Installation

```bash
# Download and extract plugin
# Copy to QGIS plugins directory
cp -r geo_osam ~/.local/share/QGIS/QGIS3/profiles/default/python/plugins/

# Install dependencies
pip install torch torchvision numpy opencv-python rasterio shapely hydra-core
```

🎯 Use Cases

🏙️ Urban Planning: Building and infrastructure mapping
🌱 Environmental Monitoring: Vegetation and land cover analysis
🚗 Transportation: Vehicle and traffic analysis
🌊 Coastal Studies: Ship detection and water body mapping
🏗️ Construction: Site monitoring and progress tracking
📡 Remote Sensing: Large-scale imagery analysis

📚 Documentation

User Guide
Installation Guide
Troubleshooting
API Reference

🤝 Contributing
We welcome contributions! Please see our Contributing Guide.
📞 Support

Issues: GitHub Issues
Email: ofer@butbega.com
Documentation: Wiki

🙏 Acknowledgments

Meta AI: For the Segment Anything Model (SAM2)
QGIS Community: For the excellent GIS platform
PyTorch Team: For the deep learning framework

📄 License
This project is licensed under the GPL v2 License - see the LICENSE file for details.
🏆 Citation
If you use GeoOSAM in your research, please cite:
bibtex@software{geosam2025,
title={GeoOSAM: Advanced Segmentation for QGIS},
author={Butbega, Ofer},
year={2025},
url={https://github.com/espressouk/Geo-OSAM}
}

### **4. Documentation Folder** (RECOMMENDED)

Create `docs/` folder with:

**docs/user_guide.md**:

```markdown
# GeoOSAM User Guide

## Getting Started

[Detailed user instructions...]

## Workflow Examples

[Step-by-step tutorials...]

## Troubleshooting

[Common issues and solutions...]
```
