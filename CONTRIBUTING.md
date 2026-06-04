# Contributing to GeoOSAM

🎉 **Thank you for your interest in contributing to GeoOSAM!**

We welcome contributions from the geospatial and AI communities to help make advanced segmentation accessible to QGIS users worldwide.

## 🚀 Quick Start

### Ways to Contribute

- 🐛 **Bug Reports** - Help us identify and fix issues
- 💡 **Feature Requests** - Suggest new functionality
- 📝 **Documentation** - Improve guides and examples
- 🔧 **Code Contributions** - Fix bugs or add features
- 🧪 **Testing** - Test on different platforms and datasets
- 🌍 **Translations** - Help localize the plugin

## 📋 Before You Start

### Prerequisites

- **QGIS 3.16+** for testing
- **Python 3.7+** with PyTorch
- **Git** for version control
- **GitHub account** for collaboration

### Development Setup

```bash
# 1. Fork and clone the repository
git clone https://github.com/espressouk/GeoOSAM.git
cd geo-osam

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# 3. Install development dependencies
pip install -r requirements-dev.txt

# 4. Link to QGIS plugins directory
# Linux/Mac:
ln -s $(pwd)/geo_osam ~/.local/share/QGIS/QGIS3/profiles/default/python/plugins/geo_osam

# Windows:
mklink /D "C:\Users\%USERNAME%\AppData\Roaming\QGIS\QGIS3\profiles\default\python\plugins\geo_osam" "%CD%\geo_osam"
```

## 🐛 Reporting Bugs

### Before Reporting

1. **Search existing issues** to avoid duplicates
2. **Test with latest version** of the plugin
3. **Try with sample data** to isolate the issue
4. **Run diagnostic script** (see Troubleshooting guide)

### Bug Report Template

```markdown
**Environment:**

- OS: [Windows 11 / macOS 12 / Ubuntu 20.04]
- QGIS Version: [3.28.12]
- Plugin Version: [1.0.0]
- Python Version: [3.9.16]
- GPU: [NVIDIA RTX 4090 / Apple M2 / None]

**Bug Description:**
Clear description of what's wrong.

**Steps to Reproduce:**

1. Load raster layer (include format/size)
2. Select Buildings class
3. Click Point mode
4. Click on building
5. Error occurs

**Expected Behavior:**
What should happen.

**Actual Behavior:**
What actually happens.

**Error Messages:**
```

Paste full error message here

```

**Sample Data:**
Link to data that reproduces the issue (if possible).

**Additional Context:**
Any other relevant information.
```

## 💡 Feature Requests

### What Makes a Good Feature Request

- **Clear use case** - Who needs this and why?
- **Specific description** - What exactly should it do?
- **Mockups/examples** - Visual or code examples help
- **Compatibility** - How does it fit with existing features?

### Feature Request Template

```markdown
**Feature Summary:**
One-line description of the feature.

**Use Case:**
Detailed description of the problem this solves.

**Proposed Solution:**
How you envision this working.

**Alternative Solutions:**
Other ways this could be implemented.

**Additional Context:**
Examples, mockups, or related work.
```

## 🔧 Code Contributions

### Development Workflow

1. **Create an issue** to discuss changes first
2. **Fork the repository**
3. **Create feature branch** (`git checkout -b feature/amazing-feature`)
4. **Make changes** following our guidelines
5. **Test thoroughly** on multiple platforms
6. **Submit pull request** with detailed description

### Coding Standards

#### Python Style

```python
# Follow PEP 8
# Use type hints
def process_segmentation(image: np.ndarray, mode: str) -> Dict[str, Any]:
    """Process SAM2 segmentation with given parameters.

    Args:
        image: Input image array
        mode: Segmentation mode ("point" or "bbox")

    Returns:
        Dictionary containing masks and metadata
    """
    pass

# Use meaningful variable names
user_selected_class = "Buildings"  # Good
c = "Buildings"  # Bad

# Handle errors gracefully
try:
    result = sam_predictor.predict(image)
except Exception as e:
    self.show_error(f"Segmentation failed: {e}")
    return None
```

#### QGIS Integration

```python
# Always check for valid layers
layer = iface.activeLayer()
if not isinstance(layer, QgsRasterLayer):
    self.show_warning("Please select a raster layer")
    return

# Use QGIS message system
iface.messageBar().pushMessage(
    "Success",
    "Segmentation completed",
    level=Qgis.Info,
    duration=3
)

# Thread long operations
worker = SAM2Worker(params)
worker.finished.connect(self.on_completed)
worker.start()
```

#### Documentation

```python
# Document all public methods
class GeoOSAMControlPanel:
    def add_custom_class(self, name: str, color: str, description: str = "") -> bool:
        """Add a new segmentation class.

        Args:
            name: Unique class name
            color: RGB color in format "R,G,B" (0-255)
            description: Optional class description

        Returns:
            True if class added successfully, False if name already exists

        Example:
            >>> panel.add_custom_class("Solar Panels", "255,255,0", "Rooftop solar")
            True
        """
```

### Testing

GeoOSAM has an automated test suite covering the helpers module and the mask-to-vector pipeline. Tests run without a QGIS installation — QGIS modules are stubbed by `tests/conftest.py`.

```bash
# Install test dependencies
pip install pytest numpy opencv-python rasterio shapely

# Run all tests
python3 -m pytest tests/ -v
```

**Test coverage:**
- `tests/test_helpers.py` — All 9 detection helper classes (factory, morphology, merge, dedupe, validate_object)
- `tests/test_mask_pipeline.py` — Mask-to-vector conversion, `filter_contained_masks`, `merge_nearby_masks_class_aware`, `dedupe_or_merge_masks_smart`

When adding new features, add corresponding tests. New helper subclasses must have at least `validate_object` and `apply_morphology` covered.

#### Manual Testing Checklist

- [ ] Plugin loads without errors
- [ ] Point segmentation works on various imagery
- [ ] BBox segmentation works on various imagery
- [ ] Export creates valid shapefiles
- [ ] GPU/CPU detection works correctly
- [ ] Error messages are helpful
- [ ] UI is responsive during processing

### Platform Testing

**Essential:** Test on at least 2 platforms before submitting:

- **Windows 10/11** - Most common user platform
- **Ubuntu 20.04+** - Common for GIS professionals
- **macOS 12+** - Including Apple Silicon if possible

## 📝 Documentation Contributions

### What Needs Documentation

- **User guides** with step-by-step workflows
- **API documentation** for developers
- **Troubleshooting** common issues
- **Examples** with real datasets
- **Video tutorials** (especially valuable)

### Documentation Style

````markdown
# Use clear headings

## Descriptive section titles

### Specific feature explanations

**Bold** for important concepts
`code` for technical terms

```python
# Code blocks for examples
```
````

> **Tip:** Include screenshots for UI instructions
>
> **Warning:** Highlight potential issues

````

## 🧪 Testing Guidelines

### Test Data
Use publicly available datasets:
- **Sentinel-2** imagery from Copernicus Open Access Hub
- **Landsat** imagery from USGS Earth Explorer
- **OpenAerialMap** high-resolution imagery
- **Sample datasets** included in plugin

### Performance Testing
```python
# Benchmark segmentation performance
import time

start = time.time()
result = sam_predictor.predict(image)
duration = time.time() - start

print(f"Segmentation took {duration:.2f}s")
print(f"Image size: {image.shape}")
print(f"Device: {device}")
````

### Edge Cases to Test

- **Very large images** (>10K pixels)
- **Very small images** (<500 pixels)
- **Different formats** (GeoTIFF, JP2, PNG)
- **Various projections** (UTM, Geographic, etc.)
- **Single vs multi-band** imagery
- **Poor quality imagery** (low resolution, poor contrast)

## 🌍 Internationalization

### Adding Translations

```python
# Use tr() for user-facing strings
self.label.setText(self.tr("Select segmentation class"))

# Create translation files
pylupdate5 geo_osam.pro
linguist geo_osam_es.ts  # Spanish translation

# Generate compiled translations
lrelease geo_osam_es.ts
```

### Languages Needed

Priority languages for GIS community:

- **Spanish** - Large global GIS community
- **French** - Strong in Africa/Canada
- **German** - Major European market
- **Portuguese** - Brazil and other markets
- **Chinese** - Huge user base

## 🚀 Release Process

### Version Numbering

We use [Semantic Versioning](https://semver.org/):

- **1.0.0** - Major release
- **1.2.0** - Minor features
- **1.0.1** - Bug fixes

### Release Checklist

- [ ] All tests pass
- [ ] Documentation updated
- [ ] CHANGELOG.md updated
- [ ] Version bumped in metadata.txt
- [ ] Tagged release on GitHub
- [ ] Uploaded to QGIS Plugin Repository

## 📞 Getting Help

### Questions?

- **GitHub Discussions** - For general questions
- **Issues** - For bug reports and feature requests
- **Email** - geoosamplugin@gmail.com for sensitive matters

### Development Chat

- **Discord/Slack** - [Link to community chat if available]
- **QGIS Community** - General QGIS development help

## 🏆 Recognition

### Contributors

All contributors are recognized in:

- **CONTRIBUTORS.md** file
- **Plugin about dialog**
- **Release notes**
- **Social media announcements**

### Types of Contributions

We recognize all types of contributions:

- 💻 **Code** - Bug fixes and features
- 📖 **Documentation** - Guides and examples
- 🎨 **Design** - UI/UX improvements
- 🐛 **Testing** - Bug reports and testing
- 💬 **Community** - Helping users and discussions
- 🌍 **Translation** - Localization work

## 📜 License

By contributing, you agree that your contributions will be licensed under the same GPL v2 license as the project.

---

**Ready to contribute? We're excited to work with you!** 🎉

**Questions?** Open an issue or reach out at geoosamplugin@gmail.com
