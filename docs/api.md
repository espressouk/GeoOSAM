# GeoOSAM API Reference

## 📋 Overview

GeoOSAM is built with a modular architecture for extensibility and maintainability. This document covers the public API for developers who want to extend or integrate with the plugin.

## 🏗️ Architecture

```
geo_osam/
├── geo_osam.py              # Main plugin class
├── geo_osam_dialog.py       # Control panel and core logic
├── sam2/                    # SAM2 model integration
│   ├── build_sam.py
│   ├── sam2_image_predictor.py
│   └── configs/
└── resources/               # UI resources
```

## 📚 Core Classes

### SegSam (Main Plugin Class)

```python
class SegSam:
    """Main QGIS Plugin Implementation."""

    def __init__(self, iface: QgsInterface):
        """Initialize plugin with QGIS interface."""

    def initGui(self) -> None:
        """Create menu entries & toolbar icons."""

    def unload(self) -> None:
        """Remove plugin UI elements on unload."""

    def show_control_panel(self) -> None:
        """Show the GeoOSAM control panel."""

    def toggle_control_panel(self) -> None:
        """Toggle control panel visibility."""
```

#### Methods

##### `__init__(iface: QgsInterface)`

Initializes the plugin with QGIS interface reference.

**Parameters:**

- `iface`: QGIS interface object

**Example:**

```python
plugin = SegSam(iface)
```

##### `show_control_panel()`

Creates and displays the control panel as a dockable widget.

**Returns:** None

**Side Effects:**

- Creates `GeoOSAMControlPanel` instance
- Docks panel to right side of QGIS
- Updates QGIS message bar

### GeoOSAMControlPanel (Core Functionality)

```python
class GeoOSAMControlPanel(QtWidgets.QDockWidget):
    """Enhanced control panel with output folder selection, undo, and more classes."""

    def __init__(self, iface: QgsInterface, parent=None):
        """Initialize control panel."""

    # Public Methods
    def set_output_folder(self, folder_path: str) -> bool:
        """Set custom output folder for shapefiles."""

    def add_custom_class(self, name: str, color: str, description: str = "") -> bool:
        """Add a new segmentation class."""

    def export_class_layer(self, class_name: str, output_path: str = None) -> bool:
        """Export specific class layer to shapefile."""

    def get_segmentation_stats(self) -> dict:
        """Get current segmentation statistics."""
```

#### Properties

##### `DEFAULT_CLASSES`

Dictionary of pre-defined segmentation classes:

```python
DEFAULT_CLASSES = {
    'Buildings': {'color': '220,20,60', 'description': 'Residential and commercial buildings'},
    'Roads': {'color': '105,105,105', 'description': 'Streets, highways, and pathways'},
    'Vegetation': {'color': '34,139,34', 'description': 'Trees, grass, and vegetation'},
    'Water': {'color': '30,144,255', 'description': 'Rivers, lakes, and water bodies'},
    'Agriculture': {'color': '255,215,0', 'description': 'Farmland and crops'},
    'Parking': {'color': '255,140,0', 'description': 'Parking lots and areas'},
    'Industrial': {'color': '138,43,226', 'description': 'Industrial buildings and areas'},
    'Residential': {'color': '255,20,147', 'description': 'Residential areas'},
    'Commercial': {'color': '0,191,255', 'description': 'Commercial areas'},
    'Vehicle': {'color': '255,69,0', 'description': 'Cars, trucks, and vehicles'},
    'Ship': {'color': '0,206,209', 'description': 'Ships, boats, and vessels'},
    'Other': {'color': '148,0,211', 'description': 'Unclassified objects'}
}
```

#### Core Methods

##### `set_output_folder(folder_path: str) -> bool`

Set custom output folder for shapefile exports.

**Parameters:**

- `folder_path` (str): Absolute path to output directory

**Returns:**

- `bool`: True if successful, False otherwise

**Example:**

```python
panel = GeoOSAMControlPanel(iface)
success = panel.set_output_folder("/home/user/my_segments")
```

##### `add_custom_class(name: str, color: str, description: str = "") -> bool`

Add a new segmentation class to the available options.

**Parameters:**

- `name` (str): Class name (must be unique)
- `color` (str): RGB color in format "R,G,B" (0-255)
- `description` (str): Optional class description

**Returns:**

- `bool`: True if successful, False if class exists

**Example:**

```python
panel.add_custom_class("Solar Panels", "255,255,0", "Rooftop solar installations")
```

##### `export_class_layer(class_name: str, output_path: str = None) -> bool`

Export a specific class layer to shapefile.

**Parameters:**

- `class_name` (str): Name of class to export
- `output_path` (str, optional): Custom output path, uses default if None

**Returns:**

- `bool`: True if export successful

**Example:**

```python
success = panel.export_class_layer("Buildings", "/path/to/buildings.shp")
```

##### `get_segmentation_stats() -> dict`

Get current segmentation statistics.

**Returns:**

- `dict`: Statistics including feature counts, classes, etc.

**Example:**

```python
stats = panel.get_segmentation_stats()
print(f"Total segments: {stats['total_segments']}")
print(f"Active classes: {stats['active_classes']}")
```

## 🔧 Utility Functions

### Performance Configuration

#### `setup_pytorch_performance() -> int`

Configure PyTorch for optimal performance.

**Returns:**

- `int`: Number of threads configured

**Example:**

```python
from geo_osam_dialog import setup_pytorch_performance
threads = setup_pytorch_performance()
print(f"Using {threads} CPU threads")
```

#### `detect_best_device() -> str`

Detect best available device for inference.

**Returns:**

- `str`: Device identifier ("cuda", "mps", or "cpu")

**Example:**

```python
from geo_osam_dialog import detect_best_device
device = detect_best_device()
print(f"Using device: {device}")
```

### Model Management

#### `auto_download_checkpoint() -> bool`

Automatically download SAM2 checkpoint if missing.

**Returns:**

- `bool`: True if checkpoint available after operation

**Example:**

```python
from geo_osam_dialog import auto_download_checkpoint
if auto_download_checkpoint():
    print("Checkpoint ready")
```

## 🧵 Threading Classes

### OptimizedSAM2Worker

```python
class OptimizedSAM2Worker(QThread):
    """Worker thread for SAM2 inference."""

    # Signals
    finished = pyqtSignal(object)  # Emitted with result dict
    error = pyqtSignal(str)        # Emitted with error message
    progress = pyqtSignal(str)     # Emitted with progress updates

    def __init__(self, predictor, arr, mode, point_coords=None,
                 point_labels=None, box=None, mask_transform=None,
                 debug_info=None, device="cpu"):
        """Initialize worker thread."""
```

#### Signals

- **`finished(object)`**: Emitted when segmentation completes successfully
- **`error(str)`**: Emitted when an error occurs during processing
- **`progress(str)`**: Emitted with progress update messages

#### Usage Example

```python
# Create worker
worker = OptimizedSAM2Worker(
    predictor=sam_predictor,
    arr=image_array,
    mode="point",
    point_coords=[[100, 150]],
    point_labels=[1],
    device="cuda"
)

# Connect signals
worker.finished.connect(self.on_result)
worker.error.connect(self.on_error)
worker.progress.connect(self.update_status)

# Start processing
worker.start()
```

## 🗺️ Map Tools

### EnhancedPointClickTool

```python
class EnhancedPointClickTool(QgsMapTool):
    """Enhanced point selection tool with visual feedback."""

    def __init__(self, canvas: QgsMapCanvas, callback: callable):
        """Initialize point tool."""

    def canvasReleaseEvent(self, e: QgsMapMouseEvent) -> None:
        """Handle mouse click events."""

    def clear_feedback(self) -> None:
        """Clear visual feedback."""
```

### EnhancedBBoxClickTool

```python
class EnhancedBBoxClickTool(QgsMapTool):
    """Enhanced bounding box selection tool with drag feedback."""

    def __init__(self, canvas: QgsMapCanvas, callback: callable):
        """Initialize bbox tool."""

    def canvasPressEvent(self, e: QgsMapMouseEvent) -> None:
        """Handle mouse press events."""

    def canvasMoveEvent(self, e: QgsMapMouseEvent) -> None:
        """Handle mouse move events."""

    def canvasReleaseEvent(self, e: QgsMapMouseEvent) -> None:
        """Handle mouse release events."""
```

## 📊 Data Structures

### Segmentation Result

Results returned by the SAM2 worker contain:

```python
result = {
    'mask': numpy.ndarray,           # Binary segmentation mask
    'scores': numpy.ndarray,         # Confidence scores
    'logits': numpy.ndarray,         # Raw model outputs
    'mask_transform': Affine,        # Geospatial transform
    'debug_info': {                  # Processing metadata
        'mode': str,                 # "point" or "bbox"
        'class': str,                # Target class name
        'device': str,               # Processing device
        'prep_time': float,          # Preprocessing time
        'crop_size': str             # Processing dimensions
    }
}
```

### Feature Attributes

Generated features include these attributes:

```python
attributes = [
    'segment_id',      # int: Unique segment identifier
    'class_name',      # str: Assigned class name
    'class_color',     # str: RGB color code
    'method',          # str: "point" or "bbox"
    'timestamp',       # str: Creation timestamp
    'mask_file',       # str: Debug mask filename
    'crop_size',       # str: Processing dimensions
    'canvas_scale'     # float: Map scale when created
]
```

## 🔌 Extension Points

### Custom Classes

Add application-specific classes:

```python
# Define custom classes
FORESTRY_CLASSES = {
    'Deciduous': {'color': '34,139,34', 'description': 'Deciduous trees'},
    'Coniferous': {'color': '0,100,0', 'description': 'Coniferous trees'},
    'Clearcut': {'color': '139,69,19', 'description': 'Harvested areas'},
    'Access Roads': {'color': '160,82,45', 'description': 'Forest access roads'}
}

# Add to control panel
panel = GeoOSAMControlPanel(iface)
for name, info in FORESTRY_CLASSES.items():
    panel.add_custom_class(name, info['color'], info['description'])
```

### Custom Processing

Override segmentation processing:

```python
class CustomGeoOSAMPanel(GeoOSAMControlPanel):
    """Extended panel with custom processing."""

    def _process_segmentation_result(self, mask, mask_transform, debug_info):
        """Override to add custom post-processing."""

        # Custom processing here
        processed_mask = self.custom_filter(mask)

        # Call parent method
        super()._process_segmentation_result(processed_mask, mask_transform, debug_info)

    def custom_filter(self, mask):
        """Apply custom filtering to mask."""
        # Morphological operations, size filtering, etc.
        return filtered_mask
```

### Export Customization

Custom export formats:

```python
def export_to_geojson(layer, output_path):
    """Export layer to GeoJSON format."""
    from qgis.core import QgsVectorFileWriter

    error = QgsVectorFileWriter.writeAsVectorFormat(
        layer,
        output_path,
        "utf-8",
        layer.crs(),
        "GeoJSON"
    )

    return error[0] == QgsVectorFileWriter.NoError

# Usage
layer = panel.result_layers['Buildings']
export_to_geojson(layer, "/path/to/buildings.geojson")
```

## 🧪 Testing Interface

### Unit Testing

```python
import unittest
from qgis.testing import start_app
from geo_osam_dialog import GeoOSAMControlPanel

class TestGeoOSAM(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Initialize QGIS application."""
        cls.app = start_app()

    def test_class_creation(self):
        """Test custom class creation."""
        panel = GeoOSAMControlPanel(None)

        success = panel.add_custom_class("Test", "255,0,0", "Test class")
        self.assertTrue(success)

        # Test duplicate
        duplicate = panel.add_custom_class("Test", "0,255,0", "Duplicate")
        self.assertFalse(duplicate)

    def test_device_detection(self):
        """Test device detection."""
        from geo_osam_dialog import detect_best_device
        device = detect_best_device()
        self.assertIn(device, ["cuda", "mps", "cpu"])
```

### Integration Testing

```python
def test_segmentation_workflow():
    """Test complete segmentation workflow."""

    # Setup
    panel = GeoOSAMControlPanel(iface)

    # Load test raster
    test_layer = QgsRasterLayer("/path/to/test.tif", "test")
    QgsProject.instance().addMapLayer(test_layer)
    iface.setActiveLayer(test_layer)

    # Configure
    panel.current_class = "Buildings"
    panel.point = QgsPointXY(100, 200)

    # Run segmentation
    panel._run_segmentation()

    # Verify results
    assert "Buildings" in panel.result_layers
    layer = panel.result_layers["Buildings"]
    assert layer.featureCount() > 0
```

## 📖 Code Examples

### Basic Plugin Usage

```python
from qgis.core import QgsProject, QgsRasterLayer
from geo_osam import SegSam

# Initialize plugin
plugin = SegSam(iface)
plugin.initGui()

# Load raster data
raster = QgsRasterLayer("/path/to/satellite.tif", "Satellite")
QgsProject.instance().addMapLayer(raster)
iface.setActiveLayer(raster)

# Show control panel
plugin.show_control_panel()
```

### Programmatic Segmentation

```python
# Access control panel
panel = plugin.control_panel

# Set parameters
panel.current_class = "Buildings"
panel.point = QgsPointXY(longitude, latitude)

# Run segmentation
panel._run_segmentation()

# Export results
panel.export_class_layer("Buildings", "/output/buildings.shp")
```

### Batch Processing

```python
def batch_segment_buildings(raster_path, points, output_dir):
    """Segment multiple buildings from point list."""

    # Load raster
    raster = QgsRasterLayer(raster_path, "input")
    QgsProject.instance().addMapLayer(raster)
    iface.setActiveLayer(raster)

    # Initialize panel
    panel = GeoOSAMControlPanel(iface)
    panel.current_class = "Buildings"
    panel.set_output_folder(output_dir)

    # Process each point
    for i, (x, y) in enumerate(points):
        panel.point = QgsPointXY(x, y)
        panel._run_segmentation()

        # Export individual result
        output_path = f"{output_dir}/building_{i:03d}.shp"
        panel.export_class_layer("Buildings", output_path)

        # Clear for next iteration
        panel._start_new_shape()
```

## 🔗 External Integrations

### QGIS Processing Integration

```python
from qgis.core import QgsProcessingAlgorithm, QgsProcessingParameterRasterLayer

class GeoOSAMProcessingAlgorithm(QgsProcessingAlgorithm):
    """QGIS Processing algorithm wrapper for GeoOSAM."""

    def processAlgorithm(self, parameters, context, feedback):
        """Run GeoOSAM segmentation as processing algorithm."""

        # Get input raster
        raster = self.parameterAsRasterLayer(parameters, 'INPUT', context)

        # Initialize GeoOSAM
        panel = GeoOSAMControlPanel(None)

        # Configure and run
        panel.current_class = self.parameterAsString(parameters, 'CLASS', context)

        # Process points
        points = self.parameterAsMatrix(parameters, 'POINTS', context)
        for x, y in points:
            panel.point = QgsPointXY(x, y)
            panel._run_segmentation()

        # Return output layer
        return {'OUTPUT': panel.result_layers[panel.current_class]}
```

---

## 📝 Development Guidelines

### Code Style

- Follow PEP 8 for Python code
- Use type hints where possible
- Document all public methods
- Include docstring examples

### Error Handling

- Use try/catch blocks for external operations
- Provide meaningful error messages
- Log errors for debugging
- Gracefully degrade functionality

### Performance

- Use threading for long operations
- Cache expensive computations
- Optimize memory usage
- Profile critical paths

### Testing

- Write unit tests for core functions
- Include integration tests
- Test on multiple platforms
- Verify with various data formats

**For more examples and detailed documentation, see the source code comments and docstrings.** 📚
