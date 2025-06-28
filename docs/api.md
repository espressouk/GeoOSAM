# GeoOSAM API Reference

## 📋 Overview

GeoOSAM is built with a modular architecture featuring intelligent model selection between SAM 2.1 and MobileSAM based on available hardware. This document covers the public API for developers who want to extend or integrate with the plugin.

## 🏗️ Architecture

```
geo_osam/
├── geo_osam.py              # Main plugin class
├── geo_osam_dialog.py       # Control panel with intelligent model selection
├── sam2/                    # SAM2 model integration
│   ├── build_sam.py
│   ├── sam2_image_predictor.py
│   └── configs/
├── UltralyticsPredictor     # MobileSAM wrapper (embedded)
└── resources/               # UI resources
```

## 🧠 Intelligent Model Selection

### Core Logic

GeoOSAM automatically selects the optimal model based on hardware:

```python
def detect_best_device():
    """Detect best device and model combination."""
    if torch.cuda.is_available() and gpu_memory >= 4GB:
        return "cuda", "SAM2.1"
    elif torch.backends.mps.is_available():
        return "mps", "SAM2.1"
    else:
        model = "MobileSAM" if ultralytics_available else "SAM2"
        return "cpu", model
```

### Model Performance Matrix

| Hardware      | Model     | Expected Speed | Use Case              |
| ------------- | --------- | -------------- | --------------------- |
| NVIDIA GPU    | SAM 2.1   | 0.2-0.5s       | Maximum accuracy      |
| Apple Silicon | SAM 2.1   | 1-2s           | Balanced performance  |
| 24+ Core CPU  | MobileSAM | <1s            | High-end workstations |
| 8-16 Core CPU | MobileSAM | 1-2s           | Standard systems      |
| 4-8 Core CPU  | MobileSAM | 2-4s           | Budget systems        |

## 📚 Core Classes

### SegSam (Main Plugin Class)

```python
class SegSam:
    """Main QGIS Plugin Implementation with intelligent model selection."""

    def __init__(self, iface: QgsInterface):
        """Initialize plugin with QGIS interface."""

    def initGui(self) -> None:
        """Create menu entries & toolbar icons."""

    def unload(self) -> None:
        """Remove plugin UI elements on unload."""

    def show_control_panel(self) -> None:
        """Show the GeoOSAM control panel with auto device detection."""
```

### GeoOSAMControlPanel (Enhanced)

```python
class GeoOSAMControlPanel(QtWidgets.QDockWidget):
    """Enhanced control panel with intelligent model selection."""

    def __init__(self, iface: QgsInterface, parent=None):
        """Initialize with automatic device/model detection."""
        self.device, self.model_choice, self.num_cores = detect_best_device()
        self._init_sam_model()  # Initialize selected model

    # Enhanced Properties
    @property
    def current_model_info(self) -> dict:
        """Get current model and device information."""
        return {
            'device': self.device,
            'model': self.model_choice,
            'cores': self.num_cores,
            'expected_speed': self._get_expected_speed()
        }
```

#### New Model Selection Methods

##### `detect_best_device() -> tuple`

Intelligently detects optimal device and model combination.

**Returns:**

- `tuple`: (device_str, model_choice, num_cores)
  - `device_str`: "cuda", "mps", or "cpu"
  - `model_choice`: "SAM2" or "MobileSAM"
  - `num_cores`: CPU core count (None for GPU)

**Example:**

```python
device, model, cores = detect_best_device()
print(f"Using {model} on {device}")
if cores:
    print(f"CPU threading: {cores} cores")
```

##### `setup_pytorch_performance() -> int`

Configure PyTorch threading for optimal CPU performance.

**Returns:**

- `int`: Number of threads configured

**Logic:**

```python
if cores >= 16:
    optimal_threads = max(8, int(cores * 0.75))  # Use 75% for high-core
elif cores >= 8:
    optimal_threads = max(4, cores - 2)          # Leave 2 for system
else:
    optimal_threads = max(1, cores - 1)          # Leave 1 for system
```

**Example:**

```python
threads = setup_pytorch_performance()
print(f"Configured {threads} PyTorch threads")
```

## 🤖 Model Classes

### UltralyticsPredictor

```python
class UltralyticsPredictor:
    """Wrapper for Ultralytics MobileSAM with SAM2 interface compatibility."""

    def __init__(self, model):
        """Initialize with Ultralytics SAM model."""
        self.model = model  # SAM('mobile_sam.pt')
        self.features = None

    def set_image(self, image: np.ndarray) -> None:
        """Set image for segmentation (compatible with SAM2 interface)."""
        self.image = image

    def predict(self, point_coords=None, point_labels=None, box=None,
                multimask_output=False) -> tuple:
        """Predict segmentation mask (SAM2-compatible interface)."""
        # Returns: (masks, scores, logits)
```

#### Key Methods

##### `predict(point_coords=None, point_labels=None, box=None, multimask_output=False)`

Perform segmentation prediction with SAM2-compatible interface.

**Parameters:**

- `point_coords` (array): Point coordinates [[x, y]]
- `point_labels` (array): Point labels [1] (positive)
- `box` (array): Bounding box [[x1, y1, x2, y2]]
- `multimask_output` (bool): Return multiple masks (ignored for MobileSAM)

**Returns:**

- `tuple`: (masks, scores, logits)
  - `masks`: List of numpy arrays
  - `scores`: List of confidence scores
  - `logits`: None (not used by MobileSAM)

**Example:**

```python
from ultralytics import SAM

# Initialize MobileSAM
mobile_sam = SAM('mobile_sam.pt')
predictor = UltralyticsPredictor(mobile_sam)

# Set image
predictor.set_image(image_array)

# Point-based prediction
masks, scores, logits = predictor.predict(
    point_coords=[[100, 150]],
    point_labels=[1]
)

# Box-based prediction
masks, scores, logits = predictor.predict(
    box=[[50, 50, 200, 200]]
)
```

### Enhanced SAM2ImagePredictor

```python
class SAM2ImagePredictor:
    """Enhanced SAM2 predictor with performance optimizations."""

    def __init__(self, sam_model):
        """Initialize with SAM2 model and device optimization."""
        self.model = sam_model

        # Performance optimizations
        if device == "cpu":
            try:
                self.model = torch.jit.optimize_for_inference(self.model)
            except:
                pass  # Fallback gracefully
```

## 🔧 Utility Functions

### Device Detection

#### `detect_best_device() -> tuple`

Main device detection function with comprehensive hardware analysis.

**Logic Flow:**

```python
def detect_best_device():
    cores = None
    try:
        # 1. Check CUDA GPU
        if torch.cuda.is_available() and not os.getenv("GEOOSAM_FORCE_CPU"):
            gpu_props = torch.cuda.get_device_properties(0)
            if gpu_props.total_memory / 1024**3 >= 4:  # 4GB minimum
                return "cuda", "SAM2", cores

        # 2. Check Apple Silicon
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return "mps", "SAM2", cores

        # 3. CPU fallback with model selection
        else:
            model_choice = "MobileSAM" if MOBILESAM_AVAILABLE else "SAM2"
            cores = setup_pytorch_performance()
            return "cpu", model_choice, cores

    except Exception as e:
        # Graceful fallback
        device, model_choice = "cpu", "SAM2"
        cores = setup_pytorch_performance()
        return device, model_choice, cores
```

#### Environment Variables

Control device selection with environment variables:

```python
# Force CPU mode (useful for testing)
os.environ["GEOOSAM_FORCE_CPU"] = "1"

# Force GPU mode (override memory checks)
os.environ["GEOOSAM_FORCE_GPU"] = "1"
```

### Performance Configuration

#### `setup_pytorch_performance() -> int`

Advanced CPU threading configuration with hardware-aware optimization.

**Features:**

- **Core-aware scaling**: Uses 75% of cores on 16+ core systems
- **System responsiveness**: Leaves cores for OS and other apps
- **Environment setup**: Configures OpenMP, MKL, and OpenBLAS
- **Safety checks**: Handles threading conflicts gracefully

**Example:**

```python
import multiprocessing
import os
import torch

def setup_pytorch_performance():
    num_cores = multiprocessing.cpu_count()

    # Intelligent thread allocation
    if num_cores >= 16:
        optimal_threads = max(8, int(num_cores * 0.75))
    elif num_cores >= 8:
        optimal_threads = max(4, num_cores - 2)
    else:
        optimal_threads = max(1, num_cores - 1)

    # Configure PyTorch
    torch.set_num_interop_threads(min(4, optimal_threads // 2))
    torch.set_num_threads(optimal_threads)

    # Configure external libraries
    os.environ["OMP_NUM_THREADS"] = str(optimal_threads)
    os.environ["MKL_NUM_THREADS"] = str(optimal_threads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(optimal_threads)

    return optimal_threads
```

### Model Management

#### `auto_download_checkpoint() -> bool`

Enhanced checkpoint downloading with better error handling.

**Features:**

- **Cross-platform**: Bash script on Linux/Mac, Python fallback
- **Progress tracking**: Shows download progress
- **Verification**: Checks file size and integrity
- **Timeout handling**: 5-minute timeout for large downloads

#### MobileSAM Auto-Download

```python
# MobileSAM downloading is handled automatically by Ultralytics
try:
    from ultralytics import SAM
    test_model = SAM('mobile_sam.pt')  # Auto-downloads if needed
    MOBILESAM_AVAILABLE = True
except Exception:
    MOBILESAM_AVAILABLE = False
```

## 🧵 Enhanced Threading Classes

### OptimizedSAM2Worker (Updated)

```python
class OptimizedSAM2Worker(QThread):
    """Enhanced worker thread supporting both SAM2 and MobileSAM."""

    def __init__(self, predictor, arr, mode, model_choice="SAM2",
                 point_coords=None, point_labels=None, box=None,
                 mask_transform=None, debug_info=None, device="cpu"):
        """Initialize with model choice and device information."""
        super().__init__()
        self.model_choice = model_choice  # "SAM2" or "MobileSAM"
        self.device = device
        # ... other parameters

    def run(self):
        """Run inference with model-specific optimizations."""
        try:
            self.progress.emit(f"🖼️ Setting image for {self.model_choice}...")
            self.predictor.set_image(self.arr)

            self.progress.emit(f"🧠 Running {self.model_choice} inference...")

            with torch.no_grad():
                if self.mode == "point":
                    masks, scores, logits = self.predictor.predict(
                        point_coords=self.point_coords,
                        point_labels=self.point_labels,
                        multimask_output=False
                    )
                elif self.mode == "bbox":
                    masks, scores, logits = self.predictor.predict(
                        box=self.box,
                        multimask_output=False
                    )

            # Enhanced result processing
            result = {
                'mask': self._process_mask(masks[0]),
                'scores': scores,
                'logits': logits,
                'mask_transform': self.mask_transform,
                'debug_info': {
                    **self.debug_info,
                    'model': self.model_choice,
                    'device': self.device
                }
            }

            self.finished.emit(result)

        except Exception as e:
            error_msg = f"{self.model_choice} inference failed: {str(e)}"
            self.error.emit(error_msg)
```

#### Enhanced Features

- **Model-aware processing**: Different optimizations for SAM2 vs MobileSAM
- **Device-specific handling**: GPU vs CPU optimizations
- **Progress tracking**: Model-specific progress messages
- **Error context**: Enhanced error reporting with model information

## 📊 Enhanced Data Structures

### Device Information

```python
device_info = {
    'device': str,           # "cuda", "mps", "cpu"
    'model': str,            # "SAM2", "MobileSAM"
    'cores': int,            # CPU cores (None for GPU)
    'gpu_name': str,         # GPU name (None for CPU)
    'gpu_memory': float,     # GPU memory in GB (None for CPU)
    'expected_speed': str,   # Expected processing time
    'threading_info': {      # CPU threading details
        'num_threads': int,
        'interop_threads': int,
        'core_utilization': float
    }
}
```

### Enhanced Segmentation Result

```python
result = {
    'mask': numpy.ndarray,           # Binary segmentation mask
    'scores': numpy.ndarray,         # Confidence scores
    'logits': numpy.ndarray,         # Raw model outputs (None for MobileSAM)
    'mask_transform': Affine,        # Geospatial transform
    'debug_info': {                  # Enhanced processing metadata
        'mode': str,                 # "point" or "bbox"
        'class': str,                # Target class name
        'device': str,               # Processing device
        'model': str,                # Model used ("SAM2" or "MobileSAM")
        'prep_time': float,          # Preprocessing time
        'inference_time': float,     # Model inference time
        'crop_size': str,            # Processing dimensions
        'threading_info': dict       # CPU threading details
    }
}
```

## 🔌 Enhanced Extension Points

### Custom Model Selection

Override automatic model selection:

```python
import os

# Force specific configurations for testing
os.environ["GEOOSAM_FORCE_CPU"] = "1"      # Force CPU mode
os.environ["GEOOSAM_FORCE_GPU"] = "1"      # Force GPU mode
os.environ["GEOOSAM_FORCE_SAM2"] = "1"     # Force SAM2 (any device)
os.environ["GEOOSAM_FORCE_MOBILESAM"] = "1"  # Force MobileSAM

# Custom device detection
class CustomGeoOSAMPanel(GeoOSAMControlPanel):
    def _init_sam_model(self):
        """Override model selection logic."""

        # Custom logic here
        if self.custom_condition():
            self.device = "cpu"
            self.model_choice = "MobileSAM"
            self._init_mobilesam_model()
        else:
            super()._init_sam_model()
```

### Performance Monitoring

```python
class PerformanceMonitor:
    """Monitor and log model performance."""

    def __init__(self):
        self.performance_log = []

    def log_segmentation(self, result):
        """Log segmentation performance."""
        debug_info = result['debug_info']

        entry = {
            'timestamp': time.time(),
            'model': debug_info.get('model'),
            'device': debug_info.get('device'),
            'prep_time': debug_info.get('prep_time'),
            'inference_time': debug_info.get('inference_time'),
            'total_time': debug_info.get('prep_time', 0) + debug_info.get('inference_time', 0),
            'crop_size': debug_info.get('crop_size'),
            'success': len(result.get('mask', [])) > 0
        }

        self.performance_log.append(entry)

    def get_performance_stats(self):
        """Get performance statistics."""
        if not self.performance_log:
            return {}

        times = [entry['total_time'] for entry in self.performance_log if entry['success']]

        return {
            'total_segmentations': len(self.performance_log),
            'successful': sum(1 for e in self.performance_log if e['success']),
            'avg_time': sum(times) / len(times) if times else 0,
            'min_time': min(times) if times else 0,
            'max_time': max(times) if times else 0,
            'models_used': list(set(e['model'] for e in self.performance_log)),
            'devices_used': list(set(e['device'] for e in self.performance_log))
        }

# Usage
monitor = PerformanceMonitor()

# In segmentation callback:
def on_segmentation_finished(result):
    monitor.log_segmentation(result)

    # Print stats every 10 segmentations
    if len(monitor.performance_log) % 10 == 0:
        stats = monitor.get_performance_stats()
        print(f"Avg time: {stats['avg_time']:.2f}s using {stats['models_used']}")
```

### Hardware-Specific Optimizations

```python
class HardwareOptimizer:
    """Apply hardware-specific optimizations."""

    @staticmethod
    def optimize_for_device(device, model_choice):
        """Apply device-specific optimizations."""

        if device == "cuda":
            # GPU optimizations
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False

        elif device == "mps":
            # Apple Silicon optimizations
            # Some operations may fallback to CPU automatically
            pass

        elif device == "cpu":
            # CPU optimizations
            if model_choice == "MobileSAM":
                # MobileSAM-specific optimizations
                torch.set_num_threads(setup_pytorch_performance())

            # Memory optimizations
            torch.set_grad_enabled(False)

        return device, model_choice

# Usage in control panel
class OptimizedGeoOSAMPanel(GeoOSAMControlPanel):
    def _init_sam_model(self):
        """Initialize with hardware optimizations."""

        # Apply optimizations
        self.device, self.model_choice = HardwareOptimizer.optimize_for_device(
            self.device, self.model_choice
        )

        # Continue with normal initialization
        super()._init_sam_model()
```

## 🧪 Enhanced Testing Interface

### Model Selection Testing

```python
import unittest
from unittest.mock import patch, MagicMock

class TestModelSelection(unittest.TestCase):

    def test_cuda_selection(self):
        """Test CUDA GPU selection logic."""
        with patch('torch.cuda.is_available', return_value=True), \
             patch('torch.cuda.get_device_properties') as mock_props:

            # Mock sufficient GPU memory
            mock_props.return_value.total_memory = 8 * 1024**3  # 8GB

            device, model, cores = detect_best_device()

            self.assertEqual(device, "cuda")
            self.assertEqual(model, "SAM2")
            self.assertIsNone(cores)

    def test_cpu_mobilesam_selection(self):
        """Test CPU with MobileSAM selection."""
        with patch('torch.cuda.is_available', return_value=False), \
             patch('torch.backends.mps.is_available', return_value=False), \
             patch('geo_osam_dialog.MOBILESAM_AVAILABLE', True):

            device, model, cores = detect_best_device()

            self.assertEqual(device, "cpu")
            self.assertEqual(model, "MobileSAM")
            self.assertIsInstance(cores, int)
            self.assertGreater(cores, 0)

    def test_cpu_fallback(self):
        """Test CPU fallback when MobileSAM unavailable."""
        with patch('torch.cuda.is_available', return_value=False), \
             patch('torch.backends.mps.is_available', return_value=False), \
             patch('geo_osam_dialog.MOBILESAM_AVAILABLE', False):

            device, model, cores = detect_best_device()

            self.assertEqual(device, "cpu")
            self.assertEqual(model, "SAM2")
            self.assertIsInstance(cores, int)

    def test_force_cpu_environment(self):
        """Test forced CPU mode via environment variable."""
        with patch.dict(os.environ, {'GEOOSAM_FORCE_CPU': '1'}), \
             patch('torch.cuda.is_available', return_value=True):

            device, model, cores = detect_best_device()

            self.assertEqual(device, "cpu")
            # Should prefer MobileSAM on forced CPU
            self.assertIn(model, ["MobileSAM", "SAM2"])
```

### Performance Testing

```python
class TestPerformance(unittest.TestCase):

    def test_threading_setup(self):
        """Test CPU threading configuration."""
        with patch('multiprocessing.cpu_count', return_value=24):
            threads = setup_pytorch_performance()

            # Should use 75% of 24 cores = 18 threads
            self.assertEqual(threads, 18)

        with patch('multiprocessing.cpu_count', return_value=8):
            threads = setup_pytorch_performance()

            # Should use 8-2 = 6 threads
            self.assertEqual(threads, 6)

    def test_model_loading_performance(self):
        """Test model loading times."""
        import time

        # Test MobileSAM loading
        if MOBILESAM_AVAILABLE:
            start = time.time()
            from ultralytics import SAM
            model = SAM('mobile_sam.pt')
            predictor = UltralyticsPredictor(model)
            load_time = time.time() - start

            # Should load quickly
            self.assertLess(load_time, 10.0)  # 10 seconds max
```

## 📖 Enhanced Code Examples

### Complete Workflow with Model Selection

```python
from qgis.core import QgsProject, QgsRasterLayer
from geo_osam import SegSam
from geo_osam_dialog import detect_best_device, GeoOSAMControlPanel

# 1. Initialize with automatic device detection
device, model, cores = detect_best_device()
print(f"Detected: {model} on {device}")

# 2. Initialize plugin
plugin = SegSam(iface)
plugin.initGui()

# 3. Load raster data
raster = QgsRasterLayer("/path/to/satellite.tif", "Satellite")
QgsProject.instance().addMapLayer(raster)
iface.setActiveLayer(raster)

# 4. Show control panel (with auto-detected model)
plugin.show_control_panel()
panel = plugin.control_panel

# 5. Check what was actually selected
model_info = panel.current_model_info
print(f"Using: {model_info['model']} on {model_info['device']}")
print(f"Expected speed: {model_info['expected_speed']}")
```

### Programmatic Segmentation with Model Info

```python
# Access control panel with model information
panel = plugin.control_panel

# Print current configuration
print(f"Device: {panel.device}")
print(f"Model: {panel.model_choice}")
if panel.num_cores:
    print(f"CPU cores: {panel.num_cores}")

# Set parameters
panel.current_class = "Buildings"
panel.point = QgsPointXY(longitude, latitude)

# Run segmentation (uses auto-selected model)
panel._run_segmentation()

# Export results with model info in attributes
panel.export_class_layer("Buildings", "/output/buildings.shp")
```

### Batch Processing with Performance Monitoring

```python
def batch_segment_with_monitoring(raster_path, points, output_dir):
    """Batch segmentation with performance monitoring."""

    # Setup
    device, model, cores = detect_best_device()
    print(f"Batch processing with {model} on {device}")

    panel = GeoOSAMControlPanel(iface)
    panel.current_class = "Buildings"
    panel.set_output_folder(output_dir)

    # Performance tracking
    times = []

    # Process each point
    for i, (x, y) in enumerate(points):
        start_time = time.time()

        panel.point = QgsPointXY(x, y)
        panel._run_segmentation()

        process_time = time.time() - start_time
        times.append(process_time)

        print(f"Point {i+1}: {process_time:.2f}s")

        # Export individual result
        output_path = f"{output_dir}/building_{i:03d}.shp"
        panel.export_class_layer("Buildings", output_path)

        # Clear for next iteration
        panel._start_new_shape()

    # Performance summary
    avg_time = sum(times) / len(times)
    print(f"\nBatch complete:")
    print(f"  Model: {model} on {device}")
    print(f"  Average time: {avg_time:.2f}s")
    print(f"  Total time: {sum(times):.2f}s")
    print(f"  Expected range: {panel._get_expected_speed()}")
```

### Advanced Model Selection Override

```python
class CustomModelSelector:
    """Custom model selection logic."""

    @staticmethod
    def select_model_for_workload(image_size, batch_size, accuracy_priority=False):
        """Select model based on workload characteristics."""

        # Large batch processing: prefer speed
        if batch_size > 100:
            return "cpu", "MobileSAM"

        # High accuracy requirement: prefer SAM2
        if accuracy_priority:
            device, _, _ = detect_best_device()
            return device, "SAM2"

        # Small images: CPU is often sufficient
        if image_size < (512, 512):
            return "cpu", "MobileSAM"

        # Default to automatic detection
        return detect_best_device()[:2]

# Usage in custom application
class CustomGeoOSAMApp:
    def __init__(self, workload_config):
        self.workload = workload_config

        # Custom model selection
        device, model = CustomModelSelector.select_model_for_workload(
            self.workload['image_size'],
            self.workload['batch_size'],
            self.workload['accuracy_priority']
        )

        # Override environment
        if model == "MobileSAM":
            os.environ["GEOOSAM_FORCE_MOBILESAM"] = "1"

        # Initialize panel
        self.panel = GeoOSAMControlPanel(iface)
        print(f"Custom selection: {self.panel.model_choice} on {self.panel.device}")
```

---

## 📝 Development Guidelines

### Model-Aware Development

- **Test both models**: Ensure compatibility with SAM2 and MobileSAM
- **Handle device switching**: Code should work across GPU/CPU
- **Performance expectations**: Different models have different characteristics
- **Error handling**: Model-specific error scenarios

### Threading Considerations

- **CPU optimization**: Respect the threading configuration
- **GPU memory**: Handle CUDA out-of-memory gracefully
- **UI responsiveness**: Use the worker thread pattern
- **Resource cleanup**: Properly dispose of models and tensors

### Backward Compatibility

- **API consistency**: Maintain SAM2-compatible interfaces
- **Graceful fallbacks**: Handle missing dependencies
- **Version detection**: Check available models and capabilities
- **Documentation**: Update docs when adding model-specific features

**The enhanced API provides powerful model selection and performance optimization while maintaining simplicity for basic use cases.** 🚀
