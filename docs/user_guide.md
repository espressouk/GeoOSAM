# GeoOSAM User Guide

## 🎯 Quick Start

### 1. First Time Setup

1. **Load a raster layer** in QGIS (satellite/aerial imagery)
2. **Click the GeoOSAM icon** 🛰️ in the toolbar
3. **Automatic model selection** happens instantly:
   - **🎮 GPU detected**: Downloads SAM 2.1 (~160MB, one-time)
   - **💻 CPU detected**: Downloads MobileSAM (~40MB via Ultralytics)
   - **⚡ High-core CPU**: Optimized for sub-second performance
4. **Control panel opens** on the right side showing your hardware

### 2. Basic Workflow

1. **Select Output Folder** (optional - defaults to `~/GeoOSAM_shapefiles`)
2. **Choose a Class** from the dropdown (e.g., "Buildings")
3. **Point mode activates automatically** 🎯
4. **Click on objects** to segment (expect <1s on powerful systems!)
5. **Export results** as professional shapefiles

---

## 🧠 Intelligent Performance System

### Hardware Detection & Optimization

GeoOSAM automatically detects your hardware and optimizes accordingly:

| Your Hardware  | Model Used | Expected Speed | What You'll See               |
| -------------- | ---------- | -------------- | ----------------------------- |
| NVIDIA RTX GPU | SAM 2.1    | 0.2-0.5s       | 🎮 CUDA (SAM2.1)              |
| Apple M1/M2/M3 | SAM 2.1    | 1-2s           | 🍎 MPS (SAM2.1)               |
| 24+ Core CPU   | MobileSAM  | **<1s**        | 💻 CPU (MobileSAM) (24 cores) |
| 16+ Core CPU   | MobileSAM  | 1-2s           | 💻 CPU (MobileSAM) (16 cores) |
| 8-16 Core CPU  | MobileSAM  | 2-3s           | 💻 CPU (MobileSAM) (12 cores) |
| 4-8 Core CPU   | MobileSAM  | 3-5s           | 💻 CPU (MobileSAM) (6 cores)  |

**🚀 Performance Highlights:**

- **High-end CPUs**: Sub-second segmentation rivals GPU performance
- **Automatic Threading**: Uses 75% of available cores intelligently
- **MobileSAM Efficiency**: 5x smaller, exceptional multi-core scaling
- **Zero Configuration**: Works optimally out-of-the-box

---

## 📋 Detailed Instructions

### Output Settings

#### 📁 **Custom Output Folder**

- Click **"📁 Choose"** to select where shapefiles are saved
- Default: `~/GeoOSAM_shapefiles`
- Creates separate folders for shapefiles and debug masks
- **Tip**: Use project-specific folders for better organization

#### 💾 **Debug Masks** (Optional)

- Check **"💾 Save debug masks"** to save raw segmentation images
- **Default**: Disabled for optimal performance
- Useful for troubleshooting and quality control
- Files saved as PNG with timestamps and class names

### Class Selection

#### 📋 **Pre-defined Classes**

GeoOSAM includes 12 ready-to-use classes optimized for various use cases:

| Class           | Color      | Best For                            | Optimal Mode |
| --------------- | ---------- | ----------------------------------- | ------------ |
| **Buildings**   | Red        | Residential & commercial structures | Point        |
| **Roads**       | Gray       | Streets, highways, pathways         | BBox         |
| **Vegetation**  | Green      | Trees, grass, parks                 | BBox         |
| **Water**       | Blue       | Rivers, lakes, ponds                | BBox         |
| **Agriculture** | Gold       | Farmland, crops                     | BBox         |
| **Vehicle**     | Red-Orange | Cars, trucks, buses                 | Point        |
| **Ship**        | Cyan       | Boats, vessels                      | Point        |
| **Parking**     | Orange     | Parking lots, areas                 | BBox         |
| **Industrial**  | Purple     | Factories, warehouses               | Point/BBox   |
| **Residential** | Pink       | Housing areas                       | BBox         |
| **Commercial**  | Light Blue | Shopping, business districts        | BBox         |
| **Other**       | Purple     | Unclassified objects                | Point        |

#### ➕ **Adding Custom Classes**

1. Click **"➕ Add"**
2. Enter class name (e.g., "Solar Panels", "Wind Turbines")
3. Color assigned automatically from palette
4. **Best Practice**: Use descriptive names for later analysis

#### ✏️ **Editing Classes**

1. Click **"✏️ Edit"**
2. Select class to modify
3. Change name or color (RGB format: `255,0,0`)
4. Colors update automatically in map visualization
5. **Tip**: Use logical color schemes (blue for water, green for vegetation)

### Segmentation Experience

#### 🎯 **Point Mode** (Default & Recommended)

**Automatically activated when you select a class**

**Best for:** Buildings, vehicles, trees, ships, specific objects

**How it works:**

- **SAM 2.1** (GPU): Uses transformer architecture for precise boundaries
- **MobileSAM** (CPU): Uses efficient Tiny-ViT encoder for speed

**Usage:**

1. Class selection automatically activates Point mode
2. Click anywhere on the object you want to segment
3. AI automatically detects the entire object boundary
4. **Performance**: <1s on high-end systems, 0.2-5s depending on hardware

**Pro Tips:**

- Click near the center of objects for best results
- Works excellent on clearly defined objects
- Faster processing than BBox mode
- Perfect for scattered objects (individual buildings, vehicles)

#### 📦 **BBox Mode** (Available but hidden by default)

**Best for:** Large uniform areas, agricultural fields, water bodies

**How to access:** Currently hidden in UI but available in code

**Usage:**

1. Draw rectangle around target area
2. AI segments all similar objects within the box
3. Good for large-scale mapping projects

### Enhanced Workflow Features

#### ↶ **Undo Last Polygon** (New!)

- Click **"↶ Undo Last Polygon"** to remove recent additions
- Removes all polygons from the most recent segmentation operation
- **Intelligent tracking**: Knows exactly which features to remove
- Updates feature counts and layer names automatically
- **Use case**: Perfect for correcting mistakes without losing other work

#### 🔄 **Automatic Raster Selection**

- **Default behavior**: Keeps raster layer selected after segmentation
- Ensures continuous workflow without manual layer switching
- Automatically finds raster layers if none selected
- **Smart behavior**: Only switches when necessary

#### ⚡ **Real-time Performance Monitoring**

- Status panel shows actual processing times
- Device information displayed: "🎮 CUDA (SAM2.1)" or "💻 CPU (MobileSAM) (24 cores)"
- Progress updates during processing
- **Benchmark your system**: Times displayed after each segmentation

---

## 🎨 Working with Results

### Layer Management

Each class creates an intelligently named layer:

- **Format**: `SAM_Buildings (5 parts) [RGB:220,20,60]`
- **Information**: Shows feature count and color coding
- **Updates**: Names update automatically as you add features
- **Organization**: Each class gets its own layer for easy management

### Rich Attribute Data

Each polygon includes comprehensive metadata:

- **segment_id:** Unique identifier within class
- **class_name:** Assigned class name
- **class_color:** RGB color code for visualization
- **method:** Segmentation method (Point/BBox)
- **timestamp:** Precise creation time
- **mask_file:** Debug file reference (if enabled)
- **crop_size:** Processing dimensions used
- **canvas_scale:** Map zoom level when created

**Analysis Value**: Use attributes for quality control, temporal analysis, and processing statistics.

### Professional Export Options

- **💾 Export All:** Saves all classes as separate shapefiles with timestamps
- **Individual Export:** Right-click layer → Export for specific classes
- **Formats Supported:** Shapefile (recommended), GeoJSON, KML, GPX
- **Projection Handling:** Maintains original raster CRS automatically
- **Attributes Preserved**: All metadata included in exports

---

## 💡 Best Practices for Optimal Results

### 🎯 **Hardware-Specific Tips**

#### **GPU Users (NVIDIA/Apple Silicon)**

- **Expect**: 0.2-2s per segment with SAM 2.1
- **Best for**: Highest accuracy on complex objects
- **Tip**: Process larger areas due to fast speeds

#### **High-Core CPU Users (16+ cores)**

- **Expect**: <1-2s per segment with MobileSAM
- **Performance**: Rivals GPU systems
- **Tip**: Excellent for large-scale projects without GPU

#### **Standard CPU Users (4-16 cores)**

- **Expect**: 2-5s per segment with MobileSAM
- **Still efficient**: Much faster than traditional methods
- **Tip**: Process in smaller batches for best workflow

### 🎯 **Image Quality Optimization**

#### **Resolution Guidelines**

- **Optimal**: <1m/pixel for buildings, <0.5m for vehicles
- **Minimum**: 2m/pixel for large objects
- **MobileSAM advantage**: Works well even with lower resolution

#### **Image Characteristics**

- **Best**: High contrast RGB imagery
- **Good**: Multispectral with clear boundaries
- **Avoid**: Heavily compressed or blurry imagery
- **Tip**: Both models handle various image types well

### 🎯 **Efficient Segmentation Strategy**

#### **Class-by-Class Approach**

1. **Plan classes** before starting (use pre-defined when possible)
2. **Segment systematically** (all buildings, then all vehicles)
3. **Use consistent zoom** for similar object types
4. **Export frequently** to avoid data loss

#### **Click Strategy**

- **Point Mode**: Click near object centers for best boundary detection
- **Avoid edges**: Both SAM 2.1 and MobileSAM work better from object centers
- **Consistent scale**: Maintain appropriate zoom for object size
- **Quick workflow**: Modern performance allows rapid clicking

---

## 🚀 Advanced Workflows

### Urban Analysis Project

**Hardware**: Any (optimized automatically)
**Expected Time**: 100 buildings in 5-10 minutes depending on hardware

1. **Setup:** Load high-resolution urban imagery (0.5m or better)
2. **Buildings:** Select "Buildings" class, Point mode activates automatically
3. **Strategy:** Systematic clicking on building centers
4. **Vehicles:** Switch to "Vehicle" class for parking areas
5. **Quality Control:** Use Undo for any imprecise segments
6. **Export:** Professional shapefiles with full attribute data

### Environmental Monitoring

**Hardware**: CPU systems excellent for this workflow
**Expected Time**: Large areas processed efficiently with MobileSAM

1. **Setup:** Load multispectral or high-res RGB imagery
2. **Water Bodies:** Select "Water" class, works great with Point mode
3. **Vegetation:** "Vegetation" class for forest patches and parks
4. **Agriculture:** "Agriculture" class for field boundaries
5. **Analysis:** Export for change detection and temporal studies

### Disaster Response Mapping

**Hardware**: GPU preferred for speed, but CPU systems very capable
**Expected Time**: Rapid assessment possible with modern performance

1. **Setup:** Load post-event imagery
2. **Damage Assessment:** Custom classes for damage levels
3. **Infrastructure:** "Buildings" class to assess structural damage
4. **Access Routes:** "Roads" class for accessibility analysis
5. **Report Generation:** Rich attributes enable detailed reporting

### Transportation Analysis

**Hardware**: MobileSAM excellent for vehicle detection
**Expected Time**: Sub-second per vehicle on high-end systems

1. **Setup:** High-resolution imagery of transportation hubs
2. **Vehicles:** "Vehicle" class with Point mode for individual vehicles
3. **Ships:** "Ship" class for maritime facilities
4. **Infrastructure:** "Parking" class for facility analysis
5. **Traffic Analysis:** Export with timestamps for temporal analysis

---

## ⚡ Performance Optimization

### Getting Maximum Speed

#### **For All Systems**

- **Zoom appropriately**: Closer zoom = smaller processing area = faster results
- **Use Point mode**: Generally faster than BBox for individual objects
- **Batch by class**: Process all buildings, then all vehicles, etc.
- **Close other apps**: Free up system resources

#### **For CPU Systems**

- **MobileSAM advantage**: Specially optimized for CPU efficiency
- **Threading**: Plugin automatically uses optimal core count
- **Memory**: 16GB+ RAM recommended for large imagery

#### **For GPU Systems**

- **SAM 2.1 advantage**: Latest accuracy improvements
- **VRAM**: 4GB+ recommended for best performance
- **Fallback**: Automatic CPU fallback if GPU memory insufficient

### Troubleshooting Performance

#### **Slower than Expected**

1. **Check device detection**: Look at status panel for hardware info
2. **Verify model**: Should show SAM2.1 (GPU) or MobileSAM (CPU)
3. **Close applications**: Free up system resources
4. **Check zoom level**: Closer zoom = smaller processing area

#### **Model Selection Issues**

```python
# Force CPU mode if needed (in QGIS Python Console):
import os
os.environ["GEOOSAM_FORCE_CPU"] = "1"
# Restart QGIS
```

---

## ⚠️ Common Issues & Solutions

### When Segmentation Doesn't Work

#### **"No segments found"**

- **Try different click position**: Move from edge to center
- **Check image quality**: Ensure sufficient contrast
- **Verify zoom level**: Too far out can cause issues
- **Switch classes**: Some objects work better with different classes

#### **"No raster layer selected"**

- **Solution**: Plugin automatically finds raster layers
- **Manual fix**: Select any raster layer in Layers panel
- **Check layer type**: Ensure you're not on a vector layer

#### **Segmentation too slow**

- **Check hardware detection**: Status should show your actual hardware
- **Zoom in**: Reduce processing area size
- **Close apps**: Free up system resources
- **Normal ranges**: 0.2-5s depending on hardware is normal

### Model Download Issues

#### **MobileSAM download fails**

- **Automatic retry**: Ultralytics handles retries automatically
- **Internet check**: Verify connection for first-time download
- **Manual test**: Try in QGIS Python Console: `from ultralytics import SAM; SAM('mobile_sam.pt')`

#### **SAM 2.1 download fails**

- **Automatic fallback**: Plugin will retry or fallback to CPU
- **Manual download**: See installation guide for manual steps
- **Check space**: Ensure 200MB+ free space

---

## 📞 Getting Help

### Diagnostic Information

**When reporting issues, include:**

```python
# Run in QGIS Python Console for diagnostic info:
from geo_osam_dialog import detect_best_device
device, model_choice, cores = detect_best_device()
print(f"Hardware: {device.upper()}")
print(f"Model: {model_choice}")
print(f"Cores: {cores if cores else 'N/A'}")

import torch
print(f"PyTorch: {torch.__version__}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
```

### Support Channels

- **GitHub Issues:** Bug reports and feature requests
- **Email:** bkst.dev@gmail.com for direct support
- **Documentation:** Check troubleshooting guide for common solutions

### Community

- **QGIS Hub:** Rate and review the plugin
- **Share Results:** Show off your segmentation projects
- **Contribute:** Suggest new classes or improvements

---

**Happy segmenting with intelligent AI optimization!** 🛰️

Your system automatically uses the best model for your hardware - from sub-second CPU performance to cutting-edge GPU accuracy.
