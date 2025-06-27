# GeoOSAM User Guide

## 🎯 Quick Start

### 1. First Time Setup

1. **Load a raster layer** in QGIS (satellite/aerial imagery)
2. **Click the GeoOSAM icon** 🛰️ in the toolbar
3. **Wait for SAM2 model download** (automatic, ~160MB, one-time only)
4. **Control panel opens** on the right side

### 2. Basic Workflow

1. **Select Output Folder** (optional - defaults to `~/GeoOSAM_shapefiles`)
2. **Choose a Class** from the dropdown (e.g., "Buildings")
3. **Select Segmentation Mode**: Point 🎯 or BBox 📦
4. **Click on the map** to segment
5. **Export results** as shapefiles

---

## 📋 Detailed Instructions

### Output Settings

#### 📁 **Custom Output Folder**

- Click **"📁 Choose"** to select where shapefiles are saved
- Default: `~/GeoOSAM_shapefiles`
- Creates separate folders for shapefiles and debug masks

#### 💾 **Debug Masks** (Optional)

- Check **"💾 Save debug masks"** to save raw segmentation images
- Useful for troubleshooting and analysis
- Files saved as PNG with timestamps

### Class Selection

#### 📋 **Pre-defined Classes**

GeoOSAM includes 12 ready-to-use classes:

| Class           | Color      | Best For                            |
| --------------- | ---------- | ----------------------------------- |
| **Buildings**   | Red        | Residential & commercial structures |
| **Roads**       | Gray       | Streets, highways, pathways         |
| **Vegetation**  | Green      | Trees, grass, parks                 |
| **Water**       | Blue       | Rivers, lakes, ponds                |
| **Agriculture** | Gold       | Farmland, crops                     |
| **Parking**     | Orange     | Parking lots, areas                 |
| **Industrial**  | Purple     | Factories, warehouses               |
| **Residential** | Pink       | Housing areas                       |
| **Commercial**  | Light Blue | Shopping, business districts        |
| **Vehicle**     | Red-Orange | Cars, trucks, buses                 |
| **Ship**        | Cyan       | Boats, vessels                      |
| **Other**       | Purple     | Unclassified objects                |

#### ➕ **Adding Custom Classes**

1. Click **"➕ Add"**
2. Enter class name (e.g., "Solar Panels")
3. Color assigned automatically
4. Use **"✏️ Edit"** to modify colors

#### ✏️ **Editing Classes**

1. Click **"✏️ Edit"**
2. Select class to modify
3. Change name or color (RGB format: `255,0,0`)
4. Colors update automatically in map

### Segmentation Modes

#### 🎯 **Point Mode** (Recommended for most objects)

**Best for:** Buildings, vehicles, trees, specific objects

**How to use:**

1. Click **"🎯 Point Mode"**
2. Click anywhere on the object you want to segment
3. SAM automatically detects the entire object
4. Works best on clearly defined objects

**Tips:**

- Click near the center of objects
- Use for objects with clear boundaries
- Faster than BBox mode
- Great for scattered objects

#### 📦 **BBox Mode** (For large areas)

**Best for:** Agricultural fields, water bodies, large areas

**How to use:**

1. Click **"📦 BBox Mode"**
2. Click and drag to draw a rectangle
3. Release to segment everything in the box
4. SAM segments all similar objects in the area

**Tips:**

- Draw tight boxes around target areas
- Good for uniform areas (fields, water)
- Segments multiple connected objects
- Use for large-scale mapping

### Workflow Features

#### ↶ **Undo Last Polygon**

- Click **"↶ Undo Last Polygon"** to remove recent additions
- Removes all polygons from the last segmentation
- Useful for correcting mistakes
- Updates feature counts automatically

#### 🔄 **Clear Selection**

- Clears current point/bbox selection
- Keeps the tool active for next selection
- Useful to start over without changing mode

#### 🆕 **New Shape**

- Removes all current results
- Resets the workspace
- Use when starting a completely new project

#### 📡 **Reselect Raster**

- Automatically selects a raster layer
- Use if you accidentally selected a vector layer
- Ensures segmentation works on imagery

### Advanced Features

#### 🎯 **Keep Raster Selected**

- **Enabled (default):** Raster layer stays selected after segmentation
- **Disabled:** Vector result layers become active
- Recommended: Keep enabled for continuous segmentation

#### ⚙️ **Performance Optimization**

- **GPU Detection:** Automatically uses NVIDIA/Apple Silicon GPU
- **CPU Fallback:** Uses multi-core CPU if no GPU
- **Adaptive Crop Size:** Adjusts processing area based on zoom level
- **Threading:** UI stays responsive during processing

---

## 🎨 Working with Results

### Layer Management

- Each class creates a separate layer: `SAM_Buildings (5 parts) [RGB:220,20,60]`
- Layers show feature count and color coding
- Results are temporary until exported

### Attribute Data

Each polygon includes detailed information:

- **segment_id:** Unique identifier
- **class_name:** Assigned class
- **class_color:** RGB color code
- **method:** Point or BBox
- **timestamp:** When created
- **mask_file:** Debug file (if enabled)
- **crop_size:** Processing dimensions
- **canvas_scale:** Map zoom level

### Export Options

- **💾 Export All:** Saves all classes as separate shapefiles
- **Individual Export:** Right-click layer → Export
- **Formats:** Shapefile (recommended), GeoJSON, KML
- **Projection:** Maintains original raster CRS

---

## 💡 Best Practices

### For Best Results

#### 🎯 **Image Quality**

- Use high-resolution imagery (< 1m/pixel preferred)
- Ensure good contrast between objects
- RGB images work better than single-band
- Avoid heavily compressed imagery

#### 🎯 **Object Selection**

- **Point Mode:** Click near object centers
- **BBox Mode:** Draw tight, precise rectangles
- Start with obvious, well-defined objects
- Segment similar objects in batches

#### 🎯 **Class Strategy**

- Use specific classes rather than generic ones
- Create custom classes for specialized projects
- Consistent naming helps with analysis
- Color-code logically (blue for water, green for vegetation)

### Workflow Efficiency

#### 📈 **Systematic Approach**

1. **Plan your classes** before starting
2. **Segment by class type** (all buildings, then all roads)
3. **Use consistent zoom levels** for similar objects
4. **Export frequently** to avoid data loss
5. **Organize output folders** by project/date

#### 📈 **Large Projects**

- Process in smaller geographic sections
- Use consistent classification schemes
- Save projects frequently
- Consider processing time vs. accuracy trade-offs

---

## 🎯 Example Workflows

### Urban Mapping Project

1. **Setup:** Load high-res urban imagery
2. **Buildings:** Use Point mode, click on individual buildings
3. **Roads:** Use BBox mode for road segments
4. **Vehicles:** Use Point mode on parking areas
5. **Export:** Separate shapefiles for each class

### Environmental Monitoring

1. **Setup:** Load multispectral satellite imagery
2. **Water Bodies:** Use BBox mode for lakes/rivers
3. **Vegetation:** Use Point mode for forest patches
4. **Agriculture:** Use BBox mode for field boundaries
5. **Analysis:** Export for change detection studies

### Infrastructure Assessment

1. **Setup:** Load post-disaster imagery
2. **Buildings:** Point mode to assess damage
3. **Roads:** BBox mode for accessibility analysis
4. **Debris:** Custom class for damage assessment
5. **Report:** Export with detailed attributes

---

## ⚠️ Common Pitfalls

### Avoid These Mistakes

- **Don't click vector layers** - SAM needs raster imagery
- **Don't use blurry imagery** - Results will be poor
- **Don't rush segmentation** - Take time for precise clicking
- **Don't ignore the scale** - Zoom appropriately for object size
- **Don't forget to export** - Results are temporary

### When Segmentation Fails

- **Try different click positions** - Center vs. edge
- **Adjust zoom level** - Closer for small objects
- **Switch modes** - Point vs. BBox
- **Check image quality** - Contrast and resolution
- **Use Undo** - Remove poor results and try again

---

## 🚀 Advanced Tips

### Power User Features

- **Keyboard shortcuts:** Space to clear selection
- **Batch processing:** Segment similar objects quickly
- **Quality control:** Use debug masks to verify results
- **Data validation:** Check attribute tables before export

### Integration with QGIS

- **Styling:** Customize layer symbology
- **Analysis:** Use QGIS tools on segmentation results
- **Plugins:** Combine with other QGIS plugins
- **Processing:** Use in QGIS models and scripts

---

## 📞 Getting Help

### If You're Stuck

1. **Check this guide** - Most answers are here
2. **Try Troubleshooting guide** - Common solutions
3. **GitHub Issues** - Report bugs or ask questions
4. **QGIS Community** - General QGIS help

### Providing Feedback

- **GitHub Issues:** Bug reports and feature requests
- **Email:** bkst.dev@gmail.com for direct support
- **QGIS Hub:** Rate and review the plugin

**Happy segmenting!** 🛰️
