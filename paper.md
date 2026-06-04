---
title: 'GeoOSAM: An Interactive AI Segmentation Plugin for QGIS Integrating SAM 2.1 and SAM3'
tags:
  - Python
  - QGIS
  - remote sensing
  - image segmentation
  - geospatial AI
  - deep learning
  - SAM
authors:
  - name: Ofer Butbega
    orcid: 0009-0002-0724-5767
    affiliation: 1
affiliations:
  - name: Independent Researcher
    index: 1
date: 4 June 2026
bibliography: paper.bib
---

# Summary

GeoOSAM is an open-source QGIS plugin that brings Meta's Segment Anything Model 2.1 (SAM 2.1) [@ravi2024sam2] and SAM3 [@carion2025sam3] to a professional geographic information system (GIS) workflow. Through a four-tab dock panel (Segment, Detect, Results, Filters), users can delineate objects in any raster layer — satellite imagery, aerial photography, UAV captures, or online tile services — with a single point click or bounding-box draw, without writing any code. Segmented masks are immediately converted to georeferenced vector polygons and added to the QGIS layer tree as styled feature layers, ready for attribute editing, filtering, and multi-format export (GeoPackage, Shapefile, GeoJSON, FlatGeobuf).

Beyond interactive segmentation, GeoOSAM exposes SAM3's semantic capabilities through a Detect tab: users can describe objects in natural language ("find all vehicles"), click a reference object to find visually similar ones across a scene, or use any existing polygon as an exemplar query. All detection modes support automatic tiling for large rasters and can be spatially restricted to a vector polygon region of interest (ROI). The plugin has accumulated 12,423 downloads and 370 community ratings on the QGIS Plugin Repository as of June 2026, with approximately 1,600 downloads in the preceding 30 days. Downloads have been recorded in more than 20 countries spanning six continents, including the United States, Brazil, China, India, Indonesia, Germany, and Burkina Faso.

# Statement of Need

Manual digitisation of objects in remote sensing imagery remains one of the most time-consuming tasks in operational GIS workflows. Supervised classification pipelines require labelled training data and domain expertise; traditional edge-detection and thresholding approaches are brittle across sensor types and illumination conditions. The emergence of foundation models for image segmentation — particularly Meta's Segment Anything family — has dramatically lowered the barrier to high-quality object delineation, but most implementations target Python scripting environments rather than the point-and-click workflows used daily by GIS practitioners, planners, ecologists, and humanitarian mapping teams.

Existing QGIS-integrated SAM tools either lack current model support or omit the GIS-specific workflow features required for operational use. Geo-SAM [@zhao2023geosam] supports SAM 1.0 via a manual encoder pre-processing step that must be re-run each time the image or resolution changes. The `samgeo` Python package [@wu2023samgeo] and its companion QGIS plugin support SAM 2.1 and SAM3 text prompts, but neither provides exemplar-based similar-object detection, GSD-aware size and shape filters, nor a vector polygon ROI constraint for restricting tiled processing to a defined area of interest.

GeoOSAM fills this gap with a no-code, model-current plugin designed for the full GIS user spectrum: from a field ecologist digitising vegetation patches to an analyst running city-scale vehicle detection on satellite imagery. The plugin operates under a dual-tier model: all interactive segmentation and detection within the visible extent are available without charge under the GPL v2 licence, while full-raster tiled processing requires a Pro licence to sustain ongoing development.

Community feedback confirms adoption across a wide range of research and professional disciplines. Reported use cases include archaeological survey (delineation of individual stones and boulders in drone orthophotos, with users reporting the elimination of weeks of manual digitisation), ecological and biological monitoring of habitats in aerial and UAV imagery, agricultural weed detection at sub-centimetre ground sampling distances, infrastructure inspection (crack detection in high-resolution orthophotos), and environmental monitoring for litter detection in ultra-high-resolution urban imagery. Users span academic researchers, environmental consultancies, and geospatial service providers across Europe, Australia, and Asia.

# Design and Architecture

## Dual-model inference strategy

GeoOSAM automatically detects the available hardware at startup and selects the appropriate inference path:

- **GPU systems**: Meta SAM 2.1 [@ravi2024sam2], loaded via the `sam2` library with Hydra configuration [@yadan2019hydra]. Four model sizes (Tiny 156 MB through Large 898 MB) are offered; selection is exposed to the user via a dropdown. VRAM requirements per model are reported in the Performance section.
- **CPU systems**: Ultralytics SAM 2.1 [@ultralytics2024] variants (T/B/L), which are optimised for multi-threaded CPU inference and require no GPU. Benchmarking shows that performance peaks between 8 and 16 threads on a 32-core system; on systems with 16 or more cores the plugin allocates 75% of available cores, with more conservative defaults on smaller systems, to leave headroom for the QGIS UI thread.

SAM3 [@carion2025sam3] is available as a third path on GPU systems, providing automatic instance segmentation and concept-prompt modes via the Ultralytics implementation.

## Worker thread architecture

All model inference runs on `QThread` workers — `OptimizedSAM2Worker` for interactive point and bounding-box prompts, and `TiledSegmentationWorker` for full-raster detection jobs. This design keeps QGIS responsive during long operations and exposes cancellation signals so users can abort mid-tile runs without restarting the application.

## Class-specific detection helpers

The `helpers/` module implements a clean abstract base class (`BaseDetectionHelper`) with nine subclasses, one per land-cover or object class (buildings, vegetation, vehicles, vessels, water, roads, agriculture, residential, general). Each subclass overrides candidate detection, morphological post-processing, and object validation logic tuned to the visual and size characteristics of its class. The factory function `create_detection_helper()` decouples class selection from the inference pipeline.

## Mask-to-vector pipeline

Raw inference masks are converted to georeferenced vector polygons via `rasterio.features.shapes` and `shapely` [@shapely2007], with the spatial transform derived directly from the source raster's affine metadata. This ensures correct geographic coordinates regardless of the raster's projection or resolution. Per-feature attributes — class name, segment ID, area (m²), diameter (cm), circularity, and compactness — are written to the output layer using QGIS field objects, enabling downstream filtering and analysis within the native QGIS attribute table.

# Functionality

## Segment tab

The Segment tab supports point-click and bounding-box prompts. Multiple positive and negative points can be accumulated before committing a prediction (Shift+click adds, Ctrl+click subtracts), enabling users to interactively refine ambiguous boundaries. Each confirmed mask is added as a polygon to the active class layer with undo support.

## Detect tab (SAM3)

The Detect tab exposes three SAM3-powered modes:

- **Text prompt**: The user types a class description; SAM3 runs concept-prompted segmentation across the visible extent or the full raster (with tiling), and results are filtered by the detection helper for the selected class. This mode requires the Ultralytics CLIP fork (`pip install git+https://github.com/ultralytics/CLIP.git`); a runtime compatibility shim is included to handle tokenizer differences across CLIP versions (see `sam3_clip_fix.py`).
- **Find Similar**: The user clicks a reference object; GeoOSAM extracts its mask as an exemplar and runs SAM3 in a sliding-window search, returning objects with similar visual appearance.
- **Similar from Selection**: Any existing polygon in the QGIS layer can be used as an exemplar, allowing cross-session or cross-layer reference queries.

## Filters tab

Real-world size and shape filters are applied post-segmentation: minimum and maximum diameter (cm) and area (m²) calculated from the raster's ground sampling distance (GSD), plus circularity, compactness, and aspect-ratio thresholds. This dramatically reduces false positives in text-prompt runs on heterogeneous imagery.

## Export and class management

Twelve pre-defined classes with colour coding cover the most common mapping targets. Custom classes can be added at runtime. Export writes all active class layers to user-selected formats in a single operation, preserving the full attribute schema.

# Performance

All benchmarks were measured on a system with an NVIDIA GeForce RTX 4080 Laptop GPU and a 32-core CPU, using an 8-run median over a synthetic 1024×1024 RGB image after two warmup passes.

## Inference time by model (point prompt, 1024×1024)

| Model | Device | Median inference |
|---|---|---|
| SAM2.1 Tiny | GPU | 55.5 ms |
| SAM2.1 Small | GPU | 62.9 ms |
| SAM2.1 Base+ | GPU | 111.6 ms |
| SAM2.1 Large | GPU | 250.7 ms |
| SAM2.1 Tiny (Meta) | CPU (32-core) | 522 ms |
| SAM2.1_B (Ultralytics) | CPU (32-core) | 1080 ms |
| SAM3 | GPU | 799 ms |

GPU inference with SAM2.1 Tiny or Small is imperceptibly fast at interactive rates (<65 ms). CPU inference is suitable for asynchronous batch use; the Ultralytics CPU path provides the best compatibility on machines without CUDA.

## Scaling with image size (SAM2.1 Tiny, GPU)

| Image size | Median inference |
|---|---|
| 256×256 | 54.2 ms |
| 512×512 | 54.5 ms |
| 1024×1024 | 55.9 ms |
| 2048×2048 | 61.3 ms |
| 4096×4096 | 111.5 ms |

Inference time is near-constant up to 2048×2048 and doubles only at 4096×4096, demonstrating that the image encoder is the dominant cost and scales sub-linearly with resolution.

## GPU memory requirements

| Model | Peak VRAM (inference) |
|---|---|
| SAM2.1 Tiny | 597 MB |
| SAM2.1 Small | 630 MB |
| SAM2.1 Base+ | 808 MB |
| SAM2.1 Large | 1481 MB |
| SAM3 | 3160 MB |

All SAM2.1 variants run on any GPU with 2 GB VRAM; SAM3 requires approximately 4 GB.

## CPU thread scaling (SAM2.1 Tiny)

| Threads | Median inference |
|---|---|
| 1 | 3069 ms |
| 4 | 993 ms |
| 8 | 872 ms |
| 16 | 773 ms |
| 32 | 2180 ms |

Performance peaks at 8–16 threads; hyperthreading contention degrades throughput beyond 16 threads on this system, validating the plugin's 75%-of-cores default.

## Mask-to-vector conversion

Conversion of an inference mask to georeferenced QGIS polygons adds 8–35 ms for typical imagery sizes (512–2048 px), dominated by raster scan time rather than object count, and is imperceptible in interactive use.

# Comparison to Similar Tools

| Feature | GeoOSAM | Geo-SAM [@zhao2023geosam] | samgeo + QGIS plugin [@wu2023samgeo] |
|---|---|---|---|
| QGIS native UI | Yes | Yes | Yes |
| SAM 2.1 support | Yes | No (SAM 1.0 only) | Yes |
| SAM3 / text prompts | Yes | No | Yes |
| Exemplar similar-object detection | Yes | No | No |
| Automatic inference-time tiling | Yes | No† | Yes |
| Vector ROI for tiled processing | Yes | No | No |
| GSD-based size and shape filters | Yes | No | No |
| No-code QGIS operation | Yes | Yes | Yes |

†Geo-SAM supports encoder-based tiling as a separate manual pre-processing step.

# Acknowledgements

The author thanks the QGIS community for plugin infrastructure and the Meta AI Research team for releasing the SAM 2.1 model weights under the Apache 2.0 licence and the SAM3 model weights under the SAM Licence.

**AI assistance disclosure**: Portions of this paper were drafted with the assistance of Claude (Anthropic), consistent with the JOSS AI usage policy. All scientific claims, benchmark measurements, and architectural descriptions were verified and approved by the author.

# References
