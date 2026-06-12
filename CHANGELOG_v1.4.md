# GeoOSAM v1.4.3 — Vegetation & Bbox Improvements

**Release Date:** 2026-06-12
**Type:** Improvements / Bug Fixes

---

## Improvements

### Better Vegetation Masks on Multispectral Imagery
Vegetation segmentation on multispectral rasters (5+ bands) now produces cleaner masks, with noticeably less shadow and soil bleed around tree crowns and crop boundaries. Applies automatically — no setting to enable, no effect on RGB imagery or non-vegetation classes.

## Fixes

- **BBox on high-resolution imagery:** small bounding-box selections were sometimes rejected on very high-resolution rasters. Boxes now register correctly regardless of raster resolution.
- **Batch mode queue recovery:** an error during batch processing could leave the request queue stuck, requiring a restart. Errors are now surfaced and the queue recovers cleanly.

---

# GeoOSAM v1.4.2 — Pro Features & Fixes

**Release Date:** 2026-06-04
**Type:** New Features / Bug Fixes / Security

---

## What's New

### Fill Holes (Pro)
New **Fill Holes** toggle in the Filters tab Pro block. Removes interior voids from segmented polygons automatically. A configurable **Max Hole Size (px²)** threshold preserves real large voids (courtyards, irrigation pivots) while filling small SAM mask artifacts. Default: 500 px².

### SAM3 Text Prompts Fixed
Text prompts in the Detect tab now use SAM3's native concept-prompting API (`SAM3SemanticPredictor`). Previously the text parameter was silently ignored — auto-segmentation ran regardless of what was typed.

## Fixes

- Removed debug print statements from production code (`geo_osam_dialog.py`, `sam3_clip_fix.py`, `geo_osam_license.py`)
- Pro licence offline cache: improved security and reliability
- Automated test suite added (41 tests covering helpers and mask pipeline — `python3 -m pytest tests/`)

## Upgrade Note

**Pro users:** please connect online once after updating to v1.4.2 to refresh your licence cache.

---

# GeoOSAM v1.4.0 — UI Redesign & Workflow Improvements

**Release Date:** 2026-05-09
**Type:** UI Redesign / New Features / Bug Fixes

---

## What's New

### Redesigned Panel Layout

- **4-tab flat layout** — Segment, Detect, Results, Filters
- **Settings as a gear overlay** — opens in place of the tabs, no tab clutter
- **Persistent log panel** — scrollable message log always visible below the tabs, hidden when Settings is open
- **Synced class selector** — class combo in Segment and Detect tabs stay in sync
- **Minimum panel width** increased to 360px to fit all tabs comfortably
- **Settings order** — License → Model → Classes → Output
- **Tooltips removed** for a cleaner look

### Similar from Selection

Use any existing segmented polygon as the exemplar for a similarity search:

1. Select a segmented feature with the QGIS selection tool
2. Click **Similar from Selection** in the Detect tab — button enables automatically when a feature is selected
3. GeoOSAM finds matching objects across the visible extent or entire raster

### Vector Extent ROI for Entire Raster (Pro)

Pro users running **Entire Raster** mode can now restrict processing to a vector polygon extent:

- Select any polygon layer from the **Vector Extent** dropdown in the Detect tab scope section
- Processing tiles that fall outside the polygon are skipped automatically
- Significantly reduces processing time on large rasters with a focused area of interest

### Detect Tab Improvements

- **Cancel button** now appears on the Detect tab during tiled processing
- **Undo button** available on the Detect tab
- **Find Similar** is independently toggleable — click again to exit, restoring text input and Auto-Segment

---

## Code Quality

- Resolved all Pylance unused-parameter warnings (signal callbacks and unused destructuring variables)

---

## Upgrade Guide

No breaking changes. All existing classes, output layers, and settings are preserved.

1. Update via QGIS Plugin Manager or `git pull`
2. Restart QGIS
