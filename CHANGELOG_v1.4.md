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
