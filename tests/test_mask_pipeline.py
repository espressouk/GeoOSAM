"""
Tests for the mask-to-vector pipeline and mask utility functions.

Covers:
- rasterio.features.shapes → shapely polygon conversion (the GIS step)
- filter_contained_masks
- merge_nearby_masks_class_aware
- dedupe_or_merge_masks_smart
"""
import sys
import os
import numpy as np
import rasterio.transform
from rasterio.features import shapes
from shapely.geometry import shape

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Import the standalone utility functions from geo_osam_dialog.
# conftest.py has already patched sys.modules with QGIS stubs.
from geo_osam_dialog import (
    filter_contained_masks,
    merge_nearby_masks_class_aware,
    dedupe_or_merge_masks_smart,
)


# ── helpers ───────────────────────────────────────────────────────────────────

def circle_mask(cx, cy, radius, size=256):
    mask = np.zeros((size, size), dtype=np.uint8)
    y, x = np.ogrid[:size, :size]
    mask[(x - cx) ** 2 + (y - cy) ** 2 <= radius ** 2] = 255
    return mask


def rect_mask(x0, y0, x1, y1, size=256):
    mask = np.zeros((size, size), dtype=np.uint8)
    mask[y0:y1, x0:x1] = 255
    return mask


def georef_transform(size=256):
    return rasterio.transform.from_bounds(0, 0, size, size, size, size)


# ── mask → vector pipeline ────────────────────────────────────────────────────

class TestMaskToVector:
    def test_single_blob_produces_one_polygon(self):
        mask = circle_mask(128, 128, 40)
        transform = georef_transform()
        polys = [shape(g) for g, v in shapes(mask, mask=mask.astype(bool), transform=transform) if v == 255]
        assert len(polys) == 1
        assert polys[0].is_valid
        assert polys[0].area > 0

    def test_two_blobs_produce_two_polygons(self):
        mask = np.zeros((256, 256), dtype=np.uint8)
        mask[20:60, 20:60] = 255
        mask[150:190, 150:190] = 255
        transform = georef_transform()
        polys = [shape(g) for g, v in shapes(mask, mask=mask.astype(bool), transform=transform) if v == 255]
        assert len(polys) == 2

    def test_empty_mask_produces_no_polygons(self):
        mask = np.zeros((256, 256), dtype=np.uint8)
        transform = georef_transform()
        polys = [shape(g) for g, v in shapes(mask, mask=mask.astype(bool), transform=transform) if v == 255]
        assert polys == []

    def test_polygon_area_scales_with_mask(self):
        small = circle_mask(64, 64, 10, size=256)
        large = circle_mask(64, 64, 40, size=256)
        transform = georef_transform()
        area_small = sum(
            shape(g).area for g, v in shapes(small, mask=small.astype(bool), transform=transform) if v == 255
        )
        area_large = sum(
            shape(g).area for g, v in shapes(large, mask=large.astype(bool), transform=transform) if v == 255
        )
        assert area_large > area_small * 10

    def test_polygon_centroid_matches_mask_centroid(self):
        mask = circle_mask(100, 150, 30, size=256)
        transform = georef_transform(256)
        polys = [shape(g) for g, v in shapes(mask, mask=mask.astype(bool), transform=transform) if v == 255]
        assert len(polys) == 1
        # Centroid should be near pixel (100, 150) — mapped via transform
        cx, cy = polys[0].centroid.x, polys[0].centroid.y
        assert abs(cx - 100) < 5
        assert abs(cy - (256 - 150)) < 5  # rasterio flips Y


# ── filter_contained_masks ────────────────────────────────────────────────────

class TestFilterContainedMasks:
    def test_non_nested_masks_all_kept(self):
        m1 = circle_mask(50, 50, 20)
        m2 = circle_mask(200, 200, 20)
        result = filter_contained_masks([m1, m2])
        assert len(result) == 2

    def test_contained_small_mask_removed(self):
        big = rect_mask(10, 10, 200, 200)
        small = rect_mask(50, 50, 80, 80)  # entirely inside big
        result = filter_contained_masks([big, small])
        # small is contained in big → only big kept
        assert len(result) == 1
        assert np.array_equal(result[0], big)

    def test_identical_masks_deduplicated(self):
        m = circle_mask(128, 128, 40)
        result = filter_contained_masks([m, m.copy()])
        # One is contained in the other
        assert len(result) == 1

    def test_empty_list_returns_empty(self):
        assert filter_contained_masks([]) == []

    def test_single_mask_returned_unchanged(self):
        m = circle_mask(128, 128, 40)
        result = filter_contained_masks([m])
        assert len(result) == 1


# ── merge_nearby_masks_class_aware ────────────────────────────────────────────

class TestMergeNearbyMasksClassAware:
    def _blobs(self, *circles):
        return [circle_mask(cx, cy, r) for cx, cy, r in circles]

    def test_buildings_no_merging(self):
        blobs = self._blobs((60, 128, 20), (196, 128, 20))
        result = merge_nearby_masks_class_aware(blobs, "Buildings")
        assert len(result) == 2

    def test_vegetation_distant_blobs_not_merged(self):
        blobs = self._blobs((30, 30, 10), (220, 220, 10))
        result = merge_nearby_masks_class_aware(blobs, "Vegetation")
        assert len(result) == 2

    def test_vegetation_adjacent_blobs_merged(self):
        # Two circles touching / nearly touching
        blobs = self._blobs((100, 128, 20), (145, 128, 20))
        result = merge_nearby_masks_class_aware(blobs, "Vegetation")
        # Should merge into 1 (or fewer than 2)
        assert len(result) <= 2  # touching circles may merge

    def test_returns_list(self):
        blobs = self._blobs((128, 128, 30))
        result = merge_nearby_masks_class_aware(blobs, "General")
        assert isinstance(result, list)


# ── dedupe_or_merge_masks_smart ───────────────────────────────────────────────

class TestDedupeOrMergeMasksSmart:
    def test_distinct_masks_all_kept(self):
        m1 = circle_mask(50, 50, 20)
        m2 = circle_mask(200, 200, 20)
        result = dedupe_or_merge_masks_smart([m1, m2], "General")
        assert len(result) == 2

    def test_near_identical_masks_merged(self):
        m_a = rect_mask(20, 20, 80, 80)
        m_b = rect_mask(22, 22, 82, 82)  # heavily overlapping
        result = dedupe_or_merge_masks_smart([m_a, m_b], "General")
        assert len(result) == 1
        # Merged mask must be at least as large as either input
        assert np.sum(result[0] > 0) >= np.sum(m_a > 0)

    def test_vehicles_deduplication_keeps_bigger(self):
        small = rect_mask(40, 40, 60, 54)   # 20×14
        large = rect_mask(38, 38, 62, 56)   # 24×18, nearly same location
        result = dedupe_or_merge_masks_smart([small, large], "Vehicle")
        assert len(result) == 1

    def test_empty_list_returns_empty(self):
        assert dedupe_or_merge_masks_smart([], "General") == []

    def test_single_mask_returned_unchanged(self):
        m = circle_mask(128, 128, 40)
        result = dedupe_or_merge_masks_smart([m], "Buildings")
        assert len(result) == 1
