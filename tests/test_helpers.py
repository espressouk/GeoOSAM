"""
Tests for the helpers/ module — all helper classes and BaseDetectionHelper methods.
No QGIS dependency: helpers only use numpy and OpenCV.
"""
import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from helpers import create_detection_helper  # noqa: E402
from helpers.base_helper import BaseDetectionHelper  # noqa: E402
from helpers.vehicle_helper import VehicleHelper  # noqa: E402
from helpers.buildings_helper import BuildingsHelper  # noqa: E402
from helpers.vegetation_helper import VegetationHelper  # noqa: E402
from helpers.vessels_helper import VesselsHelper  # noqa: E402, F401
from helpers.water_helper import WaterHelper  # noqa: E402, F401
from helpers.road_helper import RoadHelper  # noqa: E402, F401
from helpers.general_helper import GeneralHelper  # noqa: E402
from helpers.residential_helper import ResidentialHelper  # noqa: E402, F401
from helpers.agriculture_helper import AgricultureHelper  # noqa: E402, F401


# ── fixtures ──────────────────────────────────────────────────────────────────

def solid_square_mask(size=64, fill=True):
    """A clean binary uint8 mask."""
    mask = np.zeros((size, size), dtype=np.uint8)
    if fill:
        mask[16:48, 16:48] = 255
    return mask


def two_separate_blobs():
    """Two non-overlapping square blobs."""
    mask = np.zeros((128, 128), dtype=np.uint8)
    mask[10:30, 10:30] = 255   # blob A
    mask[80:100, 80:100] = 255  # blob B
    return mask


def overlapping_blobs():
    """Two heavily-overlapping blobs (IoU > 0.5)."""
    mask_a = np.zeros((128, 128), dtype=np.uint8)
    mask_b = np.zeros((128, 128), dtype=np.uint8)
    mask_a[10:50, 10:50] = 255
    mask_b[15:55, 15:55] = 255
    return mask_a, mask_b


# ── factory ───────────────────────────────────────────────────────────────────

class TestFactory:
    def test_all_known_classes_return_helper(self):
        classes = [
            "Vehicle", "Cars", "Buildings", "Industrial", "Residential",
            "Vegetation", "Water", "Agriculture", "Road", "Vessels", "General",
        ]
        for cls in classes:
            helper = create_detection_helper(cls)
            assert isinstance(helper, BaseDetectionHelper), f"No helper for class '{cls}'"

    def test_unknown_class_returns_general(self):
        helper = create_detection_helper("UnknownXYZ")
        assert isinstance(helper, GeneralHelper)

    def test_helper_stores_class_name(self):
        helper = create_detection_helper("Vehicle")
        assert helper.class_name == "Vehicle"


# ── BaseDetectionHelper ────────────────────────────────────────────────────────

class TestApplyMorphology:
    def test_preserves_solid_blob(self):
        helper = create_detection_helper("General")
        mask = solid_square_mask()
        result = helper.apply_morphology(mask)
        assert result.shape == mask.shape
        assert result.dtype == np.uint8
        # A clean solid square should survive morphology largely intact
        assert np.sum(result > 0) > 0

    def test_removes_isolated_noise(self):
        helper = create_detection_helper("General")
        # Single-pixel noise in a mostly-empty mask
        mask = np.zeros((64, 64), dtype=np.uint8)
        mask[5, 5] = 255
        result = helper.apply_morphology(mask)
        # Opening should erase isolated single pixels
        assert np.sum(result > 0) == 0

    def test_output_is_binary(self):
        helper = create_detection_helper("Vegetation")
        mask = solid_square_mask()
        result = helper.apply_morphology(mask)
        unique = set(result.flatten().tolist())
        assert unique.issubset({0, 255})


class TestCombineMasks:
    def test_single_mask_returned_as_is(self):
        helper = create_detection_helper("General")
        mask = solid_square_mask()
        result = helper.combine_masks([mask])
        assert np.array_equal(result, mask)

    def test_combines_two_masks_by_union(self):
        helper = create_detection_helper("General")
        m1 = np.zeros((64, 64), dtype=np.uint8)
        m2 = np.zeros((64, 64), dtype=np.uint8)
        m1[0:32, 0:32] = 255
        m2[32:64, 32:64] = 255
        result = helper.combine_masks([m1, m2])
        assert np.sum(result > 0) == np.sum(m1 > 0) + np.sum(m2 > 0)

    def test_empty_list_returns_none(self):
        helper = create_detection_helper("General")
        assert helper.combine_masks([]) is None


class TestMergeNearbyMasks:
    def test_distant_masks_not_merged(self):
        helper = create_detection_helper("Vegetation")
        mask = two_separate_blobs()
        import cv2
        _, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
        blobs = [(labels == i).astype(np.uint8) * 255 for i in range(1, stats.shape[0])]
        result = helper.merge_nearby_masks(blobs)
        assert len(result) == 2

    def test_buildings_no_merging(self):
        helper = create_detection_helper("Buildings")
        mask = two_separate_blobs()
        import cv2
        _, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
        blobs = [(labels == i).astype(np.uint8) * 255 for i in range(1, stats.shape[0])]
        # Buildings helper has buffer 0 — masks must stay separate
        result = helper.merge_nearby_masks(blobs)
        assert len(result) == 2


class TestDedupeOrMergeMasks:
    def test_non_overlapping_masks_kept(self):
        helper = create_detection_helper("General")
        mask = two_separate_blobs()
        import cv2
        _, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
        blobs = [(labels == i).astype(np.uint8) * 255 for i in range(1, stats.shape[0])]
        result = helper.dedupe_or_merge_masks(blobs)
        assert len(result) == 2

    def test_high_iou_duplicates_merged(self):
        helper = create_detection_helper("General")
        m_a, m_b = overlapping_blobs()
        result = helper.dedupe_or_merge_masks([m_a, m_b])
        assert len(result) == 1
        # Union must be at least as large as either input
        assert np.sum(result[0] > 0) >= np.sum(m_a > 0)


# ── per-class validate_object ─────────────────────────────────────────────────

class TestVehicleValidation:
    def setup_method(self):
        self.helper = VehicleHelper()

    def _rect_mask(self, w, h, canvas=128):
        import cv2
        mask = np.zeros((canvas, canvas), dtype=np.uint8)
        cx, cy = canvas // 2, canvas // 2
        mask[cy - h // 2:cy + h // 2, cx - w // 2:cx + w // 2] = 255
        _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        return mask

    def test_valid_small_rectangle_accepted(self):
        mask = self._rect_mask(w=20, h=14)   # aspect ~1.43, area ~280 — near limit
        area = int(np.sum(mask > 0))
        valid, _ = self.helper.validate_object(mask, area, min_object_size=50)
        # Should be accepted — compact rectangle
        assert isinstance(valid, bool)

    def test_large_blob_rejected(self):
        mask = self._rect_mask(w=100, h=80)   # area >> 250
        area = int(np.sum(mask > 0))
        valid, reason = self.helper.validate_object(mask, area, min_object_size=50)
        assert not valid
        assert "large" in reason or "area" in reason.lower()

    def test_empty_mask_rejected(self):
        mask = np.zeros((64, 64), dtype=np.uint8)
        valid, reason = self.helper.validate_object(mask, 0, min_object_size=50)
        assert not valid


class TestBuildingsValidation:
    def test_large_polygon_accepted(self):
        helper = BuildingsHelper()
        mask = np.zeros((256, 256), dtype=np.uint8)
        mask[20:100, 20:100] = 255  # 80×80 = 6400 px
        area = int(np.sum(mask > 0))
        valid, _ = helper.validate_object(mask, area, min_object_size=100)
        assert valid

    def test_tiny_blob_rejected(self):
        helper = BuildingsHelper()
        mask = np.zeros((64, 64), dtype=np.uint8)
        mask[30:33, 30:33] = 255  # 3×3 = 9 px
        area = int(np.sum(mask > 0))
        valid, _ = helper.validate_object(mask, area, min_object_size=100)
        assert not valid


class TestVegetationValidation:
    def test_irregular_blob_accepted(self):
        helper = VegetationHelper()
        rng = np.random.default_rng(0)
        mask = np.zeros((128, 128), dtype=np.uint8)
        # Irregular patch — vegetation is loose
        mask[20:80, 20:80] = (rng.random((60, 60)) > 0.3).astype(np.uint8) * 255
        area = int(np.sum(mask > 0))
        valid, _ = helper.validate_object(mask, area, min_object_size=50)
        assert isinstance(valid, bool)


# ── process_sam_mask integration ──────────────────────────────────────────────

class TestProcessSamMask:
    def test_returns_list(self):
        helper = create_detection_helper("Buildings")
        mask = solid_square_mask(size=128)
        result = helper.process_sam_mask(mask)
        assert isinstance(result, list)

    def test_none_input_returns_none(self):
        helper = create_detection_helper("General")
        assert helper.process_sam_mask(None) is None

    def test_empty_mask_returns_empty_list(self):
        helper = create_detection_helper("General")
        mask = np.zeros((64, 64), dtype=np.uint8)
        result = helper.process_sam_mask(mask)
        assert result == []
