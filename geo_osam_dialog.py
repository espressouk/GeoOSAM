import sys
import os

# Fix for QGIS on Windows setting sys.stderr/stdout to None during plugin loading,
# which causes numpy import to crash with: AttributeError: 'NoneType' has no attribute 'write'
if sys.stderr is None:
    sys.stderr = open(os.devnull, 'w')
if sys.stdout is None:
    sys.stdout = open(os.devnull, 'w')

from hydra.core.global_hydra import GlobalHydra
from hydra import initialize_config_module, compose
from shapely.geometry import shape
from rasterio.features import shapes
import rasterio
import cv2
import numpy as np
import torch
import datetime
import pathlib
import platform
import subprocess
import urllib.request
import tempfile
import math
from PIL import Image
from qgis.PyQt.QtCore import QVariant, Qt, QThread, pyqtSignal
from qgis.core import (
    QgsProject,
    QgsRasterLayer,
    QgsRectangle,
    QgsWkbTypes,
    QgsPointXY,
    QgsVectorLayer,
    QgsFeature,
    QgsGeometry,
    QgsFillSymbol,
    QgsField,
    QgsVectorFileWriter,
    QgsMapLayerType,
    QgsDataSourceUri,
    QgsNetworkAccessManager,
    QgsRasterFileWriter,
    QgsRasterPipe,
    QgsCoordinateTransform,
    QgsMapRendererParallelJob,
    QgsMapSettings,
    Qgis
)
from qgis.gui import QgsRubberBand, QgsMapTool, QgsVertexMarker
from qgis.PyQt import QtWidgets, QtCore, QtGui

# fmt: off
plugin_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(plugin_dir)
from helpers import create_detection_helper
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

# Ultralytics SAM2.1 setup
SAM21_AVAILABLE = False

# Model size configurations
# SAM2 (Meta) - for GPU systems (September 2024 SAM 2.1 release)
SAM2_MODELS = {
    'tiny': {
        'name': 'SAM2.1 Tiny',
        'checkpoint': 'sam2.1_hiera_tiny.pt',
        'config': 'sam2.1/sam2.1_hiera_t',
        'url': 'https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt',
        'display': 'SAM2.1 Tiny (156MB, Fast)'
    },
    'small': {
        'name': 'SAM2.1 Small',
        'checkpoint': 'sam2.1_hiera_small.pt',
        'config': 'sam2.1/sam2.1_hiera_s',
        'url': 'https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt',
        'display': 'SAM2.1 Small (184MB, Balanced)'
    },
    'base': {
        'name': 'SAM2.1 Base+',
        'checkpoint': 'sam2.1_hiera_base_plus.pt',
        'config': 'sam2.1/sam2.1_hiera_b+',
        'url': 'https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt',
        'display': 'SAM2.1 Base+ (323MB, Accurate)'
    },
    'large': {
        'name': 'SAM2.1 Large',
        'checkpoint': 'sam2.1_hiera_large.pt',
        'config': 'sam2.1/sam2.1_hiera_l',
        'url': 'https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt',
        'display': 'SAM2.1 Large (898MB, Best Quality)'
    }
}

# SAM2.1 (Ultralytics) - for CPU systems
# T=Tiny, B=Base, L=Large
SAM21_MODELS = {
    'tiny': {
        'name': 'SAM2.1_T (Tiny)',
        'weights': 'sam2.1_t.pt',
        'display': 'SAM2.1_T Tiny (CPU Optimized, Fast)'
    },
    'base': {
        'name': 'SAM2.1_B (Base)',
        'weights': 'sam2.1_b.pt',
        'display': 'SAM2.1_B Base (CPU Optimized, Balanced)'
    },
    'large': {
        'name': 'SAM2.1_L (Large)',
        'weights': 'sam2.1_l.pt',
        'display': 'SAM2.1_L Large (CPU Optimized, Best Quality)'
    }
}

SAM3_MODEL = {
    'name': 'SAM3',
    'weights': 'sam3.pt',
    'display': 'SAM3 (Automatic Segmentation)'
}

SAM3_WEIGHTS_URL = "https://huggingface.co/facebook/sam3/resolve/main/sam3.pt"


# SAM3 Status (Tested 2025-12-26):
# ✅ WORKING: Auto-segmentation (instance segmentation) - Production ready!
# ✅ FIXED: Text prompts - Tokenizer bug fixed with monkey-patch (v1.3.1+)
# ✅ FIXED: Exemplar/similar mode - Fixed with same patch
# Original Issue: TypeError: 'SimpleTokenizer' object is not callable
# Tracking: https://github.com/ultralytics/ultralytics/issues/22647
# Fix: Applied in sam3_clip_fix.py (monkey-patch at startup)

# Apply SAM3 CLIP tokenizer fix
try:
    from sam3_clip_fix import apply_sam3_clip_fix, check_sam3_text_available
    if check_sam3_text_available():
        apply_sam3_clip_fix()
except ImportError:
    print("⚠️  SAM3 CLIP fix module not found (sam3_clip_fix.py)")
except Exception as e:
    print(f"⚠️  SAM3 CLIP fix failed: {e}")

try:
    from ultralytics import SAM
    test_model = SAM('sam2.1_b.pt') # Test with base model
    SAM21_AVAILABLE = True
    print("✅ Ultralytics SAM2.1 available")

    class UltralyticsPredictor:
        def __init__(self, model):
            self.model = model
            self.features = None

        def set_image(self, image):
            self.image = image

        def predict(self, point_coords=None, point_labels=None, box=None, multimask_output=False):
            try:
                if point_coords is not None:
                    if len(point_coords) > 0:
                        points = [[int(p[0]), int(p[1])] for p in point_coords]
                        labels = [int(l) for l in point_labels] if point_labels is not None else [1] * len(points)
                        # Multi-point: wrap in extra list so Ultralytics treats all points as one object
                        # Single-point: keep flat format for compatibility with all modes
                        if len(points) > 1:
                            pts_arg, lbl_arg = [points], [labels]
                        else:
                            pts_arg, lbl_arg = points, labels
                        results = self.model.predict(
                            source=self.image,
                            points=pts_arg,
                            labels=lbl_arg,
                            verbose=False
                        )
                    else:
                        return self._empty_result()

                elif box is not None:
                    if len(box) > 0:
                        bbox = box[0]
                        x1, y1, x2, y2 = int(bbox[0]), int(
                            bbox[1]), int(bbox[2]), int(bbox[3])
                        results = self.model.predict(
                            source=self.image,
                            bboxes=[[x1, y1, x2, y2]],
                            verbose=False
                        )
                    else:
                        return self._empty_result()
                else:
                    results = self.model.predict(
                        source=self.image, verbose=False)

                if results and len(results) > 0:
                    result = results[0]
                    if hasattr(result, 'masks') and result.masks is not None:
                        masks_tensor = result.masks.data
                        if len(masks_tensor) > 0:
                            mask_tensor = masks_tensor[0]
                            if hasattr(mask_tensor, 'cpu'):
                                mask = mask_tensor.cpu().numpy()
                            else:
                                mask = mask_tensor.numpy()

                            if mask.max() <= 1.0:
                                mask = (mask * 255).astype(np.uint8)
                            else:
                                mask = mask.astype(np.uint8)

                            num_pixels = np.sum(mask > 0)
                            if num_pixels > 0:
                                return [mask], [1.0], None
                            else:
                                return self._empty_result()

                return self._empty_result()

            except Exception as e:
                print(f"SAM2.1_B prediction error: {e}")
                return self._empty_result()

        def _empty_result(self):
            empty_mask = np.zeros(
                (self.image.shape[0], self.image.shape[1]), dtype=np.uint8)
            return [empty_mask], [0.0], None

except ImportError:
    print("⚠️ Ultralytics not available - install with: /usr/bin/python3 -m pip install --user ultralytics")
    SAM21_AVAILABLE = False
except Exception as e:
    print(f"⚠️ Ultralytics SAM2.1 failed: {e}")
    SAM21_AVAILABLE = False

if SAM21_AVAILABLE:
    print("   Using fast Ultralytics SAM2.1")
else:
    print("   Falling back to SAM 2 (Meta)")


# SAM3 Predictor Wrapper
class SAM3PredictorWrapper:
    """Wrapper to adapt Ultralytics SAM3 to GeoOSAM interface"""

    def __init__(self, weights_path='sam3.pt'):
        """Initialize SAM3 predictor with weights path"""
        from ultralytics import SAM

        self.weights_path = weights_path
        self.model = SAM(weights_path)
        # Ensure we don't inherit a stale predictor across calls.
        if hasattr(self.model, "predictor"):
            self.model.predictor = None
        self.current_image = None
        self.image_hash = None
        self.cached_image_path = None
        self.semantic_predictor = None
        self.semantic_error = None

    def set_image(self, image):
        """
        Set image for SAM3 prediction
        SAM3 works with file paths or arrays
        """
        import hashlib
        import tempfile
        from PIL import Image

        self.current_image = image

        # Hash image to avoid re-encoding same image
        img_hash = hashlib.md5(image.tobytes(), usedforsecurity=False).hexdigest()

        if img_hash == self.image_hash and self.cached_image_path and os.path.exists(self.cached_image_path):
            # Reuse cached image
            return

        # Cache new image to temp file (SAM3 may need file path)
        if self.cached_image_path is None or not os.path.exists(self.cached_image_path):
            fd, self.cached_image_path = tempfile.mkstemp(suffix='.jpg', prefix='geoosam_sam3_')
            os.close(fd)

        # Save RGB image
        if len(image.shape) == 2:
            # Grayscale
            img_pil = Image.fromarray(image, mode='L')
        elif image.shape[2] == 3:
            # RGB
            img_pil = Image.fromarray(image, mode='RGB')
        else:
            # Multi-channel, take first 3
            img_pil = Image.fromarray(image[:, :, :3], mode='RGB')

        img_pil.save(self.cached_image_path, quality=95)
        self.image_hash = img_hash

    def predict(self, point_coords=None, point_labels=None, box=None,
                text=None, exemplar_mode=False, multimask_output=False):
        """
        Unified predict interface supporting:
        - Point/bbox prompts (classic SAM)
        - Text prompts (SAM3 semantic)
        - Exemplar mode (SAM3 find-similar)
        """

        # Text prompt mode - use automatic instance segmentation
        # SAM3 will segment ALL objects, user filters by selected class
        if text is not None:
            try:
                print(f"🤖 SAM3 automatic instance segmentation (text prompt: '{text}' is used as filter hint)")
                print(f"📸 Image source: {self.cached_image_path}")
                print(f"🖼️ Image shape: {self.current_image.shape if self.current_image is not None else 'None'}")

                # Use automatic instance segmentation - finds ALL objects
                results = self.model.predict(
                    source=self.cached_image_path,
                    verbose=False,
                    conf=0.25,  # Confidence threshold
                    iou=0.7,  # IOU threshold for NMS
                    imgsz=1024,  # Image size
                    save=False,
                    retina_masks=True  # High quality masks
                )

                print(f"📊 Results: {len(results) if results else 0} result sets")

                extracted = self._extract_masks(results)
                print(f"✅ Extracted masks: {len(extracted[0]) if extracted and extracted[0] else 0}")
                return extracted
            except Exception as e:
                print(f"❌ SAM3 auto-segmentation error: {e}")
                import traceback
                traceback.print_exc()
                return self._empty_result()

        # Exemplar/Similar mode - use automatic instance segmentation
        # SAM3 will segment ALL objects, user can filter by selected example
        if exemplar_mode and box is not None:
            try:
                print(f"🤖 SAM3 automatic instance segmentation (similar objects mode)")
                print(f"📦 Box hint: {box}")
                print(f"📸 Image source: {self.cached_image_path}")

                if len(box) > 0:
                    bbox = box[0]
                else:
                    bbox = None

                # Try SAM3 semantic exemplar prompting first
                if bbox is not None:
                    exemplar_result = self._predict_exemplar_semantic(bbox)
                    if exemplar_result and exemplar_result[0]:
                        print(f"✅ Exemplar results: {len(exemplar_result[0])}")
                        return exemplar_result

                # Fallback to automatic instance segmentation - finds ALL objects
                self._reset_predictor_if_semantic()
                results = self.model.predict(
                    source=self.cached_image_path,
                    verbose=False,
                    conf=0.25,
                    iou=0.7,
                    imgsz=1024,
                    save=False,
                    retina_masks=True
                )

                print(f"📊 Auto-segmentation results: {len(results) if results else 0}")

                extracted = self._extract_masks(results)
                print(f"✅ Extracted masks: {len(extracted[0]) if extracted and extracted[0] else 0}")
                return extracted
            except Exception as e:
                print(f"❌ SAM3 auto-segmentation error: {e}")
                import traceback
                traceback.print_exc()
                return self._empty_result()

        # Classic point/bbox mode - SAM3 supports these too
        if point_coords is not None:
            try:
                points = [[int(p[0]), int(p[1])] for p in point_coords]
                labels = [int(l) for l in point_labels] if point_labels is not None else [1] * len(points)
                self._reset_predictor_if_semantic()
                # Multi-point: wrap in extra list so Ultralytics treats all points as one object
                # Single-point: keep flat format for compatibility with all modes
                if len(points) > 1:
                    pts_arg, lbl_arg = [points], [labels]
                else:
                    pts_arg, lbl_arg = points, labels
                results = self.model.predict(
                    source=self.current_image,
                    points=pts_arg,
                    labels=lbl_arg,
                    verbose=False,
                    save=False
                )
                return self._extract_masks(results)
            except Exception as e:
                print(f"SAM3 point prediction error: {e}")
                return self._empty_result()

        if box is not None:
            try:
                bbox = box[0]
                x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
                self._reset_predictor_if_semantic()
                results = self.model.predict(
                    source=self.current_image,
                    bboxes=[[x1, y1, x2, y2]],
                    verbose=False,
                    save=False
                )
                return self._extract_masks(results)
            except Exception as e:
                print(f"SAM3 bbox prediction error: {e}")
                return self._empty_result()

        return self._empty_result()

    def _extract_masks(self, results):
        """Extract masks from SAM3 results"""
        try:
            print(f"🔬 Extracting masks from results...")
            print(f"   Results type: {type(results)}")
            print(f"   Results length: {len(results) if results else 0}")

            if results and len(results) > 0:
                result = results[0]
                print(f"   Result[0] type: {type(result)}")
                print(f"   Has masks: {hasattr(result, 'masks')}")

                if hasattr(result, 'masks') and result.masks is not None:
                    masks_tensor = result.masks.data
                    print(f"   Masks tensor shape: {masks_tensor.shape if hasattr(masks_tensor, 'shape') else 'N/A'}")

                    # SAM3 may return multiple masks
                    masks_list = []
                    scores_list = []

                    for i, mask_tensor in enumerate(masks_tensor):
                        if hasattr(mask_tensor, 'cpu'):
                            mask = mask_tensor.cpu().numpy()
                        else:
                            mask = np.array(mask_tensor)

                        # Ensure uint8 format
                        if mask.max() <= 1.0:
                            mask = (mask * 255).astype(np.uint8)
                        else:
                            mask = mask.astype(np.uint8)

                        # Only add non-empty masks
                        num_pixels = np.sum(mask > 0)

                        if num_pixels > 0:
                            masks_list.append(mask)
                            # Get confidence if available
                            if hasattr(result, 'boxes') and hasattr(result.boxes, 'conf'):
                                try:
                                    score = float(result.boxes.conf[i]) if i < len(result.boxes.conf) else 1.0
                                except:
                                    score = 1.0
                            else:
                                score = 1.0
                            scores_list.append(score)

                    if len(masks_list) > 0:
                        print(f"✅ Extracted {len(masks_list)} masks")

                    if len(masks_list) > 0:
                        return masks_list, scores_list, None
                else:
                    print("⚠️  No masks attribute or masks is None")

            print("⚠️  Returning empty result")
            return self._empty_result()
        except Exception as e:
            print(f"❌ SAM3 mask extraction error: {e}")
            import traceback
            traceback.print_exc()
            return self._empty_result()

    def _empty_result(self):
        """Return empty result"""
        h, w = self.current_image.shape[:2] if self.current_image is not None else (512, 512)
        empty_mask = np.zeros((h, w), dtype=np.uint8)
        return [empty_mask], [0.0], None

    def _reset_predictor_if_semantic(self):
        """Avoid using semantic predictor with an interactive SAM3 model."""
        try:
            predictor = getattr(self.model, "predictor", None)
            if predictor and predictor.__class__.__name__ == "SAM3SemanticPredictor":
                self.model.predictor = None
        except Exception:
            pass

    def _predict_exemplar_semantic(self, bbox):
        """Run SAM3 semantic predictor with exemplar bbox prompt."""
        try:
            if not self._ensure_semantic_predictor():
                return None

            # Convert bbox to numpy array first to avoid slow tensor conversion warning
            import numpy as np
            bbox_array = np.array([bbox], dtype=np.float32)

            prompts = {"bboxes": bbox_array, "labels": [1]}
            self.semantic_predictor.set_prompts(prompts)
            results = self.semantic_predictor(source=self.cached_image_path, stream=False)
            return self._extract_masks(results)
        except Exception as e:
            print(f"❌ SAM3 exemplar semantic error: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _ensure_semantic_predictor(self):
        """Lazy-init SAM3 semantic predictor for exemplar prompts."""
        if self.semantic_predictor is not None:
            return True
        if self.semantic_error:
            return False
        try:
            import clip  # noqa: F401
        except Exception as e:
            self.semantic_error = f"CLIP not available: {e}"
            print(f"⚠️ SAM3 exemplar requires CLIP. {self.semantic_error}")
            return False
        try:
            from ultralytics.models.sam.build_sam3 import build_sam3_image_model
            from ultralytics.models.sam.predict import SAM3SemanticPredictor

            semantic_model = build_sam3_image_model(self.weights_path)
            predictor = SAM3SemanticPredictor(overrides={"conf": 0.25, "imgsz": 1024, "iou": 0.7})
            predictor.setup_model(model=semantic_model, verbose=False)
            self.semantic_predictor = predictor
            return True
        except Exception as e:
            self.semantic_error = str(e)
            print(f"⚠️ SAM3 semantic predictor init failed: {e}")
            return False

"""
GeoOSAM Control Panel - Enhanced SAM segmentation for QGIS
Copyright (C) 2025 by Ofer Butbega
"""

# Global threading configuration
_THREADS_CONFIGURED = False

def merge_nearby_masks_class_aware(masks, class_name, buffer_px=3):
    """Class-aware merging with different strategies per class"""

    if class_name in ['Buildings', 'Residential']:
        # For buildings: NO merging - each detection should stay separate
        return masks

    elif class_name in ['Vessels', 'Vehicle']:
        # For vehicles: minimal merging (1-2px buffer)
        buffer_px = 1

    elif class_name in ['Water', 'Agriculture', 'Vegetation']:
        # For large areas: allow more aggressive merging
        buffer_px = 5

    # Original merging logic with class-aware buffer
    kernel = np.ones((buffer_px*2+1, buffer_px*2+1), np.uint8)
    bins      = [cv2.threshold(m,127,255,cv2.THRESH_BINARY)[1] for m in masks]
    dilated   = [cv2.dilate(b, kernel, iterations=1) for b in bins]
    used      = [False]*len(bins)
    merged    = []

    for i in range(len(bins)):
        if used[i]: 
            continue
        group_mask = bins[i].copy()
        # merge in any dilated-overlap neighbors
        for j in range(i+1, len(bins)):
            if used[j]:
                continue
            # if dilated masks touch at all…
            if np.any(cv2.bitwise_and(dilated[i], dilated[j]) == 255):
                used[j] = True
                # union the original shapes
                group_mask = cv2.bitwise_or(group_mask, bins[j])
        merged.append(group_mask)
    return merged

def dedupe_or_merge_masks_smart(masks, class_name, iou_thresh=0.3, merge=True):
    """Smart deduplication based on class type"""

    if class_name in ['Buildings', 'Residential']:
        # For buildings: Only merge if VERY high overlap (likely same building)
        iou_thresh = 0.7  # Much higher threshold
        merge = False     # Don't merge, just remove duplicates

    elif class_name in ['Vehicle', 'Vessels']:
        # For vehicles: Moderate overlap allowed
        iou_thresh = 0.4
        merge = True

    elif class_name in ['Water', 'Agriculture', 'Vegetation']:
        # For large areas: Allow merging of adjacent areas
        iou_thresh = 0.1
        merge = True

    # Original logic with class-aware parameters
    bins   = [cv2.threshold(m,127,255,cv2.THRESH_BINARY)[1] for m in masks]
    used   = [False]*len(masks)
    result = []

    for i in range(len(bins)):
        if used[i]: continue
        mi = bins[i]
        union_mask = mi.copy()

        for j in range(i+1, len(bins)):
            if used[j]: continue
            mj = bins[j]
            inter = cv2.bitwise_and(mi, mj)
            uni   = cv2.bitwise_or(mi, mj)
            # IoU = area(inter) / area(union)
            if np.sum(uni==255) > 0:
                iou = np.sum(inter==255)/np.sum(uni==255)
                if iou >= iou_thresh:
                    used[j] = True
                    if merge:
                        union_mask = cv2.bitwise_or(union_mask, mj)
                    else:
                        # keep only the bigger mask by area
                        if np.sum(mj==255) > np.sum(mi==255):
                            union_mask = mj.copy()

        result.append(union_mask)
    return result

def filter_contained_masks(masks):
    keep = []
    masks_bin = [cv2.threshold(m, 127, 255, cv2.THRESH_BINARY)[1] for m in masks]
    used = [False] * len(masks)

    for i in range(len(masks)):
        if used[i]:
            continue
        mi = masks_bin[i]
        contained = False
        for j in range(len(masks)):
            if i == j or used[j]:
                continue
            mj = masks_bin[j]
            intersection = cv2.bitwise_and(mi, mj)
            # If all of mi's mask is inside mj, it's contained
            if np.sum(intersection == 255) == np.sum(mi == 255):
                contained = True
                break
        if not contained:
            keep.append(masks[i])
        else:
            used[i] = True
    return keep

def setup_pytorch_performance():
    global _THREADS_CONFIGURED

    import multiprocessing
    num_cores = multiprocessing.cpu_count()
    optimal_threads = max(4, int(num_cores * 0.75)) if num_cores >= 16 else \
        max(4, num_cores - 2) if num_cores >= 8 else \
        max(1, num_cores - 1)

    if _THREADS_CONFIGURED:
        try:
            return torch.get_num_threads()
        except:
            return optimal_threads

    # Try to configure threads, but don't fail if already initialized
    try:
        torch.set_num_interop_threads(min(4, optimal_threads // 2))
        torch.set_num_threads(optimal_threads)
        actual_threads = torch.get_num_threads()
    except RuntimeError as e:
        # Threading already configured by another plugin/process
        print(f"Note: PyTorch threading pre-configured, using existing settings")
        try:
            actual_threads = torch.get_num_threads()
        except:
            actual_threads = optimal_threads

    # Always set environment variables as backup
    os.environ["OMP_NUM_THREADS"] = str(actual_threads)
    os.environ["MKL_NUM_THREADS"] = str(actual_threads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(actual_threads)

    _THREADS_CONFIGURED = True
    return actual_threads

def auto_download_checkpoint():
    """Download SAM2 checkpoint if missing"""
    plugin_dir = os.path.dirname(os.path.abspath(__file__))
    checkpoint_dir = os.path.join(plugin_dir, "sam2", "checkpoints")
    checkpoint_path = os.path.join(checkpoint_dir, "sam2.1_hiera_tiny.pt")
    download_script = os.path.join(
        checkpoint_dir, "download_sam2_checkpoints.sh")

    if os.path.exists(checkpoint_path):
        print(f"✅ SAM2 checkpoint found")
        return True

    print(f"🔍 SAM2 checkpoint not found, downloading...")
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Try bash script first (Linux/Mac)
    if platform.system() in ['Linux', 'Darwin'] and os.path.exists(download_script):
        try:
            result = subprocess.run(['bash', download_script], cwd=checkpoint_dir,
                                    capture_output=True, text=True, timeout=300)
            if result.returncode == 0 and os.path.exists(checkpoint_path):
                print("✅ Checkpoint downloaded via script")
                return True
        except Exception as e:
            print(f"⚠️ Script failed: {e}")

    # Python fallback
    try:
        url = "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt"
        urllib.request.urlretrieve(url, checkpoint_path)
        if os.path.exists(checkpoint_path) and os.path.getsize(checkpoint_path) > 1000000:
            print("✅ Checkpoint downloaded via Python")
            return True
        else:
            print("❌ Download verification failed")
            return False
    except Exception as e:
        print(f"❌ Download failed: {e}")
        return False

def show_checkpoint_dialog(parent=None):
    """Show download dialog for SAM2 checkpoint"""
    from qgis.PyQt.QtWidgets import QMessageBox, QProgressDialog
    from qgis.PyQt.QtCore import Qt

    msg = QMessageBox(parent)
    msg.setIcon(QMessageBox.Question)
    msg.setWindowTitle("SAM2 Model Download")
    msg.setText("GeoOSAM requires the SAM2 model checkpoint (~160MB).")
    msg.setInformativeText("Would you like to download it now?")
    msg.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
    msg.setDefaultButton(QMessageBox.Yes)

    if msg.exec_() == QMessageBox.Yes:
        progress = QProgressDialog(
            "Downloading SAM2 model...", "Cancel", 0, 0, parent)
        progress.setWindowModality(Qt.WindowModal)
        progress.show()

        try:
            success = auto_download_checkpoint()
            progress.close()
            if success:
                QMessageBox.information(
                    parent, "Success", "✅ SAM2 model downloaded successfully!")
                return True
            else:
                QMessageBox.critical(
                    parent, "Download Failed", "❌ Failed to download SAM2 model.")
                return False
        except Exception as e:
            progress.close()
            QMessageBox.critical(parent, "Error", f"Download error: {e}")
            return False
    return False

def detect_best_device():
    """Detect best available device and return available model options"""
    cores = None

    # Check what base models are available
    available_models = {
        'SAM2': True,  # Always available (fallback)
        'SAM2.1': SAM21_AVAILABLE,
        'SAM3': check_sam3_available()
    }

    try:
        if torch.cuda.is_available() and not os.getenv("GEOOSAM_FORCE_CPU"):
            gpu_props = torch.cuda.get_device_properties(0)
            if gpu_props.total_memory / 1024**3 >= 3:  # 3GB minimum
                device = "cuda"

                # GPU (>3GB): Offer all SAM2 sizes + SAM3
                model_options = []

                # Add SAM2 model sizes
                for size in ['tiny', 'small', 'base', 'large']:
                    model_options.append({
                        'type': 'SAM2',
                        'size': size,
                        'id': f'SAM2_{size}',
                        'display': SAM2_MODELS[size]['display']
                    })

                # Add SAM3 if available
                if available_models['SAM3']:
                    model_options.append({
                        'type': 'SAM3',
                        'size': None,
                        'id': 'SAM3',
                        'display': SAM3_MODEL['display']
                    })

                # Default to SAM3 if available, otherwise SAM2 tiny
                default_model = 'SAM3' if available_models['SAM3'] else 'SAM2_tiny'

                print(f"🎮 GPU detected: {torch.cuda.get_device_name(0)} - {len(model_options)} models available")
                return device, default_model, cores, available_models, model_options

        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = "mps"

            # Apple Silicon: Offer all SAM2 sizes + SAM3
            model_options = []

            # Add SAM2 model sizes
            for size in ['tiny', 'small', 'base', 'large']:
                model_options.append({
                    'type': 'SAM2',
                    'size': size,
                    'id': f'SAM2_{size}',
                    'display': SAM2_MODELS[size]['display']
                })

            # Add SAM3 if available
            if available_models['SAM3']:
                model_options.append({
                    'type': 'SAM3',
                    'size': None,
                    'id': 'SAM3',
                    'display': SAM3_MODEL['display']
                })

            # Default to SAM3 if available, otherwise SAM2 tiny
            default_model = 'SAM3' if available_models['SAM3'] else 'SAM2_tiny'

            print(f"🍎 Apple Silicon GPU detected - {len(model_options)} models available")
            return device, default_model, cores, available_models, model_options

        else:
            device = "cpu"
            cores = setup_pytorch_performance()

            # CPU (or GPU <3GB): Offer SAM2.1 sizes if available, otherwise SAM2 sizes
            model_options = []

            if SAM21_AVAILABLE:
                # Add SAM2.1 model sizes (CPU optimized) - Tiny (_T), Base (_B), Large (_L)
                for size in ['tiny', 'base', 'large']:
                    model_options.append({
                        'type': 'SAM2.1',
                        'size': size,
                        'id': f'SAM2.1_{size}',
                        'display': SAM21_MODELS[size]['display']
                    })
                default_model = 'SAM2.1_base'
            else:
                # Fallback to SAM2 sizes
                for size in ['tiny', 'small', 'base']:
                    model_options.append({
                        'type': 'SAM2',
                        'size': size,
                        'id': f'SAM2_{size}',
                        'display': SAM2_MODELS[size]['display']
                    })
                default_model = 'SAM2_tiny'

            print(f"💻 CPU detected - {len(model_options)} models available ({cores} cores)")
            return device, default_model, cores, available_models, model_options

    except Exception as e:
        print(f"⚠️ Device detection failed: {e}, falling back to CPU")
        device = "cpu"
        cores = setup_pytorch_performance()

        # Fallback model options
        model_options = []
        if SAM21_AVAILABLE:
            for size in ['tiny', 'base', 'large']:
                model_options.append({
                    'type': 'SAM2.1',
                    'size': size,
                    'id': f'SAM2.1_{size}',
                    'display': SAM21_MODELS[size]['display']
                })
            default_model = 'SAM2.1_base'
        else:
            for size in ['tiny', 'small', 'base']:
                model_options.append({
                    'type': 'SAM2',
                    'size': size,
                    'id': f'SAM2_{size}',
                    'display': SAM2_MODELS[size]['display']
                })
            default_model = 'SAM2_tiny'

        return device, default_model, cores, available_models, model_options


def check_sam3_available():
    """Check if SAM3 weights and Ultralytics version are available"""
    try:
        # Check Ultralytics version supports SAM3 (>=8.3.237)
        import ultralytics
        version_str = ultralytics.__version__
        version_parts = version_str.split('.')

        # Parse version (e.g., "8.3.237" → [8, 3, 237])
        major = int(version_parts[0])
        minor = int(version_parts[1]) if len(version_parts) > 1 else 0
        patch = int(version_parts[2]) if len(version_parts) > 2 else 0

        # Check if version >= 8.3.237
        if major < 8 or (major == 8 and minor < 3) or (major == 8 and minor == 3 and patch < 237):
            print(f"⚠️ SAM3 requires Ultralytics >=8.3.237, found {version_str}")
            return False

        # Check if SAM3 weights exist in common locations
        import pathlib
        search_paths = [
            os.path.join(os.path.dirname(__file__), 'sam3.pt'),
            os.path.join(pathlib.Path.home(), '.ultralytics', 'weights', 'sam3.pt'),
            'sam3.pt'  # Current directory
        ]

        for path in search_paths:
            if os.path.exists(path):
                print(f"✅ SAM3 weights found: {path}")
                return True

        print("⚠️ SAM3 not available - weights not found (see Help for download instructions)")
        return False

    except ImportError:
        print("⚠️ SAM3 not available - Ultralytics not installed")
        return False
    except Exception as e:
        print(f"⚠️ SAM3 availability check failed: {e}")
        return False


class TiledSegmentationWorker(QThread):
    """Worker thread for processing large rasters in tiles"""
    finished = pyqtSignal(int)  # Total objects found
    error = pyqtSignal(str)
    progress = pyqtSignal(str, int, int)  # message, current_tile, total_tiles
    tile_completed = pyqtSignal(int, int)  # objects_found, tile_index
    cancelled = pyqtSignal()  # Emitted when processing is cancelled

    def __init__(self, predictor, raster_path, request_type, text_prompt=None, bbox=None,
                 current_class=None, class_color=None, tile_size=1024, overlap=128,
                 min_object_size=20, max_object_size_px=None, shape_filters=None):
        print("🔧 TiledSegmentationWorker.__init__() START")
        import sys
        sys.stdout.flush()

        super().__init__()
        print("🔧 QThread super().__init__() completed")
        sys.stdout.flush()

        self.predictor = predictor
        print(f"🔧 Predictor assigned: {type(predictor)}")
        sys.stdout.flush()

        self.raster_path = raster_path
        self.request_type = request_type
        self.text_prompt = text_prompt
        self.bbox = bbox
        self.current_class = current_class
        self.class_color = class_color
        self.tile_size = tile_size
        self.overlap = overlap
        self.min_object_size = min_object_size
        self.max_object_size_px = max_object_size_px
        self.shape_filters = shape_filters
        self.total_objects_found = 0
        self.tile_results = []  # List of (features, debug_info, transform) tuples
        self.cancel_requested = False  # Flag for cancellation

        print("🔧 TiledSegmentationWorker.__init__() COMPLETE")
        sys.stdout.flush()

    def cancel(self):
        """Request cancellation of processing"""
        print("🛑 Cancellation requested")
        self.cancel_requested = True

    def run(self):
        """Process raster in tiles"""
        import sys
        print("🔧 TiledSegmentationWorker.run() CALLED")
        sys.stdout.flush()
        print(f"🔧 Thread ID: {int(self.currentThreadId())}")
        sys.stdout.flush()

        try:
            import rasterio
            from rasterio.windows import Window
            import time
            import cv2
            import numpy as np
            print("✅ Imports successful")
        except Exception as e:
            print(f"❌ Import error in worker thread: {e}")
            import traceback
            traceback.print_exc()
            self.error.emit(f"Import error: {e}")
            return

        try:
            print("\n" + "="*80)
            print("🗺️  TILED SEGMENTATION WORKER STARTED")
            print("="*80)
            print(f"Request type: {self.request_type}")
            print(f"Text prompt: {self.text_prompt if self.text_prompt else 'N/A'}")
            print(f"Predictor type: {type(self.predictor)}")
            print(f"Raster path: {self.raster_path}")
            print("="*80 + "\n")

            with rasterio.open(self.raster_path) as src:
                width, height = src.width, src.height
                print(f"📐 Raster dimensions: {width}x{height}")

                # Calculate tile grid
                tiles = []
                for y in range(0, height, self.tile_size - self.overlap):
                    for x in range(0, width, self.tile_size - self.overlap):
                        w = min(self.tile_size, width - x)
                        h = min(self.tile_size, height - y)
                        tiles.append(Window(x, y, w, h))

                total_tiles = len(tiles)
                print(f"🔢 Total tiles to process: {total_tiles}")

                if total_tiles > 100:
                    print(f"⚠️  Large raster will create {total_tiles} tiles - this may take a while")
                    self.progress.emit(
                        f"⚠️  {total_tiles} tiles to process - this may take several minutes",
                        0, total_tiles
                    )
                    time.sleep(1)

                # For similar mode: extract exemplar from its tile first
                exemplar_crop = None
                if self.request_type == 'similar' and self.bbox is not None:
                    print("\n🎯 Extracting exemplar for similar mode...")
                    # Find which tile contains the exemplar
                    for window in tiles:
                        tile_transform = src.window_transform(window)

                        # Check if bbox is in this tile
                        corners = [
                            (self.bbox.xMinimum(), self.bbox.yMinimum()),
                            (self.bbox.xMaximum(), self.bbox.yMinimum()),
                            (self.bbox.xMaximum(), self.bbox.yMaximum()),
                            (self.bbox.xMinimum(), self.bbox.yMaximum())
                        ]
                        pixel_coords = []
                        for x, y in corners:
                            px, py = ~tile_transform * (x, y)
                            pixel_coords.append((px, py))

                        xs, ys = zip(*pixel_coords)
                        x1, x2 = min(xs), max(xs)
                        y1, y2 = min(ys), max(ys)

                        # Check if bbox is in this tile
                        if (x1 >= -10 and x2 <= window.width + 10 and
                            y1 >= -10 and y2 <= window.height + 10 and
                            x2 > x1 and y2 > y1):

                            # Read this tile
                            tile_arr = src.read([1, 2, 3], window=window, out_dtype=np.uint8)
                            tile_arr = np.moveaxis(tile_arr, 0, -1)

                            # Extract exemplar crop
                            x1 = max(0, min(tile_arr.shape[1] - 1, int(x1)))
                            y1 = max(0, min(tile_arr.shape[0] - 1, int(y1)))
                            x2 = max(0, min(tile_arr.shape[1] - 1, int(x2)))
                            y2 = max(0, min(tile_arr.shape[0] - 1, int(y2)))

                            exemplar_crop = tile_arr[y1:y2, x1:x2].copy()
                            print(f"✅ Exemplar extracted: shape={exemplar_crop.shape}")
                            break

                # Process each tile
                for tile_idx, window in enumerate(tiles):
                    # Check for cancellation request
                    if self.cancel_requested:
                        print("🛑 Processing cancelled by user")
                        self.cancelled.emit()
                        return

                    try:
                        print(f"\n--- Tile {tile_idx+1}/{total_tiles} ---")
                        print(f"Window: x={window.col_off}, y={window.row_off}, w={window.width}, h={window.height}")

                        # Read tile data
                        start_read = time.time()
                        tile_arr = src.read([1, 2, 3], window=window, out_dtype=np.uint8)
                        tile_arr = np.moveaxis(tile_arr, 0, -1)
                        print(f"✅ Tile read: {tile_arr.shape} in {time.time()-start_read:.2f}s")

                        # Update predictor with tile image
                        start_set = time.time()
                        self.predictor.set_image(tile_arr)
                        print(f"✅ Image set in predictor: {time.time()-start_set:.2f}s")

                        # Get tile transform (needed for similar mode coordinate conversion)
                        tile_transform = src.window_transform(window)

                        # Run inference based on mode
                        start_infer = time.time()
                        masks = None
                        scores = None

                        if self.request_type == 'text':
                            print(f"🔍 Running text inference: '{self.text_prompt}'")
                            masks, scores, _ = self.predictor.predict(
                                text=self.text_prompt,
                                multimask_output=False
                            )
                        elif self.request_type == 'similar':
                            # For similar mode in tiled processing:
                            # Use the extracted exemplar crop to find similar objects in EVERY tile
                            # This finds similar objects across the ENTIRE raster

                            if exemplar_crop is not None:
                                # Paste exemplar into top-left corner of tile temporarily
                                # Create a composite image with exemplar for SAM3 to use as reference
                                temp_tile = tile_arr.copy()
                                ex_h, ex_w = exemplar_crop.shape[:2]

                                # Place exemplar in top-left (or centered)
                                # Make sure it fits
                                if ex_h < tile_arr.shape[0] and ex_w < tile_arr.shape[1]:
                                    # Place at top-left
                                    temp_tile[0:ex_h, 0:ex_w] = exemplar_crop
                                    exemplar_bbox_in_tile = [0, 0, ex_w, ex_h]

                                    print(f"🎯 Using exemplar to find similar objects")
                                    print(f"   Exemplar bbox in tile: {exemplar_bbox_in_tile}")

                                    # Set the composite image with exemplar
                                    self.predictor.set_image(temp_tile)

                                    # Convert to numpy array
                                    bbox_array = np.array([exemplar_bbox_in_tile], dtype=np.float32)

                                    # Run exemplar mode
                                    masks, scores, _ = self.predictor.predict(
                                        box=bbox_array,
                                        exemplar_mode=True,
                                        multimask_output=False
                                    )

                                    # Filter out the exemplar itself from results
                                    # The exemplar was pasted at top-left, so remove masks overlapping that region
                                    if masks is not None and len(masks) > 0:
                                        filtered_masks = []
                                        filtered_scores = []
                                        exemplar_region = exemplar_bbox_in_tile  # [0, 0, ex_w, ex_h]

                                        for i, mask in enumerate(masks):
                                            # Check if this mask overlaps significantly with exemplar location
                                            # Get mask bounding box
                                            ys, xs = np.where(mask > 127)
                                            if len(xs) == 0 or len(ys) == 0:
                                                continue

                                            mask_x1, mask_x2 = xs.min(), xs.max()
                                            mask_y1, mask_y2 = ys.min(), ys.max()

                                            # Calculate overlap with exemplar region
                                            ex_x1, ex_y1, ex_x2, ex_y2 = exemplar_region
                                            overlap_x1 = max(mask_x1, ex_x1)
                                            overlap_y1 = max(mask_y1, ex_y1)
                                            overlap_x2 = min(mask_x2, ex_x2)
                                            overlap_y2 = min(mask_y2, ex_y2)

                                            if overlap_x1 < overlap_x2 and overlap_y1 < overlap_y2:
                                                # There is overlap - calculate IoU
                                                overlap_area = (overlap_x2 - overlap_x1) * (overlap_y2 - overlap_y1)
                                                mask_area = (mask_x2 - mask_x1) * (mask_y2 - mask_y1)

                                                if mask_area > 0:
                                                    overlap_ratio = overlap_area / mask_area
                                                    # If >80% of mask overlaps with exemplar, skip it (it's the exemplar itself)
                                                    if overlap_ratio > 0.8:
                                                        print(f"  Skipping mask {i+1}: exemplar itself (overlap {overlap_ratio:.1%})")
                                                        continue

                                            # Keep this mask
                                            filtered_masks.append(mask)
                                            if scores is not None and i < len(scores):
                                                filtered_scores.append(scores[i])

                                        masks = filtered_masks if filtered_masks else None
                                        scores = filtered_scores if filtered_scores else None
                                        if masks:
                                            print(f"✅ After filtering exemplar: {len(masks)} similar objects remain")
                                else:
                                    print(f"⚠️  Exemplar too large for tile")
                                    masks = None
                            else:
                                print(f"⚠️  No exemplar crop available for similar mode")
                                masks = None
                        else:
                            print(f"⚠️  Unknown request type: {self.request_type}")
                            continue

                        print(f"⏱️  Inference time: {time.time()-start_infer:.2f}s")
                        print(f"📊 Masks returned: {len(masks) if masks is not None else 0}")

                        # Convert masks to features using tile transform
                        tile_objects = 0
                        if masks is not None and len(masks) > 0:
                            # tile_transform already calculated above (line 1043)
                            print(f"🔄 Processing {len(masks)} masks...")

                            for mask_idx, mask in enumerate(masks):
                                features = self._convert_mask_to_features(mask, tile_transform)
                                if features:
                                    # Store results for main thread to add to layer
                                    debug_info = {'mode': f'TILE_{tile_idx+1}/{total_tiles}'}
                                    self.tile_results.append((features, debug_info, tile_transform))
                                    tile_objects += len(features)
                                    self.total_objects_found += len(features)

                        # Emit progress
                        progress_msg = f"🔄 Tile {tile_idx+1}/{total_tiles} - Found {self.total_objects_found} objects"
                        self.progress.emit(progress_msg, tile_idx + 1, total_tiles)
                        self.tile_completed.emit(tile_objects, tile_idx)

                    except Exception as e:
                        print(f"❌ Error processing tile {tile_idx+1}: {e}")
                        import traceback
                        traceback.print_exc()
                        continue

                # Done
                print(f"✅ Completed {total_tiles} tiles, {self.total_objects_found} objects found")
                self.finished.emit(self.total_objects_found)

        except Exception as e:
            error_msg = f"❌ Tiled segmentation error: {e}"
            print(error_msg)
            import traceback
            traceback.print_exc()
            self.error.emit(error_msg)

    def _convert_mask_to_features(self, mask, mask_transform):
        """Convert a single mask to feature dictionaries (to be converted to QgsFeature in main thread)"""
        import cv2
        try:
            # Ensure mask is uint8
            if mask.dtype != np.uint8:
                mask = (mask * 255).astype(np.uint8)

            # Threshold mask to binary and clean it up
            _, binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

            # Morphological operations to clean up the mask
            open_kernel = np.ones((3, 3), np.uint8)
            binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, open_kernel)

            close_kernel = np.ones((7, 7), np.uint8)
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, close_kernel)

            # Remove objects outside size range
            nlabels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)

            features = []
            for label in range(1, nlabels):  # Skip background (label 0)
                area = stats[label, cv2.CC_STAT_AREA]
                if area < self.min_object_size:
                    continue
                if self.max_object_size_px is not None and area > self.max_object_size_px:
                    continue

                # Extract this component
                component_mask = (labels == label).astype(np.uint8) * 255

                # Find contours
                contours, _ = cv2.findContours(component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                for contour in contours:
                    if len(contour) < 3:
                        continue

                    # Shape filter check (if enabled)
                    if self.shape_filters and self.shape_filters.get('enabled'):
                        contour_area = cv2.contourArea(contour)
                        if contour_area < 1:
                            continue
                        perimeter = cv2.arcLength(contour, True)
                        compactness = 4 * np.pi * contour_area / (perimeter * perimeter) if perimeter > 0 else 0
                        x, y, w, h = cv2.boundingRect(contour)
                        aspect_ratio = max(w, h) / max(min(w, h), 1)

                        min_circ = self.shape_filters.get('min_circularity', 0.0)
                        min_comp = self.shape_filters.get('min_compactness', 0.0)
                        max_ar = self.shape_filters.get('max_aspect_ratio', 20.0)

                        if compactness < min_comp or compactness < min_circ or aspect_ratio > max_ar:
                            continue

                    # Convert pixel coordinates to geo coordinates
                    geo_coords = []
                    for point in contour:
                        px, py = point[0]
                        geo_x, geo_y = mask_transform * (px, py)
                        geo_coords.append((geo_x, geo_y))

                    if len(geo_coords) >= 3:
                        # Return feature data as dict (will be converted to QgsFeature in main thread)
                        features.append({
                            'coords': geo_coords,
                            'area': area
                        })

            return features if features else None

        except Exception as e:
            print(f"Error converting mask to features: {e}")
            import traceback
            traceback.print_exc()
            return None


class OptimizedSAM2Worker(QThread):
    finished = pyqtSignal(object)
    error = pyqtSignal(str)
    progress = pyqtSignal(str)

    def __init__(self, predictor, arr, mode, model_choice="SAM2", point_coords=None,
                 point_labels=None, box=None, mask_transform=None, debug_info=None, device="cpu",
                 min_object_size=50, max_objects=20, arr_multispectral=None, text_prompt=None,
                 max_object_size_px=None, shape_filters=None):
        super().__init__()
        self.predictor = predictor
        self.arr = arr
        self.arr_multispectral = arr_multispectral
        self.mode = mode
        self.text_prompt = text_prompt
        self.model_choice = model_choice
        self.point_coords = point_coords
        self.point_labels = point_labels
        self.box = box
        self.mask_transform = mask_transform
        self.debug_info = debug_info or {}
        self.device = device
        self.min_object_size = min_object_size
        self.max_objects = max_objects
        self.max_object_size_px = max_object_size_px
        self.shape_filters = shape_filters

    def run(self):
        try:
            self.progress.emit(f"🖼️ Setting image for {self.model_choice}...")

            # SAFETY: Check if thread should continue
            if self.isInterruptionRequested():
                return

            self.predictor.set_image(self.arr)

            # Dispatch to appropriate segmentation method based on mode
            if self.mode == "text":
                self._run_text_segmentation()
            elif self.mode == "similar":
                self._run_similar_segmentation()
            elif self.mode == "bbox_batch":
                self._run_batch_segmentation()
            else:
                self._run_single_segmentation()

        except Exception as e:
            import traceback
            error_msg = f"{self.model_choice} inference failed: {str(e)}\n"

            # Add more specific error context
            if "truth value" in str(e).lower():
                error_msg += "\n🔧 Tip: This might be a mask array format issue. Try switching to single bbox mode first."
            elif "cuda" in str(e).lower():
                error_msg += "\n🔧 Tip: Try switching to CPU mode in device settings."

            error_msg += f"\nFull traceback:\n{traceback.format_exc()}"
            self.error.emit(error_msg)

    def _cancel_segmentation_safely(self):
        """Safely cancel running segmentation"""
        if hasattr(self, 'worker') and self.worker and self.worker.isRunning():
            print("🛑 Requesting worker interruption...")
            self.worker.requestInterruption()  # Request graceful stop

            # Give it a moment to stop gracefully
            if not self.worker.wait(2000):  # Wait 2 seconds
                print("⚠️ Worker didn't stop gracefully, terminating...")
                self.worker.terminate()
                self.worker.wait()  # Wait for termination

            self.worker.deleteLater()
            self.worker = None
            self._update_status("Segmentation cancelled", "warning")
            self._set_ui_enabled(True)

    def _run_single_segmentation(self):
        """Original single object segmentation"""
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
                    multimask_output=True
                )

                # Select best mask based on score
                if len(masks) > 1 and len(scores) > 1:
                    best_idx = np.argmax(scores)
                    masks = [masks[best_idx]]
                    scores = [scores[best_idx]]
                    if logits is not None:
                        logits = [logits[best_idx]] if isinstance(logits, list) else logits[best_idx:best_idx+1]
            else:
                raise ValueError(f"Unknown mode: {self.mode}")

        self._process_single_mask(masks[0], scores, logits)

    def _detect_object_candidates(self, image, bbox, class_name, multispectral_image=None):
        """Detect potential object locations within bbox based on class type"""
        x1, y1, x2, y2 = bbox

        # Use helper to determine if multispectral detection is supported
        helper = create_detection_helper(class_name, self.min_object_size, self.max_objects)

        # Use multi-spectral image if available and supported by the helper
        if (multispectral_image is not None and 
            hasattr(helper, 'supports_multispectral') and 
            helper.supports_multispectral()):
            detection_image = multispectral_image
            print(f"🔍 Detecting {class_name} candidates in {detection_image.shape} region (multi-spectral)")
        else:
            detection_image = image
            print(f"🔍 Detecting {class_name} candidates in {detection_image.shape} region")

        # Crop image to bbox region
        bbox_image = detection_image[y1:y2, x1:x2].copy()

        # Use helper for detection
        return helper.detect_candidates(bbox_image, bbox)

    def _run_text_segmentation(self):
        """Run SAM3 text-based segmentation"""
        self.progress.emit(f"🧠 SAM3 text inference: '{self.text_prompt}'...")

        try:
            with torch.no_grad():
                masks, scores, logits = self.predictor.predict(
                    text=self.text_prompt,
                    multimask_output=False
                )

            # Process all returned masks together
            if masks and len(masks) > 0:
                self.progress.emit(f"✅ Found {len(masks)} instances")
                # Emit all masks together as a single result for undo tracking
                self._process_all_masks(masks, scores, logits)
            else:
                self.error.emit("No objects found matching text prompt")

        except Exception as e:
            self.error.emit(f"Text segmentation error: {e}")

    def _run_similar_segmentation(self):
        """Run SAM3 exemplar-based (find similar) segmentation"""
        self.progress.emit("🧠 SAM3 similar mode: segmenting exemplar first...")

        try:
            with torch.no_grad():
                # STEP 1: First segment the object in the bbox to get a clean exemplar
                self.progress.emit("🎯 Step 1/2: Segmenting reference object...")

                exemplar_masks, exemplar_scores, _ = self.predictor.predict(
                    box=self.box,
                    multimask_output=True
                )

                if not exemplar_masks or len(exemplar_masks) == 0:
                    self.error.emit("Could not segment reference object")
                    return

                # Select best exemplar mask based on score
                if len(exemplar_masks) > 1 and exemplar_scores is not None:
                    best_idx = np.argmax(exemplar_scores)
                    exemplar_mask = exemplar_masks[best_idx]
                    print(f"✅ Exemplar segmented (score: {exemplar_scores[best_idx]:.3f})")
                else:
                    exemplar_mask = exemplar_masks[0]

                # STEP 2: Use the segmented mask as exemplar to find similar objects
                self.progress.emit("🔍 Step 2/2: Finding similar objects...")

                # Normalize exemplar_mask to uint8 (0-255) if needed
                if exemplar_mask.max() <= 1.0:
                    exemplar_mask = (exemplar_mask * 255).astype(np.uint8)
                else:
                    exemplar_mask = exemplar_mask.astype(np.uint8)

                # Get tight bbox around the segmented exemplar
                ys, xs = np.where(exemplar_mask > 127)
                if len(xs) == 0 or len(ys) == 0:
                    self.error.emit("Exemplar mask is empty")
                    return

                tight_box = np.array([[xs.min(), ys.min(), xs.max(), ys.max()]], dtype=np.float32)
                tight_width = tight_box[0][2] - tight_box[0][0]
                tight_height = tight_box[0][3] - tight_box[0][1]

                # Warn if exemplar is too small
                if tight_width < 10 or tight_height < 10:
                    print(f"⚠️  WARNING: Exemplar very small ({tight_width:.0f}x{tight_height:.0f}px) - try zooming in more")

                masks, scores, logits = self.predictor.predict(
                    box=tight_box,
                    exemplar_mode=True,
                    multimask_output=False
                )

            # Process all similar objects found together
            if masks and len(masks) >= 1:
                self.progress.emit(f"✅ Found {len(masks)} similar objects")
                # Emit all masks together as a single result for undo tracking
                self._process_all_masks(masks, scores, logits)
            else:
                self.error.emit("No similar objects found")

        except Exception as e:
            self.error.emit(f"Similar objects error: {e}")

    def _process_all_masks(self, masks, scores, logits):
        """Process multiple masks and emit as single result for undo tracking"""
        self.progress.emit("⚡ Processing masks...")

        # Convert all masks to numpy
        processed_masks = []
        for mask in masks:
            if hasattr(mask, 'cpu'):
                mask = mask.cpu().numpy()
            elif torch.is_tensor(mask):
                mask = mask.detach().cpu().numpy()

            if mask.dtype != np.uint8:
                mask = (mask * 255).astype(np.uint8)

            processed_masks.append(mask)

        # Emit all masks together
        result = {
            'masks': processed_masks,  # Multiple masks
            'scores': scores,
            'logits': logits,
            'mask_transform': self.mask_transform,
            'debug_info': {
                **self.debug_info,
                'model': self.model_choice,
                'batch_count': (len(processed_masks), len(processed_masks))
            }
        }

        self.finished.emit(result)

    def _validate_mask_for_class(self, mask, class_name, center_point):
        """Validate segmented mask based on class-specific criteria"""
        try:
            # Use helper for validation
            helper = create_detection_helper(class_name, self.min_object_size, self.max_objects)

            # Debug mask info
            mask_area = np.sum(mask > 0) if hasattr(mask, 'sum') else 0
            print(f"🔍 VALIDATION DEBUG - Class: {class_name}, Mask area: {mask_area} pixels")

            valid_masks = helper.process_sam_mask(mask)
            result = len(valid_masks) > 0
            print(f"🔍 VALIDATION RESULT: {result} (found {len(valid_masks)} valid masks)")

            return result

        except Exception as e:
            print(f"Validation error: {e}")
            return False


    def _validate_object_shape(self, mask, area):
        """Validate if the detected object has a reasonable shape"""
        try:
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                return False

            # Get the largest contour
            main_contour = max(contours, key=cv2.contourArea)

            # Basic shape validation
            x, y, w, h = cv2.boundingRect(main_contour)
            if w == 0 or h == 0:
                return False

            aspect_ratio = max(w, h) / min(w, h)

            # Calculate solidity (area / convex hull area)
            hull = cv2.convexHull(main_contour)
            hull_area = cv2.contourArea(hull)
            solidity = area / hull_area if hull_area > 0 else 0

            # Apply validation criteria
            return (
                aspect_ratio <= 10.0 and  # Not too elongated
                solidity >= 0.15 and     # Not too irregular
                area >= self.min_object_size  # Large enough
            )

        except Exception as e:
            print(f"Shape validation error: {e}")
            return False

    def _run_batch_segmentation(self):
        """Point-guided batch segmentation - detect objects then segment each individually"""
        try:
            self.progress.emit(f"🔄 Running POINT-GUIDED batch {self.model_choice} inference...")

            # Set the image ONCE for the entire process
            self.predictor.set_image(self.arr)

            # Get bbox coordinates from self.box
            bbox = self.box[0] if isinstance(self.box, list) and len(self.box) else self.box
            if bbox is None:
                self.progress.emit("❌ No bbox provided")
                result = {'individual_masks': [], 'mask_transform': self.mask_transform, 'debug_info': self.debug_info}
                self.finished.emit(result)
                return

            bbox = np.array(bbox).flatten().tolist()
            x1, y1, x2, y2 = [int(round(float(x))) for x in bbox]

            # Ensure bbox is within image bounds
            h, w = self.arr.shape[:2]
            x1 = max(0, min(w-1, x1))
            y1 = max(0, min(h-1, y1))
            x2 = max(x1+1, min(w, x2))
            y2 = max(y1+1, min(h, y2))

            print(f"🎯 Point-guided batch processing bbox: ({x1},{y1}) to ({x2},{y2}) in {w}x{h} image")

            # Detect potential object locations within bbox
            current_class = self.debug_info.get('class', 'Other')
            candidate_points = self._detect_object_candidates(self.arr, [x1, y1, x2, y2], current_class, self.arr_multispectral)

            print(f"🔍 Found {len(candidate_points)} candidate objects for class '{current_class}'")

            if not candidate_points:
                self.progress.emit("❌ No object candidates detected")
                result = {'individual_masks': [], 'mask_transform': self.mask_transform, 'debug_info': self.debug_info}
                self.finished.emit(result)
                return

            # Limit to max_objects to prevent too many detections
            candidates_to_process = candidate_points[:self.max_objects]

            # Process candidates (roads return grouped candidates, others return individual points)
            individual_masks = []
            successful_detections = 0

            for i, candidate in enumerate(candidates_to_process):
                # Handle both individual points and grouped points
                if isinstance(candidate, list):
                    # Grouped candidates (from road helper)
                    points_in_group = candidate
                    print(f"🛣️ Processing road group {i+1} with {len(points_in_group)} points")
                else:
                    # Individual point
                    points_in_group = [candidate]

                try:
                    px, py = points_in_group[0] if len(points_in_group) == 1 else (
                        int(np.mean([p[0] for p in points_in_group])),
                        int(np.mean([p[1] for p in points_in_group]))
                    )

                    self.progress.emit(f"🎯 Segmenting object {i+1}/{len(candidates_to_process)}...")
                    print(f"🔍 Processing candidate {i+1}: {len(points_in_group)} point(s)")

                    # Prepare coordinates for SAM2
                    if len(points_in_group) == 1:
                        point_coords = np.array([points_in_group[0]])
                        point_labels = np.array([1])
                    else:
                        point_coords = np.array(points_in_group)
                        point_labels = np.array([1] * len(points_in_group))

                    with torch.no_grad():
                        masks, scores, logits = self.predictor.predict(
                            point_coords=point_coords,
                            point_labels=point_labels,
                            multimask_output=False
                        )

                    if isinstance(masks, np.ndarray):
                        if len(masks.shape) > 2:
                            mask = masks[0]
                        else:
                            mask = masks
                    elif isinstance(masks, (list, tuple)) and len(masks) > 0:
                        mask = masks[0]
                    else:
                        print(f"  ❌ No valid mask returned for point {i+1}")
                        continue

                    # Convert mask to proper format
                    if hasattr(mask, 'cpu'):
                        mask = mask.cpu().numpy()
                    elif hasattr(mask, 'detach'):
                        mask = mask.detach().cpu().numpy()

                    # Ensure 2D and binary
                    if mask.ndim > 2:
                        mask = mask.squeeze()

                    if mask.dtype == bool:
                        mask = mask.astype(np.uint8) * 255
                    elif mask.max() <= 1.0:
                        mask = (mask * 255).astype(np.uint8)
                    else:
                        mask = mask.astype(np.uint8)

                    # Validate mask quality and size
                    pixel_count = np.sum(mask > 0)
                    print(f"  📊 Mask {i+1}: {pixel_count} pixels")

                    # Max size: use real-world filter if set, else 10% of image
                    image_area = self.arr.shape[0] * self.arr.shape[1]
                    if self.max_object_size_px is not None:
                        max_object_size = self.max_object_size_px
                    else:
                        max_object_size = int(image_area * 0.1)

                    if pixel_count >= self.min_object_size:
                        if pixel_count <= max_object_size:
                            print(f"  🎯 Processing class: {current_class}")

                            # Validate the mask for the current class
                            if self._validate_mask_for_class(mask, current_class, [px, py]):
                                individual_masks.append(mask)
                                successful_detections += 1
                                print(f"  ✅ ACCEPTED: {current_class} mask {i+1} ({pixel_count} pixels)")
                            else:
                                print(f"  ❌ REJECTED: {current_class} mask {i+1} failed validation")
                        else:
                            print(f"  ❌ REJECTED: Object {i+1} too large ({pixel_count} > {max_object_size}, {pixel_count/image_area*100:.1f}% of image)")
                    else:
                        print(f"  ❌ REJECTED: Object {i+1} too small ({pixel_count} < {self.min_object_size})")

                except Exception as e:
                    print(f"  ❌ Error processing candidate {i+1}: {e}")
                    continue

            # Remove any masks completely contained inside another mask
            individual_masks = filter_contained_masks(individual_masks)

            # Class-aware processing
            current_class = self.debug_info.get('class', 'Other')

            # Class-aware processing - use helper methods instead of hardcoded logic
            helper = create_detection_helper(current_class, self.min_object_size, self.max_objects)
            individual_masks = helper.merge_nearby_masks(individual_masks)
            individual_masks = helper.dedupe_or_merge_masks(individual_masks)

            print(f"🎯 Point-guided batch complete: {successful_detections}/{len(candidates_to_process)} objects successfully segmented")

            # Return results
            self.progress.emit(f"📦 Found {len(individual_masks)} individual objects (point-guided batch)")

            result = {
                'individual_masks': individual_masks,
                'mask_transform': self.mask_transform,
                'debug_info': {
                    **self.debug_info,
                    'model': self.model_choice,
                    'batch_count': len(individual_masks),
                    'individual_processing': True,
                    'detection_method': 'point_guided',
                    'candidates_found': len(candidate_points),
                    'candidates_processed': len(candidates_to_process),
                    'successful_segmentations': successful_detections,
                    'target_class': current_class,
                    'min_size_used': self.min_object_size,
                    'max_objects_used': self.max_objects
                }
            }

            self.finished.emit(result)

        except Exception as e:
            import traceback
            error_msg = f"Point-guided batch segmentation failed: {str(e)}\n{traceback.format_exc()}"
            print(f"❌ BATCH ERROR: {error_msg}")
            self.error.emit(error_msg)

    def _get_background_threshold(self, bbox_area, class_name):
        """Get class-specific background threshold"""
        if class_name in ['Vessels', 'Vehicle']:
            return bbox_area * 0.4  # Smaller threshold - reject large water areas
        elif class_name in ['Buildings', 'Industrial']:
            return bbox_area * 0.6  # Medium threshold
        elif class_name in ['Water', 'Agriculture']:
            return bbox_area * 0.9  # Large threshold - allow big areas
        else:
            return bbox_area * 0.5  # Default

    def _apply_class_specific_morphology(self, mask, class_name):
        """Apply class-specific morphological operations using helper"""
        helper = create_detection_helper(class_name, self.min_object_size, self.max_objects)
        return helper.apply_morphology(mask)

    def _validate_object_for_class(self, component_mask, component_area, class_name):
        """Class-aware object validation with optional user shape filters"""
        # Basic size filter
        if component_area < self.min_object_size:
            return False

        # Max size filter (from real-world settings or class default)
        if self.max_object_size_px is not None and component_area > self.max_object_size_px:
            return False

        # Get contour properties
        contours, _ = cv2.findContours(component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return False

        main_contour = max(contours, key=cv2.contourArea)
        if len(main_contour) < 4:
            return False

        x, y, w, h = cv2.boundingRect(main_contour)
        contour_area = cv2.contourArea(main_contour)

        if w == 0 or h == 0:
            return False

        aspect_ratio = w / h

        # Shape analysis
        hull = cv2.convexHull(main_contour)
        hull_area = cv2.contourArea(hull)
        solidity = contour_area / hull_area if hull_area > 0 else 0

        perimeter = cv2.arcLength(main_contour, True)
        compactness = 4 * np.pi * contour_area / (perimeter * perimeter) if perimeter > 0 else 0

        # USER SHAPE FILTERS (override class-specific when enabled)
        if self.shape_filters and self.shape_filters.get('enabled'):
            min_circ = self.shape_filters.get('min_circularity', 0.0)
            min_comp = self.shape_filters.get('min_compactness', 0.0)
            max_ar = self.shape_filters.get('max_aspect_ratio', 20.0)

            if compactness < min_comp:
                return False
            if min_circ > 0 and compactness < min_circ:
                # circularity ~ compactness for simple shapes
                return False
            if aspect_ratio > max_ar or (1.0 / aspect_ratio) > max_ar:
                return False

            return contour_area >= self.min_object_size * 0.5

        # CLASS-SPECIFIC VALIDATION (default when no user shape filters)
        if class_name in ['Vessels', 'Vehicle']:
            # Boats/vehicles: Prefer compact, reasonably-sized objects
            return (
                0.2 <= aspect_ratio <= 8.0 and      # Boat/car-like aspect ratio
                solidity >= 0.3 and                 # Reasonably solid
                compactness >= 0.05 and             # Not too elongated
                contour_area < 8000 and             # Not too large (reject water)
                contour_area >= self.min_object_size * 0.6  # Size validation
            )

        elif class_name in ['Buildings', 'Industrial']:
            # Buildings: Allow larger, more rectangular objects
            return (
                0.1 <= aspect_ratio <= 15.0 and     # Building-like ratios
                solidity >= 0.5 and                 # More solid than vehicles
                contour_area >= self.min_object_size * 0.8
            )

        elif class_name in ['Water', 'Agriculture']:
            # Large areas: Allow big, irregular shapes
            return (
                solidity >= 0.2 and                 # Can be irregular
                contour_area >= self.min_object_size
            )

        elif class_name == 'Vegetation':
            # Trees: Can be irregular, various sizes
            return (
                0.1 <= aspect_ratio <= 10.0 and
                solidity >= 0.15 and                # Can be very irregular
                contour_area >= self.min_object_size * 0.5
            )

        else:
            # Default validation
            return (
                0.1 <= aspect_ratio <= 20.0 and
                solidity >= 0.15 and
                compactness >= 0.02 and
                contour_area >= self.min_object_size * 0.6
            )

    def _apply_class_specific_preprocessing(self, mask, class_name):
        """Apply class-specific preprocessing to improve detection"""
        if class_name in ['Vessels', 'Vehicle']:
            # For boats/vehicles: Use opening to separate touching objects
            kernel = np.ones((3, 3), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

        elif class_name in ['Buildings', 'Industrial']:
            # For buildings: Use closing to fill gaps, less aggressive separation
            kernel = np.ones((5, 5), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

        elif class_name in ['Vegetation', 'Agriculture']:
            # For vegetation: Use gradient to find edges, then close
            kernel = np.ones((3, 3), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_GRADIENT, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

        elif class_name == 'Water':
            # For water: Minimal processing to preserve large areas
            kernel = np.ones((7, 7), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

        return mask

    def _extract_individual_objects(self, mask):
        """Class-aware individual object extraction with smart filtering"""
        try:
            # Convert to binary with proper array handling
            if hasattr(mask, 'cpu'):
                binary_mask = mask.cpu().numpy()
            elif torch.is_tensor(mask):
                binary_mask = mask.detach().cpu().numpy()
            else:
                binary_mask = np.array(mask)

            # Handle different data types
            if binary_mask.dtype == bool:
                binary_mask = binary_mask.astype(np.uint8) * 255
            elif binary_mask.dtype != np.uint8:
                if binary_mask.max() <= 1.0:
                    binary_mask = (binary_mask * 255).astype(np.uint8)
                else:
                    binary_mask = binary_mask.astype(np.uint8)

            # Ensure 2D array
            if binary_mask.ndim > 2:
                binary_mask = binary_mask.squeeze()

            if binary_mask.size == 0:
                return []

            print(f"\n🔍 CLASS-AWARE MASK ANALYSIS:")
            print(f"   Mask shape: {binary_mask.shape}")
            print(f"   Non-zero pixels: {np.sum(binary_mask > 0)}")

            # Get current class from debug info
            current_class = self.debug_info.get('class', 'Other')
            print(f"   Target class: {current_class}")

            # GET TARGET BBOX COORDINATES
            if hasattr(self, 'debug_info') and 'target_bbox' in self.debug_info:
                bbox_str = self.debug_info['target_bbox']
                import re
                bbox_match = re.match(r'\((\d+),(\d+)\)-\((\d+),(\d+)\)', bbox_str)
                if bbox_match:
                    bbox_x1, bbox_y1, bbox_x2, bbox_y2 = map(int, bbox_match.groups())
                    print(f"   Target bbox: ({bbox_x1},{bbox_y1}) to ({bbox_x2},{bbox_y2})")
                else:
                    print("   ERROR: Could not parse bbox coordinates")
                    return []
            else:
                print("   ERROR: No bbox coordinates available")
                return []

            # CROP MASK TO BBOX AREA ONLY
            print(f"   🔲 Cropping mask to bbox area only...")
            binary_mask = self._crop_mask_to_bbox(binary_mask, [bbox_x1, bbox_y1, bbox_x2, bbox_y2])
            print(f"   After bbox crop: {np.sum(binary_mask > 0)} non-zero pixels")

            if np.sum(binary_mask > 0) == 0:
                print("   No pixels within bbox area")
                return []

            # CLASS-SPECIFIC PREPROCESSING
            binary_mask = self._apply_class_specific_preprocessing(binary_mask, current_class)

            # Remove large background regions
            bbox_area = (bbox_x2 - bbox_x1) * (bbox_y2 - bbox_y1)
            background_threshold = self._get_background_threshold(bbox_area, current_class)

            num_labels_initial, labels_initial, stats_initial, _ = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
            print(f"   Initial components in bbox: {num_labels_initial-1}")

            # Filter out background regions
            filtered_mask = np.zeros_like(binary_mask)
            background_removed_count = 0

            for label_id in range(1, num_labels_initial):
                component_area = stats_initial[label_id, cv2.CC_STAT_AREA]
                if component_area > background_threshold:
                    print(f"   Removing background component: {component_area}px (> {background_threshold:.0f}px)")
                    background_removed_count += 1
                else:
                    component_mask = (labels_initial == label_id).astype(np.uint8) * 255
                    filtered_mask = cv2.bitwise_or(filtered_mask, component_mask)

            print(f"   Removed {background_removed_count} background regions")

            # CLASS-SPECIFIC MORPHOLOGICAL OPERATIONS
            final_mask = self._apply_class_specific_morphology(filtered_mask, current_class)

            print(f"   After class-specific morphology: {np.sum(final_mask > 0)} non-zero pixels")

            # Find individual objects with class-aware filtering
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(final_mask, connectivity=8)
            print(f"   Final objects in bbox: {num_labels-1}")

            individual_masks = []
            rejected_count = 0

            for label_id in range(1, num_labels):
                try:
                    component_mask = (labels == label_id).astype(np.uint8) * 255
                    component_area = stats[label_id, cv2.CC_STAT_AREA]

                    print(f"   Object {label_id}: {component_area}px", end="")

                    # CLASS-AWARE VALIDATION
                    if self._validate_object_for_class(component_mask, component_area, current_class):
                        individual_masks.append(component_mask)
                        print(" → ACCEPTED ✅")
                    else:
                        print(" → REJECTED")
                        rejected_count += 1

                except Exception as e:
                    print(f" → ERROR: {e}")
                    rejected_count += 1
                    continue

            print(f"   🎯 RESULT: {len(individual_masks)} {current_class} objects, {rejected_count} rejected")
            print(f"   ✅ Class-aware filtering applied\n")

            return individual_masks

        except Exception as e:
            print(f"❌ Error in _extract_individual_objects: {e}")
            return []

    def _crop_mask_to_bbox(self, mask, bbox_coords):
        """Crop mask to only show results within the target bbox"""
        try:
            x1, y1, x2, y2 = bbox_coords

            # Create a bbox mask - only area within selection
            bbox_mask = np.zeros_like(mask)
            bbox_mask[y1:y2+1, x1:x2+1] = 255

            # Keep only the parts of the segmentation that are within bbox
            cropped_mask = cv2.bitwise_and(mask, bbox_mask)

            return cropped_mask

        except Exception as e:
            print(f"Error cropping mask to bbox: {e}")
            return mask

    def _process_single_mask(self, mask, scores, logits, batch_count=None):
        """Process the final mask"""
        self.progress.emit("⚡ Processing mask...")

        if hasattr(mask, 'cpu'):
            mask = mask.cpu().numpy()
        elif torch.is_tensor(mask):
            mask = mask.detach().cpu().numpy()

        if mask.dtype != np.uint8:
            mask = (mask * 255).astype(np.uint8)

        result = {
            'mask': mask,
            'scores': scores,
            'logits': logits,
            'mask_transform': self.mask_transform,
            'debug_info': {
                **self.debug_info, 
                'model': self.model_choice,
                'batch_count': batch_count
            }
        }

        self.finished.emit(result)

class EnhancedPointClickTool(QgsMapTool):
    def __init__(self, canvas, cb):
        super().__init__(canvas)
        self.canvas = canvas
        self.cb = cb
        self.setCursor(QtCore.Qt.CrossCursor)

        self.point_rubber = QgsRubberBand(canvas, QgsWkbTypes.PointGeometry)
        self.point_rubber.setColor(QtCore.Qt.red)
        self.point_rubber.setIcon(QgsRubberBand.ICON_CIRCLE)
        self.point_rubber.setIconSize(12)
        self.point_rubber.setWidth(4)

        # Multi-point accumulation (Shift=positive, Ctrl=negative)
        self.accumulated_points = []  # list of (QgsPointXY, label) where label=1 positive, 0 negative
        self.point_markers = []  # QgsVertexMarker instances for visual feedback

    def canvasReleaseEvent(self, e):
        map_point = self.canvas.getCoordinateTransform().toMapCoordinates(e.pos())

        if e.modifiers() & QtCore.Qt.ShiftModifier:
            # Shift+click: accumulate positive point
            self.accumulated_points.append((map_point, 1))
            self._add_marker(map_point, positive=True)
            return
        elif e.modifiers() & QtCore.Qt.ControlModifier:
            # Ctrl+click: accumulate negative point
            self.accumulated_points.append((map_point, 0))
            self._add_marker(map_point, positive=False)
            return

        # Normal click: show red dot and trigger prediction
        self.point_rubber.reset(QgsWkbTypes.PointGeometry)
        self.point_rubber.addPoint(map_point, True)
        self.canvas.refresh()

        if self.accumulated_points:
            # Multi-point mode: add this click as positive, send all points
            self.accumulated_points.append((map_point, 1))
            points = self.accumulated_points.copy()
            self._clear_markers()
            self.accumulated_points = []
            self.cb(map_point, points)
        else:
            # Single-point mode (unchanged behavior)
            self.cb(map_point, None)

    def _add_marker(self, point, positive=True):
        """Add a +/- marker on the map"""
        marker = QgsVertexMarker(self.canvas)
        if positive:
            marker.setIconType(QgsVertexMarker.ICON_CROSS)
            marker.setColor(QtGui.QColor(0, 200, 0))  # Green for positive
        else:
            marker.setIconType(QgsVertexMarker.ICON_X)
            marker.setColor(QtGui.QColor(255, 0, 0))  # Red for negative
        marker.setIconSize(14)
        marker.setPenWidth(3)
        marker.setCenter(point)
        self.point_markers.append(marker)
        self.canvas.refresh()

    def _clear_markers(self):
        """Remove all +/- markers from the map"""
        for marker in self.point_markers:
            self.canvas.scene().removeItem(marker)
        self.point_markers = []

    def deactivate(self):
        self.point_rubber.reset(QgsWkbTypes.PointGeometry)
        self._clear_markers()
        self.accumulated_points = []
        super().deactivate()

    def clear_feedback(self):
        self.point_rubber.reset(QgsWkbTypes.PointGeometry)
        self._clear_markers()
        self.accumulated_points = []
        self.canvas.refresh()

class EnhancedBBoxClickTool(QgsMapTool):
    def __init__(self, canvas, cb):
        super().__init__(canvas)
        self.canvas = canvas
        self.cb = cb
        self.setCursor(QtCore.Qt.CrossCursor)
        self.start_point = None
        self.is_dragging = False

        self.bbox_rubber = QgsRubberBand(canvas, QgsWkbTypes.PolygonGeometry)
        self.bbox_rubber.setColor(QtCore.Qt.blue)
        self.bbox_rubber.setFillColor(QtGui.QColor(0, 0, 255, 60))
        self.bbox_rubber.setWidth(2)

    def canvasPressEvent(self, e):
        self.start_point = self.canvas.getCoordinateTransform().toMapCoordinates(e.pos())
        self.is_dragging = True
        self.bbox_rubber.reset(QgsWkbTypes.PolygonGeometry)

    def canvasMoveEvent(self, e):
        if self.is_dragging and self.start_point:
            current_point = self.canvas.getCoordinateTransform().toMapCoordinates(e.pos())
            rect = QgsRectangle(self.start_point, current_point)
            self.bbox_rubber.setToGeometry(QgsGeometry.fromRect(rect), None)
            self.canvas.refresh()

    def canvasReleaseEvent(self, e):
        if self.is_dragging and self.start_point:
            end_point = self.canvas.getCoordinateTransform().toMapCoordinates(e.pos())
            rect = QgsRectangle(self.start_point, end_point)
            # Dynamic size validation based on coordinate system
            # For geographic coordinates (degrees), use much smaller thresholds
            min_size = 0.000001 if abs(rect.width()) < 1 and abs(rect.height()) < 1 else 10

            if rect.width() > min_size and rect.height() > min_size:
                self.cb(rect)
            else:
                self.bbox_rubber.reset(QgsWkbTypes.PolygonGeometry)
                self.canvas.refresh()
        self.is_dragging = False
        self.start_point = None

    def deactivate(self):
        self.bbox_rubber.reset(QgsWkbTypes.PolygonGeometry)
        self.is_dragging = False
        self.start_point = None
        super().deactivate()

    def clear_feedback(self):
        self.bbox_rubber.reset(QgsWkbTypes.PolygonGeometry)
        self.canvas.refresh()

class Switch(QtWidgets.QAbstractButton):
    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.setCheckable(True)
        self.setFixedSize(50, 28)
    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)
        track_color = QtGui.QColor("#34D399") if self.isChecked() else QtGui.QColor("#E5E7EB")
        thumb_color = QtGui.QColor("#FFFFFF")
        painter.setBrush(track_color)
        painter.setPen(Qt.NoPen)
        painter.drawRoundedRect(self.rect(), 14, 14)
        thumb_x = self.width() - 24 if self.isChecked() else 4
        thumb_rect = QtCore.QRect(thumb_x, 4, 20, 20)
        painter.setBrush(thumb_color)
        painter.drawEllipse(thumb_rect)
    def sizeHint(self):
        return self.minimumSizeHint()

class GeoOSAMControlPanel(QtWidgets.QDockWidget):
    """Enhanced SAM segmentation control panel for QGIS"""

    DEFAULT_CLASSES = {
        'Agriculture' : {
            'color': '255,215,0',
            'description': 'Farmland and crops',
            'batch_defaults': {'min_size': 200, 'max_objects': 10, 'max_size': 0}
        },
        'Buildings'   : {
            'color': '220,20,60',
            'description': 'Residential & commercial structures',
            'batch_defaults': {'min_size': 150, 'max_objects': 20, 'max_size': 0}
        },
        'Commercial'  : {
            'color': '135,206,250',
            'description': 'Shopping and business districts',
            'batch_defaults': {'min_size': 200, 'max_objects': 15, 'max_size': 0}
        },
        'Industrial'  : {
            'color': '128,0,128',
            'description': 'Factories and warehouses',
            'batch_defaults': {'min_size': 400, 'max_objects': 8, 'max_size': 0}
        },
        'Other'       : {
            'color': '148,0,211',
            'description': 'Unclassified objects',
            'batch_defaults': {'min_size': 50, 'max_objects': 25, 'max_size': 0}
        },
        'Parking'     : {
            'color': '255,140,0',
            'description': 'Parking lots and areas',
            'batch_defaults': {'min_size': 150, 'max_objects': 15, 'max_size': 0}
        },
        'Residential' : {
            'color': '255,105,180',
            'description': 'Housing areas',
            'batch_defaults': {'min_size': 50, 'max_objects': 60, 'max_size': 0}
        },
        'Roads'       : {
            'color': '105,105,105',
            'description': 'Streets, highways, and pathways',
            'batch_defaults': {'min_size': 200, 'max_objects': 10, 'max_size': 0}
        },
        'Vessels'     : {
            'color': '0,206,209',
            'description': 'Boats, ships',
            'batch_defaults': {'min_size': 40, 'max_objects': 35, 'max_size': 2000}
        },
        'Vehicle'     : {
            'color': '255,69,0',
            'description': 'Cars, trucks, and buses',
            'batch_defaults': {'min_size': 20, 'max_objects': 50, 'max_size': 500}
        },
        'Vegetation'  : {
            'color': '34,139,34',
            'description': 'Trees, grass, and parks',
            'batch_defaults': {'min_size': 30, 'max_objects': 100, 'max_size': 0}
        },
        'Water'       : {
            'color': '30,144,255',
            'description': 'Rivers, lakes, and ponds',
            'batch_defaults': {'min_size': 500, 'max_objects': 8, 'max_size': 0}
        }
    }

    EXTRA_COLORS = [
        '50,205,50', '255,20,147', '255,165,0', '186,85,211', '0,128,128',
        '255,192,203', '165,42,42', '0,250,154', '255,0,255', '127,255,212'
    ]

    EXPORT_FORMATS = {
        'GeoPackage (.gpkg)': {'driver': 'GPKG', 'ext': '.gpkg'},
        'ESRI Shapefile (.shp)': {'driver': 'ESRI Shapefile', 'ext': '.shp'},
        'GeoJSON (.geojson)': {'driver': 'GeoJSON', 'ext': '.geojson'},
        'FlatGeobuf (.fgb)': {'driver': 'FlatGeobuf', 'ext': '.fgb'},
    }

    def __init__(self, iface, parent=None):
        super().__init__("", parent)
        self.iface = iface
        self.canvas = iface.mapCanvas()

        # Initialize device and model
        self.device, self.model_choice, self.num_cores, self.available_models, self.model_options = detect_best_device()
        self._init_sam_model()

        # Setup docking with version in title
        version = self._get_plugin_version()
        self.setWindowTitle(f"Version: {version}")
        self.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)
        self.setFeatures(QtWidgets.QDockWidget.DockWidgetMovable |
                         QtWidgets.QDockWidget.DockWidgetFloatable)

        # State variables
        self.point = None
        self.bbox = None
        self.current_mode = None
        self.result_layers = {}
        self.segment_counts = {}
        self.current_class = None
        self.classes = self.DEFAULT_CLASSES.copy()
        self.worker = None
        self.original_raster_layer = None
        self.keep_raster_selected = True

        # Output management
        self.export_save_dir = None
        self.mask_save_dir = None
        self.save_debug_masks = False

        # Undo functionality
        self.undo_stack = []

        # Status tracking for filter indicator
        self._last_status_text = "Ready to segment"
        self._last_status_type = "info"
        self._active_filter_summary = ""

        # Multi-point request context (for Shift/Ctrl point accumulation)
        self.current_request = None

        # Processing queue system
        self.processing_queue = []
        self.is_processing = False
        self.worker = None

        # Initialize
        self._init_save_directories()
        self.pointTool = EnhancedPointClickTool(self.canvas, self._point_done)
        self.bboxTool = EnhancedBBoxClickTool(self.canvas, self._bbox_done)
        self.original_map_tool = None

        # Connect to map tool changes to detect when user switches tools
        self.canvas.mapToolSet.connect(self._on_map_tool_changed)

        # Batch mode settings
        self.batch_mode_enabled = False
        self.min_object_size = 50  # Minimum pixels for valid object
        self.max_objects = 20  # Prevent too many small objects
        self.max_object_size = 0  # 0 = no class-specific max (use 10% of image heuristic)
        self.duplicate_threshold = 0.85  # Spatial overlap threshold for duplicates (very lenient for shape-based detection)

        # GSD / Resolution settings
        self.gsd_cm_per_px = 0.0  # 0 = not set; pixel-only filtering

        # Real-world size filtering (requires GSD)
        self.real_world_filter_enabled = False
        self.min_diameter_cm = 0.0
        self.max_diameter_cm = 0.0
        self.min_area_m2 = 0.0
        self.max_area_m2 = 0.0

        # Shape filtering
        self.shape_filter_enabled = False
        self.min_circularity = 0.0
        self.min_compactness = 0.0
        self.max_aspect_ratio = 10.0

        self._setup_ui()

        # Connect to selection changes for remove button
        self._connect_selection_signals()

    def _debug_current_settings(self):
        """Debug current batch settings"""
        print(f"\n🔧 CURRENT SETTINGS:")
        print(f"   Batch mode enabled: {self.batch_mode_enabled}")
        print(f"   Min object size: {self.min_object_size}px")
        print(f"   Max objects: {self.max_objects}")
        print(f"   Current class: {self.current_class}")
        print(f"   Current mode: {self.current_mode}")

    def _init_sam_model(self):
        """Initialize the selected SAM model with size variant"""
        plugin_dir = os.path.dirname(os.path.abspath(__file__))

        # Parse model choice to get type and size
        # Format: "SAM2_tiny", "SAM2.1_base", "SAM3"
        if self.model_choice == "SAM3":
            self._init_sam3_model()
        elif self.model_choice.startswith("SAM2.1_"):
            # Extract size from model choice (e.g., "SAM2.1_base" -> "base")
            size = self.model_choice.split("_", 1)[1]  # Split into ['SAM2.1', 'base']
            self._init_sam21_model(size)
        elif self.model_choice.startswith("SAM2_"):
            # Extract size from model choice (e.g., "SAM2_tiny" -> "tiny")
            size = self.model_choice.split("_", 1)[1]  # Split into ['SAM2', 'tiny']
            self._init_sam2_model(plugin_dir, size)
        else:
            # Legacy fallback
            self._init_sam2_model(plugin_dir, "tiny")

    def _init_sam2_model(self, plugin_dir, size="tiny"):
        """Initialize SAM2 model with specified size"""
        # Get model configuration
        if size not in SAM2_MODELS:
            print(f"⚠️ Unknown SAM2 size '{size}', falling back to 'tiny'")
            size = "tiny"

        model_config = SAM2_MODELS[size]
        checkpoint_filename = model_config['checkpoint']
        config_name = model_config['config']

        checkpoint_path = os.path.join(
            plugin_dir, "sam2", "checkpoints", checkpoint_filename)

        if not os.path.exists(checkpoint_path):
            # Try to download the checkpoint
            print(f"📥 Downloading {model_config['name']} checkpoint...")

            # Show progress dialog
            progress = QtWidgets.QProgressDialog(
                f"Downloading {model_config['name']}...",
                "Cancel", 0, 100, self
            )
            progress.setWindowTitle("Model Download")
            progress.setWindowModality(Qt.WindowModal)
            progress.show()

            try:
                checkpoint_dir = os.path.join(plugin_dir, "sam2", "checkpoints")
                os.makedirs(checkpoint_dir, exist_ok=True)

                def reporthook(blocknum, blocksize, totalsize):
                    if totalsize > 0:
                        percent = min(blocknum * blocksize * 100 / totalsize, 100)
                        progress.setValue(int(percent))
                        QtCore.QCoreApplication.processEvents()

                urllib.request.urlretrieve(model_config['url'], checkpoint_path, reporthook=reporthook)
                progress.close()
                print(f"✅ {model_config['name']} checkpoint downloaded")

            except Exception as e:
                progress.close()
                print(f"❌ Failed to download checkpoint: {e}")
                QtWidgets.QMessageBox.critical(
                    self,
                    "Download Failed",
                    f"Failed to download {model_config['name']}:\n{e}\n\n"
                    f"Please download manually from:\n{model_config['url']}"
                )
                raise Exception(f"SAM2 {size} checkpoint required but not available")

        if GlobalHydra.instance().is_initialized():
            GlobalHydra.instance().clear()

        try:
            with initialize_config_module(config_module="sam2.configs"):
                sam_model = build_sam2(
                    config_name, checkpoint_path, device=self.device)

                if self.device == "cuda":
                    sam_model = sam_model.cuda()

                sam_model.eval()
                if self.device == "cpu":
                    try:
                        sam_model = torch.jit.optimize_for_inference(sam_model)
                    except:
                        pass

                self.predictor = SAM2ImagePredictor(sam_model)
                print(f"✅ {model_config['name']} loaded on {self.device}")

        except Exception as e:
            print(f"❌ Failed to load {model_config['name']}: {e}")
            raise

    def _init_sam21_model(self, size="base"):
        """Initialize Ultralytics SAM2.1 model with specified size (T/B/L)"""
        # Get model configuration
        if size not in SAM21_MODELS:
            print(f"⚠️ Unknown SAM2.1 size '{size}', falling back to 'base'")
            size = "base"

        model_config = SAM21_MODELS[size]
        weights_filename = model_config['weights']

        try:
            from ultralytics import SAM

            # Check if model needs downloading (first time use)
            import pathlib
            model_cache = pathlib.Path.home() / '.ultralytics' / 'weights' / weights_filename
            if not model_cache.exists():
                print(f"📥 {model_config['name']} not cached, Ultralytics will download it automatically...")
                # Show info message
                QtWidgets.QMessageBox.information(
                    self,
                    "Model Download",
                    f"Downloading {model_config['name']} for first-time use.\n\n"
                    f"This may take a minute. The model will be cached for future use.\n\n"
                    f"Check the Python console for progress."
                )

            sam21_model = SAM(weights_filename)
            self.predictor = UltralyticsPredictor(sam21_model)
            print(f"✅ {model_config['name']} loaded successfully")
        except Exception as e:
            print(f"❌ Failed to load {model_config['name']}: {e}, falling back to SAM2")
            QtWidgets.QMessageBox.warning(
                self,
                "Model Load Failed",
                f"Failed to load {model_config['name']}:\n{e}\n\n"
                f"Falling back to SAM2 Tiny."
            )
            self.model_choice = "SAM2_tiny"
            self._init_sam2_model(os.path.dirname(os.path.abspath(__file__)), "tiny")

    def _init_sam3_model(self):
        """Initialize SAM3 model (Ultralytics SAM3)"""
        try:
            import pathlib

            # Look for sam3.pt in multiple locations
            plugin_dir = os.path.dirname(os.path.abspath(__file__))
            sam3_paths = [
                os.path.join(plugin_dir, 'sam3.pt'),
                os.path.join(pathlib.Path.home(), '.ultralytics', 'weights', 'sam3.pt'),
                'sam3.pt'  # Current directory
            ]

            weights_path = None
            for path in sam3_paths:
                if os.path.exists(path):
                    weights_path = path
                    break

            if not weights_path:
                if self._show_sam3_download_dialog():
                    for path in sam3_paths:
                        if os.path.exists(path):
                            weights_path = path
                            break
                if not weights_path:
                    raise FileNotFoundError("SAM3 weights not found - download from Hugging Face")

            self.predictor = SAM3PredictorWrapper(weights_path)
            print(f"✅ SAM3 loaded from {weights_path}")

        except FileNotFoundError as e:
            print(f"❌ SAM3 weights not found: {e}")
            # Show user-friendly dialog with download instructions
            self._show_sam3_download_dialog()
            # Fallback to SAM2.1 or SAM2
            self.model_choice = "SAM2.1_base" if SAM21_AVAILABLE else "SAM2_tiny"
            print(f"Falling back to {self.model_choice}")
            if self.model_choice.startswith("SAM2.1_"):
                size = self.model_choice.split("_", 1)[1]
                self._init_sam21_model(size)
            else:
                size = self.model_choice.split("_", 1)[1] if "_" in self.model_choice else "tiny"
                self._init_sam2_model(os.path.dirname(os.path.abspath(__file__)), size)

        except Exception as e:
            print(f"❌ Failed to load SAM3: {e}")
            # Show error dialog
            QtWidgets.QMessageBox.warning(
                self,
                "SAM3 Load Failed",
                f"Failed to load SAM3: {e}\n\nFalling back to {'SAM2.1' if SAM21_AVAILABLE else 'SAM2'}"
            )
            # Fallback
            self.model_choice = "SAM2.1_base" if SAM21_AVAILABLE else "SAM2_tiny"
            if self.model_choice.startswith("SAM2.1_"):
                size = self.model_choice.split("_", 1)[1]
                self._init_sam21_model(size)
            else:
                size = self.model_choice.split("_", 1)[1] if "_" in self.model_choice else "tiny"
                self._init_sam2_model(os.path.dirname(os.path.abspath(__file__)), size)

    def _show_sam3_download_dialog(self):
        """Show SAM3 download instructions"""
        msg = QtWidgets.QMessageBox(self)
        msg.setIcon(QtWidgets.QMessageBox.Information)
        msg.setWindowTitle("SAM3 Download Required")
        msg.setText("SAM3 weights not found")
        msg.setInformativeText(
            "SAM3 requires manual download:\n\n"
            "1. Visit: https://huggingface.co/facebook/sam3\n"
            "2. Request access (Meta approval required)\n"
            "3. Download sam3.pt file\n\n"
            "You can download directly from within GeoOSAM.\n\n"
            "File locations:\n"
            f"   • {os.path.dirname(os.path.abspath(__file__))}/sam3.pt\n"
            f"   • ~/.ultralytics/weights/sam3.pt\n"
            "   • Current directory/sam3.pt\n\n"
            "Requirements:\n"
            "• Ultralytics >=8.3.237\n"
            "• Install/update: pip install -U ultralytics\n\n"
            "Falling back to SAM2.1_B/SAM2..."
        )
        download_btn = msg.addButton("Download Now", QtWidgets.QMessageBox.AcceptRole)
        msg.addButton(QtWidgets.QMessageBox.Cancel)
        msg.exec_()

        if msg.clickedButton() == download_btn:
            return self._download_sam3_weights()
        return False

    def _refresh_model_options(self):
        """Refresh model options in UI after SAM3 download."""
        try:
            prev_choice = self.model_choice
            self.device, _, self.num_cores, self.available_models, self.model_options = detect_best_device()

            if hasattr(self, "modelComboBox") and self.modelComboBox:
                self.modelComboBox.blockSignals(True)
                self.modelComboBox.clear()
                for option in self.model_options:
                    self.modelComboBox.addItem(option['display'], option['id'])

                # Restore selection if possible
                idx = self.modelComboBox.findData(prev_choice)
                if idx >= 0:
                    self.modelComboBox.setCurrentIndex(idx)
                else:
                    self.model_choice = self.modelComboBox.currentData()
                self.modelComboBox.blockSignals(False)

            if hasattr(self, "deviceLabel") and self.deviceLabel:
                device_icon = "🎮" if "cuda" in self.device else "🖥️"
                device_info = f"{device_icon} {self.device.upper()} | {self.model_choice}"
                if getattr(self, "num_cores", None):
                    device_info += f" ({self.num_cores} cores)"
                self.deviceLabel.setText(device_info)

            if self.sam3HelpLabel:
                self.sam3HelpLabel.setVisible(not self.available_models.get('SAM3', False))
        except Exception as e:
            print(f"⚠️ Failed to refresh model options: {e}")

    def _download_sam3_weights(self):
        """Download SAM3 weights with a one-time HF token."""
        token, ok = QtWidgets.QInputDialog.getText(
            self,
            "Hugging Face Token",
            "Enter your Hugging Face access token:",
            QtWidgets.QLineEdit.Password
        )
        if not ok or not token.strip():
            return False

        try:
            weights_dir = pathlib.Path.home() / ".ultralytics" / "weights"
            weights_dir.mkdir(parents=True, exist_ok=True)
            weights_path = weights_dir / "sam3.pt"

            progress = QtWidgets.QProgressDialog(
                "Downloading SAM3 weights...", "Cancel", 0, 100, self
            )
            progress.setWindowTitle("Downloading SAM3")
            progress.setWindowModality(Qt.WindowModal)
            progress.show()

            req = urllib.request.Request(
                SAM3_WEIGHTS_URL,
                headers={"Authorization": f"Bearer {token.strip()}"}
            )
            canceled = False
            with urllib.request.urlopen(req) as response, open(weights_path, "wb") as out:
                total = response.getheader("Content-Length")
                total = int(total) if total else None
                downloaded = 0
                chunk_size = 1024 * 1024

                while True:
                    if progress.wasCanceled():
                        canceled = True
                        break
                    chunk = response.read(chunk_size)
                    if not chunk:
                        break
                    out.write(chunk)
                    downloaded += len(chunk)
                    if total:
                        progress.setValue(int(downloaded * 100 / total))
                        QtWidgets.QApplication.processEvents()

            progress.close()

            if canceled or weights_path.stat().st_size < 1_000_000:
                try:
                    weights_path.unlink(missing_ok=True)
                except Exception:
                    pass
                return False

            if weights_path.exists() and weights_path.stat().st_size > 1000000:
                QtWidgets.QMessageBox.information(
                    self, "Download Complete", f"SAM3 weights downloaded to:\n{weights_path}"
                )
                self._refresh_model_options()
                return True

            QtWidgets.QMessageBox.critical(
                self, "Download Failed", "SAM3 download failed or file is incomplete."
            )
            return False

        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Download Error", str(e))
            return False

    def _show_license_dialog(self):
        """Show license key entry/management dialog"""
        from geo_osam_license import LicenseManager

        # Create dialog
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle("GeoOSAM SAM3 Pro License")
        dialog.setMinimumWidth(500)

        layout = QtWidgets.QVBoxLayout(dialog)

        # Check if already licensed
        license_info = LicenseManager.get_license_info()
        is_licensed = license_info['type'] == 'pro'

        if is_licensed:
            # Show current license info with option to change
            info_label = QtWidgets.QLabel(
                f"<h3>✅ License Active</h3>"
                f"<p><b>Email:</b> {license_info['email']}</p>"
                f"<p><b>Status:</b> {license_info['status']}</p>"
            )
            info_label.setTextFormat(Qt.RichText)
            layout.addWidget(info_label)

            # Add spacer
            layout.addSpacing(20)

            # Buttons
            button_layout = QtWidgets.QHBoxLayout()

            change_btn = QtWidgets.QPushButton("Change License")
            change_btn.clicked.connect(lambda: self._change_license(dialog))
            button_layout.addWidget(change_btn)

            remove_btn = QtWidgets.QPushButton("Remove License")
            remove_btn.clicked.connect(lambda: self._remove_license(dialog))
            button_layout.addWidget(remove_btn)

            close_btn = QtWidgets.QPushButton("Close")
            close_btn.clicked.connect(dialog.accept)
            button_layout.addWidget(close_btn)

            layout.addLayout(button_layout)

        else:
            # Show activation form
            info_label = QtWidgets.QLabel(
                "<h3>Activate SAM3 Pro License</h3>"
                "<p>GeoOSAM is free and open-source. SAM3 Pro helps fund ongoing development.</p>"
                "<p><b>Purchase License:</b> Contact <b>geoosamplugin@gmail.com</b></p>"
                "<p><i>Note: Your email is used to validate the license key</i></p>"
            )
            info_label.setTextFormat(Qt.RichText)
            info_label.setWordWrap(True)
            layout.addWidget(info_label)

            layout.addSpacing(10)

            # Email input
            email_label = QtWidgets.QLabel("Email Address:")
            layout.addWidget(email_label)

            email_input = QtWidgets.QLineEdit()
            email_input.setPlaceholderText("your.email@example.com")
            layout.addWidget(email_input)

            layout.addSpacing(10)

            # License key input
            key_label = QtWidgets.QLabel("License Key:")
            layout.addWidget(key_label)

            key_input = QtWidgets.QLineEdit()
            key_input.setPlaceholderText("GEOSAM3-XXXXX-XXXXX-XXXXX-XXXXX")
            layout.addWidget(key_input)

            layout.addSpacing(20)

            # Feature info
            features_label = QtWidgets.QLabel(
                "<b>SAM3 Pro Features:</b><br>"
                "✅ Text prompts on entire raster (unlimited)<br>"
                "✅ Similar object detection on entire raster (unlimited)<br>"
                "✅ Automatic tile processing for large rasters"
            )
            features_label.setTextFormat(Qt.RichText)
            features_label.setStyleSheet("background-color: #f0f0f0; padding: 10px; border-radius: 5px;")
            layout.addWidget(features_label)

            layout.addSpacing(20)

            # Buttons
            button_layout = QtWidgets.QHBoxLayout()

            activate_btn = QtWidgets.QPushButton("Activate License")
            activate_btn.setDefault(True)
            activate_btn.clicked.connect(
                lambda: self._activate_license(dialog, email_input.text(), key_input.text())
            )
            button_layout.addWidget(activate_btn)

            cancel_btn = QtWidgets.QPushButton("Cancel")
            cancel_btn.clicked.connect(dialog.reject)
            button_layout.addWidget(cancel_btn)

            layout.addLayout(button_layout)

        dialog.exec_()

    def _activate_license(self, dialog, email, key):
        """Validate and activate entered license key"""
        from geo_osam_license import LicenseManager

        # Validate inputs
        email = email.strip()
        key = key.strip()

        if not email:
            QtWidgets.QMessageBox.warning(
                dialog,
                "Email Required",
                "Please enter your email address."
            )
            return

        if not key:
            QtWidgets.QMessageBox.warning(
                dialog,
                "License Key Required",
                "Please enter your license key."
            )
            return

        # Validate license
        if LicenseManager.validate_license(key, email):
            # Save license
            if LicenseManager.save_license(key, email):
                QtWidgets.QMessageBox.information(
                    dialog,
                    "License Activated",
                    f"✅ License activated successfully!\n\n"
                    f"Email: {email}\n"
                    f"You now have access to SAM3 Pro features:\n"
                    f"• Entire raster processing\n"
                    f"• Unlimited text prompts\n"
                    f"• Unlimited similar object detection"
                )

                # Update UI
                self._update_license_status()

                # Close dialog
                dialog.accept()
            else:
                QtWidgets.QMessageBox.critical(
                    dialog,
                    "Activation Failed",
                    "Failed to save license. Please try again."
                )
        else:
            QtWidgets.QMessageBox.critical(
                dialog,
                "Invalid License",
                "❌ Invalid license key for this email address.\n\n"
                "Please check:\n"
                "• Email address is correct\n"
                "• License key is copied correctly\n"
                "• No extra spaces or characters\n\n"
                "If the problem persists, contact support."
            )

    def _change_license(self, dialog):
        """Allow user to change their license"""
        from geo_osam_license import LicenseManager

        reply = QtWidgets.QMessageBox.question(
            dialog,
            "Change License",
            "Are you sure you want to change your license?\n\n"
            "Your current license will be removed.",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No
        )

        if reply == QtWidgets.QMessageBox.Yes:
            LicenseManager.clear_license()
            self._update_license_status()
            dialog.accept()
            # Reopen dialog in activation mode
            self._show_license_dialog()

    def _remove_license(self, dialog):
        """Remove the current license"""
        from geo_osam_license import LicenseManager

        reply = QtWidgets.QMessageBox.question(
            dialog,
            "Remove License",
            "Are you sure you want to remove your license?\n\n"
            "You will lose access to SAM3 Pro features:\n"
            "• Entire raster processing\n"
            "• You'll be limited to extent mode only\n\n"
            "You can re-activate later with the same key.",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No
        )

        if reply == QtWidgets.QMessageBox.Yes:
            LicenseManager.clear_license()
            self._update_license_status()
            QtWidgets.QMessageBox.information(
                dialog,
                "License Removed",
                "License removed successfully.\n\n"
                "You now have SAM3 Free tier access:\n"
                "• Extent mode (visible area) - unlimited"
            )
            dialog.accept()

    def _check_raster_access(self):
        """Check if user can access entire raster mode"""
        from geo_osam_license import LicenseManager

        if LicenseManager.has_raster_access():
            return True

        # Show upgrade dialog with HTML support
        msg = QtWidgets.QMessageBox(self)
        msg.setWindowTitle("SAM3 Pro Feature")
        msg.setTextFormat(Qt.RichText)
        msg.setText(
            "Entire raster processing is a <b>SAM3 Pro</b> feature.<br><br>"
            "<b>✓ Free Tier:</b> Extent mode (visible area) - unlimited<br>"
            "<b>⭐ Pro Tier:</b> Entire raster with auto-tiling<br><br>"
            "<i>Pro licensing helps fund development of GeoOSAM</i><br><br>"
            "Would you like to activate a license?"
        )
        msg.setStandardButtons(QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No)
        msg.setIcon(QtWidgets.QMessageBox.Information)
        reply = msg.exec_()

        if reply == QtWidgets.QMessageBox.Yes:
            self._show_license_dialog()
            # Check again after dialog
            return LicenseManager.has_raster_access()

        return False

    def _update_license_status(self):
        """Update license status label in Settings tab and banner in Segmentation tab"""
        from geo_osam_license import LicenseManager

        license_info = LicenseManager.get_license_info()

        if hasattr(self, 'licenseStatusLabel') and self.licenseStatusLabel:
            if license_info['type'] == 'pro':
                self.licenseStatusLabel.setText(f"✅ {license_info['status']}")
                self.licenseStatusLabel.setStyleSheet("color: green; font-weight: bold; font-size: 14px; padding: 5px;")
            else:
                self.licenseStatusLabel.setText(f"ℹ️ {license_info['status']}")
                self.licenseStatusLabel.setStyleSheet("color: gray; font-size: 14px; padding: 5px;")

        self._update_license_banner()

    def _set_class_combo_alert(self, alert: bool):
        if alert:
            self.classComboBox.setStyleSheet("""
                QComboBox {
                    padding: 8px 10px; font-size: 14px; border-radius: 7px;
                    border: 2px solid #F59E0B; background: #FFFBEB;
                }
                QComboBox::drop-down { border: none; }
            """)
        else:
            self.classComboBox.setStyleSheet("""
                QComboBox {
                    padding: 8px 10px; font-size: 14px; border-radius: 7px;
                    border: 1px solid #D0D5DD; background: #FFF;
                }
                QComboBox::drop-down { border: none; }
            """)

    def _on_similar_mode_toggled(self, checked):
        if checked:
            self.textPromptInput.clear()
            self.textPromptInput.setEnabled(False)
            self.segmentTextBtn.setEnabled(False)
        else:
            self.textPromptInput.setEnabled(True)

    def _on_license_banner_btn(self):
        self.tabWidget.setCurrentIndex(1)  # Always go to Settings && Filters tab

    def _update_license_banner(self):
        """Update the compact license banner on the Segmentation tab"""
        if not hasattr(self, 'licenseBanner') or not self.licenseBanner:
            return
        from geo_osam_license import LicenseManager
        license_info = LicenseManager.get_license_info()

        if license_info['type'] == 'pro':
            self.licenseBanner.setStyleSheet(
                "#licenseBanner { background: #ECFDF3; border-radius: 8px; border: 1px solid #A6F4C5; }")
            self.licenseBannerLabel.setText("✅ SAM3 Pro License Active")
            self.licenseBannerLabel.setStyleSheet("font-size: 13px; font-weight: 600; color: #027A48;")
            self.licenseBannerBtn.setVisible(False)
        else:
            self.licenseBanner.setStyleSheet(
                "#licenseBanner { background: #FFFBEB; border-radius: 8px; border: 1px solid #FDE68A; }")
            self.licenseBannerLabel.setText("🔓 Free tier — Upgrade to SAM3 Pro")
            self.licenseBannerLabel.setStyleSheet("font-size: 13px; font-weight: 600; color: #92400E;")
            self.licenseBannerBtn.setVisible(True)
            self.licenseBannerBtn.setText("Settings")
            self.licenseBannerBtn.setStyleSheet("""
                QPushButton {
                    font-size: 12px; font-weight: 600; padding: 3px 10px;
                    border-radius: 6px; border: none;
                    background: #F59E0B; color: #FFF;
                }
                QPushButton:hover { background: #D97706; }
            """)

    def _on_scope_changed(self, index):
        """Handle scope selection change"""
        from geo_osam_license import LicenseManager

        # Only check if SAM3 model is selected
        if self.model_choice != "SAM3":
            return

        scope = self.scopeComboBox.currentData()

        # If user selects 'full' without license, show upgrade dialog
        if scope == 'full' and not LicenseManager.has_raster_access():
            # Show upgrade dialog with HTML support
            msg = QtWidgets.QMessageBox(self)
            msg.setWindowTitle("SAM3 Pro Feature")
            msg.setTextFormat(Qt.RichText)
            msg.setText(
                "Entire raster processing is a <b>SAM3 Pro</b> feature.<br><br>"
                "<b>✓ Free Tier:</b> Extent mode (visible area) - unlimited<br>"
                "<b>⭐ Pro Tier:</b> Entire raster with auto-tiling<br><br>"
                "<i>Pro licensing helps fund development of GeoOSAM</i><br><br>"
                "Would you like to activate a license?"
            )
            msg.setStandardButtons(QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No)
            msg.setIcon(QtWidgets.QMessageBox.Information)
            reply = msg.exec_()

            if reply == QtWidgets.QMessageBox.Yes:
                self._show_license_dialog()

            # Revert to extent mode
            self.scopeComboBox.blockSignals(True)  # Prevent recursive calls
            self.scopeComboBox.setCurrentIndex(0)  # Back to 'aoi'
            self.scopeComboBox.blockSignals(False)

    def _init_save_directories(self):
        """Initialize output directories"""
        self.export_save_dir = pathlib.Path.home() / "GeoOSAM_output"
        self.mask_save_dir = pathlib.Path.home() / "GeoOSAM_masks"
        self.export_save_dir.mkdir(exist_ok=True)

    def _connect_selection_signals(self):
        """Connect signals for layer management (simplified - no delete button)"""
        # Connect to layer removals only (no selection tracking needed)
        QgsProject.instance().layersAdded.connect(self._on_layers_added)
        QgsProject.instance().layersRemoved.connect(self._on_layers_removed)

    def _on_layers_added(self, layers):
        """Handle when layers are added (simplified)"""
        # No need to connect selection signals anymore
        pass

    def _on_layers_removed(self, layer_ids):
        """Handle when layers are removed from the project"""
        # Clean up our tracking dictionaries
        layers_to_remove = []
        for class_name, layer in self.result_layers.items():
            try:
                # Try to access layer to see if it still exists
                if layer is None or layer.id() in layer_ids:
                    layers_to_remove.append(class_name)
            except RuntimeError:
                # Layer has been deleted
                layers_to_remove.append(class_name)

        # Remove deleted layers from our tracking
        for class_name in layers_to_remove:
            if class_name in self.result_layers:
                del self.result_layers[class_name]
            if class_name in self.segment_counts:
                del self.segment_counts[class_name]

        # Update stats
        self._update_stats()

    @staticmethod
    def _tip(text):
        """Wrap tooltip text in HTML with explicit white background for QGIS compatibility"""
        html_text = text.replace('\n', '<br>')
        return f'<div style="background-color:#fff; color:#1a202c; padding:4px 6px; font-size:13px;">{html_text}</div>'

    def _setup_ui(self):
        # Base font size for the entire UI
        base_font_size = 13  # Readable on laptop screens
        self.setFont(QtGui.QFont("Segoe UI", 9))  # Dock title bar stays small
        print(f"UI setup using font size: {base_font_size}pt")

        # --- Dock features: standard QGIS close/float/move
        self.setFeatures(
            QtWidgets.QDockWidget.DockWidgetClosable |
            QtWidgets.QDockWidget.DockWidgetFloatable |
            QtWidgets.QDockWidget.DockWidgetMovable
        )

        # --- Container with header + tabbed layout
        container_widget = QtWidgets.QWidget()
        container_widget.setFont(QtGui.QFont("Segoe UI", base_font_size))
        container_widget.setStyleSheet("background: #f8f9fa; color: #344054;")
        self.setWidget(container_widget)

        container_layout = QtWidgets.QVBoxLayout(container_widget)
        container_layout.setSpacing(6)
        container_layout.setContentsMargins(10, 10, 10, 0)

        # Tab widget
        self.tabWidget = QtWidgets.QTabWidget()
        self.tabWidget.setStyleSheet("""
            QTabWidget::pane { border: none; background: transparent; }
            QTabBar::tab {
                font-size: 14px; font-weight: 600; padding: 7px 16px;
                border: none; background: #F2F4F7; color: #667085;
                border-top-left-radius: 8px; border-top-right-radius: 8px;
                margin-right: 2px;
            }
            QTabBar::tab:selected { background: #fff; color: #1570EF; border-bottom: 2px solid #1570EF; }
            QTabBar::tab:hover:!selected { background: #E4E7EC; }
        """)

        # Tab 1 — Segmentation (core workflow)
        tab1_scroll = QtWidgets.QScrollArea()
        tab1_scroll.setWidgetResizable(True)
        tab1_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        tab1_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        tab1_scroll.setStyleSheet("QScrollArea { border: none; background: #f8f9fa; }")
        tab1_widget = QtWidgets.QWidget()
        tab1_widget.setStyleSheet("background: transparent;")
        tab1_scroll.setWidget(tab1_widget)
        tab1_layout = QtWidgets.QVBoxLayout(tab1_widget)
        tab1_layout.setSpacing(12)
        tab1_layout.setContentsMargins(8, 10, 8, 10)

        # Tab 2 — Settings (configure once)
        tab2_scroll = QtWidgets.QScrollArea()
        tab2_scroll.setWidgetResizable(True)
        tab2_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        tab2_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        tab2_scroll.setStyleSheet("QScrollArea { border: none; background: #f8f9fa; }")
        tab2_widget = QtWidgets.QWidget()
        tab2_widget.setStyleSheet("background: transparent;")
        tab2_scroll.setWidget(tab2_widget)
        tab2_layout = QtWidgets.QVBoxLayout(tab2_widget)
        tab2_layout.setSpacing(12)
        tab2_layout.setContentsMargins(8, 10, 8, 10)

        self.tabWidget.addTab(tab1_scroll, "Segmentation")
        self.tabWidget.addTab(tab2_scroll, "Settings && Filters")

        # Keep references for adding cards later
        self._tab1_layout = tab1_layout
        self._tab2_layout = tab2_layout

        # No global tooltip styling — using HTML tooltips instead (see _tip helper)

        # Allow flexible resizing
        self.setMinimumWidth(280)
        self.setMaximumWidth(500)
        self.setMinimumHeight(500)
        self.resize(360, 700)

        # --- Card helper
        def create_card(title, icon=""):
            card = QtWidgets.QFrame()
            card.setObjectName("Card")
            card.setStyleSheet("""
                #Card {
                    background: #fff;
                    border-radius: 12px;
                    border: 1px solid #EAECF0;
                }
            """)
            shadow = QtWidgets.QGraphicsDropShadowEffect()
            shadow.setBlurRadius(14)
            shadow.setColor(QtGui.QColor(0, 0, 0, 26))
            shadow.setOffset(0, 2)
            card.setGraphicsEffect(shadow)
            card_layout = QtWidgets.QVBoxLayout(card)
            card_layout.setContentsMargins(12, 12, 12, 12)  # Reduced from 15
            card_layout.setSpacing(8)                        # Reduced from 12
            if title:
                header_layout = QtWidgets.QHBoxLayout()
                icon_label = QtWidgets.QLabel(icon)
                icon_label.setStyleSheet("font-size: 14px; margin-top: 1px;")  # Reduced from 22px
                header_label = QtWidgets.QLabel(f"<b>{title}</b>")
                header_label.setStyleSheet("font-size: 14px; color: #101828;")  # Reduced from 20px
                header_layout.addWidget(icon_label)
                header_layout.addWidget(header_label)
                header_layout.addStretch()
                card_layout.addLayout(header_layout)
            return card, card_layout

        # --- Title and Device Header (above tabs) ---
        title_label = QtWidgets.QLabel("GeoOSAM Control Panel")
        title_label.setStyleSheet("font-size: 16px; font-weight: bold; color: #1D2939;")
        title_label.setAlignment(Qt.AlignCenter)
        container_layout.addWidget(title_label)

        device_icon = "🎮" if "cuda" in self.device else "🖥️"
        device_info = f"{device_icon} {self.device.upper()} | {self.model_choice}"
        if getattr(self, "num_cores", None):
            device_info += f" ({self.num_cores} cores)"
        self.deviceLabel = QtWidgets.QLabel(device_info)
        self.deviceLabel.setStyleSheet("font-size: 14px; color: #475467;")
        self.deviceLabel.setAlignment(Qt.AlignCenter)
        container_layout.addWidget(self.deviceLabel)

        container_layout.addWidget(self.tabWidget)

        # --- Model Selection Card (NEW) ---
        model_card, model_layout = create_card("Model Selection", "🤖")

        self.modelComboBox = QtWidgets.QComboBox()
        self.modelComboBox.setToolTip(self._tip("Choose SAM model size based on your hardware"))
        self.modelComboBox.setStyleSheet("""
            QComboBox {
                padding: 8px 10px; font-size: 14px; border-radius: 7px;
                border: 1px solid #D0D5DD; background: #FFF;
            }
            QComboBox::drop-down { border: none; }
            QComboBox:hover { border: 1px solid #1570EF; }
        """)
        self.modelComboBox.setFocusPolicy(Qt.NoFocus)

        # Populate based on device-specific model options
        for option in self.model_options:
            self.modelComboBox.addItem(option['display'], option['id'])

        # Set current model
        current_idx = self.modelComboBox.findData(self.model_choice)
        if current_idx >= 0:
            self.modelComboBox.setCurrentIndex(current_idx)

        model_layout.addWidget(self.modelComboBox)

        # Add info label about model selection
        if "cuda" in self.device or "mps" in self.device:
            info_text = f"💡 {len(self.model_options)} GPU-optimized models available"
        else:
            info_text = f"💡 {len(self.model_options)} CPU-optimized models available"

        info_label = QtWidgets.QLabel(info_text)
        info_label.setStyleSheet("font-size: 14px; color: #475467; margin-top: 4px;")
        model_layout.addWidget(info_label)

        # Help text for SAM3 if not available (GPU only)
        self.sam3HelpLabel = None
        if ("cuda" in self.device or "mps" in self.device) and not self.available_models.get('SAM3', False):
            self.sam3HelpLabel = QtWidgets.QLabel(
                "⚠️ SAM3 not available. <a href='#sam3help' style='color: #1570EF;'>Download instructions</a>"
            )
            self.sam3HelpLabel.setStyleSheet("font-size: 14px; color: #DC6803;")
            self.sam3HelpLabel.setOpenExternalLinks(False)
            self.sam3HelpLabel.linkActivated.connect(self._show_sam3_download_dialog)
            model_layout.addWidget(self.sam3HelpLabel)

        # --- Compact license banner (above model card, SAM3 only) ---
        self.licenseBanner = None
        if self.model_choice == "SAM3" or self.available_models.get('SAM3', False):
            self.licenseBanner = QtWidgets.QWidget()
            self.licenseBanner.setObjectName("licenseBanner")
            banner_layout = QtWidgets.QHBoxLayout(self.licenseBanner)
            banner_layout.setContentsMargins(10, 6, 10, 6)
            banner_layout.setSpacing(8)

            self.licenseBannerLabel = QtWidgets.QLabel()
            self.licenseBannerLabel.setStyleSheet("font-size: 13px; font-weight: 600;")

            self.licenseBannerBtn = QtWidgets.QPushButton()
            self.licenseBannerBtn.setCursor(Qt.PointingHandCursor)
            self.licenseBannerBtn.setFocusPolicy(Qt.NoFocus)
            self.licenseBannerBtn.setAutoDefault(False)
            self.licenseBannerBtn.setDefault(False)
            self.licenseBannerBtn.clicked.connect(self._on_license_banner_btn)

            banner_layout.addWidget(self.licenseBannerLabel, stretch=1)
            banner_layout.addWidget(self.licenseBannerBtn)
            tab1_layout.addWidget(self.licenseBanner)
            self._update_license_banner()

        tab1_layout.addWidget(model_card)

        # --- SAM3 License Status (only show if SAM3 available or selected) ---
        self.licenseCard = None
        self.licenseStatusLabel = None
        self.manageLicenseBtn = None

        if self.model_choice == "SAM3" or self.available_models.get('SAM3', False):
            license_card, license_layout = create_card("SAM3 Pro License", "🔑")
            self.licenseCard = license_card

            # License status label
            self.licenseStatusLabel = QtWidgets.QLabel()
            self.licenseStatusLabel.setWordWrap(True)
            self.licenseStatusLabel.setStyleSheet("font-size: 14px; padding: 5px;")
            license_layout.addWidget(self.licenseStatusLabel)

            # Manage License button
            self.manageLicenseBtn = QtWidgets.QPushButton("Manage License")
            self.manageLicenseBtn.setCursor(Qt.PointingHandCursor)
            self.manageLicenseBtn.setStyleSheet("""
                QPushButton {
                    font-size: 14px; padding: 6px 16px; border-radius: 8px;
                    background: #1570EF; color: #FFF; border: none;
                }
                QPushButton:hover { background: #1366D6; }
            """)
            self.manageLicenseBtn.setAutoDefault(False)
            self.manageLicenseBtn.setDefault(False)
            self.manageLicenseBtn.setFocusPolicy(Qt.NoFocus)
            self.manageLicenseBtn.clicked.connect(self._show_license_dialog)
            license_layout.addWidget(self.manageLicenseBtn)

            # Update license status
            self._update_license_status()

            tab2_layout.addWidget(license_card)

        # --- Output Settings ---
        output_card, output_layout = create_card("Output Settings", "📂")
        folder_layout = QtWidgets.QHBoxLayout()
        self.outputFolderLabel = QtWidgets.QLabel("Default folder")
        self.outputFolderLabel.setStyleSheet("font-size: 14px; color: #475467;")  # Reduced from 18px
        self.selectFolderBtn = QtWidgets.QPushButton("Choose")
        self.selectFolderBtn.setCursor(Qt.PointingHandCursor)
        self.selectFolderBtn.setStyleSheet("""
            QPushButton {
                font-size: 14px; padding: 6px 16px; border-radius: 8px;
                background: #FFF; border: 1px solid #D0D5DD;
            }
            QPushButton:hover { background: #F9FAFB; }
        """)  # Reduced font-size and padding
        self.selectFolderBtn.setAutoDefault(False)
        self.selectFolderBtn.setDefault(False)
        self.selectFolderBtn.setFocusPolicy(Qt.NoFocus)
        folder_layout.addWidget(self.outputFolderLabel)
        folder_layout.addStretch()
        folder_layout.addWidget(self.selectFolderBtn)
        output_layout.addLayout(folder_layout)

        format_layout = QtWidgets.QHBoxLayout()
        format_label = QtWidgets.QLabel("Export format")
        format_label.setStyleSheet("font-size: 14px; color: #475467;")
        self.formatComboBox = QtWidgets.QComboBox()
        self.formatComboBox.setStyleSheet("""
            QComboBox {
                padding: 8px 10px; font-size: 14px; border-radius: 7px;
                border: 1px solid #D0D5DD; background: #FFF; color: #344054;
            }
            QComboBox::drop-down { border: none; }
            QComboBox:hover { border: 1px solid #1570EF; }
        """)
        self.formatComboBox.setFocusPolicy(Qt.NoFocus)
        for fmt in self.EXPORT_FORMATS:
            self.formatComboBox.addItem(fmt)
        format_layout.addWidget(format_label)
        format_layout.addStretch()
        format_layout.addWidget(self.formatComboBox)
        output_layout.addLayout(format_layout)

        debug_layout = QtWidgets.QHBoxLayout()
        debug_label = QtWidgets.QLabel("Save debug masks")
        debug_label.setStyleSheet("font-size: 14px;")  # Reduced from 18px
        self.saveDebugSwitch = Switch()
        debug_layout.addWidget(debug_label)
        debug_layout.addStretch()
        debug_layout.addWidget(self.saveDebugSwitch)
        output_layout.addLayout(debug_layout)
        tab2_layout.addWidget(output_card)

        # --- GSD / Resolution Card (Tab 2) ---
        gsd_card, gsd_layout = create_card("GSD / Resolution", "📐")

        gsd_row = QtWidgets.QHBoxLayout()
        gsd_label = QtWidgets.QLabel("GSD:")
        gsd_label.setStyleSheet("font-size: 14px; color: #475467;")
        self.gsdSpinBox = QtWidgets.QDoubleSpinBox()
        self.gsdSpinBox.setRange(0.01, 10000.0)
        self.gsdSpinBox.setDecimals(2)
        self.gsdSpinBox.setValue(0.01)
        self.gsdSpinBox.setSuffix(" cm/px")
        self.gsdSpinBox.setSpecialValueText("Not set")
        self.gsdSpinBox.setStyleSheet("""
            QDoubleSpinBox {
                padding: 6px 8px; font-size: 14px; border-radius: 7px;
                border: 1px solid #D0D5DD; background: #FFF; color: #344054;
            }
            QDoubleSpinBox:hover { border: 1px solid #1570EF; }
        """)
        self.gsdSpinBox.setToolTip(self._tip("Ground Sample Distance — centimeters per pixel.\nAuto-detect reads from raster metadata."))
        self.gsdSpinBox.setFixedWidth(110)
        self.gsdSpinBox.setFocusPolicy(Qt.NoFocus)

        self.autoGsdBtn = QtWidgets.QPushButton("Auto-Detect")
        self.autoGsdBtn.setCursor(Qt.PointingHandCursor)
        self.autoGsdBtn.setStyleSheet("""
            QPushButton {
                font-size: 14px; padding: 5px 12px; border-radius: 7px;
                background: #1570EF; color: #FFF; border: none;
            }
            QPushButton:hover { background: #1849A9; }
        """)
        self.autoGsdBtn.setAutoDefault(False)
        self.autoGsdBtn.setDefault(False)
        self.autoGsdBtn.setFocusPolicy(Qt.NoFocus)

        gsd_row.addWidget(gsd_label)
        gsd_row.addWidget(self.gsdSpinBox)
        gsd_row.addStretch()
        gsd_row.addWidget(self.autoGsdBtn)
        gsd_layout.addLayout(gsd_row)

        self.gsdSourceLabel = QtWidgets.QLabel("Not set — click Auto-Detect or enter manually")
        self.gsdSourceLabel.setStyleSheet("font-size: 14px; color: #9CA3AF; font-style: italic;")
        gsd_layout.addWidget(self.gsdSourceLabel)

        tab2_layout.addWidget(gsd_card)

        # --- Size Filtering Card (Tab 2) ---
        size_card, size_layout = create_card("Size Filtering", "📏")

        sf_toggle_row = QtWidgets.QHBoxLayout()
        sf_toggle_label = QtWidgets.QLabel("Real-world filtering")
        sf_toggle_label.setStyleSheet("font-size: 14px;")
        self.realWorldFilterSwitch = Switch()
        self.realWorldFilterSwitch.setEnabled(False)  # Disabled until GSD is set
        sf_toggle_row.addWidget(sf_toggle_label)
        sf_toggle_row.addStretch()
        sf_toggle_row.addWidget(self.realWorldFilterSwitch)
        size_layout.addLayout(sf_toggle_row)

        self.realWorldFilterFrame = QtWidgets.QFrame()
        self.realWorldFilterFrame.setStyleSheet("""
            QFrame { background: #F9FAFB; border: 1px solid #EAECF0; border-radius: 8px; }
        """)
        rwf_layout = QtWidgets.QGridLayout(self.realWorldFilterFrame)
        rwf_layout.setContentsMargins(8, 8, 8, 8)
        rwf_layout.setSpacing(6)

        spinbox_style = """
            QDoubleSpinBox {
                padding: 4px 6px; font-size: 14px; border-radius: 5px;
                border: 1px solid #D0D5DD; background: #FFF; color: #344054;
            }
        """

        # Min diameter
        rwf_layout.addWidget(QtWidgets.QLabel("Min diameter:"), 0, 0)
        self.minDiameterSpinBox = QtWidgets.QDoubleSpinBox()
        self.minDiameterSpinBox.setRange(0, 100000)
        self.minDiameterSpinBox.setDecimals(1)
        self.minDiameterSpinBox.setSuffix(" cm")
        self.minDiameterSpinBox.setStyleSheet(spinbox_style)
        self.minDiameterSpinBox.setFixedWidth(95)
        self.minDiameterSpinBox.setFocusPolicy(Qt.NoFocus)
        rwf_layout.addWidget(self.minDiameterSpinBox, 0, 1)
        self.minDiameterPixelLabel = QtWidgets.QLabel("(set GSD)")
        self.minDiameterPixelLabel.setStyleSheet("font-size: 14px; color: #9CA3AF;")
        rwf_layout.addWidget(self.minDiameterPixelLabel, 0, 2)

        # Max diameter
        rwf_layout.addWidget(QtWidgets.QLabel("Max diameter:"), 1, 0)
        self.maxDiameterSpinBox = QtWidgets.QDoubleSpinBox()
        self.maxDiameterSpinBox.setRange(0, 100000)
        self.maxDiameterSpinBox.setDecimals(1)
        self.maxDiameterSpinBox.setSuffix(" cm")
        self.maxDiameterSpinBox.setStyleSheet(spinbox_style)
        self.maxDiameterSpinBox.setFixedWidth(95)
        self.maxDiameterSpinBox.setFocusPolicy(Qt.NoFocus)
        rwf_layout.addWidget(self.maxDiameterSpinBox, 1, 1)
        self.maxDiameterPixelLabel = QtWidgets.QLabel("(set GSD)")
        self.maxDiameterPixelLabel.setStyleSheet("font-size: 14px; color: #9CA3AF;")
        rwf_layout.addWidget(self.maxDiameterPixelLabel, 1, 2)

        # Min area
        rwf_layout.addWidget(QtWidgets.QLabel("Min area:"), 2, 0)
        self.minAreaSpinBox = QtWidgets.QDoubleSpinBox()
        self.minAreaSpinBox.setRange(0, 1e9)
        self.minAreaSpinBox.setDecimals(2)
        self.minAreaSpinBox.setSuffix(" m\u00B2")
        self.minAreaSpinBox.setStyleSheet(spinbox_style)
        self.minAreaSpinBox.setFixedWidth(95)
        self.minAreaSpinBox.setFocusPolicy(Qt.NoFocus)
        rwf_layout.addWidget(self.minAreaSpinBox, 2, 1)
        self.minAreaPixelLabel = QtWidgets.QLabel("(set GSD)")
        self.minAreaPixelLabel.setStyleSheet("font-size: 14px; color: #9CA3AF;")
        rwf_layout.addWidget(self.minAreaPixelLabel, 2, 2)

        # Max area
        rwf_layout.addWidget(QtWidgets.QLabel("Max area:"), 3, 0)
        self.maxAreaSpinBox = QtWidgets.QDoubleSpinBox()
        self.maxAreaSpinBox.setRange(0, 1e9)
        self.maxAreaSpinBox.setDecimals(2)
        self.maxAreaSpinBox.setSuffix(" m\u00B2")
        self.maxAreaSpinBox.setStyleSheet(spinbox_style)
        self.maxAreaSpinBox.setFixedWidth(95)
        self.maxAreaSpinBox.setFocusPolicy(Qt.NoFocus)
        rwf_layout.addWidget(self.maxAreaSpinBox, 3, 1)
        self.maxAreaPixelLabel = QtWidgets.QLabel("(set GSD)")
        self.maxAreaPixelLabel.setStyleSheet("font-size: 14px; color: #9CA3AF;")
        rwf_layout.addWidget(self.maxAreaPixelLabel, 3, 2)

        # Style grid labels
        for i in range(4):
            label = rwf_layout.itemAtPosition(i, 0).widget()
            label.setStyleSheet("font-size: 14px; color: #475467;")

        self.realWorldFilterFrame.setVisible(False)
        size_layout.addWidget(self.realWorldFilterFrame)

        self.gsdRequiredLabel = QtWidgets.QLabel("Requires GSD to be set above")
        self.gsdRequiredLabel.setStyleSheet("font-size: 14px; color: #DC6803; font-style: italic;")
        size_layout.addWidget(self.gsdRequiredLabel)

        tab2_layout.addWidget(size_card)

        # --- Shape Filters Card (Tab 2) ---
        shape_card, shape_layout = create_card("Shape Filtering", "🔷")

        shape_toggle_row = QtWidgets.QHBoxLayout()
        shape_toggle_label = QtWidgets.QLabel("Enable shape filtering")
        shape_toggle_label.setStyleSheet("font-size: 14px;")
        self.shapeFilterSwitch = Switch()
        shape_toggle_row.addWidget(shape_toggle_label)
        shape_toggle_row.addStretch()
        shape_toggle_row.addWidget(self.shapeFilterSwitch)
        shape_layout.addLayout(shape_toggle_row)

        self.shapeFilterFrame = QtWidgets.QFrame()
        self.shapeFilterFrame.setStyleSheet("""
            QFrame { background: #F9FAFB; border: 1px solid #EAECF0; border-radius: 8px; }
        """)
        shf_layout = QtWidgets.QGridLayout(self.shapeFilterFrame)
        shf_layout.setContentsMargins(8, 8, 8, 8)
        shf_layout.setSpacing(6)

        # Min circularity
        circ_label = QtWidgets.QLabel("Min circularity:")
        circ_label.setStyleSheet("font-size: 14px; color: #475467;")
        shf_layout.addWidget(circ_label, 0, 0)
        self.circularitySpinBox = QtWidgets.QDoubleSpinBox()
        self.circularitySpinBox.setRange(0.0, 1.0)
        self.circularitySpinBox.setSingleStep(0.05)
        self.circularitySpinBox.setDecimals(2)
        self.circularitySpinBox.setValue(0.0)
        self.circularitySpinBox.setStyleSheet(spinbox_style)
        self.circularitySpinBox.setFixedWidth(70)
        self.circularitySpinBox.setFocusPolicy(Qt.NoFocus)
        self.circularitySpinBox.setToolTip(self._tip("4*pi*area/perimeter\u00B2\n1.0 = perfect circle, 0.0 = any shape"))
        shf_layout.addWidget(self.circularitySpinBox, 0, 1)

        # Min compactness
        comp_label = QtWidgets.QLabel("Min compactness:")
        comp_label.setStyleSheet("font-size: 14px; color: #475467;")
        shf_layout.addWidget(comp_label, 1, 0)
        self.compactnessSpinBox = QtWidgets.QDoubleSpinBox()
        self.compactnessSpinBox.setRange(0.0, 1.0)
        self.compactnessSpinBox.setSingleStep(0.05)
        self.compactnessSpinBox.setDecimals(2)
        self.compactnessSpinBox.setValue(0.0)
        self.compactnessSpinBox.setStyleSheet(spinbox_style)
        self.compactnessSpinBox.setFixedWidth(70)
        self.compactnessSpinBox.setFocusPolicy(Qt.NoFocus)
        self.compactnessSpinBox.setToolTip(self._tip("area / bounding_box_area\n1.0 = fills bbox perfectly, 0.0 = any shape"))
        shf_layout.addWidget(self.compactnessSpinBox, 1, 1)

        # Max aspect ratio
        ar_label = QtWidgets.QLabel("Max aspect ratio:")
        ar_label.setStyleSheet("font-size: 14px; color: #475467;")
        shf_layout.addWidget(ar_label, 2, 0)
        self.aspectRatioSpinBox = QtWidgets.QDoubleSpinBox()
        self.aspectRatioSpinBox.setRange(1.0, 20.0)
        self.aspectRatioSpinBox.setSingleStep(0.5)
        self.aspectRatioSpinBox.setDecimals(1)
        self.aspectRatioSpinBox.setValue(10.0)
        self.aspectRatioSpinBox.setStyleSheet(spinbox_style)
        self.aspectRatioSpinBox.setFixedWidth(70)
        self.aspectRatioSpinBox.setFocusPolicy(Qt.NoFocus)
        self.aspectRatioSpinBox.setToolTip(self._tip("max(w,h)/min(w,h)\n1.0 = square, high = elongated"))
        shf_layout.addWidget(self.aspectRatioSpinBox, 2, 1)

        shape_hint = QtWidgets.QLabel("Filters objects by geometric shape after segmentation")
        shape_hint.setStyleSheet("font-size: 14px; color: #9CA3AF; font-style: italic;")
        shf_layout.addWidget(shape_hint, 3, 0, 1, 2)

        self.shapeFilterFrame.setVisible(False)
        shape_layout.addWidget(self.shapeFilterFrame)

        tab2_layout.addWidget(shape_card)

        # --- Class Selection ---
        class_card, class_layout = create_card("Class Selection", "🏷️")

        # Class dropdown
        self.classComboBox = QtWidgets.QComboBox()
        self.classComboBox.addItem("-- Select Class --", None)
        for class_name in self.classes.keys():
            self.classComboBox.addItem(class_name, class_name)
        self.classComboBox.setStyleSheet("""
            QComboBox {
                padding: 8px 10px; font-size: 14px; border-radius: 7px;
                border: 2px solid #F59E0B; background: #FFFBEB;
            }
            QComboBox::drop-down { border: none; }
        """)  # Reduced padding and font-size
        self.classComboBox.setFocusPolicy(Qt.NoFocus)
        class_layout.addWidget(self.classComboBox)

        # Current class label
        self.currentClassLabel = QtWidgets.QLabel("No class selected")
        self.currentClassLabel.setWordWrap(True)
        self.currentClassLabel.setStyleSheet("""
            font-weight: 600; padding: 12px; margin: 4px; 
            border: 2px solid #D0D5DD; 
            background-color: #F9FAFB; 
            color: #667085;
            border-radius: 8px; font-size: 14px;
        """)  # Reduced padding and font-size
        class_layout.addWidget(self.currentClassLabel)

        # Add/Edit buttons
        class_btn_layout = QtWidgets.QHBoxLayout()
        self.addClassBtn = QtWidgets.QPushButton("➕ Add")
        self.editClassBtn = QtWidgets.QPushButton("✏️ Edit")
        for btn in [self.addClassBtn, self.editClassBtn]:
            btn.setCursor(Qt.PointingHandCursor)
            btn.setAutoDefault(False)
            btn.setDefault(False)
            btn.setFocusPolicy(Qt.NoFocus)
            btn.setStyleSheet("""
                QPushButton {
                    font-size: 14px; padding: 8px; border-radius: 8px;
                    background: #FFF; border: 1px solid #D0D5DD;
                }
                QPushButton:hover { background: #F9FAFB; }
            """)  # Reduced font-size and padding
            class_btn_layout.addWidget(btn)
        class_layout.addLayout(class_btn_layout)
        tab1_layout.addWidget(class_card)

        # --- SAM3 Modes Card (SAM3 only: text prompt + find similar) ---
        self.textPromptCard, text_prompt_layout = create_card("SAM3 Modes", "🤖")
        self.textPromptCard.setVisible(self.model_choice == "SAM3")

        # Row 1: [input - full width]
        self.textPromptInput = QtWidgets.QLineEdit()
        self.textPromptInput.setPlaceholderText("Text prompt — e.g. building, car, tree")
        self.textPromptInput.setStyleSheet("""
            QLineEdit {
                padding: 8px 10px; font-size: 14px; border-radius: 7px;
                border: 1px solid #D0D5DD; background: #FFF;
            }
            QLineEdit:focus { border: 2px solid #1570EF; }
            QLineEdit:disabled {
                background: #F3F4F6; color: #9CA3AF;
                border: 1px solid #E5E7EB;
            }
        """)
        self.textPromptInput.setFocusPolicy(Qt.StrongFocus)
        text_prompt_layout.addWidget(self.textPromptInput)

        # Row 2: [🤖 Auto-Segment]  or  [👁 Find Similar]
        self.segmentTextBtn = QtWidgets.QPushButton("🤖 Auto-Segment")
        self.segmentTextBtn.setCursor(Qt.PointingHandCursor)
        self.segmentTextBtn.setEnabled(False)  # Enabled only when text is entered
        self.segmentTextBtn.setStyleSheet("""
            QPushButton {
                font-size: 14px; font-weight: 600; padding: 8px 12px;
                border-radius: 8px; color: #FFF;
                background: #1570EF; border: 1px solid #1570EF;
            }
            QPushButton:hover { background: #1849A9; }
            QPushButton:disabled { background: #D0D5DD; border: 1px solid #D0D5DD; color: #9CA3AF; }
        """)
        self.segmentTextBtn.setAutoDefault(False)
        self.segmentTextBtn.setFocusPolicy(Qt.NoFocus)

        self.similarModeBtn = QtWidgets.QPushButton("👁 Find Similar")
        self.similarModeBtn.setCheckable(True)
        self.similarModeBtn.setVisible(self.model_choice == "SAM3")
        self.similarModeBtn.setToolTip("")
        self.similarModeBtn.setFocusPolicy(Qt.NoFocus)

        or_lbl = QtWidgets.QLabel("or")
        or_lbl.setStyleSheet("font-size: 12px; color: #9CA3AF;")
        or_lbl.setFixedWidth(20)
        or_lbl.setAlignment(Qt.AlignCenter)

        btn_row = QtWidgets.QHBoxLayout()
        btn_row.setSpacing(4)
        btn_row.addWidget(self.segmentTextBtn, stretch=1)
        btn_row.addWidget(or_lbl)
        btn_row.addWidget(self.similarModeBtn, stretch=1)
        text_prompt_layout.addLayout(btn_row)

        # Row 3: Scope
        self.scopeComboBox = QtWidgets.QComboBox()
        self.scopeComboBox.addItem("Visible Extent (AOI)", "aoi")
        self.scopeComboBox.addItem("Entire Raster (Auto-slice)", "full")
        self.scopeComboBox.setStyleSheet("""
            QComboBox {
                font-size: 14px; padding: 6px; border-radius: 6px;
                border: 1px solid #D0D5DD; background: #FFF;
            }
            QComboBox:hover { border-color: #1570EF; }
        """)
        self.scopeComboBox.setFocusPolicy(Qt.NoFocus)
        self.scopeComboBox.currentIndexChanged.connect(self._on_scope_changed)

        scope_row = QtWidgets.QHBoxLayout()
        scope_row.setSpacing(8)
        scope_row.addWidget(QtWidgets.QLabel("Scope:"))
        scope_row.itemAt(0).widget().setStyleSheet("font-size: 14px; color: #475467;")
        scope_row.addWidget(self.scopeComboBox, stretch=1)
        text_prompt_layout.addLayout(scope_row)

        tab1_layout.addWidget(self.textPromptCard)

        # --- Enhanced Segmentation Mode ---
        mode_card, mode_layout = create_card("Segmentation Mode", "🎯")

        # Point mode button (existing)
        self.pointModeBtn = QtWidgets.QPushButton("Point Mode")
        self.pointModeBtn.setCheckable(True)
        self.pointModeBtn.setChecked(True)

        # Enhanced BBox mode button
        self.bboxModeBtn = QtWidgets.QPushButton("BBox Mode")
        self.bboxModeBtn.setCheckable(True)
        self.bboxModeBtn.setVisible(True)

        # Button group for click-based modes only — Auto-Segment is an action, not a mode
        self.mode_button_group = QtWidgets.QButtonGroup()
        self.mode_button_group.addButton(self.pointModeBtn)
        self.mode_button_group.addButton(self.bboxModeBtn)
        self.mode_button_group.addButton(self.similarModeBtn)
        self.mode_button_group.setExclusive(True)

        mode_btn_style = """
            QPushButton {
                font-size: 14px; font-weight: 600; padding: 10px;
                border-radius: 8px; border: 1px solid #D0D5DD;
                background: #FFF; color: #344054;
            }
            QPushButton:hover { background: #F9FAFB; }
            QPushButton:checked {
                color: #FFF; background: #1570EF; border: 1px solid #1570EF;
            }
        """
        self.pointModeBtn.setStyleSheet(mode_btn_style)
        self.bboxModeBtn.setStyleSheet(mode_btn_style)
        self.similarModeBtn.setStyleSheet(mode_btn_style)

        # NEW: Batch mode toggle
        batch_layout = QtWidgets.QHBoxLayout()
        batch_label = QtWidgets.QLabel("Batch Segmentation")
        batch_label.setStyleSheet("font-size: 14px;")
        batch_label.setToolTip(self._tip("Find multiple objects in bbox area"))
        self.batchModeSwitch = Switch()
        self.batchModeSwitch.setToolTip(self._tip("Enable to find multiple objects in bbox"))
        batch_layout.addWidget(batch_label)
        batch_layout.addStretch()
        batch_layout.addWidget(self.batchModeSwitch)

        # NEW: Batch settings (initially hidden) - ENHANCED WITH TOOLTIPS
        self.batchSettingsFrame = QtWidgets.QFrame()
        self.batchSettingsFrame.setStyleSheet("""
            QFrame {
                background: #F9FAFB; 
                border: 1px solid #E5E7EB; 
                border-radius: 6px; 
                margin: 2px;
            }
        """)
        self.batchSettingsFrame.setSizePolicy(
            QtWidgets.QSizePolicy.Preferred, 
            QtWidgets.QSizePolicy.Maximum
        )

        batch_settings_layout = QtWidgets.QVBoxLayout(self.batchSettingsFrame)
        batch_settings_layout.setContentsMargins(8, 6, 8, 6)  # Reduced margins
        batch_settings_layout.setSpacing(3)  # Reduced spacing

        # Min object size setting - ENHANCED WITH TOOLTIPS
        size_layout = QtWidgets.QHBoxLayout()
        size_layout.setSpacing(4)
        size_label = QtWidgets.QLabel("Min size:")
        size_label.setStyleSheet("font-size: 14px; color: #667085;")
        size_label.setFixedWidth(70)
        self.minSizeSpinBox = QtWidgets.QSpinBox()
        self.minSizeSpinBox.setRange(10, 500)
        self.minSizeSpinBox.setValue(50)
        self.minSizeSpinBox.setSuffix("px")
        self.minSizeSpinBox.setFixedWidth(100)
        self.minSizeSpinBox.setStyleSheet("""
            QSpinBox { 
                font-size: 14px; padding: 2px; 
                border: 1px solid #D0D5DD; border-radius: 3px; 
            }
        """)
        # ENHANCED: Add helpful tooltip with class recommendations
        self.minSizeSpinBox.setToolTip("")
        size_layout.addWidget(size_label)
        size_layout.addWidget(self.minSizeSpinBox)
        size_layout.addStretch()

        # Max objects setting - ENHANCED WITH TOOLTIPS
        max_layout = QtWidgets.QHBoxLayout()
        max_layout.setSpacing(4)
        max_label = QtWidgets.QLabel("Max obj:")
        max_label.setStyleSheet("font-size: 14px; color: #667085;")
        max_label.setFixedWidth(70)
        self.maxObjectsSpinBox = QtWidgets.QSpinBox()
        self.maxObjectsSpinBox.setRange(1, 50)
        self.maxObjectsSpinBox.setValue(20)
        self.maxObjectsSpinBox.setFixedWidth(100)
        self.maxObjectsSpinBox.setStyleSheet("""
            QSpinBox { 
                font-size: 14px; padding: 2px; 
                border: 1px solid #D0D5DD; border-radius: 3px; 
            }
        """)
        # ENHANCED: Add helpful tooltip with class recommendations
        self.maxObjectsSpinBox.setToolTip("")
        max_layout.addWidget(max_label)
        max_layout.addWidget(self.maxObjectsSpinBox)
        max_layout.addStretch()

        batch_settings_layout.addLayout(size_layout)
        batch_settings_layout.addLayout(max_layout)

        # ENHANCED: Add helpful hints label
        self.classHintsLabel = QtWidgets.QLabel("Auto-adjusts based on selected class")
        self.classHintsLabel.setStyleSheet("font-size: 14px; color: #9CA3AF; font-style: italic;")
        self.classHintsLabel.setAlignment(Qt.AlignCenter)
        batch_settings_layout.addWidget(self.classHintsLabel)

        # Initially hidden and properly sized
        self.batchSettingsFrame.setVisible(False)
        self.batchSettingsFrame.setMaximumHeight(80)  # Slightly increased for hints label

        # Add all to mode layout
        mode_layout.addWidget(self.pointModeBtn)
        mode_layout.addWidget(self.bboxModeBtn)
        mode_layout.addLayout(batch_layout)
        mode_layout.addWidget(self.batchSettingsFrame)
        tab1_layout.addWidget(mode_card)

        # --- Status & Controls Card ---
        status_card, status_layout = create_card("Status & Controls", "⚙️")
        self.statusLabel = QtWidgets.QLabel("Ready to segment")
        self.statusLabel.setWordWrap(True)
        self.statusLabel.setStyleSheet("""
            padding: 10px; border-radius: 8px; font-size: 16px; font-weight: 500;
            background: #ECFDF3; color: #027A48; border: 1px solid #D1FADF;
        """)  # Reduced padding and font-size
        status_layout.addWidget(self.statusLabel)

        self.statsLabel = QtWidgets.QLabel("Total Segments: 0 | Classes: 0")
        self.statsLabel.setStyleSheet(
            "font-size: 14px; color: #475467; margin-top: 3px; margin-bottom: 3px;")  # Reduced from 18px
        status_layout.addWidget(self.statsLabel)

        # Filter info is shown inside the status label (no separate indicator)

        self.progressBar = QtWidgets.QProgressBar()
        self.progressBar.setRange(0, 0)
        self.progressBar.setVisible(False)
        self.progressBar.setTextVisible(False)
        self.progressBar.setStyleSheet("""
            QProgressBar {
                border: 1px solid #D0D5DD; border-radius: 8px;
                background-color: #F2F4F7; height: 8px;
            }
            QProgressBar::chunk {
                background-color: #1570EF; border-radius: 8px;
            }
        """)  # Reduced height
        status_layout.addWidget(self.progressBar)

        self.undoBtn = QtWidgets.QPushButton("⟲ Undo Last Polygon")
        self.undoBtn.setEnabled(False)
        self.undoBtn.setCursor(Qt.PointingHandCursor)
        self.undoBtn.setStyleSheet("""
            QPushButton {
                font-size: 14px; font-weight: 600; padding: 10px;
                border-radius: 8px; background: #DC2626; color: #FFF;
                border: 1px solid #DC2626;
            }
            QPushButton:hover { background: #B91C1C; }
            QPushButton:disabled {
                background: #F2F4F7; color: #98A2B3; border-color: #EAECF0;
            }
        """)  # Reduced font-size and padding

        self.undoBtn.setAutoDefault(False)
        self.undoBtn.setDefault(False)
        self.undoBtn.setFocusPolicy(Qt.NoFocus)

        export_btn_style = """
            QPushButton {
                font-size: 14px; font-weight: 600; padding: 10px;
                border-radius: 8px; color: #FFF;
                background: #027A48; border: 1px solid #027A48;
            }
            QPushButton:hover { background: #039855; }
        """

        self.exportBtn = QtWidgets.QPushButton("💾 Export All")
        self.exportBtn.setCursor(Qt.PointingHandCursor)
        self.exportBtn.setStyleSheet(export_btn_style)
        self.exportBtn.setAutoDefault(False)
        self.exportBtn.setDefault(False)
        self.exportBtn.setFocusPolicy(Qt.NoFocus)

        self.exportSelectedBtn = QtWidgets.QPushButton("💾 Export Selected")
        self.exportSelectedBtn.setCursor(Qt.PointingHandCursor)
        self.exportSelectedBtn.setStyleSheet(export_btn_style)
        self.exportSelectedBtn.setAutoDefault(False)
        self.exportSelectedBtn.setDefault(False)
        self.exportSelectedBtn.setFocusPolicy(Qt.NoFocus)
        self.exportSelectedBtn.setToolTip(self._tip("Export layers selected in the QGIS Layers panel"))

        export_row = QtWidgets.QHBoxLayout()
        export_row.setSpacing(6)
        export_row.addWidget(self.exportBtn, stretch=1)
        export_row.addWidget(self.exportSelectedBtn, stretch=1)

        status_layout.addWidget(self.undoBtn)
        status_layout.addLayout(export_row)
        tab1_layout.addWidget(status_card)

        tab1_layout.addStretch()
        tab2_layout.addStretch()
        self.setFocusPolicy(Qt.ClickFocus)

        # Enable proper resizing
        self.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        container_widget.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Minimum)

        # Force initial layout
        self.adjustSize()

        # Connect all the signals (keeping original connections)
        self.selectFolderBtn.clicked.connect(self._select_output_folder)
        self.saveDebugSwitch.toggled.connect(self._on_debug_toggle)
        self.addClassBtn.clicked.connect(self._add_new_class)
        self.editClassBtn.clicked.connect(self._edit_classes)
        self.classComboBox.currentTextChanged.connect(self._on_class_changed)
        self.pointModeBtn.clicked.connect(self._activate_point_tool)
        self.bboxModeBtn.clicked.connect(self._activate_bbox_tool)
        self.undoBtn.clicked.connect(self._undo_last_polygon)
        self.exportBtn.clicked.connect(self._export_all_classes)
        self.exportSelectedBtn.clicked.connect(self._export_selected_layers)
        self.batchModeSwitch.toggled.connect(self._on_batch_mode_toggle)
        self.minSizeSpinBox.valueChanged.connect(self._on_batch_settings_changed)
        self.maxObjectsSpinBox.valueChanged.connect(self._on_batch_settings_changed)

        # SAM3 feature connections
        self.modelComboBox.currentIndexChanged.connect(self._on_model_changed)
        self.segmentTextBtn.clicked.connect(self._run_text_segmentation)
        self.textPromptInput.returnPressed.connect(self._run_text_segmentation)
        self.textPromptInput.textChanged.connect(
            lambda text: self.segmentTextBtn.setEnabled(bool(text.strip()))
        )
        self.similarModeBtn.clicked.connect(self._activate_similar_tool)
        self.similarModeBtn.toggled.connect(self._on_similar_mode_toggled)

        # GSD / Size / Shape filter connections
        self.autoGsdBtn.clicked.connect(self._auto_detect_gsd)
        self.gsdSpinBox.valueChanged.connect(self._on_gsd_changed)
        self.realWorldFilterSwitch.toggled.connect(self._on_real_world_filter_toggle)
        self.shapeFilterSwitch.toggled.connect(self._on_shape_filter_toggle)
        self.minDiameterSpinBox.valueChanged.connect(self._update_real_world_filter_hints)
        self.maxDiameterSpinBox.valueChanged.connect(self._update_real_world_filter_hints)
        self.minAreaSpinBox.valueChanged.connect(self._update_real_world_filter_hints)
        self.maxAreaSpinBox.valueChanged.connect(self._update_real_world_filter_hints)
        self.circularitySpinBox.valueChanged.connect(lambda: self._update_filter_indicator())
        self.compactnessSpinBox.valueChanged.connect(lambda: self._update_filter_indicator())
        self.aspectRatioSpinBox.valueChanged.connect(lambda: self._update_filter_indicator())

        # Refresh model options in case SAM3 was downloaded during init
        self._refresh_model_options()

    def _select_output_folder(self):
        folder = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Select Output Folder", str(self.export_save_dir.parent))

        if folder:
            self.export_save_dir = pathlib.Path(
                folder) / "GeoOSAM_output"
            self.mask_save_dir = pathlib.Path(folder) / "GeoOSAM_masks"
            self.export_save_dir.mkdir(exist_ok=True)
            if self.save_debug_masks:
                self.mask_save_dir.mkdir(exist_ok=True)

            short_path = "..." + str(self.export_save_dir)[-35:] if len(
                str(self.export_save_dir)) > 40 else str(self.export_save_dir)
            self.outputFolderLabel.setText(short_path)
            self._update_status(
                f"📁 Output folder: {self.export_save_dir}", "info")

    def _on_debug_toggle(self, checked):
        self.save_debug_masks = checked
        if checked:
            self.mask_save_dir.mkdir(exist_ok=True)
            self._update_status("💾 Debug masks will be saved", "info")
        else:
            self._update_status("🚫 Debug masks disabled", "info")

    def _clear_widget_focus(self):
        """Clear focus from all widgets and return it to map canvas"""
        # Give focus back to the map canvas so space bar works for map tools
        self.canvas.setFocus()
        QtWidgets.QApplication.processEvents()

    def _reset_batch_defaults(self):
        """Reset to generic batch defaults when no class is selected"""
        default_min_size = 50
        default_max_objects = 20

        self.minSizeSpinBox.setValue(default_min_size)
        self.maxObjectsSpinBox.setValue(default_max_objects)
        self.min_object_size = default_min_size
        self.max_objects = default_max_objects

    def _on_class_changed(self):
        selected_data = self.classComboBox.currentData()
        if selected_data:
            self.current_class = selected_data
            self._set_class_combo_alert(False)
            class_info = self.classes[selected_data]
            self.currentClassLabel.setText(f"Current: {selected_data}")

            color = class_info['color']
            try:
                r, g, b = [int(c.strip()) for c in color.split(',')]
                self.currentClassLabel.setStyleSheet(
                    f"font-weight: 600; padding: 12px; margin: 4px; "
                    f"border: 3px solid rgb({r},{g},{b}); "
                    f"background-color: rgba({r},{g},{b}, 30); "
                    f"color: rgb({max(0, r-50)},{max(0, g-50)},{max(0, b-50)}); "
                    f"border-radius: 8px; font-size: 16px;")
            except:
                self.currentClassLabel.setStyleSheet(
                    f"font-weight: 600; padding: 12px; border: 2px solid rgb({color}); "
                    f"background-color: rgba({color}, 50); font-size: 16px;")

            # NEW: Apply class-specific batch defaults
            self._apply_class_batch_defaults(class_info)

            self._activate_point_tool()
        else:
            self.current_class = None
            self.currentClassLabel.setText("No class selected")
            self.currentClassLabel.setStyleSheet("""
                font-weight: 600; padding: 12px; margin: 4px; 
                border: 2px solid #D0D5DD; 
                background-color: #F9FAFB; 
                color: #667085;
                border-radius: 8px; font-size: 16px;
            """)

            # Reset to default values when no class selected
            self._reset_batch_defaults()

        self._clear_widget_focus()

    def _apply_class_batch_defaults(self, class_info):
        """Apply recommended batch settings for the selected class"""
        if 'batch_defaults' in class_info:
            defaults = class_info['batch_defaults']

            # Update spinbox values
            self.minSizeSpinBox.setValue(defaults.get('min_size', 50))
            self.maxObjectsSpinBox.setValue(defaults.get('max_objects', 20))

            # Update internal settings
            self.min_object_size = defaults.get('min_size', 50)
            self.max_objects = defaults.get('max_objects', 20)
            self.max_object_size = defaults.get('max_size', 0)

            # Show helpful message about applied defaults
            class_name = self.current_class
            min_size = defaults.get('min_size', 50)
            max_objects = defaults.get('max_objects', 20)

            if self.batch_mode_enabled:
                self._update_status(
                    f"🎯 Applied {class_name} defaults: {min_size}px min, {max_objects} max objects", "info")

    def _add_new_class(self):
        class_name, ok = QtWidgets.QInputDialog.getText(
            self, 'Add Class', 'Enter class name:')
        if ok and class_name and class_name not in self.classes:
            used_colors = [info['color'] for info in self.classes.values()]
            available_colors = [
                c for c in self.EXTRA_COLORS if c not in used_colors]

            if available_colors:
                color = available_colors[0]
            else:
                import random
                color = f"{random.randint(100,255)},{random.randint(100,255)},{random.randint(100,255)}"

            description = f'Custom class: {class_name}'

            # NEW: Add default batch settings for new classes
            self.classes[class_name] = {
                'color': color, 
                'description': description,
                'batch_defaults': {'min_size': 50, 'max_objects': 20, 'max_size': 0}  # Generic defaults
            }

            self.classComboBox.addItem(class_name, class_name)
            self._update_status(
                f"Added class: {class_name} (RGB:{color}) with default batch settings", "info")

    def _edit_classes(self):
        class_list = list(self.classes.keys())
        if not class_list:
            self._update_status("No classes to edit", "warning")
            return

        class_name, ok = QtWidgets.QInputDialog.getItem(
            self, 'Edit Classes', 'Select class to edit:', class_list, 0, False)

        if ok and class_name:
            current_info = self.classes[class_name]
            new_name, ok2 = QtWidgets.QInputDialog.getText(
                self, 'Edit Class Name', f'Edit name for {class_name}:', text=class_name)

            if ok2 and new_name:
                current_color = current_info['color']
                new_color, ok3 = QtWidgets.QInputDialog.getText(
                    self, 'Edit Color', f'Edit color for {new_name} (R,G,B):', text=current_color)

                if ok3 and new_color:
                    try:
                        parts = [int(p.strip()) for p in new_color.split(',')]
                        if len(parts) == 3 and all(0 <= p <= 255 for p in parts):
                            if new_name != class_name:
                                del self.classes[class_name]

                            self.classes[new_name] = {
                                'color': new_color,
                                'description': current_info.get('description', f'Class: {new_name}')
                            }
                            self._refresh_class_combo()
                            self._update_status(
                                f"Updated {new_name} with RGB({new_color})", "info")
                        else:
                            self._update_status(
                                "Invalid color format! Use R,G,B (0-255)", "error")
                    except ValueError:
                        self._update_status(
                            "Invalid color format! Use R,G,B (0-255)", "error")

    def _on_batch_mode_toggle(self, checked):
        """Handle batch mode toggle with class-aware defaults"""
        self.batch_mode_enabled = checked
        self.batchSettingsFrame.setVisible(checked)

        if checked:
            self.bboxModeBtn.setText("BBox Batch Mode")

            # Apply current class defaults if a class is selected
            if self.current_class and self.current_class in self.classes:
                class_info = self.classes[self.current_class]
                self._apply_class_batch_defaults(class_info)

            self._update_status("🔄 Batch mode: Will find multiple objects in bbox", "info")
        else:
            self.bboxModeBtn.setText("BBox Mode") 
            self._update_status("📦 Single mode: Will segment entire bbox", "info")

        # Better layout handling
        QtWidgets.QApplication.processEvents()
        if hasattr(self, 'widget') and self.widget():
            self.widget().adjustSize()
            self.widget().updateGeometry()
        self.updateGeometry()
        self._clear_widget_focus()

    def _on_batch_settings_changed(self):
        """Update batch settings"""
        self.min_object_size = self.minSizeSpinBox.value()
        self.max_objects = self.maxObjectsSpinBox.value()
        self._clear_widget_focus()

    # --- GSD / Size / Shape filter methods ---

    def _auto_detect_gsd(self):
        """Read GSD from the active raster layer metadata"""
        import math
        current_layer = self.iface.activeLayer()
        if not isinstance(current_layer, QgsRasterLayer) or not current_layer.isValid():
            self._update_status("Select a raster layer first", "warning")
            return

        try:
            import rasterio
            with rasterio.open(current_layer.source()) as src:
                x_res, y_res = src.res
                avg_res = (abs(x_res) + abs(y_res)) / 2.0

                crs = src.crs
                if crs and crs.is_geographic:
                    center_lat = (src.bounds.top + src.bounds.bottom) / 2.0
                    meters_per_deg_lon = 111320 * math.cos(math.radians(center_lat))
                    meters_per_deg_lat = 111320
                    x_m = abs(x_res) * meters_per_deg_lon
                    y_m = abs(y_res) * meters_per_deg_lat
                    gsd_meters = (x_m + y_m) / 2.0
                else:
                    gsd_meters = avg_res

                gsd_cm = gsd_meters * 100.0
                self.gsd_cm_per_px = gsd_cm
                self.gsdSpinBox.setValue(gsd_cm)
                self.gsdSourceLabel.setText(
                    f"Auto: {gsd_cm:.2f} cm/px from {current_layer.name()[:25]}")
                self._update_status(f"GSD: {gsd_cm:.2f} cm/px detected", "info")
                self._update_real_world_filter_hints()

        except Exception as e:
            self._update_status(f"GSD auto-detect failed: {e}", "error")

    def _on_gsd_changed(self, value):
        """Handle manual GSD spinbox change"""
        self.gsd_cm_per_px = value if value > 0.01 else 0.0
        if self.gsd_cm_per_px > 0:
            self.gsdSourceLabel.setText(f"Manual: {value:.2f} cm/px")
            self.realWorldFilterSwitch.setEnabled(True)
            self._update_status(f"GSD: {value:.2f} cm/px", "info")
        else:
            self.gsdSourceLabel.setText("Not set")
            # Disable and turn off size filter when GSD is cleared
            if self.real_world_filter_enabled:
                self.realWorldFilterSwitch.setChecked(False)
            self.realWorldFilterSwitch.setEnabled(False)
            self._update_status("Ready to segment", "info")
        self._update_real_world_filter_hints()
        self._update_filter_indicator()

    def _on_real_world_filter_toggle(self, checked):
        """Show/hide real-world size filter settings"""
        self.real_world_filter_enabled = checked
        self.realWorldFilterFrame.setVisible(checked)
        self._update_filter_indicator()

    def _on_shape_filter_toggle(self, checked):
        """Show/hide shape filter settings"""
        self.shape_filter_enabled = checked
        self.shapeFilterFrame.setVisible(checked)
        self._update_filter_indicator()

    def _update_filter_indicator(self):
        """Update filter summary and refresh status label to show it"""
        parts = []

        if self.real_world_filter_enabled and self.gsd_cm_per_px > 0:
            size_parts = []
            if self.minDiameterSpinBox.value() > 0:
                size_parts.append(f"min {self.minDiameterSpinBox.value():.0f}cm")
            if self.maxDiameterSpinBox.value() > 0:
                size_parts.append(f"max {self.maxDiameterSpinBox.value():.0f}cm")
            if self.minAreaSpinBox.value() > 0:
                size_parts.append(f"min {self.minAreaSpinBox.value():.1f}m²")
            if self.maxAreaSpinBox.value() > 0:
                size_parts.append(f"max {self.maxAreaSpinBox.value():.1f}m²")
            if size_parts:
                parts.append("Size: " + ", ".join(size_parts))

        if self.shape_filter_enabled:
            shape_parts = []
            if self.circularitySpinBox.value() > 0:
                shape_parts.append(f"circ≥{self.circularitySpinBox.value():.1f}")
            if self.compactnessSpinBox.value() > 0:
                shape_parts.append(f"comp≥{self.compactnessSpinBox.value():.1f}")
            if self.aspectRatioSpinBox.value() < 10:
                shape_parts.append(f"AR≤{self.aspectRatioSpinBox.value():.1f}")
            if shape_parts:
                parts.append("Shape: " + ", ".join(shape_parts))

        self._active_filter_summary = " | ".join(parts) if parts else ""
        # Refresh status label to include/remove filter line
        self._update_status(self._last_status_text, self._last_status_type)

    def _update_real_world_filter_hints(self, _value=None):
        """Update pixel-equivalent hint labels next to each spinbox"""
        import math
        gsd = self.gsd_cm_per_px

        self.min_diameter_cm = self.minDiameterSpinBox.value()
        self.max_diameter_cm = self.maxDiameterSpinBox.value()
        self.min_area_m2 = self.minAreaSpinBox.value()
        self.max_area_m2 = self.maxAreaSpinBox.value()

        if gsd > 0:
            self.gsdRequiredLabel.setVisible(False)
        else:
            self.gsdRequiredLabel.setVisible(True)

        if gsd <= 0:
            for label in [self.minDiameterPixelLabel, self.maxDiameterPixelLabel,
                          self.minAreaPixelLabel, self.maxAreaPixelLabel]:
                label.setText("(set GSD)")
            return

        for spinbox, label in [(self.minDiameterSpinBox, self.minDiameterPixelLabel),
                               (self.maxDiameterSpinBox, self.maxDiameterPixelLabel)]:
            val = spinbox.value()
            if val > 0:
                label.setText(f"(~{val / gsd:.0f} px)")
            else:
                label.setText("(off)")

        gsd_m = gsd / 100.0
        px_area_m2 = gsd_m * gsd_m
        for spinbox, label in [(self.minAreaSpinBox, self.minAreaPixelLabel),
                               (self.maxAreaSpinBox, self.maxAreaPixelLabel)]:
            val = spinbox.value()
            if val > 0:
                label.setText(f"(~{val / px_area_m2:.0f} px\u00B2)")
            else:
                label.setText("(off)")

        self._update_filter_indicator()

    def _get_effective_pixel_filters(self):
        """Convert real-world size filters to pixel thresholds.
        Returns dict with min_area_px/max_area_px, or None if not active."""
        import math
        if not self.real_world_filter_enabled or self.gsd_cm_per_px <= 0:
            return None

        gsd = self.gsd_cm_per_px
        result = {}

        if self.min_diameter_cm > 0:
            d_px = self.min_diameter_cm / gsd
            result['min_area_px'] = math.pi * (d_px / 2) ** 2
        if self.max_diameter_cm > 0:
            d_px = self.max_diameter_cm / gsd
            result['max_area_px'] = math.pi * (d_px / 2) ** 2

        gsd_m = gsd / 100.0
        px_area_m2 = gsd_m * gsd_m

        if self.min_area_m2 > 0:
            min_from_area = self.min_area_m2 / px_area_m2
            result['min_area_px'] = max(result.get('min_area_px', 0), min_from_area)
        if self.max_area_m2 > 0:
            max_from_area = self.max_area_m2 / px_area_m2
            if 'max_area_px' in result:
                result['max_area_px'] = min(result['max_area_px'], max_from_area)
            else:
                result['max_area_px'] = max_from_area

        return result if result else None

    def _run_text_segmentation(self):
        """Run segmentation using text prompt (SAM3 only)"""
        print("✅ GeoOSAM v1.3.0 - Text segmentation method (FIXED)")
        if self.model_choice != "SAM3":
            QtWidgets.QMessageBox.warning(
                self, "Not Available",
                "Text prompts require SAM3 model.\n\n"
                "Please select SAM3 from the Model Selection dropdown."
            )
            return

        text_prompt = self.textPromptInput.text().strip()
        if not text_prompt:
            self._update_status("⚠️ Enter text prompt first", "warning")
            return

        if not self.current_class:
            self._update_status("⚠️ Select a class first", "warning")
            return

        # Check if raster layer exists
        current_layer = self.iface.activeLayer()
        if not isinstance(current_layer, QgsRasterLayer) or not current_layer.isValid():
            self._update_status("⚠️ No valid raster layer selected", "error")
            return

        # Get scope setting
        scope = self.scopeComboBox.currentData()  # 'aoi' or 'full'

        # Check license for full raster mode
        if scope == 'full':
            if not self._check_raster_access():
                return  # User denied or no license

        # Determine bbox based on scope
        bbox = None
        if scope == 'aoi':
            # Use visible extent as AOI
            bbox = self.canvas.extent()
        # else scope == 'full': bbox stays None, will trigger auto-tiling

        # Create request with text prompt and scope
        request = {
            'type': 'text',
            'text': text_prompt,
            'point': None,
            'bbox': bbox,
            'scope': scope,  # 'aoi' or 'full'
            'class': self.current_class,
            'timestamp': datetime.datetime.now()
        }

        self._add_to_queue(request)
        # Auto-Segment is not a map mode — deactivate any active click tool
        self.mode_button_group.setExclusive(False)
        for btn in (self.pointModeBtn, self.bboxModeBtn, self.similarModeBtn):
            btn.setChecked(False)
        self.mode_button_group.setExclusive(True)
        self.iface.actionPan().trigger()
        self._clear_widget_focus()

    def _on_model_changed(self, index):
        """Handle model selection change"""
        new_model = self.modelComboBox.currentData()

        if new_model == self.model_choice:
            return

        # Confirm change (requires reload)
        reply = QtWidgets.QMessageBox.question(
            self,
            "Change Model?",
            f"Switch from {self.model_choice} to {new_model}?\n\n"
            "This will reload the model.\n"
            "Your existing layers and features will be preserved.",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No
        )

        if reply == QtWidgets.QMessageBox.Yes:
            self.model_choice = new_model
            self._reload_model()
            self._update_ui_for_model()
        else:
            # Revert dropdown
            current_idx = self.modelComboBox.findData(self.model_choice)
            if current_idx >= 0:
                self.modelComboBox.setCurrentIndex(current_idx)

    def _reload_model(self):
        """Reload SAM model"""
        try:
            self._update_status("Reloading model...", "processing")
            self.predictor = None
            self._init_sam_model()
            self._update_status(f"✅ {self.model_choice} loaded successfully", "success")
        except Exception as e:
            self._update_status(f"❌ Model load failed: {e}", "error")
            QtWidgets.QMessageBox.critical(
                self, "Error",
                f"Failed to load {self.model_choice}:\n{e}"
            )

    def _update_ui_for_model(self):
        """Update UI based on selected model capabilities"""
        # Check if current model is SAM3 (model_choice is now "SAM3" or "SAM2_tiny", etc.)
        is_sam3 = self.model_choice == "SAM3"

        # Show/hide text prompt card
        self.textPromptCard.setVisible(is_sam3)

        # Show/hide license card
        if self.licenseCard:
            self.licenseCard.setVisible(is_sam3)
            if is_sam3:
                self._update_license_status()

        # Update similar mode button visibility
        self.similarModeBtn.setVisible(is_sam3)
        if not is_sam3 and self.current_mode == "similar":
            # Revert to point mode if similar mode was active
            self.pointModeBtn.setChecked(True)
            self.current_mode = "point"

        # Update device label
        device_icon = "🎮" if "cuda" in self.device else "🖥️"
        device_info = f"{device_icon} {self.device.upper()} | {self.model_choice}"
        if getattr(self, "num_cores", None):
            device_info += f" ({self.num_cores} cores)"

        if hasattr(self, "deviceLabel") and self.deviceLabel:
            self.deviceLabel.setText(device_info)

    def _refresh_class_combo(self):
        current_class = self.current_class
        self.classComboBox.clear()
        self.classComboBox.addItem("-- Select Class --", None)

        for class_name, class_info in self.classes.items():
            self.classComboBox.addItem(class_name, class_name)

        if current_class and current_class in self.classes:
            index = self.classComboBox.findData(current_class)
            if index >= 0:
                self.classComboBox.setCurrentIndex(index)

    def _detect_tile_layer_type(self, layer):
        """Detect if layer is a tile service (XYZ, WMS, WMTS) and return type"""
        if not isinstance(layer, QgsRasterLayer):
            return None

        try:
            provider_type = layer.providerType()
            data_source = layer.dataProvider().dataSourceUri()

            if provider_type == "wms":
                # All tile services use "wms" provider in QGIS
                data_source_lower = data_source.lower()
                data_source_upper = data_source.upper()

                if "type=xyz" in data_source_lower:
                    return "XYZ"
                elif "service=WMS" in data_source_upper:
                    return "WMS" 
                elif "service=WMTS" in data_source_upper or "wmts" in data_source_lower:
                    return "WMTS"
                elif "tilematrixset" in data_source_lower:
                    return "WMTS"
                else:
                    # Generic tile service
                    return "TILE"

            return None  # Not a tile service

        except Exception as e:
            print(f"Error detecting tile layer type: {e}")
            return None

    def _cache_tile_layer_as_raster(self, layer):
        """Cache tile layer by rendering current canvas view to a temporary GeoTIFF"""
        try:
            # Get current canvas
            canvas = self.iface.mapCanvas()

            # Create a temporary raster by rendering just this layer
            from qgis.core import QgsProject

            # Use map renderer instead of canvas manipulation to avoid flickering
            from qgis.core import QgsMapRendererParallelJob, QgsMapSettings

            # Create map settings for just this layer
            settings = QgsMapSettings()
            settings.setLayers([layer])
            settings.setExtent(canvas.extent())
            settings.setDestinationCrs(canvas.mapSettings().destinationCrs())
            settings.setOutputSize(canvas.size())
            settings.setBackgroundColor(QtGui.QColor(255, 255, 255, 0))

            # Render to image without touching canvas
            job = QgsMapRendererParallelJob(settings)
            job.start()
            job.waitForFinished()

            if job.errors():
                raise Exception(f"Render errors: {'; '.join(job.errors())}")

            # Get rendered image and save temporarily
            image = job.renderedImage()
            temp_file = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
            temp_img_path = temp_file.name
            temp_file.close()

            image.save(temp_img_path)

            # Convert to GeoTIFF with proper georeferencing
            temp_tif = tempfile.NamedTemporaryFile(suffix='.tif', delete=False)
            temp_tif_path = temp_tif.name
            temp_tif.close()

            # Get canvas extent and CRS
            extent = canvas.extent()
            crs = canvas.mapSettings().destinationCrs()

            # Open the PNG and convert to GeoTIFF
            from PIL import Image
            import rasterio
            from rasterio.transform import from_bounds

            with Image.open(temp_img_path) as img:
                img_array = np.array(img)
                if len(img_array.shape) == 3:
                    # Convert to rasterio format (bands, height, width)
                    img_array = np.transpose(img_array, (2, 0, 1))

                    # Handle RGBA vs RGB
                    if img_array.shape[0] == 4:
                        # Drop alpha channel, keep only RGB
                        img_array = img_array[:3, :, :]
                        band_count = 3
                    else:
                        band_count = img_array.shape[0]
                else:
                    band_count = 1

                height, width = img.size[1], img.size[0]

                # Create transform
                transform = from_bounds(
                    extent.xMinimum(), extent.yMinimum(),
                    extent.xMaximum(), extent.yMaximum(),
                    width, height
                )

                # Write GeoTIFF
                with rasterio.open(
                    temp_tif_path, 'w',
                    driver='GTiff',
                    height=height,
                    width=width,
                    count=band_count,
                    dtype=img_array.dtype,
                    crs=crs.toWkt(),
                    transform=transform
                ) as dst:
                    if len(img_array.shape) == 3:
                        dst.write(img_array)
                    else:
                        dst.write(img_array, 1)

            # No need to restore visibility since we didn't change it

            # Clean up PNG
            try:
                os.unlink(temp_img_path)
            except:
                pass

            print(f"✅ Successfully cached tiles to: {temp_tif_path}")
            return temp_tif_path

        except Exception as e:
            print(f"❌ Tile caching error: {e}")
            import traceback
            traceback.print_exc()

            # No visibility cleanup needed since we didn't change it
            return None

    def _validate_class_selection(self):
        """Enhanced validation that properly handles layer switching"""
        if not self.current_class:
            self._update_status("Please select a class first!", "warning")
            self._set_class_combo_alert(True)
            return False

        current_layer = self.iface.activeLayer()
        if not isinstance(current_layer, QgsRasterLayer) or not current_layer.isValid():
            self._update_status(
                "Please select a valid raster layer first!", "warning")
            return False

        # Check if this is a tile layer and show appropriate message
        tile_type = self._detect_tile_layer_type(current_layer)
        if tile_type:
            self._update_status(f"🌐 {tile_type} tile layer detected - will cache tiles for processing", "info")

        # ALWAYS update the raster layer reference when validating
        self.original_raster_layer = current_layer

        # Clear any existing feedback when switching layers
        if hasattr(self, 'pointTool'):
            self.pointTool.clear_feedback()
        if hasattr(self, 'bboxTool'):
            self.bboxTool.clear_feedback()

        return True

    def _activate_point_tool(self):
        if not self._validate_class_selection():
            return

        self.current_mode = 'point'
        self.original_map_tool = self.canvas.mapTool()

        # Disable batch mode for point mode (doesn't make sense for single points)
        if self.batch_mode_enabled:
            self.batchModeSwitch.setChecked(False)
            self.batch_mode_enabled = False
            self.batchSettingsFrame.setVisible(False)

        # Disable batch mode switch in point mode
        self.batchModeSwitch.setEnabled(False)

        self.pointModeBtn.setChecked(True)
        self._update_status(
            f"Point mode active for [{self.current_class}]. Click on map to segment.", "processing")
        self.canvas.setMapTool(self.pointTool)
        self._clear_widget_focus()

    def _activate_bbox_tool(self):
        if not self._validate_class_selection():
            return

        self.current_mode = 'bbox'
        self.original_map_tool = self.canvas.mapTool()

        # Re-enable batch mode switch for bbox mode
        self.batchModeSwitch.setEnabled(True)

        self.bboxModeBtn.setChecked(True)
        self._update_status(
            f"BBox mode active for [{self.current_class}]. Click and drag to segment.", "processing")
        self.canvas.setMapTool(self.bboxTool)
        self._clear_widget_focus()

    def _activate_similar_tool(self):
        """Activate similar object detection mode (SAM3 exemplar)"""
        if self.model_choice != "SAM3":
            QtWidgets.QMessageBox.warning(
                self, "Not Available",
                "Similar Objects mode requires SAM3 model.\n\n"
                "Please select SAM3 from the Model Selection dropdown."
            )
            # Revert to point mode
            self.pointModeBtn.setChecked(True)
            return

        if not self._validate_class_selection():
            self.pointModeBtn.setChecked(True)
            return

        self.current_mode = "similar"
        self.original_map_tool = self.canvas.mapTool()

        # Disable batch mode for similar mode
        self.batchModeSwitch.setEnabled(False)

        self.similarModeBtn.setChecked(True)
        self._update_status(
            f"Similar Objects mode: Click on reference object for [{self.current_class}]",
            "processing"
        )
        # Use point tool to select reference object
        self.canvas.setMapTool(self.pointTool)
        self._clear_widget_focus()

    def _on_map_tool_changed(self, new_tool, old_tool):
        """Detect when user switches away from plugin tools to external QGIS tools"""
        # Check if the new tool is NOT one of our plugin tools
        if new_tool != self.pointTool and new_tool != self.bboxTool:
            # User switched to a different tool (e.g., hand/pan tool)
            # Uncheck all mode buttons to reflect that our tools are inactive
            # Uncheck all buttons when user switches to an external QGIS tool
            self.mode_button_group.setExclusive(False)
            self.segmentTextBtn.setChecked(False)
            self.pointModeBtn.setChecked(False)
            self.bboxModeBtn.setChecked(False)
            self.similarModeBtn.setChecked(False)
            self.mode_button_group.setExclusive(True)

    def _point_done(self, pt, multi_points=None):
        # Handle similar mode differently - convert point to exemplar bbox
        if self.current_mode == "similar":
            bbox = self._point_to_exemplar_bbox(pt)
            # Get scope setting for similar mode
            scope = self.scopeComboBox.currentData() if hasattr(self, 'scopeComboBox') else 'aoi'

            # Check license for full raster mode
            if scope == 'full':
                if not self._check_raster_access():
                    return  # User denied or no license

            request = {
                'type': 'similar',
                'point': pt,
                'bbox': bbox,  # Used as exemplar
                'scope': scope,  # 'aoi' or 'full'
                'class': self.current_class,
                'timestamp': datetime.datetime.now()
            }
        else:
            # Regular point mode
            request = {
                'type': 'point',
                'point': pt,
                'bbox': None,
                'multi_points': multi_points,  # None for single, list of (QgsPointXY, label) for multi
                'class': self.current_class,
                'timestamp': datetime.datetime.now()
            }
        self._add_to_queue(request)

    def _point_to_exemplar_bbox(self, pt):
        """Convert point click to exemplar bounding box for similar mode"""
        # Create small bbox around point for exemplar
        # Size varies by class - larger for buildings, smaller for vehicles
        if self.current_class in ['Buildings', 'Residential']:
            pixel_radius = 50  # 100x100 px bbox for large objects
        elif self.current_class in ['Vehicle', 'Vessels']:
            pixel_radius = 25  # 50x50 px bbox for small objects
        else:
            pixel_radius = 35  # Default 70x70 px

        # Convert map point to pixel coordinates
        transform = self.canvas.getCoordinateTransform()
        pixel_pt = transform.transform(pt)

        # Create bbox in pixel space
        x1_px = int(pixel_pt.x() - pixel_radius)
        y1_px = int(pixel_pt.y() - pixel_radius)
        x2_px = int(pixel_pt.x() + pixel_radius)
        y2_px = int(pixel_pt.y() + pixel_radius)

        # Convert back to map coordinates
        pt1 = transform.toMapCoordinates(x1_px, y1_px)
        pt2 = transform.toMapCoordinates(x2_px, y2_px)

        from qgis.core import QgsRectangle
        return QgsRectangle(pt1, pt2)

    def _bbox_done(self, rect):
        # Add to queue instead of blocking
        request = {
            'type': 'bbox',
            'point': None,
            'bbox': rect,
            'class': self.current_class,
            'timestamp': datetime.datetime.now()
        }
        self._add_to_queue(request)

    def _add_to_queue(self, request):
        """Add a request to the processing queue"""
        self.processing_queue.append(request)
        queue_position = len(self.processing_queue)

        if request['type'] == 'point':
            pt = request['point']
            self._update_status(
                f"🔄 Queued point ({pt.x():.1f}, {pt.y():.1f}) for [{request['class']}] - Position {queue_position}",
                "info")
        elif request['type'] == 'text':
            text = request.get('text', 'unknown')
            self._update_status(
                f"🔄 Queued text prompt '{text}' for [{request['class']}] - Position {queue_position}",
                "info")
        elif request['type'] == 'similar':
            pt = request['point']
            self._update_status(
                f"🔄 Queued similar objects at ({pt.x():.1f}, {pt.y():.1f}) for [{request['class']}] - Position {queue_position}",
                "info")
        else:  # bbox
            rect = request['bbox']
            self._update_status(
                f"🔄 Queued bbox ({rect.width():.1f}×{rect.height():.1f}) for [{request['class']}] - Position {queue_position}",
                "info")

        # Start processing if not already running
        self._process_queue()

    def _process_queue(self):
        """Process the next item in the queue"""
        if self.is_processing or not self.processing_queue:
            return

        # Get next request
        request = self.processing_queue.pop(0)
        remaining = len(self.processing_queue)

        # Set current request data
        self.point = request['point']
        self.bbox = request['bbox']
        self.current_class = request['class']
        self.text_prompt = request.get('text', None)  # Extract text prompt if present
        self.request_type = request['type']  # Store request type for mode determination
        self.request_scope = request.get('scope', 'aoi')  # Extract scope (aoi or full)
        self.current_request = request  # Store full request for modifier access in result callback

        # Update status with queue info
        if request['type'] == 'point':
            pt = request['point']
            status_msg = f"Processing point ({pt.x():.1f}, {pt.y():.1f}) for [{request['class']}]"
        elif request['type'] == 'text':
            text = request.get('text', 'unknown')
            status_msg = f"Processing text prompt '{text}' for [{request['class']}]"
        elif request['type'] == 'similar':
            pt = request['point']
            status_msg = f"Processing similar objects at ({pt.x():.1f}, {pt.y():.1f}) for [{request['class']}]"
        else:  # bbox
            rect = request['bbox']
            status_msg = f"Processing bbox ({rect.width():.1f}×{rect.height():.1f}) for [{request['class']}]"

        if remaining > 0:
            status_msg += f" - {remaining} more in queue"

        self._update_status(status_msg, "processing")

        # Start segmentation
        self._run_segmentation()

    def _run_segmentation(self):
        """Enhanced segmentation that ensures current layer is used"""
        # Prevent multiple simultaneous requests
        if self.is_processing:
            self._update_status("Processing already in progress, please wait...", "warning")
            return

        # Cancel any existing worker
        if self.worker and self.worker.isRunning():
            self._cancel_segmentation_safely()

        # Set processing state
        self.is_processing = True

        # DEBUG: Verify settings are correct
        if self.batch_mode_enabled and self.current_mode == 'bbox':
            self._debug_current_settings()

        if not self.current_class:
            self._update_status("No class selected", "error")
            self.is_processing = False
            return

        # Get the CURRENT active layer (not stored reference)
        current_layer = self.iface.activeLayer()
        if not isinstance(current_layer, QgsRasterLayer) or not current_layer.isValid():
            self._update_status("Please select a valid raster layer", "error")
            self.is_processing = False
            return

        # Update stored reference to current layer
        self.original_raster_layer = current_layer

        # Validation: need point, bbox, or text prompt
        has_prompt = (self.point is not None or self.bbox is not None or
                     (hasattr(self, 'text_prompt') and self.text_prompt))
        if not has_prompt:
            self._update_status("No selection or text prompt found", "error")
            self.is_processing = False
            return

        # Check if this is a full raster request for text/similar modes
        if hasattr(self, 'request_type') and self.request_type in ['text', 'similar']:
            scope = getattr(self, 'request_scope', 'aoi')
            if scope == 'full':
                # Use semantic predictor for full raster (text/similar modes)
                self._run_tiled_segmentation(current_layer)
                return

        import time
        start_time = time.time()

        self._set_ui_enabled(False)

        # Update status based on mode
        if self.current_mode == 'bbox' and self.batch_mode_enabled:
            self._update_status(f"🔄 Batch processing on layer: {current_layer.name()[:30]}...", "processing")
        else:
            self._update_status(f"🚀 Processing on layer: {current_layer.name()[:30]}...", "processing")

        try:
            # Use the current_layer (not self.original_raster_layer)
            result = self._prepare_optimized_segmentation_data(current_layer)
            if result is None:
                self._set_ui_enabled(True)
                return

            # Handle both RGB and multi-spectral data
            if len(result) == 7:
                arr, mask_transform, debug_info, input_coords, input_labels, input_box, arr_multispectral = result
            else:
                arr, mask_transform, debug_info, input_coords, input_labels, input_box = result
                arr_multispectral = None
            prep_time = time.time() - start_time

            # Add layer info to debug
            debug_info['source_layer'] = current_layer.name()
            debug_info['layer_crs'] = current_layer.crs().authid()
            debug_info['batch_mode'] = self.batch_mode_enabled and self.current_mode == 'bbox'

        except Exception as e:
            self._update_status(f"Error preparing data from {current_layer.name()}: {e}", "error")
            self._set_ui_enabled(True)
            return

        # Determine processing mode based on request type
        if hasattr(self, 'request_type'):
            # Use stored request type from queue
            if self.request_type == 'text':
                mode = "text"
            elif self.request_type == 'similar':
                mode = "similar"
            elif self.request_type == 'bbox' and self.batch_mode_enabled:
                mode = "bbox_batch"
            elif self.request_type == 'bbox':
                mode = "bbox"
            else:  # point
                mode = "point"
        else:
            # Fallback for backward compatibility (direct calls, not from queue)
            mode = "point" if self.point is not None else "bbox"
            if mode == "bbox" and self.batch_mode_enabled:
                mode = "bbox_batch"

        # Continue with worker thread...
        # Compute effective size filters (real-world overrides pixel-based)
        real_world_filters = self._get_effective_pixel_filters()
        effective_min_size = self.min_object_size
        effective_max_size = None
        if real_world_filters:
            if 'min_area_px' in real_world_filters:
                effective_min_size = max(effective_min_size, int(real_world_filters['min_area_px']))
            if 'max_area_px' in real_world_filters:
                effective_max_size = int(real_world_filters['max_area_px'])
        # Apply class-level max_size if no real-world override
        if effective_max_size is None and self.max_object_size > 0:
            effective_max_size = self.max_object_size

        self.worker = OptimizedSAM2Worker(
            predictor=self.predictor,
            arr=arr,
            mode=mode,
            model_choice=self.model_choice,
            point_coords=input_coords,
            point_labels=input_labels,
            box=input_box,
            mask_transform=mask_transform,
            debug_info={**debug_info, 'prep_time': prep_time},
            device=self.device,
            # Pass batch settings
            min_object_size=effective_min_size,
            arr_multispectral=arr_multispectral,
            max_objects=self.max_objects,
            text_prompt=getattr(self, 'text_prompt', None),
            max_object_size_px=effective_max_size,
            shape_filters={
                'enabled': self.shape_filter_enabled,
                'min_circularity': self.circularitySpinBox.value(),
                'min_compactness': self.compactnessSpinBox.value(),
                'max_aspect_ratio': self.aspectRatioSpinBox.value(),
            } if self.shape_filter_enabled else None
        )

        self.worker.finished.connect(self._on_segmentation_finished)
        self.worker.error.connect(self._on_segmentation_error)
        self.worker.progress.connect(self._on_segmentation_progress)
        self.worker.start()

    def _run_semantic_text_full_raster(self, raster_layer):
        """
        Use SAM3 semantic predictor to find objects matching text prompt.
        For large rasters, uses tiled processing to avoid GPU memory issues.
        Handles both AOI (extent) and full raster modes.
        """
        import rasterio
        import numpy as np
        from rasterio.windows import from_bounds

        # Check if we're processing AOI or full raster
        scope = getattr(self, 'request_scope', 'aoi')

        self._set_ui_enabled(False)
        if scope == 'full':
            self._update_status(f"🔍 Finding '{self.text_prompt}' across entire raster: {raster_layer.name()[:30]}...", "processing")
            print("\n" + "="*80)
            print("🔍 SEMANTIC TEXT MODE - FULL RASTER")
        else:
            self._update_status(f"🔍 Finding '{self.text_prompt}' in current extent: {raster_layer.name()[:30]}...", "processing")
            print("\n" + "="*80)
            print("🔍 SEMANTIC TEXT MODE - CURRENT EXTENT (AOI)")

        print("="*80)
        print(f"Raster: {raster_layer.source()}")
        print(f"Scope: {scope}")
        print(f"Text prompt: '{self.text_prompt}'")
        print("="*80)

        try:
            with rasterio.open(raster_layer.source()) as src:
                # Determine processing extent
                if scope == 'full':
                    # Use full raster bounds
                    extent = src.bounds
                    xmin, ymin, xmax, ymax = extent.left, extent.bottom, extent.right, extent.top
                    window = None  # Process entire raster
                    print(f"📏 Processing full raster")
                else:
                    # Use map extent for AOI
                    extent = self.canvas.extent()
                    xmin, ymin, xmax, ymax = (
                        extent.xMinimum(), extent.yMinimum(),
                        extent.xMaximum(), extent.yMaximum()
                    )
                    # Calculate window for this extent
                    window = from_bounds(xmin, ymin, xmax, ymax, src.transform)
                    print(f"📏 Processing extent: ({xmin:.2f}, {ymin:.2f}) to ({xmax:.2f}, {ymax:.2f})")

                # Get dimensions
                if window:
                    height, width = int(window.height), int(window.width)
                else:
                    height, width = src.height, src.width

                print(f"📏 Processing size: {width}x{height}")

                # Check if image is too large for single-pass processing
                # SAM3 semantic predictor uses a lot of GPU memory
                max_size_semantic = 2048  # Conservative limit for semantic predictor
                if max(width, height) > max_size_semantic:
                    print(f"⚠️  Image too large for single-pass semantic processing")
                    print(f"   Switching to tiled processing...")
                    # Close rasterio connection before switching to tiled mode
                    pass

            # Use tiled processing for large rasters
            if max(width, height) > max_size_semantic:
                self.is_processing = False
                self._set_ui_enabled(True)
                self._run_tiled_text_segmentation(raster_layer)
                return

            # For smaller images, continue with semantic predictor
            with rasterio.open(raster_layer.source()) as src:

                # Determine if we need to downsample
                max_size = 4096  # Maximum dimension for processing
                if max(width, height) > max_size:
                    scale = max_size / max(width, height)
                    out_width = int(width * scale)
                    out_height = int(height * scale)
                    print(f"⚠️  Large area - downsampling to {out_width}x{out_height}")
                else:
                    out_width, out_height = width, height
                    scale = 1.0

                # Read raster data
                print("📖 Reading raster data...")
                if src.count >= 3:
                    bands = [1, 2, 3]
                elif src.count == 2:
                    bands = [1, 1, 2]
                else:
                    bands = [1, 1, 1]

                # Read with or without window
                if window:
                    if scale < 1.0:
                        arr = src.read(bands, window=window,
                                     out_shape=(len(bands), out_height, out_width),
                                     out_dtype=np.uint8)
                    else:
                        arr = src.read(bands, window=window, out_dtype=np.uint8)
                    # Get transform for this window
                    mask_transform = src.window_transform(window)
                else:
                    if scale < 1.0:
                        arr = src.read(bands, out_shape=(len(bands), out_height, out_width),
                                     out_dtype=np.uint8)
                    else:
                        arr = src.read(bands, out_dtype=np.uint8)
                    mask_transform = src.transform

                arr = np.moveaxis(arr, 0, -1)
                print(f"✅ Raster loaded: {arr.shape}")

                # Adjust transform for downsampling if needed
                if scale < 1.0:
                    from rasterio import Affine
                    mask_transform = mask_transform * Affine.scale(1/scale)

                # Set image in predictor
                print("🧠 Setting image in SAM3 predictor...")
                self.predictor.set_image(arr)

                # Use semantic predictor with text prompt
                print(f"🔍 Running SAM3 semantic predictor with text: '{self.text_prompt}'...")
                self._update_status(f"🔍 SAM3 finding '{self.text_prompt}'...", "processing")

                masks, scores, logits = self.predictor.predict(
                    text=self.text_prompt,
                    multimask_output=False
                )

                print(f"✅ Found {len(masks) if masks else 0} objects matching '{self.text_prompt}'")

                if masks and len(masks) > 0:
                    # Process results - collect all features first
                    all_features = []
                    total_masks = len(masks)
                    print(f"🔄 Processing {total_masks} masks...")
                    self.progressBar.setRange(0, total_masks)
                    self.progressBar.setVisible(True)

                    from qgis.PyQt.QtWidgets import QApplication
                    for i, mask in enumerate(masks):
                        self._update_status(f"🔄 Processing mask {i+1}/{total_masks}...", "processing")
                        self.progressBar.setValue(i + 1)
                        QApplication.processEvents()

                        # Convert mask to features
                        features = self._convert_mask_to_features(mask, mask_transform)

                        if features:
                            all_features.extend(features)

                    self.progressBar.setVisible(False)

                    # Add all features to layer with undo tracking
                    if all_features:
                        debug_info = {'mode': 'TEXT_EXTENT'}
                        self._add_features_to_layer(all_features, debug_info, len(all_features))
                        self._update_status(f"✅ Found {len(all_features)} objects matching '{self.text_prompt}'", "success")
                        print(f"\n{'='*80}")
                        print(f"✅ COMPLETED: {len(all_features)} objects found matching '{self.text_prompt}'")
                        print(f"{'='*80}\n")
                    else:
                        self._update_status(f"⚠️  No features created from masks", "warning")
                else:
                    self._update_status(f"⚠️  No objects found matching '{self.text_prompt}'", "warning")
                    print(f"⚠️  No objects detected matching text prompt")

        except Exception as e:
            error_msg = f"❌ Semantic text error: {e}"
            print(error_msg)
            import traceback
            traceback.print_exc()
            self._update_status(error_msg, "error")
        finally:
            self.is_processing = False
            self._set_ui_enabled(True)

    def _run_semantic_similar_full_raster(self, raster_layer):
        """
        Use SAM3 semantic predictor to find similar objects.
        For large rasters, uses tiled processing to avoid GPU memory issues.
        Handles both AOI (extent) and full raster modes.
        """
        import rasterio
        import numpy as np
        from rasterio.windows import from_bounds

        # Check if we're processing AOI or full raster
        scope = getattr(self, 'request_scope', 'aoi')

        self._set_ui_enabled(False)
        if scope == 'full':
            self._update_status(f"🎯 Finding similar objects across entire raster: {raster_layer.name()[:30]}...", "processing")
            print("\n" + "="*80)
            print("🎯 SEMANTIC SIMILAR MODE - FULL RASTER")
        else:
            self._update_status(f"🎯 Finding similar objects in current extent: {raster_layer.name()[:30]}...", "processing")
            print("\n" + "="*80)
            print("🎯 SEMANTIC SIMILAR MODE - CURRENT EXTENT (AOI)")

        print("="*80)
        print(f"Raster: {raster_layer.source()}")
        print(f"Scope: {scope}")
        print(f"Exemplar bbox: {self.bbox}")
        print("="*80)

        try:
            with rasterio.open(raster_layer.source()) as src:
                # Determine processing extent
                if scope == 'full':
                    # Use full raster bounds
                    extent = src.bounds
                    xmin, ymin, xmax, ymax = extent.left, extent.bottom, extent.right, extent.top
                    window = None  # Process entire raster
                    print(f"📏 Processing full raster")
                else:
                    # Use map extent for AOI
                    extent = self.canvas.extent()
                    xmin, ymin, xmax, ymax = (
                        extent.xMinimum(), extent.yMinimum(),
                        extent.xMaximum(), extent.yMaximum()
                    )
                    # Calculate window for this extent
                    window = from_bounds(xmin, ymin, xmax, ymax, src.transform)
                    print(f"📏 Processing extent: ({xmin:.2f}, {ymin:.2f}) to ({xmax:.2f}, {ymax:.2f})")

                # Get dimensions
                if window:
                    height, width = int(window.height), int(window.width)
                else:
                    height, width = src.height, src.width

                print(f"📏 Processing size: {width}x{height}")

                # Check if image is too large for single-pass processing
                # SAM3 semantic predictor uses a lot of GPU memory
                max_size_semantic = 2048  # Conservative limit for semantic predictor
                if max(width, height) > max_size_semantic:
                    print(f"⚠️  Image too large for single-pass semantic processing")
                    print(f"   Switching to tiled processing...")
                    # Close rasterio connection before switching to tiled mode
                    pass

            # Use tiled processing for large rasters
            if max(width, height) > max_size_semantic:
                self.is_processing = False
                self._set_ui_enabled(True)
                self._run_tiled_similar_segmentation(raster_layer)
                return

            # For smaller images, continue with semantic predictor
            with rasterio.open(raster_layer.source()) as src:

                # Determine if we need to downsample
                max_size = 4096  # Maximum dimension for processing
                if max(width, height) > max_size:
                    scale = max_size / max(width, height)
                    out_width = int(width * scale)
                    out_height = int(height * scale)
                    print(f"⚠️  Large area - downsampling to {out_width}x{out_height}")
                else:
                    out_width, out_height = width, height
                    scale = 1.0

                # Read raster data
                print("📖 Reading raster data...")
                if src.count >= 3:
                    bands = [1, 2, 3]
                elif src.count == 2:
                    bands = [1, 1, 2]
                else:
                    bands = [1, 1, 1]

                # Read with or without window
                if window:
                    if scale < 1.0:
                        arr = src.read(bands, window=window,
                                     out_shape=(len(bands), out_height, out_width),
                                     out_dtype=np.uint8)
                    else:
                        arr = src.read(bands, window=window, out_dtype=np.uint8)
                    # Get transform for this window
                    mask_transform = src.window_transform(window)
                else:
                    if scale < 1.0:
                        arr = src.read(bands, out_shape=(len(bands), out_height, out_width),
                                     out_dtype=np.uint8)
                    else:
                        arr = src.read(bands, out_dtype=np.uint8)
                    mask_transform = src.transform

                arr = np.moveaxis(arr, 0, -1)
                print(f"✅ Raster loaded: {arr.shape}")

                # Adjust transform for downsampling if needed
                if scale < 1.0:
                    from rasterio import Affine
                    mask_transform = mask_transform * Affine.scale(1/scale)

                # Convert exemplar bbox from geo to pixel coordinates
                if self.bbox is not None:
                    corners = [
                        (self.bbox.xMinimum(), self.bbox.yMinimum()),
                        (self.bbox.xMaximum(), self.bbox.yMinimum()),
                        (self.bbox.xMaximum(), self.bbox.yMaximum()),
                        (self.bbox.xMinimum(), self.bbox.yMaximum())
                    ]

                    pixel_coords = []
                    for x, y in corners:
                        px, py = ~mask_transform * (x, y)
                        pixel_coords.append((px, py))

                    xs, ys = zip(*pixel_coords)
                    x1, x2 = min(xs), max(xs)
                    y1, y2 = min(ys), max(ys)

                    # Clamp to image bounds
                    x1 = max(0, min(arr.shape[1] - 1, int(x1)))
                    y1 = max(0, min(arr.shape[0] - 1, int(y1)))
                    x2 = max(0, min(arr.shape[1] - 1, int(x2)))
                    y2 = max(0, min(arr.shape[0] - 1, int(y2)))

                    pixel_bbox = [x1, y1, x2, y2]
                    print(f"🎯 Exemplar bbox (pixels): {pixel_bbox}")

                    # Set image in predictor
                    print("🧠 Setting image in SAM3 predictor...")
                    self.predictor.set_image(arr)

                    # Use semantic predictor with exemplar mode
                    print("🔍 Running SAM3 semantic predictor (finding similar objects)...")
                    self._update_status("🔍 SAM3 finding similar objects...", "processing")

                    masks, scores, logits = self.predictor.predict(
                        box=[pixel_bbox],
                        exemplar_mode=True,
                        multimask_output=False
                    )

                    print(f"✅ Found {len(masks) if masks else 0} similar objects")

                    if masks and len(masks) > 0:
                        # Process results - collect all features first
                        all_features = []
                        total_masks = len(masks)
                        print(f"🔄 Processing {total_masks} masks...")
                        self.progressBar.setRange(0, total_masks)
                        self.progressBar.setVisible(True)

                        from qgis.PyQt.QtWidgets import QApplication
                        for i, mask in enumerate(masks):
                            self._update_status(f"🔄 Processing mask {i+1}/{total_masks}...", "processing")
                            self.progressBar.setValue(i + 1)
                            QApplication.processEvents()

                            # Convert mask to features
                            features = self._convert_mask_to_features(mask, mask_transform)

                            if features:
                                all_features.extend(features)

                        self.progressBar.setVisible(False)

                        # Add all features to layer with undo tracking
                        if all_features:
                            debug_info = {'mode': 'SIMILAR_EXTENT'}
                            self._add_features_to_layer(all_features, debug_info, len(all_features))
                            self._update_status(f"✅ Found {len(all_features)} similar objects", "success")
                            print(f"\n{'='*80}")
                            print(f"✅ COMPLETED: {len(all_features)} similar objects found")
                            print(f"{'='*80}\n")
                        else:
                            self._update_status("⚠️  No features created from masks", "warning")
                    else:
                        self._update_status("⚠️  No similar objects found", "warning")
                        print("⚠️  No similar objects detected")
                else:
                    self._update_status("❌ No exemplar bbox provided", "error")
                    print("❌ No bbox available for similar mode")

        except Exception as e:
            error_msg = f"❌ Semantic similar error: {e}"
            print(error_msg)
            import traceback
            traceback.print_exc()
            self._update_status(error_msg, "error")
        finally:
            self.is_processing = False
            self._set_ui_enabled(True)

    def _run_tiled_text_segmentation(self, raster_layer):
        """Run tiled segmentation for text prompt mode (large rasters)"""
        print("\n" + "="*80)
        print("🔧 TILED TEXT MODE - Processing large raster in tiles")
        print("="*80)

        # Use the TiledSegmentationWorker
        self._set_ui_enabled(False)
        self._update_status(f"🗺️  Finding '{self.text_prompt}' in tiles: {raster_layer.name()[:30]}...", "processing")

        # Track initial feature IDs before processing (for undo)
        result_layer = self._get_or_create_class_layer(self.current_class)
        if result_layer and result_layer.isValid():
            self._tiled_initial_feature_ids = set(f.id() for f in result_layer.getFeatures())
        else:
            self._tiled_initial_feature_ids = set()

        try:
            # Get class color
            class_color = self.classes.get(self.current_class, {}).get('color', '128,128,128')

            # Compute effective size filters for tiled worker
            real_world_filters = self._get_effective_pixel_filters()
            tiled_min = self.min_object_size
            tiled_max = None
            if real_world_filters:
                if 'min_area_px' in real_world_filters:
                    tiled_min = max(tiled_min, int(real_world_filters['min_area_px']))
                if 'max_area_px' in real_world_filters:
                    tiled_max = int(real_world_filters['max_area_px'])
            if tiled_max is None and self.max_object_size > 0:
                tiled_max = self.max_object_size

            self.tiled_worker = TiledSegmentationWorker(
                predictor=self.predictor,
                raster_path=raster_layer.source(),
                request_type='text',
                text_prompt=self.text_prompt if hasattr(self, 'text_prompt') else None,
                bbox=None,
                current_class=self.current_class,
                class_color=class_color,
                tile_size=1024,
                overlap=128,
                min_object_size=tiled_min,
                max_object_size_px=tiled_max,
                shape_filters={
                    'enabled': self.shape_filter_enabled,
                    'min_circularity': self.circularitySpinBox.value(),
                    'min_compactness': self.compactnessSpinBox.value(),
                    'max_aspect_ratio': self.aspectRatioSpinBox.value(),
                } if self.shape_filter_enabled else None
            )

            # Connect signals
            self.tiled_worker.finished.connect(self._on_tiled_segmentation_finished)
            self.tiled_worker.error.connect(self._on_tiled_segmentation_error)
            self.tiled_worker.progress.connect(self._on_tiled_segmentation_progress)
            self.tiled_worker.tile_completed.connect(self._on_tile_completed)
            self.tiled_worker.cancelled.connect(self._on_tiled_segmentation_cancelled)

            # Create and show cancel button
            self._create_cancel_button()

            # Start worker
            self.tiled_worker.start()
            print("✅ Tiled worker started")

        except Exception as e:
            print(f"❌ Error creating tiled worker: {e}")
            import traceback
            traceback.print_exc()
            self._update_status(f"❌ Failed to start tiled processing: {e}", "error")
            self.is_processing = False
            self._set_ui_enabled(True)

    def _run_tiled_similar_segmentation(self, raster_layer):
        """Run tiled segmentation for similar objects mode (large rasters)"""
        print("\n" + "="*80)
        print("🔧 TILED SIMILAR MODE - Processing large raster in tiles")
        print("="*80)

        # Use the TiledSegmentationWorker
        self._set_ui_enabled(False)
        self._update_status(f"🗺️  Processing similar objects in tiles: {raster_layer.name()[:30]}...", "processing")

        # Track initial feature IDs before processing (for undo)
        result_layer = self._get_or_create_class_layer(self.current_class)
        if result_layer and result_layer.isValid():
            self._tiled_initial_feature_ids = set(f.id() for f in result_layer.getFeatures())
        else:
            self._tiled_initial_feature_ids = set()

        try:
            # Get class color
            class_color = self.classes.get(self.current_class, {}).get('color', '128,128,128')

            # Compute effective size filters for tiled worker
            real_world_filters = self._get_effective_pixel_filters()
            tiled_min = self.min_object_size
            tiled_max = None
            if real_world_filters:
                if 'min_area_px' in real_world_filters:
                    tiled_min = max(tiled_min, int(real_world_filters['min_area_px']))
                if 'max_area_px' in real_world_filters:
                    tiled_max = int(real_world_filters['max_area_px'])
            if tiled_max is None and self.max_object_size > 0:
                tiled_max = self.max_object_size

            self.tiled_worker = TiledSegmentationWorker(
                predictor=self.predictor,
                raster_path=raster_layer.source(),
                request_type='similar',
                text_prompt=None,
                bbox=self.bbox if hasattr(self, 'bbox') else None,
                current_class=self.current_class,
                class_color=class_color,
                tile_size=1024,
                overlap=128,
                min_object_size=tiled_min,
                max_object_size_px=tiled_max,
                shape_filters={
                    'enabled': self.shape_filter_enabled,
                    'min_circularity': self.circularitySpinBox.value(),
                    'min_compactness': self.compactnessSpinBox.value(),
                    'max_aspect_ratio': self.aspectRatioSpinBox.value(),
                } if self.shape_filter_enabled else None
            )

            # Connect signals
            self.tiled_worker.finished.connect(self._on_tiled_segmentation_finished)
            self.tiled_worker.error.connect(self._on_tiled_segmentation_error)
            self.tiled_worker.progress.connect(self._on_tiled_segmentation_progress)
            self.tiled_worker.tile_completed.connect(self._on_tile_completed)
            self.tiled_worker.cancelled.connect(self._on_tiled_segmentation_cancelled)

            # Create and show cancel button
            self._create_cancel_button()

            # Start worker
            self.tiled_worker.start()
            print("✅ Tiled worker started")

        except Exception as e:
            print(f"❌ Error creating tiled worker: {e}")
            import traceback
            traceback.print_exc()
            self._update_status(f"❌ Failed to start tiled processing: {e}", "error")
            self.is_processing = False
            self._set_ui_enabled(True)

    def _run_tiled_segmentation(self, raster_layer):
        """Run segmentation on entire raster using auto-tiling (SAM3 text/similar modes)"""
        # Verify SAM3
        if self.model_choice != "SAM3":
            self._update_status("⚠️  Tiled segmentation requires SAM3", "error")
            self.is_processing = False
            self._set_ui_enabled(True)
            return

        # For SIMILAR mode: check size first, then decide
        if self.request_type == 'similar':
            self._run_semantic_similar_full_raster(raster_layer)
            return

        # For TEXT mode: check size first, then decide
        if self.request_type == 'text':
            self._run_semantic_text_full_raster(raster_layer)
            return

        self._set_ui_enabled(False)
        self._update_status(f"🗺️  Starting tiled segmentation on: {raster_layer.name()[:30]}...", "processing")

        import sys
        print("\n" + "="*80)
        print("🔧 CREATING TILED WORKER")
        print("="*80)
        print(f"Raster path: {raster_layer.source()}")
        print(f"Request type: {self.request_type}")
        print(f"Text prompt: {self.text_prompt if hasattr(self, 'text_prompt') else None}")
        print(f"Predictor: {type(self.predictor)}")
        print("="*80)
        sys.stdout.flush()

        # Create worker for background processing
        try:
            print("🔧 Creating TiledSegmentationWorker instance...")
            sys.stdout.flush()

            # Get class color
            class_color = self.classes.get(self.current_class, {}).get('color', '128,128,128')

            # Compute effective size filters for tiled worker
            real_world_filters = self._get_effective_pixel_filters()
            tiled_min = self.min_object_size
            tiled_max = None
            if real_world_filters:
                if 'min_area_px' in real_world_filters:
                    tiled_min = max(tiled_min, int(real_world_filters['min_area_px']))
                if 'max_area_px' in real_world_filters:
                    tiled_max = int(real_world_filters['max_area_px'])
            if tiled_max is None and self.max_object_size > 0:
                tiled_max = self.max_object_size

            self.tiled_worker = TiledSegmentationWorker(
                predictor=self.predictor,
                raster_path=raster_layer.source(),
                request_type=self.request_type,
                text_prompt=self.text_prompt if hasattr(self, 'text_prompt') else None,
                bbox=self.bbox if hasattr(self, 'bbox') else None,
                current_class=self.current_class,
                class_color=class_color,
                tile_size=1024,
                overlap=128,
                min_object_size=tiled_min,
                max_object_size_px=tiled_max,
                shape_filters={
                    'enabled': self.shape_filter_enabled,
                    'min_circularity': self.circularitySpinBox.value(),
                    'min_compactness': self.compactnessSpinBox.value(),
                    'max_aspect_ratio': self.aspectRatioSpinBox.value(),
                } if self.shape_filter_enabled else None
            )
            print("✅ Worker created successfully")
        except Exception as e:
            print(f"❌ Error creating worker: {e}")
            import traceback
            traceback.print_exc()
            self._update_status(f"❌ Failed to create worker: {e}", "error")
            self.is_processing = False
            self._set_ui_enabled(True)
            return

        # Connect signals
        self.tiled_worker.finished.connect(self._on_tiled_segmentation_finished)
        self.tiled_worker.error.connect(self._on_tiled_segmentation_error)
        self.tiled_worker.progress.connect(self._on_tiled_segmentation_progress)
        self.tiled_worker.tile_completed.connect(self._on_tile_completed)
        print("✅ Signals connected")

        # Start processing in background thread
        print("🚀 Starting worker thread...")
        self.tiled_worker.start()
        print("✅ Worker thread started")

    def _on_tiled_segmentation_progress(self, message, current_tile, total_tiles):
        """Handle progress updates from tiled worker"""
        self._update_status(message, "processing")

    def _on_tile_completed(self, objects_found, tile_index):
        """Handle completion of individual tile - add features to layer"""
        # Process results from worker's tile_results list
        if hasattr(self.tiled_worker, 'tile_results') and self.tiled_worker.tile_results:
            # Get the latest result
            for features_data, debug_info, transform in self.tiled_worker.tile_results:
                if features_data:
                    # Convert feature dicts to QgsFeature objects
                    from qgis.core import QgsFeature, QgsGeometry, QgsPointXY
                    qgs_features = []

                    for feat_data in features_data:
                        feature = QgsFeature()

                        # Convert coords to QgsPointXY
                        qgs_points = [QgsPointXY(x, y) for x, y in feat_data['coords']]

                        # Create polygon geometry
                        if len(qgs_points) >= 3:
                            geom = QgsGeometry.fromPolygonXY([qgs_points])
                            feature.setGeometry(geom)
                            qgs_features.append(feature)

                    if qgs_features:
                        # Temporarily disable undo tracking for individual tiles
                        # (will be tracked as single operation at the end)
                        undo_was_enabled = self.undoBtn.isEnabled()
                        undo_stack_size = len(self.undo_stack)

                        self._add_features_to_layer(qgs_features, debug_info, len(qgs_features))

                        # Remove the undo entry that was just added (we'll add one big entry at the end)
                        if len(self.undo_stack) > undo_stack_size:
                            self.undo_stack.pop()

                        # Keep undo button disabled during tiled processing
                        if not undo_was_enabled:
                            self.undoBtn.setEnabled(False)

            # Clear processed results
            self.tiled_worker.tile_results.clear()

    def _on_tiled_segmentation_finished(self, total_objects):
        """Handle completion of tiled segmentation"""

        # Quick deduplication pass - only for similar mode where duplicates are common
        if hasattr(self.tiled_worker, 'request_type') and self.tiled_worker.request_type == 'similar':
            self._deduplicate_layer_features(iou_threshold=0.5)

        # Add all tiled operation features to undo stack as a single operation
        # Use the difference between current and initial feature IDs (accounts for deduplication)
        if hasattr(self, '_tiled_initial_feature_ids'):
            result_layer = self._get_or_create_class_layer(self.current_class)
            if result_layer and result_layer.isValid():
                # Get current feature IDs after all processing (including deduplication)
                current_feature_ids = set(f.id() for f in result_layer.getFeatures())
                # Find new features (ones that weren't there before)
                new_feature_ids = list(current_feature_ids - self._tiled_initial_feature_ids)

                if new_feature_ids:
                    self.undo_stack.append((self.current_class, new_feature_ids))
                    self.undoBtn.setEnabled(True)
                    print(f"✅ Added {len(new_feature_ids)} features to undo stack (single operation)")
                else:
                    print(f"⚠️  No new features were added")

            # Clean up initial tracking
            delattr(self, '_tiled_initial_feature_ids')

        self._update_status(f"✅ Tiled segmentation complete - Found {total_objects} objects", "success")

        # Cleanup
        self.is_processing = False
        self._set_ui_enabled(True)
        self._remove_cancel_button()

        if hasattr(self, 'tiled_worker'):
            self.tiled_worker.deleteLater()
            self.tiled_worker = None

        # Process next in queue
        self._process_queue()

    def _on_tiled_segmentation_error(self, error_msg):
        """Handle errors from tiled worker"""
        self._update_status(f"❌ {error_msg}", "error")

        # Cleanup
        self.is_processing = False
        self._set_ui_enabled(True)
        self._remove_cancel_button()

        if hasattr(self, 'tiled_worker'):
            self.tiled_worker.deleteLater()
            self.tiled_worker = None

        # Process next in queue
        self._process_queue()

    def _on_tiled_segmentation_cancelled(self):
        """Handle cancellation of tiled processing"""
        print("⚠️ Tiled processing cancelled by user")
        self._update_status("⚠️ Processing cancelled by user", "warning")

        # Add any features that were created before cancellation to undo stack
        # This allows user to undo partial results from cancelled operation
        if hasattr(self, '_tiled_initial_feature_ids'):
            result_layer = self._get_or_create_class_layer(self.current_class)
            if result_layer and result_layer.isValid():
                # Get current feature IDs after partial processing
                current_feature_ids = set(f.id() for f in result_layer.getFeatures())
                # Find new features (ones that were added before cancellation)
                new_feature_ids = list(current_feature_ids - self._tiled_initial_feature_ids)

                if new_feature_ids:
                    self.undo_stack.append((self.current_class, new_feature_ids))
                    print(f"✅ Added {len(new_feature_ids)} partially completed features to undo stack")
                else:
                    print(f"⚠️  No features were added before cancellation")

            # Clean up initial tracking
            delattr(self, '_tiled_initial_feature_ids')

        # Cleanup
        self.is_processing = False
        self._set_ui_enabled(True)
        self._remove_cancel_button()

        if hasattr(self, 'tiled_worker'):
            self.tiled_worker.deleteLater()
            self.tiled_worker = None

        # Process next in queue
        self._process_queue()

    def _create_cancel_button(self):
        """Create and show cancel button for tiled processing (replaces undo button during processing)"""
        # Simply hide undo button and show cancel button (swap them)
        if not hasattr(self, 'cancelTiledBtn'):
            # Create cancel button with same style as undo button
            self.cancelTiledBtn = QtWidgets.QPushButton("🛑 Cancel Processing")
            self.cancelTiledBtn.setCursor(Qt.PointingHandCursor)
            self.cancelTiledBtn.setStyleSheet("""
                QPushButton {
                    font-size: 14px; font-weight: 600; padding: 10px;
                    border-radius: 8px; background: #DC2626; color: #FFF;
                    border: 1px solid #DC2626;
                }
                QPushButton:hover { background: #B91C1C; }
                QPushButton:disabled {
                    background: #FCA5A5; color: #FFFFFF;
                    border: 1px solid #DC2626;
                }
            """)
            self.cancelTiledBtn.setAutoDefault(False)
            self.cancelTiledBtn.setDefault(False)
            self.cancelTiledBtn.setFocusPolicy(Qt.NoFocus)
            self.cancelTiledBtn.clicked.connect(self._cancel_tiled_processing)

            # Find the undo button's parent layout and insert cancel button at same position
            parent_layout = self.undoBtn.parent().layout()
            if parent_layout:
                # Get undo button index
                for i in range(parent_layout.count()):
                    item = parent_layout.itemAt(i)
                    if item and item.widget() == self.undoBtn:
                        parent_layout.insertWidget(i, self.cancelTiledBtn)
                        break

        # Hide undo button and show cancel button
        self.undoBtn.setVisible(False)

        # Reset button state
        self.cancelTiledBtn.setEnabled(True)
        self.cancelTiledBtn.setText("🛑 Cancel Processing")
        self.cancelTiledBtn.setVisible(True)
        print("✅ Cancel button shown (undo button hidden)")

    def _remove_cancel_button(self):
        """Remove/hide cancel button after tiled processing (restore undo button)"""
        if hasattr(self, 'cancelTiledBtn'):
            self.cancelTiledBtn.setVisible(False)
            print("✅ Cancel button hidden")

        # Restore undo button visibility
        self.undoBtn.setVisible(True)

    def _cancel_tiled_processing(self):
        """Cancel ongoing tiled processing"""
        if hasattr(self, 'tiled_worker') and self.tiled_worker:
            print("🛑 User clicked cancel button")
            self.tiled_worker.cancel()
            # Disable cancel button to prevent double-clicks
            if hasattr(self, 'cancelTiledBtn'):
                self.cancelTiledBtn.setEnabled(False)
                self.cancelTiledBtn.setText("Cancelling...")

    def _convert_mask_to_features(self, mask, mask_transform):
        """Convert a single mask to QgsFeature objects with size/shape filtering"""
        # Compute effective size filters
        real_world_filters = self._get_effective_pixel_filters()
        effective_min = self.min_object_size
        effective_max = None
        if real_world_filters:
            if 'min_area_px' in real_world_filters:
                effective_min = max(effective_min, int(real_world_filters['min_area_px']))
            if 'max_area_px' in real_world_filters:
                effective_max = int(real_world_filters['max_area_px'])
        if effective_max is None and self.max_object_size > 0:
            effective_max = self.max_object_size

        # Build shape filters dict
        sf = None
        if self.shape_filter_enabled:
            sf = {
                'enabled': True,
                'min_circularity': self.circularitySpinBox.value(),
                'min_compactness': self.compactnessSpinBox.value(),
                'max_aspect_ratio': self.aspectRatioSpinBox.value(),
            }

        try:
            # Threshold mask to binary and clean it up
            _, binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

            # Morphological operations to clean up the mask
            open_kernel = np.ones((3, 3), np.uint8)
            binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, open_kernel)

            close_kernel = np.ones((7, 7), np.uint8)
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, close_kernel)

            # Remove objects outside size range
            nlabels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
            cleaned = np.zeros(binary.shape, dtype=np.uint8)
            for i in range(1, nlabels):
                area = stats[i, cv2.CC_STAT_AREA]
                if area < effective_min:
                    continue
                if effective_max is not None and area > effective_max:
                    continue
                cleaned[labels == i] = 255
            binary = cleaned

        except Exception as e:
            print(f"Error thresholding/refining mask: {e}")
            return []

        # Convert mask to features
        features = []
        try:
            for geom, _ in shapes(binary, mask=binary > 0, transform=mask_transform):
                shp_geom = shape(geom)
                if not shp_geom.is_valid:
                    shp_geom = shp_geom.buffer(0)
                if shp_geom.is_empty:
                    continue

                # Shape filter check using shapely geometry
                if sf and sf.get('enabled'):
                    shp_area = shp_geom.area
                    shp_perimeter = shp_geom.length
                    if shp_perimeter > 0:
                        compactness = 4 * 3.14159265 * shp_area / (shp_perimeter * shp_perimeter)
                    else:
                        compactness = 0
                    bx, by, bxx, byy = shp_geom.bounds
                    bw, bh = bxx - bx, byy - by
                    aspect_ratio = max(bw, bh) / max(min(bw, bh), 1e-9)

                    if compactness < sf.get('min_compactness', 0.0):
                        continue
                    if compactness < sf.get('min_circularity', 0.0):
                        continue
                    if aspect_ratio > sf.get('max_aspect_ratio', 20.0):
                        continue

                # Convert shapely geometry to QGIS geometry
                if hasattr(shp_geom, 'exterior'):
                    coords = list(shp_geom.exterior.coords)
                    qgs_points = []
                    for coord in coords:
                        if len(coord) >= 2:
                            qgs_points.append(QgsPointXY(coord[0], coord[1]))
                    if len(qgs_points) >= 3:
                        qgs_geom = QgsGeometry.fromPolygonXY([qgs_points])
                    else:
                        continue
                else:
                    try:
                        wkt_str = shp_geom.wkt
                        qgs_geom = QgsGeometry.fromWkt(wkt_str)
                    except Exception as e:
                        print(f"⚠️ WKT conversion failed: {e}")
                        continue

                if not qgs_geom.isNull() and not qgs_geom.isEmpty():
                    f = QgsFeature()
                    f.setGeometry(qgs_geom)
                    features.append(f)

        except Exception as e:
            print(f"Error processing geometries: {str(e)}")
            return []

        return features


    def _get_adaptive_bbox_padding(self, bbox_area):
        """Calculate adaptive padding based on bbox size to reduce background inclusion"""
        # For geographic coordinates (small areas), use different thresholds
        if bbox_area < 1:  # Geographic coordinates in degrees
            if bbox_area > 0.1:         # Very large geographic area
                return 0.02             # 2% padding
            elif bbox_area > 0.01:      # Large geographic area  
                return 0.05             # 5% padding
            elif bbox_area > 0.001:     # Medium geographic area
                return 0.1              # 10% padding
            elif bbox_area > 0.000001:  # Small geographic area
                return 0.2              # 20% padding
            else:                       # Very small geographic area
                return 0.3              # 30% padding
        else:  # Projected coordinates (large areas)
            if bbox_area > 500000:      # Very large area
                return 0.02             # 2% padding
            elif bbox_area > 100000:    # Large area  
                return 0.05             # 5% padding
            elif bbox_area > 50000:     # Medium area
                return 0.1              # 10% padding
            else:                       # Small area
                return 0.2              # 20% padding

    def _deduplicate_layer_features(self, iou_threshold=0.5):
        """
        Quick deduplication pass on the current class layer.
        Removes duplicate features from tile overlap zones.

        Args:
            iou_threshold: IoU threshold for considering features as duplicates
        """
        try:
            # Get the current class layer
            result_layer = self._get_or_create_class_layer(self.current_class)
            if not result_layer or not result_layer.isValid():
                return

            feature_count = result_layer.featureCount()
            if feature_count <= 1:
                return

            print(f"🔍 Deduplicating {feature_count} features...")

            # Get all features
            features = list(result_layer.getFeatures())
            to_delete = []

            # Find duplicates based on IoU
            for i in range(len(features)):
                if features[i].id() in to_delete:
                    continue

                geom_i = features[i].geometry()
                if not geom_i or geom_i.isEmpty():
                    continue

                for j in range(i + 1, len(features)):
                    if features[j].id() in to_delete:
                        continue

                    geom_j = features[j].geometry()
                    if not geom_j or geom_j.isEmpty():
                        continue

                    # Calculate IoU
                    intersection = geom_i.intersection(geom_j)
                    if not intersection.isEmpty():
                        area_i = geom_i.area()
                        area_j = geom_j.area()
                        area_intersection = intersection.area()

                        # Check if one is contained in the other (95% overlap)
                        if area_i > 0 and area_intersection / area_i > 0.95:
                            to_delete.append(features[i].id())
                            break
                        elif area_j > 0 and area_intersection / area_j > 0.95:
                            to_delete.append(features[j].id())
                            continue

                        # Check IoU for overlapping duplicates
                        if area_i > 0 and area_j > 0:
                            area_union = area_i + area_j - area_intersection
                            if area_union > 0:
                                iou = area_intersection / area_union
                                if iou > iou_threshold:
                                    # Keep the larger one
                                    if area_i >= area_j:
                                        to_delete.append(features[j].id())
                                    else:
                                        to_delete.append(features[i].id())
                                        break

            # Delete duplicates
            if to_delete:
                result_layer.startEditing()
                result_layer.deleteFeatures(to_delete)
                result_layer.commitChanges()
                print(f"✅ Removed {len(to_delete)} duplicates, {result_layer.featureCount()} features remaining")

        except Exception as e:
            print(f"⚠️  Deduplication failed: {e}")
            import traceback
            traceback.print_exc()

    def _add_features_to_layer(self, features, debug_info, object_count, filename=None):
        """Add features to the appropriate class layer"""
        current_raster = self.iface.activeLayer()
        if isinstance(current_raster, QgsRasterLayer):
            self.original_raster_layer = current_raster

        try:
            result_layer = self._get_or_create_class_layer(self.current_class)
            if not result_layer or not result_layer.isValid():
                self._update_status("Failed to create or access layer", "error")
                return

            # Get the next available segment ID
            next_segment_id = self._get_next_segment_id(result_layer, self.current_class)

            # Enhanced attributes with layer tracking
            timestamp_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            crop_info = debug_info.get('crop_size', 'unknown') if self.current_mode == 'bbox' else debug_info.get('actual_crop', 'unknown')
            class_color = self.classes.get(self.current_class, {}).get('color', '128,128,128')
            canvas_scale = self.canvas.scale()
            source_layer_name = debug_info.get('source_layer', 'unknown')
            layer_crs = debug_info.get('layer_crs', 'unknown')

            # Add batch info
            batch_info = f"batch_{object_count}_objects" if debug_info.get('individual_processing') else "single"

            # Set enhanced attributes for features
            import math as _math
            for i, feat in enumerate(features):
                geom = feat.geometry()
                area_m2 = geom.area() if geom and not geom.isEmpty() else 0.0
                diameter_cm = 2.0 * _math.sqrt(area_m2 / _math.pi) * 100.0 if area_m2 > 0 else 0.0
                perimeter = geom.length() if geom and not geom.isEmpty() else 0.0
                circularity = (4.0 * _math.pi * area_m2) / (perimeter * perimeter) if perimeter > 0 else 0.0
                feat.setAttributes([
                    next_segment_id + i,
                    self.current_class,
                    class_color,
                    batch_info,
                    timestamp_str,
                    filename or "debug_disabled",
                    crop_info,
                    canvas_scale,
                    source_layer_name,
                    layer_crs,
                    round(area_m2, 4),
                    round(diameter_cm, 2),
                    round(circularity, 4)
                ])

            # Add features and track for undo
            result_layer.startEditing()
            success = result_layer.dataProvider().addFeatures(features)
            result_layer.commitChanges()

            if success:
                # Update tracking
                self.segment_counts[self.current_class] = next_segment_id + len(features) - 1

                # Enhanced undo tracking with layer info
                all_features = list(result_layer.getFeatures())
                new_feature_ids = [f.id() for f in all_features[-len(features):]]
                self.undo_stack.append((self.current_class, new_feature_ids))
                self.undoBtn.setEnabled(True)

            result_layer.updateExtents()
            result_layer.triggerRepaint()

            # Keep the source raster selected
            if self.keep_raster_selected and self.original_raster_layer:
                self.iface.setActiveLayer(self.original_raster_layer)

            # Update layer name with source info
            total_features = result_layer.featureCount()
            color_info = f" [RGB:{class_color}]"
            source_info = f" [{source_layer_name[:10]}]" if source_layer_name != 'unknown' else ""
            new_layer_name = f"SAM_{self.current_class}{source_info} ({total_features}){color_info}"
            result_layer.setName(new_layer_name)

            # Clear visual feedback
            if self.current_mode == 'point':
                self.pointTool.clear_feedback()
            elif self.current_mode == 'bbox':
                self.bboxTool.clear_feedback()

        except Exception as e:
            self._update_status(f"Error adding features: {e}", "error")
            return

    def _prepare_optimized_segmentation_data(self, rlayer):
        # Check if this is a tile layer that needs caching
        tile_type = self._detect_tile_layer_type(rlayer)
        cached_path = None

        if tile_type:
            # For tile layers, cache the current extent as a temporary raster
            self._update_status(f"🌐 {tile_type} tile layer - caching current view", "processing")
            cached_path = self._cache_tile_layer_as_raster(rlayer)
            if cached_path:
                rpath = cached_path
                self._update_status("✅ Tile caching complete", "info")
            else:
                self._update_status("❌ Failed to cache tiles - check console for details", "error")
                return None
        else:
            rpath = rlayer.source()

        adaptive_crop_size = self._get_adaptive_crop_size()

        try:
            with rasterio.open(rpath) as src:
                # Handle multi-band images
                band_count = src.count

                # Determine which bands to use - preserve all bands for multi-spectral
                if band_count >= 5:
                    # Multi-spectral: read all bands for advanced processing
                    bands_to_read = list(range(1, band_count + 1))
                    print(f"📡 Multi-spectral mode: reading all {band_count} bands")
                elif band_count >= 3:
                    bands_to_read = [1, 2, 3]
                elif band_count == 2:
                    bands_to_read = [1, 1, 2]
                elif band_count == 1:
                    bands_to_read = [1, 1, 1]
                else:
                    self._update_status("No bands found in raster", "error")
                    return None

                # TEXT/SIMILAR MODE (extent-based crop)
                if hasattr(self, 'request_type') and self.request_type in ['text', 'similar']:
                    scope = getattr(self, 'request_scope', 'aoi')
                    if scope == 'full':
                        # Use full raster bounds
                        extent = src.bounds
                    else:
                        # Use map extent for AOI
                        extent = self.canvas.extent()
                    try:
                        if scope == 'full':
                            xmin, ymin, xmax, ymax = extent.left, extent.bottom, extent.right, extent.top
                        else:
                            xmin, ymin, xmax, ymax = (
                                extent.xMinimum(), extent.yMinimum(),
                                extent.xMaximum(), extent.yMaximum()
                            )

                        window = rasterio.windows.from_bounds(
                            xmin, ymin, xmax, ymax,
                            src.transform
                        )
                        # Limit to raster bounds
                        window = window.intersection(
                            rasterio.windows.Window(0, 0, src.width, src.height)
                        )

                        # Limit size for performance
                        # Similar mode is more memory-intensive, use smaller max size
                        if self.request_type == 'similar':
                            max_size = 1024  # More conservative for similar mode (GPU memory intensive)
                        else:
                            max_size = 2048  # Text mode can handle larger

                        scale = 1.0
                        if window.width > max_size or window.height > max_size:
                            scale = min(max_size / window.width, max_size / window.height)
                            out_width = int(window.width * scale)
                            out_height = int(window.height * scale)
                            arr = src.read(bands_to_read, window=window,
                                         out_shape=(len(bands_to_read), out_height, out_width),
                                         out_dtype=np.uint8)
                        else:
                            arr = src.read(bands_to_read, window=window, out_dtype=np.uint8)

                        arr = np.moveaxis(arr, 0, -1)
                        input_coords = None
                        input_labels = None
                        input_box = None
                        mask_transform = src.window_transform(window)

                        # CRITICAL: Adjust transform if we downsampled
                        if scale < 1.0:
                            from rasterio import Affine
                            mask_transform = mask_transform * Affine.scale(1/scale)
                            print(f"⚠️  Downsampled to {arr.shape[1]}x{arr.shape[0]} (scale={scale:.3f}), adjusted transform")

                        # For SIMILAR mode, convert exemplar bbox to pixel coords in this crop
                        if self.request_type == 'similar' and self.bbox is not None:
                            corners = [
                                (self.bbox.xMinimum(), self.bbox.yMinimum()),
                                (self.bbox.xMaximum(), self.bbox.yMinimum()),
                                (self.bbox.xMaximum(), self.bbox.yMaximum()),
                                (self.bbox.xMinimum(), self.bbox.yMaximum())
                            ]
                            pixel_coords = []
                            for x, y in corners:
                                px, py = ~mask_transform * (x, y)
                                pixel_coords.append((px, py))

                            xs, ys = zip(*pixel_coords)
                            x1, x2 = min(xs), max(xs)
                            y1, y2 = min(ys), max(ys)

                            x1 = max(0, min(arr.shape[1] - 1, int(x1)))
                            y1 = max(0, min(arr.shape[0] - 1, int(y1)))
                            x2 = max(0, min(arr.shape[1] - 1, int(x2)))
                            y2 = max(0, min(arr.shape[0] - 1, int(y2)))

                            # Expand if bbox is too small
                            if (x2 - x1) < 5 or (y2 - y1) < 5:
                                print(f"⚠️  Bbox too small ({x2-x1}x{y2-y1}px), expanding to minimum size")
                                center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
                                x1 = max(0, center_x - 10)
                                y1 = max(0, center_y - 10)
                                x2 = min(arr.shape[1] - 1, center_x + 10)
                                y2 = min(arr.shape[0] - 1, center_y + 10)

                            input_box = np.array([[x1, y1, x2, y2]])

                        debug_info = {
                            'mode': self.request_type.upper(),
                            'class': self.current_class,
                            'actual_crop': f"{arr.shape[1]}x{arr.shape[0]}",
                            'bands_used': f"{band_count} -> {len(bands_to_read)}",
                            'device': self.device
                        }

                        return arr, mask_transform, debug_info, input_coords, input_labels, input_box

                    except Exception as e:
                        self._update_status(f"Error processing {self.request_type} mode extent: {e}", "error")
                        return None

                elif self.point is not None:  # POINT MODE
                    try:
                        row, col = src.index(self.point.x(), self.point.y())
                        center_pixel_x, center_pixel_y = col, row
                    except Exception as e:
                        self._update_status(f"Point is outside raster bounds: {e}", "error")
                        return None

                    crop_size = adaptive_crop_size
                    half_size = crop_size // 2

                    # For multi-point: expand crop to cover all points
                    multi_points = self.current_request.get('multi_points') if self.current_request else None
                    if multi_points:
                        all_px = [center_pixel_x]
                        all_py = [center_pixel_y]
                        for map_pt, _ in multi_points:
                            try:
                                r, c = src.index(map_pt.x(), map_pt.y())
                                all_px.append(c)
                                all_py.append(r)
                            except Exception:
                                pass
                        # Crop must contain all points with padding
                        px_min, px_max = min(all_px), max(all_px)
                        py_min, py_max = min(all_py), max(all_py)
                        span_x = px_max - px_min
                        span_y = py_max - py_min
                        # Ensure crop is at least adaptive_crop_size and covers all points + padding
                        needed = max(crop_size, int(max(span_x, span_y) * 1.5))
                        half_size = needed // 2
                        center_pixel_x = (px_min + px_max) // 2
                        center_pixel_y = (py_min + py_max) // 2

                    x_min = max(0, center_pixel_x - half_size)
                    y_min = max(0, center_pixel_y - half_size)
                    x_max = min(src.width, center_pixel_x + half_size)
                    y_max = min(src.height, center_pixel_y + half_size)

                    if x_max <= x_min or y_max <= y_min:
                        self._update_status("Invalid crop area for point", "error")
                        return None

                    window = rasterio.windows.Window(
                        x_min, y_min, x_max - x_min, y_max - y_min)

                    try:
                        # Use float32 for multi-spectral to preserve reflectance values
                        if band_count >= 5:
                            arr = src.read(bands_to_read, window=window, out_dtype=np.float32)
                        else:
                            arr = src.read(bands_to_read, window=window, out_dtype=np.uint8)
                        if arr.size == 0:
                            self._update_status("Empty crop area", "error")
                            return None
                    except Exception as e:
                        self._update_status(f"Error reading raster: {e}", "error")
                        return None

                    # Handle different band configurations
                    if band_count == 1:
                        arr = np.stack([arr[0], arr[0], arr[0]], axis=0)
                    elif band_count == 2:
                        arr = np.stack([arr[0], arr[0], arr[1]], axis=0)
                    elif band_count >= 5:
                        # Multi-spectral: keep all bands as-is
                        pass
                    # For 3-4 bands, arr is already correct

                    arr = np.moveaxis(arr, 0, -1)

                    # Normalize - preserve multi-spectral data ranges
                    if band_count >= 5:
                        # For multi-spectral, normalize each band independently
                        normalized_bands = []
                        for i in range(arr.shape[2]):
                            band = arr[:, :, i].astype(np.float32)
                            if band.max() > band.min():
                                band_norm = ((band - band.min()) / (band.max() - band.min()) * 255).astype(np.uint8)
                            else:
                                band_norm = np.zeros_like(band, dtype=np.uint8)
                            normalized_bands.append(band_norm)
                        arr = np.stack(normalized_bands, axis=2)
                    else:
                        # Standard normalization for RGB
                        if arr.max() > arr.min():
                            arr_min, arr_max = arr.min(), arr.max()
                            arr = ((arr.astype(np.float32) - arr_min) /
                                (arr_max - arr_min) * 255).astype(np.uint8)
                        else:
                            arr = np.zeros_like(arr, dtype=np.uint8)

                    # Build input coordinates — single or multi-point
                    multi_points = self.current_request.get('multi_points') if self.current_request else None

                    if multi_points:
                        # Multi-point mode: convert all map points to pixel coords
                        coords_list = []
                        labels_list = []
                        for map_pt, label in multi_points:
                            try:
                                r, c = src.index(map_pt.x(), map_pt.y())
                                rel_x = max(0, min(arr.shape[1] - 1, c - x_min))
                                rel_y = max(0, min(arr.shape[0] - 1, r - y_min))
                                coords_list.append([rel_x, rel_y])
                                labels_list.append(label)
                            except Exception:
                                pass  # Skip points outside raster bounds
                        if not coords_list:
                            self._update_status("No valid points in raster bounds", "error")
                            return None
                        input_coords = np.array(coords_list)
                        input_labels = np.array(labels_list)
                    else:
                        # Single-point mode (unchanged)
                        relative_x = center_pixel_x - x_min
                        relative_y = center_pixel_y - y_min
                        relative_x = max(0, min(arr.shape[1] - 1, relative_x))
                        relative_y = max(0, min(arr.shape[0] - 1, relative_y))
                        input_coords = np.array([[relative_x, relative_y]])
                        input_labels = np.array([1])

                    input_box = None
                    mask_transform = src.window_transform(window)

                    debug_info = {
                        'mode': 'POINT',
                        'class': self.current_class,
                        'actual_crop': f"{arr.shape[1]}x{arr.shape[0]}",
                        'bands_used': f"{band_count} -> {len(bands_to_read)}",
                        'device': self.device
                    }

                else:  # BBOX MODE - SMART HYBRID VERSION
                    # Calculate bbox dimensions in geographic coordinates
                    bbox_width = self.bbox.width()
                    bbox_height = self.bbox.height()
                    bbox_area = bbox_width * bbox_height

                    # SMART HYBRID: Adaptive padding based on area size
                    # For geographic coordinates, use much smaller thresholds
                    large_area_threshold = 0.000001 if bbox_area < 1 else 50000

                    if bbox_area > large_area_threshold:  # Large areas get adaptive padding
                        padding_factor = self._get_adaptive_bbox_padding(bbox_area)
                        print(f"🎯 LARGE area ({bbox_area:.0f}): adaptive padding {padding_factor*100:.1f}%")

                        # Extra reduction for batch mode on large areas
                        if hasattr(self, 'batch_mode_enabled') and self.batch_mode_enabled:
                            padding_factor *= 0.6
                            print(f"🔄 Batch mode: further reduced to {padding_factor*100:.1f}%")

                    else:  # Small areas use original fixed logic
                        padding_factor = 0.3  # 30% for small areas
                        print(f"📍 SMALL area ({bbox_area:.0f}): fixed padding {padding_factor*100:.1f}%")

                    # FIXED: Define max_crop_size based on area and device
                    if bbox_area > 1000000:  # Very large area (1M map units²)
                        max_crop_size = 2048
                    elif bbox_area > 100000:   # Large area 
                        max_crop_size = 1536
                    elif bbox_area > 10000:    # Medium area
                        max_crop_size = 1024
                    else:  # Small area
                        max_crop_size = 768

                    # Adjust max_crop_size based on device capability
                    if self.device == "cuda":
                        max_crop_size = min(max_crop_size * 1.5, 2048)  # Increase for GPU
                    elif self.device == "cpu" and self.model_choice.startswith("SAM2.1_"):
                        max_crop_size = min(max_crop_size, 1024)  # Limit for CPU SAM2.1

                    print(f"📐 Max crop size: {max_crop_size}px (device: {self.device})")

                    # Create padded bbox for context
                    padded_bbox = QgsRectangle(
                        self.bbox.xMinimum() - bbox_width * padding_factor,
                        self.bbox.yMinimum() - bbox_height * padding_factor,
                        self.bbox.xMaximum() + bbox_width * padding_factor,
                        self.bbox.yMaximum() + bbox_height * padding_factor
                    )

                    try:
                        # Use from_bounds correctly
                        padded_window = rasterio.windows.from_bounds(
                            padded_bbox.xMinimum(), padded_bbox.yMinimum(),
                            padded_bbox.xMaximum(), padded_bbox.yMaximum(),
                            src.transform
                        )

                        # Ensure window is within raster bounds
                        padded_window = padded_window.intersection(
                            rasterio.windows.Window(0, 0, src.width, src.height)
                        )

                    except Exception as e:
                        self._update_status(f"Error creating bbox window: {e}", "error")
                        return None

                    if padded_window.width <= 0 or padded_window.height <= 0:
                        self._update_status("Invalid bbox dimensions", "error")
                        return None

                    # Check if crop would be too large and downsample if needed
                    if padded_window.width > max_crop_size or padded_window.height > max_crop_size:
                        # Calculate downsampling factor
                        scale_factor = min(
                            max_crop_size / padded_window.width,
                            max_crop_size / padded_window.height
                        )

                        # Read with downsampling
                        out_width = int(padded_window.width * scale_factor)
                        out_height = int(padded_window.height * scale_factor)

                        try:
                            # Use float32 for multi-spectral to preserve reflectance values
                            if band_count >= 5:
                                arr = src.read(
                                    bands_to_read, 
                                    window=padded_window, 
                                    out_shape=(len(bands_to_read), out_height, out_width),
                                    out_dtype=np.float32
                                )
                            else:
                                arr = src.read(
                                    bands_to_read, 
                                    window=padded_window, 
                                    out_shape=(len(bands_to_read), out_height, out_width),
                                    out_dtype=np.uint8
                                )
                            print(f"🔽 Downsampled large bbox: {padded_window.width}x{padded_window.height} -> {out_width}x{out_height}")
                        except Exception as e:
                            self._update_status(f"Error reading downsampled raster: {e}", "error")
                            return None
                    else:
                        # Read at full resolution
                        try:
                            # Use float32 for multi-spectral to preserve reflectance values
                            if band_count >= 5:
                                arr = src.read(bands_to_read, window=padded_window, out_dtype=np.float32)
                            else:
                                arr = src.read(bands_to_read, window=padded_window, out_dtype=np.uint8)
                        except Exception as e:
                            self._update_status(f"Error reading raster: {e}", "error")
                            return None

                    if arr.size == 0:
                        self._update_status("Empty crop area", "error")
                        return None

                    # Handle different band configurations
                    if band_count == 1:
                        arr = np.stack([arr[0], arr[0], arr[0]], axis=0)
                    elif band_count == 2:
                        arr = np.stack([arr[0], arr[0], arr[1]], axis=0)
                    elif band_count >= 5:
                        # Multi-spectral: keep all bands as-is
                        pass
                    # For 3-4 bands, arr is already correct

                    arr = np.moveaxis(arr, 0, -1)

                    # Normalize - preserve multi-spectral data ranges
                    if band_count >= 5:
                        # For multi-spectral, normalize each band independently
                        normalized_bands = []
                        for i in range(arr.shape[2]):
                            band = arr[:, :, i].astype(np.float32)
                            if band.max() > band.min():
                                band_norm = ((band - band.min()) / (band.max() - band.min()) * 255).astype(np.uint8)
                            else:
                                band_norm = np.zeros_like(band, dtype=np.uint8)
                            normalized_bands.append(band_norm)
                        arr = np.stack(normalized_bands, axis=2)
                    else:
                        # Standard normalization for RGB
                        if arr.max() > arr.min():
                            arr_min, arr_max = arr.min(), arr.max()
                            arr = ((arr.astype(np.float32) - arr_min) /
                                (arr_max - arr_min) * 255).astype(np.uint8)
                        else:
                            arr = np.zeros_like(arr, dtype=np.uint8)

                    # Calculate bbox coordinates in the cropped image
                    padded_transform = src.window_transform(padded_window)

                    # Account for downsampling in transform
                    if 'scale_factor' in locals():
                        from affine import Affine
                        # Adjust transform for downsampling
                        a, b, c, d, e, f = padded_transform[:6]
                        padded_transform = Affine(a/scale_factor, b, c, d, e/scale_factor, f)

                    try:
                        # Convert bbox corners to pixel coordinates correctly
                        corners = [
                            (self.bbox.xMinimum(), self.bbox.yMinimum()),  # bottom-left
                            (self.bbox.xMaximum(), self.bbox.yMinimum()),  # bottom-right  
                            (self.bbox.xMaximum(), self.bbox.yMaximum()),  # top-right
                            (self.bbox.xMinimum(), self.bbox.yMaximum())   # top-left
                        ]

                        pixel_coords = []
                        for x, y in corners:
                            px, py = ~padded_transform * (x, y)
                            pixel_coords.append((px, py))

                        # Find bounding rectangle of all transformed corners
                        xs, ys = zip(*pixel_coords)
                        x1, x2 = min(xs), max(xs)
                        y1, y2 = min(ys), max(ys)

                        # Convert to integers and clamp to image bounds
                        x1 = max(0, min(arr.shape[1]-1, int(x1)))
                        y1 = max(0, min(arr.shape[0]-1, int(y1))) 
                        x2 = max(0, min(arr.shape[1]-1, int(x2)))
                        y2 = max(0, min(arr.shape[0]-1, int(y2)))

                        # Ensure minimum bbox size
                        if (x2 - x1) < 5 or (y2 - y1) < 5:
                            center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
                            x1 = max(0, center_x - 10)
                            y1 = max(0, center_y - 10)
                            x2 = min(arr.shape[1]-1, center_x + 10)
                            y2 = min(arr.shape[0]-1, center_y + 10)

                    except Exception as e:
                        self._update_status(f"Error converting bbox coordinates: {e}", "error")
                        print(f"Debug - bbox: {self.bbox.toString()}")
                        print(f"Debug - transform: {padded_transform}")
                        return None

                    # Set SAM inputs
                    input_box = np.array([[x1, y1, x2, y2]])
                    input_coords = None
                    input_labels = None
                    mask_transform = padded_transform

                    debug_info = {
                        'mode': 'SMART_HYBRID_BBOX',
                        'class': self.current_class,
                        'original_bbox': f"{bbox_width:.1f}x{bbox_height:.1f}",
                        'bbox_area': bbox_area,
                        'padding_strategy': 'adaptive' if bbox_area > 50000 else 'fixed',
                        'crop_size': f"{arr.shape[1]}x{arr.shape[0]}",
                        'padding_factor': f"{padding_factor:.3f}",
                        'target_bbox': f"({x1},{y1})-({x2},{y2})",
                        'target_size': f"{x2-x1}x{y2-y1}",
                        'bands_used': f"{band_count} -> {len(bands_to_read)}",
                        'downsampled': 'scale_factor' in locals(),
                        'max_crop_size': max_crop_size,
                        'device': self.device
                    }

                # Create RGB version for SAM2 and keep multi-spectral for vegetation detection
                if band_count >= 5:
                    # Create RGB version for SAM2 (use first 3 bands)
                    arr_rgb = arr[:, :, :3].copy()
                    # Keep full multi-spectral for vegetation detection
                    arr_multispectral = arr.copy()
                    return arr_rgb, mask_transform, debug_info, input_coords, input_labels, input_box, arr_multispectral
                else:
                    return arr, mask_transform, debug_info, input_coords, input_labels, input_box, None

        except Exception as e:
            self._update_status(f"Error accessing raster data: {e}", "error")
            return None
        finally:
            # Clean up temporary cached file
            if cached_path and os.path.exists(cached_path):
                try:
                    os.unlink(cached_path)
                    print(f"🧹 Cleaned up temp file: {os.path.basename(cached_path)}")
                except Exception as e:
                    print(f"Warning: Could not clean up temp file {cached_path}: {e}")

    def _get_adaptive_crop_size(self):
        canvas_scale = self.canvas.scale()

        # Base sizes based on device capability
        if self.device == "cuda":
            base_size = 1024  # Increased for CUDA
        elif self.device == "mps":
            base_size = 768   # Good for Apple Silicon
        else:
            # CPU: SAM2.1 models are optimized for smaller sizes
            base_size = 512 if self.model_choice.startswith("SAM2.1_") else 640

        # Adjust based on map scale for better context
        if canvas_scale > 500000:      # Very zoomed out - use larger crops
            crop_size = min(base_size * 2, 2048)
        elif canvas_scale > 100000:    # Zoomed out
            crop_size = int(base_size * 1.5)
        elif canvas_scale > 10000:     # Medium zoom
            crop_size = base_size
        elif canvas_scale > 1000:      # Zoomed in
            crop_size = int(base_size * 0.8)
        else:                          # Very zoomed in
            crop_size = max(256, int(base_size * 0.6))

        # Ensure reasonable bounds
        crop_size = max(256, min(crop_size, 2048))

        return crop_size

    def _on_segmentation_finished(self, result):
        try:
            import time
            start_process_time = time.time()

            self._update_status("✨ Processing results...", "processing")

            debug_info = result['debug_info']

            # FIXED: Better detection of batch vs single results
            # Check for 'individual_masks' key instead of just debug flag
            if 'individual_masks' in result:
                # This is batch processing - result structure: {'individual_masks': [...], 'mask_transform': ..., 'debug_info': ...}
                print(f"🔄 BATCH: Processing {len(result['individual_masks'])} individual masks")
                self._process_individual_batch_results(result, result['mask_transform'], debug_info)
            elif 'masks' in result:
                # This is text/similar mode with multiple masks - process all together for single undo
                print(f"🔄 TEXT/SIMILAR: Processing {len(result['masks'])} masks together")
                masks = result['masks']
                mask_transform = result['mask_transform']
                self._process_multiple_masks_result(masks, mask_transform, debug_info)
            elif 'mask' in result:
                # This is single processing - result structure: {'mask': array, 'mask_transform': ..., 'debug_info': ...}
                # (Reduced logging - was causing spam)
                mask = result['mask']
                mask_transform = result['mask_transform']
                self._process_single_mask_result(mask, mask_transform, debug_info)
            else:
                # Error case - unexpected result structure
                raise KeyError(f"Unexpected result structure. Expected 'mask', 'masks' or 'individual_masks', got keys: {list(result.keys())}")

            process_time = time.time() - start_process_time
            prep_time = debug_info.get('prep_time', 0)
            total_time = prep_time + process_time

            count = debug_info.get('batch_count')
            if isinstance(count, tuple):
                count = count[0]
            count_str = f"{count} objects found" if count else "Done"
            self._update_status(f"✅ {count_str} · {total_time:.1f}s", "info")
            self._clear_widget_focus()

        except Exception as e:
            import traceback
            error_msg = f"Error processing results: {str(e)}\n"
            error_msg += f"Result keys: {list(result.keys()) if isinstance(result, dict) else 'Not a dict'}\n"
            error_msg += f"Debug info: {debug_info if 'debug_info' in locals() else 'None'}\n"
            error_msg += f"Full traceback:\n{traceback.format_exc()}"
            self._update_status(error_msg, "error")
        finally:
            self._set_ui_enabled(True)
            self.is_processing = False  # Reset processing state
            if hasattr(self, 'worker') and self.worker:
                self.worker.deleteLater()
                self.worker = None

            # Process next item in queue
            self._process_queue()

    def _get_next_segment_id(self, layer, class_name):
        """Get the next available segment ID for a class"""
        if layer.featureCount() == 0:
            return 1

        # Find the highest existing segment_id
        max_id = 0
        for feature in layer.getFeatures():
            try:
                segment_id = feature.attribute("segment_id")
                if segment_id is not None and isinstance(segment_id, int):
                    max_id = max(max_id, segment_id)
            except:
                pass

        return max_id + 1

    def _update_segment_count_for_class(self, layer, class_name):
        """Update segment count based on actual highest segment_id in layer"""
        try:
            if not layer or not layer.isValid():
                return

            max_id = 0
            for feature in layer.getFeatures():
                try:
                    segment_id = feature.attribute("segment_id")
                    if segment_id is not None and isinstance(segment_id, int):
                        max_id = max(max_id, segment_id)
                except:
                    pass

            self.segment_counts[class_name] = max_id
        except RuntimeError:
            # Layer has been deleted
            if class_name in self.segment_counts:
                del self.segment_counts[class_name]

    def _process_segmentation_result(self, mask_or_result, mask_transform, debug_info):
        """Enhanced to handle both single and individual batch results"""

        # Check if this is individual batch processing
        if debug_info.get('individual_processing', False):
            # FIXED: Handle batch result object correctly
            if isinstance(mask_or_result, dict) and 'individual_masks' in mask_or_result:
                return self._process_individual_batch_results(mask_or_result, mask_transform, debug_info)
            else:
                raise ValueError(f"Expected batch result with 'individual_masks', got: {type(mask_or_result)}")

        # Original single mask processing
        return self._process_single_mask_result(mask_or_result, mask_transform, debug_info)

    def _check_spatial_duplicates(self, new_features, existing_layer, overlap_threshold=0.5):
        """Check for spatial duplicates against existing features in the layer"""
        if not existing_layer or not existing_layer.isValid():
            return new_features

        # Get existing features in the area
        filtered_features = []

        for new_feature in new_features:
            new_geom = new_feature.geometry()
            is_duplicate = False

            # Check against existing features
            for existing_feature in existing_layer.getFeatures():
                existing_geom = existing_feature.geometry()

                # Calculate intersection
                if new_geom.intersects(existing_geom):
                    intersection = new_geom.intersection(existing_geom)
                    intersection_area = intersection.area()
                    new_area = new_geom.area()

                    # If overlap is significant, consider it a duplicate
                    if new_area > 0 and (intersection_area / new_area) > overlap_threshold:
                        is_duplicate = True
                        break

            if not is_duplicate:
                filtered_features.append(new_feature)

        removed_count = len(new_features) - len(filtered_features)
        if removed_count > 0:
            print(f"🚫 Removed {removed_count} spatial duplicates (overlap > {overlap_threshold*100}%)")

        return filtered_features

    def _process_individual_batch_results(self, result_data, mask_transform, debug_info):
        """Process multiple individual masks from batch segmentation - INDIVIDUAL OBJECTS ONLY"""
        print(f"🔍 _process_individual_batch_results called")
        print(f"  📊 Result data keys: {list(result_data.keys())}")
        print(f"  📊 Current class: {self.current_class}")

        individual_masks = result_data.get('individual_masks', [])
        print(f"  📊 Individual masks found: {len(individual_masks)}")

        if not individual_masks:
            print(f"❌ No individual masks found in result_data")
            self._update_status("No individual objects found", "warning")
            return

        print(f"✅ Processing {len(individual_masks)} individual masks")

        # Initialize batch undo tracking
        self._current_batch_undo = []

        # Save debug info
        filename_base = None
        if self.save_debug_masks:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            class_prefix = f"{self.current_class}_" if self.current_class else ""
            filename_base = f"batch_{class_prefix}bbox_{self.bbox.width():.1f}x{self.bbox.height():.1f}_{timestamp}"

        # Process each individual mask SEPARATELY - NO COMBINING
        successful_objects = 0

        for obj_idx, mask in enumerate(individual_masks):
            try:
                # Save individual debug mask if enabled
                if self.save_debug_masks and filename_base:
                    individual_filename = f"{filename_base}_obj{obj_idx+1}.png"
                    individual_filename = "".join(c for c in individual_filename if c.isalnum() or c in "._-")
                    mask_path = self.mask_save_dir / individual_filename
                    try:
                        cv2.imwrite(str(mask_path), mask)
                    except Exception as e:
                        print(f"Failed to save debug mask for object {obj_idx+1}: {e}")

                # FIXED: Process this individual mask and add directly to layer
                print(f"🔍 Converting mask {obj_idx+1} to features...")
                features = self._convert_mask_to_features(mask, mask_transform)
                print(f"  📊 Generated {len(features) if features else 0} features")

                if features:
                    # 🚫 Check for spatial duplicates before adding
                    print(f"  🚫 Checking for spatial duplicates...")
                    result_layer = self._get_or_create_class_layer(self.current_class)
                    print(f"  📋 Result layer: {result_layer.name() if result_layer else 'None'}")

                    features = self._check_spatial_duplicates(features, result_layer, self.duplicate_threshold)
                    print(f"  ✅ After duplicate check: {len(features)} features remain")

                    if features:  # Only add if not duplicates
                        print(f"  📍 Adding {len(features)} features to layer...")
                        # Add each individual object immediately to avoid combining
                        self._add_individual_features_to_layer(features, debug_info, obj_idx + 1)
                        successful_objects += 1
                        print(f"  ✅ Successfully added object {obj_idx+1}")
                    else:
                        print(f"  ❌ No features to add after duplicate filtering")
                else:
                    print(f"  ❌ No features generated from mask {obj_idx+1}")

            except Exception as e:
                print(f"Error processing individual object {obj_idx+1}: {e}")
                continue


        if successful_objects == 0:
            self._update_status("No valid features generated from objects", "warning")
            return

        # Clear visual feedback after batch processing completes
        if self.current_mode == 'point':
            self.pointTool.clear_feedback()
        elif self.current_mode == 'bbox':
            self.bboxTool.clear_feedback()

        # Add batch results to undo stack
        if hasattr(self, '_current_batch_undo') and self._current_batch_undo:
            self.undo_stack.append((self.current_class, self._current_batch_undo))
            self.undoBtn.setEnabled(True)

        # Update status
        undo_hint = " (↶ Undo available)" if successful_objects > 0 else ""
        source_info = debug_info.get('source_layer', 'unknown')[:15]
        self._update_status(
            f"✅ Added {successful_objects} individual [{self.current_class}] objects from {source_info}!{undo_hint}", "info")
        self._update_stats()

    def _add_individual_features_to_layer(self, features, debug_info, object_number):
        """Add individual features to layer separately to avoid combining"""
        current_raster = self.iface.activeLayer()
        if isinstance(current_raster, QgsRasterLayer):
            self.original_raster_layer = current_raster

        try:
            result_layer = self._get_or_create_class_layer(self.current_class)
            if not result_layer or not result_layer.isValid():
                self._update_status("Failed to create or access layer", "error")
                return

            # Get the next available segment ID
            next_segment_id = self._get_next_segment_id(result_layer, self.current_class)

            # Enhanced attributes with layer tracking
            timestamp_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            crop_info = debug_info.get('crop_size', 'unknown') if self.current_mode == 'bbox' else debug_info.get('actual_crop', 'unknown')
            class_color = self.classes.get(self.current_class, {}).get('color', '128,128,128')
            canvas_scale = self.canvas.scale()
            source_layer_name = debug_info.get('source_layer', 'unknown')
            layer_crs = debug_info.get('layer_crs', 'unknown')

            # Add batch info with object number
            batch_info = f"batch_obj_{object_number}"

            # Set enhanced attributes for features
            import math as _math
            for i, feat in enumerate(features):
                geom = feat.geometry()
                area_m2 = geom.area() if geom and not geom.isEmpty() else 0.0
                diameter_cm = 2.0 * _math.sqrt(area_m2 / _math.pi) * 100.0 if area_m2 > 0 else 0.0
                perimeter = geom.length() if geom and not geom.isEmpty() else 0.0
                circularity = (4.0 * _math.pi * area_m2) / (perimeter * perimeter) if perimeter > 0 else 0.0
                feat.setAttributes([
                    next_segment_id + i,
                    self.current_class,
                    class_color,
                    batch_info,
                    timestamp_str,
                    f"batch_obj_{object_number}.png" if self.save_debug_masks else "debug_disabled",
                    crop_info,
                    canvas_scale,
                    source_layer_name,
                    layer_crs,
                    round(area_m2, 4),
                    round(diameter_cm, 2),
                    round(circularity, 4)
                ])

            # Add features and track for undo
            result_layer.startEditing()
            success = result_layer.dataProvider().addFeatures(features)
            result_layer.commitChanges()

            if success:
                # Update tracking
                self.segment_counts[self.current_class] = next_segment_id + len(features) - 1

                # Track for undo - INDIVIDUAL TRACKING
                all_features = list(result_layer.getFeatures())
                new_feature_ids = [f.id() for f in all_features[-len(features):]]

                # Store individual object for undo (not combined)
                if hasattr(self, '_current_batch_undo'):
                    self._current_batch_undo.extend(new_feature_ids)
                else:
                    self._current_batch_undo = new_feature_ids

            result_layer.updateExtents()
            result_layer.triggerRepaint()

            # Keep the source raster selected
            if self.keep_raster_selected and self.original_raster_layer:
                self.iface.setActiveLayer(self.original_raster_layer)

            # Update layer name with source info
            total_features = result_layer.featureCount()
            color_info = f" [RGB:{class_color}]"
            source_info = f" [{source_layer_name[:10]}]" if source_layer_name != 'unknown' else ""
            new_layer_name = f"SAM_{self.current_class}{source_info} ({total_features}){color_info}"
            result_layer.setName(new_layer_name)

        except Exception as e:
            self._update_status(f"Error adding individual features: {e}", "error")
            return

    def _process_multiple_masks_result(self, masks, mask_transform, debug_info):
        """Process multiple masks together for text/similar mode (single undo)"""
        # Optional max-size filter for text/similar mode (only when real-world filtering is enabled)
        max_area_px = None
        if self.real_world_filter_enabled and self.gsd_cm_per_px > 0:
            real_world_filters = self._get_effective_pixel_filters()
            if real_world_filters and 'max_area_px' in real_world_filters:
                max_area_px = int(real_world_filters['max_area_px'])

        if max_area_px is not None:
            filtered_masks = []
            filtered_out = 0
            for mask in masks:
                try:
                    area_px = int(np.sum(mask > 0))
                except Exception:
                    area_px = 0
                if area_px <= max_area_px:
                    filtered_masks.append(mask)
                else:
                    filtered_out += 1
            if filtered_out > 0:
                print(f"📏 Real-world max-area filter: removed {filtered_out} masks > {max_area_px}px²")
            masks = filtered_masks

        # Convert all masks to features
        all_features = []
        for i, mask in enumerate(masks):
            features = self._convert_mask_to_features(mask, mask_transform)
            if features:
                all_features.extend(features)
                print(f"  Mask {i+1}/{len(masks)}: {len(features)} features")

        if not all_features:
            self._update_status("No segments found", "warning")
            return

        # Add ALL features to layer in one call (single undo entry)
        self._add_features_to_layer(all_features, debug_info, len(masks))

        # Update status
        undo_hint = " (↶ Undo available)"
        source_info = debug_info.get('source_layer', 'unknown')[:15]
        self._update_status(
            f"✅ Added {len(all_features)} [{self.current_class}] polygons from {len(masks)} objects on {source_info}!{undo_hint}", "info")
        self._update_stats()

    def _process_single_mask_result(self, mask, mask_transform, debug_info):
        """Process a single combined mask (original behavior)"""
        # Save mask image for traceability (ONLY if debug enabled)
        filename = None
        if self.save_debug_masks:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            class_prefix = f"{self.current_class}_" if self.current_class else ""

            if self.current_mode == 'point':
                filename = f"mask_{class_prefix}point_{self.point.x():.1f}_{self.point.y():.1f}_{timestamp}.png"
            else:
                filename = f"mask_{class_prefix}bbox_{self.bbox.width():.1f}x{self.bbox.height():.1f}_{timestamp}.png"

            filename = "".join(c for c in filename if c.isalnum() or c in "._-")
            mask_path = self.mask_save_dir / filename

            try:
                cv2.imwrite(str(mask_path), mask)
            except Exception as e:
                self._update_status(f"Failed to save debug mask: {e}", "warning")
                filename = "save_failed"

        # Convert mask to features
        features = self._convert_mask_to_features(mask, mask_transform)
        if not features:
            self._update_status("No segments found", "warning")
            return

        # Add features to layer
        self._add_features_to_layer(features, debug_info, 1, filename)

        # Update status
        multi_points = self.current_request.get('multi_points') if self.current_request else None
        point_info = f" ({len(multi_points)} points)" if multi_points else ""
        undo_hint = " (↶ Undo available)" if len(features) > 0 else ""
        source_info = debug_info.get('source_layer', 'unknown')[:15]
        self._update_status(
            f"✅ Added {len(features)} [{self.current_class}] polygons{point_info} from {source_info}!{undo_hint}", "info")
        self._update_stats()

    def _get_or_create_class_layer(self, class_name):
        """Enhanced layer creation that uses current raster CRS"""
        # Check if we have a tracked layer for this class
        if class_name in self.result_layers:
            layer = self.result_layers[class_name]
            try:
                if layer and layer.isValid():
                    return layer
            except RuntimeError:
                del self.result_layers[class_name]
                if class_name in self.segment_counts:
                    del self.segment_counts[class_name]

        # Get the CURRENT active raster layer for CRS
        current_raster = self.iface.activeLayer()
        if not isinstance(current_raster, QgsRasterLayer) or not current_raster.isValid():
            self._update_status("No valid raster layer selected", "error")
            return None

        # Update our stored reference
        self.original_raster_layer = current_raster

        class_info = self.classes.get(class_name, {'color': '128,128,128'})
        color = class_info['color']

        # Use current raster's CRS and add layer info to name
        raster_name = current_raster.name()[:15]  # Truncate long names
        layer_name = f"SAM_{class_name}_{raster_name}_{datetime.datetime.now():%H%M%S}"

        layer = QgsVectorLayer(
            f"Polygon?crs={current_raster.crs().authid()}", 
            layer_name, 
            "memory"
        )

        if not layer.isValid():
            self._update_status(f"Failed to create layer with CRS {current_raster.crs().authid()}", "error")
            return None

        layer.dataProvider().addAttributes([
            QgsField("segment_id", QVariant.Int),
            QgsField("class", QVariant.String),
            QgsField("class_color", QVariant.String),
            QgsField("method", QVariant.String),
            QgsField("timestamp", QVariant.String),
            QgsField("mask_file", QVariant.String),
            QgsField("crop_size", QVariant.String),
            QgsField("canvas_scale", QVariant.Double),
            QgsField("source_layer", QVariant.String),  # Track source raster
            QgsField("layer_crs", QVariant.String),     # Track CRS used
            QgsField("area_m2", QVariant.Double),
            QgsField("diameter_cm", QVariant.Double),
            QgsField("circularity", QVariant.Double)
        ])
        layer.updateFields()

        self._apply_class_style(layer, class_name)

        QgsProject.instance().addMapLayer(layer)
        self.result_layers[class_name] = layer
        self.segment_counts[class_name] = 0

        # Keep the current raster selected
        if self.keep_raster_selected and current_raster:
            self.iface.setActiveLayer(current_raster)

        return layer

    def _apply_class_style(self, layer, class_name):
        try:
            class_info = self.classes.get(class_name, {'color': '128,128,128'})
            color = class_info['color']

            try:
                r, g, b = [int(c.strip()) for c in color.split(',')]
            except:
                r, g, b = 128, 128, 128

            symbol = QgsFillSymbol.createSimple({
                'color': f'{r},{g},{b},180',
                'outline_color': f'{r},{g},{b},255',
                'outline_width': '1.5',
                'outline_style': 'solid'
            })

            layer.renderer().setSymbol(symbol)
            layer.setOpacity(0.85)
            layer.triggerRepaint()

        except Exception as e:
            print(f"Color application failed for {class_name}: {e}")

    def _undo_last_polygon(self):
        if not self.undo_stack:
            self._update_status("No polygons to undo", "warning")
            return

        class_name, feature_ids = self.undo_stack.pop()

        if class_name not in self.result_layers:
            self._update_status(f"Class layer {class_name} not found", "error")
            return

        layer = self.result_layers[class_name]

        try:
            layer.startEditing()
            removed_count = 0
            for feature_id in feature_ids:
                if layer.deleteFeature(feature_id):
                    removed_count += 1

            layer.commitChanges()
            layer.updateExtents()
            layer.triggerRepaint()

            # Update segment count based on actual features (FIXED)
            self._update_segment_count_for_class(layer, class_name)

            total_features = layer.featureCount()
            class_color = self.classes.get(class_name, {}).get('color', '128,128,128')
            color_info = f" [RGB:{class_color}]"
            new_layer_name = f"SAM_{class_name} ({total_features} parts){color_info}"
            layer.setName(new_layer_name)

            self._update_stats()
            self._update_status(f"↶ Undid {removed_count} polygons from [{class_name}]", "info")

            if not self.undo_stack:
                self.undoBtn.setEnabled(False)

        except Exception as e:
            self._update_status(f"Failed to undo: {e}", "error")
            if layer.isEditable():
                layer.rollBack()

    def _export_all_classes(self):
        # Collect all SAM_ vector layers from the project (not just current-session ones)
        layers_to_export = []
        for layer in QgsProject.instance().mapLayers().values():
            try:
                if (isinstance(layer, QgsVectorLayer) and
                        layer.isValid() and
                        layer.name().startswith("SAM_") and
                        layer.featureCount() > 0):
                    layers_to_export.append(layer)
            except RuntimeError:
                continue

        if not layers_to_export:
            self._update_status("No segments to export!", "warning")
            return

        exported_count = 0
        for layer in layers_to_export:
            class_name = layer.name()  # Use full layer name as file name
            if self._export_layer(layer, class_name):
                exported_count += 1

        if exported_count > 0:
            self._update_status(
                f"💾 Exported {exported_count} layer(s) to {self.export_save_dir}", "info")
        else:
            self._update_status("Export failed — check output folder", "warning")

    def _export_selected_layers(self):
        selected = self.iface.layerTreeView().selectedLayers()
        layers_to_export = []
        for layer in selected:
            try:
                if (isinstance(layer, QgsVectorLayer) and
                        layer.isValid() and
                        layer.name().startswith("SAM_") and
                        layer.featureCount() > 0):
                    layers_to_export.append(layer)
            except RuntimeError:
                continue

        if not layers_to_export:
            self._update_status("No SAM layers selected in Layers panel", "warning")
            return

        exported_count = 0
        for layer in layers_to_export:
            if self._export_layer(layer, layer.name()):
                exported_count += 1

        if exported_count > 0:
            self._update_status(
                f"💾 Exported {exported_count} layer(s) to {self.export_save_dir}", "info")
        else:
            self._update_status("Export failed — check output folder", "warning")

    def _export_layer(self, layer, class_name):
        try:
            fmt_name = self.formatComboBox.currentText()
            fmt = self.EXPORT_FORMATS[fmt_name]
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            export_name = f"SAM_{class_name}_{timestamp}{fmt['ext']}"
            export_path = str(self.export_save_dir / export_name)

            error = QgsVectorFileWriter.writeAsVectorFormat(
                layer, export_path, "utf-8", layer.crs(), fmt['driver'])

            if error[0] == QgsVectorFileWriter.NoError:
                print(f"💾 Exported {class_name}: {export_path}")
                return True
            else:
                print(f"❌ Export failed for {class_name}: {error}")
                return False

        except Exception as e:
            print(f"❌ Export error for {class_name}: {e}")
            return False

    def _update_stats(self):
        """Update statistics display"""
        total_segments = 0
        total_classes = 0

        try:
            # Check all layers in project, not just tracked ones
            all_layers = QgsProject.instance().mapLayers().values()
            for layer in all_layers:
                try:
                    if (isinstance(layer, QgsVectorLayer) and 
                        layer.isValid() and 
                        layer.name().startswith("SAM_") and 
                        layer.featureCount() > 0):
                        total_segments += layer.featureCount()
                        total_classes += 1
                except RuntimeError:
                    # Layer being deleted, skip
                    continue

            self.statsLabel.setText(
                f"Total Segments: {total_segments} | Classes: {total_classes}")
        except Exception as e:
            # Fallback to simple display
            self.statsLabel.setText("Total Segments: ? | Classes: ?")

    def _on_segmentation_progress(self, message):
        self._update_status(message, "processing")

    def _on_segmentation_error(self, error_message):
        self._update_status(f"❌ {error_message}", "error")
        self._set_ui_enabled(True)
        self.is_processing = False  # Reset processing state
        if hasattr(self, 'worker') and self.worker:
            self.worker.deleteLater()
            self.worker = None

        # Process next item in queue
        self._process_queue()

    def _clear_queue(self):
        """Clear the processing queue"""
        cleared_count = len(self.processing_queue)
        self.processing_queue.clear()
        if cleared_count > 0:
            self._update_status(f"🗑️ Cleared {cleared_count} items from queue", "info")

    def _get_queue_status(self):
        """Get current queue status"""
        return f"Queue: {len(self.processing_queue)} pending"

    def _cancel_segmentation(self):
        if hasattr(self, 'worker') and self.worker and self.worker.isRunning():
            self.worker.terminate()
            self.worker.wait()
            self.worker.deleteLater()
            self.worker = None
            self._update_status("Segmentation cancelled", "warning")
            self._set_ui_enabled(True)
            self.is_processing = False  # Reset processing state

    def _set_ui_enabled(self, enabled):
        self.pointModeBtn.setEnabled(enabled)
        self.bboxModeBtn.setEnabled(enabled)
        self.classComboBox.setEnabled(enabled)
        self.addClassBtn.setEnabled(enabled)
        self.editClassBtn.setEnabled(enabled)
        self.exportBtn.setEnabled(enabled)
        self.selectFolderBtn.setEnabled(True)
        self.saveDebugSwitch.setEnabled(True)

        if enabled and self.undo_stack:
            self.undoBtn.setEnabled(True)
        elif not enabled:
            pass  # Keep undo available during processing
        else:
            self.undoBtn.setEnabled(False)

        if hasattr(self, 'progressBar'):
            self.progressBar.setVisible(not enabled)

        if not enabled:
            self.setCursor(Qt.WaitCursor)
        else:
            self.setCursor(Qt.ArrowCursor)

    def _update_status(self, message, status_type="info"):
        # Store for filter indicator refresh
        self._last_status_text = message
        self._last_status_type = status_type

        color_styles = {
            "info": "background: #ECFDF3; color: #027A48; border: 1px solid #D1FADF;",
            "warning": "background: #FFFBEB; color: #DC6803; border: 1px solid #FED7AA;",
            "error": "background: #FEF2F2; color: #DC2626; border: 1px solid #FECACA;",
            "processing": "background: #EFF8FF; color: #1570EF; border: 1px solid #B2DDFF;"
        }
        color_style = color_styles.get(status_type, color_styles["info"])

        # Append active filter summary if any
        filter_summary = getattr(self, '_active_filter_summary', '')
        display = message
        if filter_summary:
            display = f"{message}\n🔹 Filters: {filter_summary}"

        self.statusLabel.setText(display)
        self.statusLabel.setStyleSheet(f"""
            padding: 14px; border-radius: 8px; font-size: 16px; font-weight: 500;
            {color_style}
        """) 

    def closeEvent(self, event):
        """Handle close event to clean up tools"""
        try:
            # Reset to original map tool if we changed it
            if self.original_map_tool:
                self.canvas.setMapTool(self.original_map_tool)

            # Clean up rubber bands
            if hasattr(self, 'pointTool'):
                self.pointTool.clear_feedback()
            if hasattr(self, 'bboxTool'):
                self.bboxTool.clear_feedback()

        except Exception as e:
            print(f"Error during cleanup: {e}")

        super().closeEvent(event)

    def _get_plugin_version(self):
        """Read version from metadata.txt"""
        try:
            import os
            metadata_path = os.path.join(os.path.dirname(__file__), 'metadata.txt')
            with open(metadata_path, 'r') as f:
                for line in f:
                    if line.startswith('version='):
                        return line.split('=')[1].strip()
            return "1.0.0"  # Fallback version
        except Exception:
            return "1.0.0"  # Fallback version

class SegSamDialog(QtWidgets.QDialog):
    def __init__(self, iface, parent=None):
        super().__init__(parent)
        self.iface = iface
        self.control_panel = None

        layout = QtWidgets.QVBoxLayout(self)
        label = QtWidgets.QLabel("GeoOSAM Control Panel")
        label.setStyleSheet(
            "font-size: 18px; font-weight: bold; padding: 10px;")
        layout.addWidget(label)

        show_panel_btn = QtWidgets.QPushButton("Show Control Panel")
        show_panel_btn.clicked.connect(self._show_control_panel)
        layout.addWidget(show_panel_btn)

        close_btn = QtWidgets.QPushButton("Close")
        close_btn.clicked.connect(self.close)
        layout.addWidget(close_btn)

        # Get version from metadata
        version = self._get_plugin_version()
        self.setWindowTitle(f"Version: {version}")
        self.resize(280, 140)

    def _show_control_panel(self):
        if not self.control_panel:
            self.control_panel = GeoOSAMControlPanel(self.iface)
            self.iface.addDockWidget(
                Qt.RightDockWidgetArea, self.control_panel)
        self.control_panel.show()
        self.control_panel.raise_()
        self.close()

    def _get_plugin_version(self):
        """Read version from metadata.txt"""
        try:
            import os
            metadata_path = os.path.join(os.path.dirname(__file__), 'metadata.txt')
            with open(metadata_path, 'r') as f:
                for line in f:
                    if line.startswith('version='):
                        return line.split('=')[1].strip()
            return "1.0.0"  # Fallback version
        except Exception:
            return "1.0.0"  # Fallback version
