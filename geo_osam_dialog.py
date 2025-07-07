import sys
import os
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
    QgsVectorFileWriter
)
from qgis.gui import QgsRubberBand, QgsMapTool
from qgis.PyQt import QtWidgets, QtCore, QtGui

# fmt: off
plugin_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(plugin_dir)
from helpers import create_detection_helper
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

# Ultralytics SAM2.1_B setup
SAM21B_AVAILABLE = False

try:
    from ultralytics import SAM
    test_model = SAM('sam2.1_b.pt') # mobile_sam.pt
    SAM21B_AVAILABLE = True
    print("✅ Ultralytics SAM2.1_B available")

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
                        point = point_coords[0]
                        x, y = int(point[0]), int(point[1])
                        results = self.model.predict(
                            source=self.image,
                            points=[[x, y]],
                            labels=[1],
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
                print(f"MobileSAM prediction error: {e}")
                return self._empty_result()

        def _empty_result(self):
            empty_mask = np.zeros(
                (self.image.shape[0], self.image.shape[1]), dtype=np.uint8)
            return [empty_mask], [0.0], None

except ImportError:
    print("⚠️ Ultralytics not available - install with: /usr/bin/python3 -m pip install --user ultralytics")
    SAM21B_AVAILABLE = False
except Exception as e:
    print(f"⚠️ Ultralytics SAM2.1_B failed: {e}")
    SAM21B_AVAILABLE = False

if SAM21B_AVAILABLE:
    print("   Using fast Ultralytics SAM2.1_B")
else:
    print("   Falling back to SAM 2.1")

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
    if _THREADS_CONFIGURED:
        return torch.get_num_threads()

    import multiprocessing
    num_cores = multiprocessing.cpu_count()
    optimal_threads = max(4, int(num_cores * 0.75)) if num_cores >= 16 else \
        max(4, num_cores - 2) if num_cores >= 8 else \
        max(1, num_cores - 1)

    torch.set_num_interop_threads(min(4, optimal_threads // 2))
    torch.set_num_threads(optimal_threads)

    os.environ["OMP_NUM_THREADS"] = str(optimal_threads)
    os.environ["MKL_NUM_THREADS"] = str(optimal_threads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(optimal_threads)

    _THREADS_CONFIGURED = True
    return optimal_threads

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
        url = "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2.1_hiera_tiny.pt"
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
    """Detect best available device and model"""
    cores = None
    try:
        if torch.cuda.is_available() and not os.getenv("GEOOSAM_FORCE_CPU"):
            gpu_props = torch.cuda.get_device_properties(0)
            if gpu_props.total_memory / 1024**3 >= 3:  # 3GB minimum
                device = "cuda"
                model_choice = "SAM2"
                print(
                    f"🎮 GPU detected: {torch.cuda.get_device_name(0)} - using SAM2")
                return device, model_choice, cores

        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device, model_choice = "mps", "SAM2"
            print("🍎 Apple Silicon GPU detected - using SAM2")
            return device, model_choice, cores

        else:
            device = "cpu"
            model_choice = "SAM2.1_B" if SAM21B_AVAILABLE else "SAM2"
            cores = setup_pytorch_performance()
            print(f"💻 CPU detected - using {model_choice} ({cores} cores)")
            return device, model_choice, cores

    except Exception as e:
        print(f"⚠️ Device detection failed: {e}, falling back to CPU")
        device, model_choice = "cpu", "SAM2.1_B" if SAM21B_AVAILABLE else "SAM2"
        cores = setup_pytorch_performance()
        return device, model_choice, cores


class OptimizedSAM2Worker(QThread):
    finished = pyqtSignal(object)
    error = pyqtSignal(str)
    progress = pyqtSignal(str)

    def __init__(self, predictor, arr, mode, model_choice="SAM2", point_coords=None,
                 point_labels=None, box=None, mask_transform=None, debug_info=None, device="cpu",
                 min_object_size=50, max_objects=20, arr_multispectral=None):
        super().__init__()
        self.predictor = predictor
        self.arr = arr
        self.arr_multispectral = arr_multispectral
        self.mode = mode
        self.model_choice = model_choice
        self.point_coords = point_coords
        self.point_labels = point_labels
        self.box = box
        self.mask_transform = mask_transform
        self.debug_info = debug_info or {}
        self.device = device
        self.min_object_size = min_object_size
        self.max_objects = max_objects

    def run(self):
        try:
            self.progress.emit(f"🖼️ Setting image for {self.model_choice}...")

            # SAFETY: Check if thread should continue
            if self.isInterruptionRequested():
                return

            self.predictor.set_image(self.arr)

            if self.mode == "bbox_batch":
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

        # Use multi-spectral image for vegetation detection if available
        if multispectral_image is not None and class_name == "Vegetation":
            detection_image = multispectral_image
            print(f"🔍 Detecting {class_name} candidates in {detection_image.shape} region (multi-spectral)")
        else:
            detection_image = image
            print(f"🔍 Detecting {class_name} candidates in {detection_image.shape} region")

        # Crop image to bbox region
        bbox_image = detection_image[y1:y2, x1:x2].copy()

        # Use helper for detection
        helper = create_detection_helper(class_name, self.min_object_size, self.max_objects)
        return helper.detect_candidates(bbox_image, bbox)




    def _validate_mask_for_class(self, mask, class_name, center_point):
        """Validate segmented mask based on class-specific criteria"""
        try:
            # Use helper for validation
            helper = create_detection_helper(class_name, self.min_object_size, self.max_objects)
            valid_masks = helper.process_sam_mask(mask)
            return len(valid_masks) > 0

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

            # Segment each candidate point individually
            individual_masks = []
            successful_detections = 0

            # Limit to max_objects to prevent too many detections
            candidates_to_process = candidate_points[:self.max_objects]

            for i, (px, py) in enumerate(candidates_to_process):
                try:
                    self.progress.emit(f"🎯 Segmenting object {i+1}/{len(candidates_to_process)}...")

                    print(f"🔍 Processing candidate {i+1}: point ({px}, {py})")

                    # Run point segmentation using existing SAM2 pipeline
                    point_coords = np.array([[px, py]])
                    point_labels = np.array([1])

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

                    # Calculate reasonable max size (10% of image area)
                    image_area = self.arr.shape[0] * self.arr.shape[1]
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

            # Class-aware processing - use existing logic for all classes
            individual_masks = merge_nearby_masks_class_aware(individual_masks, current_class, buffer_px=1)
            individual_masks = dedupe_or_merge_masks_smart(individual_masks, current_class)

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

            print(f"   Result keys: {list(result.keys())}")
            print(f"   Final individual_processing: {result['debug_info'].get('individual_processing')}")

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
        """Class-aware object validation"""
        # Basic size filter
        if component_area < self.min_object_size:
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

        # CLASS-SPECIFIC VALIDATION
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

    def canvasReleaseEvent(self, e):
        map_point = self.canvas.getCoordinateTransform().toMapCoordinates(e.pos())
        self.point_rubber.reset(QgsWkbTypes.PointGeometry)
        self.point_rubber.addPoint(map_point, True)
        self.canvas.refresh()
        self.cb(map_point)

    def deactivate(self):
        self.point_rubber.reset(QgsWkbTypes.PointGeometry)
        super().deactivate()

    def clear_feedback(self):
        self.point_rubber.reset(QgsWkbTypes.PointGeometry)
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
            'batch_defaults': {'min_size': 200, 'max_objects': 10}
        },
        'Buildings'   : {
            'color': '220,20,60',   
            'description': 'Residential & commercial structures',
            'batch_defaults': {'min_size': 150, 'max_objects': 20}
        },
        'Commercial'  : {
            'color': '135,206,250', 
            'description': 'Shopping and business districts',
            'batch_defaults': {'min_size': 200, 'max_objects': 15}
        },
        'Industrial'  : {
            'color': '128,0,128',   
            'description': 'Factories and warehouses',
            'batch_defaults': {'min_size': 400, 'max_objects': 8}
        },
        'Other'       : {
            'color': '148,0,211',   
            'description': 'Unclassified objects',
            'batch_defaults': {'min_size': 50, 'max_objects': 25}
        },
        'Parking'     : {
            'color': '255,140,0',   
            'description': 'Parking lots and areas',
            'batch_defaults': {'min_size': 150, 'max_objects': 15}
        },
        'Residential' : {
            'color': '255,105,180', 
            'description': 'Housing areas',
            'batch_defaults': {'min_size': 50, 'max_objects': 60}
        },
        'Roads'       : {
            'color': '105,105,105', 
            'description': 'Streets, highways, and pathways',
            'batch_defaults': {'min_size': 200, 'max_objects': 10}
        },
        'Vessels'     : {
            'color': '0,206,209',   
            'description': 'Boats, ships',
            'batch_defaults': {'min_size': 40, 'max_objects': 35}
        },
        'Vehicle'     : {
            'color': '255,69,0',    
            'description': 'Cars, trucks, and buses',
            'batch_defaults': {'min_size': 20, 'max_objects': 50}
        },
        'Vegetation'  : {
            'color': '34,139,34',   
            'description': 'Trees, grass, and parks',
            'batch_defaults': {'min_size': 30, 'max_objects': 100}
        },
        'Water'       : {
            'color': '30,144,255',  
            'description': 'Rivers, lakes, and ponds',
            'batch_defaults': {'min_size': 500, 'max_objects': 8}   # Large areas
        }
    }

    EXTRA_COLORS = [
        '50,205,50', '255,20,147', '255,165,0', '186,85,211', '0,128,128',
        '255,192,203', '165,42,42', '0,250,154', '255,0,255', '127,255,212'
    ]

    def __init__(self, iface, parent=None):
        super().__init__("", parent)
        self.iface = iface
        self.canvas = iface.mapCanvas()

        # Initialize device and model
        self.device, self.model_choice, self.num_cores = detect_best_device()
        self._init_sam_model()

        # Setup docking
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
        self.shapefile_save_dir = None
        self.mask_save_dir = None
        self.save_debug_masks = False

        # Undo functionality
        self.undo_stack = []

        # Initialize
        self._init_save_directories()
        self.pointTool = EnhancedPointClickTool(self.canvas, self._point_done)
        self.bboxTool = EnhancedBBoxClickTool(self.canvas, self._bbox_done)
        self.original_map_tool = None

        # Batch mode settings
        self.batch_mode_enabled = False
        self.min_object_size = 50  # Minimum pixels for valid object
        self.max_objects = 20  # Prevent too many small objects
        self.duplicate_threshold = 0.85  # Spatial overlap threshold for duplicates (very lenient for shape-based detection)

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
        """Initialize the selected SAM model"""
        plugin_dir = os.path.dirname(os.path.abspath(__file__))

        if self.model_choice == "SAM2.1_B":
            self._init_sam21b_model()
        else:
            self._init_sam2_model(plugin_dir)

    def _init_sam2_model(self, plugin_dir):
        """Initialize SAM2 model"""
        checkpoint_path = os.path.join(
            plugin_dir, "sam2", "checkpoints", "sam2.1_hiera_tiny.pt")

        if not os.path.exists(checkpoint_path):
            if not auto_download_checkpoint():
                if not show_checkpoint_dialog(self):
                    raise Exception(
                        "SAM2 checkpoint required but not available")

        if GlobalHydra.instance().is_initialized():
            GlobalHydra.instance().clear()

        try:
            with initialize_config_module(config_module="sam2.configs"):
                sam_model = build_sam2(
                    "sam2.1/sam2.1_hiera_t", checkpoint_path, device=self.device)

                if self.device == "cuda":
                    sam_model = sam_model.cuda()

                sam_model.eval()
                if self.device == "cpu":
                    try:
                        sam_model = torch.jit.optimize_for_inference(sam_model)
                    except:
                        pass

                self.predictor = SAM2ImagePredictor(sam_model)
                print(f"✅ SAM2 model loaded on {self.device}")

        except Exception as e:
            print(f"❌ Failed to load SAM2: {e}")
            raise

    def _init_sam21b_model(self):
        """Initialize Ultralytics SAM2.1_B model"""
        try:
            from ultralytics import SAM
            sam21b_model = SAM('sam2.1_b.pt')  # mobile_sam.pt
            self.predictor = UltralyticsPredictor(sam21b_model)
            print(f"✅ Ultralytics SAM2.1_B loaded successfully")
        except Exception as e:
            print(f"❌ Failed to load SAM2.1_B: {e}, falling back to SAM2")
            self.model_choice = "SAM2"
            self._init_sam2_model(os.path.dirname(os.path.abspath(__file__)))

    def _init_save_directories(self):
        """Initialize output directories"""
        self.shapefile_save_dir = pathlib.Path.home() / "GeoOSAM_shapefiles"
        self.mask_save_dir = pathlib.Path.home() / "GeoOSAM_masks"
        self.shapefile_save_dir.mkdir(exist_ok=True)

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

    def _setup_ui(self):
        # Force small font size regardless of DPI detection
        base_font_size = 9  # Normal size
        self.setFont(QtGui.QFont("Segoe UI", base_font_size))
        print(f"UI setup using forced font size: {base_font_size}pt")

        # --- Dock features: standard QGIS close/float/move
        self.setFeatures(
            QtWidgets.QDockWidget.DockWidgetClosable |
            QtWidgets.QDockWidget.DockWidgetFloatable |
            QtWidgets.QDockWidget.DockWidgetMovable
        )

        # --- Scrollable, responsive area
        scroll_area = QtWidgets.QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)  # Disable horizontal scroll
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)     # Only show vertical when needed
        scroll_area.setStyleSheet("QScrollArea { border: none; background: #f8f9fa; }")
        self.setWidget(scroll_area)

        main_widget = QtWidgets.QWidget()
        main_widget.setFont(QtGui.QFont("Segoe UI", base_font_size))
        scroll_area.setWidget(main_widget)

        main_layout = QtWidgets.QVBoxLayout(main_widget)
        main_layout.setSpacing(12)                    # Reduced spacing
        main_layout.setContentsMargins(15, 15, 15, 15)  # Reduced margins
        main_widget.setStyleSheet("background: transparent; color: #344054;")

        # Allow flexible resizing
        self.setMinimumWidth(300)
        self.setMaximumWidth(450)   # Add max width back
        self.setMinimumHeight(500)
        self.resize(350, 700)       # Set initial size

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
                icon_label.setStyleSheet("font-size: 12px; margin-top: 1px;")  # Reduced from 22px
                header_label = QtWidgets.QLabel(f"<b>{title}</b>")
                header_label.setStyleSheet("font-size: 13px; color: #101828;")  # Reduced from 20px
                header_layout.addWidget(icon_label)
                header_layout.addWidget(header_label)
                header_layout.addStretch()
                card_layout.addLayout(header_layout)
            return card, card_layout

        # --- Title and Device Header ---
        title_label = QtWidgets.QLabel("GeoOSAM Control Panel")
        title_label.setStyleSheet("font-size: 14px; font-weight: bold; color: #1D2939;")
        title_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(title_label)

        device_icon = "🎮" if "cuda" in self.device else "🖥️"
        device_info = f"{device_icon} {self.device.upper()} | {self.model_choice}"
        if getattr(self, "num_cores", None):
            device_info += f" ({self.num_cores} cores)"
        device_label = QtWidgets.QLabel(device_info)
        device_label.setStyleSheet("font-size: 12px; color: #475467;")  # Reduced from 18px
        device_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(device_label)

        separator = QtWidgets.QFrame()
        separator.setFrameShape(QtWidgets.QFrame.HLine)
        separator.setStyleSheet("border-top: 1px solid #EAECF0;")
        main_layout.addWidget(separator)

        # --- Output Settings ---
        output_card, output_layout = create_card("Output Settings", "📂")
        folder_layout = QtWidgets.QHBoxLayout()
        self.outputFolderLabel = QtWidgets.QLabel("Default folder")
        self.outputFolderLabel.setStyleSheet("font-size: 11px; color: #475467;")  # Reduced from 18px
        self.selectFolderBtn = QtWidgets.QPushButton("Choose")
        self.selectFolderBtn.setCursor(Qt.PointingHandCursor)
        self.selectFolderBtn.setStyleSheet("""
            QPushButton {
                font-size: 11px; padding: 6px 16px; border-radius: 8px;
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

        debug_layout = QtWidgets.QHBoxLayout()
        debug_label = QtWidgets.QLabel("Save debug masks")
        debug_label.setStyleSheet("font-size: 11px;")  # Reduced from 18px
        self.saveDebugSwitch = Switch()
        debug_layout.addWidget(debug_label)
        debug_layout.addStretch()
        debug_layout.addWidget(self.saveDebugSwitch)
        output_layout.addLayout(debug_layout)
        main_layout.addWidget(output_card)

        # --- Class Selection ---
        class_card, class_layout = create_card("Class Selection", "🏷️")

        # Class dropdown
        self.classComboBox = QtWidgets.QComboBox()
        self.classComboBox.addItem("-- Select Class --", None)
        for class_name in self.classes.keys():
            self.classComboBox.addItem(class_name, class_name)
        self.classComboBox.setStyleSheet("""
            QComboBox {
                padding: 8px 10px; font-size: 11px; border-radius: 7px;
                border: 1px solid #D0D5DD; background: #FFF;
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
            border-radius: 8px; font-size: 11px;
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
                    font-size: 11px; padding: 8px; border-radius: 8px;
                    background: #FFF; border: 1px solid #D0D5DD;
                }
                QPushButton:hover { background: #F9FAFB; }
            """)  # Reduced font-size and padding
            class_btn_layout.addWidget(btn)
        class_layout.addLayout(class_btn_layout)
        main_layout.addWidget(class_card)

        # --- Enhanced Segmentation Mode ---
        mode_card, mode_layout = create_card("Segmentation Mode", "🎯")

        # Point mode button (existing)
        self.pointModeBtn = QtWidgets.QPushButton("Point Mode")
        self.pointModeBtn.setCheckable(True)
        self.pointModeBtn.setChecked(True)
        self.pointModeBtn.setProperty("active", True)

        # Enhanced BBox mode button
        self.bboxModeBtn = QtWidgets.QPushButton("BBox Mode")
        self.bboxModeBtn.setCheckable(True)
        self.bboxModeBtn.setVisible(True)

        # Button group for mutual exclusion
        self.mode_button_group = QtWidgets.QButtonGroup()
        self.mode_button_group.addButton(self.pointModeBtn)
        self.mode_button_group.addButton(self.bboxModeBtn)
        self.mode_button_group.setExclusive(True)

        mode_btn_style = """
            QPushButton {
                font-size: 12px; font-weight: 600; padding: 10px;
                border-radius: 8px; border: 1px solid #D0D5DD;
                background: #FFF;
            }
            QPushButton:hover { background: #F9FAFB; }
            QPushButton[active="true"] {
                color: #FFF; background: #1570EF; border: 1px solid #1570EF;
            }
        """
        self.pointModeBtn.setStyleSheet(mode_btn_style)
        self.bboxModeBtn.setStyleSheet(mode_btn_style)

        # NEW: Batch mode toggle
        batch_layout = QtWidgets.QHBoxLayout()
        batch_label = QtWidgets.QLabel("Batch Segmentation")
        batch_label.setStyleSheet("font-size: 11px;")
        batch_label.setToolTip("Find multiple objects in bbox area")
        self.batchModeSwitch = Switch()
        self.batchModeSwitch.setToolTip("Enable to find multiple objects in bbox")
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
        size_label.setStyleSheet("font-size: 10px; color: #667085;")
        size_label.setFixedWidth(50)  # Fixed width to prevent layout shift
        self.minSizeSpinBox = QtWidgets.QSpinBox()
        self.minSizeSpinBox.setRange(10, 500)
        self.minSizeSpinBox.setValue(50)
        self.minSizeSpinBox.setSuffix("px")
        self.minSizeSpinBox.setFixedWidth(70)  # Fixed width
        self.minSizeSpinBox.setStyleSheet("""
            QSpinBox { 
                font-size: 10px; padding: 2px; 
                border: 1px solid #D0D5DD; border-radius: 3px; 
            }
        """)
        # ENHANCED: Add helpful tooltip with class recommendations
        self.minSizeSpinBox.setToolTip("Minimum object size in pixels\n• Buildings: ~100px\n• Vehicles: ~15px\n• Vessels: ~30px\n• Trees: ~25px")
        size_layout.addWidget(size_label)
        size_layout.addWidget(self.minSizeSpinBox)
        size_layout.addStretch()

        # Max objects setting - ENHANCED WITH TOOLTIPS
        max_layout = QtWidgets.QHBoxLayout()
        max_layout.setSpacing(4)
        max_label = QtWidgets.QLabel("Max obj:")
        max_label.setStyleSheet("font-size: 10px; color: #667085;")
        max_label.setFixedWidth(50)  # Fixed width
        self.maxObjectsSpinBox = QtWidgets.QSpinBox()
        self.maxObjectsSpinBox.setRange(1, 50)
        self.maxObjectsSpinBox.setValue(20)
        self.maxObjectsSpinBox.setFixedWidth(50)  # Fixed width
        self.maxObjectsSpinBox.setStyleSheet("""
            QSpinBox { 
                font-size: 10px; padding: 2px; 
                border: 1px solid #D0D5DD; border-radius: 3px; 
            }
        """)
        # ENHANCED: Add helpful tooltip with class recommendations
        self.maxObjectsSpinBox.setToolTip("Maximum objects to detect\n• Vehicles: ~40\n• Vessels: ~30\n• Trees: ~35\n• Buildings: ~15")
        max_layout.addWidget(max_label)
        max_layout.addWidget(self.maxObjectsSpinBox)
        max_layout.addStretch()

        batch_settings_layout.addLayout(size_layout)
        batch_settings_layout.addLayout(max_layout)

        # ENHANCED: Add helpful hints label
        self.classHintsLabel = QtWidgets.QLabel("Auto-adjusts based on selected class")
        self.classHintsLabel.setStyleSheet("font-size: 9px; color: #9CA3AF; font-style: italic;")
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
        main_layout.addWidget(mode_card)

        # --- Status & Controls Card ---
        status_card, status_layout = create_card("Status & Controls", "⚙️")
        self.statusLabel = QtWidgets.QLabel("Ready to segment")
        self.statusLabel.setWordWrap(True)
        self.statusLabel.setStyleSheet("""
            padding: 10px; border-radius: 8px; font-size: 14px; font-weight: 500;
            background: #ECFDF3; color: #027A48; border: 1px solid #D1FADF;
        """)  # Reduced padding and font-size
        status_layout.addWidget(self.statusLabel)

        self.statsLabel = QtWidgets.QLabel("Total Segments: 0 | Classes: 0")
        self.statsLabel.setStyleSheet(
            "font-size: 10px; color: #475467; margin-top: 3px; margin-bottom: 3px;")  # Reduced from 18px
        status_layout.addWidget(self.statsLabel)

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
                font-size: 11px; font-weight: 600; padding: 10px;
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

        self.exportBtn = QtWidgets.QPushButton("💾 Export All")
        self.exportBtn.setCursor(Qt.PointingHandCursor)
        self.exportBtn.setStyleSheet("""
            QPushButton {
                font-size: 11px; font-weight: 600; padding: 10px;
                border-radius: 8px; color: #FFF;
                background: #027A48; border: 1px solid #027A48;
            }
            QPushButton:hover { background: #039855; }
        """)  # Reduced font-size and padding

        self.exportBtn.setAutoDefault(False)
        self.exportBtn.setDefault(False)
        self.exportBtn.setFocusPolicy(Qt.NoFocus)

        status_layout.addWidget(self.undoBtn)
        status_layout.addWidget(self.exportBtn)
        main_layout.addWidget(status_card)

        main_layout.addStretch()
        self.setFocusPolicy(Qt.ClickFocus)

        # Enable proper resizing
        self.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        main_widget.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Minimum)

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
        self.batchModeSwitch.toggled.connect(self._on_batch_mode_toggle)
        self.minSizeSpinBox.valueChanged.connect(self._on_batch_settings_changed)
        self.maxObjectsSpinBox.valueChanged.connect(self._on_batch_settings_changed)

    def _select_output_folder(self):
        folder = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Select Output Folder for Shapefiles", str(self.shapefile_save_dir.parent))

        if folder:
            self.shapefile_save_dir = pathlib.Path(
                folder) / "GeoOSAM_shapefiles"
            self.mask_save_dir = pathlib.Path(folder) / "GeoOSAM_masks"
            self.shapefile_save_dir.mkdir(exist_ok=True)
            if self.save_debug_masks:
                self.mask_save_dir.mkdir(exist_ok=True)

            short_path = "..." + str(self.shapefile_save_dir)[-35:] if len(
                str(self.shapefile_save_dir)) > 40 else str(self.shapefile_save_dir)
            self.outputFolderLabel.setText(short_path)
            self._update_status(
                f"📁 Output folder: {self.shapefile_save_dir}", "info")

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
                    f"border-radius: 8px; font-size: 14px;")
            except:
                self.currentClassLabel.setStyleSheet(
                    f"font-weight: 600; padding: 12px; border: 2px solid rgb({color}); "
                    f"background-color: rgba({color}, 50); font-size: 14px;")

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
                border-radius: 8px; font-size: 14px;
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
                'batch_defaults': {'min_size': 50, 'max_objects': 20}  # Generic defaults
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

    def _validate_class_selection(self):
        """Enhanced validation that properly handles layer switching"""
        if not self.current_class:
            self._update_status("Please select a class first!", "warning")
            return False

        current_layer = self.iface.activeLayer()
        if not isinstance(current_layer, QgsRasterLayer) or not current_layer.isValid():
            self._update_status(
                "Please select a valid raster layer first!", "warning")
            return False

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

        # Update button states
        self.pointModeBtn.setProperty("active", True)
        self.bboxModeBtn.setProperty("active", False)
        self.pointModeBtn.style().polish(self.pointModeBtn)
        self.bboxModeBtn.style().polish(self.bboxModeBtn)

        self._update_status(
            f"Point mode active for [{self.current_class}]. Click on map to segment.", "processing")
        self.canvas.setMapTool(self.pointTool)

    def _activate_bbox_tool(self):
        if not self._validate_class_selection():
            return

        self.current_mode = 'bbox'
        self.original_map_tool = self.canvas.mapTool()

        # Re-enable batch mode switch for bbox mode
        self.batchModeSwitch.setEnabled(True)

        # Update button states
        self.pointModeBtn.setProperty("active", False)
        self.bboxModeBtn.setProperty("active", True)
        self.pointModeBtn.style().polish(self.pointModeBtn)
        self.bboxModeBtn.style().polish(self.bboxModeBtn)

        self._update_status(
            f"BBox mode active for [{self.current_class}]. Click and drag to segment.", "processing")
        self.canvas.setMapTool(self.bboxTool)

    def _point_done(self, pt):
        self.point = pt
        self.bbox = None
        self._update_status(
            f"Processing point at ({pt.x():.1f}, {pt.y():.1f})...", "processing")
        self._run_segmentation()

    def _bbox_done(self, rect):
        self.bbox = rect
        self.point = None
        self._update_status(
            f"Processing bbox ({rect.width():.1f}×{rect.height():.1f})...", "processing")
        self._run_segmentation()

    def _run_segmentation(self):
        """Enhanced segmentation that ensures current layer is used"""
        # DEBUG: Verify settings are correct
        if self.batch_mode_enabled and self.current_mode == 'bbox':
            self._debug_current_settings()

        if not self.current_class:
            self._update_status("No class selected", "error")
            return

        # Get the CURRENT active layer (not stored reference)
        current_layer = self.iface.activeLayer()
        if not isinstance(current_layer, QgsRasterLayer) or not current_layer.isValid():
            self._update_status("Please select a valid raster layer", "error")
            return

        # Update stored reference to current layer
        self.original_raster_layer = current_layer

        if self.point is None and self.bbox is None:
            self._update_status("No selection found", "error")
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

        # Determine processing mode
        mode = "point" if self.point is not None else "bbox"
        if mode == "bbox" and self.batch_mode_enabled:
            mode = "bbox_batch"

        # Continue with worker thread...
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
            min_object_size=self.min_object_size,
            arr_multispectral=arr_multispectral,
            max_objects=self.max_objects
        )

        self.worker.finished.connect(self._on_segmentation_finished)
        self.worker.error.connect(self._on_segmentation_error)
        self.worker.progress.connect(self._on_segmentation_progress)
        self.worker.start()

    def _convert_mask_to_features(self, mask, mask_transform):
        """Convert a single mask to QgsFeature objects"""
        try:
            # Threshold mask to binary and clean it up
            _, binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

            # Morphological operations to clean up the mask
            open_kernel = np.ones((3, 3), np.uint8)
            binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, open_kernel)

            close_kernel = np.ones((7, 7), np.uint8)
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, close_kernel)

            # Remove small objects
            nlabels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
            min_size = 20
            cleaned = np.zeros(binary.shape, dtype=np.uint8)
            for i in range(1, nlabels):
                if stats[i, cv2.CC_STAT_AREA] >= min_size:
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
            for i, feat in enumerate(features):
                feat.setAttributes([
                    next_segment_id + i,
                    self.current_class,
                    class_color,
                    batch_info,  # Use batch_info instead of just mode
                    timestamp_str,
                    filename or "debug_disabled",
                    crop_info,
                    canvas_scale,
                    source_layer_name,
                    layer_crs
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

                if self.point is not None:  # POINT MODE
                    try:
                        row, col = src.index(self.point.x(), self.point.y())
                        center_pixel_x, center_pixel_y = col, row
                    except Exception as e:
                        self._update_status(f"Point is outside raster bounds: {e}", "error")
                        return None

                    crop_size = adaptive_crop_size
                    half_size = crop_size // 2

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
                    elif self.device == "cpu" and self.model_choice == "MobileSAM":
                        max_crop_size = min(max_crop_size, 1024)  # Limit for CPU MobileSAM

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

    def _get_adaptive_crop_size(self):
        canvas_scale = self.canvas.scale()

        # Base sizes based on device capability
        if self.device == "cuda":
            base_size = 1024  # Increased for CUDA
        elif self.device == "mps":
            base_size = 768   # Good for Apple Silicon
        else:
            base_size = 512 if self.model_choice == "MobileSAM" else 640

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
            elif 'mask' in result:
                # This is single processing - result structure: {'mask': array, 'mask_transform': ..., 'debug_info': ...}
                print(f"📦 SINGLE: Processing single mask")
                mask = result['mask']
                mask_transform = result['mask_transform']
                self._process_single_mask_result(mask, mask_transform, debug_info)
            else:
                # Error case - unexpected result structure
                raise KeyError(f"Unexpected result structure. Expected 'mask' or 'individual_masks', got keys: {list(result.keys())}")

            process_time = time.time() - start_process_time
            prep_time = debug_info.get('prep_time', 0)
            total_time = prep_time + process_time

            model_info = f"({debug_info.get('model', 'SAM')} on {self.device.upper()})"
            batch_info = ""
            if debug_info.get('batch_count'):
                if 'individual_masks' in result:
                    batch_info = f" - {debug_info['batch_count']} individual objects found"
                else:
                    batch_info = f" - {debug_info['batch_count']} objects found"
            self._update_status(
                f"✅ Completed in {total_time:.1f}s {model_info}{batch_info}! Click again to add more.", "info")

        except Exception as e:
            import traceback
            error_msg = f"Error processing results: {str(e)}\n"
            error_msg += f"Result keys: {list(result.keys()) if isinstance(result, dict) else 'Not a dict'}\n"
            error_msg += f"Debug info: {debug_info if 'debug_info' in locals() else 'None'}\n"
            error_msg += f"Full traceback:\n{traceback.format_exc()}"
            self._update_status(error_msg, "error")
        finally:
            self._set_ui_enabled(True)
            if hasattr(self, 'worker') and self.worker:
                self.worker.deleteLater()
                self.worker = None

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
            for i, feat in enumerate(features):
                feat.setAttributes([
                    next_segment_id + i,
                    self.current_class,
                    class_color,
                    batch_info,  # Individual object identifier
                    timestamp_str,
                    f"batch_obj_{object_number}.png" if self.save_debug_masks else "debug_disabled",
                    crop_info,
                    canvas_scale,
                    source_layer_name,
                    layer_crs
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
        undo_hint = " (↶ Undo available)" if len(features) > 0 else ""
        source_info = debug_info.get('source_layer', 'unknown')[:15]
        self._update_status(
            f"✅ Added {len(features)} [{self.current_class}] polygons from {source_info}!{undo_hint}", "info")
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
            QgsField("class_name", QVariant.String),
            QgsField("class_color", QVariant.String),
            QgsField("method", QVariant.String),
            QgsField("timestamp", QVariant.String),
            QgsField("mask_file", QVariant.String),
            QgsField("crop_size", QVariant.String),
            QgsField("canvas_scale", QVariant.Double),
            QgsField("source_layer", QVariant.String),  # Track source raster
            QgsField("layer_crs", QVariant.String)      # Track CRS used
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
        if not self.result_layers:
            self._update_status("No segments to export!", "warning")
            return

        exported_count = 0
        for class_name, layer in self.result_layers.items():
            if layer and layer.featureCount() > 0:
                if self._export_layer_to_shapefile(layer, class_name):
                    exported_count += 1

        if exported_count > 0:
            self._update_status(
                f"💾 Exported {exported_count} class(es) to {self.shapefile_save_dir}", "info")
        else:
            self._update_status("No segments found to export!", "warning")

    def _export_layer_to_shapefile(self, layer, class_name):
        try:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            shapefile_name = f"SAM_{class_name}_{timestamp}.shp"
            shapefile_path = str(self.shapefile_save_dir / shapefile_name)

            error = QgsVectorFileWriter.writeAsVectorFormat(
                layer, shapefile_path, "utf-8", layer.crs(), "ESRI Shapefile")

            if error[0] == QgsVectorFileWriter.NoError:
                print(f"💾 Exported {class_name}: {shapefile_path}")
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
        if hasattr(self, 'worker') and self.worker:
            self.worker.deleteLater()
            self.worker = None

    def _cancel_segmentation(self):
        if hasattr(self, 'worker') and self.worker and self.worker.isRunning():
            self.worker.terminate()
            self.worker.wait()
            self.worker.deleteLater()
            self.worker = None
            self._update_status("Segmentation cancelled", "warning")
            self._set_ui_enabled(True)

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
        color_styles = {
            "info": "background: #ECFDF3; color: #027A48; border: 1px solid #D1FADF;",
            "warning": "background: #FFFBEB; color: #DC6803; border: 1px solid #FED7AA;",
            "error": "background: #FEF2F2; color: #DC2626; border: 1px solid #FECACA;",
            "processing": "background: #EFF8FF; color: #1570EF; border: 1px solid #B2DDFF;"
        }
        color_style = color_styles.get(status_type, color_styles["info"])
        self.statusLabel.setText(message)
        self.statusLabel.setStyleSheet(f"""
            padding: 14px; border-radius: 8px; font-size: 14px; font-weight: 500;
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

        self.setWindowTitle("GeoOSAM")
        self.resize(280, 140)

    def _show_control_panel(self):
        if not self.control_panel:
            self.control_panel = GeoOSAMControlPanel(self.iface)
            self.iface.addDockWidget(
                Qt.RightDockWidgetArea, self.control_panel)
        self.control_panel.show()
        self.control_panel.raise_()
        self.close()