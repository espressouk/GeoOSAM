"""
Base Detection Helper

Common SAM and detection functionality shared by all object detection helpers.
"""

import cv2
import numpy as np
from abc import ABC, abstractmethod


class BaseDetectionHelper(ABC):
    """Base class for all object detection helpers with common SAM functionality"""
    
    def __init__(self, class_name, min_object_size=50, max_objects=25):
        self.class_name = class_name
        self.min_object_size = min_object_size
        self.max_objects = max_objects
    
    @abstractmethod
    def detect_candidates(self, bbox_image, bbox):
        """Override in subclasses for class-specific detection"""
        pass
    
    @abstractmethod
    def validate_object(self, component_mask, component_area, min_object_size):
        """Override in subclasses for class-specific validation"""
        pass
    
    def apply_morphology(self, mask):
        """Common morphological operations - override for class-specific needs"""
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        return mask
    
    def should_merge_masks(self):
        """Override in subclasses - return True if this class should merge nearby masks"""
        return False
    
    def process_sam_mask(self, mask, predictor=None):
        """Common SAM mask processing logic"""
        if mask is None:
            return None
        
        # Apply class-specific morphology
        mask = self.apply_morphology(mask)
        
        # Find connected components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
        
        valid_masks = []
        for i in range(1, num_labels):  # Skip background (0)
            component_mask = (labels == i).astype(np.uint8) * 255
            component_area = stats[i, cv2.CC_STAT_AREA]
            
            # Use class-specific validation
            is_valid, reason = self.validate_object(component_mask, component_area, self.min_object_size)
            
            if is_valid:
                valid_masks.append(component_mask)
            else:
                print(f"    ❌ Rejected component {i}: {reason}")
        
        return valid_masks
    
    def combine_masks(self, masks):
        """Common mask combination logic"""
        if not masks:
            return None
        
        if len(masks) == 1:
            return masks[0]
        
        # Combine all masks
        combined = np.zeros_like(masks[0])
        for mask in masks:
            combined = cv2.bitwise_or(combined, mask)
        
        return combined
    
    def get_basic_shape_metrics(self, contour):
        """Get basic shape metrics for validation"""
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = max(w, h) / max(min(w, h), 1)
        
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        contour_area = cv2.contourArea(contour)
        solidity = contour_area / hull_area if hull_area > 0 else 0
        
        perimeter = cv2.arcLength(contour, True)
        
        return {
            'bbox': (x, y, w, h),
            'aspect_ratio': aspect_ratio,
            'area': contour_area,
            'solidity': solidity,
            'perimeter': perimeter
        }
    
    def get_merge_buffer_size(self):
        """Get buffer size for merging nearby objects. Override in subclasses."""
        return 3  # Default buffer size
    
    def get_iou_threshold(self):
        """Get IoU threshold for deduplication. Override in subclasses."""
        return 0.3  # Default IoU threshold
    
    def should_merge_duplicates(self):
        """Whether to merge overlapping objects or just remove duplicates. Override in subclasses."""
        return True  # Default: merge duplicates
    
    def merge_nearby_masks(self, masks):
        """Merge nearby masks using class-specific buffer size"""
        buffer_px = self.get_merge_buffer_size()
        
        if buffer_px == 0:
            # No merging for this class
            return masks
            
        # Original merging logic with class-aware buffer
        kernel = np.ones((buffer_px*2+1, buffer_px*2+1), np.uint8)
        bins = [cv2.threshold(m, 127, 255, cv2.THRESH_BINARY)[1] for m in masks]
        dilated = [cv2.dilate(b, kernel, iterations=1) for b in bins]
        used = [False] * len(bins)
        merged = []

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
    
    def dedupe_or_merge_masks(self, masks):
        """Smart deduplication using class-specific parameters"""
        iou_thresh = self.get_iou_threshold()
        merge = self.should_merge_duplicates()
        
        # Original logic with class-aware parameters
        bins = [cv2.threshold(m, 127, 255, cv2.THRESH_BINARY)[1] for m in masks]
        used = [False] * len(masks)
        result = []

        for i in range(len(bins)):
            if used[i]: 
                continue
            mi = bins[i]
            union_mask = mi.copy()

            for j in range(i+1, len(bins)):
                if used[j]: 
                    continue
                mj = bins[j]
                inter = cv2.bitwise_and(mi, mj)
                uni = cv2.bitwise_or(mi, mj)
                # IoU = area(inter) / area(union)
                if np.sum(uni == 255) > 0:
                    iou = np.sum(inter == 255) / np.sum(uni == 255)
                    if iou >= iou_thresh:
                        used[j] = True
                        if merge:
                            union_mask = cv2.bitwise_or(union_mask, mj)
                        else:
                            # keep only the bigger mask by area
                            if np.sum(mj == 255) > np.sum(mi == 255):
                                union_mask = mj.copy()
                                
            result.append(union_mask)
        return result