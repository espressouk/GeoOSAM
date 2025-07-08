"""
Vessels Detection Helper

Specialized helper for vessel detection using the previous working logic.
"""

import cv2
import numpy as np
from .base_helper import BaseDetectionHelper


class VesselsHelper(BaseDetectionHelper):
    """Specialized helper for vessel detection"""
    
    def __init__(self, class_name="Vessels", min_object_size=20, max_objects=50):
        super().__init__(class_name, min_object_size, max_objects)
    
    def detect_candidates(self, bbox_image, bbox):
        """Vessel-specific bright object detection"""
        return self._detect_bright_objects(bbox_image, bbox)
    
    def validate_object(self, component_mask, component_area, min_object_size):
        """Vessel-specific validation"""
        contours, _ = cv2.findContours(component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return False, "no contours"
        
        main_contour = max(contours, key=cv2.contourArea)
        metrics = self.get_basic_shape_metrics(main_contour)
        
        # Vessel validation - balanced filtering (original working logic)
        is_valid = (
            metrics['aspect_ratio'] <= 5.0 and      # Vessels shouldn't be too elongated
            metrics['solidity'] >= 0.4 and          # Should be reasonably solid
            metrics['area'] <= 2000 and             # Reasonable max size
            metrics['area'] >= min_object_size * 0.7  # Reasonable minimum
        )
        
        if not is_valid:
            if metrics['aspect_ratio'] > 5.0:
                return False, f"bad aspect ratio ({metrics['aspect_ratio']:.1f} > 5.0)"
            elif metrics['solidity'] < 0.4:
                return False, f"not solid enough ({metrics['solidity']:.2f} < 0.4)"
            elif metrics['area'] > 2000:
                return False, f"too large ({metrics['area']} > 2000)"
            elif metrics['area'] < min_object_size * 0.7:
                return False, f"too small ({metrics['area']} < {min_object_size * 0.7})"
        
        return True, "valid vessel"
    
    def apply_morphology(self, mask):
        """Vessel-specific morphology"""
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)  # Remove noise
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)  # Fill gaps
        return mask
    
    def should_merge_masks(self):
        return True  # Vessels can have minimal merging
    
    def get_background_threshold(self, bbox_area):
        """Vessel-specific background threshold"""
        return bbox_area * 0.7  # High threshold for boats on water
    
    def process_sam_mask(self, mask, predictor=None):
        """Vessel-specific SAM mask processing with strict shoreline rejection"""
        if mask is None:
            return None
        
        # Apply class-specific morphology
        mask = self.apply_morphology(mask)
        
        # Find connected components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
        
        # For vessels: if ANY component is larger than 5000 pixels, reject entire mask (likely shoreline)
        max_vessel_size = 5000
        for i in range(1, num_labels):  # Skip background (0)
            component_area = stats[i, cv2.CC_STAT_AREA]
            if component_area > max_vessel_size:
                print(f"    ❌ ENTIRE MASK REJECTED: Contains large component {i} ({component_area} > {max_vessel_size}) - likely shoreline")
                return []
        
        # If no large components found, proceed with normal validation
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
    
    def _detect_bright_objects(self, bbox_image, bbox):
        """Detect bright objects (vessels) using adaptive thresholding"""
        x1, y1, x2, y2 = bbox
        
        # Convert to grayscale
        if len(bbox_image.shape) == 3:
            gray = cv2.cvtColor(bbox_image, cv2.COLOR_RGB2GRAY)
        else:
            gray = bbox_image
        
        print(f"  📊 Image stats: min={gray.min()}, max={gray.max()}, mean={gray.mean():.1f}")
        
        # Adaptive thresholding to find bright objects
        # Use mean + standard deviation to find objects brighter than background
        mean_val = gray.mean()
        std_val = gray.std()
        threshold = min(255, mean_val + 1.5 * std_val)  # Objects significantly brighter than average
        
        print(f"  🎯 Using threshold: {threshold:.1f} (mean + 1.5*std)")
        
        _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
        
        # Morphological operations to clean up and separate objects
        kernel = np.ones((3, 3), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)  # Remove noise
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)  # Fill gaps
        
        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        candidates = []
        for contour in contours:
            area = cv2.contourArea(contour)
            
            if area >= self.min_object_size:
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    
                    full_x = x1 + cx
                    full_y = y1 + cy
                    candidates.append((full_x, full_y))
        
        print(f"  🚢 Found {len(candidates)} vessel candidates")
        return candidates