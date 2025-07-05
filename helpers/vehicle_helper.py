"""
Vehicle Detection Helper

Specialized helper for vehicle detection using bright object detection.
"""

import cv2
import numpy as np
from .base_helper import BaseDetectionHelper


class VehicleHelper(BaseDetectionHelper):
    """Specialized helper for vehicle detection"""
    
    def __init__(self, class_name="Vehicle", min_object_size=20, max_objects=50):
        super().__init__(class_name, min_object_size, max_objects)
    
    def detect_candidates(self, bbox_image, bbox):
        """Vehicle-specific bright object detection"""
        return self._detect_bright_objects(bbox_image, bbox)
    
    def validate_object(self, component_mask, component_area, min_object_size):
        """Vehicle-specific validation"""
        contours, _ = cv2.findContours(component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return False, "no contours"
        
        main_contour = max(contours, key=cv2.contourArea)
        metrics = self.get_basic_shape_metrics(main_contour)
        
        # Vehicle/vessel validation
        is_valid = (
            metrics['aspect_ratio'] <= 6.0 and      # Boats/vehicles shouldn't be too elongated
            metrics['solidity'] >= 0.3 and          # Should be reasonably solid
            metrics['area'] <= 8000 and             # Not too large (reject water areas)
            metrics['area'] >= min_object_size * 0.6  # Minimum size
        )
        
        if not is_valid:
            if metrics['aspect_ratio'] > 6.0:
                return False, f"bad aspect ratio ({metrics['aspect_ratio']:.1f} > 6.0)"
            elif metrics['solidity'] < 0.3:
                return False, f"not solid enough ({metrics['solidity']:.2f} < 0.3)"
            elif metrics['area'] > 8000:
                return False, f"too large ({metrics['area']} > 8000)"
            elif metrics['area'] < min_object_size * 0.6:
                return False, f"too small ({metrics['area']} < {min_object_size * 0.6})"
        
        return True, "valid vehicle"
    
    def apply_morphology(self, mask):
        """Vehicle-specific morphology"""
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)  # Remove noise
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)  # Fill gaps
        return mask
    
    def should_merge_masks(self):
        return True  # Vehicles can have minimal merging
    
    def get_background_threshold(self, bbox_area):
        """Vehicle-specific background threshold"""
        return bbox_area * 0.4  # Smaller threshold - reject large water areas
    
    def _detect_bright_objects(self, bbox_image, bbox):
        """Detect bright objects (vehicles) using adaptive thresholding"""
        x1, y1, x2, y2 = bbox
        
        # Convert to grayscale
        if len(bbox_image.shape) == 3:
            gray = cv2.cvtColor(bbox_image, cv2.COLOR_RGB2GRAY)
        else:
            gray = bbox_image
        
        print(f"  📊 Image stats: min={gray.min()}, max={gray.max()}, mean={gray.mean():.1f}")
        
        # Adaptive thresholding
        mean_val = gray.mean()
        std_val = gray.std()
        threshold = min(255, mean_val + 1.5 * std_val)  # Objects significantly brighter than average
        
        print(f"  🎯 Using threshold: {threshold:.1f} (mean + 1.5*std)")
        
        _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
        
        # Morphological operations
        kernel = np.ones((3, 3), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)  # Remove noise
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)  # Fill gaps
        
        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        candidates = []
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # Size filtering
            min_area = self.min_object_size * 0.5  # Slightly smaller for detection
            max_area = 10000 if self.class_name == 'Vessels' else 20000
            
            if min_area <= area <= max_area:
                # Get centroid
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    
                    # Convert back to full image coordinates
                    full_x = x1 + cx
                    full_y = y1 + cy
                    candidates.append((full_x, full_y))
        
        print(f"  🚗 Found {len(candidates)} bright object candidates")
        return candidates