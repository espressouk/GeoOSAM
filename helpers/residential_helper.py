"""
Residential Detection Helper

Specialized helper for residential building detection.
"""

import cv2
import numpy as np
from .base_helper import BaseDetectionHelper


class ResidentialHelper(BaseDetectionHelper):
    """Specialized helper for residential building detection"""
    
    def __init__(self, class_name="Residential", min_object_size=75, max_objects=40):
        super().__init__(class_name, min_object_size, max_objects)
    
    def detect_candidates(self, bbox_image, bbox):
        """Residential-specific bright object detection (matches original behavior)"""
        return self._detect_bright_objects(bbox_image, bbox)
    
    def validate_object(self, component_mask, component_area, min_object_size):
        """Residential-specific validation"""
        contours, _ = cv2.findContours(component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return False, "no contours"
        
        main_contour = max(contours, key=cv2.contourArea)
        metrics = self.get_basic_shape_metrics(main_contour)
        
        min_threshold = max(min_object_size * 0.5, 25)
        max_threshold = 20000
        
        is_valid = (
            metrics['aspect_ratio'] <= 12.0 and     # Allow longer buildings
            metrics['solidity'] >= 0.4 and          # More lenient solidity
            metrics['area'] >= min_threshold and     # Use relaxed size threshold
            metrics['area'] <= max_threshold         # Prevent oversized masks
        )
        
        if not is_valid:
            if metrics['aspect_ratio'] > 12.0:
                return False, f"bad aspect ratio ({metrics['aspect_ratio']:.1f} > 12.0)"
            elif metrics['solidity'] < 0.4:
                return False, f"not solid enough ({metrics['solidity']:.2f} < 0.4)"
            elif metrics['area'] < min_threshold:
                return False, f"below threshold ({metrics['area']} < {min_threshold})"
            elif metrics['area'] > max_threshold:
                return False, f"too large ({metrics['area']} > {max_threshold})"
        
        return True, "valid residential"
    
    def apply_morphology(self, mask):
        """Building-specific morphology"""
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        return mask
    
    def should_merge_masks(self):
        return False  # Buildings don't merge - each detection should stay separate
    
    def get_background_threshold(self, bbox_area):
        """Residential-specific background threshold"""
        return bbox_area * 0.6  # Medium threshold
    
    def _detect_bright_objects(self, bbox_image, bbox):
        """Detect bright objects (residential buildings) using adaptive thresholding"""
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
        
        print(f"  🏠 Found {len(candidates)} bright residential candidates")
        return candidates