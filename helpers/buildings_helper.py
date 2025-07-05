"""
Buildings Detection Helper

Specialized helper for building detection using rectangular/edge detection.
"""

import cv2
import numpy as np
from .base_helper import BaseDetectionHelper


class BuildingsHelper(BaseDetectionHelper):
    """Specialized helper for building detection using rectangular detection"""
    
    def __init__(self, class_name="Buildings", min_object_size=75, max_objects=40):
        super().__init__(class_name, min_object_size, max_objects)
    
    def detect_candidates(self, bbox_image, bbox):
        """Buildings-specific rectangular detection"""
        return self._detect_rectangular_objects(bbox_image, bbox)
    
    def validate_object(self, component_mask, component_area, min_object_size):
        """Buildings-specific validation"""
        contours, _ = cv2.findContours(component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return False, "no contours"
        
        main_contour = max(contours, key=cv2.contourArea)
        metrics = self.get_basic_shape_metrics(main_contour)
        
        # Buildings validation (from original)
        is_valid = (
            metrics['aspect_ratio'] <= 8.0 and
            metrics['solidity'] >= 0.5 and          # Buildings should be more solid
            metrics['area'] >= min_object_size
        )
        
        if not is_valid:
            if metrics['aspect_ratio'] > 8.0:
                return False, f"bad aspect ratio ({metrics['aspect_ratio']:.1f} > 8.0)"
            elif metrics['solidity'] < 0.5:
                return False, f"not solid enough ({metrics['solidity']:.2f} < 0.5)"
            elif metrics['area'] < min_object_size:
                return False, f"below threshold ({metrics['area']} < {min_object_size})"
        
        return True, "valid building"
    
    def apply_morphology(self, mask):
        """Building-specific morphology"""
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        return mask
    
    def should_merge_masks(self):
        return False  # Buildings don't merge - each detection should stay separate
    
    def get_background_threshold(self, bbox_area):
        """Buildings-specific background threshold"""
        return bbox_area * 0.6  # Medium threshold
    
    def _detect_rectangular_objects(self, bbox_image, bbox):
        """Detect rectangular objects (buildings) using edge detection"""
        x1, y1, x2, y2 = bbox
        
        # Convert to grayscale
        if len(bbox_image.shape) == 3:
            gray = cv2.cvtColor(bbox_image, cv2.COLOR_RGB2GRAY)
        else:
            gray = bbox_image
        
        # Edge detection
        edges = cv2.Canny(gray, 50, 150)
        
        # Find contours from edges
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        candidates = []
        for contour in contours:
            area = cv2.contourArea(contour)
            
            if area >= self.min_object_size:
                # Check if contour is roughly rectangular
                epsilon = 0.02 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                
                if len(approx) >= 4:  # Roughly rectangular
                    M = cv2.moments(contour)
                    if M["m00"] != 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                        
                        full_x = x1 + cx
                        full_y = y1 + cy
                        candidates.append((full_x, full_y))
        
        print(f"  🏢 Found {len(candidates)} rectangular building candidates")
        return candidates