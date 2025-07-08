"""
General Detection Helper

Generic helper for general object detection - serves as default for new/unknown classes.
"""

import cv2
import numpy as np
from .base_helper import BaseDetectionHelper


class GeneralHelper(BaseDetectionHelper):
    """General helper for generic object detection"""
    
    def __init__(self, class_name="General", min_object_size=50, max_objects=30):
        super().__init__(class_name, min_object_size, max_objects)
    
    def detect_candidates(self, bbox_image, bbox):
        """General-purpose object detection using brightness"""
        return self._detect_bright_objects(bbox_image, bbox)
    
    def validate_object(self, component_mask, component_area, min_object_size):
        """General object validation"""
        contours, _ = cv2.findContours(component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return False, "no contours"
        
        main_contour = max(contours, key=cv2.contourArea)
        metrics = self.get_basic_shape_metrics(main_contour)
        
        # General validation - moderate requirements
        is_valid = (
            metrics['aspect_ratio'] <= 10.0 and     # Not extremely elongated
            metrics['solidity'] >= 0.3 and          # Reasonably solid
            metrics['area'] >= min_object_size * 0.7  # Close to minimum size
        )
        
        if not is_valid:
            if metrics['aspect_ratio'] > 10.0:
                return False, f"too elongated ({metrics['aspect_ratio']:.1f} > 10.0)"
            elif metrics['solidity'] < 0.3:
                return False, f"not solid enough ({metrics['solidity']:.2f} < 0.3)"
            elif metrics['area'] < min_object_size * 0.7:
                return False, f"too small ({metrics['area']} < {min_object_size * 0.7})"
        
        return True, "valid general object"
    
    def apply_morphology(self, mask):
        """General morphology operations"""
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 4))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        return mask
    
    def should_merge_masks(self):
        return True  # General objects can be merged if adjacent
    
    def get_background_threshold(self, bbox_area):
        """General background threshold"""
        return bbox_area * 0.6  # Medium threshold
    
    def _detect_bright_objects(self, bbox_image, bbox):
        """General bright object detection"""
        x1, y1, x2, y2 = bbox
        
        # Convert to grayscale
        if len(bbox_image.shape) == 3:
            gray = cv2.cvtColor(bbox_image, cv2.COLOR_RGB2GRAY)
        else:
            gray = bbox_image
        
        print(f"  📊 Image stats: min={gray.min()}, max={gray.max()}, mean={gray.mean():.1f}")
        
        # General object detection approach
        # Step 1: Noise reduction
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Step 2: Adaptive thresholding
        mean_val = blurred.mean()
        std_val = blurred.std()
        
        # Use moderate threshold for general objects
        threshold = min(255, mean_val + 1.0 * std_val)
        
        # Also use Otsu's method for comparison
        otsu_thresh, _ = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Use the more permissive threshold
        final_threshold = min(threshold, otsu_thresh * 0.9)
        
        print(f"  🔍 General thresholds: adaptive={threshold:.1f}, otsu={otsu_thresh}, final={final_threshold:.1f}")
        
        # Create binary mask
        _, binary = cv2.threshold(blurred, final_threshold, 255, cv2.THRESH_BINARY)
        
        # Step 3: Morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
        
        # Step 4: Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        candidates = []
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # General size filtering
            if area >= self.min_object_size and area <= 10000:
                # Basic shape validation
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = max(w, h) / max(min(w, h), 1)
                
                # General shape requirements
                if aspect_ratio <= 8.0 and w >= 3 and h >= 3:
                    M = cv2.moments(contour)
                    if M["m00"] != 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                        
                        full_x = x1 + cx
                        full_y = y1 + cy
                        candidates.append((full_x, full_y))
        
        print(f"  🔍 Found {len(candidates)} general object candidates")
        return candidates