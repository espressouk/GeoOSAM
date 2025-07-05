"""
Vegetation Detection Helper

Specialized helper for vegetation detection using texture analysis.
"""

import cv2
import numpy as np
from .base_helper import BaseDetectionHelper


class VegetationHelper(BaseDetectionHelper):
    """Specialized helper for vegetation detection"""
    
    def __init__(self, class_name="Vegetation", min_object_size=30, max_objects=40):
        super().__init__(class_name, min_object_size, max_objects)
    
    def detect_candidates(self, bbox_image, bbox):
        """Vegetation-specific texture detection"""
        return self._detect_textured_objects(bbox_image, bbox)
    
    def validate_object(self, component_mask, component_area, min_object_size):
        """Vegetation-specific validation"""
        contours, _ = cv2.findContours(component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return False, "no contours"
        
        main_contour = max(contours, key=cv2.contourArea)
        metrics = self.get_basic_shape_metrics(main_contour)
        
        # Vegetation can be irregular, so more lenient validation
        is_valid = (
            metrics['aspect_ratio'] <= 5.0 and      # Not too elongated
            metrics['solidity'] >= 0.15 and         # Can be very irregular
            metrics['area'] >= min_object_size * 0.5  # Smaller threshold for detection
        )
        
        if not is_valid:
            if metrics['aspect_ratio'] > 5.0:
                return False, f"bad aspect ratio ({metrics['aspect_ratio']:.1f} > 5.0)"
            elif metrics['solidity'] < 0.15:
                return False, f"not solid enough ({metrics['solidity']:.2f} < 0.15)"
            elif metrics['area'] < min_object_size * 0.5:
                return False, f"too small ({metrics['area']} < {min_object_size * 0.5})"
        
        return True, "valid vegetation"
    
    def apply_morphology(self, mask):
        """Vegetation-specific morphology"""
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        return mask
    
    def should_merge_masks(self):
        return True  # Vegetation can merge
    
    def get_background_threshold(self, bbox_area):
        """Vegetation-specific background threshold"""
        return bbox_area * 0.9  # Large threshold - allow big areas
    
    def _detect_textured_objects(self, bbox_image, bbox):
        """Detect textured objects (vegetation) using local standard deviation"""
        x1, y1, x2, y2 = bbox
        print(f"  🌿 Analyzing texture in {x2-x1}x{y2-y1}px bbox")
        
        # Convert to grayscale for texture analysis
        if len(bbox_image.shape) == 3:
            gray = cv2.cvtColor(bbox_image, cv2.COLOR_RGB2GRAY)
        else:
            gray = bbox_image
        
        # Apply texture detection using local standard deviation
        kernel_size = 9
        kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size * kernel_size)
        mean = cv2.filter2D(gray.astype(np.float32), -1, kernel)
        sqr_mean = cv2.filter2D((gray.astype(np.float32))**2, -1, kernel)
        texture = np.sqrt(sqr_mean - mean**2)
        
        # Normalize texture to 0-255
        if texture.max() > texture.min():
            texture_norm = ((texture - texture.min()) / (texture.max() - texture.min()) * 255).astype(np.uint8)
        else:
            texture_norm = np.zeros_like(texture, dtype=np.uint8)
        
        # Adaptive threshold
        threshold = np.percentile(texture_norm, 70)
        _, binary = cv2.threshold(texture_norm, threshold, 255, cv2.THRESH_BINARY)
        
        # Morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
        
        # Find contours and extract candidates
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        candidates = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area >= self.min_object_size * 0.5:  # Slightly smaller threshold for detection
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    full_x = x1 + cx
                    full_y = y1 + cy
                    candidates.append((full_x, full_y))
        
        print(f"  🌿 Found {len(candidates)} vegetation texture candidates")
        return candidates