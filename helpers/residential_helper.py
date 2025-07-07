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
        """Enhanced residential building detection combining brightness with geometric features"""
        x1, y1, x2, y2 = bbox
        
        # Convert to grayscale
        if len(bbox_image.shape) == 3:
            gray = cv2.cvtColor(bbox_image, cv2.COLOR_RGB2GRAY)
        else:
            gray = bbox_image
        
        print(f"  📊 Image stats: min={gray.min()}, max={gray.max()}, mean={gray.mean():.1f}")
        
        # BEST PRACTICE 1: Noise reduction with Gaussian blur
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        
        # BEST PRACTICE 2: Multi-modal thresholding approach
        mean_val = blurred.mean()
        std_val = blurred.std()
        
        # Primary threshold: Statistical approach for bright objects
        threshold_bright = min(255, mean_val + 1.2 * std_val)
        
        # Secondary threshold: Otsu's method for adaptive segmentation
        otsu_thresh, _ = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Combine approaches: Use brighter threshold to avoid vegetation
        final_threshold = max(threshold_bright, otsu_thresh * 0.8)
        
        print(f"  🎯 Thresholds: bright={threshold_bright:.1f}, otsu={otsu_thresh}, final={final_threshold:.1f}")
        
        _, binary = cv2.threshold(blurred, final_threshold, 255, cv2.THRESH_BINARY)
        
        # BEST PRACTICE 3: Progressive morphological operations
        # Start with small kernel to preserve building separation
        small_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        medium_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        
        # Remove small noise while preserving building shapes
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, small_kernel, iterations=1)
        # Fill small gaps in building roofs
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, medium_kernel, iterations=1)
        
        # BEST PRACTICE 4: Contour analysis with geometric filtering
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        candidates = []
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # Size filtering with upper bound to exclude large vegetation areas
            if area >= self.min_object_size and area <= 8000:
                # BEST PRACTICE 5: Shape analysis for building characteristics
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = max(w, h) / max(min(w, h), 1)
                
                # Calculate perimeter-to-area ratio (buildings are more compact)
                perimeter = cv2.arcLength(contour, True)
                compactness = (4 * np.pi * area) / (perimeter * perimeter) if perimeter > 0 else 0
                
                # Buildings have reasonable aspect ratios and are reasonably compact
                if (aspect_ratio < 5.0 and           # Not extremely elongated
                    compactness > 0.12 and          # Reasonably compact shape
                    w >= 5 and h >= 5):             # Minimum building size
                    
                    M = cv2.moments(contour)
                    if M["m00"] != 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                        
                        full_x = x1 + cx
                        full_y = y1 + cy
                        candidates.append((full_x, full_y))
        
        print(f"  🏠 Found {len(candidates)} enhanced residential candidates")
        return candidates