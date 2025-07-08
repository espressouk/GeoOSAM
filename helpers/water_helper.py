"""
Water Detection Helper

Specialized helper for water body detection using dark object detection and blue channel analysis.
"""

import cv2
import numpy as np
from .base_helper import BaseDetectionHelper


class WaterHelper(BaseDetectionHelper):
    """Specialized helper for water body detection using dark object detection"""
    
    def __init__(self, class_name="Water", min_object_size=200, max_objects=20):
        super().__init__(class_name, min_object_size, max_objects)
    
    def detect_candidates(self, bbox_image, bbox):
        """Water-specific dark object detection"""
        return self._detect_dark_objects(bbox_image, bbox)
    
    def validate_object(self, component_mask, component_area, min_object_size):
        """Water-specific validation"""
        contours, _ = cv2.findContours(component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return False, "no contours"
        
        main_contour = max(contours, key=cv2.contourArea)
        metrics = self.get_basic_shape_metrics(main_contour)
        
        # Water validation - water bodies are often large and relatively smooth
        is_valid = (
            metrics['area'] >= min_object_size and
            metrics['solidity'] >= 0.4 and          # Water bodies should be reasonably solid
            metrics['aspect_ratio'] <= 15.0         # Allow elongated water bodies (rivers, lakes)
        )
        
        if not is_valid:
            if metrics['area'] < min_object_size:
                return False, f"below threshold ({metrics['area']} < {min_object_size})"
            elif metrics['solidity'] < 0.4:
                return False, f"not solid enough ({metrics['solidity']:.2f} < 0.4)"
            elif metrics['aspect_ratio'] > 15.0:
                return False, f"too elongated ({metrics['aspect_ratio']:.1f} > 15.0)"
        
        return True, "valid water body"
    
    def apply_morphology(self, mask):
        """Water-specific morphology - smooth and fill water bodies"""
        # Use larger kernel for water bodies to smooth edges
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        return mask
    
    def should_merge_masks(self):
        return True  # Water bodies can be merged if they're connected
    
    def get_background_threshold(self, bbox_area):
        """Water-specific background threshold"""
        return bbox_area * 0.4  # Lower threshold for water (larger areas expected)
    
    def _detect_dark_objects(self, bbox_image, bbox):
        """Enhanced water detection combining darkness with blue channel analysis"""
        x1, y1, x2, y2 = bbox
        
        # Convert to grayscale
        if len(bbox_image.shape) == 3:
            gray = cv2.cvtColor(bbox_image, cv2.COLOR_RGB2GRAY)
            # Extract blue channel for water detection
            blue_channel = bbox_image[:, :, 2]  # Assuming RGB format
        else:
            gray = bbox_image
            blue_channel = gray
        
        print(f"  📊 Image stats: min={gray.min()}, max={gray.max()}, mean={gray.mean():.1f}")
        
        # BEST PRACTICE 1: Noise reduction with Gaussian blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        blue_blurred = cv2.GaussianBlur(blue_channel, (5, 5), 0)
        
        # BEST PRACTICE 2: Sea and water body detection approach
        mean_val = blurred.mean()
        std_val = blurred.std()
        
        # Simple water detection - just catch very dark areas
        # Based on logs: mean around 16, water should be much darker
        
        # Use very low threshold to catch dark water areas
        water_threshold = min(mean_val * 0.8, 20)  # Very permissive - catch anything darker than 80% of mean
        
        print(f"  🎯 Simple water threshold: {water_threshold:.1f} (mean={mean_val:.1f})")
        
        # Create binary mask for dark objects (inverted threshold)
        _, binary = cv2.threshold(blurred, water_threshold, 255, cv2.THRESH_BINARY_INV)
        
        # BEST PRACTICE 3: Minimal morphological operations
        # Just basic cleanup to avoid eliminating water areas
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)
        
        # BEST PRACTICE 4: Contour analysis with water-specific filtering
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        candidates = []
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # Size filtering - very permissive to catch any water areas
            if area >= 100:
                # BEST PRACTICE 5: Shape analysis for water body characteristics
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = max(w, h) / max(min(w, h), 1)
                
                # Calculate shape metrics
                perimeter = cv2.arcLength(contour, True)
                compactness = (4 * np.pi * area) / (perimeter * perimeter) if perimeter > 0 else 0
                
                # Calculate solidity
                hull = cv2.convexHull(contour)
                hull_area = cv2.contourArea(hull)
                solidity = area / hull_area if hull_area > 0 else 0
                
                # Very permissive shape filtering - just basic sanity check
                if (w >= 5 and h >= 5):             # Minimum size check only
                    
                    M = cv2.moments(contour)
                    if M["m00"] != 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                        
                        full_x = x1 + cx
                        full_y = y1 + cy
                        candidates.append((full_x, full_y))
        
        print(f"  💧 Found {len(candidates)} water body candidates")
        return candidates