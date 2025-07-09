"""
Agriculture Detection Helper

Specialized helper for agriculture detection using texture analysis similar to vegetation.
"""

import cv2
import numpy as np
from .base_helper import BaseDetectionHelper


class AgricultureHelper(BaseDetectionHelper):
    """Specialized helper for agriculture detection"""
    
    def __init__(self, class_name="Agriculture", min_object_size=50, max_objects=30):
        super().__init__(class_name, min_object_size, max_objects)
    
    def detect_candidates(self, bbox_image, bbox):
        """Agriculture-specific texture detection"""
        return self._detect_textured_objects(bbox_image, bbox)
    
    def validate_object(self, component_mask, component_area, min_object_size):
        """Agriculture-specific validation"""
        contours, _ = cv2.findContours(component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return False, "no contours"
        
        main_contour = max(contours, key=cv2.contourArea)
        metrics = self.get_basic_shape_metrics(main_contour)
        
        # Agriculture can be more regular than wild vegetation (fields, crops)
        is_valid = (
            metrics['aspect_ratio'] <= 8.0 and      # Allow elongated fields
            metrics['solidity'] >= 0.2 and          # More regular than wild vegetation
            metrics['area'] >= min_object_size * 0.7  # Slightly higher threshold than vegetation
        )
        
        if not is_valid:
            if metrics['aspect_ratio'] > 8.0:
                return False, f"bad aspect ratio ({metrics['aspect_ratio']:.1f} > 8.0)"
            elif metrics['solidity'] < 0.2:
                return False, f"not solid enough ({metrics['solidity']:.2f} < 0.2)"
            elif metrics['area'] < min_object_size * 0.7:
                return False, f"too small ({metrics['area']} < {min_object_size * 0.7})"
        
        return True, "valid agriculture"
    
    def apply_morphology(self, mask):
        """Agriculture-specific morphology"""
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 4))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        return mask
    
    def should_merge_masks(self):
        return True  # Agriculture fields can be merged if adjacent
    
    def get_background_threshold(self, bbox_area):
        """Agriculture-specific background threshold"""
        return bbox_area * 0.7  # Medium-high threshold (fields can be large)
    
    def _detect_textured_objects(self, bbox_image, bbox):
        """Detect agricultural areas using texture analysis"""
        x1, y1, x2, y2 = bbox
        
        # Convert to grayscale
        if len(bbox_image.shape) == 3:
            gray = cv2.cvtColor(bbox_image, cv2.COLOR_RGB2GRAY)
        else:
            gray = bbox_image
        
        print(f"  📊 Image stats: min={gray.min()}, max={gray.max()}, mean={gray.mean():.1f}")
        
        # Agricultural texture detection
        # Step 1: Gaussian blur for noise reduction
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Step 2: Texture detection using Laplacian
        laplacian = cv2.Laplacian(blurred, cv2.CV_64F)
        laplacian = np.absolute(laplacian)
        
        # Step 3: Statistical thresholding for agricultural areas
        mean_val = blurred.mean()
        std_val = blurred.std()
        
        # Agriculture often has medium brightness (not too dark, not too bright)
        lower_thresh = max(0, mean_val - 0.5 * std_val)
        upper_thresh = min(255, mean_val + 0.8 * std_val)
        
        print(f"  🌾 Agriculture thresholds: lower={lower_thresh:.1f}, upper={upper_thresh:.1f}")
        
        # Create mask for medium brightness areas
        mask_brightness = cv2.inRange(blurred, lower_thresh, upper_thresh)
        
        # Create mask for textured areas
        texture_thresh = np.percentile(laplacian, 60)  # Areas with some texture
        mask_texture = (laplacian > texture_thresh).astype(np.uint8) * 255
        
        # Combine brightness and texture masks
        combined_mask = cv2.bitwise_and(mask_brightness, mask_texture)
        
        # Morphological operations to clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel, iterations=1)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        
        # Find contours
        contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        candidates = []
        for contour in contours:
            area = cv2.contourArea(contour)
            
            if area >= self.min_object_size:
                # Get shape metrics for agricultural validation
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = max(w, h) / max(min(w, h), 1)
                
                # Agriculture can be rectangular fields
                if aspect_ratio < 12.0 and w >= 4 and h >= 4:  # Allow elongated fields
                    M = cv2.moments(contour)
                    if M["m00"] != 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                        
                        full_x = x1 + cx
                        full_y = y1 + cy
                        candidates.append((full_x, full_y))
        
        print(f"  🌾 Found {len(candidates)} agriculture candidates")
        return candidates
    
    def get_merge_buffer_size(self):
        """Agriculture: Allow more aggressive merging"""
        return 5
    
    def get_iou_threshold(self):
        """Agriculture: Allow merging of adjacent areas"""
        return 0.1
    
    def should_merge_duplicates(self):
        """Agriculture: Merge duplicates"""
        return True