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
        """Buildings-specific bright object detection"""
        return self._detect_bright_objects(bbox_image, bbox)
    
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
    
    def get_merge_buffer_size(self):
        """Buildings: NO merging - each detection should stay separate"""
        return 0
    
    def get_iou_threshold(self):
        """Buildings: Only merge if VERY high overlap (likely same building)"""
        return 0.7
    
    def should_merge_duplicates(self):
        """Buildings: Don't merge, just remove duplicates"""
        return False
    
    def get_background_threshold(self, bbox_area):
        """Buildings-specific background threshold"""
        return bbox_area * 0.6  # Medium threshold
    
    def _detect_bright_objects(self, bbox_image, bbox):
        """Enhanced building detection combining brightness with geometric features"""
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
            
            # Size filtering with stricter upper bound to exclude land patches
            if area >= self.min_object_size and area <= 4000:  # Reduced from 8000 to exclude large land areas
                # BEST PRACTICE 5: Enhanced shape analysis for building characteristics
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = max(w, h) / max(min(w, h), 1)
                
                # Calculate shape metrics to distinguish buildings from land
                perimeter = cv2.arcLength(contour, True)
                compactness = (4 * np.pi * area) / (perimeter * perimeter) if perimeter > 0 else 0
                
                # Calculate solidity (convex hull ratio) - buildings are more solid
                hull = cv2.convexHull(contour)
                hull_area = cv2.contourArea(hull)
                solidity = area / hull_area if hull_area > 0 else 0
                
                # Buildings have reasonable shape properties vs irregular land patches
                if (aspect_ratio < 4.0 and           # Buildings aren't too elongated
                    compactness > 0.15 and          # Buildings are more compact than land patches
                    solidity > 0.6 and              # Buildings are more solid/regular than land
                    w >= 5 and h >= 5 and           # Minimum building size
                    area / (w * h) > 0.4):          # Good fill ratio - buildings fill their bounding box better
                    
                    M = cv2.moments(contour)
                    if M["m00"] != 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                        
                        full_x = x1 + cx
                        full_y = y1 + cy
                        candidates.append((full_x, full_y))
        
        print(f"  🏢 Found {len(candidates)} enhanced building candidates")
        return candidates