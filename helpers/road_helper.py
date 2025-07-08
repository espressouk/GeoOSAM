"""
Road Detection Helper

Specialized helper for road detection using linear feature detection.
"""

import cv2
import numpy as np
from .base_helper import BaseDetectionHelper


class RoadHelper(BaseDetectionHelper):
    """Specialized helper for road detection"""
    
    def __init__(self, class_name="Road", min_object_size=100, max_objects=20):
        super().__init__(class_name, min_object_size, max_objects)
    
    def detect_candidates(self, bbox_image, bbox):
        """Road-specific linear feature detection"""
        return self._detect_linear_objects(bbox_image, bbox)
    
    def validate_object(self, component_mask, component_area, min_object_size):
        """Road-specific validation"""
        contours, _ = cv2.findContours(component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return False, "no contours"
        
        main_contour = max(contours, key=cv2.contourArea)
        metrics = self.get_basic_shape_metrics(main_contour)
        
        # Roads can be individual segments or networks
        is_valid = (
            metrics['aspect_ratio'] >= 1.5 and      # Roads should have some elongation
            metrics['aspect_ratio'] <= 30.0 and     # Allow for road networks
            metrics['solidity'] >= 0.2 and          # Roads can be irregular (networks)
            metrics['area'] >= min_object_size * 0.6  # Lower threshold for road segments
        )
        
        if not is_valid:
            if metrics['aspect_ratio'] < 1.5:
                return False, f"not elongated enough ({metrics['aspect_ratio']:.1f} < 1.5)"
            elif metrics['aspect_ratio'] > 30.0:
                return False, f"too elongated ({metrics['aspect_ratio']:.1f} > 30.0)"
            elif metrics['solidity'] < 0.2:
                return False, f"not solid enough ({metrics['solidity']:.2f} < 0.2)"
            elif metrics['area'] < min_object_size * 0.6:
                return False, f"too small ({metrics['area']} < {min_object_size * 0.6})"
        
        return True, "valid road"
    
    def apply_morphology(self, mask):
        """Road-specific morphology"""
        # Use rectangular kernel for linear features
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        return mask
    
    def should_merge_masks(self):
        return True  # Roads can be merged if connected
    
    def get_background_threshold(self, bbox_area):
        """Road-specific background threshold"""
        return bbox_area * 0.5  # Medium threshold
    
    def _detect_linear_objects(self, bbox_image, bbox):
        """Detect roads using linear feature detection"""
        x1, y1, x2, y2 = bbox
        
        # Convert to grayscale
        if len(bbox_image.shape) == 3:
            gray = cv2.cvtColor(bbox_image, cv2.COLOR_RGB2GRAY)
        else:
            gray = bbox_image
        
        print(f"  📊 Image stats: min={gray.min()}, max={gray.max()}, mean={gray.mean():.1f}")
        
        # Road detection approach
        # Step 1: Gaussian blur for noise reduction
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        
        # Step 2: Edge detection for linear features
        edges = cv2.Canny(blurred, 50, 150, apertureSize=3)
        
        # Step 3: Hough Line Transform to detect linear features
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, minLineLength=30, maxLineGap=10)
        
        # Step 4: Create mask from detected lines
        line_mask = np.zeros_like(gray)
        if lines is not None:
            for line in lines:
                x1_line, y1_line, x2_line, y2_line = line[0]
                cv2.line(line_mask, (x1_line, y1_line), (x2_line, y2_line), 255, 3)
        
        # Step 5: Also use brightness-based detection for paved roads
        mean_val = blurred.mean()
        std_val = blurred.std()
        
        # Roads can be bright (asphalt) or medium brightness (concrete)
        road_threshold_low = max(0, mean_val - 0.3 * std_val)
        road_threshold_high = min(255, mean_val + 1.0 * std_val)
        
        print(f"  🛣️ Road thresholds: low={road_threshold_low:.1f}, high={road_threshold_high:.1f}")
        
        # Create mask for medium-bright areas (roads)
        mask_brightness = cv2.inRange(blurred, road_threshold_low, road_threshold_high)
        
        # Combine line detection with brightness
        combined_mask = cv2.bitwise_or(line_mask, mask_brightness)
        
        # Step 6: Aggressive morphological operations to connect road segments
        # Use larger kernels to connect road fragments
        kernel_rect_h = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3))  # Horizontal roads
        kernel_rect_v = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 15))  # Vertical roads
        kernel_square = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))   # General connectivity
        
        # Apply morphology in both directions with more iterations
        morph1 = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel_rect_h, iterations=3)
        morph2 = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel_rect_v, iterations=3)
        morph3 = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel_square, iterations=2)
        
        # Combine all directions
        combined_mask = cv2.bitwise_or(morph1, morph2)
        combined_mask = cv2.bitwise_or(combined_mask, morph3)
        
        # Final dilation to merge nearby road segments
        kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_DILATE, kernel_dilate, iterations=2)
        
        # Find contours
        contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        candidates = []
        for contour in contours:
            area = cv2.contourArea(contour)
            
            if area >= max(200, self.min_object_size):  # Increase minimum size for road networks
                # Get shape metrics for road validation
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = max(w, h) / max(min(w, h), 1)
                
                # Roads should be elongated but allow for road networks
                if aspect_ratio >= 1.5 and aspect_ratio <= 50.0 and w >= 8 and h >= 8:
                    M = cv2.moments(contour)
                    if M["m00"] != 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                        
                        full_x = x1 + cx
                        full_y = y1 + cy
                        candidates.append((full_x, full_y))
        
        print(f"  🛣️ Found {len(candidates)} road candidates")
        return candidates