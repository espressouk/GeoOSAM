"""
Vehicle Detection Helper

Specialized helper for vehicle detection optimized for batch mode.
"""

import cv2
import numpy as np
from .base_helper import BaseDetectionHelper


class VehicleHelper(BaseDetectionHelper):
    """Specialized helper for vehicle detection in batch mode"""
    
    def __init__(self, class_name="Vehicle", min_object_size=15, max_objects=60):
        super().__init__(class_name, min_object_size, max_objects)
    
    def detect_candidates(self, bbox_image, bbox):
        """Vehicle-specific bright object detection optimized for batch"""
        return self._detect_vehicle_objects(bbox_image, bbox)
    
    def validate_object(self, component_mask, component_area, min_object_size):
        """Vehicle-specific validation"""
        contours, _ = cv2.findContours(component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return False, "no contours"
        
        main_contour = max(contours, key=cv2.contourArea)
        metrics = self.get_basic_shape_metrics(main_contour)
        
        # Vehicle-specific validation - vehicles are very small, rectangular, solid
        is_valid = (
            metrics['aspect_ratio'] >= 1.2 and      # Vehicles have some elongation
            metrics['aspect_ratio'] <= 3.0 and      # But not too elongated
            metrics['solidity'] >= 0.7 and          # Vehicles are very solid rectangles
            metrics['area'] <= 250 and              # Vehicles are very small
            metrics['area'] >= min_object_size * 0.9  # Close to minimum size
        )
        
        if not is_valid:
            if metrics['aspect_ratio'] < 1.2:
                return False, f"too square ({metrics['aspect_ratio']:.1f} < 1.2)"
            elif metrics['aspect_ratio'] > 3.0:
                return False, f"too elongated ({metrics['aspect_ratio']:.1f} > 3.0)"
            elif metrics['solidity'] < 0.7:
                return False, f"not solid enough ({metrics['solidity']:.2f} < 0.7)"
            elif metrics['area'] > 250:
                return False, f"too large ({metrics['area']} > 250)"
            elif metrics['area'] < min_object_size * 0.9:
                return False, f"too small ({metrics['area']} < {min_object_size * 0.9})"
        
        return True, "valid vehicle"
    
    def apply_morphology(self, mask):
        """Vehicle-specific morphology"""
        # Use rectangular kernel for vehicle shapes
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        return mask
    
    def should_merge_masks(self):
        return False  # Vehicles should stay separate - don't merge
    
    def get_merge_buffer_size(self):
        """Vehicle: minimal merging (1-2px buffer)"""
        return 1
    
    def get_iou_threshold(self):
        """Vehicle: Moderate overlap allowed"""
        return 0.4
    
    def should_merge_duplicates(self):
        """Vehicle: Merge duplicates"""
        return True
    
    def get_background_threshold(self, bbox_area):
        """Vehicle-specific background threshold"""
        return bbox_area * 0.8  # High threshold - vehicles are small objects
    
    def _detect_vehicle_objects(self, bbox_image, bbox):
        """Enhanced vehicle detection optimized for batch mode"""
        x1, y1, x2, y2 = bbox
        
        # Convert to grayscale
        if len(bbox_image.shape) == 3:
            gray = cv2.cvtColor(bbox_image, cv2.COLOR_RGB2GRAY)
        else:
            gray = bbox_image
        
        print(f"  📊 Image stats: min={gray.min()}, max={gray.max()}, mean={gray.mean():.1f}")
        
        # BEST PRACTICE 1: Minimal noise reduction (preserve small vehicle details)
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        
        # BEST PRACTICE 2: Multi-threshold approach for vehicles
        mean_val = blurred.mean()
        std_val = blurred.std()
        
        # Vehicles are typically bright objects (metal, paint reflects light)
        threshold_bright = min(255, mean_val + 1.5 * std_val)  # Bright vehicles
        threshold_medium = min(255, mean_val + 0.8 * std_val)  # Medium bright vehicles
        
        # Otsu's method for adaptive segmentation
        otsu_thresh, _ = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Use most permissive threshold to catch various vehicle colors
        final_threshold = min(threshold_medium, otsu_thresh * 0.9)
        
        print(f"  🚗 Vehicle thresholds: bright={threshold_bright:.1f}, medium={threshold_medium:.1f}, otsu={otsu_thresh}, final={final_threshold:.1f}")
        
        # Create binary mask for bright objects
        _, binary = cv2.threshold(blurred, final_threshold, 255, cv2.THRESH_BINARY)
        
        # BEST PRACTICE 3: Minimal morphological operations to preserve vehicle shapes
        # Use small kernel to avoid merging separate vehicles
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        
        # Fill small gaps in vehicles
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)
        # Remove small noise
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
        
        # BEST PRACTICE 4: Vehicle-specific contour analysis
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        candidates = []
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # Size filtering - vehicles are very small objects
            if area >= self.min_object_size and area <= 300:
                # BEST PRACTICE 5: Vehicle shape analysis
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = max(w, h) / max(min(w, h), 1)
                
                # Calculate shape metrics for vehicle characteristics
                perimeter = cv2.arcLength(contour, True)
                compactness = (4 * np.pi * area) / (perimeter * perimeter) if perimeter > 0 else 0
                
                # Calculate solidity (vehicles are solid rectangles)
                hull = cv2.convexHull(contour)
                hull_area = cv2.contourArea(hull)
                solidity = area / hull_area if hull_area > 0 else 0
                
                # Calculate extent (fill ratio of bounding box)
                extent = area / (w * h) if w * h > 0 else 0
                
                # Vehicle characteristics - very small, compact, rectangular
                if (aspect_ratio >= 1.2 and           # Vehicles have some elongation
                    aspect_ratio <= 3.0 and          # But not too elongated
                    compactness > 0.3 and            # Vehicles are compact objects
                    solidity > 0.7 and               # Vehicles are very solid shapes
                    extent > 0.6 and                 # Vehicles fill their bounding box well
                    w >= 3 and h >= 3 and            # Minimum vehicle dimensions
                    w <= 20 and h <= 20):            # Maximum vehicle dimensions (very small)
                    
                    M = cv2.moments(contour)
                    if M["m00"] != 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                        
                        full_x = x1 + cx
                        full_y = y1 + cy
                        candidates.append((full_x, full_y))
        
        print(f"  🚗 Found {len(candidates)} vehicle candidates")
        return candidates