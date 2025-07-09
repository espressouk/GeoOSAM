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
        """Road-specific linear feature detection with grouping for connected networks"""
        candidates = self._detect_linear_objects(bbox_image, bbox)
        
        # Group nearby candidates for connected processing
        grouped_candidates = self._group_nearby_candidates(candidates, max_distance=100)
        
        print(f"🛣️ Road helper: {len(candidates)} candidates -> {len(grouped_candidates)} groups")
        
        return grouped_candidates
    
    def validate_object(self, component_mask, component_area, min_object_size):
        """Road-specific validation"""
        contours, _ = cv2.findContours(component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return False, "no contours"
        
        main_contour = max(contours, key=cv2.contourArea)
        metrics = self.get_basic_shape_metrics(main_contour)
        
        # Roads can be individual segments or networks - be permissive
        is_valid = (
            metrics['aspect_ratio'] >= 1.3 and      # Allow slightly less elongated shapes for networks
            metrics['aspect_ratio'] <= 50.0 and     # Very permissive for complex road networks
            metrics['solidity'] >= 0.15 and         # Very permissive for irregular networks
            metrics['area'] >= min_object_size * 0.5  # Lower threshold for road segments
        )
        
        if not is_valid:
            if metrics['aspect_ratio'] < 1.3:
                return False, f"not elongated enough ({metrics['aspect_ratio']:.1f} < 1.3)"
            elif metrics['aspect_ratio'] > 50.0:
                return False, f"too elongated ({metrics['aspect_ratio']:.1f} > 50.0)"
            elif metrics['solidity'] < 0.15:
                return False, f"not solid enough ({metrics['solidity']:.2f} < 0.15)"
            elif metrics['area'] < min_object_size * 0.5:
                return False, f"too small ({metrics['area']} < {min_object_size * 0.5})"
        
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
        """Smart road detection: find center lines, then expand to full road width"""
        x1, y1, x2, y2 = bbox
        
        # Convert to grayscale
        if len(bbox_image.shape) == 3:
            gray = cv2.cvtColor(bbox_image, cv2.COLOR_RGB2GRAY)
        else:
            gray = bbox_image
        
        print(f"  📊 Image stats: min={gray.min()}, max={gray.max()}, mean={gray.mean():.1f}")
        
        # Step 1: Find center lines using skeletonization approach
        # Roads are typically medium brightness
        mean_val = gray.mean()
        std_val = gray.std()
        
        # Initial road detection (adaptive thresholds for different road types)
        # Roads can be darker or brighter than background
        road_threshold_low = max(0, mean_val - 1.2 * std_val)
        road_threshold_high = min(255, mean_val + 1.2 * std_val)
        
        print(f"  🛣️ Road brightness range: {road_threshold_low:.1f} - {road_threshold_high:.1f}")
        
        # Create binary mask for road-like areas (more inclusive)
        # Try both darker and brighter roads
        dark_roads = cv2.inRange(gray, 0, int(mean_val - 0.3 * std_val))
        bright_roads = cv2.inRange(gray, int(mean_val + 0.3 * std_val), 255)
        medium_roads = cv2.inRange(gray, int(road_threshold_low), int(road_threshold_high))
        
        # Combine all road possibilities
        road_mask = cv2.bitwise_or(dark_roads, bright_roads)
        road_mask = cv2.bitwise_or(road_mask, medium_roads)
        
        # Step 2: Clean up the mask to get road shapes
        kernel_clean = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        road_mask = cv2.morphologyEx(road_mask, cv2.MORPH_OPEN, kernel_clean, iterations=1)
        road_mask = cv2.morphologyEx(road_mask, cv2.MORPH_CLOSE, kernel_clean, iterations=2)
        
        # Step 3: Find skeletons (center lines) of road areas
        try:
            from skimage import morphology
            # Convert to binary for skeletonization
            binary = road_mask > 0
            # Get skeleton (center lines)
            skeleton = morphology.skeletonize(binary)
            skeleton = (skeleton * 255).astype(np.uint8)
        except ImportError:
            # Fallback: Use morphological erosion to find centerlines
            print("  ⚠️ skimage not available, using morphological centerline detection")
            
            # Create different directional kernels to detect centerlines
            kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 7))  # Detect horizontal centerlines
            kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 1))  # Detect vertical centerlines
            
            # Erode with different kernels to get centerlines
            centerline_h = cv2.morphologyEx(road_mask, cv2.MORPH_ERODE, kernel_h, iterations=3)
            centerline_v = cv2.morphologyEx(road_mask, cv2.MORPH_ERODE, kernel_v, iterations=3)
            
            # Combine centerlines
            skeleton = cv2.bitwise_or(centerline_h, centerline_v)
            
            # Clean up skeleton
            kernel_clean = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            skeleton = cv2.morphologyEx(skeleton, cv2.MORPH_CLOSE, kernel_clean, iterations=1)
        
        # Step 4: Expand skeleton back to realistic road width
        # Use different kernels for different road orientations
        kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 7))  # Horizontal roads
        kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 15))  # Vertical roads
        kernel_d = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))  # Diagonal roads
        
        # Expand skeleton in all directions
        expanded_h = cv2.morphologyEx(skeleton, cv2.MORPH_DILATE, kernel_h, iterations=1)
        expanded_v = cv2.morphologyEx(skeleton, cv2.MORPH_DILATE, kernel_v, iterations=1)
        expanded_d = cv2.morphologyEx(skeleton, cv2.MORPH_DILATE, kernel_d, iterations=1)
        
        # Combine all expansions
        final_mask = cv2.bitwise_or(expanded_h, expanded_v)
        final_mask = cv2.bitwise_or(final_mask, expanded_d)
        
        # Intersect with original road areas to avoid expanding into non-road areas
        final_mask = cv2.bitwise_and(final_mask, road_mask)
        
        # Final cleanup
        final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_CLOSE, kernel_clean, iterations=1)
        
        # Find contours
        contours, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        candidates = []
        for contour in contours:
            area = cv2.contourArea(contour)
            
            if area >= self.min_object_size * 0.3:  # Lower threshold for center-line approach
                # Check if it's road-like (elongated)
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = max(w, h) / max(min(w, h), 1)
                
                # Roads should be elongated and have minimum dimensions
                if aspect_ratio >= 1.2 and w >= 5 and h >= 5:  # More permissive for center-line
                    M = cv2.moments(contour)
                    if M["m00"] != 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                        
                        full_x = x1 + cx
                        full_y = y1 + cy
                        candidates.append((full_x, full_y))
        
        print(f"  🛣️ Found {len(candidates)} road candidates using center-line approach")
        return candidates
    
    def _group_nearby_candidates(self, candidate_points, max_distance=100):
        """Group nearby candidate points using spatial clustering"""
        if len(candidate_points) <= 1:
            return candidate_points
        
        # Convert to numpy array for easier processing
        points = np.array(candidate_points)
        n_points = len(points)
        
        # Calculate distance matrix
        distances = np.zeros((n_points, n_points))
        for i in range(n_points):
            for j in range(i + 1, n_points):
                dist = np.sqrt((points[i][0] - points[j][0])**2 + (points[i][1] - points[j][1])**2)
                distances[i][j] = dist
                distances[j][i] = dist
        
        # Simple clustering: group points within max_distance
        groups = []
        used = set()
        
        for i in range(n_points):
            if i in used:
                continue
                
            # Start a new group
            group = [candidate_points[i]]
            used.add(i)
            
            # Find all points within max_distance
            for j in range(n_points):
                if j not in used and distances[i][j] <= max_distance:
                    group.append(candidate_points[j])
                    used.add(j)
            
            groups.append(group)
        
        return groups