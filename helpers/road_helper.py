"""
Road Detection Helper

Specialized helper for road detection using OpenCV algorithms.
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
        """Simple, guaranteed OpenCV road detection"""
        x1, y1, x2, y2 = bbox
        
        # Convert to grayscale
        if len(bbox_image.shape) == 3:
            gray = cv2.cvtColor(bbox_image, cv2.COLOR_RGB2GRAY)
        else:
            gray = bbox_image
        
        print(f"  🛣️ OPENCV: Simple road detection on {gray.shape[1]}x{gray.shape[0]}px image")
        print(f"  📊 Image stats: min={gray.min()}, max={gray.max()}, mean={gray.mean():.1f}")
        
        # Step 1: Enhanced edge detection for road boundaries
        # Use adaptive thresholding for better edge detection
        mean_val = gray.mean()
        std_val = gray.std()
        
        # Dynamic Canny thresholds based on image statistics
        lower_thresh = max(50, int(mean_val - 0.5 * std_val))
        upper_thresh = min(200, int(mean_val + 0.5 * std_val))
        
        edges = cv2.Canny(gray, lower_thresh, upper_thresh, apertureSize=3)
        print(f"  🔍 EDGES: Adaptive thresholds ({lower_thresh}, {upper_thresh}), found {np.sum(edges > 0)} edge pixels")
        
        # Step 2: Detect both straight and curved road segments
        final_mask = np.zeros(gray.shape, dtype=np.uint8)
        
        # Part A: Hough lines for straight road segments
        lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=30, minLineLength=50, maxLineGap=10)
        
        straight_pixels = 0
        
        if lines is not None:
            print(f"  📏 HOUGH: Found {len(lines)} straight line segments")
            # Draw thick lines to represent straight roads
            for line in lines:
                x1_line, y1_line, x2_line, y2_line = line[0]
                # Calculate line length and angle
                length = np.sqrt((x2_line - x1_line)**2 + (y2_line - y1_line)**2)
                
                # Only keep longer lines (likely roads, not noise)
                if length > 40:
                    # Draw thick line to represent road width
                    thickness = min(15, max(8, int(length / 20)))  # Adaptive thickness
                    cv2.line(final_mask, (x1_line, y1_line), (x2_line, y2_line), 255, thickness)
                    straight_pixels += thickness * length
            
            print(f"  ✅ STRAIGHT: Generated {int(straight_pixels)} straight road pixels")
        else:
            print(f"  ❌ STRAIGHT: No straight road lines detected")
        
        # Part B: Curved road detection using contour analysis
        print(f"  🌀 CURVES: Detecting curved road segments")
        
        # Create a mask for potential curved roads
        # Use morphological operations to find road-like curves
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        edges_closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel_close, iterations=1)
        
        # Find contours that could be curved roads
        curve_contours, _ = cv2.findContours(edges_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        curved_pixels = 0
        curve_count = 0
        
        for contour in curve_contours:
            # Analyze contour for road-like curves
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)
            
            if area > 100 and perimeter > 80:  # Minimum size for road curves
                # Check if it's elongated and smooth (road-like)
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = max(w, h) / max(min(w, h), 1)
                
                # Calculate curve smoothness
                epsilon = 0.02 * perimeter
                approx = cv2.approxPolyDP(contour, epsilon, True)
                
                # Road curves should be elongated and not too angular
                if aspect_ratio > 2.0 and len(approx) > 4:
                    # Calculate road width based on area and perimeter
                    estimated_width = min(20, max(6, int(area / (perimeter / 2))))
                    
                    # Draw the curved road
                    cv2.drawContours(final_mask, [contour], -1, 255, estimated_width)
                    curved_pixels += area
                    curve_count += 1
        
        print(f"  🌀 CURVES: Found {curve_count} curved segments, {int(curved_pixels)} curve pixels")
        
        # Part C: Final cleanup and combination
        if straight_pixels > 0 or curved_pixels > 0:
            # Clean up the combined mask
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
            final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_OPEN, kernel, iterations=1)
            
            total_pixels = np.sum(final_mask > 0)
            print(f"  ✅ OPENCV: Combined road mask with {total_pixels} pixels (straight + curves)")
        else:
            print(f"  ❌ OPENCV: No roads detected (straight or curved)")
        
        # Find contours from OpenCV road mask
        contours, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        candidates = []
        print(f"  🛣️ OPENCV: Found {len(contours)} road segments")
        
        for i, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            
            # Simple validation - just check minimum size
            if area >= self.min_object_size * 0.5:  # More permissive for OpenCV detection
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = max(w, h) / max(min(w, h), 1)
                
                print(f"  🔍 ROAD SEGMENT {i}: area={area:.1f}px, {w}x{h}px, AR={aspect_ratio:.1f}")
                
                # Simple validation - roads should be somewhat elongated
                if aspect_ratio >= 1.5 and w >= 10 and h >= 10:
                    # Use center point for SAM
                    M = cv2.moments(contour)
                    if M["m00"] != 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                        
                        full_x = x1 + cx
                        full_y = y1 + cy
                        candidates.append((full_x, full_y))
                        print(f"  ✅ ROAD SEGMENT {i}: ACCEPTED -> center=({full_x},{full_y})")
                    else:
                        print(f"  ❌ ROAD SEGMENT {i}: Invalid moments")
                else:
                    print(f"  ❌ ROAD SEGMENT {i}: Failed validation (AR={aspect_ratio:.1f} < 1.5 or too small)")
            else:
                print(f"  ❌ ROAD SEGMENT {i}: Too small ({area:.1f} < {self.min_object_size * 0.5:.1f})")
        
        print(f"  🛣️ OPENCV: Found {len(candidates)} road candidates")
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

    def get_merge_buffer_size(self):
        """Roads: Allow moderate merging"""
        return 4
    
    def get_iou_threshold(self):
        """Roads: Allow merging of connected road segments"""
        return 0.2
    
    def should_merge_duplicates(self):
        """Roads: Merge connected segments"""
        return True