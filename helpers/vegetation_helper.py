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
        
        # Auto-detect and normalize image values
        bbox_image = self._normalize_image_values(bbox_image)
        
        # Smart band selection and processing
        gray = self._prepare_vegetation_bands(bbox_image)
        
        # Apply texture detection using local standard deviation
        print(f"  🔍 TEXTURE: Processing band shape: {gray.shape}, dtype: {gray.dtype}, range: {gray.min()} to {gray.max()}")
        
        kernel_size = 9
        kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size * kernel_size)
        mean = cv2.filter2D(gray.astype(np.float32), -1, kernel)
        sqr_mean = cv2.filter2D((gray.astype(np.float32))**2, -1, kernel)
        # Ensure non-negative values before sqrt to avoid numerical errors
        variance = np.maximum(sqr_mean - mean**2, 0.0)
        texture = np.sqrt(variance)
        
        print(f"  🔍 TEXTURE: Raw texture range: {texture.min():.4f} to {texture.max():.4f}")
        
        # Normalize texture to 0-255
        if texture.max() > texture.min():
            texture_norm = ((texture - texture.min()) / (texture.max() - texture.min()) * 255).astype(np.uint8)
            print(f"  🔧 TEXTURE: Normalized texture range: {texture_norm.min()} to {texture_norm.max()}")
        else:
            texture_norm = np.zeros_like(texture, dtype=np.uint8)
            print(f"  ⚠️ TEXTURE: Constant texture detected - using zero array")
        
        # Adaptive threshold
        threshold = np.percentile(texture_norm, 70)
        print(f"  🎯 TEXTURE: 70th percentile threshold: {threshold}")
        _, binary = cv2.threshold(texture_norm, threshold, 255, cv2.THRESH_BINARY)
        print(f"  🎯 TEXTURE: Binary pixels above threshold: {np.sum(binary > 0)}/{binary.size} ({100*np.sum(binary > 0)/binary.size:.1f}%)")
        
        # Morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
        
        # Find contours and extract candidates
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        print(f"  🔍 CONTOURS: Found {len(contours)} total contours")
        
        candidates = []
        min_area_threshold = self.min_object_size * 0.5
        print(f"  🎯 CONTOURS: Min area threshold: {min_area_threshold}px")
        
        for i, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            if area >= min_area_threshold:
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    full_x = x1 + cx
                    full_y = y1 + cy
                    candidates.append((full_x, full_y))
                    print(f"    ✅ CONTOUR {i}: area={area:.1f}px, center=({cx},{cy}) -> ({full_x},{full_y})")
                else:
                    print(f"    ❌ CONTOUR {i}: area={area:.1f}px, invalid moments")
            else:
                print(f"    ❌ CONTOUR {i}: area={area:.1f}px < {min_area_threshold}px")
        
        print(f"  🌿 FINAL: Found {len(candidates)} vegetation texture candidates")
        return candidates
    
    def _normalize_image_values(self, image):
        """Auto-detect and normalize image values to 0-255 range"""
        print(f"  🔍 NORM: Input image shape: {image.shape}, dtype: {image.dtype}")
        
        if image.dtype == np.uint8:
            print(f"  ✅ NORM: Already 8-bit, range: {image.min()} to {image.max()}")
            return image  # Already 8-bit
        
        # Check if values are in reflectance range (0-1) or need scaling
        img_min, img_max = image.min(), image.max()
        print(f"  📊 NORM: Value range: {img_min:.6f} to {img_max:.6f}")
        
        # Check for no-data values
        unique_vals = np.unique(image)
        if len(unique_vals) < 10:
            print(f"  ⚠️ NORM: Few unique values detected: {unique_vals[:10]}")
        
        if img_max <= 1.0 and img_min >= 0.0:
            # Reflectance values - scale to 0-255
            normalized = (image * 255).astype(np.uint8)
            print(f"  🔧 NORM: Scaled reflectance to 0-255, result range: {normalized.min()} to {normalized.max()}")
        else:
            # Other range - normalize to 0-255
            if img_max > img_min:
                normalized = ((image - img_min) / (img_max - img_min) * 255).astype(np.uint8)
                print(f"  🔧 NORM: Normalized from [{img_min:.4f}, {img_max:.4f}] to 0-255, result: {normalized.min()} to {normalized.max()}")
            else:
                normalized = np.zeros_like(image, dtype=np.uint8)
                print(f"  ⚠️ NORM: Constant values detected - using zero array")
        
        return normalized
    
    def _prepare_vegetation_bands(self, bbox_image):
        """Smart band selection for vegetation detection"""
        print(f"  🔍 BANDS: Input shape: {bbox_image.shape}")
        
        if len(bbox_image.shape) == 2:
            # Single band - use as is
            print(f"  📡 BANDS: Single band image")
            return bbox_image
        
        bands = bbox_image.shape[2] if len(bbox_image.shape) == 3 else 1
        print(f"  📡 BANDS: Processing {bands}-band image")
        
        if bands >= 5:
            # Multi-spectral: Use NDVI-like calculation
            # Assume bands: Blue, Green, Red, NIR, RedEdge (common UAV setup)
            red_band = bbox_image[:, :, 2]   # Band 3 (Red)
            nir_band = bbox_image[:, :, 3]   # Band 4 (NIR)
            
            print(f"  🔍 BANDS: Red band range: {red_band.min():.4f} to {red_band.max():.4f}")
            print(f"  🔍 BANDS: NIR band range: {nir_band.min():.4f} to {nir_band.max():.4f}")
            
            # Calculate NDVI-like index
            red_f = red_band.astype(np.float32)
            nir_f = nir_band.astype(np.float32)
            
            # Avoid division by zero
            denominator = red_f + nir_f
            ndvi = np.zeros_like(red_f)
            mask = denominator > 0
            valid_pixels = np.sum(mask)
            print(f"  🔍 BANDS: Valid pixels for NDVI: {valid_pixels}/{mask.size} ({100*valid_pixels/mask.size:.1f}%)")
            
            if valid_pixels > 0:
                ndvi[mask] = (nir_f[mask] - red_f[mask]) / denominator[mask]
                ndvi_min, ndvi_max = ndvi[mask].min(), ndvi[mask].max()
                print(f"  🔍 BANDS: NDVI range: {ndvi_min:.4f} to {ndvi_max:.4f}")
                
                # Normalize NDVI to 0-255
                ndvi_norm = ((ndvi + 1) * 127.5).astype(np.uint8)  # NDVI range -1 to 1
                print(f"  🌿 BANDS: Using NDVI calculation, result range: {ndvi_norm.min()} to {ndvi_norm.max()}")
                return ndvi_norm
            else:
                print(f"  ⚠️ BANDS: No valid pixels for NDVI, falling back to green band")
                return bbox_image[:, :, 1]  # Green band fallback
            
        elif bands >= 3:
            # RGB: Enhanced green channel processing
            if bands == 3:
                # Standard RGB - use green channel (best for vegetation)
                green = bbox_image[:, :, 1]
                print(f"  🌿 Using green channel for vegetation detection")
                return green
            else:
                # RGB + extra bands - combine green and NIR if available
                green = bbox_image[:, :, 1]
                nir = bbox_image[:, :, 3] if bands > 3 else green
                # Weighted combination
                combined = (0.6 * green.astype(np.float32) + 0.4 * nir.astype(np.float32)).astype(np.uint8)
                print(f"  🌿 Using green+NIR combination")
                return combined
        else:
            # Single band or grayscale
            return cv2.cvtColor(bbox_image, cv2.COLOR_RGB2GRAY) if len(bbox_image.shape) == 3 else bbox_image