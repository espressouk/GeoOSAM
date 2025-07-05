"""
Geo-OSAM Detection Helpers

Clean architecture for class-specific object detection logic.
Each helper contains proven, isolated logic for different object types.
"""

from .base_helper import BaseDetectionHelper
from .vegetation_helper import VegetationHelper
from .residential_helper import ResidentialHelper
from .vehicle_helper import VehicleHelper
from .buildings_helper import BuildingsHelper

def create_detection_helper(class_name, min_object_size=50, max_objects=25):
    """Factory to create appropriate helper for each class"""
    if class_name == 'Vegetation':
        return VegetationHelper(class_name, min_object_size, max_objects)
    elif class_name in ['Buildings', 'Industrial']:
        return BuildingsHelper(class_name, min_object_size, max_objects)
    elif class_name == 'Residential':
        return ResidentialHelper(class_name, min_object_size, max_objects)
    elif class_name in ['Vehicle', 'Vessels']:
        return VehicleHelper(class_name, min_object_size, max_objects)
    else:
        # Default helper for other classes - use vehicle helper as fallback (bright objects)
        return VehicleHelper(class_name, min_object_size, max_objects)

__all__ = [
    'BaseDetectionHelper',
    'VegetationHelper', 
    'ResidentialHelper',
    'VehicleHelper',
    'BuildingsHelper',
    'create_detection_helper'
]