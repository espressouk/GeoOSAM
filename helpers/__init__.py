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
from .water_helper import WaterHelper
from .agriculture_helper import AgricultureHelper
from .road_helper import RoadHelper
from .general_helper import GeneralHelper
from .vessels_helper import VesselsHelper

def create_detection_helper(class_name, min_object_size=50, max_objects=25):
    """Factory to create appropriate helper for each class"""
    if class_name == 'Vegetation':
        return VegetationHelper(class_name, min_object_size, max_objects)
    elif class_name in ['Buildings', 'Industrial']:
        return BuildingsHelper(class_name, min_object_size, max_objects)
    elif class_name == 'Residential':
        return ResidentialHelper(class_name, min_object_size, max_objects)
    elif class_name in ['Vehicle', 'Cars']:
        return VehicleHelper(class_name, min_object_size, max_objects)
    elif class_name == 'Vessels':
        return VesselsHelper(class_name, min_object_size, max_objects)
    elif class_name == 'Water':
        return WaterHelper(class_name, min_object_size, max_objects)
    elif class_name == 'Agriculture':
        return AgricultureHelper(class_name, min_object_size, max_objects)
    elif class_name == 'Road':
        return RoadHelper(class_name, min_object_size, max_objects)
    else:
        # Default helper for other classes - use general helper as fallback
        return GeneralHelper(class_name, min_object_size, max_objects)

__all__ = [
    'BaseDetectionHelper',
    'VegetationHelper', 
    'ResidentialHelper',
    'VehicleHelper',
    'BuildingsHelper',
    'WaterHelper',
    'AgricultureHelper',
    'RoadHelper',
    'GeneralHelper',
    'VesselsHelper',
    'create_detection_helper'
]