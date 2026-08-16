"""
BathySurrogate Package
Modular environmental surrogate framework for satellite-derived bathymetry.
"""

__version__ = '1.0.0'

from .utils import validate_data, normalize_data, create_spatiotemporal_grid, calculate_uncertainty
from .preprocessor import DataPreprocessor
from .classic_models import ClassicModels
from .s3gm_wrapper import S3GMWrapper
from .main import HybridBathymetrySystem
from .gpu_memory import GPUMemoryManager

__all__ = [
    'HybridBathymetrySystem',
    'validate_data',
    'normalize_data',
    'create_spatiotemporal_grid',
    'calculate_uncertainty',
    'DataPreprocessor',
    'ClassicModels',
    'S3GMWrapper',
    'GPUMemoryManager'
]
