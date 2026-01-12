import os
import sys
from typing import Dict, Any, Optional, List, Tuple, Union
import numpy as np
import logging
import ee
from scipy.interpolate import RegularGridInterpolator
import torch
import cv2

# Import custom modules
from .preprocessor import DataPreprocessor
from .classic_models import ClassicModels
from .s3gm_wrapper import S3GMWrapper
from .utils import validate_data, create_spatiotemporal_grid
from .gpu_memory import GPUMemoryManager

logger = logging.getLogger(__name__)

class HybridBathymetrySystem:
    """Hybrid bathymetry estimation system"""
    
    def __init__(
        self,
        region: ee.Geometry,
        time_range: Dict[str, range],
        config_path: Optional[str] = None,
        use_gpu: bool = True
    ):
        """
        Initialize the system
        
        Args:
            region: Study area
            time_range: Time range configuration
            config_path: S3GM configuration file path
            use_gpu: Whether to use GPU
        """
        try:
            self.region = region
            self.time_range = time_range
            self.device = torch.device('cuda' if use_gpu and torch.cuda.is_available() else 'cpu')
            
            # Initialize modules
            self.preprocessor = DataPreprocessor(region)
            self.classic_models = None  # Set to None first, wait for external setting
            self.s3gm = S3GMWrapper(
                config_path or 'configs/s3gm_default.yaml'
            )
            
            logger.info(f"System initialization completed, using device: {self.device}")
            
        except Exception as e:
            logger.error(f"System initialization failed: {str(e)}")
            raise
            
    def _preprocess_data(
        self,
        sentinel_data: Dict[str, np.ndarray],
        gebco_data: np.ndarray,
        sparse_measurements: np.ndarray,
        measurement_coordinates: np.ndarray
    ) -> Dict[str, Any]:
        """Data preprocessing"""
        try:
            # Validate input data
            validate_data(
                sentinel_data, 
                gebco_data,
                sparse_measurements,
                measurement_coordinates
            )
            
            # Use preprocessor to process data
            processed_data = self.preprocessor.process(
                sentinel_data=sentinel_data,
                gebco_data=gebco_data,
                sparse_measurements=sparse_measurements,
                measurement_coordinates=measurement_coordinates,
                is_ee_image=False
            )
            
            logger.info("Data preprocessing completed")
            return processed_data
            
        except Exception as e:
            logger.error(f"Data preprocessing failed: {str(e)}")
            raise
            
    def process(
        self,
        sentinel_data: Dict[str, np.ndarray],
        gebco_data: np.ndarray,
        measurements: np.ndarray,
        measurement_coordinates: np.ndarray,
        target_years: Optional[List[int]] = None,
        is_ee_image: bool = False
    ) -> Dict[str, np.ndarray]:
        """Process input data and generate prediction results"""
        try:
            # 1. Preprocess data
            processed_data = self._preprocess_data(
                sentinel_data,
                gebco_data,
                measurements,
                measurement_coordinates
            )
            
            # 2. Use classic model for prediction
            classic_results = self.classic_models.predict(processed_data['sentinel'], 'rf')
            
            # 3. Normalize classic model prediction results
            normalized_classic, classic_stats = self.preprocessor._normalize_data(
                classic_results,
                data_type='classic'
            )
            
            # Log data range before and after normalization
            logger.info("Data range before and after normalization:")
            logger.info(f"GEBCO data - original range: [{np.nanmin(gebco_data):.2f}, {np.nanmax(gebco_data):.2f}]")
            logger.info(f"GEBCO data - after normalization: [{np.nanmin(processed_data['gebco']):.2f}, {np.nanmax(processed_data['gebco']):.2f}]")
            logger.info(f"Classic model prediction - original range: [{np.nanmin(classic_results):.2f}, {np.nanmax(classic_results):.2f}]")
            logger.info(f"Classic model prediction - after normalization: [{np.nanmin(normalized_classic):.2f}, {np.nanmax(normalized_classic):.2f}]")
            
            # 4. Prepare S3GM input data
            s3gm_input = {
                'sentinel-classic': normalized_classic,
                'gebco': processed_data['gebco'],
                'measurements': {
                    'depths': processed_data['measurements']['values'],
                    'coordinates': measurement_coordinates
                },
                'years': target_years or [2018, 2019, 2020, 2021, 2022, 2023],
                'stats': {
                    'gebco': processed_data['stats']['gebco'],
                    'classic': classic_stats
                }
            }
            
            # 5. Execute S3GM prediction
            s3gm_output = self.s3gm.predict(s3gm_input)
            
            return {
                'bathymetry': s3gm_output['depth'],
                'confidence': s3gm_output['confidence']
            }
            
        except Exception as e:
            logger.error(f"Processing failed: {str(e)}")
            raise
            
    def validate(
        self,
        results: Dict[str, np.ndarray],
        validation_data: np.ndarray,
        validation_coordinates: np.ndarray
    ) -> Dict[str, float]:
        """
        Validate prediction results
        
        Args:
            results: Prediction results
            validation_data: Validation data
            validation_coordinates: Validation point coordinates
            
        Returns:
            Dictionary of validation metrics
        """
        try:
            metrics = {}
            
            # Calculate metrics for each model result
            for model_name, result in results.items():
                model_metrics = self._calculate_metrics(
                    predictions=result,
                    ground_truth=validation_data,
                    coordinates=validation_coordinates
                )
                metrics[model_name] = model_metrics
                
            return metrics
            
        except Exception as e:
            logger.error(f"Result validation failed: {str(e)}")
            raise
            
    def _calculate_metrics(
        self,
        predictions: np.ndarray,
        ground_truth: np.ndarray,
        coordinates: np.ndarray
    ) -> Dict[str, float]:
        """Calculate evaluation metrics"""
        try:
            # Extract prediction values at validation point locations
            pred_values = self._extract_values_at_coordinates(predictions, coordinates)
            
            # Calculate metrics
            metrics = {
                'rmse': np.sqrt(np.mean((pred_values - ground_truth) ** 2)),
                'mae': np.mean(np.abs(pred_values - ground_truth)),
                'r2': 1 - np.sum((ground_truth - pred_values) ** 2) / \
                      np.sum((ground_truth - np.mean(ground_truth)) ** 2)
            }
            
            return metrics
            
        except Exception as e:
            logger.error(f"Metrics calculation failed: {str(e)}")
            raise
            
    def _extract_values_at_coordinates(
        self,
        data: np.ndarray,
        coordinates: np.ndarray
    ) -> np.ndarray:
        """Extract values at specified coordinate locations"""
        try:
            values = np.zeros(len(coordinates))
            
            for i, (y, x) in enumerate(coordinates):
                # Ensure coordinates are integers
                y_idx = int(round(y))
                x_idx = int(round(x))
                
                # Boundary check
                y_idx = np.clip(y_idx, 0, data.shape[0] - 1)
                x_idx = np.clip(x_idx, 0, data.shape[1] - 1)
                
                values[i] = data[y_idx, x_idx]
                
            return values
            
        except Exception as e:
            logger.error(f"Coordinate value extraction failed: {str(e)}")
            raise
            
    def cleanup(self):
        """Clean up resources"""
        try:
            # Clean up GPU memory
            if torch.cuda.is_available():
                GPUMemoryManager.clear_gpu_memory()
                
            logger.info("System resource cleanup completed")
            
        except Exception as e:
            logger.error(f"Resource cleanup failed: {str(e)}")
            raise
            
    def set_classic_models(self, classic_models):
        """Set classic models"""
        self.classic_models = classic_models
        self.s3gm.set_classic_models(classic_models)  # Also update classic models in S3GM wrapper
        logger.info("Classic models set")