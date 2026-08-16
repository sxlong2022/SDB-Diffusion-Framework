import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)

def validate_data(
    sentinel_data: Dict[str, np.ndarray],
    gebco_data: np.ndarray,
    sparse_measurements: np.ndarray,
    measurement_coordinates: np.ndarray
) -> bool:
    """Validate input data"""
    try:
        # Validate Sentinel data
        required_bands = ['blue', 'green']
        for band in required_bands:
            if band not in sentinel_data:
                raise ValueError(f"Missing required band: {band}")
                
        # Validate GEBCO data
        if gebco_data.ndim != 3:  # [T, H, W]
            raise ValueError("GEBCO data dimension incorrect")
            
        # Validate sparse measurements
        if sparse_measurements.ndim != 1:
            raise ValueError("Sparse measurement data dimension incorrect")
            
        if measurement_coordinates.shape[1] != 2:
            raise ValueError("Measurement coordinates must be 2D (x,y)")
            
        if len(sparse_measurements) != len(measurement_coordinates):
            raise ValueError("Number of measurements does not match number of coordinates")
            
        return True
        
    except Exception as e:
        logger.error(f"Data validation failed: {str(e)}")
        raise

def create_spatiotemporal_grid(
    coordinates: np.ndarray,
    values: np.ndarray,
    shape: Tuple[int, int],
    time_range: Optional[List[int]] = None
) -> np.ndarray:
    """Create spatiotemporal grid"""
    try:
        H, W = shape
        if time_range is None:
            time_range = list(range(2018, 2024))
        T = len(time_range)
        
        # Create grid
        grid = np.zeros((T, H, W))
        
        # Normalize coordinates
        norm_coords = coordinates.copy()
        norm_coords[:, 0] = norm_coords[:, 0] * (H - 1)
        norm_coords[:, 1] = norm_coords[:, 1] * (W - 1)
        
        # Fill values
        for t in range(T):
            for val, (y, x) in zip(values, norm_coords):
                y_idx = int(round(y))
                x_idx = int(round(x))
                if 0 <= y_idx < H and 0 <= x_idx < W:
                    grid[t, y_idx, x_idx] = val
                    
        return grid
        
    except Exception as e:
        logger.error(f"Spatiotemporal grid creation failed: {str(e)}")
        raise

def normalize_data(
    data: np.ndarray,
    min_val: Optional[float] = None,
    max_val: Optional[float] = None
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Data normalization
    
    Args:
        data: Input data
        min_val: Minimum value (optional)
        max_val: Maximum value (optional)
        
    Returns:
        normalized_data: Normalized data
        stats: Statistics (min, max)
    """
    try:
        # Handle NaN values
        data = np.nan_to_num(data, nan=np.nanmean(data))
        
        # Get data range
        if min_val is None:
            min_val = np.percentile(data, 1)  # Use 1st percentile to avoid outliers
        if max_val is None:
            max_val = np.percentile(data, 99)  # Use 99th percentile to avoid outliers
            
        # Normalize to [0,1] range
        normalized_data = (data - min_val) / (max_val - min_val + 1e-8)
        normalized_data = np.clip(normalized_data, 0, 1)
        
        stats = {
            'min': float(min_val),
            'max': float(max_val)
        }
        
        return normalized_data, stats
        
    except Exception as e:
        logger.error(f"Data normalization failed: {str(e)}")
        raise

def calculate_uncertainty(
    predictions: Dict[str, np.ndarray],
    measurements: Optional[Dict[str, np.ndarray]] = None,
    confidence_threshold: float = 0.8
) -> np.ndarray:
    """
    Calculate prediction uncertainty
    
    Args:
        predictions: Predictions from different models
        measurements: Measured data (optional)
        confidence_threshold: Confidence threshold
        
    Returns:
        uncertainty: Uncertainty estimate
    """
    try:
        # 1. Calculate inter-model standard deviation
        all_predictions = np.stack([pred for pred in predictions.values()])
        model_std = np.std(all_predictions, axis=0)
        
        # 2. Calculate deviation from measurements (if available)
        measurement_error = np.zeros_like(model_std)
        if measurements is not None and 'depths' in measurements and 'coordinates' in measurements:
            depths = measurements['depths']
            coords = measurements['coordinates']
            
            for depth, (y, x) in zip(depths, coords):
                y_idx = int(y * model_std.shape[0])
                x_idx = int(x * model_std.shape[1])
                
                # Calculate prediction error at each measurement point
                for pred in predictions.values():
                    error = np.abs(pred[y_idx, x_idx] - depth)
                    measurement_error[y_idx, x_idx] = max(
                        measurement_error[y_idx, x_idx],
                        error
                    )
        
        # 3. Combined uncertainty estimate
        uncertainty = (model_std + measurement_error) / 2
        
        # 4. Normalize
        uncertainty = (uncertainty - uncertainty.min()) / (uncertainty.max() - uncertainty.min() + 1e-8)
        
        # 5. Apply confidence threshold
        uncertainty[uncertainty > confidence_threshold] = confidence_threshold
        
        return uncertainty
        
    except Exception as e:
        logger.error(f"Uncertainty calculation failed: {str(e)}")
        raise