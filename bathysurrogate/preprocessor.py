import numpy as np
import logging
from typing import Dict, Any, Optional, Tuple
import glob
import os
import numpy as np

try:
    import ee
except ImportError:
    ee = None

logger = logging.getLogger(__name__)

def init_gee(project_id: str = 'YOUR_GEE_PROJECT_ID'):
    """Initialize Google Earth Engine API safely"""
    if ee is not None:
        try:
            ee.Initialize(project=project_id)
        except Exception:
            try:
                ee.Authenticate()
                ee.Initialize(project=project_id)
            except Exception as e:
                logger.warning(f"GEE Initialization warning: {e}")

class DataPreprocessor:
    """Data preprocessor"""
    
    def __init__(self, region: Optional[Any] = None):
        """
        Initialize preprocessor
        
        Args:
            region: Study area (optional)
        """
        self.region = region
        
    def process(
        self,
        sentinel_data: Dict[str, np.ndarray],
        gebco_data: np.ndarray,
        sparse_measurements: np.ndarray,
        measurement_coordinates: np.ndarray,        
        is_ee_image: bool = False
    ) -> Dict[str, Any]:
        """Data preprocessing"""
        try:
            # 1. Remote sensing data doesn't need normalization (classic model handles internally)
            
            # 2. GEBCO data normalization (sign conversion: negative depth -> positive depth, positive land -> negative land)
            # Note: No longer need manual sign conversion, handled in _normalize_data
            normalized_gebco, gebco_stats = self._normalize_data(
                gebco_data, 
                data_type='gebco'
            )
            
            # 3. Normalize sparse measurement data (nautical chart data)
            normalized_measurements, measurement_stats = self._normalize_data(
                sparse_measurements,
                data_type='chart'
            )
            
            # 4. Create measurement grid
            measurement_grid = self._create_measurement_grid(
                normalized_measurements,
                measurement_coordinates,
                gebco_data.shape[-2:]
            )
            
            return {
                'sentinel': sentinel_data,
                'gebco': normalized_gebco,
                'measurements': {
                    'values': normalized_measurements,
                    'coordinates': measurement_coordinates,
                    'grid': measurement_grid
                },
                'stats': {
                    'gebco': gebco_stats,
                    'measurements': measurement_stats
                }
            }
            
        except Exception as e:
            logger.error(f"Data preprocessing failed: {str(e)}")
            raise
            
    def _normalize_data(self, data: np.ndarray, data_type: str) -> Tuple[np.ndarray, Dict[str, float]]:
        """Use Min-Max normalization to map valid physical depths to [-1, 1] range"""
        try:
            normalized_data = data.astype(np.float32).copy() 
            
            # Set physical range and special values
            min_phys = 0.0
            max_phys = 90.0
            land_value_norm = 1.5  # Normalized land/invalid value
            eps = 1e-6  # Prevent division by zero
            
            stats = self._get_default_stats()
            stats['min_phys'] = min_phys
            stats['max_phys'] = max_phys
            stats['land_value'] = land_value_norm

            if data_type == 'gebco':
                # GEBCO: data < 0 is depth, data >= 0 is land
                sea_mask = data < 0
                land_mask = ~sea_mask
                if not np.any(sea_mask):
                    logger.warning(f"No valid depth data in GEBCO data")
                    normalized_data.fill(land_value_norm) 
                    return normalized_data, stats
                
                sea_depths_phys = -data[sea_mask]  # Convert to positive physical depth
                
                # Record original statistics
                stats.update({
                    'median': float(np.median(sea_depths_phys)), 
                    'iqr': float(np.percentile(sea_depths_phys, 75) - np.percentile(sea_depths_phys, 25)), 
                    'q25': float(np.percentile(sea_depths_phys, 25)), 
                    'q75': float(np.percentile(sea_depths_phys, 75)),
                    'min': float(sea_depths_phys.min()), 
                    'max': float(sea_depths_phys.max())
                })
                
                # Apply Min-Max normalization to [-1, 1]
                normalized_sea = 2 * (sea_depths_phys - min_phys) / (max_phys - min_phys + eps) - 1
                # Clip to ensure values are within [-1, 1] for numerical stability
                normalized_sea = np.clip(normalized_sea, -1.0, 1.0)
                
                normalized_data[sea_mask] = normalized_sea
                normalized_data[land_mask] = land_value_norm  # Set land to special value

            elif data_type in ['classic', 'chart']:
                # Classic/Chart: data > 0 is depth, data <= 0 is invalid/land
                valid_mask = data > 0
                invalid_mask = ~valid_mask
                if not np.any(valid_mask):
                    logger.warning(f"No valid depth data in {data_type} data")
                    normalized_data.fill(land_value_norm)  # Fill with invalid value
                    return normalized_data, stats

                valid_depths_phys = data[valid_mask]  # Already positive physical depth
                
                # Record original statistics
                stats.update({
                    'median': float(np.median(valid_depths_phys)), 
                    'iqr': float(np.percentile(valid_depths_phys, 75) - np.percentile(valid_depths_phys, 25)), 
                    'q25': float(np.percentile(valid_depths_phys, 25)), 
                    'q75': float(np.percentile(valid_depths_phys, 75)),
                    'min': float(valid_depths_phys.min()), 
                    'max': float(valid_depths_phys.max()),
                    'invalid_value': land_value_norm  # Use unified land_value_norm
                })
                
                # Apply Min-Max normalization to [-1, 1]
                normalized_valid = 2 * (valid_depths_phys - min_phys) / (max_phys - min_phys + eps) - 1
                normalized_valid = np.clip(normalized_valid, -1.0, 1.0)

                normalized_data[valid_mask] = normalized_valid
                normalized_data[invalid_mask] = land_value_norm  # Set invalid/land to special value

            # Final check for NaNs introduced during processing
            if np.isnan(normalized_data).any():
                 logger.warning(f"{data_type} contains NaN after normalization, replacing with land/invalid value {land_value_norm}")
                 normalized_data = np.nan_to_num(normalized_data, nan=land_value_norm)

            self._check_normalized_data(normalized_data, data_type, stats)

            return normalized_data, stats

        except Exception as e:
            logger.error(f"Data normalization failed ({data_type}): {str(e)}")
            raise

    def _check_normalized_data(self, data: np.ndarray, data_type: str, stats: Dict[str, float]):
        """Enhanced normalized data check"""
        # Check for NaNs before processing stats
        if np.isnan(data).any():
            logger.warning(f"{data_type} data contains NaN during check (after filling?)")

        land_value = stats.get('land_value', 1.5)  # Get land/invalid value from stats
        # Use isclose to check land/invalid values
        valid_mask = ~np.isclose(data, land_value) & ~np.isnan(data)  # Exclude land/invalid and NaNs

        logger.info(f"{data_type} normalized data distribution:")
        land_ratio = np.mean(np.isclose(data, land_value))
        logger.info(f"- Land/invalid value ({land_value:.1f}) ratio: {land_ratio:.2%}")

        if np.any(valid_mask):
            valid_data = data[valid_mask]
            percentiles = np.percentile(valid_data, [0, 25, 50, 75, 100]) if len(valid_data) > 0 else ["N/A"]*5
            logger.info(f"- Valid value range: [{valid_data.min():.4f}, {valid_data.max():.4f}]")
            logger.info(f"- Valid value percentiles [0, 25, 50, 75, 100]: {percentiles}")
            logger.info(f"- Original physical value range: [{stats.get('min', 'N/A'):.2f}, {stats.get('max', 'N/A'):.2f}]")
            if 'iqr' in stats:
                logger.info(f"- Original physical value IQR: {stats['iqr']:.2f}")
        else:
            logger.warning(f"- No valid values in {data_type} data (non-land/invalid)")


    def _get_default_stats(self) -> Dict[str, float]:
        """Extended default statistics"""
        # Provide more robust defaults, especially min/max
        return {
            'median': 0.0, 'iqr': 1.0, 'q25': -0.5, 'q75': 0.5,
            'land_value': 1.5, 'invalid_value': 1.5,  # Keep both for clarity
            'min': 0.0, 'max': 0.0
        }
            
    def _create_measurement_grid(
        self,
        measurements: np.ndarray,
        coordinates: np.ndarray,
        shape: Tuple[int, int]
    ) -> np.ndarray:
        """Create measurement grid"""
        try:
            H, W = shape
            grid = np.zeros((H, W))
            
            # Convert normalized coordinates to pixel coordinates
            pixel_coords = np.zeros_like(coordinates)
            pixel_coords[:, 0] = coordinates[:, 0] * (H - 1)  # y coordinate
            pixel_coords[:, 1] = coordinates[:, 1] * (W - 1)  # x coordinate
            
            # Round to nearest integer coordinates
            pixel_coords = np.round(pixel_coords).astype(int)
            
            # Boundary check
            pixel_coords[:, 0] = np.clip(pixel_coords[:, 0], 0, H - 1)
            pixel_coords[:, 1] = np.clip(pixel_coords[:, 1], 0, W - 1)
            
            # Fill grid
            for coord, value in zip(pixel_coords, measurements):
                grid[coord[0], coord[1]] = value
            
            logger.info(f"Created measurement grid with shape {shape}")
            return grid
            
        except Exception as e:
            logger.error(f"Measurement grid creation failed: {str(e)}")
            raise
            
    def get_sparse_points(self) -> Dict[str, np.ndarray]:
        """Get nautical chart sparse observation points"""
        try:
            if self.region is None:
                raise ValueError("Study region not set, please specify region parameter during initialization")
            
            # Get study area bounds
            bounds = self.region.bounds().getInfo()
            min_lon = bounds['coordinates'][0][0][0]  # West boundary
            min_lat = bounds['coordinates'][0][0][1]  # South boundary
            max_lon = bounds['coordinates'][0][2][0]  # East boundary
            max_lat = bounds['coordinates'][0][2][1]  # North boundary
        
            logger.info(f"Retrieving nautical chart data within region [{min_lon:.3f}, {min_lat:.3f}, {max_lon:.3f}, {max_lat:.3f}]...")
        
            # Read nautical chart data
            chart_files = glob.glob(os.path.join('Official_nautical_chart', '*_sample.xyz'))
            if not chart_files:
                raise FileNotFoundError("Nautical chart data files not found")
        
            points_data = []
            for file in chart_files:
                data = np.loadtxt(file)
                # Keep only points within study area
                mask = ((data[:, 0] >= min_lon) & (data[:, 0] <= max_lon) & 
                       (data[:, 1] >= min_lat) & (data[:, 1] <= max_lat))
                points_data.append(data[mask])
            
            if not points_data:
                raise ValueError("No nautical chart data points found in specified region")
            
            points_data = np.vstack(points_data)
        
            # Extract coordinates and depths
            lon = points_data[:, 0]
            lat = points_data[:, 1]
            depths = points_data[:, 2]
        
            # Convert geographic coordinates to normalized grid coordinates (0-1 range)
            norm_coords = np.zeros((len(lon), 2))
            norm_coords[:, 0] = (lat - min_lat) / (max_lat - min_lat)
            norm_coords[:, 1] = (lon - min_lon) / (max_lon - min_lon)
        
            logger.info(f"Found {len(depths)} nautical chart observation points in study region")
            logger.info(f"Depth range: {np.min(depths):.2f} to {np.max(depths):.2f} m")
        
            return {
                'depths': depths,  # Return original depth values without normalization
                'coordinates': norm_coords
            }
        
        except Exception as e:
            logger.error(f"Failed to retrieve nautical chart data: {str(e)}")
            raise