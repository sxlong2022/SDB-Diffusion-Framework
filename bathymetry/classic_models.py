import numpy as np
from typing import Dict, Any
import logging
import yaml
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold

logger = logging.getLogger(__name__)

class ClassicModels:
    """Classic bathymetry inversion models"""
    
    def __init__(self, config_path: str = 'configs/classic_models.yaml'):
        """Initialize classic bathymetry inversion models"""
        self.config_path = Path(config_path)
        self.load_config()
        self.llm_model = LinearRegression()
        self.rf_model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=3,
            random_state=42
        )
    
    def load_config(self) -> None:
        """Load configuration file"""
        try:
            if not self.config_path.exists():
                logger.warning(f"Config file {self.config_path} does not exist, using default parameters")
                self._set_default_params()
                return
                
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            # Get RF model parameters
            rf_config = config.get('rf_params', {})
            model_params = rf_config.get('model_params', {})
            
            # Update RF model parameters
            self.rf_model = RandomForestRegressor(
                n_estimators=model_params.get('n_estimators', 500),
                max_depth=model_params.get('max_depth', 25),
                min_samples_split=model_params.get('min_samples_split', 2),
                min_samples_leaf=model_params.get('min_samples_leaf', 1),
                max_features=model_params.get('max_features', 'sqrt'),
                bootstrap=model_params.get('bootstrap', True),
                random_state=model_params.get('random_state', 42),
                n_jobs=model_params.get('n_jobs', -1),
                oob_score=model_params.get('oob_score', True)
            )
            
            # Save feature engineering parameters
            self.rf_params = {
                'feature_engineering': rf_config.get('feature_engineering', {
                    'normalization': {'use_standard_scaling': True},
                    'band_ratio': {'use_log_transform': True},
                    'depth_index': {'use_normalized_bands': True}
                }),
                'stats': {},  # Will be updated during training
                'feature_importances': {}  # Will be updated during training
            }
            
        except Exception as e:
            logger.error(f"Failed to load config file: {str(e)}")
            self._set_default_params()
    
    def _set_default_params(self) -> None:
        """Set default parameters"""
        self.llm_params = {}
        self.rf_params = {}

    def train_rf(self, blue_band: np.ndarray, green_band: np.ndarray, depth: np.ndarray) -> float:
        """Train Random Forest model"""
        try:
            # 1. Data validation
            valid_mask = np.logical_and.reduce((
                ~np.isnan(blue_band),
                ~np.isnan(green_band),
                ~np.isnan(depth),
                blue_band > 0,
                green_band > 0,
                depth >= 0.1,
                depth <= 75.0
            ))
        
            # 2. Enhanced feature engineering
            # 2.1 Basic features (standardized)
            blue_norm = (blue_band - np.mean(blue_band[valid_mask])) / np.std(blue_band[valid_mask])
            green_norm = (green_band - np.mean(green_band[valid_mask])) / np.std(green_band[valid_mask])
        
            # 2.2 Band ratio features
            band_ratio = np.log(blue_band / green_band)
            band_ratio_norm = (band_ratio - np.mean(band_ratio[valid_mask])) / np.std(band_ratio[valid_mask])
        
            # 2.3 Log-transformed features
            log_blue = np.log(blue_band)
            log_green = np.log(green_band)
            log_ratio = log_blue - log_green
        
            # 2.4 Depth-related features
            depth_index = (blue_norm - green_norm) / (blue_norm + green_norm)
        
            X = np.column_stack((
                blue_norm[valid_mask],
                green_norm[valid_mask],
                band_ratio_norm[valid_mask],
                log_blue[valid_mask],
                log_green[valid_mask],
                log_ratio[valid_mask],
                depth_index[valid_mask]
            ))
            feature_names = ['blue', 'green', 'band_ratio', 'log_blue', 
                            'log_green', 'log_ratio', 'depth_index']
            y = depth[valid_mask]
        
            # 3. Optimized model parameters
            self.rf_model = RandomForestRegressor(
                n_estimators=500,  # Increase number of trees
                max_depth=25,      # Appropriately increase depth
                min_samples_split=2,
                min_samples_leaf=1,
                max_features='sqrt',
                bootstrap=True,
                random_state=42,
                n_jobs=-1,
                oob_score=True     # Enable out-of-bag scoring
            )
        
            # 4. Train model
            self.rf_model.fit(X, y)
            r2 = self.rf_model.score(X, y)
        
            # 5. Save feature importances and statistics
            importances = dict(zip(feature_names, self.rf_model.feature_importances_))
            self.rf_params = {
                'stats': {
                    'blue_mean': float(np.mean(blue_band[valid_mask])),
                    'blue_std': float(np.std(blue_band[valid_mask])),
                    'green_mean': float(np.mean(green_band[valid_mask])),
                    'green_std': float(np.std(green_band[valid_mask])),
                    'band_ratio_mean': float(np.mean(band_ratio[valid_mask])),
                    'band_ratio_std': float(np.std(band_ratio[valid_mask]))
                },
                'feature_importances': importances,
                'oob_score': float(self.rf_model.oob_score_)
            }
        
            # 6. Log training information
            logger.info(f"Training depth range: [{np.min(y):.2f}, {np.max(y):.2f}] m")
            logger.info(f"Blue band range: [{np.min(blue_band[valid_mask]):.2f}, {np.max(blue_band[valid_mask]):.2f}]")
            logger.info(f"Green band range: [{np.min(green_band[valid_mask]):.2f}, {np.max(green_band[valid_mask]):.2f}]")
            logger.info(f"Out-of-bag score: {self.rf_params['oob_score']:.4f}")
            for name, importance in importances.items():
                logger.info(f"Feature importance - {name}: {importance:.4f}")
            logger.info(f"Valid training samples: {len(y)}")
        
            return r2
        
        except Exception as e:
            logger.error(f"Random Forest training failed: {str(e)}")
            raise

    def predict_rf(self, blue_band: np.ndarray, green_band: np.ndarray) -> np.ndarray:
        """Random Forest model prediction"""
        try:
            # Get parameters and statistics
            stats = self.rf_params['stats']
        
            # 1. Feature engineering
            # 1.1 Basic features (standardized)
            blue_norm = (blue_band - stats['blue_mean']) / stats['blue_std']
            green_norm = (green_band - stats['green_mean']) / stats['green_std']
        
            # 1.2 Band ratio features
            band_ratio = np.log(blue_band / green_band)
            band_ratio_norm = (band_ratio - stats['band_ratio_mean']) / stats['band_ratio_std']
        
            # 1.3 Log-transformed features
            log_blue = np.log(blue_band)
            log_green = np.log(green_band)
            log_ratio = log_blue - log_green
        
            # 1.4 Depth-related features
            depth_index = (blue_norm - green_norm) / (blue_norm + green_norm)
        
            H, W = blue_band.shape
            predictions = np.zeros((H, W))
           
            # 2. Data validation
            valid_mask = np.logical_and.reduce((
                ~np.isnan(blue_band),
                ~np.isnan(green_band),
                blue_band > 0,
                green_band > 0
            ))
        
            if np.any(valid_mask):
                # 3. Build feature matrix
                X = np.column_stack((
                    blue_norm[valid_mask],
                    green_norm[valid_mask],
                    band_ratio_norm[valid_mask],
                    log_blue[valid_mask],
                    log_green[valid_mask],
                    log_ratio[valid_mask],
                    depth_index[valid_mask]
                ))
            
                # 4. Model prediction
                pred = self.rf_model.predict(X)
            
                # 5. Fill predictions back to original shape
                predictions[valid_mask] = pred
            
                # 6. Apply depth range limits
                predictions[predictions < 0.1] = 0  # Minimum depth limit
                predictions[predictions > 75.0] = 75.0  # Maximum depth limit
            
                # 7. Land mask
                land_mask = ~valid_mask
                predictions[land_mask] = 0
        
            return predictions
        
        except Exception as e:
            logger.error(f"Random Forest prediction failed: {str(e)}")
            raise

    def predict(self, sentinel_data: Dict[str, np.ndarray], method: str = 'llm') -> np.ndarray:
        """Predict water depth"""
        try:
            blue = sentinel_data['blue']   
            green = sentinel_data['green'] 
            T, H, W = blue.shape
            
            predictions = np.zeros((T, H, W))
            
            for t in range(T):
                if method.lower() == 'llm':
                    predictions[t] = self.predict_llm(blue[t], green[t])
                elif method.lower() == 'rf':
                    predictions[t] = self.predict_rf(blue[t], green[t])
                else:
                    raise ValueError(f"Unsupported method: {method}")
            
            return predictions[:, np.newaxis, :, :]
            
        except Exception as e:
            logger.error(f"Classic model prediction failed: {str(e)}")
            raise