import sys
import os
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import logging
import yaml
import argparse
from scipy.interpolate import interp1d, RegularGridInterpolator
import cv2
from tqdm import tqdm
from .s3gm_config import S3GMConfig, load_config
from .classic_models import ClassicModels
from .utils import normalize_data
from torch.amp import autocast, GradScaler
import torch.nn as nn
from .preprocessor import DataPreprocessor
from torch.utils.checkpoint import checkpoint
import math

# Add S3GM code path
s3gm_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'S3GM', 'Code')
if not os.path.exists(s3gm_path):
    raise ImportError(f"S3GM code path does not exist: {s3gm_path}")
sys.path.append(s3gm_path)

# Import S3GM modules
from models.unet_video import UNetVideoModel
from sampler.sde import VESDE, VPSDE
from models.ema import ExponentialMovingAverage
from sampler.utils import complete_video_pc_dps, LangevinCorrector

logger = logging.getLogger(__name__)

class S3GMWrapper:
    """S3GM model wrapper"""
    
    def __init__(
        self,
        config_path: Optional[str] = None,
        classic_models: Optional[ClassicModels] = None
    ):
        """Initialize S3GM wrapper"""
        try:
            # 1. Load configuration
            self.config = load_config(config_path) if config_path else S3GMConfig()
            logger.info(f"Loaded config sampling.inner_loop value: {self.config.sampling.get('inner_loop', 'not found in sampling dict')}")
            
            # 2. Ensure classic model instance exists
            self.classic_models = classic_models or ClassicModels()
            
            # 3. Set device
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
            # 4. Initialize model - Note: UNetVideoModel internally adds +1 to in_channels
            input_channels = self.config.num_components
            logger.info(f"Model initialization: input_channels={input_channels} (UNet internally adds +1)")
            
            self.model = UNetVideoModel(
                in_channels=input_channels,
                model_channels=self.config.model_channels,
                out_channels=self.config.num_components,
                num_res_blocks=self.config.num_res_blocks,
                attention_resolutions=self.config.attention_resolutions,
                dropout=self.config.dropout,
                channel_mult=self.config.channel_mult,
                dims=self.config.dims,
                num_heads=self.config.num_heads,
                use_rpe_net=self.config.use_rpe_net,
                use_mixed_activation=self.config.range_adaptation['use_mixed_activation'],
                land_value=self.config.range_adaptation['land_value'],
                init_scale=self.config.range_adaptation['init_scale'],
                init_shift=self.config.range_adaptation['init_shift']
            ).to(self.device)
            
            # Log initialized model parameters
            logger.info(f"Model parameter count: {sum(p.numel() for p in self.model.parameters())}")
            logger.info(f"Model input channels: {self.model.in_channels}")
            logger.info(f"Model output channels: {self.model.out_channels}")
            
            # Log configuration
            logger.info(f"Range adaptation config:")
            for key, value in self.config.range_adaptation.items():
                logger.info(f"  - {key}: {value}")
            
            # 5. Initialize SDE
            sde_type = self.config.sde_type.lower()
            if sde_type == 'vpsde':
                self.sde = VPSDE(config=self.config)
            elif sde_type == 'vesde':
                self.sde = VESDE(config=self.config)
            else:
                raise NotImplementedError(f"SDE type {sde_type} not supported.")

            logger.info(f"VPSDE/VESDE initialized with: beta_min={getattr(self.sde, 'beta_0', 'N/A')}, beta_max={getattr(self.sde, 'beta_1', 'N/A')}, sigma_min={getattr(self.sde, 'sigma_min', 'N/A')}, sigma_max={getattr(self.sde, 'sigma_max', 'N/A')}, num_scales={self.sde.N}")
            
            # 5. Add score_fn (needed before corrector)
            self.score_fn = self._get_score_fn()
            
            # 6. Don't create corrector instance at initialization
            self.corrector = None
        
            # 7. Add EMA
            if hasattr(self.config, 'use_ema') and self.config.use_ema:
                self.ema = ExponentialMovingAverage(
                    self.model.parameters(),
                    decay=self.config.ema_rate
                )
            
            logger.info("S3GM wrapper initialization completed")
            
        except Exception as e:
            logger.error(f"S3GM wrapper initialization failed: {str(e)}")
            raise

    def set_classic_models(self, classic_models: ClassicModels) -> None:
        """Set classic models"""
        self.classic_models = classic_models
        logger.info("S3GM wrapper classic models updated")

    def _prepare_input_data(self, normalized_classic, gebco_data, measurements):
        """Prepare input data"""
        # Data validation function
        def check_data(data, name):
            # Ensure data is torch.Tensor type
            if isinstance(data, np.ndarray):
                data = torch.from_numpy(data)
            elif not isinstance(data, torch.Tensor):
                data = torch.tensor(data)
            
            if torch.isnan(data).any():
                logger.error(f"{name} contains NaN")
                return False
            if torch.isinf(data).any():
                logger.error(f"{name} contains Inf")
                return False
            logger.info(f"{name} range: [{data.min():.3f}, {data.max():.3f}]")
            return True
        
        try:
            # Check each input
            check_data(normalized_classic, "normalized_classic")
            check_data(gebco_data, "gebco_data")
            if measurements is not None:
                check_data(measurements['depths'], "measurements depths")
            
            #print(f"GEBCO data shape before processing: {gebco_data.shape}")
            
            B = 1  # Batch size
            T = len(normalized_classic)  # Number of time steps
            H, W = gebco_data.shape[-2:]  # Spatial dimensions
        
            # Check normalized_classic dimensions
            # print(f"Classic results shape: {normalized_classic.shape}")
            if len(normalized_classic.shape) != 4:
                raise ValueError(f"Classic results dimension error: {normalized_classic.shape}, should be [T, 1, H, W]")
            
            input_tensor = torch.zeros((B, T, self.config.num_components, H, W))
            # print(f"Input tensor initial shape: {input_tensor.shape}")  # Should be (1, 6, 5, 64, 64)
                
            # 1. Classic model results (keep positive values)
            input_tensor[0, :, 0:1] = torch.from_numpy(normalized_classic)
        
            # 2. GEBCO data (convert to positive values)
            gebco_tensor = torch.from_numpy(gebco_data)
            if len(gebco_tensor.shape) == 3:
                gebco_tensor = gebco_tensor.unsqueeze(1)
            elif len(gebco_tensor.shape) == 2:
                gebco_tensor = gebco_tensor.unsqueeze(0).unsqueeze(0).repeat(T, 1, 1, 1)
            input_tensor[0, :, 1:2] = gebco_tensor
            # print(f"Input tensor shape after GEBCO: {input_tensor.shape}")
        
            # 3. Measurement point data (nautical chart data, keep positive values)
            if measurements is not None:
                depth_grid = torch.zeros((H, W))
                mask_grid = torch.zeros((H, W))
            
                coords = measurements['coordinates']  # Assuming range in [0,1]
                depths = measurements['depths']
            
                # Add coordinate validation
                print(f"Coordinates range: [{coords.min()}, {coords.max()}]")
                print(f"Number of measurement points: {len(depths)}")
            
                for depth, (y, x) in zip(depths, coords):
                    # Map [0,1] range to [0,H-1] and [0,W-1]
                    i = min(int(y * (H-1)), H-1)
                    j = min(int(x * (W-1)), W-1)
                    depth_grid[i, j] = depth
                    mask_grid[i, j] = 1.0
            
                # Check number of filled points
                print(f"Number of filled points in depth grid: {(depth_grid != 0).sum()}")
                print(f"Number of filled points in mask grid: {(mask_grid != 0).sum()}")
            
                # Expand dimensions and repeat
                input_tensor[0, :, 2:3] = depth_grid.unsqueeze(0).unsqueeze(0).repeat(T, 1, 1, 1)
                input_tensor[0, :, 3:4] = mask_grid.unsqueeze(0).unsqueeze(0).repeat(T, 1, 1, 1)
            
                print(f"Input tensor shape after measurements: {input_tensor.shape}")
        
            # 4. Unified mask
            unified_mask = torch.ones((T, 1, H, W))  # Create unified mask
            #print(f"Unified mask shape: {unified_mask.shape}")
            #print(f"Unified mask value range: [{unified_mask.min()}, {unified_mask.max()}]")

            input_tensor[0, :, 4:5] = unified_mask
            #print(f"Final input tensor shape: {input_tensor.shape}")
            #print(f"Channel 5 (unified mask) value range: [{input_tensor[0, :, 4:5].min()}, {input_tensor[0, :, 4:5].max()}]")

            # Validate value ranges for all channels
            for i in range(5):
                channel_data = input_tensor[0, :, i:i+1]
                print(f"Channel {i+1} value range: [{channel_data.min():.4f}, {channel_data.max():.4f}]")
            
            return input_tensor.to(self.device)
        
        except Exception as e:
            logger.error(f"Input data preparation failed: {str(e)}")
            raise
            
    def _create_measurement_grid(self, depths, coordinates, shape):
        """Create measurement point grid"""
        try:
            H, W = shape
            grid = np.zeros((2, H, W))  # [2, H, W] for depths and positions
            
            # Normalize coordinates
            norm_coords = coordinates.copy()
            norm_coords[:, 0] = (norm_coords[:, 0] - norm_coords[:, 0].min()) / \
                               (norm_coords[:, 0].max() - norm_coords[:, 0].min()) * (H - 1)
            norm_coords[:, 1] = (norm_coords[:, 1] - norm_coords[:, 1].min()) / \
                               (norm_coords[:, 1].max() - norm_coords[:, 1].min()) * (W - 1)
            
            # Fill depth values
            for depth, (y, x) in zip(depths, norm_coords):
                i, j = int(y), int(x)
                grid[0, i, j] = depth
                grid[1, i, j] = 1  # Position marker
                
            return grid
            
        except Exception as e:
            logger.error(f"Measurement grid creation failed: {str(e)}")
            raise
            
    def _create_temporal_mask(self, measurements):
        """Create temporal mask"""
        try:
            mask = np.zeros((self.config.num_frames, 
                           self.config.image_size, 
                           self.config.image_size))
            
            # Create mask based on measurement point positions
            if 'coordinates' in measurements:
                coords = measurements['coordinates']
                for t in range(self.config.num_frames):
                    for y, x in coords:
                        i = int(y * self.config.image_size)
                        j = int(x * self.config.image_size)
                        mask[t, i, j] = 1
                        
            return mask
            
        except Exception as e:
            logger.error(f"Temporal mask creation failed: {str(e)}")
            raise

    def _transform_pretrain(self, x, mode='forward'):
        """Data scale transformation
        
        Args:
            x: Input data [already mean-std normalized, range in [-1,1], land is 1.5]
            mode: 'forward' or 'inverse'
        """
        try:
            if mode == 'forward':
                # Data is already normalized, just log the range
                logger.info(f"Pretrain data range: [{x.min().item():.4f}, {x.max().item():.4f}]")
                return x
                
            elif mode == 'inverse':
                # Keep land marker value unchanged
                land_mask = (x == 1.5)
                # Other values are already in [-1,1] range, no conversion needed
                return x
                
        except Exception as e:
            logger.error(f"Data transformation failed: {str(e)}")
            raise

    def _prepare_pretrain_data(self, classic_data, gebco_data):
        """Prepare pretraining data
        
        Args:
            classic_data: Classic model results (normalized to [0,1])
            gebco_data: GEBCO data (normalized to [0,1])
        """
        try:
            B = 1  # Batch size
            T = len(classic_data)  # Number of time steps
            H, W = gebco_data.shape[-2:]  # Spatial dimensions
            
            # Create input tensor
            input_tensor = torch.zeros((B, T, self.config.num_components, H, W))
            
            # Get weights
            classic_weight = self.config.input_weights['classic']
            gebco_weight = self.config.input_weights['gebco']
            
            # 1. Classic model results (apply weights)
            classic_tensor = torch.from_numpy(classic_data).unsqueeze(1) * classic_weight # [T, 1, H, W]
            input_tensor[0, :, 0:1] = classic_tensor
            
            # 2. GEBCO data (apply weights)
            gebco_tensor = torch.from_numpy(gebco_data)
            if len(gebco_tensor.shape) == 3:  # [T, H, W]
                gebco_tensor = gebco_tensor.unsqueeze(1)  # [T, 1, H, W]
            elif len(gebco_tensor.shape) == 2:  # [H, W]
                gebco_tensor = gebco_tensor.unsqueeze(0).unsqueeze(0).repeat(T, 1, 1, 1)  # [T, 1, H, W]
            gebco_tensor = gebco_tensor * gebco_weight
            input_tensor[0, :, 1:2] = gebco_tensor
            
            # 3. Set other channels to zero (nautical chart data not used in pretraining)
            input_tensor[0, :, 2:4] = 0.0
            
            # 4. Unified mask
            input_tensor[0, :, 4:5] = 1.0
            
            # 5. Apply data transformation
            transformed_tensor = self._transform_pretrain(input_tensor.to(self.device))
            
            logger.info(f"Pretrain data preparation complete:")
            logger.info(f"- Input tensor shape: {transformed_tensor.shape}")
            logger.info(f"- Classic model channel (weighted) range: [{transformed_tensor[0, :, 0:1].min().item():.4f}, {transformed_tensor[0, :, 0:1].max().item():.4f}]")
            logger.info(f"- GEBCO channel (weighted) range: [{transformed_tensor[0, :, 1:2].min().item():.4f}, {transformed_tensor[0, :, 1:2].max().item():.4f}]")
            
            return transformed_tensor
            
        except Exception as e:
            logger.error(f"Pretrain data preparation failed: {str(e)}")
            raise

    def pretrain(self, classic_data, gebco_data, save_path):
        """Execute pretraining
        
        Args:
            classic_data: Classic model results (normalized to [-5,5])
            gebco_data: GEBCO data (normalized to [-5,5])
            save_path: Model save path
        """
        try:
            # Check input data
            if np.isnan(classic_data).any() or np.isnan(gebco_data).any():
                raise ValueError("Input data contains NaN values")
            
            # Check model initialization state
            for name, param in self.model.named_parameters():
                if torch.all(param == 0):
                    # logger.warning(f"{name} weights are all zeros, reinitializing")
                    if 'weight' in name:
                        nn.init.xavier_normal_(param)
                    elif 'bias' in name:
                        nn.init.zeros_(param)
            
            # Prepare pretraining data (includes data transformation)
            pretrain_tensor = self._prepare_pretrain_data(classic_data, gebco_data)
            
            # Execute pretraining - use new pretraining function
            success = self._run_proper_pretrain(pretrain_tensor, num_epochs=1500)
            
            if not success:
                logger.error("Pretraining failed, cannot continue")
                raise RuntimeError("Pretraining failed")
            
            # Create save directory (if not exists)
            save_dir = os.path.dirname(save_path)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
                logger.info(f"Created model save directory: {save_dir}")
            
            # Validate model before saving
            self._check_model_weights()
            torch.save(self.model.state_dict(), save_path)
            logger.info(f"Pretrained model saved to: {save_path}")
            
            # Log current model configuration
            logger.info(f"Model configuration:")
            logger.info(f"  - Range adaptation: {self.config.range_adaptation['enabled']}")
            logger.info(f"  - Mixed activation: {self.config.range_adaptation['use_mixed_activation']}")
            logger.info(f"  - Land marker value: {self.config.range_adaptation['land_value']}")
            
            # Validate model
            logger.info(f"Validating model...")
            self.model.eval()
            with torch.no_grad():
                test_input = pretrain_tensor[:, :1].clone()
                test_t = torch.ones(1, device=self.device) * 0.5
                B, T = test_input.shape[:2]
                latent_mask = torch.ones([B, T, 1, 1, 1]).float().to(self.device)
                obs_mask = torch.zeros([B, T, 1, 1, 1]).float().to(self.device)
                
                test_score, _ = self.model(
                    x=test_input,
                    x0=test_input,
                    timesteps=test_t,
                    latent_mask=latent_mask,
                    obs_mask=obs_mask,
                    frame_indices=torch.arange(T, device=self.device).expand(B, -1)
                )
                logger.info(f"Model test output range: [{test_score.min().item():.4f}, {test_score.max().item():.4f}]")
            
        except Exception as e:
            logger.error(f"Pretraining failed: {str(e)}")
            raise

    def load_pretrained(self, model_path: str):
        """Load pretrained model"""
        try:
            logger.info(f"Loading pretrained model: {model_path}")
            state_dict = torch.load(model_path, map_location=self.device)
            
            # Handle model structure changes - allow missing parameters for new layers
            model_dict = self.model.state_dict()
            
            # Filter out mismatched keys
            pretrained_dict = {k: v for k, v in state_dict.items() if k in model_dict}
            
            # Check for missing keys
            missing_keys = [k for k in model_dict.keys() if k not in pretrained_dict]
            if missing_keys:
                logger.warning(f"Model structure updated, following parameters will be randomly initialized:")
                for k in missing_keys:
                    logger.warning(f" - {k}")
            
            # Update current model parameters
            model_dict.update(pretrained_dict)
            
            # Load updated parameters
            self.model.load_state_dict(model_dict, strict=False)
            
            # Log loaded model info
            logger.info(f"Successfully loaded pretrained model")
            self._check_model_weights()
            return True
        except Exception as e:
            logger.error(f"Failed to load pretrained model: {str(e)}")
            raise

    def conditional_sampling(self, measurements, measurement_coordinates, years, classic_data, gebco_data):
        """Execute conditional sampling
        
        Args:
            measurements: Nautical chart depth data (normalized)
            measurement_coordinates: Nautical chart coordinates
            years: List of years
            classic_data: Classic model results (normalized)
            gebco_data: GEBCO data (normalized)
        """
        try:
            # Prepare conditional sampling data
            input_tensor = self._prepare_condition_data(
                measurements, measurement_coordinates, years, classic_data, gebco_data
            )
            
            # Execute sampling
            results = self._run_sampling(input_tensor)
            
            # Check results (ensure it's a tensor)
            if isinstance(results, np.ndarray):
                results_tensor = torch.from_numpy(results)
            else:
                results_tensor = results
            
            logger.info(f"Conditional sampling result shape: {results_tensor.shape}")
            if torch.isnan(results_tensor).any():
                nan_ratio = torch.isnan(results_tensor).sum().item() / results_tensor.numel()
                logger.warning(f"Conditional sampling result contains NaN values! NaN ratio: {nan_ratio:.2%}")
                
                # Log NaN ratio for each channel
                for c in range(results_tensor.shape[2]):
                    channel_nan_ratio = torch.isnan(results_tensor[..., c, :, :]).sum().item() / \
                                      (results_tensor.shape[0] * results_tensor.shape[1] * 
                                       results_tensor.shape[3] * results_tensor.shape[4])
                    logger.warning(f"Channel {c} NaN ratio: {channel_nan_ratio:.2%}")
            
            valid_values = results_tensor[~torch.isnan(results_tensor)]
            if len(valid_values) > 0:
                logger.info(f"Conditional sampling result range (excluding NaN): [{valid_values.min().item():.4f}, {valid_values.max().item():.4f}]")
            else:
                logger.error("No valid values in conditional sampling result!")
            
            return results
            
        except Exception as e:
            logger.error(f"Conditional sampling failed: {str(e)}")
            raise

    def _prepare_condition_data(self, measurements, measurement_coordinates, years, classic_data, gebco_data):
        """Prepare conditional sampling data"""
        try:
            # Ensure input data is correct
            H, W = self.config.image_size, self.config.image_size
            T = len(years)
            
            # Get weights
            classic_weight = self.config.input_weights['classic']
            gebco_weight = self.config.input_weights['gebco']
            
            # Log accurate dimension info
            logger.info(f"Preparing condition data: years={years}, num_components={self.config.num_components}")
            
            # Ensure input_tensor dimensions are correct
            # Format: [batch, time, channels, height, width]
            input_tensor = torch.zeros((1, T, self.config.num_components, H, W), 
                                     dtype=torch.float32, device=self.device)
            
            # 0. Load classic model results (apply weights)
            classic_tensor = torch.from_numpy(classic_data).unsqueeze(1) * classic_weight # [T, 1, H, W]
            input_tensor[0, :, 0:1] = classic_tensor.to(self.device)
            logger.info(f"- Classic model data (weighted) range: [{input_tensor[0, :, 0:1].min().item():.4f}, {input_tensor[0, :, 0:1].max().item():.4f}]")

            # 1. Load GEBCO data (apply weights)
            gebco_tensor = torch.from_numpy(gebco_data)
            if len(gebco_tensor.shape) == 3:  # [T, H, W]
                gebco_tensor = gebco_tensor.unsqueeze(1)  # [T, 1, H, W]
            elif len(gebco_tensor.shape) == 2:  # [H, W]
                gebco_tensor = gebco_tensor.unsqueeze(0).unsqueeze(0).repeat(T, 1, 1, 1)  # [T, 1, H, W]
            gebco_tensor = gebco_tensor * gebco_weight
            input_tensor[0, :, 1:2] = gebco_tensor.to(self.device)
            logger.info(f"- GEBCO data (weighted) range: [{input_tensor[0, :, 1:2].min().item():.4f}, {input_tensor[0, :, 1:2].max().item():.4f}]")
            
            # Create depth grid and mask grid
            depth_grid = torch.zeros((H, W), device=self.device)
            mask_grid = torch.zeros((H, W), device=self.device)
            
            # Fill measurement points
            for depth, (y, x) in zip(measurements, measurement_coordinates):
                i = min(int(y * (H-1)), H-1)
                j = min(int(x * (W-1)), W-1)
                
                # Explicitly convert numpy float to python float
                try:
                    depth_float = float(depth) 
                    depth_grid[i, j] = depth_float 
                except TypeError as e:
                    logger.error(f"Cannot convert depth value {depth} (type: {type(depth)}) to float. Error: {e}")
                    # Handle error, e.g., skip this point or assign a default value
                    depth_grid[i, j] = 0.0 # Or some other default like stats['invalid_value']
                
                mask_grid[i, j] = 1.0 # Assigning 1.0 (python float) is okay
            
            # Record original mask
            original_mask = mask_grid.cpu().numpy()
            valid_observations = np.sum(original_mask > 0)
            logger.info(f"Original mask observation point count: {valid_observations}")
            
            # Total mask point counter
            total_mask_points = 0
            
            # For all time steps:
            for t, year in enumerate(years):
                # Nautical chart depth data (normalized)
                input_tensor[0, t, 2:3] = depth_grid.unsqueeze(0)
                
                # Set observation mask for all years with different strengths
                # 2023 uses full mask
                if year == 2023:
                    input_tensor[0, t, 3:4] = mask_grid.unsqueeze(0)
                    total_mask_points += valid_observations
                    logger.info(f"Year {year} uses full mask, {valid_observations} observation points (100%)")
                else:
                    # Other years use partial mask (improved strategy)
                    # Ensure even 2018 has at least 20% observation points
                    year_factor = 0.2 + 0.8 * (year - 2018) / 5.0  # Range from 0.2 to 1.0
                    
                    # Generate deterministic mask (no random dropout, take first N points)
                    n_points = int(valid_observations * year_factor)
                    n_points = max(1, n_points)  # Ensure at least 1 point
                    
                    # Create new mask
                    partial_mask = torch.zeros_like(mask_grid)
                    
                    # Get observation point coordinates from original mask
                    obs_indices = torch.nonzero(mask_grid, as_tuple=True)
                    if len(obs_indices[0]) > 0:
                        # Keep only first n_points observation points
                        selected_indices = list(zip(obs_indices[0][:n_points].tolist(), 
                                                    obs_indices[1][:n_points].tolist()))
                        
                        for i, j in selected_indices:
                            partial_mask[i, j] = 1.0
                    
                    input_tensor[0, t, 3:4] = partial_mask.unsqueeze(0)
                    points_count = partial_mask.sum().item()
                    total_mask_points += points_count
                    logger.info(f"Year {year} retained {points_count:.0f} observation points ({year_factor*100:.0f}%)")
            
            # Unified mask
            input_tensor[0, :, 4:5] = 1.0
            
            # Calculate spatial weights based on original mask
            def calculate_spatial_weights(mask, decay=None):
                """Calculate distance-based spatial weights using integer mask"""
                if decay is None:
                    decay = self.config.spatial_decay
                    
                # Ensure mask is numpy array
                if isinstance(mask, torch.Tensor):
                    mask = mask.cpu().numpy()
                    
                # Ensure mask is 2D array with values 0 or 1
                mask = mask.squeeze()
                # Binarize mask to 0/1
                mask = (mask > 0.5).astype(np.float32)
                
                H, W = mask.shape  # Should be 2D now
                y_coords = np.linspace(0, 1, H)
                x_coords = np.linspace(0, 1, W)
                grid_y, grid_x = np.meshgrid(y_coords, x_coords, indexing='ij')
                
                # Get observation point positions
                obs_indices = np.where(mask > 0)  # Ensure condition is explicit
                obs_y, obs_x = obs_indices[0], obs_indices[1]  # Extract row and column indices separately
                
                # Ensure there are observation points
                if len(obs_y) == 0:
                    logger.warning(f"No observation points found, using uniform weights")
                    return np.ones((H, W)) * 0.5
                    
                # Convert to normalized coordinates
                obs_y = obs_y / (H - 1)
                obs_x = obs_x / (W - 1)
                
                # Initialize weight matrix
                weights = np.zeros((H, W))
                
                # Calculate influence of each observation point
                for y, x in zip(obs_y, obs_x):
                    # Calculate distance to current observation point
                    dist = np.sqrt((grid_y - y)**2 + (grid_x - x)**2)
                    # Calculate weight using exponential decay
                    weight = np.exp(-dist / decay)
                    # Update weight matrix (take maximum)
                    weights = np.maximum(weights, weight)
                
                # Normalize weights to [0.3, 1] range, ensuring some weight even far from observation points
                if weights.max() > weights.min():
                    weights = 0.3 + 0.7 * (weights - weights.min()) / (weights.max() - weights.min())
                else:
                    weights.fill(0.5)
                    
                return weights

            # Calculate temporal weights
            def calculate_time_weights(num_frames, current_frame, decay=None):
                """Calculate time-based weights"""
                if decay is None:
                    decay = self.config.time_decay
                    
                weights = np.zeros(num_frames)
                for i in range(num_frames):
                    time_diff = abs(i - current_frame)
                    weights[i] = np.exp(-time_diff * decay)
                
                # Normalize to [0.2, 1] range, ensuring historical data has influence
                weights = 0.2 + 0.8 * (weights - weights.min()) / (weights.max() - weights.min())
                return weights

            # New mask weight calculation logic
            # Calculate spatial weights once using original mask
            spatial_weights = calculate_spatial_weights(original_mask)
            
            # Apply different temporal weights for each time step
            for t in range(T):
                logger.info(f"Processing time step {t}, year {years[t]}")
                time_weights = calculate_time_weights(T, t)
                
                # Apply weights to mask
                for frame in range(T):
                    mask_weights = spatial_weights * time_weights[frame]
                    # Normalize mask_weights to [0, 1]
                    min_w, max_w = mask_weights.min(), mask_weights.max()
                    if max_w > min_w:
                        normalized_mask_weights = (mask_weights - min_w) / (max_w - min_w)
                    else:
                        normalized_mask_weights = np.ones_like(mask_weights) * 0.5 # If all weights are same, set to 0.5
                    # Apply normalized weights
                    input_tensor[0, frame, 3] = torch.from_numpy(normalized_mask_weights).to(self.device)
            
            return input_tensor
            
        except Exception as e:
            logger.error(f"Failed to prepare conditional sampling data: {str(e)}")
            raise

    def _run_diffusion(self, input_tensor, transform_fn=None):
        """Run diffusion process"""
        try:
            # Extract condition info from input_tensor
            # Assuming input_tensor shape is [1, num_frames, channels, height, width]
            # with depth channel at index 2, mask channel at index 3
            
            # Extract and preprocess condition data
            depth_channel = input_tensor[:, :, 2:3]  # Depth channel
            mask_channel = input_tensor[:, :, 3:4]  # Mask channel
            
            # Store condition data for score_fn use
            self.condition_data = {
                'depth': depth_channel.clone(),  # Observation depth
                'mask': mask_channel.clone(),    # Observation mask
                'input_tensor': input_tensor.clone()  # Complete input
            }
            
            # Check condition data
            logger.info(f"Diffusion process input check:")
            logger.info(f"- Input tensor shape: {input_tensor.shape}")
            logger.info(f"- Depth channel range: [{depth_channel.min().item():.4f}, {depth_channel.max().item():.4f}]")
            logger.info(f"- Total 1s in mask channel: {mask_channel.sum().item()}")
            
            # Ensure SDE parameters are on correct device
            if hasattr(self.sde, 'discrete_betas'):
                self.sde.discrete_betas = self.sde.discrete_betas.to(self.device)
            if hasattr(self.sde, 'alphas'):
                self.sde.alphas = self.sde.alphas.to(self.device)
            if hasattr(self.sde, 'alphas_cumprod'):
                self.sde.alphas_cumprod = self.sde.alphas_cumprod.to(self.device)

            # Define network forward function
            def net_fn(x, t):
                B, T = x.shape[:2]
                
                # Prepare default masks
                latent_mask = torch.ones([B, T, 1, 1, 1]).float().to(self.device)
                obs_mask = torch.zeros([B, T, 1, 1, 1]).float().to(self.device)
                
                # Try to extract mask channel from input_tensor (without logging)
                try:
                    if isinstance(input_tensor, torch.Tensor):
                        if input_tensor.shape[2] > 3:  # Check if there are enough channels
                            # Extract observation mask channel (index 3) and check for non-zero values
                            mask_channel = input_tensor[0, :, 3:4].to(self.device)
                            
                            # Process each time step separately
                            for t_idx in range(T):
                                if t_idx < mask_channel.shape[0]:  # Ensure index is valid
                                    # Check if current time step has observation points
                                    if torch.any(mask_channel[t_idx] > 0):
                                        # Mark time step as having observations
                                        obs_mask[0, t_idx, 0, 0, 0] = 1.0
                except Exception as e:
                    # Handle errors silently
                    pass
                
                # Monitor input state
                if torch.isnan(x).any():
                    logger.error(f"net_fn input x contains NaN - shape: {x.shape}")
                    logger.error(f"Timestep t: {t.item() if isinstance(t, torch.Tensor) else t}")
                    raise ValueError("net_fn input contains NaN")

                with torch.no_grad():
                    score, _ = self.model(
                        x=x,
                        x0=x,
                        timesteps=t,
                        latent_mask=latent_mask,
                        obs_mask=obs_mask,
                        frame_indices=torch.arange(T, device=self.device).expand(B, -1)
                    )
                    
                    # Check if score is all NaN
                    if torch.isnan(score).any():
                        nan_locations = torch.isnan(score)
                        nan_count = nan_locations.sum().item()
                        logger.error(f"Score contains {nan_count} NaN values at timestep {t}")
                        
                        # Only calculate statistics if non-NaN values exist
                        valid_score = score[~torch.isnan(score)]
                        if valid_score.numel() > 0:
                            logger.error(f"Score stats: min={valid_score.min().item():.4f}, "
                                       f"max={valid_score.max().item():.4f}, "
                                       f"mean={valid_score.mean().item():.4f}")
                        else:
                            logger.error("Score is all NaN values!")
                        
                        # Log NaN positions
                        nan_indices = nan_locations.nonzero()
                        if len(nan_indices) > 0:
                            logger.error(f"First NaN position: {nan_indices[0].tolist()}")
                        
                        raise ValueError(f"Score contains NaN at timestep {t}")
                    
                    # Only log score range in debug mode
                    if self.config.stability.get('debug', False):
                        logger.debug(f"Score range at timestep {t}: [{score.min().item():.4f}, {score.max().item():.4f}]")
                
                return score

            # Ensure input tensor is on correct device and convert type
            if isinstance(input_tensor, np.ndarray):
                x = torch.from_numpy(input_tensor).to(self.device, non_blocking=True)
            else:
                x = input_tensor.to(self.device, non_blocking=True)
            
            if x.dtype != torch.float32:
                x = x.float()
            
            # Detailed input data check
            if torch.isnan(x).any() or torch.isinf(x).any():
                nan_count = torch.isnan(x).sum().item()
                inf_count = torch.isinf(x).sum().item()
                logger.error(f"Input tensor contains {nan_count} NaN and {inf_count} Inf values")
                logger.error(f"Input stats: min={x[~torch.isnan(x) & ~torch.isinf(x)].min().item():.4f}, "
                            f"max={x[~torch.isnan(x) & ~torch.isinf(x)].max().item():.4f}")
                raise ValueError("Input tensor contains NaN or Inf values")
            
            # Clear GPU cache
            torch.cuda.empty_cache()
            
            # Add input check
            logger.info(f"Diffusion process input check:")
            logger.info(f"- Input tensor shape: {x.shape}")
            if transform_fn:
                test_output = transform_fn(x.cpu().numpy())
                if isinstance(test_output, torch.Tensor):
                    logger.info(f"- Transform function output shape: {test_output.shape}")
                else:
                    logger.info(f"- Transform function output shape: {test_output.shape}")

            # Execute diffusion process
            try:
                # Use torch.amp.autocast context manager
                with torch.amp.autocast('cuda', enabled=False):
                    result_np, _ = complete_video_pc_dps(
                        self.config,
                        net_fn,
                        self.sde,
                        x.cpu().numpy(),
                        transform=transform_fn,
                        corrector=LangevinCorrector,
                        continuous=self.config.sampling['continuous'],
                        n_steps=self.config.sampling['inner_loop'],
                        probability_flow=self.config.probability_flow,
                        snr=self.config.sampling['snr'],
                        eps=self.config.eps,
                        device=self.device
                    )
            except RuntimeError as e:
                logger.error(f"Diffusion process runtime error: {str(e)}")
                logger.error(f"- Configuration info:")
                logger.error(f"  - num_frames: {self.config.num_frames}")
                logger.error(f"  - num_components: {self.config.num_components}")
                logger.error(f"  - sampling steps: {self.config.sampling['num_steps']}")
                raise
            
            # Result processing section
            def process_results(result_tensor):
                # First check if it's a tensor, convert to numpy if so
                if isinstance(result_tensor, torch.Tensor):
                    result_np = result_tensor.cpu().numpy()
                else:
                    result_np = result_tensor  # Already numpy array

                # Identify land regions (using land_value from config)
                land_mask = np.abs(result_np - self.config.land_value) < 0.1
                
                # Only log data range, no clipping
                valid_data = result_np[~land_mask & ~np.isnan(result_np)]
                if len(valid_data) > 0:
                    orig_min, orig_max = np.min(valid_data), np.max(valid_data)
                    logger.info(f"Original result range: [{orig_min:.4f}, {orig_max:.4f}]")
                
                # Keep land region values unchanged
                result_np[land_mask] = self.config.land_value
                
                return result_np

            # Process results after diffusion process ends
            result = result_np  # Save original result first
            
            # Apply transform function if available
            if transform_fn is not None:
                result = transform_fn(result)
            
            # Final process_results call, just for logging range
            result = process_results(result)

            return result # Return process_results result directly

        except Exception as e:
            logger.error(f"Diffusion process failed: {str(e)}")
            raise

    def _run_sampling(self, input_tensor):
        """Execute conditional sampling"""
        try:
            def adaptive_transform_sampling(x):
                """Adaptive data transform function for sampling"""
                if isinstance(x, np.ndarray):
                    x = torch.from_numpy(x).to(self.device)
                
                # Identify land
                land_mask = torch.abs(x - 1.5) < 0.1
                
                # Handle NaN and infinity values with wider range
                x = torch.where(land_mask, x, 
                               torch.nan_to_num(x, nan=0.0, posinf=20.0, neginf=-20.0))
                
                # Log data range
                if not hasattr(adaptive_transform_sampling, "logged"):
                    if torch.isfinite(x).any():
                        min_val = x[torch.isfinite(x)].min().item()
                        max_val = x[torch.isfinite(x)].max().item()
                        logger.info(f"Sampling data range: [{min_val:.4f}, {max_val:.4f}]")
                    adaptive_transform_sampling.logged = True
                
                return x

            # Adjust input_tensor
            try:
                nf = self.config.num_frames
                ns = self.config.sampling['num_steps']
                ol = self.config.sampling['overlap']
                b = max(1, int(ns // max(1, (nf - ol))) + 1)  # Prevent division by zero
                ns_real = b * (nf - ol) + ol
            
                # Create adjusted input_tensor
                adjusted_input_tensor = torch.zeros(
                    (1, ns_real, input_tensor.shape[2], input_tensor.shape[3], input_tensor.shape[4]), 
                    device=self.device
                )
            
                # Fill adjusted_input_tensor
                for i in range(b):
                    i_inv = b - i - 1
                    start_idx = i_inv * (nf - ol)
                    end_idx = min(start_idx + nf, ns_real)
                    if start_idx < ns_real:
                        src_end = min(nf, end_idx-start_idx)
                        if src_end > 0:
                            src_data = input_tensor[:, :src_end, :input_tensor.shape[2]]
                            adjusted_input_tensor[:, start_idx:start_idx+src_end, :input_tensor.shape[2]] = src_data
            
                # Copy auxiliary channels
                if input_tensor.shape[2] < adjusted_input_tensor.shape[2]:
                    aux_channels = input_tensor[:, 0:1, input_tensor.shape[2]:]
                    adjusted_input_tensor[:, :, input_tensor.shape[2]:] = aux_channels
            
                logger.info(f"Adjusted input_tensor shape: {input_tensor.shape} -> {adjusted_input_tensor.shape}")
            except Exception as e:
                logger.error(f"Error adjusting input_tensor: {e}")
                # Use original input_tensor on error
                adjusted_input_tensor = input_tensor
        
            # Execute diffusion process
            return self._run_diffusion(
                adjusted_input_tensor,
                transform_fn=adaptive_transform_sampling
            )
        
        except Exception as e:
            logger.error(f"Conditional sampling execution failed: {str(e)}")
            raise

    def _run_proper_pretrain(self, input_tensor, num_epochs=1500, batch_size=1, save_interval=50):
        """Execute pretraining using standard diffusion loss function"""
        try:
            # Ensure input is tensor and on correct device
            if isinstance(input_tensor, np.ndarray):
                x = torch.from_numpy(input_tensor).to(self.device)
            else:
                x = input_tensor.to(self.device)
            
            if x.dtype != torch.float32:
                x = x.float()
            
            # Use more robust optimizer configuration
            optimizer = torch.optim.AdamW(
                self.model.parameters(), 
                lr=1e-5,  # Lower learning rate to 1e-5
                weight_decay=1e-4
            )
            
            # Use cosine annealing learning rate schedule
            T_max = num_epochs # Total epochs as cosine period length
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, 
                T_max=T_max, 
                eta_min=1e-7 # Minimum learning rate
            )
            # Learning rate warmup parameters
            warmup_epochs = 10
            initial_lr = 1e-5
            
            # Initialize GradScaler for mixed precision
            scaler = GradScaler('cuda', enabled=self.config.use_amp)
            
            # Improved early stopping logic
            min_loss = float('inf')
            patience = 50  # Increased patience (from 8 to 50)
            patience_counter = 0
            window_size = 5  # Use window average loss
            loss_window = []
            
            # Pretraining loop
            for epoch in range(num_epochs):
                self.model.train()
                
                # Implement learning rate warmup
                if epoch < warmup_epochs:
                    lr_scale = (epoch + 1) / warmup_epochs
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = initial_lr * lr_scale
                    current_lr = initial_lr * lr_scale
                    if (epoch + 1) % 10 == 0: # Log LR at warmup end and every 10 epochs
                        logger.info(f"Epoch {epoch+1}/{num_epochs}: Warmup - LR set to {current_lr:.2e}")
                elif epoch == warmup_epochs: # After warmup, restore optimizer LR, let scheduler take over
                     for param_group in optimizer.param_groups:
                          param_group['lr'] = initial_lr
                     current_lr = initial_lr
                     logger.info(f"Epoch {epoch+1}/{num_epochs}: Warmup finished - LR set to {current_lr:.2e}")       
                else:
                     current_lr = optimizer.param_groups[0]['lr'] # Get current LR for logging
                     if (epoch + 1) % 100 == 0: # Log LR every 100 epochs
                          logger.info(f"Epoch {epoch+1}/{num_epochs}: Current LR = {current_lr:.2e}")

                optimizer.zero_grad() # Move to loop start
                
                # Use autocast context manager for forward pass
                with autocast('cuda', enabled=self.config.use_amp):
                    # Randomly generate timesteps
                    t = torch.rand(batch_size, device=self.device) * (1.0 - 1e-5) + 1e-5
                    
                    # Add random noise
                    mean, std = self.sde.marginal_prob(x, t)
                    z = torch.randn_like(x)
                    perturbed_x = mean + std[:, None, None, None, None] * z
                    perturbed_x.requires_grad_(True)
                    
                    # Create random mask for conditional training
                    random_mask = torch.zeros_like(x[:, :, 3:4])
                    random_mask.bernoulli_(0.2)  # 20% of points as condition
                    
                    # Extract these points as condition
                    condition_x = x.clone()
                    
                    # Prepare model input
                    B, T = x.shape[:2]
                    latent_mask = torch.ones([B, T, 1, 1, 1]).float().to(self.device)
                    obs_mask = random_mask
                    
                    # Define simple wrapper to handle non-Tensor and fixed parameters
                    def model_forward_wrapper(px, cx, ts, lm, om, fi):
                        # return_attn_weights=False is fixed
                        return self.model(x=px, x0=cx, timesteps=ts, latent_mask=lm, obs_mask=om, frame_indices=fi, return_attn_weights=False)

                    # Use checkpoint to call wrapper
                    # Note: checkpoint doesn't directly return attn_weights, so we use _ for second return value
                    score, _ = checkpoint(
                        model_forward_wrapper, 
                        perturbed_x, 
                        condition_x, 
                        t, 
                        latent_mask, 
                        obs_mask, 
                        torch.arange(T, device=self.device).expand(B, -1),
                        use_reentrant=True # Use default use_reentrant
                    )
                    
                    # Calculate loss
                    if self.config.parameterization == 'v':
                        # v-parameterization loss
                        # Use marginal_prob to get std (sigma_t)
                        _, std_t = self.sde.marginal_prob(x, t) # mean is not needed here
                        sigma_t = std_t[:, None, None, None, None]
                        # Calculate alpha_t = sqrt(1 - sigma_t^2)
                        alpha_t = torch.sqrt(1. - sigma_t**2 + self.config.eps) # Add eps to avoid sqrt(0)
                        
                        # Calculate v target
                        v_target = alpha_t * z - sigma_t * x # z is noise epsilon, x is clean data x0
                        loss = torch.mean((score - v_target) ** 2)
                    else:
                        # epsilon-parameterization loss (original logic)
                        _, std_t = self.sde.marginal_prob(x, t)
                        target = -z / std_t[:, None, None, None, None]
                        loss = torch.mean((score - target) ** 2)
                
                # Check for divergence
                is_divergent = loss.item() > 1000 or np.isnan(loss.item())
                
                if not is_divergent:
                    # 使用 scaler 进行反向传播和优化 (仅在损失正常时)
                    scaler.scale(loss).backward() # 1. Scale loss and backprop

                    # 2. Unscale gradients before clipping
                    scaler.unscale_(optimizer)

                    # 3. Clip gradients (optional but recommended)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                    # 4. Optimizer step. scaler.step() automatically checks for infs/NaNs
                    scaler.step(optimizer)

                    # 5. Update the scale for next iteration
                    scaler.update()
                else:
                    logger.warning(f"Epoch {epoch+1}:检测到损失发散 (loss={loss.item():.4e})，跳过梯度更新")
                
                # 记录和早停 (使用原始loss值，即使发散也记录NaN或大值)
                loss_value_for_tracking = loss.item()
                loss_window.append(loss_value_for_tracking)
                if len(loss_window) > window_size:
                    loss_window.pop(0)
                
                # 计算平均损失时排除 NaN/Inf
                finite_losses = [l for l in loss_window if np.isfinite(l)]
                if not finite_losses:
                    avg_loss = float('inf') # 如果窗口内全是无效值
                else:    
                    avg_loss = sum(finite_losses) / len(finite_losses)

                if (epoch + 1) % 10 == 0:
                    logger.info(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss_value_for_tracking:.4f}, Avg Loss: {avg_loss:.4f}")
                    
                    # 使用平均损失进行早停判断 (仅当 avg_loss 是有效数值时)
                    if np.isfinite(avg_loss):
                        if avg_loss < min_loss * 0.95:  # 只有明显改进才重置
                            min_loss = avg_loss
                            patience_counter = 0
                            logger.info(f"  (Avg loss improved to {min_loss:.4f})")
                        else:
                            patience_counter += 1
                            logger.info(f"  (Avg loss did not improve significantly for {patience_counter}/{patience} epochs)")
                            
                        if patience_counter >= patience:
                            logger.info(f"Early stopping at epoch {epoch+1}")
                            break
                
                # --- 修改：在每个 epoch 结束时调用 CosineAnnealingLR --- 
                # if np.isfinite(avg_loss):
                #     scheduler.step(avg_loss)  # ReduceLROnPlateau 使用平均损失
                # 只在预热期结束后才更新 CosineAnnealingLR 调度器
                if epoch >= warmup_epochs:
                    scheduler.step()
                # --------------------------------------------------
            
            return True
        
        except Exception as e:
            logger.error(f"预训练失败: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return False

    def _validate_model(self, x, condition_x):
        """验证模型性能"""
        self.model.eval()
        with torch.no_grad():
            test_t = torch.ones(1, device=self.device) * 0.5
            B, T = x.shape[:2]
            
            # 创建验证用的掩码
            latent_mask = torch.ones([B, T, 1, 1, 1]).float().to(self.device)
            obs_mask = torch.zeros([B, T, 1, 1, 1]).float().to(self.device)
            
            # 测试模型输出
            test_score, _ = self.model(
                x=x,
                x0=condition_x,
                timesteps=test_t,
                latent_mask=latent_mask,
                obs_mask=obs_mask,
                frame_indices=torch.arange(T, device=self.device).expand(B, -1)
            )
            
            # 记录验证结果
            logger.info(f"验证 - 模型输出范围: [{test_score.min().item():.4f}, {test_score.max().item():.4f}]")

    def _check_model_weights(self):
        """检查模型权重的有效性，并以更低的日志级别报告"""
        try:
            total_params = 0
            zero_params = 0
            inf_params = 0
            nan_params = 0
            
            for name, param in self.model.named_parameters():
                total_params += param.numel()
                zero_count = (param == 0).sum().item()
                zero_params += zero_count
                inf_params += torch.isinf(param).sum().item()
                nan_params += torch.isnan(param).sum().item()
                
                # 只在调试级别记录单个参数的零值情况
                if zero_count == param.numel() and logger.isEnabledFor(logging.DEBUG):
                    logger.debug(f"{name} 权重全为0")
                    # 使用正态分布重新初始化
                    if 'bias' in name:
                        nn.init.zeros_(param)
                    else:
                        nn.init.normal_(param, mean=0.0, std=0.02)
            
            # 只在出现严重问题时记录警告
            if inf_params > 0 or nan_params > 0:
                logger.error("模型权重中存在无穷值或NaN!")
                logger.error(f"- 无穷值参数比例: {inf_params/total_params:.2%}")
                logger.error(f"- NaN值参数比例: {nan_params/total_params:.2%}")
                return False
            
            # 将权重统计信息改为INFO级别的单行日志
            if zero_params/total_params > 0.1:  # 如果零值参数比例超过10%才记录
                logger.info(f"模型参数统计 - 总数: {total_params}, 零值比例: {zero_params/total_params:.2%}")
            
            return True
            
        except Exception as e:
            logger.error(f"检查模型权重失败: {str(e)}")
            raise

    def _manage_memory(self):
        """内存管理"""
        torch.cuda.empty_cache()
        torch.cuda.synchronize()  # 确保所有CUDA操作完成

    def _get_score_fn(self):
        """获取评分函数"""
        # 获取y数据（观测深度）和mask（观测掩码）
        # 假设在_run_diffusion中接收input_tensor，其中包含深度和掩码通道
        
        def score_fn(x, t):
            """评分函数，计算给定噪声水平t下的分数"""
            # x: [batch_size, num_frames, channels, height, width]
            # t: [batch_size]
            
            # 获取原始shape
            shape = x.shape
            
            # 确保t是一个向量
            t = t.view(-1)
            
            # 从input_tensor中提取观测数据和掩码
            # 假设在self.condition_data已经存储了预处理好的条件数据
            if hasattr(self, 'condition_data') and self.condition_data is not None:
                # 获取条件数据
                x0 = self.condition_data['depth'].to(x.device)  # 深度通道，形状应与x匹配
                obs_mask = self.condition_data['mask'].to(x.device)  # 掩码通道，指示观测点位置
                
                # 记录日志（首次调用时）
                if not hasattr(score_fn, 'logged_condition'):
                    logger.info(f"条件深度范围: [{x0.min().item():.4f}, {x0.max().item():.4f}]")
                    logger.info(f"观测点数量: {obs_mask.sum().item()}")
                    score_fn.logged_condition = True
            else:
                # 如果没有条件数据，使用零初始化
                x0 = torch.zeros_like(x)
                obs_mask = torch.zeros((shape[0], shape[1], 1, shape[3], shape[4]), device=x.device)
                logger.warning("没有找到条件数据，使用零初始化!")
            
            # 强化条件表示
            # 1. 增加条件掩码通道的权重，使模型更关注观测点
            enhanced_obs_mask = obs_mask * 2.0  # 增大掩码权重
            
            # 2. 创建条件差异通道 - 帮助模型理解观测点与当前预测的差异
            with torch.no_grad():
                _, x_noisy, _ = self.sde.marginal_prob(x0, t.reshape(-1, 1, 1, 1, 1))
                condition_diff = (x - x_noisy) * obs_mask  # 指示条件点的差异方向
            
            # 统一掩码，用于注意力机制
            latent_mask = torch.zeros_like(obs_mask)
            
            # 调用模型计算分数，传递增强的条件信息
            score, _ = self.model(
                x, 
                x0=x0, 
                timesteps=t, 
                obs_mask=enhanced_obs_mask,  # 增强的掩码
                latent_mask=latent_mask,
                frame_indices=None,  # 提供帧索引信息
                return_attn_weights=False
            )
            
            return score
        
        return score_fn