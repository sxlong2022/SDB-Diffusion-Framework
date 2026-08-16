import sys
import os
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import logging
import yaml
import argparse
from scipy.interpolate import interp1d, RegularGridInterpolator
from tqdm import tqdm
from .s3gm_config import S3GMConfig, load_config
from .classic_models import ClassicModels
from .utils import normalize_data
from torch.amp import autocast, GradScaler
import torch.nn as nn
from .preprocessor import DataPreprocessor
from torch.utils.checkpoint import checkpoint
import math

# Add S3GM code directory to sys.path
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
    """S3GM Generative Diffusion Model Wrapper."""
    
    def __init__(
        self,
        config_path: Optional[str] = None,
        classic_models: Optional[ClassicModels] = None
    ):
        """Initialize S3GM model wrapper."""
        try:
            # 1. Load configuration
            self.config = load_config(config_path) if config_path else S3GMConfig()
            # Confirm inner_loop value loaded from config
            logger.info(f"Loaded sampling.inner_loop value: {self.config.sampling.get('inner_loop', 'Not found')}")
            # *************************************
            
            # 2. Ensure surrogate model instance exists
            self.classic_models = classic_models or ClassicModels()
            
            # 3. Configure hardware device
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
            # 4. Initialize model (UNetVideoModel internally appends +1 channel)
            input_channels = self.config.num_components
            logger.info(f"Model initialized: input channels={input_channels} (+1 in UNet)")
            
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
            
            # Log initialized parameter count
            logger.info(f"Total model parameters: {sum(p.numel() for p in self.model.parameters())}")
            logger.info(f"Model input channels: {self.model.in_channels}")
            logger.info(f"Model output channels: {self.model.out_channels}")
            
            # Log configuration
            logger.info(f"Range adaptation configuration:")
            for key, value in self.config.range_adaptation.items():
                logger.info(f"  - {key}: {value}")
            
            # 5. Initialize SDE
            sde_type = self.config.sde_type.lower()
            if sde_type == 'vpsde':
                # Pass full config object to SDE
                self.sde = VPSDE(config=self.config)
            # Internal processing step
            elif sde_type == 'vesde':
                self.sde = VESDE(config=self.config)
            else:
                raise NotImplementedError(f"SDE type {sde_type} not supported.")
            # -------------------------

            # Verify SDE parameter initialization
            logger.info(f"SDE initialized: beta_min={getattr(self.sde, 'beta_0', 'N/A')}, beta_max={getattr(self.sde, 'beta_1', 'N/A')}, sigma_min={getattr(self.sde, 'sigma_min', 'N/A')}, sigma_max={getattr(self.sde, 'sigma_max', 'N/A')}, num_scales={self.sde.N}")
            
            # Configure score function before corrector
            self.score_fn = self._get_score_fn()
            
            # Lazy initialize corrector
            self.corrector = None
        
            # 7. Exponential Moving Average (EMA)
            if hasattr(self.config, 'use_ema') and self.config.use_ema:
                self.ema = ExponentialMovingAverage(
                    self.model.parameters(),
                    decay=self.config.ema_rate
                )
            
            logger.info("S3GM wrapper initialization complete")
            
        except Exception as e:
            logger.error(f"S3GM wrapper initialization failed: {str(e)}")
            raise

    def set_classic_models(self, classic_models: ClassicModels) -> None:
        """Set surrogate models."""
        self.classic_models = classic_models
        logger.info("Surrogate model updated in S3GM wrapper")

    def _prepare_input_data(self, normalized_classic, gebco_data, measurements):
        """Prepare input tensor."""
        # Numerical validity check
        def check_data(data, name):
            # Ensure tensor type
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
            # Validate input components
            check_data(normalized_classic, "normalized_classic")
            check_data(gebco_data, "gebco_data")
            if measurements is not None:
                check_data(measurements['depths'], "measurements depths")
            
            #print(f"GEBCO data shape before processing: {gebco_data.shape}")
            
            B = 1  # Batch size
            T = len(normalized_classic)  # Number of frames
            H, W = gebco_data.shape[-2:]  # Spatial dimensions
        
            # Check normalized classic surrogate dimension
            # print(f"Classic results shape: {normalized_classic.shape}")
            if len(normalized_classic.shape) != 4:
                raise ValueError(f"Surrogate dimension error: {normalized_classic.shape}, expected [T, 1, H, W]")
            
            input_tensor = torch.zeros((B, T, self.config.num_components, H, W))
            # Internal processing step
                
            # 1. Surrogate prediction channel
            input_tensor[0, :, 0:1] = torch.from_numpy(normalized_classic)
        
            # 2. GEBCO bathymetric prior channel
            gebco_tensor = torch.from_numpy(gebco_data)
            if len(gebco_tensor.shape) == 3:
                gebco_tensor = gebco_tensor.unsqueeze(1)
            elif len(gebco_tensor.shape) == 2:
                gebco_tensor = gebco_tensor.unsqueeze(0).unsqueeze(0).repeat(T, 1, 1, 1)
            input_tensor[0, :, 1:2] = gebco_tensor
            # print(f"Input tensor shape after GEBCO: {input_tensor.shape}")
        
            # 3. Nautical chart sounding sparse channel
            if measurements is not None:
                depth_grid = torch.zeros((H, W))
                mask_grid = torch.zeros((H, W))
            
                coords = measurements['coordinates']  # Normalized coordinates [0, 1]
                depths = measurements['depths']
            
                # Verify coordinate bounds
                print(f"Coordinates range: [{coords.min()}, {coords.max()}]")
                print(f"Number of measurement points: {len(depths)}")
            
                for depth, (y, x) in zip(depths, coords):
                    # Map [0, 1] to pixel grid coordinates
                    i = min(int(y * (H-1)), H-1)
                    j = min(int(x * (W-1)), W-1)
                    depth_grid[i, j] = depth
                    mask_grid[i, j] = 1.0
            
                # Verify populated points
                print(f"Number of filled points in depth grid: {(depth_grid != 0).sum()}")
                print(f"Number of filled points in mask grid: {(mask_grid != 0).sum()}")
            
                # Expand dimensions across time frames
                input_tensor[0, :, 2:3] = depth_grid.unsqueeze(0).unsqueeze(0).repeat(T, 1, 1, 1)
                input_tensor[0, :, 3:4] = mask_grid.unsqueeze(0).unsqueeze(0).repeat(T, 1, 1, 1)
            
                print(f"Input tensor shape after measurements: {input_tensor.shape}")
        
            # 4. Unified spatial domain mask
            unified_mask = torch.ones((T, 1, H, W))
            #print(f"Unified mask shape: {unified_mask.shape}")
            #print(f"Unified mask value range: [{unified_mask.min()}, {unified_mask.max()}]")

            input_tensor[0, :, 4:5] = unified_mask
            #print(f"Final input tensor shape: {input_tensor.shape}")
            #print(f"Channel 5 (unified mask) value range: [{input_tensor[0, :, 4:5].min()}, {input_tensor[0, :, 4:5].max()}]")

            # Validate range across all input channels
            for i in range(5):
                channel_data = input_tensor[0, :, i:i+1]
                print(f"Channel {i+1} value range: [{channel_data.min():.4f}, {channel_data.max():.4f}]")
            
            return input_tensor.to(self.device)
        
        except Exception as e:
            logger.error(f"Input data preparation failed: {str(e)}")
            raise
            
    def _create_measurement_grid(self, depths, coordinates, shape):
        """Create sparse sounding observation grid."""
        try:
            H, W = shape
            grid = np.zeros((2, H, W))  # [2, H, W] for depths and positions
            
            # Normalize coordinate grid
            norm_coords = coordinates.copy()
            norm_coords[:, 0] = (norm_coords[:, 0] - norm_coords[:, 0].min()) / \
                               (norm_coords[:, 0].max() - norm_coords[:, 0].min()) * (H - 1)
            norm_coords[:, 1] = (norm_coords[:, 1] - norm_coords[:, 1].min()) / \
                               (norm_coords[:, 1].max() - norm_coords[:, 1].min()) * (W - 1)
            
            # Populate sounding depth values
            for depth, (y, x) in zip(depths, norm_coords):
                i, j = int(y), int(x)
                grid[0, i, j] = depth
                grid[1, i, j] = 1  # Observation flag
                
            return grid
            
        except Exception as e:
            logger.error(f"Sounding grid creation failed: {str(e)}")
            raise
            
    def _create_temporal_mask(self, measurements):
        """Create temporal observation mask."""
        try:
            mask = np.zeros((self.config.num_frames, 
                           self.config.image_size, 
                           self.config.image_size))
            
            # Build mask based on coordinate points
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
            x: Input data normalized to [-1, 1], land mapped to 1.5
            mode: 'forward' or 'inverse'
        """
        try:
            if mode == 'forward':
                # Log data ranges
                logger.info(f"Pretraining data range: [{x.min().item():.4f}, {x.max().item():.4f}]")
                return x
                
            elif mode == 'inverse':
                # Preserve land mask values
                land_mask = (x == 1.5)
                # Marine values already normalized to [-1, 1]
                return x
                
        except Exception as e:
            logger.error(f"Data transformation failed: {str(e)}")
            raise

    def _prepare_pretrain_data(self, classic_data, gebco_data):
        """Prepare pretraining tensor
        
        Args:
            classic_data: Surrogate prediction array (normalized)
            gebco_data: GEBCO bathymetric grid (normalized)
        """
        try:
            B = 1  # Batch size
            T = len(classic_data)  # Number of frames
            H, W = gebco_data.shape[-2:]  # Spatial dimensions
            
            # Create input tensor
            input_tensor = torch.zeros((B, T, self.config.num_components, H, W))
            
            # Retrieve channel weights
            classic_weight = self.config.input_weights['classic']
            gebco_weight = self.config.input_weights['gebco']
            
            # 1. Surrogate channel with weighting
            classic_tensor = torch.from_numpy(classic_data).unsqueeze(1) * classic_weight # [T, 1, H, W]
            input_tensor[0, :, 0:1] = classic_tensor
            
            # 2. GEBCO bathymetric prior channel with weighting
            gebco_tensor = torch.from_numpy(gebco_data)
            if len(gebco_tensor.shape) == 3:  # [T, H, W]
                gebco_tensor = gebco_tensor.unsqueeze(1)  # [T, 1, H, W]
            elif len(gebco_tensor.shape) == 2:  # [H, W]
                gebco_tensor = gebco_tensor.unsqueeze(0).unsqueeze(0).repeat(T, 1, 1, 1)  # [T, 1, H, W]
            gebco_tensor = gebco_tensor * gebco_weight
            input_tensor[0, :, 1:2] = gebco_tensor
            
            # 3. Composite depth channel: RF surrogate prior + sparse chart sounding injection
            # Pretraining embeds sparse sounding measurements into Channel 2
            # U-Net learns spatial mapping from sparse soundings to full domain
            # U-Net learns spatial mapping from sparse soundings to full domain
            depth_weight = self.config.input_weights.get('depth', 1.0)
            classic_norm_full = classic_tensor.squeeze(1) / classic_weight  # Unweighted [-1, 1] scale
            fused_depth = classic_norm_full.clone()
            # Pretraining embeds sparse sounding measurements into Channel 2
            try:
                from bathymetry.preprocessor import DataPreprocessor
                preprocessor = DataPreprocessor(region=[122.35, 30.62, 122.6, 30.8])
                nautical_charts = preprocessor.get_sparse_points()
                chart_depths = nautical_charts['depths']
                chart_coords = nautical_charts['coordinates']
            # Pretraining embeds sparse sounding measurements into Channel 2
                depth_grid = classic_norm_full[0].clone()  # Base RF surrogate grid
                mask_grid = torch.zeros(H, W)
                for idx, (depth, (y, x)) in enumerate(zip(chart_depths, chart_coords)):
                    i = min(int(y * (H-1)), H-1)
                    j = min(int(x * (W-1)), W-1)
            # Pretraining embeds sparse sounding measurements into Channel 2
            # Execute reverse diffusion trajectory
            # Physical depth normalized to [-1, 1] for Channel 2
            # Internal processing step
                    depth_norm = float(depth) / 45.0 - 1.0  # (depth-0)/(90-0)*2-1
                    depth_grid[i, j] = depth_norm
                    mask_grid[i, j] = 1.0
            # Internal processing step
                for t in range(T):
                    input_tensor[0, t, 2:3] = depth_grid.unsqueeze(0) * depth_weight
            # Spatial observation mask
                    input_tensor[0, t, 3:4] = mask_grid.unsqueeze(0)
                logger.info(f"Pretraining depth channel populated with {int(mask_grid.sum())} sounding points")
            except Exception as e:
            # Fallback to pure RF surrogate prior
                logger.warning(f"Sounding embedding failed, falling back to pure RF prior: {e}")
                input_tensor[0, :, 2:3] = fused_depth.unsqueeze(1) * depth_weight
                input_tensor[0, :, 3:4] = 1.0
            
            # 4. Observation mask channel (1=sounding, 0=unobserved)

            # 5. Spatial domain mask (all ones during pretraining)
            input_tensor[0, :, 4:5] = 1.0
            
            # Apply tensor transformation
            transformed_tensor = self._transform_pretrain(input_tensor.to(self.device))
            
            logger.info(f"Pretraining data prepared:")
            logger.info(f"- Input tensor shape: {transformed_tensor.shape}")
            logger.info(f"- Surrogate channel (weighted) range: [{transformed_tensor[0, :, 0:1].min().item():.4f}, {transformed_tensor[0, :, 0:1].max().item():.4f}]")
            logger.info(f"- GEBCO channel (weighted) range: [{transformed_tensor[0, :, 1:2].min().item():.4f}, {transformed_tensor[0, :, 1:2].max().item():.4f}]")
            
            return transformed_tensor
            
        except Exception as e:
            logger.error(f"Pretraining data preparation failed: {str(e)}")
            raise

    def pretrain(self, classic_data, gebco_data, save_path):
        """Execute diffusion model pretraining
        
        Args:
            classic_data: Surrogate prediction array (normalized)
            gebco_data: GEBCO bathymetry array (normalized)
            save_path: Destination checkpoint path
        """
        try:
            # Check data validity
            if np.isnan(classic_data).any() or np.isnan(gebco_data).any():
                raise ValueError("Input data contains NaN values")
            
            # Check data validity
            for name, param in self.model.named_parameters():
                if torch.all(param == 0):
            # Check model parameter initialization
                    if 'weight' in name:
                        nn.init.xavier_normal_(param)
                    elif 'bias' in name:
                        nn.init.zeros_(param)
            
            # Internal processing step
            pretrain_tensor = self._prepare_pretrain_data(classic_data, gebco_data)
            
            # Internal processing step
            success = self._run_proper_pretrain(pretrain_tensor, num_epochs=1500)
            
            if not success:
                logger.error("Pretraining failed, cannot continue")
                raise RuntimeError("Pretraining failed")
            
            # Internal processing step
            save_dir = os.path.dirname(save_path)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
                logger.info(f"Created checkpoint directory: {save_dir}")
            
            # Internal processing step
            self._check_model_weights()
            torch.save(self.model.state_dict(), save_path)
            logger.info(f"Pretrained model saved to: {save_path}")
            
            # Internal processing step
            logger.info(f"Model configuration:")
            logger.info(f"  - Range adaptation: {self.config.range_adaptation['enabled']}")
            logger.info(f"  - Mixed activation: {self.config.range_adaptation['use_mixed_activation']}")
            logger.info(f"  - Land mask flag: {self.config.range_adaptation['land_value']}")
            
            # Internal processing step
            logger.info(f"Validating model output...")
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
                logger.info(f"Model forward test output range: [{test_score.min().item():.4f}, {test_score.max().item():.4f}]")
            
        except Exception as e:
            logger.error(f"Pretraining failed: {str(e)}")
            raise

    def load_pretrained(self, model_path: str):
        """Load pretrained model weights."""
        try:
            logger.info(f"Loading pretrained weights from: {model_path}")
            state_dict = torch.load(model_path, map_location=self.device)
            
            # Internal processing step
            model_dict = self.model.state_dict()
            
            # Internal processing step
            pretrained_dict = {k: v for k, v in state_dict.items() if k in model_dict}
            
            # Check data validity
            missing_keys = [k for k in model_dict.keys() if k not in pretrained_dict]
            if missing_keys:
                logger.warning(f"Model architecture changed, the following parameters will be initialized randomly:")
                for k in missing_keys:
                    logger.warning(f" - {k}")
            
            # Internal processing step
            model_dict.update(pretrained_dict)
            
            # Internal processing step
            self.model.load_state_dict(model_dict, strict=False)
            
            # Internal processing step
            logger.info(f"Successfully loaded pretrained model")
            self._check_model_weights()
            return True
        except Exception as e:
            logger.error(f"Failed to load pretrained model: {str(e)}")
            raise

    def conditional_sampling(self, measurements, measurement_coordinates, years, classic_data, gebco_data, fold_id: Optional[int] = None):
        """Execute conditional diffusion sampling
        
        Args:
            measurements: Normalized chart sounding measurements
            measurement_coordinates: Sounding coordinates (normalized [0, 1])
            years: List of annual time steps
            classic_data: Surrogate prediction array (normalized)
            gebco_data: GEBCO bathymetry grid (normalized)
            fold_id: Spatial CV fold ID (optional; if provided, masks holdout test points)
        """
        try:
            # Execute reverse diffusion trajectory
            input_tensor = self._prepare_condition_data(
                measurements, measurement_coordinates, years, classic_data, gebco_data, fold_id=fold_id
            )
            
            # Execute reverse diffusion trajectory
            results = self._run_sampling(input_tensor)
            
            # Check data validity
            if isinstance(results, np.ndarray):
                results_tensor = torch.from_numpy(results)
            else:
                results_tensor = results
            
            logger.info(f"Sampling result shape: {results_tensor.shape}")
            if torch.isnan(results_tensor).any():
                nan_ratio = torch.isnan(results_tensor).sum().item() / results_tensor.numel()
                logger.warning(f"Sampling output contains NaNs! Ratio: {nan_ratio:.2%}")
                
            # Internal processing step
                for c in range(results_tensor.shape[2]):
                    channel_nan_ratio = torch.isnan(results_tensor[..., c, :, :]).sum().item() / \
                                      (results_tensor.shape[0] * results_tensor.shape[1] * 
                                       results_tensor.shape[3] * results_tensor.shape[4])
                    logger.warning(f"Channel {c} NaN ratio: {channel_nan_ratio:.2%}")
            
            valid_values = results_tensor[~torch.isnan(results_tensor)]
            if len(valid_values) > 0:
                logger.info(f"Sampling result range (excluding NaNs): [{valid_values.min().item():.4f}, {valid_values.max().item():.4f}]")
            else:
                logger.error("No valid marine values found in sampling output!")
            
            return results
            
        except Exception as e:
            logger.error(f"Conditional sampling failed: {str(e)}")
            raise

    def _prepare_condition_data(self, measurements, measurement_coordinates, years, classic_data, gebco_data, fold_id: Optional[int] = None):
        """Prepare conditional sampling tensors."""
        try:
            # Internal processing step
            H, W = self.config.image_size, self.config.image_size
            T = len(years)
            
            # Retrieve channel weights
            classic_weight = self.config.input_weights['classic']
            gebco_weight = self.config.input_weights['gebco']
            
            # Internal processing step
            logger.info(f"Preparing conditional tensor: years={years}, channels={self.config.num_components}")
            
            # Internal processing step
            # Internal processing step
            input_tensor = torch.zeros((1, T, self.config.num_components, H, W), 
                                     dtype=torch.float32, device=self.device)
            
            # Spatial and temporal decay weights
            classic_tensor = torch.from_numpy(classic_data).unsqueeze(1) * classic_weight # [T, 1, H, W]
            input_tensor[0, :, 0:1] = classic_tensor.to(self.device)
            logger.info(f"- Surrogate channel (weighted) range: [{input_tensor[0, :, 0:1].min().item():.4f}, {input_tensor[0, :, 0:1].max().item():.4f}]")

            # Spatial and temporal decay weights
            gebco_tensor = torch.from_numpy(gebco_data)
            if len(gebco_tensor.shape) == 3:  # [T, H, W]
                gebco_tensor = gebco_tensor.unsqueeze(1)  # [T, 1, H, W]
            elif len(gebco_tensor.shape) == 2:  # [H, W]
                gebco_tensor = gebco_tensor.unsqueeze(0).unsqueeze(0).repeat(T, 1, 1, 1)  # [T, 1, H, W]
            gebco_tensor = gebco_tensor * gebco_weight
            input_tensor[0, :, 1:2] = gebco_tensor.to(self.device)
            logger.info(f"- GEBCO channel (weighted) range: [{input_tensor[0, :, 1:2].min().item():.4f}, {input_tensor[0, :, 1:2].max().item():.4f}]")
            
            # Spatial observation mask
            # Pretraining embeds sparse sounding measurements into Channel 2
            # Physical depth normalized to [-1, 1] for Channel 2
            # Spatial and temporal decay weights
            # Internal processing step
            depth_grid = torch.from_numpy(classic_data[-1]).to(self.device).float().clone()
            mask_grid = torch.zeros((H, W), device=self.device)
            
            # Internal processing step
            fold_ids = None
            if fold_id is not None:
                folds_path = 'intermediate_results/validation/spatial_folds.npz'
                if os.path.exists(folds_path):
                    folds_data = np.load(folds_path)
                    fold_ids = folds_data['fold_ids']
                    logger.info(f"S3GM DPS guidance mask: masking {np.sum(fold_ids == fold_id)} test points for Fold {fold_id}")
                else:
                    raise FileNotFoundError(f"Spatial fold definition file not found: {folds_path}")
            
            # Internal processing step
            for idx, (depth, (y, x)) in enumerate(zip(measurements, measurement_coordinates)):
                if fold_id is not None and fold_ids is not None and fold_ids[idx] == fold_id:
                    continue
                
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
            
            # Spatial observation mask
            original_mask = mask_grid.cpu().numpy()
            valid_observations = np.sum(original_mask > 0)
            logger.info(f"Valid sounding observations count: {valid_observations}")
            
            # Spatial observation mask
            total_mask_points = 0
            
            # Internal processing step
            for t, year in enumerate(years):
            # Pretraining embeds sparse sounding measurements into Channel 2
                input_tensor[0, t, 2:3] = depth_grid.unsqueeze(0)
                
            # Spatial observation mask
            # Spatial observation mask
                if year == 2023:
                    input_tensor[0, t, 3:4] = mask_grid.unsqueeze(0)
                    total_mask_points += valid_observations
                    logger.info(f"Year {year}: full observation mask with {valid_observations} soundings (100%)")
                else:
            # Spatial observation mask
            # Spatial observation mask
                    year_factor = 0.2 + 0.8 * (year - 2018) / 5.0
                    
            # Spatial observation mask
                    n_points = int(valid_observations * year_factor)
                    n_points = max(1, n_points)
                    
            # Spatial observation mask
                    partial_mask = torch.zeros_like(mask_grid)
                    
            # Spatial observation mask
                    obs_indices = torch.nonzero(mask_grid, as_tuple=True)
                    if len(obs_indices[0]) > 0:
            # Spatial observation mask
                        selected_indices = list(zip(obs_indices[0][:n_points].tolist(), 
                                                    obs_indices[1][:n_points].tolist()))
                        
                        for i, j in selected_indices:
                            partial_mask[i, j] = 1.0
                    
                    input_tensor[0, t, 3:4] = partial_mask.unsqueeze(0)
                    points_count = partial_mask.sum().item()
                    total_mask_points += points_count
                    logger.info(f"Year {year}: retained {points_count:.0f} soundings ({year_factor*100:.0f}%)")
            
            # Spatial observation mask
            input_tensor[0, :, 4:5] = 1.0
            
            # Spatial observation mask
            def calculate_spatial_weights(mask, decay=None):
                """Compute distance-based spatial decay weights."""
                if decay is None:
                    decay = self.config.spatial_decay
                    
            # Internal processing step
                if isinstance(mask, torch.Tensor):
                    mask = mask.cpu().numpy()
                    
            # Internal processing step
                mask = mask.squeeze()
            # Internal processing step
                mask = (mask > 0.5).astype(np.float32)
                
                H, W = mask.shape
                y_coords = np.linspace(0, 1, H)
                x_coords = np.linspace(0, 1, W)
                grid_y, grid_x = np.meshgrid(y_coords, x_coords, indexing='ij')
                
            # Spatial observation mask
                obs_indices = np.where(mask > 0)
                obs_y, obs_x = obs_indices[0], obs_indices[1]
                
            # Spatial observation mask
                if len(obs_y) == 0:
                    logger.warning(f"No soundings found, applying uniform spatial weights")
                    return np.ones((H, W)) * 0.5
                    
            # Physical depth normalized to [-1, 1] for Channel 2
                obs_y = obs_y / (H - 1)
                obs_x = obs_x / (W - 1)
                
            # Check model parameter initialization
                weights = np.zeros((H, W))
                
            # Spatial observation mask
                for y, x in zip(obs_y, obs_x):
            # Spatial observation mask
                    dist = np.sqrt((grid_y - y)**2 + (grid_x - x)**2)
            # Spatial and temporal decay weights
                    weight = np.exp(-dist / decay)
            # Spatial and temporal decay weights
                    weights = np.maximum(weights, weight)
                
            # Physical depth normalized to [-1, 1] for Channel 2
                if weights.max() > weights.min():
                    weights = 0.3 + 0.7 * (weights - weights.min()) / (weights.max() - weights.min())
                else:
                    weights.fill(0.5)
                    
                return weights

            # Spatial and temporal decay weights
            def calculate_time_weights(num_frames, current_frame, decay=None):
                """Compute temporal decay weights."""
                if decay is None:
                    decay = self.config.time_decay
                    
                weights = np.zeros(num_frames)
                for i in range(num_frames):
                    time_diff = abs(i - current_frame)
                    weights[i] = np.exp(-time_diff * decay)
                
            # Physical depth normalized to [-1, 1] for Channel 2
                weights = 0.2 + 0.8 * (weights - weights.min()) / (weights.max() - weights.min())
                return weights

            # Spatial observation mask
            # Spatial observation mask
            spatial_weights = calculate_spatial_weights(original_mask)
            
            # Spatial and temporal decay weights
            for t in range(T):
                logger.info(f"Processing frame {t}, Year {years[t]}")
                time_weights = calculate_time_weights(T, t)
                
            # Spatial observation mask
                for frame in range(T):
                    mask_weights = spatial_weights * time_weights[frame]
            # Physical depth normalized to [-1, 1] for Channel 2
                    min_w, max_w = mask_weights.min(), mask_weights.max()
                    if max_w > min_w:
                        normalized_mask_weights = (mask_weights - min_w) / (max_w - min_w)
                    else:
                        normalized_mask_weights = np.ones_like(mask_weights) * 0.5
                    # ******************************************
            # Physical depth normalized to [-1, 1] for Channel 2
                    input_tensor[0, frame, 3] = torch.from_numpy(normalized_mask_weights).to(self.device)
            
            return input_tensor
            
        except Exception as e:
            logger.error(f"Failed to prepare condition sampling data: {str(e)}")
            raise

    def _run_diffusion(self, input_tensor, transform_fn=None):
        """Execute diffusion reverse trajectory."""
        try:
            # Internal processing step
            # Internal processing step
            # Spatial observation mask
            
            # Internal processing step
            depth_channel = input_tensor[:, :, 2:3]
            mask_channel = input_tensor[:, :, 3:4]
            
            # Internal processing step
            self.condition_data = {
                'depth': depth_channel.clone(),
                'mask': mask_channel.clone(),
                'input_tensor': input_tensor.clone()
            }
            
            # Check data validity
            logger.info(f"Diffusion process input inspection:")
            logger.info(f"- Input tensor shape: {input_tensor.shape}")
            logger.info(f"- Depth channel range: [{depth_channel.min().item():.4f}, {depth_channel.max().item():.4f}]")
            logger.info(f"- Observation mask active count: {mask_channel.sum().item()}")
            
            # Internal processing step
            if hasattr(self.sde, 'discrete_betas'):
                self.sde.discrete_betas = self.sde.discrete_betas.to(self.device)
            if hasattr(self.sde, 'alphas'):
                self.sde.alphas = self.sde.alphas.to(self.device)
            if hasattr(self.sde, 'alphas_cumprod'):
                self.sde.alphas_cumprod = self.sde.alphas_cumprod.to(self.device)

            # Internal processing step
            def net_fn(x, t):
                B, T = x.shape[:2]
                
            # Spatial observation mask
                latent_mask = torch.ones([B, T, 1, 1, 1]).float().to(self.device)
                obs_mask = torch.zeros([B, T, 1, 1, 1]).float().to(self.device)
                
            # Spatial observation mask
                try:
                    if isinstance(input_tensor, torch.Tensor):
                        if input_tensor.shape[2] > 3:
            # Check data validity
                            mask_channel = input_tensor[0, :, 3:4].to(self.device)
                            
            # Internal processing step
                            for t_idx in range(T):
                                if t_idx < mask_channel.shape[0]:
            # Check data validity
                                    if torch.any(mask_channel[t_idx] > 0):
            # Internal processing step
                                        obs_mask[0, t_idx, 0, 0, 0] = 1.0
                except Exception as e:
            # Internal processing step
                    pass
                
            # Internal processing step
                if torch.isnan(x).any():
                    logger.error(f"net_fn input x contains NaN - shape: {x.shape}")
                    logger.error(f"Timestep t: {t.item() if isinstance(t, torch.Tensor) else t}")
                    raise ValueError("net_fn input contains NaN")

            # Internal processing step
            # Internal processing step
            # Hard conditioning: soundings injected directly at observation points
                with torch.no_grad():
            # Pretraining embeds sparse sounding measurements into Channel 2
            # Spatial observation mask
            # Spatial observation mask
            # U-Net learns spatial mapping from sparse soundings to full domain
            # Internal processing step
                    cond_x = input_tensor.to(self.device)
                    if cond_x.shape[1] != x.shape[1]:
            # Internal processing step
            # Internal processing step
                        cond_x = cond_x[:, :x.shape[1]].clone()
                        if cond_x.shape[1] < x.shape[1]:
                            pad = x.shape[1] - cond_x.shape[1]
                            cond_x = torch.cat([cond_x, cond_x[:, -1:].repeat(1, pad, 1, 1, 1)], dim=1)
            # Spatial observation mask
                    obs_mask_pixel = cond_x[:, :, 3:4] > 0.5  # [B, T, 1, H, W]
                    score, _ = self.model(
                        x=x,
                        x0=cond_x,
                        timesteps=t,
                        latent_mask=latent_mask,
                        obs_mask=obs_mask_pixel.float(),
                        frame_indices=torch.arange(T, device=self.device).expand(B, -1)
                    )
                
                return score

            # Internal processing step
            if isinstance(input_tensor, np.ndarray):
                x = torch.from_numpy(input_tensor).to(self.device, non_blocking=True)
            else:
                x = input_tensor.to(self.device, non_blocking=True)
            
            if x.dtype != torch.float32:
                x = x.float()
            
            # Check data validity
            if torch.isnan(x).any() or torch.isinf(x).any():
                nan_count = torch.isnan(x).sum().item()
                inf_count = torch.isinf(x).sum().item()
                logger.error(f"Input tensor contains {nan_count} NaNs and {inf_count} Infs")
                logger.error(f"Input stats: min={x[~torch.isnan(x) & ~torch.isinf(x)].min().item():.4f}, max={x[~torch.isnan(x) & ~torch.isinf(x)].max().item():.4f}")
                raise ValueError("Input tensor contains NaN or Inf values")
            
            # Internal processing step
            torch.cuda.empty_cache()
            
            # Check data validity
            logger.info(f"Diffusion process input inspection:")
            logger.info(f"- Input tensor shape: {x.shape}")
            if transform_fn:
                test_output = transform_fn(x.cpu().numpy())
                if isinstance(test_output, torch.Tensor):
                    logger.info(f"- Transform function output shape: {test_output.shape}")
                else:
                    logger.info(f"- Transform function output shape: {test_output.shape}")
            # Execute reverse diffusion trajectory
            try:
            # Internal processing step
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
                logger.error(f"Diffusion runtime error: {str(e)}")
                logger.error(f"- Configuration details:")
                logger.error(f"  - num_frames: {self.config.num_frames}")
                logger.error(f"  - num_components: {self.config.num_components}")
                logger.error(f"  - sampling steps: {self.config.sampling['num_steps']}")
                raise
            
            # Internal processing step
            def process_results(result_tensor):
            # Check data validity
                if isinstance(result_tensor, torch.Tensor):
                    result_np = result_tensor.cpu().numpy()
                else:
                    result_np = result_tensor

            # Internal processing step
                land_mask = np.abs(result_np - self.config.land_value) < 0.1
                
            # Internal processing step
                valid_data = result_np[~land_mask & ~np.isnan(result_np)]
                if len(valid_data) > 0:
                    orig_min, orig_max = np.min(valid_data), np.max(valid_data)
                    logger.info(f"Raw prediction range: [{orig_min:.4f}, {orig_max:.4f}]")
                
            # Internal processing step
                result_np[land_mask] = self.config.land_value
                
                return result_np

            # Execute reverse diffusion trajectory
            result = result_np
            
            # Internal processing step
            if transform_fn is not None:
                result = transform_fn(result)
            
            # Internal processing step
            result = process_results(result)

            return result

        except Exception as e:
            logger.error(f"Diffusion process failed: {str(e)}")
            raise

    def _run_sampling(self, input_tensor):
        """Execute conditional diffusion sampling"""
        try:
            def adaptive_transform_sampling(x):
                """Adaptive data transformation for sampling."""
                if isinstance(x, np.ndarray):
                    x = torch.from_numpy(x).to(self.device)
                
            # Internal processing step
                land_mask = torch.abs(x - 1.5) < 0.1
                
            # Internal processing step
                x = torch.where(land_mask, x, 
                               torch.nan_to_num(x, nan=0.0, posinf=20.0, neginf=-20.0))
                
            # Internal processing step
                if not hasattr(adaptive_transform_sampling, "logged"):
                    if torch.isfinite(x).any():
                        min_val = x[torch.isfinite(x)].min().item()
                        max_val = x[torch.isfinite(x)].max().item()
                        logger.info(f"Sampling data range: [{min_val:.4f}, {max_val:.4f}]")
                    adaptive_transform_sampling.logged = True
                
                return x

            # Internal processing step
            try:
                nf = self.config.num_frames
                ns = self.config.sampling['num_steps']
                ol = self.config.sampling['overlap']
                b = max(1, int(ns // max(1, (nf - ol))) + 1)
                ns_real = b * (nf - ol) + ol
            
            # Internal processing step
                adjusted_input_tensor = torch.zeros(
                    (1, ns_real, input_tensor.shape[2], input_tensor.shape[3], input_tensor.shape[4]), 
                    device=self.device
                )
            
            # Internal processing step
                for i in range(b):
                    i_inv = b - i - 1
                    start_idx = i_inv * (nf - ol)
                    end_idx = min(start_idx + nf, ns_real)
                    if start_idx < ns_real:
                        src_end = min(nf, end_idx-start_idx)
                        if src_end > 0:
                            src_data = input_tensor[:, :src_end, :input_tensor.shape[2]]
                            adjusted_input_tensor[:, start_idx:start_idx+src_end, :input_tensor.shape[2]] = src_data
            
            # Internal processing step
                if input_tensor.shape[2] < adjusted_input_tensor.shape[2]:
                    aux_channels = input_tensor[:, 0:1, input_tensor.shape[2]:]
                    adjusted_input_tensor[:, :, input_tensor.shape[2]:] = aux_channels
            
                logger.info(f"Adjusted input tensor shape: {input_tensor.shape} -> {adjusted_input_tensor.shape}")
            except Exception as e:
                logger.error(f"Error adjusting input tensor: {e}")
            # Internal processing step
                adjusted_input_tensor = input_tensor
        
            # Execute reverse diffusion trajectory
            return self._run_diffusion(
                adjusted_input_tensor,
                transform_fn=adaptive_transform_sampling
            )
        
        except Exception as e:
            logger.error(f"Conditional sampling failed: {str(e)}")
            raise

    def _run_proper_pretrain(self, input_tensor, num_epochs=1500, batch_size=1, save_interval=50):
        """Execute pretraining with diffusion objective."""
        try:
            # Internal processing step
            if isinstance(input_tensor, np.ndarray):
                x = torch.from_numpy(input_tensor).to(self.device)
            else:
                x = input_tensor.to(self.device)
            
            if x.dtype != torch.float32:
                x = x.float()
            
            # Internal processing step
            optimizer = torch.optim.AdamW(
                self.model.parameters(), 
                lr=1e-5,
                weight_decay=1e-4
            )
            
            # Step CosineAnnealingLR scheduler after warmup
            # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            #     optimizer,
            #     mode='min',
            # Internal processing step
            #     patience=50,
            #     min_lr=1e-7,
            # )
            T_max = num_epochs
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, 
                T_max=T_max, 
                eta_min=1e-7
            )
                # Learning rate warmup step
            warmup_epochs = 10
            initial_lr = 1e-5
            # ------------------------
            
            # Check model parameter initialization
            scaler = GradScaler('cuda', enabled=self.config.use_amp)
            
            # Track running loss window for early stopping
            min_loss = float('inf')
            patience = 50
            patience_counter = 0
            window_size = 5
            loss_window = []
            
            # Internal processing step
            for epoch in range(num_epochs):
                self.model.train()
                
                # Learning rate warmup step
                if epoch < warmup_epochs:
                    lr_scale = (epoch + 1) / warmup_epochs
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = initial_lr * lr_scale
                    current_lr = initial_lr * lr_scale
                    if (epoch + 1) % 10 == 0:
                        logger.info(f"Epoch {epoch+1}/{num_epochs}: Warmup - LR set to {current_lr:.2e}")
                elif epoch == warmup_epochs:
                     for param_group in optimizer.param_groups:
                          param_group['lr'] = initial_lr
                     current_lr = initial_lr
                     logger.info(f"Epoch {epoch+1}/{num_epochs}: Warmup finished - LR set to {current_lr:.2e}")       
                else:
                     current_lr = optimizer.param_groups[0]['lr']
                     if (epoch + 1) % 100 == 0:
                          logger.info(f"Epoch {epoch+1}/{num_epochs}: Current LR = {current_lr:.2e}")
                # ------------------------

                optimizer.zero_grad()
                
            # Internal processing step
                with autocast('cuda', enabled=self.config.use_amp):
            # Internal processing step
                    t = torch.rand(batch_size, device=self.device) * (1.0 - 1e-5) + 1e-5
                    
            # Internal processing step
                    mean, std = self.sde.marginal_prob(x, t)
                    z = torch.randn_like(x)
                    perturbed_x = mean + std[:, None, None, None, None] * z
                    perturbed_x.requires_grad_(True)
                    
            # Spatial observation mask
            # Pretraining embeds sparse sounding measurements into Channel 2
            # U-Net learns spatial mapping from sparse soundings to full domain
            # Spatial observation mask
                    obs_mask_base = (x[:, :, 3:4] > 0.5).float()
            # Spatial observation mask
                    random_mask = torch.zeros_like(x[:, :, 3:4])
            # Spatial observation mask
                    for t_idx in range(x.shape[1]):
                        obs_positions = (obs_mask_base[0, t_idx] > 0.5).nonzero()
                        if len(obs_positions) > 0:
                            n_keep = max(1, int(len(obs_positions) * 0.7))
                            perm = torch.randperm(len(obs_positions))[:n_keep]
                            for pi in perm:
                                yy, xx = obs_positions[pi][0].item(), obs_positions[pi][1].item()
                                random_mask[0, t_idx, 0, yy, xx] = 1.0
                    
            # Internal processing step
                    condition_x = x.clone()
                    
            # Internal processing step
                    B, T = x.shape[:2]
                    latent_mask = torch.ones([B, T, 1, 1, 1]).float().to(self.device)
                    obs_mask = random_mask
                    
            # Internal processing step
                    def model_forward_wrapper(px, cx, ts, lm, om, fi):
            # Internal processing step
                        return self.model(x=px, x0=cx, timesteps=ts, latent_mask=lm, obs_mask=om, frame_indices=fi, return_attn_weights=False)

            # Internal processing step
            # Internal processing step
                    score, _ = checkpoint(
                        model_forward_wrapper, 
                        perturbed_x, 
                        condition_x, 
                        t, 
                        latent_mask, 
                        obs_mask, 
                        torch.arange(T, device=self.device).expand(B, -1),
                        use_reentrant=True
                    )
                    
                    # Compute v-parameterization loss
                    if self.config.parameterization == 'v':
                        # v-parameterization loss
            # Internal processing step
                        _, std_t = self.sde.marginal_prob(x, t) # mean is not needed here
                        sigma_t = std_t[:, None, None, None, None]
            # Internal processing step
                        alpha_t = torch.sqrt(1. - sigma_t**2 + self.config.eps)
                        
            # Internal processing step
                        v_target = alpha_t * z - sigma_t * x
                        loss = torch.mean((score - v_target) ** 2)
                    else:
            # Internal processing step
                        _, std_t = self.sde.marginal_prob(x, t)
                        target = -z / std_t[:, None, None, None, None]
                        loss = torch.mean((score - target) ** 2)
                
            # Check data validity
                is_divergent = loss.item() > 1000 or np.isnan(loss.item())
                
                if not is_divergent:
                    # Compute v-parameterization loss
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
                    logger.warning(f"Epoch {epoch+1}: Loss divergence detected (loss={loss.item():.4e}), skipping gradient update")
                
            # Track running loss window for early stopping
                loss_value_for_tracking = loss.item()
                loss_window.append(loss_value_for_tracking)
                if len(loss_window) > window_size:
                    loss_window.pop(0)
                
            # Track running loss window for early stopping
                finite_losses = [l for l in loss_window if np.isfinite(l)]
                if not finite_losses:
                    avg_loss = float('inf')
                else:    
                    avg_loss = sum(finite_losses) / len(finite_losses)

                if (epoch + 1) % 10 == 0:
                    logger.info(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss_value_for_tracking:.4f}, Avg Loss: {avg_loss:.4f}")
                    
            # Track running loss window for early stopping
                    if np.isfinite(avg_loss):
                        if avg_loss < min_loss * 0.95:
                            min_loss = avg_loss
                            patience_counter = 0
                            logger.info(f"  (Avg loss improved to {min_loss:.4f})")
                        else:
                            patience_counter += 1
                            logger.info(f"  (Avg loss did not improve significantly for {patience_counter}/{patience} epochs)")
                            
                        if patience_counter >= patience:
                            logger.info(f"Early stopping at epoch {epoch+1}")
                            break
                
            # Step CosineAnnealingLR scheduler after warmup
                # if np.isfinite(avg_loss):
            # Track running loss window for early stopping
            # Step CosineAnnealingLR scheduler after warmup
                if epoch >= warmup_epochs:
                    scheduler.step()
                # --------------------------------------------------
            
            return True
        
        except Exception as e:
            logger.error(f"Pretraining failed: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return False

    def _validate_model(self, x, condition_x):
        """Validate model forward output."""
        self.model.eval()
        with torch.no_grad():
            test_t = torch.ones(1, device=self.device) * 0.5
            B, T = x.shape[:2]
            
            # Spatial observation mask
            latent_mask = torch.ones([B, T, 1, 1, 1]).float().to(self.device)
            obs_mask = torch.zeros([B, T, 1, 1, 1]).float().to(self.device)
            
            # Internal processing step
            test_score, _ = self.model(
                x=x,
                x0=condition_x,
                timesteps=test_t,
                latent_mask=latent_mask,
                obs_mask=obs_mask,
                frame_indices=torch.arange(T, device=self.device).expand(B, -1)
            )
            
            # Internal processing step
            logger.info(f"Validation forward output range: [{test_score.min().item():.4f}, {test_score.max().item():.4f}]")

    def _check_model_weights(self):
        """Inspect parameter weights for NaNs/Infs."""
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
                
            # Internal processing step
                if zero_count == param.numel() and logger.isEnabledFor(logging.DEBUG):
                    logger.debug(f"{name} weights are all zero")
            # Check model parameter initialization
                    if 'bias' in name:
                        nn.init.zeros_(param)
                    else:
                        nn.init.normal_(param, mean=0.0, std=0.02)
            
            # Internal processing step
            if inf_params > 0 or nan_params > 0:
                logger.error("Model weights contain Infs or NaNs!")
                logger.error(f"- Inf parameter ratio: {inf_params/total_params:.2%}")
                logger.error(f"- NaN parameter ratio: {nan_params/total_params:.2%}")
                return False
            
            # Spatial and temporal decay weights
            if zero_params/total_params > 0.1:
                logger.info(f"Model parameter stats - Total: {total_params}, Zero ratio: {zero_params/total_params:.2%}")
            
            return True
            
        except Exception as e:
            logger.error(f"Parameter check failed: {str(e)}")
            raise

    def _manage_memory(self):
        """GPU memory cleanup."""
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    def _get_score_fn(self):
        """Construct score evaluation function."""
            # Spatial observation mask
            # Spatial observation mask
        
        def score_fn(x, t):
            """Evaluate score at noise level t."""
            # x: [batch_size, num_frames, channels, height, width]
            # t: [batch_size]
            
            # Internal processing step
            shape = x.shape
            
            # Internal processing step
            t = t.view(-1)
            
            # Spatial observation mask
            # Internal processing step
            if hasattr(self, 'condition_data') and self.condition_data is not None:
            # Internal processing step
                x0 = self.condition_data['depth'].to(x.device)
                obs_mask = self.condition_data['mask'].to(x.device)
                
            # Internal processing step
                if not hasattr(score_fn, 'logged_condition'):
                    logger.info(f"Conditioning depth range: [{x0.min().item():.4f}, {x0.max().item():.4f}]")
                    logger.info(f"Active sounding count: {obs_mask.sum().item()}")
                    score_fn.logged_condition = True
            else:
            # Check model parameter initialization
                x0 = torch.zeros_like(x)
                obs_mask = torch.zeros((shape[0], shape[1], 1, shape[3], shape[4]), device=x.device)
                logger.warning("No conditioning data found, zero initializing!")
            
            # Internal processing step
            # Spatial observation mask
            enhanced_obs_mask = obs_mask * 2.0
            
            # Spatial observation mask
            with torch.no_grad():
                _, x_noisy, _ = self.sde.marginal_prob(x0, t.reshape(-1, 1, 1, 1, 1))
                condition_diff = (x - x_noisy) * obs_mask
            
            # Spatial observation mask
            latent_mask = torch.zeros_like(obs_mask)
            
            # Internal processing step
            score, _ = self.model(
                x, 
                x0=x0, 
                timesteps=t, 
                obs_mask=enhanced_obs_mask,
                latent_mask=latent_mask,
                frame_indices=None,
                return_attn_weights=False
            )
            
            return score
        
        return score_fn