import sys
import os
import logging
import argparse
import numpy as np
import ee
ee.Authenticate()
ee.Initialize(project='YOUR_GEE_PROJECT_ID')
from datetime import datetime
from data_acquisition_preprocessing import (
    get_sentinel2_images,
    load_gebco_data
)
from miwc import apply_miwc
from bathymetry.main import HybridBathymetrySystem
from bathymetry.classic_models import ClassicModels
from bathymetry.preprocessor import DataPreprocessor
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import zoom
import math
import torch
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from joblib import dump, load
from typing import List, Dict, Optional, Tuple, Union
from PIL import Image
import matplotlib.gridspec
import matplotlib.ticker as mticker
from scipy.stats import wilcoxon

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def setup_logging():
    """Configure logging system"""
    if len(logging.getLogger().handlers) > 0:
        return
        
    log_dir = 'logging'
    os.makedirs(log_dir, exist_ok=True)
    
    # Default to INFO level, change to DEBUG for detailed initialization info
    logging.basicConfig(
        level=logging.INFO,  # or logging.DEBUG for detailed info
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Bathymetry System Processing')
    parser.add_argument(
        '--stage',
        type=str,
        choices=['1', '1.5', '1.8','2', '3', '4', '5'],
        help='Processing stage: 1=Data preprocessing, 1.5=Classic model training, 1.8=Classic model validation, 2=S3GM model processing, 3=Post-processing and visualization, 4=Statistical significance analysis, 5=Detailed performance analysis by depth and terrain zones'
    )
    return parser.parse_args()

def stage1_preprocessing(aoi, gebco_years, sentinel_years):
    """Stage 1: Data preprocessing"""
    try:
        # Create time series data storage structure
        sentinel_time_series = {
            'blue': np.zeros((6, 64, 64)),  # [T, H, W]
            'green': np.zeros((6, 64, 64))
        }
        gebco_time_series = np.zeros((6, 64, 64))  # [T, H, W]
        
        # Process data for each year
        for idx, (gebco_year, sentinel_year) in enumerate(zip(gebco_years, sentinel_years)):
            logger.info(f"Processing GEBCO data released in {gebco_year} (corresponding to {sentinel_year} observations)...")
            
            # GEBCO data processing
            gebco_file = os.path.join(
                'GEBCO_Bathymetry',
                'GEBCO_26_Dec_2024_86912dfafafa',
                f'gebco_{gebco_year}_n30.8_s30.62_w122.35_e122.6.nc'
            )
            depth_data = load_gebco_data(gebco_file)
            if depth_data.shape != (64, 64):
                # Create target grid
                target_h, target_w = 64, 64
                orig_h, orig_w = depth_data.shape
    
                # Create source and target coordinate grids
                y_orig = np.linspace(0, 1, orig_h)
                x_orig = np.linspace(0, 1, orig_w)
                y_target = np.linspace(0, 1, target_h)
                x_target = np.linspace(0, 1, target_w)
    
                # Use bilinear interpolation for resampling
                interpolator = RegularGridInterpolator(
                    (y_orig, x_orig), 
                    depth_data,
                    method='linear',
                    bounds_error=False,
                    fill_value=np.nan
                )
    
                # Create target coordinate points
                xx, yy = np.meshgrid(x_target, y_target)
                points = np.stack([yy.ravel(), xx.ravel()], axis=1)
    
                # Perform interpolation
                depth_data = interpolator(points).reshape(target_h, target_w)
    
                logger.info(f"Resampled GEBCO data to target size: {depth_data.shape}")
            
            gebco_time_series[idx] = depth_data
            logger.info(f"GEBCO data range (after resampling): [{depth_data.min():.4f}, {depth_data.max():.4f}]")
            
            # Sentinel-2 data processing
            date_range = (f'{sentinel_year}-01-01', f'{sentinel_year}-12-31')
            image_collection = get_sentinel2_images(aoi, date_range)
            processed_image = apply_miwc(
                image_collection=image_collection,
                aoi=aoi,
                p_value=4,
                band_to_process=['nir'], 
                scale=10
            )
            
            def get_band_array(image, band_name, aoi):
                """Get array data for specified band"""
                try:
                    # Get available band names
                    available_bands = image.bandNames().getInfo()
                    logger.info(f"Available bands: {available_bands}")
                    
                    # First resample image to target resolution
                    target_scale = math.sqrt(aoi.area().getInfo() / (64 * 64))  # Calculate target resolution
                    resampled_image = image.resample('bilinear').reproject(
                        crs=image.projection(),
                        scale=target_scale
                    )
                    
                    # Select band (using list format)
                    band_data = resampled_image.select([band_name])
                    
                    # Get data
                    data = band_data.reduceRegion(
                        reducer=ee.Reducer.toList(),
                        geometry=aoi,
                        scale=target_scale,
                        maxPixels=1e13
                    ).get(band_name).getInfo()
                    
                    if not data:
                        logger.error(f"Failed to retrieve {band_name} band data")
                        return None
                        
                    # Convert to numpy array and reshape
                    temp_grid = np.array(data)
                    logger.info(f"Retrieved data size: {temp_grid.size}")
                    
                    # Calculate nearest square grid size
                    grid_size = int(np.sqrt(temp_grid.size))
                    temp_grid = temp_grid[:grid_size*grid_size].reshape(grid_size, grid_size)
                    
                    # Use scipy.ndimage for precise resampling to 64x64
                    resampled_grid = zoom(temp_grid, (64/grid_size, 64/grid_size), order=1)
                    logger.info(f"Resampled image size: {resampled_grid.shape}")
                    
                    return resampled_grid
                    
                except Exception as e:
                    logger.error(f"Error retrieving {band_name} band data: {str(e)}")
                    return None
            
            # Ensure processed_image is ee.Image type and contains necessary bands
            processed_image = ee.Image(processed_image)
            available_bands = processed_image.bandNames().getInfo()
            logger.info(f"Available bands after MIWC processing: {available_bands}")

            if not all(band in available_bands for band in ['blue', 'green']):
                logger.warning("Missing required bands after MIWC processing")
                continue

            # Get band data using list format
            blue_array = get_band_array(processed_image, 'blue', aoi)
            green_array = get_band_array(processed_image, 'green', aoi)
            
            # Data validation
            if blue_array is None or green_array is None:
                logger.error("Unable to retrieve valid band data")
                continue
                
            if blue_array.shape != green_array.shape:
                logger.error(f"Band shapes inconsistent: blue={blue_array.shape}, green={green_array.shape}")
                continue
                
            if np.all(np.isnan(blue_array)) or np.all(np.isnan(green_array)):
                logger.error("Band data is all NaN")
                continue
            
            sentinel_time_series['blue'][idx] = blue_array
            sentinel_time_series['green'][idx] = green_array
            logger.info(f"Sentinel-2 blue band range (after resampling): [{blue_array.min():.4f}, {blue_array.max():.4f}]")
            logger.info(f"Sentinel-2 green band range (after resampling): [{green_array.min():.4f}, {green_array.max():.4f}]")
            
            logger.info(f"Completed preprocessing and storage for {sentinel_year} data")
            
        # Save preprocessing results
        output_dir = 'intermediate_results'
        os.makedirs(output_dir, exist_ok=True)
        np.save(os.path.join(output_dir, 'sentinel_time_series.npy'), sentinel_time_series)
        np.save(os.path.join(output_dir, 'gebco_time_series.npy'), gebco_time_series)
        
        return sentinel_time_series, gebco_time_series
        
    except Exception as e:
        logger.error(f"Stage 1 processing failed: {str(e)}")
        raise

def stage1_5_model_training(sentinel_time_series):
    """Stage 1.5: Train classic models"""
    try:
        logger.info("Starting classic model training...")
        classic_models = ClassicModels()
        
        # Initialize preprocessor to get nautical chart data
        aoi = ee.Geometry.Rectangle([122.35, 30.62, 122.6, 30.8])
        preprocessor = DataPreprocessor(region=aoi)
        nautical_charts = preprocessor.get_sparse_points()
        
        # Use original positive depths and apply depth range limits
        depths = np.abs(nautical_charts['depths'])
        valid_depth_mask = (depths >= 0.1) & (depths <= 75.0)
        
        if not np.any(valid_depth_mask):
            raise ValueError("No data points within valid depth range")
            
        nautical_charts['depths'] = depths[valid_depth_mask]
        nautical_charts['coordinates'] = nautical_charts['coordinates'][valid_depth_mask]
        
        logger.info(f"Original depth range: {depths.min():.2f} to {depths.max():.2f} m")
        logger.info(f"Valid depth range: {nautical_charts['depths'].min():.2f} to {nautical_charts['depths'].max():.2f} m")
        logger.info(f"Number of valid depth points: {np.sum(valid_depth_mask)}")
        
        # Use latest year (2023) remote sensing data
        t = -1  # Last time point
        blue = sentinel_time_series['blue'][t]
        green = sentinel_time_series['green'][t]
        
        # Extract band values at nautical chart locations
        H, W = blue.shape
        coords = nautical_charts['coordinates']
        depths = nautical_charts['depths']
        
        blue_values = []
        green_values = []
        valid_depths = []
        
        for i, (y, x) in enumerate(coords):
            y_idx = int(y * (H - 1))
            x_idx = int(x * (W - 1))
            if 0 <= y_idx < H and 0 <= x_idx < W:
                blue_values.append(blue[y_idx, x_idx])
                green_values.append(green[y_idx, x_idx])
                valid_depths.append(depths[i])
        
        blue_values = np.array(blue_values)
        green_values = np.array(green_values)
        valid_depths = np.array(valid_depths)
        
        # Train Random Forest model
        r2_rf = classic_models.train_rf(blue_values, green_values, valid_depths)
        logger.info(f"Random Forest model training completed, R²: {r2_rf:.4f}")
        
        # Save model parameters and model itself
        os.makedirs('intermediate_results/model_params', exist_ok=True)
        np.save('intermediate_results/model_params/rf_params.npy', classic_models.rf_params)
        dump(classic_models.rf_model, 'intermediate_results/model_params/rf_model.joblib')
        
        return classic_models
        
    except Exception as e:
        logger.error(f"Classic model training failed: {str(e)}")
        raise

def stage1_8_model_validation(sentinel_time_series):
    """Stage 1.8: Classic model validation"""
    try:
        logger.info("Starting classic model validation...")
        classic_models = load_trained_classic_models()
        
        # 2. Create output directory
        output_dir = 'results/classic_models'
        os.makedirs(output_dir, exist_ok=True)

        # --- Load Land Mask (similar to stage 3) ---
        mask_file = 'results/s3gm_visualization/land_water_mask.tif' # Path to mask
        land_mask_array = None
        try:
            mask_image = Image.open(mask_file)
            H_target, W_target = 64, 64
            mask_image_resized = mask_image.resize((W_target, H_target), Image.NEAREST)
            land_mask_array = np.array(mask_image_resized)
            logger.info(f"Successfully loaded and resized land/water mask for RF plot from {mask_file}")
        except FileNotFoundError:
            logger.warning(f"Land/water mask file not found: {mask_file}. RF time series plot will not have overlay.")
        except Exception as e:
            logger.error(f"Error loading or processing land/water mask for RF plot: {e}. Skipping overlay.")
        # --------------------------------------------
        
        # Load nautical charts needed for coordinates
        try:
            import ee
            if not ee.data._credentials:
                 ee.Authenticate()
                 ee.Initialize(project='YOUR_GEE_PROJECT_ID')
        except ImportError:
             logger.error("Google Earth Engine Python API (ee) not found. Cannot get nautical charts.")
             raise
        except Exception as e:
             logger.error(f"GEE initialization failed: {e}")
             raise
        aoi = ee.Geometry.Rectangle([122.35, 30.62, 122.6, 30.8])
        preprocessor = DataPreprocessor(region=aoi)
        nautical_charts = preprocessor.get_sparse_points()

        # 4. Predict for all years
        depths = []
        years = range(2018, 2024)
        for t, year in enumerate(years):
            blue = sentinel_time_series['blue'][t]
            green = sentinel_time_series['green'][t]
            depth = classic_models.predict_rf(blue, green)
            depths.append(depth)
            logger.info(f"Completed prediction for year {year}")
        
        # 5. Create and save time series composite plot (pass land mask and chart coords)
        chart_coords_rf = nautical_charts['coordinates']  # Get coordinates
        create_time_series_plot(depths, years, 'rf', output_dir, land_mask_array=land_mask_array, chart_coords=chart_coords_rf)
        
        # 6. Compare 2023 predictions with nautical chart data
        # Get 2023 prediction results and actual nautical chart data
        t = -1  # 2023 index
        predicted_depth = depths[t]
        # H, W defined earlier during mask loading or prediction
        if 'H' not in locals(): H, W = predicted_depth.shape

        # Extract predicted values at nautical chart locations (nautical_charts is already loaded)
        coords = nautical_charts['coordinates']
        true_depths = np.abs(nautical_charts['depths'])

        predicted_values = []
        valid_true_depths = []

        for i, (y, x) in enumerate(coords):
            y_idx = int(y * (H - 1))
            x_idx = int(x * (W - 1))
            if 0 <= y_idx < H and 0 <= x_idx < W:
                pred_value = predicted_depth[y_idx, x_idx]
                if np.isfinite(pred_value) and pred_value > 0:
                    predicted_values.append(pred_value)
                    valid_true_depths.append(true_depths[i])

        predicted_values = np.array(predicted_values)
        valid_true_depths = np.array(valid_true_depths)

        # Calculate validation metrics
        rmse = np.nan
        mae = np.nan
        r2 = np.nan
        if len(valid_true_depths) > 0 and len(predicted_values) > 0:
            rmse = np.sqrt(np.mean((predicted_values - valid_true_depths) ** 2))
            mae = np.mean(np.abs(predicted_values - valid_true_depths))
            denom = np.sum((valid_true_depths - np.mean(valid_true_depths)) ** 2)
            if denom > 1e-9: # Avoid division by zero
                r2 = 1 - np.sum((valid_true_depths - predicted_values) ** 2) / denom
            else:
                r2 = 0.0 # Or handle as undefined
        logger.info("RF Validation Results (2023):")
        logger.info(f"RMSE: {rmse:.2f} m")
        logger.info(f"MAE: {mae:.2f} m")
        logger.info(f"R²: {r2:.4f}")
        logger.info(f"Number of points: {len(valid_true_depths)}")
        # Create and save Scatter Plot
        FONTSIZE_MAIN_LABELS = 12
        FONTSIZE_MAIN_TICKS = 10
        FONTSIZE_TEXT_BOX = 10 # Slightly smaller for text box
        fig_scatter = plt.figure(figsize=(7, 5.5)) # Match RF plot aspect ratio
        ax_scatter = fig_scatter.add_subplot(1, 1, 1)
        if len(valid_true_depths) > 0 and len(predicted_values) > 0 and np.isfinite(r2):
            scatter = ax_scatter.scatter(valid_true_depths, predicted_values, alpha=0.6, s=30, label='Samples')
            ideal_line, = ax_scatter.plot([0, 75], [0, 75], 'r--', linewidth=1.5, label='Ideal fit')  # Match RF plot
            ax_scatter.set_xlim(0, 75)
            ax_scatter.set_ylim(0, 75)
            # Use RF metrics, format matches RF plot (no N)
            text_str = f'RMSE: {rmse:.2f} m\nMAE: {mae:.2f} m\nR²: {r2:.4f}'
            ax_scatter.text(0.95, 0.05, text_str,
                      transform=ax_scatter.transAxes, fontsize=FONTSIZE_TEXT_BOX,
                      verticalalignment='bottom', horizontalalignment='right',
                      bbox=dict(boxstyle='round,pad=0.5', fc='white', alpha=0.8))
            ax_scatter.legend(loc='upper left') # Match RF plot
        else:
            ax_scatter.text(0.5, 0.5, 'No valid data for scatter plot', ha='center', va='center')
            ax_scatter.set_xlim(0, 75) # Still set limits even if no data
            ax_scatter.set_ylim(0, 75)

        ax_scatter.set_xlabel('Measured Depth (m)', fontsize=FONTSIZE_MAIN_LABELS)
        ax_scatter.set_ylabel('Predicted Depth (m)', fontsize=FONTSIZE_MAIN_LABELS)
        ax_scatter.tick_params(axis='both', which='major', labelsize=FONTSIZE_MAIN_TICKS)
        ax_scatter.grid(True, linestyle='--', alpha=0.6) # Match RF plot
        # No title, matching RF plot
        fig_scatter.tight_layout()
        save_path_scatter = os.path.join(output_dir, 'rf_validation_2023.jpg') # Correct filename
        plt.savefig(save_path_scatter, dpi=600, bbox_inches='tight')
        logger.info(f"RF Scatter plot saved to: {save_path_scatter}")
        plt.close(fig_scatter) # Close the specific figure

    except Exception as e:
        logger.error(f"Classic model validation failed: {str(e)}")
        raise

def create_time_series_plot(depths, years, model_name, output_dir, land_mask_array=None, chart_coords=None):
    """Create time series composite plot"""
    try:
        # Increase font sizes for better visibility when shrunk in Word column
        FONTSIZE_MAIN_LABELS = 14  # Axes labels, colorbar label
        FONTSIZE_MAIN_TICKS = 12   # Tick labels, base for subplot title
        
        plt.rcParams.update({'font.size': FONTSIZE_MAIN_TICKS})
        fig = plt.figure(figsize=(12, 8))

        num_years = len(years)
        num_cols = 3
        num_rows = (num_years + num_cols - 1) // num_cols
        
        all_valid_depths = []
        for depth in depths:
            valid_depths = depth[~np.isnan(depth) & (depth > 0)]
            if len(valid_depths) > 0:
                all_valid_depths.extend(valid_depths)
        vmin = 0
        vmax = 75

        extent = [122.35, 122.6, 30.62, 30.8]
        im = None

        def lon_formatter(x, pos): return f'{x:.1f}°E'
        def lat_formatter(y, pos): return f'{y:.1f}°N'
        
        for i, (depth, year) in enumerate(zip(depths, years)):
            ax = plt.subplot(num_rows, num_cols, i + 1)
            # Add interpolation for smoother look
            current_im = ax.imshow(depth, cmap='ocean', vmin=vmin, vmax=vmax, extent=extent, origin='upper', aspect='auto', interpolation='bicubic') 
            if i == 0: im = current_im

            # Conditional overlay for 2023 only
            if year == 2023 and chart_coords is not None and len(chart_coords) > 0:
                 # Convert normalized [0,1] coordinates (y, x) to plot coordinates (lon, lat)
                 min_lon, max_lon, min_lat, max_lat = extent
                 plot_lon = min_lon + chart_coords[:, 1] * (max_lon - min_lon)
                 plot_lat = min_lat + chart_coords[:, 0] * (max_lat - min_lat)
 
                 # Plot points as small open circles (thinner edge)
                 ax.scatter(plot_lon, plot_lat, s=8, facecolors='none', edgecolors='white', linewidths=0.4, alpha=0.9, label='Chart Points')
                 # Add legend only to the 2023 subplot
                 #ax.legend(loc='upper right', fontsize=FONTSIZE_MAIN_TICKS - 2, frameon=True, framealpha=0.7, facecolor='white')
            # --------------------------

            ax.set_title(str(year), fontsize=FONTSIZE_MAIN_TICKS + 2) # Now 14pt
            ax.tick_params(axis='both', which='major', labelsize=FONTSIZE_MAIN_TICKS)

            ax.xaxis.set_major_locator(mticker.MultipleLocator(0.1))
            ax.yaxis.set_major_locator(mticker.MultipleLocator(0.1))
            ax.xaxis.set_major_formatter(mticker.FuncFormatter(lon_formatter))
            ax.yaxis.set_major_formatter(mticker.FuncFormatter(lat_formatter))
            
            # Only show tick labels on the outer plots
            if i >= num_years - num_cols:
                pass
            else:
                ax.tick_params(axis='x', labelbottom=False)

            if i % num_cols == 0:
                pass
            else:
                 ax.tick_params(axis='y', labelleft=False)

            # --- Overlay land mask ---
            if land_mask_array is not None:
                land_color = [0.5, 0.3, 0.1, 0.9] # RGBA for dark brown, more opaque
                water_color = [0, 0, 0, 0]
                cmap_land = colors.ListedColormap([land_color, water_color])
                bounds = [-0.5, 0.5, 1.5]
                norm_land = colors.BoundaryNorm(bounds, cmap_land.N)
                ax.imshow(land_mask_array, cmap=cmap_land, norm=norm_land, interpolation='nearest', zorder=10, extent=extent, origin='upper', aspect='auto')
            # -------------------------

        # Add colorbar
        if im:
            fig.tight_layout(rect=[0, 0.05, 1, 0.95])
            cax = fig.add_axes([0.15, 0.03, 0.7, 0.03])
            cbar = fig.colorbar(im, cax=cax, orientation='horizontal')
            cbar.set_label('Depth (m)', fontsize=FONTSIZE_MAIN_LABELS)
            cbar.ax.tick_params(labelsize=FONTSIZE_MAIN_TICKS)

        # Save the time series plot
        save_path = os.path.join(output_dir, f'{model_name}_time_series.jpg')
        plt.savefig(save_path, dpi=600, bbox_inches='tight')
        logger.info(f"Time series plot saved to: {save_path}")
        plt.close(fig) # Close the figure
        
    except Exception as e:
        logger.error(f"Failed to create time series plot: {str(e)}")
        raise

def stage2_s3gm_processing(system, sentinel_time_series, gebco_time_series, sentinel_years):
    """Stage 2: S3GM model processing"""
    try:
        # Create necessary directories
        results_dir = 'results'
        models_dir = os.path.join(results_dir, 's3gm_pretrained_models')  
        time_series_dir = os.path.join(results_dir, 's3gm_time_series') 
        os.makedirs(results_dir, exist_ok=True)
        os.makedirs(models_dir, exist_ok=True)
        os.makedirs(time_series_dir, exist_ok=True)
        # Stage 2.1: Pretraining
        pretrained_model_path = os.path.join(models_dir, 's3gm_pretrained.pth')
        if not os.path.exists(pretrained_model_path):
            logger.info("Executing Stage 2.1: Pretraining phase")
            
            # Load trained classic models first
            classic_models = load_trained_classic_models()
            system.set_classic_models(classic_models)
            
            # Generate predictions using classic models
            classic_predictions = []
            for year in sentinel_years:
                blue = sentinel_time_series['blue'][year - 2018]
                green = sentinel_time_series['green'][year - 2018]
                depth = system.classic_models.predict_rf(blue, green)
                if np.isnan(depth).any() or np.isinf(depth).any():
                    logger.warning(f"Classic model prediction for {year} contains invalid values")
                    depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
                classic_predictions.append(depth)
            
            # Convert predictions to time series array (T, H, W)
            classic_time_series = np.stack(classic_predictions)
            
            # Normalize data
            preprocessor = DataPreprocessor()
            normalized_classic, classic_stats = preprocessor._normalize_data(classic_time_series, 'classic')
            normalized_gebco, gebco_stats = preprocessor._normalize_data(gebco_time_series, 'gebco')
            
            # Save statistics for later denormalization
            np.save('intermediate_results/classic_stats.npy', classic_stats)
            np.save('intermediate_results/gebco_stats.npy', gebco_stats)
            
            # Save normalized data
            np.save('intermediate_results/classic_normalized.npy', normalized_classic)
            np.save('intermediate_results/gebco_normalized.npy', normalized_gebco)
            logger.info("Saved normalized classic model and GEBCO data")
            
            pretrain_data = {
                'classic': normalized_classic,  # Now in [0,1] range
                'gebco': normalized_gebco,     # Now in [0,1] range
            }
            
            # Execute pretraining (model is saved automatically inside pretrain function)
            system.s3gm.pretrain(
                classic_data=pretrain_data['classic'],
                gebco_data=pretrain_data['gebco'],
                save_path=pretrained_model_path
            )
        else:
            logger.info("Loading existing pretrained model")
            system.s3gm.load_pretrained(pretrained_model_path)
        
        # Log range adaptation parameters
        logger.info("Model configuration:")
        logger.info(f"  - Range adaptation: {system.s3gm.config.range_adaptation['enabled']}")
        logger.info(f"  - Mixed activation: {system.s3gm.config.range_adaptation['use_mixed_activation']}")
        logger.info(f"  - Land marker value: {system.s3gm.config.range_adaptation['land_value']}")
        
        # Validate model before Stage 2.2
        logger.info("Validating model...")
        test_input = torch.randn(1, 6, 5, 64, 64).to(system.s3gm.device)
        timesteps = torch.zeros(1, dtype=torch.long).to(system.s3gm.device)
        with torch.no_grad():
            test_output = system.s3gm.model(
                test_input,
                x0=test_input,
                timesteps=timesteps,
                obs_mask=torch.zeros(1, 6, 1, 1, 1).to(system.s3gm.device),
                latent_mask=torch.ones(1, 6, 1, 1, 1).to(system.s3gm.device),
                frame_indices=torch.arange(6).unsqueeze(0).to(system.s3gm.device)
            )
            # Get first element of model output (main output)
            test_output = test_output[0]
            # Validation logic
            if torch.all(torch.eq(test_output, torch.zeros_like(test_output))):
                raise ValueError("Pretrained model output is all zeros, retraining required")
            logger.info(f"Model test output range: [{test_output.min().item():.4f}, {test_output.max().item():.4f}]")
            
        # Stage 2.2: Conditional sampling
        logger.info("Executing Stage 2.2: Conditional sampling phase")
        # Get 2023 nautical chart data
        aoi = ee.Geometry.Rectangle([122.35, 30.62, 122.6, 30.8])
        preprocessor = DataPreprocessor(region=aoi)
        nautical_charts = preprocessor.get_sparse_points()
        
        # Normalize nautical chart data
        preprocessor = DataPreprocessor()
        normalized_depths, chart_stats = preprocessor._normalize_data(
            nautical_charts['depths'], 
            data_type='chart'
        )
        
        # Save nautical chart statistics (for later analysis)
        os.makedirs('intermediate_results', exist_ok=True)
        np.save('intermediate_results/chart_stats.npy', chart_stats)
        logger.info(f"Nautical chart statistics saved: {chart_stats}")
        
        # Update nautical_charts dictionary
        nautical_charts_normalized = {
            'depths': normalized_depths,
            'coordinates': nautical_charts['coordinates']
        }
        
        # Load preprocessed data
        # Note: Assuming classic_data and gebco_data are already normalized
        classic_data_normalized = np.load('intermediate_results/classic_normalized.npy')
        gebco_data_normalized = np.load('intermediate_results/gebco_normalized.npy')

        # Execute conditional sampling - pass classic model and GEBCO data
        results = system.s3gm.conditional_sampling(
            measurements=nautical_charts_normalized['depths'],
            measurement_coordinates=nautical_charts_normalized['coordinates'],
            years=sentinel_years,
            classic_data=classic_data_normalized,
            gebco_data=gebco_data_normalized
        )
        
        # Check results shape and content before saving
        if isinstance(results, torch.Tensor):
            results = results.cpu().numpy()
        
        # Save results with chart_stats
        save_results(
            depths=results,
            years=sentinel_years,
            output_dir=time_series_dir,
            chart_stats=chart_stats
        )
        
    except Exception as e:
        logger.error(f"Stage 2 processing failed: {str(e)}")
        raise

def stage3_postprocessing():
    """Stage 3: Post-processing analysis"""
    try:
        output_dir = 'results/s3gm_visualization'
        os.makedirs(output_dir, exist_ok=True)

        depths = []
        years = range(2018, 2024)
        for year in years:
            depth = np.load(f'results/s3gm_time_series/bathymetry_{year}.npy')
            depths.append(depth)

        mask_file = 'results/s3gm_visualization/land_water_mask.tif' # Corrected path
        land_mask_array = None
        try:
            mask_image = Image.open(mask_file)
            H, W = depths[0].shape
            mask_image_resized = mask_image.resize((W, H), Image.NEAREST)
            land_mask_array = np.array(mask_image_resized)
            logger.info(f"Successfully loaded and resized land/water mask from {mask_file}")
        except FileNotFoundError:
            logger.warning(f"Land/water mask file not found: {mask_file}. Skipping overlay.")
        except Exception as e:
            logger.error(f"Error loading or processing land/water mask: {e}. Skipping overlay.")

        try:
            import ee
            if not ee.data._credentials:
                 ee.Authenticate()
                 ee.Initialize(project='YOUR_GEE_PROJECT_ID')
        except ImportError:
             logger.error("Google Earth Engine Python API (ee) not found. Cannot get nautical charts.")
             raise
        except Exception as e:
             logger.error(f"GEE initialization failed: {e}")
             raise

        aoi = ee.Geometry.Rectangle([122.35, 30.62, 122.6, 30.8])
        preprocessor = DataPreprocessor(region=aoi)
        nautical_charts = preprocessor.get_sparse_points()

        chart_coords_s3gm = nautical_charts['coordinates']  # Get coordinates
        # Create time-series composite plot, passing the mask and chart coords
        create_time_series_plot(depths, years, 's3gm', output_dir, land_mask_array, chart_coords=chart_coords_s3gm)

        # Validation part (Now starts here)
        t = -1
        predicted_depth = depths[t]
        # H, W defined earlier during mask loading

        coords = nautical_charts['coordinates']
        true_depths = np.abs(nautical_charts['depths'])

        predicted_values = []
        valid_true_depths = []

        # Extract valid points using the original predicted_depth before masking for plotting
        for i, (y, x) in enumerate(coords):
            y_idx = int(y * (H - 1))
            x_idx = int(x * (W - 1))
            if 0 <= y_idx < H and 0 <= x_idx < W:
                pred_value = predicted_depth[y_idx, x_idx]
                if np.isfinite(pred_value) and pred_value > 0:
                    predicted_values.append(pred_value)
                    valid_true_depths.append(true_depths[i])

        predicted_values = np.array(predicted_values)
        valid_true_depths = np.array(valid_true_depths)

        if len(valid_true_depths) > 0 and len(predicted_values) > 0:
            rmse = np.sqrt(np.mean((predicted_values - valid_true_depths) ** 2))
            mae = np.mean(np.abs(predicted_values - valid_true_depths))
            denom = np.sum((valid_true_depths - np.mean(valid_true_depths)) ** 2)
            r2 = 1 - np.sum((valid_true_depths - predicted_values) ** 2) / denom if denom != 0 else -np.inf
        else:
            rmse, mae, r2 = np.nan, np.nan, np.nan
            logger.warning("No valid overlapping points found for validation.")

        logger.info("S3GM Model Validation Results:")
        logger.info(f"RMSE: {rmse:.2f} m" if np.isfinite(rmse) else "RMSE: N/A")
        logger.info(f"MAE: {mae:.2f} m" if np.isfinite(mae) else "MAE: N/A")
        logger.info(f"R²: {r2:.4f}" if np.isfinite(r2) else "R²: N/A")
        logger.info(f"Number of validation points: {len(valid_true_depths)}")

        # Create scatter plot with the same style as RF validation
        FONTSIZE_MAIN_LABELS = 12
        FONTSIZE_MAIN_TICKS = 10
        FONTSIZE_TEXT_BOX = 10 # Slightly smaller for text box
        fig_scatter = plt.figure(figsize=(7, 5.5)) # Match RF plot aspect ratio
        ax_scatter = fig_scatter.add_subplot(1, 1, 1)
        if len(valid_true_depths) > 0 and len(predicted_values) > 0 and np.isfinite(r2):
            scatter = ax_scatter.scatter(valid_true_depths, predicted_values, alpha=0.6, s=30, label='Samples')
            ideal_line, = ax_scatter.plot([0, 75], [0, 75], 'r--', linewidth=1.5, label='Ideal fit')  # Match RF plot
            ax_scatter.set_xlim(0, 75)
            ax_scatter.set_ylim(0, 75)
            # Use S3GM metrics, format matches RF plot (no N)
            text_str = f'RMSE: {rmse:.2f} m\nMAE: {mae:.2f} m\nR²: {r2:.4f}' 
            ax_scatter.text(0.95, 0.05, text_str,
                      transform=ax_scatter.transAxes, fontsize=FONTSIZE_TEXT_BOX,
                      verticalalignment='bottom', horizontalalignment='right',
                      bbox=dict(boxstyle='round,pad=0.5', fc='white', alpha=0.8))
            ax_scatter.legend(loc='upper left') # Match RF plot
        else:
            ax_scatter.text(0.5, 0.5, 'No valid data for scatter plot', ha='center', va='center')
            ax_scatter.set_xlim(0, 75) # Still set limits even if no data
            ax_scatter.set_ylim(0, 75)

        ax_scatter.set_xlabel('Measured Depth (m)', fontsize=FONTSIZE_MAIN_LABELS)
        ax_scatter.set_ylabel('Predicted Depth (m)', fontsize=FONTSIZE_MAIN_LABELS)
        ax_scatter.tick_params(axis='both', which='major', labelsize=FONTSIZE_MAIN_TICKS)
        ax_scatter.grid(True, linestyle='--', alpha=0.6) # Match RF plot
        # No title, matching RF plot
        fig_scatter.tight_layout()
        save_path_scatter = os.path.join(output_dir, 's3gm_validation_2023.jpg') # Correct filename
        plt.savefig(save_path_scatter, dpi=600, bbox_inches='tight')
        logger.info(f"S3GM Scatter plot saved to: {save_path_scatter}")
        plt.close(fig_scatter) # Close the specific figure

        # --- End of S3GM Scatter Plot ---

        def lon_formatter(x, pos): return f'{x:.1f}°E'
        def lat_formatter(y, pos): return f'{y:.1f}°N'

        # --- Generate Difference Map (S3GM - RF) for 2023 ---
        try:
            # Load RF 2023 results (assuming stage 1.8 was run and saved rf_2023.npy)
            # First, check if rf_time_series.npy exists to get the path structure
            rf_output_dir = 'results/classic_models'
            # Attempt to load the RF 2023 prediction (assuming it was saved individually, modify if needed)
            # Let's assume rf_depths were saved in stage1.8 similarly to how s3gm depths are saved
            # If not, we need to reload sentinel data and run predict_rf for 2023 again
            # ---- SAFER APPROACH: Load RF results if available ----
            rf_2023 = None
            rf_depths_list = []
            try:
                 # Load all RF depths if saved as a list/stack
                 # This depends on how stage1_8 saved its predictions.
                 # Let's assume it saved depths similar to stage 3
                 rf_years = range(2018, 2024)
                 for rf_year in rf_years:
                    rf_depth_file = os.path.join(rf_output_dir, f'rf_{rf_year}.npy') # Assuming this naming
                    if os.path.exists(rf_depth_file):
                       rf_depths_list.append(np.load(rf_depth_file))
                 if len(rf_depths_list) == len(rf_years):
                     rf_2023 = rf_depths_list[-1] # Get the 2023 data
                 else:
                    logger.warning(f"Could not find all RF annual prediction files in {rf_output_dir}. Trying to load specifically rf_2023.npy if it exists.")
                    # Fallback: Check if a single file was saved
                    specific_rf_file = os.path.join(rf_output_dir, 'rf_2023.npy') # Check for this specific name convention
                    if os.path.exists(specific_rf_file):
                        rf_2023 = np.load(specific_rf_file)

            except Exception as load_err:
                 logger.warning(f"Could not load pre-saved RF results for 2023: {load_err}. Skipping difference map.")

            s3gm_2023 = depths[-1] # Get the S3GM 2023 result

            if rf_2023 is not None and s3gm_2023 is not None:
                logger.info("Calculating S3GM - RF difference map for 2023.")
                difference_map = s3gm_2023 - rf_2023

                # Apply land mask to the difference map
                if land_mask_array is not None:
                    difference_map[land_mask_array == 0] = np.nan # Mask land pixels

                # Determine symmetric color limits
                max_abs_diff = np.nanmax(np.abs(difference_map))
                diff_vmin = -max_abs_diff
                diff_vmax = max_abs_diff

                # Plotting
                FONTSIZE_MAIN_LABELS = 12 # Match other plots if needed
                FONTSIZE_MAIN_TICKS = 10
                fig_diff = plt.figure(figsize=(7, 5.5)) # Match scatter plot aspect ratio
                ax_diff = fig_diff.add_subplot(1, 1, 1)
                extent = [122.35, 122.6, 30.62, 30.8] # Same extent as other maps

                im_diff = ax_diff.imshow(difference_map, cmap='coolwarm', vmin=diff_vmin, vmax=diff_vmax, extent=extent, origin='upper', aspect='auto', interpolation='nearest')

                # Apply land mask overlay visualization (optional, but good for context)
                if land_mask_array is not None:
                    land_color = [0.5, 0.3, 0.1, 0.9] # Dark brown
                    water_color = [0, 0, 0, 0] # Transparent
                    cmap_land = colors.ListedColormap([land_color, water_color])
                    bounds = [-0.5, 0.5, 1.5]
                    norm_land = colors.BoundaryNorm(bounds, cmap_land.N)
                    ax_diff.imshow(land_mask_array, cmap=cmap_land, norm=norm_land, interpolation='nearest', zorder=10, extent=extent, origin='upper', aspect='auto')

                #ax_diff.set_title('Difference: S3GM - RF (2023)', fontsize=FONTSIZE_MAIN_TICKS + 2)
                ax_diff.tick_params(axis='both', which='major', labelsize=FONTSIZE_MAIN_TICKS)
                ax_diff.xaxis.set_major_locator(mticker.MultipleLocator(0.1))
                ax_diff.yaxis.set_major_locator(mticker.MultipleLocator(0.1))
                ax_diff.xaxis.set_major_formatter(mticker.FuncFormatter(lon_formatter)) # Use formatters defined earlier
                ax_diff.yaxis.set_major_formatter(mticker.FuncFormatter(lat_formatter))

                # Add colorbar
                cbar = fig_diff.colorbar(im_diff, ax=ax_diff, orientation='vertical', fraction=0.046, pad=0.04)
                cbar.set_label('Depth Difference (m)', fontsize=FONTSIZE_MAIN_LABELS)
                cbar.ax.tick_params(labelsize=FONTSIZE_MAIN_TICKS)

                fig_diff.tight_layout()
                save_path_diff = os.path.join(output_dir, 's3gm_rf_difference_2023.jpg')
                plt.savefig(save_path_diff, dpi=600, bbox_inches='tight')
                logger.info(f"Difference map saved to: {save_path_diff}")
                plt.close(fig_diff)
            else:
                logger.warning("RF or S3GM 2023 results not available. Skipping difference map generation.")

        except Exception as e:
            logger.error(f"Failed to generate difference map: {str(e)}")
        # --- End of Difference Map ---

    except Exception as e:
        logger.error(f"Post-processing analysis failed: {str(e)}")
        raise

def load_trained_classic_models():
    """Load trained classic models and parameters"""
    try:
        classic_models = ClassicModels()
        
        # Load parameters
        rf_params = np.load('intermediate_results/model_params/rf_params.npy', allow_pickle=True).item()
        classic_models.rf_params = rf_params
        
        # Load Random Forest model
        classic_models.rf_model = load('intermediate_results/model_params/rf_model.joblib')
        
        return classic_models
        
    except Exception as e:
        logger.error(f"Failed to load classic models: {str(e)}")
        raise

def denormalize_bathymetry(normalized_data: np.ndarray, stats: Dict[str, float]) -> np.ndarray:
    """Denormalize bathymetry data using Min-Max and enforce physical constraints"""
    try:
        # Get physical range and special values from stats
        min_phys = stats.get('min_phys', 0.0)
        max_phys = stats.get('max_phys', 90.0)
        land_value_norm = stats.get('land_value', 1.5)
        eps = 1e-6

        # Identify land regions (using isclose to handle floating point errors)
        land_mask = np.isclose(normalized_data, land_value_norm)
        
        # Check actual data range (for debugging)
        valid_data = normalized_data[~land_mask & ~np.isnan(normalized_data)]
        if len(valid_data) > 0:
            actual_min, actual_max = valid_data.min(), valid_data.max()
            logger.info(f"Valid data range before denormalization: [{actual_min:.4f}, {actual_max:.4f}]")
            # Warn if values exceed [-1, 1] range
            if actual_min < -1.0 - eps or actual_max > 1.0 + eps:
                 logger.warning(f"  Note: Valid data range exceeds expected [-1, 1] interval!")
        else:
            logger.warning("No valid (non-land/NaN) data found before denormalization")

        # Execute denormalization: phys = ((norm + 1) / 2) * (max_phys - min_phys) + min_phys
        depth_denorm = ((normalized_data + 1) / 2) * (max_phys - min_phys) + min_phys

        # Handle special regions: set land areas to 0m (or NaN as needed)
        depth_denorm = np.where(land_mask, 0.0, depth_denorm)
        
        # Enforce physical constraints: depth must be non-negative
        sea_mask = ~land_mask
        if np.any(depth_denorm[sea_mask] < 0):
            neg_count = np.sum(depth_denorm[sea_mask] < 0)
            total_count = np.sum(sea_mask)
            logger.warning(f"Negative depth values detected, {(neg_count/total_count)*100:.2f}% of sea pixels (should not occur)")
            logger.warning(f"Negative value range: [{depth_denorm[sea_mask & (depth_denorm < 0)].min():.2f}, 0) m")
            depth_denorm[sea_mask] = np.maximum(depth_denorm[sea_mask], 0.0)
        
        # Handle possible NaN values
        depth_denorm = np.nan_to_num(depth_denorm, nan=0.0)
        
        # Final check and logging
        if np.any(sea_mask):
            sea_depths = depth_denorm[sea_mask]
            logger.info(f"Depth range after denormalization (sea): [{sea_depths.min():.2f}, {sea_depths.max():.2f}]")
        else:
            logger.warning("No sea pixels after denormalization")
            
        return depth_denorm
        
    except Exception as e:
        logger.error(f"Denormalization failed: {str(e)}")
        raise

def save_results(depths, years, output_dir, chart_stats):
    """Save prediction results
    
    Args:
        depths: Tensor with shape [1, T, C, H, W], where:
            - 1: batch size
            - T: number of time steps (years)
            - C: number of channels (components)
            - H, W: image height and width
        output_dir: Output directory
        chart_stats: Statistics of nautical chart data
    """
    try:
        os.makedirs(output_dir, exist_ok=True)
        
        # Denormalize and save prediction results for each year
        for t, year in enumerate(years):
            # Get depth data for current year (channel 2 is depth data)
            depth_data = depths[0, t, 2]  # Shape should be [H, W]
            
            # Check data before denormalization
            logger.info(f"Year {year} before denormalization:")
            logger.info(f"- Data range: [{depth_data.min():.4f}, {depth_data.max():.4f}]")
            logger.info(f"- Non-zero value distribution: {np.percentile(depth_data[depth_data != 0], [25, 50, 75])}")
            
            # Denormalize using unified chart statistics
            depth_denorm = denormalize_bathymetry(depth_data, chart_stats)
            
            # Check after denormalization
            logger.info(f"Year {year} after denormalization:")
            logger.info(f"- Data range: [{depth_denorm.min():.4f}, {depth_denorm.max():.4f}]")
            logger.info(f"- Valid value distribution: {np.percentile(depth_denorm[depth_denorm > 0], [25, 50, 75])}")
            
            # Save denormalized results
            output_path = os.path.join(output_dir, f'bathymetry_{year}.npy')
            np.save(output_path, depth_denorm)
            
            logger.info(f"Saved prediction results for {year}: {output_path}")
            logger.info(f"Year {year} depth range: [{depth_denorm.min():.2f}, {depth_denorm.max():.2f}] m")
            logger.info(f"Year {year} land pixel ratio: {np.mean(np.abs(depth_data - 1.5) < 0.1):.2%}")
            
    except Exception as e:
        logger.error(f"Failed to save results: {str(e)}")
        raise

def stage4_statistical_analysis():
    """Stage 4: Statistical significance analysis"""
    try:
        logger.info("Starting Stage 4: Statistical significance analysis")

        # 1. Load ground truth nautical chart data
        logger.info("Loading nautical chart data...")
        aoi = ee.Geometry.Rectangle([122.35, 30.62, 122.6, 30.8])
        preprocessor = DataPreprocessor(region=aoi)
        nautical_charts = preprocessor.get_sparse_points()
        true_depths_all = np.abs(nautical_charts['depths'])
        coords = nautical_charts['coordinates']
        logger.info(f"Successfully loaded {len(true_depths_all)} nautical chart measurement points")

        # 2. Load 2023 model prediction results
        logger.info("Loading RF and S3GM 2023 prediction results...")
        rf_pred_path = 'results/classic_models/rf_2023.npy'
        s3gm_pred_path = 'results/s3gm_time_series/bathymetry_2023.npy'

        if not os.path.exists(rf_pred_path):
            logger.error(f"RF prediction file not found: {rf_pred_path}. Please run Stage 1.8 first.")
            return
        if not os.path.exists(s3gm_pred_path):
            logger.error(f"S3GM prediction file not found: {s3gm_pred_path}. Please run Stage 2 and 3 first.")
            return
            
        rf_preds_map = np.load(rf_pred_path)
        s3gm_preds_map = np.load(s3gm_pred_path)
        logger.info("Prediction results loaded successfully")

        # 3. Align data
        H, W = rf_preds_map.shape
        rf_preds_aligned = []
        s3gm_preds_aligned = []
        true_depths_aligned = []

        for i, (y, x) in enumerate(coords):
            y_idx, x_idx = int(y * (H - 1)), int(x * (W - 1))
            if 0 <= y_idx < H and 0 <= x_idx < W:
                rf_val = rf_preds_map[y_idx, x_idx]
                s3gm_val = s3gm_preds_map[y_idx, x_idx]
                
                # Keep only points where all models and ground truth are valid
                if np.isfinite(rf_val) and np.isfinite(s3gm_val) and rf_val > 0 and s3gm_val > 0:
                    rf_preds_aligned.append(rf_val)
                    s3gm_preds_aligned.append(s3gm_val)
                    true_depths_aligned.append(true_depths_all[i])

        rf_preds_aligned = np.array(rf_preds_aligned)
        s3gm_preds_aligned = np.array(s3gm_preds_aligned)
        true_depths_aligned = np.array(true_depths_aligned)
        logger.info(f"After alignment, valid validation points for statistical analysis: {len(true_depths_aligned)}")

        # 4. Calculate absolute errors
        rf_abs_errors = np.abs(rf_preds_aligned - true_depths_aligned)
        s3gm_abs_errors = np.abs(s3gm_preds_aligned - true_depths_aligned)

        # 5. Perform Wilcoxon signed-rank test
        # H0: Median of error differences is 0
        # H1: RF error > S3GM error (S3GM error is smaller)
        logger.info("Performing Wilcoxon signed-rank test...")
        stat, p_value = wilcoxon(rf_abs_errors, s3gm_abs_errors, alternative='greater')
        logger.info("--- Wilcoxon Signed-Rank Test Results ---")
        logger.info(f"Statistic: {stat:.4f}")
        logger.info(f"P-value: {p_value:.6f}")
        if p_value < 0.05:
            logger.info("Interpretation: P-value < 0.05, we reject the null hypothesis. S3GM model prediction error is statistically significantly lower than RF model.")
        else:
            logger.info("Interpretation: P-value >= 0.05, we cannot reject the null hypothesis. Insufficient statistical evidence that S3GM model is significantly better than RF model.")

        # 6. Use Bootstrapping to calculate 95% confidence interval for MAE improvement
        logger.info("\nUsing Bootstrapping to calculate 95% confidence interval for MAE improvement...")
        diff_mae = rf_abs_errors - s3gm_abs_errors # MAE_RF - MAE_S3GM
        n_iterations = 10000
        bootstrap_means = []
        for _ in range(n_iterations):
            sample_indices = np.random.choice(len(diff_mae), size=len(diff_mae), replace=True)
            bootstrap_means.append(np.mean(diff_mae[sample_indices]))
        
        confidence_interval = np.percentile(bootstrap_means, [2.5, 97.5])
        mean_improvement = np.mean(diff_mae)
        logger.info("--- 95% Confidence Interval for MAE Improvement (MAE_RF - MAE_S3GM) ---")
        logger.info(f"Mean improvement: {mean_improvement:.4f} m")
        logger.info(f"95% confidence interval: [{confidence_interval[0]:.4f}, {confidence_interval[1]:.4f}] m")
        if confidence_interval[0] > 0:
            logger.info("Interpretation: Confidence interval is entirely above zero, further confirming S3GM model provides robust and significant accuracy improvement.")
        else:
            logger.info("Interpretation: Confidence interval includes zero, indicating the model improvement may not be robust.")

    except Exception as e:
        logger.error(f"Stage 4 statistical analysis failed: {str(e)}")
        raise

def stage5_in_depth_analysis():
    """Stage 5: Model performance analysis by depth and terrain zones"""
    try:
        logger.info("Starting Stage 5: Detailed performance analysis by depth and terrain zones")

        # 1. Load data
        logger.info("Loading nautical chart, RF and S3GM 2023 prediction results...")
        aoi = ee.Geometry.Rectangle([122.35, 30.62, 122.6, 30.8])
        preprocessor = DataPreprocessor(region=aoi)
        nautical_charts = preprocessor.get_sparse_points()
        true_depths_all = np.abs(nautical_charts['depths'])
        coords = nautical_charts['coordinates']

        rf_pred_path = 'results/classic_models/rf_2023.npy'
        s3gm_pred_path = 'results/s3gm_time_series/bathymetry_2023.npy'

        if not os.path.exists(rf_pred_path):
            rf_pred_path_alt = 'results/classic_models/rf_time_series.npy'
            if not os.path.exists(rf_pred_path_alt):
                 logger.error(f"RF prediction file not found: {rf_pred_path} or {rf_pred_path_alt}")
                 return
            else: # If saved as a list, load and get the last element
                 rf_preds_map = np.load(rf_pred_path_alt, allow_pickle=True)[-1]
        else:
            rf_preds_map = np.load(rf_pred_path)

        s3gm_preds_map = np.load(s3gm_pred_path)
        
        # 2. Align data
        H, W = rf_preds_map.shape
        aligned_data = []
        for i, (y, x) in enumerate(coords):
            y_idx, x_idx = int(y * (H - 1)), int(x * (W - 1))
            if 0 <= y_idx < H and 0 <= x_idx < W:
                rf_val = rf_preds_map[y_idx, x_idx]
                s3gm_val = s3gm_preds_map[y_idx, x_idx]
                if np.isfinite(rf_val) and np.isfinite(s3gm_val) and rf_val > 0 and s3gm_val > 0:
                    aligned_data.append({
                        'true': true_depths_all[i],
                        'rf': rf_val,
                        's3gm': s3gm_val,
                        'y_idx': y_idx,
                        'x_idx': x_idx
                    })
        
        logger.info(f"After alignment, valid points for analysis: {len(aligned_data)}")

        # 3. Analysis by depth zones
        logger.info("\n--- Model performance analysis by depth zones ---")
        depth_bins = {
            'Shallow (0-10m)': (0, 10),
            'Intermediate (10-30m)': (10, 30),
            'Deep (>30m)': (30, 100)
        }
        
        print("\n| Depth Range          | Model | N    | RMSE (m) | MAE (m)  |")
        print("|----------------------|-------|------|----------|----------|")

        for name, (min_d, max_d) in depth_bins.items():
            bin_data = [p for p in aligned_data if min_d < p['true'] <= max_d]
            if not bin_data: continue

            true = np.array([p['true'] for p in bin_data])
            rf = np.array([p['rf'] for p in bin_data])
            s3gm = np.array([p['s3gm'] for p in bin_data])
            
            rf_rmse = np.sqrt(np.mean((rf - true)**2))
            rf_mae = np.mean(np.abs(rf - true))
            s3gm_rmse = np.sqrt(np.mean((s3gm - true)**2))
            s3gm_mae = np.mean(np.abs(s3gm - true))
            
            print(f"| {name:<20} | RF    | {len(true):<4} | {rf_rmse:<8.2f} | {rf_mae:<8.2f} |")
            print(f"| {name:<20} | S3GM  | {len(true):<4} | {s3gm_rmse:<8.2f} | {s3gm_mae:<8.2f} |")

        # 4. Analysis by terrain slope
        logger.info("\n--- Model performance analysis by terrain slope ---")
        grad_y, grad_x = np.gradient(s3gm_preds_map)
        slope_map = np.sqrt(grad_y**2 + grad_x**2)
        
        for p in aligned_data:
            p['slope'] = slope_map[p['y_idx'], p['x_idx']]

        slopes = np.array([p['slope'] for p in aligned_data])
        slope_quantiles = np.percentile(slopes, [33.3, 66.6])

        slope_bins = {
            'Low Slope': (0, slope_quantiles[0]),
            'Medium Slope': (slope_quantiles[0], slope_quantiles[1]),
            'High Slope': (slope_quantiles[1], np.inf)
        }
        
        print("\n| Topography           | Model | N    | RMSE (m) | MAE (m)  |")
        print("|----------------------|-------|------|----------|----------|")

        for name, (min_s, max_s) in slope_bins.items():
            bin_data = [p for p in aligned_data if min_s <= p['slope'] < max_s]
            if not bin_data: continue
            
            true = np.array([p['true'] for p in bin_data])
            rf = np.array([p['rf'] for p in bin_data])
            s3gm = np.array([p['s3gm'] for p in bin_data])

            rf_rmse = np.sqrt(np.mean((rf - true)**2))
            rf_mae = np.mean(np.abs(rf - true))
            s3gm_rmse = np.sqrt(np.mean((s3gm - true)**2))
            s3gm_mae = np.mean(np.abs(s3gm - true))
            
            print(f"| {name:<20} | RF    | {len(true):<4} | {rf_rmse:<8.2f} | {rf_mae:<8.2f} |")
            print(f"| {name:<20} | S3GM  | {len(true):<4} | {s3gm_rmse:<8.2f} | {s3gm_mae:<8.2f} |")
            
    except Exception as e:
        logger.error(f"Stage 5 detailed performance analysis failed: {str(e)}")
        raise

def main():
    try:
        logger.info("Starting bathymetry inversion program...")
        
        args = parse_args()
        
        # Initialize basic parameters
        aoi = ee.Geometry.Rectangle([122.35, 30.62, 122.6, 30.8])
        gebco_years = range(2019, 2025)
        sentinel_years = range(2018, 2024)
        
        if args.stage == '1':
            logger.info("Executing Stage 1: Data preprocessing")
            stage1_preprocessing(aoi, gebco_years, sentinel_years)
            
        elif args.stage == '1.5':
            logger.info("Executing Stage 1.5: Classic model training")
            sentinel_time_series = np.load('intermediate_results/sentinel_time_series.npy', allow_pickle=True).item()
            classic_models = stage1_5_model_training(sentinel_time_series)
        
        elif args.stage == '1.8':
            logger.info("Executing Stage 1.8: Classic model validation")
            sentinel_time_series = np.load('intermediate_results/sentinel_time_series.npy', allow_pickle=True).item()
            stage1_8_model_validation(sentinel_time_series)
                
        elif args.stage == '2':
            logger.info("Executing Stage 2: S3GM model processing")
            # Load preprocessed data and trained model parameters
            sentinel_time_series = np.load('intermediate_results/sentinel_time_series.npy', allow_pickle=True).item()
            gebco_time_series = np.load('intermediate_results/gebco_time_series.npy')
            
            # Initialize system
            system = HybridBathymetrySystem(
                region=aoi,
                time_range={'gebco': gebco_years, 'sentinel': sentinel_years}
            )
            
            # Execute Stage 2
            stage2_s3gm_processing(system, sentinel_time_series, gebco_time_series, sentinel_years)

        elif args.stage == '3':
            logger.info("Executing Stage 3: Post-processing and visualization")
            stage3_postprocessing()

        elif args.stage == '4':
            logger.info("Executing Stage 4: Statistical significance analysis")
            stage4_statistical_analysis()

        elif args.stage == '5':
            logger.info("Executing Stage 5: Detailed performance analysis by depth and terrain zones")
            stage5_in_depth_analysis()

    except Exception as e:
        logger.error(f"Error occurred during program execution: {str(e)}")
        raise

if __name__ == '__main__':
    setup_logging()  # Only call once here
    main()

# Run Stage 1 (Data preprocessing)
# python run_bathymetry.py --stage 1

# Run Stage 1.5 (Classic model training)
# python run_bathymetry.py --stage 1.5

# Run Stage 1.8 (Classic model validation)
# python run_bathymetry.py --stage 1.8

# Run Stage 2 (S3GM model processing)
# python run_bathymetry.py --stage 2

# Run Stage 3 (Post-processing and visualization)
# python run_bathymetry.py --stage 3

# Run Stage 4 (Statistical analysis)
# python run_bathymetry.py --stage 4

# Run Stage 5 (Detailed performance analysis)
# python run_bathymetry.py --stage 5