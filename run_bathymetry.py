#!/usr/bin/env python3
"""
BathySurrogate: An Open-Source Environmental Surrogate Framework for Satellite-Derived Bathymetry
Pipeline Execution Script

Usage:
    python run_bathymetry.py --stage <num>

Stages:
    1: Data Preprocessing (Sentinel-2 MIWC composite + GEBCO grids)
    1.5: Random Forest Surrogate Model Training (Full & 5-Fold Spatial CV)
    1.8: Random Forest Surrogate Model Validation (Out-of-Fold 5-Fold CV)
    2: S3GM Spatio-Temporal Generative Diffusion Model Conditional Sampling
    3: Post-processing and Spatial Visualization
    4: Statistical Significance Analysis (Wilcoxon Signed-Rank Test)
    5: Zoning & Terrain Performance Analysis
"""

import sys
import time
import os
import logging
import argparse
import math
from datetime import datetime
from typing import List, Dict, Optional, Tuple, Union

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.gridspec
import matplotlib.ticker as mticker
from PIL import Image
from joblib import dump, load
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import zoom
from scipy.stats import wilcoxon

from bathysurrogate.main import HybridBathymetrySystem
from bathysurrogate.classic_models import ClassicModels
from bathysurrogate.preprocessor import DataPreprocessor
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def setup_logging():
    """Configure logging system."""
    if len(logging.getLogger().handlers) > 0:
        return
        
    log_dir = 'logging'
    os.makedirs(log_dir, exist_ok=True)
    
    # Default level is INFO; use DEBUG for detailed tracing
    logging.basicConfig(
        level=logging.INFO,  # or logging.DEBUG for verbose output
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='BathySurrogate: Environmental Surrogate Computational Framework')
    parser.add_argument(
        '--stage',
        type=str,
        choices=['1', '1.5', '1.8','2', '3', '4', '5'],
        help='Processing stage: 1=Preprocessing, 1.5=Surrogate Training, 1.8=Surrogate Validation, 2=S3GM Sampling, 3=Post-processing, 4=Statistical Tests, 5=Zoning Analysis'
    )
    return parser.parse_args()

def stage1_preprocessing(aoi_coords, gebco_years, sentinel_years):
    """Stage 1: Data Acquisition & Preprocessing."""
    try:
        import ee
        ee.Authenticate()
        ee.Initialize(project='YOUR_GEE_PROJECT_ID')
        aoi = ee.Geometry.Rectangle(aoi_coords)
        from data_acquisition_preprocessing import (
            get_sentinel2_images,
            load_gebco_data
        )
        from miwc import apply_miwc
        # Initialize time-series storage structures
        sentinel_time_series = {
            'blue': np.zeros((6, 64, 64)),  # [T, H, W]
            'green': np.zeros((6, 64, 64))
        }
        gebco_time_series = np.zeros((6, 64, 64))  # [T, H, W]
        
        # Process each annual observation frame
        for idx, (gebco_year, sentinel_year) in enumerate(zip(gebco_years, sentinel_years)):
            logger.info(f"Processing GEBCO grid {gebco_year} (corresponding to {sentinel_year} observations)...")
            
            # GEBCO bathymetry grid processing
            gebco_file = os.path.join(
                'GEBCO_Bathymetry',
                'GEBCO_26_Dec_2024_86912dfafafa',
                f'gebco_{gebco_year}_n30.8_s30.62_w122.35_e122.6.nc'
            )
            depth_data = load_gebco_data(gebco_file)
            if depth_data.shape != (64, 64):
                # Create target computational grid
                target_h, target_w = 64, 64
                orig_h, orig_w = depth_data.shape
    
                # Create source and target coordinate meshes
                y_orig = np.linspace(0, 1, orig_h)
                x_orig = np.linspace(0, 1, orig_w)
                y_target = np.linspace(0, 1, target_h)
                x_target = np.linspace(0, 1, target_w)
    
                # Bilinear interpolation resampling
                interpolator = RegularGridInterpolator(
                    (y_orig, x_orig), 
                    depth_data,
                    method='linear',
                    bounds_error=False,
                    fill_value=np.nan
                )
    
                # Create target coordinate query points
                xx, yy = np.meshgrid(x_target, y_target)
                points = np.stack([yy.ravel(), xx.ravel()], axis=1)
    
                # Execute interpolation
                depth_data = interpolator(points).reshape(target_h, target_w)
    
                logger.info(f"Resampled GEBCO grid to target resolution: {depth_data.shape}")
            
            gebco_time_series[idx] = depth_data
            logger.info(f"GEBCO depth range (post-resampling): [{depth_data.min():.4f}, {depth_data.max():.4f}]")
            
            # Sentinel-2 optical imagery processing
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
                """Extract raster array for specified optical band."""
                try:
                    # Retrieve available spectral bands
                    available_bands = image.bandNames().getInfo()
                    logger.info(f"Available spectral bands: {available_bands}")
                    
                    # Reproject and resample image to target resolution
                    target_scale = math.sqrt(aoi.area().getInfo() / (64 * 64))  # Compute target ground sampling distance
                    resampled_image = image.resample('bilinear').reproject(
                        crs=image.projection(),
                        scale=target_scale
                    )
                    
                    # Select spectral band
                    band_data = resampled_image.select([band_name])
                    
                    # Fetch image region array
                    data = band_data.reduceRegion(
                        reducer=ee.Reducer.toList(),
                        geometry=aoi,
                        scale=target_scale,
                        maxPixels=1e13
                    ).get(band_name).getInfo()
                    
                    if not data:
                        logger.error(f"{band_name}band data retrieval failed")
                        return None
                        
                    # Convert to NumPy array and reshape
                    temp_grid = np.array(data)
                    logger.info(f"Retrieved array size: {temp_grid.size}")
                    
                    # Compute square grid dimensions
                    grid_size = int(np.sqrt(temp_grid.size))
                    temp_grid = temp_grid[:grid_size*grid_size].reshape(grid_size, grid_size)
                    
                    # Resample to 64x64 grid
                    resampled_grid = zoom(temp_grid, (64/grid_size, 64/grid_size), order=1)
                    logger.info(f"Resampled grid shape: {resampled_grid.shape}")
                    
                    return resampled_grid
                    
                except Exception as e:
                    logger.error(f"Failed to extract {band_name} band data: {str(e)}")
                    return None
            
            # Verify processed composite image validity
            processed_image = ee.Image(processed_image)
            available_bands = processed_image.bandNames().getInfo()
            logger.info(f"Available bands post-MIWC composite: {available_bands}")

            if not all(band in available_bands for band in ['blue', 'green']):
                logger.warning("Missing required bands in MIWC composite")
                continue

            # Extract Blue and Green bands
            blue_array = get_band_array(processed_image, 'blue', aoi)
            green_array = get_band_array(processed_image, 'green', aoi)
            
            # Validate extracted arrays
            if blue_array is None or green_array is None:
                logger.error("Failed to retrieve valid spectral band arrays")
                continue
                
            if blue_array.shape != green_array.shape:
                logger.error(f"Band shape mismatch: blue={blue_array.shape}, green={green_array.shape}")
                continue
                
            if np.all(np.isnan(blue_array)) or np.all(np.isnan(green_array)):
                logger.error("Band data contains only NaNs")
                continue
            
            sentinel_time_series['blue'][idx] = blue_array
            sentinel_time_series['green'][idx] = green_array
            logger.info(f"Sentinel-2 Blue band range (resampled): [{blue_array.min():.4f}, {blue_array.max():.4f}]")
            logger.info(f"Sentinel-2 Green band range (resampled): [{green_array.min():.4f}, {green_array.max():.4f}]")
            
            logger.info(f"Completed preprocessing for year {sentinel_year}")
            
        # Save preprocessed arrays
        output_dir = 'intermediate_results'
        os.makedirs(output_dir, exist_ok=True)
        np.save(os.path.join(output_dir, 'sentinel_time_series.npy'), sentinel_time_series)
        np.save(os.path.join(output_dir, 'gebco_time_series.npy'), gebco_time_series)
        
        return sentinel_time_series, gebco_time_series
        
    except Exception as e:
        logger.error(f"Stage 1 processing failed: {str(e)}")
        raise

def stage1_5_model_training(sentinel_time_series):
    """Stage 1.5: Surrogate Model Training (Full & 5-Fold Spatial CV)."""
    try:
        logger.info("Starting surrogate model training...")
        classic_models = ClassicModels()
        
        # Initialize preprocessor to retrieve sounding ground truth
        aoi = [122.35, 30.62, 122.6, 30.8]
        preprocessor = DataPreprocessor(region=aoi)
        
        # 1. Ensure and load spatially blocked fold partitions
        folds_path = 'intermediate_results/validation/spatial_folds.npz'
        if not os.path.exists(folds_path):
            os.makedirs(os.path.dirname(folds_path), exist_ok=True)
            logger.info("Spatial folds not found, generating new blocked 5-fold partitions...")
            preprocessor.generate_spatial_folds(n_splits=5, seed=2026)
            
        folds_data = np.load(folds_path)
        all_coords = folds_data['coordinates']
        all_depths = folds_data['depths']
        fold_ids = folds_data['fold_ids']
        
        # Extract optical features for all sounding locations
        t = -1  # Latest time frame (2023)
        blue = sentinel_time_series['blue'][t]
        green = sentinel_time_series['green'][t]
        H, W = blue.shape
        
        # 2. Train full-dataset model (strictly for cartographic visualization products)
        blue_values_all = []
        green_values_all = []
        valid_depths_all = []
        for i, (y, x) in enumerate(all_coords):
            y_idx = int(y * (H - 1))
            x_idx = int(x * (W - 1))
            if 0 <= y_idx < H and 0 <= x_idx < W:
                blue_values_all.append(blue[y_idx, x_idx])
                green_values_all.append(green[y_idx, x_idx])
                valid_depths_all.append(all_depths[i])
                
        blue_values_all = np.array(blue_values_all)
        green_values_all = np.array(green_values_all)
        valid_depths_all = np.array(valid_depths_all)
        
        logger.info("Training full Random Forest model for product cartography...")
        # Full model uses 12 enhanced features (GEBCO + spatial coords + nearest soundings)
        gebco_2023 = np.load('intermediate_results/gebco_time_series.npy')[5]
        coords_all = np.array([c for i, c in enumerate(all_coords)
                               if 0 <= int(c[0]*(H-1)) < H and 0 <= int(c[1]*(W-1)) < W])
        r2_rf_all = classic_models.train_rf(blue_values_all, green_values_all, valid_depths_all,
                                            gebco_band=gebco_2023, coords=coords_all, use_enhanced=True)
        logger.info(f"Full Random Forest model trained successfully, R^2: {r2_rf_all:.4f}")
        
        os.makedirs('intermediate_results/model_params', exist_ok=True)
        np.save('intermediate_results/model_params/rf_params.npy', classic_models.rf_params)
        dump(classic_models.rf_model, 'intermediate_results/model_params/rf_model.joblib')
        
        # 3. Train Spatially Blocked 5-Fold Cross-Validation models
        for k in range(5):
            logger.info(f"Training spatial fold {k}/5 surrogate model...")
            train_mask = (fold_ids != k)
            train_coords = all_coords[train_mask]
            train_depths = all_depths[train_mask]
            
            blue_values_k = []
            green_values_k = []
            valid_depths_k = []
            for i, (y, x) in enumerate(train_coords):
                y_idx = int(y * (H - 1))
                x_idx = int(x * (W - 1))
                if 0 <= y_idx < H and 0 <= x_idx < W:
                    blue_values_k.append(blue[y_idx, x_idx])
                    green_values_k.append(green[y_idx, x_idx])
                    valid_depths_k.append(train_depths[i])
                    
            blue_values_k = np.array(blue_values_k)
            green_values_k = np.array(green_values_k)
            valid_depths_k = np.array(valid_depths_k)
            
            fold_classic = ClassicModels()
            # Enhanced feature training: GEBCO + coords + nearest soundings
            gebco_2023 = np.load('intermediate_results/gebco_time_series.npy')[5]  # 2023 GEBCO grid
            train_coords_valid = np.array([c for i, c in enumerate(train_coords)
                                           if 0 <= int(c[0]*(H-1)) < H and 0 <= int(c[1]*(W-1)) < W])
            r2_rf_k = fold_classic.train_rf(blue_values_k, green_values_k, valid_depths_k,
                                            gebco_band=gebco_2023, coords=train_coords_valid, use_enhanced=True)
            logger.info(f"Fold {k} Random Forest model trained successfully, R2: {r2_rf_k:.4f}")
            
            # Save fold-specific models and parameter metadata
            dump(fold_classic.rf_model, f'intermediate_results/model_params/rf_model_fold_{k}.joblib')
            np.save(f'intermediate_results/model_params/rf_params_fold_{k}.npy', fold_classic.rf_params)
            
        return classic_models
    except Exception as e:
        logger.error(f"Surrogate model training failed: {str(e)}")
        raise

def stage1_8_model_validation(sentinel_time_series, gebco_time_series=None):
    try:
        logger.info("Starting classic model validation...")
        classic_models = load_trained_classic_models()
        
        # 2. Create output directories
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
        
        aoi = [122.35, 30.62, 122.6, 30.8]
        preprocessor = DataPreprocessor(region=aoi)
        nautical_charts = preprocessor.get_sparse_points()

        # 4. Predict bathymetry across all years
        depths = []
        years = range(2018, 2024)
        for t, year in enumerate(years):
            blue = sentinel_time_series['blue'][t]
            green = sentinel_time_series['green'][t]
            depth = classic_models.predict_rf(blue, green)
            depths.append(depth)
            logger.info(f"Completed prediction for year {year}")
        
        # 5. Create and save annual bathymetric synthesis map
        chart_coords_rf = nautical_charts['coordinates'] # Get coordinates
        create_time_series_plot(depths, years, 'rf', output_dir, land_mask_array=land_mask_array, chart_coords=chart_coords_rf)
        
        # 6. Out-of-Fold (OOF) 5-Fold Spatial Cross-Validation Evaluation
        logger.info("Executing Enhanced Random Forest Out-of-Fold (OOF) Spatial CV Evaluation...")
        folds_path = 'intermediate_results/validation/spatial_folds.npz'
        if not os.path.exists(folds_path):
            raise FileNotFoundError(f"Spatial fold data not found: {folds_path}. Please run Stage 1.5 first.")
            
        folds_data = np.load(folds_path)
        all_coords = folds_data['coordinates']
        all_depths = folds_data['depths']
        fold_ids = folds_data['fold_ids']
        
        predicted_values_oof = []
        true_depths_oof = []
        point_ids_oof = []
        
        # Optical features for 2023 used for validation
        t_latest = -1
        blue_2023 = sentinel_time_series['blue'][t_latest]
        green_2023 = sentinel_time_series['green'][t_latest]
        H_rf, W_rf = blue_2023.shape
        
        for k in range(5):
            # Load Fold k model
            fold_classic = load_trained_classic_models_fold(k)
            fold_mask = (fold_ids == k)
            test_coords = all_coords[fold_mask]
            test_depths = all_depths[fold_mask]
            test_indices = np.where(fold_mask)[0]
            
            # Predict on Fold k test soundings
            fold_pred_map = fold_classic.predict_rf(blue_2023, green_2023, gebco_band=gebco_time_series[t_latest])
            
            for idx_in_test, (y, x) in enumerate(test_coords):
                y_idx = int(y * (H_rf - 1))
                x_idx = int(x * (W_rf - 1))
                if 0 <= y_idx < H_rf and 0 <= x_idx < W_rf:
                    pred_val = fold_pred_map[y_idx, x_idx]
                    if np.isfinite(pred_val) and pred_val > 0:
                        predicted_values_oof.append(pred_val)
                        true_depths_oof.append(test_depths[idx_in_test])
                        point_ids_oof.append(test_indices[idx_in_test])
                        
        predicted_values_oof = np.array(predicted_values_oof)
        true_depths_oof = np.array(true_depths_oof)
        point_ids_oof = np.array(point_ids_oof)
        
        # Save RF OOF prediction records
        os.makedirs('results/validation', exist_ok=True)
        oof_df_path = 'results/validation/oof_predictions_rf.csv'
        np.savetxt(
            oof_df_path,
            np.column_stack((point_ids_oof, true_depths_oof, predicted_values_oof)),
            header='point_id,observed_depth,predicted_depth',
            comments='',
            delimiter=','
        )
        logger.info(f"RF OOF predictions saved to: {oof_df_path}")
        
        # Compute OOF performance metrics
        if len(true_depths_oof) > 0 and len(predicted_values_oof) > 0:
            rmse = np.sqrt(np.mean((predicted_values_oof - true_depths_oof) ** 2))
            mae = np.mean(np.abs(predicted_values_oof - true_depths_oof))
            denom = np.sum((true_depths_oof - np.mean(true_depths_oof)) ** 2)
            r2 = 1 - np.sum((true_depths_oof - predicted_values_oof) ** 2) / denom if denom > 1e-9 else 0.0
        else:
            rmse, mae, r2 = np.nan, np.nan, np.nan
            logger.warning("No valid RF OOF evaluation pairs found.")
            
        logger.info("Enhanced RF Spatially Blocked Cross-Validation (OOF) Results:")
        logger.info(f"OOF RMSE: {rmse:.2f} m")
        logger.info(f"OOF MAE: {mae:.2f} m")
        logger.info(f"OOF R2: {r2:.4f}")
        logger.info(f"Evaluated Soundings Count: {len(true_depths_oof)}")
        
        # Plot RF OOF scatter validation figure
        FONTSIZE_MAIN_LABELS = 12
        FONTSIZE_MAIN_TICKS = 10
        FONTSIZE_TEXT_BOX = 10
        fig_scatter = plt.figure(figsize=(7, 5.5))
        ax_scatter = fig_scatter.add_subplot(1, 1, 1)
        if len(true_depths_oof) > 0 and len(predicted_values_oof) > 0 and np.isfinite(r2):
            scatter = ax_scatter.scatter(true_depths_oof, predicted_values_oof, alpha=0.6, s=30, label='OOF Samples')
            ideal_line, = ax_scatter.plot([0, 75], [0, 75], 'r--', linewidth=1.5, label='Ideal fit')
            ax_scatter.set_xlim(0, 75)
            ax_scatter.set_ylim(0, 75)
            text_str = f'OOF RMSE: {rmse:.2f} m\nOOF MAE: {mae:.2f} m\nOOF R2: {r2:.4f}'
            ax_scatter.text(0.95, 0.05, text_str,
                      transform=ax_scatter.transAxes, fontsize=FONTSIZE_TEXT_BOX,
                      verticalalignment='bottom', horizontalalignment='right',
                      bbox=dict(boxstyle='round,pad=0.5', fc='white', alpha=0.8))
            ax_scatter.legend(loc='upper left')
        else:
            ax_scatter.text(0.5, 0.5, 'No valid OOF data', ha='center', va='center')
            ax_scatter.set_xlim(0, 75)
            ax_scatter.set_ylim(0, 75)
            
        ax_scatter.set_xlabel('Measured Depth (m)', fontsize=FONTSIZE_MAIN_LABELS)
        ax_scatter.set_ylabel('Predicted Depth (m)', fontsize=FONTSIZE_MAIN_LABELS)
        ax_scatter.tick_params(axis='both', which='major', labelsize=FONTSIZE_MAIN_TICKS)
        ax_scatter.grid(True, linestyle='--', alpha=0.6)
        fig_scatter.tight_layout()
        save_path_scatter = os.path.join(output_dir, 'rf_validation_2023.jpg')
        plt.savefig(save_path_scatter, dpi=600, bbox_inches='tight')
        logger.info(f"RF OOF scatter plot saved to: {save_path_scatter}")
        plt.close(fig_scatter)

    except Exception as e:
        logger.error(f"Classic model validation failed: {str(e)}")
        raise

def create_time_series_plot(depths, years, model_name, output_dir, land_mask_array=None, chart_coords=None):
    """Create annual time-series composite figure."""
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

        def lon_formatter(x, pos): return f'{x:.1f}\u00b0E'
        def lat_formatter(y, pos): return f'{y:.1f}\u00b0N'
        
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

        # Add horizontal colorbar
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
        logger.error(f"Failed to create time-series synthesis figure: {str(e)}")
        raise

def stage2_s3gm_processing(system, sentinel_time_series, gebco_time_series, sentinel_years):
    """Stage 2: S3GM Spatio-Temporal Generative Diffusion Sampling."""
    try:
        # Create directories
        results_dir = 'results'
        models_dir = os.path.join(results_dir, 's3gm_pretrained_models')  
        time_series_dir = os.path.join(results_dir, 's3gm_time_series') 
        os.makedirs(results_dir, exist_ok=True)
        os.makedirs(models_dir, exist_ok=True)
        os.makedirs(time_series_dir, exist_ok=True)
        # Stage 2.1: S3GM Generative Pretraining
        pretrained_model_path = os.path.join(models_dir, 's3gm_pretrained.pth')
        if not os.path.exists(pretrained_model_path):
            logger.info("Executing Stage 2.1: S3GM Generative Pretraining Phase")
            
            # Load trained surrogate model
            classic_models = load_trained_classic_models()
            system.set_classic_models(classic_models)
            
            # Generating prior depth field with surrogate model
            classic_predictions = []
            for year in sentinel_years:
                blue = sentinel_time_series['blue'][year - 2018]
                green = sentinel_time_series['green'][year - 2018]
                depth = system.classic_models.predict_rf(blue, green, gebco_band=gebco_time_series[year - 2018])
                if np.isnan(depth).any() or np.isinf(depth).any():
                    logger.warning(f"{year}surrogate prediction contains invalid values")
                    depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
                classic_predictions.append(depth)
            
            # Stacking annual surrogate predictions into time-series tensor (T, H, W)
            classic_time_series = np.stack(classic_predictions)
            
            # Normalizing datasets for diffusion network
            preprocessor = DataPreprocessor()
            normalized_classic, classic_stats = preprocessor._normalize_data(classic_time_series, 'classic')
            normalized_gebco, gebco_stats = preprocessor._normalize_data(gebco_time_series, 'gebco')
            
            # Saving normalization parameters for post-processing
            np.save('intermediate_results/classic_stats.npy', classic_stats)
            np.save('intermediate_results/gebco_stats.npy', gebco_stats)
            
            # Save normalized data arrays
            np.save('intermediate_results/classic_normalized.npy', normalized_classic)
            np.save('intermediate_results/gebco_normalized.npy', normalized_gebco)
            logger.info("Saved normalized surrogate predictions and GEBCO grids")
            
            pretrain_data = {
                'classic': normalized_classic,  # Normalized scale
                'gebco': normalized_gebco,     # Normalized scale
            }
            
            # Running pretraining (model checkpoint saved automatically)
            system.s3gm.pretrain(
                classic_data=pretrain_data['classic'],
                gebco_data=pretrain_data['gebco'],
                save_path=pretrained_model_path
            )
        else:
            logger.info("Loading existing pretrained S3GM model checkpoint")
            system.s3gm.load_pretrained(pretrained_model_path)
        
        # Log range adaptation configuration
        logger.info("Model configuration:")
        logger.info(f"  - Range adaptation: {system.s3gm.config.range_adaptation['enabled']}")
        logger.info(f"  - Mixed activation: {system.s3gm.config.range_adaptation['use_mixed_activation']}")
        logger.info(f"  - Land flag: {system.s3gm.config.range_adaptation['land_value']}")
        
        # Validate model forward output before Stage 2.2
        logger.info("Validating model output...")
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
            # Extract primary model output
            test_output = test_output[0]
            # Verify non-zero output
            if torch.all(torch.eq(test_output, torch.zeros_like(test_output))):
                raise ValueError("Pretrained model output is all zeros; re-training required")
            logger.info(f"Model test output range: [{test_output.min().item():.4f}, {test_output.max().item():.4f}]")
            
        # Stage 2.2: Conditional Posterior Sampling
        logger.info("Executing Stage 2.2: S3GM Conditional Posterior Sampling (DPS)")
        # Loading 2023 sounding measurements
        aoi = [122.35, 30.62, 122.6, 30.8]
        preprocessor = DataPreprocessor(region=aoi)
        nautical_charts = preprocessor.get_sparse_points()
        
        # Normalizing sounding measurements
        normalized_depths, chart_stats = preprocessor._normalize_data(
            nautical_charts['depths'], 
            data_type='chart'
        )
        
        # Save chart sounding normalization statistics
        os.makedirs('intermediate_results', exist_ok=True)
        np.save('intermediate_results/chart_stats.npy', chart_stats)
        logger.info(f"Sounding normalization parameters saved: {chart_stats}")
        
        # Updating normalized sounding dictionary
        nautical_charts_normalized = {
            'depths': normalized_depths,
            'coordinates': nautical_charts['coordinates']
        }
        
        # Normalizing GEBCO bathymetric grids
        normalized_gebco, gebco_stats = preprocessor._normalize_data(gebco_time_series, 'gebco')
        np.save('intermediate_results/gebco_normalized.npy', normalized_gebco)
        np.save('intermediate_results/gebco_stats.npy', gebco_stats)

        # 2.2.1 Full cartographic production sampling (skipped in diagnostic mode)
        logger.info("Diagnostic mode: skipping full-dataset reconstruction sampling...")
        
        # 2.2.2 5-Fold Cross-Validation conditional sampling (Fold 0 diagnostic)
        for k in range(1):
            logger.info(f"Executing Fold {k}/5 conditional sampling...")
            
            # Generate surrogate predictions with Fold k model
            fold_classic = load_trained_classic_models_fold(k)
            classic_predictions_k = []
            for year in sentinel_years:
                blue = sentinel_time_series['blue'][year - 2018]
                green = sentinel_time_series['green'][year - 2018]
                depth = fold_classic.predict_rf(blue, green, gebco_band=gebco_time_series[year - 2018])
                if np.isnan(depth).any() or np.isinf(depth).any():
                    depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
                classic_predictions_k.append(depth)
            classic_time_series_k = np.stack(classic_predictions_k)
            
            normalized_classic_k, classic_stats_k = preprocessor._normalize_data(classic_time_series_k, 'classic')
            np.save(f'intermediate_results/classic_normalized_fold_{k}.npy', normalized_classic_k)
            np.save(f'intermediate_results/classic_stats_fold_{k}.npy', classic_stats_k)
            
            # Conditional sampling with holdout test points masked (fold_id=k)
            results_k = system.s3gm.conditional_sampling(
                measurements=nautical_charts_normalized['depths'],
                measurement_coordinates=nautical_charts_normalized['coordinates'],
                years=sentinel_years,
                classic_data=normalized_classic_k,
                gebco_data=normalized_gebco,
                fold_id=k
            )
            
            if isinstance(results_k, torch.Tensor):
                results_k = results_k.cpu().numpy()
                
            fold_output_dir = os.path.join(results_dir, 's3gm_time_series', f'fold_{k}')
            save_results(
                depths=results_k,
                years=sentinel_years,
                output_dir=fold_output_dir,
                chart_stats=chart_stats
            )
            logger.info(f"Fold {k} conditional sampling completed, results saved to {fold_output_dir}!")
        
    except Exception as e:
        logger.error(f"Stage 2 processing failed: {str(e)}")
        raise

def stage3_postprocessing():
    """Stage 3: Post-processing and Spatial Uncertainty Visualization."""
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

        aoi = [122.35, 30.62, 122.6, 30.8]
        preprocessor = DataPreprocessor(region=aoi)
        nautical_charts = preprocessor.get_sparse_points()

        chart_coords_s3gm = nautical_charts['coordinates'] # Get coordinates
        # Create time-series composite plot, passing the mask and chart coords
        create_time_series_plot(depths, years, 's3gm', output_dir, land_mask_array, chart_coords=chart_coords_s3gm)

        # Validation part (Now starts here - Out-of-Fold 5-Fold Spatial CV Validation)
        logger.info("Executing S3GM Out-of-Fold (OOF) Spatial CV Evaluation...")
        folds_path = 'intermediate_results/validation/spatial_folds.npz'
        if not os.path.exists(folds_path):
            raise FileNotFoundError(f"Spatial fold data not found: {folds_path}. Please run Stage 1.5 first.")
            
        folds_data = np.load(folds_path)
        all_coords = folds_data['coordinates']
        all_depths = folds_data['depths']
        fold_ids = folds_data['fold_ids']
        
        predicted_values_oof = []
        true_depths_oof = []
        point_ids_oof = []
        
        # Load fold predictions and extract holdout test soundings
        for k in range(5):
            fold_pred_path = f'results/s3gm_time_series/fold_{k}/bathymetry_2023.npy'
            if not os.path.exists(fold_pred_path):
                logger.warning(f"Fold {k} prediction file not found: {fold_pred_path}, skipping fold")
                continue
                
            fold_pred_map = np.load(fold_pred_path)
            fold_mask = (fold_ids == k)
            test_coords = all_coords[fold_mask]
            test_depths = all_depths[fold_mask]
            test_indices = np.where(fold_mask)[0]
            
            for idx_in_test, (y, x) in enumerate(test_coords):
                y_idx = int(y * (H - 1))
                x_idx = int(x * (W - 1))
                if 0 <= y_idx < H and 0 <= x_idx < W:
                    pred_val = fold_pred_map[y_idx, x_idx]
                    if np.isfinite(pred_val) and pred_val > 0:
                        predicted_values_oof.append(pred_val)
                        true_depths_oof.append(test_depths[idx_in_test])
                        point_ids_oof.append(test_indices[idx_in_test])
                        
        predicted_values_oof = np.array(predicted_values_oof)
        true_depths_oof = np.array(true_depths_oof)
        point_ids_oof = np.array(point_ids_oof)
        
        # Save OOF predictions for statistical comparison
        os.makedirs('results/validation', exist_ok=True)
        oof_df_path = 'results/validation/oof_predictions_s3gm.csv'
        np.savetxt(
            oof_df_path,
            np.column_stack((point_ids_oof, true_depths_oof, predicted_values_oof)),
            header='point_id,observed_depth,predicted_depth',
            comments='',
            delimiter=','
        )
        logger.info(f"S3GM OOF prediction results saved to: {oof_df_path}")
        
        if len(true_depths_oof) > 0 and len(predicted_values_oof) > 0:
            rmse = np.sqrt(np.mean((predicted_values_oof - true_depths_oof) ** 2))
            mae = np.mean(np.abs(predicted_values_oof - true_depths_oof))
            denom = np.sum((true_depths_oof - np.mean(true_depths_oof)) ** 2)
            r2 = 1 - np.sum((true_depths_oof - predicted_values_oof) ** 2) / denom if denom != 0 else -np.inf
        else:
            rmse, mae, r2 = np.nan, np.nan, np.nan
            logger.warning("No valid OOF evaluation pairs found.")
            
        logger.info("S3GM Spatially Blocked CV (OOF) Results:")
        logger.info(f"OOF RMSE: {rmse:.2f} m" if np.isfinite(rmse) else "OOF RMSE: N/A")
        logger.info(f"OOF MAE: {mae:.2f} m" if np.isfinite(mae) else "OOF MAE: N/A")
        logger.info(f"OOF R2: {r2:.4f}" if np.isfinite(r2) else "OOF R2: N/A")
        logger.info(f"Independent Evaluated Soundings Count: {len(true_depths_oof)}")
        
        # Plot OOF independent validation scatter plot
        FONTSIZE_MAIN_LABELS = 12
        FONTSIZE_MAIN_TICKS = 10
        FONTSIZE_TEXT_BOX = 10
        fig_scatter = plt.figure(figsize=(7, 5.5))
        ax_scatter = fig_scatter.add_subplot(1, 1, 1)
        if len(true_depths_oof) > 0 and len(predicted_values_oof) > 0 and np.isfinite(r2):
            scatter = ax_scatter.scatter(true_depths_oof, predicted_values_oof, alpha=0.6, s=30, label='OOF Samples')
            ideal_line, = ax_scatter.plot([0, 75], [0, 75], 'r--', linewidth=1.5, label='Ideal fit')
            ax_scatter.set_xlim(0, 75)
            ax_scatter.set_ylim(0, 75)
            text_str = f'OOF RMSE: {rmse:.2f} m\nOOF MAE: {mae:.2f} m\nOOF R2: {r2:.4f}'
            ax_scatter.text(0.95, 0.05, text_str,
                      transform=ax_scatter.transAxes, fontsize=FONTSIZE_TEXT_BOX,
                      verticalalignment='bottom', horizontalalignment='right',
                      bbox=dict(boxstyle='round,pad=0.5', fc='white', alpha=0.8))
            ax_scatter.legend(loc='upper left')
        else:
            ax_scatter.text(0.5, 0.5, 'No valid OOF data', ha='center', va='center')
            ax_scatter.set_xlim(0, 75)
            ax_scatter.set_ylim(0, 75)
            
        ax_scatter.set_xlabel('Measured Depth (m)', fontsize=FONTSIZE_MAIN_LABELS)
        ax_scatter.set_ylabel('Predicted Depth (m)', fontsize=FONTSIZE_MAIN_LABELS)
        ax_scatter.tick_params(axis='both', which='major', labelsize=FONTSIZE_MAIN_TICKS)
        ax_scatter.grid(True, linestyle='--', alpha=0.6)
        fig_scatter.tight_layout()
        save_path_scatter = os.path.join(output_dir, 's3gm_validation_2023.jpg')
        plt.savefig(save_path_scatter, dpi=600, bbox_inches='tight')
        logger.info(f"S3GM OOF scatter plot saved to: {save_path_scatter}")
        plt.close(fig_scatter)
        # --- End of S3GM Scatter Plot ---

        def lon_formatter(x, pos): return f'{x:.1f}\u00b0E'
        def lat_formatter(y, pos): return f'{y:.1f}\u00b0N'

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
        logger.error(f"Stage 3 post-processing analysis failed: {str(e)}")
        raise

def load_trained_classic_models():
    """Load trained full surrogate model and parameters."""
    try:
        classic_models = ClassicModels()
        
        # Load parameters
        rf_params = np.load('intermediate_results/model_params/rf_params.npy', allow_pickle=True).item()
        classic_models.rf_params = rf_params
        
        # Load Random Forest model
        classic_models.rf_model = load('intermediate_results/model_params/rf_model.joblib')
        
        return classic_models
        
    except Exception as e:
        logger.error(f"Failed to load surrogate model: {str(e)}")
        raise

def load_trained_classic_models_fold(fold_id):
    """Load fold-specific surrogate model and parameters."""
    try:
        classic_models = ClassicModels()
        rf_params = np.load(f'intermediate_results/model_params/rf_params_fold_{fold_id}.npy', allow_pickle=True).item()
        classic_models.rf_params = rf_params
        classic_models.rf_model = load(f'intermediate_results/model_params/rf_model_fold_{fold_id}.joblib')
        return classic_models
    except Exception as e:
        logger.error(f"Failed to load Fold {fold_id} surrogate model: {str(e)}")
        raise

def denormalize_bathymetry(normalized_data: np.ndarray, stats: Dict[str, float]) -> np.ndarray:
    """Denormalize bathymetry data to physical meters and enforce physical constraints."""
    try:
        # Extract physical depth range and special value flags
        min_phys = stats.get('min_phys', 0.0)
        max_phys = stats.get('max_phys', 90.0)
        land_value_norm = stats.get('land_value', 1.5)
        eps = 1e-6

        # Identify land pixels with tolerance
        land_mask = np.isclose(normalized_data, land_value_norm)
        
        # Check data range
        valid_data = normalized_data[~land_mask & ~np.isnan(normalized_data)]
        if len(valid_data) > 0:
            actual_min, actual_max = valid_data.min(), valid_data.max()
            logger.info(f"Pre-denormalization valid range: [{actual_min:.4f}, {actual_max:.4f}]")
            # Check for values outside [-1, 1]
            if actual_min < -1.0 - eps or actual_max > 1.0 + eps:
                 logger.warning(f"  Note: Valid data range exceeds expected [-1, 1] interval.")
        else:
            logger.warning("No valid marine data found prior to denormalization.")

        # Denormalization: phys = ((norm + 1) / 2) * (max_phys - min_phys) + min_phys
        depth_denorm = ((normalized_data + 1) / 2) * (max_phys - min_phys) + min_phys

        # Set land pixels to 0m (or NaN)
        depth_denorm = np.where(land_mask, 0.0, depth_denorm)
        
        # Enforce non-negative physical depth constraint
        sea_mask = ~land_mask
        if np.any(depth_denorm[sea_mask] < 0):
            neg_count = np.sum(depth_denorm[sea_mask] < 0)
            total_count = np.sum(sea_mask)
            logger.warning(f"Negative depths detected across marine pixels: {(neg_count/total_count)*100:.2f}% (clipped to zero)")
            logger.warning(f"Negative value range: [{depth_denorm[sea_mask & (depth_denorm < 0)].min():.2f}, 0) m")
            depth_denorm[sea_mask] = np.maximum(depth_denorm[sea_mask], 0.0)
        
        # Handle NaN values
        depth_denorm = np.nan_to_num(depth_denorm, nan=0.0)  # Map NaNs to 0m
        
        # Final range check and logging
        if np.any(sea_mask):
            sea_depths = depth_denorm[sea_mask]
            logger.info(f"Post-denormalization marine depth range: [{sea_depths.min():.2f}, {sea_depths.max():.2f}]")
        else:
            logger.warning("No marine pixels found after denormalization.")
            
        return depth_denorm
        
    except Exception as e:
        logger.error(f"Denormalization failed: {str(e)}")
        raise

def save_results(depths, years, output_dir, chart_stats):
    """Save denormalized bathymetry prediction outputs.
    
    Args:
        depths: Tensor of shape [1, T, C, H, W]
            - 1: batch size
            - T: Number of time frames (years)
            - C: Number of channels
            - H, W: Height and width
        output_dir: Destination directory
        chart_stats: Sounding normalization statistics
    """
    try:
        os.makedirs(output_dir, exist_ok=True)
        
        # Denormalize and save annual prediction rasters
        for t, year in enumerate(years):
            # Extract Channel 2 (depth channel)
            depth_data = depths[0, t, 2]  # Shape [H, W]
            
            # Inspect data pre-denormalization
            logger.info(f"{year}Pre-denormalization stats:")
            logger.info(f"- Data range: [{depth_data.min():.4f}, {depth_data.max():.4f}]")
            logger.info(f"- Non-zero distribution: {np.percentile(depth_data[depth_data != 0], [25, 50, 75])}")
            
            # Denormalize using chart statistics
            depth_denorm = denormalize_bathymetry(depth_data, chart_stats)
            
            # Inspect post-denormalization data
            logger.info(f"{year}Post-denormalization stats:")
            logger.info(f"- Data range: [{depth_denorm.min():.4f}, {depth_denorm.max():.4f}]")
            logger.info(f"- Valid value distribution: {np.percentile(depth_denorm[depth_denorm > 0], [25, 50, 75])}")
            
            # Save denormalized raster files
            output_path = os.path.join(output_dir, f'bathymetry_{year}.npy')
            np.save(output_path, depth_denorm)
            
            logger.info(f"Saved{year}prediction output: {output_path}")
            logger.info(f"{year}depth range: [{depth_denorm.min():.2f}, {depth_denorm.max():.2f}] m")
            logger.info(f"{year}land pixel ratio: {np.mean(np.abs(depth_data - 1.5) < 0.1):.2%}")
            
    except Exception as e:
        logger.error(f"Failed to save results: {str(e)}")
        raise

def stage4_statistical_analysis():
    """Stage 4: Statistical Significance Analysis."""
    try:
        logger.info("Executing Stage 4: Statistical Significance Testing")

        # 1. Load ground truth sounding measurements
        logger.info("Loading ground truth sounding measurements...")
        aoi = [122.35, 30.62, 122.6, 30.8]
        preprocessor = DataPreprocessor(region=aoi)
        nautical_charts = preprocessor.get_sparse_points()
        true_depths_all = np.abs(nautical_charts['depths'])
        coords = nautical_charts['coordinates']
        logger.info(f"Successfully loaded {len(true_depths_all)} sounding points")

        # 2. Load 2023 prediction maps for surrogate and S3GM
        logger.info("Loading 2023 prediction maps for RF surrogate and S3GM...")
        rf_pred_path = 'results/classic_models/rf_2023.npy'
        s3gm_pred_path = 'results/s3gm_time_series/bathymetry_2023.npy'

        if not os.path.exists(rf_pred_path):
            logger.error(f"RF prediction file not found: {rf_pred_path}. Please run Stage 1.8 first.")
            return
        if not os.path.exists(s3gm_pred_path):
            logger.error(f"S3GM prediction file not found: {s3gm_pred_path}. Please run Stages 2 and 3 first.")
            return
            
        rf_preds_map = np.load(rf_pred_path)
        s3gm_preds_map = np.load(s3gm_pred_path)
        logger.info("Prediction rasters loaded successfully")

        # 3. Align prediction rasters with sounding coordinates
        H, W = rf_preds_map.shape
        rf_preds_aligned = []
        s3gm_preds_aligned = []
        true_depths_aligned = []

        for i, (y, x) in enumerate(coords):
            y_idx, x_idx = int(y * (H - 1)), int(x * (W - 1))
            if 0 <= y_idx < H and 0 <= x_idx < W:
                rf_val = rf_preds_map[y_idx, x_idx]
                s3gm_val = s3gm_preds_map[y_idx, x_idx]
                
                # Retain points where all predictions and observations are valid
                if np.isfinite(rf_val) and np.isfinite(s3gm_val) and rf_val > 0 and s3gm_val > 0:
                    rf_preds_aligned.append(rf_val)
                    s3gm_preds_aligned.append(s3gm_val)
                    true_depths_aligned.append(true_depths_all[i])

        rf_preds_aligned = np.array(rf_preds_aligned)
        s3gm_preds_aligned = np.array(s3gm_preds_aligned)
        true_depths_aligned = np.array(true_depths_aligned)
        logger.info(f"Aligned valid soundings for statistical testing: {len(true_depths_aligned)}")

        # 4. Compute absolute errors
        rf_abs_errors = np.abs(rf_preds_aligned - true_depths_aligned)
        s3gm_abs_errors = np.abs(s3gm_preds_aligned - true_depths_aligned)

        # 5. Execute Wilcoxon signed-rank test
        # H0: Median of paired differences is zero
        # H1: RF error > S3GM error (S3GM error is smaller)
        logger.info("Computing block-level Wilcoxon signed-rank test...")
        stat, p_value = wilcoxon(rf_abs_errors, s3gm_abs_errors, alternative='greater')
        logger.info("--- Wilcoxon Signed-Rank Test Results ---")
        logger.info(f"Statistic: {stat:.4f}")
        logger.info(f"P-value: {p_value:.6f}")
        if p_value < 0.05:
            logger.info("Interpretation: p < 0.05. Statistically significant error reduction.")
        else:
            logger.info("Interpretation: p >= 0.05. Differences not statistically significant.")

        # 6. Compute 95% bootstrap confidence interval for MAE improvement
        logger.info("\nComputing 95% Bootstrap Confidence Interval for MAE delta...")
        diff_mae = rf_abs_errors - s3gm_abs_errors # MAE_RF - MAE_S3GM
        n_iterations = 10000
        bootstrap_means = []
        for _ in range(n_iterations):
            sample_indices = np.random.choice(len(diff_mae), size=len(diff_mae), replace=True)
            bootstrap_means.append(np.mean(diff_mae[sample_indices]))
        
        confidence_interval = np.percentile(bootstrap_means, [2.5, 97.5])
        mean_improvement = np.mean(diff_mae)
        logger.info("--- 95% Confidence Interval for MAE Improvement (MAE_RF - MAE_S3GM) ---")
        logger.info(f"Mean MAE improvement: {mean_improvement:.4f} m")
        logger.info(f"95% Confidence Interval: [{confidence_interval[0]:.4f}, {confidence_interval[1]:.4f}] m")
        if confidence_interval[0] > 0:
            logger.info("Interpretation: CI is strictly positive, confirming robust precision improvement.")
        else:
            logger.info("Interpretation: CI includes zero.")

    except Exception as e:
        logger.error(f"Stage 4 statistical analysis failed: {str(e)}")
        raise

def stage5_in_depth_analysis():
    """Stage 5: Detailed Performance Stratification Across Depth and Terrain Slope Zones."""
    try:
        logger.info("Executing Stage 5: Zoning and Terrain Stratification Analysis")

        # 1. Load ground truth soundings and predictions
        logger.info("Loading soundings, RF, and S3GM 2023 outputs...")
        aoi = [122.35, 30.62, 122.6, 30.8]
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
        
        # 2. Align data points
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
        
        logger.info(f"Aligned valid soundings for stratification analysis: {len(aligned_data)}")

        # 3. Stratify performance across depth zones
        logger.info("\n--- Performance Stratification Across Depth Zones ---")
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

        # 4. Stratify performance across seabed slopes
        logger.info("\n--- Performance Stratification Across Seabed Slopes ---")
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
        logger.error(f"Stage 5 performance analysis failed: {str(e)}")
        raise

def main():
    try:
        logger.info("Starting BathySurrogate computational pipeline...")
        
        args = parse_args()
        if not args.stage:
            logger.error("Please specify the --stage parameter.")
            return
            
        start_time = time.time()
        
        # Initialize basic study area parameters
        aoi = [122.35, 30.62, 122.6, 30.8]
        gebco_years = range(2019, 2025)
        sentinel_years = range(2018, 2024)
        
        if args.stage == '1':
            logger.info("Executing Stage 1: Data Preprocessing")
            stage1_preprocessing(aoi, gebco_years, sentinel_years)
            
        elif args.stage == '1.5':
            logger.info("Executing Stage 1.5: Surrogate Model Training")
            sentinel_time_series = np.load('intermediate_results/sentinel_time_series.npy', allow_pickle=True).item()
            classic_models = stage1_5_model_training(sentinel_time_series)
        
        elif args.stage == '1.8':
            logger.info("Executing Stage 1.8: Surrogate Model Validation")
            sentinel_time_series = np.load('intermediate_results/sentinel_time_series.npy', allow_pickle=True).item()
            gebco_time_series = np.load('intermediate_results/gebco_time_series.npy')
            stage1_8_model_validation(sentinel_time_series, gebco_time_series)
                
        elif args.stage == '2':
            logger.info("Executing Stage 2: S3GM Diffusion Sampling")
            # Load preprocessed data and trained model weights
            sentinel_time_series = np.load('intermediate_results/sentinel_time_series.npy', allow_pickle=True).item()
            gebco_time_series = np.load('intermediate_results/gebco_time_series.npy')
            
            # Initialize Hybrid Bathymetry System
            system = HybridBathymetrySystem(
                region=aoi,
                time_range={'gebco': gebco_years, 'sentinel': sentinel_years}
            )
            
            # Execute Stage 2
            stage2_s3gm_processing(system, sentinel_time_series, gebco_time_series, sentinel_years)

        elif args.stage == '3':
            logger.info("Executing Stage 3: Post-processing and Visualization")
            stage3_postprocessing()

        elif args.stage == '4':
            logger.info("Executing Stage 4: Statistical Significance Analysis")
            stage4_statistical_analysis()

        elif args.stage == '5':
            logger.info("Executing Stage 5: Zoning Performance Analysis")
            stage5_in_depth_analysis()

        elapsed = time.time() - start_time
        logger.info(f"Stage {args.stage} completed. Elapsed time: {elapsed:.2f} s ({elapsed/60.0:.2f} min).")
        
        os.makedirs('results/validation', exist_ok=True)
        with open('results/validation/execution_benchmarks.txt', 'a', encoding='utf-8') as f:
            f.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Stage {args.stage}: {elapsed:.2f} s ({elapsed/60.0:.2f} min)\n")
    except Exception as e:
        logger.error(f"Pipeline error: {str(e)}")
        raise
if __name__ == '__main__':
    setup_logging()  # Initialize logging once
    main()

# Run Stage 1 (Data Preprocessing)
# python run_bathymetry.py --stage 1

# Run Stage 1.5 (Surrogate Model Training)
# python run_bathymetry.py --stage 1.5

# Run Stage 1.8 (Surrogate Model Validation)
# python run_bathymetry.py --stage 1.8

# Run Stage 2 (S3GM Diffusion Sampling)
# python run_bathymetry.py --stage 2

# Run Stage 3 (Post-processing & Visualization)
# python run_bathymetry.py --stage 3

# Run Stage 4 (Statistical Significance Testing)
# python run_bathymetry.py --stage 4

# Run Stage 5 (Zoning Performance Analysis)
# python run_bathymetry.py --stage 5