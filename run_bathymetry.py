import sys
import os
import logging
import argparse
import numpy as np
import ee
ee.Authenticate()
ee.Initialize(project='fast-banner-452901-c8')
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

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def setup_logging():
    """配置日志系统"""
    if len(logging.getLogger().handlers) > 0:
        return
        
    log_dir = 'logging'
    os.makedirs(log_dir, exist_ok=True)
    
    # 默认使用 INFO 级别，需要详细初始化信息时可改为 DEBUG
    logging.basicConfig(
        level=logging.INFO,  # 或 logging.DEBUG 查看详细信息
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='水深测量系统处理程序')
    parser.add_argument(
        '--stage',
        type=str,
        choices=['1', '1.5', '1.8','2', '3'],
        help='处理阶段：1=数据预处理，1.5=经典模型训练，1.8=经典模型验证，2=S3GM模型处理，3=结果后处理与可视化'
    )
    return parser.parse_args()

def stage1_preprocessing(aoi, gebco_years, sentinel_years):
    """第一阶段：数据预处理"""
    try:
        # 创建时序数据存储结构
        sentinel_time_series = {
            'blue': np.zeros((6, 64, 64)),  # [T, H, W]
            'green': np.zeros((6, 64, 64))
        }
        gebco_time_series = np.zeros((6, 64, 64))  # [T, H, W]
        
        # 处理每年的数据
        for idx, (gebco_year, sentinel_year) in enumerate(zip(gebco_years, sentinel_years)):
            logger.info(f"处理 {gebco_year} 年发布的 GEBCO 数据（对应 {sentinel_year} 年的观测）...")
            
            # GEBCO数据处理
            gebco_file = os.path.join(
                'GEBCO_Bathymetry',
                'GEBCO_26_Dec_2024_86912dfafafa',
                f'gebco_{gebco_year}_n30.8_s30.62_w122.35_e122.6.nc'
            )
            depth_data = load_gebco_data(gebco_file)
            if depth_data.shape != (64, 64):
                # 创建目标网格
                target_h, target_w = 64, 64
                orig_h, orig_w = depth_data.shape
    
                # 创建源和目标坐标网格
                y_orig = np.linspace(0, 1, orig_h)
                x_orig = np.linspace(0, 1, orig_w)
                y_target = np.linspace(0, 1, target_h)
                x_target = np.linspace(0, 1, target_w)
    
                # 使用双线性插值进行重采样
                interpolator = RegularGridInterpolator(
                    (y_orig, x_orig), 
                    depth_data,
                    method='linear',
                    bounds_error=False,
                    fill_value=np.nan
                )
    
                # 创建目标坐标点
                xx, yy = np.meshgrid(x_target, y_target)
                points = np.stack([yy.ravel(), xx.ravel()], axis=1)
    
                # 执行插值
                depth_data = interpolator(points).reshape(target_h, target_w)
    
                logger.info(f"已将GEBCO数据重采样至目标尺寸: {depth_data.shape}")
            
            gebco_time_series[idx] = depth_data
            logger.info(f"GEBCO数据范围 (重采样后): [{depth_data.min():.4f}, {depth_data.max():.4f}]")
            
            # Sentinel-2数据处理
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
                """获取指定波段的数组数据"""
                try:
                    # 获取实际可用的波段名称
                    available_bands = image.bandNames().getInfo()
                    logger.info(f"可用波段: {available_bands}")
                    
                    # 首先将图像重采样到目标分辨率
                    target_scale = math.sqrt(aoi.area().getInfo() / (64 * 64))  # 计算目标分辨率
                    resampled_image = image.resample('bilinear').reproject(
                        crs=image.projection(),
                        scale=target_scale
                    )
                    
                    # 选择波段（使用列表格式）
                    band_data = resampled_image.select([band_name])
                    
                    # 获取数据
                    data = band_data.reduceRegion(
                        reducer=ee.Reducer.toList(),
                        geometry=aoi,
                        scale=target_scale,
                        maxPixels=1e13
                    ).get(band_name).getInfo()
                    
                    if not data:
                        logger.error(f"{band_name}波段数据获取失败")
                        return None
                        
                    # 转换为numpy数组并重塑
                    temp_grid = np.array(data)
                    logger.info(f"获取的数据大小: {temp_grid.size}")
                    
                    # 计算最接近的方形网格尺寸
                    grid_size = int(np.sqrt(temp_grid.size))
                    temp_grid = temp_grid[:grid_size*grid_size].reshape(grid_size, grid_size)
                    
                    # 使用scipy.ndimage进行精确重采样到64x64
                    resampled_grid = zoom(temp_grid, (64/grid_size, 64/grid_size), order=1)
                    logger.info(f"重采样后的图像尺寸: {resampled_grid.shape}")
                    
                    return resampled_grid
                    
                except Exception as e:
                    logger.error(f"获取{band_name}波段数据时发生错误: {str(e)}")
                    return None
            
            # 确保processed_image是ee.Image类型并包含必要的波段
            processed_image = ee.Image(processed_image)
            available_bands = processed_image.bandNames().getInfo()
            logger.info(f"MIWC处理后的可用波段: {available_bands}")

            if not all(band in available_bands for band in ['blue', 'green']):
                logger.warning("MIWC处理后缺少必要的波段")
                continue

            # 使用列表格式获取波段数据
            blue_array = get_band_array(processed_image, 'blue', aoi)
            green_array = get_band_array(processed_image, 'green', aoi)
            
            # 数据验证
            if blue_array is None or green_array is None:
                logger.error("无法获取有效的波段数据")
                continue
                
            if blue_array.shape != green_array.shape:
                logger.error(f"波段形状不一致: blue={blue_array.shape}, green={green_array.shape}")
                continue
                
            if np.all(np.isnan(blue_array)) or np.all(np.isnan(green_array)):
                logger.error("波段数据全为NaN")
                continue
            
            sentinel_time_series['blue'][idx] = blue_array
            sentinel_time_series['green'][idx] = green_array
            logger.info(f"Sentinel-2 blue波段范围 (重采样后): [{blue_array.min():.4f}, {blue_array.max():.4f}]")
            logger.info(f"Sentinel-2 green波段范围 (重采样后): [{green_array.min():.4f}, {green_array.max():.4f}]")
            
            logger.info(f"完成 {sentinel_year} 年数据的预处理和存储")
            
        # 保存预处理结果
        output_dir = 'intermediate_results'
        os.makedirs(output_dir, exist_ok=True)
        np.save(os.path.join(output_dir, 'sentinel_time_series.npy'), sentinel_time_series)
        np.save(os.path.join(output_dir, 'gebco_time_series.npy'), gebco_time_series)
        
        return sentinel_time_series, gebco_time_series
        
    except Exception as e:
        logger.error(f"第一阶段处理失败: {str(e)}")
        raise

def stage1_5_model_training(sentinel_time_series):
    """第1.5阶段：训练经典模型"""
    try:
        logger.info("开始训练经典模型...")
        classic_models = ClassicModels()
        
        # 初始化预处理器获取海图数据
        aoi = ee.Geometry.Rectangle([122.35, 30.62, 122.6, 30.8])
        preprocessor = DataPreprocessor(region=aoi)
        nautical_charts = preprocessor.get_sparse_points()
        
        # 使用原始正值深度，并应用深度范围限制
        depths = np.abs(nautical_charts['depths'])
        valid_depth_mask = (depths >= 0.1) & (depths <= 75.0)
        
        if not np.any(valid_depth_mask):
            raise ValueError("没有在有效深度范围内的数据点")
            
        nautical_charts['depths'] = depths[valid_depth_mask]
        nautical_charts['coordinates'] = nautical_charts['coordinates'][valid_depth_mask]
        
        logger.info(f"原始深度范围: {depths.min():.2f} 到 {depths.max():.2f} 米")
        logger.info(f"有效深度范围: {nautical_charts['depths'].min():.2f} 到 {nautical_charts['depths'].max():.2f} 米")
        logger.info(f"有效深度点数: {np.sum(valid_depth_mask)}")
        
        # 使用最新年份(2023)的遥感数据
        t = -1  # 最后一个时间点
        blue = sentinel_time_series['blue'][t]
        green = sentinel_time_series['green'][t]
        
        # 提取海图位置对应的波段值
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
        
        # 训练随机森林模型
        r2_rf = classic_models.train_rf(blue_values, green_values, valid_depths)
        logger.info(f"随机森林模型训练完成，R²: {r2_rf:.4f}")
        
        # 保存模型参数和模型本身
        os.makedirs('intermediate_results/model_params', exist_ok=True)
        np.save('intermediate_results/model_params/rf_params.npy', classic_models.rf_params)
        dump(classic_models.rf_model, 'intermediate_results/model_params/rf_model.joblib')
        
        return classic_models
        
    except Exception as e:
        logger.error(f"经典模型训练失败: {str(e)}")
        raise

def stage1_8_model_validation(sentinel_time_series):
    """第1.8阶段：经典模型验证"""
    try:
        logger.info("Starting classic model validation...")
        classic_models = load_trained_classic_models()
        
        # 2. 创建输出目录
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
                 ee.Initialize(project='fast-banner-452901-c8')
        except ImportError:
             logger.error("Google Earth Engine Python API (ee) not found. Cannot get nautical charts.")
             raise
        except Exception as e:
             logger.error(f"GEE initialization failed: {e}")
             raise
        aoi = ee.Geometry.Rectangle([122.35, 30.62, 122.6, 30.8])
        preprocessor = DataPreprocessor(region=aoi)
        nautical_charts = preprocessor.get_sparse_points()

        # 4. 预测所有年份
        depths = []
        years = range(2018, 2024)
        for t, year in enumerate(years):
            blue = sentinel_time_series['blue'][t]
            green = sentinel_time_series['green'][t]
            depth = classic_models.predict_rf(blue, green)
            depths.append(depth)
            logger.info(f"Completed prediction for year {year}")
        
        # 5. 创建并保存时序合成图 (pass land mask and chart coords)
        chart_coords_rf = nautical_charts['coordinates'] # Get coordinates
        create_time_series_plot(depths, years, 'rf', output_dir, land_mask_array=land_mask_array, chart_coords=chart_coords_rf)
        
        # 6. 2023年预测结果与海图数据对比
        # 获取2023年的预测结果和实际海图数据
        t = -1  # 2023年的索引
        predicted_depth = depths[t]
        # H, W defined earlier during mask loading or prediction
        if 'H' not in locals(): H, W = predicted_depth.shape

        # 提取海图位置对应的预测值 (nautical_charts is already loaded)
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

        # 计算验证指标
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
    """创建时序合成图"""
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

        # 添加colorbar
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
        logger.error(f"创建时序合成图失败: {str(e)}")
        raise

def stage2_s3gm_processing(system, sentinel_time_series, gebco_time_series, sentinel_years):
    """第二阶段：S3GM模型处理"""
    try:
        # 创建必要的目录
        results_dir = 'results'
        models_dir = os.path.join(results_dir, 's3gm_pretrained_models')  
        time_series_dir = os.path.join(results_dir, 's3gm_time_series') 
        os.makedirs(results_dir, exist_ok=True)
        os.makedirs(models_dir, exist_ok=True)
        os.makedirs(time_series_dir, exist_ok=True)
        # Stage 2.1: 预训练
        pretrained_model_path = os.path.join(models_dir, 's3gm_pretrained.pth')
        if not os.path.exists(pretrained_model_path):
            logger.info("执行Stage 2.1: 预训练阶段")
            
            # 先加载训练好的经典模型
            classic_models = load_trained_classic_models()
            system.set_classic_models(classic_models)
            
            # 使用经典模型生成预测结果
            classic_predictions = []
            for year in sentinel_years:
                blue = sentinel_time_series['blue'][year - 2018]
                green = sentinel_time_series['green'][year - 2018]
                depth = system.classic_models.predict_rf(blue, green)
                if np.isnan(depth).any() or np.isinf(depth).any():
                    logger.warning(f"{year}年经典模型预测结果包含无效值")
                    depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
                classic_predictions.append(depth)
            
            # 将预测结果转换为时序数组 (T, H, W)
            classic_time_series = np.stack(classic_predictions)
            
            # 对数据进行标准化处理
            preprocessor = DataPreprocessor()
            normalized_classic, classic_stats = preprocessor._normalize_data(classic_time_series, 'classic')
            normalized_gebco, gebco_stats = preprocessor._normalize_data(gebco_time_series, 'gebco')
            
            # 保存统计信息，用于后续反标准化
            np.save('intermediate_results/classic_stats.npy', classic_stats)
            np.save('intermediate_results/gebco_stats.npy', gebco_stats)
            
            # *** 新增：保存归一化后的数据 ***
            np.save('intermediate_results/classic_normalized.npy', normalized_classic)
            np.save('intermediate_results/gebco_normalized.npy', normalized_gebco)
            logger.info("已保存归一化后的经典模型和GEBCO数据")
            
            pretrain_data = {
                'classic': normalized_classic,  # 现在是[0,1]范围
                'gebco': normalized_gebco,     # 现在是[0,1]范围
            }
            
            # 执行预训练（预训练函数内部会自动保存模型）
            system.s3gm.pretrain(
                classic_data=pretrain_data['classic'],
                gebco_data=pretrain_data['gebco'],
                save_path=pretrained_model_path
            )
        else:
            logger.info("加载已有预训练模型")
            system.s3gm.load_pretrained(pretrained_model_path)
        
        # 添加范围适配参数日志
        logger.info("模型配置信息:")
        logger.info(f"  - 范围适配: {system.s3gm.config.range_adaptation['enabled']}")
        logger.info(f"  - 混合激活函数: {system.s3gm.config.range_adaptation['use_mixed_activation']}")
        logger.info(f"  - 陆地标记值: {system.s3gm.config.range_adaptation['land_value']}")
        
        # 在Stage 2.2之前验证模型
        logger.info("验证模型...")
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
            # 获取模型输出的第一个元素（主要输出）
            test_output = test_output[0]
            # 修改验证逻辑
            if torch.all(torch.eq(test_output, torch.zeros_like(test_output))):
                raise ValueError("预训练模型输出全为0，需要重新训练")
            logger.info(f"模型测试输出范围: [{test_output.min().item():.4f}, {test_output.max().item():.4f}]")
            
        # Stage 2.2: 条件采样
        logger.info("执行Stage 2.2: 条件采样阶段")
        # 获取2023年海图数据
        aoi = ee.Geometry.Rectangle([122.35, 30.62, 122.6, 30.8])
        preprocessor = DataPreprocessor(region=aoi)
        nautical_charts = preprocessor.get_sparse_points()
        
        # 对海图数据进行标准化
        preprocessor = DataPreprocessor()
        normalized_depths, chart_stats = preprocessor._normalize_data(
            nautical_charts['depths'], 
            data_type='chart'
        )
        
        # 保存海图数据的统计信息（用于后续分析）
        os.makedirs('intermediate_results', exist_ok=True)
        np.save('intermediate_results/chart_stats.npy', chart_stats)
        logger.info(f"海图数据统计信息已保存: {chart_stats}")
        
        # 更新nautical_charts字典
        nautical_charts_normalized = {
            'depths': normalized_depths,
            'coordinates': nautical_charts['coordinates']
        }
        
        # 加载预处理数据
        # 注意：这里假设 classic_data 和 gebco_data 已经是归一化后的
        # 你可能需要先加载它们，然后使用 preprocessor 进行归一化
        classic_data_normalized = np.load('intermediate_results/classic_normalized.npy') # 假设已保存归一化的经典模型数据
        gebco_data_normalized = np.load('intermediate_results/gebco_normalized.npy') # 假设已保存归一化的GEBCO数据

        # 执行条件采样 - 传入经典模型和GEBCO数据
        results = system.s3gm.conditional_sampling(
            measurements=nautical_charts_normalized['depths'],
            measurement_coordinates=nautical_charts_normalized['coordinates'],
            years=sentinel_years,
            classic_data=classic_data_normalized, # 传入归一化的经典数据
            gebco_data=gebco_data_normalized      # 传入归一化的GEBCO数据
        )
        
        # 在保存结果之前，检查results的形状和内容
        if isinstance(results, torch.Tensor):
            results = results.cpu().numpy()
        
        # 保存结果时传入chart_stats
        save_results(
            depths=results,
            years=sentinel_years,
            output_dir=time_series_dir,
            chart_stats=chart_stats  # 确保传入chart_stats
        )
        
    except Exception as e:
        logger.error(f"第二阶段处理失败: {str(e)}")
        raise

def stage3_postprocessing():
    """第三阶段：后处理分析"""
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
                 ee.Initialize(project='fast-banner-452901-c8')
        except ImportError:
             logger.error("Google Earth Engine Python API (ee) not found. Cannot get nautical charts.")
             raise
        except Exception as e:
             logger.error(f"GEE initialization failed: {e}")
             raise

        aoi = ee.Geometry.Rectangle([122.35, 30.62, 122.6, 30.8])
        preprocessor = DataPreprocessor(region=aoi)
        nautical_charts = preprocessor.get_sparse_points()

        chart_coords_s3gm = nautical_charts['coordinates'] # Get coordinates
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
        logger.error(f"后处理分析失败: {str(e)}")
        raise

def load_trained_classic_models():
    """加载训练好的经典模型和参数"""
    try:
        classic_models = ClassicModels()
        
        # 加载参数
        rf_params = np.load('intermediate_results/model_params/rf_params.npy', allow_pickle=True).item()
        classic_models.rf_params = rf_params
        
        # 加载随机森林模型
        classic_models.rf_model = load('intermediate_results/model_params/rf_model.joblib')
        
        return classic_models
        
    except Exception as e:
        logger.error(f"加载经典模型失败: {str(e)}")
        raise

def denormalize_bathymetry(normalized_data: np.ndarray, stats: Dict[str, float]) -> np.ndarray:
    """使用 Min-Max 反标准化水深数据，并强制物理约束"""
    try:
        # 从 stats 中获取物理范围和特殊值
        min_phys = stats.get('min_phys', 0.0)
        max_phys = stats.get('max_phys', 90.0)
        land_value_norm = stats.get('land_value', 1.5)
        eps = 1e-6

        # 识别陆地区域 (使用 isclose 以处理浮点误差)
        land_mask = np.isclose(normalized_data, land_value_norm)
        
        # 检查实际数据范围（用于调试）
        valid_data = normalized_data[~land_mask & ~np.isnan(normalized_data)]
        if len(valid_data) > 0:
            actual_min, actual_max = valid_data.min(), valid_data.max()
            logger.info(f"反标准化前有效数据范围: [{actual_min:.4f}, {actual_max:.4f}]")
            # 对超出 [-1, 1] 的值发出警告
            if actual_min < -1.0 - eps or actual_max > 1.0 + eps:
                 logger.warning(f"  注意: 有效数据范围超出预期的 [-1, 1] 区间！")
        else:
            logger.warning("反标准化前没有找到有效（非陆地/NaN）数据")

        # 执行反标准化计算: phys = ((norm + 1) / 2) * (max_phys - min_phys) + min_phys
        depth_denorm = ((normalized_data + 1) / 2) * (max_phys - min_phys) + min_phys

        # 处理特殊区域：将陆地区域设为0米（或根据需要设为NaN）
        depth_denorm = np.where(land_mask, 0.0, depth_denorm)
        
        # 强制物理约束：水深必须非负 (理论上 Min-Max 不会产生负值，除非原始数据或操作有问题)
        sea_mask = ~land_mask
        if np.any(depth_denorm[sea_mask] < 0):
            neg_count = np.sum(depth_denorm[sea_mask] < 0)
            total_count = np.sum(sea_mask)
            logger.warning(f"检测到负水深值，占海域像素的 {(neg_count/total_count)*100:.2f}% (理论上不应发生)")
            logger.warning(f"负值范围: [{depth_denorm[sea_mask & (depth_denorm < 0)].min():.2f}, 0) 米")
            depth_denorm[sea_mask] = np.maximum(depth_denorm[sea_mask], 0.0)
        
        # 处理可能的 NaN 值 (例如，如果输入本身包含NaN)
        depth_denorm = np.nan_to_num(depth_denorm, nan=0.0) # 将NaN也设为0米
        
        # 最终检查和日志
        if np.any(sea_mask):
            sea_depths = depth_denorm[sea_mask]
            logger.info(f"反标准化后深度范围 (海域): [{sea_depths.min():.2f}, {sea_depths.max():.2f}]")
        else:
            logger.warning("反标准化后没有海域像素")
            
        return depth_denorm
        
    except Exception as e:
        logger.error(f"反标准化失败: {str(e)}")
        raise

def save_results(depths, years, output_dir, chart_stats):
    """保存预测结果
    
    Args:
        depths: 形状为[1, T, C, H, W]的张量，其中:
            - 1: batch size
            - T: 时间步数（年份数）
            - C: 通道数（组件数）
            - H, W: 图像高度和宽度
        output_dir: 输出目录
        chart_stats: 海图数据的统计信息
    """
    try:
        os.makedirs(output_dir, exist_ok=True)
        
        # 对每一年的预测结果进行反标准化和保存
        for t, year in enumerate(years):
            # 获取当前年份的深度数据（通道2是深度数据）
            depth_data = depths[0, t, 2]  # 形状应为[H, W]
            
            # 在反标准化之前检查数据
            logger.info(f"{year}年反标准化前:")
            logger.info(f"- 数据范围: [{depth_data.min():.4f}, {depth_data.max():.4f}]")
            logger.info(f"- 非零值分布: {np.percentile(depth_data[depth_data != 0], [25, 50, 75])}")
            
            # 使用统一的海图统计信息进行反标准化
            depth_denorm = denormalize_bathymetry(depth_data, chart_stats)
            
            # 反标准化后检查
            logger.info(f"{year}年反标准化后:")
            logger.info(f"- 数据范围: [{depth_denorm.min():.4f}, {depth_denorm.max():.4f}]")
            logger.info(f"- 有效值分布: {np.percentile(depth_denorm[depth_denorm > 0], [25, 50, 75])}")
            
            # 保存反标准化后的结果
            output_path = os.path.join(output_dir, f'bathymetry_{year}.npy')
            np.save(output_path, depth_denorm)
            
            logger.info(f"已保存{year}年的预测结果: {output_path}")
            logger.info(f"{year}年深度范围: [{depth_denorm.min():.2f}, {depth_denorm.max():.2f}] 米")
            logger.info(f"{year}年陆地像素比例: {np.mean(np.abs(depth_data - 1.5) < 0.1):.2%}")
            
    except Exception as e:
        logger.error(f"保存结果失败: {str(e)}")
        raise

def main():
    try:
        # 设置日志系统（移除这里的调用）
        # setup_logging()  # 删除这行
        
        logger.info("开始执行水深反演程序...")
        
        args = parse_args()
        
        # 初始化基本参数
        aoi = ee.Geometry.Rectangle([122.35, 30.62, 122.6, 30.8])
        gebco_years = range(2019, 2025)
        sentinel_years = range(2018, 2024)
        
        if args.stage == '1':
            logger.info("执行第一阶段：数据预处理")
            stage1_preprocessing(aoi, gebco_years, sentinel_years)
            
        elif args.stage == '1.5':
            logger.info("执行第1.5阶段：经典模型训练")
            sentinel_time_series = np.load('intermediate_results/sentinel_time_series.npy', allow_pickle=True).item()
            classic_models = stage1_5_model_training(sentinel_time_series)
        
        elif args.stage == '1.8':
            logger.info("执行第1.8阶段：经典模型验证")
            sentinel_time_series = np.load('intermediate_results/sentinel_time_series.npy', allow_pickle=True).item()
            stage1_8_model_validation(sentinel_time_series)
                
        elif args.stage == '2':
            logger.info("执行第二阶段：S3GM模型处理")
            # 加载预处理数据和训练好的模型参数
            sentinel_time_series = np.load('intermediate_results/sentinel_time_series.npy', allow_pickle=True).item()
            gebco_time_series = np.load('intermediate_results/gebco_time_series.npy')
            
            # 初始化系统
            system = HybridBathymetrySystem(
                region=aoi,
                time_range={'gebco': gebco_years, 'sentinel': sentinel_years}
            )
            
            # 执行第二阶段
            stage2_s3gm_processing(system, sentinel_time_series, gebco_time_series, sentinel_years)

        elif args.stage == '3':
            logger.info("执行第三阶段：结果后处理与可视化")
            stage3_postprocessing()


    except Exception as e:
        logger.error(f"程序执行过程中发生错误: {str(e)}")
        raise

if __name__ == '__main__':
    setup_logging()  # 只在这里调用一次
    main()

# 运行第一阶段（数据预处理）
# python run_bathymetry.py --stage 1

# 运行第1.5阶段（经典模型训练）
# python run_bathymetry.py --stage 1.5

# 运行第1.8阶段（经典模型验证）
# python run_bathymetry.py --stage 1.8

# 运行第二阶段（S3GM模型处理）
# python run_bathymetry.py --stage 2

# 运行第三阶段（结果后处理与可视化）
# python run_bathymetry.py --stage 3