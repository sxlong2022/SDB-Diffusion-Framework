import numpy as np
import logging
from typing import Dict, Any, Optional, Tuple
import ee
ee.Authenticate()
ee.Initialize(project='YOUR_GEE_PROJECT_ID')
import glob
import os

logger = logging.getLogger(__name__)

class DataPreprocessor:
    """数据预处理器"""
    
    def __init__(self, region: Optional[ee.Geometry] = None):
        """
        初始化预处理器
        
        Args:
            region: 研究区域（可选）
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
        """数据预处理"""
        try:
            # 1. 遥感数据不需要标准化（经典模型内部会处理）
            
            # 2. GEBCO数据标准化（转换符号：负值水深->正值水深，正值陆地->负值陆地）
            # 注意：不再需要手动转换符号，在_normalize_data中处理
            normalized_gebco, gebco_stats = self._normalize_data(
                gebco_data, 
                data_type='gebco'
            )
            
            # 3. 标准化稀疏观测点数据（海图数据）
            normalized_measurements, measurement_stats = self._normalize_data(
                sparse_measurements,
                data_type='chart'
            )
            
            # 4. 创建测量点网格
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
            logger.error(f"数据预处理失败: {str(e)}")
            raise
            
    def _normalize_data(self, data: np.ndarray, data_type: str) -> Tuple[np.ndarray, Dict[str, float]]:
        """使用 Min-Max 标准化将有效物理水深映射到 [-1, 1] 区间"""
        try:
            normalized_data = data.astype(np.float32).copy() 
            
            # 设定物理范围和特殊值
            min_phys = 0.0
            max_phys = 90.0
            land_value_norm = 1.5 # 标准化后的陆地/无效值
            eps = 1e-6 # 防止除零
            
            stats = self._get_default_stats()
            stats['min_phys'] = min_phys
            stats['max_phys'] = max_phys
            stats['land_value'] = land_value_norm

            if data_type == 'gebco':
                # GEBCO: data < 0 是水深, data >= 0 是陆地
                sea_mask = data < 0
                land_mask = ~sea_mask
                if not np.any(sea_mask):
                    logger.warning(f"GEBCO数据中没有有效的水深数据")
                    normalized_data.fill(land_value_norm) 
                    return normalized_data, stats
                
                sea_depths_phys = -data[sea_mask] # 转换为正物理深度
                
                # 记录原始统计信息
                stats.update({
                    'median': float(np.median(sea_depths_phys)), 
                    'iqr': float(np.percentile(sea_depths_phys, 75) - np.percentile(sea_depths_phys, 25)), 
                    'q25': float(np.percentile(sea_depths_phys, 25)), 
                    'q75': float(np.percentile(sea_depths_phys, 75)),
                    'min': float(sea_depths_phys.min()), 
                    'max': float(sea_depths_phys.max())
                })
                
                # 应用 Min-Max 标准化到 [-1, 1]
                normalized_sea = 2 * (sea_depths_phys - min_phys) / (max_phys - min_phys + eps) - 1
                # Clip to ensure values are within [-1, 1] for numerical stability, though ideally they should be.
                normalized_sea = np.clip(normalized_sea, -1.0, 1.0)
                
                normalized_data[sea_mask] = normalized_sea
                normalized_data[land_mask] = land_value_norm # 陆地设为特殊值

            elif data_type in ['classic', 'chart']:
                # Classic/Chart: data > 0 是水深, data <= 0 是无效/陆地
                valid_mask = data > 0
                invalid_mask = ~valid_mask
                if not np.any(valid_mask):
                    logger.warning(f"{data_type}数据中没有有效的水深数据")
                    normalized_data.fill(land_value_norm) # Fill with invalid value
                    return normalized_data, stats

                valid_depths_phys = data[valid_mask] # 已经是正物理深度
                
                # 记录原始统计信息
                stats.update({
                    'median': float(np.median(valid_depths_phys)), 
                    'iqr': float(np.percentile(valid_depths_phys, 75) - np.percentile(valid_depths_phys, 25)), 
                    'q25': float(np.percentile(valid_depths_phys, 25)), 
                    'q75': float(np.percentile(valid_depths_phys, 75)),
                    'min': float(valid_depths_phys.min()), 
                    'max': float(valid_depths_phys.max()),
                    'invalid_value': land_value_norm # 使用统一的 land_value_norm
                })
                
                # 应用 Min-Max 标准化到 [-1, 1]
                normalized_valid = 2 * (valid_depths_phys - min_phys) / (max_phys - min_phys + eps) - 1
                normalized_valid = np.clip(normalized_valid, -1.0, 1.0)

                normalized_data[valid_mask] = normalized_valid
                normalized_data[invalid_mask] = land_value_norm # 无效/陆地设为特殊值

            # Final check for NaNs introduced during processing
            if np.isnan(normalized_data).any():
                 logger.warning(f"{data_type} 标准化后包含NaN值，将替换为陆地/无效值 {land_value_norm}")
                 normalized_data = np.nan_to_num(normalized_data, nan=land_value_norm)

            self._check_normalized_data(normalized_data, data_type, stats)

            return normalized_data, stats

        except Exception as e:
            logger.error(f"数据标准化失败 ({data_type}): {str(e)}")
            raise

    def _check_normalized_data(self, data: np.ndarray, data_type: str, stats: Dict[str, float]):
        """增强的标准化数据检查"""
        # Check for NaNs before processing stats
        if np.isnan(data).any():
            logger.warning(f"{data_type}数据检查时发现NaN值 (在填充后?)")

        land_value = stats.get('land_value', 1.5) # Get land/invalid value from stats
        # 使用 isclose 检查陆地/无效值
        valid_mask = ~np.isclose(data, land_value) & ~np.isnan(data) # Exclude land/invalid and NaNs

        logger.info(f"{data_type} 标准化后数据分布:")
        land_ratio = np.mean(np.isclose(data, land_value))
        logger.info(f"- 陆地/无效值 ({land_value:.1f}) 比例: {land_ratio:.2%}")

        if np.any(valid_mask):
            valid_data = data[valid_mask]
            percentiles = np.percentile(valid_data, [0, 25, 50, 75, 100]) if len(valid_data) > 0 else ["N/A"]*5
            logger.info(f"- 有效值范围: [{valid_data.min():.4f}, {valid_data.max():.4f}]")
            logger.info(f"- 有效值分位数 [0, 25, 50, 75, 100]: {percentiles}")
            logger.info(f"- 原始物理值范围: [{stats.get('min', 'N/A'):.2f}, {stats.get('max', 'N/A'):.2f}]")
            if 'iqr' in stats:
                logger.info(f"- 原始物理值IQR: {stats['iqr']:.2f}")
        else:
            logger.warning(f"- {data_type}数据中没有有效值 (非陆地/无效值)")


    def _get_default_stats(self) -> Dict[str, float]:
        """扩展的默认统计值"""
        # Provide more robust defaults, especially min/max
        return {
            'median': 0.0, 'iqr': 1.0, 'q25': -0.5, 'q75': 0.5,
            'land_value': 1.5, 'invalid_value': 1.5, # Keep both for clarity
            'min': 0.0, 'max': 0.0
        }
            
    def _create_measurement_grid(
        self,
        measurements: np.ndarray,
        coordinates: np.ndarray,
        shape: Tuple[int, int]
    ) -> np.ndarray:
        """创建测量点网格"""
        try:
            H, W = shape
            grid = np.zeros((H, W))
            
            # 将归一化坐标转换为像素坐标
            pixel_coords = np.zeros_like(coordinates)
            pixel_coords[:, 0] = coordinates[:, 0] * (H - 1)  # y坐标
            pixel_coords[:, 1] = coordinates[:, 1] * (W - 1)  # x坐标
            
            # 四舍五入到最近的整数坐标
            pixel_coords = np.round(pixel_coords).astype(int)
            
            # 边界检查
            pixel_coords[:, 0] = np.clip(pixel_coords[:, 0], 0, H - 1)
            pixel_coords[:, 1] = np.clip(pixel_coords[:, 1], 0, W - 1)
            
            # 填充网格
            for coord, value in zip(pixel_coords, measurements):
                grid[coord[0], coord[1]] = value
            
            logger.info(f"创建了形状为 {shape} 的测量点网格")
            return grid
            
        except Exception as e:
            logger.error(f"测量点网格创建失败: {str(e)}")
            raise
            
    def get_sparse_points(self) -> Dict[str, np.ndarray]:
        """获取海图稀疏观测点数据"""
        try:
            if self.region is None:
                raise ValueError("未设置研究区域，请在初始化时指定region参数")
            
            # 获取研究区域边界
            bounds = self.region.bounds().getInfo()
            min_lon = bounds['coordinates'][0][0][0]  # 西边界
            min_lat = bounds['coordinates'][0][0][1]  # 南边界
            max_lon = bounds['coordinates'][0][2][0]  # 东边界
            max_lat = bounds['coordinates'][0][2][1]  # 北边界
        
            logger.info(f"正在获取区域 [{min_lon:.3f}, {min_lat:.3f}, {max_lon:.3f}, {max_lat:.3f}] 内的海图数据...")
        
            # 读取海图数据
            chart_files = glob.glob(os.path.join('Official_nautical_chart', '*_sample.xyz'))
            if not chart_files:
                raise FileNotFoundError("找不到海图数据文件")
        
            points_data = []
            for file in chart_files:
                data = np.loadtxt(file)
                # 仅保留研究区域内的点
                mask = ((data[:, 0] >= min_lon) & (data[:, 0] <= max_lon) & 
                       (data[:, 1] >= min_lat) & (data[:, 1] <= max_lat))
                points_data.append(data[mask])
            
            if not points_data:
                raise ValueError("在指定区域内未找到任何海图数据点")
            
            points_data = np.vstack(points_data)
        
            # 提取坐标和深度
            lon = points_data[:, 0]
            lat = points_data[:, 1]
            depths = points_data[:, 2]
        
            # 将地理坐标转换为归一化网格坐标 (0-1范围)
            norm_coords = np.zeros((len(lon), 2))
            norm_coords[:, 0] = (lat - min_lat) / (max_lat - min_lat)
            norm_coords[:, 1] = (lon - min_lon) / (max_lon - min_lon)
        
            logger.info(f"在研究区域内找到 {len(depths)} 个海图观测点")
            logger.info(f"深度范围: {np.min(depths):.2f} 到 {np.max(depths):.2f} 米")
        
            return {
                'depths': depths,  # 返回原始深度值，不进行标准化
                'coordinates': norm_coords
            }
        
        except Exception as e:
            logger.error(f"获取海图数据失败: {str(e)}")
            raise