import numpy as np
import logging
from typing import Dict, Any, Optional, Tuple
import ee
ee.Authenticate()
ee.Initialize(project='fast-banner-452901-c8')
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
        """改进的标准化处理：使用分位数统计进行标准化，同时保持陆地区分"""
        try:
            normalized_data = np.zeros_like(data)
        
            if data_type == 'gebco':
                # GEBCO数据：需要分别处理海域和陆地
                sea_mask = data < 0
                if not np.any(sea_mask):
                    logger.warning(f"GEBCO数据中没有有效的水深数据")
                    return normalized_data, self._get_default_stats()
            
                # 将负值转换为正值
                sea_depths = -data[sea_mask]
            
                # 使用分位数统计
                q25, q75 = np.percentile(sea_depths, [25, 75])
                iqr = q75 - q25
                median = np.median(sea_depths)
            
                # 使用IQR进行标准化，映射到[-5,5]范围
                normalized_sea = (sea_depths - median) / (iqr + 1e-6) * 10  # 添加epsilon避免除零
                normalized_sea = np.clip(normalized_sea, -5, 5)  # 限制在±5个IQR范围内
                normalized_data[sea_mask] = normalized_sea
            
                # 陆地区域设为特殊值（如1.5）
                land_mask = ~sea_mask
                normalized_data[land_mask] = 1.5  # 使用超出[-1,1]范围的值表示陆地
            
                stats = {
                    'median': float(median),
                    'iqr': float(iqr),
                    'q25': float(q25),
                    'q75': float(q75),
                    'land_value': 1.5,
                    'min': float(sea_depths.min()),
                    'max': float(sea_depths.max())
                }
            
            elif data_type in ['classic', 'chart']:
                # 经典模型和海图数据：只处理有效值
                valid_mask = data > 0
                if not np.any(valid_mask):
                    logger.warning(f"{data_type}数据中没有有效的水深数据")
                    return normalized_data, self._get_default_stats()
            
                valid_depths = data[valid_mask]
            
                # 使用分位数统计
                q25, q75 = np.percentile(valid_depths, [25, 75])
                iqr = q75 - q25
                median = np.median(valid_depths)
            
                # 使用IQR进行标准化
                normalized_valid = (valid_depths - median) / (iqr + 1e-6) * 10
                normalized_valid = np.clip(normalized_valid, -5, 5)
                normalized_data[valid_mask] = normalized_valid
            
                # 无效区域（陆地或无数据）设为特殊值
                invalid_mask = ~valid_mask
                normalized_data[invalid_mask] = 1.5
            
                stats = {
                    'median': float(median),
                    'iqr': float(iqr),
                    'q25': float(q25),
                    'q75': float(q75),
                    'invalid_value': 1.5,
                    'min': float(valid_depths.min()),
                    'max': float(valid_depths.max())
                }
        
            # 增强数据检查
            self._check_normalized_data(normalized_data, data_type, stats)
        
            return normalized_data, stats
        
        except Exception as e:
            logger.error(f"数据标准化失败: {str(e)}")
            raise

    def _check_normalized_data(self, data: np.ndarray, data_type: str, stats: Dict[str, float]):
        """增强的标准化数据检查"""
        # 检查NaN值
        if np.isnan(data).any():
            logger.warning(f"{data_type}数据中存在NaN值")
    
        # 检查有效值范围（排除陆地标记值1.5）
        valid_mask = (data != 1.5) & (data != 0)
    
        # 输出详细的数据分布信息
        land_ratio = np.mean(data == 1.5)
        zero_ratio = np.mean(data == 0)
        if np.any(valid_mask):
            valid_data = data[valid_mask]
            percentiles = np.percentile(valid_data, [0, 25, 50, 75, 100])
            logger.info(f"{data_type}数据分布: ")
            logger.info(f"- 陆地比例: {land_ratio:.2%}")
            logger.info(f"- 零值比例: {zero_ratio:.2%}")
            logger.info(f"- 分位数 [0, 25, 50, 75, 100]: {percentiles}")
            logger.info(f"- 原始数据范围: [{stats['min']:.2f}, {stats['max']:.2f}]")
            if 'iqr' in stats:
                logger.info(f"- IQR: {stats['iqr']:.2f}")
        else:
            logger.warning(f"{data_type}数据中没有有效值")

    def _get_default_stats(self) -> Dict[str, float]:
        """扩展的默认统计值"""
        return {
            'median': 0.0,
            'iqr': 1.0,
            'q25': -0.5,
            'q75': 0.5,
            'land_value': 1.5,
            'min': 0.0,
            'max': 0.0
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