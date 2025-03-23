import sys
import os
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Union

# 添加S3GM代码路径
s3gm_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'S3GM', 'Code')
if not os.path.exists(s3gm_path):
    raise ImportError(f"S3GM代码路径不存在: {s3gm_path}")
sys.path.append(s3gm_path)

# 现在可以导入S3GM模块
from sampler.sde import VESDE, VPSDE

logger = logging.getLogger(__name__)

def validate_data(
    sentinel_data: Dict[str, np.ndarray],
    gebco_data: np.ndarray,
    sparse_measurements: np.ndarray,
    measurement_coordinates: np.ndarray
) -> bool:
    """验证输入数据的有效性"""
    try:
        # 验证Sentinel数据
        required_bands = ['blue', 'green']
        for band in required_bands:
            if band not in sentinel_data:
                raise ValueError(f"缺少必要的波段: {band}")
                
        # 验证GEBCO数据
        if gebco_data.ndim != 3:  # [T, H, W]
            raise ValueError("GEBCO数据维度不正确")
            
        # 验证稀疏观测点
        if sparse_measurements.ndim != 1:
            raise ValueError("稀疏观测点数据维度不正确")
            
        if measurement_coordinates.shape[1] != 2:
            raise ValueError("观测点坐标必须是二维的(x,y)")
            
        if len(sparse_measurements) != len(measurement_coordinates):
            raise ValueError("观测点数量与坐标数量不匹配")
            
        return True
        
    except Exception as e:
        logger.error(f"数据验证失败: {str(e)}")
        raise

def create_spatiotemporal_grid(
    coordinates: np.ndarray,
    values: np.ndarray,
    shape: Tuple[int, int],
    time_range: Optional[List[int]] = None
) -> np.ndarray:
    """创建时空网格"""
    try:
        H, W = shape
        if time_range is None:
            time_range = list(range(2018, 2024))
        T = len(time_range)
        
        # 创建网格
        grid = np.zeros((T, H, W))
        
        # 归一化坐标
        norm_coords = coordinates.copy()
        norm_coords[:, 0] = norm_coords[:, 0] * (H - 1)
        norm_coords[:, 1] = norm_coords[:, 1] * (W - 1)
        
        # 填充值
        for t in range(T):
            for val, (y, x) in zip(values, norm_coords):
                y_idx = int(round(y))
                x_idx = int(round(x))
                if 0 <= y_idx < H and 0 <= x_idx < W:
                    grid[t, y_idx, x_idx] = val
                    
        return grid
        
    except Exception as e:
        logger.error(f"时空网格创建失败: {str(e)}")
        raise

def normalize_data(
    data: np.ndarray,
    min_val: Optional[float] = None,
    max_val: Optional[float] = None
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    数据标准化
    
    Args:
        data: 输入数据
        min_val: 最小值（可选）
        max_val: 最大值（可选）
        
    Returns:
        normalized_data: 标准化后的数据
        stats: 统计信息（最小值、最大值）
    """
    try:
        # 处理NaN值
        data = np.nan_to_num(data, nan=np.nanmean(data))
        
        # 获取数据范围
        if min_val is None:
            min_val = np.percentile(data, 1)  # 使用1%分位数避免异常值影响
        if max_val is None:
            max_val = np.percentile(data, 99)  # 使用99%分位数避免异常值影响
            
        # 标准化到[0,1]区间
        normalized_data = (data - min_val) / (max_val - min_val + 1e-8)
        normalized_data = np.clip(normalized_data, 0, 1)
        
        stats = {
            'min': float(min_val),
            'max': float(max_val)
        }
        
        return normalized_data, stats
        
    except Exception as e:
        logger.error(f"数据标准化失败: {str(e)}")
        raise

def calculate_uncertainty(
    predictions: Dict[str, np.ndarray],
    measurements: Optional[Dict[str, np.ndarray]] = None,
    confidence_threshold: float = 0.8
) -> np.ndarray:
    """
    计算预测结果的不确定性
    
    Args:
        predictions: 不同模型的预测结果
        measurements: 实测数据（可选）
        confidence_threshold: 置信度阈值
        
    Returns:
        uncertainty: 不确定性估计
    """
    try:
        # 1. 计算模型间的标准差
        all_predictions = np.stack([pred for pred in predictions.values()])
        model_std = np.std(all_predictions, axis=0)
        
        # 2. 计算与测量点的偏差（如果有测量数据）
        measurement_error = np.zeros_like(model_std)
        if measurements is not None and 'depths' in measurements and 'coordinates' in measurements:
            depths = measurements['depths']
            coords = measurements['coordinates']
            
            for depth, (y, x) in zip(depths, coords):
                y_idx = int(y * model_std.shape[0])
                x_idx = int(x * model_std.shape[1])
                
                # 计算每个模型在测量点的预测误差
                for pred in predictions.values():
                    error = np.abs(pred[y_idx, x_idx] - depth)
                    measurement_error[y_idx, x_idx] = max(
                        measurement_error[y_idx, x_idx],
                        error
                    )
        
        # 3. 综合不确定性估计
        uncertainty = (model_std + measurement_error) / 2
        
        # 4. 归一化
        uncertainty = (uncertainty - uncertainty.min()) / (uncertainty.max() - uncertainty.min() + 1e-8)
        
        # 5. 应用置信度阈值
        uncertainty[uncertainty > confidence_threshold] = confidence_threshold
        
        return uncertainty
        
    except Exception as e:
        logger.error(f"不确定性计算失败: {str(e)}")
        raise