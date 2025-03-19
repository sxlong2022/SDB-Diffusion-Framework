import numpy as np
import torch
import logging
from typing import Dict, Optional, Union
from scipy.interpolate import RegularGridInterpolator

logger = logging.getLogger(__name__)

class DataFusionModule:
    """数据融合模块"""
    
    def __init__(self, config: Optional[Dict] = None):
        """
        初始化数据融合模块
        
        Args:
            config: 融合配置参数
        """
        self.config = config or {
            'classic_weight': 0.3,    # 经典模型权重
            's3gm_weight': 0.4,       # S3GM模型权重
            'gebco_weight': 0.2,      # GEBCO数据权重
            'measurement_weight': 0.1, # 测量点权重
            'spatial_decay': 0.1,     # 空间衰减系数
            'confidence_threshold': 0.8 # 置信度阈值
        }
        
    def fuse(
        self,
        classic_results: np.ndarray,
        s3gm_results: np.ndarray,
        gebco_data: np.ndarray,
        measurements: Dict[str, np.ndarray]
    ) -> np.ndarray:
        """
        融合多源数据
        
        Args:
            classic_results: 经典模型结果
            s3gm_results: S3GM模型结果
            gebco_data: GEBCO数据
            measurements: 测量点数据
            
        Returns:
            融合后的结果
        """
        try:
            # 1. 数据标准化
            classic_norm = self._normalize_data(classic_results)
            s3gm_norm = self._normalize_data(s3gm_results)
            gebco_norm = self._normalize_data(gebco_data)
            
            # 2. 计算置信度图
            confidence_map = self._calculate_confidence_map(
                classic_norm,
                s3gm_norm,
                gebco_norm,
                measurements
            )
            
            # 3. 加权融合
            fused_result = (
                self.config['classic_weight'] * classic_norm +
                self.config['s3gm_weight'] * s3gm_norm +
                self.config['gebco_weight'] * gebco_norm
            )
            
            # 4. 测量点约束
            fused_result = self._apply_measurement_constraints(
                fused_result,
                measurements,
                confidence_map
            )
            
            # 5. 后处理
            fused_result = self._postprocess(fused_result)
            
            return fused_result
            
        except Exception as e:
            logger.error(f"数据融合失败: {str(e)}")
            raise
            
    def _normalize_data(self, data: np.ndarray) -> np.ndarray:
        """数据标准化"""
        return (data - np.mean(data)) / np.std(data)
        
    def _calculate_confidence_map(
        self,
        classic_data: np.ndarray,
        s3gm_data: np.ndarray,
        gebco_data: np.ndarray,
        measurements: Dict[str, np.ndarray]
    ) -> np.ndarray:
        """计算置信度图"""
        try:
            H, W = classic_data.shape
            confidence_map = np.ones((H, W))
            
            # 基于测量点计算空间衰减
            if 'coordinates' in measurements:
                for y, x in measurements['coordinates']:
                    y_idx, x_idx = int(y * H), int(x * W)
                    y_grid, x_grid = np.meshgrid(
                        np.arange(H),
                        np.arange(W),
                        indexing='ij'
                    )
                    distance = np.sqrt(
                        (y_grid - y_idx)**2 + 
                        (x_grid - x_idx)**2
                    )
                    decay = np.exp(-self.config['spatial_decay'] * distance)
                    confidence_map *= (1 + decay)
                    
            # 归一化置信度图
            confidence_map = confidence_map / confidence_map.max()
            
            return confidence_map
            
        except Exception as e:
            logger.error(f"置信度图计算失败: {str(e)}")
            raise
            
    def _apply_measurement_constraints(
        self,
        fused_data: np.ndarray,
        measurements: Dict[str, np.ndarray],
        confidence_map: np.ndarray
    ) -> np.ndarray:
        """应用测量点约束"""
        try:
            result = fused_data.copy()
            
            if 'depths' in measurements and 'coordinates' in measurements:
                depths = measurements['depths']
                coords = measurements['coordinates']
                
                for depth, (y, x) in zip(depths, coords):
                    y_idx = int(y * result.shape[0])
                    x_idx = int(x * result.shape[1])
                    
                    # 在测量点位置应用强约束
                    if confidence_map[y_idx, x_idx] > self.config['confidence_threshold']:
                        result[y_idx, x_idx] = depth
                        
            return result
            
        except Exception as e:
            logger.error(f"测量点约束应用失败: {str(e)}")
            raise
            
    def _postprocess(self, data: np.ndarray) -> np.ndarray:
        """结果后处理"""
        try:
            # 1. 去除异常值
            data = np.clip(
                data,
                np.percentile(data, 1),
                np.percentile(data, 99)
            )
            
            # 2. 平滑处理
            from scipy.ndimage import gaussian_filter
            data = gaussian_filter(data, sigma=0.5)
            
            return data
            
        except Exception as e:
            logger.error(f"后处理失败: {str(e)}")
            raise