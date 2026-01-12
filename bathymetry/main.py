import os
import sys
from typing import Dict, Any, Optional, List, Tuple, Union
import numpy as np
import logging
import ee
from scipy.interpolate import RegularGridInterpolator
import torch
import cv2

# Import custom modules
from .preprocessor import DataPreprocessor
from .classic_models import ClassicModels
from .s3gm_wrapper import S3GMWrapper
from .utils import validate_data, create_spatiotemporal_grid
from .gpu_memory import GPUMemoryManager

logger = logging.getLogger(__name__)

class HybridBathymetrySystem:
    """Hybrid bathymetry estimation system"""
    
    def __init__(
        self,
        region: ee.Geometry,
        time_range: Dict[str, range],
        config_path: Optional[str] = None,
        use_gpu: bool = True
    ):
        """
        Initialize the system
        
        Args:
            region: Study area
            time_range: Time range configuration
            config_path: S3GM configuration file path
            use_gpu: Whether to use GPU
        """
        try:
            self.region = region
            self.time_range = time_range
            self.device = torch.device('cuda' if use_gpu and torch.cuda.is_available() else 'cpu')
            
            # Initialize modules
            self.preprocessor = DataPreprocessor(region)
            self.classic_models = None  # Set to None first, wait for external setting
            self.s3gm = S3GMWrapper(
                config_path or 'configs/s3gm_default.yaml'
            )
            
            logger.info(f"System initialization completed, using device: {self.device}")
            
        except Exception as e:
            logger.error(f"系统初始化失败: {str(e)}")
            raise
            
    def _preprocess_data(
        self,
        sentinel_data: Dict[str, np.ndarray],
        gebco_data: np.ndarray,
        sparse_measurements: np.ndarray,
        measurement_coordinates: np.ndarray
    ) -> Dict[str, Any]:
        """数据预处理"""
        try:
            # 验证输入数据
            validate_data(
                sentinel_data, 
                gebco_data,
                sparse_measurements,
                measurement_coordinates
            )
            
            # 使用预处理器处理数据
            processed_data = self.preprocessor.process(
                sentinel_data=sentinel_data,
                gebco_data=gebco_data,
                sparse_measurements=sparse_measurements,
                measurement_coordinates=measurement_coordinates,
                is_ee_image=False
            )
            
            logger.info("数据预处理完成")
            return processed_data
            
        except Exception as e:
            logger.error(f"数据预处理失败: {str(e)}")
            raise
            
    def process(
        self,
        sentinel_data: Dict[str, np.ndarray],
        gebco_data: np.ndarray,
        measurements: np.ndarray,
        measurement_coordinates: np.ndarray,
        target_years: Optional[List[int]] = None,
        is_ee_image: bool = False
    ) -> Dict[str, np.ndarray]:
        """处理输入数据并生成预测结果"""
        try:
            # 1. 预处理数据
            processed_data = self._preprocess_data(
                sentinel_data,
                gebco_data,
                measurements,
                measurement_coordinates
            )
            
            # 2. 使用经典模型进行预测
            classic_results = self.classic_models.predict(processed_data['sentinel'], 'rf')
            
            # 3. 标准化经典模型预测结果
            normalized_classic, classic_stats = self.preprocessor._normalize_data(
                classic_results,
                data_type='classic'
            )
            
            # 记录数据标准化前后的范围
            logger.info("数据标准化前后的范围:")
            logger.info(f"GEBCO数据 - 原始范围: [{np.nanmin(gebco_data):.2f}, {np.nanmax(gebco_data):.2f}]")
            logger.info(f"GEBCO数据 - 标准化后: [{np.nanmin(processed_data['gebco']):.2f}, {np.nanmax(processed_data['gebco']):.2f}]")
            logger.info(f"经典模型预测 - 原始范围: [{np.nanmin(classic_results):.2f}, {np.nanmax(classic_results):.2f}]")
            logger.info(f"经典模型预测 - 标准化后: [{np.nanmin(normalized_classic):.2f}, {np.nanmax(normalized_classic):.2f}]")
            
            # 4. 准备S3GM输入数据
            s3gm_input = {
                'sentinel-classic': normalized_classic,
                'gebco': processed_data['gebco'],
                'measurements': {
                    'depths': processed_data['measurements']['values'],
                    'coordinates': measurement_coordinates
                },
                'years': target_years or [2018, 2019, 2020, 2021, 2022, 2023],
                'stats': {
                    'gebco': processed_data['stats']['gebco'],
                    'classic': classic_stats
                }
            }
            
            # 5. 执行S3GM预测
            s3gm_output = self.s3gm.predict(s3gm_input)
            
            return {
                'bathymetry': s3gm_output['depth'],
                'confidence': s3gm_output['confidence']
            }
            
        except Exception as e:
            logger.error(f"处理失败: {str(e)}")
            raise
            
    def validate(
        self,
        results: Dict[str, np.ndarray],
        validation_data: np.ndarray,
        validation_coordinates: np.ndarray
    ) -> Dict[str, float]:
        """
        验证预测结果
        
        Args:
            results: 预测结果
            validation_data: 验证数据
            validation_coordinates: 验证点坐标
            
        Returns:
            验证指标字典
        """
        try:
            metrics = {}
            
            # 计算各个模型结果的指标
            for model_name, result in results.items():
                model_metrics = self._calculate_metrics(
                    predictions=result,
                    ground_truth=validation_data,
                    coordinates=validation_coordinates
                )
                metrics[model_name] = model_metrics
                
            return metrics
            
        except Exception as e:
            logger.error(f"结果验证失败: {str(e)}")
            raise
            
    def _calculate_metrics(
        self,
        predictions: np.ndarray,
        ground_truth: np.ndarray,
        coordinates: np.ndarray
    ) -> Dict[str, float]:
        """计算评估指标"""
        try:
            # 在验证点位置提取预测值
            pred_values = self._extract_values_at_coordinates(predictions, coordinates)
            
            # 计算指标
            metrics = {
                'rmse': np.sqrt(np.mean((pred_values - ground_truth) ** 2)),
                'mae': np.mean(np.abs(pred_values - ground_truth)),
                'r2': 1 - np.sum((ground_truth - pred_values) ** 2) / \
                      np.sum((ground_truth - np.mean(ground_truth)) ** 2)
            }
            
            return metrics
            
        except Exception as e:
            logger.error(f"指标计算失败: {str(e)}")
            raise
            
    def _extract_values_at_coordinates(
        self,
        data: np.ndarray,
        coordinates: np.ndarray
    ) -> np.ndarray:
        """在指定坐标位置提取值"""
        try:
            values = np.zeros(len(coordinates))
            
            for i, (y, x) in enumerate(coordinates):
                # 确保坐标为整数
                y_idx = int(round(y))
                x_idx = int(round(x))
                
                # 边界检查
                y_idx = np.clip(y_idx, 0, data.shape[0] - 1)
                x_idx = np.clip(x_idx, 0, data.shape[1] - 1)
                
                values[i] = data[y_idx, x_idx]
                
            return values
            
        except Exception as e:
            logger.error(f"坐标值提取失败: {str(e)}")
            raise
            
    def cleanup(self):
        """清理资源"""
        try:
            # 清理GPU内存
            if torch.cuda.is_available():
                GPUMemoryManager.clear_gpu_memory()
                
            logger.info("系统资源清理完成")
            
        except Exception as e:
            logger.error(f"资源清理失败: {str(e)}")
            raise
            
    def set_classic_models(self, classic_models):
        """设置经典模型"""
        self.classic_models = classic_models
        self.s3gm.set_classic_models(classic_models)  # 同时更新S3GM包装器中的经典模型
        logger.info("经典模型已设置")