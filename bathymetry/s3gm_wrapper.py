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
from torch.cuda.amp import autocast
import torch.nn as nn
from .preprocessor import DataPreprocessor
from torch.utils.checkpoint import checkpoint
import math

# 添加S3GM代码路径
s3gm_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'S3GM', 'Code')
if not os.path.exists(s3gm_path):
    raise ImportError(f"S3GM代码路径不存在: {s3gm_path}")
sys.path.append(s3gm_path)

# 导入S3GM模块
from models.unet_video import UNetVideoModel
from sampler.sde import VESDE, VPSDE
from models.ema import ExponentialMovingAverage
from sampler.utils import complete_video_pc_dps, LangevinCorrector

logger = logging.getLogger(__name__)

class S3GMWrapper:
    """S3GM模型包装器"""
    
    def __init__(
        self,
        config_path: Optional[str] = None,
        classic_models: Optional[ClassicModels] = None
    ):
        """初始化S3GM包装器"""
        try:
            # 1. 加载配置
            self.config = load_config(config_path) if config_path else S3GMConfig()
            
            # 2. 确保经典模型实例存在
            self.classic_models = classic_models or ClassicModels()
            
            # 3. 设置设备
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
            # 4. 初始化模型 - 注意：UNetVideoModel内部会将in_channels+1
            input_channels = self.config.num_components
            logger.info(f"模型初始化：输入通道数={input_channels} (UNet内部会+1)")
            
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
                land_value=self.config.range_adaptation['land_value']
            ).to(self.device)
            
            # 添加日志记录已初始化的模型参数
            logger.info(f"模型参数计数: {sum(p.numel() for p in self.model.parameters())}")
            logger.info(f"模型输入通道: {self.model.in_channels}")
            logger.info(f"模型输出通道: {self.model.out_channels}")
            
            # 记录配置
            logger.info(f"范围适配配置:")
            for key, value in self.config.range_adaptation.items():
                logger.info(f"  - {key}: {value}")
            
            # 5. 初始化SDE
            sde_config = {
                'beta_min': self.config.beta_min,
                'beta_max': self.config.beta_max,
                'N': self.config.num_scales,
            }
            logger.info(f"初始化VPSDE配置: {sde_config}")
            self.sde = VPSDE(config=sde_config)
            logger.info(f"VPSDE初始化后参数: beta_min={self.sde.beta_0}, beta_max={self.sde.beta_1}, num_scales={self.config.num_scales}")
            
            # 添加参数验证
            if hasattr(self.sde, 'beta_max') and self.sde.beta_max != self.config.beta_max:
                logger.warning(f"SDE初始化后beta_max不匹配: 配置值={self.config.beta_max}, 实际值={self.sde.beta_max}")
                # 强制设置正确的值
                self.sde.beta_max = self.config.beta_max
                logger.info(f"已强制更新SDE的beta_max为配置值: {self.config.beta_max}")
            
            # 5. 添加score_fn（需要在corrector之前）
            self.score_fn = self._get_score_fn()
            
            # 6. 不在初始化时创建corrector实例
            self.corrector = None  # 移除这里的corrector初始化
        
            # 7. 添加EMA
            if hasattr(self.config, 'use_ema') and self.config.use_ema:
                self.ema = ExponentialMovingAverage(
                    self.model.parameters(),
                    decay=self.config.ema_rate
                )
            
            logger.info("S3GM包装器初始化完成")
            
        except Exception as e:
            logger.error(f"S3GM包装器初始化失败: {str(e)}")
            raise

    def set_classic_models(self, classic_models: ClassicModels) -> None:
        """设置经典模型"""
        self.classic_models = classic_models
        logger.info("S3GM包装器已更新经典模型")

    def _prepare_input_data(self, normalized_classic, gebco_data, measurements):
        """准备输入数据"""
        # 修改数值检查函数
        def check_data(data, name):
            # 确保数据是torch.Tensor类型
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
            # 检查每个输入
            check_data(normalized_classic, "normalized_classic")
            check_data(gebco_data, "gebco_data")
            if measurements is not None:
                check_data(measurements['depths'], "measurements depths")
            
            #print(f"GEBCO data shape before processing: {gebco_data.shape}")
            
            B = 1  # 批次大小
            T = len(normalized_classic)  # 时间步数
            H, W = gebco_data.shape[-2:]  # 空间维度
        
            # 检查normalized_classic维度
            # print(f"Classic results shape: {normalized_classic.shape}")
            if len(normalized_classic.shape) != 4:
                raise ValueError(f"Classic results维度错误: {normalized_classic.shape}, 应为[T, 1, H, W]")
            
            input_tensor = torch.zeros((B, T, self.config.num_components, H, W))
            # print(f"Input tensor initial shape: {input_tensor.shape}")  # 应该是 (1, 6, 5, 64, 64)
                
            # 1. 经典模型结果（保持正值）
            input_tensor[0, :, 0:1] = torch.from_numpy(normalized_classic)
        
            # 2. GEBCO数据（转为正值）
            gebco_tensor = torch.from_numpy(gebco_data)
            if len(gebco_tensor.shape) == 3:
                gebco_tensor = gebco_tensor.unsqueeze(1)
            elif len(gebco_tensor.shape) == 2:
                gebco_tensor = gebco_tensor.unsqueeze(0).unsqueeze(0).repeat(T, 1, 1, 1)
            input_tensor[0, :, 1:2] = gebco_tensor
            # print(f"Input tensor shape after GEBCO: {input_tensor.shape}")
        
            # 3. 测量点数据（海图数据，保持正值）
            if measurements is not None:
                depth_grid = torch.zeros((H, W))
                mask_grid = torch.zeros((H, W))
            
                coords = measurements['coordinates']  # 假设范围在[0,1]
                depths = measurements['depths']
            
                # 添加坐标验证
                print(f"Coordinates range: [{coords.min()}, {coords.max()}]")
                print(f"Number of measurement points: {len(depths)}")
            
                for depth, (y, x) in zip(depths, coords):
                    # 将[0,1]范围映射到[0,H-1]和[0,W-1]
                    i = min(int(y * (H-1)), H-1)
                    j = min(int(x * (W-1)), W-1)
                    depth_grid[i, j] = depth
                    mask_grid[i, j] = 1.0
            
                # 检查填充的点数
                print(f"Number of filled points in depth grid: {(depth_grid != 0).sum()}")
                print(f"Number of filled points in mask grid: {(mask_grid != 0).sum()}")
            
                # 扩展维度并重复
                input_tensor[0, :, 2:3] = depth_grid.unsqueeze(0).unsqueeze(0).repeat(T, 1, 1, 1)
                input_tensor[0, :, 3:4] = mask_grid.unsqueeze(0).unsqueeze(0).repeat(T, 1, 1, 1)
            
                print(f"Input tensor shape after measurements: {input_tensor.shape}")
        
            # 4. 统一掩码
            unified_mask = torch.ones((T, 1, H, W))  # 创建统一掩码
            #print(f"Unified mask shape: {unified_mask.shape}")
            #print(f"Unified mask value range: [{unified_mask.min()}, {unified_mask.max()}]")

            input_tensor[0, :, 4:5] = unified_mask
            #print(f"Final input tensor shape: {input_tensor.shape}")
            #print(f"Channel 5 (unified mask) value range: [{input_tensor[0, :, 4:5].min()}, {input_tensor[0, :, 4:5].max()}]")

            # 验证所有通道的值范围
            for i in range(5):
                channel_data = input_tensor[0, :, i:i+1]
                print(f"Channel {i+1} value range: [{channel_data.min():.4f}, {channel_data.max():.4f}]")
            
            return input_tensor.to(self.device)
        
        except Exception as e:
            logger.error(f"输入数据准备失败: {str(e)}")
            raise
            
    def _create_measurement_grid(self, depths, coordinates, shape):
        """创建测量点网格"""
        try:
            H, W = shape
            grid = np.zeros((2, H, W))  # [2, H, W] for depths and positions
            
            # 归一化坐标
            norm_coords = coordinates.copy()
            norm_coords[:, 0] = (norm_coords[:, 0] - norm_coords[:, 0].min()) / \
                               (norm_coords[:, 0].max() - norm_coords[:, 0].min()) * (H - 1)
            norm_coords[:, 1] = (norm_coords[:, 1] - norm_coords[:, 1].min()) / \
                               (norm_coords[:, 1].max() - norm_coords[:, 1].min()) * (W - 1)
            
            # 填充深度值
            for depth, (y, x) in zip(depths, norm_coords):
                i, j = int(y), int(x)
                grid[0, i, j] = depth
                grid[1, i, j] = 1  # 位置标记
                
            return grid
            
        except Exception as e:
            logger.error(f"测量点网格创建失败: {str(e)}")
            raise
            
    def _create_temporal_mask(self, measurements):
        """创建时间掩码"""
        try:
            mask = np.zeros((self.config.num_frames, 
                           self.config.image_size, 
                           self.config.image_size))
            
            # 根据测量点位置创建掩码
            if 'coordinates' in measurements:
                coords = measurements['coordinates']
                for t in range(self.config.num_frames):
                    for y, x in coords:
                        i = int(y * self.config.image_size)
                        j = int(x * self.config.image_size)
                        mask[t, i, j] = 1
                        
            return mask
            
        except Exception as e:
            logger.error(f"时间掩码创建失败: {str(e)}")
            raise

    def _transform_pretrain(self, x, mode='forward'):
        """数据尺度转换
        
        Args:
            x: 输入数据 [已经是mean-std标准化，范围在[-1,1]，陆地为1.5]
            mode: 'forward' 或 'inverse'
        """
        try:
            if mode == 'forward':
                # 数据已经是标准化的，只需要记录范围
                logger.info(f"预训练数据范围: [{x.min().item():.4f}, {x.max().item():.4f}]")
                return x
                
            elif mode == 'inverse':
                # 保持陆地标记值不变
                land_mask = (x == 1.5)
                # 其他值已经在[-1,1]范围内，不需要转换
                return x
                
        except Exception as e:
            logger.error(f"数据转换失败: {str(e)}")
            raise

    def _prepare_pretrain_data(self, classic_data, gebco_data):
        """准备预训练数据
        
        Args:
            classic_data: 经典模型结果 (已标准化到[0,1])
            gebco_data: GEBCO数据 (已标准化到[0,1])
        """
        try:
            B = 1  # 批次大小
            T = len(classic_data)  # 时间步数
            H, W = gebco_data.shape[-2:]  # 空间维度
            
            # 创建输入张量
            input_tensor = torch.zeros((B, T, self.config.num_components, H, W))
            
            # 1. 经典模型结果
            classic_tensor = torch.from_numpy(classic_data).unsqueeze(1)  # [T, 1, H, W]
            input_tensor[0, :, 0:1] = classic_tensor
            
            # 2. GEBCO数据
            gebco_tensor = torch.from_numpy(gebco_data)
            if len(gebco_tensor.shape) == 3:  # [T, H, W]
                gebco_tensor = gebco_tensor.unsqueeze(1)  # [T, 1, H, W]
            elif len(gebco_tensor.shape) == 2:  # [H, W]
                gebco_tensor = gebco_tensor.unsqueeze(0).unsqueeze(0).repeat(T, 1, 1, 1)  # [T, 1, H, W]
            input_tensor[0, :, 1:2] = gebco_tensor
            
            # 3. 其他通道置零（预训练阶段不使用海图数据）
            input_tensor[0, :, 2:4] = 0.0
            
            # 4. 统一掩码
            input_tensor[0, :, 4:5] = 1.0
            
            # 5. 应用数据转换
            transformed_tensor = self._transform_pretrain(input_tensor.to(self.device))
            
            logger.info(f"预训练数据准备完成:")
            logger.info(f"- 输入张量形状: {transformed_tensor.shape}")
            logger.info(f"- 经典模型通道范围: [{transformed_tensor[0, :, 0:1].min().item():.4f}, {transformed_tensor[0, :, 0:1].max().item():.4f}]")
            logger.info(f"- GEBCO通道范围: [{transformed_tensor[0, :, 1:2].min().item():.4f}, {transformed_tensor[0, :, 1:2].max().item():.4f}]")
            
            return transformed_tensor
            
        except Exception as e:
            logger.error(f"预训练数据准备失败: {str(e)}")
            raise

    def pretrain(self, classic_data, gebco_data, save_path):
        """执行预训练
        
        Args:
            classic_data: 经典模型结果 (已标准化到[-5,5])
            gebco_data: GEBCO数据 (已标准化到[-5,5])
            save_path: 模型保存路径
        """
        try:
            # 检查输入数据
            if np.isnan(classic_data).any() or np.isnan(gebco_data).any():
                raise ValueError("输入数据包含NaN值")
            
            # 检查模型初始化状态
            for name, param in self.model.named_parameters():
                if torch.all(param == 0):
                    logger.warning(f"{name} 权重全为0，重新初始化")
                    if 'weight' in name:
                        nn.init.xavier_normal_(param)
                    elif 'bias' in name:
                        nn.init.zeros_(param)
            
            # 准备预训练数据（包含数据转换）
            pretrain_tensor = self._prepare_pretrain_data(classic_data, gebco_data)
            
            # 执行预训练 - 使用新的预训练函数
            success = self._run_proper_pretrain(pretrain_tensor, num_epochs=1000)
            
            if not success:
                logger.error("预训练失败，无法继续")
                raise RuntimeError("预训练失败")
            
            # 创建保存目录（如果不存在）
            save_dir = os.path.dirname(save_path)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
                logger.info(f"创建模型保存目录: {save_dir}")
            
            # 保存模型前进行验证
            self._check_model_weights()
            torch.save(self.model.state_dict(), save_path)
            logger.info(f"预训练模型已保存到: {save_path}")
            
            # 记录当前模型配置
            logger.info(f"模型配置信息:")
            logger.info(f"  - 范围适配: {self.config.range_adaptation['enabled']}")
            logger.info(f"  - 混合激活函数: {self.config.range_adaptation['use_mixed_activation']}")
            logger.info(f"  - 陆地标记值: {self.config.range_adaptation['land_value']}")
            
            # 验证模型
            logger.info(f"验证模型...")
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
                logger.info(f"模型测试输出范围: [{test_score.min().item():.4f}, {test_score.max().item():.4f}]")
            
        except Exception as e:
            logger.error(f"预训练失败: {str(e)}")
            raise

    def load_pretrained(self, model_path: str):
        """加载预训练模型
        
        Args:
            model_path: 预训练模型权重文件路径
        """
        try:
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"预训练模型文件不存在: {model_path}")
            
            # 加载模型权重
            state_dict = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            self.model.eval()  # 设置为评估模式
            
            logger.info(f"成功加载预训练模型: {model_path}")
            # 添加模型权重检查
            if not self._check_model_weights():
                raise ValueError("预训练模型权重存在问题，请检查模型文件")
            
        except Exception as e:
            logger.error(f"加载预训练模型失败: {str(e)}")
            raise
            
    def conditional_sampling(self, measurements, measurement_coordinates, years):
        """执行条件采样"""
        try:
            # 准备条件采样数据
            input_tensor = self._prepare_condition_data(measurements, measurement_coordinates, years)
            
            # 执行采样
            results = self._run_sampling(input_tensor)
            
            # 检查结果（确保是tensor）
            if isinstance(results, np.ndarray):
                results_tensor = torch.from_numpy(results)
            else:
                results_tensor = results
            
            logger.info(f"条件采样结果形状: {results_tensor.shape}")
            if torch.isnan(results_tensor).any():
                nan_ratio = torch.isnan(results_tensor).sum().item() / results_tensor.numel()
                logger.warning(f"条件采样结果包含NaN值! NaN比例: {nan_ratio:.2%}")
                
                # 记录每个通道的NaN比例
                for c in range(results_tensor.shape[2]):
                    channel_nan_ratio = torch.isnan(results_tensor[..., c, :, :]).sum().item() / \
                                      (results_tensor.shape[0] * results_tensor.shape[1] * 
                                       results_tensor.shape[3] * results_tensor.shape[4])
                    logger.warning(f"通道 {c} 的NaN比例: {channel_nan_ratio:.2%}")
            
            valid_values = results_tensor[~torch.isnan(results_tensor)]
            if len(valid_values) > 0:
                logger.info(f"条件采样结果范围（排除NaN）: [{valid_values.min().item():.4f}, {valid_values.max().item():.4f}]")
            else:
                logger.error("条件采样结果中没有有效值！")
            
            return results
            
        except Exception as e:
            logger.error(f"条件采样失败: {str(e)}")
            raise

    def _prepare_condition_data(self, measurements, measurement_coordinates, years):
        """准备条件采样数据"""
        try:            
            H, W = self.config.image_size, self.config.image_size
            T = len(years)
            
            # 验证输入数据
            if not measurements.size or not measurement_coordinates.size:
                raise ValueError("测量数据为空")
            
            # 创建输入张量
            input_tensor = torch.zeros((1, T, self.config.num_components, H, W), 
                                     dtype=torch.float32, device=self.device)
            
            # 创建深度网格和掩码网格
            depth_grid = torch.zeros((H, W), device=self.device)
            mask_grid = torch.zeros((H, W), device=self.device)
            
            # 填充测量点
            for depth, (y, x) in zip(measurements, measurement_coordinates):
                i = min(int(y * (H-1)), H-1)
                j = min(int(x * (W-1)), W-1)
                
                # Explicitly convert numpy float to python float
                try:
                    depth_float = float(depth) 
                    depth_grid[i, j] = depth_float 
                except TypeError as e:
                    logger.error(f"无法将深度值 {depth} (类型: {type(depth)}) 转换为 float. Error: {e}")
                    # Handle error, e.g., skip this point or assign a default value
                    depth_grid[i, j] = 0.0 # Or some other default like stats['invalid_value']
                
                mask_grid[i, j] = 1.0 # Assigning 1.0 (python float) is okay
            
            # 记录原始掩码 - 新增
            original_mask = mask_grid.cpu().numpy()
            valid_observations = np.sum(original_mask > 0)
            logger.info(f"原始掩码观测点数量: {valid_observations}")
            
            # 总掩码点计数器
            total_mask_points = 0
            
            # 对所有时间步：
            for t, year in enumerate(years):
                # 海图深度数据（已标准化）
                input_tensor[0, t, 2:3] = depth_grid.unsqueeze(0)
                
                # 修改：为所有年份设置观测掩码，但强度不同
                # 2023年使用完整掩码
                if year == 2023:
                    input_tensor[0, t, 3:4] = mask_grid.unsqueeze(0)
                    total_mask_points += valid_observations
                    logger.info(f"{year}年使用完整掩码，{valid_observations}个观测点 (100%)")
                else:
                    # 其他年份使用部分掩码（随机选择部分观测点）
                    year_factor = (year - 2018) / 5.0  # 0.0到1.0
                    partial_mask = mask_grid.clone()
                    # 生成随机掩码（随机丢弃一部分观测点）
                    rand_mask = torch.rand_like(partial_mask) < year_factor
                    partial_mask = partial_mask * rand_mask
                    input_tensor[0, t, 3:4] = partial_mask.unsqueeze(0)
                    points_count = partial_mask.sum().item()
                    total_mask_points += points_count
                    logger.info(f"{year}年保留了{points_count:.0f}个观测点 ({year_factor*100:.0f}%)")
            
            # 统一掩码
            input_tensor[0, :, 4:5] = 1.0
            
            # 记录数据统计信息
            logger.info(f"条件采样数据准备完成:")
            logger.info(f"- 输入张量形状: {input_tensor.shape}")
            logger.info(f"- 深度通道范围: [{input_tensor[0, :, 2:3].min().item():.4f}, {input_tensor[0, :, 2:3].max().item():.4f}]")
            logger.info(f"- 掩码通道中1的总数: {total_mask_points:.0f}")
            logger.info(f"- 2023年的观测点数量: {valid_observations}")
            
            # 修改空间权重计算逻辑 - 继续使用原始逻辑
            spatial_weights = None
            
            # 计算基于原始掩码的空间权重
            def calculate_spatial_weights(mask, decay=None):
                """计算基于距离的空间权重"""
                if decay is None:
                    decay = self.config.spatial_decay
                    
                # 首先确保mask是numpy数组
                if isinstance(mask, torch.Tensor):
                    mask = mask.cpu().numpy()
                    
                # 确保mask是2D数组 - 关键修复
                mask = mask.squeeze()  # 移除所有大小为1的维度
                
                H, W = mask.shape  # 现在应该是2D的
                y_coords = np.linspace(0, 1, H)
                x_coords = np.linspace(0, 1, W)
                grid_y, grid_x = np.meshgrid(y_coords, x_coords, indexing='ij')
                
                # 获取观测点位置
                obs_indices = np.where(mask > 0)  # 确保条件明确
                obs_y, obs_x = obs_indices[0], obs_indices[1]  # 分别提取行和列索引
                
                # 确保有观测点
                if len(obs_y) == 0:
                    logger.warning(f"没有找到观测点，使用均匀权重")
                    return np.ones((H, W)) * 0.5
                    
                # 转换为归一化坐标
                obs_y = obs_y / (H - 1)
                obs_x = obs_x / (W - 1)
                
                # 初始化权重矩阵
                weights = np.zeros((H, W))
                
                # 计算每个观测点的影响
                for y, x in zip(obs_y, obs_x):
                    # 计算到当前观测点的距离
                    dist = np.sqrt((grid_y - y)**2 + (grid_x - x)**2)
                    # 使用指数衰减计算权重
                    weight = np.exp(-dist / decay)
                    # 更新权重矩阵（取最大值）
                    weights = np.maximum(weights, weight)
                
                # 归一化权重到[0.3, 1]范围，保证即使远离观测点也有一定权重
                if weights.max() > weights.min():
                    weights = 0.3 + 0.7 * (weights - weights.min()) / (weights.max() - weights.min())
                else:
                    weights.fill(0.5)
                    
                return weights

            # 计算时间权重
            def calculate_time_weights(num_frames, current_frame, decay=None):
                """计算基于时间的权重"""
                if decay is None:
                    decay = self.config.time_decay
                    
                weights = np.zeros(num_frames)
                for i in range(num_frames):
                    time_diff = abs(i - current_frame)
                    weights[i] = np.exp(-time_diff * decay)
                
                # 归一化到[0.2, 1]范围，确保历史数据也有影响
                weights = 0.2 + 0.8 * (weights - weights.min()) / (weights.max() - weights.min())
                return weights

            # 新的掩码权重计算逻辑
            # 只使用原始掩码计算一次空间权重
            spatial_weights = calculate_spatial_weights(original_mask)
            
            # 对每个时间步应用不同的时间权重
            for t in range(T):
                logger.info(f"处理时间步 {t}, 年份 {years[t]}")
                time_weights = calculate_time_weights(T, t)
                
                # 将权重应用到掩码中
                for frame in range(T):
                    mask_weights = spatial_weights * time_weights[frame]
                    input_tensor[0, frame, 3] = torch.from_numpy(mask_weights).to(self.device)
            
            return input_tensor
            
        except Exception as e:
            logger.error(f"准备条件采样数据失败: {str(e)}")
            raise

    def _run_diffusion(
        self, 
        input_tensor: Union[np.ndarray, torch.Tensor], 
        transform_fn=None
    ) -> np.ndarray:
        """运行扩散过程"""
        try:
            # 确保SDE参数在正确设备上
            if hasattr(self.sde, 'discrete_betas'):
                self.sde.discrete_betas = self.sde.discrete_betas.to(self.device)
            if hasattr(self.sde, 'alphas'):
                self.sde.alphas = self.sde.alphas.to(self.device)
            if hasattr(self.sde, 'alphas_cumprod'):
                self.sde.alphas_cumprod = self.sde.alphas_cumprod.to(self.device)

            # 定义网络前向传播函数
            def net_fn(x, t):
                B, T = x.shape[:2]
                
                # 准备默认掩码
                latent_mask = torch.ones([B, T, 1, 1, 1]).float().to(self.device)
                obs_mask = torch.zeros([B, T, 1, 1, 1]).float().to(self.device)
                
                # 尝试从input_tensor中提取掩码通道（但不记录任何日志）
                try:
                    if isinstance(input_tensor, torch.Tensor):
                        if input_tensor.shape[2] > 3:  # 检查是否有足够的通道
                            # 提取观测掩码通道(索引3)并检查是否有非零值
                            mask_channel = input_tensor[0, :, 3:4].to(self.device)
                            
                            # 对每个时间步分别处理
                            for t_idx in range(T):
                                if t_idx < mask_channel.shape[0]:  # 确保索引有效
                                    # 检查当前时间步是否有观测点
                                    if torch.any(mask_channel[t_idx] > 0):
                                        # 将时间步标记为有观测
                                        obs_mask[0, t_idx, 0, 0, 0] = 1.0
                except Exception as e:
                    # 错误时不记录任何日志，静默处理
                    pass
                
                # 监控输入状态
                if torch.isnan(x).any():
                    logger.error(f"net_fn输入x存在NaN - 形状: {x.shape}")
                    logger.error(f"时间步t: {t.item() if isinstance(t, torch.Tensor) else t}")
                    raise ValueError("net_fn输入包含NaN")

                with torch.no_grad():
                    score, _ = self.model(
                        x=x,
                        x0=x,
                        timesteps=t,
                        latent_mask=latent_mask,
                        obs_mask=obs_mask,
                        frame_indices=torch.arange(T, device=self.device).expand(B, -1)
                    )
                    
                    # 检查score是否全为NaN
                    if torch.isnan(score).any():
                        nan_locations = torch.isnan(score)
                        nan_count = nan_locations.sum().item()
                        logger.error(f"Score contains {nan_count} NaN values at timestep {t}")
                        
                        # 只有在存在非NaN值时才计算统计信息
                        valid_score = score[~torch.isnan(score)]
                        if valid_score.numel() > 0:
                            logger.error(f"Score统计: min={valid_score.min().item():.4f}, "
                                       f"max={valid_score.max().item():.4f}, "
                                       f"mean={valid_score.mean().item():.4f}")
                        else:
                            logger.error("Score全为NaN值!")
                        
                        # 记录NaN出现的位置
                        nan_indices = nan_locations.nonzero()
                        if len(nan_indices) > 0:
                            logger.error(f"首个NaN位置: {nan_indices[0].tolist()}")
                        
                        raise ValueError(f"Score contains NaN at timestep {t}")
                    
                    # 仅在debug模式下记录score范围
                    if self.config.stability.get('debug', False):
                        logger.debug(f"Score range at timestep {t}: [{score.min().item():.4f}, {score.max().item():.4f}]")
                
                return score

            # 确保输入张量在正确的设备上并进行类型转换
            if isinstance(input_tensor, np.ndarray):
                x = torch.from_numpy(input_tensor).to(self.device, non_blocking=True)
            else:
                x = input_tensor.to(self.device, non_blocking=True)
            
            if x.dtype != torch.float32:
                x = x.float()
            
            # 输入数据的详细检查
            if torch.isnan(x).any() or torch.isinf(x).any():
                nan_count = torch.isnan(x).sum().item()
                inf_count = torch.isinf(x).sum().item()
                logger.error(f"输入张量包含 {nan_count} 个NaN和 {inf_count} 个Inf值")
                logger.error(f"输入统计: min={x[~torch.isnan(x) & ~torch.isinf(x)].min().item():.4f}, "
                            f"max={x[~torch.isnan(x) & ~torch.isinf(x)].max().item():.4f}")
                raise ValueError("输入张量包含NaN或Inf值")
            
            # 清理GPU缓存
            torch.cuda.empty_cache()
            
            # 添加输入检查
            logger.info(f"扩散过程输入检查:")
            logger.info(f"- 输入张量形状: {x.shape}")
            if transform_fn:
                test_output = transform_fn(x.cpu().numpy())
                if isinstance(test_output, torch.Tensor):
                    logger.info(f"- 变换函数输出形状: {test_output.shape}")
                else:
                    logger.info(f"- 变换函数输出形状: {test_output.shape}")

            # 执行扩散过程
            try:
                # 使用torch.amp.autocast上下文管理器
                with torch.amp.autocast('cuda', enabled=False):
                    result_np, _ = complete_video_pc_dps(
                        self.config,
                        net_fn,
                        self.sde,
                        x.cpu().numpy(),
                        transform=transform_fn,
                        corrector=LangevinCorrector,
                        continuous=self.config.sampling['continuous'],
                        n_steps=self.config.sampling['num_steps'],
                        probability_flow=self.config.probability_flow,
                        snr=self.config.sampling['snr'],
                        eps=1e-12,
                        device=self.device
                    )
            except RuntimeError as e:
                logger.error(f"扩散过程运行时错误: {str(e)}")
                logger.error(f"- 配置信息:")
                logger.error(f"  - num_frames: {self.config.num_frames}")
                logger.error(f"  - num_components: {self.config.num_components}")
                logger.error(f"  - sampling steps: {self.config.sampling['num_steps']}")
                raise
            
            # 修改结果处理部分
            def process_results(result_tensor):
                # 首先检查是否是张量，如果是则转换为numpy
                if isinstance(result_tensor, torch.Tensor):
                    result_np = result_tensor.cpu().numpy()
                else:
                    result_np = result_tensor  # 已经是numpy数组

                # 识别陆地区域（使用配置中的land_value）
                land_mask = np.abs(result_np - self.config.land_value) < 0.1
                
                # 只记录数据范围，不做任何裁剪
                valid_data = result_np[~land_mask & ~np.isnan(result_np)]
                if len(valid_data) > 0:
                    orig_min, orig_max = np.min(valid_data), np.max(valid_data)
                    logger.info(f"原始结果范围: [{orig_min:.4f}, {orig_max:.4f}]")
                
                # 保持陆地区域的值不变
                result_np[land_mask] = self.config.land_value
        
            return result_np

            # 在扩散过程结束后处理结果
            result = result_np  # 先保存原始结果
            
            # 如果有变换函数，应用它
            if transform_fn is not None:
                result = transform_fn(result)
            
            # 最后进行一次process_results，但只是为了记录范围
            result = process_results(result)
            
            return result
            
        except Exception as e:
            logger.error(f"扩散过程失败: {str(e)}")
            raise

    def _run_sampling(self, input_tensor):
        """执行条件采样"""
        try:
            def adaptive_transform_sampling(x):
                """采样时的自适应数据转换函数 (移除硬裁剪)"""
                if isinstance(x, np.ndarray):
                    x = torch.from_numpy(x).to(self.device)

                # 安全地获取掩码
                try:
                    mask = torch.zeros((x.shape[-2], x.shape[-1]), device=self.device)
                    if isinstance(input_tensor, torch.Tensor) and len(input_tensor.shape) >= 5:
                        if input_tensor.shape[1] > 0 and input_tensor.shape[2] > 3:
                            mask = input_tensor[0, -1, 3:4].squeeze()
                except Exception as e:
                    logger.debug(f"获取掩码时出错: {e}，使用默认掩码")
                    mask = torch.zeros((x.shape[-2], x.shape[-1]), device=self.device)
            
                # 识别陆地
                land_mask = torch.abs(x - 1.5) < 0.1
                
                # 处理NaN和无穷值 (保持，但可以用更宽的范围，例如匹配x0_hat?)
                # 例如： torch.nan_to_num(x, nan=0.0, posinf=20.0, neginf=-20.0)
                # 或者暂时保持 +/- 5
                x = torch.where(land_mask, x, torch.nan_to_num(x, nan=0.0, posinf=5.0, neginf=-5.0))

                # 记录数据范围 - 只在第一次调用时记录
                if not hasattr(adaptive_transform_sampling, "logged"):
                    if torch.isfinite(x).any():
                        min_val = x[torch.isfinite(x)].min().item()
                        max_val = x[torch.isfinite(x)].max().item()
                        logger.info(f"采样数据范围 (transform_fn): [{min_val:.4f}, {max_val:.4f}]") # 修改日志标签
                    adaptive_transform_sampling.logged = True

                return x

            # 调整input_tensor
            try:
                nf = self.config.num_frames
                ns = self.config.sampling['num_steps']
                ol = self.config.sampling['overlap']
                b = max(1, int(ns // max(1, (nf - ol))) + 1)  # 防止除零
                ns_real = b * (nf - ol) + ol
            
                # 创建调整后的input_tensor
                adjusted_input_tensor = torch.zeros(
                    (1, ns_real, input_tensor.shape[2], input_tensor.shape[3], input_tensor.shape[4]), 
                    device=self.device
                )
            
                # 填充adjusted_input_tensor
                for i in range(b):
                    i_inv = b - i - 1
                    start_idx = i_inv * (nf - ol)
                    end_idx = min(start_idx + nf, ns_real)
                    if start_idx < ns_real:
                        src_end = min(nf, end_idx-start_idx)
                        if src_end > 0:
                            src_data = input_tensor[:, :src_end, :input_tensor.shape[2]]
                            adjusted_input_tensor[:, start_idx:start_idx+src_end, :input_tensor.shape[2]] = src_data
            
                # 复制辅助通道
                if input_tensor.shape[2] < adjusted_input_tensor.shape[2]:
                    aux_channels = input_tensor[:, 0:1, input_tensor.shape[2]:]
                    adjusted_input_tensor[:, :, input_tensor.shape[2]:] = aux_channels
            
                logger.info(f"调整input_tensor形状: {input_tensor.shape} -> {adjusted_input_tensor.shape}")
            except Exception as e:
                logger.error(f"调整input_tensor时出错: {e}")
                # 在出错时使用原始input_tensor
                adjusted_input_tensor = input_tensor
        
            # 执行扩散过程
            return self._run_diffusion(
                adjusted_input_tensor,
                transform_fn=adaptive_transform_sampling
            )
        
        except Exception as e:
            logger.error(f"条件采样运行失败: {str(e)}")
            raise

    def _run_proper_pretrain(self, input_tensor, num_epochs=300, batch_size=1, save_interval=50):
        """使用标准扩散损失函数执行预训练"""
        try:
            # 确保输入是tensor并在正确设备上
            if isinstance(input_tensor, np.ndarray):
                x = torch.from_numpy(input_tensor).to(self.device)
            else:
                x = input_tensor.to(self.device)
            
            if x.dtype != torch.float32:
                x = x.float()
            
            # 初始化优化器
            optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)
            
            # 添加学习率调度器
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
            
            # 训练循环
            for epoch in tqdm(range(num_epochs), desc="预训练进度"):
                self.model.train()
                optimizer.zero_grad()
                
                # 随机生成时间步
                t = torch.rand(batch_size, device=self.device) * (1.0 - 1e-5) + 1e-5
                
                # 添加随机噪声
                mean, std = self.sde.marginal_prob(x, t)
                z = torch.randn_like(x)
                perturbed_x = mean + std[:, None, None, None, None] * z
                
                # 准备模型输入参数
                B, T = perturbed_x.shape[:2]
                latent_mask = torch.ones([B, T, 1, 1, 1]).float().to(self.device)
                obs_mask = torch.zeros([B, T, 1, 1, 1]).float().to(self.device)
                
                # 前向传播，计算损失
                score, _ = self.model(
                    x=perturbed_x,
                    x0=perturbed_x,  # 在预训练阶段，x0 和 x 相同
                    timesteps=t,
                    latent_mask=latent_mask,
                    obs_mask=obs_mask,
                    frame_indices=torch.arange(T, device=self.device).expand(B, -1)
                )
                
                # 计算损失：预测噪声 z
                losses = torch.square(score * std[:, None, None, None, None] + z)
                loss = torch.mean(losses)
                
                # 监控数据范围（每50个epoch）
                if epoch % 25 == 0 or epoch == num_epochs - 1:
                    print(f"数据监控[第{epoch}步]: 范围=[{perturbed_x.min().item():.4f}, {perturbed_x.max().item():.4f}], "
                          f"均值={perturbed_x.mean().item():.4f}, 标准差={perturbed_x.std().item():.4f}, "
                          f"loss={loss.item():.6f}")
                
                # 反向传播和优化
                loss.backward()
                
                # 梯度裁剪以增加稳定性
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                optimizer.step()
                scheduler.step()
                
                # 定期保存检查点
                if (epoch + 1) % save_interval == 0 or epoch == num_epochs - 1:
                    # 记录当前模型配置
                    logger.info(f"模型配置信息:")
                    logger.info(f"  - 范围适配: {self.config.range_adaptation['enabled']}")
                    logger.info(f"  - 混合激活函数: {self.config.range_adaptation['use_mixed_activation']}")
                    logger.info(f"  - 陆地标记值: {self.config.range_adaptation['land_value']}")
                    
                    # 验证模型
                    self.model.eval()
                    
                    # 使用一个简单的样本测试模型输出
                    with torch.no_grad():
                        test_t = torch.ones(1, device=self.device) * 0.5
                        test_x = torch.randn_like(x)
                        test_score, _ = self.model(
                            x=test_x,
                            x0=test_x,
                            timesteps=test_t,
                            latent_mask=latent_mask,
                            obs_mask=obs_mask,
                            frame_indices=torch.arange(T, device=self.device).expand(B, -1)
                        )
                        logger.info(f"模型测试输出范围: [{test_score.min().item():.4f}, {test_score.max().item():.4f}]")
            
            return True
        
        except Exception as e:
            logger.error(f"预训练失败: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return False

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
        """获取得分函数"""
        def score_fn(x, t):
            """计算得分函数
            Args:
                x: 输入张量 [B, T, C, H, W]
                t: 时间步
            """
            with torch.set_grad_enabled(self.config.requires_grad):
                B = x.shape[0]
                latent_mask = torch.ones([B, self.config.num_frames, 1, 1, 1]).float().to(self.device)
                obs_mask = torch.zeros([B, self.config.num_frames, 1, 1, 1]).float().to(self.device)
                frame_indices = torch.stack([torch.arange(self.config.num_frames) for _ in range(B)]).to(self.device)
                
                # 修改调用方式，使用字典传递额外参数
                score, _ = self.model(
                    x, 
                    timesteps=t,
                    latent_mask=latent_mask,
                    obs_mask=obs_mask,
                    frame_indices=frame_indices
                )
                return score
        
        return score_fn