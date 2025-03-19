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
            
            # 4. 初始化模型
            self.model = UNetVideoModel(
                in_channels=self.config.num_components,
                model_channels=self.config.model_channels,
                out_channels=self.config.num_components,
                num_res_blocks=self.config.num_res_blocks,
                attention_resolutions=self.config.attention_resolutions,
                dropout=self.config.dropout,
                channel_mult=self.config.channel_mult,
                dims=self.config.dims,
                num_heads=self.config.num_heads,
                use_rpe_net=self.config.use_rpe_net
            ).to(self.device)
            
            # 5. 初始化SDE
            sde_config = {
                'beta_min': self.config.beta_min,
                'beta_max': self.config.beta_max,
                'N': self.config.num_scales,
                'sigma_min': self.config.sigma_min,
                'sigma_max': self.config.sigma_max
            }
            logger.info(f"初始化SDE配置: {sde_config}")
            self.sde = VESDE(config=sde_config)
            logger.info(f"SDE初始化后参数: sigma_min={self.sde.sigma_min}, sigma_max={self.sde.sigma_max}, num_scales={self.config.num_scales}")
            
            # 添加参数验证
            if hasattr(self.sde, 'sigma_max') and self.sde.sigma_max != self.config.sigma_max:
                logger.warning(f"SDE初始化后sigma_max不匹配: 配置值={self.config.sigma_max}, 实际值={self.sde.sigma_max}")
                # 强制设置正确的值
                self.sde.sigma_max = self.config.sigma_max
                logger.info(f"已强制更新SDE的sigma_max为配置值: {self.config.sigma_max}")
            
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
            classic_data: 经典模型结果 (已标准化到[0,1])
            gebco_data: GEBCO数据 (已标准化到[0,1])
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
            
            # 执行预训练
            self._run_pretrain(pretrain_tensor)
            
            # 创建保存目录（如果不存在）
            save_dir = os.path.dirname(save_path)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
                logger.info(f"创建模型保存目录: {save_dir}")
            
            # 保存模型前进行验证
            self._check_model_weights()
            torch.save(self.model.state_dict(), save_path)
            logger.info(f"预训练模型已保存到: {save_path}")
            
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
        """准备条件采样数据
        
        Args:
            measurements: 已标准化的海图深度数据
            measurement_coordinates: 测量点坐标（[0,1]范围）
            years: 目标年份列表
        """
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
                depth_grid[i, j] = depth
                mask_grid[i, j] = 1.0
            
            # 对所有时间步：
            for t, year in enumerate(years):
                # 海图深度数据（已标准化）
                input_tensor[0, t, 2:3] = depth_grid.unsqueeze(0)
                # 2023年的掩码
                if year == 2023:
                    input_tensor[0, t, 3:4] = mask_grid.unsqueeze(0)
            
            # 统一掩码
            input_tensor[0, :, 4:5] = 1.0
            
            # 记录数据统计信息
            logger.info(f"条件采样数据准备完成:")
            logger.info(f"- 输入张量形状: {input_tensor.shape}")
            logger.info(f"- 深度通道范围: [{input_tensor[0, :, 2:3].min().item():.4f}, {input_tensor[0, :, 2:3].max().item():.4f}]")
            logger.info(f"- 掩码通道中1的数量: {(input_tensor[0, :, 3:4] > 0).sum().item()}")
            
            return input_tensor
            
        except Exception as e:
            logger.error(f"条件采样数据准备失败: {str(e)}")
            raise

    def _run_diffusion(
        self, 
        input_tensor: Union[np.ndarray, torch.Tensor], 
        transform_fn=None
    ) -> np.ndarray:
        """执行扩散过程"""
        try:
            # 确保SDE参数在正确设备上
            if hasattr(self.sde, 'discrete_sigmas'):
                self.sde.discrete_sigmas = self.sde.discrete_sigmas.to(self.device)
            if hasattr(self.sde, 'discrete_betas'):
                self.sde.discrete_betas = self.sde.discrete_betas.to(self.device)

            # 定义网络前向传播函数
            def net_fn(x, t):
                B, T = x.shape[:2]
                
                # 监控输入状态
                if torch.isnan(x).any():
                    logger.error(f"net_fn输入x存在NaN - 形状: {x.shape}")
                    logger.error(f"时间步t: {t.item() if isinstance(t, torch.Tensor) else t}")
                    raise ValueError("net_fn输入包含NaN")
                
                # 准备掩码和帧索引
                latent_mask = torch.ones([T, 1, 1, 1]).float().to(self.device)
                obs_mask = torch.zeros([T, 1, 1, 1]).float().to(self.device)
                frame_indices = torch.arange(T, device=self.device)
                
                latent_mask = latent_mask.expand(B, -1, -1, -1, -1)
                obs_mask = obs_mask.expand(B, -1, -1, -1, -1)
                frame_indices = frame_indices.expand(B, -1)

                with torch.no_grad():
                    score, _ = self.model(
                        x=x,
                        x0=x,
                        timesteps=t,
                        latent_mask=latent_mask,
                        obs_mask=obs_mask,
                        frame_indices=frame_indices
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
            
            # 检查结果
            if np.isnan(result_np).any():
                nan_count = np.isnan(result_np).sum()
                logger.error(f"扩散结果包含 {nan_count} 个NaN值")
                logger.error(f"结果统计: min={np.nanmin(result_np):.4f}, "
                           f"max={np.nanmax(result_np):.4f}")
                raise ValueError("扩散结果包含NaN")
        
            return result_np
            
        except Exception as e:
            logger.error(f"扩散过程执行失败: {str(e)}")
            if 'x' in locals():
                logger.error(f"输入张量形状: {x.shape}")
                logger.error(f"输入范围: [{x.min().item():.4f}, {x.max().item():.4f}]")
            logger.error(f"channel_modal配置: {self.config.channel_modal}")
            raise

    def _run_pretrain(self, input_tensor):
        """执行预训练"""
        try:
            def adaptive_transform_pretrain(x):
                """预训练时的自适应数据转换函数"""
                if isinstance(x, np.ndarray):
                    x = torch.from_numpy(x).to(self.device)
            
                # 获取掩码（第4个通道）
                mask = input_tensor[0, -1, 3:4]  # 使用2023年的掩码
                land_mask = torch.abs(x - self.config.land_value) < 0.1
                sea_mask = ~land_mask
            
                if sea_mask.any():
                    sea_data = x[sea_mask]
                
                    # 使用分位数统计代替均值和标准差
                    q25 = torch.quantile(sea_data, 0.25)
                    q75 = torch.quantile(sea_data, 0.75)
                    iqr = q75 - q25
                    median = torch.median(sea_data)
                
                    # 计算空间相关性权重
                    H, W = x.shape[-2:]
                    y_coords = torch.linspace(0, 1, H, device=self.device)
                    x_coords = torch.linspace(0, 1, W, device=self.device)
                    y_grid, x_grid = torch.meshgrid(y_coords, x_coords, indexing='ij')
                    coords = torch.stack([y_grid, x_grid], dim=-1)
                
                    # 计算每个点与观测点的距离权重
                    obs_points = (mask > 0).nonzero(as_tuple=True)
                    if len(obs_points[0]) > 0:
                        obs_coords = torch.stack([
                            obs_points[0].float() / (H-1),
                            obs_points[1].float() / (W-1)
                        ], dim=-1)
                    
                        # 计算空间权重
                        coords_flat = coords.reshape(-1, 2)
                        distances = torch.cdist(coords_flat, obs_coords)
                        min_distances = distances.min(dim=1)[0]
                    
                        # 使用自适应空间衰减
                        spatial_decay = self.config.spatial_decay * (1 + torch.exp(-min_distances))
                        spatial_weights = torch.exp(-spatial_decay * min_distances).reshape(H, W)
                    
                        # 根据空间权重调整阈值
                        threshold_factor = torch.lerp(
                            torch.tensor(2.0, device=self.device),
                            torch.tensor(3.0, device=self.device),
                            spatial_weights
                        )
                    
                        # 计算动态阈值
                        curr_min = median - threshold_factor * iqr
                        curr_max = median + threshold_factor * iqr
                    
                        # 应用渐进式缩放
                        scale_factor = torch.lerp(
                            torch.tensor(0.95, device=self.device),
                            torch.tensor(1.0, device=self.device),
                            spatial_weights
                        )
                    
                        # 应用自适应转换
                        x = torch.where(land_mask, x,
                                      torch.where(x < curr_min,
                                                curr_min + (x - curr_min) * scale_factor,
                                                torch.where(x > curr_max,
                                                          curr_max + (x - curr_max) * scale_factor,
                                                          x)))
                    
                        # 应用空间权重平滑
                        x = x * (1 - spatial_weights) + x * spatial_weights
            
                # 处理NaN和无穷值
                x = torch.where(land_mask, x, torch.nan_to_num(x, nan=0.0))
            
                # 添加梯度裁剪
                if self.config.stability.get('grad_clip', None):
                    x = torch.clamp(x, -self.config.stability['grad_clip'], 
                                  self.config.stability['grad_clip'])
            
                return x

            # 在执行扩散过程前调整input_tensor
            nf = self.config.num_frames
            ns = self.config.sampling['num_steps']
            ol = self.config.sampling['overlap']
            b = int(ns // (nf - ol) + 1)
            ns_real = b * (nf - ol) + ol
        
            # 创建调整后的input_tensor
            adjusted_input_tensor = torch.zeros(
                (1, ns_real, input_tensor.shape[2], input_tensor.shape[3], input_tensor.shape[4]), 
                device=self.device
            )
        
            # 按照x_to_sample的逻辑填充adjusted_input_tensor
            for i in range(b):
                i_inv = b - i - 1
                start_idx = i_inv * (nf - ol)
                end_idx = start_idx + nf
                # 复制主要通道数据
                adjusted_input_tensor[:, start_idx:end_idx, :input_tensor.shape[2]] = input_tensor[:, :, :input_tensor.shape[2]]
        
            # 复制掩码等辅助通道
            adjusted_input_tensor[:, :, input_tensor.shape[2]:] = input_tensor[:, 0:1, input_tensor.shape[2]:]
        
            logger.info(f"预训练阶段调整input_tensor形状: {input_tensor.shape} -> {adjusted_input_tensor.shape}")
        
            # 执行扩散过程
            return self._run_diffusion(
                adjusted_input_tensor,
                transform_fn=adaptive_transform_pretrain
            )
        
        except Exception as e:
            logger.error(f"预训练运行失败: {str(e)}")
            raise

    def _run_sampling(self, input_tensor):
        """执行条件采样"""
        try:
            def adaptive_transform_sampling(x):
                """采样时的自适应数据转换函数"""
                if isinstance(x, np.ndarray):
                    x = torch.from_numpy(x).to(self.device)
            
                # 获取掩码（第4个通道）
                mask = input_tensor[0, -1, 3:4]  # 使用2023年的掩码
                land_mask = torch.abs(x - self.config.land_value) < 0.1
                sea_mask = ~land_mask
            
                if sea_mask.any():
                    sea_data = x[sea_mask]
                
                    # 使用分位数统计
                    q25 = torch.quantile(sea_data, 0.25)
                    q75 = torch.quantile(sea_data, 0.75)
                    iqr = q75 - q25
                    median = torch.median(sea_data)
                
                    # 计算空间相关性权重
                    H, W = x.shape[-2:]
                    y_coords = torch.linspace(0, 1, H, device=self.device)
                    x_coords = torch.linspace(0, 1, W, device=self.device)
                    y_grid, x_grid = torch.meshgrid(y_coords, x_coords, indexing='ij')
                    coords = torch.stack([y_grid, x_grid], dim=-1)
                
                    # 计算每个点与观测点的距离权重
                    obs_points = (mask > 0).nonzero(as_tuple=True)
                    if len(obs_points[0]) > 0:
                        obs_coords = torch.stack([
                            obs_points[0].float() / (H-1),
                            obs_points[1].float() / (W-1)
                        ], dim=-1)
                    
                        # 改进的空间权重计算
                        coords_flat = coords.reshape(-1, 2)
                        distances = torch.cdist(coords_flat, obs_coords)
                        min_distances = distances.min(dim=1)[0]
                    
                        # 计算观测点密度
                        density = torch.exp(-min_distances).mean()
                    
                        # 根据密度调整空间衰减率
                        adaptive_decay = self.config.spatial_decay * (1 + torch.exp(-density))
                        spatial_weights = torch.exp(-adaptive_decay * min_distances).reshape(H, W)
                    
                        # 计算时间权重（考虑年份差异）
                        time_weights = torch.tensor(
                            [np.exp(-self.config.time_decay * abs(i - len(self.config.time_range['sentinel']) + 1))
                             for i in range(len(self.config.time_range['sentinel']))],
                            device=self.device
                        )
                    
                        # 结合时空权重
                        combined_weights = spatial_weights * time_weights.view(-1, 1, 1)
                    
                        # 动态阈值调整
                        base_threshold = 2.0
                        max_threshold = 3.0
                        threshold_factor = torch.lerp(
                            torch.tensor(base_threshold, device=self.device),
                            torch.tensor(max_threshold, device=self.device),
                            combined_weights
                        )
                    
                        # 计算自适应阈值
                        curr_min = median - threshold_factor * iqr
                        curr_max = median + threshold_factor * iqr
                    
                        # 渐进式缩放因子
                        scale_factor = torch.lerp(
                            torch.tensor(0.9, device=self.device),
                            torch.tensor(1.0, device=self.device),
                            combined_weights
                        )
                    
                        # 应用自适应转换
                        x = torch.where(land_mask, x,
                                      torch.where(x < curr_min,
                                                curr_min + (x - curr_min) * scale_factor,
                                                torch.where(x > curr_max,
                                                          curr_max + (x - curr_max) * scale_factor,
                                                          x)))
                    
                        # 应用组合权重平滑
                        x = x * (1 - combined_weights) + x * combined_weights
            
                # 处理NaN和无穷值
                x = torch.where(land_mask, x, torch.nan_to_num(x, nan=0.0))
            
                # 添加梯度裁剪
                if self.config.stability.get('grad_clip', None):
                    x = torch.clamp(x, -self.config.stability['grad_clip'], 
                                  self.config.stability['grad_clip'])
            
                return x

            # 调整input_tensor
            nf = self.config.num_frames
            ns = self.config.sampling['num_steps']
            ol = self.config.sampling['overlap']
            b = int(ns // (nf - ol) + 1)
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
                end_idx = start_idx + nf
                adjusted_input_tensor[:, start_idx:end_idx, :input_tensor.shape[2]] = input_tensor[:, :, :input_tensor.shape[2]]
        
            adjusted_input_tensor[:, :, input_tensor.shape[2]:] = input_tensor[:, 0:1, input_tensor.shape[2]:]
        
            logger.info(f"调整input_tensor形状: {input_tensor.shape} -> {adjusted_input_tensor.shape}")
        
            # 执行扩散过程
            return self._run_diffusion(
                adjusted_input_tensor,
                transform_fn=adaptive_transform_sampling
            )
        
        except Exception as e:
            logger.error(f"条件采样运行失败: {str(e)}")
            raise

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
                score = torch.clamp(score, -1.0, 1.0)  # 添加梯度裁剪
                return score
        
        return score_fn