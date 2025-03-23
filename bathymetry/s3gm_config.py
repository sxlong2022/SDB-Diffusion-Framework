"""S3GM模型配置文件"""
from dataclasses import dataclass, field
from typing import Tuple, Optional, List
import os
import numpy as np
import torch
import yaml

@dataclass
class S3GMConfig:
    """S3GM模型配置类"""
    # 基础参数
    num_components: int = 5  # 1(经典) + 1(GEBCO) + 2(海图) + 1(掩码)
    image_size: int = 64
    num_frames: int = 6     # 2018-2023
    version: str = "bathymetry_v0"
    data: str = "bathymetry"
    
    # 数据处理参数
    land_value: float = 1.5  # 添加这一行，与preprocessor.py保持一致
    
    # 模型架构参数
    model_channels: int = 128
    num_res_blocks: int = 5   
    attention_resolutions: Tuple[int, ...] = (8, 16, 32)
    dropout: float = 0.2
    channel_mult: Tuple[int, ...] = (1, 2, 4, 4)
    dims: int = 2
    num_heads: int = 8
    use_checkpoint: bool = True
    use_scale_shift_norm: bool = True
    
    # 通道模态配置
    channel_modal: List[int] = field(
        default_factory=lambda: [1, 1, 2, 1]  # [经典模型, GEBCO, 海图(深度+掩码), 统一掩码]
    )
    num_modals: int = 1
    
    # 扩散模型参数
    beta_min: float = 0.001
    beta_max: float = 5.0
    num_scales: int = 300
    ema_rate: float = 0.999
    predictor: str = "reverse_diffusion"
    corrector: str = "langevin"
    probability_flow: bool = True
    
    # 采样参数
    sampling: dict = field(default_factory=lambda: {
        'alpha': 0.85,
        'inner_loop': 100,
        'snr': 0.2,
        'continuous': True,
        'num_steps': 20,
        'overlap': 2
    })
    
    # RPE相关参数
    beta: int = 32
    use_rpe_net: bool = True
    time_embed_dim: int = 256
        
    # 时序处理参数
    time_decay: float = 0.12
    spatial_decay: float = 0.05
    
    # 性能优化参数
    use_fp16: bool = False
    use_amp: bool = False
    memory_efficient: bool = True
    gradient_checkpointing: bool = True
    use_reentrant: bool = False  # 添加checkpoint重入参数
    requires_grad: bool = True   # 新增，控制是否在训练时计算梯度
    
    # EMA相关参数
    use_ema: bool = True
    ema_decay: float = 0.9999
    
    # SDE参数
    sigma_min: float = 0.002
    sigma_max: float = 0.2
    
    # 数值稳定性参数
    eps: float = 1.0e-12
    
    # 采样相关参数
    num_samples: int = 1
    num_samples_train: int = 1000
    num_samples_val: int = 1000
    train_split: float = 0.9
    
    # 物理约束参数
    physics_guide: bool = False  # 是否启用物理引导
    
    # 稳定性参数
    stability: dict = field(default_factory=lambda: {
        'grad_clip': 1.0,
        'check_grad_interval': 5,
        'debug': True  # 添加debug选项控制日志输出
    })
    
    # 添加新的裁剪参数
    clipping: dict = field(default_factory=lambda: {
        'use_dynamic': True,     # 是否使用动态裁剪
        'percentile_range': [1, 99],  # 百分位数范围
        'expansion_factor': 1.2,  # 扩展系数
        'land_threshold': 0.1    # 陆地判定阈值
    })
    
    # 添加新的权重参数
    weights: dict = field(default_factory=lambda: {
        'spatial_min': 0.3,      # 空间权重最小值
        'spatial_max': 1.0,      # 空间权重最大值
        'temporal_min': 0.2,     # 时间权重最小值
        'temporal_max': 1.0      # 时间权重最大值
    })
    
    def __post_init__(self):
        """初始化后的处理"""
        # 不需要调用super().__post_init__()
        # 直接进行必要的初始化
        if isinstance(self.attention_resolutions, list):
            self.attention_resolutions = tuple(self.attention_resolutions)
        if isinstance(self.channel_mult, list):
            self.channel_mult = tuple(self.channel_mult)
        
        # 初始化VESDE参数
        self.discrete_betas = torch.linspace(self.beta_min/self.num_scales, self.beta_max/self.num_scales, self.num_scales)
        self.alphas = 1. - self.discrete_betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        
        # 其他检查保持不变
        assert self.num_components > 0, "num_components必须大于0"
        assert self.image_size > 0, "image_size必须大于0"
        assert self.num_frames > 0, "num_frames必须大于0"
        assert len(self.channel_modal) > 0, "channel_modal不能为空"
        assert self.eps > 0, "eps必须大于0"
        
        # 验证SDE参数
        assert 0 < self.sigma_min < self.sigma_max, "sigma_min必须小于sigma_max且都为正数"
        
        # 验证采样参数
        assert self.sampling['inner_loop'] > 0, "inner_loop必须为正数"
        assert 0 < self.sampling['snr'] < 1, "snr必须在0到1之间"
        
        # 计算时间权重
        self.time_weights = [
            np.exp(-self.time_decay * i) for i in range(self.num_frames)
        ]
        
        # EMA参数验证
        assert 0.99 <= self.ema_decay < 1, "EMA衰减率必须在0.99到1之间"
        
    @classmethod
    def from_dict(cls, config_dict: dict) -> 'S3GMConfig':
        """从字典创建配置"""
        return cls(**{k: v for k, v in config_dict.items() if hasattr(cls, k)})

    @classmethod
    def from_yaml(cls, path: str) -> 'S3GMConfig':
        """从YAML文件加载配置"""
        with open(path, 'r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f)
        return cls.from_dict(config_dict)

    def __getattr__(self, name):
        """处理参数访问"""
        if name in self.sampling:
            return self.sampling[name]
        elif name in self.stability:  # 添加对stability字典的访问
            return self.stability[name]
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")

def get_default_config() -> S3GMConfig:
    """获取默认配置"""
    return S3GMConfig()

def load_config(config_path: Optional[str] = None) -> S3GMConfig:
    """从文件加载配置"""
    if config_path is None:
        config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                'configs', 's3gm_default.yaml')
    else:
        if not config_path.endswith('.yaml'):
            config_path = f"{config_path}.yaml"
        if not os.path.dirname(config_path):
            config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                    'configs', config_path)
    
    try:
        return S3GMConfig.from_yaml(config_path)
    except Exception as e:
        print(f"加载配置文件失败: {str(e)}")
        print("使用默认配置")
        return S3GMConfig()

def validate_config_consistency():
    """验证配置文件与默认值的一致性"""
    yaml_config = S3GMConfig.from_yaml('configs/s3gm_default.yaml')
    default_config = S3GMConfig()
    
    for field in fields(S3GMConfig):
        yaml_value = getattr(yaml_config, field.name)
        default_value = getattr(default_config, field.name)
        if yaml_value != default_value:
            logger.warning(
                f"配置不一致: {field.name}\n"
                f"  yaml值: {yaml_value}\n"
                f"  默认值: {default_value}"
            )