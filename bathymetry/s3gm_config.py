"""S3GM模型配置文件"""
from dataclasses import dataclass, field
from typing import Tuple, Optional, List, Dict, Any
import os
import numpy as np
import torch
import yaml

@dataclass
class S3GMConfig:
    # 模型基本参数
    num_components: int = 5  # 1(经典) + 1(GEBCO) + 2(海图) + 1(掩码)
    image_size: int = 64
    num_frames: int = 6     # 2018-2023
    version: str = "bathymetry_v0"
    data: str = "bathymetry"
    parameterization: str = "v"

    # 模型架构参数
    model_channels: int = 128
    num_res_blocks: int = 4
    attention_resolutions: Tuple[int, ...] = (4, 8, 16)
    dropout: float = 0.2
    channel_mult: Tuple[int, ...] = (1, 2, 4, 4)
    dims: int = 2
    num_heads: int = 8
    use_checkpoint: bool = True
    use_scale_shift_norm: bool = True

    # 通道模态配置
    channel_modal: List[int] = field(default_factory=lambda: [1, 1, 2, 1])  # [经典模型, GEBCO, 海图(深度+掩码), 统一掩码]
    num_modals: int = 1

    # SDE配置
    sde_type: str = 'vpsde'
    beta_min: float = 0.1
    beta_max: float = 10.0
    num_scales: int = 1000

    # 扩散模型参数
    ema_rate: float = 0.999
    predictor: str = "reverse_diffusion"
    corrector: str = "langevin"
    probability_flow: bool = True

    # 采样参数
    sampling: Dict[str, Any] = field(default_factory=lambda: {
        'alpha': 1,
        'gamma1': 5,
        'gamma2': 5,
        'gamma_spatial': 0.5,
        'inner_loop': 5,
        'snr': 0.05,
        'continuous': True,
        'num_steps': 3,
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
    use_fp16: bool = True
    use_amp: bool = True
    memory_efficient: bool = True
    gradient_checkpointing: bool = True
    use_reentrant: bool = False
    batch_size_override: int = 1
    requires_grad: bool = True

    # EMA相关参数
    use_ema: bool = True
    ema_decay: float = 0.9999
    
    # VESDE参数
    sigma_min: float = 0.002
    sigma_max: float = 10
    
    # 数值稳定性参数
    eps: float = 1.0e-12

    # 采样相关参数
    num_samples: int = 1
    num_samples_train: int = 1000
    num_samples_val: int = 1000
    train_split: float = 0.9

    # 数据处理参数
    land_value: float = 1.5

    # 物理约束参数
    physics_guide: bool = False

    # 稳定性参数
    stability: Dict[str, Any] = field(default_factory=lambda: {
        'grad_clip': 1.0,
        'check_grad_interval': 5,
        'debug': True,
        'monitor_data_ranges': True,
        'x0_hat_clamp': True,
        'score_clamp_range': 20.0,
        'gradient_clip_dps': 10.0
    })

    # 裁剪参数
    clipping: Dict[str, Any] = field(default_factory=lambda: {
        'use_dynamic': True,
        'percentile_range': [1, 99],
        'expansion_factor': 1.2,
        'land_threshold': 0.1
    })

    # 权重参数
    weights: Dict[str, Any] = field(default_factory=lambda: {
        'spatial_min': 0.3,
        'spatial_max': 1.0,
        'temporal_min': 0.2,
        'temporal_max': 1.0
    })

    # 范围适配配置
    range_adaptation: Dict[str, Any] = field(default_factory=lambda: {
        'enabled': True,
        'use_mixed_activation': True,
        'land_value': 1.5,
        'init_scale': 1.0,
        'init_shift': 0.0,
        'attention_scale_factor': 0.8
    })

    # 输入通道权重
    input_weights: Dict[str, Any] = field(default_factory=lambda: {
        'classic': 0.7,
        'gebco': 0.3
    })

    def __post_init__(self):
        # 类型和结构修正
        if isinstance(self.attention_resolutions, list):
            self.attention_resolutions = tuple(self.attention_resolutions)
        if isinstance(self.channel_mult, list):
            self.channel_mult = tuple(self.channel_mult)
        # 其他一致性检查
        assert self.num_components > 0
        assert self.image_size > 0
        assert self.num_frames > 0
        assert len(self.channel_modal) > 0
        assert self.eps > 0
        assert self.sampling['inner_loop'] > 0
        assert 0 < self.sampling['snr'] < 1
        assert 0 < self.ema_decay < 1

    @classmethod
    def from_dict(cls, config_dict: dict) -> 'S3GMConfig':
        instance = cls()
        for k, v in config_dict.items():
            if hasattr(instance, k):
                current_value = getattr(instance, k)
                if isinstance(current_value, dict) and isinstance(v, dict):
                    current_value.update(v)
                elif k == 'parameterization' and isinstance(v, str):
                    setattr(instance, k, v)
                else:
                    setattr(instance, k, v)
        return instance

    @classmethod
    def from_yaml(cls, path: str) -> 'S3GMConfig':
        with open(path, 'r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f)
        return cls.from_dict(config_dict)

    def __getattr__(self, name):
        if name in self.sampling:
            return self.sampling[name]
        elif name in self.stability:
            return self.stability[name]
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")

def get_default_config() -> S3GMConfig:
    return S3GMConfig()

def load_config(config_path: Optional[str] = None) -> S3GMConfig:
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
        print(f"INFO - 尝试加载配置文件: {config_path}")
        if not os.path.exists(config_path):
            print(f"ERROR - 配置文件不存在于路径: {config_path}")
        return S3GMConfig.from_yaml(config_path)
    except Exception as e:
        print(f"ERROR - 加载配置文件 '{config_path}' 失败: {str(e)}")
        import traceback
        print(traceback.format_exc())
        print("INFO - 回退到使用默认配置")
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