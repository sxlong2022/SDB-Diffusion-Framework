"""
Various utilities for neural networks.
"""

import math

import torch as th
import torch.nn as nn


# PyTorch 1.7 has SiLU, but we support PyTorch 1.5.
class SiLU(nn.Module):
    def forward(self, x):
        return x * th.sigmoid(x)


# 新增：混合激活函数 - 适应更广范围的输入值
class MixedRangeActivation(nn.Module):
    """
    混合激活函数，更适合处理超出[0,1]范围的数据
    结合了SiLU和GELU的特性，对正负值都有良好的响应
    """
    def __init__(self):
        super().__init__()
        # 可学习的通道适应参数
        self.neg_scale = nn.Parameter(th.ones(1) * 0.2)
        self.pos_scale = nn.Parameter(th.ones(1))
    
    def forward(self, x):
        # 将输入分为正值部分和负值部分单独处理
        pos_mask = (x > 0).float()
        neg_mask = 1.0 - pos_mask
        
        # 正值区域：使用SiLU
        pos_out = x * th.sigmoid(x) * self.pos_scale
        
        # 负值区域：使用GELU变体，对负值更友好
        neg_out = 0.5 * x * (1 + th.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * th.pow(x, 3)))) * self.neg_scale
        
        # 组合正负部分
        return pos_out * pos_mask + neg_out * neg_mask


# 新增：输入数据范围适配器
class InputRangeAdapter(nn.Module):
    """
    输入范围适配器，用于处理超出[0,1]范围的数据
    """
    def __init__(self, in_channels, out_channels=None, land_value=1.5, init_scale=0.4, init_shift=0.1):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels or in_channels
        self.land_value = land_value
        
        # 通道适应参数 - 注意：保持为成员变量，但在forward中动态匹配
        self.channel_scales = nn.Parameter(th.ones(1, 1, 1, 1))
        self.channel_shifts = nn.Parameter(th.zeros(1, 1, 1, 1))
        
        # 使用传入的参数初始化
        # nn.init.constant_(self.channel_scales, val=0.4)
        # nn.init.constant_(self.channel_shifts, val=0.1)
        nn.init.constant_(self.channel_scales, val=init_scale)
        nn.init.constant_(self.channel_shifts, val=init_shift)
        
        # 值范围映射层
        if self.out_channels != self.in_channels:
            self.proj = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1)
        else:
            self.proj = nn.Identity()
    
    def forward(self, x):
        # 获取实际输入通道数
        actual_channels = x.shape[1]
        
        # 识别陆地区域（值接近land_value）
        land_mask = (th.abs(x - self.land_value) < 0.1).float()
        
        # 对水域部分应用范围适配（确保陆地值不受影响）
        water_mask = 1.0 - land_mask
        adapted_x = x * water_mask  # 保留水域部分
        
        # 应用通道缩放和偏移（只对水域部分）- 广播到所有通道
        scales = self.channel_scales.expand(-1, actual_channels, -1, -1)
        shifts = self.channel_shifts.expand(-1, actual_channels, -1, -1)
        
        adapted_x = adapted_x * scales + shifts
        
        # 恢复陆地值
        adapted_x = adapted_x + x * land_mask
        
        # 投影到所需通道数
        return self.proj(adapted_x), land_mask


class GroupNorm32(nn.GroupNorm):
    def forward(self, x):
        return super().forward(x.float()).type(x.dtype)


def conv_nd(dims, *args, **kwargs):
    """
    Create a 1D, 2D, or 3D convolution module.
    """
    if dims == 1:
        return nn.Conv1d(*args, **kwargs)
    elif dims == 2:
        return nn.Conv2d(*args, **kwargs)
    elif dims == 3:
        return nn.Conv3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")


def linear(*args, **kwargs):
    """
    Create a linear module.
    """
    return nn.Linear(*args, **kwargs)


def avg_pool_nd(dims, *args, **kwargs):
    """
    Create a 1D, 2D, or 3D average pooling module.
    """
    if dims == 1:
        return nn.AvgPool1d(*args, **kwargs)
    elif dims == 2:
        return nn.AvgPool2d(*args, **kwargs)
    elif dims == 3:
        return nn.AvgPool3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")


def update_ema(target_params, source_params, rate=0.99):
    """
    Update target parameters to be closer to those of source parameters using
    an exponential moving average.

    :param target_params: the target parameter sequence.
    :param source_params: the source parameter sequence.
    :param rate: the EMA rate (closer to 1 means slower).
    """
    for targ, src in zip(target_params, source_params):
        targ.detach().mul_(rate).add_(src, alpha=1 - rate)


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


def scale_module(module, scale):
    """
    Scale the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().mul_(scale)
    return module


def mean_flat(tensor, mask=None):
    """
    Take the mean over all non-batch dimensions.
    """
    if mask is not None:
        tensor = tensor * mask
    return tensor.mean(dim=list(range(1, len(tensor.shape))))


def normalization(channels):
    """
    Make a standard normalization layer.

    :param channels: number of input channels.
    :return: an nn.Module for normalization.
    """
    return GroupNorm32(32, channels)


def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.

    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = th.exp(
        -math.log(max_period) * th.arange(start=0, end=half, dtype=th.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = th.cat([th.cos(args), th.sin(args)], dim=-1)
    if dim % 2:
        embedding = th.cat([embedding, th.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


def checkpoint(func, inputs, params, flag):
    """
    Evaluate a function without caching intermediate activations, allowing for
    reduced memory at the expense of extra compute in the backward pass.

    :param func: the function to evaluate.
    :param inputs: the argument sequence to pass to `func`.
    :param params: a sequence of parameters `func` depends on but does not
                   explicitly take as arguments.
    :param flag: if False, disable gradient checkpointing.
    """
    if flag:
        args = tuple(inputs) + tuple(params)
        return CheckpointFunction.apply(func, len(inputs), *args)
    else:
        return func(*inputs)


class CheckpointFunction(th.autograd.Function):
    @staticmethod
    def forward(ctx, run_function, length, *args):
        ctx.run_function = run_function
        ctx.input_tensors = list(args[:length])
        ctx.input_params = list(args[length:])
        with th.no_grad():
            output_tensors = ctx.run_function(*ctx.input_tensors)
        return output_tensors

    @staticmethod
    def backward(ctx, *output_grads):
        ctx.input_tensors = [x.detach().requires_grad_(True) for x in ctx.input_tensors]
        with th.enable_grad():
            # Fixes a bug where the first op in run_function modifies the
            # Tensor storage in place, which is not allowed for detach()'d
            # Tensors.
            shallow_copies = [x.view_as(x) for x in ctx.input_tensors]
            output_tensors = ctx.run_function(*shallow_copies)
        input_grads = th.autograd.grad(
            output_tensors,
            ctx.input_tensors + ctx.input_params,
            output_grads,
            allow_unused=True,
        )
        del ctx.input_tensors
        del ctx.input_params
        del output_tensors
        return (None, None) + input_grads
