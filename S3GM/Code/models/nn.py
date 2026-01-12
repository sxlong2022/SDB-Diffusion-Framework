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


# New: Mixed activation function - adapts to wider range of input values
class MixedRangeActivation(nn.Module):
    """
    Mixed activation function, better suited for handling data outside [0,1] range
    Combines characteristics of SiLU and GELU, with good response to both positive and negative values
    """
    def __init__(self):
        super().__init__()
        # Learnable channel adaptation parameters
        self.neg_scale = nn.Parameter(th.ones(1) * 0.2)
        self.pos_scale = nn.Parameter(th.ones(1))
    
    def forward(self, x):
        # Separate input into positive and negative parts for individual processing
        pos_mask = (x > 0).float()
        neg_mask = 1.0 - pos_mask
        
        # Positive region: use SiLU
        pos_out = x * th.sigmoid(x) * self.pos_scale
        
        # Negative region: use GELU variant, more friendly to negative values
        neg_out = 0.5 * x * (1 + th.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * th.pow(x, 3)))) * self.neg_scale
        
        # Combine positive and negative parts
        return pos_out * pos_mask + neg_out * neg_mask


# New: Input data range adapter
class InputRangeAdapter(nn.Module):
    """
    Input range adapter for handling data outside [0,1] range
    """
    def __init__(self, in_channels, out_channels=None, land_value=1.5, init_scale=0.4, init_shift=0.1):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels or in_channels
        self.land_value = land_value
        
        # Channel adaptation parameters - Note: keep as member variables, but dynamically match in forward
        self.channel_scales = nn.Parameter(th.ones(1, 1, 1, 1))
        self.channel_shifts = nn.Parameter(th.zeros(1, 1, 1, 1))
        
        # Initialize with passed parameters
        # nn.init.constant_(self.channel_scales, val=0.4)
        # nn.init.constant_(self.channel_shifts, val=0.1)
        nn.init.constant_(self.channel_scales, val=init_scale)
        nn.init.constant_(self.channel_shifts, val=init_shift)
        
        # Value range mapping layer
        if self.out_channels != self.in_channels:
            self.proj = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1)
        else:
            self.proj = nn.Identity()
    
    def forward(self, x):
        # Get actual input channel count
        actual_channels = x.shape[1]
        
        # Identify land regions (values close to land_value)
        land_mask = (th.abs(x - self.land_value) < 0.1).float()
        
        # Apply range adaptation to water regions (ensure land values are not affected)
        water_mask = 1.0 - land_mask
        adapted_x = x * water_mask  # Keep water region
        
        # Apply channel scaling and offset (only to water region) - broadcast to all channels
        scales = self.channel_scales.expand(-1, actual_channels, -1, -1)
        shifts = self.channel_shifts.expand(-1, actual_channels, -1, -1)
        
        adapted_x = adapted_x * scales + shifts
        
        # Restore land values
        adapted_x = adapted_x + x * land_mask
        
        # Project to required channel count
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
