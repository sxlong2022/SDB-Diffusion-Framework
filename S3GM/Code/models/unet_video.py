from abc import abstractmethod

import math

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F

from .fp16_util import convert_module_to_f16, convert_module_to_f32
from .nn import (
    SiLU,
    MixedRangeActivation,
    InputRangeAdapter,
    conv_nd,
    linear,
    avg_pool_nd,
    zero_module,
    normalization,
    timestep_embedding,
    checkpoint,
)
from .rpe import RPEAttention


class TimestepBlock(nn.Module):
    """
    Any module where forward() takes timestep embeddings as a second argument.
    """

    @abstractmethod
    def forward(self, x, emb):
        """
        Apply the module to `x` given `emb` timestep embeddings.
        """


class TimestepEmbedAttnThingsSequential(nn.Sequential, TimestepBlock):
    """
    A sequential module that passes extra things to the children that
    support it as an extra input.
    """
    def forward(self, x, emb, attn_mask, T=1, frame_indices=None, attn_weights_list=None):
        for layer in self:
            if isinstance(layer, TimestepBlock):
                kwargs = dict(emb=emb)
                kwargs['emb'] = emb
            elif isinstance(layer, FactorizedAttentionBlock):
                kwargs = dict(
                    temb=emb,
                    attn_mask=attn_mask,
                    T=T,
                    frame_indices=frame_indices,
                    attn_weights_list=attn_weights_list,
                )
            else:
                kwargs = {}
            x = layer(x, **kwargs)
        return x


class Upsample(nn.Module):
    """
    An upsampling layer with an optional convolution.

    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 upsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, dims=2):
        super().__init__()
        self.channels = channels
        self.use_conv = use_conv
        self.dims = dims
        if use_conv:
            self.conv = conv_nd(dims, channels, channels, 3, padding=1)

    def forward(self, x):
        assert x.shape[1] == self.channels
        if self.dims == 3:
            x = F.interpolate(
                x, (x.shape[2], x.shape[3] * 2, x.shape[4] * 2), mode="nearest"
            )
        else:
            x = F.interpolate(x, scale_factor=2, mode="nearest")
        if self.use_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    """
    A downsampling layer with an optional convolution.

    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 downsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, dims=2):
        super().__init__()
        self.channels = channels
        self.use_conv = use_conv
        self.dims = dims
        stride = 2 if dims != 3 else (1, 2, 2)
        if use_conv:
            self.op = conv_nd(dims, channels, channels, 3, stride=stride, padding=1)
        else:
            self.op = avg_pool_nd(stride)

    def forward(self, x):
        assert x.shape[1] == self.channels
        return self.op(x)


class ResBlock(TimestepBlock):
    """
    A residual block that can optionally change the number of channels.

    :param channels: the number of input channels.
    :param emb_channels: the number of timestep embedding channels.
    :param dropout: the rate of dropout.
    :param out_channels: if specified, the number of out channels.
    :param use_conv: if True and out_channels is specified, use a spatial
        convolution instead of a smaller 1x1 convolution to change the
        channels in the skip connection.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param use_checkpoint: if True, use gradient checkpointing on this module.
    """

    def __init__(
        self,
        channels,
        emb_channels,
        dropout,
        out_channels=None,
        use_conv=False,
        use_scale_shift_norm=False,
        dims=2,
        use_checkpoint=False,
        use_mixed_activation=True,
    ):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_checkpoint = use_checkpoint
        self.use_scale_shift_norm = use_scale_shift_norm
        self.use_mixed_activation = use_mixed_activation

        activation_fn = MixedRangeActivation if use_mixed_activation else SiLU
        
        self.in_layers = nn.Sequential(
            normalization(channels),
            activation_fn(),
            conv_nd(dims, channels, self.out_channels, 3, padding=1),
        )
        self.emb_layers = nn.Sequential(
            activation_fn(),
            linear(
                emb_channels,
                2 * self.out_channels if use_scale_shift_norm else self.out_channels,
            ),
        )
        self.out_layers = nn.Sequential(
            normalization(self.out_channels),
            activation_fn(),
            nn.Dropout(p=dropout),
            zero_module(
                conv_nd(dims, self.out_channels, self.out_channels, 3, padding=1)
            ),
        )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = conv_nd(
                dims, channels, self.out_channels, 3, padding=1
            )
        else:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 1)

    def forward(self, x, emb):
        """
        Apply the block to a Tensor, conditioned on a timestep embedding.

        :param x: an [N x C x ...] Tensor of features.
        :param emb: an [N x emb_channels] Tensor of timestep embeddings.
        :return: an [N x C x ...] Tensor of outputs.
        """
        return checkpoint(
            self._forward, (x, emb), self.parameters(), self.use_checkpoint
        )

    def _forward(self, x, emb):
        h = self.in_layers(x)
        emb_out = self.emb_layers(emb).type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]
        if self.use_scale_shift_norm:
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            scale, shift = th.chunk(emb_out, 2, dim=1)
            h = out_norm(h) * (1 + scale) + shift
            h = out_rest(h)
        else:
            h = h + emb_out
            h = self.out_layers(h)
        return self.skip_connection(x) + h


class FactorizedAttentionBlock(nn.Module):

    def __init__(self, channels, num_heads, use_rpe_net, time_embed_dim=None, use_checkpoint=False, use_mixed_activation=True):
        super().__init__()
        self.spatial_attention = RPEAttention(
            channels=channels, num_heads=num_heads, use_checkpoint=use_checkpoint,
            use_rpe_q=False, use_rpe_k=False, use_rpe_v=False,
            use_mixed_activation=use_mixed_activation
        )
        self.temporal_attention = RPEAttention(
            channels=channels, num_heads=num_heads, use_checkpoint=use_checkpoint,
            time_embed_dim=time_embed_dim, use_rpe_net=use_rpe_net,
            use_mixed_activation=use_mixed_activation
        )
        
        self.range_adjust = nn.Sequential(
            normalization(channels),
            MixedRangeActivation() if use_mixed_activation else SiLU(),
            zero_module(conv_nd(2, channels, channels, 1))
        )

    def forward(self, x, attn_mask, temb, T, frame_indices=None, attn_weights_list=None):
        # --- Prepare temporal attention mask ---
        # Input attn_mask has shape like [B, T, 1, H, W] or [B, T, 1, H]
        # We need a mask of shape [B, T] for temporal attention.
        # Reduce spatial dimensions: check if *any* spatial location is active (value > 0)
        if attn_mask is not None:
            # Determine the batch size B based on the input x shape and T
            if len(x.shape) == 4: # Input to FactorizedAttentionBlock is [BT, C, H, W]
                BT = x.shape[0]
                B = BT // T if T > 0 else 1
            elif len(x.shape) == 3: # Input is [BT, C, H]
                BT = x.shape[0]
                B = BT // T if T > 0 else 1
            else: # Fallback or error
                B = attn_mask.shape[0]

            # Check if attn_mask shape is [B, T, ...]
            if attn_mask.shape[0] == B and attn_mask.shape[1] == T:
                 temporal_attn_mask = attn_mask.view(B, T, -1).any(dim=-1) # Shape [B, T]
                 # Ensure the mask is boolean or float for multiplication later
                 temporal_attn_mask = temporal_attn_mask.float()
            else:
                 # Handle cases where attn_mask might already be reshaped or incorrect
                 print(f"Warning: Unexpected attn_mask shape {attn_mask.shape} in FactorizedAttentionBlock. Expected first two dims: ({B}, {T}). Using default mask.")
                 temporal_attn_mask = th.ones(B, T, device=x.device, dtype=th.float32) # Default to allow all interactions

        else:
            # If no mask provided, assume all time steps are valid
            if len(x.shape) == 4: BT = x.shape[0]; B = BT // T if T > 0 else 1
            elif len(x.shape) == 3: BT = x.shape[0]; B = BT // T if T > 0 else 1
            else: B = 1 # Fallback
            temporal_attn_mask = th.ones(B, T, device=x.device, dtype=th.float32)


        if len(x.shape) == 4:
            BT, C, H, W = x.shape
            B = BT//T if T > 0 else 1 # Recalculate B safely

            x = x + self.range_adjust(x)

            x = x.view(B, T, C, H, W).permute(0, 3, 4, 2, 1)  # B, H, W, C, T
            x = x.reshape(B, H*W, C, T)

            # --- Pass the correctly shaped mask to temporal_attention ---
            x = self.temporal_attention(x,
                                        temb,
                                        frame_indices,
                                        attn_mask=temporal_attn_mask, # Shape [B, T]
                                        attn_weights_list=None if attn_weights_list is None else attn_weights_list['temporal'],)

            x = x.view(B, H, W, C, T).permute(0, 4, 3, 1, 2)  # B, T, C, H, W
            x = x.reshape(B, T, C, H*W)

            x = self.spatial_attention(x,
                                    temb,
                                    frame_indices=None,
                                    attn_weights_list=None if attn_weights_list is None else attn_weights_list['spatial'])

            x = x.reshape(BT, C, H, W)

        elif len(x.shape) == 3:
            BT, C, H = x.shape
            B = BT//T if T > 0 else 1 # Recalculate B safely

            x = x + self.range_adjust(x.unsqueeze(-1)).squeeze(-1)

            x = x.view(B, T, C, H).permute(0, 3, 2, 1)  # B, H, C, T
            x = x.reshape(B, H, C, T)

            # --- Pass the correctly shaped mask to temporal_attention ---
            x = self.temporal_attention(x,
                                        temb,
                                        frame_indices,
                                        attn_mask=temporal_attn_mask, # Shape [B, T]
                                        attn_weights_list=None if attn_weights_list is None else attn_weights_list['temporal'],)

            x = x.view(B, H, C, T).permute(0, 3, 2, 1)  # B, T, C, H
            x = x.reshape(B, T, C, H)

            x = self.spatial_attention(x,
                                    temb,
                                    frame_indices=None,
                                    attn_weights_list=None if attn_weights_list is None else attn_weights_list['spatial'])

            x = x.reshape(BT, C, H)

        return x


class UNetVideoModel(nn.Module):
    """
    The full UNet model with attention and timestep embedding.

    :param in_channels: channels in the input Tensor.
    :param model_channels: base channel count for the model.
    :param out_channels: channels in the output Tensor.
    :param num_res_blocks: number of residual blocks per downsample.
    :param attention_resolutions: a collection of downsample rates at which
        attention will take place. May be a set, list, or tuple.
        For example, if this contains 4, then at 4x downsampling, attention
        will be used.
    :param dropout: the dropout probability.
    :param channel_mult: channel multiplier for each level of the UNet.
    :param conv_resample: if True, use learned convolutions for upsampling and
        downsampling.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param use_checkpoint: use gradient checkpointing to reduce memory usage.
    :param num_heads: the number of attention heads in each attention layer.
    """

    def __init__(
        self,
        in_channels,
        model_channels,
        out_channels,
        num_res_blocks,
        attention_resolutions,
        image_size=None,
        dropout=0,
        channel_mult=(1, 2, 4, 8),
        conv_resample=True,
        dims=2,
        use_checkpoint=False,
        num_heads=1,
        num_heads_upsample=-1,
        use_scale_shift_norm=False,
        use_rpe_net=False,
        use_mixed_activation=True,
        land_value=1.5,
        init_scale=0.4,
        init_shift=0.1
    ):
        super().__init__()

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        self.in_channels = in_channels + 1
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.use_checkpoint = use_checkpoint
        self.num_heads = num_heads
        self.num_heads_upsample = num_heads_upsample
        self.use_rpe_net = use_rpe_net
        self.use_mixed_activation = use_mixed_activation
        self.land_value = land_value

        self.input_adapter = InputRangeAdapter(
            in_channels,
            in_channels,
            land_value=land_value,
            init_scale=init_scale,
            init_shift=init_shift
        )

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            MixedRangeActivation() if use_mixed_activation else SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        self.input_blocks = nn.ModuleList(
            [
                TimestepEmbedAttnThingsSequential(
                    conv_nd(dims, self.in_channels, model_channels, 3, padding=1)
                )
            ]
        )
        input_block_chans = [model_channels]
        ch = model_channels
        ds = 1
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=mult * model_channels,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                        use_mixed_activation=use_mixed_activation,
                    )
                ]
                ch = mult * model_channels
                if ds in attention_resolutions:
                    layers.append(
                        FactorizedAttentionBlock(
                            ch, use_checkpoint=use_checkpoint, num_heads=num_heads, use_rpe_net=use_rpe_net, time_embed_dim=time_embed_dim, use_mixed_activation=use_mixed_activation,
                        )
                    )
                self.input_blocks.append(TimestepEmbedAttnThingsSequential(*layers))
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                self.input_blocks.append(
                    TimestepEmbedAttnThingsSequential(Downsample(ch, conv_resample, dims=dims))
                )
                input_block_chans.append(ch)
                ds *= 2
        
        self.middle_block = TimestepEmbedAttnThingsSequential(
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
                use_mixed_activation=use_mixed_activation,
            ),
            FactorizedAttentionBlock(
                ch, 
                use_checkpoint=use_checkpoint, 
                num_heads=num_heads, 
                use_rpe_net=use_rpe_net, 
                time_embed_dim=time_embed_dim,
                use_mixed_activation=use_mixed_activation
            ),
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
                use_mixed_activation=use_mixed_activation,
            ),
        )

        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                layers = [
                    ResBlock(
                        ch + input_block_chans.pop(),
                        time_embed_dim,
                        dropout,
                        out_channels=model_channels * mult,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                        use_mixed_activation=use_mixed_activation,
                    )
                ]
                ch = model_channels * mult
                if ds in attention_resolutions:
                    layers.append(
                        FactorizedAttentionBlock(
                            ch,
                            use_checkpoint=use_checkpoint,
                            num_heads=num_heads_upsample,
                            use_rpe_net=use_rpe_net,
                            time_embed_dim=time_embed_dim,
                            use_mixed_activation=use_mixed_activation,
                        )
                    )
                if level and i == num_res_blocks:
                    layers.append(Upsample(ch, conv_resample, dims=dims))
                    ds //= 2
                self.output_blocks.append(TimestepEmbedAttnThingsSequential(*layers))

        self.out = nn.Sequential(
            normalization(ch),
            MixedRangeActivation() if use_mixed_activation else SiLU(),
            zero_module(conv_nd(dims, model_channels, out_channels, 3, padding=1)),
        )

        # 添加条件引导层
        self.condition_guide = nn.Sequential(
            conv_nd(dims, self.in_channels-1, model_channels, 3, padding=1),
            MixedRangeActivation() if use_mixed_activation else SiLU(),
            conv_nd(dims, model_channels, model_channels, 1)
        )

    def convert_to_fp16(self):
        """
        Convert the torso of the model to float16.
        """
        self.input_blocks.apply(convert_module_to_f16)
        self.middle_block.apply(convert_module_to_f16)
        self.output_blocks.apply(convert_module_to_f16)

    def convert_to_fp32(self):
        """
        Convert the torso of the model to float32.
        """
        self.input_blocks.apply(convert_module_to_f32)
        self.middle_block.apply(convert_module_to_f32)
        self.output_blocks.apply(convert_module_to_f32)

    @property
    def inner_dtype(self):
        """
        Get the dtype used by the torso of the model.
        """
        return next(self.input_blocks.parameters()).dtype

    def forward(self, x, *, x0, timesteps, frame_indices=None,
                obs_mask=None, latent_mask=None, return_attn_weights=False
            ):
        """
        Apply the model to an input batch.

        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param y: an [N] Tensor of labels, if class-conditional.
        :return: an [N x C x ...] Tensor of outputs.
        """
        if len(x.shape) == 5:
            B, T, C, H, W = x.shape
        elif len(x.shape) == 4:
            B, T, C, H = x.shape
        timesteps = timesteps.view(B, 1).expand(B, T)
        
        # 检查输入维度并提供默认值
        if obs_mask is None:
            # obs_mask = th.zeros_like(x[:, :, :1]) # 这可能导致维度不匹配，如果x是5D
             obs_mask = th.zeros((B, T, 1) + x.shape[3:], device=x.device, dtype=x.dtype)
        if latent_mask is None:
            # latent_mask = th.zeros_like(x[:, :, :1])
             latent_mask = th.zeros((B, T, 1) + x.shape[3:], device=x.device, dtype=x.dtype)
        
        # 计算 attn_mask
        attn_mask = (obs_mask + latent_mask).clip(max=1)

        # 创建观测点指示器
        indicator_template = th.ones_like(x[:, :, :1, :, :]) if len(x.shape) == 5 else th.ones_like(x[:, :, :1, :])
        obs_indicator = indicator_template * obs_mask
        
        # 简化 cat 操作，避免维度问题
        combined_x = th.cat([
            x*(1-obs_mask) + x0*obs_mask,  # 主通道（观测点用x0替换）
            obs_indicator                  # 观测点指示器
        ], dim=2) # Shape: [B, T, C+1, H, W]
        
        # 记录实际输入通道数，以便后续处理
        actual_channels = combined_x.shape[2]
        if actual_channels != self.in_channels:
             # 允许 actual_channels 比 self.in_channels 少1（因为条件引导层输入是 in_channels-1）
             # 或者等于 self.in_channels (因为我们拼接了 obs_indicator)
             # 这里逻辑调整为检查是否为 C+1
             expected_combined_channels = self.in_channels # self.in_channels 是定义时的 C+1
             if actual_channels != expected_combined_channels:
                  print(f"警告: combined_x 通道数 ({actual_channels}) 与模型预期 ({expected_combined_channels}) 不匹配")

        # --- 使用 .view() 进行 Reshape ---
        reshaped_combined_x = combined_x.view(B*T, actual_channels, *combined_x.shape[3:]) # Shape: [B*T, C+1, H, W]

        # 使用输入适配器处理输入
        # 确保适配器接收和返回4D张量
        adapted_x, land_mask = self.input_adapter(reshaped_combined_x) # adapted_x Shape: [B*T, C+1, H, W]
        
        timesteps = timesteps.reshape(B*T)
        hs = []
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels)) # Shape: [B*T, emb_dim]
        
        # --- 确认 h 是 4D ---
        h = adapted_x.type(adapted_x.dtype) # Shape: [B*T, C+1, H, W]
        
        attns = {'spatial': [], 'temporal': [], 'mixed': []} if return_attn_weights else None
        
        # 以下代码处理前向传播
        for layer, module in enumerate(self.input_blocks):
            # --- 确认传入模块的 h 是 4D ---
            if len(h.shape) != 4:
                 print(f"警告: 传递给 input_block {layer} 的张量 h 不是4D，形状为 {h.shape}")
            h = module(h, emb, attn_mask, T=T, attn_weights_list=attns, frame_indices=frame_indices)
            hs.append(h)
        
        # --- 确认 middle_block 输入是 4D ---
        if len(h.shape) != 4:
            print(f"警告: 传递给 middle_block 的张量 h 不是4D，形状为 {h.shape}")
        h = self.middle_block(h, emb, attn_mask, T=T, attn_weights_list=attns, frame_indices=frame_indices)
        
        # 应用条件引导，但确保维度匹配
        try:
            # --- 确认 h 在条件引导前是 4D ---
            if len(h.shape) != 4:
                 print(f"警告: 条件引导前的张量 h 不是4D，形状为 {h.shape}")

            # 重塑x0和obs_mask以匹配h的维度
            reshaped_x0 = x0.reshape(B*T, C, *x0.shape[3:]) # x0原始通道数为C
            reshaped_mask = obs_mask.reshape(B*T, 1, *obs_mask.shape[3:]) # mask通道数为1

            # 准备条件引导的输入 (通道数为 C)
            condition_input = reshaped_x0

             # --- 确认条件引导输入是 4D ---
            if len(condition_input.shape) != 4:
                 print(f"警告: 传递给 condition_guide 的张量不是4D，形状为 {condition_input.shape}")


            # 应用条件引导
            if condition_input.shape[2:] == h.shape[2:]:  # 确保空间维度匹配
                # 注意：condition_guide 输入通道是 self.in_channels-1 = C
                condition_feature = self.condition_guide(condition_input) # 输出通道是 model_channels

                 # --- 确认条件引导输出和 h 的形状 ---
                if len(condition_feature.shape) != 4:
                    print(f"警告: condition_guide 输出不是4D，形状为 {condition_feature.shape}")
                if condition_feature.shape[1] != h.shape[1]: # 检查通道数是否匹配
                    print(f"警告: condition_guide 输出通道 ({condition_feature.shape[1]}) 与 h 通道 ({h.shape[1]}) 不匹配")


                if condition_feature.shape == h.shape:  # 最后检查确保完全匹配
                    h = h + condition_feature * reshaped_mask # 使用mask应用引导
                else:
                     # 如果形状不匹配，可能需要调整 condition_guide 的输出通道或 h 的通道
                     print(f"跳过条件引导：形状不匹配 - guide: {condition_feature.shape}, h: {h.shape}")

        except Exception as e:
            print(f"条件引导层应用失败: {e}")
            # 跳过条件引导，继续处理
        
        # 继续解码器部分
        for module in self.output_blocks:
             # --- 确认解码器输入是 4D ---
             pop_h = hs.pop()
             if len(h.shape) != 4 or len(pop_h.shape) != 4:
                  print(f"警告: 传递给 output_block 的张量不是4D - h: {h.shape}, pop_h: {pop_h.shape}")
             
             # --- 修改这里的通道数检查 ---
             # module[0] 是一个 ResBlock 实例
             # ResBlock 的输入通道数由 ch + input_block_chans.pop() 决定，
             # 并且 ResBlock 的第一个卷积层输入通道数就是 ResBlock 的输入通道数 self.channels。
             # 因此，我们可以直接使用 module[0].channels 来检查输入 ResBlock 的通道数。
             # cat_in 的通道数是 h.shape[1] + pop_h.shape[1]
             # module[0].channels 是 ResBlock 期望的输入通道数
             expected_input_channels = module[0].channels 
             actual_input_channels = h.shape[1] + pop_h.shape[1]
             if actual_input_channels != expected_input_channels: 
                  print(f"警告: output_block 输入通道数 ({actual_input_channels}) 与 ResBlock 期望 ({expected_input_channels}) 不匹配")

             cat_in = th.cat([h, pop_h], dim=1)
             h = module(cat_in, emb, attn_mask, T=T, attn_weights_list=attns, frame_indices=frame_indices)
        
        h = h.type(adapted_x.dtype)
        out = self.out(h)
        # --- 确保输出是 5D ---
        final_out = out.view(B, T, self.out_channels, *adapted_x.shape[2:])
        return final_out, attns

    def get_feature_vectors(self, x, timesteps, y=None):
        """
        Apply the model and return all of the intermediate tensors.

        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param y: an [N] Tensor of labels, if class-conditional.
        :return: a dict with the following keys:
                 - 'down': a list of hidden state tensors from downsampling.
                 - 'middle': the tensor of the output of the lowest-resolution
                             block in the model.
                 - 'up': a list of hidden state tensors from upsampling.
        """
        hs = []
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))
        result = dict(down=[], up=[])
        h = x.type(self.inner_dtype)
        for module in self.input_blocks:
            h = module(h, emb)
            hs.append(h)
            result["down"].append(h.type(x.dtype))
        h = self.middle_block(h, emb)
        result["middle"] = h.type(x.dtype)
        for module in self.output_blocks:
            cat_in = th.cat([h, hs.pop()], dim=1)
            h = module(cat_in, emb)
            result["up"].append(h.type(x.dtype))
        return result