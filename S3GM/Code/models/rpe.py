import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch as th
from .nn import zero_module, normalization, checkpoint, MixedRangeActivation


class RPENet(nn.Module):
    def __init__(self, channels, num_heads, time_embed_dim, use_mixed_activation=True):
        super().__init__()
        self.embed_distances = nn.Linear(3, channels)
        self.embed_diffusion_time = nn.Linear(time_embed_dim, channels)
        
        # 使用混合激活函数
        self.activation = MixedRangeActivation() if use_mixed_activation else nn.SiLU()
        
        self.out = nn.Linear(channels, channels)
        self.out.weight.data *= 0.
        self.out.bias.data *= 0.
        self.channels = channels
        self.num_heads = num_heads
        
        # 添加尺度适应层
        self.scale_adaptation = nn.Parameter(th.ones(1) * 0.5)
        self.shift_adaptation = nn.Parameter(th.zeros(1))

    def forward(self, temb, relative_distances):
        # 使用对数变换处理相对距离，更好地处理大范围数据
        distance_embs = th.stack(
            [th.log(1+(relative_distances).clamp(min=0)),
             th.log(1+(-relative_distances).clamp(min=0)),
             (relative_distances == 0).float()],
            dim=-1
        )  # BxTxTx3
        
        B, T, _ = relative_distances.shape
        C = self.channels
        
        # 应用缩放适应
        time_emb = self.embed_diffusion_time(temb).view(B, T, 1, C) 
        dist_emb = self.embed_distances(distance_embs)
        
        # 合并嵌入
        emb = time_emb + dist_emb
        
        # 应用尺度适应
        emb = emb * self.scale_adaptation + self.shift_adaptation
        
        # 使用激活函数
        return self.out(self.activation(emb)).view(*relative_distances.shape, self.num_heads, self.channels//self.num_heads)


class RPE(nn.Module):
    # Based on https://github.com/microsoft/Cream/blob/6fb89a2f93d6d97d2c7df51d600fe8be37ff0db4/iRPE/DeiT-with-iRPE/rpe_vision_transformer.py
    def __init__(self, channels, num_heads, time_embed_dim, use_rpe_net=False, use_mixed_activation=True):
        """ This module handles the relative positional encoding.
        Args:
            channels (int): Number of input channels.
            num_heads (int): Number of attention heads.
        """
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = channels // self.num_heads
        self.use_rpe_net = use_rpe_net
        self.scale = th.sqrt(th.tensor(self.head_dim).float()).item()  # 用于缩放注意力得分
        
        if use_rpe_net:
            self.rpe_net = RPENet(channels, num_heads, time_embed_dim, use_mixed_activation=use_mixed_activation)
        else:
            self.lookup_table_weight = nn.Parameter(
                th.zeros(2 * 32 + 1,  # 使用默认beta=32
                         self.num_heads,
                         self.head_dim))
            
        # 添加数值稳定性参数
        self.stability_eps = 1e-6

    def get_R(self, pairwise_distances, temb):
        if self.use_rpe_net:
            return self.rpe_net(temb, pairwise_distances)
        else:
            # 使用处理后的相对距离索引查找表
            clamped_indices = (pairwise_distances + 32).clamp(0, 2 * 32).long()
            return self.lookup_table_weight[clamped_indices]  # BxTxTxHx(C/H)

    def forward(self, x, pairwise_distances, temb, mode):
        if mode == "qk":
            return self.forward_qk(x, pairwise_distances, temb)
        elif mode == "v":
            return self.forward_v(x, pairwise_distances, temb)
        else:
            raise ValueError(f"Unexpected RPE attention mode: {mode}")

    def forward_qk(self, qk, pairwise_distances, temb):
        # 检查输入形状是否合理
        if qk.dim() != 5:
            raise ValueError(f"Expected 5D tensor for qk, got shape {qk.shape}")
            
        # 获取相对位置编码
        R = self.get_R(pairwise_distances, temb)
        
        # 应用稳定性检查
        if th.isnan(R).any() or th.isinf(R).any():
            print(f"警告: 检测到RPE中的R包含NaN或Inf值")
            R = th.nan_to_num(R, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # 使用einsum计算注意力分数
        attention_scores = th.einsum(
            "bdhtf,btshf->bdhts", qk, R  # BxDxHxTxT
        )
        
        # 缩放注意力分数以增强稳定性
        attention_scores = attention_scores / (self.scale + self.stability_eps)
        
        return attention_scores

    def forward_v(self, attn, pairwise_distances, temb):
        # 检查输入形状是否合理
        if attn.dim() != 5:
            raise ValueError(f"Expected 5D tensor for attn, got shape {attn.shape}")
            
        # 获取相对位置编码
        R = self.get_R(pairwise_distances, temb)
        
        # 应用稳定性检查
        if th.isnan(R).any() or th.isinf(R).any():
            print(f"警告: 检测到RPE中的R包含NaN或Inf值")
            R = th.nan_to_num(R, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # 使用einsum计算加权值
        weighted_values = th.einsum(
            "bdhts,btshf->bdhtf", attn, R  # BxDxHxTxT
        )
        
        return weighted_values


class RPEAttention(nn.Module):
    # Based on https://github.com/microsoft/Cream/blob/6fb89a2f93d6d97d2c7df51d600fe8be37ff0db4/iRPE/DeiT-with-iRPE/rpe_vision_transformer.py#L42
    def __init__(self, channels, num_heads, use_checkpoint=False,
                 time_embed_dim=None, use_rpe_net=None,
                 use_rpe_q=True, use_rpe_k=True, use_rpe_v=True,
                 use_mixed_activation=True):
        super().__init__()
        self.num_heads = num_heads
        head_dim = channels // num_heads
        # 增加scale计算的稳定性
        self.scale = (head_dim ** -0.5) * 0.8  # 稍微降低scale以增强对大值的稳定性
        self.use_checkpoint = use_checkpoint
        self.use_mixed_activation = use_mixed_activation

        self.qkv = nn.Linear(channels, channels * 3)
        self.proj_out = zero_module(nn.Linear(channels, channels))
        self.norm = normalization(channels)

        # 数值稳定性参数
        self.stability_eps = 1e-6
        
        # 添加输入范围自适应层
        self.input_adjustment = nn.Sequential(
            nn.Linear(channels, channels),
            MixedRangeActivation() if use_mixed_activation else nn.SiLU(),
            nn.Linear(channels, channels)
        )
        nn.init.zeros_(self.input_adjustment[-1].weight)
        nn.init.zeros_(self.input_adjustment[-1].bias)

        if use_rpe_q or use_rpe_k or use_rpe_v:
            assert use_rpe_net is not None
        def make_rpe_func():
            return RPE(
                channels=channels, num_heads=num_heads,
                time_embed_dim=time_embed_dim, use_rpe_net=use_rpe_net,
                use_mixed_activation=use_mixed_activation
            )
        self.rpe_q = make_rpe_func() if use_rpe_q else None
        self.rpe_k = make_rpe_func() if use_rpe_k else None
        self.rpe_v = make_rpe_func() if use_rpe_v else None

    def forward(self, x, temb, frame_indices, attn_mask=None, attn_weights_list=None):
        out, attn = checkpoint(self._forward, (x, temb, frame_indices, attn_mask), self.parameters(), self.use_checkpoint)
        if attn_weights_list is not None:
            B, D, C, T = x.shape
            attn_weights_list.append(attn.detach().view(B*D, -1, T, T).mean(dim=1).abs())  # logging attn weights
        return out

    def _forward(self, x, temb, frame_indices, attn_mask):
        B, D, C, T = x.shape
        
        # 重塑张量为批次处理格式
        x = x.reshape(B*D, C, T)
        
        # 应用归一化
        x = self.norm(x)
        
        # 重塑回原始格式
        x = x.view(B, D, C, T)
        
        # 调整输入范围（残差连接）
        x_adjusted = x + self.input_adjustment(x.transpose(-1, -2)).transpose(-1, -2)
        
        # 转置为注意力计算所需形状
        x = th.einsum("BDCT -> BDTC", x_adjusted)
        
        # 计算QKV
        qkv = self.qkv(x).reshape(B, D, T, 3, self.num_heads, C // self.num_heads)
        qkv = th.einsum("BDTtHF -> tBDHTF", qkv)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # 应用缩放
        q = q * self.scale
        
        # 计算注意力分数
        attn = (q @ k.transpose(-2, -1))  # BxDxHxTxT
        
        # 应用相对位置编码
        if frame_indices is not None and (self.rpe_q is not None or self.rpe_k is not None or self.rpe_v is not None):
            # 计算帧索引之间的相对距离
            pairwise_distances = (frame_indices.unsqueeze(-1) - frame_indices.unsqueeze(-2))
            
            # 应用RPE
            if self.rpe_k is not None:
                attn = attn + self.rpe_k(q, pairwise_distances, temb=temb, mode="qk")
            
            if self.rpe_q is not None:
                attn = attn + self.rpe_q(k * self.scale, pairwise_distances, temb=temb, mode="qk").transpose(-1, -2)
        
        # 定义带掩码的softmax函数
        def softmax(w, attn_mask):
            if attn_mask is not None:
                allowed_interactions = attn_mask.view(B, 1, T) * attn_mask.view(B, T, 1)
                allowed_interactions += (1-attn_mask.view(B, 1, T)) * (1-attn_mask.view(B, T, 1))
                inf_mask = (1-allowed_interactions)
                inf_mask[inf_mask == 1] = float('inf')
                w = w - inf_mask.view(B, 1, 1, T, T)
            
            # 应用数值稳定性处理
            w_max = th.max(w, dim=-1, keepdim=True)[0].detach()
            w = w - w_max
            w_exp = th.exp(w.float())
            
            # 检查是否有NaN或Inf
            if th.isnan(w_exp).any() or th.isinf(w_exp).any():
                print(f"警告: Softmax计算中出现NaN或Inf")
                w_exp = th.nan_to_num(w_exp, nan=0.0, posinf=1.0, neginf=0.0)
            
            # 计算softmax
            w_sum = w_exp.sum(dim=-1, keepdim=True) + self.stability_eps
            return (w_exp / w_sum).type(w.dtype)
        
        # 应用softmax
        attn = softmax(attn, attn_mask)
        
        # 计算加权和
        out = attn @ v
        
        # 应用值的相对位置编码
        if frame_indices is not None and self.rpe_v is not None:
            pairwise_distances = (frame_indices.unsqueeze(-1) - frame_indices.unsqueeze(-2))
            out = out + self.rpe_v(attn, pairwise_distances, temb=temb, mode="v")
        
        # 重塑张量
        out = th.einsum("BDHTF -> BDTHF", out).reshape(B, D, T, C)
        
        # 应用输出投影
        out = self.proj_out(out)
        
        # 残差连接
        x = x + out
        
        # 转置回原始格式
        x = th.einsum("BDTC -> BDCT", x)
        
        return x, attn