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
        
        # Use mixed activation function
        self.activation = MixedRangeActivation() if use_mixed_activation else nn.SiLU()
        
        self.out = nn.Linear(channels, channels)
        self.out.weight.data *= 0.
        self.out.bias.data *= 0.
        self.channels = channels
        self.num_heads = num_heads
        
        # Add scale adaptation layer
        self.scale_adaptation = nn.Parameter(th.ones(1) * 0.5)
        self.shift_adaptation = nn.Parameter(th.zeros(1))

    def forward(self, temb, relative_distances):
        # Use log transform for relative distances, better handling of large range data
        distance_embs = th.stack(
            [th.log(1+(relative_distances).clamp(min=0)),
             th.log(1+(-relative_distances).clamp(min=0)),
             (relative_distances == 0).float()],
            dim=-1
        )  # BxTxTx3
        
        B, T, _ = relative_distances.shape
        C = self.channels
        
        # Apply scale adaptation
        time_emb = self.embed_diffusion_time(temb).view(B, T, 1, C) 
        dist_emb = self.embed_distances(distance_embs)
        
        # Merge embeddings
        emb = time_emb + dist_emb
        
        # Apply scale adaptation
        emb = emb * self.scale_adaptation + self.shift_adaptation
        
        # Apply activation function
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
        self.scale = th.sqrt(th.tensor(self.head_dim).float()).item()  # Used for scaling attention scores
        
        if use_rpe_net:
            self.rpe_net = RPENet(channels, num_heads, time_embed_dim, use_mixed_activation=use_mixed_activation)
        else:
            self.lookup_table_weight = nn.Parameter(
                th.zeros(2 * 32 + 1,  # Use default beta=32
                         self.num_heads,
                         self.head_dim))
            
        # Numerical stability parameters
        self.stability_eps = 1e-6

    def get_R(self, pairwise_distances, temb):
        if self.use_rpe_net:
            return self.rpe_net(temb, pairwise_distances)
        else:
            # Use processed relative distance index lookup table
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
        # Check if input shape is reasonable
        if qk.dim() != 5:
            raise ValueError(f"Expected 5D tensor for qk, got shape {qk.shape}")
            
        # Get relative position encoding
        R = self.get_R(pairwise_distances, temb)
        
        # Apply stability check
        if th.isnan(R).any() or th.isinf(R).any():
            print(f"Warning: R in RPE contains NaN or Inf values")
            R = th.nan_to_num(R, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # Use einsum to calculate attention scores
        attention_scores = th.einsum(
            "bdhtf,btshf->bdhts", qk, R  # BxDxHxTxT
        )
        
        # Scale attention scores for stability
        attention_scores = attention_scores / (self.scale + self.stability_eps)
        
        return attention_scores

    def forward_v(self, attn, pairwise_distances, temb):
        # Check if input shape is reasonable
        if attn.dim() != 5:
            raise ValueError(f"Expected 5D tensor for attn, got shape {attn.shape}")
            
        # Get relative position encoding
        R = self.get_R(pairwise_distances, temb)
        
        # Apply stability check
        if th.isnan(R).any() or th.isinf(R).any():
            print(f"Warning: R in RPE contains NaN or Inf values")
            R = th.nan_to_num(R, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # Use einsum to calculate weighted values
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
        # Increase scale calculation stability
        self.scale = (head_dim ** -0.5) * 0.8  # Slightly reduce scale to enhance stability for large values
        self.use_checkpoint = use_checkpoint
        self.use_mixed_activation = use_mixed_activation

        self.qkv = nn.Linear(channels, channels * 3)
        self.proj_out = zero_module(nn.Linear(channels, channels))
        self.norm = normalization(channels)

        # Numerical stability parameters
        self.stability_eps = 1e-6
        
        # Add input range adaptation layer
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
        
        # Reshape tensor for batch processing
        x = x.reshape(B*D, C, T)
        
        # Apply normalization
        x = self.norm(x)
        
        # Reshape back to original format
        x = x.view(B, D, C, T)
        
        # Adjust input range (residual connection)
        x_adjusted = x + self.input_adjustment(x.transpose(-1, -2)).transpose(-1, -2)
        
        # Transpose to shape required for attention computation
        x = th.einsum("BDCT -> BDTC", x_adjusted)
        
        # Compute QKV
        qkv = self.qkv(x).reshape(B, D, T, 3, self.num_heads, C // self.num_heads)
        qkv = th.einsum("BDTtHF -> tBDHTF", qkv)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Apply scaling
        q = q * self.scale
        
        # Compute attention scores
        attn = (q @ k.transpose(-2, -1))  # BxDxHxTxT
        
        # Apply relative position encoding
        if frame_indices is not None and (self.rpe_q is not None or self.rpe_k is not None or self.rpe_v is not None):
            # Compute relative distances between frame indices
            pairwise_distances = (frame_indices.unsqueeze(-1) - frame_indices.unsqueeze(-2))
            
            # Apply RPE
            if self.rpe_k is not None:
                attn = attn + self.rpe_k(q, pairwise_distances, temb=temb, mode="qk")
            
            if self.rpe_q is not None:
                attn = attn + self.rpe_q(k * self.scale, pairwise_distances, temb=temb, mode="qk").transpose(-1, -2)
        
        # Define softmax function with mask
        def softmax(w, attn_mask):
            if attn_mask is not None:
                allowed_interactions = attn_mask.view(B, 1, T) * attn_mask.view(B, T, 1)
                allowed_interactions += (1-attn_mask.view(B, 1, T)) * (1-attn_mask.view(B, T, 1))
                inf_mask = (1-allowed_interactions)
                inf_mask[inf_mask == 1] = float('inf')
                w = w - inf_mask.view(B, 1, 1, T, T)
            
            # Apply numerical stability processing
            w_max = th.max(w, dim=-1, keepdim=True)[0].detach()
            w = w - w_max
            w_exp = th.exp(w.float())
            
            # Check for NaN or Inf
            if th.isnan(w_exp).any() or th.isinf(w_exp).any():
                print(f"Warning: NaN or Inf in Softmax computation")
                w_exp = th.nan_to_num(w_exp, nan=0.0, posinf=1.0, neginf=0.0)
            
            # Compute softmax
            w_sum = w_exp.sum(dim=-1, keepdim=True) + self.stability_eps
            return (w_exp / w_sum).type(w.dtype)
        
        # Apply softmax
        attn = softmax(attn, attn_mask)
        
        # Compute weighted sum
        out = attn @ v
        
        # Apply relative position encoding for values
        if frame_indices is not None and self.rpe_v is not None:
            pairwise_distances = (frame_indices.unsqueeze(-1) - frame_indices.unsqueeze(-2))
            out = out + self.rpe_v(attn, pairwise_distances, temb=temb, mode="v")
        
        # Reshape tensor
        out = th.einsum("BDHTF -> BDTHF", out).reshape(B, D, T, C)
        
        # Apply output projection
        out = self.proj_out(out)
        
        # Residual connection
        x = x + out
        
        # Transpose back to original format
        x = th.einsum("BDTC -> BDCT", x)
        
        return x, attn