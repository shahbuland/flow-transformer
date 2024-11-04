from torchtyping import TensorType
import torch
from torch import nn
import torch.nn.functional as F
import einops as eo

from .normalization import (
    RMSNorm, LayerNorm, norm,
    ScalingLayer, HeadScalingLayer, NormalizedLerp
)
from .modulation import DoubleModBlock, SimpleModulation
from .embeddings import RoPEEmbedding, RoPE2D
from .mlp import MLP, MixFFN
from ..configs import ModelConfig


def contiguous_qkv_chunk(qkv):
    q, k, v = qkv.chunk(3, dim = -1)
    return q.contiguous(), k.contiguous(), v.contiguous()

def head_split(x, n_heads, flash = False):
    if flash:
        return eo.rearrange(x, 'b n (h d) -> b n h d', h = n_heads)
    else:
        return eo.rearrange(x, 'b n (h d) -> b h n d', h = n_heads)
    
def head_merge(x, flash = False):
    if flash:
        return eo.rearrange(x, 'b n h d -> b n (h d)')
    else:
        return eo.rearrange(x, 'b h n d -> b n (h d)')

class Attn(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        d_model = config.d_model
        n_heads = config.n_heads
        self.flash = config.flash
        self.cross = config.take_label
        self.normalized = config.normalized

        self.qkv = nn.Linear(d_model, 3 * d_model, bias = False)
        self.out = nn.Linear(d_model, d_model)

        self.head_split = lambda x: head_split(x, n_heads = n_heads, flash = self.flash)
        self.head_merge = lambda x: head_merge(x, flash = self.flash)

        if self.cross:
            self.cross_qkv = nn.Linear(d_model, 3 * d_model, bias = False)
            if not self.normalized:
                self.cross_q_norm = RMSNorm(d_model // n_heads)
                self.cross_k_norm = RMSNorm(d_model // n_heads)

        if not self.normalized:
            self.q_norm = RMSNorm(d_model // n_heads)
            self.k_norm = RMSNorm(d_model // n_heads)

        if self.normalized:
            self.scale = HeadScalingLayer(n_heads, d_model, 1, d_model ** -.5)
            if self.cross:
                self.cross_scale = HeadScalingLayer(n_heads, d_model, 1, d_model ** -.5)

        self.rope = RoPEEmbedding(d_model // n_heads, flash = self.flash)

        if self.normalized:
            self.attn_scale = (d_model // n_heads) ** .5
        else:
            self.attn_scale = (d_model // n_heads) ** -.5

        if self.flash:
            from flash_attn import flash_attn_func
            self.attn = flash_attn_func
        else:
            self.attn = F.scaled_dot_product_attention
            
    def forward(self, x, c = None):
        # x [b,n,d]
        # c [b,m,d] (conditioning for cross)
        b,n,d = x.shape

        qkv = self.qkv(x)
        q,k,v = contiguous_qkv_chunk(qkv)
        q,k,v = [self.head_split(i) for i in [q,k,v]]

        if self.normalized:
            q = self.scale(norm(q))
            k = self.scale(norm(k))
        else:
            q = self.q_norm(q)
            k = self.k_norm(k)

        if self.cross:
            cross_qkv = self.cross_qkv(c)
            c_q, c_k, c_v = [self.head_split(i) for i in contiguous_qkv_chunk(cross_qkv)]
            
            if self.normalized:
                c_q = self.cross_scale(norm(c_q))
                c_k = self.cross_scale(norm(c_k))
            else:
                c_q = self.cross_q_norm(c_q)
                c_k = self.cross_k_norm(c_k)

            # note flash is [b n h d], otherwise [b h n d]
            if self.flash:
                seq_dim = 1
            else:
                seq_dim = 2

            q = torch.cat([q, c_q], dim = seq_dim)
            k = torch.cat([k, c_k], dim = seq_dim)
            v = torch.cat([v, c_v], dim = seq_dim)
        
        q,k = self.rope(q,k)

        if self.flash:
            orig_dtype = q.dtype
            flash_attn_dtype = torch.bfloat16
            attn_out = self.attn(q.to(flash_attn_dtype), k.to(flash_attn_dtype), v.to(flash_attn_dtype), softmax_scale = self.attn_scale).to(orig_dtype)
        else:
            attn_out = self.attn(q,k,v)
        
        attn_out = self.head_merge(attn_out)

        if self.cross:
            attn_out = attn_out[:,:n]
        
        return self.out(attn_out)

class DiTBlock(nn.Module):
  def __init__(self, config):
    super().__init__()

    d_model = config.d_model
    n_heads = config.n_heads
    flash = config.flash
    cross_attn = config.take_label
    self.normalized = config.normalized

    self.mod = DoubleModBlock(d_model, normalized = config.normalized)

    self.attn = Attn(config)
    self.mlp = MLP(d_model, use_scale = self.normalized)

    self.cross = cross_attn

    if self.normalized:
        self.lerp_attn = NormalizedLerp(d_model, 1/config.n_layers, d_model ** -.5)
        self.lerp_mlp = NormalizedLerp(d_model, 1/config.n_layers, d_model ** -.5)
    else:
        self.norm_1 = LayerNorm(d_model)
        self.norm_2 = LayerNorm(d_model)

  def forward(self, x : TensorType["b", "n", "d"], t_emb : TensorType["b", "d"], c = None):
    mod1, mod2 = self.mod(t_emb)

    resid_1 = x.clone() # h
    
    if not self.normalized:
        x = self.norm_1(x)

    x = mod1.first_step(x)

    if self.cross:
        attn_out = self.attn(x, c)
    else:
        attn_out = self.attn(x)    

    attn_out = mod1.second_step(attn_out) # h_A

    if self.normalized:
        x = self.lerp_attn(x, resid_1)
    else:
        x = resid_1 + mod1.second_step(attn_out)

    resid_2 = x.clone()

    if not self.normalized:
        x = self.norm_2(x)

    x = mod2.first_step(x)
    x = self.mlp(x)
    x = mod2.second_step(x) # h_M

    if self.normalized:
        x = self.lerp_mlp(x, resid_2)
    else:
        x = resid_2 + x

    return x

