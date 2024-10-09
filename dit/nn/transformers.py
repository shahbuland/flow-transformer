from torchtyping import TensorType
import torch
from torch import nn
import einops as eo

from .normalization import RMSNorm
from .modulation import DoubleModBlock, SimpleModulation
from .embeddings import RoPEEmbedding, RoPE2D
from .mlp import MLP
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

class MultiHeadAttention(nn.Module):
    def __init__(self, dim, n_heads):
        super().__init__()

        self.n_heads = n_heads
        self.scale = (dim // n_heads) ** -.5

    def forward(self, q, k, v):
        # [b, h, n, d] for all 3
        scores = torch.einsum('bhnd,bhmd->bhnm', q, k) * self.scale
        scores = torch.softmax(scores, dim = -1) # [b, h, n_in, n_out]
        out = torch.einsum('bhnm,bhmd->bhmd', scores, v)
        return out

class Attn(nn.Module):
    def __init__(self, n_heads : int, d_model : int, flash : bool = False, cross : bool = False):
        super().__init__()

        self.qkv = nn.Linear(d_model, 3 * d_model, bias = False)
        self.out = nn.Linear(d_model, d_model)

        self.head_split = lambda x: head_split(x, n_heads = n_heads, flash = flash)
        self.head_merge = lambda x: head_merge(x, flash = flash)

        if cross:
            self.cross_qkv = nn.Linear(d_model, 3 * d_model, bias = False)
            self.cross_q_norm = RMSNorm(d_model // n_heads)
            self.cross_k_norm = RMSNorm(d_model // n_heads)

        self.q_norm = RMSNorm(d_model // n_heads)
        self.k_norm = RMSNorm(d_model // n_heads)

        self.rope = RoPEEmbedding(d_model // n_heads, flash = flash)

        self.cross = cross
        self.flash = flash

        if flash:
            from flash_attn import flash_attn_func
            self.attn = flash_attn_func
        else:
            self.attn = MultiHeadAttention(d_model, n_heads)
            
    def forward(self, x, c = None):
        # x [b,n,d]
        # c [b,m,d] (conditioning for cross)
        b,n,d = x.shape

        qkv = self.qkv(x)
        q,k,v = contiguous_qkv_chunk(qkv)
        q,k,v = [self.head_split(i) for i in [q,k,v]]
        q,k = self.q_norm(q), self.k_norm(k)

        if self.cross:
            cross_qkv = self.cross_qkv(c)
            c_q, c_k, c_v = [self.head_split(i) for i in contiguous_qkv_chunk(cross_qkv)]
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
            attn_out = self.attn(q.half(), k.half(), v.half()).to(orig_dtype)
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

    self.mod = DoubleModBlock(d_model)

    self.attn = Attn(n_heads, d_model, flash, cross_attn)
    self.mlp = MLP(d_model)

    #self.norm1 = RMSNorm(d_model)
    #self.norm2 = RMSNorm(d_model)
    self.norm1 = nn.LayerNorm(d_model, elementwise_affine = False, eps = 1.0e-6)
    self.norm2 = nn.LayerNorm(d_model, elementwise_affine = False, eps = 1.0e-6)

    self.cross = cross_attn

  def forward(self, x : TensorType["b", "n", "d"], t_emb : TensorType["b", "d"], c = None):
    mod1, mod2 = self.mod(t_emb)

    resid_1 = x.clone() # Clone x to have a residual signal
    x = self.norm1(x)
    x = mod1.first_step(x)

    if self.cross:
        attn_out = self.attn(x, c)
    else:
        attn_out = self.attn(x)
    attn_out = mod1.second_step(attn_out)

    x = attn_out + resid_1
    resid_2 = x.clone()

    x = self.norm2(x)
    x = mod2.first_step(x)
    x = self.mlp(x)
    x = mod2.second_step(x)

    x = x + resid_2
    return x