import torch
from torch import nn
import einops as eo
import math

from rotary_embedding_torch import RotaryEmbedding

from .mlp import MLP

class AbsEmbedding(nn.Module):
    def __init__(self, seq_len, dim):
        super().__init__()

        self.embedding = nn.Parameter(torch.randn(seq_len, dim))
    
    def forward(self, x):
        # x: [b,n,d]
        p = eo.repeat(self.embedding, 'n d -> b n d', b = x.shape[0])
        return x + p

class RoPEEmbedding(nn.Module):
    """
    "Flat" Version of RoPE
    """
    def __init__(self, dim, flash = False):
        super().__init__()

        self.rope = RotaryEmbedding(dim // 2)
        self.flash = flash
    
    def forward(self, q, k):
        if self.flash: # [b,n,h,d] -> [b,h,n,d]
            q = q.transpose(1,2)
            k = k.transpose(1,2)
        q, k = self.rope.rotate_queries_or_keys(q), self.rope.rotate_queries_or_keys(k)
        if self.flash: # [b,h,n,d] -> [b,n,h,d]
            q = q.transpose(1,2)
            k = k.transpose(1,2)
        return q,k

class RoPE2D(nn.Module):
    def __init__(self, dim, n_patches_h, n_patches_w, flash = False):
        super().__init__()

        self.rope_h = RotaryEmbedding(dim // 4)
        self.rope_w = RotaryEmbedding(dim // 4)
        self.flash = flash

        self.n_h = n_patches_h 
        self.n_w = n_patches_w

    def forward(self, q, k):
        if self.flash: # [b,n,h,d] -> [b,h,n,d]
            q = q.transpose(1,2)
            k = k.transpose(1,2)

        q_h, q_w = q.chunk(2, dim = -1)
        k_h, k_w = k.chunk(2, dim = -1)
        
        q_h = eo.rearrange(q_h, 'b h (n_h n_w) d -> (b n_w) h n_h d', n_h = self.n_h)
        k_h = eo.rearrange(k_h, 'b h (n_h n_w) d -> (b n_w) h n_h d', n_h = self.n_h)

        q_w = eo.rearrange(q_w, 'b h (n_h n_w) d -> (b n_h) h n_w d', n_w = self.n_w)
        k_w = eo.rearrange(k_w, 'b h (n_h n_w) d -> (b n_h) h n_w d', n_w = self.n_w)

        q_h, k_h = self.rope_h.rotate_queries_or_keys(q_h), self.rope_h.rotate_queries_or_keys(k_h)
        q_w, k_w = self.rope_w.rotate_queries_or_keys(q_w), self.rope_w.rotate_queries_or_keys(k_w)

        q_h = eo.rearrange(q_h, '(b n_w) h n_h d -> b h (n_h n_w) d', n_w=self.n_w)
        k_h = eo.rearrange(k_h, '(b n_w) h n_h d -> b h (n_h n_w) d', n_w=self.n_w)

        q_w = eo.rearrange(q_w, '(b n_h) h n_w d -> b h (n_h n_w) d', n_h=self.n_h)
        k_w = eo.rearrange(k_w, '(b n_h) h n_w d -> b h (n_h n_w) d', n_h=self.n_h)

        q = torch.cat([q_h, q_w], dim=-1)
        k = torch.cat([k_h, k_w], dim=-1)

        if self.flash: # [b,h,n,d] -> [b,n,h,d]
            q = q.transpose(1,2)
            k = k.transpose(1,2)

        return q,k    

class TimestepEmbedding(nn.Module):
    def __init__(self, d_out, d_in = 512):
        super().__init__()

        self.mlp = MLP(d_in, d_out, use_scale = False)
        self.d = d_in # Assume this is even

    def forward(self, t):
        if t.ndim == 0:
            t = t.unsqueeze(0)
        # t is [B] tensor of timesteps ()
        t = t * 1000

        max_period = 10000 # This seems to always be assumed in all repos
        half = self.d // 2

        inds = torch.arange(half, device = t.device, dtype = t.dtype)
        freqs = (
            -math.log(max_period) * inds / half
        ).exp()

        embs = t[:,None] * freqs[None]
        embs = torch.cat([torch.cos(embs), torch.sin(embs)], dim = -1)

        return self.mlp(embs)