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
    def __init__(self, dim):
        super().__init__()

        self.rope = RotaryEmbedding(dim//2)

    def flip(self, x):
        return x.transpose(1,2) # (swap seq and heads or reverse)
    
    def forward(self, q, k):
        q = self.flip(q)
        k = self.flip(k)

        q,k = self.rope.rotate_queries_or_keys(q),self.rope.rotate_queries_or_keys(k)

        q = self.flip(q)
        k = self.flip(k)

        return q,k

class TimestepEmbedding(nn.Module):
    def __init__(self, d_out, d_in = 512, mult = 1000):
        super().__init__()

        self.mlp = MLP(d_in, dim_out=d_out)
        self.d = d_in # Assume this is even
        self.mult = mult

    def forward(self, t):
        if not isinstance(t, torch.Tensor):
            t = torch.tensor(t, device=self.mlp.fc_uv.weight.device, dtype=self.mlp.fc_uv.weight.dtype)
        if t.ndim == 0:
            t = t.unsqueeze(0)
        # t is [B] tensor of timesteps ()
        t = t * self.mult

        max_period = 10000 # This seems to always be assumed in all repos
        half = self.d // 2

        inds = torch.arange(half, device = t.device, dtype = t.dtype)
        freqs = (
            -math.log(max_period) * inds / half
        ).exp()

        embs = t[:,None] * freqs[None]
        embs = torch.cat([torch.cos(embs), torch.sin(embs)], dim = -1)

        return self.mlp(embs)
    
class StepEmbedding(nn.Module):
    def __init__(self, d_out, d_in=512, max_steps=128):
        super().__init__()

        self.mlp = MLP(d_in, dim_out=d_out)
        self.d = d_in
        self.max_steps = max_steps
        self.mult = 1000 / math.log2(max_steps)

    def forward(self, steps):
        if not isinstance(steps, torch.Tensor):
            steps = torch.tensor(steps, device=self.mlp.fc_uv.weight.device, dtype=self.mlp.fc_uv.weight.dtype)
        if steps.ndim == 0:
            steps = steps.unsqueeze(0)

        # Map steps to [0, log2(max_steps)]
        t = math.log2(self.max_steps) - torch.log2(steps.float())
        # Scale to [0, 1000]
        t = t * self.mult

        half = self.d // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=t.device, dtype=t.dtype) / half
        )

        embs = t[:, None] * freqs[None]
        embs = torch.cat([torch.cos(embs), torch.sin(embs)], dim=-1)

        return self.mlp(embs)