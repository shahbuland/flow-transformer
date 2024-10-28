import torch
from torch import nn

class RMSNorm(nn.Module):
    def __init__(self, d, eps = 1.0e-6):
        super().__init__()
        self.g = nn.Parameter(torch.zeros(d))
        self.eps = eps

    def forward(self, x):
        # x is [b,n,d] or [b,n,h,d]
        if x.dim() == 4:
            gain = (1 + self.g)[None,None,None,:] # Add batch, sequence, and head dims
        else:
            gain = (1 + self.g)[None,None,:] # Add batch and sequence dims

        rms = (x.float().pow(2).mean(-1, keepdim = True) + self.eps).rsqrt() # [b, n]

        x = (x * rms.to(x.dtype))
        x = x * gain

        return x

LayerNorm = lambda dim: nn.LayerNorm(dim, elementwise_affine = False, eps = 1.0e-6)