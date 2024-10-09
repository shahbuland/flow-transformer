import torch
from torch import nn

from torchtyping import TensorType

class RMSNorm(nn.Module):
    def __init__(self, d, eps = 1.0e-6):
        super().__init__()
        self.g = nn.Parameter(torch.zeros(d))
        self.eps = eps

    def forward(self, x : TensorType["b", "n", "d"]):
        gain = (1 + self.g)[None,None,:] # Add a batch and sequence dim

        rms = (x.float().pow(2).mean(-1, keepdim = True) + self.eps).rsqrt() # [b, n]

        x = (x * rms.to(x.dtype))
        x = x * gain

        return x