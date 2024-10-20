import torch
from torch import nn
import torch.nn.functional as F

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

class Norm(nn.Module):
    def __init__(self, eps = 1.0e-6):
        super().__init__()
        self.eps = eps
    
    def forward(self, x):
        return norm(x)
        #rss = (x.float().pow(2).sum(-1, keepdim = True) + self.eps).rsqrt()
        #return x * rss
    
    
LayerNorm = lambda dim: nn.LayerNorm(dim, elementwise_affine = False, eps = 1.0e-6)

def norm(data):
    return F.normalize(data, p = 2, dim = -1, eps = 1.0e-6)

def norm_layer(module : nn.Module):
    """
    Normalize linear layer along embedding dimension
    """
    module.weight.data = norm(module.weight.data)

def norm_dit_block(block : nn.Module):
    """
    Shorthand for normalizing a whole dit block
    """
    norm_layer(block.mlp.fc1)
    norm_layer(block.mlp.fc2)
    norm_layer(block.attn.qkv)
    norm_layer(block.attn.out)
    if block.attn.cross:
        norm_layer(block.attn.cross_qkv)