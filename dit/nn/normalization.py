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
    norm_layer(block.mlp.uv)
    norm_layer(block.mlp.out)
    norm_layer(block.attn.qkv)
    norm_layer(block.attn.out)
    if block.attn.cross:
        norm_layer(block.attn.cross_qkv)

class ScalingLayer(nn.Module):
    """
    Scaling layer from normalized transformer.
    Produces some scaling value "treated" with some init and scale
    """
    def __init__(self, d_model, init, scale):
        super().__init__()

        init = float(init)
        scale = float(scale)       

        self.scale = scale
        self.init = init

        self.alpha = nn.Parameter(torch.full((d_model,), scale))

    def forward(self, x):
        alpha = (self.alpha * (self.init / self.scale))
        if x.ndim == 2:
            alpha = alpha[None,:]
        else:
            alpha = alpha[None,None,:]

        return alpha * x

class HeadScalingLayer(nn.Module):
    def __init__(self, n_heads, d_model, init, scale):
        super().__init__()

        init = float(init)
        scale = float(scale)

        self.scale = scale
        self.init = init

        d_head = d_model // n_heads
        self.alpha = nn.Parameter(torch.full((n_heads, d_head), scale))

    def forward(self, x):
        # x shape: [b, n, h, d]
        alpha = (self.alpha * (self.init / self.scale))  # [h, d]
        alpha = alpha[None, None, :, :]  # [1, 1, h, d]

        return alpha * x

class NormalizedLerp(nn.Module):
    """
    Wrapper around scaling layer to simplify lerps for residual signals
    """
    def __init__(self, d_model, init, scale):
        super().__init__()

        self.scale = ScalingLayer(d_model, init, scale)
    
    def forward(self, x, res):
        return norm(res + self.scale(x - res))